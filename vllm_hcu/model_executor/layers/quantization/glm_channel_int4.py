# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM-5.3 Channel-INT4 的纯配置判断和权重布局转换。

路由 expert 使用 compressed-tensors 的 pack-quantized：每个 int32 的低位
开始连续存放 8 个 INT4，按 uint4b8 存储（数值加 8）。现有 W4A8 kernel 吃的是
SlimQuant 布局，每个字节高 4 位是偶数 K、低 4 位是奇数 K，数值为补码。这里
只做这两种布局之间的转换、AITER gemm2 布局重排，以及 checkpoint 张量名的判断。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_INT4_PACK_FACTOR = 8


def repack_slimquant_w4a8_for_aiter(weight: torch.Tensor) -> torch.Tensor:
    """Regroup high-nibble-first packed INT4 bytes for AITER W4A8.

    ``convert_compressed_tensors_int4_to_slimquant`` fixes the nibble order
    used by the checkpoint.  AITER's W4A8 MoE layout additionally groups eight
    logical INT4 values as ``[0..3]`` in the high nibbles and ``[4..7]`` in
    the low nibbles.  This conversion is deliberately separate from the
    checkpoint loader so it is applied once to the rank-3 expert tensor after
    all experts have been assembled.
    """
    if weight.dtype != torch.int8 or weight.ndim < 1 or weight.shape[-1] <= 0:
        raise ValueError(
            "AITER W4A8 repack expects a non-empty int8 packed tensor, "
            f"got dtype={weight.dtype} shape={tuple(weight.shape)}."
        )

    packed_u8 = weight.contiguous().view(torch.uint8)
    nibbles = torch.stack(
        ((packed_u8 >> 4) & 0x0F, packed_u8 & 0x0F), dim=-1
    ).reshape(*packed_u8.shape[:-1], -1)
    if nibbles.shape[-1] % 8:
        raise ValueError(
            "AITER W4A8 repack requires the logical INT4 dimension to be "
            f"divisible by 8, got shape={tuple(weight.shape)}."
        )

    blocks = nibbles.reshape(*packed_u8.shape[:-1], -1, 8)
    repacked = ((blocks[..., :4] << 4) | blocks[..., 4:]).reshape_as(packed_u8)
    return repacked.contiguous().view(torch.int8)


def repack_and_shuffle_w4a8_gemm2(
    weight: torch.Tensor,
    shuffle_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """把 SlimQuant 打包的 ``[E, N, K/2]`` expert 转成 AITER moe_c 的 gemm2 布局。

    先做 8-nibble 重排，再对每个 expert 调用 AITER 的
    ``w4a8_moe_layout_shuffle_gemm2``。结果是一次性的破坏性转换，只能在加载
    时做一次，不能在 forward 里重复做。
    """
    if weight.ndim != 3 or weight.dtype != torch.int8:
        raise ValueError(
            "W4A8 gemm2 repack expects [E, N, K/2] int8 weights, "
            f"got dtype={weight.dtype} shape={tuple(weight.shape)}."
        )
    if shuffle_fn is None:
        try:
            from aiter.ops.shuffle import (
                w4a8_moe_layout_shuffle_gemm2 as shuffle_fn,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GLM-5.3 W4A8 AITER 路径需要 "
                "aiter.ops.shuffle.w4a8_moe_layout_shuffle_gemm2；当前 AITER "
                "没有提供，可改用 --moe-backend triton。"
            ) from exc

    repacked = repack_slimquant_w4a8_for_aiter(weight).view(torch.uint8)
    shuffled = torch.stack(
        [shuffle_fn(repacked[index]) for index in range(weight.shape[0])]
    ).reshape(weight.shape)
    return shuffled.contiguous().view(torch.int8)


def is_glm_moe_dsa_model(hf_config: Any) -> bool:
    """当前配置是否是 GLM MoE DSA。"""
    if getattr(hf_config, "model_type", None) == "glm_moe_dsa":
        return True
    architectures = getattr(hf_config, "architectures", None) or ()
    return "GlmMoeDsaForCausalLM" in architectures


def is_glm_channel_int4_w4a8(hf_quant_cfg: dict[str, Any] | None) -> bool:
    """量化配置是否同时包含 channel INT4 expert 和 channel INT8 Linear。"""
    if not isinstance(hf_quant_cfg, dict):
        return False
    if hf_quant_cfg.get("quant_method") != "compressed-tensors":
        return False
    groups = hf_quant_cfg.get("config_groups")
    if not isinstance(groups, dict):
        return False
    has_int4 = False
    has_int8 = False
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights") or {}
        activations = group.get("input_activations") or {}
        group_format = group.get("format") or hf_quant_cfg.get("format")
        if (
            weights.get("num_bits") == 4
            and weights.get("strategy") == "channel"
            and weights.get("type") == "int"
            and weights.get("symmetric") is True
            and activations.get("num_bits") == 8
            and activations.get("strategy") == "token"
            and activations.get("dynamic") is True
            and activations.get("type") == "int"
            and group_format == "pack-quantized"
        ):
            has_int4 = True
        if (
            weights.get("num_bits") == 8
            and weights.get("strategy") == "channel"
            and weights.get("type") == "int"
            and activations.get("num_bits") == 8
            and activations.get("strategy") == "token"
            and activations.get("dynamic") is True
        ):
            has_int8 = True
    return has_int4 and has_int8


def normalize_routed_expert_checkpoint_name(
    name: str,
    experts_use_unpacked_weight: bool,
) -> str | None:
    """把路由 expert 的打包张量名改成现有 loader 能对上的名字。

    只在 expert 参数名是 ``w13_weight`` 而不是 ``w13_weight_packed`` 时生效：
    ``weight_packed`` 改成 ``weight``，``weight_shape`` 返回 None 让调用方跳过。
    参数本身就是 packed 命名（例如 compressed-tensors W4A16）时名字原样返回，
    它们的 ``weight_shape`` 也照常加载。
    """
    if ".mlp.experts." not in name or not experts_use_unpacked_weight:
        return name
    if name.endswith(".weight_shape"):
        return None
    if name.endswith(".weight_packed"):
        return name.replace(".weight_packed", ".weight")
    return name


def convert_compressed_tensors_int4_to_slimquant(
    packed: torch.Tensor,
) -> torch.Tensor:
    """把 compressed-tensors 的 int32 INT4 打包转成 SlimQuant 的 int8 打包。

    输入最后一维是 ``K / 8``，每个 nibble 按 uint4b8 存储（数值加 8）。输出
    最后一维是 ``K / 2``，字节内高 4 位为偶数 K、低 4 位为奇数 K，数值转成
    INT4 的补码。
    """
    if packed.dtype != torch.int32 or packed.ndim < 1 or packed.shape[-1] <= 0:
        raise ValueError(
            "compressed-tensors INT4 权重必须是最后一维为正的 int32，"
            f"收到 dtype={packed.dtype} shape={tuple(packed.shape)}。"
        )
    raw = packed.contiguous().view(torch.uint8)
    if raw.shape[-1] != packed.shape[-1] * (_INT4_PACK_FACTOR // 2):
        raise ValueError(
            "INT4 int32 展开后的字节数不正确："
            f"packed={tuple(packed.shape)} bytes={tuple(raw.shape)}。"
        )
    # 交换 nibble 后异或 0x88：每个 nibble 的最高位取反，即减去偏移 8。
    swapped = (((raw & 0x0F) << 4) | (raw >> 4)) ^ 0x88
    return swapped.view(torch.int8)
