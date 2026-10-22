# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM-5.3 Channel-INT4 的纯配置判断和权重布局转换。

路由 expert 使用 compressed-tensors 的 pack-quantized：每个 int32 的低位
开始连续存放 8 个 INT4。现有 W4A8 kernel 吃的是 SlimQuant 布局，每个字节
高 4 位是偶数 K、低 4 位是奇数 K。这里只做这两种布局之间的转换，以及
indexer 共享层、checkpoint 张量名的判断。
"""

from __future__ import annotations

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


def is_shared_indexer_layer(config: Any, layer_id: int) -> bool:
    """主干里标记为 shared 的层复用前一个 full indexer 的 top-k。

    ``indexer_types`` 只覆盖主干层。MTP 层号大于等于 ``num_hidden_layers``，
    即使配置了共享类型，也保留自己的 indexer。没有该字段时保持原行为。
    """
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is None:
        return False
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if (
        not isinstance(indexer_types, (list, tuple))
        or not isinstance(num_hidden_layers, int)
        or len(indexer_types) != num_hidden_layers
    ):
        raise ValueError(
            "indexer_types 必须为主干每一层提供一个条目："
            f"期望 {num_hidden_layers}，实际 {indexer_types!r}。"
        )
    invalid = sorted(set(indexer_types) - {"full", "shared"})
    if invalid:
        raise ValueError(f"indexer_types 只支持 full 和 shared，收到 {invalid}。")
    if not isinstance(layer_id, int) or layer_id < 0:
        raise ValueError(f"indexer 层号无效：{layer_id!r}。")
    if layer_id >= num_hidden_layers:
        return False
    if indexer_types[layer_id] != "shared":
        return False
    if "full" not in indexer_types[: layer_id + 1]:
        raise ValueError(
            f"shared indexer 层 {layer_id} 前面没有 full indexer。"
        )
    return True


def normalize_routed_expert_checkpoint_name(
    name: str,
    experts_use_unpacked_weight: bool,
) -> str | None:
    """把路由 expert 的打包张量名改成现有 loader 能对上的名字。

    返回 None 表示 ``weight_shape``，调用方直接跳过。只有参数名是
    ``w13_weight`` 而不是 ``w13_weight_packed`` 时，才把 ``weight_packed``
    改成 ``weight``。
    """
    if ".mlp.experts." not in name:
        return name
    if name.endswith(".weight_shape"):
        return None
    if experts_use_unpacked_weight and name.endswith(".weight_packed"):
        return name.replace(".weight_packed", ".weight")
    return name


def convert_compressed_tensors_int4_to_slimquant(
    packed: torch.Tensor,
) -> torch.Tensor:
    """把 compressed-tensors 的 int32 INT4 打包转成 SlimQuant 的 int8 打包。

    输入最后一维是 ``K / 8``。输出最后一维是 ``K / 2``，字节内高 4 位为偶数
    K、低 4 位为奇数 K，数值仍是 INT4 的补码。
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
    swapped = ((raw & 0x0F) << 4) | (raw >> 4)
    return swapped.view(torch.int8)
