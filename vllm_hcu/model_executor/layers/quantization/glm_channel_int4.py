# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Config checks and weight-layout conversion for GLM-5.3 channel INT4.

Routed experts use compressed-tensors pack-quantized storage: eight INT4
values are packed into each int32 from the low bits, stored as uint4b8
(value plus 8). Existing W4A8 kernels consume the SlimQuant layout, where
the high nibble of each byte is even K, the low nibble is odd K, and the
values are two's complement. This module only converts between those two
layouts, applies the AITER gemm2 reorder, and classifies checkpoint tensor
names.
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
    """Convert SlimQuant-packed ``[E, N, K/2]`` experts to the AITER moe_c gemm2 layout.

    Regroup eight nibbles first, then call AITER's
    ``w4a8_moe_layout_shuffle_gemm2`` on each expert. The result is a one-shot
    destructive conversion. It must run once at load time and must not run
    again in forward.
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
                "GLM-5.3 W4A8 AITER path requires "
                "aiter.ops.shuffle.w4a8_moe_layout_shuffle_gemm2; this AITER "
                "build does not provide it. Use --moe-backend triton instead."
            ) from exc

    repacked = repack_slimquant_w4a8_for_aiter(weight).view(torch.uint8)
    shuffled = torch.stack(
        [shuffle_fn(repacked[index]) for index in range(weight.shape[0])]
    ).reshape(weight.shape)
    return shuffled.contiguous().view(torch.int8)


def is_glm_moe_dsa_model(hf_config: Any) -> bool:
    """Return whether the config is a GLM MoE DSA model."""
    if getattr(hf_config, "model_type", None) == "glm_moe_dsa":
        return True
    architectures = getattr(hf_config, "architectures", None) or ()
    return "GlmMoeDsaForCausalLM" in architectures


def is_glm_channel_int4_w4a8(hf_quant_cfg: dict[str, Any] | None) -> bool:
    """Return whether the quant config mixes channel INT4 experts and channel INT8 linears."""
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
    """Rewrite packed routed-expert tensor names so the existing loader can match them.

    Applied only when expert parameters are named ``w13_weight`` rather than
    ``w13_weight_packed``: ``weight_packed`` becomes ``weight``, and
    ``weight_shape`` returns None so the caller skips it. Parameters that are
    already packed-named (for example compressed-tensors W4A16) are returned
    unchanged, and their ``weight_shape`` tensors are still loaded.
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
    """Convert compressed-tensors int32 INT4 packing to SlimQuant int8 packing.

    The input's last dimension is ``K / 8``. Each nibble is uint4b8 (value
    plus 8). The output's last dimension is ``K / 2``: the high nibble is
    even K, the low nibble is odd K, and the values are INT4 two's complement.
    """
    if packed.dtype != torch.int32 or packed.ndim < 1 or packed.shape[-1] <= 0:
        raise ValueError(
            "compressed-tensors INT4 weights must be int32 with a positive "
            f"last dimension, got dtype={packed.dtype} shape={tuple(packed.shape)}."
        )
    raw = packed.contiguous().view(torch.uint8)
    if raw.shape[-1] != packed.shape[-1] * (_INT4_PACK_FACTOR // 2):
        raise ValueError(
            "INT4 int32 unpacked byte count is wrong: "
            f"packed={tuple(packed.shape)} bytes={tuple(raw.shape)}."
        )
    # Swap nibbles, then XOR 0x88: flip the high bit of each nibble,
    # which subtracts the uint4b8 bias of 8.
    swapped = (((raw & 0x0F) << 4) | (raw >> 4)) ^ 0x88
    return swapped.view(torch.int8)
