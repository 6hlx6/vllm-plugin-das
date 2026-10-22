# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM-5.3 Channel-INT4 的打包转换、indexer 共享和量化识别。"""

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
    convert_compressed_tensors_int4_to_slimquant,
    is_glm_channel_int4_w4a8,
    is_shared_indexer_layer,
    normalize_routed_expert_checkpoint_name,
    repack_slimquant_w4a8_for_aiter,
)
from vllm_hcu.model_executor.layers.quantization.slimquant_facade import (
    GlmMoeDsaW4A8Facade,
)


def _pack_compressed_tensors_int4(values: torch.Tensor) -> torch.Tensor:
    """按 compressed-tensors 的低位优先规则把有符号 INT4 打进 int32。"""
    if values.shape[-1] % 8:
        raise AssertionError("K 必须是 8 的倍数")
    nibbles = (values.to(torch.int32) & 0xF).reshape(*values.shape[:-1], -1, 8)
    packed = torch.zeros(
        *values.shape[:-1],
        values.shape[-1] // 8,
        dtype=torch.int32,
    )
    for index in range(8):
        packed |= nibbles[..., index] << (4 * index)
    return packed


def _unpack_slimquant_int4(packed: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    high = ((raw >> 4) & 0xF).to(torch.int16)
    low = (raw & 0xF).to(torch.int16)
    high = torch.where(high >= 8, high - 16, high)
    low = torch.where(low >= 8, low - 16, low)
    return torch.stack((high, low), dim=-1).flatten(-2).to(torch.int8)


def _quant_config() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_int4": {
                "format": "pack-quantized",
                "weights": {
                    "num_bits": 4,
                    "strategy": "channel",
                    "type": "int",
                    "symmetric": True,
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "token",
                    "dynamic": True,
                    "type": "int",
                },
            },
            "group_int8": {
                "format": "int-quantized",
                "weights": {
                    "num_bits": 8,
                    "strategy": "channel",
                    "type": "int",
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "token",
                    "dynamic": True,
                    "type": "int",
                },
            },
        },
    }


def test_compressed_tensors_int4_matches_slimquant_nibbles() -> None:
    values = torch.tensor(
        [[1, -1, 7, -8, 0, 2, 3, 4], [-7, 5, -3, 6, 1, 1, -8, 7]],
        dtype=torch.int8,
    )
    packed = _pack_compressed_tensors_int4(values)
    assert tuple(packed.shape) == (2, 1)
    converted = convert_compressed_tensors_int4_to_slimquant(packed)
    assert converted.dtype == torch.int8
    assert tuple(converted.shape) == (2, 4)
    assert torch.equal(_unpack_slimquant_int4(converted), values)


def test_aiter_w4a8_repack_regroups_eight_nibbles() -> None:
    # SlimQuant bytes encode [0,1], [2,3], [4,5], [6,7].
    packed = torch.tensor(
        [[0x01, 0x23, 0x45, 0x67]], dtype=torch.uint8
    ).view(torch.int8)
    repacked = repack_slimquant_w4a8_for_aiter(packed)
    assert repacked.tolist() == [[0x04, 0x15, 0x26, 0x37]]


def test_int4_converter_rejects_unpacked_storage() -> None:
    with pytest.raises(ValueError, match="int32"):
        convert_compressed_tensors_int4_to_slimquant(torch.zeros(2, 4, dtype=torch.int8))


def test_glm53_quant_config_selects_mixed_w4a8() -> None:
    config = SimpleNamespace(model_type="glm_moe_dsa", architectures=["GlmMoeDsaForCausalLM"])
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        _quant_config(), None, config
    ) == "glm_moe_dsa_w4a8"
    assert is_glm_channel_int4_w4a8(_quant_config())
    int8_only = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {"group_int8": _quant_config()["config_groups"]["group_int8"]},
    }
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        int8_only, None, config
    ) is None
    other = SimpleNamespace(model_type="deepseek_v3", architectures=["DeepseekV3ForCausalLM"])
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        _quant_config(), None, other
    ) is None


def test_shared_indexer_follows_glm53_pattern_and_keeps_mtp() -> None:
    indexer_types = ["full", "full", "full"] + ["shared"] * 3
    indexer_types += ["full", "shared", "shared", "shared"] * 18
    assert len(indexer_types) == 78
    assert indexer_types.count("full") == 21
    config = SimpleNamespace(num_hidden_layers=78, indexer_types=indexer_types)
    assert [layer for layer in range(78) if not is_shared_indexer_layer(config, layer)] == [
        0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
    ]
    assert is_shared_indexer_layer(config, 3)
    assert not is_shared_indexer_layer(config, 78)
    assert not is_shared_indexer_layer(SimpleNamespace(index_topk=2048), 3)


def test_expert_checkpoint_names_drop_shape_and_rebind_packed_weights() -> None:
    packed = "model.layers.78.mlp.experts.0.gate_proj.weight_packed"
    assert normalize_routed_expert_checkpoint_name(packed, True) == (
        "model.layers.78.mlp.experts.0.gate_proj.weight"
    )
    assert normalize_routed_expert_checkpoint_name(packed, False) == packed
    assert normalize_routed_expert_checkpoint_name(
        "model.layers.3.mlp.experts.0.down_proj.weight_shape", True
    ) is None
    assert normalize_routed_expert_checkpoint_name(
        "model.layers.0.mlp.down_proj.weight", True
    ) == "model.layers.0.mlp.down_proj.weight"
