# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM-5.3 Channel-INT4 的打包转换、AITER gemm2 布局和量化识别。"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
    convert_compressed_tensors_int4_to_slimquant,
    is_glm_channel_int4_w4a8,
    normalize_routed_expert_checkpoint_name,
    repack_and_shuffle_w4a8_gemm2,
    repack_slimquant_w4a8_for_aiter,
)
from vllm_hcu.model_executor.layers.quantization.slimquant_facade import (
    GlmMoeDsaW4A8Facade,
)


def _pack_compressed_tensors_int4(values: torch.Tensor) -> torch.Tensor:
    """按 compressed-tensors 的 uint4b8（数值加 8）、低位优先规则打进 int32。"""
    if values.shape[-1] % 8:
        raise AssertionError("K 必须是 8 的倍数")
    nibbles = ((values.to(torch.int32) + 8) & 0xF).reshape(*values.shape[:-1], -1, 8)
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


def _glm_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="glm_moe_dsa", architectures=["GlmMoeDsaForCausalLM"]
    )


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


def test_int4_converter_rejects_unpacked_storage() -> None:
    with pytest.raises(ValueError, match="int32"):
        convert_compressed_tensors_int4_to_slimquant(torch.zeros(2, 4, dtype=torch.int8))


def test_aiter_w4a8_repack_regroups_eight_nibbles() -> None:
    # SlimQuant bytes encode [0,1], [2,3], [4,5], [6,7].
    packed = torch.tensor(
        [[0x01, 0x23, 0x45, 0x67]], dtype=torch.uint8
    ).view(torch.int8)
    repacked = repack_slimquant_w4a8_for_aiter(packed)
    assert repacked.tolist() == [[0x04, 0x15, 0x26, 0x37]]


def test_gemm2_repack_shuffles_each_regrouped_expert() -> None:
    weight = torch.arange(2 * 3 * 4, dtype=torch.uint8).reshape(2, 3, 4)
    weight = weight.view(torch.int8)
    seen = []

    def fake_shuffle(expert: torch.Tensor) -> torch.Tensor:
        seen.append((expert.dtype, tuple(expert.shape)))
        return expert.flip(0)

    result = repack_and_shuffle_w4a8_gemm2(weight, shuffle_fn=fake_shuffle)

    regrouped = repack_slimquant_w4a8_for_aiter(weight).view(torch.uint8)
    expected = torch.stack([regrouped[index].flip(0) for index in range(2)])
    assert seen == [(torch.uint8, (3, 4)), (torch.uint8, (3, 4))]
    assert result.dtype == torch.int8
    assert torch.equal(result.view(torch.uint8), expected)


def test_gemm2_repack_rejects_non_expert_rank() -> None:
    with pytest.raises(ValueError, match=r"\[E, N, K/2\]"):
        repack_and_shuffle_w4a8_gemm2(
            torch.zeros(3, 4, dtype=torch.int8), shuffle_fn=lambda tensor: tensor
        )


def test_expert_loaders_convert_weights_and_scales() -> None:
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        _wrap_checkpoint_loader,
    )

    loaded = []

    def record(_param, weight, *args, **kwargs):
        loaded.append(weight)

    scale_param = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
    scale_param.weight_loader = record
    _wrap_checkpoint_loader(scale_param, scale=True)
    scale = torch.tensor([[0.5], [0.25]], dtype=torch.bfloat16)
    scale_param.weight_loader(scale_param, scale, "w13_weight_scale", shard_id="w1")
    assert loaded[0].dtype == torch.float32
    assert torch.equal(loaded[0], scale.float() / 16.0)

    weight_param = torch.nn.Parameter(
        torch.zeros(2, 4, dtype=torch.int8), requires_grad=False
    )
    weight_param.weight_loader = record
    _wrap_checkpoint_loader(weight_param, scale=False)
    values = torch.tensor(
        [[1, -1, 7, -8, 0, 2, 3, 4], [-7, 5, -3, 6, 1, 1, -8, 7]],
        dtype=torch.int8,
    )
    weight_param.weight_loader(weight_param, _pack_compressed_tensors_int4(values))
    assert loaded[1].dtype == torch.int8
    assert torch.equal(_unpack_slimquant_int4(loaded[1]), values)


def test_glm53_quant_config_selects_mixed_w4a8() -> None:
    config = _glm_hf_config()
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        _quant_config(), None, config
    ) == "glm_moe_dsa_w4a8"
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        _quant_config(), "glm_moe_dsa_w4a8", config
    ) == "glm_moe_dsa_w4a8"
    assert is_glm_channel_int4_w4a8(_quant_config())
    # An explicit different -q keeps its own resolution path.
    assert GlmMoeDsaW4A8Facade.override_quantization_method(
        _quant_config(), "slimquant_marlin", config
    ) is None
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


def test_expert_checkpoint_names_drop_shape_and_rebind_packed_weights() -> None:
    packed = "model.layers.78.mlp.experts.0.gate_proj.weight_packed"
    shape = "model.layers.3.mlp.experts.0.down_proj.weight_shape"
    assert normalize_routed_expert_checkpoint_name(packed, True) == (
        "model.layers.78.mlp.experts.0.gate_proj.weight"
    )
    assert normalize_routed_expert_checkpoint_name(shape, True) is None
    # Packed-named expert parameters (compressed-tensors W4A16) keep both names.
    assert normalize_routed_expert_checkpoint_name(packed, False) == packed
    assert normalize_routed_expert_checkpoint_name(shape, False) == shape
    assert normalize_routed_expert_checkpoint_name(
        "model.layers.0.mlp.down_proj.weight", True
    ) == "model.layers.0.mlp.down_proj.weight"


def _glm_aiter_method(moe_backend: str):
    from vllm_hcu.model_executor.layers.quantization.glm_moe_dsa_w4a8 import (
        GlmMoeDsaW4A8MoEMethod,
    )

    method = GlmMoeDsaW4A8MoEMethod.__new__(GlmMoeDsaW4A8MoEMethod)
    method.moe = SimpleNamespace(moe_backend=moe_backend)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(16, dtype=torch.int8).reshape(1, 4, 4), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(8, dtype=torch.int8).reshape(1, 4, 2), requires_grad=False
    )
    return method, layer


def _patch_glm_post_load(monkeypatch, config):
    from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime
    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime,
        glm_moe_dsa_w4a8,
        slimquant_w4a8,
    )

    calls = {"repack": 0, "parent": []}

    def fake_repack(weight):
        calls["repack"] += 1
        return weight + 1

    def fake_parent(self, layer):
        calls["parent"].append(
            (
                bool(getattr(config, "need_shuffle", False)),
                bool(getattr(config, "need_shuffle_scale", False)),
            )
        )

    monkeypatch.setattr(
        deepep_runtime, "slimquant_w4a8_uses_deepep_auto", lambda moe: False
    )
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_w4a8_moe",
        lambda method, layer: config,
    )
    monkeypatch.setattr(glm_moe_dsa_w4a8, "repack_and_shuffle_w4a8_gemm2", fake_repack)
    monkeypatch.setattr(
        slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod,
        "process_weights_after_loading",
        fake_parent,
    )
    return calls


def test_glm_aiter_post_load_installs_gemm2_layout_once(monkeypatch) -> None:
    config = SimpleNamespace(need_shuffle=True, need_shuffle_scale=True)
    calls = _patch_glm_post_load(monkeypatch, config)
    method, layer = _glm_aiter_method("auto")
    original_w13 = layer.w13_weight.detach().clone()

    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)

    assert calls["repack"] == 2  # w13 and w2, only on the first call
    assert layer._glm_w4a8_gemm2_shuffled is True
    assert torch.equal(layer.w13_weight, original_w13 + 1)
    # The parent registers the gemm2 layout without generic AITER shuffles.
    assert calls["parent"] == [(False, False), (False, False)]


def test_glm_triton_post_load_keeps_slimquant_packing(monkeypatch) -> None:
    config = SimpleNamespace(need_shuffle=True, need_shuffle_scale=True)
    calls = _patch_glm_post_load(monkeypatch, config)
    method, layer = _glm_aiter_method("triton")
    original_w13 = layer.w13_weight.detach().clone()

    method.process_weights_after_loading(layer)

    assert calls["repack"] == 0
    assert not hasattr(layer, "_glm_w4a8_gemm2_shuffled")
    assert torch.equal(layer.w13_weight, original_w13)
    assert calls["parent"] == [(True, True)]


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _run_apply_with_shuffling_config(monkeypatch, *, gemm2_installed: bool):
    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime,
    )

    calls: dict[str, object] = {}

    class MoeQuantType:
        W4A8 = "w4a8"

    def get_config(**kwargs: object):
        # Larger M selects a config that asks for the generic AITER shuffle.
        return True, SimpleNamespace(
            quant_type="w4a8",
            solution_type="moe_c",
            need_shuffle=True,
            need_shuffle_scale=False,
        )

    def aiter_moe(**kwargs: object):
        calls["execute"] = kwargs
        return torch.zeros((3, 4), dtype=kwargs["output_dtype"])

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    w1 = torch.zeros((2, 8, 2), dtype=torch.int8)
    w2 = torch.zeros((2, 4, 2), dtype=torch.int8)
    layer = SimpleNamespace(
        w13_weight=w1,
        w2_weight=w2,
        swiglu_limit=None,
        global_num_experts=2,
        _expert_map=None,
        expert_mask=None,
    )
    # Load time installed the gemm2 layout from an M=1 config without shuffle.
    compressed_tensors_moe_runtime.install_aiter_moe_weight_layout(
        layer,
        SimpleNamespace(
            quant_type="w4a8",
            solution_type="moe_c",
            need_shuffle=False,
            need_shuffle_scale=False,
        ),
        logical_shape=(2, 8, 4, 4),
    )
    if gemm2_installed:
        layer._glm_w4a8_gemm2_shuffled = True
    method = SimpleNamespace(
        moe=SimpleNamespace(num_experts=2),
        moe_quant_config=SimpleNamespace(
            w1_scale=torch.ones((2, 8, 1)),
            w2_scale=torch.ones((2, 4, 1)),
            a1_scale=None,
            a2_scale=None,
        ),
    )
    compressed_tensors_moe_runtime.apply_aiter_w4a8_moe(
        method,
        layer,
        torch.ones((3, 4), dtype=torch.bfloat16),
        torch.ones((3, 2), dtype=torch.float32),
        torch.zeros((3, 2), dtype=torch.int32),
    )
    return calls


def test_glm_gemm2_layout_survives_m_dependent_aiter_configs(monkeypatch) -> None:
    calls = _run_apply_with_shuffling_config(monkeypatch, gemm2_installed=True)
    assert calls["execute"]["use_weight_shuffle"] is False


def test_unmarked_layout_still_rejects_shuffling_config(monkeypatch) -> None:
    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
        HcuCompressedTensorsMoeError,
    )

    with pytest.raises(
        HcuCompressedTensorsMoeError,
        match="AITER config layout does not match installed weights",
    ):
        _run_apply_with_shuffling_config(monkeypatch, gemm2_installed=False)
