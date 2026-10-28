# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Mixed quantization for GLM MoE DSA: channel INT4 experts and channel INT8 linears.

Routed experts in the checkpoint, including MTP, are compressed-tensors
pack-quantized INT4 with BF16 per-channel scales and dynamic per-token INT8
activations. Other linears stay on compressed-tensors channel INT8. The
published INT4 regex can miss MTP layers, so every RoutedExperts module is
loaded as INT4 when the config as a whole is this mixed scheme.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

from vllm_hcu.model_executor.layers.quantization.glm_channel_int4 import (
    convert_compressed_tensors_int4_to_slimquant,
    is_glm_channel_int4_w4a8,
    is_glm_moe_dsa_model,
    repack_and_shuffle_w4a8_gemm2,
)
from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
    SlimQuantW4A8Int8AiterMoEMethod,
)


def _disable_generic_w4a8_shuffle(config: object) -> None:
    """gemm2 is already the final layout, so skip the generic weight/scale shuffle."""
    if getattr(config, "need_shuffle", False):
        config.need_shuffle = False
    if getattr(config, "need_shuffle_scale", False):
        config.need_shuffle_scale = False


def _wrap_checkpoint_loader(param: torch.nn.Parameter, *, scale: bool) -> None:
    """Convert packing and rescale before the original expert loader runs.

    compressed-tensors scales match the real signed INT4 values. HIPC/AITER
    read the high nibble as 16 times that value, so the SlimQuant canonical
    scale is the checkpoint scale divided by 16. The Triton fallback
    multiplies by 16 again at the call site.
    """
    original = param.weight_loader

    def loader(loaded_param, loaded_weight, *args, **kwargs):
        if scale:
            loaded_weight = loaded_weight.to(dtype=loaded_param.dtype) / 16.0
        elif loaded_weight.dtype == torch.int32:
            loaded_weight = convert_compressed_tensors_int4_to_slimquant(
                loaded_weight
            )
        return original(loaded_param, loaded_weight, *args, **kwargs)

    param.weight_loader = loader


class GlmMoeDsaW4A8MoEMethod(SlimQuantW4A8Int8AiterMoEMethod):
    """Load packed INT4 experts and keep computation on SlimQuant W4A8."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        for name in ("w13_weight", "w2_weight"):
            _wrap_checkpoint_loader(getattr(layer, name), scale=False)
        for name in ("w13_weight_scale", "w2_weight_scale"):
            _wrap_checkpoint_loader(getattr(layer, name), scale=True)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """Fold experts into the AITER W4A8 gemm2 layout and skip the generic shuffle.

        An 8-nibble regroup alone is still SlimQuant's pairwise byte order.
        AITER moe_c expects that regroup followed by
        ``w4a8_moe_layout_shuffle_gemm2``. When ``need_shuffle`` is set, the
        parent also calls ``aiter_moe_shfl_weight``, which is a different
        layout; stacking the two still yields garbage on a single run.
        Triton (BoltOps) and DeepEP keep the unshuffled SlimQuant packing.
        """
        backend = getattr(self.moe, "moe_backend", "auto")
        from vllm_hcu.model_executor.layers.fused_moe.deepep_runtime import (
            slimquant_w4a8_uses_deepep_auto,
        )

        if backend == "triton" or slimquant_w4a8_uses_deepep_auto(self.moe):
            super().process_weights_after_loading(layer)
            return

        from vllm_hcu.model_executor.layers.quantization.compressed_tensors_moe_runtime import (
            prewarm_aiter_w4a8_moe,
        )

        aiter_config = prewarm_aiter_w4a8_moe(self, layer)
        if aiter_config is None:
            super().process_weights_after_loading(layer)
            return

        if not getattr(layer, "_glm_w4a8_gemm2_shuffled", False):
            for name in ("w13_weight", "w2_weight"):
                parameter = getattr(layer, name)
                parameter.data.copy_(repack_and_shuffle_w4a8_gemm2(parameter.data))
            layer._glm_w4a8_gemm2_shuffled = True

        # Prewarm returns the M=1 config. Clear the generic shuffle so the
        # parent registers the gemm2 layout and does not call
        # aiter_moe_shfl_weight.
        _disable_generic_w4a8_shuffle(aiter_config)
        super().process_weights_after_loading(layer)


class GlmMoeDsaChannelW4A8Config(CompressedTensorsConfig):
    """GLM MoE DSA mixed quant: INT4 experts, original INT8 for other linears."""

    @classmethod
    def get_name(cls) -> str:
        return "glm_moe_dsa_w4a8"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GlmMoeDsaChannelW4A8Config":
        parsed = super().from_config(config)
        if type(parsed) is cls:
            return parsed
        # If the parent hardcodes the base class, copy parsed fields onto this subclass.
        instance = cls.__new__(cls)
        instance.__dict__.update(parsed.__dict__)
        return instance

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        if user_quant not in (None, cls.get_name()):
            return None
        if not is_glm_moe_dsa_model(hf_config):
            return None
        if not is_glm_channel_int4_w4a8(hf_quant_cfg):
            return None
        return cls.get_name()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, RoutedExperts):
            return GlmMoeDsaW4A8MoEMethod(self, layer.moe_config)
        return super().get_quant_method(layer, prefix)
