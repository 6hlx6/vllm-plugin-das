# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GLM MoE DSA 的 channel INT4 expert 与 channel INT8 Linear 混合量化。

checkpoint 里路由 expert（含 MTP）是 compressed-tensors pack-quantized
INT4，scale 为 BF16 per-channel，激活动态 per-token INT8。其余 Linear 仍走
compressed-tensors 的 channel INT8。发布配置的 INT4 正则可能漏掉 MTP 层，
因此只要整份配置属于这种混合方案，所有 RoutedExperts 都按 INT4 加载。
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
)
from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
    SlimQuantW4A8Int8AiterMoEMethod,
)


def _wrap_checkpoint_loader(param: torch.nn.Parameter, *, scale: bool) -> None:
    """在原 expert loader 之前完成打包转换和 scale 类型对齐。"""
    original = param.weight_loader

    def loader(loaded_param, loaded_weight, *args, **kwargs):
        if scale:
            if loaded_weight.dtype != loaded_param.dtype:
                loaded_weight = loaded_weight.to(dtype=loaded_param.dtype)
        elif loaded_weight.dtype == torch.int32:
            loaded_weight = convert_compressed_tensors_int4_to_slimquant(
                loaded_weight
            )
        return original(loaded_param, loaded_weight, *args, **kwargs)

    param.weight_loader = loader


class GlmMoeDsaW4A8MoEMethod(SlimQuantW4A8Int8AiterMoEMethod):
    """加载 packed INT4 expert，计算仍复用 SlimQuant W4A8。"""

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
        """把 expert 收成 AITER W4A8 的 gemm2 布局，并避免再做通用 shuffle。

        只做 8-nibble 重排仍然是 SlimQuant 的逐对字节序。AITER moe_c 要的是
        这次重排之后再走 ``w4a8_moe_layout_shuffle_gemm2``。父类在
        ``need_shuffle`` 时还会调用 ``aiter_moe_shfl_weight``，那是另一套
        布局，叠加上去单跑仍然是乱码。Triton 回退继续吃未重排的打包。
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
            from vllm_hcu.model_executor.layers.quantization.kimi_k3_w4a8 import (
                repack_and_shuffle_w4a8,
            )

            for name in ("w13_weight", "w2_weight"):
                parameter = getattr(layer, name)
                parameter.data.copy_(repack_and_shuffle_w4a8(parameter.data))
            layer._glm_w4a8_gemm2_shuffled = True

        # 同一份 config 会被父类和 forward 再次取到。关掉通用 shuffle 后，
        # kernel 直接读 gemm2 布局，不再按 aiter_moe_shfl_weight 解释。
        if getattr(aiter_config, "need_shuffle", False):
            aiter_config.need_shuffle = False
        super().process_weights_after_loading(layer)


class GlmMoeDsaChannelW4A8Config(CompressedTensorsConfig):
    """GLM MoE DSA 混合量化：expert 走 INT4，其余 Linear 走原 INT8。"""

    @classmethod
    def get_name(cls) -> str:
        return "glm_moe_dsa_w4a8"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GlmMoeDsaChannelW4A8Config":
        parsed = super().from_config(config)
        if type(parsed) is cls:
            return parsed
        # 父类若写死返回基类，把已解析字段接到当前子类上。
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
