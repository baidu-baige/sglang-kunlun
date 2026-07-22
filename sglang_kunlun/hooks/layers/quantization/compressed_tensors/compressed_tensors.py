"""Kunlun compressed-tensors quantization config and methods."""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsFusedMoEMethod,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
    CompressedTensorsWNA16,
)

from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)

from .schemes import (
    INT4_SIGNED_SCALE_MULT,
    INT4_UNSIGNED_SCALE_MULT,
    INT8_SCALE_MULT,
    KunlunCompressedTensorsW8A8Int8,
    KunlunCompressedTensorsW8A8Int8MoE,
    check_standard_routed_moe_layer,
    dequant_int4,
    dequant_int4_moe,
    dynamic_quantize_int8,
    get_fused_moe_scheme,
    moe_fc,
    moe_post,
    moe_pre_sorted,
    scale_to_kernel_max,
    validate_int4_moe_shapes,
)

__all__ = [
    "INT4_SIGNED_SCALE_MULT",
    "INT4_UNSIGNED_SCALE_MULT",
    "INT8_SCALE_MULT",
    "KunlunCompressedTensorsConfig",
    "KunlunCompressedTensorsW8A8Int8",
    "KunlunCompressedTensorsW8A8Int8MoE",
    "check_standard_routed_moe_layer",
    "dequant_int4",
    "dequant_int4_moe",
    "dynamic_quantize_int8",
    "get_fused_moe_scheme",
    "moe_fc",
    "moe_post",
    "moe_pre_sorted",
    "scale_to_kernel_max",
    "validate_int4_moe_shapes",
]


class KunlunCompressedTensorsConfig(CompressedTensorsConfig):
    """Compressed-tensors config with Kunlun-specific method selection."""

    @classmethod
    def get_min_capability(cls) -> int:
        """Return the minimum device capability required by the config."""
        return 0

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:
        """Report that the quantization scheme is supported on Kunlun."""
        return True

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        """Select the Kunlun quantization method for a layer."""
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, LinearBase):
            if self.linear_fp8_config is not None:
                return Fp8LinearMethod(self.linear_fp8_config)
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return UnquantizedLinearMethod()
            if isinstance(scheme, CompressedTensorsWNA16):
                raise NotImplementedError("Not supported yet")
            elif isinstance(scheme, CompressedTensorsW8A8Int8):
                scheme = KunlunCompressedTensorsW8A8Int8(
                    strategy=scheme.strategy,
                    is_static_input_scheme=scheme.is_static_input_scheme,
                    input_symmetric=scheme.input_symmetric,
                )
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)

        if isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer=layer, layer_name=prefix)
            if layer.scheme is None:
                moe_backend = get_moe_runner_backend()
                return UnquantizedFusedMoEMethod(
                    moe_backend.is_triton_kernels(),
                    moe_backend.is_flashinfer_trtllm(),
                    moe_backend.is_deep_gemm(),
                )
            return CompressedTensorsFusedMoEMethod(self)
        return None

    def get_moe_scheme(
        self, layer: torch.nn.Module, layer_name: Optional[str] = None
    ):
        """Choose Kunlun MoE schemes while reusing upstream config parsing."""
        self._add_fused_moe_to_target_scheme_map()
        layer_name = layer_name or ""
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        all_scheme_dicts = [self.get_scheme_dict(layer, name) for name in unfused_names]
        scheme_dict = all_scheme_dicts[0] if all_scheme_dicts else None

        if not all(cur_dict == scheme_dict for cur_dict in all_scheme_dicts):
            raise ValueError(
                "All MoE projections need to have same quantization scheme but found multiple"
            )
        if scheme_dict is None:
            return None

        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")

        if (
            self._is_wNa16_group_channel(weight_quant, input_quant)
            and weight_quant.num_bits == 4
            and weight_quant.symmetric
        ):
            raise NotImplementedError("Not supported yet")
        if (
            input_quant is not None
            and self._is_dynamic_token_w4a8(weight_quant, input_quant)
            and weight_quant.type == QuantizationType.INT
            and input_quant.type == QuantizationType.INT
        ):
            raise NotImplementedError("Not supported yet")
        if (
            input_quant is not None
            and self._is_dynamic_token_w8a8(weight_quant, input_quant)
            and weight_quant.type == QuantizationType.INT
            and input_quant.type == QuantizationType.INT
        ):
            return KunlunCompressedTensorsW8A8Int8MoE(self)

        return super().get_moe_scheme(layer=layer, layer_name=layer_name)
