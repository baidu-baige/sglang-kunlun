"""Kunlun compressed-tensors quantization config and methods."""

from __future__ import annotations

from typing import Any, Dict, Optional, cast
import logging

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationType,
)

from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsFusedMoEMethod,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    WNA16_SUPPORTED_BITS,
    CompressedTensorsW8A8Int8,
    CompressedTensorsWNA16,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    is_activation_quantization_format,
)

from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
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
    KunlunCompressedTensorsW4A8Int8MoE,
    KunlunCompressedTensorsWNA16MoE,
    check_standard_routed_moe_layer,
    dequant_int4,
    dequant_int4_moe,
    dynamic_quantize_int8,
    moe_fc,
    moe_post,
    moe_pre_sorted,
    scale_to_kernel_max,
    validate_int4_moe_shapes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "INT4_SIGNED_SCALE_MULT",
    "INT4_UNSIGNED_SCALE_MULT",
    "INT8_SCALE_MULT",
    "KunlunCompressedTensorsConfig",
    "KunlunCompressedTensorsW8A8Int8",
    "KunlunCompressedTensorsW8A8Int8MoE",
    "KunlunCompressedTensorsW4A8Int8MoE",
    "KunlunCompressedTensorsWNA16MoE",
    "check_standard_routed_moe_layer",
    "dequant_int4",
    "dequant_int4_moe",
    "dynamic_quantize_int8",
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

    @classmethod
    def _quantization_scheme_map_from_config(cls, config: Dict[str, Any]):
        """Record each config group's own ``format`` in the target scheme map.

        DSv4 INT4 checkpoints declare a ``format`` on every ``config_groups``
        entry (``pack-quantized`` for W4A16 experts, ``int-quantized`` for
        W8A8 dense linears) instead of relying on the single top-level
        ``quantization_config.format``. Upstream only reads the top-level
        one, so one of the two groups always resolves to the wrong scheme.
        """
        target_scheme_map = super()._quantization_scheme_map_from_config(config)
        top_format = cast(str, config.get("format"))
        for quant_config in config.get("config_groups", dict()).values():
            group_format = cast(str, quant_config.get("format", top_format))
            for target in quant_config.get("targets") or []:
                scheme = target_scheme_map.get(target)
                if scheme is None:
                    continue
                scheme["format"] = group_format
                input_activations = quant_config.get("input_activations")
                # Upstream only parses activations against the top-level
                # format. Packed-INT4 W4A8 groups keep ``pack-quantized``
                # weights but still declare dynamic INT8 input_activations.
                if input_activations is not None and scheme.get(
                    "input_activations"
                ) is None:
                    scheme["input_activations"] = QuantizationArgs.model_validate(
                        input_activations
                    )
                elif group_format != top_format:
                    scheme["input_activations"] = (
                        QuantizationArgs.model_validate(input_activations)
                        if input_activations
                        and is_activation_quantization_format(group_format)
                        else None
                    )
        return target_scheme_map

    def _add_fused_moe_to_target_scheme_map(self) -> None:
        """Keep Linear as a fallback FusedMoE target for W8A8-only checkpoints.

        Mixed INT4 checkpoints still match expert projections by layer name
        before the class-name fallback, so this alias does not steal the
        pack-quantized expert scheme.
        """
        super()._add_fused_moe_to_target_scheme_map()
        if "FusedMoE" in self.target_scheme_map and "DeepEPMoE" not in self.target_scheme_map:
            self.target_scheme_map["DeepEPMoE"] = self.target_scheme_map["FusedMoE"]

    def get_linear_scheme(
        self, layer: torch.nn.Module, layer_name: Optional[str] = None
    ):
        """Resolve the linear scheme with the format of the matched group."""
        scheme_dict = self.get_scheme_dict(layer, layer_name)
        group_format = scheme_dict.get("format") if scheme_dict else None
        if not group_format or group_format == self.quant_format:
            return super().get_linear_scheme(layer=layer, layer_name=layer_name)

        saved_format, self.quant_format = self.quant_format, group_format
        try:
            return super().get_linear_scheme(layer=layer, layer_name=layer_name)
        finally:
            self.quant_format = saved_format

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        """Select the Kunlun quantization method for a layer."""
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

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

        for alias in ("FusedMoE", "DeepEPMoE"):
            self.target_scheme_map[alias] = scheme_dict

        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")

        if (
            input_quant is not None
            and self._is_dynamic_token_w4a8(weight_quant, input_quant)
            and weight_quant.type == QuantizationType.INT
            and input_quant.type == QuantizationType.INT
        ):
            return KunlunCompressedTensorsW4A8Int8MoE(
                scheme_dict,
                quant_format=scheme_dict.get("format", self.quant_format),
            )
        if (
            self._is_wNa16_group_channel(weight_quant, input_quant)
            and weight_quant.num_bits in WNA16_SUPPORTED_BITS
            and weight_quant.symmetric
        ):
            return KunlunCompressedTensorsWNA16MoE(
                scheme_dict,
                quant_format=scheme_dict.get("format", self.quant_format),
            )
        if (
            input_quant is not None
            and self._is_dynamic_token_w8a8(weight_quant, input_quant)
            and weight_quant.type == QuantizationType.INT
            and input_quant.type == QuantizationType.INT
        ):
            return KunlunCompressedTensorsW8A8Int8MoE(weight_quant, input_quant)

        return super().get_moe_scheme(layer=layer, layer_name=layer_name)
