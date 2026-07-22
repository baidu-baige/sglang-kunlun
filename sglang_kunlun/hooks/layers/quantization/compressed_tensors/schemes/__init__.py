"""Kunlun compressed-tensors quantization schemes."""

from ..utils import (
    INT4_SIGNED_SCALE_MULT,
    INT4_UNSIGNED_SCALE_MULT,
    INT8_SCALE_MULT,
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

from .compressed_tensors_w8a8_int8 import KunlunCompressedTensorsW8A8Int8
from .compressed_tensors_w8a8_int8_moe import KunlunCompressedTensorsW8A8Int8MoE


__all__ = [
    "INT4_SIGNED_SCALE_MULT",
    "INT4_UNSIGNED_SCALE_MULT",
    "INT8_SCALE_MULT",
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
