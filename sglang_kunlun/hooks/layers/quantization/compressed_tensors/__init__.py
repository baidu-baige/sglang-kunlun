"""Kunlun compressed-tensors quantization support."""

from .compressed_tensors import (
    KunlunCompressedTensorsConfig,
)

# Importing this module registers its ``@plugin_hook`` targets in
# ``HookRegistry``; the install happens via ``HookRegistry.apply_hooks()``.
from .kimi_k3 import kimi_k3_expert_scheme_dict  # noqa: F401
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
    "kimi_k3_expert_scheme_dict",
    "moe_fc",
    "moe_post",
    "moe_pre_sorted",
    "scale_to_kernel_max",
    "validate_int4_moe_shapes",
]
