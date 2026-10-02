"""Kimi-K3 layer-name adapters for the Kunlun compressed-tensors schemes.

What this module is, and what it deliberately is not
---------------------------------------------------
``/home/models/Kimi-K3-MoE-W4`` is a compressed-tensors checkpoint with
``format="pack-quantized"``, ``num_bits=4``, ``type="int"``,
``strategy="channel"`` (``group_size=null``), ``symmetric=true``. Only the 896
routed experts' ``w1/w2/w3`` are quantized; every other linear stays BF16.

The *implementation* of that scheme is
:class:`~sglang_kunlun.hooks.layers.quantization.compressed_tensors.schemes.KunlunCompressedTensorsWNA16MoE`
(``schemes/compressed_tensors_wna16_moe.py:29``, ``apply_weights`` at ``:245``,
backed by ``kunlun_ops.moe_fc_v3``). That is the DeepSeek-V4 path and it is
strictly more complete than the Kimi-K3 branch's own
``KunlunCompressedTensorsW4A16MoE``, whose ``apply_weights`` was a
dequantize-per-expert reference that its docstring described as "fails loudly".
Nothing here reimplements weight creation, post-load processing, or the GEMM.

This module contributes only the two things the Kimi-K3 branch had that the
DeepSeek-V4 path does not:

1. K3 leaves its non-expert linears out of ``config_groups.targets`` entirely,
   so ``find_matched_target`` raises instead of returning "unquantized". Turn
   that into ``None`` (-> ``UnquantizedLinearMethod``) for K3 decoder linears
   only.
2. K3 stores expert projections as ``experts.<i>.w1/w3/w2`` while the fused-MoE
   scheme lookup probes ``experts.0.gate_proj/up_proj/down_proj``. Map between
   them for the duration of the scheme lookup.

Known layout divergence -- verify numerically before trusting any output
-----------------------------------------------------------------------
``KunlunCompressedTensorsWNA16MoE.process_weights_after_loading``
(``schemes/compressed_tensors_wna16_moe.py:199-243``) applies the DeepSeek-V4
convention::

    packed = w.transpose(1, 2).contiguous().view(torch.int8)
    packed.bitwise_xor_(0x88)                       # unsigned -> signed nibbles
    scale = scale_to_kernel_max(scale.T, INT4_SIGNED_SCALE_MULT)   # absmax, x7

The Kimi-K3 branch's own scheme
(``compressed_tensors_w4a16_moe.py:95-136`` on branch ``kimi-k3``) only
transposed and kept the raw int32 scales. If the K3 checkpoint's
``weight_packed`` / ``weight_scale`` do not already follow the DeepSeek-V4
convention, ``moe_fc_v3`` produces **silently wrong** numbers -- no exception,
no shape error. This is the #1 item to verify numerically, per-expert, against
``sglang_kunlun/kernels/w4a16_moe.py`` (``w4a16_linear``) as a float64 oracle.
That oracle is intentionally *not* wired into the runtime path.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any, Optional

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

__all__ = [
    "get_linear_scheme_kunlun",
    "get_moe_scheme_kunlun",
    "kimi_k3_expert_scheme_dict",
]

_CT_CONFIG = (
    "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors."
    "CompressedTensorsConfig"
)

_K3_EXPERTS_PATTERN = re.compile(r"^model\.layers\.\d+\.mlp\.experts$")
# Fused-MoE probe suffix -> Kimi-K3 checkpoint suffix.
_K3_PROJECTION_MAP = {
    ".0.gate_proj": ".0.w1",
    ".0.up_proj": ".0.w3",
    ".0.down_proj": ".0.w2",
}

_MISSING = object()


def is_kimi_k3_moe_layer_name(layer_name: Any) -> bool:
    """Whether ``layer_name`` is a Kimi-K3 routed-expert container."""
    return isinstance(layer_name, str) and bool(
        _K3_EXPERTS_PATTERN.match(layer_name)
    )


@contextlib.contextmanager
def kimi_k3_expert_scheme_dict(config: Any, layer_name: Any):
    """Map K3 fused expert probe names onto the checkpoint's ``w1/w3/w2`` names.

    Both ``CompressedTensorsConfig.get_moe_scheme``
    (``srt/layers/quantization/compressed_tensors/compressed_tensors.py:800-807``)
    and :meth:`KunlunCompressedTensorsConfig.get_moe_scheme` resolve an expert
    scheme by probing ``<experts>.0.gate_proj`` / ``.0.up_proj`` /
    ``.0.down_proj`` through ``self.get_scheme_dict``. Kimi-K3's
    ``config_groups.targets`` name ``.0.w1`` / ``.0.w3`` / ``.0.w2`` instead, so
    without this remap all three probes raise "Unable to find matching target".

    A no-op for every layer name that is not a K3 expert container, so it is
    safe to wrap unconditionally.
    """
    if not is_kimi_k3_moe_layer_name(layer_name):
        yield
        return

    original = config.get_scheme_dict
    previous = config.__dict__.get("get_scheme_dict", _MISSING)

    def mapped_get_scheme_dict(layer, name=None, matched_target=None):
        """Resolve a probe name through the Kimi-K3 checkpoint naming."""
        if isinstance(name, str):
            for probe_suffix, checkpoint_suffix in _K3_PROJECTION_MAP.items():
                if name == layer_name + probe_suffix:
                    return original(
                        layer, layer_name + checkpoint_suffix, matched_target
                    )
        return original(layer, name, matched_target)

    config.get_scheme_dict = mapped_get_scheme_dict
    try:
        yield
    finally:
        if previous is _MISSING:
            config.__dict__.pop("get_scheme_dict", None)
        else:
            config.get_scheme_dict = previous


@plugin_hook(f"{_CT_CONFIG}.get_linear_scheme", type=HookType.AROUND)
def get_linear_scheme_kunlun(
    original_fn,
    self,
    layer,
    layer_name: Optional[str] = None,
    matched_target: Optional[str] = None,
):
    """Keep unmatched Kimi-K3 non-expert linears in BF16.

    ``matched_target`` is new in 0.5.19: upstream's signature is
    ``get_linear_scheme(self, layer, layer_name=None, matched_target=None)``
    (``compressed_tensors.py:906-911``) and the new ``get_lm_head_scheme``
    (``:992``) calls it with three keywords at ``:1038``. Declaring and
    forwarding it is mandatory -- otherwise the lm_head resolution
    ``TypeError``s.

    Args:
        original_fn: Upstream linear-scheme selector.
        self: Compressed-tensors configuration instance.
        layer: Linear layer being configured.
        layer_name: Fully qualified model layer name.
        matched_target: Target already resolved by a suffix-aware caller.

    Returns:
        The upstream scheme, or ``None`` for unmatched K3 non-expert linears.

    Raises:
        ValueError: Re-raised for anything outside the narrow K3 case.
    """
    try:
        return original_fn(
            self, layer, layer_name=layer_name, matched_target=matched_target
        )
    except ValueError as exc:
        is_k3_non_expert = (
            getattr(self, "quant_format", None) == "pack-quantized"
            and isinstance(layer_name, str)
            and layer_name.startswith("model.layers.")
            and ".mlp.experts" not in layer_name
        )
        if is_k3_non_expert and "Unable to find matching target" in str(exc):
            logger.debug("Kimi-K3 leaves %s unquantized", layer_name)
            return None
        raise


@plugin_hook(f"{_CT_CONFIG}.get_moe_scheme", type=HookType.AROUND)
def get_moe_scheme_kunlun(original_fn, self, layer, layer_name: Optional[str] = None):
    """Resolve Kimi-K3 expert schemes under the checkpoint's projection names.

    Args:
        original_fn: Upstream MoE-scheme selector.
        self: Compressed-tensors configuration instance.
        layer: Fused MoE layer being configured.
        layer_name: Fully qualified model layer name.

    Returns:
        Whatever the upstream selector returns for the remapped names.
    """
    with kimi_k3_expert_scheme_dict(self, layer_name):
        return original_fn(self, layer, layer_name)