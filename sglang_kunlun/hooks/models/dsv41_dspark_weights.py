"""W8A8-INT8 weight handling for the DSpark draft (MTP) loader.

`DeepseekV4ForCausalLMDSpark.load_weights` runs the same FP8-only
`_dequant_fp8_wo_a_streaming` transform as the main model, but the draft builds
`wq_a` / `wkv` / `wo_a` unquantized (`DSparkAttention` passes
`wo_a_fp8=False, wo_a_keeps_quant_config=False`, and `wq_a`/`wkv` end up without a
`weight_scale` parameter). The W8A8-INT8 checkpoint therefore leaves their
`weight_scale` tensors unmatched --

    DSpark V4 draft: unexpected weight 'mtp.0.attn.wq_a.weight_scale'
                      -> 'stages.0.self_attn.wq_a.weight_scale'

-- and the int8 weights would be used with no scale at all, the same defect class
as `wo_a` in the main model.

The pair is decided from the **scale** name only. Using the weight name is wrong:
`stages.N.self_attn.wq_a.weight` exists as a parameter (a bf16 one), so a
weight-based test calls the pair "matched", passes the weight through, and the
scale is then flushed as an orphan -- which is exactly the 72-warning symptom.

The draft loader is fed the whole checkpoint stream, so only `mtp.*` names are
considered here; the main model's own loader handles its tensors.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Optional, Tuple

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_INT8_ATTN = re.compile(r"\.attn\.(wq_a|wkv|wo_a)\.(weight|weight_scale)$")


def _needs_dequant(prefix: str, params_dict: dict, remap: Callable) -> bool:
    """Whether this `(weight, weight_scale)` pair has no quantized scale param.

    Decided from the companion `weight_scale` name, which we can compute before
    the tensor arrives; the weight param existing by itself says nothing (it is
    the bf16 placeholder the pair must fill in).
    """
    scale_name = prefix + ".weight_scale"
    return (remap(scale_name) or scale_name) not in params_dict


def dequantize_unmatched_int8_attn(
    weights: Iterable[Tuple[str, torch.Tensor]],
    params_dict: dict,
    remap: Callable[[str], Optional[str]],
    stats: Optional[dict] = None,
):
    """bf16-dequantize int8 `(weight, weight_scale)` pairs of the draft's
    unquantized attention projections; pass every other tensor through unchanged
    and in arrival order."""
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        if stats is not None:
            stats["seen"] = stats.get("seen", 0) + 1
        m = _INT8_ATTN.search(name)
        if m is None or not name.startswith("mtp."):
            yield name, tensor
            continue
        prefix, kind = name.rsplit(".", 1)
        if not _needs_dequant(prefix, params_dict, remap):
            # The draft really has a quantized scale parameter: leave the pair to
            # the quantization scheme's loader.
            if stats is not None:
                stats["matched"] = stats.get("matched", 0) + 1
            yield name, tensor
            continue
        bucket = pending.setdefault(prefix, {})
        bucket[kind] = tensor
        if "weight" in bucket and "weight_scale" in bucket:
            pending.pop(prefix, None)
            weight = bucket["weight"]
            scale = bucket["weight_scale"]
            if stats is not None:
                stats["dequant"] = stats.get("dequant", 0) + 1
            yield prefix + ".weight", (
                weight.to(torch.float32) * scale.to(torch.float32)
            ).to(torch.bfloat16)
    # A weight whose scale never arrived passes through (the scale is dropped with
    # it only if the checkpoint genuinely has none).
    for prefix, bucket in pending.items():
        for kind, tensor in bucket.items():
            yield prefix + "." + kind, tensor


@plugin_hook(
    "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.load_weights",
    type=HookType.AROUND,
)
def load_weights_around_kunlun(original_fn, self, weights):
    """load_weights_around_kunlun"""
    import os

    from sglang_kunlun.hooks.speculative.dspark_precision import (
        _strip_correction_blocks,
    )

    params_dict = dict(self.named_parameters())

    # Parameters registered on the draft include the bf16 placeholders for the
    # pairs below, so the scale's absence -- not the weight's presence -- is the
    # signal that a dequant is required.
    def remap(name: str) -> Optional[str]:
        mapped = self._remap_dspark_weight_name(name)
        if mapped is not None:
            return mapped
        # `_remap_dspark_weight_name` may not recognise the `weight_scale` suffix;
        # apply the same mtp.<i>.attn. -> stages.<i>.self_attn. mapping by hand so
        # the lookup still targets the right parameter.
        return re.sub(
            r"^mtp\.(\d+)\.attn\.", r"stages.\1.self_attn.", name
        )

    stats: Optional[dict] = (
        {} if os.environ.get("DSV4_KUNLUN_DIAG_DSPARK_W") == "1" else None
    )
    weights = _strip_correction_blocks(self, weights)
    result = original_fn(self, dequantize_unmatched_int8_attn(weights, params_dict, remap, stats))
    if stats is not None:
        print(
            "[diag-dspark-w] "
            f"seen={stats.get('seen', 0)} attn_pairs_dequant={stats.get('dequant', 0)} "
            f"attn_matched_passthrough={stats.get('matched', 0)}",
            flush=True,
        )
    return result
