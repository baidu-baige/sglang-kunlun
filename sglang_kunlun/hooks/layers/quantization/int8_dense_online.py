"""Online INT8 for the dense projections EB5's checkpoint leaves in BF16.

`SFT-V2-int8`'s `quantization_config` quantizes only the routed experts
(`ignore = ['re:(?!.*\\.mlp\\.experts).*']`), so every dense projection gets an
`UnquantizedLinearMethod` and runs a BF16 GEMM. On a 64K prefill trace
(REPORT_PD_ACG_20261113.md §13.4) `fc_bf16_rn_mn_kmn` alone is 26.4% of device
kernel time while INT8 GEMM is 0.01%, and BF16 roofline is 118 TFLOP/s against
409 for INT8.

So quantize those weights online at load time -- the checkpoint has no INT8
dense tensors -- and reuse the INT8 GEMM path `linear_apply_kunlun` already runs
for the routed experts (`kunlun_ops.quant2d` + `kunlun_ops.matmul`,
per-channel/per-token scales).

Tiers, by per-rank dense weight bytes read:
  t1 = MLP / shared experts / fc latent proj   (~24% of dense bytes)
  t2 = t1 + self_attn projections              (~95%; attn alone is ~71%)

`wo_a`, `lm_head`, `embed_tokens`, `indexer` and `mhc` always stay BF16: `wo_a`
because the attention absorb path reads `.weight` directly in a bmm, the rest
because they are tiny or read by kernels that expect BF16.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
from torch.nn import Parameter

logger = logging.getLogger(__name__)

_T1 = [
    "mlp.shared_experts",
    "mlp.fc1_latent_proj",
    "mlp.fc2_latent_proj",
    "mlp.gate_up_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
_T2 = _T1 + ["self_attn"]
_ALWAYS_BF16 = (
    "wo_a",
    "lm_head",
    "embed_tokens",
    "indexer",
    "mhc",
    # DSpark draft stages: with the W4A8 draft support from
    # dev-kunlun-0.5.23-ernie-5.5 (772d3ba) the draft's commit_kv_proj hands
    # quant2d a dtype it rejects ("quant2d not support x dtype"). The draft is
    # three layers, so INT8 there buys nothing measurable.
    "stages.",
    "commit_kv_proj",
)
_ALIGN = 128
_LOGGED = False


def selected_prefixes() -> List[str]:
    """Dense prefix substrings to quantize, from ``EB5_INT8_DENSE``.

    Empty/``off``/``0`` (the default) keeps all-BF16 dense. ``t1``/``t2`` are the
    tiers above; anything else is an explicit comma list of prefix substrings.
    """
    raw = os.environ.get("EB5_INT8_DENSE", "").strip().lower()
    if raw in {"", "0", "off", "false", "no"}:
        return []
    if raw in {"1", "t1", "tier1"}:
        return list(_T1)
    if raw in {"2", "t2", "tier2"}:
        return list(_T2)
    return [s.strip() for s in raw.split(",") if s.strip()]


def method_for(layer: torch.nn.Module, prefix: str):
    """``OnlineInt8LinearMethod`` for ``prefix``, or ``None`` to keep it BF16."""
    global _LOGGED
    groups = selected_prefixes()
    if not groups or not any(g in prefix for g in groups):
        return None
    if any(s in prefix for s in _ALWAYS_BF16):
        return None
    in_size = getattr(layer, "input_size", None)
    out_size = getattr(layer, "output_size", None)
    if in_size is None or out_size is None or in_size % _ALIGN or out_size % _ALIGN:
        # The INT8 GEMM wants 128-aligned dims; a mis-aligned layer is not
        # worth its own kernel path.
        logger.info(
            "EB5_INT8_DENSE: keeping %s in BF16 (in=%s out=%s not both "
            "divisible by %d)",
            prefix,
            in_size,
            out_size,
            _ALIGN,
        )
        return None
    if not _LOGGED:
        _LOGGED = True
        logger.info(
            "EB5_INT8_DENSE=%s: online INT8 for dense prefixes %s "
            "(routed experts still come from the checkpoint)",
            os.environ.get("EB5_INT8_DENSE"),
            groups,
        )
    return OnlineInt8LinearMethod()


class OnlineInt8LinearMethod:
    """BF16 weights in the checkpoint, INT8 weights at run time.

    `create_weights` is delegated to `UnquantizedLinearMethod` so the loader
    sees the BF16 parameter it expects; the weight is replaced by its INT8
    quantization after loading, and `apply` runs the same INT8 GEMM as
    `linear_apply_kunlun`. Scale convention follows that hook: `w_pc_max` is the
    per-output-channel amax, not amax/127, so store amax directly.
    """

    def __init__(self) -> None:
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedLinearMethod,
        )

        self._base = UnquantizedLinearMethod()
        self.prefix = ""

    def create_weights(self, *args, **kwargs):
        """Create the BF16 parameters the checkpoint loader expects."""
        return self._base.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Quantize the loaded BF16 weight to per-output-channel INT8.

        ``weight_scale`` is stored as the per-channel amax, which is what the
        kernel's ``w_pc_max`` expects.
        """
        w = layer.weight.data
        if w.dtype == torch.int8:
            return
        # Per-output-channel symmetric: amax over the input dim.
        amax = w.abs().amax(dim=1, keepdim=True).float()
        amax = torch.clamp(amax, min=1e-8)
        q = torch.round(w.float() * (127.0 / amax)).clamp_(-127, 127).to(torch.int8)
        layer.weight = Parameter(q, requires_grad=False)
        layer.weight_scale = Parameter(amax.squeeze(-1), requires_grad=False)
        # The INT8 GEMM takes an fp32 bias.
        bias = getattr(layer, "bias", None)
        if bias is not None and bias.dtype != torch.float32:
            layer.bias.data = layer.bias.data.float()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Quantize the activation per token and run the INT8 GEMM."""
        import kunlun_ops

        w_shape = layer.weight.shape
        if isinstance(x, tuple):
            x_q, x_scale = x
            out = torch.empty(
                (x_q.shape[0], w_shape[0]), dtype=torch.bfloat16, device=x_q.device
            )
        else:
            if x.shape[0] == 0:
                # DP attention leaves idle ranks with zero tokens; quant2d has
                # no meaningful row scale for an empty batch.
                return torch.empty(
                    (0, w_shape[0]), dtype=x.dtype, device=x.device
                )
            x_q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
            x_scale = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
            out = torch.empty((x.shape[0], w_shape[0]), dtype=x.dtype, device=x.device)
            kunlun_ops.quant2d(x, x_q, x_scale, force_sdnn=True)
        if bias is None:
            bias = getattr(layer, "bias", None)
        kunlun_ops.matmul(
            x_q,
            layer.weight.data,
            out,
            bias=bias,
            x_pc_max=x_scale,
            w_pc_max=layer.weight_scale.data,
        )
        return out
