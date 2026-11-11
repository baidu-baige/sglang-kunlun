# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HookRegistry registration for the DSV4 bf16 x bf16 -> fp32 linear.

Upstream ``sglang.jit_kernel.dsv4.gemm.linear_bf16_fp32`` relies on the
``torch.mm(out_dtype=...)`` overload (``aten::mm.dtype``), which the Kunlun XPU does
not provide.  The REPLACE hook below uses Kunlun's bf16 x bf16 -> fp32 ``matmul``
(``kunlun_ops.matmul``), the same type combination the CUDA path asks cuBLAS for:
inputs stay bf16, accumulation and output are fp32.

The previous fallback materialised both operands in fp32 first
(``torch.matmul(x.float(), y.float().t())``).  On this prefill pose those casts are
the dominant cost of the two hot call sites (r15 trace, per 3 steps):

    x [32768,5120] bf16 x wkv.weight        [512,5120] bf16  ->  18 calls
        cast x        11.3 ms   fp32 mm 26.8 ms
    x [ 4096,5120] bf16 x ffn.gate.weight   [384,5120] fp16  -> 120 calls
        cast x         9.6 ms   fp32 mm 15.1 ms   (weight cast 1.2 ms)

``x`` is bf16 while the *router gate* weight is fp16 at runtime (``--dtype
float16``), so the guard below used to reject it and every call took the fp32
cast chain.  Both weights are natively bf16 in the checkpoint
(``ffn.gate.weight`` BF16, ``attn.wkv.weight`` I8 dequantised to bf16), so a
bf16 weight rebuilt from the runtime fp16 copy is value-identical; it is cached
per weight tensor and rebuilt only if the parameter is updated in place.

Numerics: the bf16 inputs and the (exactly restored) bf16 weights make this the
same product as the fp32-cast chain; only the fp32 accumulation order differs.
``KL_BF16_FP32_SELFCHECK=1`` compares the first calls against the fp32
reference and disables the vendor path if the deviation exceeds the bound.

``sglang.jit_kernel.dsv4.__init__`` re-exports this symbol;
``srt.layers.attention.dsv4.compressor`` imports it at module scope and
``models.deepseek_v2`` lazily inside the call site. ``_propagate_patch`` rewrites the
already-imported bindings, and the re-export is patched for later importers.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_reported = {"active": False, "failed": False}
_selfcheck = os.environ.get("KL_BF16_FP32_SELFCHECK", "0") == "1"
_selfcheck_calls = int(os.environ.get("KL_BF16_FP32_SELFCHECK_CALLS", "4"))
_checked = 0
_disabled = False
_logged: set = set()
_MAX_REL = 1e-5
# Only the prefill-sized calls move the needle; the 1-token decode calls are
# both cheap and numerous, so they neither get checked nor logged.
_CHECK_MIN_ROWS = 1024

# id(weight) -> ((fingerprint), bf16 copy); a few MiB per layer, 40 layers.
_wcache: dict = {}


def _bf16_weight(y: torch.Tensor) -> torch.Tensor:
    """``y`` as bf16, caching the conversion of a runtime fp16 parameter."""
    if y.dtype == torch.bfloat16 and y.is_contiguous():
        return y
    key = (y.data_ptr(), tuple(y.shape), y._version)
    hit = _wcache.get(id(y))
    if hit is not None and hit[0] == key:
        return hit[1]
    yb = y.detach().to(torch.bfloat16).contiguous()
    _wcache[id(y)] = (key, yb)
    return yb


def _vendor_bf16_fp32(x: torch.Tensor, y: torch.Tensor):
    """``x @ y.T`` with bf16 inputs and an fp32 output, or ``None``."""
    global _checked, _disabled
    reason = None
    if _disabled:
        reason = "disabled"
    elif x.dtype != torch.bfloat16:
        reason = f"x.dtype={x.dtype}"
    elif y.dtype not in (torch.bfloat16, torch.float16):
        reason = f"y.dtype={y.dtype}"
    elif x.shape[-1] != y.shape[-1]:
        reason = f"k mismatch {x.shape[-1]} vs {y.shape[-1]}"
    elif x.numel() == 0:
        reason = "empty"
    elif not x.is_contiguous():
        reason = "x non-contiguous"
    if reason is not None:
        if _selfcheck and x.numel() and x.shape[0] >= 1024:
            key = ("fallback", reason, tuple(x.shape), tuple(y.shape))
            if key not in _logged:
                _logged.add(key)
                logger.warning(
                    "sglang-kunlun: linear_bf16_fp32 fallback (%s) x=%s y=%s",
                    reason,
                    tuple(x.shape),
                    tuple(y.shape),
                )
        return None
    try:
        import kunlun_ops

        yb = _bf16_weight(y)
        rows = x.reshape(-1, x.shape[-1])
        out = torch.empty(
            (rows.shape[0], yb.shape[0]), dtype=torch.float32, device=x.device
        )
        kunlun_ops.matmul(rows, yb, out)
    except Exception as exc:  # noqa: BLE001 -- fall back to the fp32 cast chain
        if not _reported["failed"]:
            _reported["failed"] = True
            logger.warning(
                "sglang-kunlun: kunlun_ops.matmul bf16 x bf16 -> fp32 failed (%s); "
                "using the fp32 cast chain",
                exc,
            )
        return None

    if _selfcheck and _checked < _selfcheck_calls and x.shape[0] >= _CHECK_MIN_ROWS:
        _checked += 1
        ref = torch.matmul(x.float(), y.float().t())
        d = (out - ref).abs().max().item()
        scale = ref.abs().max().item() or 1.0
        rel = d / scale
        if rel > _MAX_REL:
            _disabled = True
            logger.error(
                "sglang-kunlun: linear_bf16_fp32 vendor path DISABLED, "
                "max|d|=%.3g rel=%.2e > %.0e (x=%s y=%s y_dtype=%s)",
                d,
                rel,
                _MAX_REL,
                tuple(x.shape),
                tuple(y.shape),
                y.dtype,
            )
            return None
        logger.warning(
            "sglang-kunlun: linear_bf16_fp32 selfcheck ok max|d|=%.3g rel=%.2e "
            "x=%s y=%s y_dtype=%s",
            d,
            rel,
            tuple(x.shape),
            tuple(y.shape),
            y.dtype,
        )
    return out.view(*x.shape[:-1], yb.shape[0])


@plugin_hook(
    "sglang.kernels.ops.attention.dsv4.gemm.linear_bf16_fp32",
    type=HookType.REPLACE,
)
def _linear_bf16_fp32_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 linear (no ``aten::mm.dtype`` overload on Kunlun)."""
    out = _vendor_bf16_fp32(x, y)
    if out is not None:
        if not _reported["active"]:
            _reported["active"] = True
            logger.warning(
                "sglang-kunlun: linear_bf16_fp32 via kunlun_ops.matmul "
                "(bf16 x bf16 -> fp32), no fp32 operand casts"
            )
        return out
    return torch.matmul(x.to(torch.float32), y.to(torch.float32).t())
