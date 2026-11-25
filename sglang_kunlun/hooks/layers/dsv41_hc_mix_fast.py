# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Framework-side fast path for the hyper-connection ``hc_mix_stats``.

On this build ``DeepseekV4DecoderLayer._hc_mix_stats`` (deepseek_v4.py:3372)
calls ``sglang.kernels.ops.layernorm.mhc.hc_mix_stats(x_flat, hc_fn, eps)``,
which our sibling hook ``hooks/layers/mhc.py`` maps to the xspeedgate-ops
kernel.  That kernel computes, per call,

    mixes[m, :] = hc_fn @ x_flat[m, :]  *  rsqrt(mean(x_flat[m, :]**2) + eps)

with ``x_flat`` [M, 5120] bf16, ``hc_fn`` [24, 5120] fp32.  In the r12 trace it
costs ``infer_hc_mix_stats_partial<bf16>`` 879.1 ms / 237 launches (3.71 ms
each) + ``infer_hc_mix_stats_reduce`` 20.7 ms = 8.65% of device time, i.e.
~0.126 us per token row, while a plain fp32 torch gemm + rowwise norm does the
same work in ~0.03 us per row at the same geometry (vendor kernel is far off
the bandwidth roofline; microbench ``hc_mix_bench.py``).

The replacement is the canonical maths (identical to the upstream non-CUDA
else-branch) chunked over rows so the fp32 temporaries stay bounded:

    y[i] = torch.mm(x[i].float(), hc_fn.t()) * rsqrt(norm(x[i].float())**2 / K + eps)

Numerics: vs the vendor kernel ``max|diff| / max|ref|`` is ~3.6e-7 (float32
rounding on the same gemm), so this is at least as accurate as the reference.

No fp32 activation workspace (r21).  The r13 form above materialises the whole
row chunk in fp32 -- one 335 MiB cast write plus two 335 MiB reads per hc call
-- which at the real geometry (rows=4096, K=20480) is 0.76 ms, i.e. 40 ms per
prefill step over the ~53 calls.  Both the copy and one read can be dropped:

  * ``kunlun_ops.matmul_v2(x_bf16, hc_fn_fp32, out_fp32, w_trans=True)``
    supports the ``(bf16, fp32) -> fp32`` combination, so the gemm reads the
    bf16 activations directly and still accumulates in fp32 (measured vs
    ``torch.mm(x.float(), hc_fn.t())``: rel 3e-7), and
  * the RMS scale is taken from the bf16 source in 256-element blocks:
    ``vector_norm`` rounds each partial norm to bf16 (eps 2e-3), but the
    per-block roundings are independent, so the fp32 sum of the 80 block
    norms carries only ~3e-4 relative error end to end -- 10x better than
    rounding the full-row norm once, and it needs one bf16 read instead of a
    fp32 materialisation.  (``vector_norm(..., dtype=torch.float32)`` is exact
    but upcasts internally and costs 0.55 ms, i.e. no better than the copy.)

Measured at rows=4096 x K=20480: 0.760 -> 0.417 ms per call (1.82x), full chain
rel 2.9e-4 vs the fp32 form.  ``KL_HC_MIX_V2=0`` restores the fp32-workspace
chain; that chain stays as the fallback for non-bf16 inputs.

Env knobs:
  ``KL_HC_MIX_FAST=0``     restore the vendor kernel (default 1 = torch path)
  ``KL_HC_MIX_V2=0``       keep the fp32 workspace chain (r13 form)
  ``KL_HC_MIX_CHUNK``      row chunk size (default 8192; bounds the fp32 temp)
  ``KL_HC_MIX_NORM_BLOCK`` block width of the bf16 RMS reduction (default 256)
  ``KL_HC_MIX_V2_CHECKS``  how many calls to compare against the fp32 chain
                            (default 6; 0 disables the check)
  ``KL_HC_MIX_SELFCHECK=1`` compare the first ``KL_HC_MIX_SELFCHECK_CALLS``
                            calls (default 4) against the vendor kernel and log
                            the scaled max diff; a diff above 1e-5 disables the
                            fast path for good.

``hooks/layers/mhc.py`` is not modified: the previous hook is imported and used
as the fallback, and this file registers its own REPLACE hook (imported after
``mhc`` in ``hooks/layers/__init__.py``, so the last-registered REPLACE wins).
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from . import mhc as _mhc_hook

logger = logging.getLogger(__name__)

_orig = getattr(_mhc_hook, "_hc_mix_stats_torch", None)

_enabled = os.environ.get("KL_HC_MIX_FAST", "1") != "0"
_row_chunk = max(1, int(os.environ.get("KL_HC_MIX_CHUNK", "8192")))
_selfcheck = os.environ.get("KL_HC_MIX_SELFCHECK", "0") == "1"
_selfcheck_calls = int(os.environ.get("KL_HC_MIX_SELFCHECK_CALLS", "4"))

_norm_block_max = max(1, int(os.environ.get("KL_HC_MIX_NORM_BLOCK", "256")))
_v2_checks_left = int(os.environ.get("KL_HC_MIX_V2_CHECKS", "6"))
_v2_reported = False
try:  # (bf16, fp32) -> fp32 gemm; otherwise keep the fp32 workspace chain
    import kunlun_ops
except Exception:  # noqa: BLE001 -- optional native dep
    kunlun_ops = None
_v2 = os.environ.get("KL_HC_MIX_V2", "1") != "0" and kunlun_ops is not None

_state = {"calls": 0, "checked": 0, "bad": 0, "worst": 0.0, "reported": False}


def _fallback(x_flat: torch.Tensor, hc_fn: torch.Tensor, eps: float) -> torch.Tensor:
    if _orig is not None:
        return _orig(x_flat, hc_fn, eps)
    x32 = x_flat.float()
    rsqrt = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    return torch.nn.functional.linear(x32, hc_fn.float()) * rsqrt


def _die(reason: str) -> None:
    globals()["_enabled"] = False
    logger.error(
        "sglang-kunlun: hc_mix_stats torch fast path disabled (%s); "
        "restoring the xspeedgate kernel",
        reason,
    )


def _die_v2(reason: str) -> None:
    globals()["_v2"] = False
    logger.error(
        "sglang-kunlun: hc_mix_stats back to the fp32-workspace chain (%s)", reason
    )


# device index -> fp32 conversion buffer, allocated once and reused.  The CP
# gather at the end of the step asks the allocator for a single ~2.5 GiB block;
# carving a fresh 168 MiB square out of the pool on every hc call both churns
# the allocator and can split the segment that request needs.  One buffer per
# device, grown to the widest row chunk seen -- keying by (rows, k) let short
# requests pin several 168 MiB squares and pinballed the pool at the 65536
# chunk pose, which sits right on the memory edge.  Only the fp32 fallback
# chain touches it now; the r21 chain never materialises x in fp32.
_ws: dict = {}


def _workspace(x: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    idx = x.device.index
    buf = _ws.get(idx)
    if buf is not None and buf.shape[1] == k and buf.shape[0] >= rows:
        return buf
    buf = torch.empty((rows, k), dtype=torch.float32, device=x.device)
    _ws[idx] = buf
    return buf


def _norm_block(k: int) -> int:
    """Largest block width <= the configured one that divides ``k``."""
    step = min(_norm_block_max, k)
    while step > 1 and k % step:
        step -= 1
    return step


def _chain_v2(
    x_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    m: int,
    k: int,
) -> None:
    """bf16 activations straight into the fp32 gemm + blocked RMS scale."""
    block = _norm_block(k)
    for i in range(0, m, _row_chunk):
        rows = min(_row_chunk, m - i)
        xs = x_flat[i : i + rows]
        yc = out[i : i + rows]
        rc = kunlun_ops.matmul_v2(xs, hc_fn, yc, w_trans=True)
        if rc != 0:
            raise RuntimeError(f"matmul_v2(bf16,fp32)->fp32 rc={rc}")
        if block >= k:
            sq = torch.linalg.vector_norm(xs, dim=-1, keepdim=True).float()
            sq = sq.square()
        else:
            blocks = k // block
            part = torch.linalg.vector_norm(xs.view(rows, blocks, block), dim=-1)
            sq = torch.sum(part.float().square(), dim=-1, keepdim=True)
        sq = sq.div_(k).add_(eps).rsqrt_()
        yc.mul_(sq)


def _chain_fp32(
    x_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    m: int,
    k: int,
) -> None:
    """r13 form: fp32 activation workspace shared by the gemm and the norm."""
    hc_t = hc_fn.t()
    for i in range(0, m, _row_chunk):
        rows = min(_row_chunk, m - i)
        xf = _workspace(x_flat, rows, k)[:rows]
        xf.copy_(x_flat[i : i + rows])
        yc = out[i : i + rows]
        torch.mm(xf, hc_t, out=yc)
        # sum(x^2)/K; ``vector_norm`` keeps this a single pass over xf.
        sq = torch.linalg.vector_norm(xf, dim=-1, keepdim=True).square()
        sq = sq.div_(k).add_(eps).rsqrt_()
        yc.mul_(sq)


def _check_v2(
    x_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    eps: float,
    got: torch.Tensor,
    m: int,
    k: int,
    mix: int,
) -> bool:
    """One-shot equivalence check against the fp32-workspace chain."""
    global _v2_checks_left, _v2_reported
    _v2_checks_left -= 1
    ref = torch.empty((m, mix), dtype=torch.float32, device=x_flat.device)
    try:
        _chain_fp32(x_flat, hc_fn, eps, ref, m, k)
    except Exception as exc:  # noqa: BLE001 -- keep the engine alive
        _die(f"fp32 reference chain failed: {type(exc).__name__}: {exc}")
        return False
    scale = ref.abs().max().item()
    rel = (got - ref).abs().max().item() / scale if scale else 0.0
    if rel > 2e-3:
        _die_v2(f"bf16-gemm chain rel {rel:.3e} vs the fp32 workspace chain")
        return False
    if not _v2_reported:
        logger.warning(
            "sglang-kunlun: hc_mix_stats bf16-gemm + blocked RMS chain "
            "(rel %.3e vs the fp32 workspace chain, norm block %d)",
            rel,
            _norm_block(k),
        )
        _v2_reported = True
    return True


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.hc_mix_stats",
    type=HookType.REPLACE,
)
def _hc_mix_stats_fast(
    x_flat: torch.Tensor, hc_fn: torch.Tensor, eps: float
) -> torch.Tensor:
    """``hc_mix_stats`` with a chunked gemm + rowwise RMS instead."""
    if not _enabled:
        return _fallback(x_flat, hc_fn, eps)
    if (
        x_flat.dim() != 2
        or hc_fn.dim() != 2
        or x_flat.stride(1) != 1
        or hc_fn.shape[1] != x_flat.shape[1]
        or hc_fn.dtype != torch.float32
        or x_flat.shape[0] == 0
        or x_flat.dtype
        not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return _fallback(x_flat, hc_fn, eps)

    m, k = x_flat.shape
    mix = hc_fn.shape[0]
    out = torch.empty((m, mix), dtype=torch.float32, device=x_flat.device)
    use_v2 = (
        _v2
        and x_flat.dtype == torch.bfloat16
        and x_flat.is_contiguous()
        and hc_fn.is_contiguous()
    )
    if use_v2:
        try:
            _chain_v2(x_flat, hc_fn, eps, out, m, k)
        except Exception as exc:  # noqa: BLE001 -- keep the engine alive
            _die_v2(f"{type(exc).__name__}: {exc}")
            use_v2 = False
    if not use_v2:
        try:
            _chain_fp32(x_flat, hc_fn, eps, out, m, k)
        except Exception as exc:  # noqa: BLE001 -- keep the engine alive
            _die(f"{type(exc).__name__}: {exc}")
            return _fallback(x_flat, hc_fn, eps)

    _state["calls"] += 1
    if use_v2 and _v2_checks_left > 0 and not _check_v2(
        x_flat, hc_fn, eps, out, m, k, mix
    ):
        # the fp32 chain is known good; redo this call with it
        try:
            _chain_fp32(x_flat, hc_fn, eps, out, m, k)
        except Exception:  # noqa: BLE001
            return _fallback(x_flat, hc_fn, eps)
        return out
    if _selfcheck and _state["checked"] < _selfcheck_calls:
        _state["checked"] += 1
        ref = _fallback(x_flat, hc_fn, eps)
        scale = ref.abs().max().item()
        diff = (out - ref).abs().max().item()
        rel = diff / scale if scale else diff
        _state["worst"] = max(_state["worst"], rel)
        if rel > 1e-5:
            _die(f"selfcheck diff {rel:.3e} over {_selfcheck_calls} calls")
            return ref
        if not _state["reported"]:
            logger.warning(
                "sglang-kunlun: hc_mix_stats torch fast path selfcheck ok "
                "(max|diff|/max|ref| = %.3e, %d calls, chunk=%d)",
                rel,
                _state["checked"],
                _row_chunk,
            )
            _state["reported"] = True
    return out


if _enabled and _orig is not None:
    logger.warning(
        "sglang-kunlun: hc_mix_stats %s (chunk=%d rows, vendor kernel bypassed)",
        "bf16 gemm + blocked RMS (no fp32 workspace)"
        if _v2
        else "in torch fp32",
        _row_chunk,
    )
