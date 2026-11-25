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
"""Kunlun XPU fast path for the grouped wo_a bf16 projection.

``_apply_wo_a_bf16_matmul`` computes ``o[T, G, D] @ wo_a[G, R, D] -> [T, G, R]``
(here T=4096 tokens, G=8 groups, D=4096, R=1024, once per layer).  On this XPU
build every CUDA fast path in it is gated on ``_is_cuda`` (false here), so it
falls through to ``torch.einsum("tgd,grd->tgr", o, wo_a)``, which lowers to a
batched ``bmm_mn_kmn``: 4.77 ms per layer at this geometry, and it also
materialises the transposed output (a ``transpose_0213_cluster`` plus an extra
cast).  One plain ``torch.mm`` per group, written into the strided output, does
the same math at ~122 TFLOPS: 2.71 ms per layer, 1.76x faster, and the
transpose/cast kernels disappear.  Measured end-to-end on the 8x60k prefill
pose: 16.0-16.1k -> ~16.8k tok/s (+4.8%).

fp16 GEMM (r17).  On P800 the bf16 ``xblas`` GEMM tops out near 122 TFLOPS
while the fp16 one reaches ~300 TFLOPS (``bench_gemm_dtype.py``), and fp16
*accumulates in fp32*: measured against an fp32 reference the fp16 GEMM output
is 3.0e-4 relative (below the Half ulp 4.9e-4) versus 2.2e-3 for bf16.  So the
group GEMM now runs in fp16 with a cached fp16 copy of the weight:

    of = fp16(o) permuted to [G, T, D]     (one strided cast kernel)
    y16[g] = mm(of[g], W16[g].T)           (fp32 accumulate, Half out)
    out = bf16(y16 permuted back)          (one cast kernel)

2.20 ms per layer vs 2.75 ms for the bf16 form (1.25x) at this geometry.  The
stored bf16 result is not bit-identical to the bf16 GEMM (it is one bf16
rounding of a *more accurate* product), so a self-check compares against the
bf16 path on the first calls and falls back for good if it drifts.

[r35] orientation.  The per-group ``torch.mm`` above runs M=T, N=R (4096x1024),
which the xblas tiler leaves at 217 TFLOPS; the vendor ``kunlun_ops.matmul``
wants ``w`` as [N, K] anyway, so asking it for ``w16[g] @ of[g].T`` (M=R, N=T)
runs the same 275 GFLOP at 291 TFLOPS: 0.94 vs 1.27 ms per layer.  The fp16
result is [G, R, T], so the closing cast becomes a 3-cycle permute
([G,R,T] -> [T,G,R]) instead of a 2-swap; both cost the same 0.28 ms.  The
2.20 -> 1.88 ms/layer change is numerically *better* than the torch.mm chain
(3.0e-4 vs 2.6e-3 relative to an fp32 reference, the latter dominated by the
bf16 store), and its output is fp16, so the bf16 rounding downstream is
unchanged.  A torch.mm with a transposed B or a strided/mixed-dtype operand is
silently wrong on this build (rel 1.4), so only the vendor entry is used here.

Env knobs:
  ``KL_WOA_FP16=0``          disable the fp16 GEMM, keep the per-group bf16 mm
  ``KL_WOA_FP16O=0``         keep the fp16-activation layer on the original einsum
  ``KL_WOA_FP16_OUT=0``      cast the projection back to bf16 (pre-r74 behaviour)
  ``KL_WOA_MM_RT=0``         fp16 GEMM but with the [T, R] torch.mm orientation
  ``KL_WOA_PERM01=0``        no fused of-side permute+cast (torch copy_ instead)
  ``KL_WOA_VIEWS=0``         re-slice the per-group views on every call (A/B only)
  ``KL_WOA_SELFCHECK=1``     compare the first calls against the bf16 path
  ``KL_WOA_SELFCHECK_CALLS`` how many calls to check (default 3)
  ``KL_WOA_TRACE_FALLBACK=1`` log once per distinct reason a call missed the
                             fast path (diagnostic; the r34 trace shows one
                             einsum per step that should not be there)

[r68b] the closing ``copy_`` fuses the 3-cycle transpose with the fp16 -> bf16
cast, but its cast half runs the vendor ``cluster_cast_kl3<fp16, bf16>`` at only
807 GB/s (50.05 ms per 8-forward capture at the two shapes against 25.04 ms for
the transpose half).  Splitting it into a same-dtype 3-cycle ``copy_`` (1634
GB/s) plus the native ``cast_fp16_bf16`` is 228.6 vs 275.5 us per layer in
``probe_exit2.py`` (-39 us/layer); the fp16 intermediate costs one persistent
[T, G, R] workspace.  The weight cast below goes through the same helper.

[r74] the exit is handed over in fp16 and the cast disappears entirely.  The
sole consumer is ``wo_b``'s ``dynamic_quantize_int8``, which quantises from
whatever 16-bit dtype it is given -- ``kunlun_ops.quant2d`` runs at the same
speed for fp16 and bf16 (106.5 vs 107.9 us at [4096, 12288]) and keeps three
more mantissa bits.  Removing the cast took ``infer_cast_dtype<fp16,bf16>``
from 328 to 8 calls per capture: -38.62 ms/capture = -4.83 ms/fwd (0.43% of
device time, below the 0.87% e2e noise floor).  ``wo_b`` still produces bf16.

The bf16 path itself differs from the einsum only by GEMM tiling (bf16
rounding, relative ~1e-5 on the summed output).  Both paths are limited to the
plain bf16 prefill-ish fallback: decode / target-verify keep their own
kernels, non-bf16 dtypes and grads-enabled calls go to the original.

[r36] fp16 activations.  One layer per step calls this with an *fp16* ``o``
and a bf16 ``wo_a``; the original then promotes the operands to bf16, runs the
batched einsum (3.57 ms) and pays two casts (5.40 ms/step in the r35 trace).
The same vendor GEMM handles it directly when ``wo_a`` is cast to fp16: the
entry permute is then a same-dtype copy (no cast), the GEMM keeps fp32
accumulation, and the closing permute casts fp16 -> bf16 in one kernel, which
is exactly the dtype ``torch.einsum`` produces here.  1.45 vs 5.50 ms/layer.
The bf16 ``torch.mm`` group helper must never see mixed dtypes (silently
wrong, rel 1.4), so an fp16 call that loses the fast path goes to the original.

[r37] launch latency.  At this point the wo_a *device* time is at its floor
(the casts run at 1.3-1.4 TB/s, the GEMM at 291 TFLOPS), so what is left is the
host side: slicing ``w16[g] / of[g] / yt[g]`` inside the loop costs 63 us of
pure host time per layer (24 slice ops at ~2.6 us), on top of the eight ~18 us
vendor entries.  The three tensors only change when the weight is recast or a
workspace grows, so the eight view tuples are cached and only the GEMMs are
re-issued: 129.2 vs 229.5 us of empty-queue launch latency per layer, zero
device change, zero numeric change.  The grouped NT GEMM was measured as the
alternative and rejected: one launch (74.8 us) but 1.067 vs 0.962 ms of device
time per layer, i.e. +0.105 ms/layer of device for -54 us/layer of host.

[r39] weight cast is 40 x 99 us = 3.98 ms/step and cannot be cached away on this
pose.  The model walks 40 distinct wo_a weights per step (one per layer) and the
cache above keeps one entry ("same shape, different ptr" eviction), so every
step re-casts all 40 -- the r39 trace shows exactly 40 `_weight_fp16` kernels
and 40 `cluster_cast_kl3` calls per step.  Keeping one fp16 copy per weight
(40 x 67 MB = 2.7 GB) removes all 40 casts, but the pose only leaves ~1.5 GB of
allocator headroom (mem-fraction-static fills the rest) and the third
measurement round died with `torch.OutOfMemoryError` (1024 MiB request inside
the indexer score matmul) -- so the cache is deliberately left at one entry.
A cap below the live count cannot help either: FIFO over a 40-weight cycle needs
>= 40 slots, and a 32-slot cache was measured at 40 misses/step, i.e. worse
than the 40 it replaces.

[r42] the of-side permute+cast is fused again.  ``of.copy_(o.permute(1, 0, 2))``
is two full passes over the 8xT x 4096 activation (the r41 trace: bf16->fp16
``cast`` 391 us + ``transpose_0213`` 300 us per call, 27.6 ms/step over the 40
layers).  The 0/1 swap keeps the inner dimension contiguous on both sides, so
the native ``cast_permute01(src, dst)`` does it in one pass (the manual-0022
op, +1.2% on the pre-r35 layout).  It is used only when the op is registered and
the call matches its layout (dst == permute(1,0,2) of src, inner stride 1,
bf16/fp16 in either direction, same dtype included); otherwise the torch chain
is kept.  The closing 3-cycle ([G,R,T] -> [T,G,R]) is NOT expressible (the op
only does a 0/1 swap) and stays on ``copy_``.
"""

from __future__ import annotations

import inspect
import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.hooks.layers.xsg_cast import xsg_cast

try:
    import kunlun_ops
except ImportError:  # pragma: no cover -- the mm orientation falls back
    kunlun_ops = None

logger = logging.getLogger(__name__)

_reported = False
_MIN_TOKENS = 64
_GROUPED_MM = True

_fp16_ok = os.environ.get("KL_WOA_FP16", "1") != "0"
_fp16o_ok = os.environ.get("KL_WOA_FP16O", "1") != "0"
# [r74] hand the exit over in fp16 instead of casting to bf16; see the
# module docstring.  ``wo_b``'s int8 quant takes fp16 as-is.
_fp16_out = os.environ.get("KL_WOA_FP16_OUT", "1") != "0"
_mm_rt = os.environ.get("KL_WOA_MM_RT", "1") != "0" and kunlun_ops is not None
_selfcheck = os.environ.get("KL_WOA_SELFCHECK", "0") == "1"
_selfcheck_calls = int(os.environ.get("KL_WOA_SELFCHECK_CALLS", "3"))
_trace_fallback = os.environ.get("KL_WOA_TRACE_FALLBACK", "0") == "1"
_views_ok = os.environ.get("KL_WOA_VIEWS", "1") != "0"
_perm01_ok = None if os.environ.get("KL_WOA_PERM01", "1") != "0" else False
_fb_seen: set = set()
_state = {"calls": 0, "checked": 0, "worst": 0.0, "reported": False}


def _grouped_mm_enabled() -> bool:
    return _GROUPED_MM


def _die(reason: str) -> None:
    globals()["_fp16_ok"] = False
    logger.error(
        "sglang-kunlun: wo_a fp16 GEMM disabled (%s); back to the bf16 mm path",
        reason,
    )


# wo_a weight tensor -> its fp16 copy.  Keyed on (data_ptr, shape, version) so
# an in-place update (requantise, load) rebuilds instead of silently reusing.
_w16_cache: dict = {}
# device index -> fp16 activation workspace [G, T, D] (widest T seen).
_ws: dict = {}
# device index -> fp16 output workspace [G, T, R].
_ws_out: dict = {}
# device index -> fp16 [R, T]-oriented workspace [G, R, T] (r35).
_ws_rt: dict = {}
# device index -> fp16 [T, G, R] buffer holding the 3-cycle-transposed exit
# before the native cast turns it bf16 (r68b; kept fp16 by r74).
_ws_perm: dict = {}
# (w16, of, yt) fingerprint -> [(w16[g], of[g], yt[g]) for g in range(G)].
# Slicing three 3D tensors eight times costs 63 us of host time per layer on
# this build; the views stay valid as long as the three base tensors live and
# are not resized, which the fingerprint (3 x data_ptr + shape) fully captures.
_views: dict = {}


def _weight_fp16(wo_a: torch.Tensor) -> torch.Tensor:
    key = (wo_a.data_ptr(), tuple(wo_a.shape), wo_a._version)
    hit = _w16_cache.get(key)
    if hit is not None:
        return hit
    # One entry per live weight tensor: drop stale versions of the same shape.
    for k in [k for k in _w16_cache if k[1] == key[1] and k[0] != key[0]]:
        del _w16_cache[k]
    w16 = xsg_cast(wo_a, torch.float16)
    if not w16.is_contiguous():
        w16 = w16.contiguous()
    _w16_cache[key] = w16
    return w16


def _buf(store: dict, x: torch.Tensor, shape: tuple, dtype: torch.dtype):
    """Persistent workspace keyed by device; grown to the widest request.

    Allocating a fresh [G, T, D] half tensor per layer churns the allocator and
    can split the segment the end-of-step CP gather needs (see the same note in
    ``dsv41_hc_mix_fast.py``), so keep one buffer per device and device-side.
    """
    idx = x.device.index
    buf = store.get(idx)
    if buf is not None and buf.shape == shape:
        return buf
    if buf is not None and buf.numel() >= 1 and tuple(buf.shape[1:]) == shape[1:]:
        if buf.shape[0] >= shape[0]:
            return buf[: shape[0]]
    buf = torch.empty(shape, dtype=dtype, device=x.device)
    store[idx] = buf
    return buf


def _permute_cast(src: torch.Tensor, dst: torch.Tensor) -> bool:
    """``dst[n1, n0, d] = cast(src[n0, n1, d])`` in one device pass.

    Returns False when the native op is missing or cannot take this call; the
    caller then keeps ``dst.copy_(src.permute(1, 0, 2))``.
    """
    global _perm01_ok
    if _perm01_ok is None:
        try:
            import xspeedgate_ops  # noqa: F401 -- registers the torch.ops entry

            torch.ops.xspeedgate_ops.cast_permute01  # noqa: B018
            _perm01_ok = True
        except Exception:  # noqa: BLE001 -- wheel without the op
            _perm01_ok = False
            logger.warning(
                "sglang-kunlun: wo_a fused permute+cast unavailable "
                "(`xspeedgate_ops.cast_permute01` not registered); "
                "using the torch copy_ chain"
            )
    if (
        not _perm01_ok
        or src.dim() != 3
        or src.device.type != "cuda"
        or src.stride(2) != 1
        or not dst.is_contiguous()
        or dst.shape != (src.shape[1], src.shape[0], src.shape[2])
        # The op is a *cast*: bf16<->fp16 only, never same-dtype (it raises
        # "unsupported dtype pair Half -> Half").  The fp16-activation layer
        # therefore keeps the plain copy_ chain.
        or {src.dtype, dst.dtype} != {torch.bfloat16, torch.float16}
    ):
        return False
    try:
        torch.ops.xspeedgate_ops.cast_permute01(src, dst)
        return True
    except Exception as exc:  # noqa: BLE001 -- keep the engine alive
        globals()["_perm01_ok"] = False
        logger.warning(
            "sglang-kunlun: wo_a fused permute+cast disabled (%s: %s); "
            "using the torch copy_ chain",
            type(exc).__name__,
            exc,
        )
        return False


def _bf16_groups(o: torch.Tensor, wo_a: torch.Tensor) -> torch.Tensor:
    groups, rank_dim = wo_a.shape[0], wo_a.shape[1]
    out = torch.empty(
        (o.shape[0], groups, rank_dim), dtype=o.dtype, device=o.device
    )
    for g in range(groups):
        # ``o[:, g, :]`` is the mid-dim slice of the attention output and
        # ``out[:, g, :]`` is strided as well; write the GEMM into a
        # contiguous temp and copy back, which keeps the kernel on its
        # single-launch path instead of the split/strided-out fallback.
        out[:, g, :] = torch.mm(o[:, g, :], wo_a[g].transpose(0, 1))
    return out


def _fp16_groups(o: torch.Tensor, wo_a: torch.Tensor) -> torch.Tensor:
    global _mm_rt
    if _mm_rt:
        try:
            return _fp16_groups_rt(o, wo_a)
        except Exception as exc:  # noqa: BLE001 -- keep the engine alive
            globals()["_mm_rt"] = False
            logger.error(
                "sglang-kunlun: wo_a [R,T] vendored GEMM disabled (%s: %s); "
                "back to the torch.mm orientation",
                type(exc).__name__,
                exc,
            )
    return _fp16_groups_tr(o, wo_a)


def _out_dtype(o: torch.Tensor) -> torch.dtype:
    """Dtype the original einsum produces: torch yields bf16 for fp16 x bf16 here."""
    return torch.bfloat16 if o.dtype == torch.float16 else o.dtype


def _original_input(o):
    """``o`` as the upstream bf16 einsum expects it.

    The attention backend hands the extend output over in fp16 unconditionally
    (``_ATTN_OUT_KEEP_FP16``), so every call that misses the fast path -- fewer
    than ``_MIN_TOKENS`` rows, grads on, ``KL_WOA_FP16O=0``, or the fp16 GEMM
    having died -- would reach ``torch.einsum("tgd,grd->tgr", fp16, bf16)``.
    That promotes to fp32 and returns a different dtype than the bf16 path the
    caller is written against, so narrow it back first (one native cast, off
    the hot path).
    """
    if isinstance(o, torch.Tensor) and o.dtype is torch.float16:
        return xsg_cast(o, torch.bfloat16)
    return o


def _group_views(w16: torch.Tensor, of: torch.Tensor, yt: torch.Tensor) -> list:
    """The three per-group view tuples, built once per (weight, workspace) set."""
    if not _views_ok:
        return [(w16[g], of[g], yt[g]) for g in range(of.shape[0])]
    key = (
        w16.data_ptr(), of.data_ptr(), yt.data_ptr(),
        tuple(w16.shape), tuple(of.shape), tuple(yt.shape),
    )
    hit = _views.get(key)
    if hit is not None:
        return hit
    views = [(w16[g], of[g], yt[g]) for g in range(of.shape[0])]
    _views.clear()
    _views[key] = views
    return views


def _fp16_groups_rt(o: torch.Tensor, wo_a: torch.Tensor) -> torch.Tensor:
    """``w16[g] @ of[g].T`` -> [G, R, T]: M=R, N=T tiles at 291 vs 217 TFLOPS."""
    groups, rank_dim = wo_a.shape[0], wo_a.shape[1]
    tokens, _, dim = o.shape
    w16 = _weight_fp16(wo_a)
    of = _buf(_ws, o, (groups, tokens, dim), torch.float16)
    # fp16 o: same-dtype permute (cheaper than the fused cast+permute, and the
    # one pair cast_permute01 rejects by design); bf16 o: fused cast+permute.
    if o.dtype is torch.float16:
        of.copy_(o.permute(1, 0, 2))
    elif not _permute_cast(o, of):
        of.copy_(o.permute(1, 0, 2))
    yt = _buf(_ws_rt, o, (groups, rank_dim, tokens), torch.float16)
    for w, x, y in _group_views(w16, of, yt):
        kunlun_ops.matmul(w, x, y)
    # [G, R, T] fp16 -> [T, G, R] in two passes: a same-dtype 3-cycle transpose
    # (1634 GB/s) and then the native cast, because the fused
    # ``copy_(permute(2, 0, 1))`` converts at only 807 GB/s (r68b).  [r74]
    # ``_fp16_out`` returns the fp16 buffer outright: wo_b re-quantises it to
    # int8 right away and quant2d takes fp16 unchanged (see the docstring).
    tmp = _buf(_ws_perm, o, (tokens, groups, rank_dim), torch.float16)
    tmp.copy_(yt.permute(2, 0, 1))
    if _fp16_out:
        return tmp
    return xsg_cast(tmp, torch.bfloat16)


def _fp16_groups_tr(o: torch.Tensor, wo_a: torch.Tensor) -> torch.Tensor:
    groups, rank_dim = wo_a.shape[0], wo_a.shape[1]
    tokens, _, dim = o.shape
    w16 = _weight_fp16(wo_a)
    of = _buf(_ws, o, (groups, tokens, dim), torch.float16)
    # ``of`` is always fp16; when ``o`` already is fp16 (the attention output
    # now stays fp16 end to end) this is a pure mid-dim transpose and the
    # source cast disappears.
    of.copy_(o.permute(1, 0, 2))
    y16 = _buf(_ws_out, o, (groups, tokens, rank_dim), torch.float16)
    for g in range(groups):
        torch.mm(of[g], w16[g].transpose(0, 1), out=y16[g])
    # Fallback orientation only: keep the pre-r74 bf16 exit (the r74 probe was
    # taken on the [R, T] path, which is the one every delivered pose runs).
    out = torch.empty(
        (tokens, groups, rank_dim), dtype=_out_dtype(o), device=o.device
    )
    out.copy_(y16.permute(1, 0, 2))
    return out


def _shape_of(x):
    return tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__


def _fallback_reason(o, wo_a, is_decode, is_target_verify) -> str:
    if not _grouped_mm_enabled():
        return "grouped-mm-off"
    if is_decode:
        return "is_decode"
    if is_target_verify:
        return "is_target_verify"
    if torch.is_grad_enabled():
        return "grad-enabled"
    if not isinstance(o, torch.Tensor):
        return "o-not-a-tensor"
    if not isinstance(wo_a, torch.Tensor):
        return "wo_a-not-a-tensor"
    if o.dim() != 3:
        return f"o-dim-{o.dim()}"
    if o.dtype is torch.float16 and not _fp16o_ok:
        return "fp16o-off"
    if o.dtype not in (torch.bfloat16, torch.float16):
        return f"o-dtype-{o.dtype}"
    if wo_a.dtype != torch.bfloat16:
        return f"wo_a-dtype-{wo_a.dtype}"
    if o.device.type != "cuda":
        return f"device-{o.device.type}"
    if o.shape[0] < _MIN_TOKENS:
        return f"tokens-{o.shape[0]}"
    if o.shape[1:] != (wo_a.shape[0], wo_a.shape[2]):
        return "o/wo_a-shape-mismatch"
    return "unknown"


def _log_fallback(o, wo_a, is_decode, is_target_verify, kwargs) -> None:
    """One warning per distinct reason/shape that did not take the fast path."""
    reason = _fallback_reason(o, wo_a, is_decode, is_target_verify)
    key = (reason, _shape_of(o), _shape_of(wo_a))
    if key in _fb_seen:
        return
    _fb_seen.add(key)
    caller = ""
    try:
        frames = inspect.stack()[2:5]
        caller = " < ".join(
            f"{os.path.basename(f.filename)}({f.lineno}): {f.function}"
            for f in frames
        )
    except Exception:  # noqa: BLE001 -- diagnostics only
        pass
    logger.warning(
        "sglang-kunlun: wo_a fast path skipped (%s) o=%s/%s wo_a=%s/%s "
        "is_decode=%s is_target_verify=%s grad=%s kwargs=%s; caller=%s",
        reason,
        _shape_of(o),
        getattr(o, "dtype", "-"),
        _shape_of(wo_a),
        getattr(wo_a, "dtype", "-"),
        is_decode,
        is_target_verify,
        torch.is_grad_enabled(),
        sorted(kwargs),
        caller,
    )


@plugin_hook(
    "sglang.srt.models.deepseek_v4._apply_wo_a_bf16_matmul",
    type=HookType.AROUND,
)
def _apply_wo_a_bf16_matmul_kunlun(original_fn, o, wo_a, *args, **kwargs):
    global _reported
    is_decode = kwargs.get("is_decode", args[0] if len(args) > 0 else False)
    is_target_verify = kwargs.get(
        "is_target_verify", args[1] if len(args) > 1 else False
    )
    if (
        _grouped_mm_enabled()
        and not is_decode
        and not is_target_verify
        and torch.is_grad_enabled() is False
        and isinstance(o, torch.Tensor)
        and isinstance(wo_a, torch.Tensor)
        and o.dim() == 3
        and (o.dtype is torch.bfloat16 or (o.dtype is torch.float16 and _fp16o_ok))
        and wo_a.dtype == torch.bfloat16
        and o.device.type == "cuda"
        and o.shape[0] >= _MIN_TOKENS
        and o.shape[1:] == (wo_a.shape[0], wo_a.shape[2])
    ):
        out = None
        if _fp16_ok:
            try:
                out = _fp16_groups(o, wo_a)
            except Exception as exc:  # noqa: BLE001 -- keep the engine alive
                _die(f"{type(exc).__name__}: {exc}")
                out = None
        if out is None:
            if o.dtype is not torch.bfloat16:
                # The bf16 group mm would silently mix dtypes (rel 1.4 wrong).
                return original_fn(_original_input(o), wo_a, *args, **kwargs)
            out = _bf16_groups(o, wo_a)

        _state["calls"] += 1
        if (
            _selfcheck
            and _fp16_ok
            and _state["checked"] < _selfcheck_calls
        ):
            _state["checked"] += 1
            ref = (
                _bf16_groups(o, wo_a)
                if o.dtype is torch.bfloat16
                else original_fn(_original_input(o), wo_a, *args, **kwargs)
            )
            scale = ref.abs().max().item()
            diff = (out.float() - ref.float()).abs().max().item()
            rel = diff / scale if scale else diff
            _state["worst"] = max(_state["worst"], rel)
            if rel > 2e-2:
                _die(f"selfcheck rel {rel:.3e} over {_selfcheck_calls} calls")
                out = ref
            elif not _state["reported"]:
                logger.warning(
                    "sglang-kunlun: wo_a fp16 GEMM selfcheck ok "
                    "(max|diff|/max|ref| = %.3e vs the bf16 path, %d calls)",
                    rel,
                    _state["checked"],
                )
                _state["reported"] = True

        if not _reported:
            logger.warning(
                "sglang-kunlun: wo_a projection via per-group %s mm "
                "(%d groups, T=%d), replacing the batched-einsum fallback",
                "fp16" if _fp16_ok else "bf16",
                wo_a.shape[0],
                o.shape[0],
            )
            _reported = True
        return out
    if _trace_fallback:
        _log_fallback(o, wo_a, is_decode, is_target_verify, kwargs)
    return original_fn(_original_input(o), wo_a, *args, **kwargs)
