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
"""Kunlun XPU fallback for the DSA prefill-CP index kernels.

The prefill context-parallel indexer path
(``_forward_low_ratio_sources_cp`` -> ``_low_ratio_index_topk_dense``) calls two
kernels that this XPU build does not provide, and -- unlike the non-CP prefill
path (``_low_ratio_index_topk``, which guards ``_has_dense_fp4_indexer()`` and
otherwise falls back to ``_low_ratio_index_topk_torch``) -- it guards neither:

* ``_dense_fp4_mqa_logits`` needs ``deep_gemm.fp8_fp4_mqa_logits``, absent here
  (``deep_gemm`` is a stub), so the first CP prefill dies with::

      ImportError: cannot import name 'fp8_fp4_mqa_logits' from 'deep_gemm'

* ``topk_transform_ragged_v2`` dispatches to the XPU-native
  ``torch.ops.sgl_kernel.topk_transform_ragged`` only when ``is_xpu()`` is true;
  this build presents XPU as CUDA (``torch.xpu`` is absent -> ``is_xpu()`` false),
  so it falls through to the CUDA JIT, whose ``nvcc`` build fails with
  ``fatal error: concepts: No such file or directory``.

Both hooks keep the real kernel whenever it is usable and otherwise compute the
same result in torch, so they stay inert on any box that has the kernels.

* ``_dense_fp4_mqa_logits``: dequantize the packed e2m1 / ue8m0 index tensors and
  evaluate, per request window, ``sum_h relu(<q_h, k_j>) * w_h`` -- the reference
  rounding the fp4 kernel documents (bf16 dot -> relu -> * bf16 weight -> sum).
* ``topk_transform_ragged_v2``: per row, the top-k of its
  ``[start, start+seq_lens)`` window, written as ``position + out_offsets`` with
  ``-1`` padding.  The CP call site passes ``row_starts=None`` (window base 0), so
  the kernel's in-place masking of the <= 3 columns ahead of the 16-byte-aligned
  read base is a no-op there and the torch version need not reproduce it.

On top of the two kernel fallbacks, ``_low_ratio_index_topk_dense`` itself is
replaced (same code, one tail difference): the XSpeedGate ragged top-k emits
rows already ascending, so the trailing ``masked_fill + sort`` -- which only
re-orders the kernel output -- is skipped.  The candidate mask does not open
gaps (measured 0 over every call of a 61k-token sweep: it keeps ≥
``min(topk, lens)`` finite candidates, which is what the kernel selects), so
the masked layers skip it too; ``KL_TOPK_HOLE_DEBUG`` recounts gaps live.
"""

from __future__ import annotations

import functools
import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels import indexer_i8_core

logger = logging.getLogger(__name__)

_reported = False
_reported_topk = False
# R1 port: native int8 dense MQA-logits (see indexer_i8_core.py).  Off by
# default, latched off for the process on the first failure.
_native_mqa_disabled = False
_reported_native_mqa = False
# Keep the [rows, heads, window] bf16 score tensor bounded, as the torch indexer
# does for its own scores.  Env ``KL_IDX_SCORE_BUDGET_MB`` overrides the 512 MiB
# cap; a larger cap means fewer row-chunks per call (fewer launches), the row
# chunks being independent so the result is unchanged.  (1024 MiB OOMs at long
# context: the cap is a per-call temporary on top of the logits plane.)
_SCORE_BUDGET_BYTES = int(
    os.environ.get("KL_IDX_SCORE_BUDGET_MB", "512")
) * 1024 * 1024
# Row-chunk cap for the candidate-publish pooling.  The engine's own
# ``_TORCH_INDEXER_SCORE_BUDGET_BYTES`` (1 GiB, hard-coded) sizes this loop, and
# the block amax costs a temporary as large as its input slice -- at chunk 65536
# that is 1 GiB on top of a 1 GiB ``logits`` buffer and OOMs.  Chunking finer is
# result-identical (rows are independent), so use a smaller cap.
_KEEP_BUDGET_BYTES = int(
    os.environ.get("KL_IDX_KEEP_BUDGET_MB", "256")
) * 1024 * 1024
# The score blocks write ``logits[rows, :width]`` and the epilogue writes *every*
# column in ``[0, width)`` (probed on the wheel: a sentinel-filled ``out`` keeps
# zero sentinel values inside the write region, so it does not rely on a
# pre-initialised buffer).  Only the columns past a row's scored prefix need the
# ``-inf`` sentinel, and in the canonical pose ``width == max_seqlen_k`` there are
# none -- so the unconditional ``torch.full(-inf)`` was a dead 0.5 GiB write per
# indexer layer.  ``KL_IDX_LOGITS_FULL=1`` restores it (A/B + fallback).
_LOGITS_FULL = os.environ.get("KL_IDX_LOGITS_FULL", "0") not in ("", "0", "false", "False")
# Physical row width of the logits plane, rounded up to the candidate block size
# (8).  Core asks for ``ceil_align(max(lc_per_req), 4)`` = 61436 at the 60k pose,
# and 61436 % 8 == 4: the candidate publisher could then only pool
# ``[..., :61432]`` as ``unflatten(-1, (7679, 8))`` on a row whose stride is
# 61436, which is not an expressible view, so ATen materialised a contiguous body
# copy first (``strided_slice_copy2d``, 0.26 ms x 32 = 2.10 ms/forward).  With
# the row rounded up to 61440 the ``unflatten`` is a view of a contiguous plane
# and the block max runs on the pool input directly.  Bit-exact: the publisher
# pools the *physical* row and masks at block granularity (``col > last`` ->
# -inf, ``col == last`` -> +inf), so the 4 extra columns only ever land in the
# block that is forced kept, and every downstream reader (ragged top-k,
# ``mask_topk_scores``, the candidate masks) bounds itself by each row's own
# ``compress_lens``.  ``KL_IDX_PLANE_ALIGN8=0`` is the rollback.
_PLANE_ALIGN = 8 if os.environ.get("KL_IDX_PLANE_ALIGN8", "1") not in (
    "",
    "0",
    "false",
    "False",
) else 1
# Sentinel-fill of the columns past each row's scored prefix.  Only a full-row
# top-k reads them (the ragged kernels stop at ``seq_lens``); when a ragged
# top-k is installed the fill is dead work, so it is skipped unless
# ``KL_IDX_LOGIT_TAIL=1`` (restores the unconditional fill for A/B).
_TAIL_PREFILL = os.environ.get("KL_IDX_LOGIT_TAIL", "auto") not in (
    "",
    "0",
    "auto",
    "false",
    "False",
)
# Score-GEMM/epilogue dtype.  The score plane is the largest tensor the indexer
# touches (``[rows, heads, window]`` per block, written by the GEMM and read by
# the fused epilogue in one pass), so halving its element width halves both
# passes.  fp16 also has the wider-mantissa/faster-GEMM profile on this XPU
# (≈300 vs ≈122 TFLOPS bf16).  Gated on the wheel: builds whose
# ``mqa_logits_sum_heads`` only instantiate bf16 scores must keep bf16, which the
# runtime probe in ``_score_dtype`` checks on the first indexer call, so the
# default is on and an old wheel silently keeps bf16.  ``KL_IDX_SCORE_FP16=0``
# is the rollback.
_SCORE_FP16 = os.environ.get("KL_IDX_SCORE_FP16", "1") not in ("", "0", "false", "False")
_SCORE_DTYPE = None


def _score_dtype(device):
    """Element type for the score GEMM output / epilogue input (cached)."""
    global _SCORE_DTYPE
    if _SCORE_DTYPE is not None:
        return _SCORE_DTYPE
    if not _SCORE_FP16 or not _xspeedgate_op_available("mqa_logits_sum_heads"):
        _SCORE_DTYPE = torch.bfloat16
        return _SCORE_DTYPE
    try:
        width = 8
        scores = torch.zeros((1, 1, width), dtype=torch.float16, device=device)
        weights = torch.zeros((1, 1), dtype=torch.bfloat16, device=device)
        lens = torch.zeros((1,), dtype=torch.int64, device=device)
        out = torch.zeros((1, width), dtype=torch.float32, device=device)
        torch.ops.xspeedgate_ops.mqa_logits_sum_heads(scores, weights, lens, out, width)
        _SCORE_DTYPE = torch.float16
        logger.info(
            "sglang-kunlun: index score GEMM + epilogue in fp16 "
            "(KL_IDX_SCORE_FP16=0 reverts to bf16)"
        )
    except Exception as exc:  # noqa: BLE001 -- wheel has no fp16 score instance
        logger.warning(
            "sglang-kunlun: mqa_logits_sum_heads rejects fp16 scores (%s); "
            "keeping bf16 score GEMM", exc,
        )
        _SCORE_DTYPE = torch.bfloat16
    return _SCORE_DTYPE
# Skip the vestigial fp4 query-quantize round-trip.  The CP dense indexer quantizes
# the indexer query to fp4 (``_quantize_fp4_indexer_kernel``, a ~530ms/forward Triton
# launch) purely so the *real* deep_gemm ``fp8_fp4_mqa_logits`` could consume it.  On
# a build where that kernel is absent (deep_gemm is a stub, as on this XPU build) the
# torch fallback (``_dense_fp4_mqa_logits``) immediately dequantizes the fp4 query
# *back* to bf16 for its einsum -- the quantize is then a lossy round-trip that only
# degrades the query and costs a full kernel.  So this is ON BY DEFAULT exactly when
# the real kernel is absent -- no launcher env is needed, and a build that DOES ship
# the kernel keeps the quantize automatically (it is the same ``_deep_gemm_has_dense_fp4()``
# check that already selects the torch fallback).  Override the auto either way with
# ``SGLANG_KUNLUN_CP_BF16_INDEXER=1`` / ``=0``.
@functools.cache
def _bf16_idx_enabled() -> bool:
    override = os.environ.get("SGLANG_KUNLUN_CP_BF16_INDEXER")
    if override is not None:
        return override == "1"
    return not _deep_gemm_has_dense_fp4()


_LAST_Q_BF16 = None
_reported_bf16 = False

# Host-side ``(slot_offset, row_count, max_visible_len)`` plan for the *next*
# dense mqa call, one-shot.  The caller (``dsv41_topk_sort``) already knows the
# per-request window sizes on the host, so the epilogue can skip its
# ``unique_consecutive`` + per-group ``.item()`` device syncs entirely.
_LAST_EPI_PLAN = None


def _publish_epi_plan(lc_per_req, q_lens_cpu, total_slots, max_seqlen_k, num_tokens):
    """Publish the host row-group plan for the next dense mqa call.

    ``_low_ratio_index_topk_dense`` already knows every request's visible
    compressed length on the host, so the score path does not have to rediscover
    the row groups on the device (``unique_consecutive`` + two ``.tolist()``
    syncs per layer).  The plan window is the request's *newest* visible length:
    each row is clipped to its own ``lens`` inside the epilogue, so a window
    that is too wide only costs work, never correctness.  Nothing is published
    when the row counts do not tile the call exactly -- the AROUND hook would
    drop it anyway, and a leaked plan must never reach a different call.
    """
    global _LAST_EPI_PLAN
    plan, start, total = [], 0, 0
    for lc, cnt in zip(lc_per_req, q_lens_cpu):
        cnt = int(cnt)
        if cnt:
            plan.append((start, cnt, min(int(lc), total_slots - start, max_seqlen_k)))
            total += cnt
        start += int(lc)
    _LAST_EPI_PLAN = plan if total == num_tokens else None


# Every low-ratio indexer layer of a forward rebuilds the same per-row slot
# offset vector from the batch metadata (one pageable H2D + one
# ``repeat_interleave`` launch each); only ``ratio`` -- hence ``starts`` --
# varies, so the same two entries are rebuilt 8x per step.  The cached tensor is
# only ever read downstream.
_KS_CACHE = {}
_KS_CACHE_MAX = 8


def _row_slot_offsets(starts, q_lens_cpu, q_lens, num_tokens, device):
    lens_host = [int(x) for x in q_lens_cpu]
    if sum(lens_host) != num_tokens:
        # Not our batch shape: keep the literal upstream expression.
        return torch.repeat_interleave(
            torch.tensor(starts, dtype=torch.int32, device=device),
            q_lens.to(torch.int64),
            output_size=num_tokens,
        )
    key = (str(device), num_tokens, tuple(starts), tuple(lens_host))
    ks = _KS_CACHE.get(key)
    if ks is None:
        if len(_KS_CACHE) >= _KS_CACHE_MAX:
            _KS_CACHE.clear()
        ks = torch.repeat_interleave(
            torch.tensor(starts, dtype=torch.int32, device=device),
            torch.tensor(lens_host, dtype=torch.int64, device=device),
            output_size=num_tokens,
        )
        _KS_CACHE[key] = ks
    return ks


# Row blocking of one request's score rows.  ``_row_groups`` sizes a request's
# whole block by the window of its *newest* token, but a row's visible window is
# ``(pos + 1) // ratio`` -- with CP prefill the local rows of a request span the
# chunk, so that window is triangular and every early row pays for columns it
# cannot see.  Scoring fixed row blocks with each block's own max window keeps
# every row's arithmetic identical (a shorter column prefix, the rest left at the
# ``-inf`` the caller pre-fills) while roughly halving the score GEMM.
_SPLIT_ROWS = int(os.environ.get("KL_IDX_SPLIT_ROWS", "512"))
_SPLIT_MIN_LC = int(os.environ.get("KL_IDX_SPLIT_MIN_LC", "1024"))

# XPU's bf16 GEMM only takes its vectorised store path when the score tensor's
# row stride is a multiple of 16 elements.  Every live window is misaligned by
# construction -- ``min(max_lc, k_rows - start, max_seqlen_k)`` on a request
# length of 61433 / 2044 / ... -- and the fallback measures 1.30-1.39x slower
# than the same GEMM widened to the next multiple of 16 (2026-10-22 probes).
# Rounding the window up is safe on both sides: the extra columns sit past every
# row's ``lens`` so the epilogue masks them to ``-inf`` (the value the
# pre-filled ``logits`` already holds there), and the first ``max_lc`` columns
# of both the GEMM and the fused epilogue are bit-identical to the unpadded
# call at all four live window sizes.
_SCORE_NPAD = int(os.environ.get("KL_IDX_SCORE_NPAD", "16"))

# Round 47: apply that alignment to *every* block, not only the ones that fit
# inside the key cache and ``max_seqlen_k``.  A request's last block is the one
# that overruns ``k_rows`` (``start + ceil16(max_lc) > k_rows``) and, being the
# newest-token window, it is also the biggest score GEMM of the step -- so the
# old ``start + padded <= k_rows`` guard left exactly the headline kernel at
# the unaligned rate (fp16: 2710 us vs 630 us for N=61433 vs N=61440, i.e.
# 193 vs 848 GB/s of store bandwidth).  Two rows of zeros per 16 columns are
# appended to ``k`` so the oversized block reads real memory (scores there are
# exactly 0.0, and those columns are past every row's ``lens`` so the epilogue
# writes ``-inf`` over them), and the *written* width is clipped to the logits
# plane (``write_len < seq_len`` is explicitly supported by the op: columns
# >= write_len are untouched).  ``KL_IDX_SCORE_PAD_PAST_K=0`` restores round-46.
_SCORE_PAD_PAST_K = os.environ.get("KL_IDX_SCORE_PAD_PAST_K", "1") not in (
    "",
    "0",
    "false",
    "False",
)

# Round 84: call style of the score GEMM.  ``torch.matmul(q[r], kblk.T)`` treats
# the leading axis of ``q[r]`` (``[rows, H, 128]``) as a batch, so the vendor
# issues small ``[H, 128] x [128, lc]`` GEMMs; ``F.linear`` flattens the same
# operands to one ``[rows*H, 128] x [128, lc]`` GEMM -- identical arithmetic,
# different tile decomposition, and per-call 2x on this stack (measured on the
# 2026-10-22 pose: 674 -> 332 us).  fp16 is bit-identical (7/7 shapes
# ``torch.equal``); bf16 through the flat call is *slower and not bit-exact*
# (rel 3.5e-3), so the flat path is fp16-only and every other dtype keeps
# ``torch.matmul``.  ``KL_IDX_GEMM_LINEAR=0`` restores the batched call.
_SCORE_GEMM_LINEAR = os.environ.get("KL_IDX_GEMM_LINEAR", "1") not in (
    "",
    "0",
    "false",
    "False",
)


def _score_gemm(q_r, kblk, sdt):
    """``q_r @ kblk.T`` -> ``[rows, H, width]`` in ``sdt``, flattened when safe."""
    if _SCORE_GEMM_LINEAR and sdt is torch.float16:
        return torch.nn.functional.linear(q_r, kblk)
    return torch.matmul(q_r, kblk.transpose(0, 1))


def _plane_width(max_seqlen_k):
    """Physical row width: ``max_seqlen_k`` rounded up to the block size."""
    if _PLANE_ALIGN > 1:
        return (max_seqlen_k + _PLANE_ALIGN - 1) // _PLANE_ALIGN * _PLANE_ALIGN
    return max_seqlen_k


def _index_logits_init(num_tokens, max_seqlen_k, device):
    """Allocate the fp32 index-logits plane, ``-inf`` only where nothing writes.

    ``_fill_logit_tail`` fills ``[width, max_seqlen_k)`` for every scored row, so
    the whole plane ends up with exactly the values the old unconditional
    ``torch.full(-inf)`` produced -- but without writing 0.5 GiB that the block
    loop then overwrites column by column.  The row is rounded up to the
    candidate block size so the publisher can pool it without a copy (see
    ``_PLANE_ALIGN``).
    """
    width = _plane_width(max_seqlen_k)
    if _LOGITS_FULL:
        return torch.full(
            (num_tokens, width),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
    return torch.empty((num_tokens, width), dtype=torch.float32, device=device)


def _logit_tail_needed() -> bool:
    """Whether the ``-inf`` logit tail has to be materialised.

    A full-row top-k is the only consumer that reads past a row's window: the
    ragged kernels stop at ``seq_lens``, the candidate publish masks at block
    granularity (``col > last`` is -inf, ``col == last`` is forced kept -- so
    only blocks wholly inside ``[0, lens)`` contribute a value) and the consume
    path masks its own ``[0, lc)`` window.  With a ragged top-k present the
    columns past the scored prefix are never read, and filling them costs a
    fill + copy per block.
    """
    if _TAIL_PREFILL:
        return True
    return not (_xspeedgate_topk_ragged_v2_available() or _xpu_topk_ragged_available())


def _fill_logit_tail(logits, r0, r1, width, max_seqlen_k):
    """Sentinel-fill ``logits[r0:r1, width:]`` (row-chunked: XPU fills need room)."""
    if not _logit_tail_needed():
        return
    tail = max_seqlen_k - width
    if r1 <= r0 or tail <= 0:
        return
    step = max(1, _KEEP_BUDGET_BYTES // (tail * 4))
    for lo in range(r0, r1, step):
        hi = min(lo + step, r1)
        logits[lo:hi, width:].fill_(float("-inf"))


def _mask_logit_tail_inplace(scores, lens):
    """Restore the ``-inf`` tail a full-row top-k needs (cold fallback path).

    In place and row-chunked: the full-row top-k cannot tell a real score from
    an unwritten column, so every column past a row's window is masked to the
    sentinel the score chain would otherwise have written.  A ``lens.max()``
    guard would be wrong here: the widest row covering the plane says nothing
    about the narrower ones (the canonical 8x60k pose is exactly that), and a
    ``lens.min()`` guard would cost the same reduction as the mask itself.
    """
    rows, width = scores.shape
    if rows == 0 or width == 0:
        return
    cols = torch.arange(width, device=scores.device)[None, :]
    step = max(1, (_KEEP_BUDGET_BYTES // 4) // (width * 4))
    for lo in range(0, rows, step):
        hi = min(lo + step, rows)
        scores[lo:hi].masked_fill_(cols >= lens[lo:hi, None], float("-inf"))


def _score_width(max_lc, start, k_rows, max_seqlen_k):
    """Widen a score block to ``_SCORE_NPAD`` columns when both sides allow it."""
    if _SCORE_NPAD < 2:
        return max_lc
    padded = (max_lc + _SCORE_NPAD - 1) // _SCORE_NPAD * _SCORE_NPAD
    if _SCORE_PAD_PAST_K:
        # ``k`` is zero-padded past ``k_rows`` and the written width is clipped
        # to the logits plane, so the block may always round up.
        return padded
    if padded <= max_seqlen_k and start + padded <= k_rows:
        return padded
    return max_lc


def _score_blocks(ks, lens, k_rows, max_seqlen_k, plan):
    """Yield ``(slot_start, row_start, row_stop, window)`` per scoring block."""
    lens_host = None
    row0 = 0
    for start, cnt, max_lc in _row_groups(ks, lens, k_rows, max_seqlen_k, plan):
        base, row0 = row0, row0 + cnt
        if max_lc <= 0:
            # No visible keys: no block writes these rows, so they must be
            # sentinel-filled whole (the caller does that for ``window == 0``).
            yield start, base, base + cnt, 0
            continue
        if cnt > _SPLIT_ROWS and max_lc >= _SPLIT_MIN_LC:
            if lens_host is None:
                lens_host = lens.tolist()
            bounds = [
                (lo, min(lo + _SPLIT_ROWS, cnt))
                for lo in range(0, cnt, _SPLIT_ROWS)
            ]
        else:
            bounds = [(0, cnt)]
        for lo, hi in bounds:
            window = (
                max_lc
                if lens_host is None
                else min(max_lc, max(lens_host[base + lo : base + hi]))
            )
            if window <= 0:
                yield start, base + lo, base + hi, 0
                continue
            yield start, base + lo, base + hi, window


def _row_groups(ks, lens, k_rows, max_seqlen_k, plan):
    """Yield ``(slot_start, row_count, max_lc)`` per request window.

    Every row is accounted for exactly once (a group may carry ``max_lc <= 0``
    for tokens with no visible compressed positions; the caller skips those but
    must still advance its row cursor).  ``plan`` (host-computed, see
    ``_LAST_EPI_PLAN``) removes every device sync from this loop; without it the
    fallback costs one ``tolist`` instead of one ``.item()`` per group.
    """
    if plan is not None:
        for start, cnt, max_lc in plan:
            yield start, cnt, min(max_lc, k_rows - start, max_seqlen_k)
        return
    starts, counts = torch.unique_consecutive(ks, return_counts=True)
    lens_host = lens.tolist()
    row0 = 0
    for start, cnt in zip(starts.tolist(), counts.tolist()):
        max_lc = min(max(lens_host[row0 : row0 + cnt]), k_rows - start, max_seqlen_k)
        row0 += cnt
        yield start, cnt, max_lc


@functools.cache
def _deep_gemm_has_dense_fp4() -> bool:
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, "fp8_fp4_mqa_logits")


@functools.cache
def _xpu_topk_ragged_available() -> bool:
    try:
        from sglang.srt.utils import is_xpu

        if not is_xpu():
            return False
    except Exception:
        return False
    return hasattr(torch.ops.sgl_kernel, "topk_transform_ragged")


_announced_ops: set[str] = set()


def _announce_op(name: str) -> None:
    """Log the first *successful* use of each native op, once per process."""
    if name in _announced_ops:
        return
    _announced_ops.add(name)
    logger.info("sglang-kunlun: native %s active", name)


_reported_skip_sort = False


def _announce_skip_sort() -> None:
    """Log (once) that the ascending kernel output made the indexer sort redundant."""
    global _reported_skip_sort
    if _reported_skip_sort:
        return
    _reported_skip_sort = True
    logger.info(
        "sglang-kunlun: ascending ragged top-k active; skipping the indexer "
        "top-k sort"
    )


_topk_hole_calls = 0
_topk_hole_rows = 0
_topk_hole_total = 0


def _topk_hole_debug_enabled() -> bool:
    return os.environ.get("KL_TOPK_HOLE_DEBUG", "0") not in ("", "0")


def _count_topk_holes(selected: torch.Tensor, compress_lens=None, ks=None, ratio=None) -> None:
    """Count ``mask_topk_scores`` holes (``KL_TOPK_HOLE_DEBUG=1``).

    A hole is a ``-1`` that sits *before* a valid entry in the same row (the
    kernel's own padding is a valid prefix followed by ``-1``s, so plain
    ``selected < 0`` would count it).  ``mask_topk_scores`` turns a selected
    ``-inf`` position into such a ``-1``; the consumers expect padding at the
    tail, so the sort exists exactly for this case.  The counter syncs the
    device, hence opt-in only.
    """
    global _topk_hole_calls, _topk_hole_rows, _topk_hole_total
    chosen = selected >= 0
    width = selected.shape[1]
    pos = torch.arange(width, device=selected.device)
    last_valid = torch.where(chosen, pos, -1).max(dim=1).values
    invalid = (~chosen) & (pos[None, :] < last_valid[:, None])
    holes = int(invalid.sum())
    _topk_hole_calls += 1
    if holes:
        rows = int(invalid.any(dim=1).sum())
        _topk_hole_total += holes
        _topk_hole_rows += rows
        logger.info(
            "sglang-kunlun: [holes] mask punched %d entries in %d/%d rows",
            holes,
            rows,
            selected.shape[0],
        )
        if _topk_hole_calls <= 3:
            idx = invalid.any(dim=1).nonzero().flatten()[:8]
            per_row = invalid.sum(dim=1)[idx]
            info = {"rows": idx.tolist(), "holes_per_row": per_row.tolist(), "ratio": ratio}
            if compress_lens is not None:
                info["lens"] = compress_lens[idx].tolist()
            if ks is not None:
                info["ks"] = ks[idx].tolist()
            logger.info("sglang-kunlun: [holes] detail %s", info)
    if _topk_hole_calls % 200 == 0:
        logger.info(
            "sglang-kunlun: [holes] calls=%d holes=%d rows=%d",
            _topk_hole_calls,
            _topk_hole_total,
            _topk_hole_rows,
        )


@functools.cache
def _xspeedgate_op_available(name: str) -> bool:
    """Whether the installed XSpeedGate wheel registers ``torch.ops.xspeedgate_ops.<name>``.

    Two traps this avoids:

    * The namespace only exists after ``import xspeedgate_ops`` (the fragment
      registration lives in the extension's module init), so ``hasattr`` on a
      fresh process is False even though the wheel ships the op.  Importing a
      module that the engine already loaded is a cached no-op.
    * ``sglang.srt.utils.is_xpu()`` is **False** on this build (the XPU is
      presented as CUDA), which is why the upstream ``_xpu_topk_ragged_available``
      gate below never fires; gating an XSpeedGate op on it would make the op
      permanently dead.
    """
    try:
        import xspeedgate_ops  # noqa: F401 -- registers torch.ops.xspeedgate_ops
    except Exception:
        return False
    return hasattr(torch.ops.xspeedgate_ops, name)


@functools.cache
def _xspeedgate_topk_ragged_v2_available() -> bool:
    """XSpeedGate ships the ragged top-k under its Python-API name ``_v2``.

    Upstream's XPU branch calls ``torch.ops.sgl_kernel.topk_transform_ragged``,
    which this build's ``sgl_kernel`` does not bridge (only ``name`` is
    registered), so ``_xpu_topk_ragged_available()`` is False even though the
    native kernel is installed.  The XSpeedGate op takes the same XPU argument
    order (``scores, seq_lens, out_indices, out_offsets, row_starts``).
    """
    return _xspeedgate_op_available("topk_transform_ragged_v2")


# Local version tag of the delivery that changed ``topk_transform_ragged_v2``
# to emit every row in ascending window-position order (ties interleaved in
# lane order).  Older wheels share the same op signature but leave tie picks
# after each chunk's strictly-greater picks, so the indexer's trailing sort is
# only redundant at or above this tag.
_ASCENDING_TOPK_LOCAL_TAG = "xsgops20261020b"


@functools.cache
def _ascending_topk_ragged_v2_available() -> bool:
    """Whether the installed wheel's ragged top-k emits ascending rows.

    Gated on the wheel's local version tag: the op itself exposes no behaviour
    flag, and a caller that skips its own sort on a pre-fix wheel would publish
    unsorted rows.
    """
    if not _xspeedgate_topk_ragged_v2_available():
        return False
    try:
        from importlib.metadata import version

        installed = version("xspeedgate_ops")
    except Exception:  # noqa: BLE001 -- unknown provenance: keep the sort
        return False
    local = installed.split("+", 1)[1] if "+" in installed else ""
    if local.split(".", 1)[0] >= _ASCENDING_TOPK_LOCAL_TAG:
        return True
    # Newer wheels (1.7.0+) replace the xsgops<date> local tag with a commit
    # tag. Ascending emission landed long before 1.7.0, so any 1.7.0 build has
    # it; without this branch the caller would keep a redundant trailing sort.
    try:
        base = tuple(int(x) for x in installed.split("+", 1)[0].split(".")[:3])
    except ValueError:
        return False
    return base >= (1, 7, 0)


_T8_CACHE: dict = {}
_BYTE8_CACHE: dict = {}
_BYTE32_CACHE: dict = {}
_DEV_TABLE_CACHE: dict = {}
_POW2_BF16: dict = {}
# One gather on a 256-entry byte table instead of two nibble planes + an int64
# index copy.  ``KL_IDX_BYTE_LUT=0`` restores the nibble chain (A/B + guard).
_BYTE_LUT = os.environ.get("KL_IDX_BYTE_LUT", "1") not in ("", "0", "false", "False")
# Gather one int32 word per payload byte instead of two bf16 (see
# ``_bf16_byte_table32``); ``KL_IDX_BYTE_LUT32=0`` restores the pair gather.
_BYTE_LUT32 = os.environ.get("KL_IDX_BYTE_LUT32", "1") not in ("", "0", "false", "False")
_reported_byte_lut = False
# Pool the candidate blocks with ``Tensor.max`` instead of ``amax`` (see
# ``_pool_block_max``); ``KL_IDX_POOL_MAX=0`` restores the old form.
_POOL_MAX = os.environ.get("KL_IDX_POOL_MAX", "1") not in ("", "0", "false", "False")


def _dequant_table(table, device):
    """Device copy of the e2m1 table, cached per ``(device, table version)``.

    ``DSV4_DEQUANT_FP4_TABLE`` lives on the host, and copying it per call gave
    ``_bf16_dequant_table`` a fresh ``data_ptr`` every time, which missed both
    table caches and rebuilt the 256-entry byte table on every indexer layer
    (0.26 ms of host per call in the 8x60k trace).  Keeping the device copy
    alive makes the pointer stable; its contents are the same bytes.
    """
    key = (device, table.data_ptr(), table._version)
    hit = _DEV_TABLE_CACHE.get(key)
    if hit is None:
        if len(_DEV_TABLE_CACHE) > 4:
            _DEV_TABLE_CACHE.clear()
        hit = _DEV_TABLE_CACHE[key] = table.to(device)
    return hit


def _bf16_pow2_table(device):
    """256-entry ue8m0 exponent -> bf16 ``2^(e-127)``.

    Built with the expression the per-call chain used
    (``exp2(exps.float() - 127.0).to(torch.bfloat16)``), so every entry is
    bit-identical -- including ``e=0`` (the exact bf16 subnormal 2^-127) and
    ``e=255`` (inf) -- and one gather replaces the ``& 0xFF -> int32 -> float
    -> exp2 -> bf16`` launch chain.
    """
    hit = _POW2_BF16.get(device)
    if hit is None:
        exps = torch.arange(256, device=device, dtype=torch.int32)
        hit = torch.exp2(exps.float() - 127.0).to(torch.bfloat16)
        _POW2_BF16[device] = hit
    return hit


def _bf16_dequant_table(table):
    """bf16 copy of the e2m1 lookup table (16 entries, rebuilt on version bump).

    Every entry is exactly representable in bf16 (0, 0.5, 1, 1.5, 2, 3, 4, 6),
    so gathering from the bf16 table and multiplying by an exactly
    representable bf16 power of two reproduces the fp32 multiply bit for bit
    (checked for all 256 ue8m0 exponents; exponent 255 is inf/nan on both
    sides, and 0 lands on an exact bf16 subnormal).
    """
    key = (table.data_ptr(), table._version)
    hit = _T8_CACHE.get(key)
    if hit is not None:
        return hit
    _T8_CACHE.clear()
    _T8_CACHE[key] = hit = table.to(torch.bfloat16)
    return hit


def _bf16_byte_table(table):
    """256-entry table: one payload byte -> its two bf16 values ``[256, 2]``.

    A byte holds both nibbles of a pair (low nibble first), so indexing this
    table with the payload itself yields ``[n, 64, 2]`` in one gather -- no
    ``torch.stack`` of the two nibble planes and no int64 copy of every code
    (which alone was 8 bytes per code, 2.4x the payload).  Probed bit-exact
    against the nibble chain at the indexer's own shapes.
    """
    key = (table.data_ptr(), table._version)
    hit = _BYTE8_CACHE.get(key)
    if hit is not None:
        return hit
    t8 = _bf16_dequant_table(table)
    idx = torch.arange(256, device=t8.device)
    _BYTE8_CACHE.clear()
    _BYTE8_CACHE[key] = hit = torch.stack(
        [t8[idx & 0x0F], t8[idx >> 4]], dim=-1
    ).contiguous()
    return hit


def _bf16_byte_table32(table):
    """The same table as one int32 per byte: low half = low nibble's value.

    The gather above moves two bf16 per code; a table of 32-bit words moves one
    and holds the identical bytes (little-endian: ``view(int32)`` puts the first
    bf16 in the low half, so ``view(bfloat16)`` after the gather restores the
    ``[low, high]`` order the nibble chain produces).  Isolated at n=32768,
    238 -> 62 us/call; bit-exact against ``_bf16_byte_table``.
    """
    key = (table.data_ptr(), table._version)
    hit = _BYTE32_CACHE.get(key)
    if hit is not None:
        return hit
    _BYTE32_CACHE.clear()
    _BYTE32_CACHE[key] = hit = _bf16_byte_table(table).view(torch.int32).view(-1)
    return hit


def _dequant_indexer_fp4(payload, scales, table):
    """Packed index tensor -> bf16 rows.

    ``payload`` int8 [n, 64] holds one e2m1 code per nibble (low nibble = even
    element, high nibble = odd); ``scales`` int32 [n] packs four ue8m0 block
    exponents, one per 32-element group.

    The gather runs in bf16 and the scale is broadcast as one bf16 value per
    32-element group: the fp32 form materialised a ``[n, 128]`` fp32 scale
    through ``repeat_interleave`` and read/wrote twice the bytes for the same
    bits.  ``KL_IDX_BYTE_LUT=0`` restores the nibble-plane chain.
    """
    n = payload.shape[0]
    u = payload.view(torch.uint8)
    vals = None
    if _BYTE_LUT:
        try:
            if _BYTE_LUT32:
                word = _bf16_byte_table32(table)[u.to(torch.int32)]
                vals = word.view(torch.bfloat16)
            else:
                vals = _bf16_byte_table(table)[u.to(torch.int32)].flatten(1)
        except Exception as exc:  # noqa: BLE001 -- unusual byte/indices layout
            global _reported_byte_lut
            if not _reported_byte_lut:
                logger.warning(
                    "sglang-kunlun: fp4 byte-LUT dequant failed (%s); using the "
                    "nibble-plane chain", exc,
                )
                _reported_byte_lut = True
            vals = None
    if vals is not None:
        return _apply_indexer_scales(vals, scales, n)
    codes = torch.stack([u & 0x0F, (u >> 4) & 0x0F], dim=-1)  # [n, 64, 2]
    vals = _bf16_dequant_table(table)[codes.long()].flatten(1)  # [n, 128]
    return _apply_indexer_scales(vals, scales, n)


def _apply_indexer_scales(vals, scales, n):
    exps = scales.view(torch.uint8).to(torch.int32)  # [n * 4] ue8m0 bytes
    s4 = _bf16_pow2_table(scales.device)[exps].view(n, 4)  # [n, 4]
    return (vals.view(n, 4, 32) * s4.unsqueeze(-1)).view(n, 128)


_reported_epilogue = False

# Per-product rounding of the index-score epilogue (2026-10-23).  The torch chain
# rounds every clamped product to bf16 before summing over heads, and that round
# is 10 of the 16 SIMD ops the epilogue spends per 32-lane head slab -- the op's
# body, not its DMA, is the critical path.  Dropping it leaves the head sum in
# fp32, which is *more* accurate than the reference (2^-24 accumulation vs a 2^-9
# round per product): on the engine's real score distribution the mean abs error
# against an fp32-exact chain falls 6.87e-3 -> 4.17e-3 and the top-512 overlap
# against that chain rises 0.997694 -> 0.998068.  Measured at the prefill shape
# (fp16 scores, 256x32x32768): 575.9 -> 404.3 us/call, and the 60k end-to-end
# interleaved A/B (9 rounds/arm) 26773.6 -> 27175.7 tok/s.
# Wheels from 20261023a take the mode as a trailing argument; an older wheel
# raises on the probe and the call site keeps its 5-argument (RNE) form.
_MQA_ROUND_WANT = 2


@functools.cache
def _mqa_round_mode(device):
    """The rounding mode this wheel accepts, or None for the 5-arg call."""
    want = _MQA_ROUND_WANT
    if not _xspeedgate_op_available("mqa_logits_sum_heads"):
        return None
    try:
        scores = torch.zeros((1, 1, 8), dtype=torch.bfloat16, device=device)
        weights = torch.zeros((1, 1), dtype=torch.bfloat16, device=device)
        lens = torch.zeros((1,), dtype=torch.int64, device=device)
        out = torch.zeros((1, 8), dtype=torch.float32, device=device)
        torch.ops.xspeedgate_ops.mqa_logits_sum_heads(scores, weights, lens, out, 8, want)
        logger.info(
            "sglang-kunlun: index-score epilogue per-product rounding mode %d "
            "(fp32 head sum)", want,
        )
        return want
    except Exception as exc:  # noqa: BLE001 -- wheel has no round_mode argument
        logger.warning(
            "sglang-kunlun: mqa_logits_sum_heads takes no round_mode (%s); "
            "keeping the reference per-product rounding", exc,
        )
        return None


def _fused_mqa_epilogue(scores, weights, lens, out, write_len) -> bool:
    """Run the index-score epilogue on XSpeedGate when the wheel ships it.

    ``scores`` is the bf16 matmul output [R, H, L]; the op replaces
    ``clamp_min_(0) -> * weights -> sum(dim=1, fp32) -> .to(bf16).to(fp32) ->
    masked_fill(-inf)`` with one pass and writes columns [0, write_len) of
    ``out`` (pre-zeroed, rows aligned with ``scores``).  Returns False (caller
    keeps the torch chain) when the op is absent or the layout does not fit.
    """
    global _reported_epilogue
    if not _xspeedgate_op_available("mqa_logits_sum_heads"):
        return False
    if (
        scores.dtype not in (torch.bfloat16, torch.float16)
        or weights.dtype != torch.bfloat16
        or lens.dtype != torch.int64
        or out.dtype != torch.float32
        or not scores.is_contiguous()
        or not weights.is_contiguous()
        or not lens.is_contiguous()
        or not out.is_contiguous()
    ):
        return False
    try:
        mode = _mqa_round_mode(scores.device)
        if mode is None:
            torch.ops.xspeedgate_ops.mqa_logits_sum_heads(
                scores, weights, lens, out, int(write_len)
            )
        else:
            torch.ops.xspeedgate_ops.mqa_logits_sum_heads(
                scores, weights, lens, out, int(write_len), mode
            )
        _announce_op("mqa_logits_sum_heads")
        return True
    except Exception as exc:  # noqa: BLE001 -- fall back to the torch chain
        if not _reported_epilogue:
            logger.warning(
                "sglang-kunlun: xspeedgate_ops.mqa_logits_sum_heads failed (%s); "
                "using the torch index-score epilogue", exc,
            )
            _reported_epilogue = True
        return False


def _dense_fp4_mqa_logits_torch(
    q_fp4, kv_fp4, weights, ks, ke, max_seqlen_k, plan=None
):
    from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

    q_codes, q_sf = q_fp4
    k_codes, k_sf = kv_fp4
    num_tokens, num_heads, _ = q_codes.shape
    device = q_codes.device
    table = _dequant_table(DSV4_DEQUANT_FP4_TABLE, device)

    # ``-inf`` (not 0) in the <= 3 ceil-aligned padding columns beyond the group
    # max: the ragged top-k selects strictly inside each row's window, so a 0
    # there would outrank negative in-window scores in the full-row torch top-k
    # and silently drop up to 3 selected entries.
    if num_tokens == 0 or k_codes.shape[0] == 0:
        # Degenerate call: no block will run, so the plane must stay sentinel --
        # exactly the tensor the old unconditional ``torch.full`` returned.
        return torch.full(
            (num_tokens, _plane_width(max_seqlen_k)),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
    logits = _index_logits_init(num_tokens, max_seqlen_k, device)

    q = _dequant_indexer_fp4(
        q_codes.reshape(-1, q_codes.shape[-1]), q_sf.reshape(-1), table
    ).view(num_tokens, num_heads, -1)
    k = _dequant_indexer_fp4(k_codes, k_sf, table)  # [L, 128]
    sdt = _score_dtype(device)
    q = q.to(sdt).contiguous()
    k = k.to(sdt).contiguous()
    w = weights.to(torch.bfloat16)
    k_rows = k.shape[0]
    if _SCORE_PAD_PAST_K and _SCORE_NPAD > 1:
        # Zero rows past the key cache so the newest-token block (the largest
        # score GEMM of the step, and the one that overruns ``k_rows``) can
        # still round its columns up to ``_SCORE_NPAD``.  Their scores are
        # exactly 0.0 (dot with a zero key) and sit past every row's ``lens``,
        # so the epilogue writes ``-inf`` over them; the bytes are 16-31 rows
        # of zeros, versus the multi-hundred-MB score plane the block writes.
        extra = (-k_rows) % _SCORE_NPAD + _SCORE_NPAD
        k = torch.cat([k, k.new_zeros((extra, k.shape[1]))], dim=0)

    ks = ks.to(torch.long)
    lens = ke.to(torch.long) - ks  # per-row visible window
    for start, r0, r1, max_lc in _score_blocks(
        ks, lens, k_rows, max_seqlen_k, plan
    ):
        width = max(0, _score_width(max_lc, start, k_rows, max_seqlen_k))
        # ``width`` is the GEMM's column count; only ``wwidth`` of it exists in
        # the logits plane (the oversized tail block may overrun
        # ``max_seqlen_k`` as well as ``k_rows``).  The plane row is
        # block-aligned, so the sentinel fill covers the columns the score chain
        # never writes all the way to the physical row end.
        wwidth = min(width, max_seqlen_k)
        _fill_logit_tail(logits, r0, r1, wwidth, logits.size(1))
        if width <= 0:
            continue
        kblk = k[start : start + width]  # [width, 128]
        rows_per_chunk = max(1, _SCORE_BUDGET_BYTES // (num_heads * width * 4))
        for off in range(0, r1 - r0, rows_per_chunk):
            r = slice(r0 + off, r0 + min(off + rows_per_chunk, r1 - r0))
            s = _score_gemm(q[r], kblk, sdt)
            if s.dtype != sdt:
                s = s.to(sdt)
            if _fused_mqa_epilogue(s, w[r], lens[r], logits[r], wwidth):
                continue
            if wwidth != width:
                s = s[:, :, :wwidth]
            if s.dtype != torch.bfloat16:
                s = s.to(torch.bfloat16)
            s = s.clamp_min_(0.0)
            s = s * w[r].unsqueeze(-1)
            logit = s.sum(dim=1).to(torch.float32)
            # Built per fallback block rather than once per call: the fused
            # epilogue above is the path the engine takes and it never reads a
            # column index, so the full-width arange was dead work.
            cols = torch.arange(wwidth, device=device)
            logit = logit.masked_fill(
                cols[None, :] >= lens[r, None], float("-inf")
            )
            logits[r, :wwidth] = logit
    return logits


def _dense_bf16_mqa_logits_torch(
    q, kv_fp4, weights, ks, ke, max_seqlen_k, plan=None
):
    """``_dense_fp4_mqa_logits_torch`` with the query supplied directly as bf16.

    Identical to the fp4 variant except that ``q`` is the indexer's own
    ``[T, H, 128]`` bf16 query (no fp4 quantize -> dequantize round-trip); the k
    index cache is still dequantized from its stored fp4 layout.  The row/window
    chunking that keeps the score tensor within ``_SCORE_BUDGET_BYTES`` is kept so
    the long-context path cannot OOM.
    """
    from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

    # R1 port: one native int8 kernel (fp4-K dequant + int8 dot + relu + weighted head
    # sum + causal window mask) replaces the dequant -> bf16 score GEMM -> epilogue
    # chain below.  Env ``SGLANG_KUNLUN_CP_NATIVE_INDEXER=1``; any failure latches back
    # to the torch body for the rest of the process.  ``plan`` only feeds the epilogue,
    # so the native path ignores it (and leaves ``_LAST_EPI_PLAN`` untouched).
    global _native_mqa_disabled, _reported_native_mqa
    if (
        not _native_mqa_disabled
        and indexer_i8_core.enabled_from_env()
        and indexer_i8_core.available()
    ):
        try:
            table_dev = DSV4_DEQUANT_FP4_TABLE.to(q.device)
            out = indexer_i8_core.dense_mqa_logits_i8(
                q,
                kv_fp4,
                weights,
                ks,
                ke,
                max_seqlen_k,
                dequant_fn=lambda c, sf: _dequant_indexer_fp4(c, sf, table_dev),
            )
            if not _reported_native_mqa:
                logger.warning(
                    "sglang-kunlun: native I8_mqa_logits engaged for the prefill-CP "
                    "dense indexer (R1 port); the torch dequant+GEMM+epilogue chain "
                    "is bypassed"
                )
                _reported_native_mqa = True
            return out
        except Exception as exc:  # noqa: BLE001 -- latch off, keep the torch path
            _native_mqa_disabled = True
            logger.warning(
                "sglang-kunlun: native I8 dense MQA-logits failed (%s); falling back "
                "to the torch bf16 path and disabling native for this run", exc,
            )

    k_codes, k_sf = kv_fp4
    num_tokens, num_heads, _ = q.shape
    device = q.device
    table = _dequant_table(DSV4_DEQUANT_FP4_TABLE, device)

    # ``-inf`` (not 0) in the <= 3 ceil-aligned padding columns beyond the group
    # max: the ragged top-k selects strictly inside each row's window, so a 0
    # there would outrank negative in-window scores in the full-row torch top-k
    # and silently drop up to 3 selected entries.
    if num_tokens == 0 or k_codes.shape[0] == 0:
        # Degenerate call: no block will run, so the plane must stay sentinel --
        # exactly the tensor the old unconditional ``torch.full`` returned.
        return torch.full(
            (num_tokens, _plane_width(max_seqlen_k)),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
    logits = _index_logits_init(num_tokens, max_seqlen_k, device)

    q = q.to(torch.bfloat16)
    k = _dequant_indexer_fp4(k_codes, k_sf, table)  # [L, 128]
    sdt = _score_dtype(device)
    q = q.to(sdt).contiguous()
    k = k.to(sdt).contiguous()
    w = weights.to(torch.bfloat16)
    k_rows = k.shape[0]
    if _SCORE_PAD_PAST_K and _SCORE_NPAD > 1:
        # Zero rows past the key cache so the newest-token block (the largest
        # score GEMM of the step, and the one that overruns ``k_rows``) can
        # still round its columns up to ``_SCORE_NPAD``.  Their scores are
        # exactly 0.0 (dot with a zero key) and sit past every row's ``lens``,
        # so the epilogue writes ``-inf`` over them; the bytes are 16-31 rows
        # of zeros, versus the multi-hundred-MB score plane the block writes.
        extra = (-k_rows) % _SCORE_NPAD + _SCORE_NPAD
        k = torch.cat([k, k.new_zeros((extra, k.shape[1]))], dim=0)

    ks = ks.to(torch.long)
    lens = ke.to(torch.long) - ks  # per-row visible window
    for start, r0, r1, max_lc in _score_blocks(
        ks, lens, k_rows, max_seqlen_k, plan
    ):
        width = max(0, _score_width(max_lc, start, k_rows, max_seqlen_k))
        # ``width`` is the GEMM's column count; only ``wwidth`` of it exists in
        # the logits plane (the oversized tail block may overrun
        # ``max_seqlen_k`` as well as ``k_rows``).  The plane row is
        # block-aligned, so the sentinel fill covers the columns the score chain
        # never writes all the way to the physical row end.
        wwidth = min(width, max_seqlen_k)
        _fill_logit_tail(logits, r0, r1, wwidth, logits.size(1))
        if width <= 0:
            continue
        kblk = k[start : start + width]  # [width, 128]
        rows_per_chunk = max(1, _SCORE_BUDGET_BYTES // (num_heads * width * 4))
        for off in range(0, r1 - r0, rows_per_chunk):
            r = slice(r0 + off, r0 + min(off + rows_per_chunk, r1 - r0))
            s = _score_gemm(q[r], kblk, sdt)
            if s.dtype != sdt:
                s = s.to(sdt)
            if _fused_mqa_epilogue(s, w[r], lens[r], logits[r], wwidth):
                continue
            if wwidth != width:
                s = s[:, :, :wwidth]
            if s.dtype != torch.bfloat16:
                s = s.to(torch.bfloat16)
            s = s.clamp_min_(0.0)
            s = s * w[r].unsqueeze(-1)
            logit = s.sum(dim=1).to(torch.float32)
            # Built per fallback block rather than once per call: the fused
            # epilogue above is the path the engine takes and it never reads a
            # column index, so the full-width arange was dead work.
            cols = torch.arange(wwidth, device=device)
            logit = logit.masked_fill(
                cols[None, :] >= lens[r, None], float("-inf")
            )
            logits[r, :wwidth] = logit
    return logits


def _topk_transform_ragged_v2_torch(
    scores, seq_lens, *, out_offsets, out_indices, row_starts=None
):
    """Torch equivalent of the fused ragged top-k (see the module docstring).

    Row ``i`` selects the top-k of ``scores[i, start : start + seq_lens[i]]`` and
    writes ``position + out_offsets[i]`` into ``out_indices[i]``, ``-1`` padded.
    ``out_indices`` is written in place (the kernel returns ``None`` too).
    """
    num_rows, width = scores.shape
    topk = out_indices.shape[1]
    device = scores.device
    out_indices.fill_(-1)
    k = min(int(topk), width)
    if num_rows == 0 or k == 0:
        return
    lens = seq_lens.to(torch.long)
    if row_starts is None:
        start = torch.zeros(num_rows, dtype=torch.long, device=device)
    else:
        start = row_starts.to(torch.long)
    # ``scores`` carries -inf past each row's window on the CP path unless the
    # score chain skipped the sentinel fill (``KL_IDX_LOGIT_TAIL=auto`` and a
    # ragged top-k installed, see ``_logit_tail_needed``).  This full-row top-k
    # is the one consumer that reads those columns, so restore the invariant in
    # place when the fill was skipped -- and only then, so a box without a ragged
    # top-k does not pay a second full-plane pass.  Row-chunked and in place:
    # the full ``[num_rows, width]`` copy OOMs at chunk 65536 (1.84 GiB).  Only
    # the row_starts variant needs a window mask.
    if row_starts is None:
        if not _logit_tail_needed():
            _mask_logit_tail_inplace(scores, lens)
        sel = scores.topk(k, dim=-1, sorted=True).indices
    else:
        cols = torch.arange(width, device=device)[None, :]
        window = (cols >= start[:, None]) & (cols < (start + lens)[:, None])
        sel = (
            scores.masked_fill(~window, float("-inf"))
            .topk(k, dim=-1, sorted=True)
            .indices
        )
    # ``sorted=True`` ranks the finite (in-window) columns ahead of the -inf tail,
    # so the in-window entries land in the leading slots and ``-1`` pads the rest,
    # matching the kernel's documented layout.
    abs_idx = sel + out_offsets.to(torch.long)[:, None]
    # ``in_window`` is ``window[i, sel[i, :]]``.  The XPU build's ``gather_2d<bool,
    # int>`` (the kernel ``torch.gather(window, 1, sel)`` lowers to, since ``window``
    # is bool) faults with ``load/store operation exceed memory size`` once the CP
    # prefill reaches seq position 32768 -- a genuine device fault from a valid
    # index, i.e. a vendor kernel bug, not a bad ``sel``.  Recomputing the same
    # predicate directly from ``sel`` avoids the bool gather entirely.
    in_window = (sel >= start[:, None]) & (sel < (start + lens)[:, None])
    out_indices[:, :k] = torch.where(
        in_window, abs_idx, torch.full_like(abs_idx, -1)
    ).to(torch.int32)


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend._dense_fp4_mqa_logits",
    type=HookType.AROUND,
)
def _dense_fp4_mqa_logits_kunlun(
    original_fn, q_fp4, kv_fp4, weights, ks, ke, max_seqlen_k
):
    global _reported, _LAST_EPI_PLAN
    # One-shot host plan from the caller; drop it whatever path we take so it
    # can never leak into an unrelated call.
    plan = _LAST_EPI_PLAN
    _LAST_EPI_PLAN = None
    if plan is not None and sum(cnt for _s, cnt, _l in plan) != q_fp4[0].shape[0]:
        plan = None
    if _deep_gemm_has_dense_fp4():
        return original_fn(q_fp4, kv_fp4, weights, ks, ke, max_seqlen_k)
    if not _reported:
        logger.warning(
            "sglang-kunlun: deep_gemm lacks fp8_fp4_mqa_logits; using the torch "
            "dense fp4 index-logits fallback for the prefill-CP indexer"
        )
        _reported = True
    if _bf16_idx_enabled() and _LAST_Q_BF16 is not None:
        q_codes = q_fp4[0]
        num_tokens, num_heads = q_codes.shape[0], q_codes.shape[1]
        if _LAST_Q_BF16.numel() == num_tokens * num_heads * q_codes.shape[-1] * 2:
            q_bf16 = _LAST_Q_BF16.reshape(num_tokens, num_heads, -1)
            return _dense_bf16_mqa_logits_torch(
                q_bf16, kv_fp4, weights, ks, ke, max_seqlen_k, plan
            )
    return _dense_fp4_mqa_logits_torch(
        q_fp4, kv_fp4, weights, ks, ke, max_seqlen_k, plan
    )


def _try_topk_ragged_v2_native(
    scores, seq_lens, *, out_offsets, out_indices, row_starts=None
) -> bool:
    """Run the XSpeedGate ragged top-k; True iff it ran.

    The wheel's ``topk_transform_ragged_v2`` emits every row in ascending
    window-position order (``-1`` padding last), so a caller that would
    otherwise sort the output can skip that sort when this returns True.
    Returns False when the op is absent, the layout does not fit, or it
    raised -- the caller keeps its own fallback.
    """
    global _reported_topk
    topk = int(out_indices.shape[1])
    if not (
        _xspeedgate_topk_ragged_v2_available()
        and topk in (512, 1024, 2048)
        and scores.dtype == torch.float32
        and scores.is_contiguous()
        and out_indices.dtype == torch.int32
        and out_indices.stride(1) == 1
    ):
        return False
    try:
        torch.ops.xspeedgate_ops.topk_transform_ragged_v2(
            scores,
            seq_lens.to(torch.int32).contiguous(),
            out_indices,
            out_offsets.to(torch.int32).contiguous(),
            None if row_starts is None else row_starts.to(torch.int32).contiguous(),
        )
        _announce_op("topk_transform_ragged_v2")
        return True
    except Exception as exc:  # noqa: BLE001 -- fall back to the torch chain
        if not _reported_topk:
            logger.warning(
                "sglang-kunlun: xspeedgate_ops.topk_transform_ragged_v2 failed "
                "(%s); using the torch ragged top-k fallback", exc,
            )
            _reported_topk = True
        return False


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend.topk_transform_ragged_v2",
    type=HookType.AROUND,
)
def _topk_transform_ragged_v2_kunlun(
    original_fn, scores, seq_lens, *, out_offsets, out_indices, row_starts=None
):
    global _reported_topk
    if _try_topk_ragged_v2_native(
        scores,
        seq_lens,
        out_offsets=out_offsets,
        out_indices=out_indices,
        row_starts=row_starts,
    ):
        return None
    if _xpu_topk_ragged_available():
        return original_fn(
            scores,
            seq_lens,
            out_offsets=out_offsets,
            out_indices=out_indices,
            row_starts=row_starts,
        )
    if not _reported_topk:
        logger.warning(
            "sglang-kunlun: no XPU-native topk_transform_ragged; using the torch "
            "ragged top-k fallback for the prefill-CP indexer"
        )
        _reported_topk = True
    return _topk_transform_ragged_v2_torch(
        scores,
        seq_lens,
        out_offsets=out_offsets,
        out_indices=out_indices,
        row_starts=row_starts,
    )


_KREAD_ROWS = os.environ.get("KL_IDX_KREAD_ROW", "1") not in ("", "0", "false", "False")


def _fp4_rows_rowwise(buf, page_size, slots):
    """Gather the packed fp4 rows with one row index per token.

    The fp4 page row is ``page_size * 68`` uint8 -- ``page_size * 64`` payload
    bytes then ``page_size * 4`` scale bytes -- and a token's payload is 64
    contiguous bytes.  With ``page_size`` a multiple of 16 (64 and 128 here)
    the buffer divides into 64-byte rows, so ``buf.view(-1, 64)[row]`` replaces
    the upstream ``arange(64)`` broadcast + ``buf[page, cols]`` advanced index.
    That index builds an [n, 64] int64 tensor and runs it through the
    gather-index conversion, which is what dominates the readback in the 8x60k
    trace (``get_index_k_fp4`` 58.5 ms / 64 calls, of which
    ``index_select_mt`` 306.6 us, ``tensor_calc_n1_1k`` 197.2,
    ``gathernd_idx_to_gather_idx`` 189.3, ``broadcast_mtn_kl3`` 105.8 per call).

    Scales are one int32 word per token at a 16-aligned word offset, so the
    int32 view takes the same page/offset arithmetic as the store.

    Both gathers are ``index_select(0, ...)`` instead of ``t[rows]``:
    advanced indexing lowers to ``gathernd_idx_to_gather_idx`` +
    ``broadcast_mtn_kl3`` + ``index_select_mt`` here and costs 137 us of *host*
    dispatch per event -- 17.5 ms per 8x60k capture, the largest host item of
    the indexer frame.  ``index_select`` reaches the same device kernel in one
    dispatch (bit-identical: same kernel, one fewer index-format conversion).

    Returns ``None`` when the layout is not the 68-byte fp4 one.
    """
    if page_size % 16 or buf.dtype is not torch.uint8:
        return None
    slots = slots.to(torch.int64)
    page = slots // page_size
    off = slots - page * page_size
    words_per_page = page_size * 17
    rows_per_page = words_per_page // 16
    payload = (
        buf.view(torch.int8)
        .view(-1, 64)
        .index_select(0, page * rows_per_page + off)
    )
    scales = buf.view(torch.int32).view(-1).index_select(
        0, page * words_per_page + page_size * 16 + off
    )
    return payload, scales


def _read_index_k_fp4_rowwise(pool, layer_id, slots):
    """``pool.get_low_ratio_index_k_fp4`` through :func:`_fp4_rows_rowwise`.

    The buffer is fetched exactly as the upstream method fetches it (no
    ``wait_layer_transfer``: this is the same layer the store just wrote, and
    the upstream readback does not wait either).  ``None`` means "use the
    upstream method".
    """
    if not _KREAD_ROWS:
        return None
    if not (hasattr(pool, "layer_mapping") and hasattr(pool, "_indexer_pool")):
        return None
    compress_ratio, compress_layer_id, _ = pool.layer_mapping[layer_id]
    ipool = pool._indexer_pool(compress_ratio)
    buf = ipool.index_k_with_scale_buffer[compress_layer_id - ipool.start_layer]
    if buf is None:
        return None
    return _fp4_rows_rowwise(buf, ipool.page_size, slots)


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend._low_ratio_index_topk_dense",
    type=HookType.REPLACE,
)
def _low_ratio_index_topk_dense_kunlun(
    self, layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu
) -> None:
    """Pre-fill-CP dense indexer, minus the post-top-k sort.

    Mirrors the upstream ``DeepseekV4AttnBackend._low_ratio_index_topk_dense``;
    the single behavioural difference is the tail.  Upstream sorts the ragged
    top-k output because the kernel only guarantees the selected *set* (tie
    positions used to be appended after each chunk's strictly-greater picks).
    The XSpeedGate kernel now emits every row already ascending, so the sort
    is a pure permutation -- skipped here.  The sort still runs when the native
    kernel did not execute.

    The candidate-mask branch skips it too.  ``mask_topk_scores`` can clear an
    entry only when its score is ``-inf``, i.e. when a row has fewer than topk
    finite candidates; the published masks keep every finite block, i.e. at
    least ``min(16384, lens)`` candidates versus the ``min(topk, lens)`` entries
    the kernel selects, so it only ever re-clears the kernel's own ``-1`` tail.
    Measured over a 61k-token / 30-request sweep: 0 mid-row ``-1`` in every
    call.  ``KL_TOPK_HOLE_DEBUG=1`` re-runs that check on live traffic.
    """
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        quantize_fp4_indexer_tensor,
    )
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _as_int_list,
        _dense_fp4_mqa_logits,
        ceil_align,
        topk_transform_ragged_v2,
    )
    from sglang.srt.layers.attention.dsv4.candidate_indexer import CandidateMasks

    pool = self.token_to_kv_pool
    core = self.forward_metadata.core_metadata
    ratio = layer.compress_ratio
    indexer = layer.indexer
    page_indices = core.sparse_page_indices(ratio)
    raw_indices = core.sparse_raw_indices(ratio)
    page_indices.fill_(-1)
    if raw_indices is not None:
        raw_indices.fill_(-1)

    seq_lens_cpu = _as_int_list(forward_batch.seq_lens_cpu)
    assert seq_lens_cpu is not None
    device = pos.device
    # Visible compressed positions per request at its newest token; the
    # per-token count (pos + 1) // ratio bounds each row below.
    lc_per_req = [s // ratio for s in seq_lens_cpu]
    # Host row ids, not a device 0-dim tensor: ``req_to_token[device_scalar]``
    # makes ATen read the index back as a python int (two stream syncs per
    # request -- 8/forward at -w 4, the 60 ``aten::item`` pairs the sync ledger
    # sees in this frame).  ``req_pool_indices_cpu`` is the scheduler's host
    # mirror (padded in lock-step with the device tensor); the ``tolist``
    # fallback still costs one sync per call instead of two per request.
    req_cpu = getattr(forward_batch, "req_pool_indices_cpu", None)
    if req_cpu is not None and req_cpu.numel() >= len(lc_per_req):
        req_ids = req_cpu.tolist()
    else:
        req_ids = forward_batch.req_pool_indices.tolist()
    slot_chunks, starts, start = [], [], 0
    for r, lc in enumerate(lc_per_req):
        starts.append(start)
        if lc == 0:
            continue
        # Strided row slice instead of ``arange(lc) * ratio`` + a gather: the
        # computed index tensor costs a full int64 index build plus a
        # random-access select per request.  ``row[0 : (lc-1)*ratio+1 : ratio]``
        # addresses exactly the same elements.  1.85x at both live shapes
        # (1 x 61436 slots: 304 -> 164 us, 4 x 15359: 1137 -> 612 us,
        # torch.equal on the int64 result).
        row = self.req_to_token[req_ids[r]]
        slot_chunks.append(
            (row[: (lc - 1) * ratio + 1 : ratio] // ratio).to(torch.int64)
        )
        start += lc
    empty_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
    num_tokens = pos.shape[0]
    if not slot_chunks or num_tokens == 0:
        if indexer.is_candidate_source:
            self.forward_metadata.candidate_metadata = CandidateMasks(
                request_masks=[empty_mask for _ in lc_per_req]
            )
        return
    k_slots = torch.cat(slot_chunks)
    read = _read_index_k_fp4_rowwise(pool, layer.layer_id, k_slots)
    if read is None:
        read = pool.get_low_ratio_index_k_fp4(layer.layer_id, k_slots)
    k_fp4, k_sf = read

    q = indexer.queries(q_lora, layer.freqs_cis[pos])  # [T, H, 128] fp4 grid
    num_heads = q.shape[1]
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    q_fp4 = q_fp4.view(num_tokens, num_heads, 64)
    q_sf = q_sf.view(num_tokens, num_heads)
    weights = indexer.head_weights(x).float()
    compress_lens = ((pos + 1) // ratio).to(torch.int32)
    ks = _row_slot_offsets(starts, q_lens_cpu, q_lens, num_tokens, device)
    # the fused top-k reads score rows through 16-byte vectors
    epi_width = ceil_align(max(lc_per_req), 4)
    _publish_epi_plan(lc_per_req, q_lens_cpu, k_slots.numel(), epi_width, num_tokens)
    logits = _dense_fp4_mqa_logits(
        (q_fp4, q_sf),
        (k_fp4, k_sf),
        weights,
        ks,
        ks + compress_lens,
        epi_width,
    )
    if indexer.is_candidate_source or indexer.uses_candidates:
        self._publish_or_consume_candidates(
            indexer, logits, compress_lens, lc_per_req, q_lens_cpu, empty_mask
        )
    topk = indexer.index_topk
    selected = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    ran_native = _try_topk_ragged_v2_native(
        logits, compress_lens, out_offsets=ks, out_indices=selected
    )
    if not ran_native:
        topk_transform_ragged_v2(
            logits, compress_lens, out_offsets=ks, out_indices=selected
        )
    ascending = ran_native and _ascending_topk_ragged_v2_available()
    if indexer.uses_candidates and not indexer.is_candidate_source:
        # No mask_topk_scores in the normal pose: it clears a selected entry
        # only when its score is -inf, i.e. when the row has fewer than topk
        # finite candidates.  The candidate mask keeps >= min(
        # candidate_topk_blocks * candidate_block_size, lens) finite positions
        # against the min(index_topk, lens) the kernel selects, so it could only
        # re-clear the kernel's own -1 padding.
        #
        # That guarantee is exactly "candidate capacity >= index_topk".  A pose
        # whose block capacity is smaller would make the kernel fill the tail
        # from -inf positions; those must become -1 before they are mapped to a
        # page/raw index, or the attention reads tokens outside the candidate
        # set.  Keep the clear for that pose instead of assuming it away.
        capacity = indexer.candidate_topk_blocks * indexer.candidate_block_size
        if capacity < indexer.index_topk:
            from sglang.srt.layers.attention.dsv4.candidate_indexer import (
                mask_topk_scores,
            )

            selected = mask_topk_scores(logits, selected, ks)
            # Clearing entries leaves holes, so the native ascending order no
            # longer holds; take the sort branch below.
            ascending = False
        elif _topk_hole_debug_enabled():
            _count_topk_holes(selected, compress_lens, ks, ratio)
    if ascending:
        chosen = selected >= 0
        _announce_skip_sort()
    else:
        # ascending positions, padding last: the layout the consumers expect
        unselected = torch.iinfo(torch.int32).max
        selected = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values
        chosen = selected != unselected
    page_indices[:num_tokens, :topk] = torch.where(
        chosen, k_slots[selected.clamp_max(k_slots.shape[0] - 1)], -1
    ).to(torch.int32)
    if raw_indices is not None:
        raw_indices[:num_tokens, :topk] = torch.where(
            chosen, selected - ks[:, None], -1
        )


_CANDIDATE_BLOCK_MASKS_CLS = None


def _candidate_block_masks_cls():
    """``CandidateMasks`` subclass carrying block-level prefill masks.

    ``request_neg_masks`` materialises the per-position form on first use (the
    first consume layer of the forward) and caches it: expanding at consume
    time instead of publish time keeps the publish side free of the full-width
    bool tensor, and the expansion itself is bandwidth-bound either way.  The
    cached form is the *negated* mask (``True`` = drop), because that is what
    the consume layers feed to ``masked_fill_``; negating the 8x smaller block
    mask before the expansion costs 8x less than negating a position mask once
    per consume layer.
    """
    global _CANDIDATE_BLOCK_MASKS_CLS
    if _CANDIDATE_BLOCK_MASKS_CLS is None:
        from sglang.srt.layers.attention.dsv4.candidate_indexer import (
            CandidateMasks,
        )

        class _CandidateBlockMasks(CandidateMasks):
            def __init__(self, blocks, widths, block_size):
                self.blocks = blocks
                self.widths = widths
                self.block_size = block_size
                self._request_masks = None
                self._request_neg_masks = None

            def _expand(self, negate: bool):
                bs = self.block_size
                out = []
                for m, w in zip(self.blocks, self.widths):
                    if negate:
                        m = ~m
                    out.append(m.T.repeat_interleave(bs, dim=0).T.contiguous()[..., :w])
                return out

            @property
            def request_neg_masks(self):
                """Per-position drop masks (``True`` = masked), expanded once.

                ``blocks`` carries one bool per ``block_size`` positions, so the
                expansion is ``repeat_interleave`` truncated to each request's
                width (``widths``).  Only the consume layers that read positions
                pay for it, once per forward, which keeps the full-width bool
                tensor off the publish side.

                ``repeat_interleave`` on the last dim repeats *single bytes*: on
                this XPU that kernel runs at ~18 GB/s (12.3 ms for a
                ``[4096, 61440]`` bool mask), while a plain contiguous copy runs
                at ~900 GB/s.  Repeating on the transposed row dim moves 4 KB
                rows instead and the transpose back is one full-rate contiguous
                copy, so the same tensor costs 0.66 ms instead of 12.3 ms.
                """
                if self._request_neg_masks is None:
                    if self._request_masks is not None:  # assigned by a publisher
                        self._request_neg_masks = [~m for m in self._request_masks]
                    else:
                        self._request_neg_masks = self._expand(negate=True)
                return self._request_neg_masks

            @property
            def request_masks(self):
                """Kept positions, ``True`` = keep (the upstream interface)."""
                if self._request_masks is None:
                    self._request_masks = [~m for m in self.request_neg_masks]
                return self._request_masks

            @request_masks.setter
            def request_masks(self, value):
                self._request_masks = value
                self._request_neg_masks = None

        _CANDIDATE_BLOCK_MASKS_CLS = _CandidateBlockMasks
    return _CANDIDATE_BLOCK_MASKS_CLS


def _pool_block_max(blocks):
    """Max over the last dim of a ``[..., nb, block_size]`` view.

    ``Tensor.max(dim=-1)`` runs the wide reduce faster than ``amax(dim=-1)`` on
    this XPU -- a plain copy of the same 268 MB plane is ~860 GB/s while both
    reduces sit below it.  Isolated at the publisher's own shapes (fp32):
    contiguous 2048x32768 amax 1225 -> max 787 us; strided body of
    1092x61436 (row stride 61436, 61432 pooled columns) amax 1523 -> 1091 us.
    The values are identical by definition (max is exact for any reduction
    order), so this is a kernel-choice change only.  ``KL_IDX_POOL_MAX=0``
    restores ``amax``; an unusual layout falls back to it too.
    """
    if _POOL_MAX:
        try:
            return blocks.max(dim=-1).values
        except Exception:  # noqa: BLE001 -- layout the max kernel rejects
            pass
    return blocks.amax(dim=-1)


_plane_checked = False


def _pool_request_blocks(scores, lens, topk_blocks, block_size):
    """Block keep mask for one request, row-chunked under the workspace budget."""
    step = max(1, _KEEP_BUDGET_BYTES // (scores.size(1) * 4))
    t_len = scores.size(0)
    blocks = [
        _candidate_block_keep(
            scores[start : start + step],
            lens[start : start + step],
            topk_blocks,
            block_size,
        )
        for start in range(0, t_len, step)
    ]
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks)


def _announce_plane(plane_width, lc, block_size, mask, reference):
    """One-shot "I am live" marker and bitwise gate for the aligned plane.

    The first publish pools the block-aligned prefix *and* the pre-r133
    ``:lc`` slice and compares the two keep masks.  One-shot, so the steady
    state pays nothing; a mismatch logs a warning and switches this call (and
    every later one, via ``_plane_checked``) back to the reference mask.
    """
    global _plane_checked
    if _plane_checked:
        return mask
    _plane_checked = True
    ok = mask is reference or torch.equal(mask, reference)
    logger.info(
        "sglang-kunlun: candidate publish pools the block-aligned plane "
        "(plane width %d, request lc %d, block %d, aligned=%s, bitwise=%s)",
        plane_width,
        lc,
        block_size,
        plane_width % block_size == 0,
        ok,
    )
    if not ok:
        logger.warning(
            "sglang-kunlun: aligned-plane candidate mask differs from the "
            "positional reference; using the reference (KL_IDX_PLANE_ALIGN8=0 "
            "disables the aligned plane)"
        )
        return reference
    return mask


def _candidate_block_keep(scores, lens, topk_blocks, block_size):
    """``select_candidate_blocks`` up to the block-level keep mask ``[rows, nb]``.

    Same pooling and top-k, but positions past a row's length are masked at
    block granularity (``col > last`` is -inf; ``col == last`` is forced kept),
    which pools to the same values as upstream's materialised
    ``arange >= compress_lens`` position mask, and the mask is not expanded back
    to one bool per position.
    """
    width = scores.size(-1)
    # Pool the whole blocks with a view and reduce the trailing partial block on
    # its own.  ``F.pad`` here would copy every score element (~0.5-1 GiB at
    # chunk 32768/65536) just to append <block_size -inf columns; the trailing
    # reduction has the same value because a max over fewer real elements equals
    # a max over the same elements padded with -inf.
    full = width // block_size
    tail_from = full * block_size
    if full:
        # ``unflatten`` (not ``view``) so a strided plane -- a row window of the
        # wide logits tensor -- still splits without a copy; splitting a
        # contiguous axis is always an expressible view.  When the caller passes
        # the whole physical row and that row is a multiple of ``block_size``
        # (``_plane_width``), ``tail_from == width`` and this slice *is* the row,
        # so the split is a view and the reduce reads the plane in place.
        pooled = _pool_block_max(
            scores[..., :tail_from].unflatten(-1, (full, block_size))
        )
        if tail_from < width:
            pooled = torch.cat(
                (pooled, scores[..., tail_from:].amax(dim=-1, keepdim=True)), dim=-1
            )
    else:
        pooled = scores.amax(dim=-1, keepdim=True)
    num_blocks = pooled.size(-1)
    cols = torch.arange(num_blocks, device=scores.device)
    last = (lens - 1) // block_size
    pooled = pooled.masked_fill(cols > last, -torch.inf)
    pooled = pooled.masked_fill(cols == last, torch.inf)
    top = pooled.topk(min(topk_blocks, num_blocks), dim=-1, sorted=False)
    # ``sorted=False`` returns the same index set (checked against the sorted
    # form) at half the kernel time; only the set is consumed below.
    return torch.zeros_like(pooled, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend."
    "_publish_or_consume_candidates",
    type=HookType.REPLACE,
)
def _publish_or_consume_candidates_kunlun(
    self, indexer, logits, compress_lens, lc_per_req, q_lens_cpu, empty_mask
) -> None:
    """Publish the candidate blocks at block granularity.

    Upstream masks each row's unreachable tail with a materialised
    ``arange >= compress_lens`` position mask before pooling, then expands the
    block keep mask to one bool per position.  Both position-width writes go
    away here: the tail is masked at block level inside the pooling (``col >
    last``), and the keep mask stays ``[rows, num_blocks]`` until the first
    consume layer expands it once per forward.
    """
    from sglang.srt.layers.attention.dsv4.candidate_indexer import published_masks

    publish = [] if indexer.is_candidate_source else None
    consume = (
        None
        if publish is not None
        else published_masks(self.forward_metadata.candidate_metadata)
    )
    block_size = indexer.candidate_block_size
    tok_start = 0
    for b, (lc, t_len) in enumerate(zip(lc_per_req, q_lens_cpu)):
        rows = slice(tok_start, tok_start + t_len)
        tok_start += t_len
        if lc == 0 or t_len == 0:
            if publish is not None:
                publish.append(empty_mask)
            continue
        if publish is None:
            scores = logits[rows, :lc]
            step = max(1, _KEEP_BUDGET_BYTES // (lc * 4))
            # Materialises the drop mask to positions once per forward.
            neg = getattr(consume, "request_neg_masks", None)
            if neg is None:  # plain CandidateMasks from the no-work paths
                neg = [~m for m in consume.request_masks]
            # XPU's masked_fill_ allocates a temporary (~1.8x the slice), so do
            # it in row chunks: same elements, bounded workspace.
            for start in range(0, t_len, step):
                end = min(start + step, t_len)
                scores[start:end].masked_fill_(neg[b][start:end], -torch.inf)
            continue
        # Pool this request's own block-aligned prefix of the physical row.
        # When the plane row is block aligned (``_plane_width``) and the request
        # spans it, this slice *is* the row -- contiguous -- so the block max
        # reads the plane in place instead of first materialising a
        # ``[rows, lc]`` copy.  A shorter request keeps its own prefix (aligned
        # up by at most ``block_size - 1`` columns) so a mixed batch never pays
        # for the widest row; positions past that request's length are masked at
        # block granularity inside ``_candidate_block_keep`` (``col > last`` ->
        # -inf, dropped by ``values > -inf``), so extra columns cannot change
        # the keep mask.
        lcw = min(-(-lc // block_size) * block_size, logits.size(1))
        lens = compress_lens[rows, None]
        mask = _pool_request_blocks(
            logits[rows, :lcw],
            lens,
            indexer.candidate_topk_blocks,
            block_size,
        )
        reference = (
            mask
            if _plane_checked or lcw == lc
            else _pool_request_blocks(
                logits[rows, :lc],
                lens,
                indexer.candidate_topk_blocks,
                block_size,
            )
        )
        publish.append(
            _announce_plane(logits.size(1), lc, block_size, mask, reference)
        )
    if publish is not None:
        cls = _candidate_block_masks_cls()
        self.forward_metadata.candidate_metadata = cls(
            publish, [int(lc) for lc in lc_per_req], block_size
        )


@plugin_hook(
    "sglang.kernels.ops.attention.dsv4.fp4_indexer.quantize_fp4_indexer_tensor",
    type=HookType.AROUND,
)
def _quantize_fp4_indexer_tensor_trace(original_fn, x, rne=False):
    if _bf16_idx_enabled():
        global _LAST_Q_BF16, _reported_bf16
        if not _reported_bf16:
            logger.warning(
                "sglang-kunlun: keeping the bf16 indexer query, skipping the fp4 "
                "quantize -> dequantize round-trip (no fp8_fp4_mqa_logits kernel)"
            )
            _reported_bf16 = True
        n = x.shape[0]
        _LAST_Q_BF16 = x
        return (
            torch.empty((n, x.shape[-1] // 2), device=x.device, dtype=torch.int8),
            torch.empty((n,), device=x.device, dtype=torch.int32),
        )
    return original_fn(x, rne=rne)
