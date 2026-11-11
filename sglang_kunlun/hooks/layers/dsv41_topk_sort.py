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
"""Kunlun XPU fast path for the dense-indexer top-k sort tail.

Upstream ``DeepseekV4AttnBackend._low_ratio_index_topk_dense`` ends with

    unselected = torch.iinfo(torch.int32).max
    selected = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values

i.e. an ascending sort of ``[num_tokens, topk]`` int32 positions, padding last.
At the launch-best prefill pose (8x60k requests, chunk 32768) this is
``sort_large_n`` on ``[4096, 512]``: 8 calls per step, 4.55 ms each,
36.4 ms/step = 1.9% of the 1937 ms step span on this XPU build.

XPU's radix sort runs at a width-dependent speed: at ``[4096, 512]`` int32 sort
is 4.59 ms while int16 sort is 0.25 ms (the whole mask+cast+sort+cast path is
0.38 ms against 4.66 ms).  The positions being sorted are indices into the
concatenated compressed window (``k_slots``), so every valid value is
``< k_slots.numel()``; whenever that window fits in int16 (32767 slots) the sort
can run in int16 with the same ordering: the padding sentinel (32767) stays above
every valid position, so the sorted sequence is identical, and the result is cast
back to int32 where the downstream ``selected != unselected`` / ``selected - ks``
/ ``k_slots[selected]`` consumers see the same values.

At the launch-best pose the *concatenated* window does not fit int16 (it spans
several 60k requests), but each row's own window does: rows are sorted relative
to their window start (``ks``), which is an additive per-row constant and so
leaves the ascending order untouched.  The relative values are biased by -32768
to cover [0, 65535] in a signed int16.  This path cuts the tail from 4.6 ms to
0.5 ms per sort without changing a single output value.

The tail is the only behavioural difference from upstream; the rest of the
function is mirrored verbatim (upstream keeps the sort in a single inline
expression, so hooking it requires replacing the whole method).  The hook
de-registers itself if another REPLACE hook is already registered on the target
(e.g. the prefill-CP indexer fallback hook in ``dsv41_cp_indexer.py``), so
whichever plugin module lands last does not silently override the other.

``KL_TOPK_SORT_INT16=0`` restores the plain int32 sort.
"""

from __future__ import annotations

import functools
import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType, plugin_hook

logger = logging.getLogger(__name__)

_TARGET = (
    "sglang.srt.layers.attention.deepseek_v4_backend."
    "DeepseekV4AttnBackend._low_ratio_index_topk_dense"
)

# Largest int16 value; valid positions are strictly below it, so it doubles as
# the padding sentinel (the int32 path uses iinfo(int32).max the same way).
_INT16_SENTINEL = 32767
# The relative path sorts ``position - window_start`` biased by -32768, which
# keeps ascending order for any value in [0, 65535]; 65535 then doubles as the
# padding sentinel because it is above every valid biased value.
_REL_BIAS = 32768
_REL_SENTINEL = 65535

_reported = False


@functools.cache
def _int16_sort_enabled() -> bool:
    return os.environ.get("KL_TOPK_SORT_INT16", "1") != "0"


def _sort_selected(
    selected: torch.Tensor, num_slots: int, ks=None, max_rel=None
):
    """Ascending sort of ragged top-k positions, ``-1`` padding last.

    Returns ``(selected, unselected)``; ``selected != unselected`` is the
    chosen mask, exactly as on the upstream int32 path.

    ``ks`` (per-row window start) and ``max_rel`` (widest per-request window,
    host side) enable the *relative* int16 path.  Subtracting ``ks`` is an
    additive per-row constant, so the ascending order is unchanged, and the
    values then fit int16 after a -32768 bias.  This matters at the
    launch-best pose: the absolute window concatenates every request
    (``num_slots`` in the hundreds of thousands, so the absolute int16 gate
    below never fires) while one row's own window is at most 61440 slots.
    4.6 ms -> 0.5 ms per ``[4096, 512]`` sort on this build.
    """
    global _reported
    unselected = torch.iinfo(torch.int32).max
    if (
        _int16_sort_enabled()
        and ks is not None
        and max_rel is not None
        and 0 < max_rel < _REL_SENTINEL + 1
    ):
        rel = selected - ks[:, None]
        # ``selected`` is -1 padded; anything outside [0, max_rel) is padding
        # (this also covers an int32.max sentinel from a masked upstream path
        # that would otherwise overflow the int16 cast).
        rel = rel.masked_fill((rel < 0) | (rel >= max_rel), _REL_SENTINEL)
        rel = (
            (rel - _REL_BIAS)
            .to(torch.int16)
            .sort(dim=-1)
            .values.to(torch.int32)
            + _REL_BIAS
        )
        selected = (rel + ks[:, None]).masked_fill(rel == _REL_SENTINEL, unselected)
        if not _reported:
            logger.warning(
                "sglang-kunlun: dense-indexer top-k tail sorted in int16 "
                "(per-row window %d of %d absolute slots, %d rows), "
                "replacing the int32 sort",
                max_rel,
                num_slots,
                selected.shape[0],
            )
            _reported = True
    elif _int16_sort_enabled() and num_slots <= _INT16_SENTINEL:
        # int16 has no masked_fill on this build, so mask in int32 (cheap at
        # [4096, 512]) and only the sort itself runs in int16.
        selected = (
            selected.masked_fill(selected < 0, _INT16_SENTINEL)
            .to(torch.int16)
            .sort(dim=-1)
            .values.to(torch.int32)
        )
        unselected = _INT16_SENTINEL
        if not _reported:
            logger.warning(
                "sglang-kunlun: dense-indexer top-k tail sorted in int16 "
                "(window %d slots), replacing the int32 sort",
                num_slots,
            )
            _reported = True
    else:
        selected = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values
    return selected, unselected


def _publish_epi_plan(lc_per_req, q_lens_cpu, total_slots, max_seqlen_k) -> None:
    """Stash the score epilogue's per-request windows on the host.

    The epilogue otherwise derives them on the device (``unique_consecutive``
    plus one ``.item()`` per window, i.e. a stream sync per request per layer).
    ``(slot_offset, row_count, visible_len)`` mirrors that grouping; the
    epilogue drops the plan when the row counts do not add up, so a mismatch
    only costs the old syncs.
    """
    try:
        from . import dsv41_cp_indexer as cpi
    except Exception:  # noqa: BLE001 -- opportunistic channel only
        return
    if q_lens_cpu is None or len(q_lens_cpu) != len(lc_per_req):
        cpi._LAST_EPI_PLAN = None
        return
    plan, start = [], 0
    for lc, cnt in zip(lc_per_req, q_lens_cpu):
        cnt = int(cnt)
        if cnt:
            plan.append((start, cnt, min(lc, total_slots - start, max_seqlen_k)))
        start += lc
    cpi._LAST_EPI_PLAN = plan


def _low_ratio_index_topk_dense_kunlun(
    self, layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu
) -> None:
    """Mirror of the upstream method; only the tail sort differs (see module)."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        quantize_fp4_indexer_tensor,
    )
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _as_int_list,
        _dense_fp4_mqa_logits,
        ceil_align,
        topk_transform_ragged_v2,
    )
    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        CandidateMasks,
        mask_topk_scores,
    )

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
    req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)
    slot_chunks, starts, start = [], [], 0
    for r, lc in enumerate(lc_per_req):
        starts.append(start)
        if lc == 0:
            continue
        j = torch.arange(lc, device=device)
        slot_chunks.append(
            self.req_to_token[req_pool_indices[r], j * ratio].to(torch.int64) // ratio
        )
        start += lc
    empty_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
    num_tokens = pos.shape[0]
    # TODO(candidate): move this to candidate indexer
    if not slot_chunks or num_tokens == 0:
        if indexer.is_candidate_source:
            self.forward_metadata.candidate_metadata = CandidateMasks(
                request_masks=[empty_mask for _ in lc_per_req]
            )
        return
    k_slots = torch.cat(slot_chunks)
    k_fp4, k_sf = pool.get_low_ratio_index_k_fp4(layer.layer_id, k_slots)

    q = indexer.queries(q_lora, layer.freqs_cis[pos])  # [T, H, 128] fp4 grid
    num_heads = q.shape[1]
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    q_fp4 = q_fp4.view(num_tokens, num_heads, 64)
    q_sf = q_sf.view(num_tokens, num_heads)
    weights = indexer.head_weights(x).float()
    compress_lens = ((pos + 1) // ratio).to(torch.int32)
    ks = torch.repeat_interleave(
        torch.tensor(starts, dtype=torch.int32, device=device),
        q_lens.to(torch.int64),
        output_size=num_tokens,
    )
    epi_width = ceil_align(max(lc_per_req), 4)
    # Hand the score epilogue its per-request windows from the host: it would
    # otherwise pay a device sync per request (``lens.max().item()``) plus one
    # for ``unique_consecutive(...).tolist()``, 8 windows x 8 layers per step.
    _publish_epi_plan(lc_per_req, q_lens_cpu, k_slots.numel(), epi_width)
    logits = _dense_fp4_mqa_logits(
        (q_fp4, q_sf),
        (k_fp4, k_sf),
        weights,
        ks,
        ks + compress_lens,
        # the fused top-k reads score rows through 16-byte vectors
        epi_width,
    )
    if indexer.is_candidate_source or indexer.uses_candidates:
        self._publish_or_consume_candidates(
            indexer, logits, compress_lens, lc_per_req, q_lens_cpu, empty_mask
        )
    topk = indexer.index_topk
    selected = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    topk_transform_ragged_v2(logits, compress_lens, out_offsets=ks, out_indices=selected)
    if indexer.uses_candidates and not indexer.is_candidate_source:
        selected = mask_topk_scores(logits, selected, ks)
    # ascending positions, padding last: the layout the consumers expect
    selected, unselected = _sort_selected(
        selected, k_slots.numel(), ks=ks, max_rel=max(lc_per_req)
    )
    chosen = selected != unselected
    page_indices[:num_tokens, :topk] = torch.where(
        chosen, k_slots[selected.clamp_max(k_slots.shape[0] - 1)], -1
    ).to(torch.int32)
    if raw_indices is not None:
        raw_indices[:num_tokens, :topk] = torch.where(
            chosen, selected - ks[:, None], -1
        )


def _already_replaced() -> bool:
    return any(
        ht == HookType.REPLACE for ht, _hook, _src in HookRegistry._hooks.get(_TARGET, ())
    )


if _already_replaced():
    logger.info(
        "sglang-kunlun: %s already has a REPLACE hook; leaving the top-k sort "
        "tail to it",
        _TARGET,
    )
else:
    plugin_hook(_TARGET, type=HookType.REPLACE)(_low_ratio_index_topk_dense_kunlun)
