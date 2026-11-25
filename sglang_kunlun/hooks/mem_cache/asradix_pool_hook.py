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
"""Hooks for ``sglang.srt.mem_cache.deepseek_v4_memory_pool`` -- AttentionStore.

Ports the ``DeepSeekV4TokenToKVPool`` side of the reference implementation:

* ``init_as_swa_pool`` allocates the ``ChunkTailSWAPool`` staging pool. The AS
  write path can only persist a SWA window that still exists in HBM, but
  ``UnifiedRadixCache`` evicts everything older than
  ``max(sliding_window, page_size)`` as prefill advances, so every ``swa_step``
  checkpoint copies its tail window out to this pool, where it survives until
  the async write retires it.
* ``_iter_contiguous_page_buffers`` / ``get_contiguous_buf_tensor_infos`` give
  the AS content digest a tensor-level view of the page transfer layout.

Two deliberate deviations from upstream, both forced by hooking rather than
editing:

``max_context_length`` cannot be a new ``__init__`` keyword, so the
configurator hook stashes ``model_config.context_len`` in
``_PENDING_MAX_CONTEXT_LENGTH`` right before it builds the pool and the
``__init__`` hook picks it up. Same value, same single-threaded init path.

``get_contiguous_buf_infos`` keeps its stock implementation. Upstream rewrites
it in terms of the new iterator so the ptr view and the tensor view cannot
drift; that method feeds mooncake PD buffer registration, and the two views
agree by construction here because the iterator is a transcription of the same
loop. Re-expressing a live PD code path to buy a maintenance invariant is not
worth the regression risk in a plugin.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.mem_cache.as_config import as_cfg, asradix_enabled
from sglang_kunlun.mem_cache.chunk_tail_swa_pool import ChunkTailSWAPool

logger = logging.getLogger(__name__)

_POOL = "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool"

# Set by the KVCacheConfigurator hook immediately before it constructs the pool.
# ``is_draft_worker`` rides along because the stock ``__init__`` only keeps it as
# a local, and the configurator carries it as a field anyway.
_PENDING_MAX_CONTEXT_LENGTH: Optional[int] = None
_PENDING_IS_DRAFT_WORKER: bool = False


def stash_pool_context(
    max_context_length: Optional[int], is_draft_worker: bool
) -> None:
    """Hand the next pool construction the two values upstream passes in."""
    global _PENDING_MAX_CONTEXT_LENGTH, _PENDING_IS_DRAFT_WORKER
    _PENDING_MAX_CONTEXT_LENGTH = max_context_length
    _PENDING_IS_DRAFT_WORKER = is_draft_worker


def _init_as_swa_pool(self) -> None:
    """Allocate the chunk-tail SWA staging pool. Verbatim port of upstream's
    ``DeepSeekV4TokenToKVPool.init_as_swa_pool``, with the ``memory`` namespace
    reads redirected to ``as_cfg()``.

    Must run after ``_init_paged_compress_states`` so ``swa_kv_pool``'s buffers
    (and therefore ``kv_cache_total_dim``) are populated.

    The draft worker builds its own pool too: the draft shares the target's
    slot-id space (``req_to_token_pool`` / ``full_to_swa_index_mapping`` are the
    same objects), and the AS binding needs the draft pool's write pointers to
    persist the draft SWA layers alongside the target's.
    """
    from sglang.srt.runtime_context import get_schedule

    from sglang_kunlun.mem_cache.swa_step_policy import SwaStepPolicy, resolve_swa_step

    memory_args = as_cfg()
    # Draft ChunkTailSWAPools registered by the AS binding; every free must fan
    # out to them because they mirror this pool's slot ids (see
    # free_chunk_tail_swa_slots).
    self.draft_chunk_tail_pools: List[ChunkTailSWAPool] = []
    # swa_kv_pool is None on the encoder-replay and unified-kv paths; neither
    # exposes a paged SWA page to copy a half out of, so there is nothing to
    # stage. Returning here keeps those paths from an AttributeError.
    if not memory_args.enable_as_swa_pool or self.swa_kv_pool is None:
        self.chunk_tail_swa_pool = None
        self.swa_policy = None
        return

    page_size = self.page_size
    effective_swa_step = resolve_swa_step(memory_args, page_size)
    if memory_args.enable_dynamic_swa_step:
        self.swa_policy = SwaStepPolicy.from_spec(memory_args.swa_step_tiers, page_size)
    else:
        self.swa_policy = SwaStepPolicy.single(effective_swa_step, page_size)
    self.swa_step = effective_swa_step
    assert effective_swa_step % self.swa_page_size == 0, (
        f"swa_step={effective_swa_step} must be a multiple of "
        f"swa_page_size={self.swa_page_size}. This guarantees "
        "tail_start % swa_page_size == swa_win, so half=1 always."
    )

    # Sized per in-flight chunk, not per request: a slot is returned as soon as
    # its async AS write retires, so only the concurrently un-flushed chunks
    # need coverage. as_swa_buffer_coef sets how many.
    chunked_prefill_size = get_schedule().chunked_prefill_size or 0
    if chunked_prefill_size <= 0:
        chunked_prefill_size = self.max_context_length
    if not chunked_prefill_size or chunked_prefill_size <= 0:
        logger.warning(
            "chunk_tail_swa_pool disabled: neither chunked_prefill_size nor "
            "max_context_length is known, so the pool cannot be sized."
        )
        self.chunk_tail_swa_pool = None
        self.swa_policy = None
        return

    swa_win = (
        self.sliding_window
        if self.sliding_window is not None
        else self.swa_page_size // 2
    )
    slots_per_chunk = max(
        1, chunked_prefill_size // (self.swa_page_size // swa_win) // swa_win
    )
    num_pages = int(slots_per_chunk * memory_args.as_swa_buffer_coef)
    self.chunk_tail_swa_pool = ChunkTailSWAPool(
        num_pages=num_pages,
        layer_num=self.swa_kv_pool.layer_num,
        bytes_per_token=self.swa_kv_pool.kv_cache_total_dim,
        src_swa_page_size=self.swa_page_size,
        swa_win=swa_win,
        device=self.swa_kv_pool.device,
    )
    logger.info(
        "DSV4 chunk_tail_swa_pool: worker=%s, num_pages=%d, layers=%d, "
        "slot_bytes=%d, swa_win=%d, swa_step=%d, dynamic=%s, "
        "chunked_prefill_size=%d, slots_per_chunk=%d, coef=%s",
        "draft" if self.is_draft_worker else "target",
        num_pages,
        self.chunk_tail_swa_pool.layer_num,
        self.chunk_tail_swa_pool.slot_bytes,
        swa_win,
        effective_swa_step,
        memory_args.enable_dynamic_swa_step,
        chunked_prefill_size,
        slots_per_chunk,
        memory_args.as_swa_buffer_coef,
    )


def _iter_contiguous_page_buffers(self):
    """Yield one record per contiguous page buffer, in PD wire order.

    Transcribed from this engine's ``get_contiguous_buf_infos`` so the AS
    content digest hashes exactly the bytes the SDK moves.

    Each record is ``(kind, tensor, offset_bytes, transfer_bytes, item_bytes)``:
    ``kind`` is ``"content"`` (KV) or ``"index"`` (indexer); the transfer starts
    at ``tensor.data_ptr() + offset_bytes`` and covers ``transfer_bytes``; one
    FULL page occupies ``item_bytes``.
    """
    stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]
    # Registration order defines the PD wire layout: C4 KV, C4 indexer, C128 KV.
    # Keep each indexer immediately after the KV buffers of the same ratio.
    for ratio, kv_pool in self.kv_pools.items():
        if self._unified_kv:
            # Unified buffers store token rows after the SWA ring. Transfer
            # compressed pages from the offset; SWA ships as StateType.SWA_RING.
            swa_pages = self.unified_kv_pool.swa_pages
            for local_layer_id, layer_ratio in enumerate(stage_ratios):
                if layer_ratio != ratio:
                    continue
                buf = self.unified_kv_pool.kv_buffer[local_layer_id]
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                row_bytes = buf[0].nbytes
                rows_per_page = self.page_size // ratio
                compress_rows = buf.shape[0] - swa_pages
                yield (
                    "content",
                    buf,
                    swa_pages * row_bytes,
                    compress_rows * row_bytes,
                    rows_per_page * row_bytes,
                )
        elif kv_pool is not None:
            for buf in kv_pool.kv_buffer:
                assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
                yield ("content", buf, 0, buf.nbytes, buf[0].nbytes)

        indexer_pool = self.index_pools.get(ratio)
        if indexer_pool is None:
            continue
        # The transfer addresses every buffer by FULL page id; ratio-1/2 index
        # pools page at DSV41_INDEX_PAGE_SIZE, so one item is the run of index
        # pages holding a FULL page's page_size // ratio slots.
        index_pages_per_full_page = 1
        if ratio in (1, 2):
            slots_per_full_page = self.page_size // ratio
            assert slots_per_full_page % indexer_pool.page_size == 0, (
                f"ratio-{ratio} index pages of {indexer_pool.page_size} slots do "
                f"not tile a FULL page of {slots_per_full_page} slots"
            )
            index_pages_per_full_page = slots_per_full_page // indexer_pool.page_size
        for buf in indexer_pool.contiguous_page_row_buffers():
            assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
            yield (
                "index",
                buf,
                0,
                buf.nbytes,
                buf[0].nbytes * index_pages_per_full_page,
            )


def _get_contiguous_buf_tensor_infos(self):
    """Tensor-level view of the page transfer for the AS content digest.

    Returns a list of ``(kind, tensor, offset_bytes, item_bytes)`` in the exact
    same order as ``get_contiguous_buf_infos``. The digest hashes
    ``tensor.view(uint8)[offset_bytes + page_id*item_bytes :][:item_bytes]``,
    i.e. the very bytes the SDK moves for that FULL page.
    """
    return [
        (kind, buf, offset_bytes, item_bytes)
        for kind, buf, offset_bytes, _transfer_bytes, item_bytes in (
            self._iter_contiguous_page_buffers()
        )
    ]


@plugin_hook(f"{_POOL}.__init__", type=HookType.AROUND)
def pool_init_kunlun(original_fn, self, *args, **kwargs):
    """Attach the chunk-tail staging pool after the stock constructor runs.

    Unconditional, like upstream: ``init_as_swa_pool`` itself returns early when
    AttentionStore is off, and the attributes it sets
    (``chunk_tail_swa_pool``/``swa_policy``, both ``None``) are what the rest of
    the port probes with ``getattr``.
    """
    original_fn(self, *args, **kwargs)

    self.max_context_length = _PENDING_MAX_CONTEXT_LENGTH
    self.is_draft_worker = _PENDING_IS_DRAFT_WORKER

    # Bind the new methods on the instance's class once, so the AS coordinators
    # can reach them through the pool like upstream does.
    cls = type(self)
    if not hasattr(cls, "init_as_swa_pool"):
        cls.init_as_swa_pool = _init_as_swa_pool
        cls._iter_contiguous_page_buffers = _iter_contiguous_page_buffers
        cls.get_contiguous_buf_tensor_infos = _get_contiguous_buf_tensor_infos

    try:
        self.init_as_swa_pool()
    except Exception:
        if asradix_enabled():
            raise
        # With AttentionStore off a staging-pool failure must not take the
        # engine down: nothing will read these attributes.
        logger.warning("chunk_tail_swa_pool init failed (AS off)", exc_info=True)
        self.chunk_tail_swa_pool = None
        self.swa_policy = None
