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
"""Coordinator implementations for checkpointed-page cache plans."""

from __future__ import annotations

import logging
import time
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang_kunlun.mem_cache.chunk_tail_swa_pool import free_chunk_tail_swa_slots
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)

from .checkpoint_binding import PackedStateSnapshot
from .coordinator import (
    CheckpointedPageCacheCoordinator,
    _acquire_node_release,
    _contiguous_page_ids,
    _noop_release,
    _request_block_hash_ids,
)
from .types import (
    AttentionStoreEndpointBinding,
    AttentionStoreWriteContext,
    AttentionStoreWriteOwner,
)

logger = logging.getLogger(__name__)

_KDA_PENDING_HASH = "as_kda_pending_hash"
_KDA_PUBLISHED_HASH = "as_kda_published_hash"


@dataclass
class _SPAWriteTask:
    rid: str
    page_endpoint: Optional[AttentionStoreEndpointBinding]
    checkpoint_store: Any
    block_hash_ids: Sequence[int]
    page_ids: Any
    checkpoint_snapshot: Optional[PackedStateSnapshot]
    params: dict
    release_after_write: Callable[[], None] = _noop_release
    release: Callable[[], None] = _noop_release
    owner: Optional[AttentionStoreWriteOwner] = None
    # SPAG only: one chunk-tail staging slot per checkpoint of this write's page
    # range, indexed by checkpoint number from the request's start. ``-1`` means
    # "no staged window for this checkpoint" -- either an AS-hit prefix, a
    # staging-pool shortage, or a checkpoint another task already owns. Purely
    # rank-local, so it must never reach the write key / logical id.
    swa_slot_ids: Sequence[int] = ()
    # Exclusive end of the checkpoint range this task owns (it must release those
    # slots when it retires). The owned range starts at the request's
    # ``as_written_count`` as it stood when the task was prepared.
    swa_owned_end: int = 0

    @property
    def requested_blocks(self) -> int:
        """Return the number of SPA page blocks requested by this task."""
        return len(self.block_hash_ids)


@dataclass
class _KDAWriteTask:
    rid: str
    node: UnifiedTreeNode
    block_hash_ids: Sequence[int]
    page_ids: Any
    checkpoint_slot: Any
    params: dict
    producer_ready: Any = None
    request: Any = None
    write_key: Optional[tuple[int, int]] = None
    release_after_write: Callable[[], None] = _noop_release
    release: Callable[[], None] = _noop_release
    owner: Optional[AttentionStoreWriteOwner] = None

    @property
    def requested_blocks(self) -> int:
        """Return the number of KDA page blocks requested by this task."""
        return len(self.block_hash_ids)

    @property
    def endpoint_hash(self) -> int:
        """Return the terminal block hash associated with the checkpoint."""
        return int(self.block_hash_ids[-1]) if self.block_hash_ids else 0


class SPACacheCoordinator(CheckpointedPageCacheCoordinator):
    """Coordinate SPA page writes with a stable endpoint checkpoint snapshot."""

    write_thread_name = "as_spa_writer"

    def capture_finished(
        self, cache, req, is_insert: bool
    ) -> AttentionStoreWriteContext:
        """Snapshot SPA slots for a finished request."""
        # The dsv41 lineage names this ``effective_kv_committed_len``; this
        # engine's community lineage carries the identical body under the older
        # name ``owned_kv_len``. Same semantics (report only the prompt prefix
        # when strip_thinking_cache trimmed reasoning tokens), just a rename.
        owned_kv_len = getattr(req, "effective_kv_committed_len", None) or getattr(
            req, "owned_kv_len"
        )
        committed_len = owned_kv_len()
        token_ids = (req.origin_input_ids + req.output_ids)[:committed_len]
        snapshot = cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :committed_len
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=is_insert,
        )

    def capture_unfinished(self, cache, req) -> AttentionStoreWriteContext:
        """Snapshot SPA slots for an unfinished request."""
        token_ids = req.get_fill_ids()
        snapshot = cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=True,
        )

    def before_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Pack a stable SPA checkpoint before the local commit."""
        if context.is_insert and cache._is_prefill_role() and cache.ascli is not None:
            context.prepared_write = self.prepare_write(
                cache,
                context.req,
                context.token_ids,
                context.snapshot,
            )

    def after_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Retain the committed SPA node and submit its prepared write."""
        if not context.is_insert or not cache._is_prefill_role() or cache.ascli is None:
            return
        task = context.prepared_write
        if task is None:
            self.enqueue_dummy(cache, context.req.rid)
            return

        try:
            key = RadixKey(
                context.token_ids,
                context.req.extra_key,
                is_bigram=cache.tree_core.is_eagle,
            ).page_aligned(cache.page_size)
            match = UnifiedRadixCache.match_prefix(
                cache,
                MatchPrefixParams(key=key),
            )
            node = match.last_device_node
            # The post-commit re-match only publishes the device-resident,
            # page-aligned front of the committed prefix. On a warm hit part of
            # the prefix can be offloaded to host/L2, so device_indices covers
            # fewer pages than the full key -- that is legitimate, not an error.
            published_len = (
                len(match.device_indices) // cache.page_size
            ) * cache.page_size
            if node is None or published_len == 0:
                logger.warning(
                    "SPA local commit did not publish the write prefix rid=%s "
                    "is_eagle=%s page=%s len_key=%s published_indices=%s "
                    "node_is_none=%s requested_blocks=%s",
                    context.req.rid,
                    cache.tree_core.is_eagle,
                    cache.page_size,
                    len(key),
                    len(match.device_indices),
                    node is None,
                    task.requested_blocks,
                )
                raise RuntimeError("SPA local commit did not publish the write prefix")
            # WAR fix: build page_ids from the tree's canonical, ref-protected
            # slots (post-commit), not the pre-commit snapshot. _acquire_node_release
            # below protects these device_indices for the async gather's lifetime.
            cache_len = min(len(key), published_len)
            if cache_len < len(key):
                logger.debug(
                    "SPA write floored to published prefix rid=%s len_key=%s "
                    "published_indices=%s published_pages=%s requested_blocks=%s",
                    context.req.rid,
                    len(key),
                    len(match.device_indices),
                    cache_len // cache.page_size,
                    task.requested_blocks,
                )
                self._refloor_task(cache, context, task, match, cache_len)
            page_ids = self.prepare_page_ids(cache, match.device_indices[:cache_len])
            task.page_ids = page_ids

            task.release_after_write = self._make_release_after_write(
                cache, task, page_ids
            )
            task.release = _acquire_node_release(cache, node)
            # Ownership moves to the task *before* submission: if submit_write
            # fails it runs release_after_write itself, so the resources must
            # already belong to the task for that release to be the only one.
            self._commit_write_ownership(cache, context, task)
            self.submit_write(cache, task)
            context.prepared_write = None
        except Exception as exc:
            self._discard_unsubmitted(cache, task)
            context.prepared_write = None
            cache._log_as_failure(
                "write",
                rid=context.req.rid,
                requested=task.requested_blocks,
                error=exc,
            )
            self.enqueue_dummy(cache, context.req.rid)

    def _refloor_task(self, cache, context, task, match, cache_len: int) -> None:
        """Shrink a prepared task to the ``cache_len`` prefix the tree published.

        Truncates the block hashes and re-derives the checkpoint for the new
        terminal page, so the endpoint hash (``block_hash_ids[-1]``) and the
        recurrent-state snapshot still describe the same page.
        """
        task.block_hash_ids = task.block_hash_ids[: cache_len // cache.page_size]
        swa_indices = cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            match.device_indices[cache_len - cache.page_size : cache_len]
        )
        swa_page_id = int(swa_indices[0].item()) // cache.page_size
        new_snapshot = task.checkpoint_store.snapshot(
            self.state_slots(cache, context.req.kv.req_pool_idx, swa_page_id)
        )
        if task.checkpoint_snapshot is not None:
            task.checkpoint_snapshot.release()
        task.checkpoint_snapshot = new_snapshot

    def _make_release_after_write(self, cache, task, page_ids) -> Callable[[], None]:
        """Build the callback that frees everything the write staged."""
        checkpoint_snapshot = task.checkpoint_snapshot

        def _release_staging() -> None:
            try:
                page_ids.release()
            finally:
                if checkpoint_snapshot is not None:
                    checkpoint_snapshot.release()

        return _release_staging

    def _commit_write_ownership(self, cache, context, task) -> None:
        """Hand the task's rank-local resources over to the writer thread.

        Called once the task is fully built, right before ``submit_write``, so
        the task's ``release_after_write`` is the single owner from here on.
        """

    def _read_payload_args(self, cache, req, load_info, loaded):
        """Derive the plan-specific extra arguments of the load-back read.

        Returns ``(positional, keyword)``. SPA appends the checkpoint slot map
        that names where the terminal blob must land; other plans carry their
        recurrent state inside the page blocks and override this.
        """
        swa_indices = cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            loaded[-cache.page_size :]
        )
        swa_page_id = int(swa_indices[0].item()) // cache.page_size
        return (self.state_slots(cache, req.kv.req_pool_idx, swa_page_id),), {}

    def _restored_swa_len(self, cache, token_count: int) -> int:
        """Tokens at the restored tail whose SWA state this read really fills.

        Must stay page-aligned: the tree asserts ``swa_evicted_seqlen`` is a
        multiple of ``page_size``, so a plan that restores only part of the final
        SWA page still reports the whole page here. That is safe because matches
        are page-aligned too, so no reachable prefix ends inside the page, and the
        only consumer of that page is the extend that follows the restore, whose
        sliding window never reaches past the restored part.
        """
        return min(token_count, cache.page_size)

    def state_slots(self, cache, req_pool_idx: int, swa_page_id: int) -> dict[str, int]:
        """Return the logical SPA checkpoint slots for a request."""
        slots = {}
        for group in cache.ascli.layout.groups:
            if group in ("swa_page", "c4_state"):
                slots[group] = swa_page_id
            elif group in ("swa_request", "c128_state"):
                slots[group] = req_pool_idx
            else:
                raise ValueError(f"unsupported SPA checkpoint group: {group}")
        return slots

    def prepare_write(
        self,
        cache,
        req,
        token_ids: Sequence[int],
        snapshot: torch.Tensor,
    ) -> Optional[_SPAWriteTask]:
        """Prepare page identifiers and a stable SPA checkpoint snapshot."""
        if cache.ascli is None or not token_ids:
            return None
        # Offload the page-aligned prefix even when the fill length carries a
        # partial tail page. The radix tree only publishes full pages and
        # after_local_commit already page-aligns the write key, so flooring
        # here (instead of bailing on any unaligned length) lets every request
        # with at least one complete page offload and emit its digests.
        cache_len = (len(token_ids) // cache.page_size) * cache.page_size
        if cache_len == 0 or len(snapshot) < cache_len:
            return None
        token_ids = token_ids[:cache_len]
        requested = 0
        checkpoint_snapshot = None
        try:
            page_count = cache_len // cache.page_size
            swa_indices = (
                cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    snapshot[cache_len - cache.page_size : cache_len]
                )
            )
            swa_page_id = int(swa_indices[0].item()) // cache.page_size
            hashes = _request_block_hash_ids(
                req,
                token_ids,
                cache.page_size,
                page_count,
            )
            requested = len(hashes)
            checkpoint_endpoint = cache.binding.checkpoint_endpoint
            if checkpoint_endpoint is None:
                raise ValueError(
                    "SPA AttentionStore binding has no checkpoint endpoint"
                )
            checkpoint_store = checkpoint_endpoint.store
            checkpoint_snapshot = checkpoint_store.snapshot(
                self.state_slots(cache, req.kv.req_pool_idx, swa_page_id)
            )

            # WAR fix (align KDA / commit 763d75172f): do NOT build page_ids from
            # this pre-commit snapshot. super().cache_*_req (run after this, before
            # the async gather) frees the duplicate prefix slots the snapshot
            # references, and a later forward can reuse/overwrite them. page_ids are
            # built in after_local_commit from the tree's ref-protected
            # match.device_indices instead. Keep only the (already staged, copied)
            # checkpoint snapshot here.
            def release_staging() -> None:
                if checkpoint_snapshot is not None:
                    checkpoint_snapshot.release()

            return _SPAWriteTask(
                rid=req.rid,
                page_endpoint=cache.binding.page_endpoint,
                checkpoint_store=checkpoint_store,
                block_hash_ids=hashes,
                page_ids=None,
                checkpoint_snapshot=checkpoint_snapshot,
                params=getattr(req, "as_kv_cache_params", None) or {},
                release_after_write=release_staging,
            )
        except Exception as exc:
            if checkpoint_snapshot is not None:
                checkpoint_snapshot.release()
            cache._log_as_failure(
                "write",
                rid=req.rid,
                requested=requested,
                error=exc,
            )
            return None

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Load matched SPA pages and checkpoint state into unified storage."""
        anchor = params.best_match_node
        empty = torch.empty((0,), dtype=torch.int64, device=cache.device)
        if params.host_hit_length <= 0:
            return empty, anchor

        anchor, load_info, req = cache._take_attention_store_load_context(params)
        if cache.ascli is None or load_info is None or req is None:
            return empty, anchor

        requested = int(load_info["block_count"])
        started = time.perf_counter()
        token_count = requested * cache.page_size
        available = cache.token_to_kv_pool_allocator.available_size()
        if available < token_count and cache.full_evictable_size() > 0:
            cache.evict(
                EvictParams(
                    num_tokens=token_count - available + cache.page_size,
                    swa_num_tokens=cache.page_size,
                )
            )
        local_hit_len = int(load_info["start_block"]) * cache.page_size
        device = cache.token_to_kv_pool_allocator.device
        prefix_cpu = torch.tensor([local_hit_len], dtype=torch.int64)
        seq_cpu = torch.tensor([local_hit_len + token_count], dtype=torch.int64)
        last_loc_value = int(req.prefix_indices[-1].item()) if local_hit_len else -1
        try:
            loaded = cache.token_to_kv_pool_allocator.alloc_extend_swa_tail(
                prefix_cpu.to(device),
                prefix_cpu,
                seq_cpu.to(device),
                seq_cpu,
                torch.tensor([last_loc_value], dtype=torch.int64, device=device),
                token_count,
                # One whole SWA page, even though a plan may restore only its
                # upper half: alloc_extend_swa_tail maps the *last* swa_tail_len
                # full slots onto the freshly allocated SWA page, so asking for
                # less than a page would land the restored window in the page's
                # lower half and desynchronize it from the stored byte offset.
                min(token_count, cache.page_size),
            )
        except Exception as exc:
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=exc,
                started=started,
            )
            cache._as_min(0)
            return empty, anchor
        if loaded is None:
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=RuntimeError(f"cannot allocate {token_count} tokens"),
                started=started,
            )
            cache._as_min(0)
            return empty, anchor
        try:
            page_ids = _contiguous_page_ids(loaded, cache.page_size)
        except ValueError as exc:
            cache.token_to_kv_pool_allocator.free(loaded)
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=exc,
                started=started,
            )
            cache._as_min(0)
            return empty, anchor
        try:
            read_args, read_kwargs = self._read_payload_args(
                cache, req, load_info, loaded
            )
        except Exception as exc:
            cache.token_to_kv_pool_allocator.free(loaded)
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=exc,
                started=started,
            )
            cache._as_min(0)
            return empty, anchor
        if cache._as_min(1) == 0:
            cache.token_to_kv_pool_allocator.free(loaded)
            return empty, anchor
        read_error = None
        try:
            read_count = cache.ascli.read(
                load_info["hashes"],
                page_ids,
                *read_args,
                timeout=5.0,
                remote_addrs=load_info.get("remote_addrs") or None,
                **read_kwargs,
            )
        except Exception as exc:
            read_error = exc
            read_count = 0
        read_count = cache._as_min(read_count)
        if read_error is None:
            cache._log_as_result(
                "load_back",
                rid=req.rid,
                requested=requested,
                result=read_count,
                started=started,
            )
        else:
            cache._log_as_failure(
                "load_back",
                rid=req.rid,
                requested=requested,
                error=read_error,
                started=started,
            )
        if read_count != requested:
            cache.token_to_kv_pool_allocator.free(loaded)
            return empty, anchor

        total_prefix_len = local_hit_len + token_count
        token_ids = req.full_untruncated_fill_ids[:total_prefix_len]
        values = torch.cat([req.prefix_indices, loaded]).to(
            dtype=torch.int64, copy=True
        )
        restored_swa_len = self._restored_swa_len(cache, token_count)
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=cache.tree_core.is_eagle),
                value=values,
                prev_prefix_len=local_hit_len,
                # Everything before the restored window is an SWA tombstone. This
                # must describe the bytes actually read back, not the SWA pages
                # allocated: over-claiming makes the tree serve uninitialized SWA.
                swa_evicted_seqlen=local_hit_len
                + max(0, token_count - restored_swa_len),
            )
        )
        match = UnifiedRadixCache.match_prefix(
            cache,
            MatchPrefixParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=cache.tree_core.is_eagle),
                req=req,
            ),
        )
        tree_indices = match.device_indices[local_hit_len:total_prefix_len].to(
            dtype=loaded.dtype
        )
        if len(tree_indices) != token_count:
            logger.warning(
                "AS cache insert: cache_plan=%s rid=%s status=error "
                "reason=shortened_prefix expected_tokens=%s actual_tokens=%s",
                cache.attention_store_plan.name,
                req.rid,
                token_count,
                len(tree_indices),
            )
        req.as_read_pending_lock = True
        return tree_indices, match.last_device_node

    def do_write(self, cache, task: _SPAWriteTask) -> int:
        """Write SPA pages followed by their terminal checkpoint."""
        if not task.block_hash_ids:
            return 0
        started = time.perf_counter()
        try:
            page_ids = task.page_ids.materialize()
            # RAW fix (align with KDACacheCoordinator.do_write / commit 763d75172f):
            # record the producer fence here, on the writer thread right before the
            # gather, targeting the injected forward_stream -- not the scheduler's
            # current stream. _BufferPageIO.write waits on it before reading pages.
            producer_ready = self.record_device_event(
                cache.device, self._producer_stream(cache)
            )
            confirmed_pages = task.page_endpoint.store.match(
                task.block_hash_ids,
                timeout=1.0,
            )
            confirmed_pages = max(0, min(confirmed_pages, task.requested_blocks))
            if confirmed_pages < task.requested_blocks:
                task.page_endpoint.store.write(
                    task.block_hash_ids[confirmed_pages:],
                    page_ids[confirmed_pages:],
                    timeout=1.0,
                    params=task.params,
                    producer_ready=producer_ready,
                )
                confirmed_pages = task.page_endpoint.store.match(
                    task.block_hash_ids,
                    timeout=1.0,
                )
                confirmed_pages = max(0, min(confirmed_pages, task.requested_blocks))
            if confirmed_pages != task.requested_blocks:
                result = confirmed_pages
            else:
                checkpoint_hash = int(task.block_hash_ids[-1])
                checkpoint_result = task.checkpoint_store.match(
                    [checkpoint_hash], timeout=1.0
                )
                if checkpoint_result != 1:
                    checkpoint_result = task.checkpoint_store.write_snapshot(
                        checkpoint_hash,
                        task.checkpoint_snapshot,
                        timeout=1.0,
                        params=task.params,
                    )
                result = confirmed_pages if checkpoint_result == 1 else -1
            cache._log_as_result(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                result=result,
                started=started,
            )
            return result
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=exc,
                started=started,
            )
            return -1

    def enqueue_dummy(self, cache, rid: str) -> None:
        """Queue a no-op SPA task to preserve TP retirement ordering."""
        if cache.as_world_size <= 1:
            return
        self.submit_completed(
            _SPAWriteTask(
                rid=rid,
                page_endpoint=None,
                checkpoint_store=None,
                block_hash_ids=(),
                page_ids=(),
                checkpoint_snapshot=None,
                params={},
            )
        )


class SPAGCacheCoordinator(SPACacheCoordinator):
    """Coordinate SPAG page writes whose blocks carry their own SWA window.

    SPAG has no checkpoint endpoint. Every page block reserves a trailing SWA
    region and the blocks that close a ``swa_step`` checkpoint fill it from the
    request's ``ChunkTailSWAPool`` slots, so a matched page range is directly
    restorable instead of depending on a separate blob keyed on the range's final
    page.

    Slot ownership (the reason no eager pre-commit snapshot is needed here, in
    contrast to SPA's WAR in ``prepare_write``): the window was already copied
    out to the staging pool at the chunk boundary that produced it
    (``Scheduler.detect_and_may_copy_as_swa``), so nothing the local commit frees
    can invalidate it. ``req.as_swa_kv_slots`` is split at
    ``req.as_written_count``: below it the slots belong to a submitted write task
    and are released when that task retires, at or above it they belong to the
    request and are released with it.
    """

    write_thread_name = "as_spag_writer"

    @staticmethod
    def _swa_policy(cache):
        """Return the engine's ``SwaStepPolicy``, or ``None`` when SWA is off."""
        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        if getattr(kvcache, "chunk_tail_swa_pool", None) is None:
            return None
        return getattr(kvcache, "swa_policy", None)

    def state_slots(self, cache, req_pool_idx: int, swa_page_id: int) -> dict[str, int]:
        """Unreachable: SPAG stores no separate checkpoint blob."""
        raise NotImplementedError(
            "SPAG has no checkpoint endpoint; SWA rides inside the page block"
        )

    def floor_remote_terminal(self, cache, terminal_block: int) -> int:
        """Shrink the match to the last page that closes a checkpoint.

        Only checkpoint blocks carry a SWA window, so a match ending mid-stride
        would restore pages whose trailing window is a sentinel hole. The tree's
        SWA validator would then reject the range anyway, after the read already
        paid for it.
        """
        policy = self._swa_policy(cache)
        if policy is None:
            return terminal_block
        return policy.floor_to_checkpoint(terminal_block * cache.page_size) // (
            cache.page_size
        )

    def _read_payload_args(self, cache, req, load_info, loaded):
        """Name the destination SWA page for the restored trailing window.

        ``floor_remote_terminal`` guarantees the restored range ends on a
        checkpoint, so exactly one block of this read carries a window: the last
        one. ``swa_block_step`` only has to place the real offset on that block,
        and it is the same stride the write used for that checkpoint.
        """
        policy = self._swa_policy(cache)
        if policy is None:
            return (), {}
        swa_indices = cache.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            loaded[-cache.page_size :]
        )
        swa_page_id = int(swa_indices[0].item()) // cache.page_size
        if swa_page_id == 0:
            # Page 0 is the padding sink every cleared full->swa entry maps to,
            # so a real fresh allocation never lands there. Reading into it would
            # silently corrupt the sink instead of restoring anything.
            raise RuntimeError(
                "SPAG load-back resolved SWA page 0 (the padding sink); "
                "the full->swa mapping for the restored tail was cleared"
            )
        end_pos = (
            int(load_info["start_block"]) + int(load_info["block_count"])
        ) * cache.page_size
        return (), {
            "swa_ids": (swa_page_id,),
            "swa_block_step": policy.block_step_for_position(end_pos),
        }

    def _restored_swa_len(self, cache, token_count: int) -> int:
        """Keep the inherited page granularity.

        SPAG stores one ``swa_win`` window per checkpoint, i.e. only the upper
        half of the final SWA page comes back. Reporting the whole page is still
        correct -- and the only page-aligned option -- for the reasons in the base
        docstring.
        """
        return super()._restored_swa_len(cache, token_count)

    def _collect_swa_slots(self, cache, req, cache_len: int):
        """Map this write's checkpoints to staging slots, claiming the new ones.

        The returned tuple is indexed by checkpoint number from the *request's*
        start, because the SDK derives a block's checkpoint from its index alone.
        Entries this task does not own are ``-1``:

        * below ``as_written_count`` an earlier task owns (and will free) the
          slot, so reusing the id here could hand a recycled window to the Agent;
          the block itself is already stored with its window anyway.
        * at or above ``as_swa_copy_count`` nothing has been staged yet.
        """
        policy = self._swa_policy(cache)
        if policy is None:
            return (), 0
        total = policy.num_checkpoints(cache_len)
        if total <= 0:
            return (), 0
        staged = req.as_swa_kv_slots
        start = min(int(req.as_written_count), total, len(staged))
        end = min(int(req.as_swa_copy_count), total, len(staged))
        slots = [-1] * total
        for index in range(start, end):
            slots[index] = int(staged[index])
        return tuple(slots), max(start, end)

    def prepare_write(
        self,
        cache,
        req,
        token_ids: Sequence[int],
        snapshot: torch.Tensor,
    ) -> Optional[_SPAWriteTask]:
        """Prepare page identifiers and the per-checkpoint SWA slot map."""
        if cache.ascli is None or not token_ids:
            return None
        # Offload the page-aligned prefix even when the fill length carries a
        # partial tail page; the radix tree only publishes full pages.
        cache_len = (len(token_ids) // cache.page_size) * cache.page_size
        if cache_len == 0 or len(snapshot) < cache_len:
            return None
        token_ids = token_ids[:cache_len]
        requested = 0
        try:
            hashes = _request_block_hash_ids(
                req,
                token_ids,
                cache.page_size,
                cache_len // cache.page_size,
            )
            requested = len(hashes)
            swa_slot_ids, owned_end = self._collect_swa_slots(cache, req, cache_len)
            # page_ids are deliberately NOT built from this pre-commit snapshot:
            # super().cache_*_req frees the duplicate prefix slots it references
            # before the async gather runs. after_local_commit builds them from
            # the tree's ref-protected match.device_indices instead.
            return _SPAWriteTask(
                rid=req.rid,
                page_endpoint=cache.binding.page_endpoint,
                checkpoint_store=None,
                block_hash_ids=hashes,
                page_ids=None,
                checkpoint_snapshot=None,
                params=getattr(req, "as_kv_cache_params", None) or {},
                swa_slot_ids=swa_slot_ids,
                swa_owned_end=owned_end,
            )
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=req.rid,
                requested=requested,
                error=exc,
            )
            return None

    def _refloor_task(self, cache, context, task, match, cache_len: int) -> None:
        """Shrink the task to the published prefix, checkpoints included."""
        task.block_hash_ids = task.block_hash_ids[: cache_len // cache.page_size]
        policy = self._swa_policy(cache)
        if policy is None:
            return
        total = policy.num_checkpoints(cache_len)
        # Checkpoints past the published prefix stay with the request: their
        # staged windows are still valid, so the next chunk's write re-claims
        # them rather than dropping them.
        task.swa_slot_ids = task.swa_slot_ids[:total]
        task.swa_owned_end = min(task.swa_owned_end, total)

    def _commit_write_ownership(self, cache, context, task) -> None:
        """Advance the request's ownership frontier to this task's checkpoints."""
        req = context.req
        req.as_written_count = max(int(req.as_written_count), int(task.swa_owned_end))

    def _make_release_after_write(self, cache, task, page_ids) -> Callable[[], None]:
        """Also return the staged SWA windows this task owned."""
        base = super()._make_release_after_write(cache, task, page_ids)
        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        # Only slots this task claimed are non-sentinel (see _collect_swa_slots).
        slots = tuple(slot for slot in task.swa_slot_ids if slot != -1)

        released = False

        def _release_staging() -> None:
            nonlocal released
            try:
                base()
            finally:
                if not released:
                    released = True
                    free_chunk_tail_swa_slots(kvcache, slots)

        return _release_staging

    def _write_chunks(self, cache, task, confirmed_pages: int):
        """Split the unwritten block range into batches the SDK can address.

        The SDK decides which block carries a checkpoint from the block's index
        *within the batch* (``(idx + 1) % swa_block_step == 0``) and takes a
        single scalar step per call. So a batch must start on the checkpoint grid
        and may not straddle a tier boundary. ``confirmed_pages`` is floored onto
        the grid of the run it lands in: re-storing a few already-stored pages is
        idempotent, whereas starting mid-stride would shift every checkpoint of
        the batch onto the wrong block.
        """
        page_count = task.requested_blocks
        policy = self._swa_policy(cache)
        if policy is None or not task.swa_slot_ids:
            return [(confirmed_pages, page_count, (), None)]
        page_size = cache.page_size
        chunks = []
        covered = 0
        for first, last, block_step in policy.segments(0, len(task.swa_slot_ids)):
            start = policy.checkpoint_pos(first) // page_size - block_step
            end = policy.checkpoint_pos(last - 1) // page_size
            covered = end
            if end <= confirmed_pages:
                continue
            if start < confirmed_pages:
                whole_strides = (confirmed_pages - start) // block_step
                first += whole_strides
                start += whole_strides * block_step
            chunks.append((start, end, task.swa_slot_ids[first:last], block_step))
        tail_start = max(covered, confirmed_pages)
        if tail_start < page_count:
            # Trailing blocks close no checkpoint, so they carry no SWA ids and
            # the SDK sentinels their whole SWA region.
            chunks.append((tail_start, page_count, (), None))
        return chunks

    def do_write(self, cache, task: _SPAWriteTask) -> int:
        """Write SPAG page blocks together with their embedded SWA windows."""
        if not task.block_hash_ids:
            return 0
        started = time.perf_counter()
        try:
            page_ids = task.page_ids.materialize()
            # RAW fix (align with SPACacheCoordinator.do_write): record the
            # producer fence on the writer thread, targeting the injected
            # forward_stream, right before the gather.
            producer_ready = self.record_device_event(
                cache.device, self._producer_stream(cache)
            )
            store = task.page_endpoint.store
            confirmed_pages = self._match_pages(store, task)
            if confirmed_pages < task.requested_blocks:
                for start, end, swa_ids, block_step in self._write_chunks(
                    cache, task, confirmed_pages
                ):
                    store.write(
                        task.block_hash_ids[start:end],
                        page_ids[start:end],
                        timeout=1.0,
                        params=task.params,
                        producer_ready=producer_ready,
                        swa_ids=swa_ids,
                        swa_block_step=block_step,
                        logical_start=start,
                    )
                confirmed_pages = self._match_pages(store, task)
            cache._log_as_result(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                result=confirmed_pages,
                started=started,
            )
            return confirmed_pages
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=exc,
                started=started,
            )
            return -1

    @staticmethod
    def _match_pages(store, task: _SPAWriteTask) -> int:
        """Return how much of this task's page range the store already holds."""
        matched = store.match(task.block_hash_ids, timeout=1.0)
        return max(0, min(matched, task.requested_blocks))


class KDACacheCoordinator(CheckpointedPageCacheCoordinator):
    """Coordinate full-attention pages and recurrent KDA endpoint checkpoints."""

    write_thread_name = "as_kda_writer"

    def capture_finished(
        self, cache, req, is_insert: bool
    ) -> AttentionStoreWriteContext:
        """Snapshot KDA page slots for a finished request."""
        token_ids = array("q", req.origin_input_ids)
        token_ids.extend(req.output_ids)
        snapshot = cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=is_insert,
        )

    def capture_unfinished(self, cache, req) -> AttentionStoreWriteContext:
        """Snapshot KDA page slots for an unfinished request."""
        token_ids = req.get_fill_ids()
        snapshot = cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=True,
        )

    def after_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Enqueue eligible KDA state after the local commit."""
        if context.is_insert and cache._is_prefill_role() and cache.ascli is not None:
            self.enqueue(
                cache,
                context.req,
                context.token_ids,
                context.snapshot,
            )

    def enqueue(
        self,
        cache,
        req,
        token_ids: Sequence[int],
        snapshot: torch.Tensor,
    ) -> None:
        """Build and submit a KDA page-and-checkpoint write."""
        task = None
        checkpoint_slot = None
        page_ids = None
        cache_len = 0
        try:
            if not isinstance(token_ids, array) or token_ids.typecode != "q":
                token_ids = array("q", token_ids)
            match = UnifiedRadixCache.match_prefix(
                cache,
                MatchPrefixParams(key=RadixKey(token_ids, req.extra_key)),
            )
            cache_len = len(match.device_indices)
            node = match.last_device_node
            mamba_data = (
                node.component_data[ComponentType.MAMBA] if node is not None else None
            )
            if (
                cache_len < cache.attention_store_block_size
                or cache_len % cache.attention_store_block_size != 0
                or node is None
                or mamba_data.value is None
                or len(snapshot) < cache_len
            ):
                self.enqueue_dummy(cache, req.rid)
                return

            block_size = cache.attention_store_block_size
            page_count = cache_len // block_size
            hashes = _request_block_hash_ids(
                req,
                token_ids,
                block_size,
                page_count,
            )
            endpoint_hash = hashes[-1]
            if mamba_data.metadata.get(_KDA_PENDING_HASH) is not None or (
                endpoint_hash == mamba_data.metadata.get(_KDA_PUBLISHED_HASH)
            ):
                self.enqueue_dummy(cache, req.rid)
                return
            write_key = (cache_len, endpoint_hash)
            if getattr(req, "_as_kda_last_submitted_write", None) == write_key:
                self.enqueue_dummy(cache, req.rid)
                return
            checkpoint_slot = self.prepare_checkpoint_slot(mamba_data.value)
            # WAR fix: gather from the tree's canonical, ref-protected slots
            # (match.device_indices), NOT the pre-commit request snapshot.
            # super().cache_finished_req (run between capture and this enqueue)
            # frees the duplicate prefix slots kv_indices[cache_protected_len:
            # prefix_len] that live in `snapshot`; a later forward reuses and
            # overwrites them before the async writer's gather runs, so gathering
            # from `snapshot` reads corrupted (reused) KV. inc_lock_ref(node)
            # below protects device_indices, so page_ids must come from there.
            page_ids = self.prepare_page_ids(cache, match.device_indices)
            # Producer fence is recorded in do_write right before the gather; an
            # enqueue-time event is too early to cover a still-draining write.
            producer_ready = None

            def release_staging() -> None:
                page_ids.release()
                checkpoint_slot.release()

            task = _KDAWriteTask(
                rid=req.rid,
                node=node,
                block_hash_ids=hashes,
                page_ids=page_ids,
                checkpoint_slot=checkpoint_slot,
                producer_ready=producer_ready,
                params=getattr(req, "as_kv_cache_params", None) or {},
                request=req,
                write_key=write_key,
                release_after_write=release_staging,
            )
            task.release = _acquire_node_release(cache, node)
            mamba_data.metadata[_KDA_PENDING_HASH] = endpoint_hash
            req._as_kda_last_submitted_write = write_key
            try:
                self.submit_write(cache, task)
            except Exception:
                if mamba_data.metadata.get(_KDA_PENDING_HASH) == endpoint_hash:
                    del mamba_data.metadata[_KDA_PENDING_HASH]
                if getattr(req, "_as_kda_last_submitted_write", None) == write_key:
                    del req._as_kda_last_submitted_write
                raise
        except Exception as exc:
            if task is not None:
                self._discard_unsubmitted(cache, task)
            else:
                if page_ids is not None:
                    page_ids.release()
                if checkpoint_slot is not None:
                    checkpoint_slot.release()
            cache._log_as_failure(
                "write",
                rid=req.rid,
                requested=cache_len // cache.attention_store_block_size,
                error=exc,
            )
            self.enqueue_dummy(cache, req.rid)

    def do_write(self, cache, task: _KDAWriteTask) -> int:
        """Write KDA pages and recurrent state through the composite store."""
        started = time.perf_counter()
        try:
            page_ids = task.page_ids.materialize()
            # Record the producer fence here (writer thread, right before the
            # gather) capturing the producer stream's current tail, so a late
            # page write not covered by an enqueue-time event is still fenced.
            producer_ready = self.record_device_event(
                cache.device, self._producer_stream(cache)
            )
            written = cache.ascli.write(
                task.block_hash_ids,
                page_ids,
                {"checkpoint": task.checkpoint_slot.materialize()},
                timeout=1.0,
                params=task.params,
                producer_ready=producer_ready,
            )
            cache._log_as_result(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                result=written,
                started=started,
            )
            return written
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=exc,
                started=started,
            )
            return -1

    def finish(self, _cache, task: _KDAWriteTask, result: int) -> None:
        """Publish or clear the endpoint hash after KDA write completion."""
        if task.endpoint_hash == 0:
            return
        metadata = task.node.component_data[ComponentType.MAMBA].metadata
        if metadata.get(_KDA_PENDING_HASH) == task.endpoint_hash:
            del metadata[_KDA_PENDING_HASH]
        if result == task.requested_blocks:
            metadata[_KDA_PUBLISHED_HASH] = task.endpoint_hash
        elif task.request is not None and (
            getattr(task.request, "_as_kda_last_submitted_write", None)
            == task.write_key
        ):
            del task.request._as_kda_last_submitted_write

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Load matched KDA pages and recurrent state into unified storage."""
        anchor = params.best_match_node
        empty = torch.empty((0,), dtype=torch.int64, device=cache.device)
        if params.host_hit_length <= 0:
            return empty, anchor

        anchor, load_info, req = cache._take_attention_store_load_context(params)
        if cache.ascli is None or load_info is None or req is None:
            return empty, anchor

        loaded = self._read_blocks(cache, load_info)
        if loaded is None:
            return empty, anchor
        loaded_indices, value_transaction, checkpoint_allocation = loaded
        req.as_read_pending_lock = True
        return self._insert_loaded(
            cache,
            req,
            anchor,
            load_info,
            loaded_indices,
            value_transaction,
            checkpoint_allocation,
        )

    def _read_blocks(self, cache, load_info):
        block_count = int(load_info["block_count"])
        rid = load_info.get("rid", "-")
        started = time.perf_counter()
        block_size = cache.attention_store_block_size
        token_count = block_count * block_size
        value_transaction = cache.component_value_transaction()
        try:
            full_allocation = value_transaction.allocate(
                ComponentType.FULL, token_count
            )
            alloc_indices = full_allocation.value
        except Exception as exc:
            value_transaction.close()
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=block_count,
                error=exc,
                started=started,
            )
            cache._as_min(0)
            return None
        if len(alloc_indices) != token_count:
            value_transaction.close()
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=block_count,
                error=RuntimeError(
                    f"allocated {len(alloc_indices)} of {token_count} tokens"
                ),
                started=started,
            )
            cache._as_min(0)
            return None

        try:
            page_ids = _contiguous_page_ids(alloc_indices, cache.page_size)
        except ValueError as exc:
            value_transaction.close()
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=block_count,
                error=exc,
                started=started,
            )
            cache._as_min(0)
            return None

        try:
            checkpoint_allocation = value_transaction.allocate(ComponentType.MAMBA)
        except Exception as exc:
            value_transaction.close()
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=block_count,
                error=exc,
                started=started,
            )
            cache._as_min(0)
            return None
        keep_transaction = False
        try:
            host_ids = checkpoint_allocation.host_value
            if host_ids is None or len(host_ids) != 1:
                raise RuntimeError(
                    "AttentionStore KDA load-back requires a host-side checkpoint "
                    "slot ID; refusing to scalarize the scheduler GPU tensor"
                )
            state_id = int(host_ids[0])
            if cache._as_min(1) == 0:
                return None
            read_error = None
            try:
                read_count = cache.ascli.read(
                    load_info["hashes"],
                    page_ids,
                    {"checkpoint": state_id},
                    timeout=5.0,
                    remote_addrs=load_info.get("remote_addrs") or None,
                )
            except Exception as exc:
                read_error = exc
                read_count = 0

            read_count = cache._as_min(read_count)

            if read_error is None:
                cache._log_as_result(
                    "load_back",
                    rid=rid,
                    requested=block_count,
                    result=read_count,
                    started=started,
                )
            else:
                cache._log_as_failure(
                    "load_back",
                    rid=rid,
                    requested=block_count,
                    error=read_error,
                    started=started,
                )

            if read_count != block_count:
                return None
            keep_transaction = True
            return alloc_indices, value_transaction, checkpoint_allocation
        finally:
            if not keep_transaction:
                value_transaction.close()

    def _insert_loaded(
        self,
        cache,
        req,
        anchor,
        load_info,
        loaded_indices: torch.Tensor,
        value_transaction,
        checkpoint_allocation,
    ) -> Tuple[torch.Tensor, Any]:
        local_hit_len = load_info["start_block"] * cache.attention_store_block_size
        total_prefix_len = local_hit_len + len(loaded_indices)
        if len(req.prefix_indices) != local_hit_len:
            logger.error(
                "AS cache insert rejected: cache_plan=%s rid=%s "
                "reason=prefix_length_mismatch expected=%d actual=%d",
                cache.attention_store_plan.name,
                req.rid,
                local_hit_len,
                len(req.prefix_indices),
            )
            value_transaction.close()
            return loaded_indices[:0], anchor
        try:
            token_ids = req.full_untruncated_fill_ids[:total_prefix_len]
            values = torch.cat([req.prefix_indices, loaded_indices]).to(
                dtype=torch.int64, copy=True
            )
            result = cache.insert(
                InsertParams(
                    key=RadixKey(token_ids, req.extra_key),
                    value=values,
                    mamba_value=checkpoint_allocation.value,
                    prev_prefix_len=local_hit_len,
                ),
                value_transaction=value_transaction,
            )
            value_transaction.commit_insert(result)
        finally:
            # Release values not adopted by the tree.
            value_transaction.close()

        match = UnifiedRadixCache.match_prefix(
            cache,
            MatchPrefixParams(
                key=RadixKey(token_ids, req.extra_key),
                cow_mamba=True,
                req=req,
            ),
        )
        tree_indices = match.device_indices[local_hit_len:total_prefix_len].to(
            dtype=loaded_indices.dtype
        )
        if len(tree_indices) != len(loaded_indices):
            logger.warning(
                "AS cache insert: cache_plan=%s rid=%s status=error "
                "reason=shortened_prefix expected_tokens=%s actual_tokens=%s",
                cache.attention_store_plan.name,
                req.rid,
                len(loaded_indices),
                len(tree_indices),
            )
        elif load_info["hashes"]:
            match.last_device_node.component_data[ComponentType.MAMBA].metadata[
                _KDA_PUBLISHED_HASH
            ] = int(load_info["hashes"][-1])
        return tree_indices, match.last_device_node

    def enqueue_dummy(self, cache, rid: str) -> None:
        """Queue a no-op KDA task to preserve TP retirement ordering."""
        if cache.as_world_size <= 1:
            return
        self.submit_completed(
            _KDAWriteTask(
                rid=rid,
                node=cache.root_node,
                block_hash_ids=(),
                page_ids=(),
                checkpoint_slot=self.prepare_checkpoint_slot(0),
                params={},
            )
        )
