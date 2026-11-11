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
"""Coordinator for page-only AttentionStore cache plans."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from .coordinator import (
    AttentionStoreCoordinator,
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

AS_MIN_MATCH_BLOCKS = 5


@dataclass
class _PageWriteTask:
    rid: str
    endpoint: Optional[AttentionStoreEndpointBinding]
    block_hash_ids: Sequence[int]
    data_id_list: Any
    write_params: Any
    producer_ready: Any = None
    release_after_write: Callable[[], None] = _noop_release
    release: Callable[[], None] = _noop_release
    owner: Optional[AttentionStoreWriteOwner] = None

    @property
    def requested_blocks(self) -> int:
        """Return the number of page blocks requested by this task."""
        return len(self.block_hash_ids)


class FullPageCacheCoordinator(AttentionStoreCoordinator):
    """Coordinate remote match, load-back, and writes for typed page endpoints."""

    def capture_finished(
        self, cache, req, is_insert: bool
    ) -> AttentionStoreWriteContext:
        """Snapshot the committed page slots of a finished request."""
        token_count = req._cache_commit_len()
        token_ids = (req.origin_input_ids + req.output_ids)[:token_count]
        snapshot = cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :token_count
        ].clone()
        return AttentionStoreWriteContext(
            req=req,
            token_ids=token_ids,
            snapshot=snapshot,
            is_insert=is_insert,
        )

    def extend_match(
        self,
        cache,
        params: MatchPrefixParams,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        """Extend a local FULL-component match with remote page hits."""
        req = params.req
        if (
            req is None
            or req.ever_read_as
            or not req.as_block_hashs
            or cache.ascli is None
        ):
            return None
        block_size = cache.attention_store_block_size
        max_hit_length = len(params.key) // block_size * block_size
        local_hit_length = local_result.device_indices.numel()
        aligned_local_hit_length = min(
            local_hit_length // block_size * block_size, max_hit_length
        )
        if local_hit_length != aligned_local_hit_length:
            local_result = UnifiedRadixCache.match_prefix(
                cache,
                MatchPrefixParams(
                    key=RadixKey(
                        params.key.token_ids,
                        params.key.extra_key,
                        is_bigram=params.key.is_bigram,
                        limit=aligned_local_hit_length,
                    ),
                    cow_mamba=params.cow_mamba,
                    req=req,
                ),
            )
        if aligned_local_hit_length == max_hit_length:
            return local_result if local_hit_length != aligned_local_hit_length else None
        return self._match_remote(cache, req, local_result)

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Load remote FULL-component pages into UnifiedRadixCache."""
        anchor = params.best_match_node
        empty = torch.empty((0,), dtype=torch.int64, device=cache.device)
        if params.host_hit_length <= 0:
            return empty, anchor

        anchor, load_info, req = cache._take_attention_store_load_context(params)
        if cache.ascli is None or load_info is None or req is None:
            return empty, anchor

        loaded_indices = self._load_pages(cache, load_info, params.host_hit_length)
        if loaded_indices is None or len(loaded_indices) == 0:
            return empty, anchor

        req.as_read_pending_lock = True
        return self._insert_loaded(cache, req, anchor, load_info, loaded_indices)

    def _match_remote(
        self,
        cache,
        req,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        block_size = cache.attention_store_block_size
        local_hit_len = len(local_result.device_indices) // block_size * block_size
        start_block = local_hit_len // block_size
        # Router hints describe one namespace/host. Multi-node PP requests use
        # an Agent-coordinated quote before this coordinator is entered.
        remote_addrs = req.as_max_hit_host_ip if cache.pp_size <= 1 else []
        remote_len = req.as_max_hit_token_len if cache.pp_size <= 1 else []

        if remote_len and len(remote_len) == 1 and remote_len[0] > 0:
            as_hit_tokens = (
                (remote_len[0] - local_hit_len) // block_size
            ) * block_size
            if as_hit_tokens <= 0:
                return None
        else:
            requested_hashes = req.as_block_hashs.hash_ids[start_block:]
            requested = len(requested_hashes)
            started = time.perf_counter()
            match_error = None
            try:
                match_count = cache.ascli.match(requested_hashes, timeout=1.0)
            except Exception as exc:
                match_error = exc
                match_count = 0

            match_count = cache._as_min(match_count)
            max_blocks = max(
                (len(req.full_untruncated_fill_ids) - local_hit_len - 1)
                // block_size,
                0,
            )
            match_count = min(match_count, max_blocks)
            if match_count < AS_MIN_MATCH_BLOCKS:
                match_count = 0
            if match_error is None:
                cache._log_as_result(
                    "match",
                    rid=req.rid,
                    requested=requested,
                    result=match_count,
                    started=started,
                )
            else:
                cache._log_as_failure(
                    "match",
                    rid=req.rid,
                    requested=requested,
                    error=match_error,
                    started=started,
                )
            if match_count == 0:
                return None
            as_hit_tokens = match_count * block_size

        max_as_tokens = (
            (len(req.full_untruncated_fill_ids) - local_hit_len - 1)
            // block_size
            * block_size
        )
        if max_as_tokens <= 0:
            return None
        as_hit_tokens = min(as_hit_tokens, max_as_tokens)

        anchor = local_result.last_device_node
        req._as_load_info = {
            "rid": req.rid,
            "start_block": start_block,
            "as_hit_tokens": as_hit_tokens,
            "as_block_hashs": req.as_block_hashs,
            "remote_addrs": remote_addrs,
            "remote_len": remote_len,
        }
        return MatchResult(
            device_indices=local_result.device_indices,
            last_device_node=anchor,
            last_host_node=anchor,
            best_match_node=anchor,
            host_hit_length=as_hit_tokens,
        )

    def _load_pages(
        self,
        cache,
        load_info: dict,
        host_hit_length: int,
    ) -> Optional[torch.Tensor]:
        rid = load_info.get("rid", "-")
        block_size = cache.attention_store_block_size
        requested = host_hit_length // block_size
        started = time.perf_counter()
        as_hit_tokens = load_info["as_hit_tokens"]
        available = cache.token_to_kv_pool_allocator.available_size()
        if available < as_hit_tokens and cache.evictable_size() > 0:
            evict_tokens = max(as_hit_tokens - available, 0) + cache.page_size
            cache.evict(EvictParams(num_tokens=evict_tokens))

        try:
            alloc_indices = cache.token_to_kv_pool_allocator.alloc(as_hit_tokens)
        except Exception as exc:
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=requested,
                error=exc,
                started=started,
            )
            cache._as_min(0)
            return None
        if alloc_indices is None or len(alloc_indices) == 0:
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=requested,
                error=RuntimeError(
                    "allocation failed: "
                    f"need={as_hit_tokens}, "
                    f"available={cache.token_to_kv_pool_allocator.available_size()}, "
                    f"evictable={cache.evictable_size()}"
                ),
                started=started,
            )
            cache._as_min(0)
            return None
        try:
            data_ids = list(_contiguous_page_ids(alloc_indices, cache.page_size))
        except ValueError as exc:
            cache.token_to_kv_pool_allocator.free(alloc_indices)
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=requested,
                error=exc,
                started=started,
            )
            cache._as_min(0)
            return None

        if cache._as_min(1) == 0:
            cache.token_to_kv_pool_allocator.free(alloc_indices)
            return None

        start_block = load_info["start_block"]
        block_hash_ids = load_info["as_block_hashs"].hash_ids[
            start_block : start_block + len(data_ids)
        ]
        requested = len(block_hash_ids)
        remote_addrs = load_info.get("remote_addrs", [])
        remote_addr = remote_addrs[0] if remote_addrs else ""
        read_error = None
        try:
            read_count = cache.ascli.read(
                block_hash_ids,
                data_ids,
                timeout=5.0,
                remote_addrs=(remote_addr,) if remote_addr else (),
                logical_start=start_block,
            )
        except Exception as exc:
            read_error = exc
            read_count = 0
        read_count = cache._as_min(read_count)
        if read_error is None:
            cache._log_as_result(
                "load_back",
                rid=rid,
                requested=requested,
                result=read_count,
                started=started,
            )
        else:
            cache._log_as_failure(
                "load_back",
                rid=rid,
                requested=requested,
                error=read_error,
                started=started,
            )
        if read_count <= 0:
            cache.token_to_kv_pool_allocator.free(alloc_indices)
            return None

        loaded_tokens = read_count * block_size
        loaded_indices = alloc_indices[:loaded_tokens]
        if loaded_tokens < len(alloc_indices):
            cache.token_to_kv_pool_allocator.free(alloc_indices[loaded_tokens:])
        return loaded_indices

    def _insert_loaded(
        self,
        cache,
        req,
        anchor,
        load_info,
        loaded_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        block_size = cache.attention_store_block_size
        local_hit_len = load_info["start_block"] * block_size
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
            cache.token_to_kv_pool_allocator.free(loaded_indices)
            return loaded_indices[:0], anchor
        token_ids = req.full_untruncated_fill_ids[:total_prefix_len]
        bigram_len = max(0, len(token_ids) - 1) if cache.tree_core.is_eagle else len(token_ids)
        final_key_len = bigram_len // block_size * block_size
        if final_key_len == 0:
            cache.token_to_kv_pool_allocator.free(loaded_indices)
            return loaded_indices[:0], anchor

        full_values = torch.cat([req.prefix_indices, loaded_indices])
        values = full_values[:final_key_len].to(dtype=torch.int64, copy=True)
        tail_start = max(final_key_len - local_hit_len, 0)
        if tail_start < len(loaded_indices):
            cache.token_to_kv_pool_allocator.free(loaded_indices[tail_start:])
            loaded_indices = loaded_indices[:tail_start]

        cache.insert(
            InsertParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=cache.tree_core.is_eagle),
                value=values,
                prev_prefix_len=local_hit_len,
            )
        )
        match = UnifiedRadixCache.match_prefix(
            cache,
            MatchPrefixParams(
                key=RadixKey(token_ids, req.extra_key, is_bigram=cache.tree_core.is_eagle),
            ),
        )
        if match.last_device_node is None:
            return loaded_indices, anchor

        tree_full_indices = match.device_indices
        if len(tree_full_indices) >= final_key_len:
            # _insert_helper already freed the slots the tree displaced.
            loaded_indices = tree_full_indices[
                local_hit_len : local_hit_len + len(loaded_indices)
            ].to(dtype=loaded_indices.dtype)
        return loaded_indices, match.last_device_node

    def after_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Enqueue eligible page data after the local commit."""
        if context.is_insert and cache._is_prefill_role() and cache.ascli is not None:
            self.enqueue(
                cache,
                context.req,
                context.token_ids,
                context.snapshot,
            )

    def enqueue_dummy(
        self, rid: str, owner: Optional[AttentionStoreWriteOwner] = None
    ) -> None:
        """Queue a no-op task to preserve TP retirement ordering."""
        task = _PageWriteTask(
            rid=rid,
            endpoint=None,
            block_hash_ids=(),
            data_id_list=[],
            write_params={},
            owner=owner,
        )
        self.submit_completed(task, owner=owner)

    def do_write(self, cache, task: _PageWriteTask) -> int:
        """Write a prepared page task to its endpoint."""
        started = time.perf_counter()
        try:
            endpoint = task.endpoint
            if endpoint is None:
                return 0
            data_ids = task.data_id_list.materialize()
            # Record the producer fence here (writer thread, right before the
            # gather) so it captures the producer stream's current tail; a late
            # page write not covered by an enqueue-time event is still fenced.
            producer_ready = self.record_device_event(
                cache.device, self._producer_stream(cache)
            )
            written = endpoint.store.write(
                task.block_hash_ids,
                data_ids,
                timeout=1.0,
                params=task.write_params,
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

    def enqueue(
        self,
        cache,
        req,
        token_ids: Sequence[int],
        snapshot: torch.Tensor,
    ) -> None:
        """Build and submit a page write for an eligible radix prefix."""
        if cache.binding is None:
            if cache.as_world_size > 1:
                self.enqueue_dummy(req.rid)
            return
        task = None
        num_pages = 0
        try:
            block_size = cache.attention_store_block_size
            num_pages = len(token_ids) // block_size
            if num_pages <= 0:
                if cache.as_world_size > 1:
                    self.enqueue_dummy(req.rid)
                return
            aligned_tokens = num_pages * block_size
            hashes = _request_block_hash_ids(
                req,
                token_ids,
                block_size,
                num_pages,
            )
            kv_indices = snapshot[:aligned_tokens]
            match = UnifiedRadixCache.match_prefix(
                cache,
                MatchPrefixParams(
                    key=RadixKey(token_ids, req.extra_key, is_bigram=cache.tree_core.is_eagle)
                ),
            )
            node = match.last_device_node
            if node is None:
                cache._log_as_failure(
                    "write",
                    rid=req.rid,
                    requested=num_pages,
                    error=RuntimeError("no radix node matched"),
                )
                if cache.as_world_size > 1:
                    self.enqueue_dummy(req.rid)
                return
            data_ids = self.prepare_page_ids(cache, kv_indices)

            task = _PageWriteTask(
                rid=req.rid,
                endpoint=cache.binding.page_endpoint,
                block_hash_ids=hashes,
                data_id_list=data_ids,
                write_params=getattr(req, "as_kv_cache_params", None) or {},
                # Producer fence is recorded in do_write, on the writer thread
                # right before the gather -- an enqueue-time event misses page
                # writes that are still draining, so recording here is too early.
                producer_ready=None,
                release_after_write=data_ids.release,
            )
            task.release = _acquire_node_release(cache, node)
            self.submit_write(cache, task)
        except Exception as exc:
            if task is not None:
                self._discard_unsubmitted(cache, task)
            cache._log_as_failure(
                "write",
                rid=req.rid,
                requested=num_pages,
                error=exc,
            )
            if cache.as_world_size > 1:
                self.enqueue_dummy(req.rid)
