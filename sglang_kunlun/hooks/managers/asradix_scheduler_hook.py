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
"""Hooks for ``sglang.srt.managers.scheduler`` / ``Req`` -- AttentionStore.

Ports the scheduler side of aiak_sglang 1d855da5 (luno-5195): the five new
``Scheduler`` methods, the per-request AS state on ``Req``, and the four
injection points in the event loops and flush path.

Deliberate deviations, all forced by hooking rather than editing:

``as_block_hashs`` is computed here, in ``Req.__init__``, instead of in the
tokenizer manager. Upstream adds four fields to ``TokenizedGenerateReqInput`` to
carry them over ZMQ, but that is a ``msgspec.Struct`` with ``array_like=True``:
a plugin can neither add fields to it nor change the positional wire format.
``Tokens(input_ids, AS_BLOCK_SIZE)`` is a pure function of the input ids, which
the scheduler already has, so the hop is unnecessary. The caller-supplied hints
(``as_kv_cache_params``, ``as_max_hit_host_ip``, ``as_max_hit_token_len``) have
no path in without that wire change and stay empty; they are optimisation hints
and the AS match runs without them.

``retire_ascache_writes`` is driven from ``ingest_requests`` rather than being
pasted at the top of each event loop. Upstream adds the call immediately before
``self.ingest_requests()`` in ``event_loop_normal``, ``event_loop_overlap`` and
both disagg-prefill loops, so hooking that one callee reproduces the placement
in all of them -- and also covers ``scheduler_pp_mixin``, which upstream
deliberately skips and flags as a latent slot leak.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.mem_cache.as_config import as_cfg, asradix_enabled
from sglang_kunlun.mem_cache.chunk_tail_swa_pool import free_chunk_tail_swa_slots

logger = logging.getLogger(__name__)

_SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"

# One Scheduler per process. The module-level function hooks
# (``maybe_cache_unfinished_req``) need it to reach the pools, and their target
# signatures do not carry it.
_SCHEDULER_REF: Optional[Any] = None


def current_scheduler() -> Optional[Any]:
    """The Scheduler of this process, or ``None`` before it is constructed."""
    return _SCHEDULER_REF


def _init_attention_store(self, server_args) -> None:
    """Attach the AttentionStore (asradix) binding for the single-node
    (DP1/TP8, no-PP) DeepSeek-V4/V4.1 case.

    Builds one InitContext, creates the SPA/SPAG binding via the tree cache,
    refs it, and injects the model forward_stream into the write coordinator so
    its producer fence targets the stream the KV writes actually run on. On any
    failure it disables AS rather than crashing the scheduler.
    """
    try:
        from sglang.srt.runtime_context import get_device, get_parallel
        from sglang_kunlun.mem_cache.asradix.types import (
            AS_BLOCK_SIZE,
            AttentionStoreInitContext,
        )

        pod_name = os.getenv("POD_NAME")
        model_id = os.getenv("MODEL_ID")
        if pod_name is None or model_id is None:
            raise ValueError(
                "AttentionStore is enabled but POD_NAME / MODEL_ID env not set"
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        cache_plan = self.tree_cache.attention_store_plan
        layer_info = self.tp_worker.model_runner.layer_info
        target_layer_ids = tuple(
            range(int(layer_info.start_layer), int(layer_info.end_layer))
        )
        page_shard_id = int(get_parallel().attn_tp_rank)
        page_shard_num = int(get_parallel().attn_tp_size)
        draft_kv_cache, draft_layer_ids = self._as_draft_swa_layers(
            cache_plan.name, len(target_layer_ids)
        )
        gpu_count = as_cfg().per_node_gpu_num
        logger.info(
            "[AS] init: plan=%s pod=%s shard=%d/%d layers=%d draft_layers=%d "
            "page_size=%d gpu_count=%d",
            cache_plan.name,
            pod_name,
            page_shard_id,
            page_shard_num,
            len(target_layer_ids),
            len(draft_layer_ids),
            self.page_size,
            gpu_count,
        )
        ctx = AttentionStoreInitContext(
            kv_cache=kv_cache,
            req_to_token_pool=self.req_to_token_pool,
            namespace_base=model_id,
            pod_name=pod_name,
            model_version="v1",
            page_size=self.page_size,
            block_size=AS_BLOCK_SIZE,
            tp_rank=get_parallel().attn_tp_rank,
            tp_size=get_parallel().attn_tp_size,
            pp_rank=0,
            pp_size=1,
            node_pp_start=0,
            node_pp_end=1,
            dp_rank=get_parallel().attn_dp_rank,
            device_id=get_device().gpu_id,
            gpu_count=gpu_count,
            target_layer_ids=target_layer_ids,
            page_layer_ids=target_layer_ids,
            draft_kv_cache=draft_kv_cache,
            draft_layer_ids=draft_layer_ids,
            page_shard_id=page_shard_id,
            page_shard_num=page_shard_num,
            page_layer_num=len(target_layer_ids),
            endpoint_schema=dict(cache_plan.schema_factory(kv_cache)),
        )
        binding = self.tree_cache.create_attention_store_binding(ctx)
        self.tree_cache.ref_as_binding(binding)
        self.ascli = binding.cache_store
        _as_coord = getattr(self.tree_cache, "_as_coordinator", None)
        if _as_coord is not None:
            _as_coord.forward_stream = getattr(
                getattr(self.tp_worker, "model_runner", None),
                "forward_stream",
                None,
            )
        logger.info(
            "[AS] initialized: plan=%s namespace=%s forward_stream=%s",
            cache_plan.name,
            model_id,
            getattr(_as_coord, "forward_stream", None),
        )
    except Exception as exc:
        logger.error(
            "[AS] AttentionStore init failed; disabling AS: %s", exc, exc_info=True
        )
        self.enable_asradix_cache = False
        self.ascli = None


def _as_draft_swa_layers(
    self, plan_name: str, target_layer_count: int
) -> Tuple[Any, Tuple[int, ...]]:
    """Resolve the draft SWA layers the AttentionStore binding must cover.

    Returns ``(draft_kv_cache, draft_layer_ids)``, both empty when there is
    nothing to fan out to. Only SPAG consumes them: it folds the draft's SWA
    layers into the same block segment as the target's, so a restored checkpoint
    brings back both. The other plans pair ``draft_layer_ids`` with
    ``page_layer_ids`` instead, and this branch never populates those.
    """
    if plan_name != "SPAG":
        return None, ()
    draft_pool = self.draft_worker.primary_draft_kv_pool if self.draft_worker else None
    if draft_pool is None:
        if self.draft_worker is not None:
            # ngram and frozen-KV MTP have no draft KV pool at all, so there is
            # no draft SWA state to persist. Correct, but say so: the
            # alternative is wondering later why draft_layers=0.
            logger.info(
                "[AS] draft SWA fan-out disabled: %s exposes no draft KV pool",
                type(self.draft_worker).__name__,
            )
        return None, ()
    draft_swa = getattr(draft_pool, "swa_kv_pool", None)
    # The layer count MUST come from the draft SWA pool itself. Under DSpark the
    # pool holds one layer per stage while num_nextn_predict_layers stays 1, so
    # deriving it from the server arg would register a single layer and leave the
    # rest of the segment addressing the target's buffers.
    draft_swa_layers = int(getattr(draft_swa, "layer_num", 0) or 0)
    if draft_swa_layers <= 0:
        logger.warning(
            "[AS] draft SWA fan-out disabled: draft KV pool has no SWA layers"
        )
        return None, ()
    self._assert_draft_swa_geometry(draft_pool)
    return draft_pool, tuple(
        range(target_layer_count, target_layer_count + draft_swa_layers)
    )


def _assert_draft_swa_geometry(self, draft_pool) -> None:
    """Fail AS init unless the draft SWA pool is slot-compatible with target.

    ``detect_and_may_copy_as_swa`` stages each checkpoint's window into both
    pools using the *target's* staging slot id and the *target's* SWA page index.
    That is only meaningful while the two pools share one slot-id space and one
    ``full_to_swa_index_mapping``; they do today (the draft is allocated with one
    slot per target token, in the same process), but nothing in the type system
    says so, and a divergence would stage the wrong page instead of failing.
    """
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    target = self.token_to_kv_pool_allocator.get_kvcache()
    if not isinstance(target, DeepSeekV4TokenToKVPool) or not isinstance(
        draft_pool, DeepSeekV4TokenToKVPool
    ):
        raise RuntimeError(
            "AS draft SWA fan-out requires DeepSeekV4TokenToKVPool on both the "
            "target and the draft worker"
        )
    if not draft_pool.compression_ratios or not all(
        ratio == 0 for ratio in draft_pool.compression_ratios
    ):
        raise RuntimeError("AS draft SWA fan-out expects SWA-only NextN layers")
    if target._unified_kv != draft_pool._unified_kv:
        raise RuntimeError(
            "AS draft SWA fan-out requires the same unified-KV mode on the "
            "target and the draft worker"
        )
    if target._unified_kv:
        raise RuntimeError(
            "AS draft SWA fan-out requires the paged SWA pool: the unified-KV "
            "ring has no page halves to stage a window from"
        )
    if target.full_to_swa_index_mapping is not draft_pool.full_to_swa_index_mapping:
        raise RuntimeError(
            "AS draft SWA fan-out requires the target and draft pools to share "
            "one full->swa index mapping object"
        )
    target_geometry = (target.page_size, target.sliding_window)
    draft_geometry = (draft_pool.page_size, draft_pool.sliding_window)
    if target_geometry != draft_geometry:
        raise RuntimeError(
            "AS draft SWA fan-out requires identical paged SWA geometry: "
            f"target={target_geometry}, draft={draft_geometry}"
        )
    target_tail = getattr(target, "chunk_tail_swa_pool", None)
    draft_tail = getattr(draft_pool, "chunk_tail_swa_pool", None)
    if target_tail is None or draft_tail is None:
        raise RuntimeError(
            "AS draft SWA fan-out requires a chunk-tail staging pool on both "
            "workers (target=%s, draft=%s)"
            % (target_tail is not None, draft_tail is not None)
        )
    if target_tail.num_pages != draft_tail.num_pages:
        raise RuntimeError(
            "AS draft SWA staging pools must be the same size, because a slot id "
            "allocated on the target is used verbatim on the draft: "
            f"target={target_tail.num_pages}, draft={draft_tail.num_pages}"
        )


def _detect_and_may_copy_as_swa(self, req) -> None:
    """Rescue the tail SWA window at every swa_step checkpoint this request has
    now committed.

    Must run right after a chunk's forward completes: ``UnifiedRadixCache``
    evicts everything older than ``max(sliding_window, page_size)``, so a
    checkpoint's window is only still in HBM during the chunk that produced it.
    A loop (not an ``if``) because one forward can cross several checkpoints.
    """
    kvcache = self.token_to_kv_pool_allocator.get_kvcache()
    pool = getattr(kvcache, "chunk_tail_swa_pool", None)
    if pool is None:
        return
    policy = kvcache.swa_policy
    swa_win = pool.swa_win
    # This branch has no ``Req.fill_ids`` attribute (that is master-v4's name):
    # the token array lives in ``full_untruncated_fill_ids`` and
    # ``get_fill_ids()`` slices it at ``extend_range.end``, i.e. exactly the
    # prefix that this forward has just filled. Using the untruncated array here
    # would count tokens of later chunks as committed and stage a window whose
    # KV has not been written yet.
    committed_len = len(req.get_fill_ids())
    req_to_token = self.req_to_token_pool.req_to_token
    draft_pool = self.draft_worker.primary_draft_kv_pool if self.draft_worker else None
    draft_tail = getattr(draft_pool, "chunk_tail_swa_pool", None)

    while req.as_swa_copy_count < len(req.as_swa_kv_slots):
        checkpoint_no = req.as_swa_copy_count
        checkpoint_pos = policy.checkpoint_pos(checkpoint_no)
        if checkpoint_pos > committed_len:
            # Not committed yet; a later chunk will pick it up.
            break
        slot_id = req.as_swa_kv_slots[checkpoint_no]
        if slot_id == -1:
            req.as_swa_copy_count += 1
            continue
        # The last swa_win tokens before checkpoint_pos live in the UPPER half
        # (half=1) of their SWA page -- guaranteed by the
        # swa_step % swa_page_size == 0 assert in init_as_swa_pool. Resolve the
        # physical page through req_to_token -> full_to_swa rather than
        # tail_start_pos // swa_page_size, which would be wrong whenever this
        # request's SWA pages do not start at physical page 0.
        tail_start_pos = checkpoint_pos - swa_win
        full_slot = int(req_to_token[req.kv.req_pool_idx, tail_start_pos].item())
        swa_slot = int(kvcache.full_to_swa_index_mapping[full_slot].item())
        if swa_slot == 0:
            # Slot 0 is the reserved SWA padding sink and is never handed out
            # (see allocator/swa.py:25-27), so reading it back means
            # clear_full_to_swa_mapping already dropped this token's mapping:
            # the window has been evicted. Copying anyway would silently stage
            # the padding page's garbage and AS would serve it as valid KV.
            # Degrade to the -1 sentinel and recycle the staging slot.
            logger.warning(
                "chunk_tail_swa: rid=%s checkpoint#%d pos=%d tail_start=%d "
                "full_slot=%d has no live full->swa mapping (window already "
                "evicted); skipping this checkpoint",
                req.rid,
                checkpoint_no,
                checkpoint_pos,
                tail_start_pos,
                full_slot,
            )
            req.as_swa_kv_slots[checkpoint_no] = -1
            free_chunk_tail_swa_slots(kvcache, [slot_id])
            req.as_swa_copy_count += 1
            continue
        swa_page_idx = swa_slot // kvcache.swa_page_size
        pool.copy_swa_page_half(slot_id, kvcache.swa_kv_pool, swa_page_idx, half=1)
        if draft_tail is not None:
            # Same slot_id and same swa_page_idx: the draft shares the target's
            # slot-id space and the very same full_to_swa_index_mapping object
            # (asserted in _assert_draft_swa_geometry).
            draft_tail.copy_swa_page_half(
                slot_id, draft_pool.swa_kv_pool, swa_page_idx, half=1
            )
        req.as_swa_copy_count += 1


def _retire_ascache_writes(self) -> None:
    """Retire completed AttentionStore async writes in submission order.

    Every scheduler loop that can produce AttentionStore writes must call this
    once per round: retirement is the only place a write's retained tree lock is
    given back, so skipping it pins the written prefix (FULL path-to-root plus
    one SWA window) for the process lifetime.
    """
    if self.enable_asradix_cache and hasattr(self.tree_cache, "check_ascache_events"):
        self.tree_cache.check_ascache_events()


_SCHEDULER_METHODS = {
    "_init_attention_store": _init_attention_store,
    "_as_draft_swa_layers": _as_draft_swa_layers,
    "_assert_draft_swa_geometry": _assert_draft_swa_geometry,
    "detect_and_may_copy_as_swa": _detect_and_may_copy_as_swa,
    "retire_ascache_writes": _retire_ascache_writes,
}


@plugin_hook(f"{_SCHEDULER}.__init__", type=HookType.AROUND)
def scheduler_init_kunlun(original_fn, self, server_args, *args, **kwargs):
    """Attach the AttentionStore methods and bring the binding up.

    The methods go on unconditionally -- ``retire_ascache_writes`` is called
    from ``ingest_requests`` on every run and no-ops when the cache is off --
    while the binding is only built when it is on.
    """
    global _SCHEDULER_REF

    original_fn(self, server_args, *args, **kwargs)

    cls = type(self)
    for name, fn in _SCHEDULER_METHODS.items():
        if not hasattr(cls, name):
            setattr(cls, name, fn)

    self.enable_asradix_cache = asradix_enabled()
    self.ascli = None
    _SCHEDULER_REF = self

    if self.enable_asradix_cache:
        self._init_attention_store(server_args)


@plugin_hook(f"{_SCHEDULER}.ingest_requests", type=HookType.AROUND)
def ingest_requests_kunlun(original_fn, self, *args, **kwargs):
    """Retire completed AS writes once per event-loop round.

    Upstream puts the call immediately before each ``self.ingest_requests()``;
    hooking the callee covers every loop with one hook. Retirement must run
    before the round's work, not after, so the slots it frees are available to
    this round's allocations.
    """
    if getattr(self, "enable_asradix_cache", False):
        self.retire_ascache_writes()
    return original_fn(self, *args, **kwargs)


@plugin_hook(f"{_SCHEDULER}.flush_cache", type=HookType.AROUND)
def flush_cache_kunlun(original_fn, self, *args, **kwargs):
    """Drain in-flight AS async writes before the tree cache is reset.

    Upstream drains inside the ``is_fully_idle()`` branch right before
    ``tree_cache.reset()``; the same guard here keeps a busy flush from
    blocking on writes that reset will not touch.
    """
    if (
        getattr(self, "enable_asradix_cache", False)
        and hasattr(self.tree_cache, "drain_attention_store_writes")
        and self.is_fully_idle()
    ):
        self.tree_cache.drain_attention_store_writes()
    return original_fn(self, *args, **kwargs)


@plugin_hook(
    "sglang.srt.managers.schedule_batch.Req.__init__", type=HookType.AROUND
)
def req_init_kunlun(original_fn, self, *args, **kwargs):
    """Add the per-request AttentionStore state, and the block hashes.

    The fields are set unconditionally and cheaply: the AS coordinators and the
    free paths probe them on every request, and ``Req`` has no ``__slots__`` so
    there is nothing to declare. Only the hash computation is gated, because it
    walks the whole input.
    """
    original_fn(self, *args, **kwargs)

    # Populated by the AS coordinators at runtime (read latch, WAR dedup,
    # replay). The caller-supplied hint fields have no wire path into this
    # engine (see module docstring) and stay empty.
    self.as_block_hashs: Optional[Any] = None
    self.as_kv_cache_params: Optional[dict] = None
    self.as_max_hit_host_ip: List[str] = []
    self.as_max_hit_token_len: List[int] = []
    self.ever_read_as: bool = False
    # Chunk-tail SWA staging slot ids for the full lifecycle of this request.
    # One slot per swa_step checkpoint crossed; populated by
    # _alloc_chunk_tail_slots_for_batch. A value of -1 is the "no SWA data for
    # this checkpoint" sentinel (AS-hit prefix or alloc failure) and must be
    # skipped by copy / write / free.
    self.as_swa_kv_slots: List[int] = []
    # SwaStepPolicy incremental-capture water-mark: how many SWA checkpoints of
    # this request have already been copied out for async AS write. Also the
    # current write index into as_swa_kv_slots.
    self.as_swa_copy_count: int = 0
    # Number of swa_step checkpoints already submitted to an AS write task.
    # Trails as_swa_copy_count. Slots in [0, as_written_count) are owned by the
    # AS write task and freed asynchronously on retirement; slots at or after it
    # are freed synchronously in release_kv_cache.
    self.as_written_count: int = 0
    # Fail-fast flag: set True only when a genuine chunk_tail_swa_pool alloc has
    # failed (pool full). Distinct from the -1 AS-hit prefix placeholders, which
    # are NOT alloc failures. Once set, later extends skip alloc.
    self.as_swa_alloc_failed: bool = False
    self._as_cache_read_abort_reason: Optional[str] = None

    if not asradix_enabled():
        return

    try:
        from attentionstore_sdk.sdk import Tokens

        from sglang_kunlun.mem_cache.asradix.types import AS_BLOCK_SIZE

        self.as_block_hashs = Tokens(list(self.origin_input_ids), AS_BLOCK_SIZE)
    except ImportError:
        logger.warning("[AS] attentionstore_sdk unavailable; skip as_block_hashs")
    except Exception as exc:
        logger.warning("[AS] failed to build as_block_hashs: %s", exc)
