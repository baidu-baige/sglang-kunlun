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
"""Shared asynchronous lifecycle for AttentionStore coordinators."""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import struct
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)

from .transport import _once, barrier_debug_enabled, cache_digest_logger
from .types import AttentionStoreWriteContext, AttentionStoreWriteOwner

logger = logging.getLogger(__name__)


def _noop_release() -> None:
    return


def _acquire_node_release(cache, node) -> Callable[[], None]:
    lock_result = cache.inc_lock_ref(node)

    def release() -> None:
        cache.dec_lock_ref(node, lock_result.to_dec_params())

    return _once(release)


def _contiguous_page_ids(slot_ids: torch.Tensor, page_size: int) -> tuple[int, ...]:
    """Validate page-aligned contiguous slots with one device-to-host sync."""
    return _materialize_page_ids(_page_id_metadata(slot_ids, page_size), page_size)


def _page_id_metadata(slot_ids: torch.Tensor, page_size: int) -> torch.Tensor:
    """Build page bases and validity flags on the source device."""
    if page_size <= 0 or slot_ids.numel() % page_size:
        raise ValueError(
            f"slot count {slot_ids.numel()} is not aligned to page_size={page_size}"
        )
    if not slot_ids.numel():
        return torch.empty((0, 2), dtype=torch.int64, device=slot_ids.device)

    pages = slot_ids.reshape(-1, page_size)
    page_bases = pages[:, 0]
    offsets = torch.arange(page_size, dtype=pages.dtype, device=pages.device)
    valid = torch.all(pages == page_bases[:, None] + offsets, dim=1)
    valid &= page_bases.remainder(page_size) == 0
    return torch.stack((page_bases, valid.to(page_bases.dtype)), dim=1)


def _materialize_page_ids(page_meta: torch.Tensor, page_size: int) -> tuple[int, ...]:
    """Validate host page metadata and return physical page identifiers."""
    page_meta = page_meta.tolist()
    for page_index, (_, is_valid) in enumerate(page_meta):
        if not is_valid:
            raise ValueError(
                f"non-contiguous page at prefix {(page_index + 1) * page_size}"
            )
    return tuple(int(page_base) // page_size for page_base, _ in page_meta)


_COPY_EVENT_TIMEOUT_S = 1.0
_quarantined_copies: list = []
_quarantine_lock = threading.Lock()


def _event_completed(event) -> bool:
    query = getattr(event, "query", None)
    if query is None:
        return False
    try:
        return bool(query())
    except Exception:
        return False


def _wait_copy_event(event, timeout: float, operation: str) -> None:
    query = getattr(event, "query", None)
    if query is None:
        raise RuntimeError(f"{operation} event does not support bounded polling")
    deadline = time.monotonic() + max(0.0, timeout)
    while not query():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{operation} copy completion wait timed out")
        time.sleep(0.001)


def _quarantine_copy(prepared, operation: str) -> None:
    with _quarantine_lock:
        if prepared.quarantined:
            return
        prepared.quarantined = True
        _quarantined_copies.append(prepared)
    logger.warning("%s copy is incomplete; retaining its buffers", operation)


def _reap_quarantined_copies() -> None:
    with _quarantine_lock:
        _quarantined_copies[:] = [
            entry
            for entry in _quarantined_copies
            if not _event_completed(entry.ready_event)
        ]


@dataclass
class _PreparedPageIds:
    page_size: int
    metadata: torch.Tensor
    ready_event: Any = None
    source_ready_event: Any = None
    source_metadata: Optional[torch.Tensor] = None
    quarantined: bool = False

    def materialize(self) -> tuple[int, ...]:
        """Wait for the pending copy and return the resolved page ids."""
        if self.ready_event is not None:
            try:
                _wait_copy_event(self.ready_event, _COPY_EVENT_TIMEOUT_S, "AS page-id")
            except Exception:
                _quarantine_copy(self, "AS page-id")
                raise
        return _materialize_page_ids(self.metadata, self.page_size)

    def release(self) -> None:
        """Retain buffers while the page-ID copy is incomplete."""
        if self.ready_event is not None and not _event_completed(self.ready_event):
            _quarantine_copy(self, "AS page-id")


@dataclass
class _PreparedSlot:

    host: Any = None
    ready_event: Any = None
    source: Optional[torch.Tensor] = None
    value: Optional[int] = None
    quarantined: bool = False

    def materialize(self) -> int:
        """Wait for the pending copy and return the resolved slot value."""
        if self.value is None:
            if self.ready_event is not None:
                try:
                    _wait_copy_event(
                        self.ready_event, _COPY_EVENT_TIMEOUT_S, "AS checkpoint-slot"
                    )
                except Exception:
                    _quarantine_copy(self, "AS checkpoint-slot")
                    raise
            self.value = int(self.host.item())
        return self.value

    def release(self) -> None:
        """Retain buffers while the slot copy is incomplete."""
        if self.ready_event is not None and not _event_completed(self.ready_event):
            _quarantine_copy(self, "AS checkpoint-slot")


def _request_block_hash_ids(
    req, token_ids: Sequence[int], page_size: int, page_count: int
) -> tuple[int, ...]:
    """Reuse request block hashes when they cover the written prefix."""
    # Padding preserves length, so a length check alone would let the scheduler's
    # multimodal substitution (which rebinds origin_input_ids) hash away the image.
    if (
        req.as_block_hashs is not None
        and req.origin_input_ids is req.origin_input_ids_unpadded
    ):
        request_hashes = req.as_block_hashs.hash_ids
        if len(request_hashes) >= page_count:
            return tuple(map(int, request_hashes[:page_count]))

    from attentionstore_sdk.sdk import Tokens

    aligned_tokens = page_count * page_size
    return tuple(map(int, Tokens(token_ids[:aligned_tokens], page_size).hash_ids))


@dataclass
class _PendingWrite:
    future: concurrent.futures.Future
    task: Any
    logical_id: bytes
    owner: Optional[AttentionStoreWriteOwner]


class AttentionStoreCoordinator:
    """Own local-commit hooks and the complete asynchronous write lifecycle.

    Subclasses capture stable request inputs and implement type-specific I/O.
    The base class owns submission ordering and the two resource horizons:
    staging is released when the writer returns, while cache-node retention is
    released only after the task crosses the ordered maintenance frontier.
    """

    write_thread_name = "as_writer"
    pp_retirement_wait_timeout_s = 30.0

    def __init__(self) -> None:
        """Initialize an idle asynchronous write coordinator."""
        self._write_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pending_writes: list[_PendingWrite] = []
        self._page_id_copy_stream = None
        self._closed = False
        # Synchronous cache commit hooks use this only for dummy tasks and for
        # writes whose concrete task is created below the cache integration
        # layer. It is never read by the writer thread.
        self._owner_hint: Optional[AttentionStoreWriteOwner] = None
        # Injected by Scheduler._init_attention_store: the model's forward_stream,
        # i.e. the stream the MLA/KDA KV writes actually run on. The hybrid pool's
        # allocator does not carry forward_stream, so the coordinator holds it
        # directly for the producer fence.
        self.forward_stream: Any = None

    def _producer_stream(self, cache):
        """Stream the forward pass used to fill the KV pages.

        The model forward (and the KV writes) run on ``model_runner.forward_stream``;
        the page/checkpoint writers read those pages on their own stream, so the
        completion fence has to target that forward stream. Prefer the explicitly
        injected ``forward_stream`` (set by the scheduler) and fall back to the
        allocator's copy if present.
        """
        if self.forward_stream is not None:
            return self.forward_stream
        allocator = getattr(cache, "token_to_kv_pool_allocator", None)
        return getattr(allocator, "forward_stream", None)

    @staticmethod
    def record_device_event(device: Any, stream: Any = None):
        """Record completion of the producer stream that filled the pages.

        ``stream`` should be the forward stream that produced the KV data (see
        ``_producer_stream``); when omitted we fall back to the current stream.
        """
        device = torch.device(device)
        if device.type not in ("cuda", "xpu"):
            return None
        device_module = torch.get_device_module(device)
        # This may be called from the writer thread (do_write), whose current
        # device can differ from the KV pool's; pin it so the event and the
        # producer stream are created on the same device.
        with device_module.device(device):
            if stream is None:
                stream = device_module.current_stream(device)
            event = device_module.Event()
            event.record(stream)
        if barrier_debug_enabled():
            # If query() is already true right here, this event was recorded
            # against a stream that had nothing in flight -- it fences nothing.
            cache_digest_logger.info(
                "AS record_device_event: stream=%s already_done=%s",
                stream.cuda_stream if hasattr(stream, "cuda_stream") else stream,
                event.query(),
            )
        return event

    def prepare_page_ids(self, cache, slot_ids: torch.Tensor) -> _PreparedPageIds:
        """Stage validated page metadata without blocking the scheduler."""
        _reap_quarantined_copies()
        metadata = _page_id_metadata(slot_ids, cache.page_size)
        if metadata.device.type != "cuda" or not metadata.numel():
            return _PreparedPageIds(cache.page_size, metadata.cpu())

        if self._page_id_copy_stream is None:
            self._page_id_copy_stream = torch.cuda.Stream(device=metadata.device)
        source_ready = torch.cuda.Event()
        source_ready.record(torch.cuda.current_stream(metadata.device))
        host_metadata = torch.empty(
            metadata.shape,
            dtype=metadata.dtype,
            device="cpu",
            pin_memory=True,
        )
        with torch.cuda.stream(self._page_id_copy_stream):
            self._page_id_copy_stream.wait_event(source_ready)
            host_metadata.copy_(metadata, non_blocking=True)
            copy_ready = torch.cuda.Event()
            copy_ready.record(self._page_id_copy_stream)
        return _PreparedPageIds(
            cache.page_size,
            host_metadata,
            ready_event=copy_ready,
            source_ready_event=source_ready,
            source_metadata=metadata,
        )

    def prepare_checkpoint_slot(self, slot: Any) -> _PreparedSlot:
        """Stage a recurrent-state slot index without blocking the scheduler."""
        _reap_quarantined_copies()
        if not isinstance(slot, torch.Tensor):
            return _PreparedSlot(value=int(slot))
        if slot.device.type != "cuda" or slot.numel() != 1:
            return _PreparedSlot(value=int(slot.item()))
        if self._page_id_copy_stream is None:
            self._page_id_copy_stream = torch.cuda.Stream(device=slot.device)
        source_ready = torch.cuda.Event()
        source_ready.record(torch.cuda.current_stream(slot.device))
        host_slot = torch.empty(
            slot.shape, dtype=slot.dtype, device="cpu", pin_memory=True
        )
        with torch.cuda.stream(self._page_id_copy_stream):
            self._page_id_copy_stream.wait_event(source_ready)
            host_slot.copy_(slot, non_blocking=True)
            copy_ready = torch.cuda.Event()
            copy_ready.record(self._page_id_copy_stream)
        # `source` keeps the device tensor alive until the copy is consumed.
        return _PreparedSlot(host=host_slot, ready_event=copy_ready, source=slot)

    def capture_finished(
        self, _cache, _req, _is_insert: bool
    ) -> Optional[AttentionStoreWriteContext]:
        """Capture state before committing a finished request."""
        return None

    def capture_unfinished(self, _cache, _req) -> Optional[AttentionStoreWriteContext]:
        """Capture state before committing an unfinished request."""
        return None

    def before_local_commit(self, _cache, _context: AttentionStoreWriteContext) -> None:
        """Prepare an AttentionStore write before the local commit."""
        return

    def after_local_commit(self, _cache, _context: AttentionStoreWriteContext) -> None:
        """Publish a prepared write after the local commit succeeds."""
        return

    def abort_local_commit(self, cache, context: AttentionStoreWriteContext) -> None:
        """Release a prepared write after the local commit fails."""
        task = context.prepared_write
        context.prepared_write = None
        if task is not None:
            self._discard_unsubmitted(cache, task)

    def extend_match(
        self,
        cache,
        params: MatchPrefixParams,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        """Extend a local component-consistent match with an L2 hit."""
        raise NotImplementedError

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Restore an L2 hit into the components declared by the cache plan."""
        raise NotImplementedError

    def do_write(self, cache, task) -> int:
        """Execute one payload-specific AttentionStore write."""
        raise NotImplementedError

    def finish(self, _cache, _task, _result: int) -> None:
        """Apply payload-specific state after a write retires."""
        return

    def _ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._closed:
            raise RuntimeError("AttentionStore coordinator is already closed")
        if self._write_executor is None:
            self._write_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=self.write_thread_name,
            )
        return self._write_executor

    def _run_write(self, cache, task) -> int:
        try:
            return self.do_write(cache, task)
        finally:
            task.release_after_write()

    def _release_task_resource(
        self,
        cache,
        task,
        release: Callable[[], None],
        resource: str,
    ) -> Optional[Exception]:
        try:
            release()
            return None
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=task.rid,
                requested=task.requested_blocks,
                error=RuntimeError(f"write {resource} release failed: {exc}"),
            )
            return exc

    def _discard_unsubmitted(self, cache, task) -> None:
        self._release_task_resource(
            cache, task, task.release_after_write, "staging resource"
        )
        self._release_task_resource(cache, task, task.release, "retained resource")

    def _resolve_write_owner(self, cache, task) -> Optional[AttentionStoreWriteOwner]:
        """Resolve the scheduler owner attached to an asynchronous write.

        The dynamic cache context wins over request metadata: a request can be
        reused by a later microbatch while an earlier async result is being
        processed. Task metadata is the fallback for writes created outside the
        scheduler result context.
        """
        owner = getattr(cache, "_as_active_write_owner", None)
        if owner is None:
            owner = getattr(task, "owner", None)
        if owner is None:
            request = getattr(task, "request", None)
            owner = getattr(request, "_as_write_owner", None)
        if owner is None:
            owner = self._owner_hint
        if owner is not None and not isinstance(owner, AttentionStoreWriteOwner):
            raise RuntimeError(f"invalid AttentionStore write owner: {owner!r}")
        if owner is not None and owner.rid is None:
            owner = replace(owner, rid=str(getattr(task, "rid", "")))
        return owner

    @staticmethod
    def _set_task_owner(task, owner: Optional[AttentionStoreWriteOwner]) -> None:
        if owner is not None:
            task.owner = owner

    def submit_write(self, cache, task) -> None:
        """Submit a write task to the ordered background executor."""
        owner = self._resolve_write_owner(cache, task) or self._owner_hint
        if getattr(cache, "pp_world_size", getattr(cache, "pp_size", 1)) > 1:
            if owner is None:
                raise RuntimeError(
                    "AttentionStore PP write has no cache maintenance owner: "
                    f"rid={getattr(task, 'rid', '<unknown>')}"
                )
            self._set_task_owner(task, owner)
        try:
            logical_id = self._task_logical_id(task)
            future = self._ensure_executor().submit(self._run_write, cache, task)
        except Exception:
            self._discard_unsubmitted(cache, task)
            raise
        self._pending_writes.append(
            _PendingWrite(
                future=future,
                task=task,
                logical_id=logical_id,
                owner=owner,
            )
        )

    def submit_completed(self, task, result: int = 0, owner=None) -> None:
        """Queue an already completed task for ordered retirement."""
        if self._closed:
            raise RuntimeError("AttentionStore coordinator is already closed")
        owner = owner or getattr(task, "owner", None) or self._owner_hint
        if owner is not None and owner.rid is None:
            owner = replace(owner, rid=str(getattr(task, "rid", "")))
        if owner is not None:
            self._set_task_owner(task, owner)
        future = concurrent.futures.Future()
        future.set_result(result)
        self._pending_writes.append(
            _PendingWrite(
                future=future,
                task=task,
                logical_id=self._task_logical_id(task),
                owner=owner,
            )
        )

    def _resolve_result(
        self, cache, pending: _PendingWrite, timeout: Optional[float]
    ) -> int:
        try:
            return pending.future.result(timeout=timeout)
        except Exception as exc:
            cache._log_as_failure(
                "write",
                rid=pending.task.rid,
                requested=pending.task.requested_blocks,
                error=exc,
            )
            return -1

    def _retire(
        self,
        cache,
        pending_writes: Sequence[_PendingWrite],
        timeout: Optional[float],
        *,
        synchronize: bool = True,
    ) -> None:
        first_error = None
        for pending in pending_writes:
            result = self._resolve_result(cache, pending, timeout)
            if synchronize:
                result = cache._tp_min(result)
            try:
                self.finish(cache, pending.task, result)
            except Exception as exc:
                cache._log_as_failure(
                    "write",
                    rid=pending.task.rid,
                    requested=pending.task.requested_blocks,
                    error=RuntimeError(f"write completion failed: {exc}"),
                )
                first_error = first_error or exc
            release_error = self._release_task_resource(
                cache,
                pending.task,
                pending.task.release,
                "retained resource",
            )
            first_error = first_error or release_error
        retirement_succeeded = first_error is None
        if synchronize:
            retirement_succeeded = bool(cache._tp_min(int(retirement_succeeded)))
        if not retirement_succeeded:
            error = RuntimeError("AttentionStore write retirement failed")
            if first_error is not None:
                raise error from first_error
            raise error

    @staticmethod
    def _task_logical_id(task) -> bytes:
        """Return a compact, rank-stable identity for one pending write.

        The queue is compared across attention shards, so this identity must
        contain logical write inputs but not rank-local resources such as page
        IDs, node addresses, or checkpoint pool slots.  In particular, the
        request ID alone is insufficient for chunked writes: two writes for
        one request can have different prefixes.

        A digest keeps the pending-queue fingerprint bounded even for long
        prefixes.  ``write_key`` is currently used by KDA and adds the cache
        length/terminal-hash identity that is not otherwise guaranteed by the
        task type and request ID.
        """
        task_kind = f"{type(task).__module__}.{type(task).__qualname__}".encode(
            "utf-8", "surrogatepass"
        )
        rid = str(task.rid).encode("utf-8", "surrogatepass")
        block_hash_ids = tuple(
            int(block_hash_id) for block_hash_id in task.block_hash_ids
        )

        identity = hashlib.sha256()
        identity.update(b"as-task-logical-id-v2\0")
        for field in (task_kind, rid):
            identity.update(struct.pack(">Q", len(field)))
            identity.update(field)
        identity.update(struct.pack(">Q", int(task.requested_blocks)))
        identity.update(struct.pack(">Q", len(block_hash_ids)))
        for block_hash_id in block_hash_ids:
            identity.update(struct.pack(">Q", block_hash_id))

        owner = getattr(task, "owner", None)
        if owner is None:
            identity.update(b"\x00")
        else:
            identity.update(b"\x01")
            identity.update(struct.pack(">Q", int(owner.maintenance_round)))
            for value in (owner.pp_batch_plan_id, owner.mb_id):
                identity.update(struct.pack(">q", -1 if value is None else int(value)))
            rid = (owner.rid or "").encode("utf-8", "surrogatepass")
            identity.update(struct.pack(">Q", len(rid)))
            identity.update(rid)

        # KDA's write_key is a logical (cache length, terminal hash) pair.
        # Do not include checkpoint_slot: it is a rank-local Mamba pool slot.
        write_key = getattr(task, "write_key", None)
        if write_key is None:
            identity.update(b"\x00")
        else:
            identity.update(b"\x01")
            identity.update(struct.pack(">Q", len(write_key)))
            for value in write_key:
                identity.update(struct.pack(">Q", int(value)))

        return identity.digest()

    def _pending_queue_state(self) -> tuple[int, int, int]:
        digest = hashlib.sha256()
        digest.update(struct.pack(">Q", len(self._pending_writes)))
        for pending in self._pending_writes:
            digest.update(struct.pack(">Q", len(pending.logical_id)))
            digest.update(pending.logical_id)

        raw_digest = digest.digest()
        int64_mask = (1 << 63) - 1
        return (
            len(self._pending_writes),
            int.from_bytes(raw_digest[:8], "big") & int64_mask,
            int.from_bytes(raw_digest[8:16], "big") & int64_mask,
        )

    def _synchronized_pending_state(self, cache) -> int:
        count, digest_a, digest_b = self._pending_queue_state()
        bounds = torch.tensor(
            [count, -count, digest_a, -digest_a, digest_b, -digest_b],
            dtype=torch.int64,
        )
        cache._all_reduce_attn_groups(bounds, torch.distributed.ReduceOp.MIN)
        minimum, negative_maximum = map(int, bounds[:2].tolist())
        digest_bounds = list(map(int, bounds[2:].tolist()))
        if minimum != -negative_maximum:
            raise RuntimeError(
                "AttentionStore write queue length mismatch within attention stage"
            )
        if any(
            minimum_digest != -negative_maximum_digest
            for minimum_digest, negative_maximum_digest in zip(
                digest_bounds[::2], digest_bounds[1::2]
            )
        ):
            raise RuntimeError(
                "AttentionStore write queue identity or order mismatch within "
                "attention stage"
            )
        return minimum

    def _wait_for_pending_writes(
        self, cache, pending_writes: Sequence[_PendingWrite]
    ) -> None:
        deadline = time.monotonic() + self.pp_retirement_wait_timeout_s
        timed_out_pending = None
        for pending in pending_writes:
            try:
                pending.future.result(timeout=max(0.0, deadline - time.monotonic()))
            except concurrent.futures.TimeoutError as exc:
                if pending.future.done():
                    continue
                cache._log_as_failure(
                    "write",
                    rid=pending.task.rid,
                    requested=pending.task.requested_blocks,
                    error=exc,
                )
                timed_out_pending = pending
                break
            except Exception:
                pass
        stage_settled = bool(cache._tp_min(int(timed_out_pending is None)))
        if not stage_settled:
            rid = (
                timed_out_pending.task.rid
                if timed_out_pending is not None
                else "<peer attention shard>"
            )
            raise RuntimeError(
                "AttentionStore PP write retirement timed out before resource "
                f"release: rid={rid} timeout_s={self.pp_retirement_wait_timeout_s}"
            )

    def check_events(self, cache, maintenance_round: Optional[int] = None) -> None:
        """Retire writes from cache rounds completed before this round.

        The scheduler calls this before admitting the current round.  A write
        created during round N is therefore retained until round N+1, which
        prevents the current batch from releasing a node while its proxy or
        result lifecycle is still in flight.
        """
        _reap_quarantined_copies()
        is_pp = getattr(cache, "pp_world_size", getattr(cache, "pp_size", 1)) > 1
        if is_pp:
            if maintenance_round is None:
                raise RuntimeError(
                    "missing PP cache maintenance round for AttentionStore retirement"
                )
            self._drain(
                cache,
                synchronize=True,
                bounded_wait=True,
                retire_before_round=maintenance_round,
            )
            return

        if not self._pending_writes:
            return
        completed = 0
        for pending in self._pending_writes:
            if not pending.future.done():
                break
            completed += 1
        completed = cache._tp_min(completed)
        if not completed:
            return
        pending_writes = self._pending_writes[:completed]
        del self._pending_writes[:completed]
        self._retire(cache, pending_writes, timeout=1.0)

    def _drain(
        self,
        cache,
        *,
        synchronize: bool,
        bounded_wait: bool = False,
        retire_before_round: Optional[int] = None,
    ) -> None:
        if not self._pending_writes and not synchronize:
            return
        if synchronize and not self._synchronized_pending_state(cache):
            return

        if retire_before_round is None:
            retire_count = len(self._pending_writes)
        else:
            last_round = -1
            for pending in self._pending_writes:
                owner = pending.owner
                if owner is None:
                    raise RuntimeError(
                        "AttentionStore PP pending write has no cache maintenance owner: "
                        f"rid={getattr(pending.task, 'rid', '<unknown>')}"
                    )
                if owner.maintenance_round < last_round:
                    raise RuntimeError(
                        "AttentionStore pending writes are out of maintenance order: "
                        f"previous_round={last_round} current_round={owner.maintenance_round}"
                    )
                last_round = owner.maintenance_round
            retire_count = sum(
                pending.owner.maintenance_round < retire_before_round
                for pending in self._pending_writes
            )

        if retire_count == 0:
            return
        pending_writes = tuple(self._pending_writes[:retire_count])
        del self._pending_writes[:retire_count]
        if bounded_wait:
            self._wait_for_pending_writes(cache, pending_writes)
        self._retire(
            cache,
            pending_writes,
            timeout=0.0 if bounded_wait else None,
            synchronize=synchronize,
        )

    def drain(self, cache, *, synchronize: bool = True) -> None:
        """Wait for and retire every pending write."""
        is_pp = getattr(cache, "pp_world_size", getattr(cache, "pp_size", 1)) > 1
        self._drain(
            cache,
            synchronize=synchronize,
            bounded_wait=synchronize and is_pp,
            retire_before_round=None,
        )

    def before_reset(self, cache) -> None:
        """Drain pending writes before resetting the cache."""
        is_pp = getattr(cache, "pp_world_size", getattr(cache, "pp_size", 1)) > 1
        self._drain(
            cache,
            synchronize=True,
            bounded_wait=is_pp,
        )

    def before_sanity_check(self, cache) -> None:
        """Drain pending writes before checking cache invariants."""
        is_pp = getattr(cache, "pp_world_size", getattr(cache, "pp_size", 1)) > 1
        self._drain(
            cache,
            synchronize=True,
            bounded_wait=is_pp,
        )

    def shutdown(self, cache) -> None:
        """Drain writes and stop the background executor."""
        try:
            # atexit is not ordered across PP processes. Agent only publishes a
            # block after every shard is complete, so local retirement is safe.
            self._drain(cache, synchronize=False)
        finally:
            self._closed = True
            if self._write_executor is not None:
                self._write_executor.shutdown(wait=True)
                self._write_executor = None


class CheckpointedPageCacheCoordinator(AttentionStoreCoordinator):
    """Provide the shared remote-match contract for hybrid component plans."""

    def extend_match(
        self,
        cache,
        params: MatchPrefixParams,
        local_result: MatchResult,
    ) -> Optional[MatchResult]:
        """Extend a local component-consistent match with remote pages/state."""
        return cache._match_checkpointed_page_prefix(params, local_result)

    def floor_remote_terminal(self, cache, terminal_block: int) -> int:
        """Clamp a remote page match to a prefix this plan can fully restore.

        A plan whose recurrent state exists only at certain page boundaries must
        shrink the match to the last such boundary, otherwise the load-back would
        restore pages it cannot pair with any state.
        """
        return terminal_block

    def init_load_back(
        self,
        cache,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """Restore one hybrid checkpointed-page prefix into UnifiedRadixCache."""
        raise NotImplementedError
