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
"""Shared AttentionStore SDK transport adapters."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import threading
import time
from numbers import Integral
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from .types import AttentionStoreInitContext

logger = logging.getLogger(__name__)
# Cache-content diagnostic. Separate logger so it can be silenced or routed to
# its own file without touching the normal AS logs.
cache_digest_logger = logging.getLogger("asradix.cache_digest")
# Runtime override for the digest log, set through /set_internal_state. None
# means "no override", so the env default applies. Each PP rank keeps its own
# copy: the control request walks the whole PP chain, so every rank gets it.
_cache_digest_override: Optional[bool] = None


def set_cache_digest_enabled(enabled: Optional[bool]) -> None:
    """Turn the AttentionStore cache-content digest log on or off at runtime.

    The digest costs a D2H copy plus a hash per page, so it stays off by default
    and is only switched on while a cache-content problem is being chased.
    Passing ``None`` drops the override and restores the env default.
    """
    global _cache_digest_override
    _cache_digest_override = None if enabled is None else bool(enabled)
    cache_digest_logger.info(
        "AS cache digest log %s",
        "off (env default)" if enabled is None else ("on" if enabled else "off"),
    )


def cache_digest_enabled() -> bool:
    """Report whether the AttentionStore cache-content digest log is on."""
    if _cache_digest_override is not None:
        return _cache_digest_override

    from sglang.srt.environ import envs

    return envs.SGLANG_LOG_AS_CACHE_DIGEST.get()


# Runtime override for the producer-event barrier diagnostic. Same on/off
# split as the cache digest: it logs whether an event fenced anything and,
# on read, whether the page content matches what the SDK gathered.
_barrier_debug_override: Optional[bool] = None


def set_barrier_debug_enabled(enabled: Optional[bool]) -> None:
    """Turn the producer-event barrier diagnostic on or off at runtime."""
    global _barrier_debug_override
    _barrier_debug_override = None if enabled is None else bool(enabled)
    cache_digest_logger.info(
        "AS barrier debug log %s",
        "off (env default)" if enabled is None else ("on" if enabled else "off"),
    )


def barrier_debug_enabled() -> bool:
    """Report whether the producer-event barrier diagnostic is on."""
    if _barrier_debug_override is not None:
        return _barrier_debug_override

    from sglang.srt.environ import envs

    return envs.SGLANG_LOG_AS_BARRIER_DEBUG.get()


def page_digest(tensor, page_id: int, rows_per_page: int = 1) -> str:
    """MD5 the buffer rows one page occupies inside a layer buffer.

    A token-major KV buffer stores ``rows_per_page`` token rows per page, so the
    digest has to cover the whole row range the transfer moves, not just row
    ``page_id``. Bytes are taken through a uint8 view because fp8 and bf16 have
    no numpy dtype, and the digest is truncated to 12 hex chars: the log is only
    ever compared against another line from the same run.
    """
    rows_per_page = max(1, int(rows_per_page))
    start = int(page_id) * rows_per_page
    rows = tensor[start : start + rows_per_page]
    host = rows.detach().to("cpu", copy=True).contiguous()
    return hashlib.md5(host.view(torch.uint8).numpy()).hexdigest()[:12]


def wait_for_producer_event(
    event,
    timeout: Optional[float] = 1.0,
    operation: str = "AS page write",
    debug_tag: str = "",
) -> None:
    """Fence the HBM pages a transfer reads against the stream that fills them.

    Writes are executed on the coordinator's writer thread, whose current stream
    is not the one the forward pass runs on, so neither the digest snapshot nor
    the SDK gather is ordered against the producer. Without this wait either can
    read a page that is still being written.
    """
    if event is None:
        if barrier_debug_enabled():
            cache_digest_logger.info(
                "AS barrier %s: producer_ready=None (no event to wait on)",
                debug_tag,
            )
        return
    query = getattr(event, "query", None)
    if query is None:
        raise RuntimeError(f"{operation} event does not support bounded polling")
    already_done = query()
    started = time.monotonic() if barrier_debug_enabled() else 0.0
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    polls = 0
    while not query():
        polls += 1
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"{operation} producer wait timed out")
        time.sleep(0.001)
    if barrier_debug_enabled():
        # already_done=True at the top of this call means the event fenced
        # nothing: whatever kernels were still running when it was recorded
        # had already finished by the time we got here, or it was recorded
        # against a stream with nothing relevant queued on it (e.g. an
        # alt_stream side branch the default-stream event never covers).
        cache_digest_logger.info(
            "AS barrier %s: already_done=%s waited_polls=%d wait_ms=%.3f",
            debug_tag,
            already_done,
            polls,
            (time.monotonic() - started) * 1000,
        )


def _once(release: Callable[[], None]) -> Callable[[], None]:
    """Make a resource-release callback idempotent and thread-safe."""
    lock = threading.Lock()
    released = False

    def release_once() -> None:
        nonlocal released
        with lock:
            if released:
                return
            released = True
        release()

    return release_once


def _sdk_pod_config(
    ctx: AttentionStoreInitContext,
    namespace: str,
    *,
    endpoint_name: Optional[str] = None,
    model_version: Optional[str] = None,
    gpu_count: Optional[int] = None,
    pod_name: Optional[str] = None,
):
    from attentionstore_sdk.sdk import PodConfig
    from attentionstore_sdk.types import (
        PipelineCacheEndpointRole,
        PipelineCacheGroupDescriptor,
        PipelineCacheMember,
    )

    endpoint_registration = (
        None if endpoint_name is None else ctx.endpoint_registrations.get(endpoint_name)
    )
    pipeline_cache_group = None
    desired_block_count = 0
    pipeline_cache_member_id = 0
    if endpoint_registration is not None:
        desired_block_count = endpoint_registration.desired_block_count
        pipeline_cache_member_id = endpoint_registration.local_member_id
        pipeline_cache_group = PipelineCacheGroupDescriptor(
            group_id=endpoint_registration.group_id,
            generation=endpoint_registration.generation,
            endpoint_name=endpoint_registration.endpoint_name,
            endpoint_role=PipelineCacheEndpointRole[
                endpoint_registration.endpoint_role.name
            ],
            coordinator_member_id=endpoint_registration.coordinator_member_id,
            members=tuple(
                PipelineCacheMember(
                    member_id=member.member_id,
                    pp_stage_start=member.pp_stage_start,
                    pp_stage_end=member.pp_stage_end,
                    agent_address=member.agent_address,
                    bytes_per_block=member.bytes_per_block,
                    desired_block_count=member.desired_block_count,
                )
                for member in endpoint_registration.members
            ),
        )

    return PodConfig(
        namespace=namespace,
        pod_name=pod_name or ctx.pod_name,
        model_version=model_version or ctx.model_version,
        device_id=ctx.device_id,
        dp_id=ctx.dp_rank,
        gpu_count=ctx.gpu_count if gpu_count is None else gpu_count,
        desired_block_count=desired_block_count,
        pipeline_cache_group=pipeline_cache_group,
        pipeline_cache_member_id=pipeline_cache_member_id,
    )


def _standard_namespace(
    ctx: AttentionStoreInitContext,
    endpoint_name: str,
    shard_num: int,
) -> str:
    role = f"pp{ctx.node_pp_start}-{ctx.node_pp_end - 1}-of{ctx.pp_size}"
    return f"{ctx.namespace_base}:{shard_num}:{role}:{endpoint_name}"


class _PassiveStore:
    """Neutral endpoint used by PP stages with no local layers of one kind."""

    def __init__(self, layout=None):
        self.layout = layout

    @staticmethod
    def _count(block_hash_ids) -> int:
        return 1 if isinstance(block_hash_ids, Integral) else len(block_hash_ids)

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Treat every requested block as matched."""
        return self._count(block_hash_ids)

    def write(self, block_hash_ids: Sequence[int], *args, **kwargs) -> int:
        """Accept every passive write without storing data."""
        return self._count(block_hash_ids)

    def read(self, block_hash_ids: Sequence[int], *args, **kwargs) -> int:
        """Accept every passive read without loading data."""
        return self._count(block_hash_ids)

    def stop(self) -> None:
        """Stop the passive endpoint without performing cleanup."""
        return


class _BufferPageIO:
    """Transport page requests over SDK-owned typed buffer bindings."""

    def __init__(
        self,
        sdk,
        buffers,
        layer_ids: Sequence[int],
        *,
        buffer_owners: Sequence[Any] = (),
        buffer_tensors: Optional[Mapping[str, Sequence[Any]]] = None,
        buffer_pools: Optional[Mapping[str, Sequence[Any]]] = None,
        local_page_size: Optional[int] = None,
        block_size: Optional[int] = None,
        write_buffers=None,
    ):
        self._sdk = sdk
        self._buffers = buffers
        # A segment whose write source is not its read target binds different
        # pointers per direction (SPAG's SWA segment stages out of
        # ChunkTailSWAPool and restores into the live SWA pool). The SDK keeps two
        # independent buffer registrations, so one store can serve both.
        self._write_buffers = buffers if write_buffers is None else write_buffers
        self._layer_ids = tuple(map(int, layer_ids))
        self._buffer_owners = tuple(buffer_owners)
        self._buffer_tensors = {
            name: tuple(tensors) for name, tensors in (buffer_tensors or {}).items()
        }
        self._buffer_pools = {
            name: tuple(pools) for name, pools in (buffer_pools or {}).items()
        }
        self._local_page_size = local_page_size
        self._block_size = block_size
        if (local_page_size is None) != (block_size is None):
            raise ValueError("local_page_size and block_size must be set together")
        if block_size is not None:
            if local_page_size <= 0 or block_size % local_page_size:
                raise ValueError(
                    f"block_size={block_size} must be divisible by "
                    f"local_page_size={local_page_size}"
                )
            if set(self._buffer_tensors) != set(self._buffer_pools):
                raise ValueError("buffer tensor/pool families must match")
            for name, tensors in self._buffer_tensors.items():
                if len(tensors) != len(self._buffer_pools[name]):
                    raise ValueError(f"{name} buffer tensor/pool counts differ")

    @property
    def _pages_per_block(self) -> int:
        return (
            1
            if self._block_size is None
            else self._block_size // self._local_page_size
        )

    def _logical_blocks(self, block_hash_ids, page_ids):
        """Group the physical pages backing each logical AttentionStore block."""
        page_ids = tuple(map(int, page_ids))
        pages_per_block = self._pages_per_block
        if pages_per_block == 1:
            if len(block_hash_ids) != len(page_ids):
                raise ValueError("hash and page counts differ")
            return tuple(
                (int(hash_id), (page_id,))
                for hash_id, page_id in zip(block_hash_ids, page_ids)
            )
        expected = len(block_hash_ids) * pages_per_block
        if len(page_ids) != expected:
            raise ValueError(
                f"{len(block_hash_ids)} blocks of size {self._block_size} require "
                f"{expected} local pages, got {len(page_ids)}"
            )
        result = []
        for index, hash_id in enumerate(block_hash_ids):
            pages = page_ids[index * pages_per_block : (index + 1) * pages_per_block]
            result.append((int(hash_id), pages))
        return tuple(result)

    def _read_prefix(self, block_hash_ids, page_ids, read: int):
        """Narrow a read request to the blocks the Agent actually filled.

        A partial read leaves the tail buffers holding whatever was there
        before, so digesting them would report a mismatch that says nothing
        about the blocks that did come back.
        """
        count = max(0, min(int(read), len(block_hash_ids)))
        pages = count * self._pages_per_block
        return block_hash_ids[:count], page_ids[:pages]

    def _rows_per_page(self, name: str, position: int, tensor) -> int:
        """Return how many buffer rows one local page occupies.

        Mirrors the layout split in ``_uniform_page_bytes``: a token-major
        buffer (first dim covers every token slot in the pool) holds one row per
        token, while a page-major buffer already holds one row per local page.
        """
        page_size = self._local_page_size or 1
        pools = self._buffer_pools.get(name, ())
        pool = pools[position] if position < len(pools) else None
        pool_size = getattr(pool, "size", None)
        if pool_size is not None and int(tensor.shape[0]) >= int(pool_size):
            return int(page_size)
        pool_page_size = int(getattr(pool, "page_size", page_size) or page_size)
        return max(1, int(page_size) // pool_page_size)

    def _snapshot_digests(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
    ) -> Optional[list]:
        """Hash the HBM pages a transfer touches, grouped per logical block.

        Only the first and last local layer are hashed: hashing every layer costs
        a D2H copy per layer per page, and a corruption that spares both ends of
        the layer range is not a case worth paying for by default.
        """
        if not cache_digest_enabled() or not block_hash_ids:
            return None
        try:
            blocks = self._logical_blocks(block_hash_ids, page_ids)
            snapshot = []
            for name, tensors in self._buffer_tensors.items():
                if not tensors:
                    continue
                positions = [0] if len(tensors) == 1 else [0, len(tensors) - 1]
                for position in positions:
                    tensor = tensors[position]
                    rows_per_page = self._rows_per_page(name, position, tensor)
                    snapshot.append(
                        (
                            name,
                            position,
                            (
                                self._layer_ids[position]
                                if position < len(self._layer_ids)
                                else -1
                            ),
                            [
                                (
                                    hash_id,
                                    [
                                        page_digest(tensor, page, rows_per_page)
                                        for page in pages
                                    ],
                                )
                                for hash_id, pages in blocks
                            ],
                        )
                    )
            return snapshot
        except Exception as exc:  # diagnostic only, never break the transfer
            cache_digest_logger.warning("AS digest snapshot failed: %s", exc)
            return None

    def _debug_recheck_after_sync(
        self, block_hash_ids: Sequence[int], page_ids: Sequence[int], tag: str
    ) -> None:
        """After a full device sync, re-hash the same pages the write snapshot
        saw. If the barrier fenced everything, these digests must equal the
        write-time snapshot. Any page that was all-zero at write time but
        non-zero here proves the producer had NOT finished when the barrier
        released -- i.e. the event missed that page's real producer stream.
        Guarded by the barrier-debug switch because it forces a hard sync.
        """
        if not barrier_debug_enabled() or not block_hash_ids:
            return
        try:
            torch.cuda.synchronize()
            blocks = self._logical_blocks(block_hash_ids, page_ids)
            for name, tensors in self._buffer_tensors.items():
                if not tensors:
                    continue
                positions = [0] if len(tensors) == 1 else [0, len(tensors) - 1]
                for position in positions:
                    tensor = tensors[position]
                    rows_per_page = self._rows_per_page(name, position, tensor)
                    layer_id = (
                        self._layer_ids[position]
                        if position < len(self._layer_ids)
                        else -1
                    )
                    post = [
                        (
                            hash_id,
                            [
                                page_digest(tensor, page, rows_per_page)
                                for page in pages
                            ],
                        )
                        for hash_id, pages in blocks
                    ]
                    cache_digest_logger.info(
                        "AS post-sync recheck %s: family=%s layer_id=%s digests=%s",
                        tag,
                        name,
                        layer_id,
                        post,
                    )
        except Exception as exc:
            cache_digest_logger.warning("AS post-sync recheck failed: %s", exc)

    @staticmethod
    def _emit_digests(action: str, snapshot: Optional[list], transferred: int) -> None:
        """Print a previously captured digest snapshot once the transfer is done."""
        if not snapshot:
            return
        for name, position, layer_id, digests in snapshot:
            cache_digest_logger.info(
                "AS %s digest: family=%s layer_pos=%d layer_id=%s "
                "blocks=%d transferred=%s digests=%s",
                action,
                name,
                position,
                layer_id,
                len(digests),
                transferred,
                digests,
            )

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Return the matched page prefix length."""
        from attentionstore_sdk.sdk import MatchRequest

        return self._sdk.match(MatchRequest(block_hash_ids, timeout=timeout))

    @staticmethod
    def _bind_swa_ids(buffers, swa_ids):
        """Attach this call's SWA slot/page ids to the frozen buffer bindings.

        The ids are per-request (which staged window backs which checkpoint
        block), unlike the pointers, which are fixed at init.
        """
        if swa_ids is None:
            return buffers
        return dataclasses.replace(buffers, swa_ids=tuple(map(int, swa_ids)))

    def _swa_digests(self, block_hash_ids, swa_ids, swa_block_step, is_write):
        """Hook: hash the SWA region a transfer touches. Off in the base class."""
        return None

    def write(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
        timeout: float = 1.0,
        params: Optional[dict] = None,
        producer_ready=None,
        swa_ids: Optional[Sequence[int]] = None,
        swa_block_step: Optional[int] = None,
        logical_start: Optional[int] = None,
    ) -> int:
        """Write page blocks through the typed SDK buffers."""
        from attentionstore_sdk.sdk import BlockBatch, WriteRequest

        # Both the digest snapshot and the SDK gather read these pages, so the
        # fence has to come before either of them.
        wait_for_producer_event(
            producer_ready, timeout, debug_tag="write hash=%s" % (block_hash_ids[:1],)
        )
        buffers = self._bind_swa_ids(self._write_buffers, swa_ids)
        # The SDK derives each block's checkpoint position from its index within
        # this batch, so it needs to know where the batch starts.
        block_kwargs = (
            {"logical_start": logical_start} if logical_start is not None else {}
        )
        if self._pages_per_block == 1:
            self._logical_blocks(block_hash_ids, page_ids)
            # Snapshot before the SDK call: the D2H gather happens inside write(),
            # so this is the source content actually handed to the Agent.
            snapshot = self._snapshot_digests(block_hash_ids, page_ids)
            swa_snapshot = self._swa_digests(
                block_hash_ids, swa_ids, swa_block_step, True
            )
            written = self._sdk.write(
                WriteRequest(
                    blocks=BlockBatch(
                        hash_ids=block_hash_ids,
                        data_ids=page_ids,
                        layer_ids=self._layer_ids,
                        **block_kwargs,
                    ),
                    buffers=buffers,
                    timeout=timeout,
                    params=params or {},
                    h2h_copy=False,
                    swa_block_step=swa_block_step,
                )
            )
            self._emit_digests("write", snapshot, written)
            self._emit_digests("write-swa", swa_snapshot, written)
            self._debug_recheck_after_sync(
                block_hash_ids, page_ids, "write hash=%s" % (block_hash_ids[:1],)
            )
            return written
        blocks = self._logical_blocks(block_hash_ids, page_ids)
        snapshot = self._snapshot_digests(block_hash_ids, page_ids)
        written = self._sdk.write(
            WriteRequest(
                blocks=BlockBatch(
                    hash_ids=tuple(block[0] for block in blocks),
                    page_ids=tuple(block[1] for block in blocks),
                    layer_ids=self._layer_ids,
                    **block_kwargs,
                ),
                buffers=buffers,
                timeout=timeout,
                params=params or {},
                h2h_copy=False,
                swa_block_step=swa_block_step,
            )
        )
        self._emit_digests("write", snapshot, written)
        self._debug_recheck_after_sync(
            block_hash_ids, page_ids, "write hash=%s" % (block_hash_ids[:1],)
        )
        return written

    def read(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
        timeout: float = 1.0,
        remote_addrs=None,
        logical_start: Optional[int] = None,
        swa_ids: Optional[Sequence[int]] = None,
        swa_block_step: Optional[int] = None,
    ) -> int:
        """Read page blocks through the typed SDK buffers."""
        from attentionstore_sdk.sdk import BlockBatch, ReadRequest

        buffers = self._bind_swa_ids(self._buffers, swa_ids)
        if self._pages_per_block == 1:
            self._logical_blocks(block_hash_ids, page_ids)
            block_kwargs = (
                {"logical_start": logical_start} if logical_start is not None else {}
            )
            read = self._sdk.read(
                ReadRequest(
                    blocks=BlockBatch(
                        hash_ids=block_hash_ids,
                        data_ids=page_ids,
                        layer_ids=self._layer_ids,
                        **block_kwargs,
                    ),
                    buffers=buffers,
                    timeout=timeout,
                    remote_addrs=remote_addrs or (),
                    swa_block_step=swa_block_step,
                )
            )
            self._emit_digests(
                "read",
                self._snapshot_digests(
                    *self._read_prefix(block_hash_ids, page_ids, read)
                ),
                read,
            )
            if read >= len(block_hash_ids):
                # The SWA window only lands on the final requested block, so a
                # short read leaves it untouched and hashing it says nothing.
                self._emit_digests(
                    "read-swa",
                    self._swa_digests(
                        block_hash_ids, swa_ids, swa_block_step, False
                    ),
                    read,
                )
            rp_hashes, rp_pages = self._read_prefix(block_hash_ids, page_ids, read)
            self._debug_recheck_after_sync(
                rp_hashes, rp_pages, "read hash=%s" % (rp_hashes[:1],)
            )
            return read
        blocks = self._logical_blocks(block_hash_ids, page_ids)
        block_kwargs = (
            {"logical_start": logical_start} if logical_start is not None else {}
        )
        read = self._sdk.read(
            ReadRequest(
                blocks=BlockBatch(
                    hash_ids=tuple(block[0] for block in blocks),
                    page_ids=tuple(block[1] for block in blocks),
                    layer_ids=self._layer_ids,
                    **block_kwargs,
                ),
                buffers=buffers,
                timeout=timeout,
                remote_addrs=remote_addrs or (),
                swa_block_step=swa_block_step,
            )
        )
        self._emit_digests(
            "read",
            self._snapshot_digests(*self._read_prefix(block_hash_ids, page_ids, read)),
            read,
        )
        rp_hashes, rp_pages = self._read_prefix(block_hash_ids, page_ids, read)
        self._debug_recheck_after_sync(
            rp_hashes, rp_pages, "read hash=%s" % (rp_hashes[:1],)
        )
        return read

    def stop(self) -> None:
        """Stop the underlying page SDK client."""
        self._sdk.stop()


@dataclasses.dataclass(frozen=True)
class _SpagSwaDigest:
    """Tensor views for MD5-ing the SPAG SWA region on both sides of the wire.

    ``write_tensors`` are the chunk-tail staging buffers, one row per slot id, a
    row being exactly one staged window. ``read_tensors`` are the live SWA pool
    buffers, one row per SWA page id, and the window is restored into the row's
    upper half. Hashing the same ``window_bytes`` on both sides is what lets a
    write digest be compared against the read digest of the same block hash.
    """

    write_tensors: tuple
    read_tensors: tuple
    window_bytes: int


class _SpagDigestPageIO(_BufferPageIO):
    """SPAG page store that can MD5 the HBM bytes each transfer touches.

    The generic ``_BufferPageIO`` digest path is token/page-major aware but is
    driven by ``buffer_tensors``/``buffer_pools`` metadata, which SPAG does not
    supply (its data plane is a list of raw segment pointers). The DeepSeek-V4.1
    unified pool also starts each compressed-page region at a byte OFFSET past
    the SWA ring and pages at ``page_size // ratio`` rows, so the generic
    per-pool ``rows_per_page`` cannot address the right bytes.

    This subclass instead hashes exactly the bytes the SDK moves, using the
    ``(name, tensor, offset_bytes, item_bytes)`` records the pool exposes (same
    source of truth as ``get_contiguous_buf_infos``): for FULL page ``page_id`` a
    segment contributes ``tensor[offset_rows + page_id*rows :][:rows]`` where the
    row counts are derived from the byte sizes. Every segment is hashed and
    labelled by ``name`` (``content*``/``index*``) so page content and indexer
    payloads are reported separately.

    The SWA segment is hashed separately (``_swa_digests``): it is addressed by
    slot / SWA page id rather than by FULL page id, and only a subset of blocks
    carry it, so it does not fit the per-block page records above.
    """

    def __init__(
        self, sdk, buffers, layer_ids, *, digest_segments=(), swa_digest=None, **kwargs
    ):
        super().__init__(sdk, buffers, layer_ids, **kwargs)
        # Each entry: (name, tensor, offset_bytes, item_bytes).
        self._digest_segments = tuple(digest_segments)
        self._swa_digest = swa_digest

    def _snapshot_digests(self, block_hash_ids, page_ids):
        if not cache_digest_enabled() or not block_hash_ids:
            return None
        segments = self._digest_segments
        if not segments:
            return None
        try:
            pages = tuple(int(page_id) for page_id in page_ids)
            if len(pages) != len(block_hash_ids):
                # SPAG blocks are one FULL page each, so the two lists match.
                raise ValueError(
                    f"hash/page count mismatch: {len(block_hash_ids)} vs {len(pages)}"
                )
            positions = range(len(segments))
            snapshot = []
            for position in positions:
                name, tensor, offset_bytes, item_bytes = segments[position]
                row_bytes = int(tensor[0].nbytes)
                if row_bytes <= 0 or item_bytes % row_bytes or offset_bytes % row_bytes:
                    raise ValueError(
                        f"{name} byte sizes not row-aligned: row={row_bytes} "
                        f"item={item_bytes} offset={offset_bytes}"
                    )
                offset_rows = offset_bytes // row_bytes
                rows = item_bytes // row_bytes
                digests = []
                for hash_id, page_id in zip(block_hash_ids, pages):
                    start = offset_rows + page_id * rows
                    block_rows = tensor[start : start + rows]
                    host = block_rows.detach().to("cpu", copy=True).contiguous()
                    md5 = hashlib.md5(host.view(torch.uint8).numpy()).hexdigest()[:12]
                    digests.append((hash_id, [md5]))
                snapshot.append((name, position, -1, digests))
            return snapshot
        except Exception as exc:  # diagnostic only, never break the transfer
            cache_digest_logger.warning("AS SPAG digest snapshot failed: %s", exc)
            return None

    def _swa_digests(self, block_hash_ids, swa_ids, swa_block_step, is_write):
        """Hash the SWA window each checkpoint block carries.

        Keyed on the block hash of the checkpoint block so a write digest lines
        up with the read digest of the same block. Only the first and last layer
        are hashed, matching the page path's cost/benefit tradeoff.
        """
        digest = self._swa_digest
        if digest is None or not cache_digest_enabled() or not swa_ids:
            return None
        try:
            ids = tuple(int(swa_id) for swa_id in swa_ids)
            if is_write:
                step = int(swa_block_step or 0)
                if step <= 0:
                    raise ValueError(f"invalid swa_block_step={swa_block_step}")
                # Checkpoints close every ``step``-th block of the batch.
                pairs = [
                    (int(block_hash_ids[index * step + step - 1]), swa_id)
                    for index, swa_id in enumerate(ids)
                ]
            else:
                # A read restores one window, into the final requested block.
                pairs = [(int(block_hash_ids[-1]), ids[-1])]
            tensors = digest.write_tensors if is_write else digest.read_tensors
            window = int(digest.window_bytes)
            positions = [0] if len(tensors) == 1 else [0, len(tensors) - 1]
            snapshot = []
            for position in positions:
                flat = tensors[position].view(torch.uint8).reshape(-1)
                row_bytes = int(tensors[position][0].nbytes)
                # Writes stage a bare window per row; reads target the upper half
                # of a full SWA page, which is where the window was staged from.
                lead = 0 if is_write else window
                digests = []
                for hash_id, swa_id in pairs:
                    if swa_id < 0:
                        # No staged window for this checkpoint (AS-hit prefix or a
                        # staging-pool shortage); the block's SWA region is empty.
                        continue
                    start = swa_id * row_bytes + lead
                    host = flat[start : start + window].to("cpu", copy=True)
                    digests.append(
                        (hash_id, [hashlib.md5(host.numpy()).hexdigest()[:12]])
                    )
                snapshot.append(("swa", position, -1, digests))
            return snapshot
        except Exception as exc:  # diagnostic only, never break the transfer
            cache_digest_logger.warning("AS SPAG SWA digest snapshot failed: %s", exc)
            return None


class CheckpointedPageStore:
    """Page stream plus an independently stored endpoint checkpoint."""

    def __init__(
        self, page_store, checkpoint_store, *, page_match_is_joint: bool = False
    ):
        """Combine a page store with its terminal checkpoint store."""
        self.page_store = page_store
        self.checkpoint_store = checkpoint_store
        self.page_match_is_joint = page_match_is_joint
        self.layout = checkpoint_store.layout

    def match(self, block_hash_ids: Sequence[int], timeout: float = 1.0) -> int:
        """Return the longest page prefix with a terminal checkpoint."""
        page_match_started = time.perf_counter()
        page_count = self.page_store.match(block_hash_ids, timeout=timeout)
        page_match_ms = (time.perf_counter() - page_match_started) * 1000

        # A group-bound page endpoint asks the Agent group for the common terminal
        # checkpoint and pins both endpoint payloads. This also applies to a
        # singleton non-PP group. Re-querying the checkpoint endpoint would create
        # a second transaction for the same logical Match and double lease pressure.
        if self.page_match_is_joint:
            return page_count

        checkpoint_match_started = time.perf_counter()
        checkpoint_probes = 0
        matched = 0
        for count in range(page_count, 0, -1):
            checkpoint_probes += 1
            if (
                self.checkpoint_store.match(
                    [block_hash_ids[count - 1]], timeout=timeout
                )
                == 1
            ):
                matched = count
                break
        checkpoint_match_ms = (time.perf_counter() - checkpoint_match_started) * 1000
        logger.info(
            "AS checkpoint match: requested_blocks=%s page_matched_blocks=%s "
            "matched_blocks=%s checkpoint_probes=%s page_match_ms=%.2f "
            "checkpoint_match_ms=%.2f",
            len(block_hash_ids),
            page_count,
            matched,
            checkpoint_probes,
            page_match_ms,
            checkpoint_match_ms,
        )
        return matched

    def read(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
        checkpoint_slot: Mapping[str, int] | int,
        timeout: float = 1.0,
        remote_addrs=None,
    ) -> int:
        """Read pages and their terminal checkpoint as one unit."""
        if not block_hash_ids:
            return 0
        page_count = self.page_store.read(
            block_hash_ids,
            page_ids,
            timeout=timeout,
            remote_addrs=remote_addrs,
        )
        if page_count != len(block_hash_ids):
            return page_count
        state_count = self.checkpoint_store.read(
            int(block_hash_ids[-1]),
            checkpoint_slot,
            timeout=timeout,
            remote_addrs=remote_addrs,
        )
        return page_count if state_count == 1 else 0

    def write(
        self,
        block_hash_ids: Sequence[int],
        page_ids: Sequence[int],
        checkpoint_slot: Mapping[str, int] | int,
        timeout: float = 1.0,
        params: Optional[dict] = None,
        producer_ready=None,
    ) -> int:
        """Write pages and their terminal checkpoint as one unit."""
        if not block_hash_ids:
            return 0
        wait_for_producer = getattr(self.checkpoint_store, "wait_for_producer", None)
        if wait_for_producer is not None:
            wait_for_producer(producer_ready, timeout=timeout)
        self.page_store.write(block_hash_ids, page_ids, timeout=timeout, params=params)
        checkpoint_result = self.checkpoint_store.write(
            int(block_hash_ids[-1]),
            checkpoint_slot,
            timeout=timeout,
            params=params,
            producer_ready=producer_ready,
        )
        return len(block_hash_ids) if checkpoint_result == 1 else -1

    def stop(self) -> None:
        """Stop both the page and checkpoint stores."""
        try:
            self.page_store.stop()
        finally:
            self.checkpoint_store.stop()
