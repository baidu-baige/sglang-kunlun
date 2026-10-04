"""HiSparse server-argument adjustments for Kunlun.

``validate_hisparse`` documents that its dtype/backend checks "only apply to
the DSA hisparse path" because "DSv4 hisparse handles its own dtype/backend
pairing elsewhere" -- but the guard that skips them is
``is_hip and is_v4_hisparse``, so on Kunlun (``is_hip`` is False, the device
presents as CUDA) DeepSeek-V4 is still forced onto bfloat16/fp8_e4m3 and an
fp16 launch is rejected.

Kunlun serves DeepSeek-V4 with fp16 (see
``hooks/layers/attention/dsv4/dtype_restore.py``), the DSv4 C4 half-cache is a
2-byte layout for either dtype, and ``--kv-cache-dtype bfloat16`` collapses
DSPARK acceptance on this build, so honour the requested dtype for DeepSeek-V4.

The cold pool is also moved onto the XPU device here.  The swap-in has to run
inside the captured decode graph, and a CUDA graph only records device work:
``copy_items`` gathers with host index tensors, which executes at capture time
on uninitialised values.  A device-resident cold pool makes the whole swap-in
(planning, LRU, copy) plain device indexing, so it captures.  The cost is that
the cold pool -- which is the *entire* compressed KV, 21 x 18313 x 65536B =
25.2GB -- now competes with the device KV budget, so ``--mem-fraction-static``
must be lowered to make room.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

# The coordinator's own cold pool (``hisparse_coordinator.py``), distinct from
# the HiCache pools the hybrid pool assembler builds.
_HISPARSE_COLD_POOL = "dsv4_hisparse_c4"


@plugin_hook(
    "sglang.srt.arg_groups.hisparse_hook.validate_hisparse_kv_cache_dtype",
    type=HookType.AROUND,
)
def _allow_requested_kv_dtype_for_dsv4_hisparse(original_fn, server_args):
    """Skip the DSA-only KV-dtype check for DeepSeek-V4 hisparse."""
    from sglang.srt.arg_groups.overrides import model_config_of
    from sglang.srt.configs.model_config import is_deepseek_v4

    if is_deepseek_v4(model_config_of(server_args).hf_config):
        return None
    return original_fn(server_args)


@plugin_hook(
    "sglang.srt.managers.hisparse_coordinator.HiSparseCoordinator.__init__",
    type=HookType.AROUND,
)
def _grow_hisparse_row_buffers(original_fn, self, *args, **kwargs):
    """Size the row-indexed swap-in buffers for speculative verify batches.

    The coordinator sizes ``top_k_device_locs_buffer`` / ``raw_indices_buffer``
    (and the shared-index miss plan) by the request-pool slot count, assuming one
    row per request.  A DSPARK verify forward runs ``bs * num_draft_tokens``
    query rows, each selecting its own top-k, so the row-indexed buffers must
    cover that many rows.
    """
    original_fn(self, *args, **kwargs)

    import torch

    from sglang.srt.runtime_context import get_spec

    draft = int(get_spec().speculative_num_draft_tokens or 1)
    slots = self.req_to_token_pool.req_to_token.shape[0]
    rows = slots * max(1, draft)
    if getattr(self, "is_dsv4_hisparse", False) and draft > 1:
        # TARGET_VERIFY now uses a row-owned stable-slot path.  A shared-index
        # plan is request-row based and cannot safely replay this multi-row
        # miss set, so keep correctness on the synchronous path.
        self.enable_prefetch = False
    if rows <= self.top_k_device_locs_buffer.shape[0]:
        return

    device = self.device
    top_k = self.top_k
    self.top_k_device_locs_buffer = torch.full(
        (rows, top_k), -1, dtype=torch.int32, device=device
    )
    self.raw_indices_buffer = torch.full(
        (rows, top_k), -1, dtype=torch.int32, device=device
    )
    if getattr(self, "_miss_src", None) is not None:
        self._miss_src = torch.zeros((rows, top_k), dtype=torch.int64, device=device)
        self._miss_dst = torch.zeros((rows, top_k), dtype=torch.int32, device=device)
        self._miss_count = torch.zeros((rows,), dtype=torch.int32, device=device)


def move_cold_pool_to_device(coordinator) -> None:
    """Re-home the hisparse cold pool from pinned host memory onto the XPU.

    ``DeepSeekV4PagedHostPool`` builds its buffers through
    ``ALLOC_MEMORY_FUNCS`` -> ``HostTensorAllocator.allocate``, which asserts
    ``device == "cpu"``, so the pool cannot simply be constructed with
    ``device="cuda"``.  Allocate the equivalent device tensors here instead and
    drop the host ones; the pool starts empty, so nothing has to be copied.

    Called for the target worker only (``model_runner_kv_cache_mixin``): the
    DSPARK draft also builds a coordinator, but its swap-in never runs, and a
    second device-resident cold pool does not fit alongside the target's.
    """
    import torch

    if coordinator is None:
        return
    pool = getattr(coordinator, "mem_pool_host", None)
    buffers = getattr(pool, "kv_buffer", None)
    if not buffers:
        return
    gpu = pool.gpu_device
    if str(pool.device) == str(gpu):
        return

    shapes = [b.shape for b in buffers]
    dtypes = [b.dtype for b in buffers]
    total_gb = sum(s.numel() * d.itemsize for s, d in zip(shapes, dtypes)) / 1e9
    try:
        free_gb = torch.cuda.mem_get_info(gpu)[0] / 1e9
    except Exception:  # noqa: BLE001 - informational only
        free_gb = float("nan")
    logger.warning(
        "sglang-kunlun: hisparse cold pool -> device (%s, %.2f GB; %.2f GB free). "
        "The swap-in runs in the decode CUDA graph, so the cold pool cannot stay "
        "in host memory; lower --mem-fraction-static if this does not fit.",
        gpu,
        total_gb,
        free_gb,
    )

    # Replace the host buffers with device ones.  ``data_refs`` is the layer
    # order the transfer hooks walk, and ``data_ptrs`` is read by the PD
    # transfer path, so both have to be rebuilt from the new tensors.
    pool.kv_buffer = [
        torch.zeros(shape, dtype=dtype, device=gpu)
        for shape, dtype in zip(shapes, dtypes)
    ]
    pool.data_refs = list(pool.kv_buffer)
    pool.device = gpu
    pool.pin_memory = False
    pool.data_ptrs = torch.tensor(
        [b.data_ptr() for b in pool.kv_buffer], dtype=torch.uint64, device=gpu
    )


@plugin_hook(
    "sglang.srt.mem_cache.memory_pool_host.DeepSeekV4PagedHostPool.__init__",
    type=HookType.AROUND,
)
def _skip_host_register_for_hisparse_cold_pool(original_fn, self, *args, **kwargs):
    """Allocate the hisparse cold pool without ``cudaHostRegister``.

    The cold pool is host-allocated only so ``DeepSeekV4PagedHostPool.__init__``
    runs unchanged; the target's copy is re-homed onto the XPU right after
    (``move_cold_pool_to_device``) and the draft's is never used.  The
    registration is wasted work and, at 16GB, fails on XPU with a vendor error
    the cudart shim cannot even stringify (``cudaGetErrorString(): ... Invoked
    with: 712``).  Turning ``pin_memory`` off routes the pool to the plain
    mmap allocator with no registration.
    """
    pool_name = kwargs.get("pool_name", args[0] if args else None)
    if pool_name == _HISPARSE_COLD_POOL:
        kwargs["pin_memory"] = False
    return original_fn(self, *args, **kwargs)


@plugin_hook(
    "sglang.srt.managers.hisparse_coordinator.HiSparseCoordinator.admit_request_direct",
    type=HookType.AROUND,
)
def _admit_request_direct_fake_safe(original_fn, self, req):
    """Make the PD-warmup fake-bootstrap request admit safely under HiSparse.

    The boot-time PD disaggregation warmup posts one ``FAKE_BOOTSTRAP_HOST``
    request per DP rank.  ``admit_request_direct`` assumes a real RDMA'd host
    pool entry; the fake request has none, so ``_preload_to_device_buffer``
    reads uninitialised ``req_to_host_pool`` indices.  Under DP attention this
    desyncs the ranks and the warmup all_gather dead-locks (observed
    2026-10-03: ranks split between ``all_gather`` and ``recv_requests``,
    ``num_running_reqs == 0``, zero warmup POSTs completing).

    For a fake request we only allocate the device buffer and leave its tokens
    marked resident, so the swap-in reads on-device (zero) KV, the forward
    emits a token, and the DP warmup all_gather completes.  Real requests
    (``bootstrap_host != FAKE_BOOTSTRAP_HOST``) take the original path
    untouched, so eval-time accuracy is unaffected.
    """
    try:
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
    except Exception:
        return original_fn(self, req)

    if getattr(req, "bootstrap_host", None) == FAKE_BOOTSTRAP_HOST:
        self.alloc_device_buffer(req)
        req.hisparse_staging = False
        try:
            self._skip_first_backup[req.kv.req_pool_idx] = True
        except Exception:
            pass
        logger.warning(
            "sglang-kunlun: hisparse admit fast-path for fake-bootstrap warmup "
            "req %s (on-device zeros, no host preload).",
            getattr(req, "rid", "?"),
        )
        return None
    return original_fn(self, req)


# ---------------------------------------------------------------------------
# HiSparse decode-capacity gate: use the logical pool, not the device buffer pool
#
# ``ScheduleBatch.check_decode_mem`` asks the allocator
# ``check_decode_capacity(num_tokens)`` whether the next decode step fits, and on
# a False it runs ``retract_decode`` -> ``release_req`` -> (PD decode) cpu_tensor
# ``offload_kv_cache`` -> ``allocator.get_cpu_copy`` which is ``NotImplementedError``
# for the HiSparse allocator -> scheduler crash (observed 2026-10-03 on the first
# gsm8k batch).
#
# The base check is ``available_size() >= num_tokens``.  For
# ``DeepSeekV4HiSparseTokenToKVPoolAllocator`` ``available_size()`` is
# ``min(logical_attn_allocator.available_size(),
#      hisparse_attn_allocator.available_size() * compress_ratio)``.  The second
# term is the C4 *device-buffer* pool, which is an admission-time cache sized for
# concurrency: once budget-many requests are admitted it is saturated by design,
# so the min collapses to a tiny value and the gate fails *spuriously* -- even
# though the next decode token is allocated only from the logical pool
# (``alloc_decode`` -> ``logical_attn_allocator.alloc_decode``; the device buffer
# is a fixed per-request ring reused via swap-in, never grown per step).
#
# Gate HiSparse decode on the logical pool only (same units as the base compare,
# we just drop the device-buffer term from the min).  Non-HiSparse allocators are
# untouched.  This removes the spurious retraction; genuine logical-pool pressure
# still retracts as before.
# ---------------------------------------------------------------------------
@plugin_hook(
    "sglang.srt.mem_cache.allocator.base.BaseTokenToKVPoolAllocator.check_decode_capacity",
    type=HookType.AROUND,
)
def _hisparse_decode_capacity_logical_only(original_fn, self, *args, **kwargs):
    try:
        from sglang.srt.mem_cache.allocator.hisparse import (
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        )
    except Exception:
        return original_fn(self, *args, **kwargs)

    if not isinstance(self, DeepSeekV4HiSparseTokenToKVPoolAllocator):
        return original_fn(self, *args, **kwargs)

    num_tokens = kwargs.get("num_tokens")
    if num_tokens is None and args:
        num_tokens = args[0]
    tree_cache = kwargs.get("tree_cache")
    if tree_cache is None and len(args) > 1:
        tree_cache = args[1]

    try:
        self.evict_to_free_tokens(tree_cache, num_tokens)
    except Exception:
        pass
    return self.logical_attn_allocator.available_size() >= num_tokens


# ---------------------------------------------------------------------------
# HiSparse C4 compressed-KV DEVICE-POOL LEAK fix (root cause of the repeated
# "Prefill out of memory, available 0" crashes).
#
# ``DeepSeekV4HiSparseTokenToKVPoolAllocator.free_full()`` frees the logical
# pool but NOT the C4 compressed-KV device slots (``hisparse_attn_allocator``).
# So every completed request leaks its C4 device slots: the decode-batch
# ``#gpu token`` usage climbs monotonically across a run (observed 0.16 -> 1.00
# over ~400 ifeval requests at #running-req=1, #prealloc-req=0) until the C4
# device pool is full, after which ``alloc_extend`` returns None and
# ``alloc_paged_token_slots_extend`` fail-louds -> scheduler exit -3. Short
# prompts (gsm8k/ifeval) finish many requests quickly and hit the wall fast;
# long prompts (longbench) finish few and usually complete before leaking out.
#
# The sibling non-SWA allocator's ``free()`` already releases BOTH pools
# (``logical_attn_allocator.free`` + ``self.free_hisparse``). The SWA-hybrid
# allocator's ``free_full`` path simply omitted the ``free_hisparse`` call.
# Mirror it: after the logical full-free actually happens (immediate path),
# release the mapped C4 device slots. Deferred frees are re-driven through
# ``free_full`` by ``free_group_end`` with ``free_group=None``, so they land
# here too and are released on flush.
# ---------------------------------------------------------------------------
@plugin_hook(
    "sglang.srt.mem_cache.allocator.hisparse.DeepSeekV4HiSparseTokenToKVPoolAllocator.free_full",
    type=HookType.AROUND,
)
def _hisparse_free_full_release_c4(original_fn, self, free_indices, *args, **kwargs):
    original_fn(self, free_indices, *args, **kwargs)
    try:
        if free_indices is None or free_indices.numel() == 0:
            return
        # Only release C4 device slots once the logical free has actually been
        # applied (immediate path); the deferred path re-enters here on flush.
        if getattr(self, "free_group", None) is None:
            self.free_hisparse(free_indices)
    except Exception:
        # Never let the leak-fix break the free path.
        pass


# Same C4 leak, the ``free()`` path: completed requests mostly release through
# ``free()`` (not ``free_full``), and it too drops only the logical slots. Mirror
# the sibling non-SWA allocator (whose single ``free`` releases both pools) by
# also releasing the mapped C4 device slots. ``free_hisparse`` is idempotent
# (clears the full->hisparse mapping and filters indices<=0), so SWA-only indices
# with no C4 mapping are a safe no-op.
@plugin_hook(
    "sglang.srt.mem_cache.allocator.hisparse.DeepSeekV4HiSparseTokenToKVPoolAllocator.free",
    type=HookType.AROUND,
)
def _hisparse_free_release_c4(original_fn, self, free_index, *args, **kwargs):
    original_fn(self, free_index, *args, **kwargs)
    try:
        if free_index is None or free_index.numel() == 0:
            return
        if getattr(self, "free_group", None) is None:
            self.free_hisparse(free_index)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DSV4 HiSparse C4 device-pool leak: release the request's orphaned C4 pages at
# completion, BEFORE the coordinator clears the full->hisparse mapping.
#
# The per-step spec-decode allocation (``alloc_for_spec_decode`` -> hisparse
# ``alloc_extend``) maps a fresh C4 device page for each draft token.  The DSV4
# decode remap (``map_last_loc_to_buffer``) only re-points the *newest* entry at
# the request's fixed device-buffer slot; the other draft pages keep their
# freshly allocated mapping.  On completion the coordinator clears the whole
# request's mapping (``full_to_hisparse_device_index_mapping[compressed_locs]=0``)
# *before* ``release_kv_cache`` reaches ``allocator.free_full``/``free`` -- so by
# the time the (leak-fix) free hooks run, the mapping is already zero and
# ``free_hisparse`` is a no-op.  Those C4 pages are therefore never returned:
# measured 2026-10-04 the C4 device pool drains monotonically (~20 tokens per
# short ifeval request), exhausting the full/32-sized C4 pool after a few hundred
# requests and fail-louding the scheduler in ``alloc_for_spec_decode``.
#
# Free them here instead, while the mapping is still live, excluding the device
# buffer slots (freed separately by the coordinator's own ``request_finished``,
# so re-freeing them would double-free).  A request's slots are dead at this
# point, so -- unlike the mid-flight remap -- this is safe.
#
# Ordering matters: the coordinator's ``request_finished`` first waits for the
# decode producer stream and any pending backup (its own lines ~891-893) before
# it releases anything, because the previous overlapped batch may still be
# reading this request's KV.  So *capture* the orphan set up front (the mapping
# is still populated) but *free* it only after the original returns, i.e. after
# that synchronisation.  Freeing before it (as an earlier revision did) races
# the in-flight forward and corrupts generation.
# ---------------------------------------------------------------------------
@plugin_hook(
    "sglang.srt.managers.hisparse_coordinator.HiSparseCoordinator.request_finished",
    type=HookType.AROUND,
)
def _hisparse_free_orphan_c4_on_finish(original_fn, self, req, *args, **kwargs):
    orphan = None
    try:
        if getattr(self, "is_dsv4_hisparse", False):
            import torch

            mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
            allocated_len = int(req.kv.kv_allocated_len)
            if allocated_len > 0:
                locs = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :allocated_len
                ]
                compressed_locs = (
                    self.mem_pool_device.translate_loc_from_full_to_compressed(locs)
                )
                mapped = mapping[compressed_locs]
                keep = mapped == 0
                cap = int(self.req_device_buffer_size[req.kv.req_pool_idx])
                if cap > 0:
                    devbuf = self.req_to_device_buffer[req.kv.req_pool_idx, :cap]
                    keep = keep | torch.isin(mapped, devbuf)
                orphan = torch.unique(mapped[~keep])
    except Exception:
        orphan = None

    out = original_fn(self, req, *args, **kwargs)

    try:
        if orphan is not None and orphan.numel() > 0:
            self.token_to_kv_pool_allocator.free_hisparse_indices(orphan)
    except Exception:
        # Never let the leak-fix break request teardown.
        pass
    return out


# ---------------------------------------------------------------------------
# HiSparse PD-decode admission: bound by the DEVICE logical pool, not the
# host-backed capacity.
#
# ``DecodePreallocQueue._check_if_req_exceed_kv_capacity`` aborts a bootstrapped
# request only when it is larger than the admission capacity.  For HiSparse that
# capacity is ``model_runner.max_token_pool_size`` -- the HOST-backed logical
# capacity (host_to_device_ratio x the device pool), which is far larger than the
# device logical pool actually used to grow KV each decode step
# (``max_total_num_tokens`` = 131584 at mem-fraction 0.66).  So an over-long
# longbench prompt (idx 89 ~590k, idx 117 ~220k tokens -- the prefill side runs at
# the model's native 1048576 context and transfers the full KV) passes the host
# check, is admitted, then overflows the device pool mid-decode inside
# ``alloc_for_spec_decode`` and fail-louds the scheduler (observed 2026-10-04:
# "Prefill out of memory ... allocate 256 tokens, available 0" -> scheduler exit -3
# -> SIGTERM the server).  Such a prompt can NEVER fit the device pool with
# generation headroom, so admission looping on it would also stall the whole eval.
#
# Bound HiSparse admission by the device pool minus the generation reserve
# (SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION, the same headroom the prealloc gate
# reserves) and abort the oversized request fast with BAD_REQUEST -- the eval
# harness runs with --ignore-errors, so those few cases are recorded and skipped
# instead of crashing the server.  Everything that CAN fit (<= device pool - reserve,
# e.g. idx 160 ~102k) is left to the normal admission path untouched.
# ---------------------------------------------------------------------------
@plugin_hook(
    "sglang.srt.disaggregation.decode.DecodePreallocQueue._check_if_req_exceed_kv_capacity",
    type=HookType.AROUND,
)
def _hisparse_prealloc_capacity_device_bound(original_fn, self, req, *args, **kwargs):
    import logging

    logger = logging.getLogger("sglang_kunlun.hisparse")
    try:
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None or not getattr(scheduler, "enable_hisparse", False):
            return original_fn(self, req, *args, **kwargs)

        device_capacity = int(self.max_total_num_tokens)
        # Reserve the full longbench generation (16384) + spec-draft slack, NOT the
        # default CLIP (4096): a single admitted request grows by its generation, so
        # input + full-gen must fit the device pool or it over-subscribes mid-decode.
        reserve = 16896

        input_len = int(self._rebootstrap_prefill_len(req))
        if input_len + reserve > device_capacity:
            from http import HTTPStatus

            from sglang.srt.disaggregation.decode import prepare_abort

            message = (
                f"Request {req.rid} input ({input_len} tokens) + generation reserve "
                f"({reserve}) exceeds the HiSparse decode DEVICE pool ({device_capacity}); "
                f"it cannot be served without overflowing the pool. Aborting."
            )
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
    except Exception:
        # Never let the guard break admission -- fall back to upstream behaviour.
        return original_fn(self, req, *args, **kwargs)

    return original_fn(self, req, *args, **kwargs)
