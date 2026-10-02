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
