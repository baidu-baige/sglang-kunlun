"""Shared host KV storage for DeepSeek V4 MLA HiCache.

This feature is opt-in.  When enabled, every rank in one TP group maps the
same host allocation, while only the group leader performs D2H writes.  It is
intentionally limited to the DSV4 HiCache stack; the normal SGLang host pool
and allocator remain unchanged otherwise.
"""

from __future__ import annotations

import atexit
import ctypes
import glob
import hashlib
import logging
import mmap
import os
import platform
import random
import socket
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

ENABLED = get_bool_env_var("HICACHE_SHARED_HOST_KV", "false")
VERIFY = get_bool_env_var("HICACHE_SHARED_VERIFY", "false")
SHARED_DIR = os.environ.get("HICACHE_SHARED_DIR", "/dev/shm")
_HUGE_PAGE_SIZE = 2 * 1024 * 1024
_MADV_HUGEPAGE = 14
_NMAX = 256
_NWORDS = _NMAX // (8 * ctypes.sizeof(ctypes.c_ulong))
_MEMPOLICY_SYSCALLS = {
    "x86_64": {"set_mempolicy": 238, "get_mempolicy": 239},
    "aarch64": {"set_mempolicy": 237, "get_mempolicy": 236},
}
_MEMPOLICY_SYS = _MEMPOLICY_SYSCALLS.get(platform.machine())

_active = False
_tp_rank = 0
_tp_group = None
_tag = "sgl_hicache_shared"
_allocation_id = 0
_mappings: list["_Mapping"] = []


@dataclass
class _Mapping:
    tensor: Optional[torch.Tensor]
    mapping: mmap.mmap
    registered: bool


def is_enabled() -> bool:
    """Return whether shared host KV is enabled by environment configuration."""
    return ENABLED


def active() -> bool:
    """Return whether shared host KV is currently active for this rank."""
    return _active


def is_leader() -> bool:
    """Return whether this rank owns writes to the shared host KV."""
    return _active and _tp_rank == 0


def skip_write() -> bool:
    """Return whether this rank must skip host KV writes."""
    return _active and _tp_rank != 0


def _barrier() -> None:
    if _tp_group is not None:
        torch.distributed.barrier(group=_tp_group)


def _all_true(value: bool) -> bool:
    if _tp_group is None:
        return value
    result = torch.tensor([int(value)], dtype=torch.int32)
    torch.distributed.all_reduce(
        result, op=torch.distributed.ReduceOp.MIN, group=_tp_group
    )
    return bool(result.item())


def _same_node() -> bool:
    if _tp_group is None:
        return True
    name = platform.node() or socket.gethostname()
    digest = int(hashlib.sha1(name.encode()).hexdigest()[:15], 16)
    value = torch.tensor([digest], dtype=torch.int64)
    minimum, maximum = value.clone(), value.clone()
    torch.distributed.all_reduce(
        minimum, op=torch.distributed.ReduceOp.MIN, group=_tp_group
    )
    torch.distributed.all_reduce(
        maximum, op=torch.distributed.ReduceOp.MAX, group=_tp_group
    )
    return int(minimum.item()) == int(maximum.item())


def _host_register(tensor: torch.Tensor) -> None:
    cudart = getattr(torch.cuda, "cudart", None)
    if cudart is None:
        raise RuntimeError("torch.cuda.cudart is unavailable")
    api = cudart()
    register = getattr(api, "cudaHostRegister", None)
    if register is None:
        raise RuntimeError("cudaHostRegister is unavailable in the Kunlun runtime")
    size = tensor.numel() * tensor.element_size()
    rc = int(register(tensor.data_ptr(), size, 0))
    if rc != 0:
        raise RuntimeError(f"cudaHostRegister failed: rc={rc}, size={size}")


def _host_unregister(tensor: torch.Tensor) -> None:
    try:
        api = torch.cuda.cudart()
        unregister = getattr(api, "cudaHostUnregister", None)
        if unregister is not None:
            unregister(tensor.data_ptr())
    except Exception:
        logger.debug("Failed to unregister shared host KV", exc_info=True)


def _advise_hugepage(mapping: mmap.mmap, length: int) -> None:
    """Request transparent huge pages for a shared-memory mapping.

    ``/dev/shm`` must be mounted with ``huge=advise`` (or ``huge=always``).
    This is only a kernel hint: the mapping may still be backed by 4 KiB
    pages because of memory pressure, alignment, or runtime restrictions.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        madvise = libc.madvise
        madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        madvise.restype = ctypes.c_int
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        rc = madvise(
            ctypes.c_void_p(address),
            ctypes.c_size_t(length),
            _MADV_HUGEPAGE,
        )
        if rc != 0:
            logger.warning(
                "madvise(MADV_HUGEPAGE) failed for shared host KV: errno=%d",
                ctypes.get_errno(),
            )
    except Exception:
        logger.warning(
            "madvise(MADV_HUGEPAGE) is unavailable; shared host KV may use "
            "4 KiB pages",
            exc_info=True,
        )


def _online_numa_nodes() -> list[int]:
    nodes = []
    for path in glob.glob("/sys/devices/system/node/node[0-9]*"):
        try:
            nodes.append(int(os.path.basename(path)[4:]))
        except ValueError:
            pass
    return sorted(nodes) or [0]


def _numa_node_mask():
    mask = (ctypes.c_ulong * _NWORDS)()
    for node in _online_numa_nodes():
        if node < _NMAX:
            mask[node // 64] |= 1 << (node % 64)
    return mask


def _mempolicy_syscall(number, *args):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc.syscall(ctypes.c_long(number), *args)


def _set_mempolicy_interleave():
    """Temporarily interleave newly faulted shmem pages across NUMA nodes."""
    if _MEMPOLICY_SYS is None:
        return None
    try:
        previous_mode = ctypes.c_int(0)
        previous_mask = (ctypes.c_ulong * _NWORDS)()
        rc = _mempolicy_syscall(
            _MEMPOLICY_SYS["get_mempolicy"],
            ctypes.byref(previous_mode),
            ctypes.cast(previous_mask, ctypes.c_void_p),
            ctypes.c_ulong(_NMAX),
            ctypes.c_void_p(0),
            ctypes.c_ulong(0),
        )
        saved = (
            (previous_mode.value, previous_mask)
            if rc == 0
            else ("__default__", None)
        )
        rc = _mempolicy_syscall(
            _MEMPOLICY_SYS["set_mempolicy"],
            ctypes.c_int(3),  # MPOL_INTERLEAVE
            ctypes.cast(_numa_node_mask(), ctypes.c_void_p),
            ctypes.c_ulong(_NMAX),
        )
        if rc != 0:
            logger.warning(
                "set_mempolicy(MPOL_INTERLEAVE) failed for shared host KV: errno=%d",
                ctypes.get_errno(),
            )
            return None
        return saved
    except Exception:
        logger.warning(
            "NUMA interleave is unavailable for shared host KV",
            exc_info=True,
        )
        return None


def _restore_mempolicy(saved) -> None:
    if saved is None or _MEMPOLICY_SYS is None:
        return
    try:
        mode, mask = saved
        if mode == "__default__" or mode == 0:
            _mempolicy_syscall(
                _MEMPOLICY_SYS["set_mempolicy"],
                ctypes.c_int(0),
                ctypes.c_void_p(0),
                ctypes.c_ulong(0),
            )
        else:
            _mempolicy_syscall(
                _MEMPOLICY_SYS["set_mempolicy"],
                ctypes.c_int(mode),
                ctypes.cast(mask, ctypes.c_void_p),
                ctypes.c_ulong(_NMAX),
            )
    except Exception:
        logger.debug("Failed to restore NUMA memory policy", exc_info=True)


def configure(tp_rank: int, tp_group) -> None:
    """Enable shared mode for the current DSV4 TP group.

    The capability checks and the tag exchange are collective.  If the feature
    is explicitly enabled but the host setup is invalid, all ranks fail before
    any host pool is constructed.
    """
    global _active, _tp_rank, _tp_group, _tag
    _tp_rank, _tp_group = tp_rank, tp_group
    _active = False
    if not ENABLED:
        return

    world_size = (
        torch.distributed.get_world_size(group=tp_group) if tp_group is not None else 1
    )
    if world_size <= 1:
        return

    local_ok = os.path.isdir(SHARED_DIR) and os.access(SHARED_DIR, os.W_OK)
    local_ok = local_ok and callable(getattr(torch.cuda, "cudart", None))
    if not _same_node():
        local_ok = False
    if not _all_true(local_ok):
        raise RuntimeError(
            "HICACHE_SHARED_HOST_KV requires a writable shared directory, "
            "a same-node TP group, and torch.cuda.cudart host registration support"
        )

    seed = random.SystemRandom().getrandbits(62) if tp_rank == 0 else 0
    tag = torch.tensor([seed], dtype=torch.int64)
    torch.distributed.all_reduce(
        tag, op=torch.distributed.ReduceOp.MAX, group=tp_group
    )
    _tag = f"sgl_hicache_{int(tag.item()):016x}"
    _active = True
    logger.info(
        "Shared DSV4 host KV enabled: tp_rank=%d, dir=%s, tag=%s, numa=interleave",
        _tp_rank,
        SHARED_DIR,
        _tag,
    )


def maybe_shared_alloc(dims, dtype, device: str) -> Optional[torch.Tensor]:
    """Allocate a shared, DMA-registered CPU tensor or return ``None``."""
    if not _active or device != "cpu":
        return None

    global _allocation_id
    _allocation_id += 1
    number = 1
    for dim in dims:
        number *= int(dim)
    nbytes = number * torch.empty((), dtype=dtype).element_size()
    span = (
        (nbytes + _HUGE_PAGE_SIZE - 1) // _HUGE_PAGE_SIZE
    ) * _HUGE_PAGE_SIZE
    signature = f"{_tag}:{_allocation_id}:{tuple(dims)}:{dtype}"
    suffix = hashlib.sha1(signature.encode()).hexdigest()[:12]
    path = os.path.join(SHARED_DIR, f"{_tag}_{_allocation_id}_{suffix}")

    fd = -1
    mapping = None
    tensor = None
    buffer = None
    local_ok = True
    if _tp_rank == 0:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, span)
        except Exception:
            local_ok = False
            logger.exception("Failed to create shared DSV4 host KV file %s", path)
    _barrier()

    if local_ok:
        try:
            if _tp_rank != 0:
                fd = os.open(path, os.O_RDWR)
            mapping = mmap.mmap(
                fd,
                span,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except Exception:
            local_ok = False
            logger.exception("Failed to map shared DSV4 host KV file %s", path)
        finally:
            if fd >= 0:
                os.close(fd)
                fd = -1
    _barrier()

    if _tp_rank == 0:
        try:
            os.unlink(path)
        except OSError:
            pass

    if local_ok and mapping is not None:
        saved_policy = _set_mempolicy_interleave()
        try:
            _advise_hugepage(mapping, span)
            buffer = (ctypes.c_char * nbytes).from_buffer(mapping)
            tensor = torch.frombuffer(buffer, dtype=dtype, count=number).view(*dims)
            _host_register(tensor)
        except Exception:
            local_ok = False
            logger.exception(
                "Failed to register shared DSV4 host KV allocation %s", path
            )
        finally:
            _restore_mempolicy(saved_policy)

    if fd >= 0:
        os.close(fd)

    if not _all_true(local_ok):
        if tensor is not None:
            _host_unregister(tensor)
            tensor = None
        buffer = None
        if mapping is not None:
            try:
                mapping.close()
            except BufferError:
                logger.debug(
                    "Shared host KV mapping still has exported views after "
                    "allocation failure"
                )
        raise RuntimeError(
            "Shared DSV4 host KV allocation failed on at least one TP rank"
        )

    assert tensor is not None and mapping is not None
    _mappings.append(_Mapping(tensor, mapping, True))
    return tensor


def shared_alloc_or_default(
    dims, *, dtype, device, pin_memory, allocator
) -> torch.Tensor:
    """Return a shared host tensor, or fall back to upstream host allocation."""
    shared = maybe_shared_alloc(dims, dtype, device)
    if shared is not None:
        return shared
    from sglang.srt.mem_cache.pool_host.common import alloc_with_host_register

    return alloc_with_host_register(
        dims,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
        allocator=allocator,
    )


def verify_indices(host_indices) -> None:
    """Fail collectively if TP ranks disagree on shared host slots."""
    if not (_active and VERIFY and _tp_group is not None):
        return
    digest = int(
        hashlib.sha1(host_indices.detach().cpu().contiguous().numpy().tobytes())
        .hexdigest()[:15],
        16,
    )
    value = torch.tensor([digest], dtype=torch.int64)
    minimum, maximum = value.clone(), value.clone()
    torch.distributed.all_reduce(
        minimum, op=torch.distributed.ReduceOp.MIN, group=_tp_group
    )
    torch.distributed.all_reduce(
        maximum, op=torch.distributed.ReduceOp.MAX, group=_tp_group
    )
    if int(minimum.item()) != int(maximum.item()):
        raise RuntimeError("TP ranks allocated different shared DSV4 host KV slots")


def shutdown() -> None:
    """Release all shared mappings; safe to call more than once."""
    while _mappings:
        mapping = _mappings.pop()
        if mapping.registered and mapping.tensor is not None:
            _host_unregister(mapping.tensor)
        mapping.tensor = None
        try:
            mapping.mapping.close()
        except (BufferError, OSError):
            logger.debug("Failed to close shared host KV mapping", exc_info=True)


atexit.register(shutdown)
