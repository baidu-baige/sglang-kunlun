"""Hooks for ``sglang.srt.utils.common``.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/utils/common.py
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List, Optional

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


def _is_kunlun() -> bool:
    try:
        import torch_xmlir  # noqa: F401
        return True
    except Exception:
        return False


def _xpusmi_path() -> Optional[str]:
    try:
        import pkg_resources

        return (
            pkg_resources.get_distribution("xmlir").location
            + "/torch_xmlir/xre/bin"
        )
    except Exception:
        return None



# @plugin_hook(
#     "sglang.srt.utils.common.support_triton",
#     type=HookType.REPLACE,
# )
# def support_triton_kunlun(backend: str) -> bool:
#     """Always False on Kunlun; preserves upstream's backend-blacklist gate."""
#     if backend in ("torch_native", "intel_amx"):
#         return False
#     return False  # Kunlun never returns True

# ---------------------------------------------------------------------------
# get_nvgpu_memory_capacity → query Kunlun XPU memory through xpu-smi
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.utils.common.get_nvgpu_memory_capacity",
    type=HookType.REPLACE,
)
def get_nvgpu_memory_capacity_kunlun():
    """Mock nvidia-smi by parsing ``xpu-smi`` output for Kunlun."""
    bin_dir = _xpusmi_path()
    if bin_dir is None:
        return None
    try:
        cmd = (
            f"{bin_dir}/xpu-smi "
            "| awk '/^\\| *[0-9]+ +P[0-9]00 OAM/{getline; "
            'gsub(/.*\\/ |MiB.*/, ""); print}\''
        )
        result = subprocess.run(
            [cmd],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        out = result.stdout.strip().splitlines()
        if not out:
            return None
        return int(out[0].strip())
    except (FileNotFoundError, ValueError, OSError) as e:
        logger.warning("xpu-smi memory-capacity query failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# set_gpu_proc_affinity → bind worker process to NUMA-local CPUs
# ---------------------------------------------------------------------------


def _parse_cpu_ranges(cpu_ranges: str) -> List[int]:
    """Parse CPU ranges like ``0-15,128-143`` into core IDs."""
    core_ids: List[int] = []
    for part in cpu_ranges.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            core_ids.extend(range(start, end + 1))
        else:
            core_ids.append(int(part))
    return core_ids


def _get_gpu_topo_affinity(gpu_id: int):
    """Return ``(numa_node, cpu_cores)`` from ``xpu-smi topo -m``."""
    try:
        result = subprocess.run(
            ["xpu-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if not parts or parts[0] != f"XPU{gpu_id}":
                continue

            # xpu-smi prints CPU_Affinity and NUMA Affinity as the last two
            # columns; parse from the right so the XPU/NIC matrix width can vary.
            cpu_affinity = parts[-2]
            numa_node = int(parts[-1])
            return numa_node, _parse_cpu_ranges(cpu_affinity)
    except Exception as e:
        logger.warning("Failed to get XPU topology affinity: %s", e)
    return None, []


def _get_numa_cores(node: int) -> List[int]:
    """Return CPU core IDs belonging to NUMA ``node`` (lscpu fallback)."""
    import psutil

    try:
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, check=True
        )
        output = result.stdout
        cpu_ranges = None
        for line in output.splitlines():
            if line.startswith(f"NUMA node{node} CPU(s):"):
                cpu_ranges = line.split(":", 1)[1].strip()
                break
        if cpu_ranges is None:
            raise ValueError(f"NUMA node {node} not found")
        return _parse_cpu_ranges(cpu_ranges)
    except Exception as e:
        logger.warning("Failed to get NUMA cores: %s", e)
        return list(range(psutil.cpu_count(logical=False)))


def _contiguous_ranges(cores: List[int]) -> List[List[int]]:
    """Split a sorted core list into its contiguous blocks (SMT halves stay apart)."""
    blocks: List[List[int]] = []
    for c in sorted(cores):
        if blocks and c == blocks[-1][-1] + 1:
            blocks[-1].append(c)
        else:
            blocks.append([c])
    return blocks


def _topo_cpu_affinity_strings():
    """``{gpu_id: cpu_affinity_string}`` from a single ``xpu-smi topo -m`` call."""
    out = {}
    try:
        result = subprocess.run(
            ["xpu-smi", "topo", "-m"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if not parts or not parts[0].startswith("XPU"):
                continue
            try:
                gid = int(parts[0][3:])
            except ValueError:
                continue
            out[gid] = parts[-2]
    except Exception as e:  # noqa: BLE001 - topology is advisory
        logger.warning("Failed to read XPU topology for affinity split: %s", e)
    return out


def _split_shared_numa(gpu_id: int, cores: List[int]) -> List[int]:
    """Give each XPU sharing a NUMA node its own slice of that node's cores.

    `xpu-smi topo -m` reports the whole NUMA-local core list per XPU, so on a
    host where two XPUs hang off one node (measured on the Hygon C86 P455 box:
    8 XPUs over 4 nodes, DP0/DP1 both pinned to `16-31,144-159`) both schedulers
    get the same 16 physical cores and fight over them. Decode is a lockstep
    collective, so that contention shows up as arrival skew inside the deepep
    dispatch/combine kernels -- measured `mean-min` 6.36 ms/step on Hygon versus
    2.90 ms on the Intel reference, with isolated 10 ms spikes.

    The slice keeps each contiguous block (physical cores and their SMT
    siblings are separate blocks) split in the same proportion, so a process
    keeps whole SMT pairs, and it never widens the binding past the node, so no
    cross-NUMA traffic is introduced. Measured on 206 at 256 concurrency over
    three runs each: 4921.7 tok/s mean (spread 4908-4948) with the split versus
    4841.2 (spread 4756-4932) sharing, i.e. +1.7% mean and 4x tighter spread;
    220 concurrency is a wash (4479.8 vs 4489.2). Set the env to 0 to restore
    the shared binding.
    """
    if os.environ.get(
        "SGLANG_KUNLUN_SPLIT_NUMA_AFFINITY", "1"
    ).strip().lower() in ("0", "false", "no"):
        return cores
    topo = _topo_cpu_affinity_strings()
    mine = topo.get(gpu_id)
    if mine is None:
        return cores
    peers = sorted(g for g, a in topo.items() if a == mine)
    if len(peers) <= 1 or gpu_id not in peers:
        return cores
    k, r = len(peers), peers.index(gpu_id)
    sliced: List[int] = []
    for block in _contiguous_ranges(cores):
        n = len(block)
        if n < k:
            sliced.extend(block)
            continue
        sliced.extend(block[r * n // k : (r + 1) * n // k])
    return sliced or cores


@plugin_hook(
    "sglang.srt.utils.common.set_gpu_proc_affinity",
    type=HookType.REPLACE,
)
def set_gpu_proc_affinity_kunlun(gpu_id: int):
    """Pin worker process to its XPU's NUMA-local core slice.

    0.5.22 dropped ``(pp_size, tp_size, nnodes)`` from the upstream signature —
    it now reads them from ``sglang.srt.runtime_context.get_parallel()`` inside
    the body (``srt/utils/common.py:3091``), and both call sites pass only
    ``gpu_id`` (``srt/managers/scheduler.py:6069``,
    ``benchmark/one_batch.py:915``). Mirroring the new one-argument signature is
    all this replacement needs: the Kunlun binding is derived from
    ``xpu-smi topo -m`` / NUMA topology, so it never used the parallel layout.
    If a future Kunlun policy needs it, read it the same way upstream does
    (``get_parallel().pp_size`` / ``.tp_size`` / ``.nnodes``) rather than
    re-adding parameters no caller passes.
    """
    import psutil

    pid = os.getpid()
    p = psutil.Process(pid)

    gpu_numa_node, bind_cores = _get_gpu_topo_affinity(gpu_id)
    if not bind_cores:
        bind_cores = _get_numa_cores(gpu_numa_node or 0)
    bind_cores = _split_shared_numa(gpu_id, bind_cores)

    p.cpu_affinity(bind_cores)
    logger.info(
        "Process %s gpu_id %s is running on CPUs: %s",
        pid,
        gpu_id,
        p.cpu_affinity(),
    )


# ---------------------------------------------------------------------------
# get_device / get_device_count / get_device_capability
#
# Kunlun exposes CUDA-compatible torch APIs through xpytorch SYMBOL_REWRITE.
# On hybrid hosts (Kunlun XPU + leftover NVIDIA libcuda but stale driver) the
# rewrite can be incomplete and these helpers fall through to
# ``RuntimeError: No accelerator``. Force-return the Kunlun mapping so the rest
# of sglang gets a coherent device string.
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.utils.common.get_device",
    type=HookType.REPLACE,
)
def get_device_kunlun(device_id: Optional[int] = None) -> str:
    """Report ``cuda`` on Kunlun — torch_xmlir uses SYMBOL_REWRITE to map
    Kunlun XPU through CUDA APIs, so sglang must use ``cuda`` as the device
    string throughout (the Intel XPU path will fail)."""
    if device_id is None:
        return "cuda"
    return f"cuda:{device_id}"


@plugin_hook(
    "sglang.srt.utils.common.get_device_count",
    type=HookType.REPLACE,
)
def get_device_count_kunlun() -> int:
    """Use ``torch.cuda.device_count()`` (via torch_xmlir SYMBOL_REWRITE) first,
    then fall back to ``xpu-smi`` discovery."""
    import torch

    try:
        n = int(torch.cuda.device_count())
        if n > 0:
            return n
    except Exception:
        pass

    bin_dir = _xpusmi_path()
    if bin_dir:
        try:
            result = subprocess.run(
                [f"{bin_dir}/xpu-smi", "discovery"],
                capture_output=True,
                text=True,
                check=False,
            )
            count = 0
            for line in result.stdout.splitlines():
                if line.strip().startswith("|") and "P800" in line:
                    count += 1
            if count:
                return count
        except Exception:
            pass

    # Last-resort: trust the launch-script ``--tp`` value if exposed via env.
    tp = os.environ.get("WORLD_SIZE") or os.environ.get("KUNLUN_DEVICE_COUNT")
    if tp:
        try:
            return int(tp)
        except ValueError:
            pass
    return 1


@plugin_hook(
    "sglang.srt.utils.common.get_device_capability",
    type=HookType.REPLACE,
)
def get_device_capability_kunlun(device_id: int = 0):
    """Return a CUDA-compatible capability for SGLang quantization gates."""
    return (8, 0)


@plugin_hook(
    "sglang.srt.utils.common.get_device_name",
    type=HookType.REPLACE,
)
def get_device_name_kunlun(device_id: int = 0) -> str:
    """Stable identifier used for cache keys & logs."""
    return "Kunlun-XPU"


# ---------------------------------------------------------------------------
# async_h2d → stage the small host vectors through pinned memory
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.utils.common.async_h2d",
    type=HookType.REPLACE,
)
def async_h2d_kunlun(args, *, dtype, device):
    """Enqueue the list copy instead of draining the stream for it.

    Upstream only stages through pinned memory when
    ``current_platform.is_pin_memory_available()`` says so, and
    ``KunlunDeviceMixin`` inherits the ``DeviceMixin`` default of ``False``.
    That turns every call into ``torch.tensor(args).to(device)`` -- a pageable,
    host-blocking copy that first waits for everything already queued on the
    stream. Measured on this deployment: 31 ms per call behind a loaded queue,
    and the P-side 64K prefill step pays it nine times (once per compressed
    layer in ``_make_low_ratio_prefill_indexer_inputs``, plus
    ``get_tail_row_indices``) -- ~212 ms of an 838 ms step with the host
    stalled and the device draining.

    Pinned staging makes the same copy asynchronous (0.06 ms behind the same
    queue) and leaves the values bit-identical; it is what the CUDA platform
    does. The staging buffer is kept alive on the returned tensor so it cannot
    be recycled while the copy is still in flight.

    Pinned host memory is only unavailable on this platform for the multi-GB
    registrations that fail with vendor error 712 (see
    ``hooks/hisparse.py:_skip_host_register_for_hisparse_cold_pool``); these
    vectors are a few hundred bytes. ``SGLANG_KUNLUN_ASYNC_H2D=0`` restores the
    blocking pageable copy.
    """
    import torch

    if os.environ.get("SGLANG_KUNLUN_ASYNC_H2D", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return torch.tensor(args, dtype=dtype).to(device)
    host = torch.tensor(args, dtype=dtype, pin_memory=True)
    out = host.to(device, non_blocking=True)
    out._kunlun_h2d_staging = host
    return out

