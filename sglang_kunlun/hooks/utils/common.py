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


@plugin_hook(
    "sglang.srt.utils.common.set_gpu_proc_affinity",
    type=HookType.REPLACE,
)
def set_gpu_proc_affinity_kunlun(
    pp_size: int,
    tp_size: int,
    nnodes: int,
    gpu_id: int,
):
    """Pin worker process to its XPU's NUMA-local core slice."""
    import psutil

    pid = os.getpid()
    p = psutil.Process(pid)

    gpu_numa_node, bind_cores = _get_gpu_topo_affinity(gpu_id)
    if not bind_cores:
        bind_cores = _get_numa_cores(gpu_numa_node or 0)

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

