# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Hooks for ``sglang.srt.distributed.parallel_state``.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/distributed/parallel_state.py

REPLACE ``init_model_parallel_group`` with a Kunlun-flavoured
``GroupCoordinator`` factory (use_pynccl=False, use_xpu_communicator=True,
use_npu_communicator=True, use_hpu_communicator=True).
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import List, Optional

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
import logging
logger = logging.getLogger(__name__)


# ---- CPU affinity binding (mirrors the deploy-sglang-pd profile patch) -------
# The profiler's per-rank CUPTI tracing needs each scheduler pinned to the CPU
# cores of its own XPU's NUMA node. Without the pin, all 8 ranks' host threads
# roam the whole core set; the trace capture then perturbs the device enough to
# fault the first profiled forward (run-039: error 700 at
# _kunlun_compressed_attention). The profile driver does this by AST-injecting
# _bhta_bind_cpu into parallel_state.py; we do the same from the plugin hook.
def _bhta_parse_cpu_ranges(range_str: str) -> list:
    """Parse a CPU range string into a list of (start, end) tuples."""
    ranges = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            ranges.append((int(start), int(end)))
        else:
            val = int(part)
            ranges.append((val, val))
    return ranges


def _bhta_split_ranges_for_xpus(ranges: list, xpu_index: int, total_xpus: int) -> list:
    """Split each range evenly among the XPUs sharing it and return this XPU's cores."""
    cores = []
    for start, end in ranges:
        range_size = end - start + 1
        chunk_size = range_size // total_xpus
        chunk_start = start + xpu_index * chunk_size
        if xpu_index == total_xpus - 1:
            chunk_end = end
        else:
            chunk_end = chunk_start + chunk_size - 1
        cores.extend(range(chunk_start, chunk_end + 1))
    return cores


def _bhta_parse_xpu_topo() -> dict:
    """Parse ``xpu-smi topo -m`` into a local_rank -> [cpu cores] mapping."""
    result = subprocess.run(
        ["xpu-smi", "topo", "-m"], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"xpu-smi topo -m failed: {result.stderr.strip()}")
    xpu_pattern = re.compile(r"^XPU(\d+)\s")
    affinity_pattern = re.compile(r"(\d+-\d+(?:,\d+-\d+)*)\s+\d+$")
    affinity_groups: dict = {}
    for line in result.stdout.strip().splitlines():
        if "NUMA Affinity" in line:
            continue
        match = xpu_pattern.match(line)
        if not match:
            continue
        aff_match = affinity_pattern.search(line)
        if not aff_match:
            continue
        affinity_groups.setdefault(aff_match.group(1), []).append(int(match.group(1)))
    cpu_map = {}
    for affinity_str, xpu_ids in affinity_groups.items():
        ranges = _bhta_parse_cpu_ranges(affinity_str)
        xpu_ids_sorted = sorted(xpu_ids)
        n = len(xpu_ids_sorted)
        for i, xpu_id in enumerate(xpu_ids_sorted):
            cpu_map[xpu_id] = _bhta_split_ranges_for_xpus(ranges, i, n)
    return cpu_map


def _bhta_bind_cpu(local_rank: int) -> None:
    """Bind the current process to the CPU cores of its XPU's NUMA node."""
    try:
        cores = _bhta_parse_xpu_topo().get(local_rank)
        if cores:
            os.sched_setaffinity(0, cores)
            logger.info(f"[rank {local_rank}] bind cpu {cores}")
        else:
            logger.info(f"[rank {local_rank}] no CPU affinity found, skipping bind")
    except Exception as e:  # noqa: BLE001 - never let binding break startup
        logger.warning(f"[rank {local_rank}] CPU binding failed: {e}")


@plugin_hook(
    "sglang.srt.distributed.parallel_state.init_distributed_environment",
    type=HookType.BEFORE,
)
def prepare_kunlun_profiler_environment(*args, **kwargs):
    """Configure Kunlun profiler env and CPU affinity before distributed init.

    Accept arbitrary args/kwargs so upstream signature changes to
    ``init_distributed_environment`` (e.g. the 0.5.14 ``recovered_rank``
    parameter) do not break this BEFORE hook. ``local_rank`` is read by name
    (kwarg) or by position (4th positional arg), matching the upstream
    signature ``(world_size, rank, distributed_init_method, local_rank, ...)``.
    """
    local_rank = kwargs.get("local_rank")
    if local_rank is None and len(args) >= 4:
        local_rank = args[3]
    if local_rank is None:
        local_rank = -1

    if "XPU_ENABLE_PROFILER_TRACING" in os.environ and os.environ["XPU_ENABLE_PROFILER_TRACING"] == "1":
        logger.info(f"enable trace for Kunlun profiler")
        device_id = local_rank % 8
        os.environ["XPU_CUPTI_ENABLE_DEVICE"] = str(device_id)
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        if local_rank >= 0:
            _bhta_bind_cpu(local_rank)


@plugin_hook(
    "sglang.srt.distributed.parallel_state.init_model_parallel_group",
    type=HookType.REPLACE,
)
def kunlun_init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_pynccl: Optional[bool] = None,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
    use_mscclpp_allreduce: Optional[bool] = None,
    use_torch_symm_mem_allreduce: Optional[bool] = None,
    recovered_rank: bool = False,
    rank_offset: int = 0,
    max_world_size: Optional[int] = None,
):
    """Kunlun: init GroupCoordinator with XPU/NPU/HPU communicators."""
    from sglang.srt.distributed.parallel_state import (
        _ENABLE_CUSTOM_ALL_REDUCE,
        _ENABLE_MSCCLPP_ALL_REDUCE,
        _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE,
        GroupCoordinator,
    )

    if use_custom_allreduce is None:
        use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
    if use_mscclpp_allreduce is None:
        use_mscclpp_allreduce = _ENABLE_MSCCLPP_ALL_REDUCE
    if use_torch_symm_mem_allreduce is None:
        use_torch_symm_mem_allreduce = _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE

    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_pymscclpp=use_mscclpp_allreduce,
        use_custom_allreduce=use_custom_allreduce,
        use_torch_symm_mem_all_reduce=use_torch_symm_mem_allreduce,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_npu_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        recovered_rank=recovered_rank,
        rank_offset=rank_offset,
        max_world_size=max_world_size,
    )
