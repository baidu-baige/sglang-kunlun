"""Hooks for ``sglang.srt.distributed.parallel_state``.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/distributed/parallel_state.py

REPLACE ``init_model_parallel_group`` with a Kunlun-flavoured
``GroupCoordinator`` factory (use_pynccl=False, use_xpu_communicator=True,
use_npu_communicator=True, use_hpu_communicator=True).
"""

from __future__ import annotations

import os
from typing import List, Optional

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
import logging
logger = logging.getLogger(__name__)


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
    )
