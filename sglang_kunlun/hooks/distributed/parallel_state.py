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


# ── 补上 pynccl 被关掉之后的集合通信缺口 ─────────────────────────────────
#
# 本模块把 `init_model_parallel_group` 的 `use_pynccl` 强制成 False（昆仑不支持
# pynccl 的 CUDA private API，见 `bootstrap/pre_shim.py`）。但上游有两个集合通信
# 是 **pynccl-only**：
#
#   `parallel_state.py:1213`  `reduce_scatterv`  → "pynccl is required for reduce_scatterv"
#   `parallel_state.py:1400`  `all_gatherv`      → "pynccl is required for all_gatherv"
#
# 而且它们的 `with pynccl_comm.change_state(enable=True)` 写在 assert **之前**，
# 所以 pynccl 为 None 时抛的是
#     AttributeError: 'NoneType' object has no attribute 'change_state'
# 而不是那句可读的 assert 信息 —— 排查时容易误判成别的 None。
#
# 谁需要它：**DP attention 的 MoE 同步路径**
# （`models/deepseek_v4.py:3639` `_run_moe_ffn_dp_sync` 里的
# `get_tp_group().reduce_scatterv(...)`）。8 卡 DP + MoE 必然走到，
# 不补这个缺口 D 节点会在 CUDA graph capture 期直接挂。
#
# `_deterministic_reduce_scatterv`（`parallel_state.py:1186`）是同一语义的 torch 实现
# （`all_reduce` + `narrow`，逐位相同的结果），而 `all_reduce` 自身有完整的非 pynccl
# 回退（hpu/xpu/npu 通信器 → 最终 `torch.distributed.all_reduce`），
# 所以 pynccl 缺失时直接复用它。
#
# `SGLANG_KUNLUN_REDUCE_SCATTERV_FALLBACK=0` 可关掉，回到上游行为（会在 DP 下报错）。

_KUNLUN_RSV_WARNED: set = set()


@plugin_hook(
    "sglang.srt.distributed.parallel_state.GroupCoordinator.reduce_scatterv",
    type=HookType.AROUND,
)
def reduce_scatterv_without_pynccl(original_fn, self, input_, output=None, sizes=None):
    """pynccl 缺失时用 `_deterministic_reduce_scatterv` 顶掉 pynccl-only 实现。"""
    import logging
    import os

    if os.environ.get("SGLANG_KUNLUN_REDUCE_SCATTERV_FALLBACK", "1") != "1":
        return original_fn(self, input_, output, sizes)

    if getattr(self, "pynccl_comm", None) is not None:
        return original_fn(self, input_, output, sizes)

    key = getattr(self, "unique_name", id(self))
    if key not in _KUNLUN_RSV_WARNED:
        _KUNLUN_RSV_WARNED.add(key)
        logging.getLogger(__name__).info(
            "reduce_scatterv: 该组 pynccl_comm 为 None（昆仑关闭 pynccl），"
            "改用 _deterministic_reduce_scatterv（all_reduce + narrow）。"
            "group=%s world_size=%s",
            getattr(self, "unique_name", "?"),
            getattr(self, "world_size", "?"),
        )
    return self._deterministic_reduce_scatterv(input_, output, sizes)
