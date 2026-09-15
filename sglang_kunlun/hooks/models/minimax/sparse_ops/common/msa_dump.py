"""把 CUDA graph 里 kunlun msa 算子的**真实**入参 dump 到磁盘，供离线单测复放。

为什么不能在算子调用点直接存值：CUDA graph replay 期 Python 不执行，算子调用点只在
capture 期跑一次，那时 buffer 里全是 capture 用的假数据。所以这里分两步：

* capture 期：只**登记**入参张量的引用。这些张量要么是后端的静态 metadata buffer，
  要么是 graph 私有内存池里的中间结果，replay 会原地写入真实值，地址不变。
* replay 之后：从 ``CudaGraphRunner.replay`` 的 AFTER hook 里读同一批 buffer，此时
  内容就是这次真实 decode 的值，clone 到 CPU 存盘。

``--disable-cuda-graph`` 的 eager 路径下直接在调用点存盘。

用法::

    export SGLANG_MSA_DUMP_DIR=/home/zx/msa_dump   # 只有设了才开启，其余全是可选
    export SGLANG_MSA_DUMP_REPLAY_AT=8   # 从某个 bs 的第 N 次 replay 起检查快照
    export SGLANG_MSA_DUMP_MIN_KV_LEN=0  # 只保存 KV 长度达到阈值的 replay
    export SGLANG_MSA_DUMP_LAYER=0       # 取一次 forward 里第几个稀疏层的调用
    export SGLANG_MSA_DUMP_KV=0          # 是否连 KV cache 的值一起存（默认只存 spec）
    export SGLANG_MSA_DUMP_MAX_MB=64     # 超过这个大小的张量只存 spec
    export SGLANG_MSA_DUMP_RANKS=0       # 参与 dump 的 TP rank，逗号分隔，all = 全部

每个 ``(算子, batch_size)`` 存一个文件 ``{op}_bs{bs}_rank{r}.pt``，内容见 ``_save``。
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# 只存 spec 不存值的大张量（KV cache 动辄几十 GB，单测里随机填即可，值不影响耗时）
_KV_ARGS = frozenset({"k_cache", "v_cache"})


@lru_cache(maxsize=1)
def _dump_dir() -> Optional[str]:
    path = os.environ.get("SGLANG_MSA_DUMP_DIR", "").strip()
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def msa_dump_enabled() -> bool:
    return _dump_dir() is not None


@lru_cache(maxsize=1)
def _replay_at() -> int:
    return max(int(os.environ.get("SGLANG_MSA_DUMP_REPLAY_AT", "8")), 1)


@lru_cache(maxsize=1)
def _min_kv_len() -> int:
    return max(int(os.environ.get("SGLANG_MSA_DUMP_MIN_KV_LEN", "0")), 0)


@lru_cache(maxsize=1)
def _layer_index() -> int:
    return max(int(os.environ.get("SGLANG_MSA_DUMP_LAYER", "0")), 0)


@lru_cache(maxsize=1)
def _dump_kv() -> bool:
    return os.environ.get("SGLANG_MSA_DUMP_KV", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


@lru_cache(maxsize=1)
def _max_bytes() -> int:
    return int(float(os.environ.get("SGLANG_MSA_DUMP_MAX_MB", "64")) * 1024 * 1024)


@lru_cache(maxsize=1)
def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


@lru_cache(maxsize=1)
def _rank_enabled() -> bool:
    spec = os.environ.get("SGLANG_MSA_DUMP_RANKS", "0").strip().lower()
    if spec in ("all", "*"):
        return True
    return _rank() in {int(x) for x in spec.split(",") if x.strip()}


# (op, bs) -> capture 期登记的 kwargs（存的是张量引用，值等 replay 后再读）
_pending: Dict[tuple, Dict[str, Any]] = {}
# (op, bs) -> 该 key 的调用计数，用来挑第 _layer_index() 个稀疏层
_seen: Dict[tuple, int] = defaultdict(int)
# 已经存过盘的 (op, bs)
_done: set = set()
# bs -> replay 次数
_replays: Dict[int, int] = defaultdict(int)


def record(op: str, kwargs: Dict[str, Any], bs: int) -> None:
    """登记（capture 期）或直接落盘（eager 期）一次算子调用的入参。

    ``kwargs`` 必须就是传给 ``kunlun_ops.<op>`` 的那个 dict，单测按名字原样回灌。
    """
    if not msa_dump_enabled() or not _rank_enabled():
        return
    key = (op, bs)
    if key in _done or key in _pending:
        return
    index = _seen[key]
    _seen[key] = index + 1
    if index != _layer_index():
        return

    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    if get_is_capture_mode():
        # 只留引用，等 replay 写入真实值后再读
        _pending[key] = dict(kwargs)
        logger.info("msa dump: armed %s bs=%d, waiting for replay", op, bs)
    else:
        _save(op, bs, kwargs)
        _done.add(key)


def flush(bs: int) -> None:
    """replay 之后把某个 bs 登记的所有算子快照落盘。"""
    keys = [key for key in _pending if key[1] == bs]
    if not keys:
        return

    min_kv_len = _min_kv_len()
    if min_kv_len:
        # 以主 sparse attention 的 live device LOD 为准，确保跳过服务启动
        # warmup，并让两个算子保存的是同一次目标长度 replay。
        kv_lod = next(
            (
                _pending[key].get("lod_seqlens_kv_xpu")
                for key in keys
                if key[0] == "msa_sparse_attention"
            ),
            None,
        )
        if (
            not isinstance(kv_lod, torch.Tensor)
            or kv_lod.numel() == 0
            or int(kv_lod[-1].item()) < min_kv_len
        ):
            return

    torch.cuda.synchronize()
    for key in keys:
        kwargs = _pending.pop(key)
        _save(key[0], key[1], kwargs)
        _done.add(key)


def _pack(name: str, value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        # int / float / bool / None 原样存，其余存 repr（单测不会用到）
        return value if isinstance(value, (int, float, bool, type(None))) else repr(value)
    spec = {
        "__spec__": True,
        "name": name,
        "shape": tuple(value.shape),
        "dtype": value.dtype,
        "contiguous": value.is_contiguous(),
        "nbytes": value.numel() * value.element_size(),
    }
    if name in _KV_ARGS and not _dump_kv():
        return spec
    if spec["nbytes"] > _max_bytes():
        return spec
    return value.detach().to("cpu", copy=True)


def _save(op: str, bs: int, kwargs: Dict[str, Any]) -> None:
    payload = {
        "op": op,
        "bs": bs,
        "rank": _rank(),
        "layer_index": _layer_index(),
        "kwargs": {name: _pack(name, value) for name, value in kwargs.items()},
    }
    path = os.path.join(_dump_dir(), f"{op}_bs{bs}_rank{_rank()}.pt")
    torch.save(payload, path)
    logger.info("msa dump: saved %s", path)


def _after_cuda_graph_replay(result, self, *args, **kwargs):
    """AFTER hook：数够 replay 次数就快照，取的是热态 decode 的真实 buffer。"""
    bs = getattr(self, "bs", None)
    if bs is None:
        return result
    _replays[bs] += 1
    if _replays[bs] >= _replay_at():
        flush(bs)
    return result


if msa_dump_enabled():
    # 这个模块在 plugin 加载期被 decode 算子模块 import，早于 HookRegistry.apply_hooks，
    # 所以此处注册能生效。关闭 dump 时完全不注册，replay 路径零额外开销。
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.model_executor.cuda_graph_runner.CudaGraphRunner.replay",
        _after_cuda_graph_replay,
        HookType.AFTER,
    )
