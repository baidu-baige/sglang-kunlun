"""上游 API 变更的兼容层。"""

from __future__ import annotations

from typing import Any

_MISSING = object()


def server_arg(name: str, default: Any = _MISSING) -> Any:
    """读取 ServerArgs 字段当前生效的值。

    `get_global_server_args()` 在 dsv4.1 上已退休（它返回的是算子的原始输入，
    解析管线决定过的字段读回来是旧值且不报错）。上游让调用方改读 namespace bag
    （`get_exec().kernel.…` / `get_schedule().…`），但那要求每个调用点都知道自己的
    字段属于哪个 namespace；昆仑侧读的字段很分散，所以统一走 `resolution_result` ——
    与 bag 投影同一套规则，不需要知道 namespace 路径。
    """
    from sglang.srt.arg_groups.overrides import resolution_result
    from sglang.srt.runtime_context import get_server_args

    if default is _MISSING:
        return resolution_result(get_server_args(), name)
    return resolution_result(get_server_args(), name, default)
