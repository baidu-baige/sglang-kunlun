"""上游 API 变更的兼容层。"""

from __future__ import annotations

from typing import Any

_MISSING = object()


def memfd_create(name, flags=1):
    """Linux memfd with os-compatible flags (default MFD_CLOEXEC).

    Some Python builds omit os.memfd_create despite libc/kernel support.
    This is an explicit call-site fallback, never a global os monkey patch.

    Args:
        name: File name accepted by os.fsencode; embedded nulls are rejected.
        flags: Unsigned memfd flags, defaulting to MFD_CLOEXEC.

    Returns:
        An open file descriptor owned by the caller.
    """
    import ctypes
    import errno
    import operator
    import os
    import sys

    native = getattr(os, "memfd_create", None)
    if native is not None:
        return native(name, flags)
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "memfd_create requires Linux")
    encoded = os.fsencode(name)
    if b"\0" in encoded:
        raise ValueError("embedded null byte")
    flags = operator.index(flags)
    if not 0 <= flags <= 0xFFFFFFFF:
        raise OverflowError("flags must fit an unsigned int")
    libc = ctypes.CDLL(None, use_errno=True)
    create = getattr(libc, "memfd_create", None)
    if create is None:
        # Do not guess architecture-dependent syscall numbers.
        raise OSError(errno.ENOSYS, "libc does not expose memfd_create")
    create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    create.restype = ctypes.c_int
    fd = create(encoded, flags)
    if fd == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


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
