"""Hooks for ``sglang.srt.layers.attention.dsv4.compressor``.

target 与上游文件一一对应：Compressor 的构造后 dtype 恢复。
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from .dtype_restore import restore_requested_fp16_parameter_dtype


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.compressor.Compressor.__init__",
    type=HookType.AFTER,
)
def initialize_compressor_parameter_dtype_kunlun(result, self, *args, **kwargs):
    """Restore the requested dtype for the compressor gate projection."""
    restore_requested_fp16_parameter_dtype(self.wkv_gate)
    return result
