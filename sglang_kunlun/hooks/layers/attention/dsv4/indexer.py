"""Hooks for ``sglang.srt.layers.attention.dsv4.indexer``.

target 与上游文件一一对应：C4Indexer 的构造后 dtype 恢复。
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from .dtype_restore import restore_requested_fp16_parameter_dtype


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.indexer.C4Indexer.__init__",
    type=HookType.AFTER,
)
def initialize_c4_indexer_parameter_dtype_kunlun(result, self, *args, **kwargs):
    """Restore the requested dtype for C4 indexer projections."""
    restore_requested_fp16_parameter_dtype(self.wq_b, self.weights_proj)
    return result
