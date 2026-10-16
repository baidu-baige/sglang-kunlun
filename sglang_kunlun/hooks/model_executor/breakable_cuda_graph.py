"""Hooks for the breakable-CUDA-graph runtime helpers.

Kunlun exposes the XPU behind a CUDA-shaped torch API, so upstream's
``cuda-python`` introspection of stream-capture state is unusable here: the
bindings import fine, but every runtime call comes back with
``cudaErrorInsufficientDriver`` because there is no NVIDIA driver behind them.
``torch.cuda.is_current_stream_capturing()`` does work, so ask that instead.

The second hook keeps a capture error attributable: ``_end_current_segment``
asserts an open segment, which is wrong when the failure happened *between*
segments (inside an eager break body), and the assert then replaces the real
exception.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_CAPTURE_MODULE = (
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph"
)


@plugin_hook(f"{_CAPTURE_MODULE}._is_stream_capturing", type=HookType.REPLACE)
def is_stream_capturing_kunlun(stream) -> bool:
    """Capture-state query through the portable torch API."""
    from sglang.srt.utils import get_device_module

    device_module = get_device_module()
    with device_module.stream(stream):
        return device_module.is_current_stream_capturing()


@plugin_hook(
    f"{_CAPTURE_MODULE}.BreakableCUDAGraphCapture._end_current_segment",
    type=HookType.AROUND,
)
def end_current_segment_kunlun(original, self, *args, **kwargs):
    """Skip closing when no segment is open (nothing was begun to close)."""
    if self._current_graph is None:
        return None
    return original(self, *args, **kwargs)
