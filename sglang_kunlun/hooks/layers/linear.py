"""REPLACE ``ColumnParallelLinear.__init__`` to enforce float32 bias for w8a8.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/linear.py.

Reason: ``kunlun_ops.matmul`` currently requires bias to be float32 when
the linear method is W8A8Int8. The patch wraps the original __init__ so the
bias parameter is recreated with the correct dtype after the upstream init.
"""

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.linear.ColumnParallelLinear.__init__",
    type=HookType.AROUND,
)
def column_parallel_linear_init_around(original_fn, self, *args, **kwargs):
    """Run upstream init, then upgrade bias dtype for w8a8."""
    original_fn(self, *args, **kwargs)
    if getattr(self, "bias", None) is None:
        return

    # Inline import to avoid circular import at hook-registration time.
    from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8LinearMethod
    from sglang.srt.utils import set_weight_attrs

    if not isinstance(getattr(self, "quant_method", None), W8A8Int8LinearMethod):
        return

    self.bias = Parameter(
        torch.empty(self.output_size_per_partition, dtype=torch.float32)
    )
    set_weight_attrs(
        self.bias,
        {
            "output_dim": 0,
            "weight_loader": self.weight_loader,
        },
    )
