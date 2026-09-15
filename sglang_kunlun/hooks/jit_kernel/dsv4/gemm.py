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
"""HookRegistry registration for the DSV4 bf16 x bf16 -> fp32 linear.

Upstream ``sglang.jit_kernel.dsv4.gemm.linear_bf16_fp32`` relies on the
``torch.mm(out_dtype=...)`` overload (``aten::mm.dtype``), which the Kunlun XPU does
not provide. The REPLACE hook below does a plain fp32 matmul instead.

``sglang.jit_kernel.dsv4.__init__`` re-exports this symbol;
``srt.layers.attention.dsv4.compressor`` imports it at module scope and
``models.deepseek_v2`` lazily inside the call site. ``_propagate_patch`` rewrites the
already-imported bindings, and the re-export is patched for later importers.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.jit_kernel.dsv4.gemm.linear_bf16_fp32",
    type=HookType.REPLACE,
)
def _linear_bf16_fp32_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Plain fp32 matmul (no ``aten::mm.dtype`` overload on Kunlun)."""
    return torch.matmul(x.to(torch.float32), y.to(torch.float32).t())
