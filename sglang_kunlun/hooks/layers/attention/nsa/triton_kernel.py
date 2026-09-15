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
"""Function-level hooks for NSA Triton kernel helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.triton_kernel.act_quant",
    type=HookType.REPLACE,
)
def act_quant_kunlun(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise INT8 quantization for Kunlun XPU.

    Kunlun XPU does not support ``float8_e4m3fn`` as a first-class dtype;
    the fused indexer / attention kernels (``c4a_mqa_logits``,
    ``c4a_paged_mqa_logits``, ``c4a_paged_mqa_logits_with_mixed_cache``) all
    consume ``int8`` for Q with a per-row float32 scale.

    Uses ``kunlun_ops.quant2d`` (kunlun native per-row absmax INT8 quant).

    Returns:
      y_int8: same shape as ``x``, dtype ``torch.int8``.
      y_max:  ``[x.shape[0], -1]`` float32; per-block abs-max value.
              Note: this is the raw max, not the true scale;
              ``true_scale = y_max / 127``.
    """
    import kunlun_ops

    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)

    x_shape = x.shape
    x_blocks = x.view(-1, block_size).contiguous()
    y_int8 = torch.empty(x_blocks.shape, dtype=torch.int8, device=x.device)
    y_max = torch.empty(x_blocks.shape[0], dtype=torch.float32, device=x.device)
    kunlun_ops.quant2d(x_blocks, y_int8, y_max, force_sdnn=True)

    return y_int8.view(x_shape), y_max.view(x_shape[0], -1)
