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
"""HookRegistry registration for the DSV4 fused HC head.

Upstream ``sglang.srt.layers.mhc_head.fused_hc_head`` runs the rmsnorm + sigmoid
gating + weighted sum as a tilelang JIT kernel. The REPLACE hook below keeps the
GEMM in torch (``nofc`` = the Kunlun op has no FC step) and fuses the rest into
``kunlun_ops.fused_dpsk_v4_hc_head_nofc``.

``sglang.srt.models.deepseek_v4`` imports the symbol lazily inside the call site,
so it resolves the patched module attribute at call time.
"""

from __future__ import annotations

import kunlun_ops
import torch
import torch.nn.functional as F

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.mhc_head.fused_hc_head",
    type=HookType.REPLACE,
)
def _fused_hc_head_kunlun(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    """Kunlun-native fused HC Head via ``kunlun_ops.fused_dpsk_v4_hc_head_nofc``.

    Step 1: GEMM (linear) — done separately (nofc = no FC in the kunlun op)
    Step 2: ``fused_dpsk_v4_hc_head_nofc`` fuses rmsnorm rsqrt + sigmoid gating + weighted sum
    """
    shape, dtype = x.size(), x.dtype
    if shape[0] == 0:
        return torch.empty((0, shape[-1]), dtype=dtype, device=x.device)

    hc_mult = shape[1]
    hidden_size = shape[2]

    # Step 1: GEMM (linear) — produce fp32 gemm_out [B, hc_mult]
    x_flat = x.flatten(1).float()
    gemm_out = F.linear(x_flat, hc_fn.float())  # [B, hc_mult] fp32

    # Step 2: fused rmsnorm + sigmoid + weighted_sum via kunlun native op
    hs_flat = x.contiguous()  # [B, hc_mult, hidden_size]
    out = torch.empty((shape[0], hidden_size), dtype=hs_flat.dtype, device=x.device)

    # Ensure hc_scale/hc_base are on the right device
    scale = hc_scale.to(x.device)
    base = hc_base.to(x.device)

    kunlun_ops.fused_dpsk_v4_hc_head_nofc(
        hs_flat,       # [B, hc_mult, hidden_size] bf16
        gemm_out,      # [B, hc_mult] fp32
        scale,         # [1] fp32
        base,          # [1] fp32
        out,           # [B, hidden_size] bf16
        hidden_size,   # int
        norm_eps,      # float
        hc_eps,        # float
        hc_mult,       # int
    )
    return out.to(dtype)
