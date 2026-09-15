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
"""HookRegistry registration for the MHC split-Sinkhorn gate kernel.

Upstream ``sglang.srt.layers.mhc.hc_split_sinkhorn`` dispatches to a tilelang
JIT kernel that cannot build on the Kunlun XPU. The REPLACE hook below keeps the
same (pre, post, comb) contract: ``pre``/``post`` are per-token sigmoid gates and
``comb`` is Sinkhorn-normalized.

``sglang.srt.models.deepseek_v4`` imports the symbol lazily inside the call site,
so it resolves the patched module attribute at call time.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.mhc.hc_split_sinkhorn",
    type=HookType.REPLACE,
)
def _hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Kunlun fused split-Sinkhorn, falling back to pure torch."""
    b, s, _ = mixes.size()
    hc = int(hc_mult)
    # Kunlun fused MHC split-Sinkhorn: same pre/post/comb definition and
    # Sinkhorn iteration count as the torch reference below, but the 20-iteration
    # row/col normalization loop runs in a single fused kernel. Falls back to the
    # torch implementation if the fused op is unavailable.
    try:
        import kunlun_ops

        _m = mixes.reshape(-1, (2 + hc) * hc).contiguous()
        _nr = _m.shape[0]
        _pre = torch.empty((_nr, hc), dtype=torch.float32, device=mixes.device)
        _post = torch.empty((_nr, hc), dtype=torch.float32, device=mixes.device)
        _comb = torch.empty((_nr, hc * hc), dtype=torch.float32, device=mixes.device)
        kunlun_ops.mhc_split_sinkhorn(
            _m,
            hc_scale.float().contiguous(),
            hc_base.float().contiguous(),
            _pre,
            _post,
            _comb,
            hc,
            int(sinkhorn_iters),
            float(eps),
        )
        return (
            _pre.reshape(b, s, hc),
            _post.reshape(b, s, hc),
            _comb.reshape(b, s, hc, hc),
        )
    except Exception:
        pass

    m = mixes.reshape(-1, (2 + hc) * hc).float()
    scale = hc_scale.float()
    base = hc_base.float()
    n = m.shape[0]

    pre = torch.sigmoid(m[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (m[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(n, hc, hc)

    # row softmax over k (dim=2)
    row_max = comb.amax(dim=2, keepdim=True)
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=2, keepdim=True)
    comb = comb / row_sum + eps

    # initial column normalization over j (dim=1)
    col_sum = comb.sum(dim=1, keepdim=True)
    comb = comb / (col_sum + eps)

    for _ in range(int(sinkhorn_iters) - 1):
        row_sum = comb.sum(dim=2, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=1, keepdim=True)
        comb = comb / (col_sum + eps)

    pre = pre.reshape(b, s, hc)
    post = post.reshape(b, s, hc)
    comb = comb.reshape(b, s, hc, hc)
    return pre, post, comb
