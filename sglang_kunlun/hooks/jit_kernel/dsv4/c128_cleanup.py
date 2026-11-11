# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Kunlun hook for ``sglang.kernels.ops.attention.dsv4.c128_cleanup``.

``clear_unaccepted_c128_draft_states`` resets the c128 compressed draft-state
rows for speculative (MTP/EAGLE) tokens that the verify step did not accept:
for every draft slot at or beyond the accepted length it zeros the first half
of the row and sets the second half to ``-inf``. Upstream implements this with
a triton kernel that the XPU triton backend cannot lower
(``operand #0 does not dominate this use`` -> ``PassManager::run failed``), so
replace it with a vectorized pure-torch equivalent. This runs in the eager
verify post-step (not inside a CUDA graph), so dynamic-shape masking is fine.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.kernels.ops.attention.dsv4.c128_cleanup.clear_unaccepted_c128_draft_states",
    type=HookType.REPLACE,
)
def clear_unaccepted_c128_draft_states_kunlun(
    state: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    ring_size: int,
    num_draft_tokens: int,
) -> None:
    """Vectorized torch equivalent of the triton c128 draft-state cleanup."""
    half = state.shape[-1] // 2
    device = state.device

    draft = torch.arange(num_draft_tokens, device=device, dtype=torch.int64)  # [T]
    # A draft slot is unaccepted when draft_offset >= accept_len for its request.
    unaccepted = draft.unsqueeze(0) >= accept_lens.to(torch.int64).view(-1, 1)  # [B,T]

    req = req_pool_indices.to(torch.int64).view(-1, 1)  # [B,1]
    seq = seq_lens.to(torch.int64).view(-1, 1)          # [B,1]
    slot = (seq + draft.unsqueeze(0)) % ring_size        # [B,T]
    rows = (req * ring_size + slot)[unaccepted]          # [N] flat state rows

    # Empty selection -> no-op; matches the per-element `if <accept: return`.
    state[rows, :half] = 0.0
    state[rows, half:] = float("-inf")
