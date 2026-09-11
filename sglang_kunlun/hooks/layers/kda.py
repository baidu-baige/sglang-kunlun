"""Kunlun overrides for Kimi-K3's nvcc-JIT fast paths.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.kernels.ops.attention.kda_fused_decode.covered",
    type=HookType.REPLACE,
)
def disable_k3_fused_decode_kunlun(*args, **kwargs) -> bool:
    """Never claim the fully-fused KDA decode kernel is usable."""
    del args, kwargs
    return False


@plugin_hook(
    "sglang.kernels.ops.kimi_k3.mla_output_gate.kimi_k3_mla_output_gate",
    type=HookType.REPLACE,
)
def kimi_k3_mla_output_gate_kunlun(
    x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Torch form of the MLA output gate, preserving the two-step rounding."""
    return x * torch.sigmoid(gate)


@plugin_hook(
    "sglang.srt.models.kimi_k3._add3",
    type=HookType.REPLACE,
)
def kimi_k3_add3_kunlun(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    *,
    prefetch_bc: bool = False,
) -> torch.Tensor:
    """Two rounded additions without the NVIDIA add3 JIT."""
    del prefetch_bc
    result = a + b
    return result if c is None else result + c


@plugin_hook(
    "sglang.srt.models.kimi_k3.KimiK3DeltaAttention.forward_qkvbfg_fused",
    type=HookType.AROUND,
)
def kda_forward_qkvbfg_fused_kunlun(original_fn, self, hidden_states: torch.Tensor):
    """Torch projections instead of the ``kimi_k3_tiny_gemm`` JIT."""
    if not self.use_full_rank_gate or self._bfa_w is None:
        return original_fn(self, hidden_states)

    fused_states, _ = self.fused_qkvg_proj(hidden_states)
    qkv, g_proj_states = torch.split(fused_states, self.split_sizes, dim=-1)

    n_fa, n_b = self._bfa_fa_size, self._bfa_b_size
    bfa = F.linear(hidden_states, self._bfa_w)
    forget_gate = F.linear(bfa[..., :n_fa], self._bfa_f_b_w)
    beta = bfa[..., n_fa : n_fa + n_b]
    return qkv, beta, forget_gate, g_proj_states


@plugin_hook(
    "sglang.srt.models.kimi_k3.KimiK3MLAAttention._precompute_output_gate",
    type=HookType.REPLACE,
)
def skip_k3_output_gate_precompute_kunlun(self, hidden_states: torch.Tensor) -> None:
    """Keep the MLA output gate on the main stream."""
    del hidden_states
    self._gate_precomputed = None
