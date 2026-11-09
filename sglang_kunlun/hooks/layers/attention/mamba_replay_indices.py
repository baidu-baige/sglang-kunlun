"""Kunlun override for the mamba decode-graph replay-prep state-indices path.

Upstream ``sglang.kernels.ops.mamba.mamba_state_indices_triton.
fused_replay_state_indices`` fuses the ``_replay_metadata`` reference chain into
a single **Triton** launch. Triton on Kunlun XPU falls through to the stock
NVIDIA driver, which asserts ``libcuda.so cannot found`` the moment the kernel is
launched — this fires only on the decode CUDA-graph *capture / replay* path
(``HybridLinearAttnBackend.init_forward_metadata_out_graph -> _replay_metadata``),
which eager never reaches. The plugin already redirects ``kunlun_backend`` /
``kunlun_deepseek_v4_backend`` ``init_forward_metadata_out_graph``, but the GDN
(linear-attention) backend routed through this helper was never migrated.

This REPLACE swaps the Triton launch for a pure-torch equivalent that reproduces
the documented reference chain **bit-for-bit** (integer index math, no floating
point):

    req_pool_indices[valid_bs:total_bs] = 0        # side effect: zero padded rows
    out[:valid_bs]          = mapping[req_pool_indices[:valid_bs]]   # gather
    out[valid_bs:total_bs]  = -1                    # padding sentinel
    return out[:total_bs]

The gather of the valid rows uses the *original* ``req_pool_indices`` (the fused
kernel loads them with ``mask=valid`` before the zeroing store), so results are
identical to both the Triton kernel and the aten ``else`` branch in
``_replay_metadata``. ``translate_mamba_indices`` is identity for the static
``HybridReqToTokenPool`` (the only pool that takes this fast path — the unified
pool sets ``_fused_state_indices_ok`` False), so no translate step is needed here.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.kernels.ops.mamba.mamba_state_indices_triton.fused_replay_state_indices",
    type=HookType.REPLACE,
)
def fused_replay_state_indices_kunlun(
    *,
    req_pool_indices: torch.Tensor,
    mamba_index_mapping: torch.Tensor,
    out_state_indices: torch.Tensor,
    valid_bs: int,
    total_bs: int,
) -> torch.Tensor:
    """Torch equivalent of the Triton ``fused_replay_state_indices``.

    Fills ``out_state_indices[:total_bs]`` in place (gather + ``-1`` padding
    sentinel) and zeros the padded rows ``[valid_bs, total_bs)`` of the static
    ``req_pool_indices`` buffer, then returns the ``out_state_indices[:total_bs]``
    view. Mirrors the fused kernel exactly: valid rows gather the *original*
    ``req_pool_indices``; padded rows never gather.
    """
    # TORCH_FALLBACK[L5] fused_replay_state_indices: Triton kernel has no Kunlun XPU
    #   backend (launch hits stock NVIDIA driver -> libcuda missing) -> pure-torch
    #   integer index prep (gather + -1 sentinel + padded-row zeroing) on-device.
    # perf: once per decode cuda-graph replay | out (total_bs,) int32, total_bs<=max_bs
    #   | want: a Kunlun/XPU impl of the fused replay-prep (or the aten else-branch
    #   validated on XPU) so this host-authored index prep can be dropped.
    # Valid rows: mapping gather using the original (pre-zeroing) req ids.
    if valid_bs > 0:
        req_valid = req_pool_indices[:valid_bs].to(torch.long)
        gathered = mamba_index_mapping[req_valid].to(out_state_indices.dtype)
        out_state_indices[:valid_bs] = gathered

    # Padded rows: -1 sentinel (mamba kernels skip state_idx < 0), and reproduce
    # the reference chain's side effect of zeroing those req_pool_indices rows so
    # captured gathers stay in-bounds.
    if valid_bs < total_bs:
        out_state_indices[valid_bs:total_bs] = -1
        req_pool_indices[valid_bs:total_bs] = 0

    return out_state_indices[:total_bs]
