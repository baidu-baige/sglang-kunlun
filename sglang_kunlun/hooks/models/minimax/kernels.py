"""Kunlun kernel replacements for the MiniMax-M3 path.

Two groups, both registered at import time (``register_jit_op`` records a spec;
``kernel_ops.install()`` applies it, so this package must be imported before
``install()`` — see ``sglang_kunlun/hooks/registry.py``):

* ``minimax_store_kv_index`` / ``minimax_qknorm_rope`` — upstream
  ``sglang.kernels.ops.*`` helpers replaced by XSpeedGate launches;
* ``moe_fused_gate`` is NOT registered here: the main plugin already covers
  ``sglang.kernels.ops.moe.moe_fused_gate`` for the same upstream contract.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from sglang_kunlun.kernels.kernel_ops import register_jit_op


@register_jit_op("sglang.kernels.ops.kvcache.minimax_store_kv_index", "store_kv_index")
def store_kv_index(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v: torch.Tensor | None,
    idx_v_cache: torch.Tensor | None,
    indices: torch.Tensor,
    *,
    num_kv_heads: int,
    head_bytes: int,
) -> None:
    """Store the MiniMax-M3 main and index caches in one XSpeedGate launch."""
    import xspeedgate_ops  # noqa: F401 - 注册 xspeedgate_ops schema

    torch.ops.xspeedgate_ops.store_kv_index(
        k,
        v,
        k_cache,
        v_cache,
        idx_k,
        idx_k_cache,
        idx_v,
        idx_v_cache,
        indices,
        num_kv_heads,
        head_bytes,
    )


@register_jit_op(
    "sglang.kernels.ops.attention.minimax_qknorm_rope", "minimax_qknorm_rope_grouped"
)
def minimax_qknorm_rope_grouped(
    qkv: torch.Tensor,
    groups: Sequence[Tuple[torch.Tensor, int, int]],
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply grouped Gemma QKNorm + partial NeoX RoPE in place."""
    weights = []
    offsets = []
    counts = []
    for weight, offset, count in groups:
        if count > 0:
            weights.append(weight)
            offsets.append(int(offset))
            counts.append(int(count))

    if not 1 <= len(weights) <= 4:
        raise ValueError(
            f"MiniMax QKNorm+RoPE requires 1..4 non-empty groups, got {len(weights)}"
        )

    import xspeedgate_ops  # noqa: F401 - 注册 xspeedgate_ops schema

    torch.ops.xspeedgate_ops.fused_gemma_qknorm_rope_grouped(
        qkv,
        weights,
        offsets,
        counts,
        cos_sin_cache,
        positions,
        float(eps),
    )
    return qkv


@register_jit_op("sglang.kernels.ops.attention.minimax_qknorm_rope", "minimax_qknorm_rope")
def minimax_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    nq: int,
    nk: int,
    nv: int,
    eps: float,
) -> torch.Tensor:
    """Apply main Q/K Gemma QKNorm + partial NeoX RoPE in place."""

    import xspeedgate_ops  # noqa: F401 - 注册 xspeedgate_ops schema

    torch.ops.xspeedgate_ops.fused_gemma_qknorm_rope(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        int(nq),
        int(nk),
        int(nv),
        float(eps),
    )
    return qkv


# NOTE: ``sglang.kernels.ops.moe.moe_fused_gate`` is already registered by the
# main plugin (``sglang_kunlun/kernels/kernel_ops.py`` ->
# ``moe_fused_gate_dsv4_kunlun``), which forwards to
# ``kunlun_ops.moe_fused_gate_dsv4`` and covers the MiniMax-M3 contract too:
# ``scoring_func="sigmoid"`` with ``num_fused_shared_experts`` from the config.
# Registering it again here raises "jit op already registered" (kernel_ops
# rejects duplicate keys), so the MiniMax package deliberately leaves it alone.
