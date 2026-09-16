"""Kunlun kernel replacements for the MiniMax-M3 path.

Two groups, both registered at import time (``register_jit_op`` records a spec;
``kernel_ops.install()`` applies it, so this package must be imported before
``install()`` — see ``sglang_kunlun/hooks/registry.py``):

* ``minimax_store_kv_index`` / ``minimax_qknorm_rope`` — upstream
  ``sglang.kernels.ops.*`` helpers replaced by XSpeedGate launches;
* ``moe_fused_gate`` is NOT registered here either: the main plugin owns the
  single registration and dispatches the MiniMax contract (sigmoid + output
  scaling) into :func:`moe_fused_gate_minimax` below.
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


def moe_fused_gate_minimax(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
):
    """MiniMax-M3 fused gate (sigmoid + optional fused shared experts).

    Called by the plugin's ``moe_fused_gate`` replacement only for the MiniMax
    contract (``scoring_func="sigmoid"`` with the routed scaling applied on the
    output); the DSV4 ``sqrtsoftplus`` contract keeps using
    ``kunlun_ops.moe_fused_gate_dsv4``.
    """
    if not renormalize:
        raise NotImplementedError(
            "Kunlun MiniMax moe_fused_gate requires renormalize=True"
        )
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    import kunlun_ops

    num_tokens, num_experts = scores.shape
    weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=scores.device)
    indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=scores.device)

    if num_fused_shared_experts:
        import xspeedgate_ops  # noqa: F401 - 注册 xspeedgate_ops schema

        if hasattr(torch.ops.xspeedgate_ops, "minimax_m3_moe_fused_gate"):
            return torch.ops.xspeedgate_ops.minimax_m3_moe_fused_gate(
                scores,
                bias,
                1,
                1,
                topk,
                num_fused_shared_experts,
                routed_scaling_factor,
                apply_routed_scaling_factor_on_output,
            )
        # Older xspeedgate_ops (e.g. 1.2.0) has no M3 gate; the kunlun gate
        # implements the same contract including the fused shared experts.
        kunlun_ops.moe_fused_gate(
            scores,
            bias,
            1,
            1,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor,
            weights,
            indices,
            torch.empty((12, num_experts), dtype=torch.int32, device=scores.device),
        )
    else:
        kunlun_ops.moe_fused_gate(
            scores,
            bias,
            1,
            1,
            topk,
            0,
            routed_scaling_factor,
            weights,
            indices,
            torch.empty((12, num_experts), dtype=torch.int32, device=scores.device),
        )

    if apply_routed_scaling_factor_on_output:
        weights *= routed_scaling_factor
    return weights, indices
