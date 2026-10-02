"""K3 MoE router implementations and oracle comparison helpers."""

from __future__ import annotations

import torch


def route_k3_torch(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int = 16,
):
    """K3 sigmoid+bias selection with un-biased score normalization."""
    if router_logits.ndim != 2:
        raise ValueError("router_logits must be 2-D")
    if correction_bias.ndim != 1 or correction_bias.shape[0] != router_logits.shape[1]:
        raise ValueError("correction_bias must match the expert dimension")
    scores = torch.sigmoid(router_logits.float())
    ids = torch.topk(scores + correction_bias.float(), top_k, dim=-1).indices
    weights = torch.gather(scores, -1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, ids.to(torch.int32)


def route_k3_xspeedgate(router_logits, correction_bias, top_k=16):
    """Use XSpeedGate when available; otherwise return the torch oracle."""
    try:
        import xspeedgate_ops

        return xspeedgate_ops.ops.minimax_m3_moe_fused_gate(
            router_logits.contiguous(),
            correction_bias.contiguous(),
            1,
            1,
            top_k,
            0,
            1.0,
            False,
        )
    except (ImportError, AttributeError):
        return route_k3_torch(router_logits, correction_bias, top_k)
