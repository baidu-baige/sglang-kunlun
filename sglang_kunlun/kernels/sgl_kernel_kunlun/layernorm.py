"""Kunlun-backed implementations of layernorm ``sgl_kernel`` APIs."""

from __future__ import annotations

import torch


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm and return a new output tensor."""

    import kunlun_ops

    out = torch.empty_like(x)
    kunlun_ops.rmsnorm(
        x,
        weight,
        out,
        eps,
        False,
        True,
        None,
        None,
        None,
    )
    return out


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """Apply fused residual add + RMSNorm in-place following sgl_kernel API."""

    import kunlun_ops

    out = torch.empty_like(x)
    kunlun_ops.add_rmsnorm(
        x,
        residual,
        weight,
        out,
        eps,
        False,
        True,
        None,
        None,
        residual,
        None,
    )
    x.copy_(out)
