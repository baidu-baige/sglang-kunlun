"""Kunlun-backed implementations of layernorm ``sgl_kernel`` APIs."""

from __future__ import annotations

import torch


def _resolve_add_rmsnorm_inplace():
    """Return ``torch.ops.xspeedgate_ops.add_rmsnorm_inplace``, or None to use kunlun_ops."""
    return getattr(torch.ops.xspeedgate_ops, "add_rmsnorm_inplace", None)


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
    """Apply fused residual add + RMSNorm in-place following sgl_kernel API.

    ``xspeedgate_ops.add_rmsnorm_inplace`` writes both outputs in place, so the
    scratch ``out`` buffer and the trailing ``x.copy_(out)`` that
    ``kunlun_ops.add_rmsnorm`` needs are gone. Falls back to the kunlun_ops
    path when the vendor op is unavailable.
    """

    op = _resolve_add_rmsnorm_inplace()
    if op is not None:
        op(x, residual, weight, eps)
        return

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
