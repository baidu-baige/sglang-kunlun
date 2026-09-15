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


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply Gemma RMSNorm and return a new output tensor with the input shape."""

    import kunlun_ops

    input_for_norm = x.contiguous()
    normalized = torch.empty_like(input_for_norm)
    kunlun_ops.gemma_rmsnorm(
        input_for_norm,
        weight.contiguous(),
        normalized,
        eps,
    )
    return normalized


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """Add residual and apply Gemma RMSNorm in-place following sgl_kernel API."""

    import kunlun_ops

    # MiniMax 的 hidden states 和 residual 在 prefill、decode 及graph replay中均为连续张量。
    # 如果上游布局变化破坏该约定，则直接报错；算子的原地别名不支持带stride的视图。
    if not x.is_contiguous() or not residual.is_contiguous():
        raise ValueError(
            "Kunlun gemma_fused_add_rmsnorm requires contiguous x and residual, "
            f"got x.shape={tuple(x.shape)}, x.stride={x.stride()}, "
            f"residual.shape={tuple(residual.shape)}, "
            f"residual.stride={residual.stride()}"
        )

    kunlun_ops.gemma_add_rmsnorm(
        x,
        residual,
        weight.contiguous(),
        x,
        eps,
        residual_output=residual,
        store_output_before_norm=True,
    )
