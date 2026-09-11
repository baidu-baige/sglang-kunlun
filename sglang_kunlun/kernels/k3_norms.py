"""Torch-native K3 normalization primitives."""

from __future__ import annotations

from typing import Optional

import torch


def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    """RMSNorm with FP32 statistics and input-dtype output."""
    orig_dtype = x.dtype
    values = x.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    values = values * torch.rsqrt(variance + eps)
    if weight is not None:
        values = values * weight.float()
    return values.to(orig_dtype)


def rms_norm_with_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    post_residual_addition: Optional[torch.Tensor] = None,
):
    """Return RMSNorm(x + residual) and the updated residual."""
    merged = x.float() + residual.float()
    if post_residual_addition is not None:
        merged = merged + post_residual_addition.float()
    residual_out = merged.to(residual.dtype)
    return rms_norm(merged.to(x.dtype), weight, eps), residual_out


def l2_norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Normalize each final-dimension vector using FP32 accumulation."""
    orig_dtype = x.dtype
    values = x.float()
    result = values * torch.rsqrt(values.square().sum(dim=-1, keepdim=True) + eps)
    return result.to(output_dtype or orig_dtype)
