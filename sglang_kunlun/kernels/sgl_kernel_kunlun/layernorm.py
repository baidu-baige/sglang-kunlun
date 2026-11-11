# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Kunlun-backed implementations of layernorm ``sgl_kernel`` APIs."""

from __future__ import annotations

import torch


# ``weight.to(x.dtype)`` for a norm weight is a *static* conversion: the
# parameter never changes after load, yet on XPU one ``aten::to`` costs ~50-70 us
# of *host* time (74 us measured on this stack) even when it is a no-op, and the
# per-call cast showed up as 158 launches per prefill step in the -w4 trace.
# Cache the converted tensor, keyed on (data_ptr, version, dtype) so an
# in-place load rebuilds instead of reusing stale bytes.
_W_CAST: dict = {}


def _cast_weight(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if weight.dtype == dtype:
        return weight
    key = (weight.data_ptr(), weight._version, dtype)
    hit = _W_CAST.get(key)
    if hit is None or hit.shape != weight.shape:
        hit = weight.to(dtype)
        if len(_W_CAST) > 512:
            _W_CAST.clear()
        _W_CAST[key] = hit
    return hit


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure-torch RMSNorm, dtype-robust (compute in fp32, cast back to x)."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm and return a new output tensor.

    ``kunlun_ops.rmsnorm`` (add_rmsnorm) rejects some x/weight/output dtype
    combinations on the XPU (e.g. a bf16 norm weight against an fp16 hidden
    state on the DSV4.1 hc path). Align the weight to x's dtype first; if the
    native kernel still refuses the combo, fall back to a pure-torch RMSNorm so
    the forward stays functional.
    """

    import kunlun_ops

    w = _cast_weight(weight, x.dtype)
    out = torch.empty_like(x)
    try:
        kunlun_ops.rmsnorm(
            x,
            w,
            out,
            eps,
            False,
            True,
            None,
            None,
            None,
        )
    except RuntimeError:
        return _torch_rmsnorm(x, weight, eps)
    return out


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """Apply fused residual add + RMSNorm in-place following sgl_kernel API."""

    import kunlun_ops

    w = _cast_weight(weight, x.dtype)
    out = torch.empty_like(x)
    try:
        kunlun_ops.add_rmsnorm(
            x,
            residual,
            w,
            out,
            eps,
            False,
            True,
            None,
            None,
            residual,
            None,
        )
    except RuntimeError:
        # dtype/kernel unsupported: torch add+rmsnorm, matching in-place contract
        residual.add_(x)
        x.copy_(_torch_rmsnorm(residual, weight, eps))
        return
    x.copy_(out)
