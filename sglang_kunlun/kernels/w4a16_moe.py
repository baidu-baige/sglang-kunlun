"""Torch-native W4A16 reference helpers for Kunlun Kimi-K3.

These helpers are the correctness oracle for ``K3-OP-002``. They operate on
packed INT32 tensors on the active XPU and only unpack the requested expert or
weight, so the validation path does not materialize all 896 experts in BF16.

The checkpoint uses eight signed 4-bit values per INT32 word. The low nibble is
read first and converted from two's-complement INT4 before applying one scale
per output channel. The nibble convention is intentionally isolated here so
an XSpeedGate layout adapter can be compared against the same oracle.
"""

from __future__ import annotations

from typing import Optional

import torch

PACK_FACTOR = 8
INT4_MASK = 0xF


def unpack_int4_weight(packed: torch.Tensor, logical_k: Optional[int] = None) -> torch.Tensor:
    """Unpack ``[..., K // 8]`` int32 words to ``[..., K]`` signed int4 values."""
    # compressed-tensors symmetric int4 stores ``q + INT4_ZERO_POINT``.
    INT4_ZERO_POINT = 8
    if packed.dtype != torch.int32:
        raise TypeError(f"packed INT4 weights must be int32, got {packed.dtype}")
    words = packed.to(torch.int64).unsqueeze(-1)
    shifts = torch.arange(PACK_FACTOR, device=packed.device, dtype=torch.int64)
    values = ((words >> (shifts * 4)) & INT4_MASK).to(torch.int32) - INT4_ZERO_POINT
    values = values.reshape(*packed.shape[:-1], packed.shape[-1] * PACK_FACTOR)
    if logical_k is not None:
        if logical_k < 0 or logical_k > values.shape[-1]:
            raise ValueError(
                f"logical_k={logical_k} is invalid for unpacked width {values.shape[-1]}"
            )
        values = values[..., :logical_k]
    return values


def dequantize_int4_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    logical_k: Optional[int] = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Return an output-channel-major dequantized weight."""
    values = unpack_int4_weight(packed, logical_k=logical_k).to(torch.float32)
    if scales.ndim == values.ndim - 1:
        scales = scales.unsqueeze(-1)
    if scales.shape[:-1] != values.shape[:-1]:
        raise ValueError(
            f"scale shape {tuple(scales.shape)} does not match weight shape "
            f"{tuple(values.shape)}"
        )
    if scales.shape[-1] not in (1, values.shape[-1]):
        raise ValueError(
            f"scale last dimension must be 1 or K={values.shape[-1]}, "
            f"got {scales.shape[-1]}"
        )
    return (values * scales.to(device=values.device, dtype=torch.float32)).to(
        output_dtype
    )


def w4a16_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    logical_k: Optional[int] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute ``x @ dequantized_weight.T`` for one expert."""
    if x.shape[-1] != (logical_k or packed.shape[-1] * PACK_FACTOR):
        raise ValueError(
            f"activation width {x.shape[-1]} does not match packed K "
            f"{logical_k or packed.shape[-1] * PACK_FACTOR}"
        )
    weight = dequantize_int4_weight(
        packed, scales, logical_k=logical_k, output_dtype=torch.float32
    )
    result = torch.matmul(x.to(torch.float32), weight.transpose(-1, -2))
    return result.to(output_dtype or x.dtype)
