"""Kunlun-backed implementations for selected ``sgl_kernel.elementwise`` APIs."""

from __future__ import annotations

from typing import Optional

import torch


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert (
        input.shape[:-1] == output.shape[:-1]
    ), f"{input.shape[:-1]} != {output.shape[:-1]}"
    assert (
        input.shape[-1] == 2 * output.shape[-1]
    ), f"{input.shape[-1]} != {2 * output.shape[-1]}"


def silu_and_mul(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply SiLU to the first half of input and multiply with the second half."""
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )

    import kunlun_ops

    kunlun_ops.swiglu(
        x=input,
        y=out,
    )
    return out
