"""Kunlun compressed-tensors W8A8 linear scheme."""

from __future__ import annotations

from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)

from ..utils import INT8_SCALE_MULT, dynamic_quantize_int8, scale_to_kernel_max

__all__ = ["KunlunCompressedTensorsW8A8Int8"]


class KunlunCompressedTensorsW8A8Int8(CompressedTensorsW8A8Int8):
    """Dynamic per-token W8A8 scheme backed by Kunlun INT8 matmul."""

    def __init__(
        self,
        strategy: str,
        is_static_input_scheme: bool,
        input_symmetric: bool,
    ) -> None:
        """Initialize and validate the Kunlun W8A8 scheme."""
        super().__init__(strategy, is_static_input_scheme, input_symmetric)
        strategy_value = (
            strategy.value if isinstance(strategy, QuantizationStrategy) else strategy
        )
        if (
            strategy_value != QuantizationStrategy.CHANNEL.value
            or is_static_input_scheme
            or not input_symmetric
        ):
            raise NotImplementedError(
                "Kunlun compressed-tensors W8A8 linear requires dynamic "
                "per-token activations and symmetric per-channel weights."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        """Return the minimum device capability required by the scheme."""
        return 0

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare INT8 weights and convert scales to integer-domain maxima."""
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            scale_to_kernel_max(layer.weight_scale.data, INT8_SCALE_MULT),
            requires_grad=False,
        )
        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply dynamic INT8 quantization and Kunlun matmul."""
        import kunlun_ops

        if isinstance(x, tuple):
            x_q, x_scale = x
            output_dtype = torch.bfloat16
        else:
            output_dtype = x.dtype
            x_q, x_scale = dynamic_quantize_int8(x, keepdim=False)

        output_size = layer.weight.shape[0]
        out = torch.empty(
            (x_q.shape[0], output_size),
            dtype=output_dtype,
            device=x_q.device,
        )
        kunlun_ops.matmul(
            x_q,
            layer.weight.data,
            out,
            bias=bias.to(torch.float32) if bias is not None else None,
            x_pc_max=x_scale,
            w_pc_max=layer.weight_scale.data,
        )
        return out
