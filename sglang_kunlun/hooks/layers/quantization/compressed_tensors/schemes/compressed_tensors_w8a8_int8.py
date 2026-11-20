# Copyright 2026 Baidu, Inc.
# Licensed under the Apache License, Version 2.0.
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
"""Kunlun compressed-tensors W8A8 linear scheme."""

from __future__ import annotations

import os
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)

from ..utils import INT8_SCALE_MULT, dynamic_quantize_int8, scale_to_kernel_max

__all__ = ["KunlunCompressedTensorsW8A8Int8"]

# [r141] wq_b emits fp16 directly for prefill rows.
#
# wq_b's output *is* q, and the very next consumer of q is the attention backend,
# whose window/compressed cache is fp16: today the int8 GEMM writes bf16 and
# `_kunlun_compressed_attention` immediately casts it back (`xspeedgate_ops::
# cast_bf16_fp16`, 122 calls / 27.96 ms per 2-forward capture, 586 us per layer
# at [7680,64,512] = 1.72 TB/s, i.e. at the copy roof -- r70 closed only the
# consumer side).  The same `kunlun_ops.matmul` accepts an fp16 destination and
# is faster with one: [7680,1280]x[32768,1280] int8 -> bf16 1501 us, -> fp16
# 1052 us, -> bf16 again 1505 us (30% reproducible), and rope/dlpack both take
# fp16 (`flashinfer_rotary_embedding` verified fp16 in-place, and it is *more*
# accurate: 1.85e-3 vs 1.53e-2 against an fp32 reference).  Net ~21 layers x
# (586 us cast + ~450 us GEMM) ~= 20 ms/forward.
# Numerics: strictly better.  The fp32 accumulator now rounds once (to fp16)
# instead of twice (bf16, then fp16 in the cast); no new overflow surface,
# because the removed cast already forced q into fp16 range.
# Scope: prefill/extend rows only (decode keeps its static bf16 q_out buffer and
# its cuda-graph contract), and only for wq_b's own shape; every other W8A8
# linear (wq_a [1280,5120], wo_b [5120,8192], ...) is untouched.
# Rollback: KL_WQB_FP16=0 (module-level env, takes effect on cold start).
_WQB_FP16 = os.environ.get("KL_WQB_FP16", "1") != "0"
_WQB_WEIGHT_SHAPE = (32768, 1280)   # n_local_heads * head_dim, q_lora_rank
_WQB_MIN_ROWS = 64                  # extend/prefill only
_WQB_LOGGED = False


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

        global _WQB_LOGGED

        # An idle/empty batch (zero rows) must not reach the Kunlun matmul: the
        # backend does not promise a zero-row shape. Return the empty result.
        inputs = x[0] if isinstance(x, tuple) else x
        if inputs.shape[0] == 0:
            return torch.empty(
                (0, layer.weight.shape[0]),
                dtype=torch.bfloat16 if isinstance(x, tuple) else x.dtype,
                device=inputs.device,
            )

        if isinstance(x, tuple):
            x_q, x_scale = x
            output_dtype = torch.bfloat16
        else:
            output_dtype = x.dtype
            x_q, x_scale = dynamic_quantize_int8(x, keepdim=False)

        output_size = layer.weight.shape[0]
        if (
            _WQB_FP16
            and x_q.shape[0] >= _WQB_MIN_ROWS
            and tuple(layer.weight.shape) == _WQB_WEIGHT_SHAPE
        ):
            output_dtype = torch.float16
            if not _WQB_LOGGED:
                _WQB_LOGGED = True
                print("[r141] wq_b fp16-out active rows=%d" % x_q.shape[0],
                      flush=True)
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
