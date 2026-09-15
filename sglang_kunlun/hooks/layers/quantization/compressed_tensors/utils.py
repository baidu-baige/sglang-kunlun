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
"""Shared Kunlun compressed-tensors utilities."""

from __future__ import annotations

from typing import Optional

import torch

INT4_UNSIGNED_SCALE_MULT = 15.0
INT4_SIGNED_SCALE_MULT = 7.0
INT8_SCALE_MULT = 127.0

__all__ = [
    "INT4_SIGNED_SCALE_MULT",
    "INT4_UNSIGNED_SCALE_MULT",
    "INT8_SCALE_MULT",
    "check_standard_routed_moe_layer",
    "dequant_int4",
    "dequant_int4_moe",
    "dequant_int4_to_int8",
    "dynamic_quantize_int8",
    "moe_fc",
    "moe_post",
    "moe_pre_sorted",
    "scale_to_kernel_max",
    "validate_int4_moe_shapes",
]


def scale_to_kernel_max(
    scale: torch.Tensor,
    multiplier: float,
) -> torch.Tensor:
    """Convert a checkpoint quantization scale to a kernel maximum value."""
    return scale.to(torch.float32).mul(multiplier)


def check_standard_routed_moe_layer(layer: torch.nn.Module) -> None:
    """Validate assumptions shared by the sorted Kunlun MoE implementations."""
    if getattr(layer, "num_fused_shared_experts", 0):
        raise NotImplementedError(
            "Kunlun quantized MoE only supports routed experts. Disable shared "
            "expert fusion when shared experts use a different quantization."
        )
    if getattr(layer, "use_triton_kernels", False):
        raise NotImplementedError(
            "Kunlun quantized MoE requires standard top-k dispatch and does not "
            "support the triton-kernel MoE runner backend."
        )


def dynamic_quantize_int8(
    x: torch.Tensor,
    *,
    keepdim: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize activations to per-token INT8 values."""
    import kunlun_ops

    x = x.contiguous()
    x_int8 = torch.empty_like(x, dtype=torch.int8)
    scale_shape = (x.shape[0], 1) if keepdim else (x.shape[0],)
    x_scale = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
    kunlun_ops.quant2d(x=x, y=x_int8, max=x_scale, force_sdnn=True)
    return x_int8, x_scale


def dequant_int4_to_int8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert uncompressed group-wise INT4 weights to per-channel INT8."""
    import kunlun_ops

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "INT4 weight and scale must be two-dimensional: "
            f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}."
        )
    if weight.dtype != torch.int8:
        raise TypeError(f"INT4 weight must use torch.int8 storage, got {weight.dtype}.")

    output_size, input_size = weight.shape
    if group_size <= 0 or input_size % group_size != 0:
        raise ValueError(
            "INT4 input size must be divisible by group_size: "
            f"input_size={input_size}, group_size={group_size}."
        )
    expected_scale_shape = (output_size, input_size // group_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Invalid INT4 scale shape {tuple(scale.shape)}; "
            f"expected {expected_scale_shape}."
        )

    dequant_weight = (
        weight.reshape(output_size, -1, group_size).to(torch.float16)
        * scale.to(torch.float16).unsqueeze(-1)
    ).reshape(output_size, input_size).contiguous()
    weight_int8 = torch.empty_like(weight, dtype=torch.int8)
    weight_max = torch.empty(output_size, dtype=torch.float32, device=weight.device)
    kunlun_ops.quant2d(
        x=dequant_weight,
        y=weight_int8,
        max=weight_max,
        force_sdnn=True,
    )
    return weight_int8, weight_max


def dequant_int4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize packed symmetric INT4 weights with the Kunlun kernel."""
    import kunlun_ops

    if weight_packed.ndim != 2:
        raise ValueError(
            f"W4A16 packed weight must be two-dimensional, got {weight_packed.ndim}."
        )
    if weight_packed.dtype != torch.int32:
        raise TypeError(
            f"W4A16 packed weight must use torch.int32, got {weight_packed.dtype}."
        )
    output_size = weight_packed.shape[0]
    input_size = weight_packed.shape[1] * 8
    if weight_scale.ndim != 2 or weight_scale.shape[0] != output_size:
        raise ValueError(
            "W4A16 weight and scale shapes are incompatible: "
            f"weight={tuple(weight_packed.shape)}, "
            f"scale={tuple(weight_scale.shape)}."
        )

    num_groups = weight_scale.shape[1]
    if num_groups == 0 or input_size % num_groups != 0:
        raise ValueError(
            "W4A16 input size must be divisible by the scale group count: "
            f"input_size={input_size}, num_groups={num_groups}."
        )
    group_size = input_size // num_groups
    kernel_scale = scale_to_kernel_max(
        weight_scale.repeat_interleave(group_size, dim=1),
        INT4_UNSIGNED_SCALE_MULT,
    )
    weight = torch.empty(
        (output_size, input_size),
        dtype=torch.float16,
        device=weight_packed.device,
    )
    kunlun_ops.dequant_int4(
        weight_packed.contiguous().view(torch.int8),
        kernel_scale,
        None,
        weight,
        1,
        int4_signed=False,
        use_mode_fast=False,
    )
    return weight


def validate_int4_moe_shapes(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Validate packed W4A16 expert tensors and return their dimensions."""
    if weight_packed.ndim != 3 or weight_scale.ndim != 3:
        raise ValueError(
            "W4A16 MoE weight and scale must be three-dimensional: "
            f"weight={tuple(weight_packed.shape)}, "
            f"scale={tuple(weight_scale.shape)}."
        )
    if weight_packed.dtype != torch.int32:
        raise TypeError(
            f"W4A16 MoE packed weight must use torch.int32, got {weight_packed.dtype}."
        )

    num_experts, packed_input_size, output_size = weight_packed.shape
    scale_experts, num_groups, scale_output_size = weight_scale.shape
    if (scale_experts, scale_output_size) != (num_experts, output_size):
        raise ValueError(
            "W4A16 MoE weight and scale shapes are incompatible: "
            f"weight={tuple(weight_packed.shape)}, "
            f"scale={tuple(weight_scale.shape)}."
        )
    input_size = packed_input_size * 8
    if num_groups == 0 or input_size % num_groups != 0:
        raise ValueError(
            "W4A16 MoE input size must be divisible by the scale group count: "
            f"input_size={input_size}, num_groups={num_groups}."
        )
    return num_experts, output_size, packed_input_size, num_groups


def dequant_int4_moe(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize packed expert INT4 weights with the Kunlun kernel."""
    num_experts, output_size, packed_input_size, num_groups = (
        validate_int4_moe_shapes(weight_packed, weight_scale)
    )
    packed = weight_packed.transpose(1, 2).contiguous()
    scales = weight_scale.transpose(1, 2).contiguous()
    weight = dequant_int4(
        packed.reshape(num_experts * output_size, packed_input_size),
        scales.reshape(num_experts * output_size, num_groups),
    )
    return weight.reshape(num_experts, output_size, packed_input_size * 8)


def moe_pre_sorted(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    *,
    index_have_neg: bool,
    block_statistic: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort routed tokens and return kernel metadata for expert projections."""
    import kunlun_ops

    top_k = topk_ids.shape[1]
    if block_statistic is None or block_statistic.shape != (12, num_experts):
        block_statistic = torch.empty(
            (12, num_experts), dtype=torch.int32, device=x.device
        )
        kunlun_ops.gen_block_statistic(topk_ids, block_statistic)
    sorted_x = torch.empty(
        (x.shape[0] * top_k, x.shape[1]), dtype=x.dtype, device=x.device
    )
    sorted_tokens_idx = torch.zeros(
        x.shape[0] * top_k, dtype=torch.int32, device=x.device
    )
    expert_m = torch.zeros(num_experts, dtype=torch.int32, device=x.device)
    sorted_tokens_num_lod = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=x.device
    )
    kunlun_ops.moe_pre_sorted(
        x=x,
        topk_index=topk_ids,
        block_statistic=block_statistic,
        moe_expand=sorted_x,
        moe_index=sorted_tokens_idx,
        expert_m=expert_m,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        index_have_neg=index_have_neg,
    )
    return sorted_x, sorted_tokens_idx, sorted_tokens_num_lod


def moe_fc(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor],
    sorted_tokens_num_lod: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    output_shape: tuple[int, ...],
    output_dtype: torch.dtype,
    *,
    quantize_input: bool,
    packed_int4: bool = False,
    moe_topk: Optional[int] = None,
) -> torch.Tensor:
    """Run a sorted Kunlun MoE fully connected operation."""
    import kunlun_ops

    if moe_topk is None:
        moe_topk = topk_ids.shape[1]
    x_scale = None
    if quantize_input:
        x, x_scale = dynamic_quantize_int8(x, keepdim=True)
        x_scale = x_scale.reshape(x_scale.shape[0], -1)
    use_grouped_int8 = quantize_input
    kernel_dtype = torch.bfloat16 if use_grouped_int8 else output_dtype
    output = torch.empty(output_shape, dtype=kernel_dtype, device=x.device)
    y = output.reshape(-1, output.shape[-1])
    kunlun_ops.moe_fc_v3(
        x=x,
        weight=weight,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=1 if use_grouped_int8 else moe_topk,
        y=y,
        x_perchannel_max=x_scale,
        w_perchannel_max=weight_scale,
        tgemm_type="int8_wo_t" if use_grouped_int8 else None,
        use_pack_int4=packed_int4,
        sort_mode=True,
    )
    return output


def moe_post(
    down: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    output_shape: torch.Size,
) -> torch.Tensor:
    """Combine sorted expert outputs with router weights."""
    import kunlun_ops

    output = torch.empty(output_shape, dtype=down.dtype, device=down.device)
    dequant_scale = torch.ones_like(topk_weights, dtype=torch.float32)
    kunlun_ops.moe_post(
        x=down.reshape(-1, down.shape[-1]),
        moe_index=sorted_tokens_idx.view_as(topk_weights),
        normed_scale=topk_weights,
        dequant_scale=dequant_scale,
        y=output,
    )
    return output
