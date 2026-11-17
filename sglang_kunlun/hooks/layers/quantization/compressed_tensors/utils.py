"""Shared Kunlun compressed-tensors utilities."""

from __future__ import annotations

import os
from typing import Optional

import torch

INT4_UNSIGNED_SCALE_MULT = 15.0
INT4_SIGNED_SCALE_MULT = 7.0
INT8_SCALE_MULT = 127.0

# DSV4_MOE_PRESORT_TORCH=1 swaps moe_pre_sorted for a torch-native implementation (see
# `_moe_pre_sorted_torch`). Motivation in T022: on the `kunlun_ops` path,
# `infer_moe_ffn_pre_sorted_split_m` goes out of bounds on dspark's verify batch and
# crashes the device, while pre-sort itself is pure index math (sort + gather + prefix
# sum); torch writes it short and fast, which neatly separates "metadata miscomputed"
# from "the GEMM itself is wrong".
_PRESORT_TORCH = os.environ.get("DSV4_MOE_PRESORT_TORCH", "0") not in (
    "0", "", "false", "False"
)

# DSV4_MOE_QUANT_BEFORE_SORT=0 restores quantizing the expanded [m*top_k, n] tensor
# inside `moe_fc`. A token's top_k copies share one per-token INT8 scale, so
# quantizing x ([m, n]) first and sorting the int8 rows reads/writes a third of the
# bytes: the fp16 `sorted_x` (1006 MB at m=32768, n=5120, top_k=6) never has to be
# materialized or read back. Measured on one P800 at that shape: 3.54 ms -> 1.41 ms
# for pre-sort + quantize together, bit-identical int8 payload and scales.
_QUANT_BEFORE_SORT = os.environ.get("DSV4_MOE_QUANT_BEFORE_SORT", "1") not in (
    "0", "", "false", "False"
)

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
    "quantize_before_sort_enabled",
    "scale_to_kernel_max",
    "sorted_row_scales",
    "validate_int4_moe_shapes",
]


def quantize_before_sort_enabled() -> bool:
    """Whether callers should quantize activations before the MoE expand."""
    return _QUANT_BEFORE_SORT


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
    if x.shape[0] == 0:
        return x_int8, x_scale
    kunlun_ops.quant2d(x=x, y=x_int8, max=x_scale, force_sdnn=True)
    return x_int8, x_scale


def sorted_row_scales(
    x_scale: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Place per-token activation scales in `moe_pre_sorted` row order.

    `sorted_tokens_idx` maps a (token, top-k) lane to its row in the sorted tensor,
    so the per-token scale is scattered along the same permutation the rows took.
    Under EP the lanes owned by another rank carry index -1; they are routed to one
    extra slot that is dropped afterwards, which keeps the scatter free of any host
    sync and stops -1 from aliasing row 0.
    """
    nk = sorted_tokens_idx.numel()
    lane_scale = (
        x_scale.reshape(-1, 1).expand(-1, top_k).reshape(-1).to(torch.float32)
    )
    buf = torch.zeros(nk + 1, dtype=torch.float32, device=x_scale.device)
    pos = sorted_tokens_idx.to(torch.int64)
    buf.scatter_(0, torch.where(pos >= 0, pos, torch.full_like(pos, nk)), lane_scale)
    return buf[:nk].reshape(nk, 1)


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


def _moe_pre_sorted_torch(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    *,
    index_have_neg: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch-native pre-sort with the same contract as `kunlun_ops.moe_pre_sorted`.

    Semantics of the three return values (inferred from `moe_post`; see the docstring there):
      * `sorted_x[r]`            -- the hidden of the token corresponding to sorted row r
      * `sorted_tokens_idx[i*K+k]` -- which sorted row lane (token i, expert k) lands in;
                                      **-1** when routed to another EP rank
      * `sorted_tokens_num_lod`  -- a length-E+1 prefix sum; local expert e occupies
                                    the row range `[lod[e], lod[e+1])`

    The implementation sets the key of an "external lane" to `num_experts` so it sorts
    last, thus local rows naturally occupy `[0, lod[E])`; the trailing rows belong to no
    expert and the GEMM will not read them (they are dirty just like the Kunlun kernel;
    we also do not zero them here, keeping behavior consistent).

    No host sync anywhere (no `.item()`/`nonzero()`); shapes depend only on
    `topk_ids.shape`, so this can go into a CUDA graph.
    """
    n, k = topk_ids.shape
    nk = n * k
    dev = x.device
    # The chosen key is deliberately kept int32 and uses a same-shape masked_fill: on
    # Kunlun, `torch.where` with a 0-dim other goes through the broadcast kernel, and the
    # int64 broadcast of a long prefill (16384 tokens x top-k) times out the device task
    # (KL_XID_KERNEL_EXCEPTION, where.cpp -> broadcast_calc_common). Semantics are
    # identical to the original where.
    flat = topk_ids.reshape(-1).to(torch.int32)
    if index_have_neg:
        owned = flat >= 0
        key = flat.masked_fill(~owned, num_experts).to(torch.int64)
    else:
        owned = None
        key = flat.to(torch.int64)
    # stable keeps the original lane order within an expert; row order itself does not
    # affect the final result (moe_post reads back via sorted_tokens_idx), but a stable
    # sort makes A/B comparison easier to inspect.
    order = torch.argsort(key, stable=True)
    lane_token = torch.arange(n, device=dev).repeat_interleave(k)
    sorted_x = x.index_select(0, lane_token.index_select(0, order)).contiguous()

    # Inverse permutation: lane -> sorted row number
    pos = torch.empty(nk, dtype=torch.int32, device=dev)
    pos.scatter_(0, order, torch.arange(nk, dtype=torch.int32, device=dev))
    if owned is not None:
        pos = pos.masked_fill(owned.logical_not(), -1)

    # Row counts per key -> take the prefix sum of only the first num_experts buckets
    # (bucket num_experts is external lanes)
    counts = torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
    counts.scatter_add_(0, key, torch.ones(nk, dtype=torch.int32, device=dev))
    lod = torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
    lod[1:] = counts[:num_experts].cumsum(0).to(torch.int32)
    return sorted_x, pos, lod


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

    if _PRESORT_TORCH:
        return _moe_pre_sorted_torch(
            x, topk_ids, num_experts, index_have_neg=index_have_neg
        )
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
    x_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a sorted Kunlun MoE fully connected operation.

    `x_scale` lets the caller hand in an already INT8-quantized `x` together with
    its per-token scales in sorted-row order (see `sorted_row_scales`); the GEMM
    still runs the grouped-INT8 path, only the local quantize step is skipped.
    """
    import kunlun_ops

    if moe_topk is None:
        moe_topk = topk_ids.shape[1]
    if x_scale is not None:
        assert quantize_input, "x_scale is only meaningful on the grouped-INT8 path"
        assert x.dtype == torch.int8, "x_scale implies a pre-quantized INT8 x"
        x_scale = x_scale.reshape(x_scale.shape[0], -1)
    elif quantize_input:
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


_LANE_IDS: dict = {}


def _lane_ids(like: torch.Tensor) -> torch.Tensor:
    """Cached 0..numel-1 laid out like ``like``, used to spread masked lanes."""
    key = (like.numel(), like.device, like.dtype)
    buf = _LANE_IDS.get(key)
    if buf is None:
        buf = torch.arange(like.numel(), device=like.device, dtype=like.dtype)
        _LANE_IDS[key] = buf
    return buf.view_as(like)


def moe_post(
    down: torch.Tensor,
    sorted_tokens_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    output_shape: torch.Size,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Combine sorted expert outputs with router weights.

    The two dtype constraints come from kunlun_ops.moe_post: normed_scale must be "same
    as x or fp32", and y must be the same as x. On the int8 grouped path moe_fc always
    produces `down` in bf16 (see kernel_dtype above); then if the model is fp16,
    topk_weights is fp16, and the bf16 + fp16 combination is rejected by the op with
    "moe_post not support x/normed_scale dtype". So normed_scale is uniformly promoted to
    fp32 (measured legal for both bf16 and fp16 x), and the output is cast back to the
    model dtype per output_dtype.

    Under EP, lanes (token, top-k) routed to another rank have `sorted_tokens_idx` = -1 and
    no corresponding row written in `down` (`moe_fc` allocates with `torch.empty`).
    `kunlun_ops.moe_post` does not skip those lanes, so each rank multiplies uninitialized
    memory by the real router weights and adds it in, and the EP all-reduce then accumulates
    8 copies -- measured routed contribution 7x too large, cos 0.48 vs the reference (shared
    expert cos 0.9999, single-op `moe_fc` cos 0.99996, both correct).

    So mask external lanes out here: zero the weights and clamp the index to 0 so negative
    rows are never addressed.
    """
    import kunlun_ops

    out_dtype = down.dtype if output_dtype is None else output_dtype
    output = torch.empty(output_shape, dtype=down.dtype, device=down.device)
    moe_index = sorted_tokens_idx.view_as(topk_weights)
    owned = moe_index >= 0
    # Unconditional elementwise ops, no host sync, safe for CUDA graph capture.
    normed_scale = topk_weights.to(torch.float32) * owned
    # Masked lanes are multiplied by 0, so which row they read cannot change the
    # result -- but it does change the cost. clamp_min(0) sent all of them to row
    # 0: on P800 that turns 86016 of the 98304 gathers (EP8) into requests for one
    # 10 KB address, which serialises on the few memory channels that address maps
    # to. Measured at m=16384, top_k=6, n=5120: 2.12 ms clamped vs 0.77 ms spread,
    # i.e. a masked lane costs 3x a real one purely because of the collision.
    #
    # They must stay inside [0, n_written) though. `moe_fc` allocates `down` with
    # torch.empty and writes only the owned rows, so the tail can hold NaN/Inf bit
    # patterns and 0 * NaN = NaN: spreading over the whole buffer poisoned 87.5% of
    # the output (measured) and killed the service downstream. owned.sum() is
    # exactly the number of written rows and stays on device, so no host sync and
    # the expression is still CUDA-graph capturable. Verified bit-identical to
    # clamp_min(0) with a NaN-filled tail.
    n_written = owned.sum().clamp_min(1).to(moe_index.dtype)
    moe_index = torch.where(owned, moe_index, _lane_ids(moe_index) % n_written)
    dequant_scale = torch.ones_like(topk_weights, dtype=torch.float32)
    kunlun_ops.moe_post(
        x=down.reshape(-1, down.shape[-1]),
        moe_index=moe_index,
        normed_scale=normed_scale,
        dequant_scale=dequant_scale,
        y=output,
    )
    if output.dtype is not out_dtype:
        output = output.to(out_dtype)
    return output
