"""REPLACE W8A8Int8 linear/MoE methods with kunlun_ops-backed forwards.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/quantization/w8a8_int8.py.

The mimo branch installs four monkey patches:
  * ``W8A8Int8LinearMethod.process_weights_after_loading``
  * ``W8A8Int8LinearMethod.apply``
  * ``W8A8Int8MoEMethod.process_weights_after_loading``
  * ``W8A8Int8MoEMethod.apply``
  * ``W8A8Int8Config.get_quant_method``
each of which we expose here as a method-level ``REPLACE`` plugin hook.
"""

from __future__ import annotations

from typing import Any, Optional

import os
import torch
from torch.nn.parameter import Parameter

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
import logging
logger = logging.getLogger(__name__)


def _clamp_fp16_moe_output(output: torch.Tensor) -> torch.Tensor:
    if output.dtype is not torch.float16:
        return output
    limit = int(os.environ.get("SGLANG_FP16_LIMIT_IN_MOE", "10"))
    return output.clamp(min=-limit, max=limit)


# ---------------------------------------------------------------------------
# Linear method
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.quantization.w8a8_int8.W8A8Int8LinearMethod."
    "process_weights_after_loading",
    type=HookType.REPLACE,
)
def linear_process_weights_after_loading_kunlun(
    self, layer: torch.nn.Module
) -> None:
    """Bake the per-channel scale into integer-domain (×127) up front."""
    layer.weight = Parameter(layer.weight.data, requires_grad=False)
    layer.weight_scale = Parameter(
        layer.weight_scale.data * 127, requires_grad=False
    )
    # from debug.dsv4_probe_bridge import dump_selected_linear_parameters

    # dump_selected_linear_parameters(layer)


@plugin_hook(
    "sglang.srt.layers.quantization.w8a8_int8.W8A8Int8LinearMethod.apply",
    type=HookType.REPLACE,
)
def linear_apply_kunlun(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    """W8A8 int8 linear forward backed by ``kunlun_ops.matmul``."""
    import kunlun_ops  # local-only dep

    # from debug.dsv4_probe_bridge import dump_linear_stage

    w_shape = layer.weight.shape
    # dsv4_layer_id = getattr(layer, "_dsv4_layer_id", None)
    # if dsv4_layer_id is not None and not isinstance(x, tuple):
    #     from sglang.srt.models.deepseek_v4 import (
    #         _record_dsv4_decode_stage,
    #         _record_dsv4_decode_static,
    #     )

    #     _record_dsv4_decode_stage(dsv4_layer_id, "wo_b_input", x)
    #     _record_dsv4_decode_static(dsv4_layer_id, "wo_b_weight", layer.weight.data)
    #     _record_dsv4_decode_static(
    #         dsv4_layer_id, "wo_b_weight_scale", layer.weight_scale.data
    #     )
    if isinstance(x, tuple):
        x_q, x_scale = x
        out = torch.empty(
            (x_q.shape[0], w_shape[0]),
            dtype=torch.bfloat16,
            device=x_q.device,
        )
    else:
        x_shape = x.shape
        x_q = torch.empty(x_shape, dtype=torch.int8, device=x.device)
        x_scale = torch.empty(x_shape[0], dtype=torch.float32, device=x.device)
        out = torch.empty(
            (x_shape[0], w_shape[0]), dtype=x.dtype, device=x.device
        )
        # dump_linear_stage(layer, "quant2d.input.x", x)
        kunlun_ops.quant2d(x, x_q, x_scale, force_sdnn=True)
        # dump_linear_stage(layer, "quant2d.output.x_q", x_q)
        # dump_linear_stage(layer, "quant2d.output.x_scale", x_scale)
    # if dsv4_layer_id is not None:
    #     _record_dsv4_decode_stage(dsv4_layer_id, "wo_b_x_q", x_q)
    #     _record_dsv4_decode_stage(dsv4_layer_id, "wo_b_x_scale", x_scale)

    # dump_linear_stage(layer, "matmul.input.x_q", x_q)
    # dump_linear_stage(layer, "matmul.input.weight", layer.weight.data)
    # dump_linear_stage(layer, "matmul.input.x_pc_max", x_scale)
    # dump_linear_stage(layer, "matmul.input.w_pc_max", layer.weight_scale.data)
    kunlun_ops.matmul(
        x_q,
        layer.weight.data,
        out,
        bias=bias,
        x_pc_max=x_scale,
        w_pc_max=layer.weight_scale.data,
    )
    # dump_linear_stage(layer, "matmul.output", out)
    # if dsv4_layer_id is not None:
    #     _record_dsv4_decode_stage(dsv4_layer_id, "wo_b_matmul_output", out)
    return out


# ---------------------------------------------------------------------------
# MoE method
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.quantization.w8a8_int8.W8A8Int8MoEMethod."
    "process_weights_after_loading",
    type=HookType.REPLACE,
)
def moe_process_weights_after_loading_kunlun(
    self, layer: torch.nn.Module
) -> None:
    """Bake the per-channel scale into integer-domain (×127) for MoE."""
    layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
    layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
    layer.w13_weight_scale = Parameter(
        layer.w13_weight_scale.data * 127, requires_grad=False
    )
    layer.w2_weight_scale = Parameter(
        layer.w2_weight_scale.data * 127, requires_grad=False
    )


@plugin_hook(
    "sglang.srt.layers.quantization.w8a8_int8.W8A8Int8MoEMethod.apply",
    type=HookType.REPLACE,
)
def moe_apply_kunlun(
    self,
    layer: torch.nn.Module,
    dispatch_output: "Any",
):
    """W8A8 int8 fused-MoE forward backed by ``kunlun_ops``."""
    import kunlun_ops  # local-only dep
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    from debug.dsv4_probe_bridge import (
        dump_selected_moe_rows,
        dump_selected_moe_tensor,
    )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids
    top_k = self.moe_runner_config.top_k
    num_experts = self.moe_runner_config.num_experts
    num_tokens, hidden_size = hidden_states.shape
    hidden_dim = layer.w2_weight.shape[1]
    device = hidden_states.device
    dump_selected_moe_rows(
        layer, "input.hidden_states", hidden_states, num_tokens, top_k
    )
    dump_selected_moe_rows(layer, "input.topk_ids", topk_ids, num_tokens, top_k)
    dump_selected_moe_rows(
        layer, "input.topk_weights", topk_weights, num_tokens, top_k
    )
    # Keep the exact preprocessing contract: expand token rows first,
    # then quantize the M * top_k matrix consumed by both grouped GEMMs.
    block_statistic = torch.zeros(
        12, num_experts, dtype=torch.int32, device=device
    )
    kunlun_ops.gen_block_statistic(topk_ids, block_statistic)
    moe_expand = torch.empty(
        num_tokens * top_k,
        hidden_size,
        dtype=hidden_states.dtype,
        device=device,
    )
    expert_m = torch.zeros(num_experts, dtype=torch.int32, device=device)
    sorted_tokens_num_lod = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=device
    )
    sorted_tokens_idx = torch.zeros(
        num_tokens * top_k, dtype=torch.int32, device=device
    )
    kunlun_ops.moe_pre_sorted(
        hidden_states,
        topk_ids,
        block_statistic,
        moe_expand,
        sorted_tokens_idx,
        expert_m,
        sorted_tokens_num_lod,
    )
    dump_selected_moe_tensor(layer, "pre_sort.block_statistic", block_statistic)
    dump_selected_moe_tensor(layer, "pre_sort.moe_expand", moe_expand)
    dump_selected_moe_tensor(layer, "pre_sort.sorted_tokens_idx", sorted_tokens_idx)
    dump_selected_moe_tensor(layer, "pre_sort.expert_m", expert_m)
    dump_selected_moe_tensor(
        layer, "pre_sort.sorted_tokens_num_lod", sorted_tokens_num_lod
    )
    x_q = torch.empty_like(moe_expand, dtype=torch.int8)
    x_scale = torch.empty(
        (num_tokens * top_k, 1), dtype=torch.float32, device=device
    )
    kunlun_ops.quant2d(moe_expand, x_q, x_scale, force_sdnn=True)
    dump_selected_moe_rows(layer, "fc1.x_q", x_q, num_tokens, top_k)
    dump_selected_moe_rows(layer, "fc1.x_scale", x_scale, num_tokens, top_k)
    gate_up = torch.empty(
        num_tokens,
        top_k,
        layer.w13_weight.shape[1],
        dtype=hidden_states.dtype,
        device=device,
    )
    kunlun_ops.moe_fc(
        x=x_q,
        x_perchannel_max=x_scale,
        weight=layer.w13_weight,
        w_perchannel_max=layer.w13_weight_scale,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=top_k,
        y=gate_up,
        topk_ids=topk_ids,
        act=None,
    )
    dump_selected_moe_rows(layer, "fc1.gate_up_raw", gate_up, num_tokens, top_k)
    swiglu_limit = self.moe_runner_config.swiglu_limit
    if swiglu_limit is not None:
        half = gate_up.shape[-1] // 2
        gate_up[..., :half].clamp_(max=float(swiglu_limit))
        gate_up[..., half:].clamp_(
            min=-float(swiglu_limit), max=float(swiglu_limit)
        )
    dump_selected_moe_rows(
        layer, "fc1.gate_up_clamped", gate_up, num_tokens, top_k
    )
    activated = torch.empty(
        *gate_up.shape[:-1],
        gate_up.shape[-1] // 2,
        dtype=gate_up.dtype,
        device=device,
    )
    kunlun_ops.swiglu(gate_up, activated)
    dump_selected_moe_rows(layer, "swiglu.output", activated, num_tokens, top_k)
    activated = activated.reshape(num_tokens * top_k, -1)
    activated_q = torch.empty_like(activated, dtype=torch.int8)
    activated_scale = torch.empty(
        (num_tokens * top_k, 1), dtype=torch.float32, device=device
    )
    kunlun_ops.quant2d(
        activated, activated_q, activated_scale, force_sdnn=True
    )
    dump_selected_moe_rows(
        layer, "fc2.activated_q", activated_q, num_tokens, top_k
    )
    dump_selected_moe_rows(
        layer, "fc2.activated_scale", activated_scale, num_tokens, top_k
    )
    expert_output = torch.empty(
        num_tokens,
        top_k,
        hidden_dim,
        dtype=hidden_states.dtype,
        device=device,
    )
    kunlun_ops.moe_fc(
        x=activated_q,
        x_perchannel_max=activated_scale,
        weight=layer.w2_weight,
        w_perchannel_max=layer.w2_weight_scale,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=top_k,
        y=expert_output,
        topk_ids=topk_ids,
        act=None,
    )
    dump_selected_moe_rows(
        layer, "fc2.expert_output", expert_output, num_tokens, top_k
    )
    dequant_scale = torch.ones(
        (num_tokens, top_k), dtype=torch.float32, device=device
    )
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=hidden_states.dtype, device=device
    )
    kunlun_ops.moe_post(
        expert_output,
        sorted_tokens_idx.view(num_tokens, top_k),
        topk_weights,
        dequant_scale,
        output,
    )
    dump_selected_moe_rows(layer, "post.output_raw", output, num_tokens, top_k)
    routed_scaling_factor = self.moe_runner_config.routed_scaling_factor
    if routed_scaling_factor not in (None, 1.0):
        output.mul_(routed_scaling_factor)
    dump_selected_moe_rows(layer, "post.output_scaled", output, num_tokens, top_k)
    output = _clamp_fp16_moe_output(output)
    dump_selected_moe_rows(layer, "post.output_final", output, num_tokens, top_k)
    return StandardCombineInput(hidden_states=output)


# ---------------------------------------------------------------------------
# Config dispatch
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.quantization.w8a8_int8.W8A8Int8Config.get_quant_method",
    type=HookType.REPLACE,
)
def w8a8_int8_config_get_quant_method_kunlun(
    self,
    layer: torch.nn.Module,
    prefix: str,
):
    """Re-implementation of ``W8A8Int8Config.get_quant_method`` that mirrors
    the upstream behaviour but uses the local imports the mimo patch relies
    on. Keeping the explicit override avoids relying on the upstream
    method's class-attribute lookup, which can drift between releases.
    """
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedFusedMoEMethod,
        UnquantizedLinearMethod,
    )
    from sglang.srt.layers.quantization.w8a8_int8 import (
        W8A8Int8LinearMethod,
        W8A8Int8MoEMethod,
    )

    if isinstance(layer, LinearBase):
        if should_ignore_layer(
            prefix,
            ignore=self.ignore,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()
        return W8A8Int8LinearMethod(self)
    if isinstance(layer, FusedMoE):
        if should_ignore_layer(
            prefix,
            ignore=self.ignore,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedFusedMoEMethod()
        return W8A8Int8MoEMethod(self)
    return None
