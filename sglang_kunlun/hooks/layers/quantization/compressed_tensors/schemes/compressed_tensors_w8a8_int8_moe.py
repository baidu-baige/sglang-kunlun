"""Kunlun compressed-tensors MoE schemes."""

from __future__ import annotations

import os

import torch
import xspeedgate_ops
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.environ import envs
from sglang.srt.utils import set_weight_attrs

from ..utils import (
    INT8_SCALE_MULT,
    check_standard_routed_moe_layer,
    moe_fc,
    moe_post,
    moe_pre_sorted,
    scale_to_kernel_max,
)

__all__ = [
    "KunlunCompressedTensorsW8A8Int8MoE",
]


class KunlunCompressedTensorsW8A8Int8MoE(CompressedTensorsMoEScheme):
    """Kunlun W8A8 MoE for INT8 channel weights and dynamic token activations."""

    def __init__(self, weight_quant, input_quant) -> None:
        """Initialize the Kunlun W8A8 MoE quantization scheme."""
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )
        self.moe_runner_config = None

    @classmethod
    def get_min_capability(cls) -> int:
        """Return the minimum device capability required by the scheme."""
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create INT8 expert weights and per-channel scales."""
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        check_standard_routed_moe_layer(layer)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config) -> None:
        """Store the MoE runner configuration for later execution."""
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare loaded INT8 weights and scales for execution."""
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.data, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.data, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            scale_to_kernel_max(layer.w13_weight_scale.data, INT8_SCALE_MULT),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            scale_to_kernel_max(layer.w2_weight_scale.data, INT8_SCALE_MULT),
            requires_grad=False,
        )

    def apply_weights(self, layer: torch.nn.Module, dispatch_output):
        """Apply the W8A8 MoE weights to dispatched tokens."""
        import kunlun_ops
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        if x.shape[0] == 0:
            return StandardCombineInput(hidden_states=x.new_empty(x.shape))
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights.to(x.dtype)
        topk_ids = topk_output.topk_ids.to(torch.int32)
        top_k = topk_ids.shape[1]
        index_have_neg = getattr(layer, "moe_ep_size", 1) > 1
        num_experts = layer.w13_weight.shape[0]
        sorted_x, sorted_tokens_idx, sorted_tokens_num_lod = moe_pre_sorted(
            x,
            topk_ids,
            num_experts,
            index_have_neg=index_have_neg,
        )

        num_tokens = x.shape[0]
        gate_up = moe_fc(
            sorted_x,
            layer.w13_weight,
            layer.w13_weight_scale,
            sorted_tokens_num_lod,
            sorted_tokens_idx,
            topk_ids,
            (num_tokens, top_k, layer.w13_weight.shape[1]),
            x.dtype,
            quantize_input=True,
        )
        # TODO: 待验证 clamp_swiglu_dynamic_quant 融合算子
        # # fused (optional clamp) + silu_and_mul + per-row int8 quant for second FC.
        # is_2604b = envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B"
        # swiglu_limit = 10.0 if is_2604b else 0.0
        # if is_2604b:
        #     deepseek_v4_moe_code_path_checker.observed += 1

        # x_q, x_scale = xspeedgate_ops.ops.clamp_swiglu_dynamic_quant(
        #     y, float(swiglu_limit)
        # )
        # del y, moe_expand

        if self.moe_runner_config.swiglu_limit is not None:
            limit = float(self.moe_runner_config.swiglu_limit)
            gate, up = gate_up.chunk(2, dim=-1)
            gate.clamp_(max=limit)
            up.clamp_(min=-limit, max=limit)
        hidden = torch.empty(
            gate_up.shape[:-1] + (gate_up.shape[-1] // 2,),
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        kunlun_ops.swiglu(gate_up, hidden)
        # moe_fc quantizes per row, so flatten (num_tokens, top_k, ...) back to
        # (num_tokens * top_k, ...) before the second projection.
        hidden = hidden.reshape(num_tokens * top_k, -1)

        down = moe_fc(
            hidden,
            layer.w2_weight,
            layer.w2_weight_scale,
            sorted_tokens_num_lod,
            sorted_tokens_idx,
            topk_ids,
            (num_tokens, top_k, layer.w2_weight.shape[1]),
            x.dtype,
            quantize_input=True,
        )

        output = moe_post(down, sorted_tokens_idx, topk_weights, x.shape)

        routed_scaling_factor = self.moe_runner_config.routed_scaling_factor
        if routed_scaling_factor not in (None, 1.0):
            output.mul_(routed_scaling_factor)
        if output.dtype is torch.float16:
            limit = int(os.environ.get("SGLANG_FP16_LIMIT_IN_MOE", "10"))
            output = output.clamp(min=-limit, max=limit)
        return StandardCombineInput(hidden_states=output)
