"""Kunlun compressed-tensors W4A16 (pack-quantized) fused MoE scheme."""

from __future__ import annotations

import os

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    WNA16_SUPPORTED_BITS,
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

from ..utils import (
    INT4_SIGNED_SCALE_MULT,
    check_standard_routed_moe_layer,
    moe_fc,
    moe_post,
    moe_pre_sorted,
    scale_to_kernel_max,
)

# SwiGLU clamping for routed experts: the fused op vs "2 clamp_ + swiglu".
# DSV4.1 requires clamping gate/up to swiglu_limit, otherwise BOS/attention-sink
# activations amplify across layers into fp16 Inf/NaN and greedy decoding collapses to
# repeated token 0. The two paths are mathematically identical; the fused version saves
# two passes of large-tensor reads/writes. Set DSV4_MOE_SWIGLU_CLAMP_FUSION=0 to switch
# back to the fallback path for comparison.
_FUSED_SWIGLU_CLAMP = os.environ.get("DSV4_MOE_SWIGLU_CLAMP_FUSION", "1") == "1"


__all__ = ["KunlunCompressedTensorsWNA16MoE"]


class KunlunCompressedTensorsWNA16MoE(CompressedTensorsMoEScheme):
    """Packed INT4 fused MoE backed by Kunlun ``moe_fc_v3``."""

    quantize_activations = False

    def __init__(self, scheme_dict, quant_format: str | None = None) -> None:
        """Read the MoE scheme/format from the matched expert target."""
        config = scheme_dict.get("weights")
        quant_format = scheme_dict.get("format", quant_format)

        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        self.sym = config.symmetric
        self.moe_runner_config = None

        if not config.symmetric:
            raise ValueError("Only symmetric quantization is supported for Kunlun MoE")
        if not (
            quant_format == CompressionFormat.pack_quantized.value
            and self.num_bits in WNA16_SUPPORTED_BITS
        ):
            raise ValueError(
                "For Fused MoE layers, only "
                f"{CompressionFormat.pack_quantized.value} is supported for the "
                f"following bits: {WNA16_SUPPORTED_BITS}, got {quant_format} / "
                f"{self.num_bits} bits"
            )

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
        """Create packed INT4 expert weights in the GPTQ/compressed-tensors layout."""
        check_standard_routed_moe_layer(layer)
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        load_full_w2 = (
            self.actorder is not None
            and self.actorder != "static"
            and self.group_size != -1
        )
        if load_full_w2:
            w2_scales_size = intermediate_size_per_partition * layer.moe_tp_size
        else:
            w2_scales_size = intermediate_size_per_partition

        strategy_value = (
            self.strategy.value
            if isinstance(self.strategy, QuantizationStrategy)
            else self.strategy
        )
        if strategy_value == QuantizationStrategy.CHANNEL.value:
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        # Channel quantization without act-order has no checkpoint group indices.
        # Keep the attribute/shape/loader interface until postprocessing deletes
        # these unused placeholders, but do not advertise them as checkpoint
        # parameters. Zero initialization also avoids retaining undefined data.
        internal_indices = (
            strategy_value == QuantizationStrategy.CHANNEL.value
            and self.actorder is None
        )
        for name, width in (
            ("w13_weight_g_idx", hidden_size),
            ("w2_weight_g_idx", intermediate_size_per_partition),
            ("w13_g_idx_sort_indices", hidden_size),
            ("w2_g_idx_sort_indices", intermediate_size_per_partition),
        ):
            if internal_indices:
                indices = torch.zeros(num_experts, width, dtype=torch.int32)
                layer.register_buffer(name, indices, persistent=False)
            else:
                indices = torch.nn.Parameter(
                    torch.empty(num_experts, width, dtype=torch.int32),
                    requires_grad=False,
                )
                layer.register_parameter(name, indices)
            set_weight_attrs(indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config) -> None:
        """Store the MoE runner configuration for later execution."""
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack GPTQ-style INT4 weights into the layout Kunlun ``moe_fc_v3`` wants.

        Marlin repacking / act-order is not used on Kunlun, so the related
        buffers are dropped. ``bitwise_xor_(0x88)`` rebiases the two packed
        unsigned INT4 nibbles to signed INT4, and the scales are converted to
        a per-channel absmax (x7) because the kernel consumes a max rather
        than a scale.
        """
        for unused in (
            "w13_weight_shape",
            "w2_weight_shape",
            "w13_weight_g_idx",
            "w2_weight_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
        ):
            if hasattr(layer, unused):
                delattr(layer, unused)

        strategy_value = (
            self.strategy.value
            if isinstance(self.strategy, QuantizationStrategy)
            else self.strategy
        )
        if strategy_value != QuantizationStrategy.CHANNEL.value:
            raise ValueError(
                "Kunlun W4A16 fused MoE only supports channel-wise scales, got "
                f"{self.strategy}"
            )

        from sglang_kunlun.hooks.layers.moe import mega_moe

        if mega_moe.mega_moe_enabled() and mega_moe.convert_layer_weights(layer):
            # MegaMoE wants gate/up interleaved by 256, moe_fc_v3 wants them
            # contiguous. The expert weights are far too large to keep both, so
            # a converted layer commits to the MegaMoE layout and its forward
            # must go through MegaMoE. `convert_layer_weights` returns False for
            # the DSPARK draft model, which stays on DeepEP.
            return

        for weight_name, scale_name in (
            ("w13_weight_packed", "w13_weight_scale"),
            ("w2_weight_packed", "w2_weight_scale"),
        ):
            weight = getattr(layer, weight_name)
            packed = weight.data.transpose(1, 2).contiguous().view(torch.int8)
            packed.bitwise_xor_(0x88)
            weight.data = packed

            scale = getattr(layer, scale_name)
            scale.data = scale_to_kernel_max(
                scale.data.transpose(1, 2).contiguous(),
                INT4_SIGNED_SCALE_MULT,
            )

    def apply_weights(self, layer: torch.nn.Module, dispatch_output):
        """Apply packed INT4 MoE weights to dispatched tokens."""
        import kunlun_ops
        import xspeedgate_ops  # noqa: F401  # registers torch.ops.xspeedgate_ops
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        if x.shape[0] == 0:
            return StandardCombineInput(hidden_states=x.new_empty(x.shape))

        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights.to(x.dtype)
        topk_ids = topk_output.topk_ids.to(torch.int32)
        top_k = topk_ids.shape[1]
        index_have_neg = getattr(layer, "moe_ep_size", 1) > 1
        num_experts = layer.w13_weight_packed.shape[0]
        sorted_x, sorted_tokens_idx, sorted_tokens_num_lod = moe_pre_sorted(
            x,
            topk_ids,
            num_experts,
            index_have_neg=index_have_neg,
            block_statistic=getattr(topk_output, "block_statistic", None),
        )

        num_tokens = x.shape[0]
        gate_up = moe_fc(
            sorted_x,
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            sorted_tokens_num_lod,
            sorted_tokens_idx,
            topk_ids,
            (num_tokens, top_k, layer.w13_weight_packed.shape[1]),
            x.dtype,
            quantize_input=self.quantize_activations,
            packed_int4=True,
        )

        limit = (
            float(self.moe_runner_config.swiglu_limit)
            if self.moe_runner_config is not None
            and self.moe_runner_config.swiglu_limit is not None
            else None
        )
        if limit is not None and _FUSED_SWIGLU_CLAMP:
            # One op does clamp + silu + mul. The fallback path is 2 clamp_ + 1 swiglu,
            # which reads/writes the large [num_tokens, top_k, 2I] tensor two extra times
            # -- in the P-side 16k prefill trace, clamp alone is 115 ms / 2.0% (240 calls
            # = 2 per layer).
            # Semantics match the fallback path word for word, pinned by
            # test_dsv4_silu_and_mul_clamp_matches_058_contract in test_kernel_ops.py:
            #     silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
            # Use the op's return value directly as hidden, saving the output.copy_() in the
            # dsv4_silu_and_mul_clamp_kunlun wrapper.
            hidden = torch.ops.xspeedgate_ops.silu_and_mul_with_swiglu_limit(
                gate_up.reshape(-1, gate_up.shape[-1]), limit
            )
        else:
            if limit is not None:
                gate, up = gate_up.chunk(2, dim=-1)
                gate.clamp_(max=limit)
                up.clamp_(min=-limit, max=limit)
            hidden = torch.empty(
                gate_up.shape[:-1] + (gate_up.shape[-1] // 2,),
                dtype=gate_up.dtype,
                device=gate_up.device,
            )
            kunlun_ops.swiglu(gate_up, hidden)
        hidden = hidden.reshape(num_tokens * top_k, -1)

        down = moe_fc(
            hidden,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            sorted_tokens_num_lod,
            sorted_tokens_idx,
            topk_ids,
            (num_tokens, top_k, layer.w2_weight_packed.shape[1]),
            x.dtype,
            quantize_input=self.quantize_activations,
            packed_int4=True,
        )

        # On the int8 grouped path, `down` is bf16; cast it back to the model dtype here,
        # otherwise the fp16 clamp check below is skipped and bf16 leaks into the fp16
        # layers that follow.
        output = moe_post(
            down, sorted_tokens_idx, topk_weights, x.shape, output_dtype=x.dtype
        )

        routed_scaling_factor = getattr(
            self.moe_runner_config, "routed_scaling_factor", None
        )
        if routed_scaling_factor not in (None, 1.0):
            output.mul_(routed_scaling_factor)
        if output.dtype is torch.float16:
            limit = int(os.environ.get("SGLANG_FP16_LIMIT_IN_MOE", "10"))
            output = output.clamp(min=-limit, max=limit)
        return StandardCombineInput(hidden_states=output)
