"""Kunlun MoE top-k hooks.

Two functional differences vs upstream:
  * The fused sigmoid+grouped-topk+norm JIT kernel
    (``sglang.jit_kernel.grouped_topk.grouped_topk``) is replaced with
    ``kunlun_ops.moe_sigmoid_group_topk_norm`` — the upstream JIT path nvcc-
    compiles ``moe/grouped_topk.cuh`` with ``-std=c++20`` which the Kunlun
    nvcc (11.7) rejects. The replacement returns ``(topk_values, topk_indices)``
    exactly like upstream; the per-block routing statistic that Kunlun's MoE
    needs is regenerated on demand in ``unquant.py`` via
    ``kunlun_ops.gen_block_statistic`` (so dropping it here is fine).
  * ``select_experts`` returns a Kunlun-specific ``StandardTopKOutput``
    that carries the extra ``block_statistic`` field alongside the
    standard ``topk_weights`` / ``topk_ids`` / ``router_logits``.
"""
from __future__ import annotations

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    TopKOutputFormat,
    biased_grouped_topk,
    biased_topk_impl,
    _use_aiter,
    _post_process_topk_ids,
    grouped_topk,
    fused_topk_native,
    fused_topk,
    fused_topk_softmax_torch_raw_logits
)
from typing import Optional, NamedTuple, Tuple
from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
import kunlun_ops
from sglang_kunlun.kernels.moe_router import route_k3_torch


class StandardTopKOutput(NamedTuple):
    """Standard top-k output format."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor
    block_statistic: torch.Tensor

    @property
    def format(self) -> TopKOutputFormat:
        """TopKOutputFormat"""
        return TopKOutputFormat.STANDARD


# NOTE: the JIT replacement for ``sglang.jit_kernel.grouped_topk.grouped_topk``
# lives in ``sglang_kunlun/kernels/kernel_ops.py`` (registered + installed via
# ``kernel_ops.install()`` BEFORE the hook modules are imported). Registering it
# here would be too late — install() has already run by the time this module is
# imported, so the upstream nvcc-JIT kernel would not get patched.


def _kunlun_biased_grouped_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: Optional[bool],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused sigmoid + biased grouped top-k + renorm on Kunlun"""
    if num_fused_shared_experts:
        raise NotImplementedError(
            "Kunlun biased grouped top-k does not implement fused shared experts; "
            "launch with --disable-shared-experts-fusion"
        )
    if not renormalize:
        raise NotImplementedError(
            "kunlun_ops.moe_sigmoid_group_topk_norm always renormalizes; "
            "renormalize=False is not supported"
        )

    scale = 1.0
    if routed_scaling_factor is not None and apply_routed_scaling_factor_on_output:
        scale = float(routed_scaling_factor)

    num_tokens, num_experts = gating_output.shape
    topk_weights = torch.empty(
        num_tokens, topk, dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        num_tokens, topk, dtype=torch.int32, device=gating_output.device
    )
    if num_tokens == 0:
        return topk_weights, topk_ids

    # The kernel also emits a per-block routing statistic; the MoE method
    # regenerates it on demand, so it is discarded here.
    block_statistic = torch.empty(
        12, num_experts + 1, dtype=torch.int32, device=gating_output.device
    )
    kunlun_ops.moe_sigmoid_group_topk_norm(
        x=gating_output,
        topk_index=topk_ids,
        norm_score=topk_weights,
        block_statistic=block_statistic,
        bias=correction_bias.float(),
        scale=scale,
        n_group=num_expert_group,
        topk_group=topk_group,
    )
    return topk_weights, topk_ids


@plugin_hook(
    "sglang.srt.layers.moe.topk.select_experts",
    type=HookType.REPLACE,
)
def select_experts_kunlun(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: TopKConfig,
    *,
    layer_id: Optional[int] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
) -> StandardTopKOutput:
    """select_experts"""
    top_k = topk_config.top_k
    use_grouped_topk = topk_config.use_grouped_topk
    topk_group = topk_config.topk_group
    num_expert_group = topk_config.num_expert_group
    renormalize = topk_config.renormalize
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    custom_routing_function = topk_config.custom_routing_function
    correction_bias = topk_config.correction_bias
    torch_native = topk_config.torch_native
    routed_scaling_factor = topk_config.routed_scaling_factor
    apply_routed_scaling_factor_on_output = (
        topk_config.apply_routed_scaling_factor_on_output
    )

    scoring_func = topk_config.scoring_func

    router_logits, correction_bias = (
        expert_location_dispatch.transform_select_experts_inputs(
            router_logits=router_logits,
            correction_bias=correction_bias,
            info=expert_location_dispatch_info,
        )
    )

    # DeepSeek V2/V3/R1 series models use grouped_top_k
    # remove num_fused_shared_experts from grouped_topk/biased_grouped_topk
    num_routed_topk = top_k - num_fused_shared_experts
    block_statistic = None
    use_k3_router = (
        use_grouped_topk
        and custom_routing_function is None
        and correction_bias is not None
        and num_fused_shared_experts == 0
        and num_expert_group == 1
        and topk_group == 1
        and top_k == 16
        and router_logits.shape[-1] == 896
    )
    if use_k3_router:
        topk_weights, topk_ids = route_k3_torch(
            router_logits, correction_bias, top_k
        )
    elif use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        else:
            topk_weights, topk_ids = _kunlun_biased_grouped_topk(
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
    elif torch_native and custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )
    elif custom_routing_function is None:
        if scoring_func == "sqrtsoftplus":
            from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

            topk_weights, topk_ids = moe_fused_gate(
                router_logits,
                correction_bias,
                topk=num_routed_topk if _use_aiter else top_k,
                scoring_func=scoring_func,
                num_fused_shared_experts=num_fused_shared_experts,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            )
        elif (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            and scoring_func == "softmax"
            and correction_bias is None
        ):
            # flashinfer_trtllm_routed uses raw-logits topk
            assert not apply_routed_scaling_factor_on_output, "Not implemented"
            topk_weights, topk_ids = fused_topk_softmax_torch_raw_logits(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
            )
        else:
            assert not apply_routed_scaling_factor_on_output, "Not implemented"
            # Qwen3MOE uses fused_topk
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=num_routed_topk if _use_aiter else top_k,
                renormalize=renormalize,
                correction_bias=correction_bias,
                scoring_func=scoring_func,
            )
    else:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in custom_routing_function"
        assert expert_location_dispatch_info is None
        assert not apply_routed_scaling_factor_on_output, "Not implemented"
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=num_routed_topk if _use_aiter else top_k,
            renormalize=renormalize,
        )

    topk_ids, topk_weights, _ = _post_process_topk_ids(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk_config=topk_config,
        router_logits=router_logits,
        num_token_non_padded=num_token_non_padded,
        layer_id=layer_id,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )

    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits, block_statistic)