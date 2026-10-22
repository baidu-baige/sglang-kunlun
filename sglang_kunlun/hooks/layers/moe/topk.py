"""Kunlun MoE top-k hooks.

One functional difference vs upstream:
  * The fused sigmoid+grouped-topk+norm JIT kernel
    (``sglang.jit_kernel.grouped_topk.grouped_topk``) is replaced with
    ``kunlun_ops.moe_sigmoid_group_topk_norm`` — the upstream JIT path nvcc-
    compiles ``moe/grouped_topk.cuh`` with ``-std=c++20`` which the Kunlun
    nvcc (11.7) rejects. The replacement returns ``(topk_values, topk_indices)``
    exactly like upstream; the per-block routing statistic that Kunlun's MoE
    needs is regenerated on demand in ``unquant.py`` via
    ``kunlun_ops.gen_block_statistic`` (so dropping it here is fine).

``select_experts`` MUST return the upstream
``sglang.srt.layers.moe.topk.StandardTopKOutput``. Defining a same-named
NamedTuple here (e.g. to carry an extra ``block_statistic`` field) silently
breaks ``TopKOutputChecker.format_is_standard``, which is an ``isinstance``
check against the upstream class: the EP local-expert-id remapping in
``StandardDispatcher.dispatch`` is then skipped with no error at all, the
Kunlun MoE kernels receive GLOBAL expert ids where they expect local ones,
index their per-expert buffers out of bounds and fault the device (which in
turn wedges the C2C fabric). Consumers that want the per-block routing
statistic regenerate it on demand via ``kunlun_ops.gen_block_statistic``.
"""
from __future__ import annotations

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKConfig,
    biased_grouped_topk,
    biased_grouped_topk_impl,
    biased_topk_impl,
    _use_aiter,
    _post_process_topk_ids,
    grouped_topk,
    fused_topk_native,
    fused_topk,
    fused_topk_softmax_torch_raw_logits
)
from typing import Optional, Tuple
from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
import kunlun_ops
from sglang_kunlun.kernels.moe_router import route_k3_torch


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
    # 0.5.22 added these two keyword-only parameters, and ``TopK.forward_cuda``
    # passes BOTH unconditionally on every MoE forward
    # (``srt/layers/moe/topk.py:789``-791; ``forward_native`` passes
    # ``dynamic_expert_bias`` too). Omitting them is a TypeError on the active
    # MoE path, so they are mirrored here.
    #  * ``dynamic_expert_bias``: a per-token [num_tokens, num_experts] routing
    #    bias produced by the multi-gate (``srt/layers/multi_gate.py:190``,
    #    used by bailing_moe / bailing_moe_v3). Upstream overrides
    #    ``topk_config.correction_bias`` with it and routes through
    #    ``biased_grouped_topk_impl`` (sigmoid only) — see below.
    #  * ``router_logits_partials``: fp32 split-K partials from the ROCm decode
    #    router gate (``srt/models/deepseek_v2.py:1090``), whose fixed-order sum
    #    IS the logits. Only the aiter sqrtsoftplus gate consumes them; every
    #    other route first reduces them with ``rocm_router_reduce_partials``, a
    #    HIP Triton kernel. Rejected explicitly below.
    dynamic_expert_bias: Optional[torch.Tensor] = None,
    router_logits_partials: Optional[torch.Tensor] = None,
) -> StandardTopKOutput:
    """select_experts"""
    top_k = topk_config.top_k
    use_grouped_topk = topk_config.use_grouped_topk
    topk_group = topk_config.topk_group
    num_expert_group = topk_config.num_expert_group
    renormalize = topk_config.renormalize
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    custom_routing_function = topk_config.custom_routing_function
    # 0.5.22: a per-token ``dynamic_expert_bias`` supersedes the static
    # per-expert ``topk_config.correction_bias`` (upstream topk.py:2563).
    correction_bias = (
        topk_config.correction_bias
        if dynamic_expert_bias is None
        else dynamic_expert_bias
    )
    torch_native = topk_config.torch_native
    routed_scaling_factor = topk_config.routed_scaling_factor
    apply_routed_scaling_factor_on_output = (
        topk_config.apply_routed_scaling_factor_on_output
    )

    scoring_func = topk_config.scoring_func

    if router_logits_partials is not None:
        # Kunlun has no ROCm split-K router gate and cannot run the HIP Triton
        # reducer (``rocm_router_reduce_partials``) that upstream falls back to,
        # so unreduced partials would silently leave ``router_logits`` as an
        # unfilled buffer. Fail loudly instead.
        raise NotImplementedError(
            "Kunlun select_experts does not support router_logits_partials "
            "(ROCm split-K router gate); pass pre-reduced router_logits"
        )

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
    if dynamic_expert_bias is not None:
        # 0.5.22 branch (upstream topk.py:2644): a per-token bias cannot go
        # through the fused per-expert-bias gates, so upstream takes the pure
        # torch ``biased_grouped_topk_impl`` and requires sigmoid scoring.
        # That reference impl is plain torch, so it runs as-is on Kunlun —
        # mirror it exactly rather than approximating it with
        # ``kunlun_ops.moe_sigmoid_group_topk_norm`` (whose ``bias`` is
        # per-expert [num_experts], not per-token).
        if scoring_func != "sigmoid":
            raise ValueError(
                "Per-token expert bias is only supported with sigmoid routing"
            )
        topk_weights, topk_ids = biased_grouped_topk_impl(
            hidden_states=hidden_states,
            gating_output=router_logits,
            correction_bias=correction_bias,
            topk=top_k,
            renormalize=renormalize,
            num_expert_group=num_expert_group or 1,
            topk_group=topk_group or 1,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )
    elif use_k3_router:
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
        elif scoring_func == "sigmoid" and apply_routed_scaling_factor_on_output:
            # MiniMax-M3 contract only: sigmoid scoring plus the routed scaling
            # applied on the output weights. This exact combination used to hit
            # the ``assert not apply_routed_scaling_factor_on_output`` below, so
            # nothing that worked before reaches this branch — every other
            # sigmoid caller (flag False) keeps the previous fused_topk path.
            # Upstream routes it through the JIT fused gate (which carries both
            # the scaling flag and the fused shared experts); the plugin
            # replaces that kernel with the MiniMax implementation.
            from sglang.kernels.ops.moe.moe_fused_gate import (
                moe_fused_gate as _jit_moe_fused_gate,
            )

            bias_fp32 = (
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else torch.zeros(
                    router_logits.shape[1],
                    dtype=torch.float32,
                    device=router_logits.device,
                )
            )
            topk_weights, topk_ids = _jit_moe_fused_gate(
                router_logits,
                bias_fp32,
                num_routed_topk if _use_aiter else top_k,
                scoring_func="sigmoid",
                num_fused_shared_experts=num_fused_shared_experts,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
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

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)