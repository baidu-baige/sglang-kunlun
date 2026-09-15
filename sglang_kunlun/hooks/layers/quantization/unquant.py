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
"""REPLACE ``UnquantizedFusedMoEMethod.apply`` with a kunlun_ops moe path.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/quantization/unquant.py.

The mimo branch monkey-patches ``UnquantizedFusedMoEMethod.apply`` to a
kunlun_ops-backed pipeline (gen_block_statistic -> moe_pre_sorted ->
moe_fc -> swiglu -> moe_fc -> moe_post). Here we surface the same
behaviour through a method-level ``REPLACE`` plugin hook.

The hook expects ``dispatch_output.topk_output`` to be a 4-tuple
``(topk_weights, topk_ids, router_logits, block_statistic)`` as produced
by the kunlun ``select_experts`` REPLACE in ``layers/moe/topk.py``
(Wave 3). On a non-Kunlun host the original upstream apply is used, so
nothing changes there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


@plugin_hook(
    "sglang.srt.layers.quantization.unquant.UnquantizedFusedMoEMethod.apply",
    type=HookType.REPLACE,
)
def unquantized_fused_moe_apply_kunlun(
    self,
    layer: torch.nn.Module,
    dispatch_output: "StandardDispatchOutput",
) -> "CombineInput":
    """Kunlun fused-MoE forward for the unquantized path."""
    import kunlun_ops  # local-only dep
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if len(topk_output) == 4:
        topk_weights, topk_ids, _router_logits, block_statistic = topk_output
    elif len(topk_output) == 3:
        topk_weights, topk_ids, _router_logits = topk_output
        block_statistic = None
    else:
        raise ValueError(
            f"Unsupported topk_output format in kunlun unquant MoE path: {len(topk_output)}"
        )
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    top_k = topk_ids.shape[1]
    global_num_experts, _up_gate_size, _ = layer.w13_weight.shape
    M, N = hidden_states.shape
    hidden_dim = layer.w2_weight.shape[1]

    cache: Any = torch.empty(
        M * top_k * max(layer.w13_weight.shape[1], layer.w2_weight.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    moe_expand = torch.empty(
        (M * top_k, N), dtype=hidden_states.dtype, device=hidden_states.device
    )
    expert_m = torch.zeros(
        global_num_experts, dtype=torch.int32, device=hidden_states.device
    )
    sorted_tokens_num_lod = torch.zeros(
        global_num_experts + 1, dtype=torch.int32, device=hidden_states.device
    )
    sorted_tokens_idx = torch.zeros(
        M * top_k, dtype=torch.int32, device=hidden_states.device
    )
    if block_statistic is None or block_statistic.shape != (12, global_num_experts):
        block_statistic = torch.empty(
            (12, global_num_experts),
            dtype=torch.int32,
            device=hidden_states.device,
        )

    kunlun_ops.gen_block_statistic(topk_ids, block_statistic)
    kunlun_ops.moe_pre_sorted(
        x=hidden_states,
        topk_index=topk_ids,
        block_statistic=block_statistic,
        moe_expand=moe_expand,
        moe_index=sorted_tokens_idx,
        expert_m=expert_m,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
    )

    y = torch.empty(
        M,
        top_k,
        layer.w13_weight.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    moe_expand = moe_expand.view(M * top_k, hidden_dim)
    kunlun_ops.moe_fc(
        x=moe_expand,
        x_perchannel_max=None,
        weight=layer.w13_weight,
        w_perchannel_max=None,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=top_k,
        y=y,
        topk_ids=topk_ids,
        act=None,
    )

    d = y.shape[-1] // 2
    out1 = torch.empty(y.shape[:-1] + (d,), dtype=y.dtype, device=y.device)
    kunlun_ops.swiglu(x=y, y=out1)
    out1 = out1.reshape(-1, out1.shape[-1])

    out = cache[: M * top_k * layer.w2_weight.shape[1]].view(
        (M, top_k, layer.w2_weight.shape[1]),
    )
    kunlun_ops.moe_fc(
        x=out1,
        x_perchannel_max=None,
        weight=layer.w2_weight,
        w_perchannel_max=None,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=top_k,
        y=out,
        topk_ids=topk_ids,
        act=None,
    )

    dequant_scale = torch.ones(
        [M, top_k], dtype=torch.float32, device=out.device
    )
    sorted_tokens_idx = sorted_tokens_idx.view(M, top_k)
    kunlun_ops.moe_post(
        x=out,
        moe_index=sorted_tokens_idx,
        normed_scale=topk_weights,
        dequant_scale=dequant_scale,
        y=hidden_states,
    )

    return StandardCombineInput(hidden_states=hidden_states)


@plugin_hook(
    "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply",
    type=HookType.AROUND,
)
def unquantized_linear_apply_kunlun(original_fn, self, layer, x, bias=None):
    """Cast activation/bias to the weight dtype before the Kunlun-rewritten
    ``F.linear`` / ``fc_fusion``, which requires ``x.dtype == weight.dtype``.

    The MTP/NextN draft feeds fp32 activations into fp16 unquantized weights,
    crashing with ``fc_fusion execution failed: lhs and rhs must have the same
    data type. Got lhs: Float, rhs: Half``. No-op when dtypes already match.
    """
    w = getattr(layer, "weight", None)
    if w is not None and isinstance(x, torch.Tensor) and x.dtype != w.dtype:
        x = x.to(w.dtype)
        if bias is not None and isinstance(bias, torch.Tensor) and bias.dtype != w.dtype:
            bias = bias.to(w.dtype)
    return original_fn(self, layer, x, bias)
