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

import torch
from torch.nn.parameter import Parameter

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


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

    w_shape = layer.weight.shape
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
        kunlun_ops.quant2d(x, x_q, x_scale, force_sdnn=True)

    kunlun_ops.matmul(
        x_q,
        layer.weight.data,
        out,
        bias=bias,
        x_pc_max=x_scale,
        w_pc_max=layer.weight_scale.data,
    )
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

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids
    _router_logits = topk_output.router_logits
    top_k = self.moe_runner_config.top_k
    num_experts = self.moe_runner_config.num_experts
    gateup_output_n = layer.w13_weight.shape[1]
    hidden_size = layer.w2_weight.shape[1]
    num_tokens = x.shape[0]
    device = x.device

    cache: Any = torch.empty(
        num_tokens * top_k * max(gateup_output_n, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    sorted_tokens_num_lod = torch.empty(
        [num_experts + 1], dtype=torch.int32, device=device
    )
    # kunlun_ops.moe_fc requires a FLAT 1-D index of length num_tokens*top_k
    # (its assertion: sorted_tokens_idx.shape[0] == y.shape[0]*y.shape[1]).
    sorted_tokens_idx = torch.empty(
        num_tokens * top_k, dtype=torch.int32, device=device
    )
    kunlun_ops.moe_sorted_topk_idx(
        topk_ids,
        topk_ids.shape[0],
        top_k,
        num_experts,
        sorted_tokens_num_lod,
        sorted_tokens_idx,
    )

    # quant input
    x_q = torch.empty_like(x, dtype=torch.int8, device=device)
    x_scale = torch.empty(x.shape[0], dtype=torch.float32, device=device)
    kunlun_ops.quant2d(x, x_q, x_scale, force_sdnn=False)

    # up
    intermediate_cache1 = cache[: num_tokens * top_k * gateup_output_n].view(
        (num_tokens, top_k, gateup_output_n),
    )
    kunlun_ops.moe_fc(
        x=x_q,
        weight=layer.w13_weight,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=top_k,
        y=intermediate_cache1,
        act=None,
        x_perchannel_max=x_scale,
        w_perchannel_max=layer.w13_weight_scale,
        topk_ids=topk_ids,
        sort_mode=False,
    )

    # act
    intermediate_cache2 = torch.empty(
        (num_tokens, top_k, gateup_output_n // 2),
        dtype=torch.bfloat16,
        device=device,
    )
    kunlun_ops.swiglu(intermediate_cache1, intermediate_cache2)

    # quant intermediate
    intermediate_cache2_q = torch.empty(
        (num_tokens * top_k, gateup_output_n // 2),
        dtype=torch.int8,
        device=device,
    )
    intermediate_cache2_scale = torch.empty(
        num_tokens * top_k, dtype=torch.float32, device=device
    )
    kunlun_ops.quant2d(
        intermediate_cache2,
        intermediate_cache2_q,
        intermediate_cache2_scale,
        force_sdnn=False,
    )

    # down
    intermediate_cache3 = cache[: num_tokens * top_k * hidden_size].view(
        (num_tokens, top_k, hidden_size),
    )
    kunlun_ops.moe_fc(
        x=intermediate_cache2_q,
        weight=layer.w2_weight,
        sorted_tokens_num_lod=sorted_tokens_num_lod,
        sorted_tokens_idx=sorted_tokens_idx,
        moe_topk=1,
        y=intermediate_cache3,
        act=None,
        x_perchannel_max=intermediate_cache2_scale,
        w_perchannel_max=layer.w2_weight_scale,
        topk_ids=topk_ids,
        topk_w=topk_weights,
        sort_mode=False,
    )

    out = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    kunlun_ops.reduce_sum(intermediate_cache3, 1, out)
    # Apply DSV4 routed_scaling_factor on the routed-expert output. Stock sglang
    # MoE runners do ``output *= routed_scaling_factor`` (e.g. deep_gemm/flashinfer);
    # this kunlun REPLACE path omitted it, leaving the routed-expert output a factor
    # of routed_scaling_factor (1.5 for DSV4-Flash) too small -> ~33% magnitude
    # divergence vs the reference (cos ~1.0, rel ~0.33). Re-apply it here.
    rsf = getattr(self.moe_runner_config, "routed_scaling_factor", None)
    if rsf is not None and rsf != 1.0:
        out.mul_(rsf)
    return StandardCombineInput(hidden_states=out)


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
