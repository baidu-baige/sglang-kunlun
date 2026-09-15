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
"""HookRegistry registrations for the DSV4 MoE JIT kernels.

Upstream ``sglang.jit_kernel.dsv4.moe`` builds tvm_ffi CUDA JIT modules, which are
unavailable on the Kunlun XPU. Three symbols are REPLACE'd here:

- ``silu_and_mul_clamp`` -> native fused op
  ``silu(clamp(g, max=lim)) * clamp(u, -lim, lim)``, where ``g, u`` is the gate_up split.
- ``hash_topk`` -> fused Kunlun hash MoE routing, with a pure-torch fallback.
- ``mask_topk_ids`` -> static-shape ``where`` + in-place ``copy_`` (graph-capturable).
  The JIT build of this one fails earlier than the others: tvm_ffi ``load_inline``
  compiles ``#include <concepts>``, unavailable on gcc-9 ("fatal error: concepts: No
  such file or directory"). ``srt.layers.moe.topk._mask_topk_ids_padded_region``
  reaches it via the ``_is_cuda`` branch (Kunlun masquerades as CUDA) whenever
  ``num_token_non_padded`` is set, e.g. deepep dispatch padding on the PD prefill CP
  path.

``sglang.jit_kernel.dsv4.__init__`` re-exports both symbols and the call sites
(``models.deepseek_v2``, ``srt.layers.moe.hash_topk``, ``srt.layers.moe.topk``) import
them from there. ``HookRegistry._apply_target`` resolves the parent package first (so
the re-export binding exists in ``sys.modules``) and then ``_propagate_patch`` rewrites
that stale binding, so a later ``from sglang.jit_kernel.dsv4 import ...`` picks up these
versions.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.jit_kernel.dsv4.moe.silu_and_mul_clamp",
    type=HookType.REPLACE,
)
def _silu_and_mul_clamp_kunlun(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    """Native xspeedgate fused op.

    Schema: ``silu_and_mul_with_swiglu_limit(Tensor gate_up, float limit) -> Tensor``
    """
    result = torch.ops.xspeedgate_ops.silu_and_mul_with_swiglu_limit(
        input, float(swiglu_limit)
    )
    output.copy_(result)


@plugin_hook(
    "sglang.jit_kernel.dsv4.moe.hash_topk",
    type=HookType.REPLACE,
)
def _hash_topk_torch(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
):
    """Kunlun fused hash MoE routing, falling back to pure torch."""
    assert scoring_func == "sqrtsoftplus"
    num_tokens = router_logits.size(0)
    num_routed_experts = router_logits.size(1)
    topk_routed = tid2eid.size(1)
    topk_fused = topk_routed + int(num_fused_shared_experts)
    dev = router_logits.device

    topk_weights = torch.empty(
        (num_tokens, topk_fused), dtype=torch.float32, device=dev
    )
    topk_ids = torch.empty((num_tokens, topk_fused), dtype=torch.int32, device=dev)
    if num_tokens == 0:
        return topk_weights, topk_ids

    # Kunlun fused hash MoE routing (moe_hash_topk_fused): same softplus+sqrt +
    # renormalize + fused shared-expert semantics. input_tokens must be int32.
    # Torch fallback below.
    try:
        import kunlun_ops

        kunlun_ops.moe_hash_topk_fused(
            router_logits,
            topk_weights,
            topk_ids,
            int(topk_routed),
            True,
            input_ids[:num_tokens].to(torch.int32).contiguous(),
            tid2eid,
            None,
            float(routed_scaling_factor),
            0,
            int(num_fused_shared_experts),
        )
        return topk_weights, topk_ids
    except Exception:
        pass

    token_id = input_ids[:num_tokens].to(torch.int64)          # (T,)
    expert_id = tid2eid[token_id].to(torch.int64)              # (T, topk_routed)
    logit = torch.gather(router_logits.float(), 1, expert_id)  # (T, topk_routed)

    softplus = logit.clamp(min=0.0) + torch.log1p(torch.exp(-logit.abs()))
    weight = torch.sqrt(softplus)
    routed_sum = weight.sum(dim=1, keepdim=True)
    routed_weight = weight / routed_sum

    topk_ids[:, :topk_routed] = expert_id.to(torch.int32)
    topk_weights[:, :topk_routed] = routed_weight

    if int(num_fused_shared_experts) > 0:
        shared_w = 1.0 / float(routed_scaling_factor)
        for j in range(int(num_fused_shared_experts)):
            topk_ids[:, topk_routed + j] = num_routed_experts + j
            topk_weights[:, topk_routed + j] = shared_w

    return topk_weights, topk_ids


@plugin_hook(
    "sglang.jit_kernel.dsv4.moe.mask_topk_ids",
    type=HookType.REPLACE,
)
def _mask_topk_ids_torch(
    topk_ids: torch.Tensor,
    num_token_non_padded: torch.Tensor,
) -> None:
    """Set rows >= ``num_token_non_padded`` to -1, in-place and graph-capturable."""
    idx = torch.arange(topk_ids.shape[0], device=topk_ids.device).unsqueeze(-1)
    new = torch.where(idx < num_token_non_padded, topk_ids,
                      torch.full_like(topk_ids, -1))
    topk_ids.copy_(new)
