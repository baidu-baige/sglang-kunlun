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
"""REPLACE ``create_moe_dispatcher`` to align dispatcher selection with mimo.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/moe/fused_moe_triton/layer.py.

Two functional differences vs upstream:
  * Drops the ``is_mori()`` and ``is_nixl()`` branches that the mimo
    deploy doesn't support.
  * Sets ``return_recv_hook=get_deepep_mode().is_normal()`` instead of
    a hard-coded ``True``.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.moe.fused_moe_triton.layer.create_moe_dispatcher",
    type=HookType.REPLACE,
)
def create_moe_dispatcher_kunlun(moe_runner_config):
    """Kunlun MoE dispatcher selection."""
    from sglang.srt.batch_overlap.two_batch_overlap import (
        MaybeTboDeepEPDispatcher,
    )
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatcher,
    )

    a2a_backend = get_moe_a2a_backend()
    if a2a_backend.is_none():
        return StandardDispatcher(moe_runner_config)
    if a2a_backend.is_deepep() or a2a_backend.is_mooncake():
        return MaybeTboDeepEPDispatcher(
            group=get_tp_group().device_group,
            router_topk=moe_runner_config.top_k,
            permute_fusion=True,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
            params_dtype=moe_runner_config.params_dtype,
            deepep_mode=get_deepep_mode(),
            async_finish=True,
            return_recv_hook=get_deepep_mode().is_normal(), # tbo暂时不支持
        )
    if a2a_backend.is_ascend_fuseep():
        from sglang.srt.layers.moe.token_dispatcher import NpuFuseEPDispatcher

        return NpuFuseEPDispatcher(
            group=get_tp_group().device_group,
            router_topk=moe_runner_config.top_k,
            permute_fusion=True,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
            params_dtype=moe_runner_config.params_dtype,
        )
    raise NotImplementedError(f"Unsupported a2a backend: {a2a_backend}")



@plugin_hook(
    "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE._weight_loader_impl",
    type=HookType.AROUND,
)
def weight_loader_impl_kunlun(
    original_fn, self, param, loaded_weight, weight_name, shard_id, expert_id
):
    """Compressed-tensors stores expert projections as ``[out, in // pack]`` with
    ``[out, num_groups]`` scales, while the Kunlun packed-INT4 schemes register
    ``is_transposed=True`` params (``[in // pack, out]`` / ``[num_groups, out]``).
    Upstream bridges the two by transposing ``loaded_weight``, but only when the
    scheme class name is exactly one of its own ``CompressedTensorsWNA16*MoE``
    (see ``_weight_loader_impl`` in the upstream module), so the Kunlun
    subclasses never got the flip and per-channel scales failed to copy
    (``[4096, 1]`` into ``[1, 4096]``).
    """
    from sglang_kunlun.hooks.layers.quantization.compressed_tensors.schemes import (
        KunlunCompressedTensorsWNA16MoE,
    )

    if (
        isinstance(getattr(self, "scheme", None), KunlunCompressedTensorsWNA16MoE)
        and "zero" not in weight_name
        and loaded_weight.dim() == 2
    ):
        loaded_weight = loaded_weight.t().contiguous()

    return original_fn(
        self,
        param=param,
        loaded_weight=loaded_weight,
        weight_name=weight_name,
        shard_id=shard_id,
        expert_id=expert_id,
    )
