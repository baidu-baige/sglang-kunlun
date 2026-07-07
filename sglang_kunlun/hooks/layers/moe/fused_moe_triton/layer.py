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
