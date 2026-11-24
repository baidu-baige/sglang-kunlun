"""Relax ``check_mega_moe_compat`` for Kunlun: skip the CUDA SM90/SM100 gate.

Moved out of base sglang (``arg_groups/mega_moe_hook.py``) into the plugin so the
base tree stays pristine across re-pulls. Kunlun presents as CUDA SM80
(``is_cuda=True, device_sm=80``), which upstream's SM90/SM100 capability gate
rejects, but Kunlun runs its own hydrax ``int8_int4_mega_moe`` kernel over xshmem
symmetric memory, so that gate does not apply. This hook only exists in the
Kunlun plugin, so it is inherently Kunlun-scoped. The two-batch / single-batch
overlap constraint is preserved (the megamoe kernel has no overlap split).
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.arg_groups.mega_moe_hook.check_mega_moe_compat",
    type=HookType.AROUND,
)
def check_mega_moe_compat_kunlun(original_fn, server_args):
    """Skip upstream's SM90/SM100 capability gate for Kunlun megamoe."""
    from sglang.srt.arg_groups.model_override_base import resolving_view

    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "megamoe":
        return original_fn(server_args)
    if cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap:
        # The fused kernel has no dispatch_a / combine_a split for TBO / SBO.
        raise ValueError(
            "--moe-a2a-backend megamoe has no two-batch / single-batch overlap "
            "decomposition; disable --enable-two-batch-overlap and "
            "--enable-single-batch-overlap."
        )
    # Kunlun runs its own MegaMoE kernel and reports SM80; the upstream
    # CUDA SM90/SM100 / HIP gate would reject it, so skip it here.
    return
