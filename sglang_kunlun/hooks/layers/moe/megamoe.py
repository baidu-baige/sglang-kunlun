"""KLX MegaMoE wiring on 0.5.23.

The fused op itself lives in ``sglang_kunlun.hooks.layers.moe.klx_megamoe`` (migrated from
the 0.5.8 fork, commits ``b89a38acd`` / ``b48757625``). Upstream 0.5.23 already
knows the ``--moe-a2a-backend megamoe`` value, forces expert parallelism to span
the TP group, and keeps ``StandardDispatcher`` (the fused op does its own
dispatch), so what is left is the Kunlun side of the contract:

* the symmetric workspace has to be sized before bkcl allocates the xshmem heap,
  which happens after argument resolution but before any worker starts;
* the w4a8 expert weights have to be rewritten into hydrax's gate/up-interleaved
  layout once, at weight load;
* ``DeepseekV2MoE.forward`` has to take the fused path instead of the packed-int4
  GEMM chain.

Why hooks and not an edit to each site: all three targets are upstream 0.5.23
code, and the plugin already owns this kind of patch. The only 0.5.23-side edit
is the new ``klx_megamoe`` module, which does not exist upstream.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


def _is_megamoe() -> bool:
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    return get_moe_a2a_backend().is_megamoe()


def _on_kunlun() -> bool:
    """Whether this process serves on the KLX device.

    The upstream mega-moe hooks gate on CUDA SM90 / SM100 because their fused
    kernels are DeepGEMM's; the KLX build runs hydrax's ``int8_int4_mega_moe``
    instead, on the same argument value. The platform flag is what the rest of
    the plugin keys on, so use it here too.
    """
    import os

    return os.environ.get("SGLANG_USE_XPU") == "1"


@plugin_hook(
    "sglang.srt.arg_groups.mega_moe_hook.check_mega_moe_compat",
    type=HookType.AROUND,
)
def megamoe_compat_kunlun(original, server_args, *args, **kwargs):
    """Keep the overlap check, drop the CUDA-only platform requirement.

    TBO/SBO really is impossible here: the fused op has no dispatch_a /
    combine_a split to overlap. The SM90/SM100 gate describes DeepGEMM kernels
    this build never reaches, so on KLX only the overlap check applies.
    """
    if not _on_kunlun():
        return original(server_args, *args, **kwargs)

    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    if cfg.moe_a2a_backend != "megamoe":
        return None
    if cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap:
        raise ValueError(
            "--moe-a2a-backend megamoe has no two-batch / single-batch overlap "
            "decomposition; disable --enable-two-batch-overlap and "
            "--enable-single-batch-overlap."
        )
    return None


@plugin_hook(
    "sglang.srt.arg_groups.mega_moe_hook.validate_mega_moe_token_budget",
    type=HookType.AROUND,
)
def megamoe_token_budget_kunlun(original, server_args, *args, **kwargs):
    """Skip upstream's per-rank token budget: KLX derives its own capacity.

    ``validate_mega_moe_token_budget`` compares the batch against
    ``SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK``. The KLX path sizes
    the symmetric workspace from the resolved batch instead
    (``klx_megamoe.megamoe_capacity_for``), and the fused op validates the row
    count at call time, so the DeepGEMM knob does not describe this runtime.
    """
    if _on_kunlun():
        return None
    return original(server_args, *args, **kwargs)


@plugin_hook(
    "sglang.srt.arg_groups.overrides._a2a_backend_overrides",
    type=HookType.AROUND,
)
def megamoe_xshmem_env(original, view, *args, **kwargs):
    """Size the symmetric heap the fused op allocates out of.

    bkcl allocates ``XSHMEM_SYMMETRIC_SIZE`` up front and the workspace is taken
    from it, so the value has to be in the environment before the workers start
    -- this runs while the args are still being resolved. It is derived from the
    resolved batch shape rather than configured, and it sits OUTSIDE
    ``--mem-fraction-static``, i.e. every byte reserved here is HBM the KV cache
    does not get, which is why it is computed instead of defaulted high.
    """
    import os

    result = original(view, *args, **kwargs)

    if getattr(view, "moe_a2a_backend", None) != "megamoe":
        return result

    from sglang_kunlun.hooks.layers.moe.klx_megamoe import (
        megamoe_capacity_for,
        megamoe_xshmem_size,
    )

    if os.environ.get("XSHMEM_MODE") is None:
        os.environ["XSHMEM_MODE"] = "1"
        logger.warning("MegaMoE: XSHMEM_MODE is set to 1.")

    needed = megamoe_xshmem_size(view)
    current = os.environ.get("XSHMEM_SYMMETRIC_SIZE")
    if current is None:
        os.environ["XSHMEM_SYMMETRIC_SIZE"] = str(needed)
        logger.warning(
            "MegaMoE: XSHMEM_SYMMETRIC_SIZE set to %d bytes (%.2f GiB) for a "
            "%d-token per-rank capacity.",
            needed,
            needed / 2**30,
            megamoe_capacity_for(view),
        )
    elif int(current) < needed:
        logger.warning(
            "MegaMoE: XSHMEM_SYMMETRIC_SIZE=%s is below the %d bytes (%.2f GiB) "
            "the workspace needs; the symmetric allocation will fail.",
            current,
            needed,
            needed / 2**30,
        )
    return result


@plugin_hook(
    "sglang_kunlun.hooks.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_wna16_moe.KunlunCompressedTensorsWNA16MoE."
    "process_weights_after_loading",
    type=HookType.AFTER,
)
def megamoe_build_weights(*args, **kwargs):
    """Rewrite the w4a8 experts into the hydrax layout and allocate the workspace.

    Weight loading is the last collective point outside any graph capture, so
    both the in-place gate/up interleave and ``get_symm_buffer`` (an xshmem
    allocation plus a peer-pointer exchange) are anchored here instead of on the
    first forward, which would land inside a decode-graph capture.
    """
    if not _is_megamoe():
        return

    from sglang_kunlun.hooks.layers.moe.klx_megamoe import (
        build_megamoe_expert_weights,
        megamoe_init_symm_buffer,
    )

    # The hook is applied to the scheme method, but which argument carries the
    # MoE layer depends on how the registry binds `self`; pick whichever argument
    # actually holds the packed weights instead of assuming a position.
    candidates = list(args) + list(kwargs.values())
    layer = next(
        (c for c in candidates if hasattr(c, "w13_weight_packed")), None
    )
    if layer is None:
        # Log instead of returning silently: a silent return here is
        # indistinguishable from "the fused path was never selected".
        logger.warning(
            "MegaMoE: process_weights_after_loading saw no packed w4a8 layer "
            "(args=%s); the fused path stays off.",
            [type(a).__name__ for a in candidates],
        )
        return
    logger.warning("MegaMoE: building hydrax expert layout for %s", type(layer).__name__)
    build_megamoe_expert_weights(layer)
    megamoe_init_symm_buffer(layer)


@plugin_hook(
    "sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward",
    type=HookType.AROUND,
)
def megamoe_forward(original, moe, hidden_states, *args, **kwargs):
    """Take the fused path when this layer's experts were rewritten for hydrax.

    A layer whose weights were not rewritten -- an unquantized or differently
    quantized MoE, e.g. an MTP layer -- keeps the ordinary path, because its
    weights are still in the layout those GEMMs want.
    """
    if _is_megamoe():
        from sglang_kunlun.hooks.layers.moe.klx_megamoe import (
            forward_megamoe_deepseek,
            should_use_megamoe,
        )

        if should_use_megamoe(moe):
            return forward_megamoe_deepseek(
                moe,
                hidden_states,
                kwargs.get("forward_batch"),
                kwargs.get("gemm_output_zero_allocator"),
            )
    return original(moe, hidden_states, *args, **kwargs)
