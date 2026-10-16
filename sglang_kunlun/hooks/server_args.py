"""Hooks for ``sglang.srt.server_args``.

Kunlun captures prefill with the segmented (breakable) CUDA-graph backend; the
torch.compile-based ``tc_piecewise`` path does not exist here. A request for it --
``--cuda-graph-backend-prefill=tc_piecewise`` or the deprecated
``--enforce-piecewise-cuda-graph`` -- would otherwise fail during capture and
lose the prefill graph entirely, so it is served by ``breakable`` with the
configured capture sizes (same shapes, same skipped auto-disable cascade).

Why a hook and not ``KunlunSRTPlatform``: ``ServerArgs._handle_cuda_graph_config``
resolves the CUDA-graph config *before* ``apply_server_args_defaults`` runs, while
plugin hooks are applied ahead of ServerArgs construction (``launch_server`` loads
plugins first). This also covers every input path -- convenience flag, legacy
alias, and the ``--cuda-graph-config`` JSON.

The second hook keeps an explicitly requested ``breakable`` prefill graph alive
under ``--deepep-mode normal``: upstream's ``_handle_a2a_moe`` disables both
phases there for full-graph capture, which does not apply to the Kunlun BCG path
(the DeepEP NORMAL a2a runs as an eager node). See the hook for the details.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.srt.arg_groups.cuda_graph_hook.handle_cuda_graph_config",
    type=HookType.AROUND,
)
def handle_cuda_graph_config_kunlun(original, server_args, *args, **kwargs):
    """Serve an unsupported prefill backend request with one Kunlun can capture.

    0.5.19 moved this out of ``ServerArgs`` into the arg-resolution pipeline
    (``arg_groups/cuda_graph_hook.py``), and resolution results are no longer
    plain field writes: the pipeline records decisions in the declaration stash
    (``server_args._resolved_overrides``) and every reader projects them through
    ``resolving_view``. So the hook reads the resolved config the same way the
    pipeline's own steps do and writes its decision back through
    ``declare_resolution`` -- mutating ``server_args.cuda_graph_config`` would
    land on a field nothing reads (it is ``None`` after resolution).
    """
    result = original(server_args, *args, **kwargs)

    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        with_phase,
    )

    config = resolving_view(server_args).cuda_graph_config
    if config is not None and config.prefill.backend == Backend.TC_PIECEWISE:
        logger.warning(
            "Prefill CUDA graph backend 'tc_piecewise' is not available on Kunlun "
            "(no torch.compile capture path); using 'breakable' with the configured "
            "capture sizes instead."
        )
        declare_resolution(
            server_args,
            "kunlun_handle_cuda_graph_config",
            cuda_graph_config=with_phase(
                config, Phase.PREFILL, backend=Backend.BREAKABLE
            ),
        )
        config = resolving_view(server_args).cuda_graph_config
    if config is not None and config.prefill.backend == Backend.BREAKABLE:
        _enable_cp_v2_for_dsv4_interleave(server_args)
    else:
        # [fix] CP-v2 (and with it the DSA prefill CP row split) must not depend on
        # the prefill CUDA graph being breakable. With --disable-prefill-cuda-graph
        # the backend resolves to DISABLED, this branch was skipped, SGLANG_ENABLE_CP_V2
        # stayed False, the row split was switched off and the C4 indexer read the full
        # sequence's pages instead of the CP-local shard (~8x more device work;
        # 8192-chunk 11.5k -> 1.5k tok/s).
        _enable_cp_v2_for_dsv4_interleave(server_args)
    return result


def _cp_prefill_requested(cfg, server_args) -> bool:
    """Whether this run asks for prefill context parallelism.

    0.5.19 keeps the deprecated ``--enable-dsa-prefill-context-parallel`` /
    ``--enable-prefill-context-parallel`` aliases alive and resolves them into
    ``enable_prefill_cp`` only as a *declaration*, so read all three names from
    both the resolving view and the raw record.
    """

    for name in (
        "enable_prefill_cp",
        "enable_prefill_context_parallel",
        "enable_dsa_prefill_context_parallel",
    ):
        if getattr(cfg, name, False) or getattr(server_args, name, False):
            return True
    return False


def _enable_cp_v2_for_dsv4_interleave(server_args) -> None:
    """Put a breakable-capture DSV4 prefill CP run on the CP-v2 rails."""
    from sglang.srt.arg_groups.overrides import model_config_of, resolving_view
    from sglang.srt.environ import envs

    if envs.SGLANG_ENABLE_CP_V2.get():
        return
    cfg = resolving_view(server_args)
    if not (
        _cp_prefill_requested(cfg, server_args)
        and getattr(cfg, "cp_strategy", None) == "interleave"
    ):
        return
    # 0.5.19 dropped ServerArgs.get_model_config(); the pipeline's own helper is
    # the resolver-safe way to read the model config.
    architectures = model_config_of(server_args).hf_config.architectures or []
    if not any(arch.startswith("DeepseekV4") for arch in architectures):
        return

    envs.SGLANG_ENABLE_CP_V2.set(True)
    logger.warning(
        "Enabling CP-v2 for %s prefill CP (cp_strategy=interleave) because a "
        "breakable prefill CUDA graph was requested: the graph gate and the "
        "CP-local capture inputs need it. The model keeps sharding its own "
        "rows; the CP-v2 strategy shard/gather stay inert.",
        "/".join(architectures),
    )


@plugin_hook(
    "sglang.srt.arg_groups.moe_hook.handle_a2a_moe",
    type=HookType.AROUND,
)
def handle_a2a_moe_deepep_normal_kunlun(original, server_args, *args, **kwargs):
    """Keep an explicitly requested breakable prefill graph under DeepEP NORMAL.

    0.5.19 keeps the same rule (``deepep_mode=normal`` disables both phases) and
    the same ``server_args._cuda_graph_config_locked`` marker, but the handler is
    the free function ``arg_groups.moe_hook.handle_a2a_moe`` and the decision
    travels through the declaration stash (see the hook above).
    """
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        with_phase,
    )

    config_before = resolving_view(server_args).cuda_graph_config
    locked = getattr(server_args, "_cuda_graph_config_locked", set())
    requested_breakable = (
        config_before is not None
        and (Phase.PREFILL, "backend") in locked
        and config_before.prefill.backend == Backend.BREAKABLE
    )

    result = original(server_args, *args, **kwargs)

    config_after = resolving_view(server_args).cuda_graph_config
    if (
        requested_breakable
        and config_after is not None
        and config_after.prefill.backend != Backend.BREAKABLE
    ):
        declare_resolution(
            server_args,
            "kunlun_handle_a2a_moe",
            cuda_graph_config=with_phase(
                config_after, Phase.PREFILL, backend=Backend.BREAKABLE
            ),
        )
        logger.warning(
            "Keeping the explicitly requested breakable prefill CUDA graph with "
            "deepep_mode=`normal`; the DeepEP NORMAL a2a runs as an eager graph "
            "break."
        )
    return result
