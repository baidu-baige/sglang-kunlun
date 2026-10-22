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
    return result


# NOTE: 0.5.22 removed ``SGLANG_ENABLE_CP_V2`` (strategy-based prefill context
# parallelism is the only generic implementation now), so the 0.5.19-era
# ``_enable_cp_v2_for_dsv4_interleave`` forcing is gone with it. The DSA CP row
# split is kept authoritative by ``hooks/layers/cp/bcg.py``
# (``is_dsa_enable_prefill_cp``), which reads the explicit
# ``--enable-dsa-prefill-context-parallel`` flag instead of an env switch.


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
