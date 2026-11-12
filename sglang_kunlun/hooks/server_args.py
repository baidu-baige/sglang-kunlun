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


@plugin_hook(
    "sglang.srt.arg_groups.model_hook._dsa_kv_cache_dtype_default",
    type=HookType.AROUND,
)
@plugin_hook(
    "sglang.srt.arg_groups.overrides._dsa_kv_cache_dtype_default",
    type=HookType.AROUND,
)
def tolerate_kunlun_int8_kv_cache(original, view, *args, **kwargs):
    """Let the Kunlun int8 KV cache through the DSA dtype gate.

    ``_dsa_kv_cache_dtype_default`` asserts bf16/fp8_e4m3 because that is what
    the CUDA DSA kernels accept. Kunlun's ``KunlunDSAAttnBackend`` carries its
    own int8 KV path -- the 0.5.8 fork serves GLM-5.3 w4a8 with
    ``--kv-cache-dtype int8`` -- so on this platform the value is legal and
    needs no defaulting; returning ``{}`` leaves it untouched. This mirrors
    what upstream already does for XPU and NPU (``overrides.py:500-508``),
    which Kunlun cannot use because it registers as ``PlatformEnum.OOT`` with
    ``device_type="cuda"``, so ``get_platform().is_xpu`` is False.

    **Two targets on purpose.** ``arg_groups/model_hook.py:11-14`` does
    ``from sglang.srt.arg_groups.overrides import _dsa_kv_cache_dtype_default``
    at import time and then calls that captured reference
    (``run_post_process_pass(server_args, _dsa_kv_cache_dtype_default)``,
    ``model_hook.py:343``). Patching only the defining module left that stale
    binding pointing at the original, and the boot died with
    ``AssertionError: DeepSeek DSA only supports bf16/bfloat16 or fp8_e4m3
    kv_cache_dtype`` even though the log showed the hook applied. The registry's
    ``_propagate_patch`` is supposed to rebind such references, but it only
    walks modules already in ``sys.modules`` when the hook lands, and whether
    ``model_hook`` is imported by then depends on import order. Registering the
    consumer's own attribute as a second target makes it order-independent.
    """
    # Both values `configure_kv_cache_dtype_kunlun` can map onto a Kunlun pool
    # (`hooks/mem_cache/common.py`): int8 (the 0.5.8 production value) and fp16
    # (the model dtype). Anything else keeps upstream's verdict.
    if getattr(view, "kv_cache_dtype", None) in ("int8", "fp16"):
        return {}
    return original(view, *args, **kwargs)


@plugin_hook(
    "sglang.srt.arg_groups.overrides._tp_lm_head_all_to_all_default",
    type=HookType.AROUND,
)
def decline_tp_lm_head_all_to_all_kunlun(original, view, *args, **kwargs):
    """Keep the decode LM head on the communication-free DP path.

    0.5.23 added `_tp_lm_head_all_to_all_default` (`overrides.py:1400-1418`),
    which *infers* `enable_tp_lm_head_all_to_all` rather than defaulting it off:
    it turns on for exactly this topology -- `disaggregation_mode == "decode"`,
    `attn_dp_size > 1`, `tp_size == attn_dp_size`, `attn_cp_size == 1`, no
    `enable_dp_lm_head`. Nobody passes the flag; upstream derives it. The
    capture path then demands a PyNCCL communicator
    (`RuntimeError: --enable-tp-lm-head-all-to-all requires an available
    PyNCCL communicator for CUDA graph capture.`), and this platform has none:
    the Kunlun `init_model_parallel_group` REPLACE hardcodes `use_pynccl=False`
    because the pynccl collectives are not available here
    (`hooks/distributed/parallel_state.py:93` and its comment block).

    So the feature is structurally impossible on Kunlun, and declining it
    restores the 0.5.8 production behaviour -- a vocab-parallel LM head sharded
    over the full TP group with no all-to-all. An explicit CLI value still
    wins: upstream returns `{}` early when the user set it, and this hook only
    overrides the *inferred* value.
    """
    result = original(view, *args, **kwargs)
    if result.get("enable_tp_lm_head_all_to_all"):
        logger.info(
            "Kunlun: declining the inferred --enable-tp-lm-head-all-to-all; "
            "this platform runs without a PyNCCL communicator, so the capture "
            "path cannot use it. The LM head stays vocab-parallel over the TP "
            "group, as the 0.5.8 production configuration runs it."
        )
        return {"enable_tp_lm_head_all_to_all": False}
    return result
