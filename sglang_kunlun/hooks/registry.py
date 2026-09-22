"""General plugin entry point for Kunlun hooks.

``register_all()`` imports each module that owns ``@plugin_hook(...)``
registrations. The order is intentional: foundational hooks and attention
module redirects must be registered before higher-level subsystem hooks.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

HOOK_MODULES = (
    "sglang_kunlun.hooks.utils.common",
    "sglang_kunlun.hooks.layers",
    "sglang_kunlun.hooks.mem_cache",
    "sglang_kunlun.hooks.model_executor",
    "sglang_kunlun.hooks.distributed",
    "sglang_kunlun.hooks.constrained",
    "sglang_kunlun.hooks.disaggregation",
    "sglang_kunlun.models",
    "sglang_kunlun.hooks.models",
    "sglang_kunlun.hooks.speculative",
    "sglang_kunlun.hooks.production_precision",
    "sglang_kunlun.hooks.mtp_production",
    "sglang_kunlun.hooks.ragged_draft_extend",
    "sglang_kunlun.hooks.function_call",
    "sglang_kunlun.hooks.parser",
)

# Diagnostics live outside ``sglang_kunlun``. They register last so that no
# production hook can resolve against a probe-wrapped function, and they are
# optional: a deployment without the ``debug`` package simply skips them.
DEBUG_HOOK_MODULES = (
    "debug.tensor_dump_hooks",
    "debug.dsv4_mtp_nextn_probes",
    "debug.mtp_alignment_plugin",
    "debug.ragged_draft_attention_probe",
    # T002: CP × MoE 四个观测点探针。只在 DSV41_CP_PROBE_DIR 设置时才注册 hook，
    # 未设置时模块本身不 import torch/sglang、完全空转。
    "debug.dsv41_cp_moe_probe",
)


def register_all() -> None:
    """Import every hook module so decorators register with ``HookRegistry``."""
    from sglang_kunlun.bootstrap import _kunlun_pre_shim
    from sglang_kunlun.bootstrap import env_compat

    _kunlun_pre_shim()
    env_compat.install()

    # Gate off Kimi-K3 fused features that have no Kunlun equivalent. Must run
    # before any ServerArgs/model code reads the corresponding env variables.
    from sglang_kunlun.platform.k3_capabilities import disable_unsupported_k3_features

    disable_unsupported_k3_features()
    from sglang_kunlun.kernels import deep_geem_hook as _deep_geem_hook  # noqa: F401
    from sglang_kunlun.kernels import flashinfer_hook as _flashinfer_hook  # noqa: F401
    from sglang_kunlun.kernels import k3_conv_registry as _k3_conv_registry  # noqa: F401
    from sglang_kunlun.kernels import kernel_ops, kernel_registry

    # V4.1 low-ratio (compress_ratio 1/2) kernel replacements. Must be imported
    # before kernel_ops.install(), which snapshots the registration tables.
    from sglang_kunlun.kernels import dsv41_low_ratio as _dsv41_low_ratio  # noqa: F401
    from sglang_kunlun.kernels import dsv41_mhc as _dsv41_mhc  # noqa: F401

    # Must precede kernel_ops.install(): ``sglang.kernels.selector`` caches
    # resolved callables (``lru_cache`` on ``_resolve``), so the override has to
    # be in place before any op is resolved for the first time.
    kernel_registry.install()

    legacy_jit_specs = [
        key
        for key, spec in kernel_ops._JIT_OPS.items()
        if spec.module_path.startswith("sglang.jit_kernel.")
    ]
    for key in legacy_jit_specs:
        kernel_ops._JIT_OPS.pop(key)
    if legacy_jit_specs:
        logger.info(
            "sglang-kunlun: ignored %d removed sglang.jit_kernel registrations",
            len(legacy_jit_specs),
        )
    kernel_ops.install()
    kernel_registry.log_resolution_report_once()
    for module_name in HOOK_MODULES:
        importlib.import_module(module_name)
    debug_modules = 0
    for module_name in DEBUG_HOOK_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            logger.debug("sglang-kunlun: debug hooks unavailable: %s", module_name)
        else:
            debug_modules += 1
    logger.info(
        "sglang-kunlun: %d hook modules registered, %d debug hook modules "
        "registered, %d triton ops, %d jit ops and %d kernel-registry overrides "
        "installed",
        len(HOOK_MODULES),
        debug_modules,
        len(kernel_ops.registered_triton_ops()),
        len(kernel_ops.registered_jit_ops()),
        len(kernel_registry.registered_overrides()),
    )
