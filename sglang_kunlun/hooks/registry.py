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
    "sglang_kunlun.hooks.speculative",
    "sglang_kunlun.hooks.production_precision",
    "sglang_kunlun.hooks.mtp_production",
    "sglang_kunlun.hooks.ragged_draft_extend",
)

# Diagnostics live outside ``sglang_kunlun``. They register last so that no
# production hook can resolve against a probe-wrapped function, and they are
# optional: a deployment without the ``debug`` package simply skips them.
DEBUG_HOOK_MODULES = (
    "debug.tensor_dump_hooks",
    "debug.dsv4_mtp_nextn_probes",
)


def register_all() -> None:
    """Import every hook module so decorators register with ``HookRegistry``."""
    from sglang_kunlun.bootstrap import _kunlun_pre_shim

    _kunlun_pre_shim()
    from sglang_kunlun.kernels import deep_geem_hook as _deep_geem_hook  # noqa: F401
    from sglang_kunlun.kernels import flashinfer_hook as _flashinfer_hook  # noqa: F401
    from sglang_kunlun.kernels import kernel_ops

    kernel_ops.install()
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
        "registered, %d triton ops and %d jit ops installed",
        len(HOOK_MODULES),
        debug_modules,
        len(kernel_ops.registered_triton_ops()),
        len(kernel_ops.registered_jit_ops()),
    )
