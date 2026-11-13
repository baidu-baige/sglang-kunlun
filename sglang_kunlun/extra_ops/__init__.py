"""XSpeedGate operators shipped as a second extension module.

The installed wheel is the only place `sparse_mixed_attn_int8` exists, so it is
never reinstalled. The extension registers with `TORCH_LIBRARY_FRAGMENT_EXPAND`,
so this module adds its operators to `torch.ops.xspeedgate_ops` alongside the
installed wheel: keep the .so file name `xspeedgate_ops.cpython-*.so`, but do
not name this package `xspeedgate_ops` or it shadows the wheel.

Import is best effort; without it the plugin runs the torch fallbacks.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Operators this module is expected to add, for the post-import audit.
EXPECTED_OPS = ("fused_dsa_metadata_copy", "gather_spec_extras")

loaded = False

try:
    from . import xspeedgate_ops  # noqa: F401  # registers the fragment
except Exception as exc:  # noqa: BLE001 - absent/mismatched module is legitimate
    logger.info(
        "sglang-kunlun: extra XSpeedGate operators not loaded (%s: %s); "
        "the torch fallbacks stay in use.",
        type(exc).__name__,
        exc,
    )
else:
    loaded = True


def registered_ops() -> set:
    """Which of :data:`EXPECTED_OPS` are actually on ``torch.ops`` now."""
    import torch

    names = {schema.name for schema in torch._C._jit_get_all_schemas()}
    return {op for op in EXPECTED_OPS if f"xspeedgate_ops::{op}" in names}
