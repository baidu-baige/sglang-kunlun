"""C's XSpeedGate operators, shipped as a second extension module.

The installed ``xspeedgate_ops`` wheel is a prebuilt binary and is the ONLY
place `sparse_mixed_attn_int8` exists (see HANDOVER_TPOT.md section 2), so it
must never be reinstalled. The XSpeedGate entry point registers with
``TORCH_LIBRARY_FRAGMENT_EXPAND(TORCH_EXTENSION_NAME, m)``, and
``TORCH_EXTENSION_NAME`` is the last segment of the extension name, so a second
``xspeedgate_ops.cpython-*.so`` loaded from a DIFFERENT package adds its
operators to the same ``torch.ops.xspeedgate_ops`` namespace with the installed
wheel still loaded. Measured: 256 ops -> 257, nothing lost.

Two naming rules, both load-bearing:
  * the shared object's file name must stay ``xspeedgate_ops.cpython-*.so``
    (that is what ``PyInit_xspeedgate_ops`` is called), and
  * the *package* directory must NOT be named ``xspeedgate_ops``, or it shadows
    the installed wheel and the process ends up with only these operators.

Import is best effort: without it the plugin runs the torch fallbacks, so a
missing or ABI-mismatched module must never fail a boot.
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
