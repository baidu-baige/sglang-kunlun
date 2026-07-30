"""Out-of-tree DSV4 precision-alignment debug hooks for sglang-kunlun.

This package holds every probe, dump and diagnostic log that was previously
inlined into the production modules of ``sglang_kunlun``.  It lives next to the
package (not inside it) for two reasons:

* ``pyproject.toml`` only packages ``sglang_kunlun*``, so nothing here ships in
  a wheel and production deployments cannot execute it;
* the production hot paths keep a single guarded one-liner per probe site.

Usage is unchanged from before the refactor: the same ``DSV4_*`` environment
variables enable the same probes, which write the same files with the same
payload keys.  Production modules reach this package through
``sglang_kunlun.debug_bridge``:

    if _DEBUG:
        debug_attention.capture("attention.operator_begin", locals())

``_DEBUG`` is resolved once at import time (see :func:`probes_requested`), so a
process that exports none of the variables below pays a single boolean test per
probe site and never builds a ``locals()`` dictionary.  Export ``DSV4_DEBUG=1``
to arm the probes that are driven by a module constant instead of an
environment variable (``attention.ENABLE_ACCURACY_DUMPS``,
``kernels.ENABLE_ACCURACY_DUMPS``).
"""

from __future__ import annotations

import os

from . import allocator, attention, kernels, mem_cache

__all__ = [
    "allocator",
    "attention",
    "kernels",
    "mem_cache",
    "PROBE_ENV_VARS",
    "probes_requested",
    "install_mtp_nextn_probes",
]

# Every environment variable that arms at least one probe.  Keeping the list
# here means the production side never mentions a DSV4_* name.
PROBE_ENV_VARS = (
    "DSV4_DEBUG",
    "DSV4_ACCURACY_DUMP_DIR",
    "DSV4_ALLOC_EXTEND_PROBE_DIR",
    "DSV4_C4_ATTN_METADATA_LOG",
    "DSV4_DECODE_ATTENTION_ALIAS_LAYER",
    "DSV4_DECODE_ATTENTION_ALIAS_OPERATOR_ONLY",
    "DSV4_DECODE_LAYER_DUMP",
    "DSV4_DECODE_PROBE_CACHE_ROWS",
    "DSV4_DECODE_PROBE_LAYERS",
    "DSV4_IFEVAL_MTP_DIAG_DIR",
    "DSV4_MTP_PROBE",
    "DSV4_MTP_TENSOR_DUMP_DIR",
    "DSV4_MTP_VERIFY_LAYER_DUMP",
    "DSV4_PREFILL_BACKEND_DEVICE_PROBE",
    "DSV4_PREFILL_BACKEND_PROBE",
)


def probes_requested() -> bool:
    """Return whether this process asked for any probe."""
    return any(os.environ.get(name) for name in PROBE_ENV_VARS)


def install_mtp_nextn_probes() -> None:
    """Install the single-token MTP boundary probes on the NextN modules."""
    from . import mtp_nextn

    mtp_nextn.install()
