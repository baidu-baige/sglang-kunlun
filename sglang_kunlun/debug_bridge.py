"""Bridge from the production hooks to the optional ``debug`` package.

The precision-alignment probes live in ``<repo>/debug`` (outside the installed
package).  Production modules mark a probe site with a single guarded line::

    if _DEBUG:
        debug_attention.capture("attention.operator_begin", locals())

``DEBUG_ENABLED`` is resolved once at import time and is ``False`` unless the
debug package is present *and* the process exported one of its ``DSV4_*``
variables, so a normal serving process pays one boolean test per site.  The
handlers on the debug side then evaluate the same variables as the inlined code
did, write the same files and use the same payload keys.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_MODULE_NAME = "sglang_kunlun_debug"
_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "debug"


class _NoopNamespace:
    """Stand-in whose every attribute is a hook that does nothing."""

    @staticmethod
    def _noop(*args, **kwargs):
        return None

    def __getattr__(self, name):
        return self._noop


def _load_debug_package():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    init_file = _PACKAGE_DIR / "__init__.py"
    if not init_file.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        init_file,
        submodule_search_locations=[str(_PACKAGE_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - debug tooling must never break serving
        del sys.modules[_MODULE_NAME]
        logger.warning("ignoring unusable debug package at %s", _PACKAGE_DIR, exc_info=True)
        return None
    return module


_debug = _load_debug_package()
_noop_namespace = _NoopNamespace()

# Resolved once at import time: production probe sites are guarded by this flag,
# so a process that requested no probe never builds a ``locals()`` dictionary.
DEBUG_ENABLED = _debug is not None and _debug.probes_requested()

allocator = getattr(_debug, "allocator", _noop_namespace)
attention = getattr(_debug, "attention", _noop_namespace)
kernels = getattr(_debug, "kernels", _noop_namespace)
mem_cache = getattr(_debug, "mem_cache", _noop_namespace)


def install_mtp_nextn_probes() -> None:
    """Install the NextN MTP boundary probes when the debug package is present."""
    if _debug is not None:
        _debug.install_mtp_nextn_probes()
