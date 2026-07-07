"""Attention hooks for Kunlun XPU.

Side effects on import:
1. Triton driver pre-shim (backend="cuda") for fla platform detection.
2. NSA backend module redirect to Kunlun reimplementation.
3. Imports nsa, fla submodules for redirect/hook side effects.
"""

from __future__ import annotations

import logging
import sys

from sglang_kunlun import _kunlun_pre_shim

logger = logging.getLogger(__name__)


def _redirect_nsa_backend() -> None:
    """Substitute upstream nsa_backend module with our Kunlun impl."""
    target = "sglang.srt.layers.attention.nsa_backend"
    if target in sys.modules:
        # Upstream already imported — too late to redirect cleanly.
        logger.warning(
            "%s already imported before sglang_kunlun plugin; "
            "Kunlun NSA backend redirect skipped",
            target,
        )
        return
    try:
        from . import _nsa_backend_impl  # noqa: F401

        sys.modules[target] = _nsa_backend_impl
        logger.info("Redirected %s -> sglang_kunlun._nsa_backend_impl", target)
    except Exception as e:  # pragma: no cover
        logger.exception("Failed to redirect nsa_backend: %s", e)

# 把 Triton active driver 的 get_current_target().backend 伪装成 "cuda"，避免crash
_kunlun_pre_shim()
_redirect_nsa_backend()

# Submodules with their own hook / setattr-time side effects.
from . import attention_registry  # noqa: E402,F401
from . import nsa  # noqa: E402,F401
from . import fla  # noqa: E402,F401