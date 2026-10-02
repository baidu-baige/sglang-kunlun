"""Per-operator switches for validated Kunlun kernel replacements."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_KUNLUN = "kunlun"
_NATIVE = "native"

# Promote an operator only after its interface and four-layer gates pass.
_DEFAULTS = {
    # Keep native selected until the candidate survives both validation gates.
    # A native XPU crash cannot be recovered by the Python bridge fallback.
    "MLA_PREFILL": _NATIVE,
    "MLA_DECODE": _NATIVE,
    "MLA_DECODE_Q_CACHE": _NATIVE,
}

_warned: set[str] = set()
_traced: set[str] = set()


def use_kunlun_op(name: str) -> bool:
    """Return whether the named Kunlun operator is selected."""
    if name not in _DEFAULTS:
        raise KeyError(
            f"unknown Kunlun op switch {name!r}, expected one of {sorted(_DEFAULTS)}"
        )
    variable = f"SGLANG_KUNLUN_OP_{name}"
    value = os.getenv(variable, _DEFAULTS[name]).strip().lower()
    if value not in (_KUNLUN, _NATIVE):
        raise ValueError(
            f"{variable} must be {_KUNLUN!r} or {_NATIVE!r}, got {value!r}"
        )
    return value == _KUNLUN


def warn_once(key: str, message: str) -> None:
    """Log one fallback diagnostic per process and operator."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message)


def trace_once(key: str, message: str) -> None:
    """Log one success-path trace proving that the candidate really ran."""
    if key in _traced:
        return
    _traced.add(key)
    logger.warning("[op-trace] %s", message)
