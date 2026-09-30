"""Kunlun fla utils patch.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/attention/fla/__init__.py

Community ``get_available_device`` queries the triton driver to detect the
backend. On Kunlun (no triton driver), this raises and falls back to ``cpu``,
breaking ``_check_platform``. We force ``"cuda"`` so platform resolves to
``nvidia`` and the SYMBOL_REWRITE-backed CUDA path is taken.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# GLM-5.3-Flash-era sglang moved fla under sglang.kernels.ops; keep the legacy
# path as a fallback so the plugin works against both trees.
_FLA_UTILS_PATHS = (
    "sglang.kernels.ops.attention.fla.utils",
    "sglang.srt.layers.attention.fla.utils",
)


def _import_fla_utils():
    """Return the fla utils module from whichever path this sglang exposes."""

    for path in _FLA_UTILS_PATHS:
        try:
            return importlib.import_module(path), path
        except ImportError:
            continue
    raise ImportError(f"none of {_FLA_UTILS_PATHS} is importable")


try:
    import torch

    _fla_utils, _fla_utils_path = _import_fla_utils()

    def _get_available_device() -> str:
        return "cuda"

    _fla_utils.get_available_device = _get_available_device
    _fla_utils.device = "cuda"
    _fla_utils.device_torch_lib = getattr(torch, "cuda")
    _fla_utils.device_platform = "nvidia"
    _fla_utils.is_amd = False
    _fla_utils.is_intel = False
    logger.info("Kunlun: %s.get_available_device -> 'cuda'", _fla_utils_path)
except Exception as e:  # pragma: no cover
    logger.warning("fla utils kunlun patch failed: %s", e)
