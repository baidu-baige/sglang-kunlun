"""Kunlun fla utils patch.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/attention/fla/__init__.py

Community ``get_available_device`` queries the triton driver to detect the
backend. On Kunlun (no triton driver), this raises and falls back to ``cpu``,
breaking ``_check_platform``. We force ``"cuda"`` so platform resolves to
``nvidia`` and the SYMBOL_REWRITE-backed CUDA path is taken.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


try:
    import torch
    import sglang.srt.layers.attention.fla.utils as _fla_utils

    def _get_available_device() -> str:
        return "cuda"

    _fla_utils.get_available_device = _get_available_device
    _fla_utils.device = "cuda"
    _fla_utils.device_torch_lib = getattr(torch, "cuda")
    _fla_utils.device_platform = "nvidia"
    _fla_utils.is_amd = False
    _fla_utils.is_intel = False
    logger.info("Kunlun: fla.utils.get_available_device -> 'cuda'")
except Exception as e:  # pragma: no cover
    logger.warning("fla utils kunlun patch failed: %s", e)
