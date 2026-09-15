# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
