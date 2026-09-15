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
from . import deepseek_v4_backend  # noqa: E402,F401