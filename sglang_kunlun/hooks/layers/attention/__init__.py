"""Attention hooks for Kunlun XPU.

Side effects on import:
1. Triton driver pre-shim (backend="cuda") for fla platform detection.
2. Imports attention_registry (registers kunlun / kunlun_compressed / kunlun_nsa).
3. Imports nsa, fla submodules for hook side effects.
"""

from __future__ import annotations

import logging

from sglang_kunlun import _kunlun_pre_shim

logger = logging.getLogger(__name__)

# 把 Triton active driver 的 get_current_target().backend 伪装成 "cuda"，避免crash
_kunlun_pre_shim()

# Submodules with their own hook / setattr-time side effects.
from . import attention_registry  # noqa: E402,F401
from . import kunlun_deepseek_v4_backend  # noqa: E402,F401
from . import nsa  # noqa: E402,F401
from . import fla  # noqa: E402,F401
