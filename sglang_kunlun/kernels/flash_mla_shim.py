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
"""Kunlun adapter for ``sgl_kernel.flash_mla``.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Optional, Tuple

import torch

@dataclasses.dataclass
class FlashMLASchedMeta:
    """Sentinel scheduler-metadata object.

    The pure-torch path does no tile scheduling, so this carries no state.
    Mirrors the real ``sgl_kernel.flash_mla.FlashMLASchedMeta`` surface used by
    the backend (which only ever stores it and passes it back in).
    """

    have_initialized: bool = False
    config: Optional[object] = None
    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None


def get_mla_metadata(*args, **kwargs) -> Tuple["FlashMLASchedMeta", None]:
    """Return a sentinel; matches ``flash_mla.get_mla_metadata()[0]`` usage."""
    return FlashMLASchedMeta(), None


def install() -> None:
    """Bind the pure-torch shim onto the plugin's ``sgl_kernel.flash_mla`` stub."""
    mod = sys.modules.get("sgl_kernel.flash_mla")
    if mod is None:
        import sgl_kernel  # noqa: F401  (installs stub package)

        mod = sys.modules.get("sgl_kernel.flash_mla")
    if mod is None:
        return
    mod.FlashMLASchedMeta = FlashMLASchedMeta
    mod.get_mla_metadata = get_mla_metadata
    # Also expose on the root package for ``import sgl_kernel.flash_mla as fm``.
    root = sys.modules.get("sgl_kernel")
    if root is not None:
        root.flash_mla = mod
