# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Kunlun device operations shared by SGLang platform interfaces."""

from __future__ import annotations

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum


class KunlunDeviceMixin(DeviceMixin):
    """Kunlun device mixin."""

    _enum = PlatformEnum.OOT
    device_name = "kunlun"
    device_type = "cuda"

    def get_dispatch_key_name(self) -> str:
        """Return the MultiPlatformOp forward_cuda"""
        return "cuda"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """Get device total memory."""
        import torch

        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """Make everything deterministic."""
        if seed is None:
            return
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
