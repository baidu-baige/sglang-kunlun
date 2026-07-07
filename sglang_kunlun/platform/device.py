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
