"""Kunlun method hook for the upstream DeepSeek V4 NextN model."""

from __future__ import annotations

import os

import torch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _record_mtp_nextn_probe(name, tensor):
    if (
        os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR") is None
        or not isinstance(tensor, torch.Tensor)
        or tensor.shape[0] != 1
    ):
        return
    backend = get_attn_backend()
    step = getattr(backend, "speculative_step_id", 0)
    prefix = f"step{step}_layer0.nextn"
    probe = getattr(backend, "_mtp_tensor_probe", {})
    probe[f"{prefix}.{name}"] = tensor.clone()
    backend._mtp_tensor_probe = probe


@plugin_hook(
    "sglang.srt.models.deepseek_v4_nextn.DeepseekV4ModelNextN.hc_head",
    type=HookType.REPLACE,
)
def hc_head_kunlun(
    self,
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
):
    """Run the fused Kunlun MHC head and preserve the empty-tensor contract."""
    if x.numel() == 0:
        return x.new_empty((0, x.shape[-1]))
    return torch.ops.xspeedgate_ops.mhc_head(
        x.contiguous(),
        hc_fn,
        hc_scale,
        hc_base,
        self.rms_norm_eps,
        self.hc_eps,
    )
