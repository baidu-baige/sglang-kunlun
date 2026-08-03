"""Plugin hooks for the DeepSeek-V4 FP16/BF16 index cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    target="sglang.srt.layers.attention.dsv4.index_buf_accessor.NopeFp8RopeBf16Pack",
    type=HookType.REPLACE,
)
@dataclass
class NopeFp8RopeBf16Pack:
    """Store an unquantized half-precision key or the FP8/BF16 split pack."""

    k_nope_fp8: torch.Tensor
    k_rope_bf16: torch.Tensor | None
    scale_k_nope_ue8m0: torch.Tensor | None
    kv_cache_dtype: torch.dtype

    def __post_init__(self):
        if self.kv_cache_dtype in (torch.bfloat16, torch.float16):
            assert self.k_nope_fp8.shape[-1] == 448 + 64
            assert self.k_rope_bf16 is None
            assert self.scale_k_nope_ue8m0 is None
            assert self.k_nope_fp8.dtype is self.kv_cache_dtype
        else:
            assert self.k_nope_fp8.shape[-1] == 448
            assert self.k_rope_bf16.shape[-1] == 64
            assert self.scale_k_nope_ue8m0.shape[-1] == 7

    def slice_pack(self, _slice: Any) -> "NopeFp8RopeBf16Pack":
        is_half_cache = self.kv_cache_dtype in (torch.bfloat16, torch.float16)
        return NopeFp8RopeBf16Pack(
            k_nope_fp8=self.k_nope_fp8[_slice],
            k_rope_bf16=self.k_rope_bf16[_slice] if not is_half_cache else None,
            scale_k_nope_ue8m0=(
                self.scale_k_nope_ue8m0[_slice] if not is_half_cache else None
            ),
            kv_cache_dtype=self.kv_cache_dtype,
        )


@plugin_hook(
    target="sglang.srt.layers.attention.dsv4.index_buf_accessor._set_k_and_s_triton",
    type=HookType.REPLACE,
)
def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    page_size: int,
):
    """Store an unquantized FP16/BF16 key in the paged cache."""
    k_val = nope_fp8_rope_bf16_pack.k_nope_fp8
    assert k_val.dtype in (torch.bfloat16, torch.float16), (
        f"in this kernel, kv dtype is bf16/fp16, got {k_val.dtype}"
    )
    torch.ops.xspeedgate_ops.set_k_and_s_v4(buf, loc, k_val, page_size)
