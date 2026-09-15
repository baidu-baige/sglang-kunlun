"""Plugin hooks for the DeepSeek-V4 FP16/BF16 index cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels.dsv4_mixed_kv import (
    GROUP_SIZE as _INT8_GROUP_SIZE,
    NOPE_DIM as _INT8_NOPE_DIM,
    ROPE_DIM as _INT8_ROPE_DIM,
    STRIDE as _INT8_STRIDE,
    move_scale_to_tail,
)

# DSV4-Flash int8 mixed tiling: 448 nope + 64 rope, nope quantized in groups
# of 64 (fp32 scale per group). The stride has to match the pool allocation
# (``dsv4_create_buffer_kunlun``) and the backend's ``reshape(-1, stride)``.


@plugin_hook(
    target="sglang.kernels.ops.attention.dsv4.index_buf_accessor.NopeFp8RopeBf16Pack",
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
        elif self.kv_cache_dtype == torch.int8:
            assert self.k_nope_fp8.shape[-1] == 448 + 64
            assert self.k_rope_bf16 is None
            assert self.scale_k_nope_ue8m0 is None
            assert self.k_nope_fp8.dtype == torch.bfloat16
        else:
            assert self.k_nope_fp8.shape[-1] == 448
            assert self.k_rope_bf16.shape[-1] == 64
            assert self.scale_k_nope_ue8m0.shape[-1] == 7

    def slice_pack(self, _slice: Any) -> "NopeFp8RopeBf16Pack":
        """Return the selected rows while preserving the cache representation."""
        is_half_cache = self.kv_cache_dtype in (
            torch.bfloat16,
            torch.float16,
            torch.int8,
        )
        return NopeFp8RopeBf16Pack(
            k_nope_fp8=self.k_nope_fp8[_slice],
            k_rope_bf16=self.k_rope_bf16[_slice] if not is_half_cache else None,
            scale_k_nope_ue8m0=(
                self.scale_k_nope_ue8m0[_slice] if not is_half_cache else None
            ),
            kv_cache_dtype=self.kv_cache_dtype,
        )


@plugin_hook(
    target="sglang.kernels.ops.attention.dsv4.index_buf_accessor._set_k_and_s_triton",
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
    if (
        nope_fp8_rope_bf16_pack.kv_cache_dtype == torch.int8
        and buf.shape[1] == page_size * _INT8_STRIDE
    ):
        # Both conditions are required. ``buf.shape[1]`` alone would misfire if
        # some half-cache pool ever happened to be _INT8_STRIDE bytes wide, and
        # then an fp16/bf16 buf would be viewed as int8 and silently written in
        # the wrong layout -- garbled output rather than a crash.
        import kunlun_ops

        packed = k_val.to(torch.bfloat16).contiguous()
        flat = buf.view(torch.int8).reshape(-1, _INT8_STRIDE)
        # Clamp both ends like ``dsv4_set_k_and_s_kunlun`` does for bf16/fp16:
        # swa_loc can carry a stale/garbage value, and an unclamped negative here
        # would wrap to the tail rows of the page table.
        loc_i32 = (
            loc.clamp(min=0, max=flat.shape[0] - 1).to(torch.int32).contiguous()
        )
        kunlun_ops.quantize_mla_kv_cache_split(
            packed,
            loc_i32,
            flat,
            _INT8_NOPE_DIM,
            _INT8_ROPE_DIM,
            _INT8_GROUP_SIZE,
        )
        # The kernel emits nope(int8) + scale first and rope(bf16) after it, so
        # that tail has to be rotated into the [nope 448][rope 128][scale 28]
        # order ``mixed_int8_bytes_per_token`` defines.
        move_scale_to_tail(flat, loc_i32)
        return
    assert k_val.dtype in (torch.bfloat16, torch.float16), (
        f"in this kernel, kv dtype is bf16/fp16, got {k_val.dtype}"
    )
    torch.ops.xspeedgate_ops.set_k_and_s_v4(buf, loc, k_val, page_size)
