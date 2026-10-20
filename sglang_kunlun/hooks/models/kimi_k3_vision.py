"""Kimi-K3 vision tower hooks for Kunlun —— native 实现 + 按图大小选注意力后端。

昆仑 XPU 上 triton 退化为普通函数，故视觉侧算子给 native torch 实现：
1. ViT 融合 QK complex RoPE（apply_fused_qk_complex_rope / precompile_...）：
   native 交织复数旋转，与 triton kernel 及上游 kimi_k3_vl.apply_rope 等价。
2. 图像 normalize+patchify（normalize_and_patchify）：复用上游 native
   _normalize_and_patchify_torch。

ViT 注意力：**不再需要 --mm-attention-backend**。按 sglang 源码同款 shape-aware 逻辑分流：
大图（max_seqlen>1536 或 max_seqlen*total_tokens>=3e6）-> kunlun_mm，小图 -> sdpa。
kunlun_mm 是昆仑大图注意力后端的命名占位（对标 B300 的 fa4），**当前用分段 SDPA 实现**，
待昆仑 fa4 等价算子就位后，只需改 VisionKunlunMMAttention.forward 内部即可替换。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


# ========================= ViT RoPE =========================
def _native_fused_rope(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """native 交织复数 RoPE，等价 triton _fused_qk_complex_rope_kernel / 上游 apply_rope。"""
    fc = freqs_cis.unsqueeze(-2)  # (..., 1, head_dim//2) 对 head 广播
    q_c = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_c = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    q_out = torch.view_as_real(q_c * fc).flatten(-2)
    k_out = torch.view_as_real(k_c * fc).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


@plugin_hook(
    "sglang.kernels.ops.attention.vision_rope.apply_fused_qk_complex_rope",
    type=HookType.REPLACE,
)
def kunlun_apply_fused_qk_complex_rope(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ """
    return _native_fused_rope(q, k, freqs_cis)


@plugin_hook(
    "sglang.kernels.ops.attention.vision_rope.precompile_fused_qk_complex_rope",
    type=HookType.REPLACE,
)
def kunlun_precompile_fused_qk_complex_rope(
    *, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
) -> bool:
    """Native 预热（避开 XPU 不支持的 complex fill），真正跑一次 native RoPE。"""
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    qkv = torch.zeros((1, 3, num_heads, head_dim), dtype=dtype, device=device)
    q, k, _ = torch.unbind(qkv, dim=1)
    freqs = torch.view_as_complex(
        torch.ones((1, head_dim // 2, 2), dtype=torch.float32, device=device)
    )
    _native_fused_rope(q, k, freqs)
    return True


# ========================= 图像 patchify =========================
@plugin_hook(
    "sglang.kernels.ops.mm.process.image.normalize_and_patchify",
    type=HookType.REPLACE,
)
def kunlun_normalize_and_patchify(
    image, image_scale, image_bias, patch_size, padded_height, padded_width
):
    """强制走上游 native torch patchify，避开 triton kernel（与之数值等价）。"""
    from sglang.kernels.ops.mm.process.image import _normalize_and_patchify_torch

    return _normalize_and_patchify_torch(
        image, image_scale, image_bias, patch_size, padded_height, padded_width
    )


# ========================= ViT 注意力：大图 kunlun_mm / 小图 sdpa =========================
class VisionKunlunMMAttention(nn.Module):
    """昆仑大图 ViT 注意力后端（命名占位，对标 B300 的 fa4）。

    当前用上游分段 SDPA（sdpa_varlen_attention）实现，数值与小图 sdpa 一致；
    待昆仑 fa4 等价变长注意力算子就位后，仅替换本 forward 内部即可。
    """

    def __init__(self, **kwargs) -> None:  # 接收 use_data_parallel / workspace_buffer 等
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens=None,
        bsz: int = 1,
        seq_len: Optional[int] = None,
        forward_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        """变长 ViT 注意力：返回 [tot, H, D]。"""
        from sglang.srt.models.kimi_k3_vl import sdpa_varlen_attention

        cu = cu_seqlens
        if cu is None and forward_metadata is not None:
            cu = forward_metadata.cu_seqlens
        out = sdpa_varlen_attention(q, k, v, cu)  # [tot, H*D]
        return out.view(q.shape[0], q.shape[1], q.shape[2])  # [tot, H, D]


@plugin_hook(
    "sglang.srt.models.kimi_k3_vl._resolve_mm_attention_backend",
    type=HookType.REPLACE,
)
def kunlun_resolve_mm_attention_backend(
    configured_backend, *, max_seqlen, total_tokens, device, fa4_available=None
):
    """昆仑同款 shape-aware 选择：auto 时大图 kunlun_mm、小图 sdpa；显式后端则透传。"""
    if configured_backend != "auto":
        return configured_backend
    from sglang.srt.models.kimi_k3_vl import (
        _SM103_FA4_MIN_ATTENTION_WORK as _WORK,
        _SM103_TRITON_MAX_SEQLEN as _MAXLEN,
    )

    is_big = max_seqlen > _MAXLEN or max_seqlen * total_tokens >= _WORK
    return "kunlun_mm" if is_big else "sdpa"


@plugin_hook(
    "sglang.srt.models.kimi_k3_vl.MoonViTEncoderLayer.__init__",
    type=HookType.AROUND,
)
def kunlun_moonvit_layer_init(original_fn, self, *args, **kwargs):
    """让 ViT block 构造出 kunlun_mm 后端模块（auto/sdpa 时都建它，供大图路径使用）。

    注意：仅改本层的 attention_backend（影响 ModuleDict 构造）；按图大小的 per-forward
    选择仍由 encoder 的 attention_backend='auto' 驱动 _resolve_mm_attention_backend。
    """
    from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL

    QKV_BACKEND_IMPL.setdefault("kunlun_mm", VisionKunlunMMAttention)
    if kwargs.get("attention_backend") in (None, "auto", "sdpa"):
        kwargs["attention_backend"] = "kunlun_mm"
    return original_fn(self, *args, **kwargs)
