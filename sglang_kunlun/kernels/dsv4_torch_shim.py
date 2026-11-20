# Copyright 2026 Baidu, Inc.
# Licensed under the Apache License, Version 2.0.
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
"""Pure-PyTorch replacements for DSV4 tvm_ffi JIT kernels on Kunlun XPU.

All kernels here are registered via register_jit_op() in kernel_ops.py so they
patch the upstream sglang.jit_kernel.dsv4.* symbols at import time.
"""
from __future__ import annotations

import contextvars
import functools
import math
from typing import Literal, Optional, Tuple, Union

import kunlun_ops
import os
import torch
import torch.nn.functional as F
import xspeedgate_ops  # noqa: F401  (registers torch.ops.xspeedgate_ops.*)

from sglang_kunlun.hooks.layers.attention.nsa.triton_kernel import (
    act_quant_kunlun,
)


# ---------------------------------------------------------------------------
# fused_q_norm_rope
# Upstream: sglang.jit_kernel.dsv4.elementwise.fused_q_norm_rope
# CUDA: FusedQNormRopeKernel — rmsnorm-self (no weight) + RoPE on last rope_dim
# ---------------------------------------------------------------------------

def _fused_q_norm_rope_torch(
    q_input: torch.Tensor,   # [B, num_q_heads, head_dim]
    q_output: torch.Tensor,  # [B, num_q_heads, head_dim]  (written in-place)
    eps: float,
    freqs_cis: torch.Tensor, # [max_pos, rope_dim//2] complex64
    positions: torch.Tensor, # [B] int32/int64
) -> None:
    """rmsnorm-self (no weight) + RoPE on last rope_dim dims."""
    # Kunlun fused RMSNorm(self) + GPT-J RoPE (dpsk_v4_norm_rope_gptj, mode=2).
    # norm_weight=ones => weightless RMSNorm; handle = per-row int64 positions
    # (position repeated across the H heads of each token). Torch fallback below.
    try:
        import kunlun_ops
        _B, _H, _D = q_input.shape
        _kv = q_input.reshape(_B * _H, _D).contiguous()
        _w = torch.ones(_D, dtype=_kv.dtype, device=_kv.device)
        _handle = positions.reshape(-1).to(torch.int64).repeat_interleave(_H).contiguous()
        kunlun_ops.dpsk_v4_norm_rope_gptj(_kv, _w, _handle, freqs_cis, 2, 0, float(eps))
        q_output.copy_(_kv.reshape(_B, _H, _D).to(q_output.dtype))
        return
    except Exception:
        pass
    q = q_input.float()  # [B, H, D]
    # rmsnorm-self: no weight
    rms = q.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    q = q * rms

    head_dim = q.shape[-1]
    rope_dim = freqs_cis.shape[-1] * 2  # complex -> real pairs

    freqs = freqs_cis[positions.long()]   # [B, rope_dim//2] complex64
    # Split q into nope and rope parts
    q_nope = q[..., :head_dim - rope_dim]   # [B, H, nope_dim]
    q_rope = q[..., head_dim - rope_dim:]   # [B, H, rope_dim]

    # Apply RoPE: complex multiply
    B, H, _ = q_rope.shape
    q_rope_c = q_rope.view(B, H, -1, 2)  # [B, H, rope_dim/2, 2]
    # freqs: [B, rope_dim//2] complex  -> [B, 1, rope_dim//2] complex
    f = freqs.unsqueeze(1)  # [B, 1, rope_dim//2]
    # view as float pairs
    f_r = f.real.unsqueeze(-1)  # [B, 1, rope_dim//2, 1]
    f_i = f.imag.unsqueeze(-1)
    xr = q_rope_c[..., :1]  # [B, H, rope_dim/2, 1]
    xi = q_rope_c[..., 1:]
    q_rope_rot = torch.cat([xr * f_r - xi * f_i, xr * f_i + xi * f_r], dim=-1)  # [B, H, rope_dim/2, 2]
    q_rope_rot = q_rope_rot.view(B, H, rope_dim)

    q_out = torch.cat([q_nope, q_rope_rot], dim=-1)
    q_output.copy_(q_out.to(q_input.dtype))


def _e4m3_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Quantize to FP8-E4M3 and return the raw uint8 byte representation.

    Kunlun XPU has no device-side float32->float8_e4m3fn conversion kernel, so
    the cast is done on CPU and only the resulting bytes are moved back.
    """
    x_bytes = (
        x.detach().float().clamp(-448.0, 448.0).cpu().to(torch.float8_e4m3fn).view(torch.uint8)
    )
    return x_bytes.to(x.device)


def _to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Quantize to a float8_e4m3fn tensor (bitcast from CPU-produced bytes)."""
    return _e4m3_to_uint8(x).view(torch.float8_e4m3fn)


def _fused_k_norm_rope_flashmla_torch(
    kv: torch.Tensor,        # [B, head_dim=512] or [B, q_lora_rank+head_dim] (non-contiguous)
    kv_weight: torch.Tensor, # [head_dim]
    eps: float,
    freqs_cis: torch.Tensor, # [max_pos, rope_dim//2] complex64
    positions: torch.Tensor, # [B] int32/int64
    out_loc: torch.Tensor,   # [B] int32 — cache slot index
    kvcache: torch.Tensor,   # [npages, page_bytes] uint8
    page_size: int,
) -> None:
    """rmsnorm WITH kv_weight + RoPE + write to FlashMLA paged cache.

    Page layout per token (576 bytes):
      bytes   0..447  : 448 bytes = 224 FP8-E4M3 pairs (nope part, warps 0-6)
      bytes 448..575  : 128 bytes =  64 BF16 pairs (rope part, warp 7)
    Scale layout in page tail:
      page_ptr + page_size*576 + in_page*8: 7 valid UE8M0 bytes (one per nope warp)
    """
    if kv.shape[0] == 0:
        return

    head_dim = 512
    rope_dim = 64
    nope_dim = head_dim - rope_dim  # 448

    # Ensure we have [B, head_dim] by taking the last head_dim columns
    if kv.shape[-1] != head_dim:
        kv_input = kv[..., -head_dim:].contiguous()
    else:
        kv_input = kv.contiguous()

    B = kv_input.shape[0]
    kv_f = kv_input.float()  # [B, 512]
    w_f = kv_weight.float()  # [512]

    # rmsnorm WITH weight
    rms = kv_f.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    kv_normed = kv_f * rms * w_f.unsqueeze(0)  # [B, 512]

    # Split nope / rope
    kv_nope = kv_normed[:, :nope_dim]   # [B, 448]
    kv_rope = kv_normed[:, nope_dim:]   # [B, 64]

    # Apply RoPE on rope part
    freqs = freqs_cis[positions.long()]  # [B, 32] complex64
    kv_rope_c = kv_rope.view(B, -1, 2)  # [B, 32, 2]
    f_r = freqs.real  # [B, 32]
    f_i = freqs.imag
    xr = kv_rope_c[:, :, 0]
    xi = kv_rope_c[:, :, 1]
    kv_rope_rot = torch.stack([xr * f_r - xi * f_i, xr * f_i + xi * f_r], dim=-1)  # [B, 32, 2]
    kv_rope_rot = kv_rope_rot.view(B, rope_dim)  # [B, 64]

    # === bf16 KV store (align-to-reference): write 512 bf16 via the native,
    # cuda-graph-capturable set_k_and_s_v4 (no fp8 quant / no python-loop byte
    # write). Active when the cache buffer is bf16/fp16 (--kv-cache-dtype bfloat16).
    if kvcache.dtype in (torch.bfloat16, torch.float16):
        kv_out = torch.cat([kv_nope, kv_rope_rot], dim=-1).to(kvcache.dtype).contiguous()  # [B,512]
        _np = kvcache.shape[0]
        _mx = _np * page_size - 1
        # out_loc is already int64 on this path; the unconditional cast was a
        # no-op ``aten::to`` dispatch per layer (80 per prefill step).
        _loc = out_loc if out_loc.dtype is torch.int64 else out_loc.to(torch.int64)
        _loc = _clamp_loc_i64(_loc, _mx)
        torch.ops.xspeedgate_ops.set_k_and_s_v4(kvcache, _loc, kv_out, page_size)
        return

    # FP8-E4M3 quantization for nope part: 7 warp groups of 64 elements
    # Each warp group: 64 elements -> 1 UE8M0 scale byte + 64 FP8 bytes
    nope_warp_groups = nope_dim // 64  # = 7
    kv_nope_reshaped = kv_nope.view(B, nope_warp_groups, 64)  # [B, 7, 64]

    # per-warp abs max
    abs_max = kv_nope_reshaped.abs().amax(dim=-1)  # [B, 7]
    abs_max = abs_max.clamp(min=1e-4)
    scale_raw = abs_max / 448.0  # FP8_E4M3_MAX = 448

    # UE8M0 encoding: byte = floor(log2(scale)) + 127
    log2_scale = torch.log2(scale_raw)
    scale_ue8m0 = (torch.floor(log2_scale) + 127.0).clamp(0, 254).to(torch.uint8)  # [B, 7]
    # inverse scale (quant multiplier). Dequant reconstructs value = fp8 * 2^(s-127),
    # so quantization must divide by that same rounded scale: inv = 2^(127 - s).
    inv_scale = (2.0 ** (127.0 - scale_ue8m0.float()))  # [B, 7]

    # quantize: divide by scale_raw, clamp, cast to float8_e4m3fn
    kv_nope_scaled = (kv_nope_reshaped * inv_scale.unsqueeze(-1)).clamp(-448, 448)  # [B, 7, 64]
    kv_nope_fp8 = _to_e4m3(kv_nope_scaled)  # [B, 7, 64]

    # Now write to kvcache
    page_bits = int(math.log2(page_size))
    page_mask = page_size - 1

    out_loc_i32 = out_loc.to(torch.int32)
    pages = (out_loc_i32 >> page_bits).long()    # [B]
    offsets = (out_loc_i32 & page_mask).long()   # [B]

    # page layout: kvcache[page, 576*page_size + 8*in_page:...] for scales
    # Each token occupies 576 bytes starting at offset*576 within the page.
    kvcache_u8 = kvcache.view(torch.uint8)  # [npages, page_bytes]
    page_bytes = kvcache.shape[1]

    for b in range(B):
        p = int(pages[b].item())
        off = int(offsets[b].item())
        # token data pointer = page * page_bytes + off * 576
        token_start = p * page_bytes + off * 576

        # Write nope part (FP8): bytes 0..447
        nope_fp8_flat = kv_nope_fp8[b].view(torch.uint8).flatten()  # 448 bytes
        kvcache_u8.view(-1)[token_start:token_start + nope_dim].copy_(nope_fp8_flat)

        # Write rope part (BF16): bytes 448..575
        rope_bf16 = kv_rope_rot[b].to(torch.bfloat16)  # [64] = 128 bytes
        rope_bytes = rope_bf16.view(torch.uint8)  # 128 bytes
        kvcache_u8.view(-1)[token_start + nope_dim:token_start + 576].copy_(rope_bytes)

        # Write UE8M0 scales at: page_ptr + page_size*576 + off*8
        scale_start = p * page_bytes + page_size * 576 + off * 8
        scales_b = scale_ue8m0[b]  # [7] uint8
        kvcache_u8.view(-1)[scale_start:scale_start + nope_warp_groups].copy_(scales_b)


# [host-diet r85] ``out_loc.clamp()`` below is one int64 allocation + one kernel
# per layer (80 per prefill forward) whose result dies with the next call, so it
# writes into a scratch instead.
#
# [rel-7th qps-oom] r85 那版按 ``(numel, device)`` 缓存：QPS 流量下每个不同的 B
# 各留一份 ``int64[numel]``（8·numel 字节）且永不释放，和 attention 的
# ``_attn_aux`` 是同一种无界增长（9 分钟约 370 个不同 B），会挤掉大请求里
# ``dsv41_cp_indexer._expand()`` 需要的连续大块（实测 ``torch.OutOfMemoryError``）。
# 改成**每个 device 一块 grow-only 缓冲**：按需从头部切出 n 元素的连续视图，
# 容量只增不减，稳定态只有一块 ``int64[max_numel]``。
_CLAMP_I64_CACHE: dict = {}
_CLAMP_OUT = os.environ.get("KL_CLAMP_OUT", "1").lower() not in (
    "0",
    "",
    "false",
    "off",
    "none",
)


def _grow_cap(old: int, need: int) -> int:
    """grow-only 容量：够用即可，否则按 1.25 倍留余量（别让每个新 B 都重分配）。"""
    return need if old <= 0 else max(need, (old * 5 + 3) // 4)


def _clamp_loc_i64(loc: torch.Tensor, upper: int) -> torch.Tensor:
    if not _CLAMP_OUT:
        return loc.clamp(min=0, max=upper)
    # 只有一个逻辑 tag（clamp 输出），key 就只留 device，沿用原来的 ``str(device)``。
    key = str(loc.device)
    n = loc.numel()
    buf = _CLAMP_I64_CACHE.get(key)
    if buf is None or buf.numel() < n:
        old = 0 if buf is None else buf.numel()
        # 先放掉旧缓冲的引用再分配：峰值是「一块」而不是「两块」。
        buf = None
        _CLAMP_I64_CACHE.pop(key, None)
        buf = torch.empty(_grow_cap(old, n), dtype=torch.int64, device=loc.device)
        _CLAMP_I64_CACHE[key] = buf
    # 连续前缀视图：1-D、stride 1，与旧的 ``torch.empty(n)`` 形状/步长完全一致。
    out = buf[:n]
    torch.clamp(loc, min=0, max=upper, out=out)
    return out


def _fused_k_norm_rope_flashmla_kunlun(
    kv: torch.Tensor,        # [B, head_dim=512] or [B, q_lora_rank+head_dim]
    kv_weight: torch.Tensor, # [head_dim]
    eps: float,
    freqs_cis: torch.Tensor, # [max_pos, rope_dim//2] complex64
    positions: torch.Tensor, # [B] int32/int64
    out_loc: torch.Tensor,   # [B] int32
    kvcache: torch.Tensor,   # [npages, page_bytes]
    page_size: int,
) -> None:
    """Kunlun-native fused K norm + RoPE + paged cache store.

    Step 1: ``sgl_kernel_kunlun.rmsnorm`` (with weight) → kv_normed [B, 512]
    Step 2: ``_apply_rope_xspeedgate`` on rope part [B, 64]
    Step 3: store to paged cache (bf16: ``set_k_and_s_v4``; FP8: torch quant+write)
    """
    if kv.shape[0] == 0:
        return

    head_dim = 512
    rope_dim = 64
    nope_dim = head_dim - rope_dim  # 448

    if kv.shape[-1] != head_dim:
        kv_input = kv[..., -head_dim:].contiguous()
    else:
        kv_input = kv.contiguous()

    B = kv_input.shape[0]

    # --- Step 1: RMSNorm with weight (reuse sgl_kernel_kunlun.rmsnorm) ---
    from sglang_kunlun.kernels.sgl_kernel_kunlun import rmsnorm as _kunlun_rmsnorm

    kv_normed = _kunlun_rmsnorm(kv_input, kv_weight, eps)


    # --- Step 2: RoPE on rope part (reuse _apply_rope_xspeedgate) ---
    kv_rope = kv_normed[:, nope_dim:]   # [B, 64] view of the normed row

    # flashinfer_rotary_embedding requires 3D query [B, H, rope_dim];
    # unsqueeze to [B, 1, 64].  Pass key=None.  ``inplace`` rotates the tail of
    # ``kv_normed`` directly (it is a private rmsnorm output), so no cat is
    # needed below: ``kv_normed`` already *is* the [B, 512] rope-applied row.
    _apply_rope_xspeedgate(
        kv_rope.unsqueeze(1), None, freqs_cis, positions, inverse=False, inplace=True,
    )


    # --- Step 3: store to paged cache ---
    if kvcache.dtype in (torch.bfloat16, torch.float16):
        kv_out = kv_normed if kv_normed.dtype == kvcache.dtype else kv_normed.to(kvcache.dtype)
        if not kv_out.is_contiguous():
            kv_out = kv_out.contiguous()
        _np = kvcache.shape[0]
        _mx = _np * page_size - 1
        # out_loc is already int64 on this path; the unconditional cast was a
        # no-op ``aten::to`` dispatch per layer (80 per prefill step).
        _loc = out_loc if out_loc.dtype is torch.int64 else out_loc.to(torch.int64)
        _loc = _clamp_loc_i64(_loc, _mx)
        torch.ops.xspeedgate_ops.set_k_and_s_v4(kvcache, _loc, kv_out, page_size)
        return

    raise NotImplementedError(
        f"_fused_k_norm_rope_flashmla_kunlun only supports bf16/fp16 kvcache, "
        f"got {kvcache.dtype}"
    )


# ---------------------------------------------------------------------------
# fused_rope_inplace
# Upstream: sglang.jit_kernel.dsv4.elementwise.fused_rope_inplace
# Applied after attention on the rope-part of output (inverse=True)
# ---------------------------------------------------------------------------

def _fused_rope_inplace_torch(
    q: torch.Tensor,           # [..., rope_dim]
    k: Optional[torch.Tensor], # [..., rope_dim] or None
    freqs_cis: torch.Tensor,   # [max_pos, rope_dim//2] complex64
    positions: torch.Tensor,   # [B] int32/int64
    inverse: bool = False,
) -> None:
    """Apply (or invert) RoPE in-place.

    CUDA-graph-capture safe: avoid the float32<->bf16 round-trip cast and the
    ``cat``/``view_as`` reshape that lower to strided ``copy_`` / ``as_strided``
    ops the XPU graph capturer rejects (copy_kernel.cpp:407 / as_strided assert).
    Compute in x's native dtype with ``stack`` + ``reshape`` (contiguous) and a
    single in-place ``copy_``.
    """
    def _apply(x, freqs, inv):
        B = x.shape[0]
        rope_dim = x.shape[-1]
        x_c = x.view(B, -1, rope_dim // 2, 2)  # [B, H, D/2, 2]
        xr = x_c[..., 0]  # [B, H, D/2]
        xi = x_c[..., 1]
        f_r = freqs.real.to(x.dtype).unsqueeze(1)  # [B, 1, D/2]
        f_i = freqs.imag.to(x.dtype).unsqueeze(1)
        if inv:
            f_i = -f_i
        o_r = xr * f_r - xi * f_i
        o_i = xr * f_i + xi * f_r
        # reshape back to x's ORIGINAL (possibly multi-head 3D) shape, not
        # (B, rope_dim): x may be [B, H, rope_dim] so a fixed (B, rope_dim)
        # drops the head dim.
        out = torch.stack([o_r, o_i], dim=-1).reshape(x.shape)
        x.copy_(out)

    freqs = freqs_cis[positions.long()]  # [B, rope_dim//2] complex64
    _apply(q, freqs, inverse)
    if k is not None:
        _apply(k, freqs, inverse)


# ---------------------------------------------------------------------------
# q_rope_store
# Upstream: sglang.kernels.ops.attention.dsv4.q_rope_store.q_rope_store (Triton)
# Rotates the rope-part (last 64 dims) of each 512-wide Q head while copying the
# nope-part, writing into a (possibly padded) output buffer. The Triton kernel
# does not compile on the XPU Triton backend, so run it in pure torch.
# ---------------------------------------------------------------------------

def _q_rope_store_torch(
    q: torch.Tensor,           # [num_tokens, num_heads, 512]
    output: torch.Tensor,      # [num_tokens, num_heads, 512], same shape as q
    freqs_cis: torch.Tensor,   # [max_pos, 32] complex64
    positions: torch.Tensor,   # [num_tokens] int32/int64
) -> None:
    """Forward RoPE on the last 64 dims of each 512-wide head, nope copied.

    Mirrors the interleaved (adjacent-pair) rotation of the Triton
    ``_q_rope_store`` kernel:
        out[2i]   = x[2i]*cos_i - x[2i+1]*sin_i
        out[2i+1] = x[2i]*sin_i + x[2i+1]*cos_i
    applied to dims [448, 512); dims [0, 448) are copied through unchanged.
    """
    rope_dim = 64
    nope_dim = q.shape[-1] - rope_dim  # 448
    aliased = output.data_ptr() == q.data_ptr()

    if q.shape[0] == 0:
        if not aliased:
            output.copy_(q)
        return

    x_rope = q[..., nope_dim:]  # [T, H, 64]
    x_c = x_rope.float().reshape(*x_rope.shape[:-1], rope_dim // 2, 2)
    xr = x_c[..., 0]  # [T, H, 32]
    xi = x_c[..., 1]

    freqs = freqs_cis[positions.long()]  # [T, 32] complex64
    f_r = freqs.real.to(torch.float32).unsqueeze(1)  # [T, 1, 32]
    f_i = freqs.imag.to(torch.float32).unsqueeze(1)

    o_r = xr * f_r - xi * f_i
    o_i = xr * f_i + xi * f_r
    rotated = torch.stack([o_r, o_i], dim=-1).reshape(x_rope.shape)

    # ``rotated`` is a fresh tensor, so writing into an output that aliases q
    # is safe. Skip the redundant nope self-copy in the in-place case.
    if not aliased:
        output[..., :nope_dim].copy_(q[..., :nope_dim])
    output[..., nope_dim:].copy_(rotated.to(output.dtype))


# ---------------------------------------------------------------------------
# rope_tail_fake_quant_fp4
# Upstream: sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant
#           .rope_tail_fake_quant_fp4 (Triton, uses libdevice + fp8 casts)
# The DSV4.1 fp4 indexer's fused RoPE-on-tail + fp4 fake-quant round-trip. The
# Triton kernel does not compile on the XPU backend (make_ttxir SIGSEGV). This
# mirrors exactly the non-CUDA branch of dsv41_sparse._rope_fq4:
# ``fake_quant_fp4(rope_tail(x, freqs, rope_dim))`` (or the compressed-KV
# variant) — sglang's own numerically-matched torch references.
# ---------------------------------------------------------------------------

def _round_positive_to_e4m3_fp32(v: torch.Tensor) -> torch.Tensor:
    """Round a positive fp32 tensor to the E4M3FN grid without the fp8 dtype.

    XPU has no ``float8_e4m3fn`` copy kernel (torch_xmlir NOT_IMPLEMENTED), so
    emulate ``x.to(torch.float8_e4m3fn).float()`` for the compressed-KV scale,
    which the caller has already clamped into E4M3FN's positive finite range
    [2**-9, 448]. E4M3 has 3 mantissa bits; per binade the step is 2**(e-3),
    with subnormals (e < -6) sharing the min-normal exponent -6 (step 2**-9).
    Uses round-half-to-even (torch.round) to match the fp8 cast.
    """
    e = torch.floor(torch.log2(v)).clamp_min(-6.0)
    step = torch.exp2(e - 3.0)
    return (torch.round(v / step) * step).clamp(max=448.0)


def _fake_quant_compressed_kv_no_fp8(x: torch.Tensor) -> torch.Tensor:
    """``fake_quant_compressed_kv`` with the E4M3 scale rounding emulated in
    fp32 (the upstream ``.to(torch.float8_e4m3fn)`` is NOT_IMPLEMENTED on XPU).
    Per-16 block; one E4M3 scale per block; e2m1 fp4 round-trip; zero-preserving.
    """
    from sglang.kernels.ops.attention.dsv4.torch_quant import round_fp4

    fp4_max = 6.0
    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (amax * (1.0 / fp4_max)).clamp(min=2.0**-9, max=448.0)
    scale = _round_positive_to_e4m3_fp32(scale)
    scaled = (blocks / scale).clamp(-fp4_max, fp4_max)
    deq = round_fp4(scaled) * scale
    return deq.flatten(-2).to(x.dtype)


def _rope_tail_fake_quant_fp4_torch(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
    block_size: int = 32,
    *,
    compressed_kv: bool = False,
) -> torch.Tensor:
    """Pure-torch RoPE-tail + fp4 fake-quant, reusing sglang's torch references.

    The non-compressed path delegates to sglang's ``fake_quant_fp4`` (ue8m0, no
    fp8). The compressed-KV path reimplements ``fake_quant_compressed_kv`` with
    an fp8-free E4M3 scale rounding because XPU lacks the fp8 copy kernel.
    """
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail
    from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4

    rotated = rope_tail(x, freqs, rope_dim, inverse=inverse)
    if compressed_kv:
        return _fake_quant_compressed_kv_no_fp8(rotated)
    return fake_quant_fp4(rotated, block_size)


# ---------------------------------------------------------------------------
# c2_decode_pool
# Upstream: sglang.kernels.ops.attention.dsv4.c2_decode_pool.c2_decode_pool
#           (Triton, uses libdevice.exp/div_rn; make_ttxir SIGSEGV on XPU).
# Ratio-2 decode pair-pooling: softmax-weighted merge of the current row with
# its partner held in the compress-state ring, updating the ring in place.
# Bitwise-faithful torch reimplementation (all fp32; read partner state before
# writing, matching the kernel's load-then-store ordering).
# ---------------------------------------------------------------------------

def _c2_decode_pool_torch(
    kv: torch.Tensor,
    score: torch.Tensor,
    pos: torch.Tensor,
    raw_out_loc: torch.Tensor,
    out_loc: torch.Tensor,
    req: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
    pad_row: int,
    *,
    ring_size: int = 0,
):
    assert kv.dtype == torch.float32 and score.dtype == torch.float32
    pos_l = pos.long()
    req_l = req.long()
    odd = (pos_l % 2) == 1  # [n]

    pad = torch.full_like(req_l, pad_row)
    if ring_size:
        r = torch.where(
            (raw_out_loc == 0) | (pos_l == 0),
            pad,
            req_l * ring_size + (pos_l - 1) % ring_size,
        )
    else:
        r = torch.where(raw_out_loc == 0, pad, req_l)
    r = r.long()

    # Read partner state BEFORE any write (advanced indexing returns a copy).
    p_kv = state_kv[r]
    p_score = state_score[r]

    if ring_size:
        write_row = (req_l * ring_size + pos_l % ring_size).long()
        wmask = raw_out_loc != 0
        if bool(wmask.any()):
            wr = write_row[wmask]
            state_kv[wr] = kv[wmask]
            state_score[wr] = score[wmask]
    else:
        odd_col = odd[:, None]
        state_kv[r] = torch.where(odd_col, p_kv, kv)
        state_score[r] = torch.where(odd_col, p_score, score)

    # Softmax-weighted pair merge, fp32 (libdevice.exp/div_rn -> torch exp/div).
    m = torch.maximum(p_score, score)
    e0 = torch.exp(p_score - m)
    e1 = torch.exp(score - m)
    denom = e0 + e1
    pooled = p_kv * (e0 / denom) + kv * (e1 / denom)

    group_pos = torch.where(odd, pos_l - 1, pos_l).to(pos.dtype)
    slots = torch.where(out_loc >= 0, out_loc, torch.zeros_like(out_loc))
    return pooled, group_pos, slots


# ---------------------------------------------------------------------------
# fp4_index_logits_decode
# Upstream: sglang.kernels.ops.attention.dsv4.fp4_indexer.fp4_index_logits_decode
#           (Triton; make_ttxir SIGSEGV on XPU).
# Decode-path indexer logits: dequant the fp4 index-K cache rows at `slots`,
# then s[b,l] = sum_h relu(q[b,h].K[b,l]) * weights[b,h], -inf beyond lens[b].
# Reuses the exact fp4 dequant of pool.get_index_k_dequant and indexer.scores.
# ---------------------------------------------------------------------------

def _fp4_index_logits_decode_torch(
    q: torch.Tensor,        # [B, H, 128] bf16 (rope'd, fp4-rounded)
    weights: torch.Tensor,  # [B, H] bf16 (softmax_scale folded in)
    slots: torch.Tensor,    # [B, L] int, page pool slots (already //ratio, masked)
    lens: torch.Tensor,     # [B] int, visible compressed positions
    table: torch.Tensor,    # [num_pages, page_size*68] index-K fp4 cache buffer
    page_size: int,
) -> torch.Tensor:
    """Chunked over L to bound peak memory: the fused kernel streams the fp4 K
    cache, so materializing the full [B, L, 128] K OOMs when the KV pool has
    already claimed most HBM. Process the visible positions in blocks."""
    from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

    dev = table.device
    B, H, _ = q.shape
    L = slots.shape[1]
    p = page_size
    lut = DSV4_DEQUANT_FP4_TABLE.to(dev).float()  # [16]
    qf = q.float()                                 # [B, H, 128]
    wf = weights.float().unsqueeze(-1)             # [B, H, 1]
    slots64 = slots.to(torch.int64)
    ar64 = torch.arange(64, device=dev)
    ar4 = torch.arange(4, device=dev)
    out = torch.empty(B, L, dtype=torch.float32, device=dev)

    chunk = 1024
    for c0 in range(0, L, chunk):
        c1 = min(c0 + chunk, L)
        sc = slots64[:, c0:c1]                     # [B, C]
        page = (sc // p).unsqueeze(-1)             # [B, C, 1]
        off = sc % p                               # [B, C]
        payload_cols = (off * 64).unsqueeze(-1) + ar64       # [B, C, 64]
        scale_cols = (p * 64 + off * 4).unsqueeze(-1) + ar4  # [B, C, 4]
        u = table[page, payload_cols].view(torch.uint8)      # [B, C, 64]
        lo = lut[(u & 0x0F).long()]                # [B, C, 64]
        hi = lut[((u >> 4) & 0x0F).long()]         # [B, C, 64]
        vals = torch.stack([lo, hi], dim=-1).flatten(2)      # [B, C, 128]
        exps = table[page, scale_cols].to(torch.int32) & 0xFF  # [B, C, 4]
        scales = torch.exp2(exps.float() - 127).repeat_interleave(32, dim=-1)  # [B,C,128]
        k = vals * scales                          # [B, C, 128] fp32
        s = torch.einsum("bhd,bcd->bhc", qf, k)    # [B, H, C]
        out[:, c0:c1] = (s.relu() * wf).sum(dim=1)  # [B, C]

    ar = torch.arange(L, device=dev)
    return out.masked_fill(ar[None, :] >= lens.to(torch.int64)[:, None], float("-inf"))


_FREQS_REAL_CACHE: dict = {}


def _get_cos_sin_cache(freqs_cis: torch.Tensor) -> torch.Tensor:
    """Convert complex ``freqs_cis`` into a real cache in ``[cos | sin]``
    concatenated format, with caching keyed by data_ptr/shape/dtype/device."""
    key = (
        freqs_cis.data_ptr(),
        tuple(freqs_cis.shape),
        freqs_cis.dtype,
        freqs_cis.device,
    )
    cached = _FREQS_REAL_CACHE.get(key)
    if cached is not None:
        return cached
    # XMLIR's complex view handling requires a normal versioned tensor.
    # DSpark builds its frequencies under inference_mode, unlike the target.
    with torch.inference_mode(False):
        normal_freqs = freqs_cis.clone()
        real_part = torch.real(normal_freqs)
        imag_part = torch.imag(normal_freqs)
        freqs_real = torch.cat([real_part, imag_part], dim=-1).contiguous()
    _FREQS_REAL_CACHE[key] = freqs_real
    return freqs_real


def _apply_rope_xspeedgate(
    q: torch.Tensor,           # [B, rope_dim] or [B, H, rope_dim]
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,   # [max_pos, rope_dim//2] complex64
    positions: torch.Tensor,   # [B] int32/int64
    inverse: bool = False,
    inplace: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply GPT-J RoPE via ``xspeedgate_ops.flashinfer_rotary_embedding``.

    ``inplace=True`` lets q/k be non-contiguous views (e.g. ``q[..., -64:]`` of a
    ``[B, H, 512]`` tensor): the op takes ``token_stride``/``head_stride`` from
    the view and writes back into the *view's own storage*, so neither the
    ``.contiguous()`` copy in nor a copy back is needed. The caller must own that
    storage (or accept mutating it) -- the same in-place contract the op has
    always had; ``inplace=False`` keeps the old copy-for-you behaviour.

    Returns ``(q_out, k_out)``; with ``inplace=True`` these are the arguments
    themselves (the kernel already wrote into them).
    """
    if q.shape[0] == 0:
        return q, k
    freqs_real = _get_cos_sin_cache(freqs_cis)
    rotary_dim = q.shape[-1]
    if inplace:
        q_input = q
        k_input = k
    else:
        q_input = q.contiguous() if not q.is_contiguous() else q
        k_input = k.contiguous() if (k is not None and not k.is_contiguous()) else k
    # The 256-token launch chunking was a workaround for a reported out-of-range
    # position read ("single launch carries many tokens"). Standalone validation
    # shows the op is correct and fault-free in one launch for n_tok up to 131072
    # (bit-exact vs chunked over n_tok {2048..131072} x H {1,8,64,128}), and the
    # chunking otherwise fires ~3200 launches per 16k step plus a cat. RoPE is
    # per-token independent, so a single launch is numerically identical.
    # ``KL_ROPE_CHUNK`` restores chunking for bisection if a vendor build regresses.
    # Chunking is incompatible with ``inplace`` (it returns concatenated copies,
    # so nothing would reach the caller's storage); it is a bisection escape
    # hatch only, so it is ignored on that path.
    chunk = 0 if inplace else int(os.environ.get("KL_ROPE_CHUNK", "0"))
    n_tok = q_input.shape[0]
    if chunk <= 0 or n_tok <= chunk:
        return torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
            positions=positions,
            rotary_dim=rotary_dim,
            head_size=rotary_dim,
            cos_sin_cache=freqs_real,
            is_neox_style=False,
            query=q_input,
            key=k_input,
            offsets=None,
            inverse=inverse,
        )
    q_parts, k_parts = [], []
    for beg in range(0, n_tok, chunk):
        end = min(beg + chunk, n_tok)
        q_p, k_p = torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
            positions=positions[beg:end].contiguous(),
            rotary_dim=rotary_dim,
            head_size=rotary_dim,
            cos_sin_cache=freqs_real,
            is_neox_style=False,
            query=q_input[beg:end].contiguous(),
            key=k_input[beg:end].contiguous() if k_input is not None else None,
            offsets=None,
            inverse=inverse,
        )
        q_parts.append(q_p)
        if k_p is not None:
            k_parts.append(k_p)
    q_out = torch.cat(q_parts, dim=0)
    k_out = torch.cat(k_parts, dim=0) if k_parts else None
    return q_out, k_out


def _fused_rope_inplace_kunlun(
    q: torch.Tensor,           # [B, rope_dim] or [B, H, rope_dim]
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,   # [max_pos, rope_dim//2] complex64
    positions: torch.Tensor,   # [B] int32/int64
    inverse: bool = False,
) -> None:
    """Kunlun-native fused GPT-J RoPE via ``xspeedgate_ops.flashinfer_rotary_embedding``.

    Mirrors the reference ``fused_rope_kunlun``: converts complex ``freqs_cis`` into a
    real ``[cos | sin]`` cache (cached), then rotates q/k **in place**.  q/k are
    normally non-contiguous tail views (``q[..., -64:]``); the op runs on the
    view's strides and writes back into the caller's storage, so the old
    ``.contiguous()`` + ``copy_`` round trip (2 extra passes over the rope slice
    per call) is gone.
    """

    _apply_rope_xspeedgate(q, k, freqs_cis, positions, inverse, inplace=True)


# ---------------------------------------------------------------------------
# fused_norm_rope_inplace
# Upstream: sglang.jit_kernel.dsv4.compress_old.fused_norm_rope_inplace
# Used in _compute_kv_bf16 (DSA CP path)
# ---------------------------------------------------------------------------

def _fused_norm_rope_inplace_torch(
    kv: torch.Tensor,           # [B, head_dim] bfloat16
    weight: torch.Tensor,       # [head_dim]
    eps: float,
    freq_cis: torch.Tensor,     # [max_pos, rope_dim//2] complex64
    positions: torch.Tensor,    # [B] int32/int64
) -> None:
    """In-place rmsnorm WITH weight + RoPE."""
    head_dim = kv.shape[-1]
    rope_dim = freq_cis.shape[-1] * 2
    nope_dim = head_dim - rope_dim

    orig_dtype = kv.dtype
    kv_f = kv.float()
    w_f = weight.float()

    rms = kv_f.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    kv_f = kv_f * rms * w_f.unsqueeze(0)

    freqs = freq_cis[positions.long()]  # [B, rope_dim//2] complex
    kv_nope = kv_f[:, :nope_dim]
    kv_rope = kv_f[:, nope_dim:].view(kv_f.shape[0], -1, 2)  # [B, rope_dim/2, 2]
    f_r = freqs.real
    f_i = freqs.imag
    xr = kv_rope[:, :, 0]
    xi = kv_rope[:, :, 1]
    kv_rope_rot = torch.stack([xr * f_r - xi * f_i, xr * f_i + xi * f_r], dim=-1).view(kv_f.shape[0], rope_dim)
    kv.copy_(torch.cat([kv_nope, kv_rope_rot], dim=-1).to(orig_dtype))


def _fused_norm_rope_inplace_kunlun(
    kv: torch.Tensor,           # [B, head_dim] bf16/fp16/fp32
    weight: torch.Tensor,       # [head_dim]
    eps: float,
    freq_cis: torch.Tensor,     # [max_pos, rope_dim//2] complex64
    positions: torch.Tensor,    # [B] int32/int64
) -> None:
    """Kunlun-native fused RMSNorm + GPT-J RoPE via ``dpsk_v4_norm_rope_gptj``.

    Routes to ``kunlun_ops.dpsk_v4_norm_rope_gptj`` in DefaultForward mode
    (mode=2), which fuses RMSNorm + RoPE on the trailing ``rope_dim`` slice.
    """

    kunlun_ops.dpsk_v4_norm_rope_gptj(
        kv,
        weight,
        positions.contiguous(),
        freq_cis,
        2,        # mode = DefaultForward
        0,        # compress_ratio (unused in DefaultForward)
        eps,
    )


# ---------------------------------------------------------------------------
# fused_q_indexer_rope_hadamard_quant
# Upstream: sglang.jit_kernel.dsv4.elementwise.fused_q_indexer_rope_hadamard_quant
# CUDA: FusedQIndexerRopeHadamardQuantKernel
# NO rmsnorm; apply RoPE on last 64 dims, 128-pt Hadamard, FP8 quant
# ---------------------------------------------------------------------------

def _hadamard_128(x: torch.Tensor) -> torch.Tensor:
    """128-point normalized Hadamard transform.

    Fast Walsh-Hadamard in-place via butterfly stages.
    x: [..., 128]
    Returns same shape, normalized by 1/sqrt(128).
    """
    n = x.shape[-1]
    assert n == 128
    h = x.clone()
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            # a/b MUST be snapshotted (.clone()): they are views into h, and the
            # first assignment overwrites h[i:i+step] (== a's storage) before the
            # second line reads a. Without clone the second line computes
            # (a+b)-b == a instead of a-b — a silent in-place aliasing bug that
            # scrambles the Hadamard output (verified: indexer q_hada cos vs the
            # reference 0.39 -> 0.99997, norm no longer collapses; fixes the
            # ratio=4 DSA indexer numerical divergence).
            a = h[..., i:i + step].clone()
            b = h[..., i + step:i + 2 * step].clone()
            h[..., i:i + step] = a + b
            h[..., i + step:i + 2 * step] = a - b
        step *= 2
    return h / math.sqrt(n)


def _fused_q_indexer_rope_hadamard_quant_torch(
    q_input: torch.Tensor,    # [B, num_heads, 128]
    weight: torch.Tensor,     # [B, num_heads]
    weight_scale: float,
    freqs_cis: torch.Tensor,  # [max_pos, 32] complex64 (rope_dim=64)
    positions: torch.Tensor,  # [B] int32/int64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE on last 64 dims, 128-pt Hadamard, INT8 quant (pure-torch reference).

    Returns:
      q_int8: [B, num_heads, 128] int8
      weights_out: [B, num_heads, 1] float32  (= weight * weight_scale * q_scale)
    """
    head_dim = 128
    rope_dim = 64
    nope_dim = head_dim - rope_dim  # 64

    B, H, D = q_input.shape
    assert D == head_dim

    q_f = q_input.float()   # [B, H, 128]
    w_f = weight.float()    # [B, H]

    freqs = freqs_cis[positions.long()]  # [B, 32] complex
    # Apply RoPE on last rope_dim=64 elements only
    q_nope = q_f[:, :, :nope_dim]   # [B, H, 64]
    q_rope = q_f[:, :, nope_dim:]   # [B, H, 64]
    q_rope_c = q_rope.view(B, H, 32, 2)  # [B, H, 32, 2]
    f_r = freqs.unsqueeze(1).real   # [B, 1, 32]
    f_i = freqs.unsqueeze(1).imag
    xr = q_rope_c[:, :, :, 0]
    xi = q_rope_c[:, :, :, 1]
    q_rope_rot = torch.stack([xr * f_r - xi * f_i, xr * f_i + xi * f_r], dim=-1).view(B, H, rope_dim)

    q_after_rope = torch.cat([q_nope, q_rope_rot], dim=-1)  # [B, H, 128]

    # 128-point Hadamard (per head independently, but vectorized over B and H)
    q_hada = _hadamard_128(q_after_rope)  # [B, H, 128]

    # INT8 quant via the shared kunlun act_quant hook.
    q_hada_2d = q_hada.reshape(B * H, 128).contiguous()
    q_int8_2d, q_max_2d = act_quant_kunlun(q_hada_2d, block_size=128)
    q_int8 = q_int8_2d.reshape(B, H, 128)
    q_scale = q_max_2d.reshape(B, H, 1) / 127.0

    # weights_out = weight * weight_scale * q_scale  (q_scale shape: [B, H, 1])
    weights_out = (w_f * weight_scale).unsqueeze(-1) * q_scale  # [B, H, 1]

    return q_int8, weights_out


def _fused_q_indexer_rope_hadamard_quant_kunlun(
    q_input: torch.Tensor,    # [B, num_heads, 128]
    weight: torch.Tensor,     # [B, num_heads]
    weight_scale: float,
    freqs_cis: torch.Tensor,  # [max_pos, 32] complex64 (rope_dim=64)
    positions: torch.Tensor,  # [B] int32/int64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE on last 64 dims, 128-pt Hadamard, INT8 quant (Kunlun-native).

    Step 1: RoPE via ``_apply_rope_xspeedgate``
    Step 2: 128-pt Hadamard via ``xspeedgate_ops.hadamard_transform``
    Step 3: INT8 quant via ``act_quant_kunlun`` → ``kunlun_ops.quant2d``

    All steps run in bf16 (q_input's native dtype); float32 is only used for
    the final weight fusion arithmetic.

    Returns:
      q_int8: [B, num_heads, 128] int8
      weights_out: [B, num_heads, 1] float32  (= weight * weight_scale * q_scale)
    """
    head_dim = 128
    rope_dim = 64
    nope_dim = head_dim - rope_dim  # 64

    B, H, D = q_input.shape
    assert D == head_dim

    q = q_input.contiguous()  # [B, H, 128] bf16
    w_f = weight.float()     # [B, H] float32

    # --- Step 1: RoPE on last 64 dims via _apply_rope_xspeedgate ---
    q_nope = q[:, :, :nope_dim]   # [B, H, 64] bf16
    q_rope = q[:, :, nope_dim:]   # [B, H, 64] bf16

    q_rope_rot, _ = _apply_rope_xspeedgate(
        q_rope, None, freqs_cis, positions, inverse=False,
    )


    q_after_rope = torch.cat([q_nope, q_rope_rot], dim=-1)  # [B, H, 128] bf16

    # --- Step 2: 128-pt Hadamard via xspeedgate_ops.hadamard_transform ---
    q_hada = torch.ops.xspeedgate_ops.hadamard_transform(
        q_after_rope.contiguous(), head_dim ** -0.5,  # scale = 1/sqrt(128)
    )


    # --- Step 3: INT8 quant via act_quant_kunlun (kunlun_ops.quant2d) ---
    q_hada_2d = q_hada.reshape(B * H, head_dim).contiguous()
    q_int8_2d, q_max_2d = act_quant_kunlun(q_hada_2d, block_size=head_dim)
    q_int8 = q_int8_2d.reshape(B, H, head_dim)
    q_scale = q_max_2d.reshape(B, H, 1) # / 127.0

    # weights_out = weight * weight_scale * q_scale  (float32)
    # weights_out = (w_f * weight_scale).unsqueeze(-1) * q_scale  # [B, H, 1]
    weights_out = fused_scale_kunlun(
            weight, weight_scale, q_scale
        )

    return q_int8, weights_out


def fused_scale_kunlun(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    """fused_scale_kunlun
    """
    if len(weight.shape) > 2:
        weight = weight.squeeze(-1)
    if len(q_scale.shape) > 2:
        q_scale = q_scale.squeeze(-1)
    out = torch.ops.xspeedgate_ops.fused_scale(
        weight=weight,
        out_scale=out_scale,
        q_scale=q_scale
    )
    return out


# ---------------------------------------------------------------------------
# compress_forward: pure PyTorch softmax-pool
# Upstream: sglang.jit_kernel.dsv4.compress.compress_forward
# ---------------------------------------------------------------------------

def _compress_forward_c128_torch(
    kv_score_buffer: torch.Tensor,   # [num_pages, 128, head_dim*2]
    kv_score_input: torch.Tensor,    # [num_q/bs, head_dim*2]
    ape: torch.Tensor,               # [128, head_dim]
    plan,                            # CompressorDecodePlan / CompressorPrefillPlan
    head_dim: int,
) -> torch.Tensor:
    """Plugin-owned, CUDA-graph-capturable c128 compress_forward.

    # [CG-CAPTURE-SAFE c128 compress write]
    Self-contained replacement for upstream
    ``compressor_v2._compress_forward_c128_fallback`` so NO sglang code is modified —
    the graph-safe fix lives entirely in the plugin. Logic is identical to upstream
    (write kv_score_input into the state pool, then softmax-pool compress), except the
    DECODE state-write uses a STATIC-shape scatter (clamp + gather + torch.where +
    scatter) instead of boolean-mask ``buf[wl[valid]] = ksi[valid]`` + ``.any()``,
    which call ``nonzero()`` / ``.item()`` (data-dependent, illegal in graph capture ->
    spurious multi-TiB OOM). Numerically identical for real decode (write_locs in range).
    """
    num_total_slots = kv_score_buffer.shape[0] * kv_score_buffer.shape[1]
    num_pages = kv_score_buffer.shape[0]
    last_dim = kv_score_buffer.shape[-1]

    # Step 1: WRITE kv_score_input to state buffer
    if num_total_slots > 0:
        buf_flat = kv_score_buffer.view(-1, last_dim)
        if plan.is_decode:
            plan_raw = plan[1].view(torch.int32)  # [bs, 4]
            write_locs = plan_raw[:, 1].long()
            P = int(write_locs.shape[0])
            safe_wl = write_locs.clamp(0, num_total_slots - 1)
            cur = buf_flat[safe_wl]
            buf_flat[safe_wl] = torch.where(
                (write_locs >= 0).unsqueeze(1), kv_score_input[:P], cur
            )
        else:
            # Prefill runs eagerly (prefill cuda graph disabled) -> boolean mask is fine.
            plan_w = plan[2]  # [num_w, 8] uint8 = WritePlan
            if plan_w.shape[0] > 0:
                plan_w_raw = plan_w.view(torch.int32)  # [num_w, 2]
                ragged_ids = plan_w_raw[:, 0].long() & 0xFFFF
                write_locs = plan_w_raw[:, 1].long()
                valid_write = (write_locs >= 0) & (write_locs < num_total_slots)
                ragged_ids_safe = ragged_ids.clamp(min=0, max=kv_score_input.shape[0] - 1)
                if valid_write.any():
                    buf_flat[write_locs[valid_write]] = kv_score_input[
                        ragged_ids_safe[valid_write]
                    ]

    # Step 2: COMPRESS (read from buffer page and softmax-pool) -- unchanged, graph-safe.
    plan_c = plan[1]
    num_tokens = plan_c.shape[0]
    if num_pages == 0 or num_tokens == 0:
        return kv_score_input.new_zeros(num_tokens, head_dim)

    plan_c_raw = plan_c.view(torch.int32)  # [N, 4]
    read_page_0 = plan_c_raw[:, 2].long()
    valid_read = (read_page_0 >= 0) & (read_page_0 < num_pages)
    read_page_0_safe = torch.where(valid_read, read_page_0, torch.zeros_like(read_page_0))

    group_missing = None
    if not plan.is_decode:
        # The state pool is a paged ring, so a long prefill overwrites the slots
        # that earlier compress groups still need before step 2 reads them: the
        # compressed KV then depends on the ring offset and the output varies from
        # run to run. Every member of a compress group is a row of this call's
        # kv_score_input, so gather the group by ragged id instead of via the ring.
        cr128 = kv_score_buffer.shape[1]
        num_q_rows = kv_score_input.shape[0]
        _dev = kv_score_input.device
        rid_c = (plan_c_raw[:, 1].to(torch.int64) & 0xFFFF).to(_dev)
        g_idx = torch.arange(cr128, device=_dev)
        rows = rid_c.unsqueeze(1) - (cr128 - 1) + g_idx.unsqueeze(0)
        group_missing = (rows < 0) | (rows >= num_q_rows)
        gathered = kv_score_input.float()[
            rows.clamp(0, max(num_q_rows - 1, 0)).reshape(-1)
        ].view(-1, cr128, kv_score_input.shape[-1])
    else:
        gathered = kv_score_buffer[read_page_0_safe].float()
    kv = gathered[:, :, :head_dim].float()
    score = gathered[:, :, head_dim:].float() + ape.float().unsqueeze(0)
    if group_missing is not None:
        score = score.masked_fill(
            group_missing.unsqueeze(-1), torch.finfo(score.dtype).min
        )
    weights = score.softmax(dim=1)
    out = (weights * kv).sum(dim=1)

    if plan.is_decode:
        seq_lens = plan_c_raw[:, 0].to(torch.int32)
        is_boundary = (seq_lens % 128 == 0).unsqueeze(-1)  # [N, 1]
        out = torch.where(is_boundary, out, torch.zeros_like(out))

    return out.to(kv_score_input.dtype)


def _compress_forward_torch(
    kv_score_buffer: torch.Tensor,   # [num_pages, cr, head_dim*2] or [num_pages, cr, head_dim*4] for c4
    kv_score_input: torch.Tensor,    # [bs/num_q_tokens, head_dim*2] or head_dim*4
    ape: torch.Tensor,               # [cr, head_dim] bias for score
    plan: object,                    # CompressorDecodePlan or CompressorPrefillPlan
    *,
    head_dim: int,
    compress_ratio: int,
    out: Optional[torch.Tensor] = None,
    is_online: bool = False,
) -> torch.Tensor:
    """Softmax-pool compress_forward with plan-guided write+read from buffer."""
    # c128 path uses the plugin-owned, CUDA-graph-capturable implementation
    # (_compress_forward_c128_torch) so no sglang code is modified.

    # For c4: we have kv_score_buffer [num_indices, 8, head_dim*4] (overlap)
    # For c128: kv_score_buffer [num_indices, 128, head_dim*2]

    if is_online:
        # Online path not supported for kunlun (rare), fall back to zeros
        num_tokens = plan[1].shape[0]
        if out is None:
            out = kv_score_input.new_zeros(num_tokens, head_dim)
        return out

    if int(compress_ratio) == 4:
        return _compress_forward_c4_torch(
            kv_score_buffer,
            kv_score_input,
            ape,
            plan,
            head_dim=head_dim,
        )

    return _compress_forward_c128_torch(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        plan=plan,
        head_dim=head_dim,
    )


# ---------------------------------------------------------------------------
# compress_norm_rope_store: norm + rope + write to compress kvcache
# Upstream: sglang.jit_kernel.dsv4.compress.compress_norm_rope_store
# Used by compressor_v2._forward_compress_all_in_one Step 2
# ---------------------------------------------------------------------------


def _store_compress_kv_int8(kvcache, out_locs, kv_out, page_size=None):
    """Write compressed K into the paged compress cache.

    ``xspeedgate_ops.set_mla_kv_buffer`` does not exist in every image and the
    previous fallback silently dropped the write, so the indexer/compress caches
    stayed all-zero: the indexer then produced zero scores and the 512-topk kept
    the *oldest* 512 compressed states, which silently truncated the context of
    any prompt longer than ``topk * compress_ratio`` tokens. Do the scatter in
    torch instead. Two layouts are supported, both discovered from the buffer:

    * mixed int8: per page ``block * head_dim`` int8 keys then ``block`` float32
      per-token scales (what the c4a logits ops read back), written with
      ``kunlun_ops.set_k_and_s_triton`` when available,
    * plain bf16/fp16: per page ``block * head_dim`` elements, no scales.
    """
    if kv_out.numel() == 0 or kvcache.numel() == 0:
        return
    head_dim = kv_out.shape[-1]
    num_pages = kvcache.shape[0]
    page_elems = kvcache.numel() // num_pages
    loc = out_locs.to(torch.int64)

    if kvcache.dtype in (torch.bfloat16, torch.float16, torch.float32):
        block = page_elems // head_dim
        if block <= 0 or page_elems != block * head_dim:
            raise AssertionError(
                f"unexpected compress kv layout: page_elems={page_elems} head_dim={head_dim}"
            )
        flat = kvcache.reshape(-1)
        loc = loc.clamp(min=0, max=num_pages * block - 1)
        idx = (loc * head_dim).unsqueeze(1) + torch.arange(
            head_dim, device=flat.device, dtype=torch.int64
        ).unsqueeze(0)
        flat[idx.reshape(-1)] = kv_out.reshape(-1).to(flat.dtype)
        return

    # byte-addressed page buffer
    page_bytes = page_elems * kvcache.element_size()
    block = page_bytes // (head_dim + 4)
    if block <= 0 or page_bytes != block * (head_dim + 4):
        block_alt = page_bytes // head_dim
        if block_alt > 0 and page_bytes == block_alt * head_dim:
            block, with_scale = block_alt, False
        else:
            raise AssertionError(
                f"unexpected compress kv page stride: page_bytes={page_bytes} head_dim={head_dim}"
            )
    else:
        with_scale = True

    x = kv_out.detach().float()
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
    q8 = torch.round(x / scale).clamp(-127.0, 127.0).to(torch.int8)
    loc = loc.clamp(min=0, max=num_pages * block - 1)

    if with_scale:
        # ``kunlun_ops.set_k_and_s_triton`` writes exactly this layout (per page
        # ``page_size x k_dim`` int8 keys followed by ``page_size`` fp32 scales).
        native_store = getattr(kunlun_ops, "set_k_and_s_triton", None)
        if native_store is not None:
            native_store(
                kvcache,
                loc.to(torch.int32),
                q8,
                scale.reshape(-1).contiguous(),
                block,
            )
            return

    flat_i8 = kvcache.reshape(-1).view(torch.int8)
    page = loc // block
    off = loc % block
    k_idx = (page * page_bytes + off * head_dim).unsqueeze(1) + torch.arange(
        head_dim, device=flat_i8.device, dtype=torch.int64
    ).unsqueeze(0)
    flat_i8[k_idx.reshape(-1)] = q8.reshape(-1)

    if with_scale:
        flat_f32 = kvcache.reshape(-1).view(torch.float32)
        s_idx = page * (page_bytes // 4) + (block * head_dim) // 4 + off
        flat_f32[s_idx] = scale.reshape(-1)


def _compress_norm_rope_store_torch(
    kv: torch.Tensor,              # [num_tokens, head_dim]
    plan: object,                  # CompressorDecodePlan or CompressorPrefillPlan
    *,
    norm_weight: torch.Tensor,
    norm_eps: float,
    freq_cis: torch.Tensor,        # [max_pos, rope_dim//2] complex64
    out_loc: torch.Tensor,         # [num_c4/c128_out_tokens] int64
    kvcache: torch.Tensor,         # paged kv cache (uint8 view)
    page_size: int,
    use_fp4: bool = False,
    bf16_store: bool = False,
    kvcache_scale: torch.Tensor = None,  # [kunlun-adapt] 上游新增；fp16/bf16 KV 下不参与
    rope_cache=None,  # [kunlun-adapt] 上游新增（HIP FP4）；忽略
    fp4_k_write_metadata=None,  # [kunlun-adapt] 上游新增（FP4 写元数据）；忽略
) -> None:
    """Apply rmsnorm + RoPE then scatter to paged cache.

    For the compress path, we use the HIP-style fallback since that's tested
    pure-torch code. This calls into the existing quant_to_nope_fp8_rope_bf16_pack_triton
    or the non-fused set_extra_key_buffer path.
    """
    if kv.shape[0] == 0:
        return

    head_dim = kv.shape[-1]
    rope_dim = min(64, head_dim)  # standard for DSV4
    nope_dim = head_dim - rope_dim

    orig_dtype = kv.dtype
    kv_f = kv.float()
    w_f = norm_weight.float()

    # rmsnorm with weight
    rms = kv_f.pow(2).mean(dim=-1, keepdim=True).add(norm_eps).rsqrt()
    kv_f = kv_f * rms * w_f.unsqueeze(0)

    # Extract positions from plan
    plan_tensor = plan[1]  # plan_d or plan_c
    seq_lens_raw = plan_tensor[:, :4].contiguous().view(torch.int32).squeeze(-1)
    positions = (seq_lens_raw.to(torch.int32) - int(plan.compress_ratio)).clamp(min=0)
    positions_safe = positions.long()

    freqs = freq_cis[positions_safe]  # [N, rope_dim//2] complex

    kv_nope = kv_f[:, :nope_dim]
    kv_rope = kv_f[:, nope_dim:].view(kv_f.shape[0], -1, 2)
    f_r = freqs.real
    f_i = freqs.imag
    xr = kv_rope[:, :, 0]
    xi = kv_rope[:, :, 1]
    kv_rope_rot = torch.stack([xr * f_r - xi * f_i, xr * f_i + xi * f_r], dim=-1).view(kv_f.shape[0], rope_dim)
    kv_out = torch.cat([kv_nope, kv_rope_rot], dim=-1).to(orig_dtype)  # [N, head_dim]

    # Now store via the existing set_extra_key_buffer which calls the HIP Triton path
    # We need to get out_loc indices for plan entries
    if plan.is_decode:
        out_locs = out_loc
    else:
        # Prefill: extract ragged_ids from plan_c and remap out_loc
        plan_c_raw = plan[1].view(torch.int32)  # [num_c, 4]
        ragged_ids = plan_c_raw[:, 1].long() & 0xFFFF
        out_locs = out_loc[ragged_ids.clamp(min=0, max=out_loc.shape[0] - 1)]

    if kv_out.shape[0] == 0:
        return

    # Use the HIP-style quant+pack path if available, otherwise raw copy
    try:
        from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
            quant_to_nope_fp8_rope_bf16_pack_triton,
        )
        pack = quant_to_nope_fp8_rope_bf16_pack_triton(kv_out.bfloat16())
    except Exception:
        # Fallback: just use bfloat16 representation
        pack = kv_out.bfloat16()

    _store_compress_kv_int8(kvcache, out_locs, kv_out, page_size=page_size)


def _compress_norm_rope_store_kunlun(
    kv: torch.Tensor,              # [num_tokens, head_dim]
    plan: object,                  # CompressorDecodePlan or CompressorPrefillPlan
    *,
    norm_weight: torch.Tensor,
    norm_eps: float,
    freq_cis: torch.Tensor,        # [max_pos, rope_dim//2] complex64
    out_loc: torch.Tensor,         # [num_c4/c128_out_tokens] int64
    kvcache: torch.Tensor,         # paged kv cache (uint8 view)
    page_size: int,
    use_fp4: bool = False,
    bf16_store: bool = False,
    kvcache_scale: torch.Tensor = None,  # [kunlun-adapt] 上游新增；fp16/bf16 KV 下不参与
    rope_cache=None,  # [kunlun-adapt] 上游新增（HIP FP4）；忽略
    fp4_k_write_metadata=None,  # [kunlun-adapt] 上游新增（FP4 写元数据）；忽略
) -> None:
    """Kunlun-native fused rmsnorm + RoPE + store to compress kvcache.

    Step 1: ``sgl_kernel_kunlun.rmsnorm`` (with weight)
    Step 2: ``_apply_rope_xspeedgate`` on rope part
    Step 3: store via ``xspeedgate_ops.set_mla_kv_buffer`` (same as torch version)
    """
    if kv.shape[0] == 0:
        return

    head_dim = kv.shape[-1]
    rope_dim = 64
    nope_dim = head_dim - rope_dim

    # --- Step 1: RMSNorm with weight (reuse sgl_kernel_kunlun.rmsnorm) ---
    from sglang_kunlun.kernels.sgl_kernel_kunlun import rmsnorm as _kunlun_rmsnorm

    # rmsnorm requires bf16; keep a reference to original dtype for store
    kv_bf16 = kv.to(torch.bfloat16) if kv.dtype != torch.bfloat16 else kv
    weight_bf16 = norm_weight.to(torch.bfloat16) if norm_weight.dtype != torch.bfloat16 else norm_weight
    kv_normed = _kunlun_rmsnorm(kv_bf16, weight_bf16, norm_eps)

    # --- Step 2: RoPE on rope part (reuse _apply_rope_xspeedgate) ---
    # Extract positions from plan (same logic as torch version)
    plan_tensor = plan[1]
    seq_lens_raw = plan_tensor[:, :4].contiguous().view(torch.int32).squeeze(-1)
    positions = (seq_lens_raw.to(torch.int32) - int(plan.compress_ratio)).clamp(min=0)
    positions = positions.to(kv.device)

    kv_nope = kv_normed[:, :nope_dim]   # [N, nope_dim]
    kv_rope = kv_normed[:, nope_dim:]   # [N, 64]

    # flashinfer_rotary_embedding requires 3D; unsqueeze [N, 64] → [N, 1, 64]
    kv_rope_rot, _ = _apply_rope_xspeedgate(
        kv_rope.unsqueeze(1), None, freq_cis, positions, inverse=False,
    )
    kv_rope_rot = kv_rope_rot.squeeze(1)  # [N, 64]

    kv_out = torch.cat([kv_nope, kv_rope_rot], dim=-1)  # [N, head_dim]

    # --- Step 3: Store to paged cache (same as torch version) ---
    if plan.is_decode:
        out_locs = out_loc
    else:
        plan_c_raw = plan[1].contiguous().view(torch.int32)
        ragged_ids = plan_c_raw[:, 1].long() & 0xFFFF
        out_locs = out_loc[ragged_ids.clamp(min=0, max=out_loc.shape[0] - 1)]

    if kv_out.shape[0] == 0:
        return

    try:
        from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
            quant_to_nope_fp8_rope_bf16_pack_triton,
        )
        pack = quant_to_nope_fp8_rope_bf16_pack_triton(kv_out.bfloat16())
    except Exception:
        pack = kv_out.bfloat16()

    _store_compress_kv_int8(kvcache, out_locs, kv_out, page_size=page_size)


# ---------------------------------------------------------------------------
# create_paged_compress_data (v1 legacy compressor.py write_loc/extra_data
# builder). Upstream: sglang.jit_kernel.dsv4.attn.triton_create_paged_compress_data
# (Triton kernel, unsupported on Kunlun). Mirrors the reference
# kunlun_create_paged_compress_data: route directly to the native
# xspeedgate_ops.create_paged_compress_data kernel instead of a pure-torch
# fallback, since the op is confirmed present on Kunlun.
# ---------------------------------------------------------------------------

def _create_paged_compress_data_kunlun(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    block: int = 128,
):
    """Kunlun-native replacement for ``triton_create_paged_compress_data``.

    Routes to ``xspeedgate_ops.create_paged_compress_data`` (same op used by
    the reference ``kunlun_create_paged_compress_data``), avoiding the
    unsupported Triton kernel entirely. ``block`` is accepted for signature
    compatibility with the upstream Triton helper but unused: the native op
    does its own internal launch planning.
    """
    # xspeedgate_ops.create_paged_compress_data requires int64 for the
    # mapping tensor even though the memory pool may store it as int32.
    actual_mapping = full_to_swa_index_mapping.to(torch.int64)
    return torch.ops.xspeedgate_ops.create_paged_compress_data(
        req_pool_indices=req_pool_indices.to(torch.int64),
        seq_lens=seq_lens.to(torch.int32),
        extend_seq_lens=extend_seq_lens.to(torch.int32),
        req_to_token=req_to_token,
        full_to_swa_index_mapping=actual_mapping,
        swa_page_size=swa_page_size,
        ring_size=ring_size,
        compress_ratio=compress_ratio,
        is_overlap=is_overlap,
    )


def _compress_forward_c4_torch(
    kv_score_buffer,   # (num_pages, 4, 4*head_dim)
    kv_score_input,    # (num_q, 4*head_dim)
    ape,               # (8, head_dim) after view(-1, head_dim)
    plan,
    *,
    head_dim,
):
    hd = int(head_dim)
    num_pages = kv_score_buffer.shape[0]
    last_dim = kv_score_buffer.shape[-1]
    buf_flat = kv_score_buffer.reshape(-1, last_dim)
    total_slots = buf_flat.shape[0]

    # Step 1: write kv_score_input into the state pool
    if total_slots > 0:
        if plan.is_decode:
            # [CG-CAPTURE-SAFE compress write]
            # Static-shape scatter (no boolean mask / nonzero / .item()) so this is
            # CUDA-graph-capturable. Boolean indexing (kv_score_input[valid]) calls
            # nonzero() -> data-dependent dynamic shape, illegal during graph capture
            # (caused a spurious multi-TiB OOM). In decode each request writes its
            # current KV-score row into its own in-range state slot (wl >= 0); clamp
            # for safety and scatter unconditionally, using torch.where so any wl < 0
            # row is a no-op (writes back the current value).
            praw = plan[1].view(torch.int32)
            wl = praw[:, 1].long().to(buf_flat.device)
            P = int(wl.shape[0])
            safe_wl = wl.clamp(0, total_slots - 1)
            cur = buf_flat[safe_wl]
            buf_flat[safe_wl] = torch.where(
                (wl >= 0).unsqueeze(1), kv_score_input[:P], cur
            )
        else:
            pw = plan[2]
            if pw.shape[0] > 0:
                # [CG-CAPTURE-SAFE extend compress write]
                # Static-shape scatter (no boolean-mask / nonzero / .any()) so the
                # MTP draft-extend CUDA-graph capture is legal. Boolean indexing
                # (buf_flat[wl[valid]] = kv_score_input[ridc[valid]]) calls nonzero()
                # -> data-dependent dynamic shape, illegal during graph capture and
                # the source of the spurious multi-hundred-GiB OOM under PD+MTP.
                # Invalid rows (wl<0 or out-of-range) are routed to an appended
                # scratch slot that is discarded, so they never corrupt real state
                # slots; valid rows keep their distinct write positions. Numerically
                # identical to the original masked scatter on the eager path.
                pwr = pw.view(torch.int32)
                rid = (pwr[:, 0].long() & 0xFFFF).to(buf_flat.device)
                wl = pwr[:, 1].long().to(buf_flat.device)
                valid = (wl >= 0) & (wl < total_slots)
                ridc = rid.clamp(min=0, max=kv_score_input.shape[0] - 1)
                tgt = torch.where(
                    valid, wl.clamp(0, total_slots - 1),
                    torch.full_like(wl, total_slots),
                )
                # Duplicate write slots are already resolved on the host by
                # _make_prefill_plan_c4_torch (last token wins), so no
                # index-validating dedup is needed here - such an op would abort
                # the MTP target-verify graph capture, where the indices computed
                # above are recorded but not yet materialised.
                padded = torch.cat(
                    [buf_flat, buf_flat.new_zeros(1, last_dim)], dim=0
                )
                padded[tgt] = kv_score_input[ridc]
                buf_flat.copy_(padded[:total_slots])

    # Step 2: compress
    pc = plan[1]
    N = pc.shape[0]
    if num_pages == 0 or N == 0:
        return kv_score_input.new_zeros(N, hd)

    dev = kv_score_buffer.device
    praw = pc.view(torch.int32)
    seq_len = praw[:, 0].to(torch.int64).to(dev)
    P = seq_len - 1
    rp0 = praw[:, 2].long().to(dev)
    rp1 = praw[:, 3].long().to(dev)
    v0 = (rp0 >= 0) & (rp0 < num_pages)
    v1 = (rp1 >= 0) & (rp1 < num_pages)
    rp0s = torch.where(v0, rp0, torch.zeros_like(rp0))
    rp1s = torch.where(v1, rp1, torch.zeros_like(rp1))

    # The state pool is a ring: within one 256-token page only ``ring_size``
    # slots exist, so during a long prefill the bulk write in step 1 overwrites
    # the slots that earlier compress groups need, and step 2 would read the
    # wrong tokens (garbage compressed KV, output varying with the ring offset).
    # Every token of a compress group is present in ``kv_score_input`` of this
    # very call, so read the group members directly by ragged id on the extend
    # path; the ring write above is only needed for later decode steps.
    row_missing = None
    if not plan.is_decode:
        num_q_rows = kv_score_input.shape[0]
        rid_c = (praw[:, 1].to(torch.int64) & 0xFFFF).to(dev)
        g_idx = torch.arange(4, device=dev)
        rows_ov = rid_c.unsqueeze(1) - 7 + g_idx.unsqueeze(0)
        rows_fr = rid_c.unsqueeze(1) - 3 + g_idx.unsqueeze(0)
        row_missing = (rows_ov < 0) | (rows_ov >= num_q_rows)
        x_in = kv_score_input.float()
        ov = x_in[rows_ov.clamp(0, max(num_q_rows - 1, 0)).reshape(-1)].view(
            -1, 4, 4 * hd
        )
        fr = x_in[rows_fr.clamp(0, max(num_q_rows - 1, 0)).reshape(-1)].view(
            -1, 4, 4 * hd
        )
        kv_ov = ov[:, :, 0:hd]
        sc_ov = ov[:, :, 2 * hd:3 * hd]
        kv_fr = fr[:, :, hd:2 * hd]
        sc_fr = fr[:, :, 3 * hd:4 * hd]
    else:
        prev = kv_score_buffer[rp0s]  # (N, 4, 4hd) overlap tokens P-7..P-4
        cur = kv_score_buffer[rp1s]   # (N, 4, 4hd) fresh tokens P-3..P

        kv_ov = prev[:, :, 0:hd].float()
        sc_ov = prev[:, :, 2 * hd:3 * hd].float()
        kv_fr = cur[:, :, hd:2 * hd].float()
        sc_fr = cur[:, :, 3 * hd:4 * hd].float()

    kv = torch.cat([kv_ov, kv_fr], dim=1)          # (N, 8, hd)
    ape2 = ape.reshape(-1, hd).float()             # (8, hd)
    sc = torch.cat([sc_ov, sc_fr], dim=1) + ape2.unsqueeze(0)

    k_idx = torch.arange(8, device=kv.device)
    pos_k = P.unsqueeze(1) - 7 + k_idx.unsqueeze(0)          # (N, 8) absolute positions
    invalid = pos_k < 0
    if row_missing is not None:
        ov_page_invalid = torch.cat(
            [row_missing, torch.zeros_like(row_missing)], dim=1
        )
    else:
        ov_page_invalid = (~v0).unsqueeze(1) & (k_idx.unsqueeze(0) < 4)
    invalid = invalid | ov_page_invalid
    sc = sc.masked_fill(invalid.unsqueeze(-1), torch.finfo(sc.dtype).min)

    weights = torch.softmax(sc, dim=1)
    out = (weights * kv).sum(dim=1)                # (N, hd)

    if plan.is_decode:
        is_bnd = (seq_len % 4 == 0).unsqueeze(-1)
        out = torch.where(is_bnd, out, torch.zeros_like(out))

    return out.to(kv_score_input.dtype)
