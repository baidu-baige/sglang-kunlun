"""EB5-lite **attn ② MLA** 的算子级昆仑适配层。

与 `eb5_kunlun_adapt.py` 同一套做法：EB5 的融合 kernel 在昆仑上跑不了
（triton 后端缺失 / fork 自带 nvcc-JIT 链建不起来 / deep_gemm、sgl_kernel
在本环境整包缺失），于是把这些 kernel 换成**语义等价的 torch 拆解**，注入到
fork 的调用点上，供 `test_eb5_attn_mla.py` 的 kernel 级单测真跑。

本模块覆盖 `test_eb5_attn_mla.py` 里 23 个用例用到的 kernel：

* `sglang_per_token_group_quant_fp8{,_ue8m0}` —— per-token-group fp8 量化
* `deep_gemm.fp8_fp4_gemm_nt` / `deep_gemm.bf16_mqa_logits` —— 投影 GEMM / indexer logits
* `topk_transform_ragged_v2` —— ragged prefill fused top-k
* `jit_kernel.internal.dsa.fused_{q,k}_indexer_*` —— indexer Q/K 的 rope+hadamard+fp8
* `concat_mla_absorb_q` / `set_mla_kv_buffer` / `quantize_k_cache{,_separate}`
* `append_indexer_swa_triton` —— fork 实现漏了 padding 行语义，这里用向量化 torch 覆盖
* `GetKAndS.triton`
* `eb5_sigmoid_gate_mul`
* `sgl_kernel.flash_mla.{get_mla_metadata, flash_mla_with_kvcache}`

另有一个**运行时补丁** `_install_fp8_upcast`：昆仑 torch 没实现设备侧
`fp8 -> fp32/bf16` 的 cast（`xdnn_pytorch_wrapper/cast.cpp` 报
`UnknownScalarType`），而单测会直接对 fp8 张量调 `.float()`。补丁把
`aten::_to_copy`（CUDA key）换成"fp8 走 256 项 LUT，其余照旧 empty+copy_"的实现。

**注意**：被替换的都是"拆解实现"，因此这些用例通过只代表**拆解语义**在昆仑上
可跑通，不代表原 CUDA kernel 可跑。真正的 kernel 缺口（见报告 T1/T2/T4/T6）
需要 KLX 侧的 triton XPU 后端 / nvcc 工具链 / deep_gemm 才能消掉。
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch

_DEV = "cuda"
_HEAD_DIM = 128
_ROPE_DIM = 64
_FP8_MAX = 448.0
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_HADAMARD: Optional[torch.Tensor] = None
_FP8_LUT: dict = {}


# --------------------------------------------------------------------------- #
# fp8 基础：设备侧没有 fp8<->float 的 cast，用 LUT / CPU 往返代替
# --------------------------------------------------------------------------- #
def _is_fp8(dtype) -> bool:
    return dtype in _FP8_DTYPES


def _fp8_lut(dtype, device) -> torch.Tensor:
    """256 项 fp8 -> fp32 查表（CPU 端建好再搬到设备）。"""
    key = (dtype, str(device))
    lut = _FP8_LUT.get(key)
    if lut is None:
        lut = (
            torch.arange(256, dtype=torch.uint8)
            .view(dtype)
            .to(torch.float32)
            .to(device)
        )
        _FP8_LUT[key] = lut
    return lut


def _fp8_to_fp32(t: torch.Tensor) -> torch.Tensor:
    """设备侧 fp8 -> fp32（查表，不触发未实现的 cast）。"""
    if not _is_fp8(t.dtype):
        return t.float()
    idx = t.contiguous().view(torch.uint8).to(torch.int64)
    return _fp8_lut(t.dtype, t.device)[idx]


def _to_fp8(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp8_e4m3fn（走 CPU，昆仑设备侧 cast 未实现）。"""
    return (
        t.detach()
        .to("cpu", torch.float32)
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
        .to(t.device)
    )


def _install_fp8_upcast() -> bool:
    """让设备侧 `fp8_tensor.float()/.to(...)` 可用。

    昆仑的 `_to_copy` 对 fp8 源张量会走到 xdnn 未实现的 cast 分支。这里给
    `aten::_to_copy` 注册一个 CUDA-key 实现：fp8 源走 LUT，其余走与上游
    composite 等价的 `empty_like + copy_`。

    **默认关闭**（`EB5_KUNLUN_FP8_UPCAST=1` 打开）。原因：这是给 `aten::_to_copy`
    装一个**全局** CUDA kernel，进程里每一次 `.to()` / `.float()` / `.cpu()` 都要
    多绕一层 Python。kernel 单测需要它（用例直接做 fp8 cast），但组网推理里
    host 侧的这点开销足以把设备喂空 —— 实测会在 `xpu_wait` 里报
    ``wait for noc idle timeout`` / 卡死。组网侧 fp8 的 cast 都有各自的
    `dequantize_k_cache_paged` / `_fp8_to_fp32` 拆解，不需要这层兜底。
    """
    if os.environ.get("EB5_KUNLUN_FP8_UPCAST", "0").lower() in {
        "0",
        "false",
        "no",
        "off",
        "",
    }:
        return False
    if getattr(torch, "_eb5_fp8_upcast_installed", False):
        return True
    try:
        lib = torch.library.Library("aten", "IMPL", "CUDA")

        def _to_copy(
            self,
            *,
            dtype=None,
            layout=None,
            device=None,
            pin_memory=None,
            non_blocking=False,
            memory_format=None,
        ):
            tgt_dtype = self.dtype if dtype is None else dtype
            tgt_device = self.device if device is None else device
            if (
                _is_fp8(self.dtype)
                and not _is_fp8(tgt_dtype)
                and str(tgt_device) == str(self.device)
            ):
                out = _fp8_to_fp32(self)
                return out if tgt_dtype == torch.float32 else out.to(tgt_dtype)
            out = torch.empty_like(
                self, dtype=tgt_dtype, device=tgt_device, memory_format=memory_format
            )
            out.copy_(self, non_blocking=non_blocking)
            return out

        lib.impl("_to_copy", _to_copy)
        torch._eb5_fp8_upcast_installed = True
        return True
    except Exception as exc:  # noqa: BLE001 - 装不上就退回"用例自己用 LUT"
        print("[eb5-mla-adapt] fp8 upcast patch failed:", exc)
        return False


# --------------------------------------------------------------------------- #
# 其它昆仑运行时缺口
# --------------------------------------------------------------------------- #
def _install_runtime_shims() -> list:
    """补两个昆仑 torch 的算子缺口（都在 Python 层打，不动 dispatcher）：

    * ``Tensor.abs()``：``xdnn_pytorch_wrapper/abs.cpp`` 不支持 uint8
      （``scalar type of self : kuint8 is unsupported``），而 uint8 的绝对值就是
      自身，直接返回副本即可；其余 dtype 原样走原生实现。
    * ``torch.equal``：昆仑的 equal 不做 dtype 提升，``int32`` vs ``int64`` 会报
      ``data type not matched! kint64 using as kint32``；这里先提升到同一 dtype。
    """
    patched = []

    if not getattr(torch.Tensor, "_eb5_abs_patched", False):
        _orig_abs = torch.Tensor.abs

        def _abs(self):
            if self.dtype == torch.uint8:
                return self.clone()
            return _orig_abs(self)

        _abs._eb5_abs_patched = True  # type: ignore[attr-defined]
        torch.Tensor.abs = _abs
        patched.append("Tensor.abs(uint8)")

    if not getattr(torch.equal, "_eb5_equal_patched", False):
        _orig_equal = torch.equal

        def _equal(a, b):
            if (
                isinstance(a, torch.Tensor)
                and isinstance(b, torch.Tensor)
                and a.dtype != b.dtype
            ):
                if a.is_floating_point() or b.is_floating_point():
                    a, b = a.float(), b.float()
                else:
                    a, b = a.to(torch.int64), b.to(torch.int64)
            return _orig_equal(a, b)

        _equal._eb5_equal_patched = True  # type: ignore[attr-defined]
        torch.equal = _equal
        patched.append("torch.equal(dtype promote)")

    return patched


# --------------------------------------------------------------------------- #
# per-token-group fp8 量化
# --------------------------------------------------------------------------- #
def _group_quant_fp8(
    x: torch.Tensor, group_size: int, eps: float, ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """逐 `group_size` 段求 absmax 缩放后量化到 fp8_e4m3。

    与 `_per_token_group_quant_8bit` 的 kernel 语义一致：
    ``scale = max(abs(group), eps) / 448``；ue8m0 时 scale 再取 2 的幂
    （``exp2(ceil(log2(|scale|)))``，与 `deep_gemm.ceil_to_ue8m0` 同义）。
    """
    xf = x.float()
    *lead, k = xf.shape
    assert k % group_size == 0, (k, group_size)
    g = xf.reshape(*lead, k // group_size, group_size)
    scale = g.abs().amax(dim=-1).clamp(min=eps) / _FP8_MAX
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = _to_fp8(g / scale.unsqueeze(-1))
    return q.reshape(*lead, k), scale.to(torch.float32)


def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
):
    """torch 拆解：per-token-group fp8 量化（默认行主序 float32 scale）。"""
    if fuse_silu_and_mul:
        half = x.shape[-1] // 2
        a, b = x.float()[..., :half], x.float()[..., half:]
        x = (torch.nn.functional.silu(a) * b).to(x.dtype)
    x_q, x_s = _group_quant_fp8(x, group_size, eps, scale_ue8m0)
    if column_major_scales:
        x_s = x_s.transpose(-1, -2).contiguous().transpose(-1, -2)
    return x_q, x_s


def sglang_per_token_group_quant_fp8_ue8m0(
    x: torch.Tensor, group_size: int, eps: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch 拆解：UE8M0 变体。

    真实 kernel 把 scale 存成 TMA 对齐的 int32 打包布局；本环境 deep_gemm 整包
    缺失，scale 的消费者（`fp8_fp4_gemm_nt`）也由本模块替换，故这里采用
    `create_per_token_group_quant_fp8_output_scale` 的**行主序 float32 2 的幂**
    布局（上游注释称之为 deep_gemm 期望的取值形式），两侧自洽。
    """
    return _group_quant_fp8(x, group_size, eps, True)


# --------------------------------------------------------------------------- #
# deep_gemm：整包缺失，补两个纯 torch 实现
# --------------------------------------------------------------------------- #
def _dequant_fp8_group(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    qf = _fp8_to_fp32(q)
    sf = s.float()
    group = max(1, qf.shape[-1] // sf.shape[-1])
    return qf * sf.repeat_interleave(group, dim=-1)


def fp8_fp4_gemm_nt(
    a, b, out: torch.Tensor, recipe_a=None, recipe_b=None, **kwargs
) -> None:
    """torch 拆解：``(aq, asc) x (bq, bsc) -> out``（fp8 分组反量化后 GEMM）。"""
    aq, asc = a
    bq, bsc = b
    lhs = _dequant_fp8_group(aq, asc)
    rhs = _dequant_fp8_group(bq, bsc)
    out.copy_((lhs @ rhs.t()).to(out.dtype))


def bf16_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = True,
    **kwargs,
) -> torch.Tensor:
    """torch 拆解：lightning indexer 的 MQA logits。

    ``out[t, v] = sum_h weights[t,h] * relu(q[t,h]·kv[v])``，`[ks[t], ke[t])`
    之外按 ``clean_logits`` 置 ``-inf``。
    """
    scores = torch.einsum("thd,vd->thv", q.float(), kv.float())
    out = (weights.float().unsqueeze(-1) * torch.relu(scores)).sum(1)
    if ks is not None and ke is not None:
        idx = torch.arange(kv.shape[0], device=q.device)
        inside = (idx[None, :] >= ks.to(torch.int64)[:, None]) & (
            idx[None, :] < ke.to(torch.int64)[:, None]
        )
        if clean_logits:
            out = torch.where(inside, out, torch.full_like(out, float("-inf")))
        else:
            out = out.masked_fill(~inside, 0.0)
    return out


def _install_deep_gemm_torch() -> list:
    """把上面两个实现挂到 ``deep_gemm``（插件已装好一个空壳桩）上。"""
    import sys
    import types

    mod = sys.modules.get("deep_gemm")
    if mod is None:
        try:
            import deep_gemm as mod  # noqa: F811
        except Exception:
            mod = types.ModuleType("deep_gemm")
            sys.modules["deep_gemm"] = mod
    patched = []
    for name, impl in (
        ("fp8_fp4_gemm_nt", fp8_fp4_gemm_nt),
        ("bf16_mqa_logits", bf16_mqa_logits),
    ):
        setattr(mod, name, impl)
        patched.append(f"deep_gemm.{name}")
    return patched


# --------------------------------------------------------------------------- #
# top-k
# --------------------------------------------------------------------------- #
def topk_transform_ragged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    out_offsets: torch.Tensor,
    out_indices: torch.Tensor,
    row_starts: Optional[torch.Tensor] = None,
) -> None:
    """torch 拆解：逐行取 ``scores[i, ks : ks+seq_lens[i]]`` 的 top-k。

    写入 ``selected_position + out_offsets[i]``，不足处填 ``-1``（与 kernel 一致）。
    """
    rows, _ = scores.shape
    topk = out_indices.shape[1]
    for i in range(rows):
        sl = int(seq_lens[i])
        ks = int(row_starts[i]) if row_starts is not None else 0
        off = int(out_offsets[i])
        window = scores[i, ks : ks + max(sl, 0)].float()
        k = min(topk, window.numel())
        if k <= 0:
            out_indices[i].fill_(-1)
            continue
        idx = torch.topk(window, k, dim=-1).indices
        out_indices[i, :k] = (idx + off).to(out_indices.dtype)
        if k < topk:
            out_indices[i, k:] = -1


# --------------------------------------------------------------------------- #
# indexer Q/K：LayerNorm/RMS + RoPE(neox/gptj) + Hadamard + fp8
# --------------------------------------------------------------------------- #
def _hadamard(n: int, device) -> torch.Tensor:
    """n 阶 Sylvester-Hadamard 矩阵（与 kernel 的蝶形+shfl 网络同构）。"""
    global _HADAMARD
    if _HADAMARD is None or _HADAMARD.shape[0] != n:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat(
                [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
            )
        _HADAMARD = h
    return _HADAMARD.to(device)


def _apply_hadamard(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    h = _hadamard(head_dim, x.device)
    return (x @ h.t()) / math.sqrt(head_dim)


def _rope_leading(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    neox: bool,
) -> torch.Tensor:
    """对 x 的**前 rope_dim 维**做 RoPE；cache 行布局 ``[cos(half) | sin(half)]``。"""
    if rope_dim <= 0:
        return x
    half = rope_dim // 2
    cs = cos_sin_cache.index_select(0, positions.long().reshape(-1)).float()
    # 把 positions 的形状补成与 x 的"非最后一维"对齐，直接广播（x 可能是
    # [T, head_dim] 或 [T, heads, head_dim]）。
    extra = x.dim() - positions.dim() - 1
    shape = (*positions.shape, *([1] * extra), half) if extra > 0 else (*positions.shape, half)
    cos = cs[..., :half].reshape(shape)
    sin = cs[..., half:rope_dim].reshape(shape)
    xr, xh = x[..., :half], x[..., half:rope_dim]
    if neox:
        # NeoX（halves）：前半与后半配对
        roped = torch.cat([xr * cos - xh * sin, xh * cos + xr * sin], dim=-1)
    else:
        # GPT-J（adjacent pair）：相邻两维配对
        xp = x[..., :rope_dim].reshape(*x.shape[:-1], half, 2)
        x0, x1 = xp[..., 0], xp[..., 1]
        roped = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        roped = roped.reshape(*x.shape[:-1], rope_dim)
    return torch.cat([roped, x[..., rope_dim:]], dim=-1)


def fused_q_indexer_rope_first_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    hadamard: bool = True,
    neox_style: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch 拆解：indexer Q 的 RoPE(前 64 维) (+Hadamard) + 逐行 fp8 量化。

    ``weights_out[b, h] = weight[b, h] * weight_scale * q_scale[b, h]``，
    与 kernel 把量化 scale 折进权重输出的做法一致。
    """
    x = _rope_leading(
        q_input.float(), cos_sin_cache, positions, _ROPE_DIM, neox_style
    )
    if hadamard:
        x = _apply_hadamard(x, _HEAD_DIM)
    scale = (x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)) / _FP8_MAX
    q_fp8 = _to_fp8(x / scale)
    weights_out = (
        weight.float() * float(weight_scale) * scale.squeeze(-1)
    ).unsqueeze(-1)
    return q_fp8, weights_out.contiguous()


def fused_k_indexer_norm_rope_store(
    k_input: torch.Tensor,
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
    hadamard: bool = True,
    neox_style: bool = True,
    use_bf16_indexer: bool = False,
) -> None:
    """torch 拆解：indexer K 的 LayerNorm + RoPE + Hadamard + fp8 + paged 写回。

    页布局与 `fused_store_index_cache.cuh` 一致：每页 ``132*page_size`` 字节 =
    ``128*page_size`` fp8 key + ``4*page_size`` fp32 scale。
    """
    x = k_input.float()
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x = (x - mean) * torch.rsqrt(var + eps) * weight.float() + bias.float()
    x = _rope_leading(x, cos_sin_cache, positions, _ROPE_DIM, neox_style)
    if hadamard:
        x = _apply_hadamard(x, _HEAD_DIM)
    # kernel 的 part 4 先 round 到 bf16 再量化（与未融合路径的 scale 对齐）。
    x = x.to(torch.bfloat16).float()
    scale = (x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)) / _FP8_MAX
    q = _to_fp8(x / scale)  # [T, 128]

    if use_bf16_indexer:
        page_bytes = 256 * page_size
        per_tok = 256
    else:
        page_bytes = 132 * page_size
        per_tok = _HEAD_DIM

    loc = out_cache_loc.to(torch.int64).reshape(-1)
    n = min(loc.numel(), q.shape[0])
    if n == 0:
        return
    loc = loc[:n]
    page = torch.div(loc, page_size, rounding_mode="floor")
    off = loc % page_size
    flat_u8 = cache.reshape(-1)

    if use_bf16_indexer:
        dst = (
            (page * page_bytes)[:, None]
            + (off * per_tok)[:, None]
            + torch.arange(per_tok, device=cache.device)[None, :]
        )
        flat_u8[dst.reshape(-1)] = q[:n].to(torch.bfloat16).view(torch.uint8).reshape(-1)
        return

    dst = (
        (page * page_bytes)[:, None]
        + (off * _HEAD_DIM)[:, None]
        + torch.arange(_HEAD_DIM, device=cache.device)[None, :]
    )
    flat_u8[dst.reshape(-1)] = q[:n].view(torch.uint8).reshape(-1)

    # scale 区：页内偏移 128*page_size 起，每 token 4 字节
    cache_f32 = cache.view(torch.float32).reshape(-1)
    s_idx = page * (page_bytes // 4) + (_HEAD_DIM * page_size) // 4 + off
    cache_f32[s_idx] = scale[:n].squeeze(-1)


# --------------------------------------------------------------------------- #
# MLA 拼接 / KV buffer / K 量化
# --------------------------------------------------------------------------- #
def concat_mla_absorb_q(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch 拆解：把 q_nope 与 q_rope 沿最后一维拼接。"""
    return torch.cat([a, b], dim=-1)


def set_mla_kv_buffer(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    num_warps: int = 0,
    *,
    reserved_skip_index: int = 0,
) -> None:
    """torch 拆解：把 ``[nope | rope]`` 行按 ``loc`` 写进 paged KV buffer。"""
    n_loc = loc.shape[0]
    if n_loc == 0:
        return
    buf = kv_buffer.view(kv_buffer.shape[0], -1) if kv_buffer.dim() != 2 else kv_buffer
    nope = cache_k_nope.view(n_loc, -1) if cache_k_nope.dim() != 2 else cache_k_nope
    rope = cache_k_rope.view(n_loc, -1) if cache_k_rope.dim() != 2 else cache_k_rope
    src = torch.cat([nope, rope], dim=-1)
    if reserved_skip_index is not None and reserved_skip_index >= 0:
        keep = loc.reshape(-1) != reserved_skip_index
        buf[loc.reshape(-1)[keep]] = src[keep]
    else:
        buf[loc.reshape(-1)] = src


def _quantize_k_cache_impl(
    k_nope: torch.Tensor, k_rope: torch.Tensor, tile_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(nope_u8 [n, 512+4*ntile], rope_u8 [n, 128])，与 `_quantize_k_cache_ref` 同布局。"""
    n, dim_nope = k_nope.shape
    num_tiles = dim_nope // tile_size
    nt = k_nope.float().reshape(n, num_tiles, tile_size)
    scale = nt.abs().amax(dim=-1) / _FP8_MAX
    q = _to_fp8(nt / scale.unsqueeze(-1)).reshape(n, dim_nope)

    nope_u8 = torch.empty(
        (n, dim_nope + num_tiles * 4), dtype=torch.uint8, device=k_nope.device
    )
    nope_u8[:, :dim_nope] = q.view(torch.uint8)
    nope_u8[:, dim_nope:] = scale.to(torch.float32).view(torch.uint8)

    rope_u8 = (
        k_rope.reshape(n, -1)
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(n, -1)
    )
    return nope_u8, rope_u8


def quantize_k_cache_separate(
    k_nope: torch.Tensor, k_rope: torch.Tensor, tile_size: int = 128
):
    """torch 拆解：分别返回 nope/rope 两段（便于直接喂 `set_mla_kv_buffer`）。"""
    nope2 = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    if k_rope is None or k_rope.numel() == 0:
        rope2 = torch.empty(
            (nope2.shape[0], 0), dtype=nope2.dtype, device=nope2.device
        )
    else:
        rope2 = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope
    nope_u8, rope_u8 = _quantize_k_cache_impl(nope2, rope2, tile_size)
    return nope_u8.view(nope2.shape[0], 1, -1), rope_u8.view(rope2.shape[0], 1, -1)


def quantize_k_cache(cache_k: torch.Tensor) -> torch.Tensor:
    """torch 拆解：``_quantize_k_cache_fast_wrapped``（dv=512, tile=128）。"""
    num_blocks, block_size, _, dim = cache_k.shape
    x = cache_k.reshape(-1, dim)
    nope_u8, rope_u8 = _quantize_k_cache_impl(x[:, :512], x[:, 512:], 128)
    packed = torch.cat([nope_u8, rope_u8], dim=-1)
    return packed.view(num_blocks, block_size, 1, -1).view(torch.float8_e4m3fn)


# fork 的 `dsa_swa_index.append_indexer_swa_triton` 已是向量化 torch，但 padding 行
# （`num_rows > live_rows`，CUDA-graph 预留行）会去写 top-k 尾部列，而 kernel/ref 语义是
# 整行保持 -1；ragged 模式还会因 `row_aux[:num_rows]` 长度不足直接 shape mismatch。
# 这里用**一次 `tolist()` + 全向量化**的实现覆盖它（禁止 `int(tensor[i])` 逐元素循环）。
def append_indexer_swa_triton(
    topk_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_rows: int,
    topk: int,
    window: int,
    alignment: int,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 1,
    page_row_indices: Optional[torch.Tensor] = None,
    ragged_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """拼出 ``top-k ∪ 最近 window`` 的稀疏索引行 + 行长度（向量化 torch）。

    行长度 = ``min(max(seq_len - window, 0), topk) + min(seq_len, window)``。
    主拷贝写 ``cols < indexer_len`` 的 top-k 槽位；``cols ∈ [indexer_len+window, topk)``
    原样搬运；其余列保持 -1（**非 live 的 padding 行整行为 -1**）。window 段写在
    ``[indexer_len, indexer_len+window)``：ragged 用 ``ragged_offsets[row]+pos``，
    paged 用 ``page_table[page_row, pos//page_size]*page_size + pos%page_size``。
    """
    if topk_indices.ndim != 2 or topk_indices.shape[1] != topk:
        raise ValueError(
            f"expected top-k shape [N, {topk}], got {tuple(topk_indices.shape)}"
        )
    if topk_indices.shape[0] < num_rows:
        raise ValueError(
            f"top-k has {topk_indices.shape[0]} rows for {num_rows} launched rows"
        )

    device = topk_indices.device
    output_width = -(-(topk + window) // alignment) * alignment
    output = torch.full(
        (num_rows, output_width), -1, dtype=torch.int32, device=device
    )
    lengths = torch.zeros(num_rows, dtype=torch.int32, device=device)
    if num_rows == 0:
        return output, lengths

    live_rows = seq_lens.numel()
    n_live = min(live_rows, num_rows)
    # 一次性 D2H（每个元数据向量只同步一次），后面全是设备侧向量运算。
    seq_lens_list = seq_lens.tolist()
    seq_len = torch.zeros(num_rows, dtype=torch.int64, device=device)
    if n_live:
        seq_len[:n_live] = torch.as_tensor(
            seq_lens_list[:n_live], dtype=torch.int64, device=device
        )
    seq_len.clamp_(min=0)
    indexer_len = (seq_len - window).clamp(min=0, max=topk)
    swa_len = seq_len.clamp(max=window)

    rows = torch.arange(num_rows, dtype=torch.int64, device=device)
    cols = torch.arange(output_width, dtype=torch.int64, device=device)
    live = (rows < live_rows).unsqueeze(1)

    # ── 1) 主拷贝（只对 live 行；padding 行保持 -1，与 kernel/ref 一致）──────
    src = topk_indices[:num_rows].to(torch.int32)
    safe_cols = cols.clamp(max=max(topk - 1, 0)).unsqueeze(0).expand(num_rows, -1)
    gathered = torch.gather(src, 1, safe_cols)
    head = live & (cols.unsqueeze(0) < indexer_len.unsqueeze(1))
    tail = (
        live
        & (cols.unsqueeze(0) >= (indexer_len + window).unsqueeze(1))
        & (cols.unsqueeze(0) < topk)
    )
    output = torch.where(head | tail, gathered, output)

    # ── 2) window 段 ────────────────────────────────────────────────────────
    wo = torch.arange(window, dtype=torch.int64, device=device)
    valid = live & (wo.unsqueeze(0) < swa_len.unsqueeze(1))
    position = seq_len.unsqueeze(1) - swa_len.unsqueeze(1) + wo.unsqueeze(0)
    if ragged_offsets is not None:
        row_aux = torch.zeros(num_rows, dtype=torch.int64, device=device)
        if n_live:
            row_aux[:n_live] = torch.as_tensor(
                ragged_offsets.tolist()[:n_live], dtype=torch.int64, device=device
            )
        values = row_aux.unsqueeze(1) + position
    else:
        if page_table is None:
            raise ValueError("paged SWA index requires a page table")
        if page_row_indices is None and page_table.shape[0] < live_rows:
            # identity mode: one page-table row per live query row
            raise ValueError(
                f"page table has {page_table.shape[0]} rows for {live_rows} "
                "live query rows and no token_to_batch_idx to map them"
            )
        if page_row_indices is not None:
            page_row = torch.zeros(num_rows, dtype=torch.int64, device=device)
            if n_live:
                page_row[:n_live] = torch.as_tensor(
                    page_row_indices.tolist()[:n_live],
                    dtype=torch.int64,
                    device=device,
                )
        else:
            page_row = rows
        page_id = position // page_size
        valid = (
            valid
            & (page_row.unsqueeze(1) >= 0)
            & (page_row.unsqueeze(1) < page_table.shape[0])
            & (page_id < page_table.shape[1])
        )
        pr = page_row.clamp(min=0, max=page_table.shape[0] - 1).unsqueeze(1)
        pr = pr.expand(-1, window)
        pc = page_id.clamp(min=0, max=page_table.shape[1] - 1)
        flat = (pr * page_table.shape[1] + pc).reshape(-1)
        page = (
            page_table.reshape(-1)
            .index_select(0, flat)
            .view(num_rows, window)
            .to(torch.int32)
        )
        values = page * page_size + (position % page_size)

    dst_cols = (indexer_len.unsqueeze(1) + wo.unsqueeze(0)).clamp(
        max=output_width - 1
    )
    neg = output.new_full(values.shape, -1)
    output.scatter_(1, dst_cols, torch.where(valid, values.to(torch.int32), neg))
    lengths.copy_((indexer_len + swa_len).to(torch.int32))
    return output, lengths


def _get_k_and_s_triton(
    pool,
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_sum: int,
    max_seq_len: int,
):
    """torch 拆解：从 paged indexer 缓存里收集 K 与 scale。

    页布局 ``page_size*(head_dim+4)``：前 ``page_size*head_dim`` 是 K，其后每
    token 4 字节 scale。
    """
    page_size = pool.page_size
    head_dim = pool.index_head_dim
    device = buf.device
    k_out = torch.empty((seq_len_sum, head_dim), dtype=torch.uint8, device=device)
    s_out = torch.empty((seq_len_sum, 4), dtype=torch.uint8, device=device)
    flat = buf.reshape(-1)
    per_page = page_size * (head_dim + 4)
    s_off = page_size * head_dim
    ar_k = torch.arange(head_dim, device=device)
    ar_s = torch.arange(4, device=device)
    offset = 0
    for b in range(seq_lens.shape[0]):
        sl = int(seq_lens[b])
        if sl <= 0:
            continue
        t = torch.arange(sl, device=device)
        pg = page_indices[b, t // page_size].to(torch.int64)
        in_page = t % page_size
        k_idx = (
            (pg * per_page)[:, None]
            + (in_page * head_dim)[:, None]
            + ar_k[None, :]
        )
        k_out[offset : offset + sl] = flat[k_idx.reshape(-1)].view(sl, head_dim)
        s_idx = (
            (pg * per_page + s_off)[:, None]
            + (in_page * 4)[:, None]
            + ar_s[None, :]
        )
        s_out[offset : offset + sl] = flat[s_idx.reshape(-1)].view(sl, 4)
        offset += sl
    return k_out, s_out


def _install_get_k_and_s() -> list:
    from sglang.kernels.ops.attention.dsa.index_buf_accessor import GetKAndS

    GetKAndS.triton = classmethod(
        lambda cls, pool, buf, page_indices, seq_len_tensor, seq_len_sum, max_seq_len: (
            _get_k_and_s_triton(
                pool, buf, page_indices, seq_len_tensor, seq_len_sum, max_seq_len
            )
        )
    )
    return ["GetKAndS.triton"]


# --------------------------------------------------------------------------- #
# gated attention
# --------------------------------------------------------------------------- #
def _install_sigmoid_gate_mul() -> list:
    from sglang.srt.internal.layers.attention.eb5_sigmoid_gate_mul import (
        eb5_sigmoid_gate_mul_torch,
    )

    def eb5_sigmoid_gate_mul(x, gate, *, out=None, use_pdl=True):
        """torch 拆解：kernel 的契约就是 ``x * torch.sigmoid(gate)``（双次 round）。"""
        res = eb5_sigmoid_gate_mul_torch(x, gate)
        if out is not None:
            out.copy_(res)
            return out
        return res

    # 调用点是 `ernie5.py` 的 `from ... import eb5_sigmoid_gate_mul as
    # _eb5_sigmoid_gate_mul` —— 只改定义模块的属性和不到那个别名，必须走
    # patch_everywhere 才能把 `sglang.*` 里所有旧绑定一起重指。
    return patch_everywhere(
        "sglang.srt.internal.layers.attention.eb5_sigmoid_gate_mul",
        "eb5_sigmoid_gate_mul",
        eb5_sigmoid_gate_mul,
    )


# --------------------------------------------------------------------------- #
# FlashMLA（sgl_kernel.flash_mla）：本环境整包缺失，补 torch sparse-decode
# --------------------------------------------------------------------------- #
def _kv_raw_bytes(k_cache: torch.Tensor):
    """把 paged KV 缓冲摊成 1-D，并给出**真实的**页 / token 跨距（元素单位）。

    **注意**：不能直接用 `k_cache.reshape(-1)` 或它的 tensor stride 做 token 寻址 ——
    以 `DeepSeekV4SingleKVPool` 为例，池子按 `ceil_div(page_size*584, 576)*576` 分配每页
    （page_size=256 → 149760），而 `get_swa_key_buffer_radix` 返回的 shape
    `(num_pages, page_size, 1, 584)` / stride `(149760, 584, 584, 1)` 只是"看起来"每
    token 584 字节；`store.cuh` 实际写的是页内 **`[page_size×576 的 value][page_size×8 的
    scale]`**（token 跨距 576）。按 584 读会每 token 漂 8 字节、跨页后完全错位。

    返回 ``(raw_1d, page_size, page_stride, token_stride)``。
    """
    itemsize = k_cache.element_size()
    per = int(k_cache.shape[-1])
    page_size = int(k_cache.shape[1])
    page_stride = int(k_cache.stride(0))
    per_bytes = per * itemsize
    if per_bytes == 584:  # store.cuh 的 FlashMLA fp8 布局：value 区 token 跨距 576 B
        token_stride = 576 // itemsize
    else:  # 656（整块）/ 1024（bf16，无 padding）：页内就是 per
        token_stride = per
    raw = k_cache.as_strided(
        (int(k_cache.shape[0]), page_stride), (page_stride, 1)
    ).reshape(-1)
    # 昆仑不支持对 fp8 张量做高级索引；转成 uint8 字节视图再 gather。
    if raw.dtype in _FP8_DTYPES:
        raw = raw.contiguous().view(torch.uint8)
    return raw, page_size, page_stride, token_stride


def _gather_bytes(raw, flat_idx, width: int) -> torch.Tensor:
    """按扁平字节下标收集 ``width`` 字节的行。"""
    ar = torch.arange(width, device=raw.device)
    return raw[(flat_idx[:, :, None] + ar[None, None, :]).reshape(-1)].reshape(
        *flat_idx.shape, width
    )


def _mla_dequant_kv(k_cache: torch.Tensor, indices: torch.Tensor):
    """按 indices 收集 fp8 paged KV 并反量化成 bf16 ``[B, topk, D]``。

    两种布局（顺序不同，不能混用）：

    * **584 B**（HCA / SWA，`store.cuh::fused_store_flashmla_cache`）：页内
      ``[page_size×576 value][page_size×8 scale]``；value = ``[448 fp8 nope]
      [128 B = 64 bf16 rope]``，scale 是 7 个 UE8M0 字节（每组 64 维）。输出 512 维。
    * **656 B**（DSV4 MLA / `quantize_k_cache`）：单 token 块内 ``[512 fp8][16 B =
      4×fp32 scale][128 B rope]``。输出 576 维。
    """
    device = k_cache.device
    b, _, topk = indices.shape
    idx = indices.reshape(b, topk).to(torch.int64).clamp(min=0)
    raw, page_size, page_stride, token_stride = _kv_raw_bytes(k_cache)
    page = torch.div(idx, page_size, rounding_mode="floor")
    off = idx % page_size
    value_base = page * page_stride + off * token_stride

    if int(k_cache.shape[-1]) == 584:
        dim_nope, tile, n_tiles, rope_off = 448, 64, 7, 448
        nope = _fp8_to_fp32(
            _gather_bytes(raw, value_base, dim_nope)
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        s_base = page * page_stride + 576 * page_size + off * 8
        scale_u8 = _gather_bytes(raw, s_base, n_tiles).to(torch.int32)
        scale = torch.exp2((scale_u8 - 127).to(torch.float32))
        scale = torch.where(scale < 2.0**-126, torch.zeros_like(scale), scale)
    elif int(k_cache.shape[-1]) == 656:
        dim_nope, tile, n_tiles, rope_off = 512, 128, 4, 528
        nope = _fp8_to_fp32(
            _gather_bytes(raw, value_base, dim_nope)
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        scale = (
            _gather_bytes(raw, value_base + dim_nope, n_tiles * 4)
            .contiguous()
            .view(torch.float32)
        )
    else:
        raise NotImplementedError(f"unsupported fp8 KV token width: {k_cache.shape[-1]}")

    nope = nope * scale.repeat_interleave(tile, dim=-1)
    rope = (
        _gather_bytes(raw, value_base + rope_off, 128)
        .contiguous()
        .view(torch.bfloat16)
        .reshape(b, topk, 64)
    )
    return torch.cat([nope.to(torch.bfloat16), rope], dim=-1)


def get_mla_metadata(*args, **kwargs):
    """torch 拆解：FlashMLA 调度元数据（纯 torch 路径不做 tile 调度）。"""
    device = _DEV
    for v in list(kwargs.values()) + list(args):
        if isinstance(v, torch.Tensor):
            device = v.device
            break
    metadata = torch.zeros((1, 8), dtype=torch.int32, device=device)
    num_splits = torch.ones((1,), dtype=torch.int32, device=device)
    return metadata, num_splits


def _mla_gather_kv(k_cache: torch.Tensor, indices: torch.Tensor, is_fp8: bool):
    """按 indices 收集 paged KV 并转成 bf16（fp8 布局反量化，bf16 布局直接 gather）。"""
    b, _, topk = indices.shape
    idx = indices.reshape(b, topk).to(torch.int64).clamp(min=0)
    if is_fp8:
        return _mla_dequant_kv(k_cache, indices)
    raw, page_size, page_stride, token_stride = _kv_raw_bytes(k_cache)
    page = torch.div(idx, page_size, rounding_mode="floor")
    off = idx % page_size
    base = page * page_stride + off * token_stride
    return _gather_bytes(raw, base, token_stride)


def _rows_length(tl, bs: int, heads: int, device):
    if tl is None:
        return None
    t = tl.reshape(bs, -1).to(torch.int64)
    if t.shape[1] == 1:
        t = t.expand(bs, heads)
    return t


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    head_dim_v: Optional[int] = None,
    tile_scheduler_metadata=None,
    num_splits=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    page_size: int = 64,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    **kwargs,
):
    """torch 拆解：FlashMLA **稀疏** decode（可带 attn sink 与第二路 extra KV）。

    ``o[b,h,:] = sum_t softmax_t(scale * q[b,h]·kv[b,t]) * kv[b,t,:head_dim_v]``；
    sink 作为 V=0 的虚拟 key 只进分母（与 fork 的 sink-aware 参考写法一致）。
    返回 ``(o, lse)``，形状/dtype 与真实 API 对齐。
    """
    bs, _, heads, q_dim = q.shape
    dv = head_dim_v if head_dim_v is not None else 512
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q_dim)
    device = q.device

    kv = _mla_gather_kv(k_cache, indices, is_fp8_kvcache)
    kd = kv.shape[-1]
    valid = None
    if topk_length is not None:
        tl_ = _rows_length(topk_length, bs, heads, device)
        valid = torch.arange(kv.shape[1], device=device)[None, None, :] < tl_.unsqueeze(-1)
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        kv2 = _mla_gather_kv(extra_k_cache, extra_indices_in_kvcache, is_fp8_kvcache)
        el = _rows_length(extra_topk_length, bs, heads, device)
        if el is None:
            el = torch.full(
                (bs, heads), kv2.shape[1], dtype=torch.int64, device=device
            )
        v2 = (
            torch.arange(kv2.shape[1], device=device)[None, None, :]
            < el.unsqueeze(-1)
        )
        kv = torch.cat([kv, kv2], dim=1)
        kd = kv.shape[-1]
        valid = v2 if valid is None else torch.cat([valid, v2], dim=-1)

    q3 = q.squeeze(1)  # [B, H, q_dim]
    d = min(q_dim, kd)
    scores = torch.einsum(
        "bhd,btd->bht", q3[..., :d].float(), kv[..., :d].float()
    ) * scale

    if valid is not None:
        scores = scores.masked_fill(~valid, float("-inf"))
    if causal and cache_seqlens is not None:
        sl = cache_seqlens.reshape(bs, 1, 1).to(torch.int64)
        pos = torch.arange(kd, device=device)[None, None, :]
        scores = scores.masked_fill(pos >= sl, float("-inf"))

    if attn_sink is not None:
        sink = attn_sink.float().reshape(-1)[:heads].reshape(1, heads)
        m = torch.maximum(scores.max(dim=-1).values, sink)
        p = torch.exp(scores - m[..., None])
        denom = p.sum(dim=-1) + torch.exp(sink - m)
    else:
        m = scores.max(dim=-1).values
        p = torch.exp(scores - m[..., None])
        denom = p.sum(dim=-1)

    probs = p / denom[..., None]
    out = torch.einsum("bht,btd->bhd", probs, kv[..., :dv].float())
    lse = m + torch.log(denom.clamp(min=1e-20))
    return out.to(q.dtype).unsqueeze(1), lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
):
    """torch 拆解：FlashMLA **稀疏 prefill** 注意力（对应 `sparse_prefill_fwd`）。

    语义取自 fork 自带的 `kernels/aot/tests/test_flashmla.py::reference_torch_prefill`
    （log2 域的 max_logits / lse），并补上后端会用到的 ``attn_sink`` 与 ``topk_length``：
    sink 作为 V=0 的虚拟 key 只进分母；``topk_length`` 之外的列按无效处理。

    Args:
        q: ``[s_q, h_q, d_qk]`` bf16
        kv: ``[s_kv, h_kv, d_qk]`` 或 ``[s_kv, d_qk]`` bf16
        indices: ``[s_q, h_kv, topk]`` int32，``-1`` / ``>= s_kv`` 为无效
    Returns:
        ``(output [s_q,h_q,d_v], max_logits [s_q,h_q], lse [s_q,h_q])``（lse 为 2 底）
    """
    kvs = kv[:, 0, :] if kv.dim() == 3 else kv
    # 不要在这里整体升 fp32：``kv`` 是整块 full latent KV cache（8L 单 rank
    # 也有 ~22 GB bf16；整网更大）。``.float()`` 会再要一份等大的 fp32，
    # 2026-10-20 实测 OOM 42.51 GiB（GPU1 free 22.29 GiB）。先按 index gather
    # 出 [s_q, topk, d]（MB 级），只对这一小块升精度。
    s_q, h_q, _ = q.shape
    s_kv, d_qk = kvs.shape
    topk = indices.shape[-1]
    device = q.device

    idx = indices[:, 0, :].to(torch.int64)  # [s_q, topk]
    invalid = (idx < 0) | (idx >= s_kv)
    if topk_length is not None:
        tl = topk_length.reshape(s_q, -1)[:, 0].to(torch.int64)
        invalid = invalid | (
            torch.arange(topk, device=device)[None, :] >= tl[:, None]
        )
    gathered = kvs[idx.masked_fill(invalid, 0).reshape(-1)].reshape(
        s_q, topk, d_qk
    ).float()

    scores = torch.einsum("shd,std->sht", q.float(), gathered) * sm_scale
    scores = scores * math.log2(math.e)
    scores = scores.masked_fill(invalid.unsqueeze(1), float("-inf"))

    if attn_sink is not None:
        sink = attn_sink.float().reshape(-1)[:h_q].reshape(1, h_q)
        m = torch.maximum(scores.max(dim=-1).values, sink)
        p = torch.exp2(scores - m[..., None])
        denom = p.sum(dim=-1) + torch.exp2(sink - m)
        out = torch.einsum("sht,std->shd", p, gathered[..., :d_v]) / denom[..., None]
        lse = torch.log2(denom.clamp(min=1e-30)) + m
        max_logits = m
    else:
        lse = torch.logsumexp(scores * math.log(2), dim=-1) * math.log2(math.e)
        p = torch.exp2(scores - lse[..., None])
        out = torch.einsum("sht,std->shd", p, gathered[..., :d_v])
        max_logits = scores.max(dim=-1).values

    return out.to(q.dtype), max_logits, lse


def _install_flash_mla() -> list:
    """把上面两个实现绑到插件的 ``sgl_kernel.flash_mla`` 桩上。"""
    import sys

    mod = sys.modules.get("sgl_kernel.flash_mla")
    if mod is None:
        try:
            import sgl_kernel.flash_mla as mod  # noqa: F811
        except Exception:
            return []
    patched = []
    for name, impl in (
        ("get_mla_metadata", get_mla_metadata),
        ("flash_mla_with_kvcache", flash_mla_with_kvcache),
        ("flash_mla_sparse_fwd", flash_mla_sparse_fwd),
    ):
        setattr(mod, name, impl)
        patched.append(f"sgl_kernel.flash_mla.{name}")
    root = sys.modules.get("sgl_kernel")
    if root is not None:
        root.flash_mla = mod
        # 后端在 XPU 分支上从 sgl_kernel 根包直接 import
        root.flash_mla_sparse_fwd = flash_mla_sparse_fwd
        patched.append("sgl_kernel.flash_mla_sparse_fwd")
    return patched


# --------------------------------------------------------------------------- #
# triton MLA backend（模块级 forward 走的这条路径）
# --------------------------------------------------------------------------- #
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ=None,
):
    """torch 拆解：per-seq 的 KV split 数（`triton_backend.get_num_kv_splits` 的调度）。

    逐行复刻 `kernels/ops/attention/metadata.py::get_num_kv_splits_triton`：
    split 数取"按最短序列切"与"按设备核数切"两条路的较大者。
    """
    import math

    if int(num_seq) == 0:
        return
    seq_lens = seq_lens_ptr[: int(num_seq)].to(torch.int64)
    max_seq_len = int(seq_lens.max())
    min_seq_len = int(seq_lens.min())
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    if min_seq_len <= 0:
        min_seq_len = max(max_seq_len, 1)

    max_kv_splits_1 = min(-(-max_seq_len // min_seq_len), int(max_kv_splits))
    kv_chunk_size_1 = -(-max_seq_len // max(max_kv_splits_1, 1))

    ext_seq_len = max_seq_len / 64.0
    log2_ext = math.log2(ext_seq_len) if ext_seq_len > 0 else 0.0
    ext_device_core_count = int(int(device_core_count) * max(log2_ext, 1.0))

    num_kv_group = int(num_head) // int(num_kv_head)
    if num_kv_group == 1:
        token_grid = int(num_seq) * int(num_group) * int(num_head)
    else:
        block_h = min(16, num_kv_group)
        token_grid = (
            int(num_seq) * int(num_group) * (-(-int(num_head) // block_h))
        )
    max_kv_splits_2 = min(
        -(-ext_device_core_count // max(token_grid, 1)), int(max_kv_splits)
    )
    kv_chunk_size_2 = -(-max_seq_len // max(max_kv_splits_2, 1))

    nks = torch.maximum(
        torch.div(
            seq_lens + kv_chunk_size_1 - 1, kv_chunk_size_1, rounding_mode="floor"
        ),
        torch.div(
            seq_lens + kv_chunk_size_2 - 1, kv_chunk_size_2, rounding_mode="floor"
        ),
    )
    out = nks.repeat_interleave(int(num_group)).to(num_kv_splits_ptr.dtype)
    n = min(out.numel(), num_kv_splits_ptr.numel())
    num_kv_splits_ptr[:n] = out[:n]


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    k_scale,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    has_mla=False,
    use_pdl=False,
    page_size=1,
    score_mod=None,
    aux_tensors=None,
    **kwargs,
):
    """torch 拆解：triton MLA decode 注意力（split-KV 只是并行手段，数学不变）。

    实测调用形状：``q [T,H,Lk]``、``k_buffer [total_slots, kv_head, Lk]``（**已摊平**）、
    ``v_buffer [total_slots, kv_head, Lv]``、``kv_indices`` 是 slot id、``kv_indptr`` 是
    CSR 前缀和。逐 token 取 KV → 打分 → softmax（含 ``logit_cap`` / ``sinks``）→ 加权和。
    """
    T, H, Lk = q.shape
    kv_head_num = int(k_buffer.shape[-2])
    Lv = int(v_buffer.shape[-1])
    scale = float(sm_scale) * float(k_scale)
    qf = q.float()
    kb = k_buffer.float()
    vb = v_buffer.float()
    o_flat = o.reshape(T, H, Lv)
    sink_h = None
    if sinks is not None:
        sink_h = sinks.float().reshape(-1)[:H]

    lse_rows = None
    if attn_lse is not None:
        lse_rows = attn_lse.reshape(T, H, -1)

    for t in range(T):
        s0, s1 = int(kv_indptr[t]), int(kv_indptr[t + 1])
        if s1 <= s0:
            o_flat[t] = 0
            if lse_rows is not None:
                lse_rows[t] = 0
            continue
        slot = kv_indices[s0:s1].to(torch.int64)
        k = kb[slot]  # [n, kv_head_num, Lk]
        v = vb[slot]  # [n, kv_head_num, Lv]
        if kv_head_num == 1:
            s = (qf[t] @ k[:, 0, :].t()) * scale
            vv = v[:, 0, :]
        else:
            grp = max(H // kv_head_num, 1)
            kk = k.repeat_interleave(grp, dim=1)
            vv = v.repeat_interleave(grp, dim=1)
            s = torch.einsum("hd,nhd->hn", qf[t], kk) * scale
        if logit_cap and float(logit_cap) > 0:
            s = float(logit_cap) * torch.tanh(s / float(logit_cap))
        m = s.max(dim=-1).values
        if sink_h is not None:
            m = torch.maximum(m, sink_h)
        p = torch.exp(s - m[:, None])
        denom = p.sum(dim=-1)
        if sink_h is not None:
            denom = denom + torch.exp(sink_h - m)
        o_flat[t] = ((p / denom[:, None]) @ vv) * float(v_scale)
        if lse_rows is not None:
            lse = torch.log(denom.clamp(min=1e-30)) + m
            lse_rows[t] = float("-inf")
            lse_rows[t][:, 0] = lse
    return o


def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:
    """torch 拆解：q/k 的原地 RoPE（q/k 的最后一维就是 rope_dim）。"""
    rd = int(rope_dim) or int(cos_sin_cache.size(-1))
    q.copy_(_rope_leading(q.float(), cos_sin_cache, positions, rd, is_neox).to(q.dtype))
    k.copy_(_rope_leading(k.float(), cos_sin_cache, positions, rd, is_neox).to(k.dtype))


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """torch 拆解：``rotary_embedding``（query/key 原地）。"""
    if key is not None:
        apply_rope_inplace(query, key, cos_sin_cache, positions, is_neox=is_neox)
    else:
        rd = int(cos_sin_cache.size(-1))
        query.copy_(
            _rope_leading(query.float(), cos_sin_cache, positions, rd, is_neox).to(
                query.dtype
            )
        )


class _GridKernel:
    """让 torch 拆解也能用 ``kernel[grid](...)`` 的 Triton 调用写法。"""

    def __init__(self, impl):
        self._impl = impl

    def __getitem__(self, grid):
        return self._impl

    def __call__(self, *args, **kwargs):
        return self._impl(*args, **kwargs)


def _install_triton_mla_backend() -> list:
    patched = patch_everywhere(
        "sglang.kernels.ops.attention.metadata",
        "get_num_kv_splits_triton",
        _GridKernel(get_num_kv_splits_triton),
    )
    for attr, impl in (
        ("apply_rope_inplace", apply_rope_inplace),
        ("rotary_embedding", rotary_embedding),
    ):
        try:
            patched.extend(
                patch_everywhere("sglang.kernels.ops.attention.rope", attr, impl)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-mla-adapt] rope.{attr} failed: {exc}")
    try:
        patched.extend(
            patch_everywhere(
                "sglang.kernels.ops.attention.decode_attention",
                "decode_attention_fwd",
                decode_attention_fwd,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eb5-mla-adapt] decode_attention_fwd failed: {exc}")
    return patched


# --------------------------------------------------------------------------- #
# 注入
# --------------------------------------------------------------------------- #
def patch_everywhere(module_name: str, attr: str, impl) -> list:
    """把 ``impl`` 装到定义模块上，并修掉所有已 import 的旧绑定。

    fork 里同一符号常被多处 ``from x import f`` / ``from x import f as _g`` 在 import
    期绑死（例如 `sglang.srt.models.deepseek_v4` 持有 `fused_q_norm_rope` 的旧引用），
    只 setattr 定义模块改不到它们。这里再扫一遍所有 ``sglang.*`` 模块，按**对象身份**
    把所有仍指向旧实现的绑定重指过去。
    """
    import importlib
    import sys

    mod = importlib.import_module(module_name)
    original = getattr(mod, attr, None)
    if original is None:
        return []
    setattr(mod, attr, impl)
    patched = [f"{module_name}.{attr}"]
    for name, other in list(sys.modules.items()):
        if other is None or not name.startswith("sglang."):
            continue
        namespace = getattr(other, "__dict__", None)
        if not namespace:
            continue
        for sym, value in list(namespace.items()):
            if value is original:
                setattr(other, sym, impl)
                patched.append(f"{name}.{sym}")
    return patched


_TARGETS = (
    ("sglang.kernels.ops.quantization.fp8_kernel", "sglang_per_token_group_quant_fp8", sglang_per_token_group_quant_fp8),
    ("sglang.kernels.ops.quantization.fp8_kernel", "sglang_per_token_group_quant_fp8_ue8m0", sglang_per_token_group_quant_fp8_ue8m0),
    ("sglang.kernels.ops.attention.dsv4.topk", "topk_transform_ragged_v2", topk_transform_ragged_v2),
    ("sglang.jit_kernel.internal.dsa", "fused_q_indexer_rope_first_quant", fused_q_indexer_rope_first_quant),
    ("sglang.jit_kernel.internal.dsa", "fused_k_indexer_norm_rope_store", fused_k_indexer_norm_rope_store),
    ("sglang.kernels.ops.attention.concat_mla", "concat_mla_absorb_q", concat_mla_absorb_q),
    ("sglang.kernels.ops.kvcache.set_mla_kv_buffer", "set_mla_kv_buffer", set_mla_kv_buffer),
    ("sglang.kernels.ops.attention.dsa.quant_k_cache", "quantize_k_cache", quantize_k_cache),
    ("sglang.kernels.ops.attention.dsa.quant_k_cache", "quantize_k_cache_separate", quantize_k_cache_separate),
    ("sglang.srt.internal.layers.attention.mla_hca.dsa_swa_index", "append_indexer_swa_triton", append_indexer_swa_triton),
)


def apply() -> list:
    """把 torch 拆解 patch 到 fork 的调用点上。返回已 patch 的符号列表。"""
    import importlib

    patched = []

    if _install_fp8_upcast():
        patched.append("aten::_to_copy(fp8 upcast)")
    patched.extend(_install_runtime_shims())

    for module_name, attr, impl in _TARGETS:
        try:
            patched.extend(patch_everywhere(module_name, attr, impl))
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-mla-adapt] import {module_name} failed: {exc}")

    for hook in (
        _install_deep_gemm_torch,
        _install_get_k_and_s,
        _install_sigmoid_gate_mul,
        _install_flash_mla,
        _install_triton_mla_backend,
    ):
        try:
            patched.extend(hook())
        except Exception as exc:  # noqa: BLE001 - 单个注入失败不影响其余
            print(f"[eb5-mla-adapt] {hook.__name__} failed: {exc}")

    return patched
