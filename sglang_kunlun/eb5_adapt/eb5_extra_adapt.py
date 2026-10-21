"""EB5-lite **HCA / MoE / 前后处理** 的算子级昆仑适配层。

与 `eb5_mla_adapt.py` 同一套做法：把在昆仑上跑不了的融合 kernel（triton 后端缺失 /
fork 自带 nvcc-JIT 建不起来）换成**语义等价的 torch 拆解**，注入 fork 的调用点。

覆盖范围（对应 EB5 kernel 单测里剩余失败的用例）：

* 前处理：`compute_position_triton`、`expand_prefill_causally_triton`、
  `build_page_table_positions_triton`、`build_causal_swa_page_indices_triton`、
  `init_compression_metadata`、`plan_topk_v2`、`CompressorPrefillPlan.generate`
* HCA：`compress_norm_rope_store`、`dequantize_k_cache_paged`、
  `sparse_attn_v4_paged_prefill`、`postmix_quant`
* MoE：`mega_moe_pre_dispatch`、`fused_split_feature_topk`、`situ_and_mul`、`memcpy_triton`
* 后处理：`fused_hc_head_opt`、`alloc_decode_kernel`、`fused_multimax_selu`

其中 `build_page_table_positions` / `build_causal_swa_page_indices` 直接复用 fork 里
**已有的 torch 参考实现**（本来就是给 CPU 输入用的那条分支），只是把 `*_triton` 入口
指过去，避免重复造轮子。

**注意**：同 `eb5_mla_adapt.py` —— 通过只代表拆解语义可跑通，不代表原 CUDA kernel 可跑。
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch

_FP8_MAX = 448.0
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _import_sibling(name: str):
    """导入同目录模块，兼容包内加载与 ``EB5_KUNLUN_ADAPT_DIR`` 平铺加载。"""
    import importlib

    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ModuleNotFoundError as exc:
            # 只吞"相对路径找不到"，不要掩盖被导入模块自身的依赖缺失。
            if exc.name not in (f"{__package__}.{name}", name):
                raise
    return importlib.import_module(name)


def _dsv4_torch_shim_path() -> str:
    """定位同包 ``kernels/dsv4_torch_shim.py``（不依赖工作区落在哪个绝对路径）。"""
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(os.path.dirname(here), "kernels", "dsv4_torch_shim.py")
    if os.path.exists(local):
        return local
    try:
        spec = importlib.util.find_spec("sglang_kunlun")
    except Exception:  # noqa: BLE001 - 解析失败就返回本地路径，由调用方报错
        spec = None
    if spec is not None and spec.origin:
        candidate = os.path.join(
            os.path.dirname(spec.origin), "kernels", "dsv4_torch_shim.py"
        )
        if os.path.exists(candidate):
            return candidate
    return local


# --------------------------------------------------------------------------- #
# 复用 eb5_mla_adapt 的 fp8 基础
# --------------------------------------------------------------------------- #
def _fp8_helpers():
    return _import_sibling("eb5_mla_adapt")


def _fp8_to_fp32(t: torch.Tensor) -> torch.Tensor:
    return _fp8_helpers()._fp8_to_fp32(t)


def _to_fp8(t: torch.Tensor) -> torch.Tensor:
    return _fp8_helpers()._to_fp8(t)


def _group_quant_fp8(x, group_size, eps, ue8m0):
    return _fp8_helpers()._group_quant_fp8(x, group_size, eps, ue8m0)


# --------------------------------------------------------------------------- #
# 前处理
# --------------------------------------------------------------------------- #
def compute_position_triton(
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_seq_lens_sum,
):
    """torch 拆解：逐 batch 生成 positions + extend_start_loc。"""
    device = extend_seq_lens.device
    bs = extend_seq_lens.shape[0]
    total = int(extend_seq_lens_sum)
    positions = torch.empty(total, dtype=torch.int64, device=device)
    start_loc = torch.zeros(bs, dtype=torch.int32, device=device)
    if bs == 0 or total == 0:
        return positions, start_loc
    counts = extend_seq_lens.to(torch.int64)
    ends = torch.cumsum(counts, 0)
    start_loc[1:] = ends[:-1].to(torch.int32)
    has_prefix = extend_prefix_lens.shape[0] == bs
    base = (
        extend_prefix_lens.to(torch.int64)
        if has_prefix
        else torch.zeros(bs, dtype=torch.int64, device=device)
    )
    idx = torch.arange(total, device=device)
    bid = torch.searchsorted(ends, idx, right=True).clamp(max=bs - 1)
    off = idx - start_loc[bid].to(torch.int64)
    positions.copy_(base[bid] + off)
    return positions, start_loc


def expand_prefill_causally_triton(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    num_tokens: int,
    padded_num_tokens: Optional[int],
):
    """torch 拆解：seq_lens → 逐 token 因果长度 + 重复的 req_pool_indices。"""
    from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
        ExpandPrefillCausallyResult,
    )

    device = req_pool_indices.device
    bs = req_pool_indices.shape[0]
    total = (
        int(padded_num_tokens)
        if padded_num_tokens is not None and padded_num_tokens > num_tokens
        else int(num_tokens)
    )
    if bs == 0 or int(num_tokens) == 0:
        return ExpandPrefillCausallyResult(
            seq_lens_casual=torch.empty(total, dtype=torch.int32, device=device),
            req_pool_indices_repeated=torch.empty(
                total, dtype=req_pool_indices.dtype, device=device
            ),
        )
    counts = extend_seq_lens.to(torch.int64)
    ends = torch.cumsum(counts, 0)
    starts = ends - counts
    idx = torch.arange(int(num_tokens), device=device)
    bid = torch.searchsorted(ends, idx, right=True).clamp(max=bs - 1)
    off = idx - starts[bid]
    seq_lens_casual = (seq_lens.to(torch.int64)[bid] - counts[bid] + 1 + off).to(
        torch.int32
    )
    rpi = req_pool_indices[bid]
    if total > int(num_tokens):
        pad = total - int(num_tokens)
        seq_lens_casual = torch.cat(
            [seq_lens_casual, torch.ones(pad, dtype=torch.int32, device=device)]
        )
        rpi = torch.cat([rpi, rpi[-1:].expand(pad)])
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual, req_pool_indices_repeated=rpi
    )


def init_compression_metadata(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
):
    """torch 拆解：c4 / c128 压缩元数据（逐行对应 kernel 的 batch_id）。"""
    sl = seq_lens.to(torch.int32)
    pos = positions.to(torch.int32)
    loc = raw_out_loc.to(torch.int64)

    # 写缓冲只覆盖 num_write_tokens 行（CP 可能给 metadata 补过 padding 行）
    num_write = loc.shape[0]
    assert num_write <= sl.shape[0], (num_write, sl.shape[0])
    slw = sl[:num_write]
    c4_out_loc = torch.where((slw % 4) == 0, loc // 4, torch.zeros_like(loc))
    c128_out_loc = torch.where((slw % 128) == 0, loc // 128, torch.zeros_like(loc))

    c4_positions = pos - (pos % 4)
    c4_seq_lens_raw = sl // 4
    c4_seq_lens_clamp1 = torch.clamp(c4_seq_lens_raw, min=1)

    c128_positions = pos - (pos % 128)
    c128_seq_lens_raw = sl // 128
    c128_seq_lens_clamp1 = torch.clamp(c128_seq_lens_raw, min=1)

    c128_page_indices = None
    if compute_page_indices:
        assert page_table is not None
        assert page_size >= 128 and page_size % 128 == 0, page_size
        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        cur = c128_page_size * max_pages
        if cur > 0:
            offs = torch.arange(cur, device=sl.device)
            page_idx = torch.div(offs, c128_page_size, rounding_mode="floor")
            in_page = offs % c128_page_size
            vals = (
                page_table[:, page_idx.clamp(max=max(max_pages - 1, 0))].to(
                    torch.int32
                )
                * c128_page_size
                + in_page[None, :]
            )
            valid = offs[None, :] < c128_seq_lens_raw[:, None]
            c128_page_indices = torch.where(
                valid, vals, torch.full_like(vals, -1)
            )
        else:
            c128_page_indices = torch.empty(
                (sl.shape[0], 0), dtype=torch.int32, device=sl.device
            )

    return (
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_raw,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = 0) -> torch.Tensor:
    """torch 拆解：v2 top-k transform 的 routing plan。

    plan 只被 `topk_transform_paged_v2` 消费，而后者在本环境同样由 torch 拆解替代，
    故这里给一份全零（形状/ dtype 与 kernel 一致，row0 = {cluster_threshold,
    num_cluster_items}）即可；数值不参与结果。
    """
    return torch.zeros(
        (seq_lens.shape[0] + 1, 2), dtype=torch.int32, device=seq_lens.device
    )


def _prefill_plan_generate(
    compress_ratio,
    req_pool_indices,
    seq_lens,
    extend_lens,
    req_to_token,
    full_to_state,
    swa_page_size,
    ring_size,
    num_q_tokens,
    use_cuda_graph: bool = False,
    use_req_ring: bool = False,
):
    """torch 拆解：``CompressorPrefillPlan.generate``（row k = 第 k 个 query token）。

    CompressPlan（16B, int32[4]）= [seq_len, ragged_id, read_page_0, read_page_1]；
    WritePlan（8B, int32[2]）= [ragged_id, write_loc]。无效行以 seq_len=0 /
    write_loc=-1 标记（与 kernel 的定长契约一致）。
    """
    from sglang.kernels.ops.attention.dsv4.compress import CompressorPrefillPlan

    device = req_to_token.device
    cr = int(compress_ratio)
    sps = int(swa_page_size)
    rs = int(ring_size)
    nq = int(num_q_tokens)

    plan_c = torch.zeros(max(nq, 1), 16, dtype=torch.uint8, device=device)
    plan_w = torch.zeros(max(nq, 1), 8, dtype=torch.uint8, device=device)
    if nq == 0:
        return CompressorPrefillPlan(cr, plan_c[:0], plan_w[:0], None)

    bs = int(seq_lens.shape[0])
    seq = seq_lens.to(device=device, dtype=torch.int64)
    ext = extend_lens.to(device=device, dtype=torch.int64)
    rid_pool = req_pool_indices.to(device=device, dtype=torch.int64)

    k = torch.arange(nq, device=device)
    ends = torch.cumsum(ext, 0)
    b = torch.searchsorted(ends, k, right=True).clamp(max=max(bs - 1, 0))
    j = k - (ends - ext)[b]
    in_req = (j >= 0) & (j < ext[b])
    tok_pos = seq[b] - ext[b] + j
    pos_in_seq = tok_pos + 1

    safe_col = tok_pos.clamp(0, req_to_token.shape[1] - 1)
    safe_row = rid_pool[b].clamp(0, req_to_token.shape[0] - 1)
    full_id = req_to_token[safe_row, safe_col].to(torch.int64)
    full_id = full_id.clamp(0, full_to_state.shape[0] - 1)
    swa_loc = full_to_state[full_id].to(torch.int64)
    state_loc = torch.div(
        torch.div(swa_loc, sps, rounding_mode="floor") * rs + (swa_loc % rs),
        cr,
        rounding_mode="floor",
    )

    is_bnd = in_req & ((pos_in_seq % cr) == 0) & (pos_in_seq > 0)
    first_w = torch.div(seq[b], cr, rounding_mode="floor") * cr
    do_write = is_bnd & (tok_pos >= first_w)

    neg1 = torch.full_like(state_loc, -1)
    mb = is_bnd.to(torch.int64)
    pc = plan_c.view(torch.int32)
    pc[:, 0] = (pos_in_seq * mb).to(torch.int32)
    pc[:, 1] = (k * mb).to(torch.int32)
    pc[:, 2] = (state_loc * mb + neg1 * (1 - mb)).to(torch.int32)
    pc[:, 3] = 0

    mw = do_write.to(torch.int64)
    pw = plan_w.view(torch.int32)
    pw[:, 0] = (k * mw).to(torch.int32)
    pw[:, 1] = (state_loc * mw + neg1 * (1 - mw)).to(torch.int32)

    return CompressorPrefillPlan(cr, plan_c, plan_w, None)


def _decode_plan_generate(
    compress_ratio,
    req_pool_indices,
    req_to_token,
    full_to_state,
    seq_lens,
    swa_page_size,
    ring_size,
    use_req_ring: bool = False,
):
    """torch 拆解：``CompressorDecodePlan.generate``。

    DecodePlan（16B, int32[4]）= [seq_len, ragged_id, state_loc, 0]，逐请求一行。
    """
    from sglang.kernels.ops.attention.dsv4.compress import CompressorDecodePlan

    cr = int(compress_ratio)
    sps = int(swa_page_size)
    rs = int(ring_size)
    device = req_pool_indices.device
    bs = int(seq_lens.shape[0])
    plan_d = torch.zeros(bs, 16, dtype=torch.uint8, device=device)
    if bs == 0:
        return CompressorDecodePlan(cr, plan_d)

    seq = seq_lens.to(device=device, dtype=torch.int64)
    pos = (seq - 1).clamp(min=0)
    safe_col = pos.clamp(0, req_to_token.shape[1] - 1)
    safe_row = req_pool_indices.to(torch.int64).clamp(0, req_to_token.shape[0] - 1)
    full_id = req_to_token[safe_row, safe_col].to(torch.int64)
    full_id = full_id.clamp(0, full_to_state.shape[0] - 1)
    swa_loc = full_to_state[full_id].to(torch.int64)
    state_loc = torch.div(
        torch.div(swa_loc, sps, rounding_mode="floor") * rs + (swa_loc % rs),
        cr,
        rounding_mode="floor",
    )

    pd = plan_d.view(torch.int32)
    pd[:, 0] = seq.to(torch.int32)
    pd[:, 1] = torch.arange(bs, device=device).to(torch.int32)
    pd[:, 2] = state_loc.to(torch.int32)
    pd[:, 3] = 0
    return CompressorDecodePlan(cr, plan_d)


def _prefill_plan_generate_legacy(
    compress_ratio,
    req_pool_indices,
    seq_lens,
    extend_lens,
    num_q_tokens,
    device,
    use_cuda_graph: bool = False,
):
    """torch 拆解：``CompressorPrefillPlan.generate_legacy``（无 req_to_token 的环形布局）。

    legacy 语义：``state_loc = rid * compress_ratio + tok_pos % compress_ratio``
    （见 `test/kernels/deepseek_v4/common.py::LegacyContext.state_loc`）。
    """
    from sglang.kernels.ops.attention.dsv4.compress import CompressorPrefillPlan

    cr = int(compress_ratio)
    nq = int(num_q_tokens)
    dev = req_pool_indices.device
    plan_c = torch.zeros(max(nq, 1), 16, dtype=torch.uint8, device=dev)
    plan_w = torch.zeros(max(nq, 1), 8, dtype=torch.uint8, device=dev)
    if nq == 0:
        return CompressorPrefillPlan(cr, plan_c[:0], plan_w[:0], None)

    seq = seq_lens.reshape(-1).to(dev, torch.int64)
    ext = extend_lens.reshape(-1).to(dev, torch.int64)
    rid = req_pool_indices.reshape(-1).to(dev, torch.int64)
    bs = seq.shape[0]
    k = torch.arange(nq, device=dev)
    ends = torch.cumsum(ext, 0)
    b = torch.searchsorted(ends, k, right=True).clamp(max=max(bs - 1, 0))
    j = k - (ends - ext)[b]
    in_req = (j >= 0) & (j < ext[b])
    tok_pos = seq[b] - ext[b] + j
    pos_in_seq = tok_pos + 1

    state_loc = rid[b] * cr + (tok_pos % cr)
    is_bnd = in_req & ((pos_in_seq % cr) == 0) & (pos_in_seq > 0)
    first_w = torch.div(seq[b], cr, rounding_mode="floor") * cr
    do_write = is_bnd & (tok_pos >= first_w)

    neg1 = torch.full_like(state_loc, -1)
    mb = is_bnd.to(torch.int64)
    pc = plan_c.view(torch.int32)
    pc[:, 0] = (pos_in_seq * mb).to(torch.int32)
    pc[:, 1] = (k * mb).to(torch.int32)
    pc[:, 2] = (state_loc * mb + neg1 * (1 - mb)).to(torch.int32)
    pc[:, 3] = 0

    mw = do_write.to(torch.int64)
    pw = plan_w.view(torch.int32)
    pw[:, 0] = (k * mw).to(torch.int32)
    pw[:, 1] = (state_loc * mw + neg1 * (1 - mw)).to(torch.int32)

    return CompressorPrefillPlan(cr, plan_c, plan_w, None)


# --------------------------------------------------------------------------- #
# HCA
# --------------------------------------------------------------------------- #
def compress_norm_rope_store(
    kv: torch.Tensor,
    plan,
    *,
    norm_weight: torch.Tensor,
    norm_eps: float,
    freq_cis: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
    use_fp4: bool = False,
    bf16_store: bool = False,
    **kwargs,
) -> None:
    """torch 拆解：compressor 的 RMSNorm + RoPE + FlashMLA 布局 paged 写回。

    每 token 584 B：448 fp8 nope + 64 bf16 rope（128 B）+ 7 个 UE8M0 scale（补到 8 B）；
    page stride = ``ceil(584*page_size/576)*576``（同 ``fused_norm_rope_v2.cuh`` 的
    ``kPageBytes``）。只写 plan 标记为有效的行。
    """
    if kv.shape[0] == 0:
        return
    cr = int(plan.compress_ratio)
    is_decode = bool(getattr(plan, "is_decode", False))
    if is_decode:
        # decode：每请求一行，写位直接由 out_loc 给出。
        n = min(kv.shape[0], out_loc.numel())
        sel = torch.arange(n, device=kv.device)
        rows = kv[:n].float()
        positions = (kv.new_zeros(n, dtype=torch.int64) + cr).clamp(min=0)
        plan_rows = plan[1].contiguous().view(torch.int32)[:n]
        positions = (plan_rows[:, 0].to(torch.int64) - cr).clamp(min=0)
        locs = out_loc.to(torch.int64)[:n]
    else:
        pc = plan[1].contiguous().view(torch.int32)  # CompressPlan [N, 4]
        seq_raw = pc[:, 0]
        ragged = pc[:, 1].to(torch.int64) & 0xFFFF
        valid = seq_raw > 0
        sel = torch.nonzero(valid, as_tuple=False).flatten()
        if sel.numel() == 0:
            return
        rows = kv.index_select(0, sel).float()
        positions = (seq_raw.to(torch.int64) - cr).clamp(min=0).index_select(0, sel)
        locs = out_loc.to(torch.int64).index_select(
            0, ragged.index_select(0, sel).clamp(0, out_loc.numel() - 1)
        )

    # Step 1: RMSNorm（无 mean 项，与 kernel 的 rmsnorm 一致）
    rows = rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + norm_eps)
    rows = rows * norm_weight.float()

    # Step 2: RoPE（neox，作用在末尾 rope_dim 维）
    head_dim = rows.shape[-1]
    rope_dim = 64
    nope_dim = head_dim - rope_dim
    half = rope_dim // 2
    fr = torch.view_as_real(freq_cis).flatten(-2).float()
    cs = fr.index_select(0, positions)
    cos, sin = cs[:, :half], cs[:, half:rope_dim]
    nope = rows[:, :nope_dim]
    rope = rows[:, nope_dim:]
    rope_rot = torch.cat(
        [rope[:, :half] * cos - rope[:, half:] * sin,
         rope[:, half:] * cos + rope[:, :half] * sin],
        dim=-1,
    )
    kv_out = torch.cat([nope, rope_rot], dim=-1)  # [M, head_dim]

    # Step 3: 写回 paged cache
    page_bytes = kvcache.shape[1]
    num_pages = kvcache.shape[0]
    locs = locs.clamp(0, num_pages * page_size - 1)
    page = torch.div(locs, page_size, rounding_mode="floor")
    in_page = locs % page_size
    token_base = page * page_bytes + in_page * 576
    u8 = kvcache.contiguous().view(torch.uint8).reshape(-1)

    # nope：448 维分 7 组（64/组）做 UE8M0 fp8 量化
    tiles = 7
    nf = kv_out[:, :nope_dim].float().reshape(-1, tiles, 64)
    amax = nf.abs().amax(dim=-1).clamp(min=1e-4)
    exp = (torch.floor(torch.log2(amax / _FP8_MAX)) + 127.0).clamp(0, 254)
    inv = torch.exp2(127.0 - exp)
    q_u8 = (
        _to_fp8(nf * inv.unsqueeze(-1))
        .reshape(-1, nope_dim)
        .view(torch.uint8)
    )

    rope_bytes = (
        kv_out[:, nope_dim:].to(torch.bfloat16).contiguous().view(torch.uint8)
    ).reshape(-1, rope_dim * 2)

    ar_nope = torch.arange(nope_dim, device=u8.device)
    u8[(token_base[:, None] + ar_nope[None, :]).reshape(-1)] = q_u8.reshape(-1)
    ar_rope = torch.arange(rope_dim * 2, device=u8.device)
    u8[(token_base[:, None] + nope_dim + ar_rope[None, :]).reshape(-1)] = (
        rope_bytes.reshape(-1)
    )
    scale_base = page * page_bytes + 576 * page_size + in_page * 8
    ar_scale = torch.arange(tiles, device=u8.device)
    u8[(scale_base[:, None] + ar_scale[None, :]).reshape(-1)] = exp.to(
        torch.uint8
    ).reshape(-1)


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
):
    """torch 拆解：从 paged fp8 KV 反量化出 bf16 K（656B/token 布局）。"""
    dim_quant = quant_k_cache.shape[-1]
    assert dim_quant == 656, dim_quant
    q = quant_k_cache.reshape(-1, dim_quant)
    idx = page_table_1_flattened.to(torch.int64)
    # 昆仑不支持 fp8 / uint8 张量的 index_select，改用高级索引 gather。
    u8_all = q.contiguous().view(torch.uint8)
    u8 = u8_all[idx]
    dim_nope, dim_rope = 512, 64
    nope = _fp8_to_fp32(
        u8[:, :dim_nope].contiguous().view(torch.float8_e4m3fn)
    )
    scale = u8[:, dim_nope : dim_nope + 16].contiguous().view(torch.float32)
    nope = nope * scale.repeat_interleave(group_size, dim=-1)
    rope = u8[:, dim_nope + 16 :].contiguous().view(torch.bfloat16)
    out = torch.cat([nope.to(torch.bfloat16), rope], dim=-1)
    return out.unsqueeze(1)


def sparse_attn_v4_paged_prefill(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """torch 拆解：prefix(paged) + extend(flat) 两段稀疏 key 的 online-softmax 融合。

    sink 作为 V=0 的虚拟 key 只进分母，与 fork 的 torch 参考写法一致。
    """
    T, H, D = q.shape
    device = q.device
    out = torch.empty((T, H, D), dtype=q.dtype, device=device)
    sink = attn_sink.float()
    for t in range(T):
        p0, p1 = int(kv_indptr_prefix[t]), int(kv_indptr_prefix[t + 1])
        e0, e1 = int(kv_indptr_extend[t]), int(kv_indptr_extend[t + 1])
        parts = []
        if p1 > p0:
            pidx = kv_indices_prefix[p0:p1].to(torch.int64).clamp(min=0)
            parts.append(unified_kv.index_select(0, pidx).float())
        if e1 > e0:
            eidx = kv_indices_extend[e0:e1].to(torch.int64).clamp(min=0)
            parts.append(kv.index_select(0, eidx).float())
        if not parts:
            out[t] = 0
            continue
        kk = torch.cat(parts, dim=0)
        s = (q[t].float() @ kk.t()) * softmax_scale
        m = torch.maximum(s.max(dim=-1).values, sink)
        p = torch.exp(s - m[:, None])
        denom = p.sum(dim=-1) + torch.exp(sink - m)
        out[t] = ((p @ kk) / denom[:, None]).to(q.dtype)
    return out


def postmix_quant(
    x: torch.Tensor, mix: torch.Tensor, num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch 拆解：VHA postmix（``mix`` 矩阵乘）+ wo_a 的 fp8/UE8M0 量化。

    postmix 在 fp32 累加后**round 回 bf16** 再量化（与未融合的两段式一致）；
    scale 返回 DeepGEMM 消费的 packed UE8M0 int32 形式（4 个指数/字）。
    """
    T, G, D = x.shape
    hpg = num_heads // G
    hd = D // hpg

    xr = x.float().view(T, G, hpg, hd).reshape(T * G, hpg, hd)
    mixed = torch.matmul(xr.transpose(1, 2), mix.float().t()).transpose(1, 2)
    mixed = mixed.reshape(T, G, D).to(torch.bfloat16).float()

    scale_groups = D // 128
    mf = mixed.reshape(T, G, scale_groups, 128)
    amax = mf.abs().amax(dim=-1).clamp(min=1e-4)
    exp = (torch.floor(torch.log2(amax / _FP8_MAX)) + 127.0).clamp(0, 254)
    inv = torch.exp2(127.0 - exp)
    codes = _to_fp8(mf * inv.unsqueeze(-1)).reshape(T, G, D)

    e = exp.to(torch.int32)
    words = e.reshape(T, G, scale_groups // 4, 4)
    packed = (
        words[..., 0]
        | (words[..., 1] << 8)
        | (words[..., 2] << 16)
        | (words[..., 3] << 24)
    )
    return codes, packed.to(torch.int32)


# --------------------------------------------------------------------------- #
# MoE
# --------------------------------------------------------------------------- #
def mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int = 32,
) -> None:
    """torch 拆解：MegaMoE 的 pre-dispatch（fp8 量化 + topk 拓宽写回）。"""
    n = x.shape[0]
    if n == 0:
        return
    groups = x.shape[-1] // quant_group_size
    xf = x.float().reshape(n, groups, quant_group_size)
    amax = xf.abs().amax(dim=-1).clamp(min=1e-4)
    exp = (torch.ceil(torch.log2(amax / _FP8_MAX)) + 127.0).clamp(0, 254)
    inv = torch.exp2(127.0 - exp)
    buf_x[:n] = _to_fp8(xf * inv.unsqueeze(-1)).reshape(n, -1)

    # packed UE8M0：一个 int32 字装 4 个组指数（行主序字节布局）
    buf_x_sf.view(torch.uint8)[:n, :groups] = exp.to(torch.uint8)

    buf_topk_idx[:n] = topk_idx.to(torch.int64)
    buf_topk_weights[:n] = topk_weights


def fused_split_feature_topk(
    fused_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    topk: int,
    renormalize: bool,
    scoring_func: str,
    per_expert_scale: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    routed_scaling_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """torch 拆解：split-feature 融合路由 TopK。

    score = score(chunk0) + score(chunk1)（sigmoid/softmax），加 correction bias 后取
    TopK，renormalize 后按 per-expert scale / routed scaling factor 缩放。
    """
    T = fused_logits.shape[0]
    halves = fused_logits.float().view(T, 2, -1)
    if scoring_func == "softmax":
        scores = torch.softmax(halves, dim=-1).sum(dim=1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(halves).sum(dim=1)
    else:
        raise NotImplementedError(f"{scoring_func} is not implemented.")

    router_logits = scores
    if correction_bias is not None:
        scores = scores + correction_bias.float()
    if T == 0:
        return (
            torch.empty((0, topk), dtype=torch.float32, device=fused_logits.device),
            torch.empty((0, topk), dtype=torch.int32, device=fused_logits.device),
            router_logits,
        )

    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if per_expert_scale is not None and per_expert_scale.numel() > 0:
        # 就地展开 fork 的 apply_per_expert_scale（它的 triton kernel 在昆仑跑不了）：
        # 按 logical id 取 scale，shared / padded id（越界）取 1。
        n_experts = per_expert_scale.numel()
        safe = topk_ids.clamp(0, n_experts - 1)
        gathered = per_expert_scale.float()[safe]
        in_range = (topk_ids >= 0) & (topk_ids < n_experts)
        topk_weights = topk_weights * torch.where(
            in_range, gathered, torch.ones_like(gathered)
        )
    if routed_scaling_factor is not None:
        topk_weights = topk_weights * routed_scaling_factor

    if num_token_non_padded is not None:
        # `num_token_non_padded` 是 **device 标量**（原始 kernel 在 device 上读它）。
        # 这里不能 `int(...)`：那是 D2H 同步，本机 XPU 上会挂死（同 memcpy_triton）。
        lim = (
            torch.as_tensor(num_token_non_padded, device=topk_ids.device)
            .reshape(-1)[0]
            .to(torch.long)
        )
        rows = torch.arange(T, device=topk_ids.device) >= lim
        topk_ids = torch.where(
            rows.unsqueeze(-1), torch.full_like(topk_ids, -1), topk_ids
        )
        topk_weights = torch.where(
            rows.unsqueeze(-1), torch.zeros_like(topk_weights), topk_weights
        )

    return topk_weights, topk_ids.to(torch.int32), router_logits


def situ_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:
    """SiTU(SoftCap-GLU) 激活 —— **优先走 KLX 原生算子**。

    fork 的 `kernels/ops/kimi_k3/activation.py` 没有 XPU 分支，无条件走自带 JIT
    （nvcc），所以这里补上 XPU 分支：KLX 的 `xspeedgate_ops.situ_and_mul` 在 P800 上
    可用且数值与 torch 参考一致（实测 max|diff| ~9e-3）。取不到原生算子时退回
    torch 拆解。
    """
    hidden = input.shape[-1] // 2
    if out is None:
        out_dtype = torch.bfloat16 if input.dtype == torch.float32 else input.dtype
        out = input.new_empty(*input.shape[:-1], hidden, dtype=out_dtype)

    if input.is_cuda and input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        try:
            res = torch.ops.xspeedgate_ops.situ_and_mul(
                input.contiguous(),
                float(beta),
                float(linear_beta) if linear_beta is not None else -1.0,
            )
            out.copy_(res.reshape(out.shape).to(out.dtype))
            return out
        except Exception:  # noqa: BLE001 - 原生算子不可用时退回 torch
            pass

    gate = input[..., :hidden].float()
    up = input[..., hidden:].float()
    gate_out = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up_out = linear_beta * torch.tanh(up / linear_beta)
    else:
        up_out = up
    out.copy_((gate_out * up_out).to(out.dtype))
    return out


def memcpy_triton(dst, src, dim, offset, sz, offset_src) -> None:
    """torch 拆解：offset/size 驱动的 device 间拷贝（offset/sz 以"行"为单位）。

    ⚠️ **绝对不能** 用 `int(offset)` / `.item()` / `.cpu()` 把这两个值拉回主机：
    调用方 `sglang.srt.layers.dp_attention.memcpy()` 传进来的是 **0-dim CUDA tensor**
    （`get_dp_local_info()` 返回 `global_num_tokens_gpu` 的切片），原始 triton kernel
    是把它当 **device 指针**在 kernel 里 `tl.load` 的 —— 全程没有 D2H。
    在本机 XPU 上，这种标量 D2H 会让所有 rank 一起挂死（doc/3 记录的同一现象；
    2026-10-19 实测 4 rank 全部停在那一行，`py-spy` 可见）。

    这里改成**纯 device 侧**实现，且**形状必须全程静态**：
    - 不能 `int(offset)` / `.item()` —— 标量 D2H；
    - 也不能用布尔 mask 去索引（`x[mask]` / `nonzero`）—— 输出长度依赖 data，
      torch 会为此做一次隐式 D2H。2026-10-20 实测：4 个 scheduler 全停在
      `src.index_select(0, src_idx[mask])`，现象与 `int(offset)` 完全一样。
    所以用「固定长度的 gather + `torch.where` 回写」：gather 的索引长度恒为
    `dst.shape[0]`（Python int），越界位置 clamp 成合法索引、再用 mask 丢弃。
    代价是每次整块回写 dst（比只拷 n 行多些带宽），换来零同步。
    """
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    if dst.numel() == 0 or src.numel() == 0:
        return

    dev = dst.device
    dst_rows, src_rows = int(dst.shape[0]), int(src.shape[0])
    if dst_rows <= 0 or src_rows <= 0:
        return

    # 标量交给 device；Python int 走这条也没问题（as_tensor 不产生 D2H）。
    off = torch.as_tensor(offset, device=dev).reshape(-1)[0].to(torch.long)
    size = torch.as_tensor(sz, device=dev).reshape(-1)[0].to(torch.long)
    n = size.clamp(min=0, max=max(dst_rows, src_rows))

    # dst 的每个槽位对应哪一行 src（越界会被 clamp，随后由 mask 丢弃）。
    r = torch.arange(dst_rows, device=dev, dtype=torch.long)
    if offset_src:
        rel = off + r                     # dst[0:] ← src[off:]
        mask = (r < n) & (rel < src_rows)
    else:
        rel = r - off                     # dst[off:] ← src[0:]
        mask = (rel >= 0) & (rel < n) & (rel < src_rows)

    gathered = src[rel.clamp(0, src_rows - 1)]
    mask = mask.reshape((dst_rows,) + (1,) * (dst.dim() - 1))
    dst.copy_(torch.where(mask, gathered, dst))


def get_paged_mqa_logits_metadata_torch(
    seq_lens: torch.Tensor, page_size: int, num_sm: int
) -> torch.Tensor:
    """torch 拆解：DSV4 indexer 的 paged-MQA-logits 调度元数据。

    真实实现（``IndexerMetadataKernel``）产出一张 tile 调度表，只有 deep_gemm 的
    logits 内核会读它；纯 torch 路径不消费，故给同形状的零表
    （``[num_sm + 1, 2]`` int32），满足 metadata.py 的 ``isinstance`` 断言。
    """
    seq_lens = seq_lens.view(-1).to(torch.int32)
    return torch.zeros(
        (int(num_sm) + 1, 2), dtype=torch.int32, device=seq_lens.device
    )


def build_swa_token_ids(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa: torch.Tensor,
    swa_window: int,
    total_swa: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """torch 拆解：把每个请求的 SWA 窗口映射到物理 SWA-cache token id。"""
    num_reqs = seq_lens.shape[0]
    device = seq_lens.device
    swa_gather_lens = torch.minimum(
        seq_lens, extend_seq_lens + (swa_window - 1)
    ).to(torch.int32)
    swa_first_pos = (seq_lens - swa_gather_lens).to(torch.int32)
    swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    if num_reqs:
        swa_offsets[1:] = torch.cumsum(swa_gather_lens, dim=0).to(torch.int32)
    swa_token_ids = torch.empty(total_swa, dtype=torch.int32, device=device)
    if int(total_swa) == 0:
        return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets

    for b in range(num_reqs):
        n = int(swa_gather_lens[b])
        if n <= 0:
            continue
        start = int(swa_first_pos[b])
        off = int(swa_offsets[b])
        pos = torch.arange(start, start + n, device=device)
        full_ids = req_to_token[int(req_pool_indices[b]), pos].to(torch.int64)
        full_ids = full_ids.clamp(0, full_to_swa.numel() - 1)
        swa_token_ids[off : off + n] = full_to_swa[full_ids].to(torch.int32)
    return swa_token_ids, swa_first_pos, swa_gather_lens, swa_offsets


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    out_indices: Optional[torch.Tensor] = None,
    out_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch 拆解：把压缩区 topk 与 SWA 窗口拼成一行稀疏索引。

    前 ``topk_len`` 位写 ``topk_indices + compressed_base``，随后 ``swa_len`` 位写
    ``swa_base + 窗口内相对位置``；行宽对齐到 128。
    """
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    device = topk_indices.device
    width = -(-(int(topk) + int(window_size)) // 128) * 128
    combined_indices = (
        out_indices
        if out_indices is not None
        else torch.full((num_tokens, width), -1, dtype=torch.int32, device=device)
    )
    combined_lens = (
        out_lens
        if out_lens is not None
        else torch.zeros(num_tokens, dtype=torch.int32, device=device)
    )
    if num_tokens == 0:
        return combined_indices, combined_lens

    qsl = query_start_loc.to(torch.int64)
    base = int(qsl[0])
    for b in range(num_reqs):
        q_start = int(qsl[b]) - base
        q_end = int(qsl[b + 1]) - base
        seq_len = int(seq_lens[b])
        gather_len = int(gather_lens[b])
        cbase = int(compressed_base[b])
        sbase = int(swa_base[b])
        start_pos = seq_len - (q_end - q_start)
        gather_start = seq_len - gather_len
        for token_idx in range(q_start, q_end):
            pos = start_pos + (token_idx - q_start)
            topk_len = min((pos + 1) // int(compress_ratio), int(topk))
            swa_len = min(pos + 1, int(window_size))
            row = combined_indices[token_idx]
            if topk_len > 0:
                row[:topk_len] = topk_indices[token_idx, :topk_len] + cbase
            if swa_len > 0:
                ar = torch.arange(swa_len, device=device, dtype=torch.int32)
                row[topk_len : topk_len + swa_len] = (
                    sbase + ar + (pos - swa_len + 1 - gather_start)
                )
            combined_lens[token_idx] = topk_len + swa_len
    return combined_indices, combined_lens


def dequantize_k_cache_paged_v4(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """torch 拆解：DSV4 版 ``dequantize_k_cache_paged``（448 fp8 + 64 bf16 + 7 scale）。

    与 fork 自带的 ``dequantize_k_cache_paged_ref`` 同布局，但 fp8→fp32 走 LUT，
    绕开昆仑未实现的设备侧 fp8 cast。
    """
    dim_nope, dim_rope, tile = 448, 64, 64
    num_tiles, nope_rope_bytes, pad_scale = 7, 576, 8
    u8 = quant_k_cache.view(torch.uint8)
    bytes_per_page = u8.shape[-1]
    flat_u8 = u8.reshape(-1)
    device = quant_k_cache.device
    loc = page_table_1_flattened.to(torch.int64)
    page_idx = torch.div(loc, page_size, rounding_mode="floor")
    in_page = loc % page_size
    page_byte_base = page_idx * bytes_per_page
    token_base = page_byte_base + in_page * nope_rope_bytes

    ar_nope = torch.arange(dim_nope, device=device)
    nope_u8 = flat_u8[(token_base[:, None] + ar_nope[None, :]).reshape(-1)].reshape(
        -1, dim_nope
    )
    nope = _fp8_to_fp32(nope_u8.contiguous().view(torch.float8_e4m3fn))

    ar_s = torch.arange(num_tiles, device=device)
    s_base = page_byte_base + page_size * nope_rope_bytes + in_page * pad_scale
    scale_u8 = (
        flat_u8[(s_base[:, None] + ar_s[None, :]).reshape(-1)]
        .reshape(-1, num_tiles)
        .to(torch.int32)
    )
    scale = torch.exp2((scale_u8 - 127).to(torch.float32))
    scale = torch.where(scale < 2.0**-126, torch.zeros_like(scale), scale)
    nope = nope * scale.repeat_interleave(tile, dim=1)

    flat_bf16 = u8.view(torch.bfloat16).reshape(-1)
    r_base = torch.div(token_base + dim_nope, 2, rounding_mode="floor")
    ar_r = torch.arange(dim_rope, device=device)
    rope = flat_bf16[(r_base[:, None] + ar_r[None, :]).reshape(-1)].reshape(
        -1, dim_rope
    )

    res = torch.cat([nope.to(torch.bfloat16), rope], dim=-1)
    if out is not None:
        out[:, 0, :] = res
        return out
    return res.unsqueeze(1)


def linear_bf16_fp32(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    hpc_kernel_min_m: Optional[int] = None,
) -> torch.Tensor:
    """torch 拆解：bf16 x bf16 -> fp32 线性层。

    上游依赖 ``aten::mm.dtype``（``torch.mm(..., out_dtype=torch.float32)``），昆仑
    XPU 没有这个重载，改成 fp32 累加后转回。
    """
    return torch.mm(x.float(), y.float().t()).to(torch.float32)


def moe_fused_gate(
    scores: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    *,
    scoring_func: str = "sigmoid",
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    num_expert_group: int = 1,
    topk_group: int = 1,
    num_fused_shared_experts: int = 0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch 拆解：router TopK（sigmoid/softmax + correction bias + TopK + renorm）。"""
    s = scores.float()
    if scoring_func == "softmax":
        s = torch.softmax(s, dim=-1)
    else:
        s = torch.sigmoid(s)
    sel = s + bias.float() if bias is not None else s
    k = int(topk)
    # bias 只参与“选谁”，权重取自未加 bias 的分数（DeepSeek 系 grouped-topk 约定）。
    indices = torch.topk(sel, k, dim=-1).indices
    weights = s.gather(1, indices)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    if apply_routed_scaling_factor_on_output:
        weights = weights * float(routed_scaling_factor)
    return weights, indices.to(torch.int32)


# --------------------------------------------------------------------------- #
# KV 索引流
# --------------------------------------------------------------------------- #
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride,
    ENTRY_PAGE_SIZE: int = 1,
    TOKEN_BLOCK_PARALLEL: bool = False,
) -> None:
    """torch 拆解：把逐请求的 token id 收进扁平的 CSR ``kv_indices`` 流。"""
    device = req_to_token_ptr.device
    bs = req_pool_indices_ptr.shape[0]
    for pid in range(bs):
        rpi = int(req_pool_indices_ptr[pid])
        off = int(kv_indptr[pid])
        kv_start = int(kv_start_idx[pid]) if kv_start_idx is not None else 0
        n = int(page_kernel_lens_ptr[pid])
        if n <= 0:
            continue
        pos = torch.arange(kv_start, kv_start + n, device=device)
        if ENTRY_PAGE_SIZE == 1:
            data = req_to_token_ptr[rpi, pos]
        else:
            entry = req_to_token_ptr[rpi, pos // ENTRY_PAGE_SIZE]
            data = entry.to(torch.int64) * ENTRY_PAGE_SIZE + (pos % ENTRY_PAGE_SIZE)
        kv_indices_ptr[off : off + n] = data


# --------------------------------------------------------------------------- #
# 后处理
# --------------------------------------------------------------------------- #
def fused_hc_head_opt(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """torch 拆解：RMSNorm + hc 线性 + sigmoid-gate 加权和。"""
    T, hc_mult, hidden = x.shape
    y = torch.empty((T, hidden), dtype=x.dtype, device=x.device)
    if T == 0:
        return y
    xf = x.float().reshape(T, -1)
    sumsq = (xf * xf).sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sumsq / (hc_mult * hidden) + norm_eps)
    mix = xf @ hc_fn.float().t()
    pre = torch.sigmoid(mix * rsqrt * hc_scale.float() + hc_base.float()) + hc_eps
    y.copy_(
        torch.einsum("tm,tmd->td", pre, x.float()).to(y.dtype)
    )
    return y


def _alloc_decode_kernel_impl(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: int,
    page_size: int,
) -> None:
    """decode 每步分配新 KV slot 的索引（纯 torch，无 D2H、无 Python 逐请求循环）。

    ``new_i = ceil(seq/page) - ceil((seq-1)/page)``；需要新页的请求取
    ``free_pages[前缀和]``，否则 ``last_loc + 1``。向量化后不再依赖 ``bs_upper``。
    """
    seq_lens = seq_lens_ptr.to(torch.int64)
    pre_lens = seq_lens - 1
    num_after = (seq_lens + page_size - 1) // page_size
    num_before = (pre_lens + page_size - 1) // page_size
    new_flags = (num_after - num_before).to(torch.int64)
    needs_new = new_flags > 0

    # 分支条件用 ``free_page_ptr.numel()``（host 已知），避免 ``.item()``/``.any()``
    # 这类 D2H —— 昆仑设备半坏时会永久挂住。
    if free_page_ptr.numel() > 0:
        start = torch.cumsum(new_flags, 0) - new_flags
        page = free_page_ptr[start.clamp(min=0, max=free_page_ptr.numel() - 1)]
        val_new = page.to(torch.int64) * page_size
    else:
        val_new = torch.zeros_like(last_loc_ptr, dtype=torch.int64)
    out = torch.where(needs_new, val_new, last_loc_ptr.to(torch.int64) + 1)
    out_indices.copy_(out.to(out_indices.dtype))


def _install_alloc_decode() -> list:
    """把 ``PagedTokenToKVPoolAllocator.alloc_decode`` 整个换成 torch 实现。

    只替换 kernel 符号是不够的：调用点是 ``alloc_decode_kernel[(bs,)](...)``，
    把符号换成普通函数会 ``'function' object is not subscriptable``；而包一层
    ``__getitem__`` 又正是 doc/3 §12.6 不推荐的做法。所以这里在**方法层**替换。
    """
    import importlib

    paged = importlib.import_module("sglang.srt.mem_cache.allocator.paged")
    cls = paged.PagedTokenToKVPoolAllocator
    if getattr(cls.alloc_decode, "__eb5_kunlun__", False):
        return []

    from sglang.srt.utils import get_num_new_pages

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )
        bs = len(seq_lens)
        if bs > len(self.free_pages):
            self.merge_and_sort_free()
        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        _alloc_decode_kernel_impl(
            seq_lens, last_loc, self.free_pages, out_indices, bs, self.page_size
        )
        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
        )
        if num_new_pages > len(self.free_pages):
            return None
        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    alloc_decode.__eb5_kunlun__ = True
    cls.alloc_decode = alloc_decode
    return ["sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.alloc_decode"]


class _GridKernel:
    """让 torch 拆解也能用 ``kernel[grid](...)`` 的 Triton 调用写法。"""

    def __init__(self, impl):
        self._impl = impl

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            return self._impl(*args, **kwargs)

        return run

    def __call__(self, *args, **kwargs):
        return self._impl(*args, **kwargs)


def fused_multimax_selu(
    x: torch.Tensor, ranges: torch.Tensor, ts: torch.Tensor, *, out=None
) -> torch.Tensor:
    """torch 拆解：multimax-SELU 分段激活（每步按 bf16 舍入，对齐 eager 图）。"""
    if out is None:
        out = torch.empty_like(x)
    res = x.clone()
    res = res + ts[0] * torch.relu(ranges[0] - x)
    res = res + ts[1] * torch.relu(x - ranges[1])
    res = res + ts[2] * torch.relu(ranges[2] - x) ** 2
    res = res + ts[3] * torch.relu(x - ranges[3]) ** 2
    out.copy_(res)
    return out


# --------------------------------------------------------------------------- #
# 注入
# --------------------------------------------------------------------------- #
def build_causal_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
) -> torch.Tensor:
    """torch 拆解：因果 SWA 窗口内的 slot 索引。

    不复用 fork 的同名 torch 参考实现：那份在 gather 前只把无效位置**改写成 0 索引**
    而没在结果上回填 -1，`full_to_swa_mapping[-1]` 会绕回最后一个元素，与 kernel 语义
    不符（kernel 输出 -1 哨兵）。
    """
    device = seq_lens_casual.device
    pos = seq_lens_casual.to(torch.int64) - 1
    offsets = pos[:, None] - torch.arange(
        swa_window, dtype=torch.int64, device=device
    )[None, :]
    invalid = offsets < 0
    safe = offsets.clamp(min=0)
    raw = req_to_token[req_pool_indices_repeated.to(torch.int64)[:, None], safe]
    raw = raw.to(torch.int64).clamp(0, max(full_to_swa_mapping.numel() - 1, 0))
    swa = full_to_swa_mapping[raw].to(torch.int32)
    swa = swa.masked_fill(invalid, -1)
    padded_width = (
        (swa_window + page_index_aligned_size - 1) // page_index_aligned_size
    ) * page_index_aligned_size
    if padded_width == swa_window:
        return swa
    return torch.nn.functional.pad(swa, (0, padded_width - swa_window), value=-1)


def _patch_everywhere(module_name: str, attr: str, impl) -> list:
    """把 ``impl`` 装到定义模块上，并修掉所有已 import 的旧绑定。

    fork 里很多包 `__init__` 会 ``from .mod import f`` 再导出（例如
    `sglang.kernels.ops.attention.dsv4.compress_norm_rope_store` 既在 `.compress`
    里定义、又被包 `__init__` 重绑定）。只改定义模块的话，走包路径 import 的调用方
    仍拿到旧函数。
    """
    import importlib
    import sys

    mod = importlib.import_module(module_name)
    original = getattr(mod, attr, None)
    if original is None:
        return []
    setattr(mod, attr, impl)
    patched = [f"{module_name}.{attr}"]
    # 再扫一遍所有 sglang 模块，把**任意名字**下仍指向旧对象的绑定重指过去：
    # fork 里 `from x import f as _g`（改名导入）很常见，只按名字匹配会漏。
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


def _install_dsv4_metadata_kernels() -> list:
    """把 `*_triton` 入口指向 fork 里已有的 torch 参考实现。"""
    import sglang.kernels.ops.attention.dsv4_attn_metadata_kernels as m

    patched = []
    # 只复用 page_table_positions 那份（它对 device 张量同样正确）。
    # build_causal_swa_page_indices 的 torch 参考在 -1 哨兵上有 bug，用本模块的实现
    # （见 build_causal_swa_page_indices_triton），不在此覆盖。
    pairs = (("build_page_table_positions_triton", "build_page_table_positions"),)
    for triton_name, torch_name in pairs:
        impl = getattr(m, torch_name, None)
        if impl is not None and hasattr(m, triton_name):
            setattr(m, triton_name, impl)
            patched.append(f"{m.__name__}.{triton_name} -> {torch_name}")
    return patched


def _deep_gemm_get_num_sms() -> int:
    """torch 拆解：deep_gemm 的 SM 数（只作调度元数据的入参，纯 torch 路径不消费）。"""
    return 32


def _deep_gemm_get_paged_mqa_logits_metadata(seqlens=None, *args, **kwargs):
    """torch 拆解：paged MQA logits 的调度元数据（占位空张量）。"""
    device = seqlens.device if isinstance(seqlens, torch.Tensor) else None
    return torch.empty(0, dtype=torch.int32, device=device)


def _install_deep_gemm_extra() -> list:
    """补 deep_gemm 上被 fork 直接调用、但插件桩没提供的两个符号。

    `srt/layers/attention/dsv4/metadata.py` 无条件调 `deep_gemm.get_num_sms()` 与
    `deep_gemm.get_paged_mqa_logits_metadata(...)`（无论是否真的走 deep_gemm 内核）。
    """
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
        ("get_num_sms", _deep_gemm_get_num_sms),
        ("get_paged_mqa_logits_metadata", _deep_gemm_get_paged_mqa_logits_metadata),
    ):
        if getattr(mod, name, None) is None:
            setattr(mod, name, impl)
            patched.append(f"deep_gemm.{name}")
    return patched


def _dsv4_shim():
    """按文件路径加载插件的 dsv4_torch_shim（复用它的 compress_forward 拆解）。"""
    import importlib.util
    import sys
    import types

    mod = sys.modules.get("_eb5_dsv4_torch_shim")
    if mod is not None:
        return mod
    stub = "sglang_kunlun.hooks.layers.attention.nsa.triton_kernel"
    if stub not in sys.modules:
        # DSV4 的 indexer 量化路径会真的调它，要给能用的实现（同上，别 import 插件模块）。
        def _act_quant_kunlun(x, block_size=128, scale_fmt=None):
            import kunlun_ops

            assert x.is_contiguous(), "Input tensor must be contiguous"
            assert x.size(-1) % block_size == 0
            assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            x_shape = x.shape
            x_blocks = x.view(-1, block_size).contiguous()
            y_int8 = torch.empty(x_blocks.shape, dtype=torch.int8, device=x.device)
            y_max = torch.empty(x_blocks.shape[0], dtype=torch.float32, device=x.device)
            kunlun_ops.quant2d(x_blocks, y_int8, y_max, force_sdnn=True)
            return y_int8.view(x_shape), y_max.view(x_shape[0], -1)

        m = types.ModuleType(stub)
        m.act_quant_kunlun = _act_quant_kunlun
        sys.modules[stub] = m
    path = _dsv4_torch_shim_path()
    spec = importlib.util.spec_from_file_location("_eb5_dsv4_torch_shim", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_eb5_dsv4_torch_shim"] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_compress_forward() -> list:
    """`compress_forward`（FlashCompress{4,128}Kernel）→ 复用插件的 torch 拆解。"""
    shim = _dsv4_shim()
    impl = getattr(shim, "_compress_forward_torch", None)
    if impl is None:
        return []
    return _patch_everywhere(
        "sglang.kernels.ops.attention.dsv4.compress", "compress_forward", impl
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool = False,
):
    """torch 拆解：把 token 按 expert 分桶、并按 ``block_size`` 补齐成块。

    与 `srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py` 的
    triton 包装同契约：返回 ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``；
    无效 id（``-1`` / ``>= num_experts``）落进末尾那个额外桶，其块 ``expert_ids = -1``。
    缓冲区尺寸沿用上游算法（见其 `max_num_tokens_padded`）。
    """
    device = topk_ids.device
    nbuckets = int(num_experts) + 1
    flat = topk_ids.reshape(-1).to(torch.int64)
    numel = int(flat.numel())
    if numel < nbuckets:
        max_padded = numel * block_size
    else:
        max_padded = numel + nbuckets * (block_size - 1)
    sorted_ids = torch.full((max_padded,), numel, dtype=torch.int32, device=device)
    max_blocks = -(-max_padded // block_size)
    expert_ids = torch.full((max_blocks,), -1, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=device)
    if numel == 0:
        return sorted_ids, expert_ids, num_tokens_post_pad

    bucket = flat.clone()
    invalid = (bucket < 0) | (bucket >= int(num_experts))
    bucket[invalid] = int(num_experts)  # 末尾额外桶
    counts = torch.bincount(bucket, minlength=nbuckets)[:nbuckets]
    padded = -(-counts // block_size) * block_size
    starts = torch.cumsum(padded, 0) - padded
    total = min(int(padded.sum()), max_padded)

    order = torch.argsort(bucket, stable=True)
    exp_of = bucket[order]
    pos = starts[exp_of] + (
        torch.arange(numel, device=device) - starts[exp_of]
    )
    keep = pos < max_padded
    sorted_ids[pos[keep]] = order[keep].to(torch.int32)

    nblk = padded // block_size
    blk_exp = torch.repeat_interleave(
        torch.arange(nbuckets, device=device), nblk
    ).to(torch.int32)
    blk_exp[blk_exp == int(num_experts)] = -1  # 无效桶的块不参与 GEMM
    n = min(int(blk_exp.numel()), max_blocks)
    expert_ids[:n] = blk_exp[:n]
    num_tokens_post_pad[0] = total
    return sorted_ids, expert_ids, num_tokens_post_pad


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    bias,
    C: torch.Tensor,
    A_scale,
    B_scale,
    B_zp,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config,
    compute_type=None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape=None,
    no_combine: bool = False,
    a_use_tma: bool = False,
    b_use_tma: bool = False,
    c_sorted: bool = False,
    filter_expert: bool = True,
    fuse_sum_all_reduce: bool = False,
    router_topk: int = 1,
    fuse_add_to_output: bool = False,
    add_output_mask=None,
    mask_output: bool = False,
    lora_preserve_base: bool = False,
    fuse_swiglu: bool = False,
) -> None:
    """torch 拆解：MoE 的分块（grouped）专家 GEMM。

    逐块复刻 triton kernel 的寻址：``offs_token = sorted_token_ids[block]``（padding 值为
    ``num_tokens*top_k``），A 的行号是 ``offs_token // top_k``，块所属 expert 取
    ``expert_ids[block]``（``-1`` 跳过），``mul_routed_weight`` 时再乘
    ``topk_weights[offs_token]``。

    只支持未量化（bf16/fp16）路径；量化 / TMA / LoRA / fused-swiglu 等分支用断言挡住。
    """
    assert not (use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16), (
        "torch MoE 拆解不支持量化权重路径"
    )
    assert not (fuse_add_to_output or mask_output or fuse_sum_all_reduce or fuse_swiglu)
    assert not c_sorted, "c_sorted(TMA) 路径未实现"

    block_m = int(config["BLOCK_SIZE_M"])
    num_valid = A.shape[0] * int(top_k)
    total_pad = int(num_tokens_post_padded[0])
    if total_pad == 0:
        return
    nblocks = -(-total_pad // block_m)
    Af = A.float()
    Bf = B.float()
    tw = topk_weights.reshape(-1)
    for b in range(nblocks):
        e = int(expert_ids[b])
        if e < 0:
            continue
        rows = sorted_token_ids[b * block_m : (b + 1) * block_m].to(torch.int64)
        keep = rows < num_valid
        if not bool(keep.any()):
            continue
        rr = rows[keep]
        arows = torch.div(rr, int(top_k), rounding_mode="floor").clamp(
            max=A.shape[0] - 1
        )
        out = Af[arows] @ Bf[e].t()
        if bias is not None:
            out = out + bias.float()[e]
        if mul_routed_weight:
            out = out * tw[rr].float()[:, None]
        # C 可能是 3-D（如 [total_tokens, topk, N]），统一摊成 (-1, last) 再按行写
        C.reshape(-1, C.shape[-1]).index_copy_(0, rr, out.to(C.dtype))


def _install_fused_moe_kernel() -> list:
    return _patch_everywhere(
        "sglang.kernels.ops.moe.fused_moe_triton_kernels",
        "invoke_fused_moe_kernel",
        invoke_fused_moe_kernel,
    )


def run_activation(
    op_name: str,
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    """gated 激活 —— ``silu`` **优先走 KLX 原生** ``kunlun_ops.swiglu``。

    fork 的 `kernels/ops/activation/activation.py` 只有 JIT（nvcc）路径。``expert_ids``
    给了的话，路由到 ``-1`` 的行保持 ``out`` 原值（与 kernel 的 filtered 变体一致）。
    """
    hidden = input.shape[-1] // 2
    if out is None:
        out = input.new_empty(*input.shape[:-1], hidden)

    res = None
    if op_name == "silu" and input.is_cuda:
        try:
            import kunlun_ops

            n = input.reshape(-1, hidden * 2).shape[0]
            buf = torch.empty((n, hidden), dtype=out.dtype, device=input.device)
            kunlun_ops.swiglu(x=input.reshape(-1, hidden * 2).contiguous(), y=buf)
            res = buf.reshape(out.shape)
        except Exception:  # noqa: BLE001 - 原生不可用则退回 torch
            res = None

    if res is None:
        gate = input[..., :hidden].float()
        up = input[..., hidden:].float()
        if op_name == "silu":
            res = torch.nn.functional.silu(gate) * up
        elif op_name == "gelu":
            res = torch.nn.functional.gelu(gate) * up
        elif op_name == "gelu_tanh":
            res = torch.nn.functional.gelu(gate, approximate="tanh") * up
        else:
            raise NotImplementedError(f"Unsupported activation: {op_name}")
        res = res.to(out.dtype)

    if expert_ids is None:
        out.copy_(res)
        return out

    eids = expert_ids.to(torch.int32).reshape(-1)
    n = input.reshape(-1, hidden * 2).shape[0]
    rows = (torch.arange(n, device=input.device) // max(int(expert_step), 1)).clamp(
        max=max(eids.numel() - 1, 0)
    )
    valid = (eids[rows] >= 0).reshape(-1, 1)
    merged = torch.where(valid, res.reshape(n, hidden), out.reshape(n, hidden))
    out.copy_(merged.reshape(out.shape))
    return out


def silu_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    """`kernels/ops/activation/activation.py::silu_and_mul` 的 KLX/torch 版本。"""
    return run_activation("silu", input, out, expert_ids, expert_step)


def _install_activation() -> list:
    return _patch_everywhere(
        "sglang.kernels.ops.activation.activation", "run_activation", run_activation
    ) + _patch_everywhere(
        "sglang.kernels.ops.activation.activation", "silu_and_mul", silu_and_mul
    )


def _install_expert_location_pad() -> list:
    """绕开昆仑上会 task timeout 的 ``constant_pad1d_ncw``（int64 常量 pad）。

    ``eplb/expert_location.py::ExpertLocationMetadata._init_raw`` 里唯一一处
    ``F.pad(x, (0, n), value=-1)`` 是 3 维 int64 张量，落到底层就是
    ``xpukernel_xpu3::constant_pad1d_ncw<long>`` —— 这台机器的 XPU 上该 kernel 会
    报 ``KL_XID_KERNEL_EXCEPTION / reason[29] task timeout``，随后整个设备 session
    被 taint，紧接着的 ``xpu_wait``（就是 ``.cpu()``）永久挂住，表现为所有 rank
    卡死在 ``_init_raw``。

    这里把该模块里的 ``F`` 换成只覆盖 ``pad`` 的薄壳：命中"只 pad 最后一维的 int64
    常量 pad"时用 ``torch.cat`` 等价实现，其余调用原样转发 ``torch.nn.functional``。
    """
    import torch.nn.functional as _real_F

    import sglang.srt.eplb.expert_location as _el

    if getattr(_el, "_eb5_pad_shim", False):
        return []

    class _FPadShim:
        """``torch.nn.functional`` 的透传壳，只重写 ``pad``。"""

        def __getattr__(self, name):
            return getattr(_real_F, name)

        @staticmethod
        def pad(t, pad, mode="constant", value=None):
            if (
                mode == "constant"
                and t.is_cuda
                and t.dtype == torch.int64
                and len(pad) == 2
                and int(pad[0]) == 0
            ):
                n = int(pad[1])
                fill = 0 if value is None else value
                if n > 0:
                    tail = torch.full(
                        (*t.shape[:-1], n),
                        fill,
                        dtype=t.dtype,
                        device=t.device,
                    )
                    return torch.cat([t, tail], dim=-1)
                if n < 0:
                    return t[..., : t.shape[-1] + n].contiguous()
                return t
            return _real_F.pad(t, pad, mode=mode, value=value)

    _el.F = _FPadShim()
    _el._eb5_pad_shim = True
    return ["eplb.expert_location.F.pad -> cat"]


def _install_moe_align() -> list:
    """patch triton 包装（3 返回值版本），fused_experts 走的就是它。"""
    return _patch_everywhere(
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size",
        "moe_align_block_size",
        moe_align_block_size,
    )


def _install_compress_plan() -> list:
    """`CompressorPrefillPlan` 的两个 generate 都是 staticmethod，需按类属性替换。"""
    from sglang.kernels.ops.attention.dsv4.compress import CompressorPrefillPlan

    CompressorPrefillPlan.generate = staticmethod(_prefill_plan_generate)
    CompressorPrefillPlan.generate_legacy = staticmethod(_prefill_plan_generate_legacy)
    from sglang.kernels.ops.attention.dsv4.compress import CompressorDecodePlan

    CompressorDecodePlan.generate = staticmethod(_decode_plan_generate)
    return [
        "CompressorPrefillPlan.generate",
        "CompressorPrefillPlan.generate_legacy",
        "CompressorDecodePlan.generate",
    ]


_TARGETS = (
    ("sglang.kernels.ops.attention.position", "compute_position_triton", compute_position_triton),
    ("sglang.kernels.ops.attention.dsv4_attn_metadata_kernels", "expand_prefill_causally_triton", expand_prefill_causally_triton),
    ("sglang.kernels.ops.attention.dsv4_attn_metadata_kernels", "build_causal_swa_page_indices_triton", build_causal_swa_page_indices_triton),
    ("sglang.kernels.ops.attention.dsv4.metadata_kernel", "init_compression_metadata", init_compression_metadata),
    ("sglang.kernels.ops.attention.dsv4.topk", "plan_topk_v2", plan_topk_v2),
    ("sglang.kernels.ops.attention.dsv4.compress", "compress_norm_rope_store", compress_norm_rope_store),
    ("sglang.kernels.ops.attention.dsa.dequant_k_cache", "dequantize_k_cache_paged", dequantize_k_cache_paged),
    ("sglang.kernels.ops.attention.dsv4.dequant_k_cache", "dequantize_k_cache_paged", dequantize_k_cache_paged_v4),
    ("sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill", "sparse_attn_v4_paged_prefill", sparse_attn_v4_paged_prefill),
    ("sglang.srt.internal.layers.attention.eb5_postmix_fp8_quant", "postmix_quant", postmix_quant),
    ("sglang.kernels.ops.attention.dsv4.moe", "mega_moe_pre_dispatch", mega_moe_pre_dispatch),
    ("sglang.srt.internal.layers.moe.eb5_fused_topk", "fused_split_feature_topk", fused_split_feature_topk),
    ("sglang.kernels.ops.kimi_k3.activation", "situ_and_mul", situ_and_mul),
    ("sglang.kernels.ops.memory.memcpy_triton", "memcpy_triton", memcpy_triton),
    ("sglang.kernels.ops.kvcache.kv_indices", "create_flashinfer_kv_indices_triton", create_flashinfer_kv_indices_triton),
    ("sglang.kernels.ops.attention.dsv4.attn", "get_paged_mqa_logits_metadata", get_paged_mqa_logits_metadata_torch),
    ("sglang.srt.layers.attention.dsv4.sparse_prefill_utils", "build_swa_token_ids", build_swa_token_ids),
    ("sglang.srt.layers.attention.dsv4.sparse_prefill_utils", "combine_topk_swa_indices", combine_topk_swa_indices),
    ("sglang.kernels.ops.attention.dsv4.gemm", "linear_bf16_fp32", linear_bf16_fp32),
    ("sglang.kernels.ops.moe.moe_fused_gate", "moe_fused_gate", moe_fused_gate),
    ("sglang.srt.internal.kernels.deepseek_v4.mhc_head_opt", "fused_hc_head_opt", fused_hc_head_opt),
    ("sglang.srt.internal.models.eb5.multimax", "fused_multimax_selu", fused_multimax_selu),
)


def apply() -> list:
    """把 torch 拆解 patch 到 fork 的调用点上。返回已 patch 的符号列表。"""
    import importlib

    patched = []

    for module_name, attr, impl in _TARGETS:
        try:
            patched.extend(_patch_everywhere(module_name, attr, impl))
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-extra-adapt] import {module_name} failed: {exc}")

    # `alloc_decode_kernel` 不再走 `[grid]` 包装：改为在方法层替换
    # `PagedTokenToKVPoolAllocator.alloc_decode`（见 `_install_alloc_decode`）。
    for mod_name, attr, impl in (
        (
            "sglang.kernels.ops.kvcache.kv_indices",
            "create_flashinfer_kv_indices_triton",
            create_flashinfer_kv_indices_triton,
        ),
    ):
        try:
            mod = importlib.import_module(mod_name)
            setattr(mod, attr, _GridKernel(impl))
            patched.append(f"{mod_name}.{attr}")
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-extra-adapt] {mod_name}.{attr} failed: {exc}")

    for hook in (
        _install_alloc_decode,
        _install_deep_gemm_extra,
        _install_dsv4_metadata_kernels,
        _install_compress_forward,
        _install_moe_align,
        _install_fused_moe_kernel,
        _install_activation,
        _install_compress_plan,
        _install_expert_location_pad,
    ):
        try:
            patched.extend(hook())
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-extra-adapt] {hook.__name__} failed: {exc}")

    return patched
