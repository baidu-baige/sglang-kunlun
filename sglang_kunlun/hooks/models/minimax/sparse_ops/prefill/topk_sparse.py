"""Kunlun MiniMax prefill main sparse operation using forward metadata."""

from typing import Optional

import torch

from ..common.utils import (
    _full_page_table,
    _get_metadata_and_page_size,
    _lods,
    _paged_cache,
)


def _gqa_share_sparse_fwd_kernel(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    page_table: torch.Tensor,
    topk_idx: torch.Tensor,
    out: torch.Tensor,
    q_lod_cpu: torch.Tensor,
    q_lod: torch.Tensor,
    kv_lod_cpu: torch.Tensor,
    kv_lod: torch.Tensor,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    max_pages_per_seq: int,
    block_size_k: int,
    page_size: int,
    topk: int,
    sm_scale: float,
) -> None:
    """Sparse attention over the selected top-k blocks, writing into ``out``.

    上游按 ``block_size_q`` / ``max_seqblock_q`` 划 Triton grid 并传 40+ 个 stride；
    kunlun_ops 只吃 token 级 LOD，tiling 在算子内部完成。

    算子对 ``q`` / ``topk_idx`` 只做裸指针 static_cast，不校验 dtype 与连续性，传错会
    静默读坏内存，故所有 contiguous / dtype 归一都收在本函数内（已满足时为 no-op）。
    """
    import kunlun_ops

    kunlun_ops.msa_sparse_attention(
        q=q.contiguous(),
        k_cache=k_paged,
        v_cache=v_paged,
        block_tables=page_table,
        lod_seqlens_q_cpu=q_lod_cpu,
        lod_seqlens_q_xpu=q_lod,
        lod_seqlens_kv_cpu=kv_lod_cpu,
        lod_seqlens_kv_xpu=kv_lod,
        topk_idx=topk_idx.contiguous().to(torch.int32),
        out=out,
        topk_num=topk,
        topk_block_size=block_size_k,
        batch_num=batch_size,
        head_num=num_q_heads,
        head_num_kv=num_k_heads,
        head_dim=qk_head_dim,
        head_dim_v=v_head_dim,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        max_blocks_per_seq=max_pages_per_seq,
        paged_block_size=page_size,
        softmax_scale=sm_scale,
        prefill_len=max_seqlen_q,  # >=0 = prefill 的 q 长度，-1 才是 decode
        sink=None,  # 算子支持，但调用方已在入口拒绝非 None 的 sink
    )


@torch.no_grad()
def flash_prefill_with_gqa_share_sparse_kunlun(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: Optional[float] = None,
    use_tma: bool = True,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
) -> torch.Tensor:
    """Flash sparse prefill attention using GQA with shared top-k block indices.

    与上游的区别：page table / LOD 取自 Kunlun forward metadata 而非 ``req_to_token``
    + ``slot_ids``；``topk_idx`` 是 token-major（上游 head-major）。
    """
    # kunlun_ops 的 msa 算子按 query token 逐个算，没有 query 分块概念，故 block_size_q
    # 必须为 1；算子有 sink 形参，但 shape 约定未文档化也无测试覆盖，这里不接线，
    # MiniMaxHybridAttnBackend 恒传 sink=None，故显式拒绝而不是静默丢弃。
    if block_size_q != 1:
        raise RuntimeError(
            f"Kunlun MiniMax prefill main requires block_size_q=1, got {block_size_q}"
        )
    if sink is not None:
        raise RuntimeError("Kunlun MiniMax prefill main does not support attn sink")
    # dtype check：Kunlun 没有 fp8 KV cache 路径，故不做上游的 HIP fp8 分支
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert k_cache.dtype == q.dtype, (
        f"sparse prefill expects K cache dtype == Q dtype ({q.dtype}), got {k_cache.dtype}"
    )
    assert v_cache.dtype == k_cache.dtype
    # shape
    total_q, num_q_heads, qk_head_dim = q.shape
    metadata, page_size = _get_metadata_and_page_size()
    batch_size = cu_seqlens.shape[0] - 1
    page_table = _full_page_table(metadata.page_table, batch_size)
    q_lod_cpu, q_lod, kv_lod_cpu, kv_lod = _lods(metadata, batch_size)
    k_paged = _paged_cache(k_cache, page_size)
    v_paged = _paged_cache(v_cache, page_size)
    num_k_heads = k_paged.shape[1]
    num_v_heads, v_head_dim = v_paged.shape[1], v_paged.shape[-1]
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[1] == num_k_heads
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    # 上游此处会在 cu_seqblocks_q / max_seqblock_q 缺省时用 get_cu_seqblocks 补算，用于划
    # Triton 的 query-block grid；kunlun_ops 自己做 tiling，这两个值一律不使用。
    # output tensor
    out = torch.empty(
        (total_q, num_q_heads, v_head_dim), dtype=q.dtype, device=q.device
    )
    # launch kernel
    _gqa_share_sparse_fwd_kernel(
        q=q,
        k_paged=k_paged,
        v_paged=v_paged,
        page_table=page_table,
        topk_idx=topk_idx,
        out=out,
        q_lod_cpu=q_lod_cpu,
        q_lod=q_lod,
        kv_lod_cpu=kv_lod_cpu,
        kv_lod=kv_lod,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        max_seqlen_q=metadata.max_seq_len_q,
        max_seqlen_k=max_seqlen_k,
        max_pages_per_seq=page_table.shape[1],
        block_size_k=block_size_k,
        page_size=page_size,
        topk=topk,
        sm_scale=sm_scale,
    )
    return out
