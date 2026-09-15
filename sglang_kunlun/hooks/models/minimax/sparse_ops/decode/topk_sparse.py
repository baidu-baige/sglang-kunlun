"""Kunlun MiniMax decode main sparse operation using forward metadata."""

from typing import Optional

import torch

from ..common.msa_dump import record
from ..common.utils import (
    _full_page_table,
    _get_metadata_and_page_size,
    _paged_cache,
)


def _gqa_share_sparse_decode_kernel(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    page_table: torch.Tensor,
    topk_idx: torch.Tensor,
    out: torch.Tensor,
    q_lod: torch.Tensor,
    kv_lod: torch.Tensor,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    max_pages_per_seq: int,
    block_size: int,
    page_size: int,
    max_topk: int,
    sm_scale: float,
) -> None:
    """Sparse attention over the selected top-k blocks, writing into ``out``.

    上游用 ``_gqa_share_sparse_decode_kernel`` 出 ``NUM_TOPK_CHUNKS`` 份 o + lse 分片，
    再由 ``_merge_topk_attn_out_kernel`` 归约；kunlun_ops 在算子内部完成分片与归约，
    所以这里只有一次调用、一块最终形状的输出。

    算子对 ``q`` / ``topk_idx`` 只做裸指针 static_cast，不校验 dtype 与连续性，传错会
    静默读坏内存，故所有 contiguous / dtype 归一都收在本函数内（已满足时为 no-op）。
    """
    import kunlun_ops

    # 入参先收进 dict：dump 用的就是算子实际吃到的那一份，避免两处参数列表漂移。
    call_kwargs = dict(
        q=q.contiguous(),
        k_cache=k_paged,
        v_cache=v_paged,
        block_tables=page_table,
        # decode 走 CUDA graph，只能依赖 device 上的 live 张量，故不给 host 侧 LOD。
        # metadata 的 cu_seqlens_* 是 int32 device 张量的 1-D 切片，已连续，无需转换。
        lod_seqlens_q_cpu=None,
        lod_seqlens_q_xpu=q_lod,
        lod_seqlens_kv_cpu=None,
        lod_seqlens_kv_xpu=kv_lod,
        topk_idx=topk_idx.contiguous().to(torch.int32),
        out=out,
        topk_num=max_topk,
        topk_block_size=block_size,
        batch_num=batch_size,
        head_num=num_q_heads,
        head_num_kv=num_kv_heads,
        head_dim=head_dim,
        head_dim_v=v_head_dim,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        max_blocks_per_seq=max_pages_per_seq,
        paged_block_size=page_size,
        softmax_scale=sm_scale,
        prefill_len=-1,  # -1 = decode，>=0 = prefill 的 q 长度
        sink=None,  # 算子支持，但调用方已在入口拒绝非 None 的 sink
    )
    record("msa_sparse_attention", call_kwargs, bs=batch_size)
    kunlun_ops.msa_sparse_attention(**call_kwargs)


@torch.no_grad()
def flash_decode_with_gqa_share_sparse_kunlun(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [batch_size, ]
    slot_ids: torch.Tensor,  # [batch_size, ]
    block_size: int,
    topk_idx: torch.Tensor,  # [batch_size, num_kv_heads, topk] (token-major)
    sm_scale: Optional[float] = None,
    use_tma: bool = True,
) -> torch.Tensor:
    """Flash sparse decode attention using GQA with shared top-k block indices.

    与上游的区别：page table / LOD 取自 Kunlun forward metadata 而非 ``req_to_token``
    + ``slot_ids``；``topk_idx`` 是 token-major（上游 head-major）。
    """
    # kunlun_ops 的 msa 算子有 sink 形参，但 shape 约定未文档化也无测试覆盖，这里不接线；
    # MiniMaxHybridAttnBackend 恒传 sink=None，故显式拒绝而不是静默丢弃。
    if sink is not None:
        raise RuntimeError("Kunlun MiniMax decode main does not support attn sink")
    # dtype check：Kunlun 没有 fp8 KV cache 路径，故不做上游的 HIP fp8 分支
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert k_cache.dtype == q.dtype, (
        f"sparse decode expects K cache dtype == Q dtype ({q.dtype}), got {k_cache.dtype}"
    )
    assert v_cache.dtype == k_cache.dtype
    # shape
    batch_size, num_q_heads, head_dim = q.shape
    metadata, page_size = _get_metadata_and_page_size()
    page_table = _full_page_table(metadata.page_table, batch_size)
    q_lod = metadata.cu_seqlens_q[: batch_size + 1]
    kv_lod = metadata.cu_seqlens_k[: batch_size + 1]
    k_paged = _paged_cache(k_cache, page_size)
    v_paged = _paged_cache(v_cache, page_size)
    num_kv_heads = k_paged.shape[1]
    v_head_dim = v_paged.shape[-1]
    assert slot_ids.shape[0] == batch_size and seq_lens.shape[0] == batch_size
    assert topk_idx.shape[1] == num_kv_heads
    # 上游的 max_kv_len 取自 ``req_to_token.shape[1]``；Kunlun 取 forward metadata 里
    # 预分配 page table 的宽度（capture-invariant）。注意 page table 的列是**页**，
    # 与稀疏 block 是两个粒度：block_size 用于 topk，page_size 用于分页寻址。
    max_pages_per_seq = page_table.shape[1]
    max_kv_len = max_pages_per_seq * page_size
    # gqa
    assert num_q_heads % num_kv_heads == 0
    max_topk = topk_idx.shape[-1]
    # sm scale
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    # 上游此处分配 NUM_TOPK_CHUNKS 份 o + lse 分片，Kunlun 只要最终形状的一块输出
    out = torch.empty(
        (batch_size, num_q_heads, v_head_dim), dtype=q.dtype, device=q.device
    )
    # launch attention kernel
    _gqa_share_sparse_decode_kernel(
        q=q,
        k_paged=k_paged,
        v_paged=v_paged,
        page_table=page_table,
        topk_idx=topk_idx,
        out=out,
        q_lod=q_lod,
        kv_lod=kv_lod,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        max_seqlen_q=metadata.max_seq_len_q,
        max_seqlen_k=max_kv_len,
        max_pages_per_seq=max_pages_per_seq,
        block_size=block_size,
        page_size=page_size,
        max_topk=max_topk,
        sm_scale=sm_scale,
    )
    # 上游此处是 ``_merge_topk_attn_out_kernel`` 归约分片 + ``o_partial[0].contiguous()``
    return out
