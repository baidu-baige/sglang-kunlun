"""Kunlun MiniMax decode index operation using forward metadata."""

import os
from functools import lru_cache
from typing import Optional, Tuple

import torch

from ..common.msa_dump import record
from ..common.utils import (
    _full_page_table,
    _get_metadata,
    _paged_cache,
)


@lru_cache(maxsize=1)
def _use_kunlun_decode_topk() -> bool:
    """是否退回 ``kunlun_ops.msa_block_score_topk_transform``"""
    return os.environ.get("USE_KUNLUN_DECODE_TOPK", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def _decode_score_kernel(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    score: torch.Tensor,
    o: torch.Tensor,
    page_table: torch.Tensor,
    q_lod: torch.Tensor,
    kv_lod: torch.Tensor,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_idx_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    block_size: int,
    page_size: int,
    max_pages_per_seq: int,
    sm_scale: float,
    score_type: str,
) -> None:
    """Per-block QK score, plus index value attention into ``o``.

    上游按 ``disable_index_value`` 在 ``_decode_score_kernel`` /
    ``_decode_score_attn_kernel`` 两个 Triton kernel 间二选一，kunlun_ops 用一个算子
    融合两者。``score`` / ``o`` 必须是 fp32：算子内部 ``using TO = float`` 且只做裸指针
    ``static_cast``，传 bf16 不报错但会按 fp32 步长写坏内存。
    """
    import kunlun_ops

    kunlun_ops.msa_block_score(
        q=q.contiguous(),
        k_cache=k_paged,
        v_cache=v_paged,
        score=score,
        out=o,
        block_tables=page_table,
        # decode 走 CUDA graph，只能依赖 device 上的 live 张量，故不给 host 侧 LOD。
        # metadata 的 cu_seqlens_* 是 int32 device 张量的 1-D 切片，已连续，无需转换。
        lod_seqlens_q_cpu=None,
        lod_seqlens_q_xpu=q_lod,
        lod_seqlens_kv_cpu=None,
        lod_seqlens_kv_xpu=kv_lod,
        batch_num=batch_size,
        prefill_len=-1,  # -1 = decode，>=0 = prefill 的 q 长度
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        head_num=num_idx_heads,
        head_num_kv=num_kv_heads,
        head_dim=head_dim,
        head_dim_v=v_head_dim,
        block_size_k=block_size,
        softmax_scale=sm_scale,
        max_blocks_per_seq=max_pages_per_seq,
        page_block_size=page_size,
        sink=None,  # 算子支持，但调用方已在入口拒绝非 None 的 sink
        score_type={"max": 0, "lse": 1}[score_type],
    )


def minimax_decode_topk_kunlun(
    score: torch.Tensor,  # [batch, num_idx_heads, max_seqblock] fp32 (token-major)
    seq_lens: torch.Tensor,  # [batch] int32/int64
    q_lod: torch.Tensor,  # [batch + 1] int32 (forward metadata cu_seqlens_q)
    block_size: int,
    topk: int,
    local_blocks: int,
) -> torch.Tensor:
    """Select top-k KV block indices per (batch, index head) from decode scores.

    对应上游 ``sglang.jit_kernel.minimax_decode_topk.minimax_decode_topk``。输出
    token-major ``[batch, num_idx_heads, topk]`` int32，无效位以 ``-1`` 填充。
    """
    if (
        not _use_kunlun_decode_topk()
        and topk == 16
        and block_size == 128
        and local_blocks == 1
        and score.dtype is torch.float32
    ):
        import xspeedgate_ops  # noqa: F401  触发 torch.ops 注册

        batch_size, num_idx_heads, _ = score.shape
        topk_idx = torch.empty(
            (batch_size, num_idx_heads, topk), dtype=torch.int32, device=score.device
        )
        # decode 每请求一个 token，所以 [batch, num_idx_heads, ...] 即算子要求的
        # token-major 布局，行到序列的映射 b = row / num_heads 天然成立。
        torch.ops.xspeedgate_ops.minimax_decode_topk_block_kunlun(
            score=score,
            seq_lens=seq_lens.to(dtype=torch.int32),
            topk_idx=topk_idx,
            topk=topk,
            block_size=block_size,
        )
        return topk_idx

    import kunlun_ops

    # 算子输出即 [batch, num_idx_heads, topk] 连续 int32，调用方（topk_sparse）在进
    # msa_sparse_attention 前统一做 contiguous/int32 的边界保险，这里不重复处理。
    # 入参先收进 dict：dump 用的就是算子实际吃到的那一份，避免两处参数列表漂移。
    call_kwargs = dict(
        block_scores=score,
        # decode 走 CUDA graph，只给 device 侧
        context_qlens_cpu=None,
        context_qlens_xpu=q_lod,
        prefix_lens_cpu=None,
        # 算子要求 prefix_lens 为 int32，而 seq_lens 是 int64，这里必须转
        prefix_lens_xpu=(seq_lens - 1).to(dtype=torch.int32),
        topk=topk,
        block_size=block_size,
        max_num_blocks_per_seq=score.shape[-1],
        local_blocks=local_blocks,
    )
    record("msa_block_score_topk_transform", call_kwargs, bs=score.shape[0])
    return kunlun_ops.msa_block_score_topk_transform(**call_kwargs)


def minimax_decode_topk_page_table_kunlun(
    score: torch.Tensor,  # [batch, num_idx_heads, max_seqblock] fp32 (token-major)
    seq_lens: torch.Tensor,  # [batch] int32/int64
    q_lod: torch.Tensor,  # [batch + 1] int32 (forward metadata cu_seqlens_q)
    page_table: torch.Tensor,  # [batch, max_seqblock] int32 (forward metadata)
    block_size: int,
    topk: int,
    page_size: int,
    local_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused top-k + page-table transform, mirroring upstream
    ``sglang.jit_kernel.minimax_decode_topk.minimax_decode_topk_page_table``.

    产出 dense 后端消费的分页 page table 和每个 (batch, head) 伪请求的有效 KV 长度
    (``cache_seqlens``)。page index 为 head-minor 编码 ``base_page * num_heads + head``，
    无效块填 ``-1``。Kunlun 暂无等价融合算子，先用 torch 实现。
    """
    if block_size % page_size != 0:
        raise RuntimeError(
            f"Kunlun MiniMax dense page table requires block_size % page_size == 0, "
            f"got block_size={block_size}, page_size={page_size}"
        )
    batch_size, num_heads, blocks = score.shape
    pages_per_block = block_size // page_size
    max_pages_per_seq = page_table.shape[1]
    topk_idx = minimax_decode_topk_kunlun(
        score=score,
        seq_lens=seq_lens,
        q_lod=q_lod,
        block_size=block_size,
        topk=topk,
        local_blocks=local_blocks,
    )
    block_idx = topk_idx.to(dtype=torch.int64)

    valid_idx = block_idx >= 0
    safe_idx = block_idx.clamp(min=0, max=max(blocks - 1, 0))
    page_offsets = torch.arange(
        pages_per_block, dtype=torch.int64, device=score.device
    )
    # 稀疏 block b 覆盖 page table 的第 ``b * pages_per_block + [0, pages_per_block)``
    # 列；gather 出来的值就是真实 page id，不再需要额外换算。
    page_cols = (safe_idx[..., None] * pages_per_block + page_offsets).clamp(
        max=max_pages_per_seq - 1
    )
    dense_page_table = page_table.gather(
        1, page_cols.reshape(batch_size, -1)
    ).to(torch.int64).view(batch_size, num_heads, topk, pages_per_block)
    dense_page_table = dense_page_table * num_heads + torch.arange(
        num_heads, dtype=torch.int64, device=score.device
    )[None, :, None, None]
    valid_pages = valid_idx[..., None].expand_as(dense_page_table)
    dense_page_table = torch.where(
        valid_pages, dense_page_table, torch.full_like(dense_page_table, -1)
    ).reshape(batch_size * num_heads, topk * pages_per_block)
    selected_tokens = torch.minimum(
        seq_lens.to(torch.int64)[:, None, None] - safe_idx * block_size,
        torch.full_like(safe_idx, block_size),
    ).clamp(min=0)
    real_seq_lens = torch.where(
        valid_idx, selected_tokens, torch.zeros_like(selected_tokens)
    ).sum(dim=-1).reshape(batch_size * num_heads).to(torch.int32)
    return dense_page_table.to(torch.int32), real_seq_lens


@torch.no_grad()
def flash_decode_with_topk_idx_kunlun(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: Optional[torch.Tensor],  # paged; ignored when disable_index_value=True
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen: int,
    slot_ids: torch.Tensor,  # [batch_size, ]
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    use_tma: bool = True,
    score_type: str = "max",
    disable_index_value: bool = False,
    use_dense_main_attn: bool = False,  # NOTE: need transform idx in this case
    page_size: int = 1,
):
    """Flash decode selecting top-k KV blocks via index attention scores.

    与上游的区别：page table / LOD 取自 Kunlun forward metadata 而非
    ``req_to_token`` + ``slot_ids``；``topk_idx`` 为 token-major
    ``[batch_size, num_idx_heads, topk]``（上游是 head-major）。
    返回 ``(idx_o, topk_idx, real_seq_lens)``。
    """
    assert score_type in (
        "max",
        "lse",
    ), f"score_type must be 'max' or 'lse', got {score_type!r}"
    if not disable_index_value:
        assert v_cache is not None
    # kunlun_ops 的选块策略固定：msa_block_score_topk_transform 硬要求 local_blocks == 1
    # 且没有 init_blocks 语义；sink 算子支持但这里没接线。block_size / page_size 是通用
    # 的，由算子自己断言。
    if local_blocks != 1:
        raise RuntimeError(
            f"Kunlun MiniMax decode requires local_blocks=1, got {local_blocks}"
        )
    if init_blocks != 0:
        raise RuntimeError(
            f"Kunlun MiniMax decode does not support init_blocks, got {init_blocks}"
        )
    if sink is not None:
        raise RuntimeError("Kunlun MiniMax decode does not support attn sink")
    # shape
    batch_size, num_idx_heads, head_dim = q.shape
    metadata = _get_metadata()
    page_table = _full_page_table(metadata.page_table, batch_size)
    q_lod = metadata.cu_seqlens_q[: batch_size + 1]
    kv_lod = metadata.cu_seqlens_k[: batch_size + 1]
    k_paged = _paged_cache(k_cache, page_size)
    v_paged = k_paged if v_cache is None else _paged_cache(v_cache, page_size)
    num_kv_heads = k_paged.shape[1]
    v_head_dim = v_paged.shape[-1]
    # 上游的 max_kv_len 取自 ``req_to_token.shape[1]``；Kunlun 取 forward metadata 里
    # 预分配 page table 的宽度（capture-invariant）。page table 的列是**页**，而 score
    # 的列是**稀疏 block**，两者粒度不同（算子分别用 page_block_size / block_size_k）。
    max_pages_per_seq = page_table.shape[1]
    max_kv_len = max_pages_per_seq * page_size
    max_seqblock_k = max((max_kv_len + block_size - 1) // block_size, 1)
    assert slot_ids.shape[0] == batch_size and seq_lens.shape[0] == batch_size
    # gqa
    assert num_idx_heads % num_kv_heads == 0
    # sm scale
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    score = torch.empty(
        (batch_size, num_idx_heads, max_seqblock_k),
        dtype=torch.float32,
        device=q.device,
    )
    # 算子的 ``out`` 是必填输出，所以不管要不要 index value 都得给一块 buffer：需要时它
    # 就是返回值，不需要时当丢弃用的 scratch。上游是两个 kernel 二选一，且分片输出 o+lse
    # 后再由 ``_merge_attn_out_kernel`` 归约；Kunlun 的分片归约在算子内部完成。
    o = torch.empty(
        (batch_size, num_idx_heads, v_head_dim), dtype=torch.float32, device=q.device
    )
    _decode_score_kernel(
        q=q,
        k_paged=k_paged,
        v_paged=v_paged,
        score=score,
        o=o,
        page_table=page_table,
        q_lod=q_lod,
        kv_lod=kv_lod,
        batch_size=batch_size,
        max_seqlen_q=metadata.max_seq_len_q,
        max_seqlen_k=max_kv_len,
        num_idx_heads=num_idx_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        block_size=block_size,
        page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
        sm_scale=sm_scale,
        score_type=score_type,
    )
    # ``score`` 是预分配 page table 的全宽，超出 live 序列长度的列是脏数据，但不用在
    # Python 层清理：msa_block_score_topk_transform 会按 live 的 prefix_lens 推出
    # valid_blocks / local_block，把越界列塌成 -FLT_MAX、local block 置 1e29。

    # Fused top-k + page-table transform: emit the dense backend's page table
    # directly (page-size-aware) instead of block ids, skipping a separate gather.
    # The page table + per-query effective KV length are allocated and returned.
    real_seq_lens = None
    if use_dense_main_attn:
        topk_idx, real_seq_lens = minimax_decode_topk_page_table_kunlun(
            score=score,
            seq_lens=seq_lens,
            q_lod=q_lod,
            page_table=page_table,
            block_size=block_size,
            topk=topk,
            page_size=page_size,
            local_blocks=local_blocks,
        )
        if disable_index_value:
            return None, topk_idx, real_seq_lens
        # fall through to the idx_o (index value) merge, then return
        _skip_block_topk = True
    else:
        _skip_block_topk = False

    # get topk index
    if _skip_block_topk:
        pass
    else:
        topk_idx = minimax_decode_topk_kunlun(
            score, seq_lens, q_lod, block_size, topk, local_blocks
        )
    if disable_index_value:
        return None, topk_idx, real_seq_lens
    # 上游此处是 ``_merge_attn_out_kernel`` 归约分片 + ``o = o[0].contiguous()``；
    # kunlun_ops 已给出最终结果，只剩 fp32 -> q.dtype 转换。
    o = o.to(dtype=q.dtype)
    return o, topk_idx, real_seq_lens
