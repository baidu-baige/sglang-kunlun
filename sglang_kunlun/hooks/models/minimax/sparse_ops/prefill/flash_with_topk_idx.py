"""Kunlun MiniMax prefill index operation using forward metadata."""

import os
from functools import lru_cache
from typing import Optional, Tuple

import torch

from ..common.utils import (
    _full_page_table,
    _get_metadata_and_page_size,
    _lods,
    _paged_cache,
)


@lru_cache(maxsize=1)
def _use_kunlun_prefill_topk() -> bool:
    """是否退回 ``kunlun_ops.msa_block_score_topk_transform``"""
    return os.environ.get("USE_KUNLUN_PREFILL_TOPK", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def _flash_attn_fwd_with_block_score_kernel(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    o: torch.Tensor,
    score: torch.Tensor,
    page_table: torch.Tensor,
    q_lod_cpu: torch.Tensor,
    q_lod: torch.Tensor,
    kv_lod_cpu: torch.Tensor,
    kv_lod: torch.Tensor,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    max_pages_per_seq: int,
    block_size_k: int,
    page_size: int,
    sm_scale: float,
    score_type: str,
) -> None:
    """Per-block QK score, plus index value attention into ``o``.

    上游用一个 Triton kernel 同时出 ``score`` 和 ``o``，并靠 ``DISABLE_INDEX_VALUE``
    常量决定是否写 ``o``（此时可传 ``o=None``）；kunlun_ops 的 ``out`` 是必填输出，
    调用方在 ``disable_index_value`` 时传一块丢弃用的 scratch。

    ``score`` / ``o`` 必须是 fp32：算子内部 ``using TO = float`` 且只做裸指针
    ``static_cast``，传 bf16 不报错但会按 fp32 步长写坏内存。所有 contiguous / dtype
    归一都收在本函数内（已满足时为 no-op）。
    """
    import kunlun_ops

    kunlun_ops.msa_block_score(
        q=q.contiguous(),
        k_cache=k_paged,
        v_cache=v_paged,
        score=score,
        out=o,
        block_tables=page_table,
        lod_seqlens_q_cpu=q_lod_cpu,
        lod_seqlens_q_xpu=q_lod,
        lod_seqlens_kv_cpu=kv_lod_cpu,
        lod_seqlens_kv_xpu=kv_lod,
        batch_num=batch_size,
        prefill_len=max_seqlen_q,  # >=0 = prefill 的 q 长度，-1 才是 decode
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        head_num=num_heads,
        head_num_kv=num_kv_heads,
        head_dim=qk_head_dim,
        head_dim_v=v_head_dim,
        block_size_k=block_size_k,
        softmax_scale=sm_scale,
        max_blocks_per_seq=max_pages_per_seq,
        page_block_size=page_size,
        sink=None,  # 算子支持，但调用方已在入口拒绝非 None 的 sink
        score_type={"max": 0, "lse": 1}[score_type],
    )


def _topk_index_kernel(
    score: torch.Tensor,
    q_lod_cpu: torch.Tensor,
    q_lod: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size_k: int,
    max_seqblock_k: int,
    topk: int,
    local_blocks: int,
) -> torch.Tensor:
    """Select top-k KV block indices per (token, index head) from ``score``.

    上游预分配 ``[num_heads, all_seqblock_q, topk]`` 再由 kernel 写入；kunlun_ops 的
    算子自己分配并返回 token-major ``[total_q, num_heads, topk]`` int32，无效位填 ``-1``。
    """
    if not _use_kunlun_prefill_topk():
        # XSpeedGate 算子纯 stride 寻址但按 head-major 推循环边界，所以传
        # permute(1, 0, 2) 视图 + 如实的 stride 即可零拷贝消费 token-major 数据，
        # 输出仍是 token-major。cu_seqlens / cu_seqblocks_q 都用 q_lod：调用方已硬校验
        # block_size_q == 1，此时 query block 即 token，两者相等且 all_seqblock_q ==
        # total_q；init_blocks 同理恒为 0（入口已拒绝非 0）。
        import xspeedgate_ops  # noqa: F401  触发 torch.ops 注册

        total_q, num_heads, _ = score.shape
        topk_idx = torch.empty(
            (total_q, num_heads, topk), dtype=torch.int32, device=score.device
        )
        score_hm = score.permute(1, 0, 2)
        topk_idx_hm = topk_idx.permute(1, 0, 2)
        torch.ops.xspeedgate_ops.minimax_topk_index_kunlun(
            score=score_hm,
            topk_idx=topk_idx_hm,
            sample_interval=1,
            block_size=block_size_k,
            cu_seqlens=q_lod,
            cu_seqblocks_q=q_lod,
            prefix_lens=prefix_lens.to(dtype=torch.int32),
            topk=topk,
            init_blocks=0,
            local_blocks=local_blocks,
            stride_s_h=score_hm.stride(0),
            stride_s_n=score_hm.stride(1),
            stride_s_k=score_hm.stride(2),
            stride_ti_h=topk_idx_hm.stride(0),
            stride_ti_n=topk_idx_hm.stride(1),
            stride_ti_t=topk_idx_hm.stride(2),
            MASK_INIT=False,
            MASK_LOCAL=False,
        )
        return topk_idx

    import kunlun_ops

    return kunlun_ops.msa_block_score_topk_transform(
        block_scores=score,
        context_qlens_cpu=q_lod_cpu,
        context_qlens_xpu=q_lod,
        prefix_lens_cpu=None,
        prefix_lens_xpu=prefix_lens.to(dtype=torch.int32),
        topk=topk,
        block_size=block_size_k,
        max_num_blocks_per_seq=max_seqblock_k,
        local_blocks=local_blocks,
    )


@torch.no_grad()
def flash_prefill_with_topk_index_kunlun(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    sm_scale: Optional[float] = None,
    use_tma: bool = False,
    score_type: str = "max",
    disable_index_value: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Flash prefill with top-k index: compute sparse attention over selected KV blocks.

    与上游的区别：page table / LOD 取自 Kunlun forward metadata 而非 ``req_to_token``
    + ``slot_ids``；``score`` 与返回的 ``topk_idx`` 都是 token-major（上游 head-major）。
    """
    assert score_type in (
        "max",
        "lse",
    ), f"score_type must be 'max' or 'lse', got {score_type!r}"
    # kunlun_ops 的选块策略固定：msa_block_score 按 query token 逐个算分、没有 query 分块
    # 概念，故 block_size_q 必须为 1；msa_block_score_topk_transform 硬要求
    # local_blocks == 1 且没有 init_blocks 语义；sink 算子支持但这里没接线。
    if block_size_q != 1:
        raise RuntimeError(
            f"Kunlun MiniMax prefill requires block_size_q=1, got {block_size_q}"
        )
    if local_blocks != 1:
        raise RuntimeError(
            f"Kunlun MiniMax prefill requires local_blocks=1, got {local_blocks}"
        )
    if init_blocks != 0:
        raise RuntimeError(
            f"Kunlun MiniMax prefill does not support init_blocks, got {init_blocks}"
        )
    if sink is not None:
        raise RuntimeError("Kunlun MiniMax prefill does not support attn sink")
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert k_cache.dtype == q.dtype
    assert cu_seqlens.dtype == torch.int32
    # shape
    total_q, num_heads, qk_head_dim = q.shape
    metadata, page_size = _get_metadata_and_page_size()
    batch_size = cu_seqlens.shape[0] - 1
    q_lod_cpu, q_lod, kv_lod_cpu, kv_lod = _lods(metadata, batch_size)
    k_paged = _paged_cache(k_cache, page_size)
    if disable_index_value:
        # 上游此时可以完全不看 V；kunlun_ops 的 V 是必填输入，拿 K 顶替（不参与结果）
        v_paged = k_paged
    else:
        assert v_cache is not None and v_cache.dtype == q.dtype
        assert v_cache.shape[1] == k_cache.shape[1]
        v_paged = _paged_cache(v_cache, page_size)
    num_kv_heads = k_paged.shape[1]
    v_head_dim = v_paged.shape[-1]
    assert num_heads % num_kv_heads == 0
    assert (
        init_blocks + local_blocks <= topk
    ), "init_blocks + local_blocks must be less than topk"
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    # 上游此处会在 cu_seqblocks_q / max_seqblock_q / all_seqblock_q 缺省时用
    # get_cu_seqblocks 补算，用于划 Triton 的 query-block grid；kunlun_ops 自己做 tiling，
    # 这三个值一律不使用。
    # CUDA graph 会固化 Python 标量和临时 tensor shape，因此 max_seqlen_k 必须使用
    # sparse backend 传入的 capture-safe 上界，不能重新读取由占位 seq_len 初始化的
    # metadata.max_seq_len_k。page table 的列是**页**，
    # score 的列是**稀疏 block**，两者粒度不同，分别用 page_size / block_size_k 换算。
    # page table 保持全宽（不按 max_seq_len_k 裁剪列）：列切片非连续、会多一次拷贝，
    # 而实际读取范围由 kv_lod / max_seqlen_k 在算子内部界定。
    max_seqblock_k = max(
        (max_seqlen_k + block_size_k - 1) // block_size_k, 1
    )
    page_table = _full_page_table(metadata.page_table, batch_size)
    max_pages_per_seq = page_table.shape[1]
    # 上游 disable_index_value 时 ``o = None``；这里恒分配一块：需要时是返回值，
    # 不需要时当丢弃用的 scratch。score 为 token-major，且由算子按 valid_blocks 收敛，
    # 无需像上游那样预填 -inf。
    o = torch.empty(
        (total_q, num_heads, v_head_dim), dtype=torch.float32, device=q.device
    )
    score = torch.empty(
        (total_q, num_heads, max_seqblock_k), dtype=torch.float32, device=q.device
    )

    # launch kernel
    _flash_attn_fwd_with_block_score_kernel(
        q=q,
        k_paged=k_paged,
        v_paged=v_paged,
        o=o,
        score=score,
        page_table=page_table,
        q_lod_cpu=q_lod_cpu,
        q_lod=q_lod,
        kv_lod_cpu=kv_lod_cpu,
        kv_lod=kv_lod,
        batch_size=batch_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        max_seqlen_q=metadata.max_seq_len_q,
        max_seqlen_k=max_seqlen_k,
        max_pages_per_seq=max_pages_per_seq,
        block_size_k=block_size_k,
        page_size=page_size,
        sm_scale=sm_scale,
        score_type=score_type,
    )

    # topk extraction kernel
    topk_idx = _topk_index_kernel(
        score=score,
        q_lod_cpu=q_lod_cpu,
        q_lod=q_lod,
        prefix_lens=prefix_lens,
        block_size_k=block_size_k,
        max_seqblock_k=max_seqblock_k,
        topk=topk,
        local_blocks=local_blocks,
    )
    if disable_index_value:
        return None, topk_idx
    # 算子输出 fp32，hook 调用方要 q.dtype
    return o.to(dtype=q.dtype), topk_idx
