"""Kunlun MiniMax 稀疏注意力 orchestrator。

仅 hook 上游两个顶层 dispatch 函数。prefill/decode 的 leaf 是普通 Kunlun 调用，
全程维持 token-major 契约（上游是 head-major），因此不能作为 hook 目标。
"""

from typing import Callable, List, Optional, Tuple

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.index import (
    topk_index_reduce,
)
from sglang.kernels.ops.attention.minimax_sparse.common.utils import (
    get_cu_seqblocks,
)
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from .decode.flash_with_topk_idx import flash_decode_with_topk_idx_kunlun
from .decode.topk_sparse import flash_decode_with_gqa_share_sparse_kunlun
from .prefill.flash_with_topk_idx import flash_prefill_with_topk_index_kunlun
from .prefill.topk_sparse import flash_prefill_with_gqa_share_sparse_kunlun


@plugin_hook(
    "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse.minimax_sparse_prefill",
    type=HookType.REPLACE,
)
def minimax_sparse_prefill_kunlun(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    idx_sink: Optional[torch.Tensor],
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
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    use_msa: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    seqlens_cpu: Optional[List[int]] = None,
    # fp8 attention-GEMM scales (0.5.19+). Kunlun runs the fp16 KV / bf16 q
    # path, so they are accepted for signature parity and unused.
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_q_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
):
    """Kunlun REPLACE hook，替换上游 ``minimax_sparse_prefill``。

    流程与上游一致：index 分支（block score + top-k）-> 仅当 index head 多于
    kv head 时做 top-k 归约 -> main 稀疏注意力，返回 ``(idx_o, o)``。
    Kunlun 无 MSA 后端，``use_msa=True`` 直接抛异常。
    """
    # 为对齐上游保留：Kunlun 的 leaf 由 kunlun_ops 内部划分 block 网格，不读取
    # cu_seqblocks_q/max_seqblock_q/all_seqblock_q。block_size_q 恒为 1 时这三个
    # 值分别退化成 cu_seqlens / max_seqlen_q / 总 token 数，本就等价于已有信息。
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k, seqlens_cpu
        )

    # All seqlen is less than topk, use full attention
    # Step 1: Flash attention with topk index (using index head)
    idx_o, topk_idx = flash_prefill_with_topk_index_kunlun(
        q=idx_q,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        sink=idx_sink,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
        all_seqblock_q=all_seqblock_q,
    )
    # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
    # Kunlun 全程保持 token-major，直接就地归约 group 维（dim=2），无需像上游那样
    # 先 permute 成 head-major 再转回来；view 拆 head 维是零拷贝。
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(-1, num_kv_heads, idx_group_size, topk), dim=2
        )
    # Step 3: Sparse attention using topk index (main head).
    # Kunlun 无 MSA 后端，上游的 msa_sparse_prefill_main 分支在此直接拒绝。
    # ``sink is None`` 沿用上游写法，但实际恒成立：MiniMaxHybridAttnBackend 调用
    # 时始终传 sink=None，所以这里等价于 ``if use_msa``。
    if use_msa and sink is None:
        raise NotImplementedError("Kunlun MiniMax requires use_msa=False")
    else:
        o = flash_prefill_with_gqa_share_sparse_kunlun(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            sink=sink,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            topk_idx=topk_idx,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sm_scale=sm_scale,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
        )
    return idx_o, o


@plugin_hook(
    "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse.minimax_sparse_decode",
    type=HookType.REPLACE,
)
def minimax_sparse_decode_kunlun(
    q: torch.Tensor,
    sink: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_q: torch.Tensor,
    idx_sink: Optional[torch.Tensor],
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    dense_main_attn_fn: Optional[Callable] = None,
    page_size: int = 1,
    use_msa: bool = False,
    msa_kv_indices: Optional[torch.Tensor] = None,
    msa_plan=None,
    # fp8 attention-GEMM scales (0.5.19+); unused on the Kunlun fp16-KV path.
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_q_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Kunlun REPLACE hook，替换上游 ``minimax_sparse_decode``。

    流程与上游一致：decode index 分支（block score + top-k）-> 走
    dense-main-attn 回调，或仅当 index head 多于 kv head 时做 top-k 归约后走
    main 稀疏注意力，返回 ``(idx_o, o)``。Kunlun 无 MSA 后端，``use_msa=True``
    直接抛异常。
    """
    # Step 1: Flash decode with topk index (using index head). When the dense main
    # attention is used, the indexer emits the page table directly (fused
    # transform) instead of block ids, plus the per-query effective KV length.
    idx_o, topk_idx, real_seq_lens = flash_decode_with_topk_idx_kunlun(
        q=idx_q,
        sink=idx_sink,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        slot_ids=slot_ids,
        block_size=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        use_dense_main_attn=dense_main_attn_fn is not None,
        page_size=page_size,
    )
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if dense_main_attn_fn is not None:
        # topk_idx is the page table; real_seq_lens is the per-query cache_seqlens
        assert idx_group_size == 1
        o = dense_main_attn_fn(q, topk_idx, real_seq_lens)
    else:
        # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
        # Kunlun 全程保持 token-major，直接就地归约 group 维（dim=2），无需像上游
        # 那样先 permute 成 head-major 再转回来；view 拆 head 维是零拷贝。
        if idx_group_size > 1:
            topk_idx = topk_index_reduce(
                topk_idx.view(-1, num_kv_heads, idx_group_size, topk), dim=2
            )
        # Step 3: Sparse attention using topk index (main head).
        # Kunlun 无 MSA 后端，上游的 msa_sparse_decode_main 分支在此直接拒绝。
        # ``sink is None`` 沿用上游写法，但实际恒成立：MiniMaxHybridAttnBackend
        # 调用时始终传 sink=None，所以这里等价于 ``if use_msa``。
        if use_msa and sink is None:
            raise NotImplementedError("Kunlun MiniMax requires use_msa=False")
        else:
            o = flash_decode_with_gqa_share_sparse_kunlun(
                q=q,
                sink=sink,
                k_cache=k_cache,
                v_cache=v_cache,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                slot_ids=slot_ids,
                block_size=block_size_k,
                topk_idx=topk_idx,
                sm_scale=sm_scale,
            )
    return idx_o, o
