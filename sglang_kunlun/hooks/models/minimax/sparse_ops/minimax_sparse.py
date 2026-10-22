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
    # 0.5.22 在末尾追加了 5 个形参，``MiniMaxSparseAttnBackend._forward_prefill``
    # （``srt/layers/attention/minimax_sparse_backend.py:1609``）**全部以关键字
    # 无条件传入**，所以这里必须逐个镜像，否则每次 prefill 直接 TypeError。
    cached_topk_idx: Optional[torch.Tensor] = None,
    return_topk_idx: bool = False,
    loc_mapping: Optional[torch.Tensor] = None,
    page_size: int = 1,
    seq_lens_cpu: Optional[torch.Tensor] = None,
):
    """Kunlun REPLACE hook，替换上游 ``minimax_sparse_prefill``。

    流程与上游一致：index 分支（block score + top-k）-> 仅当 index head 多于
    kv head 时做 top-k 归约 -> main 稀疏注意力，返回 ``(idx_o, o)``。
    Kunlun 无 MSA 后端，``use_msa=True`` 直接抛异常。

    0.5.22 新增形参的处理（语义见上游 docstring 与 minimax_sparse.py:118-290）：

    * ``cached_topk_idx`` / ``return_topk_idx``：index top-k 复用。源层算完后用
      ``return_topk_idx=True`` 把归约后的 top-k 作为第三个返回值交给 backend 缓存，
      同组的 skip 层再用 ``cached_topk_idx`` 传回来，跳过 Step 1 + Step 2。
      纯调度逻辑、与算子无关，这里照搬实现。
    * ``page_size``：上游转发给 ``flash_prefill_with_topk_index``。Kunlun 的 leaf
      不收这个参数，而是从 ``_get_metadata_and_page_size()``（即
      ``backend.page_size``）自取——和 backend 传进来的 ``self.page_size`` 同源，
      故此处接收但不转发。
    * ``seq_lens_cpu``：上游只给 env 门控的 ROCm gfx95 Gluon prefill 路径用
      （sync-free scratch page 定容），Kunlun 没有该路径，接收但不使用。
    * ``loc_mapping``：HiSparse 的 pool slot 重映射表，上游一路传到
      ``flash_prefill_with_gqa_share_sparse``。Kunlun 的 main leaf 没有这个入参，
      静默忽略会读错 KV 页，故非 None 时显式拒绝。
    """
    if loc_mapping is not None:
        raise NotImplementedError(
            "Kunlun MiniMax prefill does not implement the HiSparse loc_mapping "
            "remap (upstream forwards it to flash_prefill_with_gqa_share_sparse)"
        )
    # 为对齐上游保留：Kunlun 的 leaf 由 kunlun_ops 内部划分 block 网格，不读取
    # cu_seqblocks_q/max_seqblock_q/all_seqblock_q。block_size_q 恒为 1 时这三个
    # 值分别退化成 cu_seqlens / max_seqlen_q / 总 token 数，本就等价于已有信息。
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k, seqlens_cpu
        )

    # All seqlen is less than topk, use full attention
    if cached_topk_idx is not None:
        # 0.5.22 index-cache 命中：复用同组源层归约后的 top-k，跳过 Step 1
        # （flash index attention + top-k）与 Step 2（归约）。只对
        # disable_index_value 层有效，那些层 idx_o 本就不被下游使用。
        idx_o = None
        topk_idx = cached_topk_idx
    else:
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
    # 归约后的 top-k，供 backend 缓存给同组 skip 层（0.5.22）。
    reduced_topk_idx = topk_idx
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
    if return_topk_idx:
        # 0.5.22：源层把归约后的 top-k 作为第三个返回值交出去（上游同样写法）。
        return idx_o, o, reduced_topk_idx
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
    # 0.5.22 在末尾追加了 5 个形参，``MiniMaxSparseAttnBackend._forward_decode``
    # （``srt/layers/attention/minimax_sparse_backend.py:1811``）**全部以关键字
    # 无条件传入**，所以这里必须逐个镜像，否则每次 decode 直接 TypeError。
    cached_topk_idx: Optional[torch.Tensor] = None,
    topk_out: Optional[torch.Tensor] = None,
    hisparse_swap_in_fn: Optional[Callable] = None,
    indexer_cp=None,
    packed_queries: int = 1,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Kunlun REPLACE hook，替换上游 ``minimax_sparse_decode``。

    流程与上游一致：decode index 分支（block score + top-k）-> 走
    dense-main-attn 回调，或仅当 index head 多于 kv head 时做 top-k 归约后走
    main 稀疏注意力，返回 ``(idx_o, o)``。Kunlun 无 MSA 后端，``use_msa=True``
    直接抛异常。

    0.5.22 新增形参的处理（语义见上游 minimax_sparse.py:330-420）：

    * ``cached_topk_idx`` / ``topk_out``：decode 侧的 index top-k 复用。组内源层
      传 ``topk_out``（一块地址固定的持久 buffer，CUDA-graph 安全），把归约后的
      top-k ``copy_`` 进去；skip 层把同一 buffer 当 ``cached_topk_idx`` 传回来，
      跳过 Step 1 + Step 2。纯调度逻辑、与算子无关，这里照搬实现（含上游那条
      shape 校验与 dense-main 互斥断言）。
    * ``hisparse_swap_in_fn``：HiSparse 的换入回调，上游用它的返回值
      ``hisparse_slots`` 替换 main leaf 的 KV 寻址。Kunlun 的 main leaf 没有这个
      入参，忽略会读错 KV 页，故非 None 时显式拒绝。
    * ``indexer_cp``：context-parallel 版 indexer（``MiniMaxIndexerCP``），在它
      ``supports(...)`` 且一串硬条件成立时整体顶替 Step 1。Kunlun 的 indexer 没有
      CP 实现，故只在上游真会选它的那套门槛下拒绝；``supports()`` 为假时上游也走
      普通 indexer，这里同样继续往下走。
    * ``packed_queries``：EAGLE verify 时每个 request 的 draft 行是连续排布的行数。
      上游**只**把它交给 ``indexer_cp``（minimax_sparse.py:367），``indexer_cp``
      为 None 时它在上游没有任何作用，普通 indexer 仍按逐行语义选块。因此这里接收
      但不使用：唯一真正消费它的路径已被上面的 ``indexer_cp`` 门槛拒掉，额外再拒
      反而会打断 0.5.19 下本来能跑的 verify。
    """
    if hisparse_swap_in_fn is not None:
        raise NotImplementedError(
            "Kunlun MiniMax decode does not implement the HiSparse swap-in path "
            "(upstream feeds hisparse_slots into flash_decode_with_gqa_share_sparse)"
        )
    if (
        indexer_cp is not None
        and disable_index_value
        and idx_sink is None
        and score_type == "max"
        and block_size_k == 128
        and topk == 16
        and dense_main_attn_fn is None
        and indexer_cp.supports(idx_q, idx_k_cache, max_seqlen, req_to_token)
    ):
        # 门槛与上游一致（minimax_sparse.py:344-354）：命中时上游用 CP indexer
        # 整体替换 Step 1，昆仑没有对应实现，不能静默退回普通 indexer（CP 下
        # 每张卡只持有部分 KV，选块结果会错）。
        raise NotImplementedError(
            "Kunlun MiniMax decode does not implement the context-parallel "
            "indexer (indexer_cp)"
        )
    assert dense_main_attn_fn is None or (
        cached_topk_idx is None and topk_out is None
    ), "index top-k sharing is not available on the dense main-attention path"
    # Step 1: Flash decode with topk index (using index head). When the dense main
    # attention is used, the indexer emits the page table directly (fused
    # transform) instead of block ids, plus the per-query effective KV length.
    if cached_topk_idx is not None:
        # 0.5.22 index-cache 命中：跳过 Step 1 与 Step 2（topk_idx 已是归约结果）。
        idx_o = None
        real_seq_lens = None
        topk_idx = cached_topk_idx
        _skip_reduce = True
    else:
        _skip_reduce = False
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
        if idx_group_size > 1 and not _skip_reduce:
            topk_idx = topk_index_reduce(
                topk_idx.view(-1, num_kv_heads, idx_group_size, topk), dim=2
            )
        if topk_out is not None:
            # 0.5.22：把本组的选块结果写进 skip 层要读的持久 buffer
            # （地址固定 -> CUDA-graph 安全），写法与上游一致。
            if topk_out.shape != topk_idx.shape:
                raise ValueError(
                    f"topk_out shape {tuple(topk_out.shape)} does not match "
                    f"reduced top-k shape {tuple(topk_idx.shape)}"
                )
            topk_out.copy_(topk_idx)
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
