"""Kunlun implementation of the DeepSeek sparse-attention indexer.

Single patch point: ``Indexer.forward_cuda`` / ``Indexer.forward_xpu`` are replaced
with :func:`forward_cuda` below through ``plugin_hook``. Everything else here is a
plain module-level function called directly from ``forward_cuda`` - we deliberately
do NOT hook the individual ``Indexer._get_*`` methods, so the Kunlun code path is
fully self contained and an upstream upgrade cannot silently mix our implementation
with upstream control flow.

Dependencies on the upstream ``Indexer`` instance are listed in
:data:`REQUIRED_INDEXER_ATTRS` and checked once per forward by
:func:`_resolve_deps`; that list is the whole contract to re-verify when
upgrading.

Aligned with upstream sglang 0.5.14 ``dsa/dsa_indexer.py``
(ported from the 0.5.8 ``nsa/nsa_indexer.py`` Kunlun implementation). Known, intentional
differences vs upstream:
  1. max_k logits layout + explicit ``mask_for_I8_mqa_logits`` (upstream uses a
     total_k layout);
  2. int8 quantization instead of fp8_e4m3 (hence the ``/ 127`` corrections);
  3. free-memory driven chunking of the ragged MQA logits;
  4. dual-stream is never enabled, so those upstream branches are dropped.
When upgrading, diff the *upstream* forward_cuda between versions and port the
new semantics here, rather than diffing this file.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import kunlun_ops
from einops import rearrange

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    _broadcast_indexer_topk_from_rank0,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk


if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool


# Upstream's ``_is_cuda`` branch projects the head gates with
# ``torch.mm(..., out_dtype=torch.float32)`` (``aten::mm.dtype``), which has no
# Kunlun kernel. The replacement lives in ``kernel_ops``; only the registration
# belongs here.
@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._weights_proj_bf16_in_fp32_out", type=HookType.REPLACE
)
def dsa_weights_proj_bf16_in_fp32_out_kunlun(self, x: torch.Tensor) -> torch.Tensor:
    """Project the DSA indexer head gates to fp32 without ``aten::mm.dtype``.

    Kunlun reports itself as CUDA, so upstream takes the ``_is_cuda`` branch and
    calls ``torch.mm(x, w.t(), out_dtype=torch.float32)``. That overload has no
    Kunlun kernel, so run the projection in the weight dtype and upcast, matching
    what the Kunlun indexer's own ``_get_logits_head_gate`` does.
    """

    weights, _ = self.weights_proj(x.to(self.weights_proj.weight.dtype))
    return weights.float()


# Attributes we read off the upstream Indexer instance. This is the full coupling
# surface between the Kunlun indexer and upstream.
REQUIRED_INDEXER_ATTRS = (
    "_get_k_bf16",
    "_should_chunk_mqa_logits",
    "alt_stream",
    "block_size",
    "cp_size",
    "head_dim",
    "index_topk",
    "k_norm",
    "n_heads",
    "dsa_enable_prefill_cp",
    "rope_head_dim",
    "rotary_emb",
    "softmax_scale",
    "weights_proj",
    "wk",
    "wq_b",
)


def _resolve_deps(indexer) -> None:
    """Fail loudly (once per call) if upstream renamed something we depend on."""
    missing = [a for a in REQUIRED_INDEXER_ATTRS if not hasattr(indexer, a)]
    if missing:
        raise AttributeError(
            f"kunlun dsa indexer requires Indexer attributes {missing}; upstream "
            "layout changed - update REQUIRED_INDEXER_ATTRS and the call sites"
        )


def xspeedgate_hadamard_transform(
    x: torch.Tensor, scale: Optional[float] = None
) -> torch.Tensor:
    """xspeedgate hadamard transform (replaces sgl_kernel fast_hadamard_transform).

    NOTE: this is a different kernel from :func:`hadamard_transform` above (which
    materializes the matrix and calls ``kunlun_ops.matmul``). The DSA indexer's
    q/k rotation has always used this one, so it is kept separate rather than
    unified - switching it would change indexer numerics.
    """

    if scale is None:
        return torch.ops.xspeedgate_ops.hadamard_transform(x.contiguous())
    return torch.ops.xspeedgate_ops.hadamard_transform(x.contiguous(), scale)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Rotate activations with a scaled Hadamard transform."""

    assert x.dtype == torch.bfloat16
    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return xspeedgate_hadamard_transform(x, hidden_size**-0.5)


def chunk_cu_seqlens(
    cu_seqlens: torch.Tensor,  # shape: (1, batch_size + 1)
    chunk_size: int,
):
    """Split the original sequence ranges into chunks of ``chunk_size`` rows.

    Returns a list of ``(new_cu_seqlens, interval_bounds, chunk_range)`` tuples.
    Used by the DSA ragged MQA-logits chunk path to bound peak logits memory.
    """

    cu = cu_seqlens.squeeze(0)
    total_len = cu[-1].item()

    seg_starts = cu[:-1]
    seg_ends = cu[1:]

    results = []
    for chunk_start in range(0, total_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_len)

        inter_starts = torch.clamp(seg_starts, min=chunk_start)
        inter_ends = torch.clamp(seg_ends, max=chunk_end)

        valid_mask = inter_starts < inter_ends

        valid_inter_starts = inter_starts[valid_mask]
        valid_inter_ends = inter_ends[valid_mask]
        inter_lens = valid_inter_ends - valid_inter_starts

        new_cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(inter_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        interval_bounds = torch.cat([valid_inter_starts[:1], valid_inter_ends])
        chunk_range = torch.tensor(
            [valid_inter_starts[0].item(), valid_inter_ends[-1].item()],
            dtype=torch.int32,
        )

        results.append((new_cu_seqlens, interval_bounds, chunk_range))

    return results


@torch.compile(dynamic=True)
def _get_logits_head_gate(indexer, x: torch.Tensor, q_scale: torch.Tensor):
    """ _get_logits_head_gate """
    weights, _ = indexer.weights_proj(x.to(indexer.weights_proj.weight.dtype))
    # GLM 5掉点
    weights = weights.float()
    weights = weights * indexer.n_heads**-0.5
    weights = weights * q_scale * indexer.softmax_scale
    return weights

def _get_topk_paged(
    indexer,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """ _get_topk_paged """
    if TYPE_CHECKING:
        assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

    page_size = get_token_to_kv_pool().page_size
    assert page_size == 64, "only support page size 64"

    block_tables = metadata.get_page_table_64()
    max_seq_len = block_tables.shape[1] * page_size

    kv_cache_fp8 = get_token_to_kv_pool().get_index_k_with_scale_buffer(
        layer_id=layer_id
    )

    block_kv = 64
    num_heads_kv = 1
    
    if (
        forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
    ):
        seqlens_32 = metadata.get_seqlens_expanded()
        seqlens_32_cpu = metadata.get_seqlens_expanded_cpu()
    else:
        seqlens_32 = metadata.get_seqlens_int32()
        seqlens_32_cpu = metadata.get_seqlens_int32_cpu()

    assert len(q_fp8.shape) == 3, f"q_fp8.shape: {q_fp8.shape}"
    q_fp8 = q_fp8.unsqueeze(1)

    if len(weights.shape) == 2:
        weights = weights.unsqueeze(1)
    
    # When attn_tp_size > 1 or in the MAX_LEN padding mode, padding may exist in the hidden states,
    # and it is necessary to extract the actual q length.
    q_offset = sum(metadata.get_dsa_extend_len_cpu())
        
    head_dim = get_token_to_kv_pool().index_head_dim
    k_fp8 = kv_cache_fp8[:, :page_size * head_dim].view(
        kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim
    ).to(torch.int8)
    
    k_scale = torch.ops.xspeedgate_ops.gather_paged_scale(
        kv_cache=kv_cache_fp8,
        block_tables=block_tables,
        page_size=page_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        scale_offset=page_size * head_dim,
        context_lens=seqlens_32,
    )

    # NOTE: reverted luno-4142 (1830d8ea) -- scale the gathered k scales here
    # instead of relying on the kernel's q_is_int8 argument.
    if q_fp8.dtype is torch.int8:
        k_scale /= 127
        
    fused_kv_cache = [k_fp8.contiguous(), k_scale.contiguous()]
    logits = torch.empty(
        (q_offset, q_fp8.shape[1], max_seq_len),
        dtype=torch.float32,
        device=q_fp8.device
    )

    kunlun_ops.I8_paged_mqa_logits(
        q=q_fp8[:q_offset],
        fused_kv_cache=fused_kv_cache,
        weights=weights[:q_offset],
        context_lens=[seqlens_32_cpu, seqlens_32],
        block_table=block_tables,
        max_context_len=max_seq_len,
        clean_logits=False,
        out=logits,
        use_xfa_boost=False
    )
    
    logits = logits.reshape(-1, max_seq_len)
    
    topk_result = metadata.topk_transform(logits, indexer.index_topk)
    # Restore possible padding exist in the hidden states.
    if q_offset < q_fp8.shape[0]:
        pad_len = q_fp8.shape[0] - q_offset
        padding = torch.full(
            (pad_len, topk_result.shape[1]),
            -1,
            dtype=topk_result.dtype,
            device=topk_result.device,
        )
        topk_result = torch.cat([topk_result, padding], dim=0)
    del logits
    return topk_result


def _get_topk_ragged(
    indexer,
    enable_dual_stream: bool,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata: "BaseIndexerMetadata",
    topk_result: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Kunlun ragged indexer.

    ``enable_dual_stream`` / ``topk_result`` exist for upstream 0.5.14 signature
    compatibility; the Kunlun path never uses the alternate stream and always
    allocates its own ``topk_result``.
    """
    if TYPE_CHECKING:
        assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

    assert forward_batch.forward_mode.is_extend_without_speculative()

    page_size = get_token_to_kv_pool().page_size
    assert page_size == 64, "only support page size 64"
    if len(weights.shape) == 3:
        weights = weights.squeeze(-1)
    k_fp8_list = []
    k_scale_list = []
    block_tables = metadata.get_page_table_64()

    assert (
        forward_batch.seq_lens_cpu is not None
        and forward_batch.extend_seq_lens_cpu is not None
    )

    batch_size = len(block_tables)
    token_nums, _, _ = q_fp8.shape
    device = q_fp8.device
    topk_result = torch.full(
        (token_nums, indexer.index_topk), -1, device=device, dtype=torch.int32
    )
    if batch_size == 0:
        return topk_result

    indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
    assert len(indexer_seq_lens_cpu) == batch_size
    for i in range(batch_size):
        seq_len = indexer_seq_lens_cpu[i].item()
        assert isinstance(seq_len, int)
        # Use fused Triton kernel to get both K and scale in a single call
        k_fp8 = get_token_to_kv_pool().get_index_k_continuous(
            layer_id,
            seq_len,
            block_tables[i],
        )
        k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
            layer_id,
            seq_len,
            block_tables[i],
        )

        k_fp8_list.append(k_fp8)
        k_scale_list.append(k_scale)
    
    k_fp8 = torch.cat(k_fp8_list, dim=0)
    k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
    if q_fp8.dtype is torch.int8:
        k_scale /= 127
    ks, ke = metadata.get_indexer_kvcache_range()
    seq_lens_expanded = metadata.get_seqlens_expanded()
    token_to_batch_idx = metadata.get_token_to_batch_idx()
    q_offset = ks.shape[0]
    k_offset = k_fp8.shape[0]

    # Suppose there are two requests, with extend_seq_len = [3, 2]
    # and seq_lens = [10, 4]
    # The logits matrix looks like this, with * representing the valid logits
    # and - representing the invalid logits:
    #
    #  ********--|----
    #  *********-|----
    #  **********|----
    #  ----------|***-
    #  ----------|****
    #
    # ks = [0, 0, 0, 10, 10]
    # ke = [8, 9, 10, 13, 14]

    token_nums, _, _ = q_fp8.shape
    device = q_fp8.device

    # Check if we need to chunk to avoid OOM
    need_chunk, free_mem = indexer._should_chunk_mqa_logits(q_offset, k_offset, device)
    if need_chunk:
        import logging
        logging.warning(f"[NSA] chunk path triggered: q_offset={q_offset}, \
            k_offset={k_offset}, free_mem={free_mem:.0f}")

    context_q_lens_xpu = metadata.get_cu_seqlens_q()
    context_q_lens_cpu = metadata.get_cu_seqlens_q_cpu()
    context_k_lens_xpu = metadata.get_cu_seqlens_k()
    context_k_lens_cpu = metadata.get_cu_seqlens_k_cpu()
    if not need_chunk:
        assert q_fp8[:q_offset].shape[0] != 0

        # Use max_k layout: logits shape = (q_total, max_k_per_batch).
        # I8_mqa_logits handles multi-batch correctly with this layout
        # (each batch's results written from column 0), avoiding the
        # total_k layout bug where batch 1+ are skipped.
        # This also makes fast_topk_transform_fused work correctly:
        # valid K at columns [0, k_len_b) matches LOCAL seq_lens_expanded
        # and page_table indexing, eliminating per-batch loops entirely.
        max_k = int(indexer_seq_lens_cpu.max().item())
        logits = torch.full((q_offset, max_k), -1.0, dtype=torch.float32, device=q_fp8.device)

        kunlun_ops.I8_mqa_logits(
            q=q_fp8[:q_offset],
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights[:q_offset],
            context_q_lens=(context_q_lens_cpu, context_q_lens_xpu),
            context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
            logits=logits,
            clean_logits=True,
            use_xfa_boost=False,
        )

        # Mask logits outside causal range [0, ke_local) per token.
        # In max_k layout each batch's K starts from column 0, so
        # ks_local = 0 for all tokens and ke_local = seq_lens_expanded.
        ks_local = torch.zeros(q_offset, dtype=ks.dtype, device=ks.device)
        ke_local = seq_lens_expanded.to(dtype=ks.dtype)
        torch.ops.xspeedgate_ops.mask_for_I8_mqa_logits(
            seq_len_kv=max_k,
            cu_seqlen_ks=ks_local.contiguous(),
            cu_seqlen_ke=ke_local.contiguous(),
            logits=logits,
        )

        assert logits.shape[0] == len(seq_lens_expanded)

        page_table_1 = metadata.get_page_table_1()
        cu_seqlens_q_topk = metadata.attn_metadata.cu_seqlens_q
        dst_page_table = logits.new_empty((logits.size(0), indexer.index_topk), dtype=torch.int32)
        torch.ops.xspeedgate_ops.topk_transform(
            score=logits.contiguous(),
            lengths=seq_lens_expanded.contiguous(),
            src_page_table=page_table_1.contiguous(),
            dst_page_table=dst_page_table,
            topk=indexer.index_topk,
            cu_seqlens_q=cu_seqlens_q_topk.contiguous(),
        )
        topk_result[:q_offset] = dst_page_table

        del logits
        del ks, q_fp8, k_fp8, k_scale

        return topk_result

    # Chunk path
    bytes_per_elem = 4  # float32
    bytes_per_row = k_offset * bytes_per_elem
    # Reserve 50% of free memory for logits
    max_rows = max(1, int((free_mem * 0.5) // max(bytes_per_row, 1)))
    max_rows = min(max_rows, q_offset)

    global_topk_offset = metadata.attn_metadata.topk_indices_offset

    assert (
        seq_lens_expanded.shape[0] == q_offset
    ), f"seq_lens_expanded length mismatch: {seq_lens_expanded.shape[0]} != {q_offset}"
    if global_topk_offset is not None:
        assert (
            global_topk_offset.shape[0] >= q_offset
        ), f"topk_indices_offset too short: {global_topk_offset.shape[0]} < {q_offset}"

    start = 0
    q_interval_cpu = chunk_cu_seqlens(context_q_lens_cpu, chunk_size=max_rows)
    # 重新计算，避免H2D
    q_interval_xpu = chunk_cu_seqlens(context_q_lens_xpu, chunk_size=max_rows)
    k_interval_cpu = chunk_cu_seqlens(context_k_lens_cpu, chunk_size=max_rows)
    k_interval_xpu = chunk_cu_seqlens(context_k_lens_xpu, chunk_size=max_rows)
    for i in range(0, len(q_interval_cpu)):
        chunk_context_q_lens_cpu, _, chunk_range = q_interval_cpu[i]
        chunk_context_k_lens_cpu, _, _ = k_interval_cpu[i]
        start = chunk_range[0].item()
        end   = chunk_range[1].item()

        logits_chunk = torch.full(
            (end - start, k_fp8.shape[0]),
            -1.0,
            dtype=torch.float32,
            device=q_fp8.device
        )
        chunk_context_q_lens_xpu, _, _ = q_interval_xpu[i]
        chunk_context_k_lens_xpu, _, _ = k_interval_xpu[i]
        kunlun_ops.I8_mqa_logits(
            q=q_fp8[start:end],
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights[start:end],
            context_q_lens=(chunk_context_q_lens_cpu, chunk_context_q_lens_xpu),
            context_k_lens=(chunk_context_k_lens_cpu, chunk_context_k_lens_xpu),
            logits=logits_chunk,
            clean_logits=True,
            use_xfa_boost=False,
        )

        # Mask logits outside [ks, ke) to prevent fused topk from selecting
        # invalid entries (fused kernel searches from col 0, not from ks)
        seq_len_kv = k_fp8.shape[0]
        torch.ops.xspeedgate_ops.mask_for_I8_mqa_logits(
            seq_len_kv=seq_len_kv,
            cu_seqlen_ks=ks[start:end],
            cu_seqlen_ke=ke[start:end],
            logits=logits_chunk
        )

        lengths_chunk = seq_lens_expanded[start:end]

        # RAGGED: use global offset; PAGED: construct local cu_seqlens_q per chunk
        if global_topk_offset is not None:
            # RAGGED path
            topk_offset_chunk = global_topk_offset[start:end]
            cu_seqlens_q_chunk = None
            batch_idx_chunk = None
        else:
            # PAGED path: treat each token as a length-1 sequence
            topk_offset_chunk = None
            B_chunk = logits_chunk.shape[0]
            cu_seqlens_q_chunk = torch.ones(
                B_chunk, dtype=torch.int32, device=device
            )
            batch_idx_chunk = token_to_batch_idx[start:end]

        raw_topk_chunk = metadata.topk_transform(
            logits_chunk,
            indexer.index_topk,
            ks=ks[start:end],
            cu_seqlens_q=cu_seqlens_q_chunk,
            ke_offset=ke[start:end],
            batch_idx_list=batch_idx_chunk,
            topk_indices_offset_override=topk_offset_chunk,
        )
        del logits_chunk
        topk_result[start:end] = raw_topk_chunk
    del ks, q_fp8, k_fp8, k_scale
    del q_interval_cpu, q_interval_xpu, k_interval_cpu, k_interval_xpu
    return topk_result

def _get_topk_ragged_with_cp(
    indexer,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata: "BaseIndexerMetadata",
    kv_len: int,
    actual_seq_q: int,
    cp_index: List[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    if TYPE_CHECKING:
        assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

    page_size = get_token_to_kv_pool().page_size
    assert page_size == 64, "only support page size 64"
    if len(weights.shape) == 3:
        weights = weights.squeeze(-1)
    k_fp8_list = []
    k_scale_list = []
    ks_list = []
    ke_offset_list = []
    offset = 0
    actual_seq_q_list = []
    batch_idx_list = []

    block_tables = metadata.get_page_table_64()

    assert (
        forward_batch.seq_lens_cpu is not None
        and forward_batch.extend_seq_lens_cpu is not None
    )
    if cp_index is not None:
        # TODO Multi-batch support has accuracy issues
        for batch_idx, start_seq_position, end_seq_position in cp_index:
            pre_chunk_offset = (
                forward_batch.seq_lens_cpu[batch_idx].item()
                - forward_batch.extend_seq_lens_cpu[batch_idx]
            )
            start_seq_position += pre_chunk_offset
            end_seq_position += pre_chunk_offset
            if offset == 0 and batch_idx != 0:
                offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
            k_fp8 = get_token_to_kv_pool().get_index_k_continuous(
                layer_id,
                end_seq_position,
                block_tables[batch_idx],
            )
            k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
                layer_id,
                end_seq_position,
                block_tables[batch_idx],
            )

            extend_seq_len = end_seq_position - start_seq_position
            ks = torch.full(
                (extend_seq_len,), offset, dtype=torch.int32, device="cuda"
            )
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            ke_offset = torch.arange(
                start_seq_position + 1,
                end_seq_position + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke_offset_list.append(ke_offset)
            actual_seq_q = torch.tensor(
                [extend_seq_len], dtype=torch.int32, device="cuda"
            )
            actual_seq_q_list.append(actual_seq_q)
            batch_idx_list.append(batch_idx)

        k_fp8 = torch.cat(k_fp8_list, dim=0)#.view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        k_scale /= 127
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        ke_offset = torch.cat(ke_offset_list, dim=0)
        ke = ks + ke_offset
        actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
        
        context_q_lens_xpu = torch.tensor([0, q_fp8.shape[0]], dtype=torch.int32, device="cuda")
        context_q_lens_cpu = torch.tensor([0, q_fp8.shape[0]], dtype=torch.int32, device="cpu")
        context_k_lens_xpu = torch.tensor([0, k_fp8.shape[0]], dtype=torch.int32, device="cuda")
        context_k_lens_cpu = torch.tensor([0, k_fp8.shape[0]], dtype=torch.int32, device="cpu")

        logits = torch.zeros((q_fp8.shape[0], k_fp8.shape[0]), dtype=torch.float32, device=q_fp8.device)
        kunlun_ops.I8_mqa_logits(
            q=q_fp8,
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights,
            context_q_lens=(context_q_lens_cpu, context_q_lens_xpu),
            context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
            logits=logits,
            clean_logits=True,
            use_xfa_boost=False,
        )
        topk_result = metadata.topk_transform(
            logits,
            indexer.index_topk,
            ks=ks,
            cu_seqlens_q=actual_seq_q,
            ke_offset=ke_offset,
            batch_idx_list=batch_idx_list,
        )
    else:
        kv_len = (
            forward_batch.seq_lens_cpu[0].item()
            - forward_batch.extend_seq_lens_cpu[0]
            + kv_len
        )
        k_fp8 = get_token_to_kv_pool().get_index_k_continuous(
            layer_id,
            kv_len,
            block_tables[0],
        )
        k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
            layer_id,
            kv_len,
            block_tables[0],
        )

        #k_fp8 = k_fp8.view(torch.float8_e4m3fn)
        k_scale = k_scale.view(torch.float32).squeeze(-1)
        #k_scale /= 127
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
        ke_offset = torch.arange(
            (kv_len - actual_seq_q) + 1,
            kv_len + 1,
            dtype=torch.int32,
            device="cuda",
        )
        ke = ks + ke_offset

        context_q_lens_xpu = torch.tensor([0, q_fp8.shape[0]], dtype=torch.int32, device="cuda")
        context_q_lens_cpu = torch.tensor([0, q_fp8.shape[0]], dtype=torch.int32, device="cpu")
        context_k_lens_xpu = torch.tensor([0, k_fp8.shape[0]], dtype=torch.int32, device="cuda")
        context_k_lens_cpu = torch.tensor([0, k_fp8.shape[0]], dtype=torch.int32, device="cpu")

        logits = torch.zeros((q_fp8.shape[0], k_fp8.shape[0]), dtype=torch.float32, device=q_fp8.device)
        kunlun_ops.I8_mqa_logits(
            q=q_fp8,
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights,
            context_q_lens=(context_q_lens_cpu, context_q_lens_xpu),
            context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
            logits=logits,
            clean_logits=True,
            use_xfa_boost=False,
        )
        actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
            device="cuda", non_blocking=True
        )
        topk_result = metadata.topk_transform(
            logits,
            indexer.index_topk,
            ks=ks,
            cu_seqlens_q=actual_seq_q,
            ke_offset=ke_offset,
        )

    return topk_result


def _get_q_k_bf16(
    indexer,
    q_lora: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor,
    enable_dual_stream: bool,
    forward_batch: ForwardBatch,
):
    if enable_dual_stream:
        current_stream = torch.cuda.current_stream()
        indexer.alt_stream.wait_stream(current_stream)
        # for kunlun ops
        query, _ = indexer.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=indexer.head_dim)
        q_rope, _ = torch.split(
            query,
            [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim],
            dim=-1,
        )
        with torch.cuda.stream(indexer.alt_stream):
            # TODO we should also put DeepGEMM half SM here?
            key, _ = indexer.wk(x)
            key = indexer.k_norm(key)

            k_rope, _ = torch.split(
                key,
                [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim],
                dim=-1,
            )

        current_stream.wait_stream(indexer.alt_stream)
    else:
        query, _ = indexer.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=indexer.head_dim)
        q_rope, _ = torch.split(
            query, [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim], dim=-1
        )
        key, _ = indexer.wk(x)
        key = indexer.k_norm(key)
        k_rope, _ = torch.split(
            key, [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim], dim=-1
        )

    q_rope, k_rope = indexer.rotary_emb(positions, q_rope, k_rope)

    query[..., : indexer.rope_head_dim] = q_rope
    key[..., : indexer.rope_head_dim] = k_rope

    if enable_dual_stream:
        current_stream = torch.cuda.current_stream()
        indexer.alt_stream.wait_stream(current_stream)
        query = rotate_activation(query)

        with torch.cuda.stream(indexer.alt_stream):
            key = rotate_activation(key)
        current_stream.wait_stream(indexer.alt_stream)
    else:
        query = rotate_activation(query)
        key = rotate_activation(key)

    # allgather+rerrange
    if forward_batch.attn_cp_metadata is not None and indexer.dsa_enable_prefill_cp:
        key = cp_all_gather_rerange_output(
            key.contiguous(),
            indexer.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )

    return query, key


def _forward_cuda_k_only(
    indexer,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    act_quant,
    enable_dual_stream: bool,
    metadata: "BaseIndexerMetadata",
    return_indices: bool = True,
):
    assert forward_batch.forward_mode.is_extend_without_speculative()

    # Fast path: only compute and store k cache, skip all q and weights ops
    key = indexer._get_k_bf16(x, positions, enable_dual_stream)
    key_shape = key.shape
    key = key.view(-1, indexer.block_size)
    k_fp8 = torch.empty(
        key.shape,
        device=key.device,
        dtype=torch.int8,
    )
    k_scale = torch.empty(
        [key.shape[0], 1],
        device=key.device,
        dtype=torch.float32,
    )
    kunlun_ops.quant2d(key, k_fp8, k_scale, force_sdnn=True)
    k_fp8 = k_fp8.view(key_shape)
    k_scale = k_scale.view(key_shape[0], -1)

    if not forward_batch.out_cache_loc.is_contiguous():
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

    get_token_to_kv_pool().set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=forward_batch.out_cache_loc,
        index_k=k_fp8,
        index_k_scale=k_scale,
    )

    # MHA doesn't need topk_indices
    if not return_indices:
        return None

    # MLA: every position is selected when the sequence is not longer than
    # index_topk, so the fused top-k result is just each token's own cache slots
    # in order, padded with -1. Build that page table directly instead of running
    # the topk kernel on dummy logits: on 0.5.14 the Kunlun topk_transform
    # requires the score width to fit inside the page-1 table, which
    # `[tokens, index_topk]` dummy logits violate for short sequences.
    seq_lens_expanded = metadata.get_seqlens_expanded()
    page_table_1 = metadata.get_page_table_1()
    token_to_batch_idx = metadata.get_token_to_batch_idx()
    topk = indexer.index_topk
    device = page_table_1.device
    width = page_table_1.shape[1]

    positions = torch.arange(topk, device=device)
    src_index = token_to_batch_idx.to(torch.int64).unsqueeze(1) * width + (
        positions.clamp(max=width - 1)
    )
    slots = (
        page_table_1.reshape(-1)
        .index_select(0, src_index.reshape(-1))
        .view(-1, topk)
    )
    valid = positions.unsqueeze(0) < seq_lens_expanded.unsqueeze(1)
    return torch.where(valid, slots, -torch.ones_like(slots)).to(torch.int32)


# ``MultiPlatformOp`` dispatches to ``forward_xpu`` on this platform (0.5.14 dropped
# ``forward_native``), so both entry points resolve to the Kunlun implementation.
@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer.forward_cuda",
    type=HookType.REPLACE,
)
def forward_cuda(
    indexer,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    return_indices: bool = True,
) -> Optional[torch.Tensor]:
    """forward_cuda"""
    # TODO: try to remove this patch

    from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

    if TYPE_CHECKING:
        assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)
    # a tuple like (x_fp8, x_scale[, y]). Use `x_meta` for shape/device queries.
    x_meta = x[0] if isinstance(x, tuple) else x

    # Kunlun does not support upstream's TC piecewise cuda graph path (it splits the
    # graph around store_k_cache/mqa_logits and fetches metadata inside custom ops).
    # Fail loudly rather than silently running with metadata=None.
    assert (
        not is_in_tc_piecewise_cuda_graph()
    ), "kunlun dsa indexer does not support TC piecewise cuda graph"

    metadata = get_attn_backend().get_indexer_metadata(
        layer_id, forward_batch
    )

    _resolve_deps(indexer)

    # skip DSA if attention backend choose to skip this batch
    if metadata is None:
        return None

    # Determine if should skip topk based on sequence length
    # We can only skip the logits computation if cuda graph is not involved
    skip_logits_computation = False
    if forward_batch.forward_mode.is_extend_without_speculative():
        if forward_batch.seq_lens_cpu is not None:
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            skip_logits_computation = max_kv_len <= indexer.index_topk

    # Optimization: fast path when skipping topk computation
    if skip_logits_computation and (not indexer.dsa_enable_prefill_cp):
        topk_result = _forward_cuda_k_only(indexer,
            x,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            False,  # dual stream is never enabled on kunlun
            metadata,
            return_indices,
        )
        # 0.5.14 additions: TP broadcast + state capture on every return path
        topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
        return maybe_capture_indexer_topk(layer_id, topk_result)

    query, key = _get_q_k_bf16(indexer,
            q_lora, x, positions, False, forward_batch=forward_batch
        )
    # TODO: try to remove this to act_quant
    query_shape = query.shape
    key_shape = key.shape
    query = query.view(-1, indexer.block_size)
    key = key.view(-1, indexer.block_size)
    q_fp8 = torch.empty(
        query.shape,
        device=query.device,
        dtype=torch.int8,
    )
    q_scale = torch.empty(
        [query.shape[0], 1],
        device=query.device,
        dtype=torch.float32,
    )
    k_fp8 = torch.empty(
        key.shape,
        device=key.device,
        dtype=torch.int8,
    )
    k_scale = torch.empty(
        [key.shape[0], 1],
        device=key.device,
        dtype=torch.float32,
    )

    kunlun_ops.quant2d(query, q_fp8, q_scale, force_sdnn=True)
    q_fp8 = q_fp8.view(query_shape)
    q_scale = q_scale.view(query_shape[0], -1)

    kunlun_ops.quant2d(key, k_fp8, k_scale, force_sdnn=True)
    k_fp8 = k_fp8.view(key_shape)
    k_scale = k_scale.view(key_shape[0], -1)
    
    # `_get_logits_head_gate` expects a Tensor. For tuple activations, dequantize
    # to a float tensor here (callsite), keeping `_get_logits_head_gate` backend-agnostic.
    if isinstance(x, tuple):
        assert len(x) in (
            2,
            3,
        ), "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
        x_q, x_s = x[0], x[1]
        if (
            x_s is not None
            and x_q.dim() == 2
            and x_s.dim() == 2
            and x_q.shape[0] == x_s.shape[0]
        ):
            m, n = x_q.shape
            ng = x_s.shape[1]
            if ng > 0 and n % ng == 0:
                group = n // ng
                x_for_gate = (
                    x_q.to(torch.float32)
                    .view(m, ng, group)
                    .mul_(x_s.to(torch.float32).unsqueeze(-1))
                    .view(m, n)
                    .to(torch.bfloat16)
                )
            else:
                x_for_gate = x_q.to(torch.bfloat16)
        else:
            x_for_gate = x_q.to(torch.bfloat16)
    else:
        x_for_gate = x
    weights = _get_logits_head_gate(indexer, x_for_gate, q_scale)
    # k_fp8: (seq_len, head_dim) fp8_e4m3fn
    # k_buffer: (num_total_tokens + page_size, head_dim) fp8_e4m3fn
    # k_scale: (seq_len, head_dim // block_size = 1) fp8_e4m3fn
    # k_scale_cache: (num_total_tokens + page_size, head_dim // block_size = 1) fp8_e4m3fn
    if not forward_batch.out_cache_loc.is_contiguous():
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
    get_token_to_kv_pool().set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=forward_batch.out_cache_loc,
        index_k=k_fp8,
        index_k_scale=k_scale,
    )
    seq_len = forward_batch.seq_lens_cpu[0].item()
    
    # TODO: remove this to _get_logits_head_gate
    if q_fp8.dtype is torch.int8:
        weights /= 127

    assert forward_batch.seq_lens_cpu is not None

    if (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
    ):
        topk_result = _get_topk_paged(indexer,
            forward_batch, layer_id, q_fp8, weights, metadata
        )
    else:
        if (
            forward_batch.attn_cp_metadata is not None
            and is_dsa_prefill_cp_in_seq_split()
        ):
            kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev_list[0]
            kv_len_next = forward_batch.attn_cp_metadata.kv_len_next_list[0]
            actual_seq_q_prev = forward_batch.attn_cp_metadata.actual_seq_q_prev_list[0]
            actual_seq_q_next = forward_batch.attn_cp_metadata.actual_seq_q_next_list[0]

            # TODO support mutil-batch
            # cp_batch_seq_index_prev = forward_batch.attn_cp_metadata["cp_batch_seq_index_prev"]
            # cp_batch_seq_index_next = forward_batch.attn_cp_metadata["cp_batch_seq_index_next"]
            # TODO prev, next, combined into a single call
            q_fp8_prev, q_fp8_next = torch.split(
                q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
            )
            weights_prev, weights_next = torch.split(
                weights, (weights.shape[0] + 1) // 2, dim=0
            )
            topk_result_prev = _get_topk_ragged_with_cp(indexer,
                forward_batch,
                layer_id,
                q_fp8_prev,
                weights_prev,
                metadata,
                kv_len_prev,
                actual_seq_q_prev,
            )

            topk_result_next = _get_topk_ragged_with_cp(indexer,
                forward_batch,
                layer_id,
                q_fp8_next,
                weights_next,
                metadata,
                kv_len_next,
                actual_seq_q_next,
            )
            topk_result = torch.cat([topk_result_prev, topk_result_next], dim=0)
            topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
            return maybe_capture_indexer_topk(layer_id, topk_result)
        else:
            topk_result = _get_topk_ragged(indexer,
                False,  # enable_dual_stream: never used on kunlun
                forward_batch, layer_id, q_fp8, weights, metadata
            )
    topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
    return maybe_capture_indexer_topk(layer_id, topk_result)
