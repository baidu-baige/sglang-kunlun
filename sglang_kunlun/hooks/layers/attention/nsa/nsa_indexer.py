"""Kunlun overrides for ``sglang.srt.layers.attention.nsa.nsa_indexer``.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/attention/nsa/nsa_indexer.py

This module replaces seven Indexer methods with Kunlun-specific implementations
that route through ``kunlun_ops`` and ``xspeedgate_ops``. Heavy quantization
math is expressed via ``torch.compile(dynamic=True)`` to mirror mimo.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from einops import rearrange

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

logger = logging.getLogger(__name__)


def _shape_meta(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "contiguous": tensor.is_contiguous(),
        }
    if isinstance(tensor, (list, tuple)):
        return [_shape_meta(item) for item in tensor]
    return tensor


def _small_cpu_list(tensor, limit: int = 16):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        flat = tensor.detach().flatten()
        return flat[:limit].cpu().tolist()
    return tensor


def _make_cu_seqlens_from_lens(seq_lens_cpu: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    cu_cpu = torch.nn.functional.pad(
        torch.cumsum(seq_lens_cpu.to(torch.int32), dim=0, dtype=torch.int32), (1, 0)
    )
    cu_xpu = cu_cpu.to(device=device, non_blocking=True)
    return cu_cpu, cu_xpu


# --------------------------------------------------------------------------
# Helpers (free functions — used by the patched methods below).
# --------------------------------------------------------------------------


def chunk_cu_seqlens(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
):
    """topk 按照 chunk_size 切分原始 sequence 区间."""
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


def _hadamard_transform(x: torch.Tensor, scale: float = None) -> torch.Tensor:
    if scale is None:
        return torch.ops.xspeedgate_ops.hadamard_transform(x)
    return torch.ops.xspeedgate_ops.hadamard_transform(x, scale)


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return _hadamard_transform(x, hidden_size**-0.5)


# --------------------------------------------------------------------------
# Indexer method replacements (plugin_hook REPLACE).
# --------------------------------------------------------------------------


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._project_and_scale_head_gates",
    type=HookType.REPLACE,
)
@torch.compile(dynamic=True)
def _project_and_scale_head_gates(self, x: torch.Tensor):
    weights, _ = self.weights_proj(x.to(self.weights_proj.weight.dtype))
    weights = weights.float()
    weights = weights * self.n_heads**-0.5
    return weights


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_logits_head_gate",
    type=HookType.REPLACE,
)
@torch.compile(dynamic=True)
def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
    weights, _ = self.weights_proj(x.to(self.weights_proj.weight.dtype))
    weights = weights.float()
    weights = weights * self.n_heads**-0.5
    weights = weights * q_scale * self.softmax_scale
    return weights


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged",
    type=HookType.REPLACE,
)
def _get_topk_paged(
    self,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """_get_topk_paged Kunlun replacement."""
    import kunlun_ops

    if TYPE_CHECKING:
        assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

    page_size = forward_batch.token_to_kv_pool.page_size
    assert page_size == 64, "only support page size 64"
    block_tables = metadata.get_page_table_64()
    max_seq_len = block_tables.shape[1] * page_size
    kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
        layer_id=layer_id
    )
    block_kv = 64
    num_heads_kv = 1

    if (
        forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend(include_v2=True)
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

    q_offset = sum(metadata.get_nsa_extend_len_cpu())

    head_dim = forward_batch.token_to_kv_pool.index_head_dim
    k_fp8 = (
        kv_cache_fp8[:, : page_size * head_dim]
        .view(kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim)
        .to(torch.int8)
    )
    k_scale = torch.ops.xspeedgate_ops.gather_paged_scale(
        kv_cache=kv_cache_fp8,
        block_tables=block_tables,
        page_size=page_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        scale_offset=page_size * head_dim,
        context_lens=seqlens_32,
        q_is_int8=(q_fp8.dtype == torch.int8),
    )

    fused_kv_cache = [k_fp8.contiguous(), k_scale.contiguous()]
    logits = torch.empty(
        (q_offset, q_fp8.shape[1], max_seq_len),
        dtype=torch.float32,
        device=q_fp8.device,
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
        use_xfa_boost=False,
    )

    logits = logits.reshape(-1, max_seq_len)
    topk_result = metadata.topk_transform(logits, self.index_topk)
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


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_ragged",
    type=HookType.REPLACE,
)
def _get_topk_ragged(
    self,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """_get_topk_ragged Kunlun replacement."""
    import kunlun_ops

    if TYPE_CHECKING:
        assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)
    assert forward_batch.forward_mode.is_extend_without_speculative()

    page_size = forward_batch.token_to_kv_pool.page_size
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
        (token_nums, self.index_topk), -1, device=device, dtype=torch.int32
    )
    if batch_size == 0:
        return topk_result

    indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
    assert len(indexer_seq_lens_cpu) == batch_size
    for i in range(batch_size):
        seq_len = indexer_seq_lens_cpu[i].item()
        assert isinstance(seq_len, int)
        k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
            layer_id, seq_len, block_tables[i]
        )
        k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
            layer_id, seq_len, block_tables[i]
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

    token_nums, _, _ = q_fp8.shape
    device = q_fp8.device

    need_chunk, free_mem = self._should_chunk_mqa_logits(q_offset, k_offset, device)

    context_q_lens_xpu = metadata.get_cu_seqlens_q()
    context_q_lens_cpu = metadata.get_cu_seqlens_q_cpu()
    context_k_lens_cpu, context_k_lens_xpu = _make_cu_seqlens_from_lens(
        indexer_seq_lens_cpu, device
    )
    if not need_chunk:
        assert q_fp8[:q_offset].shape[0] != 0

        max_k = int(indexer_seq_lens_cpu.max().item())
        logits = torch.full(
            (q_offset, max_k), -1.0, dtype=torch.float32, device=q_fp8.device
        )
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
        dst_page_table = logits.new_empty(
            (logits.size(0), self.index_topk), dtype=torch.int32
        )
        torch.ops.xspeedgate_ops.topk_transform(
            score=logits.contiguous(),
            lengths=seq_lens_expanded.contiguous(),
            src_page_table=page_table_1.contiguous(),
            dst_page_table=dst_page_table,
            topk=self.index_topk,
            cu_seqlens_q=cu_seqlens_q_topk.contiguous(),
        )
        del logits
        del ks, q_fp8, k_fp8, k_scale
        topk_result[:q_offset] = dst_page_table
        return topk_result

    bytes_per_elem = 4
    bytes_per_row = k_offset * bytes_per_elem
    max_rows = max(1, int((free_mem * 0.5) // max(bytes_per_row, 1)))
    max_rows = min(max_rows, q_offset)

    global_topk_offset = metadata.attn_metadata.topk_indices_offset
    assert seq_lens_expanded.shape[0] == q_offset
    if global_topk_offset is not None:
        assert global_topk_offset.shape[0] >= q_offset

    q_interval_cpu = chunk_cu_seqlens(context_q_lens_cpu, chunk_size=max_rows)
    q_interval_xpu = chunk_cu_seqlens(context_q_lens_xpu, chunk_size=max_rows)
    for i in range(0, len(q_interval_cpu)):
        chunk_context_q_lens_cpu, _, chunk_range = q_interval_cpu[i]
        start = chunk_range[0].item()
        end = chunk_range[1].item()
        chunk_context_q_lens_cpu = chunk_context_q_lens_cpu - chunk_context_q_lens_cpu[0]

        logits_chunk = torch.full(
            (end - start, k_fp8.shape[0]),
            -1.0,
            dtype=torch.float32,
            device=q_fp8.device,
        )
        chunk_context_q_lens_xpu, _, _ = q_interval_xpu[i]
        chunk_context_q_lens_xpu = chunk_context_q_lens_xpu - chunk_context_q_lens_xpu[0]
        kunlun_ops.I8_mqa_logits(
            q=q_fp8[start:end],
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights[start:end],
            context_q_lens=(chunk_context_q_lens_cpu, chunk_context_q_lens_xpu),
            context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
            logits=logits_chunk,
            clean_logits=True,
            use_xfa_boost=False,
        )

        torch.ops.xspeedgate_ops.mask_for_I8_mqa_logits(
            seq_len_kv=k_fp8.shape[0],
            cu_seqlen_ks=ks[start:end],
            cu_seqlen_ke=ke[start:end],
            logits=logits_chunk,
        )

        if global_topk_offset is not None:
            topk_offset_chunk = global_topk_offset[start:end]
            cu_seqlens_q_chunk = None
            batch_idx_chunk = None
        else:
            topk_offset_chunk = None
            B_chunk = logits_chunk.shape[0]
            cu_seqlens_q_chunk = torch.ones(
                B_chunk, dtype=torch.int32, device=device
            )
            batch_idx_chunk = token_to_batch_idx[start:end]

        raw_topk_chunk = metadata.topk_transform(
            logits_chunk,
            self.index_topk,
            ks=ks[start:end],
            cu_seqlens_q=cu_seqlens_q_chunk,
            ke_offset=ke[start:end],
            batch_idx_list=batch_idx_chunk,
            topk_indices_offset_override=topk_offset_chunk,
        )
        del logits_chunk
        topk_result[start:end] = raw_topk_chunk
    del ks, q_fp8, k_fp8, k_scale
    del q_interval_cpu, q_interval_xpu
    return topk_result


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_ragged_with_cp",
    type=HookType.REPLACE,
)
def _get_topk_ragged_with_cp(
    self,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata,
    kv_len: int,
    actual_seq_q: int,
    cp_index: List[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    """_get_topk_ragged_with_cp Kunlun replacement."""
    import kunlun_ops

    if TYPE_CHECKING:
        assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)
    page_size = forward_batch.token_to_kv_pool.page_size
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
        for batch_idx, start_seq_position, end_seq_position in cp_index:
            pre_chunk_offset = (
                forward_batch.seq_lens_cpu[batch_idx].item()
                - forward_batch.extend_seq_lens_cpu[batch_idx]
            )
            start_seq_position += pre_chunk_offset
            end_seq_position += pre_chunk_offset
            if offset == 0 and batch_idx != 0:
                offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id, end_seq_position, block_tables[batch_idx]
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id, end_seq_position, block_tables[batch_idx]
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
            actual_seq_q_t = torch.tensor(
                [extend_seq_len], dtype=torch.int32, device="cuda"
            )
            actual_seq_q_list.append(actual_seq_q_t)
            batch_idx_list.append(batch_idx)

        k_fp8 = torch.cat(k_fp8_list, dim=0)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        k_scale /= 127
        ks = torch.cat(ks_list, dim=0)
        ke_offset = torch.cat(ke_offset_list, dim=0)
        actual_seq_q = torch.cat(actual_seq_q_list, dim=0)

        context_q_lens_xpu = torch.tensor(
            [0, q_fp8.shape[0]], dtype=torch.int32, device="cuda"
        )
        context_q_lens_cpu = torch.tensor(
            [0, q_fp8.shape[0]], dtype=torch.int32, device="cpu"
        )
        context_k_lens_xpu = torch.tensor(
            [0, k_fp8.shape[0]], dtype=torch.int32, device="cuda"
        )
        context_k_lens_cpu = torch.tensor(
            [0, k_fp8.shape[0]], dtype=torch.int32, device="cpu"
        )
        logits = torch.zeros(
            (q_fp8.shape[0], k_fp8.shape[0]), dtype=torch.float32, device=q_fp8.device
        )
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
            self.index_topk,
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
        k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
            layer_id, kv_len, block_tables[0]
        )
        k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
            layer_id, kv_len, block_tables[0]
        )
        k_scale = k_scale.view(torch.float32).squeeze(-1)
        ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
        ke_offset = torch.arange(
            (kv_len - actual_seq_q) + 1,
            kv_len + 1,
            dtype=torch.int32,
            device="cuda",
        )
        context_q_lens_xpu = torch.tensor(
            [0, q_fp8.shape[0]], dtype=torch.int32, device="cuda"
        )
        context_q_lens_cpu = torch.tensor(
            [0, q_fp8.shape[0]], dtype=torch.int32, device="cpu"
        )
        context_k_lens_xpu = torch.tensor(
            [0, k_fp8.shape[0]], dtype=torch.int32, device="cuda"
        )
        context_k_lens_cpu = torch.tensor(
            [0, k_fp8.shape[0]], dtype=torch.int32, device="cpu"
        )
        logits = torch.zeros(
            (q_fp8.shape[0], k_fp8.shape[0]), dtype=torch.float32, device=q_fp8.device
        )
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
            self.index_topk,
            ks=ks,
            cu_seqlens_q=actual_seq_q,
            ke_offset=ke_offset,
        )
    return topk_result


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_q_k_bf16",
    type=HookType.REPLACE,
)
def _get_q_k_bf16(
    self,
    q_lora: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor,
    enable_dual_stream: bool,
    forward_batch: ForwardBatch,
):
    """_get_q_k_bf16 Kunlun replacement."""
    if enable_dual_stream:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
        q_rope, _ = torch.split(
            query,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )
        with torch.cuda.stream(self.alt_stream):
            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )
        current_stream.wait_stream(self.alt_stream)
    else:
        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

    q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
    query[..., : self.rope_head_dim] = q_rope
    key[..., : self.rope_head_dim] = k_rope

    if enable_dual_stream:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        query = _rotate_activation(query)
        with torch.cuda.stream(self.alt_stream):
            key = _rotate_activation(key)
        current_stream.wait_stream(self.alt_stream)
    else:
        query = _rotate_activation(query)
        key = _rotate_activation(key)

    if forward_batch.attn_cp_metadata is not None and self.nsa_enable_prefill_cp:
        from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output

        key = cp_all_gather_rerange_output(
            key.contiguous(),
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
    return query, key


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._forward_cuda_k_only",
    type=HookType.REPLACE,
)
def _forward_cuda_k_only(
    self,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    act_quant,
    enable_dual_stream: bool,
    metadata,
    return_indices: bool = True,
):
    """_forward_cuda_k_only Kunlun replacement."""
    import kunlun_ops

    assert forward_batch.forward_mode.is_extend_without_speculative()
    x_meta = x[0] if isinstance(x, tuple) else x

    key = self._get_k_bf16(x, positions, enable_dual_stream)
    key_shape = key.shape
    key = key.view(-1, self.block_size)
    k_fp8 = torch.empty(key.shape, device=key.device, dtype=torch.int8)
    k_scale = torch.empty(
        [key.shape[0], 1], device=key.device, dtype=torch.float32
    )
    kunlun_ops.quant2d(key, k_fp8, k_scale, force_sdnn=True)
    k_fp8 = k_fp8.view(key_shape)
    k_scale = k_scale.view(key_shape[0], -1)

    if not forward_batch.out_cache_loc.is_contiguous():
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

    forward_batch.token_to_kv_pool.set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=forward_batch.out_cache_loc,
        index_k=k_fp8,
        index_k_scale=k_scale,
    )

    if not return_indices:
        return None

    seq_lens_expanded = metadata.get_seqlens_expanded()
    dummy_logits = torch.zeros(
        seq_lens_expanded.shape[0],
        self.index_topk,
        dtype=torch.float32,
        device=x_meta.device,
    )
    return metadata.topk_transform(dummy_logits, self.index_topk)


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer.forward_cuda",
    type=HookType.REPLACE,
)
def forward_cuda(
    self,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    return_indices: bool = True,
) -> Optional[torch.Tensor]:
    """forward_cuda Kunlun replacement."""
    import kunlun_ops
    from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

    if TYPE_CHECKING:
        assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)
    x_meta = x[0] if isinstance(x, tuple) else x
    metadata = forward_batch.attn_backend.get_indexer_metadata(layer_id, forward_batch)
    enable_dual_stream = False

    if metadata is None:
        return None

    skip_logits_computation = False
    if forward_batch.forward_mode.is_extend_without_speculative():
        if forward_batch.seq_lens_cpu is not None:
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            skip_logits_computation = max_kv_len <= self.index_topk

    if skip_logits_computation and (not self.nsa_enable_prefill_cp):
        return self._forward_cuda_k_only(
            x,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            enable_dual_stream,
            metadata,
            return_indices,
        )

    if enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        weights = self._project_and_scale_head_gates(x)
        query, key = self._get_q_k_bf16(
            q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
        )
        q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
        with torch.cuda.stream(self.alt_stream):
            k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
        current_stream.wait_stream(self.alt_stream)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
    else:
        query, key = self._get_q_k_bf16(
            q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
        )
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(-1, self.block_size)
        key = key.view(-1, self.block_size)
        q_fp8 = torch.empty(query.shape, device=query.device, dtype=torch.int8)
        q_scale = torch.empty(
            [query.shape[0], 1], device=query.device, dtype=torch.float32
        )
        k_fp8 = torch.empty(key.shape, device=key.device, dtype=torch.int8)
        k_scale = torch.empty(
            [key.shape[0], 1], device=key.device, dtype=torch.float32
        )

        kunlun_ops.quant2d(query, q_fp8, q_scale, force_sdnn=True)
        q_fp8 = q_fp8.view(query_shape)
        q_scale = q_scale.view(query_shape[0], -1)
        kunlun_ops.quant2d(key, k_fp8, k_scale, force_sdnn=True)
        k_fp8 = k_fp8.view(key_shape)
        k_scale = k_scale.view(key_shape[0], -1)

        if isinstance(x, tuple):
            assert len(x) in (2, 3)
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
        weights = self._get_logits_head_gate(x_for_gate, q_scale)

    if not forward_batch.out_cache_loc.is_contiguous():
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
    forward_batch.token_to_kv_pool.set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=forward_batch.out_cache_loc,
        index_k=k_fp8,
        index_k_scale=k_scale,
    )
    if q_fp8.dtype is torch.int8:
        weights /= 127

    assert forward_batch.seq_lens_cpu is not None
    if len(forward_batch.seq_lens_cpu) == 0:
        return torch.full(
            (x_meta.shape[0], self.index_topk),
            -1,
            dtype=torch.int,
            device=x_meta.device,
        )

    if (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend(include_v2=True)
    ):
        return self._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)

    from sglang.srt.layers.attention.nsa.utils import is_nsa_prefill_cp_in_seq_split

    if (
        forward_batch.nsa_cp_metadata is not None
        and is_nsa_prefill_cp_in_seq_split()
    ):
        kv_len_prev = forward_batch.nsa_cp_metadata.kv_len_prev
        kv_len_next = forward_batch.nsa_cp_metadata.kv_len_next
        actual_seq_q_prev = forward_batch.nsa_cp_metadata.actual_seq_q_prev
        actual_seq_q_next = forward_batch.nsa_cp_metadata.actual_seq_q_next
        q_fp8_prev, q_fp8_next = torch.split(
            q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
        )
        weights_prev, weights_next = torch.split(
            weights, (weights.shape[0] + 1) // 2, dim=0
        )
        topk_result_prev = self._get_topk_ragged_with_cp(
            forward_batch,
            layer_id,
            q_fp8_prev,
            weights_prev,
            metadata,
            kv_len_prev,
            actual_seq_q_prev,
        )
        topk_result_next = self._get_topk_ragged_with_cp(
            forward_batch,
            layer_id,
            q_fp8_next,
            weights_next,
            metadata,
            kv_len_next,
            actual_seq_q_next,
        )
        return torch.cat([topk_result_prev, topk_result_next], dim=0)

    return self._get_topk_ragged(forward_batch, layer_id, q_fp8, weights, metadata)

