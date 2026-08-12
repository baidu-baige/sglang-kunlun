"""kunlun_nsa attention backend for sglang 0.5.14 (DeepSeek sparse attention).

Ported from the 0.5.8 subclass
``sgl_kernel/patch/layers/attention/kunlun_nsa_backend.py``. Same design: subclass
upstream and override only what really differs on Kunlun.

0.5.14 deltas that this port had to honour (see
``/home/zx/doc/ds_v32/kunlun_dsa_0514_migration_delta.md``):

* ``NativeSparseAttnBackend`` -> ``DeepseekSparseAttnBackend``, ``NSAMetadata`` ->
  ``DSAMetadata``, ``NSAIndexerMetadata`` -> ``DSAIndexerMetadata``, all ``nsa_*``
  metadata fields -> ``dsa_*``; ``TopkTransformMethod`` moved to
  ``dsa/dsa_topk_backend.py``.
* the cuda-graph seams changed: there is no
  ``init_forward_metadata_capture_cuda_graph`` / ``..._replay_cuda_graph`` any more,
  so ``cum_q_lod`` is injected in ``_build_forward_metadata_cuda_graph`` and
  ``_apply_cuda_graph_metadata`` instead.
* ``DSAIndexerMetadata`` gained ``topk_backend`` / ``paged_mqa_ctx_lens_2d`` /
  ``force_unfused_topk`` and its ``topk_transform`` now delegates to
  ``DSATopKBackend``. ``DSATopKBackend`` is a closed Enum, so a plugin cannot add a
  member - the Kunlun fused topk stays an override of ``topk_transform``.
* ``DeepseekSparseAttnMultiStepBackend`` builds ``speculative_num_steps - 1``
  backends (0.5.8 built all of them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, List, Optional

import torch

import kunlun_ops
import xspeedgate_ops  # noqa: F401  # registers torch.ops.xspeedgate_ops

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.layers.attention.dsa.utils import (
    is_dsa_prefill_cp_in_seq_split,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
    DSAIndexerMetadata,
    DSAMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


def _paged_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    src_page_table: torch.Tensor,
    topk: int,
    cu_seqlens_q: Optional[torch.Tensor],
) -> torch.Tensor:
    """Pick the top-k scored positions and map them to page-1 cache slots.

    Torch port of the fused paged top-k transform. ``xspeedgate_ops.topk_transform``
    requires ``src_page_table`` to be at least as wide as the score, but on 0.5.14
    the score width is page aligned (``page_table_64.shape[1] * page_size``) while
    the page-1 table is exactly ``max_seqlen_k`` wide, so that contract no longer
    holds. Positions are returned in ascending order (the selected *set* is what
    the sparse attention consumes) with ``-1`` padding.
    """
    rows, width = score.shape
    device = score.device
    table_width = src_page_table.shape[1]

    positions = torch.arange(width, device=device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    k = min(topk, width)
    selected = score.masked_fill(~valid, float("-inf")).topk(k, dim=1).indices
    selected, _ = selected.sort(dim=1)
    selected_valid = valid.gather(1, selected)

    if src_page_table.shape[0] == rows:
        table = src_page_table
    else:
        # Prefill/verify feed one score row per query token while the page table
        # keeps one row per request; cu_seqlens_q carries that mapping.
        assert (
            cu_seqlens_q is not None
        ), "paged topk transform needs cu_seqlens_q to map tokens to requests"
        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
        row_index = torch.repeat_interleave(
            torch.arange(q_lens.shape[0], device=device), q_lens
        )
        table = src_page_table[row_index]

    slots = table.gather(1, selected.clamp(max=table_width - 1))
    slots = torch.where(selected_valid, slots, -torch.ones_like(slots))
    if k < topk:
        slots = torch.nn.functional.pad(slots, (0, topk - k), value=-1)
    return slots.to(torch.int32)


@dataclass(frozen=True)
class KunlunDSAMetadata(DSAMetadata):
    """DSAMetadata + Kunlun specific ``cum_q_lod`` and host mirrors.

    The Kunlun kernels take both the device and the host copy of the lod/seqlen
    tensors, so the host copies are materialized once per forward batch here
    instead of once per layer.
    """

    # Cumulative q lengths in *token* space (upstream ``cu_seqlens_q`` is in
    # expanded/per-token space for verify/draft-extend).
    cum_q_lod: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Cache the host mirrors used by the Kunlun kernels."""

        def _to_cpu(name: str, tensor: Optional[torch.Tensor]):
            object.__setattr__(self, name, None if tensor is None else tensor.cpu())

        _to_cpu("cache_seqlens_int32_cpu", self.cache_seqlens_int32)
        _to_cpu("cu_seqlens_q_cpu", self.cu_seqlens_q)
        _to_cpu("cu_seqlens_k_cpu", self.cu_seqlens_k)
        _to_cpu("dsa_seqlens_expanded_cpu", self.dsa_seqlens_expanded)
        _to_cpu("dsa_cu_seqlens_q_cpu", self.dsa_cu_seqlens_q)
        _to_cpu("dsa_cu_seqlens_k_cpu", self.dsa_cu_seqlens_k)
        _to_cpu("cum_q_lod_cpu", self.cum_q_lod)


@dataclass(frozen=True)
class KunlunDSAIndexerMetadata(DSAIndexerMetadata):
    """Indexer metadata exposing the host mirrors and the Kunlun topk kernel.

    ``attn_metadata`` is always a :class:`KunlunDSAMetadata` here; it is not
    re-declared because giving it a default would break the dataclass field
    ordering inherited from upstream.
    """

    def get_seqlens_expanded_cpu(self) -> torch.Tensor:
        """Host mirror of ``dsa_seqlens_expanded``."""
        return self.attn_metadata.dsa_seqlens_expanded_cpu

    def get_cu_seqlens_k_cpu(self) -> torch.Tensor:
        """Host mirror of ``cu_seqlens_k``."""
        return self.attn_metadata.cu_seqlens_k_cpu

    def get_cu_seqlens_q(self) -> torch.Tensor:
        """Device ``cu_seqlens_q``."""
        return self.attn_metadata.cu_seqlens_q

    def get_cu_seqlens_q_cpu(self) -> torch.Tensor:
        """Host mirror of ``cu_seqlens_q``."""
        return self.attn_metadata.cu_seqlens_q_cpu

    def get_seqlens_int32_cpu(self) -> torch.Tensor:
        """Host mirror of ``cache_seqlens_int32``."""
        return self.attn_metadata.cache_seqlens_int32_cpu

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        ke_offset: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Same control flow as upstream; the PAGED branch runs the Kunlun kernel.

        Upstream 0.5.14 delegates the whole body to ``DSATopKBackend``. That enum is
        closed, so the Kunlun fused kernel is injected here instead.
        """
        if topk_indices_offset_override is not None:
            cu_topk_indices_offset = topk_indices_offset_override
            cu_seqlens_q_topk = None
        elif cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            cu_seqlens_q_topk = compute_cu_seqlens(cu_seqlens_q)
            cu_topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q_topk[:-1],
                cu_seqlens_q,
            )
        else:
            cu_seqlens_q_topk = self.attn_metadata.cu_seqlens_q
            cu_topk_indices_offset = self.attn_metadata.topk_indices_offset

        if ke_offset is not None:
            seq_lens_topk = ke_offset
        else:
            seq_lens_topk = self.get_seqlens_expanded()

        if batch_idx_list is not None:
            page_table_size_1 = self.attn_metadata.page_table_1[batch_idx_list]
        else:
            page_table_size_1 = self.attn_metadata.page_table_1

        if not envs.SGLANG_DSA_FUSE_TOPK.get() or self.force_unfused_topk:
            return self.topk_backend.topk_func(
                logits, seq_lens_topk, topk, row_starts=ks
            )
        if self.topk_transform_method == TopkTransformMethod.PAGED:
            # if fused, we return a transformed page table directly
            return _paged_topk_transform(
                score=logits,
                lengths=seq_lens_topk,
                src_page_table=page_table_size_1,
                topk=topk,
                cu_seqlens_q=cu_seqlens_q_topk,
            )
        if self.topk_transform_method == TopkTransformMethod.RAGGED:
            from sgl_kernel import fast_topk_transform_ragged_fused

            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=seq_lens_topk,
                topk_indices_offset=cu_topk_indices_offset,
                topk=topk,
                row_starts=ks,
            )
        raise RuntimeError(f"Unsupported {self.topk_transform_method = }")


class KunlunDSAAttnBackend(DeepseekSparseAttnBackend):
    """DeepSeek sparse attention backend running on Kunlun XPU."""

    # ------------------------------------------------------------------ #
    # metadata
    # ------------------------------------------------------------------ #

    def _compute_cum_q_lod(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Cumulative q lengths in token space, per forward mode."""
        batch_size = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            return self.get_device_int32_arange(batch_size + 1)
        if forward_batch.forward_mode.is_target_verify():
            extend_seq_lens = torch.full(
                (batch_size,),
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            return compute_cu_seqlens(extend_seq_lens)
        # draft_extend (v1/v2) and plain extend
        assert forward_batch.extend_seq_lens is not None
        return compute_cu_seqlens(forward_batch.extend_seq_lens)

    def _cum_q_lod_for_graph(self, bs: int, forward_mode: ForwardMode) -> torch.Tensor:
        """cum_q_lod for the cuda-graph paths (fixed draft token count)."""
        if forward_mode.is_decode_or_idle():
            # NOTE: shares storage with cu_seqlens_q, so graph replay (which updates
            # cu_seqlens_q in place) keeps cum_q_lod in sync.
            return self.forward_metadata.cu_seqlens_q
        extend_seq_lens = torch.full(
            (bs,),
            self.speculative_num_draft_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        return compute_cu_seqlens(extend_seq_lens)

    @staticmethod
    def _to_kunlun_metadata(
        metadata: DSAMetadata, cum_q_lod: torch.Tensor
    ) -> KunlunDSAMetadata:
        """Re-wrap an upstream metadata object, sharing all tensors."""
        kwargs = {f.name: getattr(metadata, f.name) for f in fields(metadata)}
        kwargs["cum_q_lod"] = cum_q_lod
        return KunlunDSAMetadata(**kwargs)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        super().init_forward_metadata(forward_batch)
        self.forward_metadata = self._to_kunlun_metadata(
            self.forward_metadata, self._compute_cum_q_lod(forward_batch)
        )

    def _build_forward_metadata_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
        actual_forward_mode: Optional[ForwardMode] = None,
    ):
        """Capture-time metadata: upstream build + Kunlun ``cum_q_lod``."""
        super()._build_forward_metadata_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            seq_lens_cpu,
            forward_mode,
            spec_info,
            out_cache_loc=out_cache_loc,
            actual_forward_mode=actual_forward_mode,
        )
        self._rewrap_graph_metadata(bs, forward_mode)

    def _apply_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        out_cache_loc: Optional[torch.Tensor] = None,
        actual_forward_mode: Optional[ForwardMode] = None,
    ):
        """Shared capture+replay body: upstream apply + Kunlun ``cum_q_lod``."""
        super()._apply_cuda_graph_metadata(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_cpu,
            forward_mode,
            spec_info,
            out_cache_loc=out_cache_loc,
            actual_forward_mode=actual_forward_mode,
        )
        self._rewrap_graph_metadata(bs, forward_mode)

    def _rewrap_graph_metadata(self, bs: int, forward_mode: ForwardMode) -> None:
        """Upgrade ``self.forward_metadata`` (and the per-bs cache) in place."""
        if isinstance(self.forward_metadata, KunlunDSAMetadata):
            # already wrapped (apply() can run right after build())
            return
        metadata = self._to_kunlun_metadata(
            self.forward_metadata, self._cum_q_lod_for_graph(bs, forward_mode)
        )
        self.forward_metadata = metadata
        cache = getattr(self, "graph_metadata", None)
        if isinstance(cache, dict) and bs in cache:
            cache[bs] = metadata

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> KunlunDSAIndexerMetadata:
        """Indexer metadata carrying the Kunlun host mirrors."""
        force_unfused = (
            self.hisparse_coordinator is not None
            and forward_batch.forward_mode.is_decode_or_idle()
        )
        return KunlunDSAIndexerMetadata(
            attn_metadata=self.forward_metadata,
            topk_transform_method=self.get_topk_transform_method(
                forward_batch.forward_mode
            ),
            topk_backend=self.dsa_topk_backend,
            paged_mqa_schedule_metadata=self.forward_metadata.paged_mqa_schedule_metadata,
            paged_mqa_ctx_lens_2d=self.forward_metadata.paged_mqa_ctx_lens_2d,
            force_unfused_topk=force_unfused,
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prefill / verify / draft-extend through the Kunlun sparse kernel."""
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                self.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        metadata = self.forward_metadata
        causal = not layer.is_cross_attention
        assert causal, "DSA is causal only"

        # Use MHA kernel if in MHA_ONE_SHOT mode
        if self.use_mha:
            assert k is not None and v is not None
            assert q_rope is None, "MHA_ONE_SHOT path should not pass q_rope"
            assert (
                layer.tp_k_head_num == layer.tp_q_head_num > 1
            ), "MHA_ONE_SHOT requires dense multi-head config"
            return self._forward_standard_mha(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
            )

        # Do absorbed multi-latent attention (MLA path)
        assert q_rope is not None
        kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope = q_rope.view(-1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim)

        # Align topk_indices with q dimensions (q may be padded: TP + partial DP)
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        # here we use page size = 1
        topk_transform_method = self.get_topk_transform_method(
            forward_batch.forward_mode
        )
        if envs.SGLANG_DSA_FUSE_TOPK.get():
            page_table_1 = topk_indices
        else:
            if topk_transform_method == TopkTransformMethod.RAGGED:
                topk_indices_offset = metadata.topk_indices_offset
                assert topk_indices_offset is not None
                mask = topk_indices != -1
                topk_indices_offset = (
                    topk_indices_offset.unsqueeze(1)
                    if topk_indices_offset.ndim == 1
                    else topk_indices_offset
                )
                topk_indices = torch.where(
                    mask, topk_indices + topk_indices_offset, topk_indices
                )
            elif topk_transform_method == TopkTransformMethod.PAGED:
                assert metadata.dsa_extend_seq_lens_list is not None
                page_table_1 = transform_index_page_table_prefill(
                    page_table=metadata.page_table_1,
                    topk_indices=topk_indices,
                    extend_lens_cpu=metadata.dsa_extend_seq_lens_list,
                    page_size=1,
                )

        q_all = torch.cat([q_nope, q_rope], dim=-1)

        # sparse_prefill_fwd_opt takes explicit q/kv lods and needs preallocated
        # output buffers.
        o_ = torch.zeros(
            [q_all.shape[0], layer.tp_q_head_num, layer.v_head_dim],
            dtype=torch.bfloat16,
            device=q_all.device,
        )
        max_logits = torch.zeros(
            [q_all.shape[0], layer.tp_q_head_num],
            dtype=torch.float32,
            device=q_all.device,
        )
        lse = torch.zeros(
            [q_all.shape[0], layer.tp_q_head_num],
            dtype=torch.float32,
            device=q_all.device,
        )

        if (
            forward_batch.attn_cp_metadata is not None
            and is_dsa_prefill_cp_in_seq_split()
        ):
            # Only use attn_cp_metadata attributes in "in seq split" mode;
            # round-robin split is handled by the override below.
            # prefix_kv_len = KV from prior chunks (0 for first chunk, >0 for 2nd+),
            # because kv_len_prev/next only cover the current chunk's CP range.
            cp_metadata = forward_batch.attn_cp_metadata
            prefix_kv_len = 0
            if (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.seq_lens_cpu is not None
            ):
                prefix_kv_len = int(forward_batch.seq_lens_cpu[0].item()) - int(
                    sum(forward_batch.extend_seq_lens_cpu)
                )

            actual_seq_q_prev = cp_metadata.actual_seq_q_prev_list[0]
            kv_len_prev_total = prefix_kv_len + cp_metadata.kv_len_prev_list[0]
            kv_len_next_total = prefix_kv_len + cp_metadata.kv_len_next_list[0]

            # lod is a cumulative offset: the kernel derives per-batch length as
            # lod[i+1] - lod[i]; batch 0 is "prev", batch 1 is "next".
            cum_q_lod_cpu = torch.tensor(
                [0, actual_seq_q_prev, q_all.shape[0]], dtype=torch.int32
            )
            kv_lod_cpu = torch.tensor(
                [0, kv_len_prev_total, kv_len_prev_total + kv_len_next_total],
                dtype=torch.int32,
            )
            cum_q_lod = cum_q_lod_cpu.to(device=q_all.device)
            kv_lod = kv_lod_cpu.to(device=q_all.device)
        else:
            cum_q_lod = metadata.cu_seqlens_q
            cum_q_lod_cpu = metadata.cu_seqlens_q_cpu
            kv_lod = metadata.cu_seqlens_k
            kv_lod_cpu = metadata.cu_seqlens_k_cpu

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            cum_q_lod = metadata.cum_q_lod
            cum_q_lod_cpu = metadata.cum_q_lod_cpu

        if is_dsa_prefill_cp_round_robin_split():
            cum_q_lod = metadata.dsa_cu_seqlens_q
            cum_q_lod_cpu = metadata.dsa_cu_seqlens_q_cpu
            kv_lod = metadata.dsa_cu_seqlens_k
            kv_lod_cpu = metadata.dsa_cu_seqlens_k_cpu

        kunlun_ops.sparse_prefill_fwd_opt(
            q=q_all,
            kv=kv_cache.contiguous().view(-1, kv_cache.size(-1)),
            indices=page_table_1.unsqueeze(1),
            out=o_,
            max_logits=max_logits,
            lse=lse,
            sm_scale=layer.scaling,
            is_causal=True,
            qlod_cpu=cum_q_lod_cpu,
            qlod_xpu=cum_q_lod,
            kvlod_cpu=kv_lod_cpu,
            kvlod_xpu=kv_lod,
        )

        # Defensive: replace NaN with 0 (e.g. when kv_lod has 0-length padding)
        o_ = torch.nan_to_num(o_, nan=0.0)

        del max_logits
        del lse
        return o_

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode through the Kunlun sparse kernel."""
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                self.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        metadata = self.forward_metadata
        causal = not layer.is_cross_attention
        assert causal, "DSA is causal only"

        kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        assert q_rope is not None, "Kunlun DSA decode requires the absorbed MLA path"
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope = q_rope.view(-1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim)

        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        if envs.SGLANG_DSA_FUSE_TOPK.get():
            page_table_1 = topk_indices
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        q_all = torch.cat([q_nope, q_rope], dim=-1)

        # fwd_kvcache_mla wants a [tokens, 1, heads, head_dim] q and preallocated
        # output buffers.
        reshape_q = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
        bs = forward_batch.batch_size
        o_ = torch.zeros(
            [bs, reshape_q.shape[1], layer.tp_q_head_num, layer.v_head_dim],
            dtype=torch.bfloat16,
            device=q_all.device,
        )
        max_logits = torch.zeros(
            [bs, reshape_q.shape[1], layer.tp_q_head_num],
            dtype=torch.float32,
            device=q_all.device,
        )
        p_sums = torch.zeros(
            [bs, reshape_q.shape[1], layer.tp_q_head_num],
            dtype=torch.float32,
            device=q_all.device,
        )
        kunlun_ops.fwd_kvcache_mla(
            q_c=reshape_q,
            kv_cache=kv_cache.view(-1, self.kv_cache_dim),
            indices=page_table_1.view(bs, reshape_q.shape[1], page_table_1.shape[-1]),
            out=o_,
            max_logits=max_logits,
            p_sums=p_sums,
            softmax_scale=layer.scaling,
            kv_lod_cpu=metadata.cache_seqlens_int32_cpu,
            kv_lod_xpu=metadata.cache_seqlens_int32,
            max_seq_kv=metadata.max_seq_len_k,
        )
        del max_logits
        del p_sums
        return o_

    def _forward_standard_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: KunlunDSAMetadata,
    ) -> torch.Tensor:
        """Dense MHA_ONE_SHOT path via ``kunlun_ops.attention``."""
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        # MHA_ONE_SHOT: k/v include all tokens (prefix + current)
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        cu_seqlens_q_cpu = metadata.cu_seqlens_q_cpu
        cu_seqlens_k_cpu = metadata.cu_seqlens_k_cpu

        assert len(cu_seqlens_q) == len(cu_seqlens_k), (
            f"batch_size mismatch: cu_seqlens_q has {len(cu_seqlens_q)-1} requests, "
            f"cu_seqlens_k has {len(cu_seqlens_k)-1} requests"
        )

        o_ = torch.empty(
            [q.shape[0], layer.tp_q_head_num, layer.v_head_dim],
            dtype=torch.bfloat16,
            device=q.device,
        )
        ds_alpha = layer.scaling * math.sqrt(layer.head_dim)
        # the kernel requires an lse buffer even though we drop it here
        softmax_lse = torch.full(
            (layer.tp_q_head_num, q.size(0)),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        kunlun_ops.attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            o_,
            is_causal=True,
            is_prefill=True,
            prefill_len=0,
            k_perchannel_scale=None,
            v_perchannel_scale=None,
            smooth=None,
            context_seq_lod_cpu=cu_seqlens_q_cpu,
            context_seq_lod_xpu=cu_seqlens_q,
            slot_mapping_cpu=None,
            slot_mapping_xpu=None,
            context_kvlen_lod_cpu=cu_seqlens_k_cpu,
            context_kvlen_lod_xpu=cu_seqlens_k,
            v_trans=False,
            v_trans_threshold=0,
            alpha=ds_alpha,
            softmax_lse=softmax_lse,
            unpadded_lse=True,
        )
        del softmax_lse
        return o_


class KunlunDSAMultiStepBackend(DeepseekSparseAttnMultiStepBackend):
    """Multi-step (MTP) wrapper around :class:`KunlunDSAAttnBackend`.

    NOTE: 0.5.14 builds ``speculative_num_steps - 1`` backends (0.5.8 built all
    ``speculative_num_steps``); this follows upstream.
    """

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                KunlunDSAAttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )
