# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlun subclass of ``sglang.srt.layers.attention.deepseek_v4_backend``.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional, Tuple, List

import cocopod  # noqa: F401  (registers torch.ops.xspeedgate_ops.*)
import kunlun_ops
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
    DSV4AttnMetadata,
    _pad_tensor_to_size,
)
from sglang.srt.layers.attention.dsv4 import indexer as _upstream_indexer
from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
    DIM_NOPE,
    DIM_ROPE,
    NOPE_ROPE_BYTES,
    NUM_SCALE_TILES,
    PADDED_SCALE_PER_TOKEN,
    TILE_SIZE,
    fp8_dtype,
)
try:
    from sglang.srt.layers.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
except ImportError:  # community main renamed it (layout-specific variants)
    from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged_dsv4 as dequantize_k_cache_paged,
    )
from sglang.srt.layers.attention.dsv4.metadata import (
    _LARGE_INDEXER_QUERY_THRESHOLD,
    PagedIndexerMetadata,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
)

# Internal-fork only: bf16 KV-cache sparse-prefill gather.  Absent on v0.5.x,
# where the sparse-prefill workspace is always fp8-dequantized.
try:
    from sglang.srt.internal.layers.attention.dsv4.gather_bf16_k_cache import (
        gather_bf16_k_cache_paged,
    )
except Exception:  # noqa: BLE001
    gather_bf16_k_cache_paged = None
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Indexer-logits bridge.
#
# The internal ``forward_c4_indexer`` selects its logits implementation from an
# env-driven ladder and calls it as a *module-level* function.  On XPU every
# rung of that ladder is either unavailable or Triton-only, so ``kernel_ops``
# registers a replacement for ``fp8_paged_mqa_logits_torch`` (the rung selected
# when ``SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`` is set) which forwards to the
# Kunlun-native implementation on the backend instance that is running.
# ``forward_c4_indexer`` below publishes that instance + batch here.
# ---------------------------------------------------------------------------
_ACTIVE_C4_BACKEND = None
_ACTIVE_C4_FORWARD_BATCH = None


def kunlun_fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata=None,
    max_seq_len=None,
    clean_logits: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Kunlun replacement for the internal indexer-logits torch fallback.

    Signature matches ``sglang.srt.layers.attention.dsv4.indexer.
    fp8_paged_mqa_logits_torch`` (positional call site), so it drops the
    deep-gemm metadata / ``clean_logits`` arguments and routes to
    :meth:`_kunlun_c4a_paged_mqa_logits`.
    """
    backend = _ACTIVE_C4_BACKEND
    forward_batch = _ACTIVE_C4_FORWARD_BATCH
    assert backend is not None and forward_batch is not None, (
        "kunlun_fp8_paged_mqa_logits_torch called outside "
        "KunlunDeepseekV4AttnBackend.forward_c4_indexer"
    )
    return backend._kunlun_c4a_paged_mqa_logits(
        q_fp8=q_fp8,
        kvcache_fp8=kvcache_fp8,
        weight=weight,
        seq_lens=seq_lens,
        page_table=page_table,
        max_seq_len=int(max_seq_len),
        forward_batch=forward_batch,
    )

_HEAD_DIM = DIM_NOPE + DIM_ROPE  # 512
_E4M3_LUT = None


def _e4m3_lut(device):
    global _E4M3_LUT
    if _E4M3_LUT is None or _E4M3_LUT.device != device:
        lut_cpu = torch.arange(256, dtype=torch.uint8).view(fp8_dtype).to(torch.float32)
        _E4M3_LUT = lut_cpu.to(device)
    return _E4M3_LUT


def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:
    return cache.reshape(-1, cache.shape[-1]).contiguous()


class KunlunDeepseekV4AttnBackend(DeepseekV4AttnBackend):
    """KunlunDeepseekV4AttnBackend
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def is_bf16_kv_cache(self):
        """Community main dropped this cached flag; derive it from the pool."""
        pool = getattr(self, "token_to_kv_pool", None)
        dtype = getattr(pool, "dtype", None)
        if dtype is None:
            dtype = getattr(pool, "store_dtype", None)
        return dtype in (torch.bfloat16, torch.float16)

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer,
        forward_batch,
        alt_streams=None,
        enable_multi_stream: bool = False,
        q_lora_ready=None,
        skip_compressor: bool = False,
        prev_topk_indices=None,
        skip_topk: bool = False,
        return_topk_indices: bool = False,
        moa_hybrid_mask=None,
    ):
        """Delegate to the internal implementation, bridging the logits kernel.

        The internal ``C4IndexerBackendMixin.forward_c4_indexer`` owns the whole
        topk / IndexCache / HiSparse / capturer tail -- and, on XPU, its
        ``topk_transform_paged`` already dispatches to ``sgl_kernel``.  The only
        piece that needs a Kunlun kernel is the indexer-logits function the
        method resolves from its env-selected ladder at call time.  Stash the
        batch and backend so the module-level replacement registered for
        ``fp8_paged_mqa_logits_torch`` can reach them, then run the internal
        method unchanged.
        """
        global _ACTIVE_C4_BACKEND, _ACTIVE_C4_FORWARD_BATCH
        _ACTIVE_C4_BACKEND = self
        _ACTIVE_C4_FORWARD_BATCH = forward_batch
        import inspect

        kwargs = dict(
            x=x,
            q_lora=q_lora,
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            alt_streams=alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
            skip_compressor=skip_compressor,
            prev_topk_indices=prev_topk_indices,
            skip_topk=skip_topk,
            return_topk_indices=return_topk_indices,
            moa_hybrid_mask=moa_hybrid_mask,
        )
        params = inspect.signature(super().forward_c4_indexer).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        try:
            return super().forward_c4_indexer(**kwargs)
        finally:
            _ACTIVE_C4_BACKEND = None
            _ACTIVE_C4_FORWARD_BATCH = None

    def _c4a_cp_extend_logits_torch(
        self,
        q_fp8: torch.Tensor,
        kvcache_fp8: torch.Tensor,
        weight: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Indexer logits for the NSA prefill context-parallel extend path.

        The kunlun ragged kernel is driven by a per-request q-lod, but under prefill
        context parallelism a rank owns a strided subset of the sequence's queries, so
        the request's extend length does not describe the local query rows; the paged
        kernel in turn returns all-zero scores for them. Zero scores make the
        downstream 512-topk keep the *oldest* 512 compressed states, which silently
        truncates the context of any prompt longer than ``topk * compress_ratio``
        tokens and degenerates the output. Compute the scores directly instead:

            score[q, k] = sum_h weight[q, h] * relu(q[q, h] . k[k]) * k_scale[k]

        which is the same formula as the reference paged fallback, evaluated for the
        local rows only. Chunked over queries to bound the temporary [chunk, H, K].
        """
        B, _sq, H, D = q_fp8.shape
        device = q_fp8.device
        scale_offset = block_size * D
        total_dim = block_size * (D + 4)

        lens = seq_lens.to(torch.int32)
        max_len = int(lens.max().item())
        if max_len <= 0:
            return torch.zeros((B, max_seq_len), dtype=torch.float32, device=device)

        num_pages = (max_len + block_size - 1) // block_size
        num_pages = min(num_pages, page_table.shape[1])
        row = int(lens.argmax().item())
        pages = page_table[row, :num_pages].to(torch.int64).clamp(min=0)

        kv_flat = kvcache_fp8.reshape(-1, total_dim)
        gathered = kv_flat[pages]
        k_int8 = (
            gathered[:, :scale_offset].reshape(-1, D)
            if gathered.dtype == torch.int8
            else gathered[:, :scale_offset].contiguous().view(torch.int8).reshape(-1, D)
        )
        k_scale = (
            gathered[:, scale_offset:].contiguous().view(torch.float32).reshape(-1)
        )
        # Only the first ``max_len`` compressed states of the page range are written
        # by this request; the rest of the last page is whatever the pool happens to
        # hold (uninitialised garbage right after start-up). Never read past the
        # valid length, and accumulate the dot products in fp32: int8 keys and
        # queries reach |q.k| ~ 127*127*D, far above the fp16 range, so an fp16
        # matmul saturates to +inf and the weighted sum below turns it into NaN,
        # which then poisons the whole attention output.
        K = min(k_int8.shape[0], num_pages * block_size, max_len)
        k_f = k_int8[:K].to(torch.float32)
        k_scale = k_scale[:K].to(torch.float32)

        q_int8 = q_fp8[:, 0]
        q_f = (q_int8 if q_int8.dtype == torch.int8 else q_int8.view(torch.int8)).to(
            torch.float32
        )

        w = weight.to(torch.float32)
        logits = torch.zeros((B, K), dtype=torch.float32, device=device)
        chunk = int(os.environ.get("KL_INDEXER_CHUNK", "128"))
        for beg in range(0, B, chunk):
            end = min(beg + chunk, B)
            dots = torch.relu(torch.matmul(q_f[beg:end], k_f.t()))  # [c, H, K]
            logits[beg:end] = torch.einsum("chk,ch->ck", dots, w[beg:end])
        logits *= k_scale.unsqueeze(0)

        positions = torch.arange(K, device=device).unsqueeze(0)
        logits = logits.masked_fill(positions >= lens.unsqueeze(1), 0.0)

        if K < max_seq_len:
            logits = F.pad(logits, (0, max_seq_len - K), value=0.0)
        else:
            logits = logits[:, :max_seq_len]
        return logits

    def _kunlun_c4a_paged_mqa_logits(
        self,
        *,
        q_fp8: torch.Tensor,
        kvcache_fp8: torch.Tensor,
        weight: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        forward_batch,
    ) -> torch.Tensor:
        """Indexer paged-mqa-logits routed to Kunlun native ops.

        Inlined ``fp8_paged_mqa_logits_torch`` replacement. Dispatch by ForwardMode:
        * decode / target_verify → ``kunlun_ops.c4a_paged_mqa_logits_with_mixed_cache``
          (paged KV, no host-side gather, CUDA-Graph safe)
        * extend / prefill       → gather KV contiguous + ``kunlun_ops.c4a_mqa_logits``
        """
        _COMPRESS_RATIO = 4
        _BLOCK_SIZE = 64

        logical_mode = getattr(
            forward_batch, "actual_forward_mode", forward_batch.forward_mode
        )
        is_extend = logical_mode.is_extend_without_speculative()
        # decode / target_verify / draft_extend / idle all use the paged op
        is_decode = not is_extend
        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)

        # Normalise seq_lens to 1-D.
        _seq_lens = seq_lens.squeeze(-1) if seq_lens.dim() == 2 else seq_lens

        B, _sq, H, D = q_fp8.shape
        assert D == 128 and kvcache_fp8.shape[1] == _BLOCK_SIZE and _sq == 1

        total_dim = _BLOCK_SIZE * (D + 4)

        # NSA prefill context parallelism hands this rank a strided subset of the
        # sequence's queries, so the request's extend length is not the number of
        # local query rows and the ragged (q-lod based) logits kernel cannot describe
        # it: it would address rows outside the logits buffer, leaving most queries
        # with all-zero scores so topk keeps only the oldest ``topk`` compressed
        # states. Compute the scores for the local rows directly instead
        # (see _c4a_cp_extend_logits_torch).
        _cp_local = False
        if not is_decode and extend_seq_lens_cpu is not None:
            if len(extend_seq_lens_cpu) > 0:
                _cp_local = int(sum(int(v) for v in extend_seq_lens_cpu)) != B

        if _cp_local:
            return self._c4a_cp_extend_logits_torch(
                q_fp8,
                kvcache_fp8,
                weight,
                _seq_lens,
                page_table,
                max_seq_len,
                block_size=_BLOCK_SIZE,
            )

        if is_decode:
            # ── Decode / Target-Verify: paged op with mixed cache layout ──
            # kunlun_ops 0.1.203+ provides ``c4a_paged_mqa_logits_with_mixed_cache``
            # which accepts the mixed KV page buffer directly (K + scale
            # interleaved per page); no host-side gather needed.
            kv_cache_flat = kvcache_fp8.reshape(-1, total_dim)

            c4_seq_lens = _seq_lens.to(torch.int32).clamp(0, max_seq_len)
            context_lens_xpu = c4_seq_lens * _COMPRESS_RATIO
            context_lens_cpu = context_lens_xpu.to("cpu", non_blocking=False)
            qlod_cpu = torch.arange(B + 1, dtype=torch.int32)
            qlod_xpu = qlod_cpu.to(q_fp8.device, non_blocking=False)
            block_table_i32 = (
                page_table.to(torch.int32)
                if page_table.dtype != torch.int32
                else page_table
            )
            q_call = q_fp8.view(dtype=torch.int8) if q_fp8.dtype != torch.int8 else q_fp8

            logits = torch.empty(
                (B, 1, max_seq_len), dtype=torch.float32, device=q_fp8.device
            )

            kunlun_ops.c4a_paged_mqa_logits_with_mixed_cache(
                q=q_call,  # [B, 1, num_heads, head_dim] int8
                weight=weight.unsqueeze(1),  # [B, 1, num_heads] float32
                k_cache=kv_cache_flat,  # mixed page buffer
                logits=logits,  # [B, 1, max_seq_len]
                max_context_len=max_seq_len * _COMPRESS_RATIO,
                qlod_cpu=qlod_cpu,
                qlod_xpu=qlod_xpu,
                context_lens_cpu=context_lens_cpu,
                context_lens_xpu=context_lens_xpu,
                block_table=block_table_i32,
                compress_ratio=_COMPRESS_RATIO,
                clean_logits=True,
            )
            return logits.squeeze(1)

        # ── Extend / Prefill: gather KV contiguous, then c4a_mqa_logits ─
        if extend_seq_lens_cpu is not None and len(extend_seq_lens_cpu) > 0:
            extend_lens = torch.as_tensor(extend_seq_lens_cpu, dtype=torch.int32)
            cum = torch.zeros(len(extend_seq_lens_cpu) + 1, dtype=torch.int64)
            cum[1:] = torch.cumsum(extend_lens.long(), dim=0)
            last_idx = (cum[1:] - 1).clamp(0, _seq_lens.shape[0] - 1)
            per_req_k_lens = _seq_lens[last_idx].to(torch.int32).cpu()
            num_requests = len(extend_seq_lens_cpu)
            qlod_cpu = torch.zeros(num_requests + 1, dtype=torch.int32)
            qlod_cpu[1:] = torch.cumsum(extend_lens, dim=0)
            max_seq_q = int(extend_lens.max().item())
            if int(qlod_cpu[-1].item()) != B:
                # NSA prefill context parallelism gives this rank only a strided
                # subset of the sequence's queries, so the request's extend length is
                # not the number of local query rows. Feeding the full length as the
                # q-lod makes the logits kernel address rows that do not exist: it
                # writes inf into part of the buffer and leaves the tail untouched, so
                # every query past ~topk compressed states keeps all-zero scores and
                # topk degenerates to "the oldest 512 states". Describe the local rows
                # instead and let the per-query mask in the topk transform apply
                # causality (which it does anyway), so drop the kernel-side mask.
                per_req_k_lens = _seq_lens.to(torch.int32).cpu().max().reshape(1)
                num_requests = 1
                qlod_cpu = torch.tensor([0, B], dtype=torch.int32)
                max_seq_q = B
                is_causal_logits = False
            else:
                is_causal_logits = True
        else:
            per_req_k_lens = _seq_lens.to(torch.int32).cpu()
            num_requests = B
            qlod_cpu = torch.arange(num_requests + 1, dtype=torch.int32)
            max_seq_q = 1
            is_causal_logits = True

        kv_cache_flat = kvcache_fp8.reshape(-1, total_dim)
        SCALE_OFFSET = _BLOCK_SIZE * D
        k_data_flat = kv_cache_flat[:, :SCALE_OFFSET].view(-1, _BLOCK_SIZE, D)
        scale_flat = (
            kv_cache_flat[:, SCALE_OFFSET:]
            .contiguous()
            .view(dtype=torch.float32)
            .reshape(-1, _BLOCK_SIZE)
        )

        total_k = int(per_req_k_lens.sum().item())
        if total_k == 0:
            return torch.zeros(
                (B, max_seq_len), dtype=torch.float32, device=q_fp8.device
            )

        k_contiguous = torch.empty((total_k, D), dtype=torch.int8, device=q_fp8.device)
        k_scale_contiguous = torch.empty(
            total_k, dtype=torch.float32, device=q_fp8.device
        )
        max_seq_k_compressed = 0
        offset = 0
        pt_rows = (qlod_cpu[1:] - 1).clamp(0, page_table.shape[0] - 1).tolist()
        for i in range(num_requests):
            seq_len_i = int(per_req_k_lens[i].item())
            if seq_len_i == 0:
                continue
            max_seq_k_compressed = max(max_seq_k_compressed, seq_len_i)
            num_blocks = (seq_len_i + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            pt_row = min(int(pt_rows[i]), page_table.shape[0] - 1)
            pages = page_table[pt_row, :num_blocks].long()
            k_pages = k_data_flat[pages].reshape(-1, D)[:seq_len_i]
            k_contiguous[offset : offset + seq_len_i] = k_pages.to(torch.int8)
            k_scale_contiguous[offset : offset + seq_len_i] = scale_flat[pages].reshape(
                -1
            )[:seq_len_i]
            offset += seq_len_i

        pre_k_lens = per_req_k_lens * _COMPRESS_RATIO
        klod_cpu = torch.zeros(num_requests + 1, dtype=torch.int32)
        klod_cpu[1:] = torch.cumsum(pre_k_lens, dim=0)
        klod_xpu = klod_cpu.to(q_fp8.device, non_blocking=False)
        qlod_xpu = qlod_cpu.to(q_fp8.device, non_blocking=False)
        com_k_start_cpu = torch.zeros(num_requests, dtype=torch.int32)
        com_k_start_xpu = com_k_start_cpu.to(q_fp8.device, non_blocking=False)

        logits = torch.zeros(
            (B, max_seq_k_compressed), dtype=torch.float32, device=q_fp8.device
        )
        # c4a_mqa_logits requires q dtype int8; upstream quant is already
        # int8 on Kunlun, defensive view for safety.
        q_2d = q_fp8[:, 0]  # [B, H, D]
        if q_2d.dtype != torch.int8:
            q_2d = q_2d.view(dtype=torch.int8)

        kunlun_ops.c4a_mqa_logits(
            q=q_2d,
            weight=weight,
            k=k_contiguous,
            k_scale=k_scale_contiguous,
            logits=logits,
            max_seq_q=max_seq_q,
            max_seq_k=max_seq_k_compressed * _COMPRESS_RATIO,
            qlod_cpu=qlod_cpu,
            qlod_xpu=qlod_xpu,
            klod_cpu=klod_cpu,
            klod_xpu=klod_xpu,
            com_k_start_cpu=com_k_start_cpu,
            com_k_start_xpu=com_k_start_xpu,
            is_causal=is_causal_logits,
            compress_ratio=_COMPRESS_RATIO,
            clean_logits=True,
        )
        if max_seq_k_compressed < max_seq_len:
            logits = F.pad(logits, (0, max_seq_len - max_seq_k_compressed), value=0.0)
        else:
            logits = logits[:, :max_seq_len]
        return logits

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs=None):
        """No-op: DSV4 has no draft-dependent verify buffers to refresh."""
        return None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ):
        """Upstream ``DeepseekV4AttnBackend.forward`` with the Kunlun attention
        """
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        core_attn_metadata = self.forward_metadata.core_attn_metadata
        token_to_kv_pool = self.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if not isinstance(core_attn_metadata, DSV4AttnMetadata):
            raise NotImplementedError("ragged attention")

        if save_kv_cache:
            self.store_cache(layer_id, swa_k, forward_batch)
        swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

        extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
        if compress_ratio == 4:
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_indices = core_attn_metadata.c4_sparse_page_indices
            extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
        elif compress_ratio == 128:
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_indices = core_attn_metadata.c128_page_indices
            extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

        swa_window_size = token_to_kv_pool.swa_page_size
        assert swa_k_cache.ndim == 2
        k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim
        # The internal pool stores even the bf16 layout in a uint8 byte buffer
        # (1024 B/token = 512 bf16); present it as bf16 so the native op sees a
        # cache whose dtype matches q.  The fp8 layout has no uniform view and
        # is handled by the torch paths instead.
        if (
            swa_k_cache.dtype == torch.uint8
            and k_cache_total_dim == 512 * 2
        ):
            swa_k_cache = swa_k_cache.view(torch.bfloat16)
            k_cache_total_dim = k_cache_total_dim // 2
        swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
            swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
        )
        if extra_k_cache is not None:
            page_sizes = {
                4: token_to_kv_pool.page_size // 4,
                128: token_to_kv_pool.page_size // 128,
            }
            # Same uint8 byte-buffer / bf16-layout situation as the SWA cache.
            if (
                extra_k_cache.dtype == torch.uint8
                and token_to_kv_pool.swa_kv_pool.kv_cache_total_dim == 512 * 2
            ):
                extra_k_cache = extra_k_cache.view(torch.bfloat16)
            extra_k_cache = extra_k_cache[
                :, : page_sizes[compress_ratio] * k_cache_total_dim
            ].view(
                extra_k_cache.shape[0],
                page_sizes[compress_ratio],
                1,
                k_cache_total_dim,
            )
        swa_page_indices = core_attn_metadata.swa_page_indices
        swa_topk_lengths = core_attn_metadata.swa_topk_lengths

        def match_num_queries(x, value):
            if x is None or x.shape[0] == q.shape[0]:
                return x
            if x.shape[0] > q.shape[0]:
                return x[: q.shape[0]]
            return _pad_tensor_to_size(x, q.shape[0], value=value)

        swa_page_indices = match_num_queries(swa_page_indices, value=0)
        swa_topk_lengths = match_num_queries(swa_topk_lengths, value=1)
        extra_indices = match_num_queries(extra_indices, value=-1)
        extra_topk_lengths = match_num_queries(extra_topk_lengths, value=1)

        if q.ndim == 3:
            q = q.unsqueeze(1)
        if swa_page_indices.ndim == 2:
            swa_page_indices = swa_page_indices.unsqueeze(1)
        if extra_indices is not None and extra_indices.ndim == 2:
            extra_indices = extra_indices.unsqueeze(1)

        assert attn_sink is not None
        # NOTE: upstream also fetches ``get_flashmla_metadata(compress_ratio)``
        # here to feed the kernel's tile scheduler. The Kunlun ops take no such
        # argument and the accessor is side-effect free, so it is dropped.
        assert (
            swa_page_indices.shape[-1] % 64 == 0
        ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
        if extra_indices is not None:
            assert (
                extra_indices.shape[-1] % 64 == 0
            ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

        if forward_batch.forward_mode.is_extend_without_speculative() and (
            q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD
            or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
        ):
            return self._forward_prefill_sparse(
                q=q,
                layer_id=layer_id,
                compress_ratio=compress_ratio,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                core_attn_metadata=core_attn_metadata,
                attn_sink=attn_sink,
            )
        # ---- inlined ``flash_mla_with_kvcache`` ---------------------------
        logical_mode = getattr(
            forward_batch, "actual_forward_mode", forward_batch.forward_mode
        )
        is_extend = logical_mode.is_extend_without_speculative()
        # Only DECODE / TARGET_VERIFY / EXTEND can drive the Kunlun native op
        # Everything else (e.g. draft-extend) uses the pure-torch
        # reference path below.
        if logical_mode.is_decode() or logical_mode.is_target_verify() or is_extend:
            return self._kunlun_compressed_attention(
                q=q,
                logical_mode=logical_mode,
                swa_k_cache=swa_k_cache,
                swa_page_indices=swa_page_indices,
                swa_window_size=swa_window_size,
                extra_k_cache=extra_k_cache,
                extra_indices=extra_indices,
                compress_ratio=compress_ratio,
                core_attn_metadata=core_attn_metadata,
                attn_sink=attn_sink,
                forward_batch=forward_batch,
            )
        else:
            # Pure-torch reference for the modes the native op cannot serve.
            n_q, _s_q, h_q, d_qk = q.shape
            q_flat = q.reshape(n_q, h_q, d_qk)
            keys, valid = self._dequant_sparse_keys(
                swa_k_cache, swa_page_indices.reshape(n_q, -1), swa_topk_lengths
            )
            if extra_k_cache is not None and extra_indices is not None:
                extra_keys, extra_valid = self._dequant_sparse_keys(
                    extra_k_cache, extra_indices.reshape(n_q, -1), extra_topk_lengths
                )
                keys = torch.cat([keys, extra_keys], dim=1)
                valid = torch.cat([valid, extra_valid], dim=1)
            return self._sparse_mla_attention(
                q_flat, keys, valid, attn_sink, self.softmax_scale, self.head_dim_v
            )

    def _kunlun_compressed_attention(
        self,
        q: torch.Tensor,
        logical_mode,
        swa_k_cache: torch.Tensor,
        swa_page_indices: torch.Tensor,
        swa_window_size: int,
        extra_k_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        compress_ratio: Literal[0, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
        forward_batch=None,
    ) -> torch.Tensor:
        """Inlined ``flash_mla_with_kvcache``: build the qlod / kvseqlen context
        the retired ContextVar wrapper used to publish, then call the Kunlun op.

        ``q`` is ``(n_q, 1, h_q, d_qk)`` and the result is
        ``(n_q, h_q, head_dim_v)``.
        """
        is_extend = logical_mode.is_extend_without_speculative()
        if is_extend:
            from sglang.srt.layers.attention.dsa.utils import (
                is_dsa_prefill_cp_round_robin_split,
            )

            is_causal = not is_dsa_prefill_cp_round_robin_split()
        else:
            is_causal = True
        # CUDA graph replays only the recorded kernels, so any Python scalar
        # derived from buffer contents here would freeze at capture time.
        is_cuda_graph_mode = logical_mode.is_decode() or logical_mode.is_target_verify()

        seq_lens = core_attn_metadata.seq_lens_casual[: q.shape[0]]
        qlod_cpu = torch.arange(seq_lens.shape[0] + 1, dtype=torch.int32)
        qlod_xpu = qlod_cpu.to(q.device, non_blocking=False)
        kvseqlen_xpu = seq_lens.to(dtype=torch.int32, device=q.device)
        # Host copy of the sequence lengths.  ``.to("cpu")`` returns uninitialised
        # memory during CUDA-graph capture, so prefer the batch's CPU mirror: the
        # decode graph runner maintains ``seq_lens_cpu`` in its own buffers and
        # refreshes it before every replay, and handing the op that very buffer
        # keeps the values live across replays.
        _n = kvseqlen_xpu.numel()
        if torch.cuda.is_current_stream_capturing():
            # Inside CUDA-graph capture the batch's CPU mirror aliases other
            # pooled buffers (share_buffers() coalesces seq_lens / seq_lens_cpu)
            # and reads back uninitialised float data, while a device->host copy
            # is illegal.  Every request of a capture batch carries the backend's
            # capture fill length, so use that; the tensor must outlive the graph
            # because the captured launch bakes its address.
            _fill = int(self.get_cuda_graph_seq_len_fill_value())
            _cache = getattr(self, "_kunlun_capture_kvseqlen_cpu", None)
            if _cache is None or _cache.numel() != _n or int(_cache[0]) != _fill:
                _cache = torch.full((_n,), _fill, dtype=torch.int32)
                self._kunlun_capture_kvseqlen_cpu = _cache
            kvseqlen_cpu = _cache
            _seq_cpu = None
        else:
            _seq_cpu = (
                getattr(forward_batch, "seq_lens_cpu", None)
                if forward_batch is not None
                else None
            )
            if _seq_cpu is not None and len(_seq_cpu) >= _n:
                _seq_cpu = _seq_cpu[:_n]
                kvseqlen_cpu = (
                    _seq_cpu
                    if _seq_cpu.dtype == torch.int32
                    else _seq_cpu.to(torch.int32)
                )
            else:
                kvseqlen_cpu = kvseqlen_xpu.to("cpu", non_blocking=False)

        if os.environ.get("KUNLUN_DEBUG_KVSEQLEN"):
            import sys as _sys

            _flat = kvseqlen_cpu.reshape(-1)
            print(
                "[kdebug] compressed_attention "
                f"fb={'yes' if forward_batch is not None else 'None'} "
                f"mirror={'yes' if _seq_cpu is not None else 'None'} "
                f"n={_flat.numel()} min={int(_flat.min())} max={int(_flat.max())} "
                f"capturing={torch.cuda.is_current_stream_capturing()} "
                f"mirror_dev={getattr(_seq_cpu, 'device', None)} "
                f"mirror_dtype={getattr(_seq_cpu, 'dtype', None)} "
                f"cpu_dev={kvseqlen_cpu.device} "
                f"mode={getattr(getattr(forward_batch, 'forward_mode', None), 'name', None)} "
                f"head={_flat[:4].tolist()}",
                file=_sys.stderr,
                flush=True,
            )

        q_dtype = q.dtype
        win_cache = _flatten_cache(swa_k_cache)
        q_3d = q.squeeze(1)
        # The internal fork's SWA pool is a *byte* buffer (uint8) in both the
        # bf16 and fp8 layouts, so matching q to the storage dtype would produce
        # a uint8 q the op rejects.  Keep q in its compute dtype and let the op
        # read the cache; only mirror the dtype for real element-typed caches.
        if win_cache.dtype not in (torch.uint8, torch.int8) and (
            q_3d.dtype != win_cache.dtype
        ):
            q_3d = q_3d.to(win_cache.dtype)
        q_3d = q_3d.contiguous()
        win_indices = swa_page_indices.squeeze(1).contiguous()
        win_size = min(swa_window_size, win_indices.shape[1])
        win_indices = win_indices[:, :win_size].contiguous()
        if extra_k_cache is None or extra_indices is None:
            com_cache = win_cache.new_empty((0, win_cache.shape[1]))
            com_indices = win_indices.new_empty((win_indices.shape[0], 0))
            com_ratio = 1
            com_topk = 0
        else:
            com_cache = _flatten_cache(extra_k_cache)
            com_indices = extra_indices.squeeze(1).contiguous()
            com_ratio = compress_ratio
            if is_cuda_graph_mode:
                max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
            else:
                max_seq_len = int(kvseqlen_cpu.max().item())
            com_topk = max(max_seq_len // com_ratio, 1)
            com_topk = min(com_topk, com_indices.shape[1])
            com_indices = com_indices[:, :com_topk].contiguous()

        # NSA prefill context parallelism leaves garbage (e.g. -8388608) in the
        # index rows of padded / out-of-window tokens. The Kunlun op only
        # understands -1 as "no key"; anything else out of range makes it read
        # unrelated cache slots, so the result would depend on the KV placement
        # of the request. Normalise out-of-range entries to -1.
        win_indices = torch.where(
            (win_indices >= 0) & (win_indices < win_cache.shape[0]),
            win_indices,
            torch.full_like(win_indices, -1),
        ).contiguous()
        if com_indices.numel() and com_cache.shape[0] > 0:
            com_indices = torch.where(
                (com_indices >= 0) & (com_indices < com_cache.shape[0]),
                com_indices,
                torch.full_like(com_indices, -1),
            ).contiguous()

        out = torch.empty(
            (q_3d.shape[0], q_3d.shape[1], self.head_dim_v),
            dtype=q_3d.dtype,
            device=q.device,
        )
        max_logits = torch.empty(q_3d.shape[:2], dtype=torch.float32, device=q.device)
        lse = torch.empty_like(max_logits)
        if q_3d.shape[1] == 64:
            torch.ops.xspeedgate_ops.compressed_attention(
                q_3d,
                win_cache,
                win_indices,
                com_cache,
                com_indices,
                out,
                max_logits,
                lse,
                qlod_cpu.contiguous(),
                qlod_xpu.contiguous(),
                kvseqlen_cpu.contiguous(),
                kvseqlen_xpu.contiguous(),
                self.softmax_scale,
                is_causal,
                win_size,
                com_ratio,
                com_topk,
                attn_sink.contiguous(),
            )
        else:
            kunlun_ops.hybrid_attention(
                q=q_3d,
                win_kv_cache=win_cache,
                win_indices=win_indices,
                com_kv_cache=com_cache,
                com_indices=com_indices,
                o=out,
                max_logits=max_logits,
                lse=lse,
                qlod_cpu=qlod_cpu,
                qlod_xpu=qlod_xpu,
                kvseqlen_cpu=kvseqlen_cpu,
                kvseqlen_xpu=kvseqlen_xpu,
                sm_scale=self.softmax_scale,
                is_causal=is_causal,
                max_window_size=win_size,
                compress_ratio=com_ratio,
                com_topk=com_topk,
                attn_sink=attn_sink.contiguous(),
            )
        return out if out.dtype == q_dtype else out.to(q_dtype)

    def _forward_prefill_sparse(
        self,
        q: torch.Tensor,
        layer_id: int,
        compress_ratio: Literal[0, 4, 128],
        forward_batch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
    ) -> torch.Tensor:
        """Upstream ``_forward_prefill_sparse`` with ``flash_mla_sparse_fwd``
        replaced by pure-torch sparse attention over the gathered workspace.
        """
        # q is (b, 1, h_q, d_qk); the sparse attention takes (s_q, h_q, d_qk).
        q_flat = q.squeeze(1)

        cache = self.forward_metadata.sparse_prefill_cache
        if cache is None:
            # Internal fork's ``SparsePrefillChunkCache.build`` additionally
            # requires ``max_seq_len`` and ``total_swa`` (the upstream v0.5.x
            # signature computed ``total_swa`` internally).  Mirror the internal
            # ``_build_sparse_prefill_chunk_cache`` computation.
            seq_lens_cpu_list = forward_batch.seq_lens_cpu.tolist()
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            total_swa = sum(
                min(int(seq_len), int(extend_len) + SWA_WINDOW - 1)
                for seq_len, extend_len in zip(
                    seq_lens_cpu_list, extend_seq_lens_cpu, strict=True
                )
            )
            _seq_flat = forward_batch.seq_lens.reshape(-1)
            _ext_flat = forward_batch.extend_seq_lens.reshape(-1)
            _n = min(_seq_flat.numel(), _ext_flat.numel())
            _ext64 = _ext_flat[: _n].to(torch.int64)
            _q_lens = _ext64.to(torch.int32)
            _starts = _seq_flat[: _n].to(torch.int64) - _ext64
            _rep_starts = torch.repeat_interleave(_starts, _ext64)
            _total = _rep_starts.numel()
            _row_ids = torch.arange(_total, device=_ext64.device, dtype=torch.int64)
            _q_pos = (
                _rep_starts
                + _row_ids
                - torch.repeat_interleave(torch.cumsum(_ext64, 0) - _ext64, _ext64)
            ).to(torch.int32)
            _want = q_flat.shape[0]
            if _q_pos.numel() < _want:
                _q_pos = torch.nn.functional.pad(
                    _q_pos, (0, _want - _q_pos.numel()), value=0
                )
            elif _q_pos.numel() > _want:
                _q_pos = _q_pos[:_want]
            cache = SparsePrefillChunkCache.build(
                seq_lens=forward_batch.seq_lens.to(torch.int32),
                extend_seq_lens=forward_batch.extend_seq_lens.to(torch.int32),
                query_lens=_q_lens,
                query_pos=_q_pos,
                req_pool_indices=forward_batch.req_pool_indices.to(torch.int32),
                req_to_token=self.req_to_token,
                full_to_swa=token_to_kv_pool.full_to_swa_index_mapping,
                swa_window_size=SWA_WINDOW,
                swa_page_size=token_to_kv_pool.swa_page_size,
                num_qo_tokens=q_flat.shape[0],
                max_seq_len=max(seq_lens_cpu_list),
                total_swa=total_swa,
            )
            self.forward_metadata.sparse_prefill_cache = cache
        # Resolve the workspace + indices for this ratio, then dequant
        # SWA + compressed regions directly into the workspace (no torch.cat).
        compressed_slice = None
        extra_k_cache = None
        extra_page_size = None
        flat_token_ids = None
        # The internal fork allocates the flat KV workspace from
        # ``self.sparse_prefill_workspace`` (a per-backend pool), not from
        # per-ratio ``cache.*_workspace`` attributes.
        if compress_ratio == 0:
            workspace = self.sparse_prefill_workspace.get(
                cache.swa_token_ids.shape[0]
            )
            combined_indices = cache.c0_combined_indices
            combined_lens = cache.c0_combined_lens
            swa_slice = workspace
        else:
            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            if compress_ratio == 128:
                assert core_attn_metadata.c128_page_indices is not None
                gather = cache.ensure_c128(core_attn_metadata.c128_page_indices)
                flat_token_ids = gather.flat_token_ids
                combined_indices = gather.combined_indices
                combined_lens = gather.combined_lens
            else:
                assert core_attn_metadata.c4_sparse_raw_indices is not None, (
                    "sparse-prefill c4 path requires c4_sparse_raw_indices "
                    "(allocated in init_flashmla_related when is_prefill=True)"
                )
                gather = cache.ensure_compressed(
                    compress_ratio, core_attn_metadata.page_table, extra_page_size
                )
                flat_token_ids = gather.flat_token_ids
                combined_indices, combined_lens = cache.combine_compressed(
                    compress_ratio,
                    core_attn_metadata.c4_sparse_raw_indices[: cache.num_qo_tokens],
                )
            n_compressed = flat_token_ids.shape[0]
            workspace = self.sparse_prefill_workspace.get(
                n_compressed + cache.swa_token_ids.shape[0]
            )
            compressed_slice = workspace[:n_compressed]
            swa_slice = workspace[n_compressed:]

        from sglang_kunlun.eb5_adapt.kernels.kernel_ops import (
            dequantize_k_cache_paged_torch,
            gather_bf16_k_cache_paged_torch,
        )

        # The bf16 pool is a plain bf16 paged buffer; the fp8 dequantizer would
        # misinterpret its bytes.  Prefer the imported symbol when available,
        # otherwise use the plugin's own torch gather.
        if self.is_bf16_kv_cache:
            populate_kv_workspace = (
                gather_bf16_k_cache_paged
                if gather_bf16_k_cache_paged is not None
                else gather_bf16_k_cache_paged_torch
            )
        else:
            populate_kv_workspace = dequantize_k_cache_paged_torch
        if compressed_slice is not None:
            populate_kv_workspace(
                extra_k_cache,
                flat_token_ids,
                page_size=extra_page_size,
                out=compressed_slice,
            )
        populate_kv_workspace(
            token_to_kv_pool.get_swa_key_buffer_radix(layer_id),
            cache.swa_token_ids,
            page_size=cache.swa_page_size,
            out=swa_slice,
        )
        # ---- inlined ``flash_mla_sparse_fwd`` (pure torch) ------------------
        # q_flat: (s_q, h_q, d_qk); workspace: (s_kv, 1, d_qk) bf16;
        # combined_indices: (s_q, topk) rebased locs into the workspace.
        s_q = q_flat.shape[0]
        kv_flat = workspace.reshape(workspace.shape[0], -1)
        idx = combined_indices.reshape(s_q, -1).to(torch.int64)
        ar = torch.arange(idx.shape[1], device=q.device).unsqueeze(0)
        # NSA prefill context parallelism pads the per-query index rows, and the
        # padded rows can carry indices past the gathered workspace. An
        # out-of-range gather reads garbage (nondeterministic NaN rows) or faults
        # with an illegal memory access, so treat out-of-range slots as invalid
        # and clamp the gather index.
        in_range = (idx >= 0) & (idx < kv_flat.shape[0])
        if combined_lens is not None:
            valid = (ar < combined_lens.to(torch.int64).unsqueeze(1)) & in_range
        else:
            valid = in_range
        safe_idx = torch.where(in_range, idx, torch.zeros_like(idx))
        keys = kv_flat[safe_idx.reshape(-1)].reshape(s_q, idx.shape[1], -1)
        _out = self._sparse_mla_attention(
            q_flat, keys, valid, attn_sink, self.softmax_scale, self.head_dim_v
        )
        return _out

    def _dequant_paged_torch(
        self,
        buf_u8_flat: torch.Tensor,
        flat_idx: torch.Tensor,
        page_size: int,
        bytes_per_page: int,
    ) -> torch.Tensor:
        """Decode N tokens from the v4 paged layout to (N, 512) bf16, pure torch."""
        device = buf_u8_flat.device
        loc = flat_idx.to(torch.int64)
        N = loc.shape[0]
        page_idx = loc // page_size
        in_page = loc % page_size
        page_byte_base = page_idx * bytes_per_page
        token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
        s_offset_bytes = page_size * NOPE_ROPE_BYTES
        token_scale_base = (
            page_byte_base + s_offset_bytes + in_page * PADDED_SCALE_PER_TOKEN
        )

        # nope fp8 bytes -> float via LUT
        nope_byte = (
            token_data_base[:, None] + torch.arange(DIM_NOPE, device=device)[None, :]
        )
        nope_u8 = buf_u8_flat[nope_byte.reshape(-1)].to(torch.int64)
        nope_f = _e4m3_lut(device)[nope_u8].reshape(N, DIM_NOPE)

        # ue8m0 scales -> 2^(s-127), zeroing subnormals
        scale_byte = (
            token_scale_base[:, None]
            + torch.arange(NUM_SCALE_TILES, device=device)[None, :]
        )
        scale_u8 = buf_u8_flat[scale_byte.reshape(-1)].reshape(N, NUM_SCALE_TILES)
        scale_pow2 = torch.exp2(scale_u8.to(torch.float32) - 127.0)
        scale_pow2 = torch.where(
            scale_pow2 < (2.0**-126), torch.zeros_like(scale_pow2), scale_pow2
        )
        scale_full = scale_pow2.repeat_interleave(TILE_SIZE, dim=1)
        nope = (nope_f * scale_full).to(torch.bfloat16)

        # rope bf16: reinterpret the same bytes as bf16 and gather
        buf_bf16 = buf_u8_flat.view(torch.bfloat16)
        rope_bf16_base = (token_data_base + DIM_NOPE) // 2
        rope_idx = (
            rope_bf16_base[:, None] + torch.arange(DIM_ROPE, device=device)[None, :]
        )
        rope = buf_bf16[rope_idx.reshape(-1)].reshape(N, DIM_ROPE)

        out = torch.empty((N, _HEAD_DIM), dtype=torch.bfloat16, device=device)
        out[:, :DIM_NOPE] = nope
        out[:, DIM_NOPE:] = rope
        return out

    def _dequant_sparse_keys(
        self,
        k_cache_4d: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize v4 FP8-packed keys at per-query token locations.

        Returns ``(keys (n_q, K, 512) bf16, valid_mask (n_q, K) bool)``; invalid
        slots are zeroed.
        """
        num_pages, page_size, _, _total_dim = k_cache_4d.shape
        # k_cache_4d is a strided slice-view of the pool's contiguous
        # (num_pages, bytes_per_page_padded) uint8 buffer. Materializing it with
        # .contiguous() triggers an unsupported strided-uint8 copy on Kunlun XPU,
        # so read from the contiguous base buffer using its real page stride.
        base = k_cache_4d
        while base._base is not None:
            base = base._base
        # If the pool buffer was realigned (sglang-kunlun PD), `base` is the padded
        # flat storage; recover the real KV region [off, numel).
        _lg = getattr(base, "_kunlun_logical", None)
        if _lg is not None and len(_lg) == 2:
            base = base.narrow(0, _lg[0], _lg[1])
        assert base.is_contiguous(), "expected contiguous base kv buffer"

        ar = torch.arange(indices.shape[1], device=indices.device).unsqueeze(0)
        idx_i64 = indices.to(torch.int64)
        valid_mask = (ar < lengths.to(torch.int64).unsqueeze(1)) & (idx_i64 >= 0)

        # bf16 KV (align-to-reference): direct 512-bf16 gather, no fp8 dequant.
        if base.dtype in (torch.bfloat16, torch.float16):
            flat = base.reshape(-1, _HEAD_DIM)
            safe = torch.where(idx_i64 >= 0, idx_i64, torch.zeros_like(idx_i64))
            safe = safe.clamp(0, flat.shape[0] - 1)  # KUNLUN_IDX_CLAMP idle/dummy safety
            keys = (
                flat[safe.reshape(-1)]
                .view(indices.shape[0], indices.shape[1], _HEAD_DIM)
                .to(torch.bfloat16)
            )
            # Slots this rank never wrote (other CP ranks' pages, or the page-aligned
            # tail) hold uninitialised memory that reads back as NaN. The mask only
            # zeroes known-invalid slots, so scrub the rest instead of letting a NaN
            # key wipe out the whole softmax row.
            keys = torch.nan_to_num(keys) * valid_mask.unsqueeze(-1)
            return keys, valid_mask

        # The pool stores bytes as uint8, but get_key_buffer().view(dtype) may hand
        # us an fp8/bf16-typed view. Reinterpret the flat storage as raw bytes.
        base_u8 = base.reshape(-1).view(torch.uint8)
        assert base_u8.numel() % num_pages == 0, "base buffer not page-aligned"
        bytes_per_page = base_u8.numel() // num_pages

        safe_idx = torch.where(idx_i64 >= 0, idx_i64, torch.zeros_like(idx_i64))
        safe_idx = safe_idx.clamp(0, max(num_pages * page_size - 1, 0))  # KUNLUN_IDX_CLAMP
        deq = self._dequant_paged_torch(
            base_u8, safe_idx.reshape(-1).to(torch.int32), page_size, bytes_per_page
        )
        keys = deq.view(indices.shape[0], indices.shape[1], _HEAD_DIM)
        return keys, valid_mask

    def _sparse_mla_attention(
        self,
        q: torch.Tensor,
        keys: torch.Tensor,
        valid_mask: torch.Tensor,
        attn_sink: Optional[torch.Tensor],
        softmax_scale: float,
        head_dim_v: int,
    ) -> torch.Tensor:
        """MLA softmax attention with per-head attention-sink virtual key.

        ``q`` is (n_q, h_q, 512), ``keys`` (n_q, K, 512) bf16 (== values in absorbed
        MLA), ``valid_mask`` (n_q, K); returns (n_q, h_q, head_dim_v) in q.dtype.
        """
        n_q, h_q, _d = q.shape
        # KUNLUN_ATTN_IDLE_GUARD: DP-attention idle/dummy forward can hand us n_q==0
        # or K==0. The degenerate softmax/reduce over a size-0 dim illegal-accesses
        # on Kunlun XPU. The idle output is unused, so return zeros of the right shape.
        if n_q == 0 or keys.shape[1] == 0:
            return q.new_zeros((n_q, h_q, head_dim_v))
        qf = q.float()
        kf = keys.float()  # (n_q, K, d)

        # scores: (n_q, h_q, K)
        scores = torch.einsum("qhd,qkd->qhk", qf, kf) * softmax_scale
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid_mask.unsqueeze(1), neg_inf)

        if attn_sink is not None:
            sink = attn_sink.float().view(1, h_q, 1).expand(n_q, h_q, 1)
            scores_ws = torch.cat([scores, sink], dim=-1)  # (n_q, h_q, K+1)
            probs = torch.softmax(scores_ws, dim=-1)[..., :-1]  # drop virtual key
        else:
            probs = torch.softmax(scores, dim=-1)

        # For a query with zero valid keys softmax over all -inf yields nan; zero it.
        any_valid = valid_mask.any(dim=-1)  # (n_q,)
        probs = torch.where(any_valid.view(n_q, 1, 1), probs, torch.zeros_like(probs))

        out = torch.einsum("qhk,qkd->qhd", probs, kf)  # (n_q, h_q, d)
        return out[..., :head_dim_v].to(q.dtype)


class KunlunDeepseekV4MultiStepBackend(DeepseekV4MultiStepBackend):
    """0.5.14 multi-step control flow composed only of Kunlun backends."""

    def __init__(self, model_runner, topk: int, speculative_num_steps: int):
        DeepseekV4AttnBackend.__init__(self, model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[KunlunDeepseekV4AttnBackend] = [
            KunlunDeepseekV4AttnBackend(
                model_runner,
                speculative_step_id=i,
                topk=topk,
                speculative_num_steps=speculative_num_steps,
            )
            for i in range(speculative_num_steps)
        ]