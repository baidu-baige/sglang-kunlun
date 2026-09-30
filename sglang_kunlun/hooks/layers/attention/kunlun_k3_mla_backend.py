"""Kunlun (P800/XPU) MLA attention path for Kimi-K3.

Why a backend subclass instead of editing ``kunlun_backend.py``:
``KunlunAttentionBackend`` is the *generic* Kunlun backend and DeepSeek-V4 rides
on its MLA branches, which still go through ``flash_attn_varlen_func`` /
``flash_attn_with_kvcache`` and merge with ``merge_state_v2_wrapper``. Rewriting
those in place would silently retarget DSV4 at K3's operator contract. Upstream
already models "this hardware runs attention differently" as a backend subclass,
and the sibling ``kunlun_kda_backend.py`` uses that shape for the linear half of
K3, so this file follows the precedent: subclass, override the two MLA entry
points, inherit everything else.

What stays in the base class: all metadata construction
(``init_forward_metadata``, the capture/replay methods, and the pointer-stable
length twins allocated in ``init_cuda_graph_state``), the KV pool, the page
table, SWA translation, and every non-operator control-flow decision. What lives
here: the MLA operator calls, and the fast paths that must not be attempted on
Kunlun.

Kunlun advertises ``device_type="cuda"`` (``platform/device.py:13``), so every
upstream ``is_cuda()`` gate answers True. The FA3/FA4 MLA kernels are therefore
*offered* to us and must be declined explicitly rather than filtered out by a
platform check -- and they must be declined, because they cannot read K3's latent
cache on XPU. There is no torch fallback here on purpose: the kimi-k3 branch
carried ``_torch_varlen_attention`` / ``_torch_paged_mla_attention`` and made them
the default, which is both the 0.5 token/s ceiling and the reason cuda-graph
capture fails (the paged fallback raises during capture). When a shape or dtype
falls outside an operator's contract this file raises and names both.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch

import kunlun_ops

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from .kunlun_backend import KunlunAttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)

# The fused decode op is hard-wired to K3's latent geometry.
_K3_KV_LORA_RANK = 512
_K3_QK_ROPE_HEAD_DIM = 64
_K3_PACKED_DIM = _K3_KV_LORA_RANK + _K3_QK_ROPE_HEAD_DIM  # 576


def _is_capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # pragma: no cover - no accelerator under unit test
        return False


def _adjacent_packed_view(
    c_kv: torch.Tensor, k_rope: torch.Tensor
) -> Optional[torch.Tensor]:
    """A single ``[.., kv_lora_rank + qk_rope_head_dim]`` view, or None.

    The latent cache stores nope and rope halves back to back, so the packed
    tensor the op wants usually already exists in memory. Proving that lets us
    skip a ``torch.cat`` -- which matters twice over: it is a per-forward
    allocation, and a variable-length ``cat`` cannot be captured in a graph.
    Returns None when the halves are not actually adjacent, and the caller then
    raises rather than silently paying for a copy.
    """
    if c_kv.dtype != k_rope.dtype or c_kv.device != k_rope.device:
        return None
    if c_kv.stride()[:-1] != k_rope.stride()[:-1]:
        return None
    if c_kv.stride(-1) != 1 or k_rope.stride(-1) != 1:
        return None
    if c_kv.shape[:-1] != k_rope.shape[:-1]:
        return None
    # k_rope must begin exactly where c_kv ends.
    expected = c_kv.storage_offset() + c_kv.shape[-1]
    if k_rope.storage_offset() != expected:
        return None
    packed_last = c_kv.shape[-1] + k_rope.shape[-1]
    return c_kv.as_strided(
        (*c_kv.shape[:-1], packed_last), c_kv.stride(), c_kv.storage_offset()
    )


class KunlunK3MLAAttnBackend(KunlunAttentionBackend):
    """Kimi-K3's MLA layers on Kunlun, native operators only."""
    supports_ragged_verify_graph: bool = False

    def __init__(self, model_runner):
        super().__init__(model_runner)
        if not self.use_mla:
            raise ValueError(
                "KunlunK3MLAAttnBackend requires an MLA model; "
                f"attention_arch={model_runner.model_config.attention_arch}"
            )
        cfg = model_runner.model_config
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # ps.tp_size, not tp_size: the parallel-state view is the one that
        # reflects attn_tp after resolution.
        self.num_q_heads_local = cfg.num_attention_heads // model_runner.ps.tp_size
        self._graph_bufs: dict[str, torch.Tensor] = {}
        logger.info(
            "KunlunK3MLAAttnBackend active: kv_lora_rank=%d qk_rope=%d v_head=%d "
            "packed=%d q_heads_local=%d (no torch fallback)",
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.kv_cache_dim,
            self.num_q_heads_local,
        )

    # ------------------------------------------------------------ graph buffers

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Allocate the decode output/staging buffers once, at max_bs."""
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        h = self.num_q_heads_local
        d = self.kv_lora_rank
        dev, dt = self.device, self.kv_cache_dtype_for_graph()
        self._graph_bufs["out"] = torch.empty((max_bs, 1, h, d), dtype=dt, device=dev)
        self._graph_bufs["q_nope"] = torch.empty(
            (max_bs, 1, h, self.kv_lora_rank), dtype=dt, device=dev
        )
        self._graph_bufs["q_rope"] = torch.empty(
            (max_bs, 1, h, self.qk_rope_head_dim), dtype=dt, device=dev
        )

    def _first_full_attn_layer_id(self) -> int:
        """A layer id the latent pool will actually accept."""
        mapping = getattr(self.token_to_kv_pool, "full_attention_layer_id_mapping", None)
        if mapping:
            return min(mapping.keys())
        return 0

    def kv_cache_dtype_for_graph(self) -> torch.dtype:
        """The dtype the latent cache is stored in."""
        buf = self.token_to_kv_pool.get_key_buffer(self._first_full_attn_layer_id())
        return buf.dtype

    def _sliced_buf(self, name: str, bs: int, *shape_tail: int) -> torch.Tensor:
        buf = self._graph_bufs.get(name)
        if buf is None or buf.shape[0] < bs:
            raise RuntimeError(
                f"KunlunK3MLAAttnBackend: graph buffer {name!r} missing or too "
                f"small for bs={bs}; init_cuda_graph_state did not run with a "
                "large enough max_bs."
            )
        return buf[:bs]

    @staticmethod
    def _ds_alpha(layer: "RadixAttention") -> float:
        """MLA softmax scale."""
        return layer.scaling * math.sqrt(layer.head_dim)

    # ------------------------------------------------------------------ helpers

    def _require_no_sinks(self, sinks, where: str) -> None:
        if sinks is not None:
            raise NotImplementedError(
                f"{where}: neither kunlun_ops.attention nor "
                "kunlun_ops.paged_attention accepts an attention sink, and K3 "
                "does not configure one. Refusing to drop it silently."
            )

    def _require_matching_dtype(
        self, q: torch.Tensor, where: str, layer: "RadixAttention"
    ) -> torch.Tensor:
        """Return this layer's latent cache without casting it."""
        cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if cache.dtype != q.dtype:
            raise ValueError(
                f"{where}: latent KV cache dtype {cache.dtype} != query dtype "
                f"{q.dtype}. Casting the cache per forward is not acceptable "
                "here; launch with --dtype float16 --kv-cache-dtype fp16."
            )
        return cache

    def _extend_lods(self, metadata):
        """The four LOD tensors ``kunlun_ops.attention`` needs."""
        extra = metadata.extra_attn_metadata
        q_lod_xpu = metadata.cu_seqlens_q
        q_lod_cpu = extra.cu_seqlens_q_cpu
        if q_lod_cpu is None:
            # Not every forward mode populates the host twin (kunlun_backend.py
            # sets it only on some paths). Extend is never graph-captured, so a
            # host sync here is acceptable; decode never reaches this helper.
            q_lod_cpu = q_lod_xpu.to("cpu", dtype=torch.int32)
        kv_lod_cpu = extra.kv_lod_cpu
        kv_lod_xpu = extra.kv_lod_xpu
        if kv_lod_cpu is None or kv_lod_xpu is None:
            raise RuntimeError(
                "KunlunK3MLAAttnBackend: extend KV LODs missing from forward "
                "metadata (kv_lod_cpu / kv_lod_xpu). init_forward_metadata did "
                "not run for this forward mode."
            )
        return q_lod_cpu, q_lod_xpu, kv_lod_cpu, kv_lod_xpu

    def _split_mla_cache_key(
        self, k: torch.Tensor, k_rope: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate nope and rope halves for the cache write."""
        if k_rope is not None:
            return k, k_rope
        return k[..., : self.kv_lora_rank], k[..., self.kv_lora_rank :]

    def _save_kv(self, layer, forward_batch, k, v, k_rope, save_kv_cache) -> None:
        if k is None or not save_kv_cache:
            return
        assert v is not None, "MLA cache write needs both k and v"
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        cache_k_nope, cache_k_rope = self._split_mla_cache_key(k, k_rope)
        self.token_to_kv_pool.set_mla_kv_buffer(
            layer,
            cache_loc.contiguous(),
            cache_k_nope.contiguous(),
            cache_k_rope.contiguous(),
        )

    # ------------------------------------------------------------------- extend

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks=None,
        **kwargs,
    ):
        """forward_extend"""
        if not self.use_mla:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache,
                q_rope=q_rope, k_rope=k_rope, sinks=sinks, **kwargs,
            )
        self._require_no_sinks(sinks, "K3 MLA extend")
        return self._forward_extend_mla(
            q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
        )

    def _forward_extend_mla(
        self, q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
    ):
        self._save_kv(layer, forward_batch, k, v, k_rope, save_kv_cache)
        metadata = self.forward_metadata

        if (
            forward_batch.attn_attend_prefix_cache is not None
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return self._forward_extend_mla_chunked_prefix(
                q, k, v, layer, forward_batch, metadata, q_rope, k_rope
            )

        h = layer.tp_q_head_num
        if q_rope is not None:
            q_ = torch.cat(
                [
                    q.view(-1, h, layer.v_head_dim),
                    q_rope.view(-1, h, self.qk_rope_head_dim),
                ],
                dim=-1,
            )
        else:
            q_ = q.contiguous().view(-1, h, layer.head_dim)

        if k_rope is not None:
            k_ = torch.cat(
                [
                    k.view(-1, layer.tp_k_head_num, self.kv_lora_rank),
                    k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim),
                ],
                dim=-1,
            )
        else:
            k_ = k.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim)
        v_ = v.contiguous().view(-1, layer.tp_k_head_num, layer.v_head_dim)

        q_lod_cpu, q_lod_xpu, kv_lod_cpu, kv_lod_xpu = self._extend_lods(metadata)
        if int(q_lod_cpu[-1]) != int(kv_lod_cpu[-1]):
            # There is paged prefix history that a dense op cannot see, and
            # answering without it would be silently wrong.
            #
            # The intended operator is
            # ``kunlun_ops.paged_attention(is_context=True, block_tables=...,
            # context_kvlens_cpu/xpu=<context-phase LOD>, prefill_len=...,
            # kv_lora_rank=..., qk_rope_head_dim=..., mla_scale=...)`` -- every
            # one of those parameters exists (kunlun_ops/_attention.py:100-122)
            # and ``context_kvlens_*`` is documented as the context-phase LOD.
            # What is *not* settled is the query layout: ``q_r`` is documented as
            # "mla 中 decoder 时输入的 q_r"  (:145), i.e. decode-only, so in
            # context mode it is unclear whether ``x`` must carry the packed
            # [nope|rope] query or the op splits it from ``kv_lora_rank``.
            # Nothing in either tree calls it this way, so wiring it now would be
            # a guess. Confirm the context-mode q contract with the kernel owners
            # before implementing.
            #
            # Until then the supported route for a long prompt is the
            # chunked-prefix cascade above, which materialises K/V per chunk.
            raise NotImplementedError(
                "K3 MLA extend: query LOD total "
                f"{int(q_lod_cpu[-1])} != KV LOD total {int(kv_lod_cpu[-1])} "
                "outside the chunked-prefix path, so there is paged prefix "
                "history kunlun_ops.attention cannot read. Do not pass "
                "--disable-chunked-prefix-cache. (Native fix: paged_attention "
                "with is_context=True; blocked on the context-mode q_r/x "
                "contract, see the comment above this raise.)"
            )

        o_ = torch.empty(
            (q_.shape[0], h, layer.v_head_dim), dtype=q_.dtype, device=q_.device
        )
        softmax_lse = torch.full(
            (h, q_.shape[0]), float("-inf"), dtype=torch.float32, device=q_.device
        )
        kunlun_ops.attention(
            q_.contiguous(),
            k_.contiguous(),
            v_.contiguous(),
            o_,
            context_seq_lod_cpu=q_lod_cpu,
            context_seq_lod_xpu=q_lod_xpu,
            context_kvlen_lod_cpu=kv_lod_cpu,
            context_kvlen_lod_xpu=kv_lod_xpu,
            alpha=self._ds_alpha(layer),
            softmax_lse=softmax_lse,
            unpadded_lse=True,
        )
        return o_.view(-1, h * layer.v_head_dim)

    def _forward_extend_mla_chunked_prefix(
        self, q, k, v, layer, forward_batch, metadata, q_rope, k_rope
    ):
        """Long-prompt path: attend one chunk of the prefix cache, or the tail."""
        h, hk = layer.tp_q_head_num, layer.tp_k_head_num
        if q_rope is not None:
            q_ = torch.cat(
                [
                    q.view(-1, h, layer.v_head_dim),
                    q_rope.view(-1, h, self.qk_rope_head_dim),
                ],
                dim=-1,
            )
        else:
            q_ = q.contiguous().view(-1, h, layer.head_dim)
        k_ = k.contiguous().view(-1, hk, layer.head_dim)
        v_ = v.contiguous().view(-1, hk, layer.v_head_dim)

        q_lod_cpu, q_lod_xpu, _, _ = self._extend_lods(metadata)

        if forward_batch.attn_attend_prefix_cache:
            chunk_idx = forward_batch.prefix_chunk_idx
            assert chunk_idx is not None and chunk_idx >= 0
            assert forward_batch.prefix_chunk_cu_seq_lens is not None
            assert forward_batch.mha_return_lse, (
                "the chunked prefix cascade needs its LSE to merge with the tail"
            )
            kv_lod_xpu = forward_batch.prefix_chunk_cu_seq_lens[chunk_idx]
            # prefix_chunk_cu_seq_lens is device-only; build the host twin from
            # prefix_chunk_seq_lens_cpu rather than syncing the device tensor.
            seq_lens_cpu = forward_batch.prefix_chunk_seq_lens_cpu[chunk_idx]
            kv_lod_cpu = torch.zeros(
                seq_lens_cpu.numel() + 1, dtype=torch.int32, device="cpu"
            )
            kv_lod_cpu[1:] = seq_lens_cpu.to(torch.int32).cumsum(dim=0)
        else:
            # Tail of the sequence, no prefix attended. One-shot mode gives the
            # full KV extent; otherwise K/V are exactly this forward's tokens.
            if forward_batch.mha_one_shot:
                _, _, kv_lod_cpu, kv_lod_xpu = self._extend_lods(metadata)
            else:
                kv_lod_cpu, kv_lod_xpu = q_lod_cpu, q_lod_xpu

        o_ = torch.empty(
            (q_.shape[0], h, layer.v_head_dim), dtype=q_.dtype, device=q_.device
        )
        softmax_lse = torch.full(
            (h, q_.shape[0]), float("-inf"), dtype=torch.float32, device=q_.device
        )
        kunlun_ops.attention(
            q_.contiguous(),
            k_.contiguous(),
            v_.contiguous(),
            o_,
            context_seq_lod_cpu=q_lod_cpu,
            context_seq_lod_xpu=q_lod_xpu,
            context_kvlen_lod_cpu=kv_lod_cpu,
            context_kvlen_lod_xpu=kv_lod_xpu,
            alpha=self._ds_alpha(layer),
            softmax_lse=softmax_lse,
            unpadded_lse=True,
        )
        out = o_.view(-1, h * layer.v_head_dim)
        if forward_batch.mha_return_lse:
            # The merge side wants [tokens, heads]; unpadded_lse gives
            # [heads, tokens]. Transpose explicitly, and let
            # merge_state_v2_wrapper clamp the -inf away.
            return out, softmax_lse.transpose(0, 1).contiguous()
        return out

    # ------------------------------------------------------------------- decode

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks=None,
        **kwargs,
    ):
        """forward_decode"""
        if not self.use_mla:
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache,
                q_rope=q_rope, k_rope=k_rope, sinks=sinks, **kwargs,
            )
        self._require_no_sinks(sinks, "K3 MLA decode")
        return self._forward_decode_mla(
            q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
        )

    def _forward_decode_mla(
        self, q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope
    ):
        self._save_kv(layer, forward_batch, k, v, k_rope, save_kv_cache)

        metadata = self.forward_metadata
        extra = metadata.extra_attn_metadata
        kv_cache = self._require_matching_dtype(q, "K3 MLA decode", layer)
        kv_cache_paged = kv_cache.view(-1, 1, self.page_size, self.kv_cache_dim)

        h = layer.tp_q_head_num
        if q_rope is not None:
            q_nope = q.view(-1, h, layer.v_head_dim)
            q_rope_v = q_rope.view(-1, h, layer.head_dim - layer.v_head_dim)
        else:
            q_all = q.contiguous().view(-1, h, layer.head_dim)
            q_nope = q_all[..., : self.kv_lora_rank]
            q_rope_v = q_all[..., self.kv_lora_rank :]

        bs = q_nope.shape[0]
        if self._graph_bufs:
            out = self._sliced_buf("out", bs)
            qn = self._sliced_buf("q_nope", bs)
            qr = self._sliced_buf("q_rope", bs)
            qn.copy_(q_nope.unsqueeze(1))
            qr.copy_(q_rope_v.unsqueeze(1))
        elif _is_capturing():
            raise RuntimeError(
                "K3 MLA decode: graph capture with no preallocated buffers. "
                "init_cuda_graph_state must run before capture."
            )
        else:
            qn = q_nope.unsqueeze(1).contiguous()
            qr = q_rope_v.unsqueeze(1).contiguous()
            out = torch.empty(
                (bs, 1, h, layer.v_head_dim), dtype=q.dtype, device=q.device
            )

        # The CPU length twin is maintained by the base class and rebound per
        # graph bucket, so no per-call .cpu() sync here.
        context_lens_cpu = extra.host_context_len_tensor
        context_lens_xpu = extra.context_len_tensor
        if context_lens_cpu is None or context_lens_xpu is None:
            raise RuntimeError(
                "K3 MLA decode: context length twins missing from metadata; "
                "the base init_forward_metadata did not populate "
                "host_context_len_tensor / context_len_tensor."
            )

        kunlun_ops.paged_attention(
            out=out,
            x=qn,
            k_cache=kv_cache_paged,
            v_cache=None,
            block_tables=metadata.page_table,
            context_lens_cpu=context_lens_cpu,
            context_lens_xpu=context_lens_xpu,
            is_context=False,
            is_causal=True,
            vo_head_dim=self.kv_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            mla_scale=layer.scaling,
            q_r=qr,
        )
        return out.view(-1, h * layer.v_head_dim)