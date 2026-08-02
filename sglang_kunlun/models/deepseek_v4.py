"""Method hooks for the upstream DeepSeek V4 model; no model classes are copied."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import kunlun_ops
from kunlun_ops import hc_post_kunlun_impl
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels.kernel_ops import dsv4_mqa_wo_a_einsum_kunlun


def _store_kv_to_swa_cache_direct(self, kv, forward_batch, attn_backend) -> None:
    """Match the dirty 0.5.14 SWA cache writer exactly."""
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    token_to_kv_pool = get_token_to_kv_pool()
    swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
    cache = token_to_kv_pool.swa_kv_pool.kv_buffer[
        token_to_kv_pool._swa_local_layer_id(self.layer_id)
    ]
    cache_tokens = cache.view(-1, kv.shape[-1])
    loc = swa_loc.contiguous().clamp(
        min=0, max=cache_tokens.shape[0] - 1
    ).long()
    cache_value = kv.reshape(kv.shape[0], -1).to(cache.dtype)
    cache_tokens.index_copy_(0, loc, cache_value)


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer._compute_kv_to_cache",
    type=HookType.REPLACE,
)
def compute_kv_to_cache_kunlun(
    self,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    attn_backend,
    qkv_a: Optional[torch.Tensor] = None,
) -> None:
    """Normalize and rotate KV, then use the dirty 0.5.14 direct writer."""
    if qkv_a is not None:
        kv = qkv_a[..., self.q_lora_rank :]
    else:
        kv, _ = self.wkv(x)
    kv = self.kv_norm(kv)

    from sglang.srt.models.deepseek_v4 import fused_rope_inplace

    fused_rope_inplace(
        kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
        None,
        self.freqs_cis,
        positions,
    )
    _store_kv_to_swa_cache_direct(self, kv, forward_batch, attn_backend)


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer._forward_prepare",
    type=HookType.AROUND,
)
def mqa_forward_prepare_058_kunlun(
    original_fn,
    self,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    attn_backend,
    q_out: Optional[torch.Tensor] = None,
    x_quant=None,
):
    """Use the 0.5.8 single-call Q/KV RoPE order for prefill and decode."""
    from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )

    use_cp = False
    if self.dsa_enable_prefill_cp:
        from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp

        use_cp = dsa_use_prefill_cp(forward_batch)

    if (
        not (
            forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_decode_or_idle()
        )
        or is_unified_kv_triton()
        or use_cp
    ):
        return original_fn(
            self,
            x,
            positions,
            forward_batch,
            attn_backend,
            q_out,
            x_quant=x_quant,
        )

    x_linear = x_quant if x_quant is not None else x
    if self.fuse_wqa_wkv:
        self._dsv4_dump_probe("wqkv_a.weight", self.wqkv_a.weight)
        qkv_a, _ = self.wqkv_a(x_linear)
        q_lora = qkv_a[..., : self.q_lora_rank]
        kv = qkv_a[..., self.q_lora_rank :]
        self._dsv4_dump_probe("wqkv_a.q_lora", q_lora)
        self._dsv4_dump_probe("wqkv_a.kv_split", kv)
    else:
        q_lora, _ = self.wq_a(x_linear)
        kv, _ = self.wkv(x_linear)

    q_lora = self.q_norm(q_lora)
    q, _ = self.wq_b(q_lora)
    q = q.view(-1, self.n_local_heads, self.head_dim)
    q_normalized = torch.empty_like(q)
    kunlun_ops.rmsnorm(
        q,
        None,
        q_normalized,
        self.eps,
        False,
        True,
        None,
        None,
        None,
    )
    q = q_normalized
    kv = self.kv_norm(kv)

    from sglang.srt.models.deepseek_v4 import fused_rope_inplace

    fused_rope_inplace(
        q[..., -self.qk_rope_head_dim :],
        kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
        self.freqs_cis,
        positions,
    )
    _store_kv_to_swa_cache_direct(self, kv, forward_batch, attn_backend)

    self._dsv4_dump_probe("pre_indexer.x", x)
    self._dsv4_dump_probe("pre_indexer.q_lora", q_lora)
    self._dsv4_dump_probe("pre_attention.q", q)
    self._dsv4_dump_probe("pre_attention.k", kv)
    self._dsv4_dump_probe("pre_attention.v", kv)
    if self.indexer is not None:
        self.indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            attn_backend=attn_backend,
        )
    if self.compressor is not None:
        self._dsv4_dump_probe("pre_compressor.x", x)
        attn_backend.forward_core_compressor(
            x,
            forward_batch,
            self.layer_id,
            self.compressor,
        )

    if q_out is not None:
        q_out.copy_(q)
        q = q_out
    return q, None


@plugin_hook(
    "sglang.srt.models.deepseek_v2.MoEGate.forward",
    type=HookType.REPLACE,
)
def moe_gate_forward_kunlun(
    self,
    hidden_states: torch.Tensor,
    gemm_output_zero_allocator=None,
    forward_batch=None,
):
    """Match the 0.5.8 Kunlun DSV4 router GEMM contract."""

    del gemm_output_zero_allocator, forward_batch
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"unsupported Kunlun MoE gate input dtype: {hidden_states.dtype}")
    if self.weight.dtype != hidden_states.dtype:
        raise TypeError(
            f"Kunlun MoE gate dtype mismatch: {hidden_states.dtype} != {self.weight.dtype}"
        )
    return hidden_states @ self.weight.T


@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4Model.hc_head",
    type=HookType.REPLACE,
)
def hc_head_kunlun(
    self,
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
):
    """Match the graph-captured 0.5.8 base-model MHC head exactly."""
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


# @plugin_hook(
#     "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_pre",
#     type=HookType.REPLACE,
# )
def hc_pre_kunlun(
    self,
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm: Optional[nn.Module] = None,
    forward_batch: Optional[ForwardBatch] = None,
):
    """Kunlun MHC pre with the 0.5.14 four-value return contract."""
    del norm, forward_batch
    shape, dtype = x.shape, x.dtype
    if x.shape[0] == 0:
        y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
        post = torch.empty((0, self.hc_mult), dtype=torch.float32, device=x.device)
        comb = torch.empty(
            (0, self.hc_mult, self.hc_mult), dtype=torch.float32, device=x.device
        )
        return y, post, comb, False

    if self.hc_mult != 4:
        raise ValueError("Kunlun hc_pre supports hc_mult=4 only")
    y, post, comb = torch.ops.xspeedgate_ops.hc_pre(
        x.contiguous(),
        hc_fn.contiguous(),
        hc_scale.contiguous(),
        hc_base.contiguous(),
        rms_eps=self.rms_norm_eps,
        hc_pre_eps=self.hc_eps,
        hc_sinkhorn_eps=self.hc_eps,
        mhc_post_mult_value=2.0,
        sinkhorn_iters=self.hc_sinkhorn_iters,
    )
    return y, post, comb, False


@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post",
    type=HookType.REPLACE,
)
def hc_post_kunlun(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    """Kunlun MHC post preserving empty shape and output dtype."""
    if x.shape[0] == 0:
        return torch.empty(
            (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
        )

    assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
    assert post.shape == (x.shape[0], self.hc_mult)
    assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)
    batch, hidden_size = x.shape[0], x.shape[-1]
    out = torch.empty(
        (batch, self.hc_mult, hidden_size), dtype=x.dtype, device=x.device
    )
    hc_post_kunlun_impl(
        post.contiguous(),
        x.contiguous(),
        comb.contiguous(),
        residual.contiguous(),
        out,
        batch,
        self.hc_mult,
        hidden_size,
    )
    return out

