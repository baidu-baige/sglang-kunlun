"""Method hooks for the upstream DeepSeek V4 model; no model classes are copied."""

from __future__ import annotations

from typing import Optional

import sys
import torch
import torch.nn.functional as F
import kunlun_ops
from kunlun_ops import hc_post_kunlun_impl
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels.kernel_ops import dsv4_mqa_wo_a_einsum_kunlun


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
    """Route DSV4 KV writes through the 0.5.8 mapping-writer contract."""
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
    attn_backend.store_cache(
        layer_id=self.layer_id,
        swa_k=kv,
        forward_batch=forward_batch,
    )


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
        qkv_a, _ = self.wqkv_a(x_linear)
        q_lora = qkv_a[..., : self.q_lora_rank]
        kv = qkv_a[..., self.q_lora_rank :]
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
    attn_backend.store_cache(
        layer_id=self.layer_id,
        swa_k=kv,
        forward_batch=forward_batch,
    )

    if self.indexer is not None:
        self.indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            attn_backend=attn_backend,
        )
    if self.compressor is not None:
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


class _TorchWoAProxy:
    def __init__(self, original_torch):
        self._original_torch = original_torch

    def __getattr__(self, name):
        return getattr(self._original_torch, name)

    def einsum(self, equation, *operands):
        if equation == "tgd,grd->tgr" and len(operands) == 2:
            return dsv4_mqa_wo_a_einsum_kunlun(operands[0], operands[1])
        return self._original_torch.einsum(equation, *operands)


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer.forward",
    type=HookType.AROUND,
)
def mqa_forward_with_kunlun_wo_a(original_fn, self, *args, **kwargs):
    module = sys.modules[original_fn.__module__]
    original_torch = module.torch
    original_local_sink = self._attn_sink_local
    if self.tp_size > 1:
        self._attn_sink_local = self.attn_sink
    module.torch = _TorchWoAProxy(original_torch)
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        module.torch = original_torch
        self._attn_sink_local = original_local_sink
