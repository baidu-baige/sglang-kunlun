"""Method hooks for the upstream DeepSeek V4 model; no model classes are copied."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels.kernel_ops import dsv4_mqa_wo_a_einsum_kunlun


def _store_kv_to_swa_cache_direct(self, kv, forward_batch, attn_backend) -> None:
    """Store normalized KV through the active DSV4 cache-pack contract."""
    attn_backend.store_cache(
        layer_id=self.layer_id,
        swa_k=kv,
        forward_batch=forward_batch,
    )


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
def mqa_forward_prepare_kunlun(
    original_fn,
    self,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    attn_backend,
    q_out: Optional[torch.Tensor] = None,
    x_quant=None,
):
    """Use the single-call Q/KV RoPE order for prefill and decode."""
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
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
    import kunlun_ops

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
    """Match the Kunlun DSV4 router GEMM contract."""

    del gemm_output_zero_allocator, forward_batch
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"unsupported Kunlun MoE gate input dtype: {hidden_states.dtype}")
    if self.weight.dtype != hidden_states.dtype:
        hidden_states = hidden_states.to(dtype=self.weight.dtype)
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
    """Match the graph-captured base-model MHC head exactly."""
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


# Golden keeps this hook disabled so the upstream Torch hc_pre runs and the
# mHC post/comb stay fp32; the Kunlun fused hc_pre returns fp16.
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
    x_flat = x.reshape(x.shape[0], -1).float()
    x_view = x_flat.view(x.shape[0], self.hc_mult, -1)
    rms = torch.rsqrt(
        x_flat.square().mean(dim=-1, keepdim=True) + self.rms_norm_eps
    )
    mixes = F.linear(x_flat, hc_fn.float()) * rms
    pre = torch.sigmoid(mixes[:, : self.hc_mult] * hc_scale[:1] + hc_base[: self.hc_mult])
    pre = pre + self.hc_eps
    post = 2.0 * torch.sigmoid(
        mixes[:, self.hc_mult : 2 * self.hc_mult] * hc_scale[1:2]
        + hc_base[self.hc_mult : 2 * self.hc_mult]
    )
    comb = (
        mixes[:, 2 * self.hc_mult :] * hc_scale[2]
        + hc_base[2 * self.hc_mult :]
    ).view(x.shape[0], self.hc_mult, self.hc_mult)

    # Sinkhorn normalization matches the reference's stabilized row/column order.
    comb = comb - comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb)
    comb = comb / comb.sum(dim=-1, keepdim=True) + self.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    for _ in range(max(self.hc_sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    y = (pre.unsqueeze(-1) * x_view).sum(dim=1).to(dtype)
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
    from kunlun_ops import hc_post_kunlun_impl

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


# ---------------------------------------------------------------------------
# compressed-tensors 量化配置的命名对齐
#
# W4A8 那份 checkpoint 的 quantization_config 是按 **checkpoint 原始命名** 导出的
# （``layers.N.attn.*`` / ``layers.N.ffn.*``、wq_a 与 wkv 未融合），而 sglang 的模块
# 路径是 ``layers.N.self_attn.*`` / ``layers.N.mlp.*``、且默认把 wq_a+wkv 融合成
# wqkv_a。两者对不上时 find_matched_target 会直接抛
# "Unable to find matching target for model.layers.0.self_attn.wqkv_a"。
#
# sglang 本来就有这个对齐通道：loader.py 里
# ``hf_to_sglang_mapper = getattr(model_class, "hf_to_sglang_mapper", None)``，
# 拿到后调 ``quant_config.apply_weight_name_mapper()``。但 DSV4 的模型类没有定义这个
# 属性，于是通道是空的。这里把它补上。
#
# 规则写成正则模式串上的子串替换（WeightsMapper 按 key 长度倒序匹配、命中一条即停），
# 所以更长的 wq_a/wkv→wqkv_a 规则会先于通用的 attn→self_attn 规则生效。
# 对已经用 sglang 命名导出的 W8A8 checkpoint 这些规则全部不命中，是无害的空操作
# （``layers\.\d+\.self_attn\.`` 里不含 ``layers\.\d+\.attn\.``）。
# ---------------------------------------------------------------------------
_DSV4_QUANT_NAME_MAP_BASE = {
    # DSpark draft 用 fuse_wqa_wkv=False，mtp 侧本来就不融合。
    r"mtp\.\d+\.attn\.": r"mtp\.\d+\.self_attn\.",
    r"layers\.\d+\.attn\.": r"layers\.\d+\.self_attn\.",
    r"layers\.\d+\.ffn\.": r"layers\.\d+\.mlp\.",
    r"mtp\.\d+\.ffn\.": r"mtp\.\d+\.mlp\.",
}

# 开融合时才把 wq_a/wkv 折到 wqkv_a 上（两者 scheme 相同，映射后合并成一个 key）。
# 关融合时必须不折，否则模型里是分开的 wq_a / wkv 两个模块、反而匹配不上。
_DSV4_QUANT_NAME_MAP_FUSED = {
    r"layers\.\d+\.attn\.wq_a$": r"layers\.\d+\.self_attn\.wqkv_a$",
    r"layers\.\d+\.attn\.wkv$": r"layers\.\d+\.self_attn\.wqkv_a$",
}


def _build_dsv4_quant_name_map(fuse_wqa_wkv: bool) -> dict:
    mapping = dict(_DSV4_QUANT_NAME_MAP_BASE)
    if fuse_wqa_wkv:
        mapping.update(_DSV4_QUANT_NAME_MAP_FUSED)
    return mapping


def _install_dsv4_quant_name_mapper() -> None:
    from sglang.srt import environ as _environ
    from sglang.srt.models.utils import WeightsMapper

    fuse = bool(_environ.envs.SGLANG_OPT_FUSE_WQA_WKV.get())
    mapper = WeightsMapper(orig_to_new_substr=_build_dsv4_quant_name_map(fuse))
    targets = []
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    targets.append(DeepseekV4ForCausalLM)
    try:
        from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark

        targets.append(DeepseekV4ForCausalLMDSpark)
    except ImportError:
        pass
    for cls in targets:
        # 已经自带 mapper 的话不覆盖（上游若后续补上，以上游为准）。
        if getattr(cls, "hf_to_sglang_mapper", None) is None:
            cls.hf_to_sglang_mapper = mapper


_install_dsv4_quant_name_mapper()

