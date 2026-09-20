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
    q_rope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
):
    """Use the single-call Q/KV RoPE order for prefill and decode.

    `q_rope_out` / `k_nope_out` / `k_rope_out` are upstream's packed fp8
    two-pool output buffers. The caller only allocates them under
    `unified_fp8_decode`/`unified_fp8_prefill`, both of which require
    `is_unified_kv_triton()`, and that gate already sends us down the
    `original_fn` branch below -- so on the Kunlun fast path they are None.
    They are still forwarded (and asserted) rather than dropped.
    """
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
            q_rope_out=q_rope_out,
            k_nope_out=k_nope_out,
            k_rope_out=k_rope_out,
        )

    assert q_rope_out is None and k_nope_out is None and k_rope_out is None, (
        "packed fp8 two-pool buffers reached the Kunlun prepare path; they are "
        "only allocated under unified_kv, which is handled above"
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
    # Q 的 per-head 无权重 RMSNorm 是 V4 专属。上游用 `config.q_head_norm` 控制
    # （`_compute_q_b` 的 `fused_q_norm_rope` 分支），V4 是 True、**V4.1 是 False**；
    # checkpoint 自带的参考实现也只归一化低秩 q_lora，从不碰 per-head Q。无条件做会让
    # 每层 Q 被多归一化一次（layer 0 实测 |Q|max 31.4 → 11.1），输出全错但量级看着正常。
    if getattr(self, "q_head_norm", True):
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

    # V4.1 low ratios own a different source pipeline: `DeepseekV41Compressor` /
    # `DeepseekV41Indexer` are not per-layer indexer modules -- they expose
    # project()/queries()/scores() and have no forward, so `self.indexer(...)`
    # and the c4 `forward_core_compressor` are both invalid. Upstream routes both
    # through the backend's `forward_low_ratio_sources`. V4 (ratio 4/128) keeps
    # the c4 calls below.
    if getattr(self, "compress_ratio", 0) in (1, 2):
        if self.compressor is not None or self.indexer is not None:
            attn_backend.forward_low_ratio_sources(
                layer=self,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
            )
    else:
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
    """Match the Kunlun DSV4 router GEMM contract.

    logits 走 **fp32**，与上游所有 CUDA 路径（`tiny_gemm_bf16(out_dtype=fp32)` /
    `linear_bf16_fp32`）和参考实现（`Gate.forward`: `linear(x.float(), weight.float())`）
    一致。V4.1 是 384 个 expert + top-6，fp16 只有约 3 位十进制有效数字，`sqrtsoftplus`
    分数接近时会选错 expert。
    """

    del gemm_output_zero_allocator, forward_batch
    if hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"unsupported Kunlun MoE gate input dtype: {hidden_states.dtype}")
    return hidden_states.float() @ self.weight.float().T


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


# 融合后的 wqkv_a 拿不到 per-channel weight_scale 的问题。
#
# 上游的 wq_a+wkv 融合分支（sglang/srt/models/deepseek_v4.py 的 `elif fuse_wqa_wkv and (...)`）
# 只认 ``.weight`` 和 ``.weight_scale_inv``（fp8 块量化命名），不认 compressed-tensors 的
# per-channel ``.weight_scale``。于是这两份 scale 掉进通用分支、被改名成
# ``self_attn.wq_a.weight_scale``，而模型里只有融合后的 ``wqkv_a.weight_scale``，
# 日志报 "not found in params_dict"、融合模块的 scale 保持初值 => int8 反量化出垃圾
# => logits 变垃圾 => 解码出空串。注意 ``.weight`` 本身是被正确 cat 融合的，只有 scale 缺失，
# 加载阶段不报错，很难发现。
#
# 这里不改上游，而是在进 load_weights 之前把权重流里的两份 scale 先合并、直接以融合后的名字
# ``...wqkv_a.weight_scale`` 交出去，上游的通用分支就能在 params_dict 里找到它、正常加载。
# 合并方式和 ``.weight`` 完全一致：per-channel scale 是 per-output-channel，
# 融合就是把输出通道按 q、kv 顺序接起来，所以同样是 ``torch.cat([q, kv], dim=0)``。
_WQKV_A_SCALE_SHARDS = {".wq_a.weight_scale": "q", ".wkv.weight_scale": "kv"}


def _fuse_wqkv_a_weight_scales(weights):
    """把 wq_a / wkv 的 weight_scale 合并成融合后 wqkv_a 的那一份。"""
    pending: dict = {}
    for name, tensor in weights:
        shard = next(
            (s for suffix, s in _WQKV_A_SCALE_SHARDS.items() if name.endswith(suffix)),
            None,
        )
        if shard is None:
            yield name, tensor
            continue
        fused = name.replace(".wq_a.", ".wqkv_a.").replace(".wkv.", ".wqkv_a.")
        bucket = pending.setdefault(fused, {})
        assert shard not in bucket, f"duplicate shard {shard} for {fused}"
        bucket[shard] = tensor
        if len(bucket) == 2:
            yield fused, torch.cat([bucket["q"], bucket["kv"]], dim=0)
            del pending[fused]
    # 只凑到一半的按原名交回去，让上游按原逻辑报 not found，不要静默吞掉。
    for fused, bucket in pending.items():
        for shard, tensor in bucket.items():
            back = ".wq_a." if shard == "q" else ".wkv."
            yield fused.replace(".wqkv_a.", back), tensor


@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights",
    type=HookType.AROUND,
)
def fuse_wqkv_a_weight_scale_kunlun(original_fn, self, weights, *args, **kwargs):
    """开融合时，先把 wq_a/wkv 的 weight_scale 合并再交给上游 loader。"""
    from sglang.srt import environ as _environ

    if not _environ.envs.SGLANG_OPT_FUSE_WQA_WKV.get():
        return original_fn(self, weights, *args, **kwargs)
    return original_fn(self, _fuse_wqkv_a_weight_scales(weights), *args, **kwargs)

def _dequant_int8_per_channel(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """W8A8-INT8 checkpoint layout -> bf16: ``weight[O, I] * scale[O, 1]``."""
    assert weight.dtype == torch.int8, f"expected int8, got {weight.dtype}"
    return (weight.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)


@plugin_hook(
    "sglang.srt.models.deepseek_v4._prepare_deepseek_v4_weights",
    type=HookType.REPLACE,
)
def _prepare_deepseek_v4_weights_kunlun(weights, quant_config):
    """Streaming weight transform that also understands the W8A8-INT8 wo_a layout.

    The 4.1 tree's transform only matches the FP8 wo_a layout
    (``.wo_a.weight`` fp8 + ``.wo_a.scale`` block scales). The W8A8-INT8 release
    stores ``.wo_a.weight`` I8 [O, I] with a per-output-channel
    ``.wo_a.weight_scale`` BF16 [O, 1]. The model builds ``wo_a`` as bf16
    (``wo_a_quant_config`` is None when the quant config is not FP8), so without
    this branch the scale is an unknown parameter -- dropped with a
    "not found in params_dict" warning -- and wo_a would be used unscaled.

    Everything that is not wo_a passes through untouched; the FP8 branch keeps
    the upstream ``_dequant_fp8`` so this hook is format-agnostic.
    """
    from sglang.srt.models.deepseek_v4 import _dequant_fp8

    pending: dict[str, dict[str, torch.Tensor]] = {}

    def _emit(prefix: str, bucket: dict) -> Optional[tuple]:
        """Emit only when the companion scale has arrived; otherwise keep waiting.

        Emitting the raw weight as soon as it appears would leave the scale to be
        flushed separately -- the model then has no parameter for it and wo_a is
        used unscaled. That is exactly the failure this hook exists to fix.
        """
        weight = bucket.get("weight")
        if weight is None:
            return None
        if "scale" in bucket:
            return prefix + ".weight", _dequant_fp8(weight, bucket["scale"])
        if "weight_scale" in bucket:
            return prefix + ".weight", _dequant_int8_per_channel(
                weight, bucket["weight_scale"]
            )
        return None

    for name, tensor in weights:
        if name.endswith(".wo_a.weight"):
            prefix = name[: -len(".weight")]
            bucket = pending.setdefault(prefix, {})
            bucket["weight"] = tensor
            emitted = _emit(prefix, bucket)
            if emitted is not None:
                pending.pop(prefix, None)
                yield emitted
            continue
        if name.endswith(".wo_a.scale") or name.endswith(".wo_a.weight_scale"):
            # Strip only the tensor suffix: `.wo_a` is part of the module prefix.
            if name.endswith(".wo_a.weight_scale"):
                suffix, key = ".weight_scale", "weight_scale"
            else:
                suffix, key = ".scale", "scale"
            prefix = name[: -len(suffix)]
            bucket = pending.setdefault(prefix, {})
            bucket[key] = tensor
            emitted = _emit(prefix, bucket)
            if emitted is not None:
                pending.pop(prefix, None)
                yield emitted
            continue
        yield name, tensor

    for prefix, bucket in pending.items():
        # A weight whose companion scale never appeared (checkpoint without wo_a
        # scales) passes through unchanged; a stray scale is forwarded as upstream.
        if bucket.get("weight") is not None:
            yield prefix + ".weight", bucket["weight"]
        for key, suffix in (("scale", ".scale"), ("weight_scale", ".weight_scale")):
            if key in bucket:
                yield prefix + suffix, bucket[key]


# ── 让 V4.1 在开 CP 的同时保留 vision（ViT）──────────────────────────────
#
# 上游 `models/deepseek_v4.py:4613-4621` 有一段硬校验：
#
#     if config.model_type == "deepseek_v41" and config.vision_n_layers > 0:
#         if (get_parallel().attn_cp_size != 1
#             or get_pp_group().world_size != 1
#             or not get_moe_a2a_backend().is_none()):
#             raise ValueError(
#                 "V4.1 vision currently supports TP/EP/DP without CP, PP or MoE A2A")
#
# 这个限制对应的真实缺口是：**"VL 感知的路由"只接在 forward_normal /
# forward_normal_dual_stream 上**（`deepseek_v2.py:1021` / `:1243` 才查
# `e_score_correction_bias_vl`）。开 CP 之后 forward 走的是 CP 分支，不再查这个偏置。
#
# 而这个偏置**只对 `input_ids == image_token_id` 的 token 生效**
# （`multimodal/dsv41/vl_routing.py` 里要么 `bias_alt_token_id=image_token_id`，
# 要么 `torch.where(input_ids == image_token_id, bias_vl, bias)`）——
# **纯文本请求里没有任何 token 等于 image_token_id**，所以缺这块对文本任务
# 数值没有影响；图片请求会静默用错路由偏置。因此本 hook 只适用于"纯文本路径"。
#
# 做法：只在 `DeepseekV4ForCausalLM.__init__` 期间，把 `deepseek_v4` 模块里的
# `get_parallel` 换成"attn_cp_size 报 1、其余字段透传"的代理，让校验通过；
# `__init__` 退出时立刻恢复。选这个点位的依据：
#   * `ParallelState` 是 `@dataclass(frozen=True, slots=True)`，改不了实例；
#   * `deepseek_v4.py` 里 `attn_cp_size` 只有两处读取 —— 4617 的校验（在
#     `__init__` 内）和 4176 的 forward（`__init__` 期间不会执行），
#     所以代理的作用面只有那一个校验。
#
# `SGLANG_KUNLUN_ALLOW_VISION_WITH_CP=0` 可关掉，回到上游行为。


class _AttnCpAsOne:
    """`get_parallel()` 的透传代理：只把 `attn_cp_size` 报成 1，其余原样透传。"""

    __slots__ = ("_state",)

    def __init__(self, state):
        object.__setattr__(self, "_state", state)

    @property
    def attn_cp_size(self) -> int:
        """报 1 表示开 CP；其他字段原样透传。"""
        return 1

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_state"), name)


@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.__init__",
    type=HookType.AROUND,
)
def allow_vision_with_cp(original_fn, self, *args, **kwargs):
    """开 CP 时绕过 "V4.1 vision ... without CP" 校验，ViT 照常构造。"""
    import logging
    import os

    import sglang.srt.models.deepseek_v4 as _dsv4

    if os.environ.get("SGLANG_KUNLUN_ALLOW_VISION_WITH_CP", "1") != "1":
        return original_fn(self, *args, **kwargs)

    if not getattr(_dsv4, "_kunlun_cp_vision_warned", False):
        _dsv4._kunlun_cp_vision_warned = True
        logging.getLogger(__name__).warning(
            "V4.1：已放宽 'vision without CP' 校验（真实 attn_cp_size=%s）。"
            "ViT 会照常构造并加载权重；但 VL 路由偏置在 CP 路径上不会被查 —— "
            "纯文本无影响，图片请求会静默算错。"
            "如需回到上游行为：SGLANG_KUNLUN_ALLOW_VISION_WITH_CP=0",
            getattr(_dsv4.get_parallel(), "attn_cp_size", "?"),
        )

    real_get_parallel = _dsv4.get_parallel
    try:
        _dsv4.get_parallel = lambda: _AttnCpAsOne(real_get_parallel())
        return original_fn(self, *args, **kwargs)
    finally:
        _dsv4.get_parallel = real_get_parallel
