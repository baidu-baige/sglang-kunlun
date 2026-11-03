"""Method hooks for the upstream DeepSeek V4 model; no model classes are copied."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels.kernel_ops import dsv4_mqa_wo_a_einsum_kunlun

from .mtp_input_correction import strip_correction_blocks


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
    q_rope: Optional[torch.Tensor] = None,
) -> None:
    """Normalize and rotate KV, then use the dirty 0.5.14 direct writer.

    0.5.22 adds `q_rope` ([T, H, head_dim]): upstream lets the same fused
    norm+rope+store launch also rope the query heads along the way. Its only call
    site is `q_rope=q if fuse_q_rope else None` in `_forward_prepare`, and
    `fuse_q_rope` can only be set true by `_hip.compute_q_b` (the non-HIP branch
    hardcodes `fuse_q_rope = False`); `fused_k_norm_rope_flashmla` also asserts
    `assert _is_hip, "only the ROCm K launch ropes q"` while the XPU branch
    directly does `assert q is None`. So on Kunlun it is always None. Fusing it
    into the same launch is not possible, but the "query heads are roped on
    return" contract can still be satisfied: attach it to the q slot of the
    `fused_rope_inplace` call below (the XPU backend already does two triton rope
    calls for q and k), instead of silently dropping a tensor whose Q was not
    rotated.
    """
    if qkv_a is not None:
        kv = qkv_a[..., self.q_lora_rank :]
    else:
        kv, _ = self.wkv(x)
    kv = self.kv_norm(kv)

    from sglang.srt.models.deepseek_v4 import fused_rope_inplace

    kv_rope = kv[..., -self.qk_rope_head_dim :].unsqueeze(1)
    if q_rope is None:
        fused_rope_inplace(kv_rope, None, self.freqs_cis, positions)
    else:
        # 0.5.22 q_rope: rotate Q and KV together (one launch upstream, two q/k args here).
        fused_rope_inplace(
            q_rope[..., -self.qk_rope_head_dim :],
            kv_rope,
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

    0.5.22 adds `q_rope_out` / `k_nope_out` / `k_rope_out`: under the unified_kv
    fp8 dual-pool (nope + rope two buffers) layout, upstream lets
    `fused_qk_norm_rope_swa_store` write the packed Q and K directly into these
    three buffers preallocated by the caller (`MQALayer.forward` only
    `new_empty`s them in the `unified_fp8_decode or unified_fp8_prefill` branch,
    and always passes them by keyword). This path must go through the upstream
    implementation, so all three are forwarded as-is to `original_fn`; the Kunlun
    fast path is itself excluded by `is_unified_kv_triton()`, so getting a
    non-None can only mean the upstream gate condition changed -- better to blow
    up than to silently drop the packed Q/K output.
    """
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
        is_unified_kv_triton,
    )

    # 0.5.22 absorbed this logic: ``MQALayer`` no longer snapshots
    # ``dsa_enable_prefill_cp`` at construction time (that field now only lives
    # on ``DeepseekV4DecoderLayer``); upstream ``_forward_prepare`` itself is
    # just one line ``use_cp = is_cp_active(forward_batch)``
    # (``deepseek_v4.py:1846``). Reuse the same check, otherwise reading an
    # attribute that no longer exists on MQALayer raises AttributeError on the
    # first prefill batch.
    from sglang.srt.layers.cp.utils import is_cp_active

    use_cp = is_cp_active(forward_batch)

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

    # The 0.5.22 unified_kv fp8 dual-pool output buffers are only meaningful on
    # the original_fn branch above (they are already routed away when
    # `is_unified_kv_triton()` is true). If they are still non-None here, the
    # upstream gate condition changed; silently ignoring them would let the
    # caller run attention / ring store on a buffer nobody ever wrote.
    if q_rope_out is not None or k_nope_out is not None or k_rope_out is not None:
        raise NotImplementedError(
            "Kunlun 的 _forward_prepare 快路径不产出 unified_kv fp8 的打包 Q/K "
            "(q_rope_out/k_nope_out/k_rope_out)；请用 SGLANG_DSV4_UNIFIED_KV_FP8=0，"
            "或让这些 batch 走上游实现。"
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
    # Per-head weightless RMSNorm on Q is V4-specific. Upstream controls it with
    # `config.q_head_norm` (the `fused_q_norm_rope` branch of `_compute_q_b`);
    # V4 is True, **V4.1 is False**. The reference implementation shipped with
    # the checkpoint also only normalizes the low-rank q_lora and never touches
    # per-head Q. Doing it unconditionally normalizes every layer's Q one extra
    # time (layer 0 measured |Q|max 31.4 → 11.1), leaving all outputs wrong
    # while the magnitude looks normal.
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

    logits go through **fp32**, matching all upstream CUDA paths
    (`tiny_gemm_bf16(out_dtype=fp32)` / `linear_bf16_fp32`) and the reference
    implementation (`Gate.forward`: `linear(x.float(), weight.float())`). V4.1
    has 384 experts + top-6; fp16 only has about 3 decimal significant digits,
    so `sqrtsoftplus` scores that are close together can pick the wrong expert.
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


# The hc_post REPLACE hook is provided uniformly by deepseek_v4_precision.py
# (that copy additionally has an fp16 nan_to_num fallback), so it is not
# registered again here.


# Quantization name mapping is installed at the loader consumption point by
# hooks/model_loader/quant_injection.py, including the 0.5.22 getter API.


# The problem that the fused wqkv_a cannot get the per-channel weight_scale.
#
# The upstream wq_a+wkv fusion branch (`elif fuse_wqa_wkv and (...)` in
# sglang/srt/models/deepseek_v4.py) only recognizes ``.weight`` and
# ``.weight_scale_inv`` (fp8 block-quant naming), not the compressed-tensors
# per-channel ``.weight_scale``. So these two scales fall into the generic
# branch and get renamed to ``self_attn.wq_a.weight_scale``, while the model
# only has the fused ``wqkv_a.weight_scale``; the log reports
# "not found in params_dict", the fused module's scale keeps its initial value
# => int8 dequant produces garbage => logits become garbage => decoding produces
# an empty string. Note that ``.weight`` itself is correctly cat-fused; only the
# scale is missing, and loading does not error out, making it hard to detect.
#
# Here we do not change upstream; instead, before entering load_weights we merge
# the two scales in the weight stream first and hand them out directly under the
# fused name ``...wqkv_a.weight_scale``, so the upstream generic branch can find
# it in params_dict and load it normally. The merge is exactly like ``.weight``:
# a per-channel scale is per-output-channel, and fusion concatenates output
# channels in q, kv order, so it is likewise ``torch.cat([q, kv], dim=0)``.
_WQKV_A_SCALE_SHARDS = {".wq_a.weight_scale": "q", ".wkv.weight_scale": "kv"}


def _fuse_wqkv_a_weight_scales(weights):
    """Merge the wq_a / wkv weight_scales into the fused wqkv_a one."""
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
    # If only half of a pair arrived, hand it back under its original name and
    # let upstream report "not found" via its own logic; do not swallow it
    # silently.
    for fused, bucket in pending.items():
        for shard, tensor in bucket.items():
            back = ".wq_a." if shard == "q" else ".wkv."
            yield fused.replace(".wqkv_a.", back), tensor


@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights",
    type=HookType.AROUND,
)
def fuse_wqkv_a_weight_scale_kunlun(original_fn, self, weights, *args, **kwargs):
    """When fusion is on, merge the wq_a/wkv weight_scales before handing them to the upstream loader."""
    from sglang.srt import environ as _environ

    # Some checkpoints ship an MTP input correction (QuaRot/R1 rotation
    # correction, applied to the target hidden that the draft receives). Upstream
    # does not recognize this module and drops it with
    # "model.norm.mtp_input_correction.blocks not found in params_dict." This
    # hook only covers models going through the DeepseekV4 loader: strip the
    # correction blocks from the weight stream and attach them to the model, then
    # apply them at the draft-extend boundary via mtp_production (or via NextN's
    # own rot_weight slot).
    weights = strip_correction_blocks(self, weights)
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


# ── Let V4.1 keep vision (ViT) while CP is enabled ─────────────────────────
#
# Upstream `models/deepseek_v4.py:4613-4621` has a hard check:
#
#     if config.model_type == "deepseek_v41" and config.vision_n_layers > 0:
#         if (get_parallel().attn_cp_size != 1
#             or get_pp_group().world_size != 1
#             or not get_moe_a2a_backend().is_none()):
#             raise ValueError(
#                 "V4.1 vision currently supports TP/EP/DP without CP, PP or MoE A2A")
#
# The real gap behind this restriction is that **"VL-aware routing" is only
# wired into forward_normal / forward_normal_dual_stream** (only
# `deepseek_v2.py:1021` / `:1243` look up `e_score_correction_bias_vl`). Once CP
# is enabled, forward goes through the CP branch and no longer looks up this
# bias.
#
# And this bias **only takes effect for tokens where `input_ids ==
# image_token_id`** (in `multimodal/dsv41/vl_routing.py`, either
# `bias_alt_token_id=image_token_id`, or
# `torch.where(input_ids == image_token_id, bias_vl, bias)`). **A pure-text
# request has no token equal to image_token_id**, so missing this piece has no
# numerical impact on text tasks; image requests would silently use the wrong
# routing bias. Therefore this hook only applies to the "pure-text path".
#
# 2026-10-22 update: the "only applies to the pure-text path" statement above no
# longer holds -- both gaps on the image path are patched in
# hooks/model_executor/cp_mm_embeds.py (merging image features under CP, and
# restoring mm hash ids to image_token_id so the VL bias selects image tokens).
# The full 900 MMMU examples already run in the normal range; see
# 文档/DSV41/改动说明.md issues 4/5.
#
# Approach: only during `DeepseekV4ForCausalLM.__init__`, replace `get_parallel`
# in the `deepseek_v4` module with a proxy that "reports attn_cp_size as 1 and
# passes everything else through", and replace `get_moe_a2a_backend` with a proxy
# that "reports is_none() as True and passes everything else through" (see
# `A2aAsNone`; it is required on the D side when deepep is enabled), so the check
# passes; restore immediately when `__init__` exits. The rationale for this
# injection point:
#   * `ParallelState` is `@dataclass(frozen=True, slots=True)`, so the instance
#     cannot be modified;
#   * in `deepseek_v4.py`, `attn_cp_size` is read in only two places -- the check
#     at 4617 (inside `__init__`) and the forward at 4176 (not executed during
#     `__init__`), so the proxy's blast radius is just that one check.
#
# `SGLANG_KUNLUN_ALLOW_VISION_WITH_CP=0` disables this and returns to upstream
# behavior.


class _AttnCpAsOne:
    """Pass-through proxy for `get_parallel()`: only reports `attn_cp_size` as 1,
    passes everything else through unchanged.
    """

    __slots__ = ("_state",)

    def __init__(self, state):
        object.__setattr__(self, "_state", state)

    @property
    def attn_cp_size(self) -> int:
        """Report 1 to indicate CP is on; pass all other fields through unchanged."""
        return 1

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_state"), name)


class A2aAsNone:
    """Pass-through proxy for `get_moe_a2a_backend()`: only reports `is_none()` as True.

    Why it also needs proxying: the third condition of the same vision gate is
    `not get_moe_a2a_backend().is_none()`, which **is hit on the D side when
    deepep is enabled** (D is dp8 -> attn_cp_size=1, so the attn_cp proxy above
    does not help). Measured: when only attn_cp is allowed through, all 8
    schedulers on the D side hit
    `ValueError: V4.1 vision currently supports TP/EP/DP without CP, PP or MoE A2A`.

    Why this replacement is safe -- the key is the **module namespace**:
      * `deepseek_v4.py:108` is `from sglang.srt.layers.moe import get_moe_a2a_backend`,
        so replacing `deepseek_v4.get_moe_a2a_backend` only affects reads inside
        that module;
      * the real place that decides what MoE to build based on the a2a backend is
        `EPMoE.__init__` (`layers/moe/ep_moe/layer.py:106,142`), which uses the
        same-named binding **in its own module** and is unaffected, so deepep's
        dispatcher is still constructed normally;
      * the other 10 `get_moe_a2a_backend()` calls in `deepseek_v4.py` are all in
        methods on the forward path (`_run_moe_ffn_dp_sync` /
        `op_mhc_prepare_attn`, etc.), which are not executed during `__init__`.
    Together: the proxy's blast radius is just that one gate condition.
    """

    __slots__ = ("_state",)

    def __init__(self, state):
        object.__setattr__(self, "_state", state)

    def is_none(self) -> bool:
        """Report True for is_none(); pass everything else through."""
        return True

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_state"), name)


@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.__init__",
    type=HookType.AROUND,
)
def allow_vision_with_cp(original_fn, self, *args, **kwargs):
    """Bypass the "V4.1 vision ... without CP" check when CP is on, so ViT is constructed normally."""
    import logging
    import os

    import sglang.srt.models.deepseek_v4 as _dsv4

    if os.environ.get("SGLANG_KUNLUN_ALLOW_VISION_WITH_CP", "1") != "1":
        return original_fn(self, *args, **kwargs)

    if not getattr(_dsv4, "_kunlun_cp_vision_warned", False):
        _dsv4._kunlun_cp_vision_warned = True
        logging.getLogger(__name__).warning(
            "V4.1：已放宽 'vision without CP/A2A' 校验"
            "（真实 attn_cp_size=%s、moe_a2a_backend=%s）。"
            "ViT 会照常构造并加载权重。放行本身带出的两个缺口已由 "
            "hooks/model_executor/cp_mm_embeds.py 补上："
            "(1) CP 下 _execute_extend_cp 不合并图片特征；"
            "(2) VL 路由偏置因为 mm hash id 而选不中图片 token。"
            "如需回到上游行为：SGLANG_KUNLUN_ALLOW_VISION_WITH_CP=0",
            getattr(_dsv4.get_parallel(), "attn_cp_size", "?"),
            _dsv4.get_moe_a2a_backend(),
        )

    real_get_parallel = _dsv4.get_parallel
    real_get_a2a = _dsv4.get_moe_a2a_backend
    try:
        _dsv4.get_parallel = lambda: _AttnCpAsOne(real_get_parallel())
        _dsv4.get_moe_a2a_backend = lambda: A2aAsNone(real_get_a2a())
        return original_fn(self, *args, **kwargs)
    finally:
        _dsv4.get_parallel = real_get_parallel
        _dsv4.get_moe_a2a_backend = real_get_a2a
