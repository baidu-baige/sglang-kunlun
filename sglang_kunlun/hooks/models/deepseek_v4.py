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
"""Hooks for DeepseekV4DecoderLayer methods (hc_post, hc_pre)."""
from __future__ import annotations

import logging
import os
from typing import Literal

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.layers.engram import EngramEmbedding, EngramHasher
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
import kunlun_ops
from types import SimpleNamespace

from sglang_kunlun.hooks.layers.xsg_cast import xsg_cast
from sglang_kunlun.hooks.utils import engram_precompute

logger = logging.getLogger(__name__)

_BF16 = torch.bfloat16


def _b16(t: torch.Tensor) -> torch.Tensor:
    """Return ``t`` as contiguous bf16, skipping the dispatch when it already is.

    The hc residual stream alternates dtype by call site, so the unconditional
    ``t.to(bf16).contiguous()`` used to fire two dispatches per operand per layer
    (``aten::to`` alone is ~74 us of host time on this XPU stack) even when the
    tensor was already bf16+contiguous.

    The fp16 -> bf16 direction goes through the native converter (818 GB/s on
    the vendor ``cluster_cast_kl3`` at [T, 4, 5120], bit-identical to ``.to``).
    """
    if t.dtype is not _BF16:
        t = xsg_cast(t, _BF16)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post",
    type=HookType.REPLACE,
)
def hc_post_kunlun(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    """Kunlun-native hc_post via ``kunlun_ops.hc_post_kunlun_impl``.

    Computes ``out = post * x + comb @ residual`` (fused).
    Only N == 4 (hc_mult) is supported on XPU.

    [r40] Rejected: ``xspeedgate_ops.mhc_post`` takes the fp32 gates directly
    (better precision, 6.6e-5 vs 4.2e-3 relative), but its kernel
    ``infer_mhc_post<bf16>`` costs 0.354 ms/call against 0.299 for
    ``hc_post_kunlun_impl<bf16,bf16>``, i.e. +4.1 ms/step for 80 calls/step,
    while the "5 casts" are only the tiny [T,4] and [T,4,4] gate tensors (the
    activations are already bf16, so ``.to(bf16)`` on them is a no-op).  The
    candidate was a phantom; this path stays.
    """
    if x.shape[0] == 0:
        return torch.empty(
            (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
        )

    batch = x.shape[0]
    N = self.hc_mult
    D = x.shape[-1]
    # kunlun_ops.hc_post_kunlun_impl is bfloat16-only, but the DSV4.1 hc residual
    # stream runs in the model dtype (float16). Bridge locally: cast the operands
    # to bf16 for the fused kernel, then cast the result back to the model dtype.
    bf16 = torch.bfloat16
    orig_dtype = x.dtype
    out = torch.empty((batch, N, D), dtype=bf16, device=x.device)
    kunlun_ops.hc_post_kunlun_impl(
        _b16(post),
        _b16(x),
        _b16(comb),
        _b16(residual),
        out,
        batch, N, D,
    )
    return out if orig_dtype == bf16 else xsg_cast(out, orig_dtype)


# --------------------------- INT4 checkpoint support ---------------------
# The compressed-tensors targets/ignore lists in the INT4 checkpoint use the
# original HF module names (`attn`/`ffn`/`w1..w3`), while the weights
# themselves are remapped by `remap_weight_name_to_dpsk_hf_format`. The mapper
# is picked up by the model loader to remap the quantization config. The keys
# look like regexes because the checkpoint targets are regex strings; 0.5.14
# WeightsMapper still matches them as plain substrings.
DeepseekV4ForCausalLM.packed_modules_mapping = {
    "wqkv_a": ["wq_a", "wkv"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

DeepseekV4ForCausalLM.hf_to_sglang_mapper = WeightsMapper(
    orig_to_new_substr={
        r"\.attn\.": r"\.self_attn\.",
        r"\.ffn\.": r"\.mlp\.",
        r"\.attn_norm": r"\.input_layernorm",
        r"\.ffn_norm": r"\.post_attention_layernorm",
    },
    orig_to_new_suffix={
        r"\.w1$": r"\.gate_proj$",
        r"\.w2$": r"\.down_proj$",
        r"\.w3$": r"\.up_proj$",
    },
)


def _fuse_wqkv_a_weight_scale(weights):
    """Concat the ``wq_a``/``wkv`` per-channel scales for the fused ``wqkv_a``.

    Upstream `load_weights` already fuses ``.weight`` and ``.weight_scale_inv``;
    the int4 checkpoint instead carries ``.weight_scale``, which is fused here
    so the generic loader path picks it up as a plain ``wqkv_a.weight_scale``.
    """
    pending = {}
    for name, tensor in weights:
        is_q = name.endswith(".wq_a.weight_scale")
        is_kv = name.endswith(".wkv.weight_scale")
        # Engram owns its own ``engram.wkv`` int8 linear whose scale is NOT part
        # of the self_attn wq_a+wkv fusion (engram has no ``wq_a``). Pass engram
        # scales through untouched — mirrors upstream load_weights' own
        # ``".engram." not in name`` guard — so ``engram.wkv.weight_scale`` reaches
        # its own ReplicatedLinear param instead of being left unpaired (was a
        # fatal ``AssertionError: unpaired wqkv_a weight scales`` at load).
        if ".engram." in name or (not is_q and not is_kv):
            yield name, tensor
            continue
        fused_name = name.replace(".wq_a." if is_q else ".wkv.", ".wqkv_a.")
        bucket = pending.setdefault(fused_name, {})
        shard: Literal["q", "kv"] = "q" if is_q else "kv"
        assert shard not in bucket, f"duplicate shard {shard} for {fused_name}"
        bucket[shard] = tensor
        if len(bucket) == 2:
            yield fused_name, torch.cat([bucket["q"], bucket["kv"]], dim=0)
            pending.pop(fused_name)
    assert not pending, f"unpaired wqkv_a weight scales: {sorted(pending)}"


def _dequant_int8_wo_a(weights):
    """Fold the per-out-channel int8 ``wo_a`` (weight + weight_scale) into bf16.

    ``wo_a`` is applied as a grouped bf16 einsum that reads ``self.wo_a.weight``
    directly (upstream ``MqaAttentionBase.forward``), so for W8A8-INT8 the linear
    is built unquantized (``params_dtype=bfloat16``) and owns no ``weight_scale``.
    The checkpoint stores int8 ``attn.wo_a.weight`` [out, in] + per-out-channel
    bf16 ``attn.wo_a.weight_scale`` [out, 1] (symmetric, no zero point). Without
    dequant the int8 weight is silently cast to bf16 (~scale x amplitude, the
    degenerate-output bug) and the scale lands as an unmatched param
    (``self_attn.wo_a.weight_scale not found in params_dict`` x40). Fold
    ``weight * weight_scale`` -> bf16 here (int8 analogue of upstream's fp8
    ``_dequant_fp8_wo_a_streaming``); the full [out, in] weight is produced before
    upstream's ColumnParallel weight_loader shards the out dim, so TP stays correct.
    """
    pending: dict = {}

    def _emit(prefix, bucket):
        w = bucket["weight"].to(torch.float32)
        s = bucket["scale"].to(torch.float32).reshape(w.shape[0], *([1] * (w.ndim - 1)))
        return prefix + ".weight", (w * s).to(torch.bfloat16)

    for name, tensor in weights:
        if name.endswith(".wo_a.weight"):
            prefix = name[: -len(".weight")]
            bucket = pending.setdefault(prefix, {})
            bucket["weight"] = tensor
            if "scale" in bucket:
                yield _emit(prefix, bucket)
                pending.pop(prefix)
            continue
        if name.endswith(".wo_a.weight_scale"):
            prefix = name[: -len(".weight_scale")]
            bucket = pending.setdefault(prefix, {})
            bucket["scale"] = tensor
            if "weight" in bucket:
                yield _emit(prefix, bucket)
                pending.pop(prefix)
            continue
        yield name, tensor
    # A bf16 checkpoint (weight only, no scale) passes straight through.
    for prefix, bucket in pending.items():
        if "weight" in bucket:
            yield prefix + ".weight", bucket["weight"]
        if "scale" in bucket:
            yield prefix + ".weight_scale", bucket["scale"]


def _rename_engram_embed_scale(weights):
    """Route the int8 engram embed per-row scale to the module's ``scale`` param.

    Checkpoint stores ``engram.embed.weight_scale`` (bf16 [N,1]); the re-declared
    param is ``scale`` and upstream never rewrites ``engram.embed`` scales, so
    rename here to bind it via ``_load_rows``.
    """
    for name, tensor in weights:
        if name.endswith(".engram.embed.weight_scale"):
            yield name[: -len(".weight_scale")] + ".scale", tensor
        else:
            yield name, tensor


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights",
    type=HookType.AROUND,
)
def load_weights_kunlun(original_fn, self, weights, is_nextn=False):
    """Pre-process the int4/int8 weight stream, then defer to upstream."""
    weights = _dequant_int8_wo_a(weights)
    weights = _rename_engram_embed_scale(weights)
    if envs.SGLANG_OPT_FUSE_WQA_WKV.get():
        weights = _fuse_wqkv_a_weight_scale(weights)
    return original_fn(self, weights, is_nextn)


@plugin_hook(
    target="sglang.srt.layers.engram.EngramEmbedding.__init__",
    type=HookType.AROUND,
)
def engram_embedding_init_kunlun(original_fn, self, num_embeddings, dim, layer_id):
    """Hold the int8 checkpoint engram table (int8 weight + per-row bf16 scale)
    instead of the pinned fp8 e4m3 payload + fp8 e8m0 block scale.

    DSV4.1-Flash W8A8-INT8 ships engram.embed as int8 [N,256] + per-row bf16
    scale [N,1]. The pinned module hard-codes fp8, which has no matching scale
    tensor AND exercises fp8 gather/convert on XPU (torch_xmlir has no fp8 cast
    -> NOT_IMPLEMENTED / native fault). Re-declare the params so the lookup runs
    in int8/bf16/f32 only; the original __init__ still computes row-sharding.
    """
    original_fn(self, num_embeddings, dim, layer_id)
    if getattr(self, "host_table", None) is not None:
        return  # host-table layout is the CUDA-only fp8 engram_gather path
    # Kunlun: zero-init (NOT torch.empty). The engram n-gram hash indexes a
    # sparse subset of this rank's rows; rows never touched by _load_rows stay
    # at their init value and, after the TP all-reduce in EngramEmbedding._lookup,
    # any non-zero garbage there is summed into the hidden state -> large,
    # run-to-run non-deterministic corruption of the first tokens. Zeroing makes
    # unloaded rows contribute 0 (correct "row not present on this rank").
    self.weight = torch.nn.Parameter(
        torch.zeros(self.rows, dim, dtype=torch.int8), requires_grad=False
    )
    self.scale = torch.nn.Parameter(
        torch.zeros(self.rows, 1, dtype=torch.bfloat16), requires_grad=False
    )
    self.weight.weight_loader = self._load_rows
    self.scale.weight_loader = self._load_rows


@plugin_hook(
    target="sglang.srt.layers.engram.EngramEmbedding._owned_rows",
    type=HookType.REPLACE,
)
def engram_owned_rows_kunlun(self, indices):
    """int8 * per-row bf16 symmetric dequant of this rank's owned rows -> bf16.

    Uses the native ``xspeedgate_ops.engram_gather`` (built for exactly this
    int8 [rows,dim] + per-row bf16 scale layout; block_size==dim is the per-row
    scale sentinel, row_lo/row_hi do the TP owned-row masking). Validated
    bit-identical to the torch dequant, which stays as the fallback.
    """
    if self.rows == 0:
        return self._empty(indices).zero_()
    dim = self.weight.shape[1]
    try:
        out = torch.ops.xspeedgate_ops.engram_gather(
            self.weight, self.scale, indices.reshape(-1).contiguous(),
            dim, dim, self.row_start, self.row_start + self.rows,
            torch.bfloat16,
        )
        return out.reshape(*indices.shape, dim)
    except Exception:
        local = indices - self.row_start
        owned = (local >= 0) & (local < self.rows)
        local = local.masked_fill(~owned, 0)
        w = self.weight[local].to(torch.float32)
        s = self.scale[local].to(torch.float32)
        values = (w * s).to(torch.bfloat16)
        return values.masked_fill(~owned.unsqueeze(-1), 0)


_ENGRAM_TRACE = os.environ.get("KL_ENGRAM_TRACE", "0") == "1"
# [r36] The op rejects fp16 weights (``unsupported (io, weight) dtype
# combination``) but the torch reference it replaces computes ``q.float() *
# k.float()`` anyway, so feeding it fp32 copies of the fp16 gate weights is an
# exact upcast: 99.997% of the output is bit-identical to the torch reference,
# the rest differs only by fp32 reduction order (mean |d| 4e-8). The copies are
# cached per Parameter (the cache holds a reference, so the ids cannot be
# recycled under it) and refreshed whenever the data pointer moves.
_ENGRAM_NATIVE = os.environ.get("KL_ENGRAM_NATIVE", "1") == "1"
try:  # the torch.ops namespace only exists once the package has been imported
    import xspeedgate_ops as _xsg_ops  # noqa: F401

    _XSG_OK = True
except Exception:  # noqa: BLE001  pragma: no cover
    _XSG_OK = False
_engram_seen: set = set()
_gate_w_cache: list = []
# [r134] Fold q*k into one 16-bit weight so the op takes its token kernel. See
# `_gate_weights_16`. `KL_ENGRAM_GATE_W16=0` restores the fp32-weight call.
_ENGRAM_GATE_W16 = os.environ.get("KL_ENGRAM_GATE_W16", "1") == "1"
_engram_gate_w16_state = None


def _gate_w16_on() -> bool:
    global _engram_gate_w16_state
    on = _ENGRAM_GATE_W16 and _XSG_OK
    if on != _engram_gate_w16_state:
        _engram_gate_w16_state = on
        print(
            "sglang-kunlun: engram gate weights = %s"
            % ("w16 fold (token kernel)" if on else "fp32 (row kernel)")
        )
    return on


def _gate_weights_f32(q_weight, k_weight):
    """fp32 copies of the fp16 gate weights, cached per (q, k) Parameter pair."""
    for entry in _gate_w_cache:
        if entry[0] is q_weight and entry[1] is k_weight and entry[2] is None:
            return entry[3], entry[4]
    pair = (
        q_weight,
        k_weight,
        None,
        q_weight.to(torch.float32).contiguous(),
        k_weight.to(torch.float32).contiguous(),
    )
    _gate_w_cache.append(pair)
    return pair[3], pair[4]


def _gate_weights_16(q_weight, k_weight, dtype):
    """[r134] The weight product as ONE 16-bit tensor (+ an all-ones second
    operand, so `wgl = qw*kw` still evaluates to the folded product exactly).

    Why it matters: the op's 16-bit branch is `dispatch16`, and at T>=2048 that
    runs `infer_fused_engram_gate_token`. It requires BOTH operands to be the same
    16-bit dtype, but the production pose hands the op fp32 weights (the exact
    upcast of the fp16 Parameters), so it always lands on
    `infer_fused_engram_gate<bfloat16,float>`: 4.17 ms/call at the production
    geometry (T=7680, HC=4, D=5120) at ~245 GB/s, 4.7x off the token kernel's
    0.88 ms / 1.16 TB/s (memory roof) -- the row kernel re-runs a full scalar
    pass over D per row and runs its inner chunk at 384 instead of 1536.

    Cost: one rounding of the weight product to `dtype`. Measured: against a
    torch reference built from the same rounded weights, this arm matches as
    tightly (max |d| 1.6e-2, mean 5.8e-8, 0.005% of elements differ) as the fp32
    arm matches the fp32 reference (max |d| 1.6e-2, mean 1.2e-7, 0.011%), i.e.
    the token kernel is not the source of any error. Against the fp32 arm the
    output moves by 4.9e-4 relative, 4x below bf16's own round-off (1.95e-3),
    with 99.2% of elements within 1 bf16 ulp. That bound is accepted for the
    release: the shipped image runs this default and the full accuracy set
    (10 datasets, incl. code/math) passed; `KL_ENGRAM_GATE_W16=0` restores the
    fp32-weight call and `KL_ENGRAM_NATIVE=0` the torch reference.
    """
    for entry in _gate_w_cache:
        if entry[0] is q_weight and entry[1] is k_weight and entry[2] == dtype:
            return entry[3], entry[4]
    wp = (
        (q_weight.to(torch.float32) * k_weight.to(torch.float32))
        .to(dtype)
        .contiguous()
    )
    pair = (q_weight, k_weight, dtype, wp, torch.ones_like(wp))
    _gate_w_cache.append(pair)
    return pair[3], pair[4]


def _engram_gate_reason(x, kv, q_weight, k_weight) -> str:
    """Why ``fused_engram_gate`` is not used (r34 trace shows the torch
    reference on every call: 64 kernels / 10.6 ms per step)."""
    if not _ENGRAM_NATIVE:
        return "KL_ENGRAM_NATIVE=0"
    if not _XSG_OK:
        return "xspeedgate_ops missing"
    if x.ndim != 3:
        return f"x.ndim={x.ndim}"
    want = (x.shape[0], (x.shape[1] + 1) * x.shape[2])
    if kv.shape != want:
        return f"kv.shape={tuple(kv.shape)} want={want}"
    if x.dtype != kv.dtype:
        return f"dtype x={x.dtype} kv={kv.dtype}"
    if x.dtype not in (torch.bfloat16, torch.float32):
        return f"x.dtype={x.dtype}"
    if q_weight.dim() != 2 or k_weight.dim() != 2:
        return f"weight-dim={q_weight.dim()}/{k_weight.dim()}"
    if q_weight.shape != k_weight.shape:
        return f"weight-shape={tuple(q_weight.shape)}/{tuple(k_weight.shape)}"
    if q_weight.dtype != k_weight.dtype:
        return f"weight-dtype={q_weight.dtype}/{k_weight.dtype}"
    if q_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"q_weight.dtype={q_weight.dtype}"
    bad = [
        n
        for n, t in (("x", x), ("kv", kv), ("q", q_weight), ("k", k_weight))
        if not t.is_contiguous()
    ]
    if bad:
        return f"non-contiguous={','.join(bad)}"
    return "op-raised"


def _log_engram_gate(x, kv, q_weight, k_weight, reason: str) -> None:
    key = (reason, tuple(x.shape), tuple(kv.shape))
    if key in _engram_seen:
        return
    _engram_seen.add(key)
    logger.warning(
        "sglang-kunlun: engram_gate fell back to torch (%s) x=%s/%s kv=%s/%s",
        reason,
        tuple(x.shape),
        x.dtype,
        tuple(kv.shape),
        kv.dtype,
    )


@plugin_hook(
    target="sglang.srt.layers.engram.engram_gate",
    type=HookType.REPLACE,
)
def engram_gate_kunlun(x, kv, q_weight, k_weight, eps, clamp_value):
    """Native XSpeedGate fused engram gate (validated against sglang's torch
    else-branch). Replaces the whole ``engram_gate`` so it does not depend on
    ``_cuda_kernels`` (the engram Triton path never runs on XPU); the torch
    reference below is the fallback for any unsupported shape/dtype.

    [r36] The gate weights ship as fp16, which the native op rejects, so it was
    silently falling back to the 64-kernel torch reference on every step
    (10.56 ms/step). fp32 copies of the weights are an exact upcast of what the
    reference computes, which brings the call to 0.54 ms.

    [r134] ... but fp32 weights also pin the op to `infer_fused_engram_gate<
    bfloat16,float>` (2 calls / 8.29 ms per forward, ~245 GB/s). Folding q*k into
    one bf16 tensor selects `infer_fused_engram_gate_token<bfloat16>` instead:
    0.88 ms/call, 1.16 TB/s, -6.58 ms/forward. See `_gate_weights_16`.
    """
    if (
        _ENGRAM_NATIVE
        and _XSG_OK
        and x.ndim == 3
        and kv.shape == (x.shape[0], (x.shape[1] + 1) * x.shape[2])
        and x.dtype == kv.dtype
        and x.dtype in (torch.bfloat16, torch.float32)
        and q_weight.dim() == 2 and k_weight.dim() == 2
        and q_weight.shape == k_weight.shape
        and q_weight.dtype == k_weight.dtype
        and q_weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.is_contiguous() for t in (x, kv, q_weight, k_weight))
    ):
        try:
            if _gate_w16_on() and x.dtype in (torch.bfloat16, torch.float16):
                wp, ones = _gate_weights_16(q_weight, k_weight, x.dtype)
                return torch.ops.xspeedgate_ops.fused_engram_gate(
                    x, kv, wp, ones, eps, clamp_value
                )
            q_f32, k_f32 = _gate_weights_f32(q_weight, k_weight)
            return torch.ops.xspeedgate_ops.fused_engram_gate(
                x, kv, q_f32, k_f32, eps, clamp_value
            )
        except Exception:
            if _ENGRAM_TRACE:
                _log_engram_gate(x, kv, q_weight, k_weight, "op-raised")
    elif _ENGRAM_TRACE:
        _log_engram_gate(
            x, kv, q_weight, k_weight,
            _engram_gate_reason(x, kv, q_weight, k_weight),
        )
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim ** -0.5
    gate = torch.sigmoid(
        torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot)
    )
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def _engram_fake_self(token_map, multipliers, primes, offsets, pad_id, image_token_id):
    """Minimal stand-in exposing the attrs EngramHasher._torch_hash_ids reads, so the
    shipped torch fallback can be reused from these module-level hooks (no ``self``)."""
    return SimpleNamespace(
        max_ngram_size=int(multipliers.shape[1]),
        image_token_id=image_token_id,
        pad_id=pad_id,
        token_map=token_map,
        multipliers=multipliers,
        primes=primes,
        offsets=offsets,
    )


@plugin_hook(
    target="sglang.srt.layers.engram.engram_hash_ids",
    type=HookType.AROUND,
)
def engram_hash_ids_native(
    original_fn, input_ids, positions, *, mode, history, token_map, multipliers,
    primes, offsets, pad_id, num_real=None, req_slots=None, block=1, row=None,
    starts=None, image_token_id=None, mm_pad_shift=0, block_t=32,
):
    """Native XSpeedGate ``engram_hash_ids`` (verify/extend). Validated bit-identical
    to the torch fallback for decode/extend; the original ``original_fn`` is the Triton
    kernel that SIGSEGVs on XPU, so the fallback re-uses the shipped torch path instead.
    """
    if engram_precompute.active():
        # Decode-graph capture/replay: the pre-replay pass already computed the
        # real ids, and the captured graph only needs this tensor's shape.
        num_tokens = int(input_ids.shape[0])
        return (
            engram_precompute.dummy_hash_ids(num_tokens),
            engram_precompute.dummy_verify_tokens(num_tokens),
        )
    nr = int(num_real) if num_real is not None else int(input_ids.shape[0])
    try:
        iids = input_ids.contiguous()
        hist = history.contiguous()
        if hist.dtype != iids.dtype:
            iids = iids.to(hist.dtype)
        return torch.ops.xspeedgate_ops.engram_hash_ids(
            iids, positions.to(torch.int32).contiguous(), int(mode), hist,
            token_map, multipliers, primes, offsets, int(pad_id), nr, req_slots,
            int(block), row, starts, image_token_id, int(mm_pad_shift),
        )
    except Exception:
        fs = _engram_fake_self(token_map, multipliers, primes, offsets, pad_id, image_token_id)
        gathered = history[req_slots] if req_slots is not None else history
        return EngramHasher._torch_hash_ids(
            fs, input_ids, positions, mode, gathered, nr, block, row, starts
        )


@plugin_hook(
    target="sglang.srt.layers.engram.EngramHasher.commit_after_verify",
    type=HookType.AROUND,
)
def engram_commit_history_kunlun(
    original_fn, self, verify_ids_2d, req_pool_indices, commit_lens
):
    """Native XSpeedGate ``engram_commit_history`` in-place history update (validated
    bit-identical to the torch ``commit_after_verify`` fallback). Only reached on the
    target-verify / spec-decode path. ``engram_force_torch_fallback`` is removed, so
    ``_cuda_kernels`` is True on XPU and the engine reaches this commit entry; the hook
    forces the native op, else falls back to the shipped torch commit.
    """
    h = self.history
    if h is not None and h.dtype == torch.int32 and h.dim() == 2 and h.stride(1) == 1:
        try:
            torch.ops.xspeedgate_ops.engram_commit_history(
                h,
                verify_ids_2d.to(torch.int32).contiguous(),
                req_pool_indices.contiguous(),
                commit_lens.to(torch.int32).contiguous(),
            )
            return None
        except Exception:
            pass
    return original_fn(self, verify_ids_2d, req_pool_indices, commit_lens)


@plugin_hook(
    target="sglang.srt.layers.engram.engram_hash_ids_and_commit",
    type=HookType.AROUND,
)
def engram_hash_ids_and_commit_native(
    original_fn, input_ids, positions, *, history, req_slots, out_cache_loc,
    token_map, multipliers, primes, offsets, pad_id, image_token_id=None,
    mm_pad_shift=0, block_t=32,
):
    """Native XSpeedGate decode hash + fused history commit (validated bit-identical
    to the torch path). Native ``engram_hash_ids`` (read-only) then scatter each live
    row's history to [token, n-2 newest predecessors] oldest-first; graph-padded rows
    (out_cache_loc == 0) are skipped, exactly like the Triton COMMIT mask.
    """
    if engram_precompute.active():
        # See engram_hash_ids_native: history is committed by the eager
        # pre-replay pass, never inside the captured graph.
        return engram_precompute.dummy_hash_ids(int(input_ids.shape[0]))
    W = int(history.shape[1])
    live = out_cache_loc != 0
    slots = req_slots.to(torch.int64)[live]
    try:
        iids = input_ids.contiguous()
        hist = history.contiguous()
        if hist.dtype != iids.dtype:
            iids = iids.to(hist.dtype)
        out, tokens = torch.ops.xspeedgate_ops.engram_hash_ids(
            iids, positions.to(torch.int32).contiguous(), 0, hist, token_map,
            multipliers, primes, offsets, int(pad_id), None, req_slots, 1,
            None, None, image_token_id, int(mm_pad_shift),
        )
        history[slots] = tokens[live][:, :W].flip(-1).to(history.dtype)
        return out
    except Exception:
        fs = _engram_fake_self(token_map, multipliers, primes, offsets, pad_id, image_token_id)
        nt = int(input_ids.shape[0])
        out, tokens = EngramHasher._torch_hash_ids(
            fs, input_ids, positions, 0, history[req_slots], nt, 1, None, None
        )
        history[slots] = tokens[live][:, :W].flip(-1).to(history.dtype)
        return out


@plugin_hook(
    target="sglang.srt.multimodal.dsv41.vl_routing.is_cuda",
    type=HookType.REPLACE,
)
def vl_routing_force_torch(*args, **kwargs):
    """Force the pure-torch MoE router in ``vl_routing.vision_topk`` on Kunlun.

    ``vision_topk`` (vl_routing.py:31) calls the Triton ``moe_fused_gate`` router
    when ``is_cuda()`` is True; torch_xmlir masquerades as CUDA so that path is
    taken and the ``_router_triton_kernel`` fails to compile on the XPU Triton
    backend. Returning False selects the shipped pure-torch routing else-branch
    (vl_routing.py:74-89), scoped to this module only (does not disturb the
    is_cuda-gated kunlun kernels elsewhere).
    """
    return False
