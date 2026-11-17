"""Kunlun hooks for ERNIE 5.5 (`sglang.srt.models.ernie55`)."""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

# MegaMoE symmetric buffer is all-to-all scratch reused every forward, so one
# buffer is shared across all MoE layers (keyed by its dims); otherwise each of
# the ~42 ERNIE MoE layers would allocate its own ~1.7GB buffer.
_MEGA_SYMM_BUFFERS = {}


def _mega_moe_cap():
    """Per-rank token capacity of the symmetric buffer (the op rounds it up).
    """
    try:
        return max(1, int(os.environ.get("SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK", "384")))
    except ValueError:
        return 384

# T gate. Decode is 1 (draft) or 2 (MTP target-verify); in prefill T is the
# prompt length and the block diagonal grows as T^2, so fall back above it.
_VHA_MAX_T = 4


def _block_diag(self, mix: torch.Tensor, t: int, dtype) -> torch.Tensor:
    """``diag(mix, ..., mix)``, cached on the module and cleared by ``fold``.

    Cached in ``o.dtype`` because ``torch.mm`` rejects the fp32 ``mix`` against
    a bf16 ``o`` (upstream's ``matmul`` silently runs the bf16 kernel); the copy
    is within 1 ULP, while promoting both sides to fp32 is twice as slow.
    """
    cache = getattr(self, "_kl_bd", None)
    if cache is None:
        cache = self._kl_bd = {}
    key = (t, dtype)
    bd = cache.get(key)
    if bd is None:
        h = mix.shape[0]
        bd = torch.zeros(t * h, t * h, dtype=dtype, device=mix.device)
        src = mix.to(dtype)
        for i in range(t):
            bd[i * h : (i + 1) * h, i * h : (i + 1) * h] = src
        cache[key] = bd
    return bd


@plugin_hook(
    "sglang.srt.models.ernie55.VhaPostMix.forward",
    type=HookType.AROUND,
)
def vha_postmix_forward_kunlun(original_fn, self, o: torch.Tensor):
    """One block-diagonal 2-D mm instead of a stride-0 broadcast bmm.

    Upstream's ``torch.matmul(mix, heads)`` expands a 2-D ``mix`` to
    ``[T, H, H]`` with stride 0 and becomes a batched GEMM: ~78 us on 8 MFLOP at
    T=2. ``out[t,i,d] = sum_j mix[i,j] o[t,j,d]`` is the same result as one 2-D
    mm against ``diag(mix, ..., mix)``. Anything unexpected falls back.
    """
    t, nh = o.shape[0], self.num_heads
    if (
        self.sharded
        or not 0 < t <= _VHA_MAX_T
        or o.numel() % (t * nh)
        or o.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return original_fn(self, o)
    try:
        bd = _block_diag(self, self.mix, t, o.dtype)
        return torch.mm(bd, o.reshape(t * nh, -1)).reshape(o.shape)
    except Exception:  # noqa: BLE001 - never take the engine down for a perf path
        logger.exception(
            "sglang-kunlun: VhaPostMix block-diagonal failed (o=%s %s); "
            "falling back to upstream",
            tuple(o.shape),
            o.dtype,
        )
        return original_fn(self, o)


@plugin_hook(
    "sglang.srt.models.ernie55.VhaPostMix.fold",
    type=HookType.AROUND,
)
def vha_postmix_fold_kunlun(original_fn, self, *args, **kwargs):
    """``fold`` rewrites ``mix`` in place, so the cached block diagonal dies."""
    out = original_fn(self, *args, **kwargs)
    if getattr(self, "_kl_bd", None):
        self._kl_bd.clear()
    return out


def _ernie55_mega_init_weights(module):
    """Convert an Ernie5_5Moe layer's expert weights to the MegaMoE SiTU layout,
    once, writing the result back into the existing Parameters so the experts keep a
    SINGLE copy (reassigning ``.data`` frees the pre-conversion tensors; only a
    transient coexists during the call). This is the DeepSeek-V4 Kunlun path's reason
    for in-place conversion -- the experts are far too large to hold twice.

    Runs on the first forward rather than at model post-load: at post-load the Kunlun
    W4A8 process-weights step has NOT run yet, so the experts are still the raw GPTQ
    layout (int32 ``[E, hidden//8, 2*ffn]``); by the first forward they are the
    processed ``[E, 2*ffn, hidden//2]`` int8 + ``[E, 2*ffn, 1]`` x7 scale that
    ``transform_weights_for_mega_moe`` consumes. Idempotent via
    ``experts._mega_moe_layout``.
    """
    from hydrax.xaccelerator.mega_moe import transform_weights_for_mega_moe

    experts = module.experts
    if getattr(experts, "_mega_moe_layout", False):
        return
    l1, l2 = transform_weights_for_mega_moe(
        (experts.w13_weight_packed.data, experts.w13_weight_scale.data.squeeze(-1)),
        (experts.w2_weight_packed.data, experts.w2_weight_scale.data.squeeze(-1)),
        activation="situ",
    )
    experts.w13_weight_packed.data, experts.w13_weight_scale.data = l1
    experts.w2_weight_packed.data, experts.w2_weight_scale.data = l2
    module._mega_beta = float(experts.moe_runner_config.gemm1_alpha)
    module._mega_linear_beta = float(experts.moe_runner_config.gemm1_clamp_limit)
    experts._mega_moe_layout = True


@plugin_hook(
    "sglang.srt.models.ernie55.Ernie5_5Moe._forward_megamoe_experts",
    type=HookType.AROUND,
)
def ernie55_forward_megamoe_experts_kunlun(original_fn, self, latent, topk_output):
    """Kunlun MegaMoE expert path for ERNIE 5.5 (fills the base stub).

    W4A8: int8-per-token activations (``kunlun_ops.quant2d``) x int4 experts,
    fused all-to-all + grouped GEMM + SiTU + combine via hydrax
    ``int8_int4_mega_moe`` over an xshmem symmetric buffer. Expert weights are
    converted to the MegaMoE SiTU layout in place at load (gate/up interleave),
    so the kernel reads the FusedMoE's packed-int4 tensors directly. Runs in the
    ``moe_latent_size`` space; the caller wraps the result
    with ``fc2_latent_proj`` + ``latent_norm`` and adds the shared expert.
    """
    import kunlun_ops
    import torch
    from hydrax.xaccelerator.mega_moe_fusion import (
        get_symm_buffer,
        int8_int4_mega_moe,
    )

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    experts = self.experts
    if not getattr(experts, "_mega_moe_layout", False):
        # One-time weight conversion, on the first forward (during warmup, before
        # CUDA-graph capture); guarded so later/captured forwards never re-run it.
        _ernie55_mega_init_weights(self)
    buf = getattr(self, "_mega_buf", None)
    if buf is None:
        # All-to-all scratch buffer: layer-shared (keyed by dims), lazily built on
        # the first forward when xshmem is ready.
        cap = self._mega_cap = _mega_moe_cap()
        key = (
            self.num_experts,
            cap,
            experts.top_k,
            latent.shape[-1],
            experts.w13_weight_packed.shape[1] // 2,
        )
        buf = _MEGA_SYMM_BUFFERS.get(key)
        if buf is None:
            buf = _MEGA_SYMM_BUFFERS[key] = get_symm_buffer(
                group=get_moe_ep_group().device_group,
                num_experts=self.num_experts,  # global routed experts (512)
                num_max_tokens_per_rank=cap,
                num_topk=experts.top_k,
                hidden=key[3],
                intermediate_hidden=key[4],
                num_shared_experts=0,  # ERNIE runs the shared expert outside the kernel
                use_fp8_dispatch=False,
                activation="situ",
                w8_group_mode=0,
            )
        self._mega_buf = buf

    hidden = latent.shape[-1]
    rows = latent.shape[0]
    if rows > self._mega_cap:
        raise RuntimeError(
            f"ERNIE MegaMoE got {rows} rows/rank but the symmetric buffer holds "
            f"{self._mega_cap}; keep per-rank tokens <= cap (CHUNKED_PREFILL<=cap "
            "or raise SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK)."
        )
    q = torch.empty((rows, hidden), dtype=torch.int8, device=latent.device)
    absmax = torch.empty((rows, 1), dtype=torch.float32, device=latent.device)
    if rows:
        # Per-token INT8 activation quant (the "A8"); 0-row idle a2a ranks skip it.
        kunlun_ops.quant2d(latent, q, absmax, force_sdnn=True)
    x_sf = absmax.view(-1)
    # topk comes from TopK in the dtypes the kernel wants on this model; convert
    # only if it ever differs (avoids a per-forward copy).
    topk_ids = topk_output.topk_ids
    if topk_ids.dtype != torch.int64:
        topk_ids = topk_ids.to(torch.int64)
    topk_weights = topk_output.topk_weights
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)

    y = torch.empty((rows, hidden), dtype=torch.bfloat16, device=latent.device)
    int8_int4_mega_moe(
        y,
        x=q,
        x_sf=x_sf,
        topk_idx=topk_ids,
        topk_weights=topk_weights,
        l1_weights=(experts.w13_weight_packed.data, experts.w13_weight_scale.data),
        l2_weights=(experts.w2_weight_packed.data, experts.w2_weight_scale.data),
        sym_buffer=buf,
        activation="situ",
        situ_beta=self._mega_beta,
        situ_linear_beta=self._mega_linear_beta,
    )
    return y.to(latent.dtype)
