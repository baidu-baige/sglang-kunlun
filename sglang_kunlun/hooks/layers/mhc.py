"""Kunlun hooks for ``sglang.kernels.ops.layernorm.mhc``.

``mhc_pre`` is the hot one: its TileLang kernel is a stub here, so upstream
falls back to the torch reference ``_mhc_pre_torch``, and it runs twice per
layer across 43 layers. The fused chain below replaces it with three
``kunlun_ops`` calls -- ``rms_qkv_gemm`` (row RMS + mixing GEMM),
``mhc_split_sinkhorn`` (splits + Sinkhorn iterations) and
``mhc_pre_weighted_sum`` (pre-weighted residual sum, tail RMSNorm folded in
when the norm weight matches dtype) -- plus host-side memos for the scratch
buffers and their views, since a view costs a dispatch on this backend.

Numerics follow ``_mhc_pre_torch``: eps *inside* the sqrt, which is also
``rms_qkv_gemm``'s convention, so the GEMM fusion is exact. Every fused path
falls back to torch when its shape/dtype/contiguity contract does not hold, and
disables itself on the first exception rather than per call.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import kunlun_ops
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


# Set to False on the first failure of the corresponding op.
_OK = {"prenorm": True, "sinkhorn": True, "combine": True, "norm": True}
_LOGGED: dict = {}


def _once(key: str, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED[key] = 1
        logger.info(msg)


def _disable(key: str, exc: BaseException) -> None:
    _OK[key] = False
    logger.warning("mhc: %s fusion disabled after %r; using torch", key, exc)


# Host memos, keyed by the full allocation key of the buffer they replace, so a
# hit can only return what `torch.empty` would have. Safe because the calls in a
# step run serially on the compute stream; tensors that outlive their call
# (`post`, `comb`) are deliberately not cached.
_XFLAT_BUF: dict = {}
_GEMM_OUT_BUF: dict = {}
_GEMM_SCRATCH: dict = {}
_GAMMA_ONES: dict = {}
_W_FP32: dict = {}
_VIEWS: dict = {}


def _fp32_cached(w: torch.Tensor) -> torch.Tensor:
    """`w.float()` memoised on `id(w)`, verified with `is` against a strong
    reference we hold so a reused address cannot serve a stale cast."""
    if w.dtype == torch.float32:
        return w
    hit = _W_FP32.get(id(w))
    if hit is not None and hit[0] is w:
        return hit[1]
    cast = w.float()
    _W_FP32[id(w)] = (w, cast)
    return cast


def _view(base: torch.Tensor, shape, slot: str) -> torch.Tensor:
    """`base.view(shape)` memoised while `base` is the same object.

    Guarded on a stored strong reference, not `view._base`, which is None under
    `torch.inference_mode`.
    """
    ent = _VIEWS.get(slot)
    if ent is not None and ent[0] is base and ent[1] == shape:
        return ent[2]
    v = base.view(shape)
    _VIEWS[slot] = (base, shape, v)
    return v


def _xflat_f32(residual_flat: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """`residual_flat.reshape(rows, k).float()` into a run-long fp32 buffer.

    `copy_` widens straight from the strided source, so the GEMM sees identical
    bits and the intermediate reshape copy disappears.
    """
    key = (rows, k, residual_flat.device)
    buf = _XFLAT_BUF.get(key)
    if buf is None:
        buf = torch.empty(rows, k, dtype=torch.float32, device=residual_flat.device)
        _XFLAT_BUF[key] = buf
    _view(buf, residual_flat.shape, "xflat").copy_(residual_flat)
    return buf


def _row_rms_recip(x: torch.Tensor, eps: float) -> torch.Tensor:
    """`rsqrt(mean(x^2, -1) + eps)`, matching `_mhc_pre_torch`."""
    return torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


# ------------------------------------------------------------------ stage 1-2
def _prenorm_gemm(
    x_flat: torch.Tensor, w_f32: torch.Tensor, eps: float
) -> Optional[torch.Tensor]:
    """`linear(x_flat, w) * rsqrt(mean(x^2) + eps)` in one launch, or None.

    `rms_qkv_gemm` computes `matmul(rms_norm(x, gamma, eps), w.t())`; the
    substitution holds because the rms factor is one scalar per row, so gamma is
    all-ones and eps goes in plain. Returns None instead of raising when the
    contract does not hold -- a non-contiguous x is accepted by the op and
    silently wrong, hence the hard guard.
    """
    if not _OK["prenorm"]:
        return None
    if (
        x_flat.dtype is not torch.float32
        or w_f32.dtype is not torch.float32
        or x_flat.dim() != 2
        or x_flat.shape[0] == 0
        or not x_flat.is_contiguous()
        or not w_f32.is_contiguous()
    ):
        return None
    rows, k = x_flat.shape
    n = w_f32.shape[0]
    try:
        dev = x_flat.device
        gamma = _GAMMA_ONES.get((k, dev))
        if gamma is None:
            gamma = torch.ones(k, dtype=torch.float32, device=dev)
            _GAMMA_ONES[(k, dev)] = gamma
        scratch = _GEMM_SCRATCH.get((rows, k, dev))
        if scratch is None:
            scratch = torch.empty((1, rows, k), dtype=torch.float32, device=dev)
            _GEMM_SCRATCH[(rows, k, dev)] = scratch
        out = _GEMM_OUT_BUF.get((rows, n, dev))
        if out is None:
            out = torch.empty((1, rows, n), dtype=torch.float32, device=dev)
            _GEMM_OUT_BUF[(rows, n, dev)] = out
        kunlun_ops.rms_qkv_gemm(
            _view(x_flat, (1, rows, k), "gemm_in"),
            w_f32,
            gamma,
            float(eps),
            True,
            out,
            scratch,
        )
        _once("prenorm", f"mhc: rms_qkv_gemm fusion active (rows={rows} k={k} n={n})")
        return _view(out, (rows, n), "gemm_out")
    except Exception as exc:  # disable once, not per call
        _disable("prenorm", exc)
        return None


# -------------------------------------------------------------------- stage 3
def _split_sinkhorn_flat(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc: int,
    sinkhorn_iters: int,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split-Sinkhorn on a flat `[n, hc*(2+hc)]` mix tensor.

    Returns `pre [n, hc]`, `post [n, hc]`, `comb [n, hc, hc]`. The fused op
    carries one epsilon and a hardcoded post multiplier, so it only runs when
    the caller's two epsilons agree and the multiplier is 2.0, as ERNIE 5.5 does.
    """
    rows = mixes.shape[0]
    fused = (
        _OK["sinkhorn"]
        and rows > 0
        and float(pre_eps) == float(sinkhorn_eps)
        and float(post_mult) == 2.0
    )
    if fused:
        try:
            m = mixes if mixes.is_contiguous() else mixes.contiguous()
            dev = mixes.device
            pre = torch.empty((rows, hc), dtype=torch.float32, device=dev)
            post = torch.empty((rows, hc), dtype=torch.float32, device=dev)
            comb = torch.empty((rows, hc * hc), dtype=torch.float32, device=dev)
            kunlun_ops.mhc_split_sinkhorn(
                m,
                _fp32_cached(hc_scale),
                _fp32_cached(hc_base),
                pre,
                post,
                comb,
                hc,
                int(sinkhorn_iters),
                float(sinkhorn_eps),
            )
            _once("sinkhorn", f"mhc: mhc_split_sinkhorn fusion active (hc={hc})")
            return pre, post, comb.view(rows, hc, hc)
        except Exception as exc:
            _disable("sinkhorn", exc)

    flat = mixes.float()
    scale = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(flat[:, :hc] * scale[0] + base[:hc]) + pre_eps
    post = post_mult * torch.sigmoid(
        flat[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    )
    comb = (flat[:, 2 * hc :] * scale[2] + base[2 * hc :]).view(-1, hc, hc)
    comb = comb.softmax(-1) + sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(max(int(sinkhorn_iters) - 1, 0)):
        comb = comb / (comb.sum(-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    return pre, post, comb


# -------------------------------------------------------------------- stage 4
def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`exact_rms_norm` spelling: fp32 accumulate, cast back to x's dtype."""
    if (
        _OK["norm"]
        and x.dim() == 2
        and x.shape[0] > 0
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float16)
        and weight.dim() == 1
        and weight.shape[0] == x.shape[-1]
        and weight.dtype in (x.dtype, torch.float32)
    ):
        try:
            out = torch.empty_like(x)
            kunlun_ops.rmsnorm(x, weight, out, eps, False, True, None, None, None)
            return out
        except Exception as exc:
            _disable("norm", exc)
    xf = x.float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * rsqrt * weight.float()).to(x.dtype)


def _hc_combine(
    x_flat: torch.Tensor,
    pre: torch.Tensor,
    hc: int,
    out_dtype: torch.dtype,
    norm_weight: Optional[torch.Tensor],
    norm_eps: Optional[float],
) -> torch.Tensor:
    """`y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]`, RMSNorm'd if asked.

    The norm lives here so the fused op can take the weight as its `scale` and
    do both steps in one pass; it rejects an fp32 scale, so the fold only runs
    when the weight already carries the residual's dtype.
    """
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    if m == 0:
        return torch.empty((m, h), dtype=out_dtype, device=x_flat.device)
    if (
        _OK["combine"]
        and x_flat.dtype in (torch.bfloat16, torch.float16)
    ):
        try:
            res = x_flat.view(m, hc, h)
            if not res.is_contiguous():
                res = res.contiguous()
            if not pre.is_contiguous():
                pre = pre.contiguous()
            fold = norm_weight is not None and norm_weight.dtype == res.dtype
            out = torch.empty((m, h), dtype=res.dtype, device=x_flat.device)
            kunlun_ops.mhc_pre_weighted_sum(
                res,
                pre,
                out,
                norm_weight if fold else None,
                float(norm_eps) if fold else 1e-6,
            )
            _once(
                "combine",
                f"mhc: mhc_pre_weighted_sum fusion active (norm_folded={fold})",
            )
            # The op already wrote res.dtype; a matching `.to()` is a dispatch.
            if out.dtype is not out_dtype:
                out = out.to(out_dtype)
            if norm_weight is not None and not fold:
                out = _rmsnorm(out, norm_weight, norm_eps)
            return out
        except Exception as exc:
            _disable("combine", exc)
    xv = x_flat.view(m, hc, h).float()
    out = (pre.float().unsqueeze(-1) * xv).sum(dim=1).to(out_dtype)
    if norm_weight is not None:
        out = _rmsnorm(out, norm_weight, norm_eps)
    return out


# ------------------------------------------------------------------- mhc_pre
def _mhc_pre_fused(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same contract as `_mhc_pre_torch` plus the optional tail RMSNorm.

    `residual` is `[s, hc, h]`; returns `post [s, hc, 1]`, `comb [s, hc, hc]`
    and `layer_input [s, h]`.
    """
    s, hc, h = residual.shape
    dtype = residual.dtype
    if s == 0:
        dev = residual.device
        return (
            torch.empty((0, hc, 1), dtype=torch.float32, device=dev),
            torch.empty((0, hc, hc), dtype=torch.float32, device=dev),
            torch.empty((0, h), dtype=dtype, device=dev),
        )

    k = hc * h
    if dtype is not torch.float32:
        x_flat = _xflat_f32(residual, s, k)
    else:
        x_flat = residual.reshape(s, k).float()

    mixes = _prenorm_gemm(x_flat, _fp32_cached(fn), rms_eps)
    if mixes is None:
        mixes = torch.nn.functional.linear(
            x_flat, _fp32_cached(fn)
        ) * _row_rms_recip(x_flat, rms_eps)

    pre, post, comb = _split_sinkhorn_flat(
        mixes,
        hc_scale,
        hc_base,
        hc,
        sinkhorn_repeat,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
    )
    layer_input = _hc_combine(
        residual.reshape(s, k), pre, hc, dtype, norm_weight, norm_eps
    )
    return post.view(s, hc, 1), comb.view(s, hc, hc), layer_input


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.hc_split_sinkhorn",
    type=HookType.REPLACE,
)
def hc_split_sinkhorn_kunlun(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Run Sinkhorn through the Kunlun op while preserving the 0.5.14 contract."""
    batch, seq_len, _ = mixes.shape
    pre = mixes.new_empty(batch, seq_len, hc_mult)
    post = mixes.new_empty(batch, seq_len, hc_mult)
    comb = mixes.new_empty(batch, seq_len, hc_mult, hc_mult)
    if mixes.numel() == 0:
        return pre, post, comb

    flat_tokens = batch * seq_len
    flat_pre = pre.view(flat_tokens, hc_mult)
    flat_post = post.view(flat_tokens, hc_mult)
    flat_comb = comb.view(flat_tokens, hc_mult * hc_mult)
    kunlun_ops.mhc_split_sinkhorn(
        mixes.reshape(flat_tokens, -1),
        hc_scale,
        hc_base,
        flat_pre,
        flat_post,
        flat_comb,
        hc_mult,
        sinkhorn_iters,
        eps,
    )
    return pre, post, comb


@plugin_hook(
    "sglang.srt.models.deepseek_v4.get_mhc_ops",
    type=HookType.REPLACE,
)
def get_mhc_ops_kunlun():
    """Resolve the MHC ops through the hooked module instead of ``sgl_kernel``.

    Upstream short-circuits to ``sgl_kernel.hc_split_sinkhorn`` on XPU, which
    returns fp16 ``post``/``comb``. Golden reaches the Kunlun replacement
    registered on the layernorm module, whose outputs follow the fp32 ``mixes``
    dtype, so route through that module to keep the mHC boundary aligned.

    ``MhcOps`` gained ``mhc_fused_post_pre``, ``mhc_pre``, ``mhc_post`` and
    ``fused_hc_head`` in 4.1 (the V4-Flash model never resolved this NamedTuple).
    XPU takes the ``_is_xpu`` branches that call ``mhc_pre`` / ``mhc_post`` /
    ``fused_hc_head``, so each is bound to the repo's own torch reference; the
    TileLang/DeepGEMM kernels behind them are gated off by this deployment's
    ``SGLANG_OPT_USE_TILELANG_MHC_PRE`` / ``..._MHC_POST`` / ``DEEPGEMM_HC_PRENORM``.
    """
    from sglang.srt.models.deepseek_v4 import MhcOps
    from sglang.srt.models import deepseek_v4 as dsv4
    from sglang.kernels.ops.layernorm import mhc as mhc_module

    def mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight=None,
        norm_eps=None,
    ):
        """Kernel-signature wrapper over the fused chain above.

        `norm_weight`/`norm_eps` are honored here (folded into the weighted
        sum); the TileLang kernel that used to be the only producer of the norm
        variant stays disabled in this deployment.
        """
        return _mhc_pre_fused(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )

    return MhcOps(
        hc_split_sinkhorn=mhc_module.hc_split_sinkhorn,
        mhc_fused_post_pre=None,
        npu_hc_pre=None,
        mhc_pre=mhc_pre_torch,
        mhc_post=mhc_post_kunlun,
        fused_hc_head=dsv4.hc_head_torch,
    )


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.mhc_pre",
    type=HookType.REPLACE,
)
def mhc_pre_kunlun(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    n_splits_pre=32,
    *,
    norm_weight=None,
    norm_eps=None,
    eps_outside_sqrt=False,
):
    """Fused replacement for the TileLang public ``mhc_pre``.

    ``n_splits``/``n_splits_pre`` only tile the TileLang kernel and have no
    meaning here. ``eps_outside_sqrt`` is ignored for the same reason the
    upstream torch reference ignores it: the reference spelling keeps eps inside
    the sqrt, and the fused chain is exact against that spelling.
    """
    return _mhc_pre_fused(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.mhc_post",
    type=HookType.REPLACE,
)
def mhc_post_kunlun(x, residual, post_layer_mix, comb_res_mix):
    """Compute mHC post with the XSG native operator."""
    post = (
        post_layer_mix.squeeze(-1)
        if post_layer_mix.dim() == residual.dim()
        else post_layer_mix
    )
    if x.shape[0] == 0:
        return torch.empty_like(residual)
    return torch.ops.xspeedgate_ops.mhc_post(
        x.contiguous(),
        residual.contiguous(),
        post.unsqueeze(-1).contiguous(),
        comb_res_mix.contiguous(),
    )


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc_head.fused_hc_head",
    type=HookType.REPLACE,
)
def fused_hc_head_kunlun(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    norm_eps,
    hc_eps,
    eps_outside_sqrt=False,
):
    """Torch-native replacement for the Triton ``fused_hc_head``.

    ERNIE 5.5's ``Ernie5_5HcHead.forward`` (mHC stream collapse) calls the public
    ``fused_hc_head`` (``mhc_head.py``), a Triton kernel (``_hc_head_kernel[grid]``)
    that is a bare stub on Kunlun. The module docstring carries the exact torch
    reference; this reproduces it, honoring ``eps_outside_sqrt`` (ERNIE passes
    True, i.e. ``1/(sqrt(mean_sq)+eps)`` like ``mhc_pre``; the DSV4 inline
    ``hc_head`` uses eps-inside ``rsqrt``). ``x`` is ``[T, hc_mult, hidden]`` and
    the output is ``[T, hidden]`` in x's dtype.
    """
    import torch.nn.functional as F

    shape = x.shape  # [T, hc_mult, hidden]
    dtype = x.dtype
    if shape[0] == 0:
        return x.new_empty((0, shape[-1]))
    xf = x.flatten(1).float()
    mean_sq = xf.square().mean(-1, keepdim=True)
    if eps_outside_sqrt:
        norm = 1.0 / (torch.sqrt(mean_sq) + norm_eps)
    else:
        norm = torch.rsqrt(mean_sq + norm_eps)
    mixes = F.linear(xf, hc_fn) * norm
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=1)
    return y.to(dtype)
