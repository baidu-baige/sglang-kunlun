# Copyright 2026 Baidu, Inc.
# Licensed under the Apache License, Version 2.0.
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
"""HookRegistry registration for the MHC split-Sinkhorn gate kernel.

Upstream ``sglang.srt.layers.mhc.hc_split_sinkhorn`` dispatches to a tilelang
JIT kernel that cannot build on the Kunlun XPU. The REPLACE hook below keeps the
same (pre, post, comb) contract: ``pre``/``post`` are per-token sigmoid gates and
``comb`` is Sinkhorn-normalized.

``sglang.srt.models.deepseek_v4`` imports the symbol lazily inside the call site,
so it resolves the patched module attribute at call time.
"""

from __future__ import annotations

import functools
import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_hc_combine_announced = False
_hc_mix_announced = False


def _f32c(t: torch.Tensor) -> torch.Tensor:
    """fp32 contiguous view/copy, skipping the dispatch when it already is.

    ``hc_scale`` / ``hc_base`` / ``hc_fn`` are static module parameters, yet the
    old unconditional ``x.float().contiguous()`` issued two ATen dispatches per
    layer per operand (``aten::to`` ~74 us host time on this XPU stack) for tensors
    that were already fp32 and contiguous.
    """
    if t.dtype is not torch.float32:
        t = t.to(torch.float32)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


@functools.cache
def _xspeedgate_hc_mix_stats_only_available() -> bool:
    """The installed xspeedgate_ops supports the ``mix == 0`` stats-only mode
    (per-token inv-rms, output ``[M, 1]``). Probed once with a 1x32 tile."""
    try:
        import xspeedgate_ops  # noqa: F401 -- registers torch.ops.xspeedgate_ops
    except Exception:
        return False
    try:
        probe_x = torch.zeros((1, 32), dtype=torch.bfloat16, device="cuda")
        probe_w = torch.zeros((0, 32), dtype=torch.float32, device="cuda")
        out = torch.ops.xspeedgate_ops.hc_mix_stats(probe_x, probe_w, 1e-6)
        return tuple(out.shape) == (1, 1)
    except Exception:
        return False


_HC_MIX_GEMM_MIN_M = 128  # crossover measured on P800 (fused wins below this)
_HC_MIX_GEMM_DTYPES = (torch.bfloat16, torch.float16)  # r66: fp16 site split too


def _hc_mix_use_gemm(x_flat: torch.Tensor, hc_fn: torch.Tensor) -> bool:
    """Split the op for large M instead of running the instruction-bound fused
    kernel (see ``_hc_mix_stats_torch`` docstring)."""
    return (
        x_flat.dim() == 2
        and x_flat.dtype in _HC_MIX_GEMM_DTYPES
        and x_flat.is_contiguous()
        and hc_fn.is_contiguous()
        and hc_fn.dtype == torch.float32
        and x_flat.shape[0] >= _HC_MIX_GEMM_MIN_M
        and _xspeedgate_hc_mix_stats_only_available()
    )


@functools.cache
def _xspeedgate_hc_combine_available() -> bool:
    """``import xspeedgate_ops`` registers the op namespace; ``hasattr`` alone is
    False on a process that has not loaded the extension yet."""
    try:
        import xspeedgate_ops  # noqa: F401 -- registers torch.ops.xspeedgate_ops
    except Exception:
        return False
    return hasattr(torch.ops.xspeedgate_ops, "hc_combine")


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.hc_split_sinkhorn",
    type=HookType.REPLACE,
)
def _hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Kunlun fused split-Sinkhorn, falling back to pure torch."""
    b, s, _ = mixes.size()
    hc = int(hc_mult)
    if b * s == 0:
        # A DP-attention rank with no local tokens (idle batch) still runs the
        # eager forward, so the MHC stats are evaluated on a 0-row tensor. xdnn's
        # amax asserts on a zero-length tensor ("[ASSERT-FAIL](0>0), 0 Tensor
        # length", amax.cpp:47) and the kunlun_ops kernel asserts on an empty row
        # count, so short-circuit: with 0 tokens the coefficients are unused and
        # only their shape matters.
        logger.debug(
            "sglang-kunlun: hc_split_sinkhorn on empty input (b=%d s=%d hc=%d); "
            "returning empty coefficients",
            b,
            s,
            hc,
        )
        return (
            mixes.new_empty((b, s, hc)),
            mixes.new_empty((b, s, hc)),
            mixes.new_empty((b, s, hc, hc)),
        )

    # Kunlun fused MHC split-Sinkhorn: same pre/post/comb definition and
    # Sinkhorn iteration count as the torch reference below, but the 20-iteration
    # row/col normalization loop runs in a single fused kernel. Falls back to the
    # torch implementation if the fused op is unavailable.
    try:
        import kunlun_ops

        _m = mixes.reshape(-1, (2 + hc) * hc)
        if not _m.is_contiguous():
            _m = _m.contiguous()
        _nr = _m.shape[0]
        _pre = torch.empty((_nr, hc), dtype=torch.float32, device=mixes.device)
        _post = torch.empty((_nr, hc), dtype=torch.float32, device=mixes.device)
        _comb = torch.empty((_nr, hc * hc), dtype=torch.float32, device=mixes.device)
        kunlun_ops.mhc_split_sinkhorn(
            _m,
            _f32c(hc_scale),
            _f32c(hc_base),
            _pre,
            _post,
            _comb,
            hc,
            int(sinkhorn_iters),
            float(eps),
        )
        return (
            _pre.reshape(b, s, hc),
            _post.reshape(b, s, hc),
            _comb.reshape(b, s, hc, hc),
        )
    except Exception:
        pass

    m = mixes.reshape(-1, (2 + hc) * hc).float()
    scale = hc_scale.float()
    base = hc_base.float()
    n = m.shape[0]

    pre = torch.sigmoid(m[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (m[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(n, hc, hc)

    # row softmax over k (dim=2)
    row_max = comb.amax(dim=2, keepdim=True)
    comb = torch.exp(comb - row_max)
    row_sum = comb.sum(dim=2, keepdim=True)
    comb = comb / row_sum + eps

    # initial column normalization over j (dim=1)
    col_sum = comb.sum(dim=1, keepdim=True)
    comb = comb / (col_sum + eps)

    for _ in range(int(sinkhorn_iters) - 1):
        row_sum = comb.sum(dim=2, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=1, keepdim=True)
        comb = comb / (col_sum + eps)

    pre = pre.reshape(b, s, hc)
    post = post.reshape(b, s, hc)
    comb = comb.reshape(b, s, hc, hc)
    return pre, post, comb


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.hc_combine",
    type=HookType.REPLACE,
)
def _hc_combine_torch(
    x_flat: torch.Tensor, pre: torch.Tensor, hc: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """Route hc_combine off triton: triton kernels are not CUDA-Graph capturable here.

    Upstream ``sglang/kernels/ops/layernorm/mhc.py::hc_combine`` launches the triton
    kernel ``_hc_combine_kernel``. On Kunlun XPU a triton kernel launch is **not
    recorded into a CUDA Graph**: at capture time torch even reports "The CUDA Graph
    is empty", and at replay the kernel writes nothing, so the output tensor keeps
    whatever the memory already held -- exactly zero, or uninitialised NaN.

    Isolated proof (torch + triton only, no sglang, single card)::

        T1 pure torch, persistent output        replay follows input   0/4 failures
        T2 triton kernel, persistent output     replay follows input   4/4 failures
        T3 triton kernel, output allocated
           inside the graph                     replay follows input   4/4 failures

    In the model this shows up as: with the decode CUDA Graph on, hidden_states is
    exactly zero on a block of layers, the w8a8 per-token dynamic quantisation then
    computes scale = absmax/127 = 0, ``x / 0`` turns the block into NaN from ~layer 13
    on, the logits become NaN and argmax collapses onto token id 0. Per-layer device
    checksums, same request and same step: 21 layers exactly zero plus 8 all-NaN with
    the graph on, versus 43 clean layers with ``--disable-cuda-graph``.

    Note the arithmetic of the triton kernel is fine -- called directly it is
    bit-stable over 200 launches and matches this torch reference to 1 fp16 ULP. The
    defect is capturability, not math, which is why no amount of Python-side
    synchronisation helps: CUDA Graph replay does not execute Python at all.

    The torch form below is plain elementwise multiply plus a reduction with fp32
    accumulation, which records into the graph like any other torch op::

        y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]

    Set ``SGLANG_KUNLUN_TRITON_HC_COMBINE=1`` to restore the upstream (defective on
    kunlun) path -- used for the negative half of the A/B proof.
    """
    import os as _os

    if _os.environ.get("SGLANG_KUNLUN_TRITON_HC_COMBINE", "0") == "1":
        import triton

        from sglang.kernels.ops.layernorm.mhc import _hc_combine_kernel

        m = x_flat.shape[0]
        h = x_flat.shape[1] // hc
        y = torch.empty((m, h), dtype=out_dtype, device=x_flat.device)
        block_h = 1024
        _hc_combine_kernel[(m, triton.cdiv(h, block_h))](
            x_flat,
            pre,
            y,
            h,
            x_flat.stride(0),
            pre.stride(0),
            pre.stride(1),
            y.stride(0),
            HC=hc,
            BLOCK_H=block_h,
        )
        return y

    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    if m == 0:
        # The fused op rejects a zero-row input; with no tokens the
        # coefficients are unused and only the shape matters.
        return x_flat.new_empty((0, h), dtype=out_dtype)
    if (
        0 < hc <= 16
        and x_flat.dim() == 2
        and x_flat.shape[1] % hc == 0
        and x_flat.is_contiguous()
        and _xspeedgate_hc_combine_available()
    ):
        try:
            y = torch.ops.xspeedgate_ops.hc_combine(
                x_flat,
                _f32c(pre.reshape(m, hc)),
                out_dtype,
            )
            global _hc_combine_announced
            if not _hc_combine_announced:
                _hc_combine_announced = True
                logger.info("sglang-kunlun: native hc_combine active")
            return y
        except Exception:
            pass
    # [KUNLUN CG-SAFE] Fall back to the fused weighted sum when the native
    # ``hc_combine`` op is unavailable: the torch expression below contains
    # dtype casts and a final ``acc.to(out_dtype)``, which lower to copy
    # kernels that can fault inside a decode capture.
    try:
        import kunlun_ops

        y = kunlun_ops.mhc_pre_weighted_sum(
            x_flat.reshape(m, hc, h), pre.reshape(m, hc)
        )
        return y if y.dtype == out_dtype else y.to(out_dtype)
    except Exception:
        pass
    acc = (
        x_flat.reshape(m, hc, h).to(torch.float32)
        * pre.reshape(m, hc, 1).to(torch.float32)
    ).sum(dim=1)
    return acc.to(out_dtype)


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.hc_mix_stats",
    type=HookType.REPLACE,
)
def _hc_mix_stats_torch(x_flat: torch.Tensor, hc_fn: torch.Tensor, eps: float) -> torch.Tensor:
    """Pure-torch replacement for the tf32x3 split-K Triton ``hc_mix_stats``.

    Upstream ``sglang/kernels/ops/layernorm/mhc.py::hc_mix_stats`` launches a
    Triton kernel whose ``tl.dot(..., input_precision="tf32x3")`` is rejected by
    the Kunlun XPU Triton backend (``allowed_dot_input_precisions=("ieee","tf32")``
    -> AssertionError). torch_xmlir reports ``x.is_cuda`` True and
    ``torch.version.cuda`` set, so ``DeepseekV4DecoderLayer._hc_mix_stats``
    (deepseek_v4.py:3370) takes the Triton branch instead of its own torch
    else-branch.     This mirrors that else-branch byte-for-byte (minus the caller's
    later ``.unsqueeze(1)``): [M,K] x [MIX,K] -> [M,MIX] fp32 =
    ``F.linear(x.float(), hc_fn) * rsqrt(mean(x^2)+eps)``. Covers both the attn
    and ffn call sites (same module symbol).

    Perf: xspeedgate-ops (1.6.0+) ships a native XPU ``hc_mix_stats`` (fused
    fc + per-token RMS, fp32 split-K). That fused kernel is instruction bound at
    ~80 GB/s; for the prefill shape ([4096, 20480] x [24, 20480]) the vendor GEMM
    runs the same 24 dots at ~715 GB/s. So for large M we split the op:
    ``kunlun_ops.matmul`` for the dots + the native op in its ``mix == 0``
    stats-only mode (per-token inv-rms, ``[M, 1]``) for the RMS factor, then one
    elementwise multiply. Same fp32 accumulation, rel err vs the fused kernel
    ~3e-7. Small M keeps the fused kernel (one launch fewer, no GEMM dispatch).
    """
    mix = hc_fn.shape[0]
    if 0 < mix <= 32:
        if _hc_mix_use_gemm(x_flat, hc_fn):
            try:
                import kunlun_ops

                fn32 = _f32c(hc_fn)
                if x_flat.dtype == torch.float16:
                    # r66: the vendor GEMM rejects mixed (fp16 x, fp32 w) and an
                    # fp16->fp32 store ("matmul unimplemented"), so mirror the 24
                    # weight rows to fp16 (3 MB read, one dispatch) and let the
                    # dots round once into fp16.  Measured on [4096,20480,24]:
                    # GEMM 137.5 us + stats 162.6 us vs 1792 us for the
                    # instruction-bound fused kernel; casting the 168 MB
                    # activation to bf16 instead costs 441 us and drops two
                    # mantissa bits.  Error vs the fp32-exact mixes: max_abs
                    # 1.25e-2 at |mixes|~7 (<=2 fp16 ulp), i.e. below the bf16
                    # granularity of the sinkhorn coefficients built from it.
                    fn16 = hc_fn if hc_fn.dtype == torch.float16 else hc_fn.to(torch.float16)
                    out = torch.empty(
                        (x_flat.shape[0], mix), dtype=torch.float16, device=x_flat.device
                    )
                    kunlun_ops.matmul(x_flat, fn16, out)
                else:
                    out = torch.empty(
                        (x_flat.shape[0], mix), dtype=torch.float32, device=x_flat.device
                    )
                    kunlun_ops.matmul(x_flat, fn32, out)
                inv_rms = torch.ops.xspeedgate_ops.hc_mix_stats(x_flat, fn32[:0], eps)
                global _hc_mix_announced
                if not _hc_mix_announced:
                    _hc_mix_announced = True
                    logger.info(
                        "sglang-kunlun: hc_mix_stats GEMM+rms path active (x dtype %s)",
                        x_flat.dtype,
                    )
                return out * inv_rms
            except Exception:
                pass
        try:
            return torch.ops.xspeedgate_ops.hc_mix_stats(x_flat, _f32c(hc_fn), eps)
        except Exception:
            pass
    x32 = x_flat.float()
    rsqrt = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    return torch.nn.functional.linear(x32, _f32c(hc_fn)) * rsqrt
