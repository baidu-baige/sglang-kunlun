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
"""Fine-grained Kunlun replacements for SGLang Triton/JIT kernel symbols."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

import xspeedgate_ops

def _debug_tensor_meta(name: str, value: object) -> str:
    """Format tensor metadata or a generic value for debug logging."""
    if isinstance(value, torch.Tensor):
        return f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
    return f"{name}: value={value!r}, type={type(value).__name__}"


KernelKey = Tuple[str, str]


@dataclass(frozen=True)
class KernelSpec:
    """Describe an upstream kernel symbol and its Kunlun replacement."""

    module_path: str
    kernel_name: str
    impl: Callable
    metadata: Mapping[str, object] = field(default_factory=dict)


class KernelLauncher:
    """Compatibility wrapper for Triton ``kernel[grid](*args)`` call sites."""

    def __init__(self, name: str, impl: Callable):
        """Initialize the launcher with an upstream name and replacement callable."""

        self.name = name
        self.impl = impl

    def __getitem__(self, grid):
        """Return a callable compatible with Triton ``kernel[grid]`` syntax."""

        def run(*args, **kwargs):
            """Invoke the Kunlun replacement implementation."""

            signature = inspect.signature(self.impl)
            if "grid" in signature.parameters:
                kwargs.setdefault("grid", grid)
            return self.impl(*args, **kwargs)

        return run

    def __call__(self, *args, **kwargs):
        """Invoke the replacement directly for non-indexed call sites."""

        return self.impl(*args, **kwargs)


_TRITON_OPS: Dict[KernelKey, KernelSpec] = {}
_JIT_OPS: Dict[KernelKey, KernelSpec] = {}
# Upstream implementations captured at patch time (before the symbol is swapped),
# so capture-gated fallbacks can still call the original in eager mode.
_UPSTREAM_ORIGINALS: Dict[KernelKey, Callable] = {}


def register_triton_op(
    module_path: str,
    kernel_name: str,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Callable:
    """Register a Kunlun replacement for an upstream Triton kernel symbol."""

    def decorator(fn: Callable) -> Callable:
        """Store the decorated function as the replacement implementation."""
        key = (module_path, kernel_name)
        if key in _TRITON_OPS:
            raise ValueError(f"triton op already registered: {key}")
        _TRITON_OPS[key] = KernelSpec(module_path, kernel_name, fn, metadata or {})
        return fn

    return decorator


def register_jit_op(
    module_path: str,
    kernel_name: str,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Callable:
    """Register a Kunlun replacement for an upstream Python JIT helper."""

    def decorator(fn: Callable) -> Callable:
        """Store the decorated function as the replacement implementation."""
        key = (module_path, kernel_name)
        if key in _JIT_OPS:
            raise ValueError(f"jit op already registered: {key}")
        _JIT_OPS[key] = KernelSpec(module_path, kernel_name, fn, metadata or {})
        return fn

    return decorator


def registered_triton_ops() -> Dict[KernelKey, KernelSpec]:
    """Return registered Triton kernel replacement specifications."""

    return dict(_TRITON_OPS)


def registered_jit_ops() -> Dict[KernelKey, KernelSpec]:
    """Return registered Python JIT helper replacement specifications."""

    return dict(_JIT_OPS)


def _replace_imported_bindings(original: object, replacement: object) -> None:
    """Patch already-imported SGLang aliases that still point to the old symbol."""

    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("sglang."):
            continue
        namespace = getattr(module, "__dict__", None)
        if not namespace:
            continue
        for symbol_name, value in list(namespace.items()):
            if value is not original:
                continue
            setattr(module, symbol_name, replacement)
            logger.info("kernel_ops: patched imported binding %s.%s", module_name, symbol_name)


def _patch_symbol(spec: KernelSpec, replacement: object) -> None:
    """Replace one upstream module symbol with its Kunlun implementation."""

    key = (spec.module_path, spec.kernel_name)
    try:
        module = importlib.import_module(spec.module_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("kernel_ops: failed to import %s.%s: %s", *key, exc)
        return
    if not hasattr(module, spec.kernel_name):
        logger.warning("kernel_ops: %s.%s not found", *key)
        return
    original = getattr(module, spec.kernel_name)
    _UPSTREAM_ORIGINALS[key] = original
    _replace_imported_bindings(original, replacement)
    setattr(module, spec.kernel_name, replacement)
    logger.info("kernel_ops: patched %s.%s", *key)


def install() -> None:
    """Install all registered Kunlun kernel replacements into upstream modules."""

    for spec in _TRITON_OPS.values():
        replacement = (
            spec.impl
            if spec.metadata.get("call_style") == "direct"
            else KernelLauncher(spec.kernel_name, spec.impl)
        )
        _patch_symbol(spec, replacement)
    for spec in _JIT_OPS.values():
        _patch_symbol(spec, spec.impl)


@register_triton_op("sglang.kernels.ops.memory.common", "write_req_to_token_pool_triton")
def write_req_to_token_pool_triton(
    req_to_token_ptr: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_tensors: torch.Tensor,
    pre_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_to_token_ptr_stride: int,
) -> None:
    """Write request token mappings into the request-to-token pool."""

    torch.ops.xspeedgate_ops.write_req_to_token_pool(
        req_to_token_ptr,
        req_pool_indices.to(torch.int32),
        prefix_tensors,
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc.to(torch.int64),
    )


@register_triton_op("sglang.kernels.ops.memory.common", "get_last_loc_kernel")
def get_last_loc_kernel(
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
    result: torch.Tensor,
    num_tokens: int,
    req_to_token_stride: int,
    BLOCK_SIZE: int,
) -> None:
    """Return the last cache location for each request prefix."""

    if prefix_lens_tensor.dtype != torch.int64:
        prefix_lens_tensor = prefix_lens_tensor.to(torch.int64)
    loc = torch.ops.xspeedgate_ops.get_last_loc(
        req_to_token, req_pool_indices_tensor[:num_tokens], prefix_lens_tensor[:num_tokens]
    )
    result[:num_tokens].copy_(loc.to(result.dtype))


@register_triton_op(
    "sglang.kernels.ops.kvcache.cache_ops",
"concat_and_cast_mha_k_kernel",
)
def concat_and_cast_mha_k_kernel(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    head_cnt: int,
    k_stride0: int,
    k_stride1: int,
    nope_stride0: int,
    nope_stride1: int,
    rope_stride0: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    """Concatenate and cast MHA key tensors into the destination buffer."""

    torch.ops.xspeedgate_ops.concat_and_cast_mha_k(k, k_nope, k_rope)


@register_triton_op(
    "sglang.kernels.ops.attention.pad",
    "seqlens_expand_kernel",
)
def seqlens_expand_kernel(
    extend_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    offsets: torch.Tensor,
    output: torch.Tensor,
    N: int,
    **kwargs,
) -> None:
    """Expand per-request sequence lengths into token positions."""

    extend_seq_lens = extend_seq_lens[:N]
    seq_lens = seq_lens[:N]
    offsets = offsets[:N]
    token_offsets = torch.arange(output.numel(), device=output.device, dtype=output.dtype)
    request_offsets = torch.repeat_interleave(offsets.to(output.dtype), extend_seq_lens)
    starts = (seq_lens - extend_seq_lens + 1).to(output.dtype)
    request_starts = torch.repeat_interleave(starts, extend_seq_lens)
    output.copy_(request_starts + token_offsets - request_offsets)


@register_triton_op(
    "sglang.kernels.ops.kvcache.kv_indices",
    "create_chunked_prefix_cache_kv_indices",
)
def create_chunked_prefix_cache_kv_indices(
    req_to_token_ptr: torch.Tensor,
    req_pool_indices_ptr: torch.Tensor,
    chunk_start_idx_ptr: torch.Tensor,
    chunk_seq_lens_ptr: torch.Tensor,
    chunk_cu_seq_lens_ptr: torch.Tensor,
    chunk_kv_indices_ptr: torch.Tensor,
    req_to_token_ptr_stride: int,
) -> None:
    """Create flattened KV indices for chunked prefix cache on Kunlun."""

    batch_size = req_pool_indices_ptr.numel()
    for batch_index in range(batch_size):
        req_pool_index = int(req_pool_indices_ptr[batch_index].item())
        chunk_start = int(chunk_start_idx_ptr[batch_index].item())
        chunk_seq_len = int(chunk_seq_lens_ptr[batch_index].item())
        if chunk_seq_len <= 0:
            continue
        chunk_kv_offset = int(chunk_cu_seq_lens_ptr[batch_index].item())
        chunk_kv_indices_ptr[
            chunk_kv_offset : chunk_kv_offset + chunk_seq_len
        ].copy_(
            req_to_token_ptr[
                req_pool_index, chunk_start : chunk_start + chunk_seq_len
            ].to(chunk_kv_indices_ptr.dtype)
        )


@register_triton_op(
    "sglang.kernels.ops.memory.memcpy_triton",
    "memcpy_triton_kernel",
)
def memcpy_triton_kernel(
    dst: torch.Tensor,
    src: torch.Tensor,
    offset: torch.Tensor,
    sz: torch.Tensor,
    offset_src: bool,
    chunk_size: int,
    BLOCK_SIZE: int,
) -> None:
    """Copy token ranges between tensors for data-parallel attention."""

    torch.ops.xspeedgate_ops.memcpy_token(dst, src, 0, offset, sz, offset_src)


# [Dropped for 0.5.14] ``overlap_utils._resolve_future_token_ids`` no longer
# exists; future-token-id resolution moved to ``FutureMap._resolve_spec_extras``
# (pure torch). This deployment also runs with --disable-overlap-schedule, so
# the path is not exercised. No Triton kernel to replace.


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.metadata_kernel",
    "init_compression_metadata",
)
def init_compression_metadata(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
):
    """Pure-torch replacement for the DSV4 compression-metadata Triton kernel.

    The upstream ``_init_compressed_attn_metadata_kernel`` is a ``triton.jit``
    kernel; the Kunlun triton backend cannot compile it (``PassManager::run
    failed``). This vectorized torch version reproduces the exact kernel logic
    (see ``sglang/srt/layers/attention/dsv4/metadata_kernel.py``).
    """
    device = seq_lens.device
    seq_lens_i64 = seq_lens.to(torch.int64)
    positions_i64 = positions.to(torch.int64)
    raw_out_loc_i64 = raw_out_loc.to(torch.int64)
    # c4/c128 out_loc are per write-token; seq_lens/positions may be longer
    # (CP pads the causal metadata). Align to raw_out_loc so XPU where gets
    # matching shapes and we reproduce the kernel's is_write_token masking.
    _num_write_tokens = raw_out_loc_i64.shape[0]
    seq_lens_w = seq_lens_i64[:_num_write_tokens]

    # --- c4 ---
    c4_should = (seq_lens_w % 4) == 0
    c4_out_loc = torch.where(
        c4_should, raw_out_loc_i64 // 4, torch.zeros_like(raw_out_loc_i64)
    ).to(torch.int64)
    c4_positions = (positions_i64 & (~3)).to(torch.int32)  # noqa: F841 (kernel discards)
    c4_seq_lens_raw = (seq_lens_i64 // 4).to(torch.int32)
    c4_seq_lens_clamp1 = torch.clamp(c4_seq_lens_raw, min=1).to(torch.int32)

    # --- c128 ---
    c128_should = (seq_lens_w % 128) == 0
    c128_out_loc = torch.where(
        c128_should, raw_out_loc_i64 // 128, torch.zeros_like(raw_out_loc_i64)
    ).to(torch.int64)
    c128_positions = (positions_i64 & (~127)).to(torch.int32)  # noqa: F841
    c128_seq_lens_raw = (seq_lens_i64 // 128).to(torch.int32)
    c128_seq_lens_clamp1 = torch.clamp(c128_seq_lens_raw, min=1).to(torch.int32)

    c128_page_indices = None
    if compute_page_indices:
        assert page_table is not None and page_size > 0
        bs = seq_lens_i64.shape[0]
        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_max_seq_len = c128_page_size * max_pages

        offsets = torch.arange(c128_max_seq_len, device=device, dtype=torch.int64)
        page_idx = offsets // c128_page_size  # [L]
        offset_in_page = offsets % c128_page_size  # [L]

        page_valid = page_idx < max_pages
        page_idx_clamped = torch.where(
            page_valid, page_idx, torch.zeros_like(page_idx)
        )
        # gather page_table[b, page_idx] -> [bs, L]
        pt = page_table.to(torch.int64)
        gathered = pt[:, page_idx_clamped]  # [bs, L]
        gathered = torch.where(
            page_valid.unsqueeze(0), gathered, torch.zeros_like(gathered)
        )
        c_page_indices = gathered * c128_page_size + offset_in_page.unsqueeze(0)

        valid = offsets.unsqueeze(0) < c128_seq_lens_raw.to(torch.int64).unsqueeze(1)
        c_page_indices = torch.where(
            valid, c_page_indices, torch.full_like(c_page_indices, -1)
        )
        c128_page_indices = c_page_indices.to(torch.int32).contiguous()

    return (
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_raw,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )


def _ceil_ue8m0_exp_torch(sf: torch.Tensor) -> torch.Tensor:
    """`_ceil_ue8m0_exp`: biased exponent, rounded up when a mantissa is
    present, clamped to [1, 254]."""
    bits = sf.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exp = exp + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return exp.clamp(1, 254)


def _fp4_e2m1_code_torch(v: torch.Tensor, rne: bool) -> torch.Tensor:
    """`_fp4_e2m1_code` / `_fp4_e2m1_code_rne`: e2m1 code with the sign in bit 3."""
    ax = v.abs().clamp_max(6.0)
    thresholds = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    if rne:
        idx = torch.zeros_like(ax, dtype=torch.int32)
        is_boundary = torch.zeros_like(ax, dtype=torch.bool)
        for t in thresholds:
            idx = idx + (ax >= t).to(torch.int32)
            is_boundary = is_boundary | (ax == t)
        # Round-half-to-even: an odd index exactly on a boundary drops to the
        # even one.
        idx = torch.where(is_boundary & ((idx & 1) == 1), idx - 1, idx)
    else:
        idx = torch.zeros_like(ax, dtype=torch.int32)
        for t in thresholds:
            idx = idx + (ax > t).to(torch.int32)
    sign = ((v < 0) & (idx != 0)).to(torch.int32)
    return idx | (sign << 3)


def _stream_capturing() -> bool:
    """True only while a cuda-graph capture is in progress on this stream.

    The torch fp4/low-ratio/fill fallbacks below are capture-only: the upstream
    Triton kernels run fine eagerly on XPU but cannot launch inside capture
    (err_code -900). Conversely the torch scatters, while numerically
    identical, corrupted the argmax input when run eagerly on XMLIR, so eager
    must keep Triton.
    """
    # [CG-DETECT] torch.cuda.is_current_stream_capturing() returns False inside
    # this build's capture (verified by trace), so the capture-only fallbacks
    # were silently skipped and the Triton kernels ran in the graph. Prefer
    # sglang's process-global capture flag, torch only as a fallback.
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if bool(get_is_capture_mode()):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _jit_original(module_path: str, name: str) -> Optional[Callable]:
    """Return the pristine upstream impl captured by ``install()`` before the
    symbol was swapped. Used by the capture-gated fallbacks to pass through to
    the original Triton kernel when not capturing."""
    return _UPSTREAM_ORIGINALS.get((module_path, name))


# ---- fp4 indexer quantize: native XSpeedGate op (r82) ----------------------
# ``xspeedgate_ops.quantize_fp4_indexer_tensor`` does the whole ue8m0 fp4
# quantize in one kernel. It is bit-exact with ``_quantize_fp4_indexer_torch``
# (12/12: three geometries x both ``rne`` arms, plus exact e2m1 midpoints,
# all zeros and a 1e-30 denormal scale) and 2.075 -> 0.114 ms/call at n=61436
# (18x per-call sync, 1.747 -> 0.081 pipelined, which matches the in-engine
# 80 us median). The torch chain it replaces was 0.76% of device time in the
# r80 census (_fp4_e2m1_code_torch 7.08 + _quantize_fp4_indexer_torch 1.63
# ms/step). ``KL_FP4_NATIVE=0`` restores the torch chain. A per-device probe
# still guards the native path because the op is bf16-only, and capture is
# skipped (the store's scatter is already capture-specific; keep that path
# exactly as delivered).
_FP4_NATIVE = os.environ.get("KL_FP4_NATIVE", "1") != "0"
_fp4_native_ok: Dict[str, bool] = {}


def _fp4_native_available(x: torch.Tensor) -> bool:
    """Probe once per device: op present, bf16 in, expected output shapes."""
    key = str(x.device)
    if key in _fp4_native_ok:
        return _fp4_native_ok[key]
    ok = False
    if x.dtype is torch.bfloat16:
        try:
            t = torch.zeros(128, 128, dtype=x.dtype, device=x.device)
            k, s = torch.ops.xspeedgate_ops.quantize_fp4_indexer_tensor(t, rne=False)
            ok = tuple(k.shape) == (128, 64) and tuple(s.shape) == (128,)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sglang-kunlun: native fp4 quantize unavailable on %s (%s);"
                " torch chain stays",
                key,
                exc,
            )
    _fp4_native_ok[key] = ok
    return ok


def _quantize_fp4_indexer_fast(
    x: torch.Tensor, rne: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Native quantize when it is safe, else the bit-exact torch chain."""
    if not _stream_capturing() and _FP4_NATIVE and _fp4_native_available(x):
        try:
            return torch.ops.xspeedgate_ops.quantize_fp4_indexer_tensor(
                x.contiguous(), rne=rne
            )
        except Exception as exc:  # noqa: BLE001
            _fp4_native_ok[str(x.device)] = False
            logger.warning(
                "sglang-kunlun: native fp4 quantize failed at runtime (%s);"
                " falling back to torch",
                exc,
            )
    return _quantize_fp4_indexer_torch(x, rne=rne)


def _quantize_fp4_indexer_torch(
    x: torch.Tensor, rne: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch ue8m0 fp4 quantize (bit-exact with the upstream Triton kernel).

    Four 32-element groups per 128-wide row; group scale is
    ``ceil_pow2_exp(max(|x|)/6)`` stored as one byte each in the int32, and the
    payload packs two 4-bit e2m1 codes per byte (low nibble = even column).
    Used both under capture (the upstream Triton kernel cannot launch there) and
    by the fused store below (the upstream 1-CTA-per-row kernel is launch-bound
    on XPU: ~1.7 ms flat here vs O(n) upstream, 10.9x at n=16384).
    """
    assert x.shape[-1] == 128
    x = x.contiguous().view(-1, 128)
    n = x.shape[0]
    x_fp4 = torch.empty((n, 64), device=x.device, dtype=torch.int8)
    x_sf = torch.empty((n,), device=x.device, dtype=torch.int32)
    if n == 0:
        return x_fp4, x_sf
    xf = x.to(torch.float32).view(n, 4, 32)
    sf = (xf.abs().amax(dim=2) / 6.0).clamp_min(1.0e-4)
    exp = _ceil_ue8m0_exp_torch(sf)
    x_sf.copy_(
        exp[:, 0] | (exp[:, 1] << 8) | (exp[:, 2] << 16) | (exp[:, 3] << 24)
    )
    scale = (exp << 23).view(torch.float32).unsqueeze(-1)
    code = _fp4_e2m1_code_torch(xf / scale, rne).view(n, 64, 2)
    x_fp4.copy_(((code[..., 0] & 0x0F) | ((code[..., 1] & 0x0F) << 4)).to(torch.int8))
    return x_fp4, x_sf


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.fp4_indexer",
    "quantize_fp4_indexer_tensor",
)
def quantize_fp4_indexer_tensor(
    x: torch.Tensor, rne: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Capture-safe fallback for the fp4 indexer quantize Triton kernel.

    The upstream ``_quantize_fp4_indexer_kernel`` launch fails inside
    cuda-graph capture on the Kunlun XPU backend (``xpuLaunchKernel ...
    err_code: -900``, hit at decode-graph capture via
    ``_low_ratio_write_group -> set_index_k_fp4``); eager Triton works, so the
    torch version below is used ONLY under capture.
    """
    if not _stream_capturing():
        original = _jit_original(
            "sglang.kernels.ops.attention.dsv4.fp4_indexer",
            "quantize_fp4_indexer_tensor",
        )
        if original is not None:
            return original(x, rne=rne)
    return _quantize_fp4_indexer_torch(x, rne=rne)


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.fp4_indexer",
    "store_fp4_index_k_cache",
)
def store_fp4_index_k_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool = False,
) -> None:
    """Fused, capture-safe replacement for ``store_fp4_index_k_cache``.

    Upstream is ``quantize_fp4_indexer_tensor`` (a 1-CTA-per-row Triton kernel,
    launch-bound on XPU) followed by ``_store_fp4_index_k_cache_kernel`` (also
    1-CTA-per-token, whose 4-wide ``tl.arange(0, 4)`` scale store is
    miscompiled: the scale bytes come out nondeterministic, two identical
    launches disagree). Both eager and capture paths therefore use a pure
    kernel quantize (the native XSpeedGate op when the arm is on and the op is
    available, else the bit-exact torch chain) plus the int32-word scatter
    below.

    The cache is ``[num_pages, page_size * 68]`` uint8: per page a
    ``page_size * 64`` payload region followed by a ``page_size * 4`` scale
    region. Viewed as int32 that is ``[num_pages, page_size * 17]``: the
    payload is 16 words per token and the scale is one packed word per token,
    so each token needs just one 16-word scatter and one scalar scatter. This
    is bit-exact with the reference layout (the packed scale already holds the
    four ue8m0 exponents in little-endian byte order) and deterministic.
    Measured vs upstream (n = tokens): full-op 4.0x @4096, 6.7x @8192,
    12.2x @16384, 26.6x @65536.
    """
    assert input.shape[-1] == 128
    assert cache.shape[1] == page_size * (64 + 4)
    assert cache.shape[1] % 4 == 0
    k_fp4, k_sf = _quantize_fp4_indexer_fast(input, rne=rne)
    n_tokens = input.numel() // input.shape[-1]
    if n_tokens == 0:
        return
    cache_i32 = cache.view(torch.int32)
    words_per_page = cache_i32.shape[1]
    loc = loc.to(torch.int64)
    if not _stream_capturing():
        # Eager: drop the padding entries (``loc == 0``). The upstream fused
        # kernel suppresses the store for ``out_loc`` of -1 (incomplete group)
        # and 0 (padding) alike, and the caller has already folded -1 into 0,
        # so dropping 0 matches its semantics. It is also ~12x faster: routing
        # every padded token to slot 0 turns the payload scatter into a
        # many-to-one write, which XPU's ``scatter_v2_large_index`` serializes
        # (17 ms vs 1.3 ms at n=32768 with 15% padding; 54 ms at 50%). Only the
        # (never-read) padding slot differed, so the real cache is bit-exact.
        keep = loc > 0
        loc = loc[keep]
        if loc.numel() == 0:
            return
        k_fp4 = k_fp4.view(torch.int32)[keep]
        k_sf = k_sf[keep]
    else:
        # Capture: boolean compaction would introduce a data-dependent shape, so
        # keep the padded tokens in and take the redundant slot-0 writes.
        k_fp4 = k_fp4.view(torch.int32)
    page = loc // page_size
    page_offset = loc - page * page_size

    # Payload: 16 contiguous int32 words per token at (page, page_offset * 16).
    flat = cache_i32.view(-1)
    word_base = page * words_per_page + page_offset * 16
    word_offsets = torch.arange(16, device=cache.device)
    flat[(word_base[:, None] + word_offsets[None, :]).reshape(-1)] = k_fp4.reshape(-1)
    # Scale: one packed int32 word per token at (page, page_size * 16 + page_offset).
    flat[page * words_per_page + page_size * 16 + page_offset] = k_sf


@register_jit_op(
    "sglang.kernels.ops.moe.fill_padded_rows",
    "_fill_padded_rows",
)
def _fill_padded_rows(
    x: torch.Tensor,
    num_token_non_padded: torch.Tensor,
    fill_value,
) -> None:
    """Capture-safe fallback for the MoE padded-row fill Triton helper.

    Upstream ``_fill_padded_rows`` launches the ``_fill_padded_rows_kernel``
    ``triton.jit`` kernel, which fails to launch inside cuda-graph capture on
    the Kunlun XPU backend (``xpuLaunchKernel ... err_code: -900``, hit at
    decode-graph capture via ``_zero_topk_weights_padded_region`` in the DSV4.1
    vl_routing / MoE topk path). Eager Triton works, so this masked-assignment
    version (which reads the pad count from device memory, no host sync) runs
    ONLY under capture.
    """
    if not _stream_capturing():
        original = _jit_original(
            "sglang.kernels.ops.moe.fill_padded_rows",
            "_fill_padded_rows",
        )
        if original is not None:
            return original(x, num_token_non_padded, fill_value)
    # Capturable fill: boolean-mask assignment (x[mask] = v) lowers to
    # nonzero() + scatter, which needs a device->host count sync that cannot
    # replay inside an XPU graph. A masked where keeps the whole write on
    # device with static shapes.
    rows = torch.arange(x.shape[0], device=x.device, dtype=torch.int64)
    pad_mask = (rows >= num_token_non_padded.reshape(1).to(torch.int64)).unsqueeze(-1)
    x.copy_(torch.where(pad_mask.expand_as(x), torch.full_like(x, fill_value), x))


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.metadata_kernel",
    "build_low_ratio_metadata",
)
def build_low_ratio_metadata(seq_lens: torch.Tensor, out_loc: torch.Tensor, topk: int):
    """Capture-safe fallback for the DSV4 low-ratio metadata Triton wrapper.

    Upstream ``build_low_ratio_metadata`` launches the ``_low_ratio_metadata``
    ``triton.jit`` kernel. On the Kunlun triton/XPU backend the launch fails
    inside cuda-graph capture with ``xpuLaunchKernel ... err_code: -900``
    (observed at decode-graph capture for every bs bucket); eager Triton works,
    so the torch version below runs ONLY under capture, mirroring the
    ``quantize_fp4_indexer_tensor`` gating. It reproduces the kernel's exact
    output contract.
    """
    if not _stream_capturing():
        original = _jit_original(
            "sglang.kernels.ops.attention.dsv4.metadata_kernel",
            "build_low_ratio_metadata",
        )
        if original is not None:
            return original(seq_lens, out_loc, topk)
    rows = seq_lens.numel()
    assert out_loc.numel() == rows
    lengths = seq_lens.to(torch.int32)
    loc = out_loc.to(torch.int64)
    len1 = lengths.clamp(min=1)
    len2 = (lengths >> 1).clamp(min=1)
    # PADDED aligns the page-index width to the next 64 multiple, matching the
    # kernel's `triton.cdiv(topk, 64) * 64`.
    padded = (topk + 63) // 64 * 64

    def _one(loc_col, lens, pad_value):
        # Fill with a device-side masked write (not torch.full): XMLIR may lower
        # scalar `full` to an H2D staging memcpy, which records a host-sourced
        # transfer into the graph - replay then re-reads freed host memory.
        page = torch.zeros((rows, padded), dtype=torch.int32,
                           device=seq_lens.device)
        page.fill_(-1)
        return (
            loc_col,
            lens.to(torch.int32),
            torch.minimum(
                lens,
                torch.full((), topk, dtype=torch.int32, device=lens.device),
            ),
            page,
        )

    c1 = _one(loc, len1, -1)
    c2 = _one(
        torch.where((lengths & 1) == 0, loc >> 1, torch.full_like(loc, -1)),
        len2,
        -1,
    )
    return (
        c1[0], c1[1], c1[2], c1[3],
        c2[0], c2[1], c2[2], c2[3],
    )


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4_attn_metadata_kernels",
    "build_page_table_positions_small",
)
def build_page_table_positions_small(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    swa_window: int,
):
    """Native replacement for the DSV4 ``_small_page_table`` Triton wrapper.

    Upstream ``build_page_table_positions_small`` launches the ``_small_page_table``
    ``triton.jit`` kernel, which cannot launch inside cuda-graph capture on XPU
    (and the Kunlun triton backend cannot compile it). ``xspeedgate_ops``
    ``page_table_positions`` has identical semantics and is capture-safe, so we
    replace the whole wrapper (keeps the upstream sglang tree pristine — this is
    the plugin-side home of what used to be an in-place core edit).
    """
    from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
        PageTablePositionsResult,
    )

    assert page_size > 0 and page_size & (page_size - 1) == 0
    lengths, positions, table, swa = torch.ops.xspeedgate_ops.page_table_positions(
        req_to_token.to(torch.int32),
        req_pool_indices_repeated.to(torch.int32),
        seq_lens_casual.to(torch.int32),
        int(max_seq_len),
        int(page_size),
        int(swa_window),
    )
    return PageTablePositionsResult(
        seq_lens_casual=lengths,
        positions_casual=positions,
        page_table=table,
        swa_topk_lengths=swa,
    )


# ---- DSV4 prefill causal expansion: cached gather instead of three scalar repeats ----
# Upstream ``expand_prefill_causally`` builds three length-num_tokens vectors with
# ``torch.repeat_interleave`` (lines 96/100/103 of the module). ``repeat_interleave_fwd``
# is a single-threaded scalar kernel: 61,436 elements cost 353.8 us each (5.76 ns/elem,
# ~1.4 GB/s) => 1.06 ms per forward for three vectors whose *token -> request* index
# depends only on the per-request extend lens. In a steady prefill pose (bs=1) that
# index is a constant, so it is built once and cached, then applied with two
# ``index_select`` gathers and one vector add (bandwidth-bound, few us each).
#
# ``(start_positions[r] + 1) + (t - start_locs[r])`` is regrouped as
# ``(start_positions[r] + 1 - start_locs[r]) + t``; every intermediate stays in int32
# and magnitudes are <= 65k, so the reassociation is exact and the output is
# bit-identical (verified in-pod against the upstream torch reference). The
# ``extend_start_loc is None`` host-loop branch and the "no host mirror and bs > 1"
# case pass straight through to the upstream original (captured by ``install()``),
# i.e. they behave exactly as before.
_EXPAND_PREFIX = "sglang.kernels.ops.attention.dsv4_attn_metadata_kernels"
_EXPAND_IDX_CACHE: Dict[Tuple[str, int, Optional[Tuple[int, ...]]], torch.Tensor] = {}
_EXPAND_IDX_CACHE_MAX = 8


def _expand_prefill_idx(
    device: torch.device,
    extend_seq_lens: torch.Tensor,
    extend_seq_lens_cpu: Optional[list],
    num_tokens: int,
) -> Optional[torch.Tensor]:
    """Return ``idx[t] = request index of token t``, or ``None`` to fall back."""
    if extend_seq_lens_cpu is not None:
        lens: Optional[Tuple[int, ...]] = tuple(int(v) for v in extend_seq_lens_cpu)
        if sum(lens) != num_tokens:
            return None
    elif extend_seq_lens.numel() == 1:
        # bs == 1: idx is all zeros regardless of the (unread) lens value.
        lens = None
    else:
        return None
    key = (str(device), int(num_tokens), lens)
    cached = _EXPAND_IDX_CACHE.get(key)
    if cached is not None:
        return cached
    if lens is None:
        idx = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    else:
        idx = torch.repeat_interleave(
            torch.arange(len(lens), dtype=torch.int64, device=device),
            torch.tensor(lens, dtype=torch.int64, device=device),
            output_size=num_tokens,
        )
    if len(_EXPAND_IDX_CACHE) >= _EXPAND_IDX_CACHE_MAX:
        _EXPAND_IDX_CACHE.pop(next(iter(_EXPAND_IDX_CACHE)))
    _EXPAND_IDX_CACHE[key] = idx
    return idx


@register_jit_op(_EXPAND_PREFIX, "expand_prefill_causally")
def expand_prefill_causally(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: Optional[torch.Tensor],
    seq_lens_cpu: Optional[list],
    extend_seq_lens_cpu: Optional[list],
    num_tokens: int,
    padded_num_tokens: Optional[int],
):
    """Cached-index replacement for the upstream ``expand_prefill_causally``."""

    from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
        ExpandPrefillCausallyResult,
    )

    if extend_start_loc is not None:
        device = req_pool_indices.device
        idx = _expand_prefill_idx(
            device, extend_seq_lens, extend_seq_lens_cpu, num_tokens
        )
        # A zero-token rank (empty CP/EP shard) can still be padded to a
        # tensor-parallel tier. ``idx`` is empty then, so ``[-1:]`` has nothing
        # to repeat over the pad rows; delegate that case to the reference
        # implementation instead of expanding an empty tensor.
        if idx is not None and idx.numel() > 0:
            base = (
                seq_lens.to(torch.int32)
                - extend_seq_lens.to(torch.int32)
                + 1
                - extend_start_loc.to(torch.int32)
            )
            seq_lens_casual = torch.arange(
                num_tokens, dtype=torch.int32, device=device
            ) + base.index_select(0, idx)
            req_pool_indices_repeated = req_pool_indices.index_select(0, idx)
            if padded_num_tokens is not None and padded_num_tokens > num_tokens:
                pad_size = padded_num_tokens - num_tokens
                seq_lens_casual = torch.nn.functional.pad(
                    seq_lens_casual, (0, pad_size), value=1
                )
                req_pool_indices_repeated = torch.cat(
                    (
                        req_pool_indices_repeated,
                        req_pool_indices_repeated[-1:].expand(pad_size),
                    )
                )
            return ExpandPrefillCausallyResult(
                seq_lens_casual=seq_lens_casual,
                req_pool_indices_repeated=req_pool_indices_repeated,
            )

    original = _jit_original(_EXPAND_PREFIX, "expand_prefill_causally")
    if original is None:
        from sglang.kernels.ops.attention import dsv4_attn_metadata_kernels as _m

        original = _m.expand_prefill_causally
    return original(
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        extend_start_loc=extend_start_loc,
        seq_lens_cpu=seq_lens_cpu,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        num_tokens=num_tokens,
        padded_num_tokens=padded_num_tokens,
    )


@register_jit_op("sglang.srt.model_executor.forward_batch_info", "clamp_position")
def clamp_position(seq_lens: torch.Tensor) -> torch.Tensor:
    """Compute non-negative last-token positions from sequence lengths."""

    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)


@register_jit_op("sglang.kernels.ops.quantization.hadamard", "hadamard_transform")
def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply Kunlun Hadamard transform replacement.

    Mirrors aiak_sglang's implementation: build the Hadamard matrix via
    ``kunlun_ops.gen_hadamard_matrix`` and apply it with ``kunlun_ops.matmul``,
    instead of the fused ``xspeedgate_ops.hadamard_transform`` op.
    """
    import kunlun_ops

    hidden_size = x.size(-1)
    hm = torch.empty((hidden_size, hidden_size), dtype=x.dtype, device=x.device)
    kunlun_ops.gen_hadamard_matrix(hm, scale)

    original_shape = x.shape
    if len(original_shape) > 2:
        x = x.view(-1, hidden_size)

    out = torch.empty_like(x, dtype=x.dtype, device=x.device)
    kunlun_ops.matmul(x, hm, out, False, True, 1.0, 0.0)

    if len(original_shape) > 2:
        out = out.view(*original_shape)

    return out


@register_jit_op("sglang.kernels.ops.activation.activation", "silu_and_mul")
def silu_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
) -> torch.Tensor:
    """Kunlun SwiGLU replacement for 0.5.14's JIT silu_and_mul.

    0.5.14 routes ``activation.SiluAndMul.forward_cuda`` through the nvcc-JIT
    ``sglang.jit_kernel.activation.silu_and_mul`` (which fails on Kunlun: nvcc
    11.7 rejects ``-std=c++20``). Replace it with ``kunlun_ops.swiglu``.

    The 0.5.14 signature adds optional ``expert_ids`` / ``expert_step`` for
    filtered MoE activation; the Kunlun SwiGLU op does not support per-expert
    filtering, so assert it is unused on this path.
    """
    import kunlun_ops

    assert expert_ids is None, (
        "Kunlun silu_and_mul does not support expert_ids filtering"
    )
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    kunlun_ops.swiglu(x=input, y=out)
    return out


@register_jit_op("sglang.srt.utils.common", "fast_topk")
def fast_topk(values: torch.Tensor, topk: int, dim: int):
    """Patch speculative fast_topk aliases onto a torch implementation on Kunlun."""

    if topk == 1:
        return torch.max(values, dim=dim, keepdim=True)
    return torch.topk(values, topk, dim=dim)


# [kunlun-adapt] 上游已移除 jit grouped_topk（新框架为纯 torch 实现）；
# kunlun 的 moe_sigmoid_group_topk_norm 优化由 select_experts hook 层面另行接入。




@register_triton_op("sglang.kernels.ops.kvcache.mla_buffer", "set_mla_kv_buffer_kernel")
def set_mla_kv_buffer_kernel(
    kv_buffer: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    loc: torch.Tensor,
    reserved_skip_index: int,
    buffer_stride: int,
    nope_stride: int,
    rope_stride: int,
    nope_dim: int,
    rope_dim: int,
    *,
    BLOCK: int,
    DCP_RANK: int = 0,
    DCP_WORLD_SIZE: int = 1,
    USE_GDC: bool = False,
) -> None:
    """Store MLA KV tensors into the paged KV buffer.

    [kunlun-adapt] 上游新增 reserved_skip_index（跳过保留槽 0 的写入）与 DCP
    constexpr（Context-Parallel 分片）。Kunlun 单机无 DCP（DCP_WORLD_SIZE==1），
    reserved 槽通过 loc 掩码过滤实现；strides 由 kunlun op 内部按连续布局处理。
    """
    from kunlun_ops import set_mla_kv_buffer_triton as impl

    # [kunlun-adapt] 上游 reserved_skip_index（跳过 CUDA-graph padding 保留槽 0）
    # 在 triton kernel 内用设备端 mask 实现；此处直通 kunlun op、不做 host 掩码
    # 过滤——`bool(keep.all())` 会产生隐式同步，在 decode CUDA-graph capture 期
    # 非法。保留槽被写入属 padding 占位语义，无正确性影响（上游跳过它仅为
    # 保守约定）；如后续 kunlun op 原生支持 skip 参数，可直接透传。
    impl(
        kv_buffer,
        loc,
        cache_k_nope.contiguous().view(cache_k_nope.shape[0], cache_k_nope.shape[-1]),
        cache_k_rope.contiguous().view(cache_k_rope.shape[0], cache_k_rope.shape[-1]),
    )


@register_triton_op("sglang.kernels.ops.kvcache.mla_buffer", "get_mla_kv_buffer_kernel")
def get_mla_kv_buffer_kernel(
    kv_buffer: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    loc: torch.Tensor,
    buffer_stride: int,
    nope_stride: int,
    rope_stride: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    """Load MLA KV tensors from the paged KV buffer."""

    torch.ops.xspeedgate_ops.get_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)


@register_triton_op("sglang.kernels.ops.memory.allocator", "alloc_extend_kernel")
def alloc_extend_kernel(
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
) -> None:
    """Allocate KV cache pages for extend batches."""

    if prefix_lens.dtype != torch.int64:
        prefix_lens = prefix_lens.to(torch.int64)
    if seq_lens.dtype != torch.int64:
        seq_lens = seq_lens.to(torch.int64)
    if last_loc.dtype != torch.int64:
        last_loc = last_loc.to(torch.int64)

    from torch_xmlir.nn.alloc_extend import Alloc_extend

    alloc_extend_op = Alloc_extend()
    xdnn_out_indices, _ = alloc_extend_op(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        page_size,
        out_indices.shape[0],
    )
    out_indices.copy_(xdnn_out_indices.to(out_indices.dtype))
    return

    # ret_value = torch.zeros(1, dtype=torch.int64, device=out_indices.device)
    # torch.ops.xspeedgate_ops.alloc_extend(
    #     prefix_lens,
    #     seq_lens,
    #     last_loc,
    #     free_pages,
    #     prefix_lens.shape[0],
    #     page_size,
    #     out_indices.shape[0],
    #     out_indices,
    #     ret_value,
    # )


@register_triton_op("sglang.kernels.ops.memory.allocator", "alloc_decode_kernel")
def alloc_decode_kernel(
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
) -> None:
    """Allocate KV cache pages for decode batches."""

    # Per xspeedgate_ops _cache.pyi: seq_lens=int64, last_loc=int32,
    # free_page=int64, out_indices=int64. Only last_loc must be int32, and
    # XPU device-side int64->int32 cast is unreliable, so round-trip via CPU.
    def _to_i32(t):
        if t.dtype == torch.int32:
            return t.contiguous()
        return t.detach().cpu().to(torch.int32).to(t.device).contiguous()

    # The Kunlun kernel indexes free_pages over the whole padded range
    # [0, bs_upper) without masking, so a shorter tensor is read out of bounds
    # It only ever consumes the first num_new_pages (<= bs) entries, 
    # so padding the tail with zeros is safe: those slots are
    # never written into out_indices. self.free_pages itself is untouched, which
    # keeps the allocator's page accounting intact.
    if free_pages.numel() < bs_upper:
        free_pages = torch.cat(
            [free_pages, free_pages.new_zeros(bs_upper - free_pages.numel())]
        )

    torch.ops.xspeedgate_ops.alloc_decode_kernel(
        seq_lens,
        _to_i32(last_loc),
        free_pages,
        out_indices.contiguous(),
        bs_upper,
        page_size,
        seq_lens.shape[0],
    )


@register_jit_op("sglang.kernels.ops.kvcache.kvcache", "store_cache")
def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_bytes: int = 0,
    num_split: int = 0,
) -> None:
    """Store key and value tensors into paged KV cache."""

    from kunlun_ops import reshape_and_cache

    reshape_and_cache(
        k.contiguous(), v.contiguous(), k_cache, v_cache, indices.to(torch.int32), None, None, 0
    )


def _copy_fused_metadata_fallback(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    nsa_cache_seqlens_src: torch.Tensor,
    seqlens_expanded_src: torch.Tensor | None,
    nsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: torch.Tensor | None,
    flashmla_num_splits_src: torch.Tensor | None,
    flashmla_metadata_src: torch.Tensor | None,
    cache_seqlens_dst: torch.Tensor,
    cu_seqlens_k_dst: torch.Tensor,
    page_table_1_dst: torch.Tensor,
    nsa_cache_seqlens_dst: torch.Tensor,
    seqlens_expanded_dst: torch.Tensor | None,
    nsa_cu_seqlens_k_dst: torch.Tensor,
    real_page_table_dst: torch.Tensor | None,
    flashmla_num_splits_dst: torch.Tensor | None,
    flashmla_metadata_dst: torch.Tensor | None,
    forward_mode: int,
    max_len: int,
    max_seqlen_k: int,
    seqlens_expanded_size: int,
) -> None:
    """Copy NSA metadata without invoking CUDA JIT fused kernels."""

    cache_seqlens_dst.copy_(cache_seqlens_src)
    cu_seqlens_k_dst[1:].copy_(cu_seqlens_k_src[1:])

    if forward_mode == 0:
        page_table_1_dst[:, :max_len].copy_(page_indices_src)
        nsa_cache_seqlens_dst.copy_(nsa_cache_seqlens_src)
    elif forward_mode == 1:
        page_table_1_dst[:, :max_seqlen_k].copy_(page_indices_src)
        if seqlens_expanded_dst is not None and seqlens_expanded_src is not None:
            seqlens_expanded_dst.copy_(seqlens_expanded_src)
        nsa_cache_seqlens_dst.copy_(nsa_cache_seqlens_src)
    elif forward_mode == 2:
        rows = page_indices_src.shape[0]
        page_table_1_dst[:rows, :max_seqlen_k].copy_(page_indices_src)
        if seqlens_expanded_dst is not None and seqlens_expanded_src is not None:
            seqlens_expanded_dst[:seqlens_expanded_size].copy_(seqlens_expanded_src)
        nsa_cache_seqlens_dst[:seqlens_expanded_size].copy_(nsa_cache_seqlens_src)

    nsa_cu_seqlens_k_dst[1 : 1 + seqlens_expanded_size].copy_(
        nsa_cu_seqlens_k_src[1 : 1 + seqlens_expanded_size]
    )

    if real_page_table_src is not None and real_page_table_dst is not None:
        rows, cols = real_page_table_src.shape
        real_page_table_dst[:rows, :cols].copy_(real_page_table_src)

    if flashmla_metadata_src is not None and flashmla_metadata_dst is not None:
        flashmla_metadata_dst[: seqlens_expanded_size + 1].copy_(
            flashmla_metadata_src[: seqlens_expanded_size + 1]
        )
    if flashmla_num_splits_src is not None and flashmla_num_splits_dst is not None:
        flashmla_num_splits_dst[: seqlens_expanded_size + 1].copy_(
            flashmla_num_splits_src[: seqlens_expanded_size + 1]
        )


@register_jit_op("sglang.kernels.ops.attention.fused_metadata_copy", "fused_metadata_copy_cuda")
def fused_metadata_copy_cuda(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    nsa_cache_seqlens_src: torch.Tensor,
    seqlens_expanded_src: torch.Tensor | None,
    nsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: torch.Tensor | None,
    flashmla_num_splits_src: torch.Tensor | None,
    flashmla_metadata_src: torch.Tensor | None,
    cache_seqlens_dst: torch.Tensor,
    cu_seqlens_k_dst: torch.Tensor,
    page_table_1_dst: torch.Tensor,
    nsa_cache_seqlens_dst: torch.Tensor,
    seqlens_expanded_dst: torch.Tensor | None,
    nsa_cu_seqlens_k_dst: torch.Tensor,
    real_page_table_dst: torch.Tensor | None,
    flashmla_num_splits_dst: torch.Tensor | None,
    flashmla_metadata_dst: torch.Tensor | None,
    forward_mode: int,
    bs: int,
    max_len: int,
    max_seqlen_k: int,
    seqlens_expanded_size: int,
) -> None:
    """Silently fall back from CUDA fused metadata copy on Kunlun."""

    _copy_fused_metadata_fallback(
        cache_seqlens_src,
        cu_seqlens_k_src,
        page_indices_src,
        nsa_cache_seqlens_src,
        seqlens_expanded_src,
        nsa_cu_seqlens_k_src,
        real_page_table_src,
        flashmla_num_splits_src,
        flashmla_metadata_src,
        cache_seqlens_dst,
        cu_seqlens_k_dst,
        page_table_1_dst,
        nsa_cache_seqlens_dst,
        seqlens_expanded_dst,
        nsa_cu_seqlens_k_dst,
        real_page_table_dst,
        flashmla_num_splits_dst,
        flashmla_metadata_dst,
        forward_mode,
        max_len,
        max_seqlen_k,
        seqlens_expanded_size,
    )


@register_jit_op("sglang.kernels.ops.attention.fused_metadata_copy", "fused_metadata_copy_multi_cuda")
def fused_metadata_copy_multi_cuda(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    nsa_cache_seqlens_src: torch.Tensor,
    nsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: torch.Tensor | None,
    flashmla_num_splits_src: torch.Tensor | None,
    flashmla_metadata_src: torch.Tensor | None,
    cache_seqlens_dst0: torch.Tensor,
    cu_seqlens_k_dst0: torch.Tensor,
    page_table_1_dst0: torch.Tensor,
    nsa_cache_seqlens_dst0: torch.Tensor,
    nsa_cu_seqlens_k_dst0: torch.Tensor,
    real_page_table_dst0: torch.Tensor | None,
    flashmla_num_splits_dst0: torch.Tensor | None,
    flashmla_metadata_dst0: torch.Tensor | None,
    cache_seqlens_dst1: torch.Tensor,
    cu_seqlens_k_dst1: torch.Tensor,
    page_table_1_dst1: torch.Tensor,
    nsa_cache_seqlens_dst1: torch.Tensor,
    nsa_cu_seqlens_k_dst1: torch.Tensor,
    real_page_table_dst1: torch.Tensor | None,
    flashmla_num_splits_dst1: torch.Tensor | None,
    flashmla_metadata_dst1: torch.Tensor | None,
    cache_seqlens_dst2: torch.Tensor,
    cu_seqlens_k_dst2: torch.Tensor,
    page_table_1_dst2: torch.Tensor,
    nsa_cache_seqlens_dst2: torch.Tensor,
    nsa_cu_seqlens_k_dst2: torch.Tensor,
    real_page_table_dst2: torch.Tensor | None,
    flashmla_num_splits_dst2: torch.Tensor | None,
    flashmla_metadata_dst2: torch.Tensor | None,
    bs: int,
    max_len: int,
    seqlens_expanded_size: int,
) -> None:
    """Silently copy multi-backend NSA metadata without CUDA JIT."""

    for dst in (
        (
            cache_seqlens_dst0,
            cu_seqlens_k_dst0,
            page_table_1_dst0,
            nsa_cache_seqlens_dst0,
            nsa_cu_seqlens_k_dst0,
            real_page_table_dst0,
            flashmla_num_splits_dst0,
            flashmla_metadata_dst0,
        ),
        (
            cache_seqlens_dst1,
            cu_seqlens_k_dst1,
            page_table_1_dst1,
            nsa_cache_seqlens_dst1,
            nsa_cu_seqlens_k_dst1,
            real_page_table_dst1,
            flashmla_num_splits_dst1,
            flashmla_metadata_dst1,
        ),
        (
            cache_seqlens_dst2,
            cu_seqlens_k_dst2,
            page_table_1_dst2,
            nsa_cache_seqlens_dst2,
            nsa_cu_seqlens_k_dst2,
            real_page_table_dst2,
            flashmla_num_splits_dst2,
            flashmla_metadata_dst2,
        ),
    ):
        _copy_fused_metadata_fallback(
            cache_seqlens_src,
            cu_seqlens_k_src,
            page_indices_src,
            nsa_cache_seqlens_src,
            None,
            nsa_cu_seqlens_k_src,
            real_page_table_src,
            flashmla_num_splits_src,
            flashmla_metadata_src,
            dst[0],
            dst[1],
            dst[2],
            dst[3],
            None,
            dst[4],
            dst[5],
            dst[6],
            dst[7],
            0,
            max_len,
            max_len,
            seqlens_expanded_size,
        )


@register_triton_op(
    "sglang.kernels.ops.speculative.multi_layer_eagle",
"rotate_input_ids_kernel",
)
def rotate_input_ids_kernel(
    input_ids: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    topk_index: torch.Tensor,
    select_index: torch.Tensor | None,
    *,
    BLOCK_SIZE: int,
) -> None:
    """Rotate Eagle draft input ids according to selected top-k indices."""

    if input_ids.dtype != torch.int64:
        input_ids = input_ids.to(torch.int64)
    if extend_seq_lens.dtype != torch.int64:
        extend_seq_lens = extend_seq_lens.to(torch.int64)
    if topk_index.dtype != torch.int64:
        topk_index = topk_index.to(torch.int64)
    torch.ops.xspeedgate_ops.rotate_input_ids_triton(
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        topk_index,
        select_index=select_index,
    )


# [kunlun-adapt] 上游 multi-layer-eagle 已重构，符号 assign_hidden_states_pool_triton 不复存在，注册移除
# [kunlun-adapt] 上游 multi-layer-eagle 已重构，符号 assign_hidden_states_pool_triton 不复存在，注册移除
def assign_hidden_states_pool_triton(
    hidden_states: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
    pool_size: int,
    num_seqs: int,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
) -> None:
    """Assign draft hidden states into the request hidden-state pool."""

    # pool_capacity = req_to_hidden_states_pool.shape[0]
    # for req in range(num_seqs):
    #     pool_idx = req_pool_indices[req]
    #     if pool_idx < 0 or pool_idx >= pool_capacity:
    #         continue
    #     extend_len = extend_seq_lens[req]
    #     start_loc = extend_start_loc[req]
    #     end_loc = start_loc + extend_len
    #     req_to_hidden_states_pool[pool_idx, :pool_size, :].copy_(
    #         hidden_states[end_loc - pool_size : end_loc, :]
    #     )

    pool_capacity = req_to_hidden_states_pool.shape[0]
    active_req_pool_indices = req_pool_indices[:num_seqs].to(torch.int64)
    write_req_pool_indices = active_req_pool_indices + 1
    valid_mask = (write_req_pool_indices >= 0) & (write_req_pool_indices < pool_capacity)
    if not bool(valid_mask.all().item()):
        valid_req_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        logger.debug(
            "assign_hidden_states_pool filters out-of-range req_pool_indices: "
            "pool_capacity=%s, num_seqs=%s, "
            "invalid_req_indices=%s, invalid_pool_indices=%s",
            pool_capacity,
            num_seqs,
            torch.nonzero(~valid_mask, as_tuple=False).flatten().detach().cpu().tolist(),
            active_req_pool_indices[~valid_mask].detach().cpu().tolist(),
        )
        active_req_pool_indices = active_req_pool_indices[valid_req_indices].contiguous()
        write_req_pool_indices = write_req_pool_indices[valid_req_indices].contiguous()
        extend_seq_lens = extend_seq_lens[:num_seqs][valid_req_indices].contiguous()
        extend_start_loc = extend_start_loc[:num_seqs][valid_req_indices].contiguous()
        num_seqs = active_req_pool_indices.numel()

    if num_seqs == 0:
        return

    for req in range(num_seqs):
        pool_idx = write_req_pool_indices[req]
        extend_len = extend_seq_lens[req]
        start_loc = extend_start_loc[req]
        end_loc = start_loc + extend_len
        req_to_hidden_states_pool[pool_idx, :pool_size, :].copy_(
            hidden_states[end_loc - pool_size : end_loc, :]
        )


# [kunlun-adapt] 上游 multi-layer-eagle 已重构，符号 rotate_input_ids_triton 不复存在，注册移除
def rotate_input_ids_triton(
    input_ids: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    topk_index: torch.Tensor,
    select_index: torch.Tensor | None = None,
):
    """Rotate draft input ids without launching the upstream Triton kernel."""

    batch_size = extend_seq_lens.shape[0]
    token_ids = topk_index.reshape(-1)[:batch_size]
    for pid in range(batch_size):
        start = int(extend_start_loc[pid].item())
        seq_len = int(extend_seq_lens[pid].item())
        if seq_len <= 0:
            continue
        if seq_len > 1:
            input_ids[start : start + seq_len - 1].copy_(
                input_ids[start + 1 : start + seq_len].clone()
            )
        if select_index is not None:
            last_pos = int(select_index[pid].item())
        else:
            last_pos = start + seq_len - 1
        input_ids[last_pos] = token_ids[pid].to(input_ids.dtype)
    return input_ids


# [kunlun-adapt] 上游 multi-layer-eagle 已重构，符号 assign_new_state_kernel 不复存在，注册移除
def assign_new_state_kernel(
    old_input_ids: torch.Tensor,
    old_positions: torch.Tensor,
    old_hidden_states: torch.Tensor,
    old_out_cache_loc: torch.Tensor,
    old_extend_seq_lens: torch.Tensor,
    old_extend_start_loc: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    next_token_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    padding_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
    step: int,
    stride_hidden_seq: int,
    stride_hidden_dim: int,
    stride_pool_req: int,
    stride_pool_step: int,
    stride_pool_dim: int,
    stride_req_token_0: int,
    stride_req_token_1: int,
    *,
    HIDDEN_DIM: int,
    BLOCK_SEQ: int,
    BLOCK_HID: int,
    grid: tuple[int, ...] | None = None,
) -> None:
    """Build the next Eagle draft state tensors from the previous state."""

    num_seqs = grid[0] if grid is not None else old_extend_seq_lens.numel()
    num_seqs = int(num_seqs)
    max_num_seqs = old_extend_seq_lens.shape[0]
    old_extend_len = old_input_ids.shape[0] // max_num_seqs
    new_extend_len = old_extend_len + 1
    hidden_dim = old_hidden_states.shape[1]
    device = old_input_ids.device

    extend_seq_lens[:num_seqs].fill_(new_extend_len)
    extend_start_loc[:num_seqs].copy_(
        old_extend_start_loc[:num_seqs]
        + torch.arange(num_seqs, dtype=old_extend_start_loc.dtype, device=device)
    )

    old_ids = old_input_ids[: num_seqs * old_extend_len].view(num_seqs, old_extend_len)
    old_pos = old_positions[: num_seqs * old_extend_len].view(num_seqs, old_extend_len)
    old_cache = old_out_cache_loc[: num_seqs * old_extend_len].view(num_seqs, old_extend_len)
    old_hidden = old_hidden_states[: num_seqs * old_extend_len].view(
        num_seqs, old_extend_len, hidden_dim
    )

    new_ids = input_ids[: num_seqs * new_extend_len].view(num_seqs, new_extend_len)
    new_pos = positions[: num_seqs * new_extend_len].view(num_seqs, new_extend_len)
    new_cache = out_cache_loc[: num_seqs * new_extend_len].view(num_seqs, new_extend_len)
    new_hidden = hidden_states[: num_seqs * new_extend_len].view(
        num_seqs, new_extend_len, hidden_dim
    )

    new_ids[:, :old_extend_len].copy_(old_ids)
    token_ids = next_token_ids[:num_seqs, 0] if next_token_ids.dim() > 1 else next_token_ids[:num_seqs]
    insert_pos = (old_extend_len - padding_lens[:num_seqs]).long().unsqueeze(1)
    new_ids.scatter_(1, insert_pos, token_ids.unsqueeze(1).to(new_ids.dtype))

    new_pos[:, 1:].copy_(old_pos)
    new_pos[:, 0] = (old_pos[:, 0] - 1).clamp(min=0)

    new_cache[:, 1:].copy_(old_cache)
    req_indices = req_pool_indices[:num_seqs].long()
    token_cols = (seq_lens[:num_seqs] - old_extend_len - 1).long()
    valid = token_cols >= 0
    first_locs = req_to_token[req_indices, token_cols.clamp(min=0)]
    new_cache[:, 0] = torch.where(valid, first_locs, new_cache[:, 0])

    new_hidden[:, 1:, :].copy_(old_hidden)
    new_hidden[:, 0, :].copy_(req_to_hidden_states_pool[req_indices + 1, -(step + 1)])


# [kunlun-adapt] 上游 multi-layer-eagle 已重构，符号 assign_new_state_triton 不复存在，注册移除
def assign_new_state_triton(
    next_token_ids: torch.Tensor,
    old_input_ids: torch.Tensor,
    old_positions: torch.Tensor,
    old_hidden_states: torch.Tensor,
    old_out_cache_loc: torch.Tensor,
    old_extend_seq_lens: torch.Tensor,
    old_extend_start_loc: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    padding_lens: torch.Tensor,
    num_seqs: int,
    step: int,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
) -> None:
    """Direct replacement for the v0.5.14 draft-extend graph state updater."""

    assign_new_state_kernel(
        old_input_ids,
        old_positions,
        old_hidden_states,
        old_out_cache_loc,
        old_extend_seq_lens,
        old_extend_start_loc,
        input_ids,
        positions,
        hidden_states,
        out_cache_loc,
        extend_seq_lens,
        extend_start_loc,
        next_token_ids,
        seq_lens,
        padding_lens,
        req_pool_indices,
        req_to_token,
        req_to_hidden_states_pool,
        step,
        old_hidden_states.stride(0),
        old_hidden_states.stride(1),
        req_to_hidden_states_pool.stride(0),
        req_to_hidden_states_pool.stride(1),
        req_to_hidden_states_pool.stride(2),
        req_to_token.stride(0),
        req_to_token.stride(1),
        HIDDEN_DIM=hidden_states.shape[1],
        BLOCK_SEQ=8,
        BLOCK_HID=64,
        grid=(num_seqs,),
    )


@register_triton_op(
    "sglang.kernels.ops.speculative.cache_locs",
    "assign_draft_cache_locs_contiguous",
)
def assign_draft_cache_locs_page_size_1(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    topk: int,
    speculative_num_steps: int,
) -> None:
    """Assign draft cache locations for the page_size==1 / topk==1 contiguous path.

    0.5.14 renamed the upstream Triton kernel to
    ``assign_draft_cache_locs_contiguous`` (in
    ``sglang.srt.speculative.triton_ops.cache_locs``), called from
    ``base_spec_worker.prepare_for_draft``. The Kunlun replacement copies
    ``topk * speculative_num_steps`` slots per request via xspeedgate_ops.
    """

    torch.ops.xspeedgate_ops.assign_draft_cache_locs_page_size_1(
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc.to(torch.int32),
        pool_len,
        topk,
        speculative_num_steps,
    )


# [Dropped for 0.5.14] ``fill_new_verified_id`` no longer exists upstream; the
# verified-id flow moved to ``FutureMap._resolve_spec_extras`` (pure torch) in
# ``sglang.srt.managers.overlap_utils`` and is not a Triton kernel. No XPU
# replacement needed.


@register_triton_op(
    "sglang.kernels.ops.speculative.eagle",
    "fill_bonus_tokens",
)
def fill_bonus_tokens(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    accept_stride: int,
) -> None:
    """Fill speculative bonus tokens without launching the upstream Triton kernel."""

    num_tokens = accept_lens.numel()
    if num_tokens == 0:
        return
    row_ids = torch.arange(num_tokens, device=accept_tokens.device, dtype=torch.int64)
    col_ids = accept_lens.to(torch.int64) - 1
    bonus_tokens[:num_tokens].copy_(
        accept_tokens[row_ids, col_ids].to(bonus_tokens.dtype)
    )


@register_triton_op("sglang.kernels.ops.speculative.eagle", "fill_accept_out_cache_loc")
def fill_accepted_out_cache_loc(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accepted_out_cache_loc: torch.Tensor,
    size_upper: int,
) -> None:
    """Collect accepted output cache locations from draft indices.

    0.5.14 renamed the upstream Triton kernel to ``fill_accept_out_cache_loc``
    (in ``sglang.srt.speculative.triton_ops.eagle``).
    """

    valid_indices = accept_index[accept_index != -1]
    if valid_indices.numel() > 0:
        accepted_out_cache_loc[: valid_indices.shape[0]] = out_cache_loc[valid_indices]


@register_triton_op(
    "sglang.kernels.ops.speculative.cache_locs",
"generate_draft_decode_kv_indices",
)
def generate_draft_decode_kv_indices(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    paged_kernel_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    positions: torch.Tensor,
    pool_len: int,
    kv_indices_stride: int,
    kv_indptr_stride: int,
    bs_upper: int,
    iter_upper: int,
    num_tokens_upper: int,
    page_size: int,
    *,
    grid: tuple[int, ...] | None = None,
) -> None:
    """Build draft-decode KV indices for the split v0.5.14 cache-locs op."""

    if grid is None:
        raise ValueError("generate_draft_decode_kv_indices requires Triton-style grid")
    num_steps, num_seqs, topk = int(grid[0]), int(grid[1]), int(grid[2])
    req_indices = req_pool_indices[:num_seqs].long()
    seq_lens = paged_kernel_lens[:num_seqs].long()
    pos = positions[: num_seqs * topk].long()

    for iters0 in range(num_steps):
        iters = iters0 + 1
        kv_indices_row = kv_indices[iters0]
        kv_indptr_row = kv_indptr[iters0]
        cum_seq_len = 0
        for bid in range(num_seqs):
            seq_len = int(seq_lens[bid].item())
            token_pool = req_to_token[req_indices[bid]]
            for topk_id in range(topk):
                kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (
                    seq_len + iters
                )
                if seq_len > 0:
                    kv_indices_row[kv_offset : kv_offset + seq_len].copy_(
                        token_pool[:seq_len].to(kv_indices_row.dtype)
                    )
                extend_offset = torch.arange(iters, device=kv_indices.device)
                if page_size == 1 or topk == 1:
                    start = seq_len + topk_id * num_steps
                else:
                    last_page_len = seq_len % page_size
                    num_new_pages_per_topk = (
                        last_page_len + num_steps + page_size - 1
                    ) // page_size
                    prefix_base = seq_len // page_size * page_size
                    start = (
                        prefix_base
                        + topk_id * num_new_pages_per_topk * page_size
                        + last_page_len
                    )
                kv_indices_row[
                    kv_offset + seq_len : kv_offset + seq_len + iters
                ].copy_(token_pool[start + extend_offset].to(kv_indices_row.dtype))
            cum_seq_len += seq_len

        kv_indptr_row[0] = 0
        for zid in range(1, num_seqs * topk + 1):
            base = int(pos[:zid].sum().item())
            kv_indptr_row[zid] = base + zid * iters


@register_triton_op("sglang.kernels.ops.speculative.cache_locs", "assign_extend_cache_locs")
def assign_extend_cache_locs(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    bs_upper: int,
) -> None:
    """Assign cache locations for accepted extended tokens.

    0.5.14 moved this Triton kernel to
    ``sglang.srt.speculative.triton_ops.cache_locs`` (wrapped by
    ``assign_extend_cache_locs_func``).
    """

    batch_size = req_pool_indices.shape[0]
    draft_token_num = out_cache_loc.numel() // batch_size if batch_size > 0 else 0
    result = torch.ops.xspeedgate_ops.assign_extend_cache_locs(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        batch_size,
        draft_token_num,
    )
    out_cache_loc.copy_(result.to(out_cache_loc.dtype))


# [Dropped for 0.5.14] ``spec_utils.create_extend_after_decode_spec_info`` was
# removed upstream; the extend-after-decode setup is inlined with torch ops in
# the modern EAGLE v2 pipeline. No Triton kernel to replace.


@register_triton_op("sglang.srt.mem_cache.allocation", "assign_req_to_token_pool")
def assign_req_to_token_pool(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    bs_upper: int,
) -> None:
    """Write accepted cache locations into the request token pool."""

    batch_size = req_pool_indices.shape[0]
    if out_cache_loc.dtype != req_to_token.dtype:
        out_cache_loc = out_cache_loc.to(req_to_token.dtype)
    torch.ops.xspeedgate_ops.assign_req_to_token_pool(
        req_pool_indices.to(torch.int32),
        req_to_token,
        start_offset.to(torch.int32),
        end_offset.to(torch.int32),
        out_cache_loc,
        pool_len,
        batch_size,
    )


@register_triton_op("sglang.kernels.ops.speculative.cache_locs", "align_evict_mask_to_page_size")
def align_evict_mask_to_page_size(
    seq_lens: torch.Tensor,
    evict_mask: torch.Tensor,
    page_size: int,
    num_draft_tokens: int,
    BLOCK_SIZE: int,
) -> None:
    """Keep partial pages from being evicted in Eagle verification."""

    torch.ops.xspeedgate_ops.align_evict_mask_to_page_size(
        seq_lens,
        evict_mask,
        page_size,
        num_draft_tokens,
        BLOCK_SIZE,
    )


@register_jit_op("sglang.kernels.ops.speculative.gather_spec_extras", "gather_spec_extras")
def gather_spec_extras(
    indices: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    hidden_states_buf: torch.Tensor | None,
):
    """Torch row-gather replacement for speculative extras on Kunlun."""

    indices = indices.to(torch.long).contiguous()
    topk_p = topk_p_buf[indices].contiguous()
    topk_index = topk_index_buf[indices].contiguous()
    bonus_tokens = output_tokens_buf[indices].contiguous()
    hidden_states = (
        hidden_states_buf[indices].contiguous()
        if hidden_states_buf is not None
        else None
    )
    return topk_p, topk_index, bonus_tokens, hidden_states


_sampler_reported = False


@register_jit_op(
    "sglang.kernels.ops.speculative.dspark.dspark_draft_model",
    "sample_step_tokens_triton",
)
def sample_step_tokens_triton(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
) -> torch.Tensor:
    """Deterministic torch sampler for one draft step.

    Under sustained web traffic the Triton combine kernel occasionally left a
    row of its int64 output unwritten; that stale float-bit pattern later
    became a draft token and was used as a Markov embedding index (illegal
    memory access). The torch path writes every row by construction.
    """
    global _sampler_reported
    if not _sampler_reported:
        _sampler_reported = True
        print("[SAMPLER] Kunlun torch step sampler active", flush=True)
    # Greedy requests carry temperature 0. Dividing by it makes the row inf ->
    # NaN and argmax then returns token 0 for every greedy draft, so the target
    # verify rejects every step. argmax over the temperature-1 softmax is the
    # same argmax, so use divisor 1 for those rows.
    temperature = temperatures.to(torch.float32)
    is_greedy = greedy_mask.to(torch.bool) | (temperature <= 0)
    divisor = torch.where(is_greedy, torch.ones_like(temperature), temperature)
    probs = torch.softmax(step_logits.float() / divisor[:, None], dim=-1)
    noise = torch.where(is_greedy[:, None], 1.0, exp_noise)
    return probs.div_(noise).argmax(dim=-1).to(torch.int64)


_accept_bad_token_reported = False


# ---- t>0 corruption fix ----
@register_jit_op(
    "sglang.kernels.ops.speculative.dspark.dspark_draft_model",
    "build_step_local_triton",
)
def build_step_local_triton(
    *, bias: torch.Tensor, base_local: torch.Tensor
) -> torch.Tensor:
    """Deterministic torch build of one draft step's local logits.

    Same contract as the triton kernel it replaces: ``base_local`` is this
    rank's [rows, per_partition] slice of the draft model's logits, ``bias``
    is the markov projection over the first ``org_width`` columns, and the
    result is their sum in float32 with the padding columns left at the base
    value.

    The triton version allocated its output with ``torch.empty`` and, under
    sustained traffic on this backend, left part of it unwritten -- the
    returned rows carried NaN and float32-max bit patterns on ~7-9% of calls.
    Those flowed through ``SoftmaxTemp`` into an all-non-finite
    ``draft_probs`` row, which is what drove the temperature>0 output
    corruption. This is the same failure, and the same remedy, as
    ``sample_step_tokens_triton`` above. Every element is written here by
    construction.
    """
    per_partition = base_local.shape[-1]
    pad = per_partition - bias.shape[-1]
    padded = (
        torch.nn.functional.pad(bias.to(torch.float32), (0, pad))
        if pad > 0
        else bias.to(torch.float32)
    )
    return base_local.to(torch.float32) + padded


@register_jit_op("sglang.kernels.ops.speculative.reject_sampling", "chain_speculative_sampling_triton")
def chain_speculative_sampling_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool,
) -> None:
    """Torch replacement for chain speculative rejection sampling on Kunlun."""
    global _accept_bad_token_reported

    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    # The chain buffers come from a persistent cache and are torch.empty; only
    # the slots the chain actually visits are written. Seed this call so any
    # unexpected read (a stale accept_index entry or a backend contract drift)
    # yields a valid token instead of an old float-bit pattern that would crash
    # the next draft's Markov embedding lookup.
    with torch.no_grad():
        accept_index[:batch_size].copy_(retrive_index[:batch_size])
        predicts[: batch_size * num_slots].copy_(
            candidates.reshape(-1).to(predicts.dtype)
        )
    for bid in range(batch_size):
        root_global_idx = int(retrive_index[bid, 0].item())
        accept_index[bid, 0] = root_global_idx
        last_accepted_global_idx = root_global_idx
        num_accept = 0
        cur_prob_row = 0
        continue_verifying = True

        step = 1
        while step < num_slots and continue_verifying:
            draft_token = int(candidates[bid, step].item())
            if draft_token < 0 or draft_token >= vocab_size:
                # A concurrent producer can leave an odd draft slot stale on
                # this backend. Never index target_probs with it: treat the
                # chain as rejected at this step and let the residual path
                # resample from the target distribution.
                if not _accept_bad_token_reported:
                    _accept_bad_token_reported = True
                    print(
                        "[ACCEPT-GUARD] out-of-range draft token rejected: "
                        f"dev={torch.cuda.current_device()} bid={bid} "
                        f"step={step} token={draft_token} "
                        f"vocab={vocab_size} dtype={candidates.dtype} "
                        f"row={candidates[bid].tolist()}",
                        flush=True,
                    )
                continue_verifying = False
                break
            p = target_probs[bid, cur_prob_row, draft_token]
            q = draft_probs[bid, cur_prob_row, draft_token]
            coin = uniform_samples[bid, step - 1]
            if bool((coin * q < p).item()):
                num_accept += 1
                cur_prob_row = step
                predicts[last_accepted_global_idx] = draft_token
                curr_global_idx = int(retrive_index[bid, step].item())
                accept_index[bid, num_accept] = curr_global_idx
                last_accepted_global_idx = curr_global_idx
                step += 1
            else:
                continue_verifying = False

        accept_token_num[bid] = num_accept

        target_row = target_probs[bid, cur_prob_row]
        if continue_verifying:
            residual = target_row
        else:
            residual = torch.clamp(target_row - draft_probs[bid, cur_prob_row], min=0)

        norm_sum = residual.sum()
        if bool((norm_sum <= 0).item()):
            final_token = vocab_size - 1
        else:
            threshold = uniform_samples_for_final_sampling[bid] * norm_sum
            above = torch.cumsum(residual, dim=0) > threshold
            if bool(above.any().item()):
                final_token = int(torch.argmax(above.to(torch.int32)).item())
            else:
                final_token = vocab_size - 1
        predicts[last_accepted_global_idx] = final_token


@register_triton_op(
    "sglang.kernels.ops.attention.position",
"compute_position_kernel",
)
def compute_position_kernel(
    positions: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    has_prefix: bool,
) -> None:
    """Compute token positions and extend start locations for a batch."""

    prefix_lens = extend_prefix_lens if has_prefix else torch.empty(0, device=extend_seq_lens.device, \
            dtype=extend_seq_lens.dtype)
    out_positions, out_start_loc = torch.ops.xspeedgate_ops.compute_position_kernel(
        prefix_lens, extend_seq_lens, positions.numel()
    )
    positions.copy_(out_positions.to(positions.dtype))
    extend_start_loc.copy_(out_start_loc.to(extend_start_loc.dtype))


@register_jit_op(
    "sglang.kernels.ops.attention.rope",
    "apply_rope_with_cos_sin_cache_inplace",
)
def apply_rope_with_cos_sin_cache_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
    fused_args: object | None = None,
) -> None:
    """Apply RoPE in-place using Kunlun's flashinfer-compatible op."""

    if fused_args is not None:
        raise NotImplementedError("fused_args is not supported on Kunlun RoPE")

    rope_dim = rope_dim or cos_sin_cache.shape[-1]
    head_size = q.shape[-1]
    q_rot, k_rot = torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
        positions=positions.flatten(),
        rotary_dim=rope_dim,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache.to(q.dtype),
        is_neox_style=is_neox,
        query=q,
        key=k,
        offsets=None,
    )
    q.copy_(q_rot.reshape_as(q))
    k.copy_(k_rot.reshape_as(k))


@register_triton_op(
    "sglang.kernels.ops.attention.dsa.index_buf_accessor",
"_get_k_triton_kernel",
)
def _get_k_triton_kernel(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    out: torch.Tensor,
    seq_len: int,
    page_size: int,
    buf_numel_per_page: int,
    index_head_dim: int,
    *,
    BLOCK_SIZE: int,
) -> None:
    """_get_k_triton_kernel"""
    result = torch.ops.xspeedgate_ops.get_k_kernel(
        buf, page_indices, seq_len, page_size, buf_numel_per_page, index_head_dim
    )
    out.copy_(result.to(out.dtype))


@register_triton_op(
    "sglang.kernels.ops.attention.dsa.index_buf_accessor",
    "_get_s_triton_kernel",
)
def _get_s_triton_kernel(
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    out: torch.Tensor,
    seq_len: int,
    page_size: int,
    buf_numel_per_page: int,
    s_offset_in_page: int,
) -> None:
    """_get_s_triton_kernel"""
    result = torch.ops.xspeedgate_ops.get_s_kernel(
        buf, page_indices, seq_len, page_size, buf_numel_per_page, s_offset_in_page
    )
    out.copy_(result.to(out.dtype))


@register_triton_op(
    "sglang.kernels.ops.attention.dsa.index_buf_accessor",
    "_set_k_and_s_triton_kernel",
)
def _set_k_and_s_triton_kernel(
    buf_fp8: torch.Tensor,
    buf_fp32: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    index_k_ptr_stride_0: int,
    *,
    PAGE_SIZE: int,
    BUF_NUMEL_PER_PAGE: int,
    NUM_K_ELEMS_PER_TOKEN: int,
    S_OFFSET_NBYTES_IN_PAGE: int,
    PRESHUFFLE_TILE: int = 0,  # [kunlun-adapt] 上游新增 constexpr；fp8 布局由 kunlun set_k_and_s_triton 内部决定
) -> None:
    """_set_k_and_s_triton_kernel"""
    from kunlun_ops import set_k_and_s_triton

    set_k_and_s_triton(
        buf=buf_fp8.view(torch.uint8).contiguous(),
        loc=loc.to(torch.int64),
        index_k=index_k.contiguous(),
        index_k_scale=index_k_scale.contiguous(),
        page_size=PAGE_SIZE,
    )


@register_triton_op(
    "sglang.kernels.ops.attention.dsa.triton_kernel",
    "_act_quant_kernel",
)
def _act_quant_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    s: torch.Tensor,
    M: int,
    N: int,
    *,
    group_size: int,
    round_scale: bool,
    BLOCK_M: int,
    BLOCK_N: int,
    **kwargs,
) -> None:
    """act_quant kernel replacement."""

    x_blocks = x.view(-1, group_size)
    scale = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    y.copy_((x_blocks / scale * 448.0).to(y.dtype).view_as(y))
    s.copy_(scale.view_as(s).to(s.dtype))


@register_triton_op(
    "sglang.kernels.ops.quantization.int8_kernel",
    "_per_token_quant_int8",
)
def _per_token_quant_int8(
    x: torch.Tensor,
    x_q: torch.Tensor,
    scales: torch.Tensor,
    x_sum: torch.Tensor | None,
    *,
    stride_x: int,
    stride_xq: int,
    N: int,
    CAL_SUM: bool,
    BLOCK: int,
    **kwargs,
) -> None:
    """per-token int8 quantization kernel replacement."""

    assert not CAL_SUM, "per_token_quant_int8: cal_sum is not supported on Kunlun."
    import kunlun_ops

    tmp_scales = torch.empty(x.shape[:-1], dtype=scales.dtype, device=x.device)
    kunlun_ops.quant2d(x, x_q, tmp_scales, force_sdnn=True)
    scales.copy_(tmp_scales.view_as(scales))
    if x_sum is not None:
        raise NotImplementedError("per_token_quant_int8: cal_sum is not supported on Kunlun.")


@register_triton_op(
    "sglang.kernels.ops.quantization.int8_kernel",
"_per_token_group_quant_int8",
)
def _per_token_group_quant_int8(
    y: torch.Tensor,
    y_q: torch.Tensor,
    y_s: torch.Tensor,
    y_stride: int,
    N: int,
    eps: float,
    *,
    int8_min: int,
    int8_max: int,
    BLOCK: int,
    **kwargs,
) -> None:
    """per-token-group int8 quantization kernel replacement."""

    raise NotImplementedError("per_token_group_quant_int8 is not supported on Kunlun.")


@register_jit_op(
    "sglang.kernels.ops.quantization.int8_kernel",
"sglang_per_token_group_quant_int8",
)
def sglang_per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
    enable_v2: Optional[bool] = None,
):
    """sglang_per_token_group_quant_int8"""
    raise NotImplementedError("sglang_per_token_group_quant_int8 is not supported on Kunlun.")


@register_triton_op(
    "sglang.kernels.ops.quantization.int8_kernel",
"_w8a8_block_int8_matmul",
)
def _w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_n: int,
    group_k: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    stride_As_m: int,
    stride_As_k: int,
    stride_Bs_k: int,
    stride_Bs_n: int,
    *,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    GROUP_SIZE_M: int,
    **kwargs,
) -> None:
    """w8a8 block int8 matmul kernel replacement."""

    raise NotImplementedError("w8a8_block_int8_matmul is not supported on Kunlun.")


# ============================================================================
# SWA page free: pure-torch replacement for get_and_clear_swa_pages. Its triton
# kernel (get_and_clear_swa_pages_kernel) fails to compile on XPU triton
# (RuntimeError: PassManager::run failed), crashing both prefill and decode when
# a request frees its SWA pages.
# ============================================================================


@register_jit_op("sglang.kernels.ops.memory.allocator", "get_and_clear_swa_pages")
def get_and_clear_swa_pages(
    full_page_representatives: torch.Tensor,
    mapping: torch.Tensor,
    page_size: int,
    check_page_mappings: bool = False,
):
    """Resolve one SWA page per FULL-page representative via mapping[full_reps],
    flag peers_mapped, then clear that full page's mapping slots. Mirrors the
    upstream get_and_clear_swa_pages_kernel semantics in torch."""
    num_pages = full_page_representatives.numel()
    swa_pages = torch.empty(num_pages, dtype=mapping.dtype, device=mapping.device)
    peers_mapped = torch.empty(num_pages, dtype=torch.bool, device=mapping.device)
    page_mappings_valid = (
        torch.empty(num_pages, dtype=torch.bool, device=mapping.device)
        if check_page_mappings
        else None
    )
    if num_pages:
        full_reps = full_page_representatives.to(torch.int64)
        swa_reps = mapping[full_reps]
        swa_pages = (swa_reps // page_size).to(mapping.dtype)
        peers_mapped = swa_reps > 0
        full_page_starts = full_reps // page_size * page_size
        offsets = torch.arange(page_size, device=mapping.device)
        mapping_offsets = full_page_starts[:, None] + offsets[None, :]
        if check_page_mappings:
            page_mapping = mapping[mapping_offsets]
            same = (page_mapping // page_size) == (swa_reps[:, None] // page_size)
            page_mappings_valid = (swa_reps > 0) & (
                ((page_mapping > 0) & ~same).sum(dim=1) == 0
            )
        mapping[mapping_offsets.reshape(-1)] = 0
    return swa_pages, peers_mapped, page_mappings_valid


# ============================================================================
# DSV4 tvm_ffi JIT kernel replacements (pure PyTorch)
# All shim implementations live in dsv4_torch_shim.py; registered here so
# they are patched into the upstream sglang.jit_kernel.dsv4.* modules at
# kernel_ops.install() time.
# ============================================================================

from sglang_kunlun.kernels.dsv4_torch_shim import (  # noqa: E402
    _fused_q_norm_rope_torch,
    _fused_k_norm_rope_flashmla_torch,
    _fused_k_norm_rope_flashmla_kunlun,
    _fused_rope_inplace_torch,
    _fused_rope_inplace_kunlun,
    _q_rope_store_torch,
    _rope_tail_fake_quant_fp4_torch,
    _c2_decode_pool_torch,
    _fp4_index_logits_decode_torch,
    _fused_norm_rope_inplace_torch,
    _fused_norm_rope_inplace_kunlun,
    _fused_q_indexer_rope_hadamard_quant_torch,
    _fused_q_indexer_rope_hadamard_quant_kunlun,
    _compress_forward_torch,
    _compress_norm_rope_store_torch,
    _compress_norm_rope_store_kunlun,
    _create_paged_compress_data_kunlun,
)


@register_jit_op("sglang.kernels.ops.attention.dsv4.elementwise", "fused_q_norm_rope")
def fused_q_norm_rope(
    q_input: torch.Tensor,
    q_output: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Pure-torch replacement: rmsnorm-self (no weight) + RoPE on Q."""
    _fused_q_norm_rope_torch(q_input, q_output, eps, freqs_cis, positions)


@register_jit_op("sglang.kernels.ops.attention.dsv4.elementwise", "fused_k_norm_rope_flashmla")
def fused_k_norm_rope_flashmla(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
    layout=None,
) -> None:
    """Kunlun-native fused K norm + RoPE + paged cache store.

    Step 1: ``kunlun_ops.rmsnorm`` (with weight)
    Step 2: ``xspeedgate_ops.flashinfer_rotary_embedding`` on rope part
    Step 3: ``xspeedgate_ops.set_k_and_s_v4`` (bf16) or torch FP8 quant+write

    ``layout`` mirrors the upstream signature (the caller now passes
    ``layout=self.kv_layout``). The Kunlun store implements only the V4 paged
    FlashMLA layout, matching the upstream XPU branch which asserts
    ``layout is KVLayout.V4`` (the V4.1 layouts are CUDA sm100 only).
    """
    if layout is not None:
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        layout = KVLayout.parse(layout)
        assert layout is KVLayout.V4, (
            "sglang_kunlun fused_k_norm_rope_flashmla only supports the V4 KV "
            f"layout (V4.1 layouts are CUDA sm100 only); got {layout}"
        )
    _fused_k_norm_rope_flashmla_kunlun(kv, kv_weight, eps, freqs_cis, positions, out_loc, kvcache, page_size)


@register_jit_op("sglang.kernels.ops.attention.dsv4.elementwise", "fused_rope_inplace")
def fused_rope_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """Kunlun-native GPT-J RoPE via ``xspeedgate_ops.flashinfer_rotary_embedding``.

    The pure-torch ``_fused_rope_inplace_torch`` is retained as reference.
    """
    _fused_rope_inplace_kunlun(q, k, freqs_cis, positions, inverse)


@register_jit_op("sglang.kernels.ops.attention.dsv4.q_rope_store", "q_rope_store")
def q_rope_store(
    q: torch.Tensor,
    output: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """XSpeedGate-native q_rope_store (RoPE on the last 64 dims of each 512-wide
    Q head, nope copied). Validated bit-identical to the torch reference; the
    pure-torch ``_q_rope_store_torch`` remains the fallback for any shape the
    native kernel rejects.
    """
    try:
        torch.ops.xspeedgate_ops.q_rope_store(q, output, freqs_cis, positions)
    except Exception:
        _q_rope_store_torch(q, output, freqs_cis, positions)


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant",
    "rope_tail_fake_quant_fp4",
)
def rope_tail_fake_quant_fp4(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
    block_size: int = 32,
    *,
    compressed_kv: bool = False,
) -> torch.Tensor:
    """XSpeedGate-native fp4 RoPE-tail fake-quant (validated bit-identical to the
    torch ref on the non-compressed path; native E4M3 on the compressed path).
    Falls back to the pure-torch reference on any unsupported shape/dtype.
    """
    try:
        xc = x.contiguous()
        d = xc.shape[-1]
        x2 = xc.reshape(-1, d)
        rows = x2.shape[0]
        t = freqs.shape[0]
        out2 = torch.empty_like(x2, dtype=torch.bfloat16)
        f_real = torch.view_as_real(freqs.contiguous()).contiguous()
        torch.ops.xspeedgate_ops.rope_tail_fake_quant_fp4_kernel(
            x2, f_real, out2,
            x2.stride(0), out2.stride(0), f_real.stride(0),
            rows // t, d, rope_dim,
            16 if compressed_kv else block_size,
            6.0 * (2.0 ** -126), bool(inverse), bool(compressed_kv), rows,
        )
        return out2.reshape(x.shape)
    except Exception:
        return _rope_tail_fake_quant_fp4_torch(
            x, freqs, rope_dim, inverse=inverse, block_size=block_size,
            compressed_kv=compressed_kv,
        )


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.c2_decode_pool", "c2_decode_pool"
)
def c2_decode_pool(
    kv: torch.Tensor,
    score: torch.Tensor,
    pos: torch.Tensor,
    raw_out_loc: torch.Tensor,
    out_loc: torch.Tensor,
    req: torch.Tensor,
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
    pad_row: int,
    *,
    ring_size: int = 0,
):
    """Pure-torch replacement for the Triton ratio-2 decode pair-pool (no XPU
    Triton). Updates state_kv/state_score in place; returns (pooled, group_pos,
    slots)."""
    try:
        return torch.ops.xspeedgate_ops.pair_pool_decode(
            kv, score, pos, raw_out_loc, out_loc, req, state_kv, state_score,
            pad_row, ring_size,
        )
    except Exception:
        return _c2_decode_pool_torch(
            kv, score, pos, raw_out_loc, out_loc, req, state_kv, state_score,
            pad_row, ring_size=ring_size,
        )


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.fp4_indexer", "fp4_index_logits_decode"
)
def fp4_index_logits_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Pure-torch replacement for the Triton fp4 decode indexer logits (no XPU
    Triton). Dequantizes the fp4 index-K cache and computes the per-position
    head-summed relu-weighted scores."""
    return _fp4_index_logits_decode_torch(
        q, weights, slots, lens, table, page_size
    )


@register_jit_op("sglang.kernels.ops.speculative.tree", "build_tree_kernel_efficient")
def build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    """Route EAGLE tree build to the Kunlun impl (upstream is a CUDA JIT that
    needs C++20 <concepts>, unavailable to nvcc's host g++ on P800)."""
    from sglang_kunlun.kernels.sgl_kernel_kunlun.speculative import (
        build_tree_kernel_efficient as _kunlun_build_tree,
    )
    _kunlun_build_tree(
        parent_list, selected_index, verified_seq_len, tree_mask, positions,
        retrive_index, retrive_next_token, retrive_next_sibling,
        topk, depth, draft_token_num, tree_mask_mode,
    )


@register_jit_op("sglang.kernels.ops.speculative.tree", "verify_tree_greedy")
def verify_tree_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    """Route EAGLE greedy verify to the Kunlun impl (upstream is a CUDA JIT)."""
    from sglang_kunlun.kernels.sgl_kernel_kunlun.speculative import (
        verify_tree_greedy as _kunlun_verify_tree_greedy,
    )
    _kunlun_verify_tree_greedy(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


@register_jit_op(
    "sglang.kernels.ops.speculative.sampling", "tree_speculative_sampling_target_only"
)
def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """Target-only tree speculative sampling on P800 昆仑芯 — pure torch.

    Upstream ``sglang.kernels.ops.speculative.sampling`` JIT-compiles
    ``sgl_kernel_jit_speculative_sampling`` (csrc/speculative/sampling.cuh ->
    ``TreeSpeculativeSamplingTargetOnly``); its header pulls in ``<concepts>``
    (C++20) which nvcc's host g++ on P800 can't find, so the build fails and the
    decode scheduler crashes (not a Triton/CUDA kernel we can run here).

    This is a faithful torch port of that CUDA kernel (threshold-based acceptance
    + ``relu(target - draft)`` residual final sampling). We do NOT reuse
    ``chain_speculative_sampling_triton`` (it ignores the thresholds and, with the
    target-only ``draft_probs == 0``, accepts every draft token -> gibberish) nor
    the ``kunlun_ops`` native op. Signature matches upstream exactly;
    ``deterministic`` is accepted for API parity.
    ``predicts``/``accept_index``/``accept_token_num`` are int32 flat/[bs,ndt]/[bs]
    buffers written in place; ``retrive_index`` holds flat indices into ``predicts``.
    """
    bs, ndt = candidates.shape
    nst = accept_index.shape[1]
    d = target_probs.shape[-1]
    ta = max(float(threshold_acc), 1e-9)
    ts = float(threshold_single)
    tp = target_probs.reshape(bs, ndt, d)
    dp = draft_probs.reshape(bs, ndt, d)
    coins = uniform_samples.reshape(bs, ndt)
    coins_f = uniform_samples_for_final_sampling.reshape(bs)
    ri = retrive_index.reshape(bs, ndt)
    rnt = retrive_next_token.reshape(bs, ndt)
    rns = retrive_next_sibling.reshape(bs, ndt)
    cand = candidates.reshape(bs, ndt)
    for bx in range(bs):
        prob_acc = 0.0
        cur_row = 0  # row within this batch's ndt target-prob rows
        coin = float(coins[bx, 0].item())
        last_acc = int(ri[bx, 0].item())
        accept_index[bx, 0] = last_acc
        num_acc = 0
        cur_index = 0
        for _ in range(1, nst):
            cur_index = int(rnt[bx, cur_index].item())
            while cur_index != -1:
                draft_index = int(ri[bx, cur_index].item())
                draft_tok = int(cand[bx, cur_index].item())
                tps = float(tp[bx, cur_row, draft_tok].item())
                prob_acc += tps
                if tps > 0.0 and (coin < prob_acc / ta or tps >= ts):
                    prob_acc = 0.0
                    cur_row = cur_index
                    coin = float(coins[bx, cur_index].item())
                    predicts[last_acc] = draft_tok
                    num_acc += 1
                    accept_index[bx, num_acc] = draft_index
                    last_acc = draft_index
                    break
                # reject: stash target mass into draft_probs, try next sibling
                dp[bx, cur_row, draft_tok] = tp[bx, cur_row, draft_tok]
                cur_index = int(rns[bx, cur_index].item())
            if cur_index == -1:
                break
        accept_token_num[bx] = num_acc
        # final sampling from relu(target - draft) at the last-accepted row
        target_row = tp[bx, cur_row]
        draft_row = dp[bx, cur_row] if num_acc != nst - 1 else torch.zeros_like(target_row)
        residual = torch.clamp(target_row - draft_row, min=0)
        total = float(residual.sum().item())
        if total <= 0.0:
            final_tok = d - 1
        else:
            gt = torch.cumsum(residual, dim=0) > (float(coins_f[bx].item()) * total)
            final_tok = int(torch.argmax(gt.to(torch.int32)).item()) if bool(gt.any().item()) else d - 1
        predicts[last_acc] = final_tok


@register_jit_op("flashinfer.sampling", "top_p_renorm_probs")
def top_p_renorm_probs(probs: torch.Tensor, top_p) -> torch.Tensor:
    """Route flashinfer's top-p renorm to the Kunlun-native op on P800 昆仑芯.

    ``dflash_utils.build_dflash_verify_target_probs`` (is_cuda branch) imports
    ``from flashinfer.sampling import top_p_renorm_probs as top_p_renorm_prob`` to
    build the verify ``target_probs`` for temperature/top-p sampling. On P800 that
    flashinfer kernel returns corrupted probabilities, so even a correct verify
    samples gibberish (temperature-only sampling is fine; adding ``top_p`` breaks
    it — verified by isolation). Patching the flashinfer source symbol also fixes
    the ``dflash_utils`` alias via install()'s imported-binding rewrite. Returns
    the renormalized probs (value-returning, unlike the in-place JIT helpers)."""
    from sglang_kunlun.kernels.sgl_kernel_kunlun.sampling import top_p_renorm_prob as _k
    return _k(probs, top_p)


@register_jit_op("flashinfer.sampling", "top_k_renorm_probs")
def top_k_renorm_probs(probs: torch.Tensor, top_k) -> torch.Tensor:
    """Kunlun-native top-k renorm replacement for flashinfer (see
    ``top_p_renorm_probs`` above). Only hit on the non-sparse ``target_probs``
    path; the sparse path does top-k via ``torch.topk`` directly."""
    from sglang_kunlun.kernels.sgl_kernel_kunlun.sampling import top_k_renorm_prob as _k
    return _k(probs, top_k)


@register_jit_op("sglang.kernels.ops.attention.dsv4.compress_old", "fused_norm_rope_inplace")
def fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Kunlun-native fused RMSNorm + GPT-J RoPE (DSA CP path).

    Routes to ``kunlun_ops.dpsk_v4_norm_rope_gptj`` (mode=2 DefaultForward).
    The pure-torch ``_fused_norm_rope_inplace_torch`` is retained as reference
    and remains callable directly for debugging/testing.
    """
    _fused_norm_rope_inplace_kunlun(kv, weight, eps, freq_cis, positions)


@register_jit_op("sglang.kernels.ops.attention.dsv4.elementwise", "fused_q_indexer_rope_hadamard_quant")
def fused_q_indexer_rope_hadamard_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
):
    """Kunlun-native: RoPE + 128-pt Hadamard + INT8 quant.

    The pure-torch ``_fused_q_indexer_rope_hadamard_quant_torch`` is retained as reference.
    """
    return _fused_q_indexer_rope_hadamard_quant_kunlun(q_input, weight, weight_scale, freqs_cis, positions)


@register_jit_op("sglang.kernels.ops.attention.dsv4.compress", "compress_forward")
def compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan,
    *,
    head_dim: int,
    compress_ratio,
    out=None,
    is_online: bool = False,
) -> torch.Tensor:
    """Pure-torch replacement: softmax-pool compress_forward."""
    return _compress_forward_torch(
        kv_score_buffer, kv_score_input, ape, plan,
        head_dim=head_dim, compress_ratio=compress_ratio, out=out, is_online=is_online,
    )


@register_jit_op("sglang.kernels.ops.attention.dsv4.compress", "compress_norm_rope_store")
def compress_norm_rope_store(
    kv: torch.Tensor,
    plan,
    *,
    norm_weight: torch.Tensor,
    norm_eps: float,
    freq_cis: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
    use_fp4: bool = False,
    bf16_store: bool = False,
    kvcache_scale: torch.Tensor = None,  # [kunlun-adapt] 上游新增；fp16/bf16 KV 下不参与
    rope_cache=None,  # [kunlun-adapt] 上游新增（HIP FP4 预计算表）；kunlun fused op 内部处理
    fp4_k_write_metadata=None,  # [kunlun-adapt] 上游新增（FP4 写元数据）；本路径不写 FP4
) -> None:
    """Kunlun-native: fused norm+rope via dpsk_v4_norm_rope_gptj + store."""
    _compress_norm_rope_store_kunlun(
        kv, plan,
        norm_weight=norm_weight, norm_eps=norm_eps, freq_cis=freq_cis,
        out_loc=out_loc, kvcache=kvcache, page_size=page_size,
        use_fp4=use_fp4, bf16_store=bf16_store,
    )


@register_jit_op("sglang.kernels.ops.attention.dsv4.attn", "triton_create_paged_compress_data")
def triton_create_paged_compress_data(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    use_req_ring: bool = False,
    block: int = 128,
):
    """Kunlun-native replacement for the v1 legacy compressor's Triton
    write_loc/extra_data builder. Routes to
    ``xspeedgate_ops.create_paged_compress_data`` (same op aiak_sglang uses)
    instead of the unsupported Triton kernel.

    ``use_req_ring`` (added by newer v1 ``compressor.py``) is accepted for
    signature compatibility and intentionally unused: the native op derives
    the ring-buffer addressing from ``ring_size`` / ``compress_ratio``.
    """
    return _create_paged_compress_data_kunlun(
        compress_ratio=compress_ratio,
        is_overlap=is_overlap,
        swa_page_size=swa_page_size,
        ring_size=ring_size,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        block=block,
    )


# ---------------------------------------------------------------------------
# TopK transform (dsv4 c4 indexer).  ``xspeedgate_ops.topk_transform`` provides
# a Kunlun-native fused topk kernel.  It does NOT output ``raw_indices`` (the
# hisparse index-recording tensor), so when the caller requests raw_indices we
# fall back to the pure-torch vectorized implementation.
# ---------------------------------------------------------------------------

@register_jit_op("sglang.kernels.ops.attention.dsv4.topk", "topk_transform_paged")
def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices=None,
) -> None:
    """Kunlun-native ``xspeedgate_ops.topk_transform`` (topk=512).

    ``xspeedgate_ops.topk_transform`` does NOT populate ``raw_indices``.
    Upstream sglang sets ``raw_indices`` in three cases:
      1. indexer capture is enabled (via ``get_global_indexer_capturer``)
      2. hisparse decode is active
      3. ``core_metadata.c4_sparse_raw_indices`` is pre-allocated (prefill)

    Cases 1 and 2 truly consume ``raw_indices`` downstream, so we fall back
    to the pure-torch vectorized path.  Case 3 is a passive prefill buffer
    that upstream downstream never reads on Kunlun (verified: no reader
    inside ``forward_c4_indexer`` past this call, and no other consumer of
    ``core_metadata.c4_sparse_raw_indices``), so we can safely ignore it and
    stay on the fast xspeedgate path.
    """
    import xspeedgate_ops  # noqa: F401

    # Detect the "real consumer" cases and fallback for them only.
    from sglang.srt.layers.attention.dsv4.indexer import get_global_indexer_capturer
    _capturer = get_global_indexer_capturer()
    _needs_raw = (out_raw_indices is not None) and (_capturer is not None)
    if _needs_raw:
        from sglang.srt.layers.attention.dsv4.indexer import (
            topk_transform_pytorch_vectorized as topk_transform_512_pytorch_vectorized,
        )
        topk_transform_512_pytorch_vectorized(
            scores, seq_lens, page_tables, out_page_indices,
            page_size, out_raw_indices,
        )
        return
    torch.ops.xspeedgate_ops.topk_transform(
        scores, seq_lens, page_tables, out_page_indices,
        512, None, page_size, True,
    )


@register_jit_op("sglang.kernels.ops.attention.dsv4.topk", "topk_transform_paged_v2")
def topk_transform_512_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
) -> None:
    """Kunlun-native topk (v2 signature drops raw_indices, keeps metadata for
    upstream compat; we ignore metadata since xspeedgate handles planning
    internally)."""
    import xspeedgate_ops  # noqa: F401

    torch.ops.xspeedgate_ops.topk_transform(
        scores, seq_lens, page_tables, out_page_indices,
        512, None, page_size, True,
    )
