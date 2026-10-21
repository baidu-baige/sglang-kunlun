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
        if module is None or not module_name.startswith(_PATCH_SCAN_PREFIXES):
            continue
        namespace = getattr(module, "__dict__", None)
        if not namespace:
            continue
        for symbol_name, value in list(namespace.items()):
            if value is not original:
                continue
            setattr(module, symbol_name, replacement)
            logger.info("kernel_ops: patched imported binding %s.%s", module_name, symbol_name)


# (module_path, kernel_name) -> the *first* object we ever saw there.  Modules
# that alias a symbol at import time (``from m import f as g``) keep their own
# reference, so the repair pass below needs the true original, not whatever
# ``getattr`` returns after an earlier patch.
_ORIGINALS: dict = {}

# Modules whose ``from m import f`` bindings must be re-pointed when a kernel is
# replaced.  The plugin's own modules alias kernels at import time too, so they
# must be scanned alongside sglang's.
_PATCH_SCAN_PREFIXES = ("sglang.", "sglang_kunlun.")


def repair_imported_bindings() -> int:
    """Re-point module-level aliases that still reference a pre-patch symbol.

    Must run *after* every module has been imported: ``_patch_symbol`` can only
    fix bindings that already exist, and sglang imports many kernel symbols with
    ``from x import y as z`` long before the model runs.
    """
    repaired = 0
    logger.warning(
        "kernel_ops: repair pass start, tracking %d originals", len(_ORIGINALS)
    )
    # New duplicate module objects may have appeared for already-patched files;
    # re-apply so every sibling carries the replacement.
    reapply_all()
    # Same idea for staticmethod targets on classes: the alias finder can leave
    # a second class object, which the hook registry never patched.
    force_plan_class_replacements()
    for (mod_path, sym), original in list(_ORIGINALS.items()):
        try:
            replacement = getattr(importlib.import_module(mod_path), sym)
        except Exception:
            continue
        if replacement is original:
            continue
        orig_code = getattr(original, "__code__", None)
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.startswith(_PATCH_SCAN_PREFIXES):
                continue
            namespace = getattr(module, "__dict__", None)
            if not namespace:
                continue
            for name, value in list(namespace.items()):
                if value is replacement:
                    continue
                same = value is original
                if not same and orig_code is not None:
                    # ``functools.wraps`` copies __wrapped__; unwrap and compare
                    # the underlying code object so wrapped aliases still match.
                    probe = getattr(value, "__wrapped__", None)
                    same = getattr(probe, "__code__", None) is orig_code or (
                        getattr(value, "__code__", None) is orig_code
                    )
                if not same:
                    continue
                # Guard against over-broad rebinds: torch.library /
                # register_custom_op wrappers can share code objects, and a
                # plain identity/code match would then rewrite unrelated
                # attributes (observed: parallel_state.inplace_all_reduce was
                # replaced by the last registered kernel).
                if name != sym and getattr(
                    value, "__name__", None
                ) != getattr(original, "__name__", None):
                    continue
                setattr(module, name, replacement)
                repaired += 1
                logger.warning(
                    "kernel_ops: repaired stale binding %s.%s", module_name, name
                )
    logger.warning(
        "kernel_ops: repaired %d stale bindings (scanned %d modules)",
        repaired,
        len(sys.modules),
    )
    return repaired


# The internal fork (baidu/wenxin/sglang v1.3.x) moved a number of kernel
# symbols out of their v0.5.x modules into ``sglang.kernels.ops.*``.  Unlike the
# ``compat_aliases`` table (which only covers modules that *disappeared*), these
# upstream modules still exist but no longer *define* the symbol, so the plugin
# registration would silently miss it.  Map the old module to the module that
# now defines the symbol so the patch still lands.
_RELOCATED_PATHS: Dict[str, Tuple[str, ...]] = {
    "sglang.srt.mem_cache.common": ("sglang.kernels.ops.memory.common",),
    "sglang.srt.layers.dp_attention": ("sglang.kernels.ops.memory.memcpy_triton",),
}


def _sibling_modules(module):
    """Every module object bound to the *same source file* as ``module``.

    ``compat_aliases``' meta-path finder can leave more than one module object
    bound to one file (measured: 25 of its 33 aliases end up duplicated at
    runtime), so the object ``importlib.import_module`` hands back is not
    necessarily the one the application imported.  Patching every sibling makes
    the replacement visible regardless of which alias path a caller used.
    """

    target = getattr(module, "__file__", None)
    if target is None:
        return [module]
    siblings = [module]
    for candidate in list(sys.modules.values()):
        if candidate is None or candidate is module:
            continue
        if getattr(candidate, "__file__", None) == target:
            siblings.append(candidate)
    return siblings


def _patch_symbol(spec: KernelSpec, replacement: object) -> None:
    """Replace one upstream module symbol with its Kunlun implementation."""

    module = None
    for mod_path in (spec.module_path,) + _RELOCATED_PATHS.get(spec.module_path, ()):
        try:
            candidate = importlib.import_module(mod_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "kernel_ops: failed to import %s.%s: %s",
                mod_path,
                spec.kernel_name,
                exc,
            )
            continue
        if not hasattr(candidate, spec.kernel_name):
            logger.warning("kernel_ops: %s.%s not found", mod_path, spec.kernel_name)
            continue
        module = candidate
        break
    if module is None:
        return
    for target in _sibling_modules(module):
        if not hasattr(target, spec.kernel_name):
            continue
        original = getattr(target, spec.kernel_name)
        if original is replacement:
            continue
        _ORIGINALS.setdefault((target.__name__, spec.kernel_name), original)
        _replace_imported_bindings(original, replacement)
        setattr(target, spec.kernel_name, replacement)
    logger.warning("kernel_ops: patched %s.%s", module.__name__, spec.kernel_name)


def force_plan_class_replacements() -> int:
    """Re-apply the compressor-plan replacements to *every* duplicate class.

    The compat-alias finder can leave a second copy of
    ``sglang.kernels.ops.attention.dsv4`` (and therefore a second
    ``CompressorPrefillPlan`` / ``CompressorDecodePlan`` class object) in
    ``sys.modules``.  ``hook_registry`` patches only the class it resolves, so
    callers that imported the class from the *other* copy keep the original
    ``generate`` and fall through to the tvm_ffi JIT (which cannot build on
    XPU).  Re-point ``generate`` on every copy we can find.

    Returns the number of class objects touched.
    """
    from sglang_kunlun.eb5_adapt.hooks.jit_kernel.dsv4.compress import (
        _decode_plan_generate_torch,
        _prefill_plan_generate_torch,
    )

    wanted = {
        "CompressorPrefillPlan": _prefill_plan_generate_torch,
        "CompressorDecodePlan": _decode_plan_generate_torch,
    }
    seen: Dict[int, str] = {}
    touched = 0
    for mod_name, module in list(sys.modules.items()):
        if module is None or not mod_name.startswith("sglang."):
            continue
        for cls_name, impl in wanted.items():
            cls = getattr(module, cls_name, None)
            if not isinstance(cls, type) or "generate" not in cls.__dict__:
                continue
            seen.setdefault(id(cls), f"{mod_name}.{cls_name}")
            raw = cls.__dict__["generate"]
            current = raw.__func__ if isinstance(raw, staticmethod) else raw
            if current is impl or getattr(current, "__wrapped__", None) is impl:
                continue
            setattr(cls, "generate", staticmethod(impl))
            touched += 1
    if seen:
        logger.warning(
            "kernel_ops: compressor-plan classes distinct=%d replaced=%d",
            len(seen),
            touched,
        )
        if os.environ.get("KUNLUN_DEBUG_PLAN_CLASSES"):
            for cid, where in seen.items():
                print(f"[kdebug] plan class {where} id={cid}", file=sys.stderr, flush=True)
    return touched


def reapply_all() -> None:
    """(Re-)install every registered Kunlun kernel replacement.

    Idempotent, and safe to call again late: the alias finder can create new
    module objects for an already-patched file while sglang imports, so a
    second pass is needed to cover those siblings.
    """

    for spec in _TRITON_OPS.values():
        replacement = (
            spec.impl
            if spec.metadata.get("call_style") == "direct"
            else KernelLauncher(spec.kernel_name, spec.impl)
        )
        _patch_symbol(spec, replacement)
    for spec in _JIT_OPS.values():
        _patch_symbol(spec, spec.impl)


def install() -> None:
    """Install all registered Kunlun kernel replacements into upstream modules."""

    reapply_all()


@register_triton_op("sglang.srt.mem_cache.common", "write_req_to_token_pool_triton")
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


@register_triton_op("sglang.srt.mem_cache.common", "get_last_loc_kernel")
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
    "sglang.srt.layers.attention.utils",
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
    "sglang.srt.layers.attention.utils",
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
    "sglang.srt.model_executor.forward_batch_deepseek_mha_mixin",
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
    "sglang.srt.layers.dp_attention",
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
    "sglang.srt.layers.attention.dsv4.metadata_kernel",
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

    # --- c4 ---
    c4_should = (seq_lens_i64 % 4) == 0
    c4_out_loc = torch.where(
        c4_should, raw_out_loc_i64 // 4, torch.zeros_like(raw_out_loc_i64)
    ).to(torch.int64)
    c4_positions = (positions_i64 & (~3)).to(torch.int32)  # noqa: F841 (kernel discards)
    c4_seq_lens_raw = (seq_lens_i64 // 4).to(torch.int32)
    c4_seq_lens_clamp1 = torch.clamp(c4_seq_lens_raw, min=1).to(torch.int32)

    # --- c128 ---
    c128_should = (seq_lens_i64 % 128) == 0
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


@register_jit_op("sglang.srt.model_executor.forward_batch_info", "clamp_position")
def clamp_position(seq_lens: torch.Tensor) -> torch.Tensor:
    """Compute non-negative last-token positions from sequence lengths."""

    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)


@register_jit_op("sglang.jit_kernel.hadamard", "hadamard_transform")
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


@register_jit_op("sglang.jit_kernel.activation", "silu_and_mul")
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


@register_jit_op("sglang.jit_kernel.grouped_topk", "grouped_topk")
def grouped_topk(
    scores: torch.Tensor,
    bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    scaling_factor: float,
):
    """Kunlun fused sigmoid+grouped-topk+norm (replaces 0.5.14's nvcc-JIT kernel).

    0.5.14's ``select_experts`` routes the CUDA single-group path through
    ``sglang.jit_kernel.grouped_topk.grouped_topk`` which nvcc-JIT compiles
    ``moe/grouped_topk.cuh`` with ``-std=c++20`` (rejected by Kunlun nvcc 11.7).
    Replace it with ``kunlun_ops.moe_sigmoid_group_topk_norm``, returning the
    same ``(topk_values, topk_indices)`` tuple. The block-statistic tensor it
    also computes is discarded here and regenerated in the MoE method.
    """
    import kunlun_ops

    m_, n_ = scores.shape
    block_statistic = torch.empty(12, n_, dtype=torch.int32, device=scores.device)
    topk_values = torch.empty(m_, topk, dtype=torch.float32, device=scores.device)
    topk_indices = torch.empty(m_, topk, dtype=torch.int32, device=scores.device)
    if m_ == 0:
        return topk_values, topk_indices
    kunlun_ops.moe_sigmoid_group_topk_norm(
        x=scores,
        topk_index=topk_indices,
        norm_score=topk_values,
        block_statistic=block_statistic,
        bias=bias.float(),
        scale=scaling_factor,
        n_group=num_expert_group,
        topk_group=topk_group,
    )
    return topk_values, topk_indices




@register_triton_op("sglang.srt.mem_cache.utils", "set_mla_kv_buffer_kernel")
def set_mla_kv_buffer_kernel(
    kv_buffer: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    loc: torch.Tensor,
    buffer_stride: int,
    nope_stride: int,
    rope_stride: int,
    nope_dim: int,
    rope_dim: int,
    *,
    BLOCK: int,
) -> None:
    """Store MLA KV tensors into the paged KV buffer."""

    from kunlun_ops import set_mla_kv_buffer_triton as impl

    impl(
        kv_buffer,
        loc,
        cache_k_nope.contiguous().view(cache_k_nope.shape[0], cache_k_nope.shape[-1]),
        cache_k_rope.contiguous().view(cache_k_rope.shape[0], cache_k_rope.shape[-1]),
    )


@register_triton_op("sglang.srt.mem_cache.utils", "get_mla_kv_buffer_kernel")
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


@register_triton_op("sglang.srt.mem_cache.triton_ops.allocator", "alloc_extend_kernel")
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


# ---------------------------------------------------------------------------
# DSV4 sparse-prefill index kernels (pure-torch reimplementations)
#
# The internal fork moved these two kernels into
# ``sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels``; the Triton
# bodies cannot compile on XPU, so we supply per-request torch equivalents.
# Registered on both the relocated path and the v0.5.x in-module path.
# ---------------------------------------------------------------------------


def _build_swa_token_ids_torch(
    out,
    swa_first_pos,
    swa_gather_lens,
    swa_offsets,
    req_pool_indices,
    req_to_token,
    req_to_token_stride,
    full_to_swa,
):
    """Map each request's SWA window to physical SWA-cache token ids."""
    num_reqs = swa_first_pos.shape[0]
    first = swa_first_pos.to(torch.int64).cpu()
    glen = swa_gather_lens.to(torch.int64).cpu()
    offs = swa_offsets.to(torch.int64).cpu()
    rpi = req_pool_indices.to(torch.int64).cpu()
    device = req_to_token.device
    for b in range(num_reqs):
        n = int(glen[b])
        if n <= 0:
            continue
        start = int(first[b])
        o = int(offs[b])
        pos = torch.arange(start, start + n, device=device)
        full_ids = req_to_token[int(rpi[b]), pos].to(torch.int64)
        out[o : o + n] = full_to_swa[full_ids].to(out.dtype)


def _combine_topk_swa_indices_torch(
    combined_indices,
    combined_indices_stride,
    combined_lens,
    topk_indices,
    topk_indices_stride,
    query_start_loc,
    seq_lens,
    gather_lens,
    compressed_base,
    swa_base,
    top_k,
    COMPRESS_RATIO,
    WINDOW_SIZE,
    PADDED_TOP_K,
):
    """Concatenate the compressed-region topk ids and the SWA ids per token."""
    num_reqs = seq_lens.shape[0]
    qsl = query_start_loc.to(torch.int64).cpu()
    base = int(qsl[0])
    device = combined_indices.device
    for b in range(num_reqs):
        q_start = int(qsl[b]) - base
        q_end = int(qsl[b + 1]) - base
        seq_len = int(seq_lens[b])
        gather_len = int(gather_lens[b])
        cbase = int(compressed_base[b])
        sbase = int(swa_base[b])
        start_pos = seq_len - (q_end - q_start)
        gather_start = seq_len - gather_len
        for token_idx in range(q_start, q_end):
            pos = start_pos + (token_idx - q_start)
            topk_len = min((pos + 1) // COMPRESS_RATIO, top_k)
            swa_len = min(pos + 1, WINDOW_SIZE)
            row = combined_indices[token_idx]
            if topk_len > 0:
                row[:topk_len] = topk_indices[token_idx, :topk_len] + cbase
            if swa_len > 0:
                ar = torch.arange(swa_len, device=device, dtype=torch.int32)
                row[topk_len : topk_len + swa_len] = (
                    sbase + ar + (pos - swa_len + 1 - gather_start)
                )
            combined_lens[token_idx] = topk_len + swa_len


@register_triton_op(
    "sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels",
    "_build_swa_token_ids_kernel",
)
@register_triton_op(
    "sglang.srt.layers.attention.dsv4.sparse_prefill_utils",
    "_build_swa_token_ids_kernel",
)
def _build_swa_token_ids_kernel(
    out_ptr,
    swa_first_pos_ptr,
    swa_gather_lens_ptr,
    swa_offsets_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    req_to_token_stride,
    full_to_swa_ptr,
) -> None:
    """XPU replacement for the DSV4 SWA token-id Triton kernel."""
    _build_swa_token_ids_torch(
        out_ptr,
        swa_first_pos_ptr,
        swa_gather_lens_ptr,
        swa_offsets_ptr,
        req_pool_indices_ptr,
        req_to_token_ptr,
        req_to_token_stride,
        full_to_swa_ptr,
    )


@register_triton_op(
    "sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels",
    "_combine_topk_swa_indices_kernel",
)
@register_triton_op(
    "sglang.srt.layers.attention.dsv4.sparse_prefill_utils",
    "_combine_topk_swa_indices_kernel",
)
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    compressed_base_ptr,
    swa_base_ptr,
    top_k,
    COMPRESS_RATIO=None,
    WINDOW_SIZE=None,
    PADDED_TOP_K=None,
) -> None:
    """XPU replacement for the DSV4 topk+SWA combine Triton kernel."""
    _combine_topk_swa_indices_torch(
        combined_indices_ptr,
        combined_indices_stride,
        combined_lens_ptr,
        topk_indices_ptr,
        topk_indices_stride,
        query_start_loc_ptr,
        seq_lens_ptr,
        gather_lens_ptr,
        compressed_base_ptr,
        swa_base_ptr,
        top_k,
        COMPRESS_RATIO,
        WINDOW_SIZE,
        PADDED_TOP_K,
    )


@register_triton_op("sglang.srt.mem_cache.triton_ops.allocator", "alloc_decode_kernel")
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


@register_jit_op("sglang.jit_kernel.kvcache", "store_cache")
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


@register_jit_op("sglang.jit_kernel.fused_metadata_copy", "fused_metadata_copy_cuda")
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


@register_jit_op("sglang.jit_kernel.fused_metadata_copy", "fused_metadata_copy_multi_cuda")
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
    "sglang.srt.speculative.multi_layer_eagle_utils",
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


@register_triton_op(
    "sglang.srt.speculative.triton_ops.multi_layer_eagle",
    "assign_hidden_states_pool_triton",
    metadata={"call_style": "direct"},
)
@register_jit_op(
    "sglang.srt.speculative.multi_layer_eagle_utils",
    "assign_hidden_states_pool_triton",
)
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


@register_triton_op(
    "sglang.srt.speculative.triton_ops.multi_layer_eagle",
    "rotate_input_ids_triton",
    metadata={"call_style": "direct"},
)
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


@register_triton_op(
    "sglang.srt.speculative.multi_layer_eagle_utils",
    "assign_new_state_kernel",
)
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


@register_triton_op(
    "sglang.srt.speculative.triton_ops.multi_layer_eagle",
    "assign_new_state_triton",
    metadata={"call_style": "direct"},
)
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
    "sglang.srt.speculative.triton_ops.cache_locs",
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
    positions=None,
    mrope=None,
    BS: int = 0,
    WRITE_POSITIONS: bool = False,
    WRITE_MROPE: bool = False,
) -> None:
    """Assign draft cache locations for the page_size==1 / topk==1 contiguous path.

    0.5.14 renamed the upstream Triton kernel to
    ``assign_draft_cache_locs_contiguous`` (in
    ``sglang.srt.speculative.triton_ops.cache_locs``), called from
    ``base_spec_worker.prepare_for_draft``. The Kunlun replacement copies
    ``topk * speculative_num_steps`` slots per request via xspeedgate_ops.
    """

    # Mirror main's Triton ``assign_draft_cache_locs_contiguous`` exactly:
    # copy ``topk * speculative_num_steps`` cache slots per request from
    # ``req_to_token`` starting at ``seq_lens[req]``.
    bs = seq_lens.shape[0]
    copy_len = int(topk) * int(speculative_num_steps)
    if bs and copy_len:
        rows = req_pool_indices.to(torch.int64)
        starts = seq_lens.to(torch.int64)
        idx = starts[:, None] + torch.arange(
            copy_len, device=seq_lens.device, dtype=torch.int64
        )[None, :]
        vals = req_to_token[rows[:, None], idx]
        n = bs * copy_len
        out_cache_loc.view(-1)[:n] = vals.reshape(-1).to(out_cache_loc.dtype)
    if WRITE_POSITIONS and positions is not None:
        bs = seq_lens.shape[0]
        n = bs * topk
        vals = seq_lens.to(positions.dtype).repeat_interleave(topk)
        positions.view(-1)[:n] = vals
        if WRITE_MROPE and mrope is not None:
            mrope.view(3, -1)[:, :n] = vals.to(mrope.dtype)


# [Dropped for 0.5.14] ``fill_new_verified_id`` no longer exists upstream; the
# verified-id flow moved to ``FutureMap._resolve_spec_extras`` (pure torch) in
# ``sglang.srt.managers.overlap_utils`` and is not a Triton kernel. No XPU
# replacement needed.


@register_triton_op("sglang.srt.speculative.triton_ops.eagle", "fill_bonus_tokens")
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


@register_triton_op("sglang.srt.speculative.triton_ops.eagle", "fill_accept_out_cache_loc")
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
    "sglang.srt.speculative.triton_ops.cache_locs",
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


@register_triton_op("sglang.srt.speculative.triton_ops.cache_locs", "assign_extend_cache_locs")
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


@register_triton_op("sglang.srt.speculative.triton_ops.cache_locs", "assign_req_to_token_pool")
@register_triton_op("sglang.srt.speculative.spec_utils", "assign_req_to_token_pool")
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


@register_triton_op("sglang.srt.speculative.spec_utils", "align_evict_mask_to_page_size")
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


@register_jit_op("sglang.srt.speculative.triton_ops.gather_spec_extras", "gather_spec_extras")
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


@register_jit_op("sglang.srt.speculative.reject_sampling", "chain_speculative_sampling_triton")
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

    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
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
    "sglang.srt.model_executor.triton_ops.position",
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


@register_jit_op("sglang.jit_kernel.rope", "apply_rope_with_cos_sin_cache_inplace")
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
    "sglang.srt.layers.attention.dsa.index_buf_accessor",
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
    "sglang.srt.layers.attention.dsa.index_buf_accessor",
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
    "sglang.srt.layers.attention.dsa.index_buf_accessor",
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
    "sglang.srt.layers.attention.dsa.triton_kernel",
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
    "sglang.srt.layers.quantization.int8_kernel",
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
    "sglang.srt.layers.quantization.int8_kernel",
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
    "sglang.srt.layers.quantization.int8_kernel",
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
    "sglang.srt.layers.quantization.int8_kernel",
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
# DSV4 tvm_ffi JIT kernel replacements (pure PyTorch)
# All shim implementations live in dsv4_torch_shim.py; registered here so
# they are patched into the upstream sglang.jit_kernel.dsv4.* modules at
# kernel_ops.install() time.
# ============================================================================

from sglang_kunlun.eb5_adapt.kernels.dsv4_torch_shim import (  # noqa: E402
    _fused_q_norm_rope_torch,
    _fused_k_norm_rope_flashmla_torch,
    _fused_k_norm_rope_flashmla_kunlun,
    _fused_rope_inplace_torch,
    _fused_rope_inplace_kunlun,
    _fused_norm_rope_inplace_torch,
    _fused_norm_rope_inplace_kunlun,
    _fused_q_indexer_rope_hadamard_quant_torch,
    _fused_q_indexer_rope_hadamard_quant_kunlun,
    _compress_forward_torch,
    _compress_norm_rope_store_torch,
    _compress_norm_rope_store_kunlun,
    _create_paged_compress_data_kunlun,
)


def clear_unaccepted_c128_draft_states_torch(
    state: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    ring_size: int,
    num_draft_tokens: int,
) -> None:
    """XPU torch replacement for ``clear_unaccepted_c128_draft_states``.

    For every (request, draft offset) that was NOT accepted, zero the first
    half of its ring slot and set the second half to ``-inf``.  Fully
    vectorized and device-side, so it stays CUDA-graph replayable.
    """
    half = state.shape[-1] // 2
    if half == 0 or req_pool_indices.numel() == 0:
        return
    device = state.device
    flat = state.reshape(-1)
    bs = req_pool_indices.numel()
    nd = int(num_draft_tokens)

    bid = torch.arange(bs, device=device)
    d = torch.arange(nd, device=device)
    acc = accept_lens.to(torch.int64)
    sel = acc[:, None] <= d[None, :]  # [bs, nd]: not accepted
    sel_b = bid[:, None].expand(bs, nd)[sel]
    sel_d = d[None, :].expand(bs, nd)[sel]
    if sel_b.numel() == 0:
        return

    seq = seq_lens.to(torch.int64)[sel_b]
    rpi = req_pool_indices.to(torch.int64)[sel_b]
    slot = (seq + sel_d) % ring_size
    row_base = (rpi * ring_size + slot) * (half * 2)

    ar = torch.arange(half, device=device)
    flat[(row_base[:, None] + ar[None, :]).reshape(-1)] = 0.0
    flat[(row_base[:, None] + half + ar[None, :]).reshape(-1)] = float("-inf")


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.c128_cleanup",
    "clear_unaccepted_c128_draft_states",
)
def clear_unaccepted_c128_draft_states(
    state: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    ring_size: int,
    num_draft_tokens: int,
) -> None:
    """XPU replacement for the c128 draft-state cleanup Triton kernel."""
    clear_unaccepted_c128_draft_states_torch(
        state,
        req_pool_indices,
        seq_lens,
        accept_lens,
        ring_size=ring_size,
        num_draft_tokens=num_draft_tokens,
    )


def assign_extend_cache_locs_uniform_func_torch(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    """XPU torch replacement for ``assign_extend_cache_locs_uniform_func``.

    Copies ``draft_token_num`` cache slots per request out of ``req_to_token``
    starting at ``start_offset``.  Device-side gather, so it stays
    CUDA-graph replayable.
    """
    out = torch.empty(
        (batch_size * draft_token_num,), dtype=torch.int64, device=device
    )
    if batch_size == 0 or draft_token_num == 0:
        return out
    rows = req_pool_indices.to(torch.int64)
    start = start_offset.to(torch.int64)
    offs = torch.arange(draft_token_num, device=req_to_token.device)
    idx = start[:, None] + offs[None, :]
    gathered = req_to_token[rows[:, None], idx]
    out.copy_(gathered.reshape(-1).to(torch.int64))
    return out


@register_jit_op(
    "sglang.kernels.ops.speculative.cache_locs",
    "assign_extend_cache_locs_uniform_func",
)
def assign_extend_cache_locs_uniform_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    """XPU replacement for the uniform EAGLE verify cache-loc assign."""
    return assign_extend_cache_locs_uniform_func_torch(
        req_pool_indices,
        req_to_token,
        start_offset,
        batch_size,
        draft_token_num,
        device,
    )


def draft_topk1_postprocess_torch(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens=None,
    draft_token_column: int = 0,
):
    """XPU torch replacement for ``draft_topk1_postprocess``.

    Argmax draft logits (topk=1 is greedy), scatter the token into the
    caller-owned ``draft_tokens`` buffer and advance ``positions`` in place.
    All steps are device ops, so it stays CUDA-graph replayable.
    """
    bs, _vocab = next_token_logits.shape
    device = next_token_logits.device
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=device)
    topk_index = torch.empty((bs, 1), dtype=torch.int64, device=device)
    if bs == 0:
        return topk_p, topk_index

    logits = next_token_logits.to(torch.float32)
    # Keep NaNs on valid lanes from winning the reduction (matches the kernel).
    logits = torch.where(logits == logits, logits, torch.full_like(logits, -1e30))
    idx = logits.argmax(dim=1)
    topk_index[:, 0] = idx
    topk_p[:, 0] = 1.0
    if draft_tokens is not None:
        draft_tokens[:, draft_token_column] = idx
    positions.add_(1)
    return topk_p, topk_index


@register_jit_op(
    "sglang.kernels.ops.speculative.topk1", "draft_topk1_postprocess"
)
def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens=None,
    draft_token_column: int = 0,
):
    """XPU replacement for the EAGLE draft topk=1 postprocess (Triton)."""
    return draft_topk1_postprocess_torch(
        next_token_logits, positions, draft_tokens, draft_token_column
    )


def gather_bf16_k_cache_paged_torch(
    k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """XPU torch replacement for ``gather_bf16_k_cache_paged``.

    Pure per-token copy out of the bf16 paged K cache (448 nope + 64 rope).
    """
    dim = 448 + 64
    k_bf16 = k_cache.view(torch.bfloat16)
    bf16_per_page = k_bf16.shape[-1]
    num_tokens = page_table_1_flattened.shape[0]

    if out is None:
        out = torch.empty(
            (num_tokens, 1, dim), dtype=torch.bfloat16, device=k_cache.device
        )

    if num_tokens == 0:
        return out

    flat = k_bf16.reshape(-1)
    loc = page_table_1_flattened.to(torch.int64)
    page = torch.div(loc, page_size, rounding_mode="floor")
    in_page = loc % page_size
    base = page * bf16_per_page + in_page * dim
    idx = base[:, None] + torch.arange(dim, device=flat.device)[None, :]
    out[:, 0, :] = flat[idx]
    return out


@register_jit_op(
    "sglang.srt.internal.layers.attention.dsv4.gather_bf16_k_cache",
    "gather_bf16_k_cache_paged",
)
def gather_bf16_k_cache_paged(
    k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """XPU replacement for the bf16 sparse-prefill K-cache gather Triton kernel."""
    return gather_bf16_k_cache_paged_torch(
        k_cache, page_table_1_flattened, page_size, out
    )


def dequantize_k_cache_paged_torch(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """XPU torch replacement for ``dequantize_k_cache_paged``.

    Decodes the DeepSeek-V4 paged KV layout (448 fp8 nope + 64 bf16 rope +
    7 UE8M0 scales padded to 8 bytes per token) with vectorized torch indexing.
    The fp8->fp32 cast goes through a 256-entry CPU lookup table because XPU has
    no device-side float8 conversion kernel.
    """
    dim_nope, dim_rope = 448, 64
    tile_size, num_scale_tiles = 64, 7
    nope_rope_bytes, padded_scale_per_token = 576, 8

    u8 = quant_k_cache.view(torch.uint8)
    bytes_per_page = u8.shape[-1]
    s_offset_bytes = page_size * nope_rope_bytes
    flat_u8 = u8.reshape(-1)
    flat_bf16 = u8.view(torch.bfloat16).reshape(-1)

    lut = (
        torch.arange(256, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
        .to(u8.device)
    )

    loc = page_table_1_flattened.to(torch.int64)
    num_tokens = loc.shape[0]
    page_idx = torch.div(loc, page_size, rounding_mode="floor")
    in_page = loc % page_size
    page_byte_base = page_idx * bytes_per_page
    token_data_base = page_byte_base + in_page * nope_rope_bytes
    token_scale_base = (
        page_byte_base + s_offset_bytes + in_page * padded_scale_per_token
    )

    device = quant_k_cache.device
    if out is None:
        out = torch.empty(
            (num_tokens, 1, dim_nope + dim_rope),
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        assert out.shape == (num_tokens, 1, dim_nope + dim_rope)
        assert out.dtype == torch.bfloat16

    if num_tokens == 0:
        return out

    ar_nope = torch.arange(dim_nope, device=device)
    nope_byte = token_data_base[:, None] + ar_nope[None, :]
    nope_f = lut[flat_u8[nope_byte].long()]

    ar_scale = torch.arange(num_scale_tiles, device=device)
    scale_byte = token_scale_base[:, None] + ar_scale[None, :]
    scale_u8 = flat_u8[scale_byte].to(torch.int32)
    scale_pow2 = torch.exp2((scale_u8 - 127).to(torch.float32))
    nope = nope_f * scale_pow2.repeat_interleave(tile_size, dim=1)

    ar_rope = torch.arange(dim_rope, device=device)
    rope_idx = ((token_data_base + dim_nope) // 2)[:, None] + ar_rope[None, :]
    rope = flat_bf16[rope_idx]

    out[:, 0, :dim_nope] = nope.to(torch.bfloat16)
    out[:, 0, dim_nope:] = rope
    return out


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.dequant_k_cache",
    "dequantize_k_cache_paged_dsv4",
)
@register_jit_op(
    "sglang.kernels.ops.attention.dsa.dequant_k_cache",
    "dequantize_k_cache_paged_dsa",
)
@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.dequant_k_cache",
    "dequantize_k_cache_paged",
)
@register_jit_op(
    "sglang.srt.layers.attention.dsv4.dequant_k_cache",
    "dequantize_k_cache_paged",
)
def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """XPU replacement for the DSV4 paged-KV dequantization Triton kernel."""
    return dequantize_k_cache_paged_torch(
        quant_k_cache, page_table_1_flattened, page_size, out
    )


@register_jit_op(
    "sglang.srt.layers.attention.dsv4.indexer", "fp8_paged_mqa_logits_torch"
)
def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata=None,
    max_seq_len=None,
    clean_logits: bool = True,
    **kwargs,
) -> torch.Tensor:
    """XPU replacement for the DSV4 indexer-logits torch fallback.

    The internal ``forward_c4_indexer`` resolves this symbol from its
    env-selected ladder at call time; ``SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=True``
    selects it on our env.  Forward to the Kunlun-native logits kernel via the
    bridge in ``deepseek_v4_backend`` (which holds the active batch/backend).
    """
    from sglang_kunlun.eb5_adapt.hooks.layers.attention.deepseek_v4_backend import (
        kunlun_fp8_paged_mqa_logits_torch,
    )

    return kunlun_fp8_paged_mqa_logits_torch(
        q_fp8,
        kvcache_fp8,
        weight,
        seq_lens,
        page_table,
        deep_gemm_metadata,
        max_seq_len,
        clean_logits,
        **kwargs,
    )


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.norm_rope", "fused_q_norm_rope"
)
@register_jit_op("sglang.jit_kernel.dsv4.elementwise", "fused_q_norm_rope")
def fused_q_norm_rope(
    q_input: torch.Tensor,
    q_output: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Pure-torch replacement: rmsnorm-self (no weight) + RoPE on Q."""
    _fused_q_norm_rope_torch(q_input, q_output, eps, freqs_cis, positions)


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.norm_rope", "fused_k_norm_rope_flashmla"
)
@register_jit_op("sglang.jit_kernel.dsv4.elementwise", "fused_k_norm_rope_flashmla")
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
    q=None,
    **kwargs,
) -> None:
    """Kunlun-native fused K norm + RoPE + paged cache store.

    Step 1: ``kunlun_ops.rmsnorm`` (with weight)
    Step 2: ``xspeedgate_ops.flashinfer_rotary_embedding`` on rope part
    Step 3: ``xspeedgate_ops.set_k_and_s_v4`` (bf16) or torch FP8 quant+write

    Dispatch on the cache dtype: the kunlun-native path only implements the
    bf16/fp16 layout; the fp8 (uint8-packed) layout is handled by the torch
    shim, which writes the 576 B/token page (448 fp8 nope + 64 bf16 rope +
    7xUE8M0 scale).  Without this dispatch the uint8 path raises.
    """
    if kvcache.dtype in (torch.bfloat16, torch.float16):
        _fused_k_norm_rope_flashmla_kunlun(
            kv, kv_weight, eps, freqs_cis, positions, out_loc, kvcache, page_size
        )
    else:
        _fused_k_norm_rope_flashmla_torch(
            kv, kv_weight, eps, freqs_cis, positions, out_loc, kvcache, page_size
        )


@register_jit_op(
    "sglang.jit_kernel.internal.dsv4.elementwise",
    "fused_k_norm_rope_flashmla_bf16",
)
def fused_k_norm_rope_flashmla_bf16(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
) -> None:
    """Internal-tree bf16 variant of ``fused_k_norm_rope_flashmla``.

    The internal fork routes the bf16/fp16 KV-cache case to a separately named
    function living in ``sglang.jit_kernel.internal.dsv4.elementwise`` (which is
    *not* covered by the ``sglang.jit_kernel.dsv4.*`` alias), so it needs its
    own registration.  Same math and same bf16 page layout as the kunlun-native
    branch above.
    """
    _fused_k_norm_rope_flashmla_kunlun(
        kv, kv_weight, eps, freqs_cis, positions, out_loc, kvcache, page_size
    )


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.norm_rope", "fused_rope_inplace"
)
@register_jit_op("sglang.jit_kernel.dsv4.elementwise", "fused_rope_inplace")
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


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.norm_rope", "fused_norm_rope_inplace"
)
@register_jit_op("sglang.jit_kernel.dsv4.compress_old", "fused_norm_rope_inplace")
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


@register_jit_op(
    "sglang.jit_kernel.internal.dsv4.elementwise",
    "fused_q_indexer_rope_hadamard_quant",
)
@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.norm_rope", "fused_q_indexer_rope_hadamard_quant"
)
@register_jit_op("sglang.jit_kernel.dsv4.elementwise", "fused_q_indexer_rope_hadamard_quant")
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


@register_jit_op("sglang.jit_kernel.dsv4.compress", "compress_forward")
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


@register_jit_op(
    "sglang.jit_kernel.internal.dsv4.compress", "compress_norm_rope_store_bf16"
)
def compress_norm_rope_store_bf16(
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
    bf16_store: bool = True,
    **kwargs,
) -> None:
    """Internal-fork bf16-store variant of ``compress_norm_rope_store``.

    Selected when ``use_bf16_store`` is true (bf16 KV cache on the latent
    compressor).  Same Kunlun norm+rope+store, writing the bf16 page layout.
    """
    _compress_norm_rope_store_kunlun(
        kv, plan,
        norm_weight=norm_weight, norm_eps=norm_eps, freq_cis=freq_cis,
        out_loc=out_loc, kvcache=kvcache, page_size=page_size,
        use_fp4=use_fp4, bf16_store=True,
    )


@register_jit_op("sglang.jit_kernel.dsv4.compress", "compress_norm_rope_store")
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
    **kwargs,
) -> None:
    """Kunlun-native: fused norm+rope via dpsk_v4_norm_rope_gptj + store.

    The internal fork's signature carries HIP/FP4-only extras
    (``kvcache_scale`` / ``rope_cache`` / ``fp4_k_write_metadata`` /
    ``fp8_2buff`` / ``kvcache_rope``); they are unused on XPU, so accept and
    drop them.
    """
    _compress_norm_rope_store_kunlun(
        kv, plan,
        norm_weight=norm_weight, norm_eps=norm_eps, freq_cis=freq_cis,
        out_loc=out_loc, kvcache=kvcache, page_size=page_size,
        use_fp4=use_fp4, bf16_store=bf16_store,
    )


@register_jit_op("sglang.jit_kernel.dsv4.attn", "triton_create_paged_compress_data")
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
    block: int = 128,
):
    """Kunlun-native replacement for the v1 legacy compressor's Triton
    write_loc/extra_data builder. Routes to
    ``xspeedgate_ops.create_paged_compress_data`` (same op aiak_sglang uses)
    instead of the unsupported Triton kernel.
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

@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.topk", "topk_transform_paged"
)
@register_jit_op("sglang.jit_kernel.dsv4.topk", "topk_transform_512")
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
            topk_transform_512_pytorch_vectorized,
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


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.topk", "topk_transform_paged_v2"
)
@register_jit_op("sglang.jit_kernel.dsv4.topk", "topk_transform_512_v2")
def topk_transform_512_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
    out_raw_indices=None,
    row_starts=None,
    page_rows=None,
    **kwargs,
) -> None:
    """Kunlun-native topk (v2 signature drops raw_indices, keeps metadata for
    upstream compat; we ignore metadata since xspeedgate handles planning
    internally).  The internal fork's ``topk_transform_paged_v2`` adds optional
    ``out_raw_indices`` / ``row_starts`` / ``page_rows``; unused on XPU."""
    import xspeedgate_ops  # noqa: F401

    torch.ops.xspeedgate_ops.topk_transform(
        scores, seq_lens, page_tables, out_page_indices,
        512, None, page_size, True,
    )


# ---------------------------------------------------------------------------
# Community main added a fused "small copy" Triton kernel used by the CUDA
# graph buffer registry.  Triton cannot compile on Kunlun, and the operation
# is a plain elementwise copy, so provide a torch implementation.
# ---------------------------------------------------------------------------
@register_jit_op("sglang.kernels.ops.memory.small_copy", "try_small_copy")
@register_jit_op("sglang.srt.mem_cache.small_copy", "try_small_copy")
def try_small_copy_kunlun(dsts, srcs):
    import torch as _torch

    if len(dsts) < 2 or len(dsts) != len(srcs):
        return False
    try:
        for dst, src in zip(dsts, srcs):
            if dst.shape != src.shape or dst.device != src.device:
                return False
            dst.copy_(src)
        return True
    except Exception:
        return False


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.attn_metadata_kernels",
    "expand_prefill_causally_triton",
)
def expand_prefill_causally_kunlun(
    *,
    req_pool_indices,
    seq_lens,
    extend_seq_lens,
    num_tokens,
    padded_num_tokens=None,
):
    """Device-side reference for the CUDA-only ``expand_prefill_causally``."""
    from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
        ExpandPrefillCausallyResult,
    )

    device = req_pool_indices.device
    bs = req_pool_indices.shape[0]
    total_tokens = (
        padded_num_tokens
        if padded_num_tokens is not None and padded_num_tokens > num_tokens
        else num_tokens
    )
    extend = extend_seq_lens.to(torch.int64)
    starts = torch.cumsum(extend, dim=0) - extend
    offs = torch.arange(total_tokens, device=device)
    is_real = offs < num_tokens
    t = torch.where(is_real, offs, torch.zeros_like(offs))
    r = torch.searchsorted(starts, t, right=True) - 1
    r = torch.where(is_real, r, torch.full_like(r, max(bs - 1, 0))).clamp_(0, max(bs - 1, 0))
    seq_len = seq_lens.to(torch.int64)[r]
    ext = extend[r]
    causal = (seq_len - ext + 1) + (t - starts[r])
    causal = torch.where(is_real, causal, torch.ones_like(causal))
    seq_lens_casual = causal.to(torch.int32)
    req_pool_indices_repeated = req_pool_indices[r]
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.attn_metadata_kernels",
    "build_page_table_positions_triton",
)
def build_page_table_positions_kunlun(
    *, req_to_token, req_pool_indices_repeated, seq_lens_casual,
    max_seq_len, page_size, swa_window,
):
    from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
        build_page_table_positions,
    )

    return build_page_table_positions(
        req_to_token=req_to_token,
        req_pool_indices_repeated=req_pool_indices_repeated,
        seq_lens_casual=seq_lens_casual,
        max_seq_len=max_seq_len,
        page_size=page_size,
        swa_window=swa_window,
    )


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.attn_metadata_kernels",
    "build_causal_swa_page_indices_triton",
)
def build_causal_swa_page_indices_kunlun(
    *, req_to_token, full_to_swa_mapping, req_pool_indices_repeated,
    seq_lens_casual, swa_window, page_index_aligned_size, swa_replay_start=None,
):
    from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
        build_causal_swa_page_indices,
    )

    return build_causal_swa_page_indices(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa_mapping,
        req_pool_indices_repeated=req_pool_indices_repeated,
        seq_lens_casual=seq_lens_casual,
        swa_window=swa_window,
        page_index_aligned_size=page_index_aligned_size,
        swa_replay_start=swa_replay_start,
    )


@register_jit_op("sglang.kernels.ops.layernorm.mhc", "hc_broadcast")
def hc_broadcast_kunlun(x, hc_mult):
    """Triton-free ``[T, H] -> [T, hc_mult, H]`` broadcast."""
    if x.numel() == 0:
        return x.new_empty(x.shape[0], hc_mult, x.shape[1])
    return x.unsqueeze(1).repeat(1, hc_mult, 1)


@register_jit_op(
    "sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels",
    "combine_topk_swa_indices",
)
def combine_topk_swa_indices_kunlun(
    topk_indices,
    query_start_loc,
    query_pos,
    seq_lens,
    gather_lens,
    compressed_base,
    swa_base,
    window_size,
    compress_ratio,
    topk,
    out_indices=None,
    out_lens=None,
    swa_indices=None,
    swa_lengths=None,
):
    """Torch implementation of the sparse-prefill top-k + SWA merge."""
    from sglang.kernels.ops.attention.dsv4.sparse_prefill_kernels import (
        combined_topk_width,
    )

    device = topk_indices.device
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    width = combined_topk_width(topk, window_size)
    combined_indices = (
        out_indices
        if out_indices is not None
        else torch.empty((num_tokens, width), dtype=torch.int32, device=device)
    )
    combined_lens = (
        out_lens
        if out_lens is not None
        else torch.empty(num_tokens, dtype=torch.int32, device=device)
    )
    if num_tokens == 0:
        return combined_indices, combined_lens

    tok = torch.arange(num_tokens, device=device)
    base = query_start_loc[0]
    qsl = (query_start_loc - base).to(torch.int64)
    owned = tok < qsl[num_reqs]

    b = (torch.searchsorted(qsl, tok.to(torch.int64), right=True) - 1).clamp(
        0, max(num_reqs - 1, 0)
    )
    gather = lambda t, fill: torch.where(owned, t.to(torch.int64)[b], torch.full_like(b, fill))
    # ``query_pos`` is per token, while the rest are per request.
    pos = torch.where(owned, query_pos.to(torch.int64), torch.zeros_like(b))
    seq_len = gather(seq_lens, 0)
    gather_len = gather(gather_lens, 0)
    comp_base = gather(compressed_base, 0)
    sw_base = gather(swa_base, 0)
    gather_start = seq_len - gather_len

    topk_len = torch.where(
        owned,
        torch.clamp((pos + 1) // compress_ratio, max=topk),
        torch.zeros_like(pos),
    )
    if swa_indices is not None:
        swa_len = torch.where(owned, swa_lengths.to(torch.int64), torch.zeros_like(pos))
    else:
        swa_len = torch.where(
            owned, torch.clamp(pos + 1, max=window_size), torch.zeros_like(pos)
        )

    off = torch.arange(width, device=device)[None, :]
    in_topk = off < topk_len[:, None]
    k = topk_indices.shape[1]
    safe_k_off = torch.clamp(off, max=max(k - 1, 0))
    topk_vals = torch.gather(
        topk_indices.to(torch.int64),
        1,
        safe_k_off.expand(num_tokens, width),
    )
    topk_vals = torch.where(in_topk, topk_vals, torch.full_like(topk_vals, -1))

    in_swa = off < (topk_len + swa_len)[:, None]
    swa_off = off - topk_len[:, None]
    safe_swa_off = torch.clamp(swa_off, min=0)
    if swa_indices is not None:
        w = swa_indices.shape[1]
        safe_swa_off = torch.clamp(safe_swa_off, max=max(w - 1, 0))
        swa_vals = torch.gather(
            swa_indices.to(torch.int64), 1, safe_swa_off
        )
        swa_vals = torch.where(swa_vals >= 0, sw_base[:, None] + swa_vals, -1)
    else:
        swa_start = sw_base + pos - swa_len + 1 - gather_start
        swa_vals = swa_start[:, None] + swa_off

    vals = torch.where(
        in_topk,
        torch.where(topk_vals >= 0, topk_vals + comp_base[:, None], topk_vals.new_full((), -1)),
        torch.where(in_swa, swa_vals, swa_vals.new_full((), -1)),
    )
    combined_indices.copy_(vals.to(torch.int32))
    combined_lens.copy_((topk_len + swa_len).to(torch.int32))
    return combined_indices, combined_lens


@register_jit_op("sglang.kernels.ops.gemm.bf16_fp32", "linear_bf16_fp32")
def linear_bf16_fp32_kunlun(x, y):
    """bf16 x (out, in)^T -> fp32 without the ``out_dtype`` mm overload."""
    return torch.mm(x.to(torch.float32), y.to(torch.float32).t())


@register_jit_op(
    "sglang.kernels.ops.activation.silu_and_mul_clamp", "silu_and_mul_clamp"
)
def silu_and_mul_clamp_kunlun(input, output, swiglu_limit):
    """SwiGLU with the DSV4 branch clamp (gate: min-clamp, up: both sides)."""
    import torch.nn.functional as F

    h = input.shape[-1] // 2
    gate = input[..., :h].clamp(max=float(swiglu_limit))
    up = input[..., h:].clamp(-float(swiglu_limit), float(swiglu_limit))
    output.copy_(F.silu(gate) * up)


@register_jit_op("sglang.kernels.ops.moe.dsv4", "hash_topk")
def hash_topk_kunlun(
    router_logits,
    input_ids,
    tid2eid,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
):
    """Pure-torch DSV4 hash routed gate (sqrt-softplus scoring)."""
    assert scoring_func == "sqrtsoftplus"
    num_tokens = router_logits.size(0)
    num_routed_experts = router_logits.size(1)
    topk_routed = tid2eid.size(1)
    topk_fused = topk_routed + num_fused_shared_experts
    topk_weights = torch.empty(
        (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
    )
    topk_ids = torch.empty(
        (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
    )
    if num_tokens == 0:
        return topk_weights, topk_ids

    rows = torch.arange(num_tokens, device=router_logits.device)
    expert_id = tid2eid[input_ids.to(torch.int64)].to(torch.int64)
    logit = router_logits[rows[:, None], expert_id].to(torch.float32)
    softplus = torch.clamp(logit, min=0) + torch.log1p(torch.exp(-logit.abs()))
    weight = torch.sqrt(softplus)
    routed_sum = weight.sum(dim=1, keepdim=True).clamp_min(1e-20)
    topk_weights[:, :topk_routed] = weight / routed_sum
    topk_ids[:, :topk_routed] = expert_id.to(torch.int32)
    if num_fused_shared_experts:
        topk_weights[:, topk_routed:] = 1.0 / float(routed_scaling_factor)
        topk_ids[:, topk_routed:] = (
            num_routed_experts
            + torch.arange(num_fused_shared_experts, device=router_logits.device)
        ).to(torch.int32)
    return topk_weights, topk_ids


@register_jit_op("sglang.kernels.ops.moe.dsv4", "mask_topk_ids")
def mask_topk_ids_kunlun(topk_ids, num_token_non_padded):
    """Rows past the padded token count are marked invalid (-1)."""
    n = num_token_non_padded.reshape(())
    rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)
    mask = (rows >= n)[:, None]
    topk_ids.copy_(torch.where(mask, torch.full_like(topk_ids, -1), topk_ids))


@register_jit_op("sglang.kernels.ops.memory.allocator", "get_and_clear_swa_pages")
def get_and_clear_swa_pages_kunlun(
    full_page_representatives, mapping, page_size, check_page_mappings=False
):
    """Torch version of the SWA page resolve + clear kernel."""
    num_pages = full_page_representatives.numel()
    device = mapping.device
    if num_pages == 0:
        empty = torch.empty(0, dtype=mapping.dtype, device=device)
        return empty, torch.empty(0, dtype=torch.bool, device=device), (
            torch.empty(0, dtype=torch.bool, device=device)
            if check_page_mappings
            else None
        )
    reps = full_page_representatives.to(torch.int64)
    swa_reps = mapping[reps]
    swa_pages = (swa_reps // page_size).to(mapping.dtype)
    peers_mapped = swa_reps > 0
    starts = reps // page_size * page_size
    offsets = starts[:, None] + torch.arange(page_size, device=device)[None, :]
    page_valid = None
    if check_page_mappings:
        page_mapping = mapping[offsets]
        same_page = (page_mapping // page_size) == (swa_reps // page_size)[:, None]
        page_valid = peers_mapped & (
            ((page_mapping > 0) & ~same_page).sum(dim=1) == 0
        )
    mapping[offsets] = 0
    return swa_pages, peers_mapped, page_valid


@register_jit_op(
    "sglang.kernels.ops.embeddings.vocab_parallel_embedding",
    "vocab_parallel_embedding",
)
def vocab_parallel_embedding_kunlun(
    input_,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    ids = input_.to(torch.int64)
    org = (ids >= org_vocab_start_index) & (ids < org_vocab_end_index)
    added = (ids >= added_vocab_start_index) & (ids < added_vocab_end_index)
    valid = org | added
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    local = torch.where(org, ids - org_vocab_start_index, ids - added_offset)
    local = torch.where(valid, local, torch.zeros_like(local))
    out = weight[local]
    return torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))


def _fill_tree_mask_full(tree_mask, seq_lens, retrive_next_token, retrive_next_sibling, nv, bs):
    """Write the FULL_MASK causal tree mask (prefix + ancestor closure).

    ``kunlun_ops.build_tree_efficient`` fills positions/retrive arrays correctly
    but leaves non-causal draft cells set, which makes every draft token attend
    to future siblings and degenerates generation.
    """
    flat = tree_mask.view(-1)
    off = 0
    nt = retrive_next_token.tolist()
    ns = retrive_next_sibling.tolist()
    seqs = seq_lens.tolist()
    for b in range(bs):
        sl = int(seqs[b])
        cols = sl + nv
        block = flat[off : off + nv * cols].view(nv, cols)
        block[:, :sl] = True
        block[:, sl:] = False
        parent = [-1] * nv
        for i in range(nv):
            c = nt[b][i]
            while c != -1:
                parent[c] = i
                c = ns[b][c]
        for q in range(nv):
            j = q
            while j != -1:
                block[q, sl + j] = True
                j = parent[j]
        off += nv * cols


@register_jit_op(
    "sglang.kernels.ops.speculative.tree", "build_tree_kernel_efficient"
)
def build_tree_kernel_efficient_kunlun(*args, **kwargs):
    from sglang_kunlun.kernels.sgl_kernel_kunlun import speculative as _spec

    ret = _spec.build_tree_kernel_efficient(*args, **kwargs)
    names = (
        "parent_list", "top_scores_index", "seq_lens", "tree_mask", "positions",
        "retrive_index", "retrive_next_token", "retrive_next_sibling",
        "topk", "spec_steps", "num_verify_tokens", "tree_mask_mode",
    )
    vals = dict(zip(names, args))
    vals.update({k: v for k, v in kwargs.items() if k in names})
    # FULL_MASK == 0 in sglang's TreeMaskMode; other modes (CPU/bitpacking) are
    # not used on this backend.
    if int(vals.get("tree_mask_mode", 0)) == 0 and "tree_mask" in vals:
        nv = int(vals["num_verify_tokens"])
        bs = int(vals["seq_lens"].numel())
        _fill_tree_mask_full(
            vals["tree_mask"], vals["seq_lens"],
            vals["retrive_next_token"], vals["retrive_next_sibling"], nv, bs,
        )
    return ret


@register_jit_op("sglang.kernels.ops.speculative.tree", "verify_tree_greedy")
def verify_tree_greedy_kunlun(*args, **kwargs):
    from sglang_kunlun.kernels.sgl_kernel_kunlun import speculative as _spec

    if args and not kwargs:
        names = (
            "predicts",
            "accept_index",
            "accept_token_num",
            "candidates",
            "retrive_index",
            "retrive_next_token",
            "retrive_next_sibling",
            "target_predict",
        )
        kwargs = dict(zip(names, args))
    return _spec.verify_tree_greedy(**kwargs)


@register_jit_op(
    "sglang.kernels.ops.speculative.eagle", "prepare_verify_commit_outputs"
)
def prepare_verify_commit_outputs_kunlun(
    predict,
    accept_index,
    accept_lens,
    seq_lens,
    num_draft_tokens=0,
    mamba_track_interval=0,
):
    """Torch version of the EAGLE verify-commit output preparation."""
    bs = accept_lens.numel()
    dev = seq_lens.device
    new_seq_lens = torch.empty(
        (bs,), device=dev, dtype=torch.promote_types(seq_lens.dtype, accept_lens.dtype)
    )
    bonus = torch.empty((bs,), device=predict.device, dtype=torch.int32)
    last_steps = track_steps = correct = select = ids = None
    if num_draft_tokens:
        last_steps = torch.empty((bs,), device=dev, dtype=torch.int64)
        if mamba_track_interval:
            track_steps = torch.empty_like(last_steps)
        correct = torch.empty_like(accept_lens, memory_format=torch.contiguous_format)
        select = torch.empty_like(last_steps)
        ids = torch.empty((predict.numel(),), device=predict.device, dtype=torch.int64)
    if bs == 0:
        if num_draft_tokens:
            return new_seq_lens, bonus, (last_steps, track_steps), (correct, select, ids)
        return new_seq_lens, bonus

    rows = torch.arange(bs, device=dev)
    accepted = accept_lens[:bs].to(torch.int64)
    length = seq_lens[:bs].to(torch.int64)
    idx = accept_index[rows, accepted - 1].to(torch.int64)
    pred = predict.reshape(-1)
    idx_norm = torch.where(idx < 0, idx + predict.numel(), idx)
    new_seq_lens[:bs] = (length + accepted).to(new_seq_lens.dtype)
    bonus[:bs] = pred[idx_norm].to(torch.int32)

    if num_draft_tokens:
        last_steps[:bs] = idx - rows * num_draft_tokens
        correct[:bs] = accepted.to(correct.dtype) - 1
        select[:bs] = rows * num_draft_tokens + accepted - 1
        # The kernel fills every row of ``ids`` (PREDICT_SIZE == predict.numel()),
        # not just the first ``bs`` — draft extend reads them all.
        _n = predict.numel()
        ids[:_n] = (
            predict.reshape(_n, -1)[:, 0].to(torch.int64)
            if predict.dim() > 1
            else pred[:_n].to(torch.int64)
        )
        if mamba_track_interval:
            post = length + accepted
            crossing = (length // mamba_track_interval) != (
                post // mamba_track_interval
            )
            tp = torch.clamp(
                post // mamba_track_interval * mamba_track_interval - length - 1,
                min=0,
            )
            track = accept_index[rows, tp].to(torch.int64) - rows * num_draft_tokens
            track_steps[:bs] = torch.where(crossing, track, track.new_full((), -1))
        return new_seq_lens, bonus, (last_steps, track_steps), (correct, select, ids)
    return new_seq_lens, bonus


@register_jit_op(
    "sglang.kernels.ops.kvcache.dsv4_c128_cleanup",
    "clear_unaccepted_c128_draft_states",
)
def clear_unaccepted_c128_draft_states_new(*args, **kwargs):
    return clear_unaccepted_c128_draft_states_torch(*args, **kwargs)


@register_jit_op(
    "sglang.kernels.ops.speculative.eagle", "prepare_draft_extend_lengths"
)
def prepare_draft_extend_lengths_kunlun(seq_lens, num_draft_tokens, front_offset):
    bs = seq_lens.numel()
    dev = seq_lens.device
    prefix = torch.clamp(seq_lens.to(torch.int64) - front_offset, min=0).to(
        torch.int32
    )
    extend = torch.full(
        (bs,), int(num_draft_tokens) + int(front_offset), dtype=torch.int32, device=dev
    )
    post = (seq_lens.to(torch.int64) + num_draft_tokens).to(seq_lens.dtype)
    return prefix, extend, post


@register_jit_op(
    "sglang.kernels.ops.speculative.eagle", "prepare_draft_extend_layout"
)
def prepare_draft_extend_layout_kunlun(seq_lens, num_draft_tokens, with_mrope):
    bs = seq_lens.numel()
    dev = seq_lens.device
    W = int(num_draft_tokens)
    prefix = torch.clamp(seq_lens.to(torch.int64), min=0).to(torch.int32)
    extend = torch.full((bs,), W, dtype=torch.int32, device=dev)
    post = (seq_lens.to(torch.int64) + W).to(seq_lens.dtype)
    positions = torch.empty((bs * W,), dtype=torch.int64, device=dev)
    starts = torch.empty((bs,), dtype=torch.int32, device=dev)
    mrope = (
        torch.empty((3, bs * W), dtype=torch.int64, device=dev) if with_mrope else None
    )
    if bs:
        rows = torch.arange(bs * W, device=dev)
        req = torch.div(rows, W, rounding_mode="floor")
        token_base = seq_lens[req].to(torch.int64)
        prefix_i = torch.clamp(token_base, min=0).to(torch.int32)
        pos = prefix_i + (rows % W).to(torch.int32)
        positions.copy_(pos.to(torch.int64))
        starts[:bs] = (torch.arange(bs, device=dev) * W).to(torch.int32)
        if mrope is not None:
            mrope[:, :] = pos.to(torch.int64).unsqueeze(0)
    return prefix, extend, post, (positions, starts, mrope)


@register_jit_op(
    "sglang.kernels.ops.speculative.eagle", "prepare_draft_extend_inputs"
)
def prepare_draft_extend_inputs_kunlun(accept_lens, predict, num_draft_tokens):
    """Torch version of the EAGLE draft-extend input preparation."""
    bs = accept_lens.numel()
    tokens = predict.numel()
    dev = accept_lens.device
    accepted = accept_lens.to(torch.int64)
    correct = (accepted - 1).to(accept_lens.dtype)
    select = (
        torch.arange(bs, device=dev, dtype=torch.int64) * num_draft_tokens
        + accepted
        - 1
    )
    if predict.dim() == 1:
        ids = predict[:tokens].to(torch.int64)
    else:
        ids = predict.reshape(tokens, -1)[:, 0].to(torch.int64)
    return correct, select, ids


@register_jit_op("sglang.kernels.ops.speculative.eagle", "build_chain_tree")
def build_chain_tree_kunlun(
    bonus_tokens,
    draft_tokens,
    seq_lens,
    tree_mask_buf=None,
    req_pool_indices=None,
    req_to_token=None,
    with_mrope=False,
):
    """Torch version of the topk=1 chain tree builder."""
    bs, depth = draft_tokens.shape
    width = depth + 1
    dev = seq_lens.device
    mask = tree_mask_buf
    if mask is None:
        mask = torch.empty((bs * width * width,), dtype=torch.bool, device=dev)
    positions = torch.empty((bs * width,), dtype=torch.int64, device=dev)
    retrieve = torch.empty((3, bs, width), dtype=torch.int64, device=dev)
    tokens = torch.empty(
        (bs * width,),
        dtype=torch.promote_types(bonus_tokens.dtype, draft_tokens.dtype),
        device=dev,
    )
    out_cache_loc = torch.empty_like(positions) if req_to_token is not None else None
    mrope = torch.empty((3, bs * width), dtype=torch.int64, device=dev) if with_mrope else None
    if bs == 0:
        return mask, positions, retrieve[0], retrieve[1], retrieve[2], tokens, out_cache_loc, mrope

    rows = torch.arange(bs, device=dev)
    cols = torch.arange(width, device=dev)
    length = seq_lens.to(torch.int64).reshape(bs, 1)
    pos = (length + cols[None, :]).reshape(-1)
    positions[:] = pos
    mask[: bs * width * width] = (
        cols[None, :] <= cols[:, None]
    ).reshape(-1).repeat(bs)
    tok = torch.empty((bs, width), dtype=tokens.dtype, device=dev)
    tok[:, 0] = bonus_tokens.reshape(-1).to(tokens.dtype)
    tok[:, 1:] = draft_tokens.to(tokens.dtype)
    tokens[:] = tok.reshape(-1)
    off = torch.arange(bs * width, device=dev)
    retrieve[0] = off.reshape(bs, width)
    next_col = torch.where(
        cols + 1 < width,
        cols + 1,
        torch.full_like(cols, -1),
    )
    retrieve[1] = next_col[None, :].expand(bs, width)
    retrieve[2] = -1
    if req_to_token is not None:
        req = req_pool_indices.to(torch.int64)
        loc = req_to_token[req[:, None], length + cols[None, :]]
        out_cache_loc[:] = loc.reshape(-1).to(out_cache_loc.dtype)
    if with_mrope:
        mrope[:, :] = pos.unsqueeze(0)
    return mask, positions, retrieve[0], retrieve[1], retrieve[2], tokens, out_cache_loc, mrope

