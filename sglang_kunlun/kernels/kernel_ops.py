"""Fine-grained Kunlun replacements for SGLang Triton/JIT kernel symbols."""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

import xspeedgate_ops

def _debug_tensor_meta(name: str, value: object) -> str:
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


@register_jit_op("sglang.srt.model_executor.forward_batch_info", "clamp_position")
def clamp_position(seq_lens: torch.Tensor) -> torch.Tensor:
    """Compute non-negative last-token positions from sequence lengths."""

    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)


@register_jit_op("sglang.jit_kernel.hadamard", "hadamard_transform")
def hadamard_transform(x: torch.Tensor, scale: float = None) -> torch.Tensor:
    """Apply Kunlun Hadamard transform replacement."""

    if scale is None:
        return torch.ops.xspeedgate_ops.hadamard_transform(x)
    return torch.ops.xspeedgate_ops.hadamard_transform(x, scale)


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


@register_jit_op("sglang.srt.layers.moe.topk", "mask_topk_ids")
def mask_topk_ids(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor) -> None:
    """Mask padded token rows in MoE top-k ids on Kunlun."""

    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    topk_ids[indices >= num_token_non_padded, :] = -1


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

    # try:
    #     from sglang.srt.environ import envs

    #     use_fast_alloc_extend = (
    #         getattr(envs, "USE_FAST_ALLOC_EXTEND_KUNLUN", None)
    #         and envs.USE_FAST_ALLOC_EXTEND_KUNLUN.get()
    #     )
    # except Exception:
    #     use_fast_alloc_extend = False

    if 0:
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

    ret_value = torch.zeros(1, dtype=torch.int64, device=out_indices.device)
    torch.ops.xspeedgate_ops.alloc_extend(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        prefix_lens.shape[0],
        page_size,
        out_indices.shape[0],
        out_indices,
        ret_value,
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

    torch.ops.xspeedgate_ops.alloc_decode_kernel(
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
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
