"""Plugin hook for DeepSeek-V4 unquantized half-precision key packing."""

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from .index_buf_accessor_v4 import NopeFp8RopeBf16Pack


def _resolve_kv_cache_dtype(kv_cache_dtype: torch.dtype | None) -> torch.dtype:
    if kv_cache_dtype is not None:
        return kv_cache_dtype

    from sglang.srt.server_args import get_global_server_args

    value = get_global_server_args().kv_cache_dtype
    if value in ("fp16", "float16", "half"):
        return torch.float16
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("int8", "torch.int8"):
        return torch.int8
    if value is None or value == "auto":
        return get_global_server_args().dtype
    raise NotImplementedError(f"Unsupported DSV4 KV-cache dtype: {value}")


@plugin_hook(
    target=(
        "sglang.kernels.ops.attention.dsv4.quant_k_cache."
        "quant_to_nope_fp8_rope_bf16_pack_triton"
    ),
    type=HookType.REPLACE,
)
def quant_to_nope_fp8_rope_bf16_pack_triton(
    k_bf16: torch.Tensor,
    kv_cache_dtype: torch.dtype | None = None,
) -> NopeFp8RopeBf16Pack:
    """Pack an FP16/BF16 key without quantization."""
    _, hidden_dim = k_bf16.shape
    assert hidden_dim == 512

    resolved_dtype = _resolve_kv_cache_dtype(kv_cache_dtype)
    if resolved_dtype == torch.int8:
        return NopeFp8RopeBf16Pack(
            k_nope_fp8=k_bf16.to(torch.bfloat16).contiguous(),
            k_rope_bf16=None,
            scale_k_nope_ue8m0=None,
            kv_cache_dtype=torch.int8,
        )
    if resolved_dtype in (torch.bfloat16, torch.float16):
        return NopeFp8RopeBf16Pack(
            k_nope_fp8=k_bf16.to(resolved_dtype),
            k_rope_bf16=None,
            scale_k_nope_ue8m0=None,
            kv_cache_dtype=resolved_dtype,
        )
    raise NotImplementedError(f"Unsupported DSV4 KV cache dtype: {kv_cache_dtype}")
