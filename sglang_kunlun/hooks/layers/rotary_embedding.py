"""rotary_embedding"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _as_head_view(t: torch.Tensor, num_tokens: int, head_size: int) -> torch.Tensor:
    """Reshape to ``[tokens, heads, head_size]`` without materialising a copy.

    ``flashinfer_rotary_embedding`` takes non-contiguous views -- it writes back
    through the original storage and only requires the last dimension to have
    stride 1 (``xspeedgate_ops/_rope.pyi:177-178, 204``: "query/key 允许是非连续
    视图 ... 算子直接写回原 storage, 无需调用方手动 .contiguous(), 无额外内存
    分配" / "query 和 key 最后一维的 stride 必须为 1, 其余维度 stride 不限").

    ``Tensor.view`` cannot express that: it raises on any input whose strides are
    not row-major, which forces every caller holding a slice (the absorbed-MLA
    ``q[..., kv_lora_rank:]`` fork, for one) to pay a ``.contiguous()`` the
    kernel never wanted.  Split the last axis instead, so a strided input keeps
    its strides and only a genuinely unusable layout falls back to a copy.
    """
    if t.dim() == 3 and t.shape[0] == num_tokens and t.shape[2] == head_size:
        return t if t.stride(2) == 1 else t.contiguous()
    if t.stride(-1) == 1 and t.shape[-1] % head_size == 0:
        # unflatten keeps the leading strides, so a view over a slice stays a
        # view; it only fails when the split needs interleaving, which the
        # stride(-1) == 1 guard above has already excluded.
        try:
            return t.reshape(num_tokens, -1, head_size)
        except RuntimeError:
            pass
    return t.contiguous().view(num_tokens, -1, head_size)


@plugin_hook(
    "sglang.srt.layers.rotary_embedding.RotaryEmbedding.forward_cuda",
    type=HookType.REPLACE,
)
def rotary_embedding_forward_kunlun(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    fused_set_kv_buffer_arg=None,
):
    """Kunlun rotary embedding via ``xspeedgate_ops.flashinfer_rotary_embedding``."""
    assert (
        fused_set_kv_buffer_arg is None
    ), "fused_set_kv_buffer_arg is not supported on Kunlun"

    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]

    # The vendor rotary requires q and k to share one dtype (`RuntimeError:
    # query and key must have same dtype`). On this platform the two arrive from
    # different producers: the EAGLE draft's projection emits the model dtype
    # while the target's attention path emits bf16 from the vendor kernels
    # (measured elsewhere in this plugin: `x=bf16 residual=fp16`). `key` is the
    # tensor that carries the stream, so q is normalised to it rather than the
    # other way round.
    if query.dtype != key.dtype:
        query = query.to(key.dtype)

    query_shape = query.shape
    query = _as_head_view(query, num_tokens, self.head_size)
    key_shape = key.shape
    key = _as_head_view(key, num_tokens, self.head_size)

    query, key = torch.ops.xspeedgate_ops.flashinfer_rotary_embedding(
        positions=positions,
        rotary_dim=self.rotary_dim,
        head_size=self.head_size,
        cos_sin_cache=self.cos_sin_cache,
        is_neox_style=self.is_neox_style,
        query=query,
        key=key,
        offsets=offsets,
    )
    query = query.reshape(query_shape)
    key = key.reshape(key_shape)
    return query, key
