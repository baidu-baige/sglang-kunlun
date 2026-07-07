"""rotary_embedding"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


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

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_size)

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
