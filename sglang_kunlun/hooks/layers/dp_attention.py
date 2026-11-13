# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Keep the DP-attention gather dtype-consistent on Kunlun.

Two dtype hazards show up on the Kunlun DeepSeek-V4 stack once DP attention is
enabled (``--dp 8 --enable-dp-attention``):

1. ``_DpGatheredBufferWrapper`` allocates its gather/scatter buffers in
   ``model_config.dtype`` (``_DpGatheredBufferWrapper.set_metadata``), which is
   ``float16`` under ``--dtype float16``. The MoE hidden states that flow
   through these buffers are ``bfloat16``: the Kunlun quantized MoE
   (``moe_fc``) *forces* its expert output to bf16 whenever activations are
   quantized, and ``moe_post``'s ``normed_scale`` is taken from the hidden
   dtype, so the whole MoE FFN has to stay bf16. An fp16 gather buffer then
   breaks both the collective (``TypeError: output tensor must have the same
   type as input tensor``) and the MoE post-combine
   (``RuntimeError: moe_post not support x/normed_scale dtype``).
   For DeepSeek-V4 the only users of ``get_global_dp_buffer`` /
   ``get_local_dp_buffer`` are the MoE FFN gather/scatter in
   ``_run_moe_ffn_dp_sync`` (the TBO path allocates from ``local.dtype``), so
   pinning the buffer to bf16 is scoped to exactly that path.

2. Even with the buffer fixed, keep a defensive cast at the ``_dp_gather``
   choke point so any remaining float mismatch degrades to a cast instead of a
   hard collective failure.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_mismatch_logged = False


@plugin_hook(
    "sglang.srt.layers.moe.utils.should_use_dp_reduce_scatterv",
    type=HookType.REPLACE,
)
def _should_use_dp_reduce_scatterv_kunlun() -> bool:
    """Disable the pynccl-backed ``reduce_scatterv`` DP combine on Kunlun.

    ``should_use_dp_reduce_scatterv`` is True for the tp=dp=ep layout and routes
    the MoE combine through ``tp_group.reduce_scatterv`` (also skipping the
    MoE-internal post-experts all-reduce). That collective is built on pynccl
    (``pynccl_comm.change_state``), which Kunlun never initialises, so the first
    decode step aborts with ``'NoneType' object has no attribute 'change_state'``.
    Returning False keeps the standard pynccl-free all_reduce + dp_scatter
    combine.

    Returns:
        bool: Always False on Kunlun.
    """
    return False


@plugin_hook(
    "sglang.srt.layers.dp_attention._DpGatheredBufferWrapper.set_metadata",
    type=HookType.AROUND,
)
def _dp_buffer_bf16_bridge(original_fn, cls, hidden_size, dtype, device):
    """Allocate the DP gather/scatter buffer in bf16, not fp16.

    The Kunlun quantized MoE FFN runs in bf16 (see module docstring), so an
    fp16 DP buffer would disagree with the hidden states on both the gather
    collective and the MoE post-combine. Any other dtype is left untouched.

    Args:
        original_fn: The wrapped ``set_metadata`` classmethod body.
        cls: The ``_DpGatheredBufferWrapper`` class (classmethod binding).
        hidden_size: Hidden size recorded on the DP flags bag.
        dtype: Requested buffer dtype (``model_config.dtype``).
        device: Buffer device.

    Returns:
        The return value of ``original_fn``.
    """
    if dtype == torch.float16:
        dtype = torch.bfloat16
    return original_fn(cls, hidden_size, dtype, device)


@plugin_hook(
    "sglang.srt.layers.dp_attention._dp_gather",
    type=HookType.AROUND,
)
def _dp_gather_dtype_bridge(
    original_fn,
    global_tokens,
    local_tokens,
    forward_batch,
    is_partial,
    cp_shard_counts=None,
):
    """Cast ``local_tokens`` to ``global_tokens.dtype`` before the DP gather.

    The global buffer is pre-allocated by the caller and must be filled in
    place, so its dtype is the one the collective has to agree on; the local
    (per-rank) tokens are the operand we are free to convert. When the two
    dtypes already match this is a pure pass-through.

    Args:
        original_fn: The wrapped ``_dp_gather`` implementation.
        global_tokens: Pre-allocated gathered buffer (output of the gather).
        local_tokens: This rank's tokens (input of the gather).
        forward_batch: The batch the gather belongs to.
        is_partial: Whether the gather is partial (replicate vs sum).

    Returns:
        The return value of ``original_fn``.
    """
    global _mismatch_logged
    if (
        global_tokens is not None
        and local_tokens is not None
        and local_tokens.dtype != global_tokens.dtype
    ):
        if not _mismatch_logged:
            _mismatch_logged = True
            logger.warning(
                "sglang-kunlun: DP gather dtype mismatch (global=%s local=%s); "
                "casting local tokens to the global buffer dtype",
                global_tokens.dtype,
                local_tokens.dtype,
            )
        local_tokens = local_tokens.to(global_tokens.dtype)
    # 0.5.21 added ``cp_shard_counts`` to _dp_gather_via_all_reduce (CP ranks
    # write their own shards); forward it only when the caller supplied it so
    # the all-gather variant keeps its four-argument signature.
    if cp_shard_counts is not None:
        return original_fn(
            global_tokens, local_tokens, forward_batch, is_partial,
            cp_shard_counts=cp_shard_counts,
        )
    return original_fn(global_tokens, local_tokens, forward_batch, is_partial)
