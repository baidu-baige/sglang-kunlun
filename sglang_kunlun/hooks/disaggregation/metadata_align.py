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
"""2MB-align the host metadata buffers that a PD peer writes into over RDMA.

``MetadataBuffers`` lays its per-request metadata out in plain CPU tensors
(``output_ids``, ``output_top_logprobs_*``, ``bootstrap_room`` ...), so the start
address is only whatever ``torch`` happened to hand out. Kunlun RDMA
(``ibv_reg_mr``) requires BOTH the MR start and length to be 2MB-aligned;
``hooks/disaggregation/mooncake_register.py`` only rounds the length up, so every
unaligned aux buffer is rejected (``register_memory`` returns -7/-202), never
reaches the RDMA metadata, and the prefill peer then fails with
``Failed to get segment descriptor`` while writing ``send_aux``.

Swap each host buffer for a 2MB-aligned view with the same shape/dtype/stride:
every reader and writer keeps using ``self.<name>[...]`` unchanged, while the
registration now covers the real region. Things that do not apply are left alone:

* ``custom_mem_pool`` builds (Mooncake NVLINK) own their allocator -- reallocating
  would move the buffer out of the pool;
* device-resident buffers (NPU/NVLINK) are already handled by their own pool.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.hooks.mem_cache._align import ALIGNMENT_2M, alloc_2m_aligned

logger = logging.getLogger(__name__)

# Everything ``MetadataBuffers.get_buf_infos`` hands to ``register_buffer_to_engine``.
_HOST_BUFFERS = (
    "output_ids",
    "cached_tokens",
    "output_token_logprobs_val",
    "output_token_logprobs_idx",
    "output_top_logprobs_val",
    "output_top_logprobs_idx",
    "output_token_sampling_mask_len",
    "output_token_sampling_mask_idx",
    "output_token_sampling_logprobs",
    "output_topk_p",
    "output_topk_index",
    "output_hidden_states",
    "output_dsa_topk_indices",
    "bootstrap_room",
    "kv_checksum",
)


def align_host_buffers(owner) -> int:
    """Point every host metadata buffer at a 2MB-aligned copy; return how many moved.

    Shapes, dtypes and row sizes are preserved, so ``get_buf_infos`` item lengths
    and ``self.<name>[idx]`` access patterns are unaffected.
    """
    moved = 0
    for name in _HOST_BUFFERS:
        buf = getattr(owner, name, None)
        if not isinstance(buf, torch.Tensor) or buf.device.type != "cpu":
            continue
        if buf.data_ptr() % ALIGNMENT_2M == 0:
            continue
        aligned = alloc_2m_aligned(tuple(buf.shape), buf.dtype, buf.device)
        aligned.copy_(buf)
        setattr(owner, name, aligned)
        moved += 1
    return moved


@plugin_hook(
    "sglang.srt.disaggregation.utils.MetadataBuffers.__init__",
    type=HookType.AFTER,
)
def align_metadata_buffers(result, self, *args, **kwargs):
    """AFTER signature: hook(result, self). Original returns None."""
    if getattr(self, "custom_mem_pool", None) is not None:
        return result
    moved = align_host_buffers(self)
    if moved:
        logger.info(
            "sglang-kunlun: 2MB-aligned %d host metadata buffer(s) for RDMA registration",
            moved,
        )
    return result
