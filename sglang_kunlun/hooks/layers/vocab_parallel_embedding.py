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
"""Kunlun XPU: disable the fused Triton vocab-parallel embedding.

Root cause (verified against checkpoint ground truth, 2026-09-18):

sglang 0.5.20 added a fused Triton kernel for vocab-parallel embedding
(``sglang/kernels/ops/embeddings/vocab_parallel_embedding.py``) and
``VocabParallelEmbedding._use_triton_embedding()`` only excludes NPU. Kunlun XPU
masquerades as CUDA, so every gate in that predicate passes and the Triton path
replaces the classic ``mask -> gather -> masked_fill_(0) -> all_reduce`` unit.

Inside the kernel the out-of-shard rows are supposed to be zeroed by::

    local_id = tl.where(valid, local_id, 0)
    vals = tl.load(weight_ptr + local_id * weight_stride0 + cols,
                   mask=col_mask & valid, other=0.0)

``valid`` is a *scalar* predicate while ``col_mask`` is a ``[BLOCK_H]`` vector.
On the kunlun Triton backend (triton 3.0.0+dd7ab66a) the scalar predicate is
lost in that broadcast, so the load degenerates to an unmasked one and every
rank returns its own local row 0 instead of zeros. After the TP all-reduce the
embedding becomes::

    embed(t) = W[t] + sum_{k != owner} W[k * vocab_per_rank]

Measured on DeepSeek-V4-Flash, TP8, token id 2398: the observed embedding
matches ``W[2398] + sum_{k=1..7} W[k*16160]`` with cosine 1.000000 (relative
error 3e-4, pure fp16 rounding), against a true row norm of 4.126 vs the
polluted 22.851. Every layer downstream therefore runs on a garbage input and
the sampler degenerates into emitting one token forever.

The non-Triton branch does the zeroing in torch and is correct, so the fix is
simply to keep kunlun off the Triton branch.

Set ``SGLANG_KUNLUN_FUSED_VOCAB_EMBEDDING=1`` to restore the upstream
(defective on kunlun) behaviour — used for the negative half of the A/B proof.
"""

import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding._use_triton_embedding",
    type=HookType.AROUND,
)
def _use_triton_embedding_kunlun(original_fn, self, input_):
    if os.environ.get("SGLANG_KUNLUN_FUSED_VOCAB_EMBEDDING", "0") == "1":
        return original_fn(self, input_)
    return False
