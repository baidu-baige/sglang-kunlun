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
"""Hooks for ``sglang.srt.layers.attention.dsv4.metadata_kernel``.

One AROUND hook on ``init_compression_metadata`` that masks the uninitialised
tail of ``c128_page_indices`` (return index 8) to -1, the "no page" marker.

Upstream allocates the page-index table with ``torch.empty(bs,
c128_max_seq_len)`` and its Triton kernel only fills the ``seq_len // 128``
entries a row actually has. On the NSA prefill context-parallel path some rows
are never written at all (padded queries), so they keep whatever the allocator
handed out - typically old float ``-inf`` bits, i.e. ``-8388608`` read as int32.
Those bogus page ids reach the Kunlun compressed-attention op, which then reads
unrelated cache slots, so a long prompt produces degenerate text depending on
the KV placement of the request.

Registered on the defining module rather than on the
``_init_compression_metadata_triton`` aliases in ``deepseek_v4_backend`` /
``deepseek_v4_backend_hip_radix``: ``_propagate_patch`` rewrites every stale
``sys.modules`` binding anyway, so alias targeting buys nothing.

AROUND (not REPLACE) so it composes with whatever produced the tuple.
``kernel_ops.install()`` runs before ``apply_hooks()`` and already substitutes a
pure-torch ``init_compression_metadata`` (the Kunlun triton backend cannot
compile the stock kernel); that version masks correctly on its own, making this
hook a cheap no-op there, but it still guards the upstream kernel.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.metadata_kernel.init_compression_metadata",
    type=HookType.AROUND,
)
def init_compression_metadata_kunlun(original_fn, *args, **kwargs):
    """Mask every ``c128_page_indices`` entry past a row's valid length to -1."""
    out = original_fn(*args, **kwargs)
    page_indices = out[8]
    seq_lens_raw = out[6]
    if page_indices is not None and page_indices.numel():
        cols = torch.arange(page_indices.shape[1], device=page_indices.device)
        valid = cols.unsqueeze(0) < seq_lens_raw.to(torch.int64).unsqueeze(1)
        page_indices.copy_(
            torch.where(valid, page_indices, torch.full_like(page_indices, -1))
        )
    return out
