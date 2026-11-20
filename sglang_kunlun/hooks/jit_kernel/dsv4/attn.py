# Copyright 2026 Baidu, Inc.
# Licensed under the Apache License, Version 2.0.
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
"""Hooks for ``sglang.kernels.ops.attention.dsv4.attn``.

``fused_store_cache`` on Kunlun (the upstream store.cuh JIT-compiles CUDA and
cannot build on the XPU host toolchain -- ``fatal error: concepts``):

- bf16/fp16 cache (SWA / flashmla extra-key): store via the native,
  cuda-graph-capturable ``torch.ops.xspeedgate_ops.set_k_and_s_v4`` (no fp8
  pack), the same op the non-CP path uses.
- uint8 indexer cache (``type="indexer"``): graph-safe torch per-token int8
  quant + paged scatter into the mixed page layout the Kunlun indexer-logits
  ops (``c4a_mqa_logits`` / ``c4a_paged_mqa_logits_with_mixed_cache``) read:
  per page = ``[page_size, D]`` int8 K block, then ``[page_size]`` fp32 scale
  block (``bytes_per_token = D + 4``). Dequant is ``k_int8 * k_scale``
  (per-token symmetric ``amax/127``), consistent with those ops and with the
  torch ``_c4a_cp_extend_logits_torch`` fallback in ``deepseek_v4_backend``.
- any other dtype: fall through to the original JIT kernel.

The target is the defining module (``...dsv4.attn``); the registry's
``_propagate_patch`` fixes the ``sglang.jit_kernel.dsv4`` re-export and the
``from ... import fused_store_cache`` binding in
``sglang.srt.mem_cache.deepseek_v4_memory_pool``.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


def _store_indexer_int8_kunlun(input, cache, indices, page_size):
    """Graph-safe torch store of fp32 indexer K into the uint8 mixed cache.

    input:   [N, D] fp32 compressed+normed+roped indexer K (D=128).
    cache:   [num_pages, page_bytes] uint8, page_bytes = page_size*(D+4).
    indices: [N] absolute slot ids (page*page_size + slot).
    Layout per page: [page_size, D] int8 K, then [page_size] fp32 scale.
    """
    N, D = input.shape
    if N == 0:
        return
    num_pages, page_bytes = cache.shape
    scale_offset = page_size * D
    assert page_bytes == scale_offset + page_size * 4, (
        f"unexpected indexer page layout: page_bytes={page_bytes}, "
        f"page_size={page_size}, D={D}"
    )
    dev = cache.device

    # Per-token symmetric int8 quant so that k_int8 * scale ~= k_fp32; the
    # logits ops multiply the stored per-token scale back in.
    amax = input.abs().amax(dim=-1, keepdim=True)                # [N,1]
    scale = (amax / 127.0).clamp_min(1e-8)                       # [N,1] fp32
    k_int8 = torch.round(input / scale).clamp_(-127, 127).to(torch.int8)  # [N,D]

    # Clamp loc to a valid slot to keep static shapes (CUDA-graph capture may
    # hand this sentinel/garbage indices; the real values arrive at replay).
    loc = indices.to(torch.int64).clamp_(0, num_pages * page_size - 1)
    pages = loc // page_size
    slots = loc % page_size
    base = pages * page_bytes                                    # [N] page byte base

    flat = cache.view(-1)                                        # contiguous uint8

    ar_d = torch.arange(D, device=dev)
    k_dst = (base + slots * D).unsqueeze(1) + ar_d.unsqueeze(0)  # [N,D]
    flat[k_dst.reshape(-1)] = k_int8.view(torch.uint8).reshape(-1)

    ar_4 = torch.arange(4, device=dev)
    s_dst = (base + scale_offset + slots * 4).unsqueeze(1) + ar_4.unsqueeze(0)  # [N,4]
    flat[s_dst.reshape(-1)] = scale.contiguous().view(torch.uint8).reshape(-1)


@plugin_hook(
    "sglang.kernels.ops.attention.dsv4.attn.fused_store_cache",
    type=HookType.AROUND,
)
def fused_store_cache_kunlun(
    original_fn, input, cache, indices, page_size, type="flashmla",
    layout=None, freqs_cis=None,
):
    """bf16/fp16 SWA fused store; uint8 indexer store; else the JIT kernel.

    ``layout``/``freqs_cis`` were added upstream when callers began passing
    ``layout=self.kv_layout``. The Kunlun stores implement only the V4 layout
    (fp8 nope + bf16 rope), which stores already-rotated kv and has no
    in-kernel RoPE, so a non-V4 layout or a freqs_cis request is rejected.
    """
    if layout is not None:
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        layout = KVLayout.parse(layout)
        assert layout is KVLayout.V4, (
            "sglang_kunlun fused_store_cache only supports the V4 KV layout "
            f"(V4.1 layouts are CUDA sm100 only); got {layout}"
        )
    assert freqs_cis is None, (
        "sglang_kunlun fused_store_cache has no in-kernel RoPE; the V4 layout "
        "stores already-rotated kv (freqs_cis must be None)"
    )
    if cache.dtype in (torch.bfloat16, torch.float16):
        kv_out = input.to(cache.dtype).contiguous()
        npages = cache.shape[0]
        loc = indices.to(torch.int64).clamp_(0, npages * page_size - 1)
        torch.ops.xspeedgate_ops.set_k_and_s_v4(cache, loc, kv_out, page_size)
        return
    if type == "indexer" and cache.dtype == torch.uint8:
        _store_indexer_int8_kunlun(input, cache, indices, page_size)
        return
    extra = {} if layout is None else {"layout": layout}
    return original_fn(
        input=input, cache=cache, indices=indices, page_size=page_size, type=type,
        **extra,
    )
