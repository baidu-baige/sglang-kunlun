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
"""HookRegistry registration for the DSV4 compress-plan generators.

Upstream ``CompressorDecodePlan.generate`` / ``CompressorPrefillPlan.generate``
(in ``sglang.jit_kernel.dsv4.compress``) call the tvm_ffi CUDA ``c_plan.cuh``
JIT kernel, which cannot build on the Kunlun XPU.

``generate`` is a ``@staticmethod`` on a ``NamedTuple`` subclass;
``HookRegistry._apply_target`` detects the staticmethod descriptor, unwraps it,
installs the REPLACE hook, and re-wraps the result in ``staticmethod`` — so the
class attribute keeps its original static-call semantics.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _make_decode_plan_torch(
    compress_ratio: int,
    req_pool_indices: torch.Tensor,   # [bs] int64
    req_to_token: torch.Tensor,       # [num_reqs, max_seq] int32
    full_to_swa: torch.Tensor,        # [num_full_slots] int64
    seq_lens: torch.Tensor,           # [bs] int64
    swa_page_size: int,
    ring_size: int,
) -> torch.Tensor:
    """Vectorized, CUDA-graph-capturable reimplementation of plan_compress_decode.

    # [CG-VECTORIZED decode plan]
    Returns plan_d: [bs, 16] uint8 (DecodePlan: uint32 seq_len, int32 write_loc,
    int32 read_page_0, int32 read_page_1). Device-only ops (no .item()/python loop)
    so the plan is recorded in the cuda graph and recomputed from the live seq_lens
    buffer at replay; index clamping makes the capture-time fill sentinel harmless.
    Numerically identical to the original per-row loop for real inputs.
    """
    bs = int(seq_lens.shape[0])
    device = seq_lens.device
    cr = int(compress_ratio)
    sps = int(swa_page_size)
    rs = int(ring_size)
    max_col = int(req_to_token.shape[1])
    n_swa = int(full_to_swa.shape[0])

    plan_d = torch.zeros(bs, 16, dtype=torch.uint8, device=device)
    pi = plan_d.view(torch.int32)  # [bs, 4]

    seq = seq_lens.to(torch.int64)
    write_pos = (seq // cr) * cr                      # [bs]
    valid = write_pos > 0                             # [bs] bool
    idx = (write_pos - 1).clamp_(0, max_col - 1)      # [bs]
    rid = req_pool_indices.to(torch.int64)
    token_loc_abs = req_to_token[rid, idx].to(torch.int64).clamp_(0, n_swa - 1)
    swa_loc = full_to_swa[token_loc_abs].to(torch.int64)
    state_loc = ((swa_loc // sps) * rs + (swa_loc % rs)) // cr   # [bs]

    seq32 = seq.to(torch.int32)
    state32 = state_loc.to(torch.int32)
    neg = torch.full((bs,), -1, dtype=torch.int32, device=device)
    pi[:, 0] = seq32
    pi[:, 1] = torch.where(valid, state32, neg)
    pi[:, 2] = torch.where(valid, state32, neg)
    pi[:, 3] = torch.zeros(bs, dtype=torch.int32, device=device)
    return plan_d


def _make_decode_plan_c4_torch(
    compress_ratio,
    req_pool_indices,
    req_to_token,
    full_to_swa,
    seq_lens,
    swa_page_size,
    ring_size,
):
    """Vectorized, CUDA-graph-capturable c4 DecodePlan builder.

    # [CG-VECTORIZED decode plan]
    [bs,16] uint8 int32 = {seq_len, write_loc, read_page_0, read_page_1}. Device-only
    ops with index clamping; numerically identical to the original per-row loop for
    real inputs, but graph-capturable (recomputes from live seq_lens at replay).
    """
    bs = int(seq_lens.shape[0])
    device = seq_lens.device
    cr = int(compress_ratio)
    sps = int(swa_page_size)
    rs = int(ring_size)
    max_col = int(req_to_token.shape[1])
    n_swa = int(full_to_swa.shape[0])

    plan_d = torch.zeros(bs, 16, dtype=torch.uint8, device=device)
    pi = plan_d.view(torch.int32)

    seq = seq_lens.to(torch.int64)
    valid = seq > 0
    pos1 = (seq - 1).clamp_(0, max_col - 1)
    pos0 = (seq - 1 - cr).clamp_(0, max_col - 1)
    rid = req_pool_indices.to(torch.int64)
    raw1 = req_to_token[rid, pos1].to(torch.int64).clamp_(0, n_swa - 1)
    raw0 = req_to_token[rid, pos0].to(torch.int64).clamp_(0, n_swa - 1)
    swa1 = full_to_swa[raw1].to(torch.int64)
    swa0 = full_to_swa[raw0].to(torch.int64)

    def cl(x):
        return (x // sps) * rs + (x % rs)

    wl = cl(swa1)                       # [bs]
    seq32 = seq.to(torch.int32)
    neg = torch.full((bs,), -1, dtype=torch.int32, device=device)
    zero = torch.zeros(bs, dtype=torch.int32, device=device)
    pi[:, 0] = torch.where(valid, seq32, zero)
    pi[:, 1] = torch.where(valid, wl.to(torch.int32), neg)
    pi[:, 2] = torch.where(valid, (cl(swa0) // cr).to(torch.int32), neg)
    pi[:, 3] = torch.where(valid, (wl // cr).to(torch.int32), neg)
    return plan_d


@plugin_hook(
    "sglang.jit_kernel.dsv4.compress.CompressorDecodePlan.generate",
    type=HookType.REPLACE,
)
def _decode_plan_generate_torch(
    compress_ratio,
    req_pool_indices,
    req_to_token,
    full_to_swa,
    seq_lens,
    swa_page_size,
    ring_size,
):
    """Pure-torch ``CompressorDecodePlan.generate`` (no tvm_ffi)."""
    from sglang.jit_kernel.dsv4.compress import CompressorDecodePlan

    _mk_dec = (
        _make_decode_plan_c4_torch
        if int(compress_ratio) == 4
        else _make_decode_plan_torch
    )
    plan_d = _mk_dec(
        int(compress_ratio),
        req_pool_indices,
        req_to_token,
        full_to_swa,
        seq_lens,
        int(swa_page_size),
        int(ring_size),
    )
    return CompressorDecodePlan(int(compress_ratio), plan_d)


@plugin_hook(
    "sglang.jit_kernel.dsv4.compress.CompressorPrefillPlan.generate",
    type=HookType.REPLACE,
)
def _prefill_plan_generate_torch(
    compress_ratio,
    req_pool_indices,
    seq_lens,
    extend_lens,
    req_to_token,
    full_to_swa,
    swa_page_size,
    ring_size,
    num_q_tokens,
    use_cuda_graph=False,
):
    """Pure-torch ``CompressorPrefillPlan.generate`` (no tvm_ffi)."""
    from sglang.jit_kernel.dsv4.compress import CompressorPrefillPlan
    from sglang_kunlun.kernels.dsv4_torch_shim import (
        _make_prefill_plan_c4_torch,
        _make_prefill_plan_torch,
    )

    _mk_pre = (
        _make_prefill_plan_c4_torch
        if int(compress_ratio) == 4
        else _make_prefill_plan_torch
    )
    plan_c, plan_w = _mk_pre(
        int(compress_ratio),
        req_pool_indices,
        seq_lens,
        extend_lens,
        req_to_token,
        full_to_swa,
        int(swa_page_size),
        int(ring_size),
        int(num_q_tokens),
    )
    return CompressorPrefillPlan(int(compress_ratio), plan_c, plan_w, None)
