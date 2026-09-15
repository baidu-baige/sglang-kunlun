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
"""HookRegistry registrations for the v1 legacy DSV4 ``compress_old`` module.

Every target here builds on a tvm_ffi CUDA JIT kernel that cannot compile on the
Kunlun XPU, so each is REPLACEd with a native Kunlun op. This is the v1
counterpart of ``compress.py`` in this package (which covers the v2 ``compress``
module).

- ``CompressorPrefillPlan.generate`` (``c_plan.cuh``) →
  ``xspeedgate_ops.plan_compress_prefill``. ``generate`` is a ``@staticmethod``
  on a ``NamedTuple`` subclass; ``HookRegistry._apply_target`` detects the
  staticmethod descriptor, unwraps ``__func__``, installs the REPLACE hook, and
  re-wraps the result in ``staticmethod`` — so the class attribute keeps its
  original static-call semantics. (This is why the legacy ``register_jit_op``
  path could not reach it.)
- ``compress_forward`` (``_jit_compress_module``) →
  ``xspeedgate_ops.compress_forward_fast``.
- ``compress_fused_norm_rope_inplace`` (``_jit_norm_rope_module``) →
  ``kunlun_ops.dpsk_v4_norm_rope_gptj``.

Both module-level functions are also re-exported by the v1
``sglang.srt.layers.attention.dsv4.compressor`` with ``from ...compress_old
import compress_forward, compress_fused_norm_rope_inplace`` (compressor.py:9-13);
``_propagate_patch`` rewrites that stale binding, so the legacy shim's manual
second patch of the ``compressor`` module is no longer needed.

Behaviour note carried over from the legacy shim: upstream's
``compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS`` branch, which
delegates to ``CompressorPrefillPlan._generate_online``, is not reproduced —
``plan_compress_prefill`` is always used, with ``is_overlap = compress_ratio == 4``.
"""

from __future__ import annotations

import kunlun_ops
import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.jit_kernel.dsv4.compress_old.CompressorPrefillPlan.generate",
    type=HookType.REPLACE,
)
def _prefill_plan_generate_v1_kunlun(
    compress_ratio,
    num_q_tokens,
    seq_lens,
    extend_lens,
    device,
    use_cuda_graph=False,
):
    """Build the v1 prefill plan via ``xspeedgate_ops.plan_compress_prefill``."""
    from sglang.jit_kernel.dsv4.compress_old import CompressorPrefillPlan

    assert seq_lens.device == extend_lens.device
    if seq_lens.dtype != torch.int64:
        seq_lens = seq_lens.to(torch.int64)
    if extend_lens.dtype != torch.int64:
        extend_lens = extend_lens.to(torch.int64)
    plan_tensor = torch.empty(
        (2, num_q_tokens, 16),
        dtype=torch.uint8,
        device=seq_lens.device,
        pin_memory=seq_lens.is_cpu,
    )
    is_overlap = compress_ratio == 4
    plan_lens = torch.ops.xspeedgate_ops.plan_compress_prefill(
        extend_lens,
        seq_lens,
        plan_tensor[0],
        plan_tensor[1],
        compress_ratio,
        is_overlap,
        use_cuda_graph,
    )
    plan_device = plan_tensor.to(device, non_blocking=True)
    L0 = int(plan_lens[0])
    L1 = int(plan_lens[1])
    return CompressorPrefillPlan(
        compress_ratio,
        plan_device[0, :L0],
        plan_device[1, :L1],
    )


# ---------------------------------------------------------------------------
# compress_forward. Mirrors aiak_sglang's compress_forward_kunlun: route
# directly to the native xspeedgate_ops.compress_forward_fast kernel (confirmed
# present on Kunlun), using the legacy calling convention
# (indices=req_pool_indices, seq_lens/compress_plan, write_plan, extra_data).
# ---------------------------------------------------------------------------

@plugin_hook(
    "sglang.jit_kernel.dsv4.compress_old.compress_forward",
    type=HookType.REPLACE,
)
def _compress_forward_v1_kunlun(
    kv_score_buffer,
    kv_score_input,
    ape,
    indices,
    plan=None,
    extra_data=None,
    *,
    head_dim,
    compress_ratio,
    out=None,
    seq_lens=None,
    extend_lens=None,
):
    """Compress via ``xspeedgate_ops.compress_forward_fast``."""
    from sglang.jit_kernel.dsv4.compress_old import (
        CompressorDecodePlan,
        compress_plan as _compress_plan_v1,
    )

    assert head_dim % 128 == 0
    num_q_tokens = kv_score_input.shape[0]
    if out is None:
        # === FIX (mirrors aiak_sglang's CP c4 cross-rank divergence fix) ===
        # compress_forward_fast does NOT fully overwrite its output buffer;
        # an uninitialized `new_empty` out leaves per-allocation HBM garbage
        # in untouched positions, which differs run-to-run and across ranks,
        # corrupting the compressed KV written to the pool. Zero-init `out`
        # so untouched positions are deterministically 0 instead of garbage.
        out = kv_score_input.new_zeros((num_q_tokens, head_dim))
    if plan is None:
        assert seq_lens is not None
        plan = _compress_plan_v1(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            kv_score_input.device,
        )
    assert plan.compress_ratio == compress_ratio, "Mismatched compress ratio in plan!"
    if isinstance(plan, CompressorDecodePlan):
        torch.ops.xspeedgate_ops.compress_forward_fast(
            kv_score_buffer, kv_score_input, out, ape, indices,
            plan.seq_lens, None, extra_data,
        )
    else:
        torch.ops.xspeedgate_ops.compress_forward_fast(
            kv_score_buffer, kv_score_input, out, ape, indices,
            plan.compress_plan, plan.write_plan, extra_data,
        )
    return out


# ---------------------------------------------------------------------------
# compress_fused_norm_rope_inplace. Mirrors aiak_sglang's
# compress_fused_norm_rope_inplace_kunlun: route directly to the native
# kunlun_ops.dpsk_v4_norm_rope_gptj fused op (confirmed present on Kunlun).
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.jit_kernel.dsv4.compress_old.compress_fused_norm_rope_inplace",
    type=HookType.REPLACE,
)
def _compress_fused_norm_rope_inplace_v1_kunlun(
    kv,
    weight,
    eps,
    freq_cis,
    plan,
):
    """RMSNorm + GPT-J RoPE in place via ``kunlun_ops.dpsk_v4_norm_rope_gptj``."""
    from sglang.jit_kernel.dsv4.compress_old import CompressorPrefillPlan

    if isinstance(plan, CompressorPrefillPlan):
        handle = plan.compress_plan.view(torch.int32)  # uint8 bytes -> int32
        mode = 0
        if plan.compress_plan.numel() == 0:
            return kv
    else:
        handle = plan.seq_lens  # already int32
        mode = 1
    kunlun_ops.dpsk_v4_norm_rope_gptj(
        kv,
        weight,
        handle,  # compress_plan or seq_lens
        freq_cis,  # complex64, kunlun_ops converts internally
        mode,  # 0=prefill, 1=decode
        plan.compress_ratio,
        eps,
    )
