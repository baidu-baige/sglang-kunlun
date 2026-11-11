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
"""R1 port · native int8 dense MQA-logits for the DSV4.1 prefill-CP indexer.

What this replaces on the new image
-----------------------------------
The rewritten ``dsv41_cp_indexer.py`` (935 lines) already routes the ragged top-k to
XSpeedGate (ascending, sort skipped) and the *epilogue* to
``xspeedgate_ops.mqa_logits_sum_heads``.  What is still torch on the dense path is the
work *before* the epilogue, per row-window block:

    k = _dequant_indexer_fp4(k_codes, k_sf, table)          # torch dequant of the whole K
    s = torch.matmul(q[r], kblk.transpose(0, 1)).to(bf16)   # bf16 score GEMM [rows, H, lc]
    -> mqa_logits_sum_heads(s, w[r], lens[r], logits[r], lc)  (or the torch epilogue)

``kunlun_ops.I8_mqa_logits`` does dequant-side int8 dot + relu + weighted head-sum inside one
kernel: q and K are re-quantised to int8, ``q_scale`` is folded into the head weights (relu
commutes with the positive per-token scales, so the fold is exact up to int8 rounding), and
the causal per-request windows come from the ``(q_lod, k_lod)`` pairs.  Same math, same
[int8 q + int8 K + fp32 k_scale] contract that R1 validated on the old stack (+37.5 %
throughput, top-512 Jaccard 0.978).

This module is sglang-free so ``acc_check_r1port.py`` can exercise exactly the shipped
numerics inside the pod.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# Same gate name as the original R1 so launchers stay compatible.
ENV_VAR = "SGLANG_KUNLUN_CP_NATIVE_INDEXER"


def enabled_from_env(default: int = 0) -> bool:
    """Return whether the native R1 dense-indexer path is enabled.

    Reads ``SGLANG_KUNLUN_CP_NATIVE_INDEXER``: unset or blank keeps ``default``;
    ``1/true/yes/on`` enables, ``0/false/no/off`` disables, and any other value
    warns once and keeps the torch path (returns False).
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None or raw.strip() == "":
        return bool(default)
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    logger.warning("%s=%r is not a bool; keeping the torch dense MQA-logits", ENV_VAR, raw)
    return False


def available() -> bool:
    """``kunlun_ops.I8_mqa_logits`` present (verified in the new image: yes)."""
    try:
        import kunlun_ops

        return hasattr(kunlun_ops, "I8_mqa_logits")
    except Exception:
        return False


def per_token_int8(x: torch.Tensor, dim: int):
    """Symmetric per-slice int8 quant along ``dim``: returns (int8, fp32 scale)."""
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12)
    scale = (amax / 127.0).to(torch.float32)
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(dim)


def lod_from_ks(counts, lens_per_seq, device):
    """``(cumulative_q_lens, cumulative_k_lens)`` as (cpu, device) int32 pairs.

    ``counts`` = q-tokens per request (from ``unique_consecutive(ks)``);
    ``lens_per_seq`` = per-request visible k length (the last row of each group, i.e. the
    causal max window).
    """
    q_cum = [0]
    for c in counts:
        q_cum.append(q_cum[-1] + int(c))
    k_cum = [0]
    for l in lens_per_seq:
        k_cum.append(k_cum[-1] + int(l))
    q_cpu = torch.tensor(q_cum, dtype=torch.int32)
    k_cpu = torch.tensor(k_cum, dtype=torch.int32)
    return (q_cpu, q_cpu.to(device)), (k_cpu, k_cpu.to(device))


def dense_mqa_logits_i8(q, kv_fp4, weights, ks, ke, max_seqlen_k,
                        dequant_fn=None):
    """Native int8 dense MQA-logits: ``logits[r, j] = sum_h relu(<q[r,h], k[s+j,h]>) * w[r,h]``.

    ``q`` is the indexer's own bf16 query ``[T, H, 128]``; ``kv_fp4`` is the packed fp4 K
    cache ``(codes, scales)``; ``ks``/``ke`` are the per-row window bounds.  ``dequant_fn``
    is supplied by the caller (the plugin's ``_dequant_indexer_fp4``) so this module does not
    depend on sglang's dequant table.

    Raises on any shape/dtype it cannot handle so the caller can fall back.
    """
    import kunlun_ops

    if dequant_fn is None:
        raise ValueError("dense_mqa_logits_i8: dequant_fn is required")
    k_codes, k_sf = kv_fp4
    num_tokens, num_heads, head_dim = q.shape
    device = q.device
    if num_tokens == 0 or k_codes.shape[0] == 0:
        raise ValueError("dense_mqa_logits_i8: empty input (caller keeps the torch path)")
    if head_dim % 8 != 0:
        raise ValueError(f"dense_mqa_logits_i8: head_dim {head_dim} not 8-aligned")

    k_bf16 = dequant_fn(k_codes, k_sf)                      # [L, 128] bf16
    ks_l = ks.to(torch.long)
    lens = ke.to(torch.long) - ks_l
    starts, counts = torch.unique_consecutive(ks_l, return_counts=True)
    if int(counts.numel()) == 0:
        raise ValueError("dense_mqa_logits_i8: empty window groups")
    row_ends = torch.cumsum(counts, 0) - 1
    lens_per_seq = lens[row_ends].tolist()

    q_i8, q_scale = per_token_int8(q.to(torch.bfloat16), dim=-1)   # [T,H,128], [T,H]
    k_i8, k_scale = per_token_int8(k_bf16, dim=-1)                 # [L,128], [L]
    weights_fused = weights.to(torch.float32) * q_scale.to(torch.float32)

    (cq_cpu, cq_xpu), (ck_cpu, ck_xpu) = lod_from_ks(
        counts.tolist(), lens_per_seq, device
    )
    logits = torch.zeros((num_tokens, max_seqlen_k), dtype=torch.float32, device=device)
    kunlun_ops.I8_mqa_logits(
        q_i8,
        (k_i8, k_scale.to(torch.float32)),
        weights_fused,
        (cq_cpu, cq_xpu),
        (ck_cpu, ck_xpu),
        logits,
        clean_logits=True,
        max_seq_q=int(counts.max().item()),
        max_seq_k=int(max_seqlen_k),
        is_causal=True,
    )
    if tuple(logits.shape) != (num_tokens, max_seqlen_k):
        raise ValueError(f"dense_mqa_logits_i8: op returned {tuple(logits.shape)}")
    return logits
