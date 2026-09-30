"""Kunlun-backed ``sgl_kernel`` attention helpers.

Only ``merge_state_v2`` lives here. The CUDA ``sgl_kernel`` provides it as a
custom op; on Kunlun ``sgl_kernel`` is the stub package
(``bootstrap/sgl_kernel_stub.py``), so every caller got a ``_StubModule`` that
raises on call. Two of them matter:

* ``hooks/layers/attention/kunlun_backend.py:29`` binds it at import time and
  wraps it in ``merge_state_v2_wrapper`` (4 call sites) -- that wrapper only
  added the -inf clamp, it never had a working op underneath.
* ``srt/models/deepseek_common/attention_forward_methods/forward_mha.py:39``
  imports it directly for the chunked-prefix cascade and calls it at ``:421``
  with the 6-argument out-variant. That is the K3 long-prompt path, and no
  plugin wrapper covers it.

The merge itself is a log-sum-exp weighted combination of two attention
partials, so a torch implementation is exact rather than approximate; it costs
a few elementwise passes per chunk boundary, not per token. Replacing it with
``kunlun_ops.attention_merge_stage`` is a performance follow-up, not a
correctness one.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# Fully-masked attention blocks produce LSE = -inf, and (-inf) - (-inf) is nan.
# Clamping to a finite floor is exact for the merge: exp(-1e30) == 0. The
# symptom when it is missing is an answer that degenerates into a run of "!"
# tokens, and it shows up both at a long prompt's chunk boundary (one side of
# the cascade can have no valid keys) and under concurrent short requests.
# Applied inside the op so the upstream forward_mha call site is covered too,
# not only the wrapper in kunlun_backend.py.
_LSE_FLOOR = -1e30


def _sanitized(lse: torch.Tensor) -> torch.Tensor:
    """LSE as float32 with -inf replaced by a finite floor (never in place)."""
    out = lse.to(torch.float32)
    return out.clamp(min=max(_LSE_FLOOR, torch.finfo(torch.float32).min))


def merge_state_v2(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention partials by their log-sum-exp weights.

    Mirrors ``sglang/kernels/aot/python/sgl_kernel/attention.py:6``: the two
    optional trailing arguments are outputs, and they are returned as well so
    both the functional and the out-variant call styles work.

    Args:
        v_a: First partial output, either ``[tokens, heads, dim]`` or the
            head-major flattened ``[tokens, heads * dim]``.
        s_a: First partial log-sum-exp, ``[tokens, heads]``.
        v_b: Second partial output, same shape as ``v_a``.
        s_b: Second partial log-sum-exp, same shape as ``s_a``.
        v_merged: Optional destination for the merged output.
        s_merged: Optional destination for the merged log-sum-exp.

    Returns:
        ``(v_merged, s_merged)``.

    Raises:
        ValueError: If the LSE shape cannot be reconciled with the value
            tensor's. A silent broadcast here would mis-attribute weights
            across heads and tokens -- the same failure the ``.T.contiguous()``
            in ``merge_state_v2_wrapper`` exists to prevent.
    """
    if tuple(v_a.shape) != tuple(v_b.shape) or tuple(s_a.shape) != tuple(s_b.shape):
        raise ValueError(
            f"merge_state_v2: shape mismatch between partials, "
            f"{tuple(v_a.shape)}/{tuple(s_a.shape)} vs "
            f"{tuple(v_b.shape)}/{tuple(s_b.shape)}"
        )

    # Two layouts reach this op. Attention backends return the head-major
    # flattened `[tokens, heads * dim]` (that is what the chunked-prefix cascade
    # in forward_mha.py:407 passes, `[208, 1536]` against a `[208, 12]` LSE),
    # while the absorbed-MLA call sites in kunlun_backend.py keep heads as their
    # own axis. The LSE always carries the head count, so it decides how to view
    # the values; head h owns the contiguous columns [h*dim, (h+1)*dim).
    view_a, view_b = v_a, v_b
    if tuple(s_a.shape) != tuple(v_a.shape[:-1]):
        if (
            v_a.ndim == s_a.ndim
            and tuple(v_a.shape[:-1]) == tuple(s_a.shape[:-1])
            and s_a.shape[-1] > 0
            and v_a.shape[-1] % s_a.shape[-1] == 0
        ):
            heads = s_a.shape[-1]
            tail = (heads, v_a.shape[-1] // heads)
            view_a = v_a.view(*v_a.shape[:-1], *tail)
            view_b = v_b.view(*v_b.shape[:-1], *tail)
        else:
            raise ValueError(
                f"merge_state_v2: lse shape {tuple(s_a.shape)} cannot be "
                f"reconciled with value shape {tuple(v_a.shape)}; expected "
                f"{tuple(v_a.shape[:-1])} or a head-major flattened last dim "
                "divisible by the head count."
            )

    sa = _sanitized(s_a)
    sb = _sanitized(s_b)
    m = torch.maximum(sa, sb)
    wa = torch.exp(sa - m)
    wb = torch.exp(sb - m)
    denom = wa + wb

    out_v = view_a * (wa / denom).unsqueeze(-1).to(view_a.dtype) + view_b * (
        wb / denom
    ).unsqueeze(-1).to(view_b.dtype)
    out_s = m + torch.log(denom)

    if v_merged is None:
        v_merged = out_v.reshape(v_a.shape).to(v_a.dtype)
    else:
        v_merged.copy_(out_v.reshape(v_merged.shape))
    if s_merged is None:
        s_merged = out_s
    else:
        s_merged.copy_(out_s)
    return v_merged, s_merged
