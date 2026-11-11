"""Kunlun hooks for ERNIE 5.5 (`sglang.srt.models.ernie55`)."""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

# T gate. Decode is 1 (draft) or 2 (MTP target-verify); in prefill T is the
# prompt length and the block diagonal grows as T^2, so fall back above it.
_VHA_MAX_T = 4


def _block_diag(self, mix: torch.Tensor, t: int, dtype) -> torch.Tensor:
    """``diag(mix, ..., mix)``, cached on the module and cleared by ``fold``.

    Cached in ``o.dtype`` because ``torch.mm`` rejects the fp32 ``mix`` against
    a bf16 ``o`` (upstream's ``matmul`` silently runs the bf16 kernel); the copy
    is within 1 ULP, while promoting both sides to fp32 is twice as slow.
    """
    cache = getattr(self, "_kl_bd", None)
    if cache is None:
        cache = self._kl_bd = {}
    key = (t, dtype)
    bd = cache.get(key)
    if bd is None:
        h = mix.shape[0]
        bd = torch.zeros(t * h, t * h, dtype=dtype, device=mix.device)
        src = mix.to(dtype)
        for i in range(t):
            bd[i * h : (i + 1) * h, i * h : (i + 1) * h] = src
        cache[key] = bd
    return bd


@plugin_hook(
    "sglang.srt.models.ernie55.VhaPostMix.forward",
    type=HookType.AROUND,
)
def vha_postmix_forward_kunlun(original_fn, self, o: torch.Tensor):
    """One block-diagonal 2-D mm instead of a stride-0 broadcast bmm.

    Upstream's ``torch.matmul(mix, heads)`` expands a 2-D ``mix`` to
    ``[T, H, H]`` with stride 0 and becomes a batched GEMM: ~78 us on 8 MFLOP at
    T=2. ``out[t,i,d] = sum_j mix[i,j] o[t,j,d]`` is the same result as one 2-D
    mm against ``diag(mix, ..., mix)``. Anything unexpected falls back.
    """
    t, nh = o.shape[0], self.num_heads
    if (
        self.sharded
        or not 0 < t <= _VHA_MAX_T
        or o.numel() % (t * nh)
        or o.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return original_fn(self, o)
    try:
        bd = _block_diag(self, self.mix, t, o.dtype)
        return torch.mm(bd, o.reshape(t * nh, -1)).reshape(o.shape)
    except Exception:  # noqa: BLE001 - never take the engine down for a perf path
        logger.exception(
            "sglang-kunlun: VhaPostMix block-diagonal failed (o=%s %s); "
            "falling back to upstream",
            tuple(o.shape),
            o.dtype,
        )
        return original_fn(self, o)


@plugin_hook(
    "sglang.srt.models.ernie55.VhaPostMix.fold",
    type=HookType.AROUND,
)
def vha_postmix_fold_kunlun(original_fn, self, *args, **kwargs):
    """``fold`` rewrites ``mix`` in place, so the cached block diagonal dies."""
    out = original_fn(self, *args, **kwargs)
    if getattr(self, "_kl_bd", None):
        self._kl_bd.clear()
    return out
