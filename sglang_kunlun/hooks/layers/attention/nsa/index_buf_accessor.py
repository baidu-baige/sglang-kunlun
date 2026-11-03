"""Function-level hooks for NSA index buffer accessors.

0.5.23 moved the module from ``sglang.kernels.ops.attention.dsa.index_buf_accessor``
to ``sglang.kernels.ops.kvcache.dsa_index_k_cache`` and merged the separate
``GetK``/``GetS`` accessors into a single ``GetKAndS`` with a different contract
(``_get_k_and_s_triton(buf, page_indices, seq_lens, seq_len_sum, max_seq_len,
page_size, index_head_dim)``). The Kunlun ``get_k_kernel``/``get_s_kernel``
replacements for the old split accessors therefore have no target left and were
dropped here; the DSA indexer gather is not on the DSV4-Flash baseline path
(``--attention-backend kunlun_compressed``), so no Kunlun ``GetKAndS`` is
provided. Reinstating the DSA indexer on Kunlun needs one.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    target="sglang.kernels.ops.kvcache.dsa_index_k_cache.SetKAndS.triton",
    type=HookType.REPLACE,
)
def set_k_and_s_triton(cls, pool, buf, loc, index_k, index_k_scale) -> None:
    """Set NSA K and S index buffers with Kunlun kernels."""
    import kunlun_ops

    # The kernel only reports "k_num != loc_num" without the counts; surface them,
    # since a mismatch here means the caller mixed a CP-sharded K with full-batch
    # cache locations (or vice versa).
    if index_k.shape[0] != loc.shape[0]:
        raise RuntimeError(
            "kunlun set_k_and_s_triton: row mismatch "
            f"index_k={tuple(index_k.shape)} index_k_scale={tuple(index_k_scale.shape)} "
            f"loc={tuple(loc.shape)} page_size={pool.page_size}"
        )

    kunlun_ops.set_k_and_s_triton(
        buf=buf.contiguous(),
        loc=loc.to(torch.int64).contiguous(),
        index_k=index_k.contiguous(),
        index_k_scale=index_k_scale.contiguous(),
        page_size=pool.page_size,
    )
