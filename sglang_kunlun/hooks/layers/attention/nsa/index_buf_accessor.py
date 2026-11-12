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

_vendor_gather_warned = False


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


@plugin_hook(
    target="sglang.kernels.ops.kvcache.dsa_index_k_cache.GetKAndS.triton",
    type=HookType.REPLACE,
)
def get_k_and_s_triton(cls, pool, buf, page_indices, seq_len_tensor, seq_len_sum, max_seq_len):
    """Gather the ragged index-K and its per-token fp32 scale.

    The module docstring above records why this replacement did not exist: the
    old split `GetK`/`GetS` accessors merged into one `GetKAndS`, and the DSA
    indexer was not on the DSV4-Flash baseline path. It is on GLM-5.3's: the
    ragged prefill branch of `_get_topk_ragged` reads it under
    `--enable-prefill-cp`, and with no backend registered the fused-op dispatch
    raises `RuntimeError: 0 compatible backends for target (cuda) ([]). There
    should only be one.` on the first prefill request.

    Both native kernels survived the upstream merge and are still in this image
    (`xspeedgate_ops::get_k_kernel`, `xspeedgate_ops::get_s_kernel`), so this is
    the 0.5.8 accessor expressed against the fused signature: the caller passes
    every request's page list at once, and the kernels take one request at a
    time, so the loop is over requests and the offsets come from
    `seq_len_tensor`.

    Layout, from `_get_k_and_s_triton_kernel` (`dsa_index_k_cache.py:368-400`):
    a page is `[index_head_dim K bytes | fp32 scales]` with
    `s_offset_in_page = page_size * index_head_dim`; request `i` occupies rows
    `[sum(seq_len[:i]), sum(seq_len[:i+1]))` of both outputs, which is exactly
    the concatenation the old per-request loop produced. K comes back as bytes
    (`uint8` in the kernel) and is reinterpreted by the caller, not cast.
    """
    import torch

    num_pages, buf_numel_per_page = buf.shape
    page_size = pool.index_page_size
    index_head_dim = pool.index_head_dim
    s_offset_in_page = page_size * index_head_dim

    device = buf.device
    k_out = torch.empty(
        (seq_len_sum, index_head_dim), dtype=torch.uint8, device=device
    )
    s_out = torch.empty((seq_len_sum, 4), dtype=torch.uint8, device=device)

    # 0.5.8 ran the two native kernels here (`index_buf_accessor.py:16,39`:
    # `xspeedgate_ops.get_k_kernel` / `get_s_kernel`, one call per request), and
    # the 0.5.8 traces show them as `xinfer3::get_k_xpu_kernel_optimized_kernel`
    # / `get_s_xpu_kernel_optimized_kernel`. They were dropped here for a
    # capture-stream error that has since been traced to something else
    # entirely (the DeepEP NORMAL notify during decode-graph capture, and the
    # fp16 draft KV pool's non-Triton scatter), so the vendor path is primary
    # again and the pure-torch gather below is the fallback.
    pages_per_req = page_indices.shape[1]
    flat_buf = buf.reshape(num_pages, buf_numel_per_page).contiguous()
    try:
        import xspeedgate_ops  # noqa: F401

        row = 0
        for i, seq_len in enumerate(seq_len_tensor.tolist()):
            if seq_len == 0:
                continue
            pages = (
                page_indices.reshape(-1)[i * pages_per_req : (i + 1) * pages_per_req]
                .contiguous()
            )
            k_out[row : row + seq_len] = torch.ops.xspeedgate_ops.get_k_kernel(
                flat_buf, pages, seq_len, page_size, buf_numel_per_page,
                index_head_dim,
            ).reshape(seq_len, index_head_dim)
            s_out[row : row + seq_len] = torch.ops.xspeedgate_ops.get_s_kernel(
                flat_buf, pages, seq_len, page_size, buf_numel_per_page,
                s_offset_in_page,
            ).reshape(seq_len, 4)
            row += seq_len
        return k_out, s_out
    except Exception as exc:  # noqa: BLE001 - fall back to the torch gather
        global _vendor_gather_warned
        if not _vendor_gather_warned:
            _vendor_gather_warned = True
            import logging

            logging.getLogger(__name__).warning(
                "xspeedgate_ops get_k/get_s unavailable (%s); using the torch "
                "gather -- the trace will lack get_k/get_s_xpu_kernel_optimized",
                exc,
            )

    # Fallback: the same arithmetic the triton kernel does
    # (`dsa_index_k_cache.py:368-400`): token `t` of request `i` reads page
    # `pages[t // page_size]` at offset `(t % page_size) * index_head_dim` for K
    # and at `s_offset_in_page + (t % page_size) * 4` for the fp32 scale.
    pages_per_req = page_indices.shape[1]
    flat_k = buf.reshape(num_pages, buf_numel_per_page)
    row = 0
    for i, seq_len in enumerate(seq_len_tensor.tolist()):
        if seq_len == 0:
            continue
        # Flatten first: some callers hand `page_indices` as `[batch, pages]`,
        # others as a flat vector; `index_select` needs a 1-D index either way.
        pages = page_indices.reshape(-1)[
            i * pages_per_req : (i + 1) * pages_per_req
        ].to(torch.int64)
        tok = torch.arange(seq_len, device=device, dtype=torch.int64)
        page_ids = pages.index_select(0, torch.div(tok, page_size, rounding_mode="floor"))
        in_page = tok % page_size
        k_idx = (in_page * index_head_dim).unsqueeze(1) + torch.arange(
            index_head_dim, device=device, dtype=torch.int64
        ).unsqueeze(0)
        k_out[row : row + seq_len] = flat_k.index_select(0, page_ids).gather(1, k_idx)
        s_idx = (s_offset_in_page + in_page * 4).unsqueeze(1) + torch.arange(
            4, device=device, dtype=torch.int64
        ).unsqueeze(0)
        s_out[row : row + seq_len] = flat_k.index_select(0, page_ids).gather(1, s_idx)
        row += seq_len
    return k_out, s_out
