"""Kunlun torch-native KV item mover for the HiSparse host<->device transfers.

The upstream hisparse kernels (``python/sglang/kernels/jit/csrc/kvcacheio/
hisparse.cuh``) address two KV layouts:

* **DSv4 page-padded C4** — a page holds 64 tokens laid out as 64 x 576B values
  followed by 64 x 8B scales, padded up to a 576B multiple (37440B per page).
  A "location" is a token index into the paged buffer, so ``index >> 6`` selects
  the page and ``index & 63`` the token slot inside it.
* **Flat byte stride** — both ends are linear buffers with ``item_size_bytes``
  stride per token (the generic MLA layout).

Both layouts are reproduced here with plain torch ops.  ``as_strided`` exposes
the value and scale runs of a padded page without materialising a page-sized
index tensor, which keeps the index set small enough for a captured graph.
"""

from __future__ import annotations

import torch

DSV4_PAGE_BITS = 6
DSV4_PAGE_SIZE = 1 << DSV4_PAGE_BITS
DSV4_PAGE_MASK = DSV4_PAGE_SIZE - 1
DSV4_VALUE_BYTES = 576
DSV4_SCALE_BYTES = 8
DSV4_SCALE_BYTE_OFFSET = DSV4_VALUE_BYTES * DSV4_PAGE_SIZE
# Padded page row: ceil((576 + 8) * 64 / 576) * 576.
DSV4_ROW_BYTES = 37440


def copy_items(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_locs: torch.Tensor,
    dst_locs: torch.Tensor,
    item_size_bytes: int,
    is_dsv4_layout: bool,
    page_size: int = DSV4_PAGE_SIZE,
) -> None:
    """Copy one KV item per ``(src_locs[i] -> dst_locs[i])`` pair, in place.

    A pair whose source and destination coincide is a no-op read/write of the
    same item, which lets callers keep a static-sized index set by parking
    irrelevant entries on a self-copy.

    ``is_dsv4_layout`` selects the paged C4 addressing.  When a page row is
    exactly ``page_size * item_size_bytes`` the buffer is a flat token array
    (the Kunlun half-cache layout) and the split value/scale addressing of
    upstream's padded row is skipped.
    """
    if src_locs.numel() == 0:
        return
    src_locs = src_locs.reshape(-1).to(torch.int64)
    dst_locs = dst_locs.reshape(-1).to(torch.int64)
    # Host<->device transfers put the two ends on different devices, so each
    # side gets its own index tensors (the cross-device assignment carries the
    # data across).
    src_locs_src = src_locs.to(src.device)
    dst_locs_dst = dst_locs.to(dst.device)

    if is_dsv4_layout and src.shape[1] != page_size * item_size_bytes:
        # Upstream's padded C4 row: 64 x 576B values, then 64 x 8B scales, then
        # padding, so values and scales are two separate runs of the page row.
        src_page = src_locs_src >> DSV4_PAGE_BITS
        dst_page = dst_locs_dst >> DSV4_PAGE_BITS
        src_off = src_locs_src & DSV4_PAGE_MASK
        dst_off = dst_locs_dst & DSV4_PAGE_MASK
        src_value = src.as_strided(
            (src.shape[0], DSV4_PAGE_SIZE, DSV4_VALUE_BYTES),
            (src.stride(0), DSV4_VALUE_BYTES, 1),
            0,
        )
        dst_value = dst.as_strided(
            (dst.shape[0], DSV4_PAGE_SIZE, DSV4_VALUE_BYTES),
            (dst.stride(0), DSV4_VALUE_BYTES, 1),
            0,
        )
        dst_value[dst_page, dst_off] = src_value[src_page, src_off].to(dst.device)

        src_scale = src.as_strided(
            (src.shape[0], DSV4_PAGE_SIZE, DSV4_SCALE_BYTES),
            (src.stride(0), DSV4_SCALE_BYTES, 1),
            DSV4_SCALE_BYTE_OFFSET,
        )
        dst_scale = dst.as_strided(
            (dst.shape[0], DSV4_PAGE_SIZE, DSV4_SCALE_BYTES),
            (dst.stride(0), DSV4_SCALE_BYTES, 1),
            DSV4_SCALE_BYTE_OFFSET,
        )
        dst_scale[dst_page, dst_off] = src_scale[src_page, src_off].to(dst.device)
        return

    # Contiguous token runs.  A DSv4 page row is exactly ``page_size`` runs of
    # ``item_size_bytes``; the generic MLA layout is one flat run per token.
    if is_dsv4_layout:
        rows, block = src.shape[0], page_size
    else:
        rows, block = src.reshape(-1).numel() // item_size_bytes, 1
    src_items = src.reshape(-1).as_strided(
        (rows, block, item_size_bytes), (block * item_size_bytes, item_size_bytes, 1), 0
    )
    dst_items = dst.reshape(-1).as_strided(
        (rows, block, item_size_bytes), (block * item_size_bytes, item_size_bytes, 1), 0
    )
    # XPU index_put_ rejects a source on another device, so gather first and
    # move the items across explicitly.
    dst_items[dst_locs_dst // block, dst_locs_dst % block] = src_items[
        src_locs_src // block, src_locs_src % block
    ].to(dst.device)
