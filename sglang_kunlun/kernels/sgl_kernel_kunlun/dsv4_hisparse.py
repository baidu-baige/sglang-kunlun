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

# One pinned staging buffer per (rows, item_bytes, dtype), grown in place and reused.
# The swap-in runs once per layer per decode step — 43 x N allocations of a fresh
# non-contiguous temp was a measurable part of the 10x throughput gap, and pinning is
# what turns the H2D from a synchronous staged copy into a DMA.
_STAGE: dict = {}


def _pinned_stage(rows: int, item_bytes: int, dtype: torch.dtype) -> torch.Tensor:
    key = (item_bytes, dtype)
    buf = _STAGE.get(key)
    if buf is None or buf.shape[0] < rows:
        # Over-allocate so a slightly larger batch does not reallocate every step.
        want = max(rows, (buf.shape[0] * 2 if buf is not None else 1024))
        try:
            buf = torch.empty((want, item_bytes), dtype=dtype, pin_memory=True)
        except Exception:
            # No pinned memory available (some containers cap it): a plain tensor is
            # still better than a per-call temp, it just loses the DMA path.
            buf = torch.empty((want, item_bytes), dtype=dtype)
        _STAGE[key] = buf
    return buf[:rows]


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
    # Address both ends in bytes: the paged host pool is uint8 while the device
    # half-cache is bf16, so a token spans a different number of elements on
    # each side.  ``view(torch.uint8)`` is metadata-only for these contiguous
    # rows.
    if src.dtype != torch.uint8:
        src = src.view(torch.uint8)
    if dst.dtype != torch.uint8:
        dst = dst.view(torch.uint8)
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

        if src.device == dst.device:
            dst_value[dst_page, dst_off] = src_value[src_page, src_off]
            dst_scale[dst_page, dst_off] = src_scale[src_page, src_off]
            return

        # Cross-device (the HiSparse swap-in: host pool -> device buffer) used to be
        # two independent `gather().to(device)` round trips, one for the values and one
        # for the scales. Both landed in pageable memory, so each was a synchronous
        # staged copy, and the gather allocated a fresh non-contiguous temp per layer
        # per step. Measured on longbench think-high at maxrun 8: 11.9 tok/s against
        # 171.6 for the same config without HiSparse — ~720ms per decode step for
        # ~103MB of KV (8 req x 512 top-k x 584B x 43 layers), i.e. ~143MB/s, an order
        # of magnitude under what one H2D of that size should reach. The link was never
        # the bottleneck; the pageable staging and the per-call allocations were.
        #
        # The fused `hisparse_copy_items` kernel cannot help here twice over: it is
        # gated on `src.device == dst.device`, and this padded row returns before
        # reaching it anyway (the DSV4 host pool row is 37440B while
        # page_size*item_size is 64*584=37376B, so the shape test above sends every
        # DSV4 swap-in down this branch).
        #
        # So: gather values and scales into ONE pinned contiguous staging buffer,
        # cross the bus once, and scatter on the device side where both ends are local.
        n = src_locs_src.numel()
        item = DSV4_VALUE_BYTES + DSV4_SCALE_BYTES
        stage_host = _pinned_stage(n, item, src.dtype)
        stage_host[:, :DSV4_VALUE_BYTES] = src_value[src_page, src_off]
        stage_host[:, DSV4_VALUE_BYTES:] = src_scale[src_page, src_off]
        stage_dev = stage_host.to(dst.device, non_blocking=True)
        dst_value[dst_page, dst_off] = stage_dev[:, :DSV4_VALUE_BYTES]
        dst_scale[dst_page, dst_off] = stage_dev[:, DSV4_VALUE_BYTES:]
        return

    # Fused XPU kernel for the contiguous layout (device<->device only): one
    # gather-copy that replaces the as_strided + advanced-index path below
    # (~24x faster than the torch gather, and CUDA-graph safe).  Falls back to
    # torch for cross-device transfers or when the op isn't built into this
    # xspeedgate_ops.  Byte offset of a token loc is loc*item_size for both the
    # DSv4 half-cache (row = page_size*item_size) and the flat layout, matching
    # the kernel's addressing.
    if (
        src.device == dst.device
        and src.is_cuda
        and src.is_contiguous()
        and dst.is_contiguous()
        and hasattr(torch.ops.xspeedgate_ops, "hisparse_copy_items")
    ):
        torch.ops.xspeedgate_ops.hisparse_copy_items(
            src, dst, src_locs_src, dst_locs_dst, item_size_bytes
        )
        return

    # Contiguous token runs.  A DSv4 page row is exactly ``page_size`` runs of
    # ``item_size_bytes``; the generic MLA layout is one flat run per token.
    # The two ends have independent token spaces (the host pool is sized by
    # host_to_device_ratio), so each side gets its own row count.
    if is_dsv4_layout:
        block = page_size
        src_rows, dst_rows = src.shape[0], dst.shape[0]
    else:
        block = 1
        src_rows = src.reshape(-1).numel() // item_size_bytes
        dst_rows = dst.reshape(-1).numel() // item_size_bytes
    src_items = src.reshape(-1).as_strided(
        (src_rows, block, item_size_bytes),
        (block * item_size_bytes, item_size_bytes, 1),
        0,
    )
    dst_items = dst.reshape(-1).as_strided(
        (dst_rows, block, item_size_bytes),
        (block * item_size_bytes, item_size_bytes, 1),
        0,
    )
    # XPU index_put_ rejects a source on another device, so gather first and
    # move the items across explicitly.
    dst_items[dst_locs_dst // block, dst_locs_dst % block] = src_items[
        src_locs_src // block, src_locs_src % block
    ].to(dst.device)
