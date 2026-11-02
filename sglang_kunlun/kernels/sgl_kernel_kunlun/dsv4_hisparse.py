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

# Staging buffers keyed by (kind, layers, item_bytes, dtype, device), grown in place and
# reused.  The transfers run once per layer per decode step — 43 x N allocations of a
# fresh non-contiguous temp was a measurable part of the throughput gap, and pinning the
# host end is what turns a crossing into a DMA instead of a staged synchronous copy.
_STAGE: dict = {}


def _stage(
    kind: str,
    layers: int,
    rows: int,
    item_bytes: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """A reusable ``(layers, rows, item_bytes)`` staging buffer.

    ``device=None`` asks for pinned host memory.  Rows are over-allocated so a
    batch that grows by one request does not reallocate every step; the returned
    view is always exactly ``rows`` long.
    """
    key = (kind, layers, item_bytes, dtype, str(device))
    buf = _STAGE.get(key)
    if buf is None or buf.shape[1] < rows:
        want = max(rows, (buf.shape[1] * 2 if buf is not None else 64))
        if device is None:
            try:
                buf = torch.empty(
                    (layers, want, item_bytes), dtype=dtype, pin_memory=True
                )
            except Exception:
                # No pinned memory available (some containers cap it): a plain tensor
                # is still better than a per-call temp, it just loses the DMA path.
                buf = torch.empty((layers, want, item_bytes), dtype=dtype)
        else:
            buf = torch.empty((layers, want, item_bytes), dtype=dtype, device=device)
        _STAGE[key] = buf
    return buf[:, :rows]


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """Byte view of a KV buffer.

    Both ends are addressed in bytes: the paged host pool is uint8 while the
    device half-cache is bf16, so a token spans a different number of elements
    on each side.  ``view`` is metadata-only for these contiguous rows.
    """
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _padded_runs(t: torch.Tensor):
    """The value and scale runs of a padded C4 page row, as strided views.

    A page row is 64 x 576B of values followed by 64 x 8B of scales, so the two
    are separate runs and need separate addressing.  ``as_strided`` exposes them
    without materialising a page-sized index tensor.
    """
    return (
        t.as_strided(
            (t.shape[0], DSV4_PAGE_SIZE, DSV4_VALUE_BYTES),
            (t.stride(0), DSV4_VALUE_BYTES, 1),
            0,
        ),
        t.as_strided(
            (t.shape[0], DSV4_PAGE_SIZE, DSV4_SCALE_BYTES),
            (t.stride(0), DSV4_SCALE_BYTES, 1),
            DSV4_SCALE_BYTE_OFFSET,
        ),
    )


def copy_items_all_layer_d2h(
    src_buffers,
    dst_buffers,
    src_locs: torch.Tensor,
    dst_locs: torch.Tensor,
    item_size_bytes: int,
    page_size: int = DSV4_PAGE_SIZE,
) -> bool:
    """Back up the same token slice of every layer across ONE device->host crossing.

    Returns ``True`` when it handled the transfer and ``False`` when the layout
    is not the padded-C4 device->host case, in which case the caller must fall
    back to the per-layer path.

    Why this exists: ``_copy_token_granular`` called ``copy_items`` once per
    layer, and every call crossed the bus twice (values, then scales) into
    pageable or strided host memory -- a *blocking* copy each time.  With 43
    layers that is 86 host stalls per decode step to move ~200KB, and all of it
    sits inside what upstream deliberately made asynchronous:
    ``HiSparseCoordinator._eager_backup_previous_token`` issues the backup on
    ``decode_backup_stream`` and only waits on ``_backup_done_event`` at the
    next step.  py-spy on the live server put 30% of scheduler CPU samples on
    that one assignment, second only to the forward's own sync.

    So gather every layer into one contiguous device buffer first (43 x 2 kernel
    launches, no host sync at all), cross once into pinned host memory, then
    scatter host-side with plain CPU indexing.  86 synchronisations per step
    become one, and the gathers stay on the backup stream where they belong.
    """
    n_layers = len(src_buffers)
    if n_layers == 0:
        return True
    src0 = _as_bytes(src_buffers[0])
    dst0 = _as_bytes(dst_buffers[0])
    # Only the padded C4 row needs the split value/scale addressing, and only a
    # real device->host pair benefits -- same-device copies are already async.
    if src0.shape[1] == page_size * item_size_bytes:
        return False
    if not src0.is_cuda or dst0.is_cuda:
        return False
    _copy_all_layer_staged(src_buffers, dst_buffers, src_locs, dst_locs)
    return True


def _copy_all_layer_staged(src_buffers, dst_buffers, src_locs, dst_locs) -> None:
    """The staged all-layer padded-C4 copy, with no device gate.

    Split out from ``copy_items_all_layer_d2h`` so the addressing can be checked
    against the per-layer path on CPU tensors, where neither end is a device.
    """
    n_layers = len(src_buffers)
    src0 = _as_bytes(src_buffers[0])
    src_locs = src_locs.reshape(-1).to(torch.int64)
    dst_locs = dst_locs.reshape(-1).to(torch.int64)
    n = src_locs.numel()
    if n == 0:
        return

    item = DSV4_VALUE_BYTES + DSV4_SCALE_BYTES
    s_locs = src_locs.to(src0.device)
    s_page = s_locs >> DSV4_PAGE_BITS
    s_off = s_locs & DSV4_PAGE_MASK

    stage_dev = _stage("d2h", n_layers, n, item, torch.uint8, src0.device)
    for i in range(n_layers):
        s_value, s_scale = _padded_runs(_as_bytes(src_buffers[i]))
        stage_dev[i, :, :DSV4_VALUE_BYTES] = s_value[s_page, s_off]
        stage_dev[i, :, DSV4_VALUE_BYTES:] = s_scale[s_page, s_off]

    # The one crossing.  Blocking on purpose: the host scatter below reads it,
    # and this is the single synchronisation the whole backup now costs.
    stage_host = _stage("d2h", n_layers, n, item, torch.uint8, None)
    stage_host.copy_(stage_dev)

    d_page = dst_locs >> DSV4_PAGE_BITS
    d_off = dst_locs & DSV4_PAGE_MASK
    for i in range(n_layers):
        d_value, d_scale = _padded_runs(_as_bytes(dst_buffers[i]))
        d_value[d_page, d_off] = stage_host[i, :, :DSV4_VALUE_BYTES]
        d_scale[d_page, d_off] = stage_host[i, :, DSV4_VALUE_BYTES:]


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

        # Cross-device.  Two directions reach this point and they want opposite
        # treatment, which the first attempt at this got wrong:
        #
        # * host -> device (the HiSparse swap-in) used to be two independent
        #   `gather().to(device)` round trips, one for values and one for scales, both
        #   landing in pageable memory, so each was a synchronous staged copy, and the
        #   gather allocated a fresh non-contiguous temp per layer per step. Measured
        #   on longbench think-high at maxrun 8: 11.9 tok/s against 171.6 for the same
        #   config without HiSparse — ~720ms per decode step for ~103MB of KV (8 req x
        #   512 top-k x 584B x 43 layers), i.e. ~143MB/s, an order of magnitude under
        #   what one H2D of that size should reach. Gathering both runs into ONE pinned
        #   contiguous buffer and crossing once fixes the staging and the allocations.
        #
        # * device -> host (the eager backup) was then *regressed* by that same change:
        #   writing a device gather into a column slice of a pinned 2-D buffer is a
        #   strided D2H, which is worse than the contiguous copy it replaced (10.7
        #   against 11.9 tok/s). The backup wants the mirror image — gather on the
        #   device, cross contiguously — and it wants it batched across all 43 layers,
        #   so it is handled by `copy_items_all_layer_d2h` instead.
        #
        # The fused `hisparse_copy_items` kernel cannot help here twice over: it is
        # gated on `src.device == dst.device`, and this padded row returns before
        # reaching it anyway (the DSV4 host pool row is 37440B while
        # page_size*item_size is 64*584=37376B, so the shape test above sends every
        # DSV4 transfer down this branch).
        if src.is_cuda and not dst.is_cuda:
            copy_items_all_layer_d2h(
                [src], [dst], src_locs, dst_locs, item_size_bytes, page_size
            )
            return

        n = src_locs_src.numel()
        item = DSV4_VALUE_BYTES + DSV4_SCALE_BYTES
        stage_host = _stage("h2d", 1, n, item, torch.uint8, None)[0]
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
