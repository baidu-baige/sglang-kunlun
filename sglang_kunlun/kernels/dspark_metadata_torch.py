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
"""Fixed-shape DSpark metadata using only torch operations.

Bad active indices raise in eager mode. During graph construction they are
masked to -1: this prevents unsafe memory access, but does NOT recover the
semantics of corrupted metadata. Mapping *values* (including -1) are outputs,
not indices, and are preserved without a second lookup.
"""

import torch


def _can_validate_values(device):
    """Determine whether device values can be validated on the host.

    Args:
        device (torch.device): Device holding the metadata tensors.

    Returns:
        bool: True outside graph construction when host reads are safe.
            False when capture is active or its state cannot be determined.
    """
    # The framework flag includes warmup, and Kunlun's stream query alone has
    # missed real capture. Conservatively skip host reads for the whole region.
    from sglang_kunlun.hooks.graph_state import is_graph_capturing

    if is_graph_capturing():
        return False
    if device.type == "cpu":
        return True
    try:
        return torch.cuda.is_current_stream_capturing() is False
    except Exception:
        return False


def _check_active(bad, validate, name):
    """Reject invalid active metadata when host validation is enabled.

    Args:
        bad (torch.Tensor): Boolean mask identifying invalid active entries.
        validate (bool): Whether reading the device predicate is permitted.
        name (str): Metadata field name used in the error message.

    Returns:
        None: Validation completes without producing a value.

    Raises:
        ValueError: Validation is enabled and an active entry is invalid.
    """
    if validate and bool(bad.any()):
        raise ValueError(f"DSpark {name}: invalid active index or context length")


def _safe_take(values, indices, active, validate, name):
    """Gather valid active indices and fill other entries with sentinel -1.

    Args:
        values (torch.Tensor): Source tensor, flattened for indexing.
        indices (torch.Tensor): Integer indices into the flattened source.
        active (torch.Tensor): Boolean mask of entries requiring a lookup.
        validate (bool): Whether invalid active indices raise on the host.
        name (str): Source field name used in validation errors.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Gathered values with the shape of
            indices and a boolean mask of valid active entries. Invalid or
            inactive entries contain -1, including when the source is empty.

    Raises:
        ValueError: Validation is enabled and an active index is out of bounds.
    """
    size = values.numel()
    good = (indices >= 0) & (indices < size)
    _check_active(active & ~good, validate, name)
    valid = active & good
    if size == 0:
        # Static shape branch: no gather is possible, even at index zero.
        return values.new_full(indices.shape, -1), torch.zeros_like(active, dtype=torch.bool)
    safe = torch.where(valid, indices, 0)
    # Zero is only a read-only safe index, never a replacement active KV.
    # Contiguous sources share storage here; noncontiguous reshape may copy.
    selected = values.reshape(-1)[safe]
    return torch.where(valid, selected, -1), valid


def build_dspark_swa_page_indices(
    *,
    req_to_token,
    full_to_swa_mapping,
    req_pool_indices_per_request,
    offsets,
    invalid,
    out_loc,
    context_lens,
    block_size,
    swa_window,
    page_index_aligned_size,
):
    """Build fixed-shape SWA page indices for each DSpark block token.

    All integer tensors must use int32 or int64 and share the offsets device.
    Eager execution validates active indices. Graph construction masks unsafe
    lookups without host reads; masking does not repair corrupted metadata.

    Args:
        req_to_token (torch.Tensor): Full-cache slot table with shape
            [request_pool_size, max_sequence_length].
        full_to_swa_mapping (torch.Tensor): One-dimensional full-to-SWA mapping.
        req_pool_indices_per_request (torch.Tensor): Request pool row indices
            with shape [batch].
        offsets (torch.Tensor): Window token offsets with shape
            [batch, swa_window].
        invalid (torch.Tensor): Boolean mask matching offsets, marking window
            entries that must not be read.
        out_loc (torch.Tensor): One-dimensional full-cache slots containing at
            least batch * block_size entries for the current blocks.
        context_lens (torch.Tensor): Window context lengths with shape [batch],
            each between zero and swa_window.
        block_size (int): Positive number of tokens in each DSpark block.
        swa_window (int): Nonnegative sliding-window width.
        page_index_aligned_size (int): Positive output-width alignment.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Contiguous int32 page indices with
            shape [batch * block_size, aligned_width] and context-plus-block
            lengths with shape [batch * block_size]. aligned_width is
            swa_window + block_size rounded up to page_index_aligned_size.
            Unused page entries contain -1; mapping values are preserved.

    Raises:
        TypeError: An input tensor has an unsupported dtype.
        ValueError: Scalar settings, tensor shapes or devices are invalid,
            lengths exceed int32 range, or eager validation detects an invalid
            active index or context length.
    """
    for name, value, minimum in (
        ("block_size", block_size, 1),
        ("swa_window", swa_window, 0),
        ("page_index_aligned_size", page_index_aligned_size, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if block_size + swa_window > torch.iinfo(torch.int32).max:
        raise ValueError("DSpark lengths exceed int32 output range")
    if offsets.ndim != 2 or offsets.shape[1] != swa_window:
        raise ValueError("offsets must have shape [batch, swa_window]")
    bs = offsets.shape[0]
    device = offsets.device
    inputs = {
        "req_to_token": req_to_token,
        "full_to_swa_mapping": full_to_swa_mapping,
        "req_pool_indices_per_request": req_pool_indices_per_request,
        "offsets": offsets,
        "out_loc": out_loc,
        "context_lens": context_lens,
    }
    for name, value in inputs.items():
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must have dtype int32 or int64")
        if value.device != device:
            raise ValueError(f"{name} must be on the offsets device")
    if invalid.dtype != torch.bool:
        raise TypeError("invalid must have dtype bool")
    if invalid.device != device or invalid.shape != offsets.shape:
        raise ValueError("invalid must match offsets shape and device")
    if req_to_token.ndim != 2 or full_to_swa_mapping.ndim != 1:
        raise ValueError("req_to_token must be 2D and full_to_swa_mapping 1D")
    if req_pool_indices_per_request.shape != (bs,) or context_lens.shape != (bs,):
        raise ValueError("request indices and context_lens must have shape [batch]")
    if out_loc.ndim != 1 or out_loc.numel() < bs * block_size:
        raise ValueError("out_loc must be 1D with at least batch * block_size entries")

    validate = _can_validate_values(device)
    width = (
        (swa_window + block_size + page_index_aligned_size - 1)
        // page_index_aligned_size * page_index_aligned_size
    )
    cl = context_lens.to(torch.int64)
    legal = (cl >= 0) & (cl <= swa_window)
    _check_active(~legal, validate, "context_lens")
    # Bounds are checked before arithmetic, avoiding overflow for corrupt int64.
    cl = torch.where(legal, cl, 0).view(bs, 1)
    k = torch.arange(width, device=device, dtype=torch.int64).view(1, width)
    in_window = legal[:, None] & (k < cl)
    in_block = legal[:, None] & (k >= cl) & (k < cl + block_size)

    # The extra column is a sentinel for all non-window output positions.
    src = torch.where(in_window, swa_window - cl + k, swa_window)
    off_pad = torch.cat((offsets.to(torch.int64), offsets.new_zeros((bs, 1))), dim=1)
    inv_pad = torch.cat((invalid, invalid.new_ones((bs, 1))), dim=1)
    off = torch.gather(off_pad, 1, src)
    active = in_window & ~torch.gather(inv_pad, 1, src)
    rp = req_pool_indices_per_request.to(torch.int64).view(bs, 1)
    rows, cols = req_to_token.shape
    good_row = (rp >= 0) & (rp < rows)
    good_col = (off >= 0) & (off < cols)
    _check_active(active & ~(good_row & good_col), validate, "req_to_token row/offset")
    valid_req = active & good_row & good_col
    # Sanitize each coordinate BEFORE multiplication / the flattened gather.
    row = torch.where(valid_req, rp, 0)
    col = torch.where(valid_req, off, 0)
    full, valid_req = _safe_take(
        req_to_token, row * cols + col, valid_req, validate, "req_to_token",
    )
    win, _ = _safe_take(
        full_to_swa_mapping, full.to(torch.int64), valid_req,
        validate, "window full_to_swa_mapping",
    )
    bcol = torch.where(in_block, k - cl, 0)
    base = torch.arange(bs, device=device, dtype=torch.int64).view(bs, 1) * block_size
    block_full, valid_block = _safe_take(
        out_loc, base + bcol, in_block, validate, "out_loc",
    )
    block, _ = _safe_take(
        full_to_swa_mapping, block_full.to(torch.int64), valid_block,
        validate, "block full_to_swa_mapping",
    )
    pages = torch.where(in_window, win, torch.where(in_block, block, -1)).to(torch.int32)
    pages = pages[:, None, :].expand(bs, block_size, width).reshape(bs * block_size, width)
    topk = torch.where(legal, cl.reshape(bs) + block_size, 0).to(torch.int32)
    topk = topk[:, None].expand(bs, block_size).reshape(bs * block_size)
    return pages.contiguous(), topk.contiguous()
