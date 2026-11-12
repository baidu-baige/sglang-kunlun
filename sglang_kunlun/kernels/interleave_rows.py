"""Torch replacement for the ``interleave_rows`` Triton kernel.

``sglang/kernels/ops/memory/interleave_rows.py`` launches
``_interleave_rows_kernel[grid](...)``, and Triton is a noop here, so the
subscript raises ``TypeError: 'function' object is not subscriptable``. It is
the prefill-CP interleave strategy's row-reorder primitive (three call sites in
``layers/cp/interleave.py``, including ``gather_kv_cache``), so without this
hook prefill CP dies on the first extend long enough to materialise global
token order.

The row mapping is copied from the kernel: for output row ``row``,

    to_rank_major:  source = row % rank_rows * cp_size + row // rank_rows
    otherwise:      source = row % cp_size * rank_rows + row // cp_size

with ``rank_rows = rank_total // cp_size``; rows whose ``source`` is past the
input are zero-filled, which is how sequence-to-rank copies get rank padding.
"""

from __future__ import annotations

import logging
import math

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_logged = False


@plugin_hook(
    "sglang.kernels.ops.memory.interleave_rows.interleave_rows",
    type=HookType.REPLACE,
)
def interleave_rows_kunlun(
    x: torch.Tensor,
    cp_size: int,
    output_rows: int,
    *,
    to_rank_major: bool = False,
) -> torch.Tensor:
    """Reorder contiguous rows between interleaved rank and sequence order."""
    global _logged

    assert x.device.type != "cpu" and x.is_contiguous() and x.ndim >= 2
    assert cp_size > 0 and output_rows >= 0
    rank_total = output_rows if to_rank_major else x.shape[0]
    sequence_total = x.shape[0] if to_rank_major else output_rows
    assert rank_total % cp_size == 0 and sequence_total <= rank_total

    if not _logged:
        _logged = True
        logger.info(
            "cp interleave_rows: torch path (cp_size=%d, in_rows=%d, out_rows=%d)",
            cp_size,
            x.shape[0],
            output_rows,
        )

    width = math.prod(x.shape[1:])
    if output_rows == 0 or width == 0:
        return x.new_empty((output_rows, *x.shape[1:]))

    rank_rows = rank_total // cp_size
    row = torch.arange(output_rows, device=x.device)
    if to_rank_major:
        source = (row % rank_rows) * cp_size + row // rank_rows
    else:
        source = (row % cp_size) * rank_rows + row // cp_size

    in_rows = x.shape[0]
    if in_rows == 0:
        return x.new_zeros((output_rows, *x.shape[1:]))
    valid = source < in_rows
    if bool(valid.all()):
        return x[source].contiguous()
    # Past-the-input rows are the rank padding and must read back as zero.
    out = x.new_zeros((output_rows, *x.shape[1:]))
    out[valid] = x[source[valid]]
    return out
