"""Kunlun implementation of ``sgl_kernel.weak_ref_tensor``.

Semantics to reproduce
----------------------
The CUDA build of ``sgl_kernel`` implements this as
``at::from_blob(data_ptr, sizes, strides, options)``: a tensor that aliases the
same storage **without owning it** (no deleter, no storage refcount). Breakable
CUDA graph depends on exactly that: the tensors handed across a graph break are
produced inside the capture-pool segments, so only the pool's own lifetime may
keep them alive. A strong Python reference would pin the block instead, the
mempool could not recycle it, and capture would grow the pool by every
break-point activation (hundreds of MiB per token bucket).

Kunlun has no such kernel, but the CUDA array interface gives the same
semantics in pure Python: ``torch.as_tensor`` on a *producer object* builds a
device tensor over the raw pointer whose only owner is that (tiny, host-side)
description. Verified on kunlun: the alias has the same ``data_ptr`` as the
source, and once the source is dropped the allocator can hand the same block
out again -- i.e. the alias holds no reference to the storage.

Two dtype wrinkles, both handled below:

* ``torch`` describes bfloat16 as ``<V2`` (void) and has no description at all
  for the fp8 family; the consumer route through numpy rejects both.
* So anything numpy cannot represent natively is described as a same-width
  integer and reinterpreted with ``Tensor.view(dtype)`` afterwards.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

__all__ = ["weak_ref_tensor"]

# Dtypes torch's CUDA-array-interface consumer maps straight through the
# buffer protocol (the set named in its own numpy conversion error).
_DIRECT_DTYPES = frozenset(
    {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.bool,
    }
)

# Same-width integer stand-in per element size (little-endian device storage).
_INT_TYPESTR = {1: "|u1", 2: "<u2", 4: "<u4", 8: "<u8"}


class _NonOwningAlias:
    """Producer object for :func:`torch.as_tensor`.

    Holds nothing but the array description: no reference to the source tensor
    and no reference to its storage, which is the whole point.
    """

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, description: dict) -> None:
        self.__cuda_array_interface__ = description


def _describe(tensor: torch.Tensor) -> Tuple[dict, Optional[torch.dtype]]:
    """Return (array description, dtype to reinterpret afterwards).

    The reinterpret dtype is set when the description uses an integer stand-in
    instead of the real element type.
    """
    itemsize = tensor.element_size()

    if tensor.dtype in _DIRECT_DTYPES:
        return dict(tensor.__cuda_array_interface__), None

    # Manual description with an integer stand-in (bf16, fp8, ...).
    integer_typestr = _INT_TYPESTR.get(itemsize)
    if integer_typestr is None:
        raise ValueError(f"unsupported element size {itemsize} for {tensor.dtype}")
    strides = None
    if not tensor.is_contiguous():
        strides = tuple(stride * itemsize for stride in tensor.stride())
    return (
        {
            "data": (tensor.data_ptr(), False),
            "shape": tuple(tensor.shape),
            "strides": strides,
            "typestr": integer_typestr,
            "version": 2,
        },
        tensor.dtype,
    )


def weak_ref_tensor(tensor: Any) -> Any:
    """Return a non-owning alias of ``tensor`` (like the CUDA kernel does).

    Falls back to returning the tensor itself when no description can be built
    (e.g. a tensor that requires grad). That fallback is always
    address-correct -- it only forgoes the memory-recycling guarantee.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.numel() == 0:
        # torch reports a null data pointer for empty tensors; nothing to alias.
        return tensor

    try:
        description, view_dtype = _describe(tensor)
    except (AttributeError, RuntimeError, ValueError):
        return tensor

    alias = torch.as_tensor(_NonOwningAlias(description), device=tensor.device)
    return alias.view(view_dtype) if view_dtype is not None else alias
