"""Kunlun implementation of ``sgl_kernel.weak_ref_tensor``.

``sglang.srt.compilation.weak_ref_tensor`` imports this symbol from ``sgl_kernel``
on CUDA/HIP/XPU, and the breakable CUDA-graph capture (the prefill graph path)
calls it on every captured argument. The Kunlun host has no CUDA sgl_kernel build
behind the stub, so without a real implementation the capture raises
``NotImplementedError`` and the server never reaches ready.

XPU has no Python-accessible weak-storage constructor to mirror the CUDA kernel
with: ``UntypedStorage._new_with_weak_ptr`` is CPU-only (``torch.cuda`` raises
"Not available for CUDA storage"), and ``torch._C._construct_storage_from_data_pointer``
takes ownership of the raw pointer — aliasing a live tensor with it frees foreign
memory on collection and corrupts the heap. So return a detached alias: same data
pointer, shape and strides, with no reference to the source tensor object. It does
keep the source *storage* alive for the alias's lifetime, so the mempool cannot
recycle captured intermediates between segments; that is a memory tradeoff, not a
correctness one — the graph keeps reading the same addresses.
"""

from __future__ import annotations

import torch


def weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Alias ``tensor``'s memory for graph capture without referencing the tensor."""
    return tensor.detach()
