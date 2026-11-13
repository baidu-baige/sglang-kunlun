"""XSpeedGate-backed implementations of the sgl_kernel kvcacheio direct ops.

Upstream's `direct` / `page_first_direct` HiCache path calls
``sgl_kernel.kvcacheio.transfer_kv_*`` directly from ``pool_host/mla.py`` and
``pool_host/base.py``. On Kunlun ``sgl_kernel`` is a stub (see
``bootstrap/sgl_kernel_stub.py``) and those calls raise ``NotImplementedError``
on the first L2 backup. The vendor ``xspeedgate_ops`` wheel exports the same
transfers with identical signatures, so this module forwards to them and
installs itself into the stub's namespace: every upstream call site then works
without a per-site REPLACE hook.

Signatures are copied from ``xspeedgate_ops/_cache.pyi``; the upstream call
sites (``pool_host/mla.py`` backup_from_device_all_layer / load_to_device)
pass keyword arguments, so the parameter names must stay identical.
"""

from __future__ import annotations

import sys
from typing import Any

import torch


def _ops() -> Any:
    """Indirection seam: the unit test replaces this to record calls."""
    return torch.ops.xspeedgate_ops


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: list[torch.Tensor],
    dst_ptrs: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> None:
    """XPU layer-first -> CPU host page-first, all layers, one launch."""
    _ops().transfer_kv_all_layer_direct_lf_pf(
        src_ptrs, dst_ptrs, src_indices, dst_indices, page_size
    )


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: list[torch.Tensor],
    dst_ptrs: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    """CPU host page-first -> one XPU layer-first buffer."""
    _ops().transfer_kv_per_layer_direct_pf_lf(
        src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size
    )


def transfer_kv_direct(
    src_layers: list[torch.Tensor],
    dst_layers: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> None:
    """XPU <-> CPU page-first, one layer pair (the `layer_first` direct op)."""
    _ops().transfer_kv_direct(
        src_layers, dst_layers, src_indices, dst_indices, page_size
    )


#: op name -> implementation, for the stub install below.
DIRECT_OPS = {
    "transfer_kv_all_layer_direct_lf_pf": transfer_kv_all_layer_direct_lf_pf,
    "transfer_kv_per_layer_direct_pf_lf": transfer_kv_per_layer_direct_pf_lf,
    "transfer_kv_direct": transfer_kv_direct,
}


def install_into_stub() -> None:
    """Put the real implementations on the ``sgl_kernel.kvcacheio`` stub module.

    The stub creates ``sgl_kernel.kvcacheio`` lazily on first attribute access;
    after this call, upstream's dotted lookups resolve to these functions
    instead of a raising ``_StubModule``.
    """
    kvcacheio = sys.modules.get("sgl_kernel.kvcacheio")
    if kvcacheio is None:
        kvcacheio = sys.modules.get("sgl_kernel")
        if kvcacheio is None:
            return  # stub not installed; upstream sgl_kernel is the real thing
        kvcacheio = getattr(kvcacheio, "kvcacheio", None)
        if kvcacheio is None:
            return
    for name, fn in DIRECT_OPS.items():
        setattr(kvcacheio, name, fn)
