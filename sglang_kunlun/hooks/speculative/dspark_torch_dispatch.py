"""Keep DSPARK on Torch except for verified Kunlun vendor routes.

Kunlun tensors report ``is_cuda`` although native Triton cannot run. Module
dispatch therefore stays pinned to Torch. Only the classes below may enter
``cls.triton``: their complete kernel paths have Kunlun replacements, and the
live bindings are checked before every call. Missing/late replacements keep
the Torch reference. The historical DSV4_DSPARK_TRITON=1 opt-in is unchanged.
"""

from __future__ import annotations

import importlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

_MODULES = (
    "dspark_attn_metadata",
    "dspark_verify_window",
    "dspark_schedule",
    "dspark_accept",
    "dspark_draft_model",
    "markov_sample",
)

# (module, class, replacement symbol, registration kind, vendor operator).
# Small accept/causal kernels and scatter were slower through their current
# wrappers on XPU. Commit layout copies whole token-table rows; leave it alone.
# No draft-model, sampling or Markov execute method is changed here.
_VENDOR_ROUTES = (
    ("dspark_attn_metadata", "ComputeDsparkWindowGather",
     "_window_gather_kernel", "triton", "window_gather_kernel"),
    ("dspark_attn_metadata", "BuildDsparkSwaPageIndices",
     "_swa_page_indices_kernel", "triton", "swa_page_indices_kernel"),
    # 0.5.23 deleted `CompactRowIndex` and `CompactVerifyIds` as dispatch
    # classes and folded both into `BuildRaggedVerifyWindow`, whose `.triton`
    # now calls `compact_row_index_triton` *and* `compact_verify_ids_triton`
    # in one path (`dspark_verify_window.py:186,218`). The two routes they used
    # to enable are therefore **not** carried over: this file's rule is that a
    # class may enter `cls.triton` only when its *complete* kernel path has
    # verified Kunlun replacements, and the merged class covers a wider path
    # than either of the originals did. `BuildRaggedVerifyWindow` stays on
    # Torch, which is this module's documented default. Re-enabling it is a
    # perf change that needs its own measurement, not a migration edit.
)


def _inputs_not_on_cuda(*args, **kwargs) -> bool:
    return False


def _supported(class_name, kwargs):
    tensors = [value for value in kwargs.values() if isinstance(value, torch.Tensor)]
    if not tensors or not all(value.is_cuda for value in tensors):
        return False
    if class_name == "BuildDsparkSwaPageIndices":
        # Avoid copying the whole int64 request table. The live request pool is
        # int32, while the SWA allocator's mapping is int64.
        return (
            kwargs["req_to_token"].dtype == torch.int32
            and kwargs["req_to_token"].stride(1) == 1
            and kwargs["full_to_swa_mapping"].dtype in (torch.int32, torch.int64)
        )
    return True


def _vendor_execute(module, class_name, symbol, kind, vendor_op):
    def execute(cls, *args, **kwargs):
        # These upstream APIs are keyword-only. Leave invalid/positional calls
        # to the reference so it retains its argument validation.
        if not args and _supported(class_name, kwargs):
            from sglang_kunlun.kernels import kernel_ops

            key = (module.__name__, symbol)
            actual = getattr(module, symbol, None)
            if kind == "triton":
                spec = kernel_ops._TRITON_OPS.get(key)
                ready = (
                    spec is not None
                    and isinstance(actual, kernel_ops.KernelLauncher)
                    and actual.impl is spec.impl
                )
            else:
                spec = kernel_ops._JIT_OPS.get(key)
                ready = spec is not None and actual is spec.impl
            if ready and hasattr(torch.ops.xspeedgate_ops, vendor_op):
                return cls.triton(**kwargs)
        return cls.torch(*args, **kwargs)

    execute.__name__ = "execute"
    execute.__qualname__ = f"{class_name}.execute"
    execute._kunlun_vendor_route = True
    return classmethod(execute)


def install() -> None:
    """Pin unverified routes and install the explicit vendor class whitelist."""
    if os.environ.get("DSV4_DSPARK_TRITON") == "1":
        logger.info("DSPARK: keeping the Triton kernels (DSV4_DSPARK_TRITON=1)")
        return
    modules = {}
    for name in _MODULES:
        module = importlib.import_module(f"sglang.kernels.ops.speculative.dspark.{name}")
        modules[name] = module
        if hasattr(module, "inputs_on_cuda"):
            module.inputs_on_cuda = _inputs_not_on_cuda
    for name, class_name, symbol, kind, vendor_op in _VENDOR_ROUTES:
        module = modules[name]
        # A vendor route is an optional *performance* opt-in, so an upstream
        # rename must cost the route, not the plugin: `getattr` raising here
        # propagates out of `load_plugins` and the server comes up with no
        # Kunlun hooks at all. 0.5.23 deleting `CompactRowIndex` did exactly
        # that -- `AttributeError: module ... has no attribute
        # 'CompactRowIndex'` during plugin import.
        cls = getattr(module, class_name, None)
        if cls is None:
            logger.warning(
                "DSPARK: vendor route %s.%s is gone upstream; staying on Torch",
                name, class_name,
            )
            continue
        if not getattr(cls.execute, "_kunlun_vendor_route", False):
            cls.execute = _vendor_execute(module, class_name, symbol, kind, vendor_op)
    logger.info("DSPARK: Torch default; guarded Kunlun routes: %s",
                ", ".join(route[1] for route in _VENDOR_ROUTES))


install()
