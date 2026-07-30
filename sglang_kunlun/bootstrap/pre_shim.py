"""Kunlun pre-import shim for CUDA-compatible runtime behavior."""

from __future__ import annotations


def _patch_fla_utils_if_loaded() -> None:
    import sys
    import torch

    module = sys.modules.get("sglang.srt.layers.attention.fla.utils")
    if module is None:
        return

    def get_available_device() -> str:
        return "cuda"

    def _check_platform() -> str:
        return "nvidia"

    module.get_available_device = get_available_device
    module._check_platform = _check_platform
    module.device = "cuda"
    module.device_torch_lib = torch.cuda
    module.device_platform = "nvidia"
    module.is_amd = False
    module.is_intel = False
    module.is_nvidia = True
    module.is_intel_alchemist = False
    module.is_nvidia_hopper = False



def _patch_torch_cuda_symm_mem_apis() -> None:
    """Provide no-op stubs for pynccl symmetric-memory CUDA private APIs.

    ``sglang.srt.distributed.device_communicators.pynccl_allocator`` does a
    top-level ``from torch.cuda.memory import (..., _cuda_beginAllocateCurrentThreadToPool,
    _cuda_endAllocateToPool, _cuda_releasePool)``. These private CUDA-graph
    mempool helpers only exist on newer upstream torch (2.6+); the Kunlun
    runtime ships an older torch (2.5.x) that lacks them, so the import crashes
    even though Kunlun never enables pynccl symmetric memory.

    We add no-op stubs so the import succeeds. They are only ever *called* from
    ``SymmetricMemoryContext`` (guarded by ``enable_symm_mem``, which Kunlun
    does not turn on), so these stubs never run on the hot path.
    """
    import torch

    mem = getattr(torch.cuda, "memory", None)
    if mem is None:
        return

    for name in (
        "_cuda_beginAllocateCurrentThreadToPool",
        "_cuda_endAllocateToPool",
        "_cuda_releasePool",
    ):
        if not hasattr(mem, name):

            def _unsupported(*_a, _name=name, **_kw):
                raise NotImplementedError(
                    f"torch.cuda.memory.{_name} is not available on the Kunlun "
                    "runtime; pynccl symmetric memory (enable_symm_mem) is "
                    "unsupported on XPU."
                )

            setattr(mem, name, _unsupported)

    # Mirror onto torch._C so the ``after_2_8_0`` branch in pynccl_allocator
    # (which calls torch._C._cuda_*) also resolves to a stub if reached.
    _C = getattr(torch, "_C", None)
    if _C is not None:
        for name in (
            "_cuda_beginAllocateCurrentThreadToPool",
            "_cuda_beginAllocateToPool",
            "_cuda_endAllocateToPool",
            "_cuda_endAllocateCurrentStreamToPool",
            "_cuda_releasePool",
        ):
            if not hasattr(_C, name):

                def _unsupported_c(*_a, _name=name, **_kw):
                    raise NotImplementedError(
                        f"torch._C.{_name} is not available on the Kunlun runtime."
                    )

                setattr(_C, name, _unsupported_c)


def _patch_optional_tilelang_import() -> None:
    """Convert missing TileLang shared-library failures into optional imports."""
    import builtins
    import functools

    current_import = builtins.__import__
    if getattr(current_import, "_kunlun_optional_tilelang", False):
        return

    @functools.wraps(current_import)
    def optional_tilelang_import(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            return current_import(name, globals, locals, fromlist, level)
        except OSError as error:
            if name == "tilelang" or name.startswith("tilelang."):
                raise ModuleNotFoundError(
                    f"Optional TileLang runtime is unavailable: {error}"
                ) from error
            raise

    optional_tilelang_import._kunlun_optional_tilelang = True
    builtins.__import__ = optional_tilelang_import


def _patch_flashinfer_optional_symbols() -> None:
    """Inject stubs for optional flashinfer symbols imported at sglang import time.

    Several sglang 0.5.14 modules do *top-level* ``from flashinfer.X import Y``
    inside their ``if _is_cuda:`` block (e.g.
    ``sglang/srt/layers/attention/vision.py`` imports
    ``cudnn_batch_prefill_with_kv_cache``). Because Kunlun masquerades as CUDA
    (``is_cuda()`` -> True), these imports run, but the Kunlun-side flashinfer
    build lacks the symbol, so the import crashes before any plugin hook (which
    is registered later via ``register_all()``) can patch it.

    We pre-inject no-op-raising stubs here so the top-level import succeeds.
    They only raise if actually *called* on the Kunlun path (which doesn't use
    these CUDA-only flashinfer kernels).
    """
    import importlib

    symbols = [
        ("flashinfer.prefill", "cudnn_batch_prefill_with_kv_cache"),
        ("flashinfer.gemm", "mm_M1_16_K7168_N256"),
    ]
    for module_path, symbol_name in symbols:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        if not hasattr(module, symbol_name):

            def _unsupported(*_a, _name=symbol_name, **_kw):
                raise RuntimeError(
                    f"flashinfer.{_name} is not supported by the Kunlun backend"
                )

            setattr(module, symbol_name, _unsupported)


def _kunlun_pre_shim() -> None:
    """Force triton's active driver to advertise ``backend == "cuda"``."""
    import os
    import sys

    is_kunlun = (
        "torch_xmlir" in sys.modules
        or "xpytorch_import_hook" in sys.modules
        or os.path.exists("/usr/local/xre")
    )
    if not is_kunlun:
        return

    try:
        _patch_optional_tilelang_import()
    except Exception:
        pass

    try:
        _patch_torch_cuda_symm_mem_apis()
    except Exception:
        pass

    try:
        _patch_flashinfer_optional_symbols()
    except Exception:
        pass

    try:
        from sglang_kunlun.bootstrap import sgl_kernel_stub
        sgl_kernel_stub.install()
    except Exception:
        pass

    try:
        import triton.runtime.driver as _driver_cfg
    except Exception:
        return

    _driver_mod = sys.modules.get("triton.runtime.driver")
    if _driver_mod is None or not hasattr(_driver_mod, "_create_driver"):
        return

    class _CudaTarget:
        backend = "cuda"

    def _patch_driver(drv):
        try:
            drv.get_current_target = lambda: _CudaTarget()
        except Exception:
            pass
        return drv

    try:
        if not getattr(_driver_mod, "_kunlun_create_wrapped", False):
            _orig_create = _driver_mod._create_driver

            def _wrapped_create():
                return _patch_driver(_orig_create())

            _driver_mod._create_driver = _wrapped_create
            _driver_mod._kunlun_create_wrapped = True

            try:
                active = _driver_cfg.active
                if active is not None and getattr(active, "_obj", None) is None:
                    active._init_fn = _wrapped_create
            except Exception:
                pass
    except Exception:
        pass

    try:
        active = _driver_cfg.active
        if active is not None:
            active.get_current_target = lambda: _CudaTarget()
            if getattr(active, "_obj", None) is not None:
                _patch_driver(active._obj)
    except Exception:
        pass

    try:
        import torch as _torch

        if hasattr(_torch, "cuda") and not getattr(_torch.cuda, "_kunlun_stubbed", False):
            _torch.cuda.get_device_name = lambda *_a, **_kw: "Kunlun-XPU"
            _torch.cuda.get_device_capability = lambda *_a, **_kw: (8, 0)
            _torch.cuda._kunlun_stubbed = True
    except Exception:
        pass

    try:
        _patch_fla_utils_if_loaded()
    except Exception:
        pass
