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
"""Kunlun pre-import shim for CUDA-compatible runtime behavior."""

from __future__ import annotations

import threading


_FLOAT16_KV_PATCH_LOCK = threading.Lock()
_GRAMMAR_BACKEND_PATCH_LOCK = threading.Lock()


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
    import os
    import sys

    # The Kunlun launcher explicitly disables flashinfer.  Do not even touch
    # its package in that mode: importing it initializes tvm_ffi and may
    # synchronously build/load a CUDA extension.
    if os.environ.get("SGLANG_IS_FLASHINFER_AVAILABLE", "").lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return

    symbols = [
        ("flashinfer.prefill", "cudnn_batch_prefill_with_kv_cache"),
        ("flashinfer.gemm", "mm_M1_16_K7168_N256"),
    ]
    for module_path, symbol_name in symbols:
        # Importing flashinfer here eagerly initializes its JIT/cubin loader
        # (via tvm_ffi). On Kunlun this can block while it probes or builds a
        # CUDA extension. The symbols are optional and the caller already
        # advertises flashinfer as unavailable, so only patch modules that
        # have been loaded by the application.
        module = sys.modules.get(module_path)
        if module is None:
            continue
        if not hasattr(module, symbol_name):

            def _unsupported(*_a, _name=symbol_name, **_kw):
                raise RuntimeError(
                    f"flashinfer.{_name} is not supported by the Kunlun backend"
                )

            setattr(module, symbol_name, _unsupported)


def _disable_xgrammar_import() -> None:
    """Prevent optional xgrammar from loading its CUDA extension on Kunlun."""
    import importlib.machinery
    import importlib.util
    import sys
    import types
    from typing import Any

    # if "xgrammar" in sys.modules:
    #     return

    if importlib.util.find_spec("xgrammar") is not None:
        return

    module = types.ModuleType("xgrammar")
    module.__file__ = "<kunlun-xgrammar-disabled>"
    module.__path__ = []
    module.__spec__ = importlib.machinery.ModuleSpec(
        "xgrammar", loader=None, is_package=True
    )
    # These names are imported by optional function-call parsing paths.  The
    # actual xgrammar backend remains unavailable and is never selected.
    module.StructuralTag = Any
    module.get_model_structural_tag = None
    sys.modules["xgrammar"] = module


def _patch_torch_xpu_availability() -> None:
    """Make sglang's generic ``is_xpu()`` detect the Kunlun shim.

    sglang 0.5.x gates its XPU dist communicators on
    ``torch.xpu.is_available()`` (``parallel_state._is_xpu``). The Kunlun shim
    maps XPU onto ``torch.cuda`` and ships a ``torch.xpu`` stub that always
    reports unavailable, so ``_is_xpu`` ends up False and dist init takes the
    pynccl path -- loading the pip NVIDIA ``libnccl.so.2``, which fails with
    "NCCL error: unhandled cuda error" on XPU devices.

    The XPU surface sglang touches is small (is_available, device_count,
    empty_cache, synchronize, ...), so delegate the whole stub onto
    ``torch.cuda`` -- which IS the XPU under the Kunlun shim. Making
    ``is_available`` alone True is NOT enough: downstream probes then call the
    other ``torch.xpu`` APIs, which raise "Torch not compiled with XPU
    enabled".
    """
    import torch

    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return
    if getattr(xpu, "_kunlun_shim", False):
        return

    cuda = torch.cuda

    def _get_device_capability(*args, **kwargs):
        # torch.xpu returns {"version": "major.minor"}; torch.cuda returns a
        # (major, minor) tuple -- adapt so sglang's ["version"] indexing works.
        major, minor = cuda.get_device_capability(*args, **kwargs)
        return {"version": f"{major}.{minor}"}

    delegates = {
        "is_available": lambda: cuda.is_available(),
        "device_count": cuda.device_count,
        "current_device": cuda.current_device,
        "get_device_name": cuda.get_device_name,
        "get_device_capability": _get_device_capability,
        "get_device_properties": cuda.get_device_properties,
        "empty_cache": cuda.empty_cache,
        "synchronize": cuda.synchronize,
        "memory_allocated": cuda.memory_allocated,
        "max_memory_allocated": cuda.max_memory_allocated,
        "reset_peak_memory_stats": cuda.reset_peak_memory_stats,
        "mem_get_info": cuda.mem_get_info,
        "manual_seed_all": cuda.manual_seed_all,
        "set_device": cuda.set_device,
        "init": lambda: None,
        "is_initialized": lambda: cuda.is_initialized(),
    }
    for name, fn in delegates.items():
        setattr(xpu, name, fn)
    xpu._kunlun_shim = True


def _patch_sglang_capability_probe() -> None:
    """Keep ``sglang.srt.utils.common.get_device_capability`` CUDA-shaped.

    The stock implementation clobbers the capability with ``(None, None)``
    whenever ``torch.xpu`` reports available (real XPU carries no cc info) --
    which breaks fp8 gates ("None >= 9"). Under the Kunlun shim the device IS
    the cuda device, so the xpu branch must keep the parsed capability instead
    of nulling it.

    Two mechanisms, both idempotent:

    1. FILE PATCH (authoritative). The pod image routes sglang imports through
       ``xpytorch_import_hook``, and a plain ``setattr`` on the module does not
       survive it -- the effective code must be changed in the installed
       ``common.py`` itself. We locate the installed file and rewrite exactly
       the two lines after the xpu-branch marker comment: drop the
       ``major, minor = None, None`` reset and int()-cast the parsed strings
       instead. A ``_kunlun_capability_file_patch`` marker comment makes it
       idempotent and re-runnable after every container restart.
    2. setattr fallback (best effort). Applied only when the file patch could
       not run (e.g. read-only install), for environments without the import
       hook, where the setattr does take effect.
    """
    import sys

    # -- 1) file patch on the installed sglang ------------------------------
    try:
        import importlib.util

        spec = importlib.util.find_spec("sglang.srt.utils.common")
        if spec is not None and spec.origin and not spec.origin.startswith("<"):
            with open(spec.origin, "r") as f:
                src = f.read()
            if "_kunlun_capability_file_patch" not in src:
                marker = (
                    "# Currently XPU version does not contain capability "
                    "information."
                )
                idx = src.find(marker)
                if idx != -1:
                    marker_end = idx + len(marker)
                    after = src[marker_end:]
                    reset = "major, minor = None, None"
                    if after.lstrip().startswith(reset):
                        patched = src[:marker_end] + after.replace(
                            reset,
                            "major, minor = int(major), int(minor)"
                            "  # _kunlun_capability_file_patch",
                            1,
                        )
                        with open(spec.origin, "w") as f:
                            f.write(patched)
    except Exception:
        pass

    # -- 2) setattr fallback (module may not be imported yet / no hook) -----
    import sys as _sys

    mod = _sys.modules.get("sglang.srt.utils.common")
    if mod is None or getattr(mod, "_kunlun_capability_patched", False):
        return

    import torch

    def get_device_capability(device_id: int = 0):
        try:
            return torch.cuda.get_device_capability(device_id)
        except Exception:
            return None

    mod.get_device_capability = get_device_capability
    mod._kunlun_capability_patched = True
    pkg = _sys.modules.get("sglang.srt.utils")
    if pkg is not None and hasattr(pkg, "get_device_capability"):
        pkg.get_device_capability = get_device_capability



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
        _patch_torch_cuda_symm_mem_apis()
    except Exception:
        pass

    try:
        _disable_xgrammar_import()
    except Exception:
        pass

    try:
        _patch_torch_xpu_availability()
    except Exception:
        pass

    try:
        _patch_sglang_capability_probe()
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


def patch_grammar_backend_default() -> None:
    """Default ``grammar_backend`` to ``"none"`` on Kunlun.
    """
    from sglang.srt.server_args import ServerArgs

    with _GRAMMAR_BACKEND_PATCH_LOCK:
        if getattr(ServerArgs, "_kunlun_grammar_backend_patched", False):
            return

        original_handle_grammar_backend = ServerArgs._handle_grammar_backend

        def _handle_grammar_backend(self):
            if self.grammar_backend is None:
                self.grammar_backend = "none"
                return
            return original_handle_grammar_backend(self)

        ServerArgs._handle_grammar_backend = _handle_grammar_backend
        ServerArgs._kunlun_grammar_backend_patched = True
