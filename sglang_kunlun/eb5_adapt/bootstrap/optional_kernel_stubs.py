# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Stubs for optional CUDA-only kernel libraries on Kunlun.

Newer sglang (including the internal fork) imports ``tilelang`` and
``deep_gemm`` at module level from its kernel/attention modules, guarded only
by ``if is_cuda():``.  ``torch_xmlir`` advertises itself as CUDA, so those
imports *do* run on Kunlun even though neither library can work there.

* ``tilelang`` — emits CUDA/TVM kernels; never selected on Kunlun.  Needs a
  *permissive* stub because modules read attributes such as
  ``tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER`` at import time.
* ``deep_gemm`` — the real package is present but its C extension fails to
  load (``undefined symbol: cuLaunchKernelEx``).  Needs a *plain* stub: this
  plugin registers its own implementations for the few symbols it uses
  (``get_num_sms``, ``get_paged_mqa_logits_metadata``) via
  ``kernels/deep_geem_hook.py``, which relies on ``hasattr`` being False.

Both stubs are only installed when the real module is absent or unimportable.
"""


from __future__ import annotations

import sys
import types

_STUB_NAME = "tilelang"


class _StubAny:
    """Permissive placeholder: attribute access and calls never fail."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "tilelang"):
        self._name = name

    def __getattr__(self, item: str) -> "_StubAny":
        return _StubAny(f"{self._name}.{item}")

    def __call__(self, *args, **kwargs) -> object:
        # ``@tilelang.jit`` (bare) and ``@tilelang.jit(...)`` (factory) both end
        # up calling the stub with the decorated function as the sole argument.
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return _StubAny(f"{self._name}()")

    def __iter__(self):
        return iter(())

    def __repr__(self) -> str:
        return f"<tilelang-stub {self._name}>"


class _CallableModule(types.ModuleType):
    """A module that is also callable.

    ``tilelang.jit`` is used both as a decorator (``@tilelang.jit(...)``) and as
    a package (``from tilelang.jit.adapter.base import ...``), so it has to be
    both a module and callable.
    """

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return _StubAny(f"{self.__name__}()")


def _make_stub_module(name: str, cls=types.ModuleType) -> types.ModuleType:
    mod = cls(name)
    mod.__file__ = "<kunlun-tilelang-stub>"
    mod.__path__ = []  # mark as a package so ``import tilelang.language`` works
    mod.__getattr__ = lambda item: _StubAny(f"{name}.{item}")  # type: ignore[attr-defined]
    return mod


def install() -> bool:
    """Install the stub unless the real tilelang is importable.

    Returns True when a stub was installed.
    """
    if _STUB_NAME in sys.modules:
        return False
    try:
        import importlib.util

        if importlib.util.find_spec(_STUB_NAME) is not None:
            return False  # real tilelang present — leave it alone
    except (ImportError, ValueError):
        pass

    top = _make_stub_module(_STUB_NAME)
    sys.modules[_STUB_NAME] = top
    # ``jit`` must be callable (decorator) *and* a package (submodule imports).
    submodules = {
        "language": types.ModuleType,
        "math": types.ModuleType,
        "jit": _CallableModule,
        "autotuner": types.ModuleType,
        "intrinsics": types.ModuleType,
        "jit.adapter": types.ModuleType,
        "jit.adapter.base": types.ModuleType,
    }
    for sub, cls in submodules.items():
        full = f"{_STUB_NAME}.{sub}"
        submod = _make_stub_module(full, cls)
        sys.modules[full] = submod
        setattr(top, sub.split(".")[0], sys.modules[f"{_STUB_NAME}.{sub.split('.')[0]}"])
    top.PassConfigKey = _StubAny("tilelang.PassConfigKey")
    return True


def install_deep_gemm() -> bool:
    """Install a *plain* ``deep_gemm`` stub when the real one cannot load.

    Plain (no ``__getattr__`` magic) so that ``deep_geem_hook``'s
    ``hasattr``-guarded symbol registration still works.
    """
    import importlib.util

    if "deep_gemm" in sys.modules:
        return False
    try:
        mod = importlib.import_module("deep_gemm")
    except Exception:
        mod = None
    if mod is not None:
        return False  # real deep_gemm loaded fine

    stub = types.ModuleType("deep_gemm")
    stub.__file__ = "<kunlun-deep-gemm-stub>"
    stub.__path__ = []
    sys.modules["deep_gemm"] = stub
    return True


def install_all() -> None:
    install()
    install_deep_gemm()
