"""Kunlun bootstrap hook loaded automatically from PYTHONPATH."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


def _enable_kunlun_attention_choice(module) -> None:
    choices = getattr(module, "ATTENTION_BACKEND_CHOICES", None)
    if choices is not None and "kunlun" not in choices:
        choices.append("kunlun")


class _ServerArgsLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module) -> None:
        self._wrapped.exec_module(module)
        _enable_kunlun_attention_choice(module)


class _ServerArgsFinder(importlib.abc.MetaPathFinder):
    _target = "sglang.srt.server_args"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._target:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            if not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _ServerArgsLoader(spec.loader)
                return spec
        return None


if os.environ.get("SGLANG_PLATFORM") == "kunlun" or os.environ.get("SGLANG_USE_XPU") == "1":
    try:
        from sglang_kunlun import _kunlun_pre_shim

        _kunlun_pre_shim()
        if "sglang.srt.server_args" in sys.modules:
            _enable_kunlun_attention_choice(sys.modules["sglang.srt.server_args"])
        elif not any(isinstance(finder, _ServerArgsFinder) for finder in sys.meta_path):
            sys.meta_path.insert(0, _ServerArgsFinder())
    except Exception as exc:
        if os.environ.get("SGLANG_KUNLUN_SHIM_DEBUG"):
            print(f"[kunlun-shim] sitecustomize failed: {exc}", file=sys.stderr)
