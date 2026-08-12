"""Kunlun bootstrap hook loaded automatically from PYTHONPATH."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


def _enable_kunlun_attention_choice(module) -> None:
    choices = getattr(module, "ATTENTION_BACKEND_CHOICES", None)
    if choices is not None:
        for backend in ("kunlun", "kunlun_compressed"):
            if backend not in choices:
                choices.append(backend)

    try:
        annotations = module.ServerArgs.__annotations__
        raw_annotation = annotations["kv_cache_dtype"]
        if isinstance(raw_annotation, str) and "'fp16'" not in raw_annotation:
            annotations["kv_cache_dtype"] = raw_annotation.replace(
                "'fp4_e2m1']", "'fp4_e2m1', 'fp16']"
            )
    except Exception:
        pass


def _enable_transformers_compatibility() -> None:
    import transformers
    from huggingface_hub import dataclasses as hub_dataclasses
    from huggingface_hub.errors import StrictDataclassDefinitionError
    from transformers import configuration_utils

    config_type = getattr(configuration_utils, "PretrainedConfig", None)
    if config_type is not None:
        if not hasattr(configuration_utils, "PreTrainedConfig"):
            configuration_utils.PreTrainedConfig = config_type
        if not hasattr(transformers, "PreTrainedConfig"):
            transformers.PreTrainedConfig = config_type

    original_strict = hub_dataclasses.strict
    if getattr(original_strict, "_sglang_kunlun_compatible", False):
        return

    def compatible_strict(cls=None, **kwargs):
        def decorate(target):
            try:
                return original_strict(target, **kwargs)
            except StrictDataclassDefinitionError:
                return target

        return decorate(cls) if cls is not None else decorate

    compatible_strict._sglang_kunlun_compatible = True
    hub_dataclasses.strict = compatible_strict


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


_startup_args = [*getattr(sys, "orig_argv", ()), *sys.argv]
_is_package_tool = any(
    arg == "pip"
    or arg.endswith("/pip")
    or arg.endswith("/pip3")
    or arg.endswith("/pip3.10")
    for arg in _startup_args
)

if (
    not _is_package_tool
    and (
        os.environ.get("SGLANG_PLATFORM") == "kunlun"
        or os.environ.get("SGLANG_USE_XPU") == "1"
    )
):
    try:
        _enable_transformers_compatibility()
        os.environ.setdefault("SGLANG_OPT_USE_TOPK_V2", "0")
        # SGLang 0.5.17 DeepSeek-V4 uses compressor_v2 as the public contract.
        # Keep an explicit user override, but never silently downgrade to v1.
        os.environ.setdefault("SGLANG_OPT_USE_COMPRESSOR_V2", "1")

        from sglang_kunlun import _kunlun_pre_shim

        _kunlun_pre_shim()
        if "sglang.srt.server_args" in sys.modules:
            _enable_kunlun_attention_choice(sys.modules["sglang.srt.server_args"])
        elif not any(isinstance(finder, _ServerArgsFinder) for finder in sys.meta_path):
            sys.meta_path.insert(0, _ServerArgsFinder())
    except Exception as exc:
        if os.environ.get("SGLANG_KUNLUN_SHIM_DEBUG"):
            print(f"[kunlun-shim] sitecustomize failed: {exc}", file=sys.stderr)
