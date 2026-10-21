# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Alias sglang module paths that were relocated between sglang releases.

``sglang-kunlun`` was written against the sglang v0.5.x layout, where DSV4 and
quantization kernels live under ``sglang.jit_kernel.dsv4`` /
``sglang.srt.layers.quantization``.  Newer sglang — including the internal fork
``baidu/wenxin/sglang`` (v1.3.x) — moved the very same modules under
``sglang.kernels.ops.*``.

Rather than rewriting every import in the plugin, register a lazy meta-path
finder that resolves the *old* dotted path to the *new* module object.  Only
paths that actually moved are listed, and the alias is a no-op when the old
path still exists (i.e. on v0.5.x), so one plugin source works on both.
"""

from __future__ import annotations

import importlib
import os
import importlib.abc
import importlib.util
import sys

# old dotted path -> new dotted path
_ALIASES = {
    # DSV4 / DSA attention kernels
    "sglang.jit_kernel.dsv4.compress":
        "sglang.kernels.ops.attention.dsv4.compress",
    "sglang.jit_kernel.dsv4.compress_old":
        "sglang.kernels.ops.attention.dsv4.compress_old",
    "sglang.srt.layers.attention.dsv4.dequant_k_cache":
        "sglang.kernels.ops.attention.dsv4.dequant_k_cache",
    "sglang.srt.layers.attention.dsv4.quant_k_cache":
        "sglang.kernels.ops.attention.dsv4.quant_k_cache",
    "sglang.srt.layers.attention.dsa.dequant_k_cache":
        "sglang.kernels.ops.attention.dsa.dequant_k_cache",
    "sglang.srt.layers.attention.dsa.quant_k_cache":
        "sglang.kernels.ops.attention.dsa.quant_k_cache",
    "sglang.srt.layers.attention.dsa.transform_index":
        "sglang.kernels.ops.attention.dsa.transform_index",
    "sglang.srt.layers.attention.dsa.index_buf_accessor":
        "sglang.kernels.ops.attention.dsa.index_buf_accessor",
    "sglang.srt.layers.attention.dsa.triton_kernel":
        "sglang.kernels.ops.attention.dsa.triton_kernel",
    "sglang.jit_kernel.flash_attention_v4":
        "sglang.kernels.ops.attention.flash_attention_v4",
    "sglang.srt.layers.attention.fla":
        "sglang.kernels.ops.attention.fla",
    "sglang.srt.layers.attention.fla.utils":
        "sglang.kernels.ops.attention.fla.utils",
    # Parent packages: a ``from a.b.c import X`` imports ``a.b`` first, so the
    # relocated package itself must resolve too.
    "sglang.jit_kernel.dsv4": "sglang.kernels.ops.attention.dsv4",
    "sglang.srt.speculative.triton_ops": "sglang.kernels.ops.speculative",
    "sglang.srt.mem_cache.triton_ops": "sglang.kernels.ops.memory",
    "sglang.srt.model_executor.triton_ops": "sglang.kernels.ops.attention",
    "sglang.srt.layers.attention.dsa": "sglang.kernels.ops.attention.dsa",
    # jit_kernel -> kernels.ops (remaining)
    "sglang.jit_kernel.kvcache": "sglang.kernels.ops.kvcache.kvcache",
    "sglang.jit_kernel.rope": "sglang.kernels.ops.attention.rope",
    "sglang.jit_kernel.hadamard": "sglang.kernels.ops.quantization.hadamard",
    "sglang.jit_kernel.dsv4.topk": "sglang.kernels.ops.attention.dsv4.topk",
    "sglang.jit_kernel.fused_metadata_copy": "sglang.kernels.ops.attention.fused_metadata_copy",
    # attention
    "sglang.srt.layers.attention.dsv4.metadata_kernel":
        "sglang.kernels.ops.attention.dsv4.metadata_kernel",
    "sglang.srt.layers.attention.utils": "sglang.kernels.ops.attention.utils",
    "sglang.srt.model_executor.triton_ops.position":
        "sglang.kernels.ops.attention.position",
    # speculative
    "sglang.srt.speculative.reject_sampling":
        "sglang.kernels.ops.speculative.reject_sampling",
    "sglang.srt.speculative.triton_ops.cache_locs":
        "sglang.kernels.ops.speculative.cache_locs",
    "sglang.srt.speculative.triton_ops.eagle":
        "sglang.kernels.ops.speculative.eagle",
    "sglang.srt.speculative.triton_ops.gather_spec_extras":
        "sglang.kernels.ops.speculative.gather_spec_extras",
    "sglang.srt.speculative.triton_ops.multi_layer_eagle":
        "sglang.kernels.ops.speculative.multi_layer_eagle",
    # quantization
    "sglang.srt.layers.quantization.fp8_kernel":
        "sglang.kernels.ops.quantization.fp8_kernel",
    "sglang.srt.layers.quantization.int8_kernel":
        "sglang.kernels.ops.quantization.int8_kernel",
    # memory
    "sglang.srt.mem_cache.triton_ops.allocator":
        "sglang.kernels.ops.memory.allocator",
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, module):
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):  # already executed under its real name
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        aliases = getattr(_AliasFinder, "aliases", _ALIASES)
        if os.environ.get("KUNLUN_ALIAS_DEBUG"):
            print(f"[alias] find_spec({fullname!r}) in_sys_modules={fullname in sys.modules}", file=sys.stderr)

        new_name = aliases.get(fullname)
        if new_name is None:
            # Prefix mapping: ``from a.b.c import X`` imports ``a.b`` first, and
            # the package machinery would then create a *separate* module object
            # for ``a.b.c`` under the old name — patching that object would not
            # affect the module the application imports via the new path.  Map
            # the whole subtree onto the relocated package instead.
            for old_prefix, new_prefix in aliases.items():
                if fullname.startswith(old_prefix + "."):
                    new_name = new_prefix + fullname[len(old_prefix):]
                    break
            else:
                return None
        try:
            module = importlib.import_module(new_name)
        except Exception:
            return None  # new path absent as well -> let the normal error surface
        sys.modules[fullname] = module  # share the very same object
        return importlib.util.spec_from_loader(fullname, _AliasLoader(module))


_installed = False


def install() -> None:
    """Idempotently append the alias finder to ``sys.meta_path``."""
    global _installed
    if _installed:
        return
    # Only alias paths that are genuinely gone; on v0.5.x nothing is aliased.
    active = {}
    for old, new in _ALIASES.items():
        try:
            if importlib.util.find_spec(old) is not None:
                continue  # old path still exists, leave it alone
        except (ImportError, AttributeError, ValueError):
            pass
        active[old] = new
    if active:
        _AliasFinder.aliases = active
        sys.meta_path.insert(0, _AliasFinder())  # must win over PathFinder
        for old_name, new_name in active.items():
            _bind_alias(old_name, new_name)
    _installed = True


def _ensure_parent(old_name: str):
    """Import (creating namespace stubs for) every parent package of a name."""
    import types as _types

    parts = old_name.split(".")
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        if pkg in sys.modules:
            continue
        try:
            importlib.import_module(pkg)
        except Exception:
            stub = _types.ModuleType(pkg)
            stub.__path__ = []
            stub.__file__ = "<kunlun-alias-stub>"
            sys.modules[pkg] = stub
            parent = sys.modules.get(".".join(parts[: i - 1]))
            if parent is not None:
                setattr(parent, parts[i - 1], stub)
    return sys.modules.get(".".join(parts[:-1]))


def _bind_alias(old_name: str, new_name: str) -> None:
    """Register ``old_name`` in ``sys.modules`` *and* on its parent package.

    ``importlib.import_module`` only needs the ``sys.modules`` entry, but the
    sglang hook registry resolves targets with ``pkgutil.resolve_name``, which
    imports the longest *existing* prefix and then ``getattr``s the rest — so
    the parent package must carry the attribute too.
    """
    try:
        module = importlib.import_module(new_name)
    except Exception:
        return
    sys.modules[old_name] = module
    parent = _ensure_parent(old_name)
    if parent is not None:
        setattr(parent, old_name.rsplit(".", 1)[1], module)
