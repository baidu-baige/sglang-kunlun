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
"""Kunlun bootstrap hook loaded automatically from PYTHONPATH."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


def _enable_kunlun_attention_choice(module) -> None:
    choices = getattr(module, "ATTENTION_BACKEND_CHOICES", None)
    if choices is not None:
        for backend in ("kunlun", "kunlun_dsv4"):
            if backend not in choices:
                choices.append(backend)
    

def _install_http_routes(module) -> None:
    """Attach the Kunlun /ready /health_forward /get_instance_info routes.

    Runs right after ``sglang.srt.entrypoints.http_server`` finishes executing,
    in *every* process that imports it — including the fresh uvicorn worker
    processes spawned for ``--tokenizer-worker-num N`` (which never run
    ``launch_server``/``load_plugins`` and therefore miss the plugin's
    launch_server hook).
    """
    try:
        from sglang_kunlun.hooks.entrypoints.http_server import install_routes

        install_routes(module)
    except Exception as exc:  # noqa: BLE001 - never block server import
        if os.environ.get("SGLANG_KUNLUN_SHIM_DEBUG"):
            print(f"[kunlun-shim] install_http_routes failed: {exc}", file=sys.stderr)


# Modules we intercept at import time → post-exec callback.
_POST_EXEC_HOOKS = {
    "sglang.srt.server_args": _enable_kunlun_attention_choice,
    "sglang.srt.entrypoints.http_server": _install_http_routes,
}


class _KunlunLoader(importlib.abc.Loader):
    def __init__(self, wrapped, on_exec):
        self._wrapped = wrapped
        self._on_exec = on_exec

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module) -> None:
        self._wrapped.exec_module(module)
        try:
            self._on_exec(module)
        except Exception:
            if os.environ.get("SGLANG_KUNLUN_SHIM_DEBUG"):
                import traceback

                traceback.print_exc()


class _KunlunFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        on_exec = _POST_EXEC_HOOKS.get(fullname)
        if on_exec is None:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            if not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _KunlunLoader(spec.loader, on_exec)
                return spec
        return None


_explicit_launch = os.environ.get("SGLANG_KUNLUN_EXPLICIT_LAUNCH") == "1"
if _explicit_launch:
    # multiprocessing ``spawn`` inherits SGLANG_PLATFORM from the explicit
    # launcher. Remove it before normal imports begin; importing torch from
    # sitecustomize is itself too early for torch_xmlir. ServerArgs is already
    # serialized into scheduler children, so they do not need these discovery
    # variables.
    os.environ.pop("SGLANG_PLATFORM", None)
    os.environ.pop("SGLANG_USE_XPU", None)

if os.environ.get("SGLANG_PLATFORM") == "kunlun" or os.environ.get("SGLANG_USE_XPU") == "1":
    try:
        from sglang_kunlun import _kunlun_pre_shim

        _kunlun_pre_shim()
        # server_args may already be imported; patch in place if so.
        if "sglang.srt.server_args" in sys.modules:
            _enable_kunlun_attention_choice(sys.modules["sglang.srt.server_args"])
        # http_server is normally imported later; if it is already present
        # (unusual), install routes now.
        if "sglang.srt.entrypoints.http_server" in sys.modules:
            _install_http_routes(sys.modules["sglang.srt.entrypoints.http_server"])
        if not any(isinstance(finder, _KunlunFinder) for finder in sys.meta_path):
            sys.meta_path.insert(0, _KunlunFinder())
    except Exception as exc:
        if os.environ.get("SGLANG_KUNLUN_SHIM_DEBUG"):
            print(f"[kunlun-shim] sitecustomize failed: {exc}", file=sys.stderr)
