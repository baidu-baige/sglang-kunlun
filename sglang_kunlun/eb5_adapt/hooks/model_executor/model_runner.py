# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Hooks for ``sglang.srt.model_executor.model_runner``.

Part of the float16 KV cache support for DeepSeek V4 on Kunlun (see
``sglang_kunlun/hooks/server_args.py`` for the full picture).

``ModelRunner.configure_kv_cache_dtype`` (model_runner.py:2337) maps the
``kv_cache_dtype`` server-arg string to a ``torch`` dtype and has no ``float16``
branch, so the value would fall through to the trailing else and raise. Resolve
it to ``torch.float16`` up front and leave every other dtype to the original.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.model_executor.model_runner.ModelRunner.configure_kv_cache_dtype",
    type=HookType.AROUND,
)
def configure_kv_cache_dtype_kunlun(original_fn, self):
    """Resolve ``float16`` / ``half`` to ``torch.float16``."""
    if self.server_args.kv_cache_dtype in ("float16", "half"):
        self.kv_cache_dtype = torch.float16
        # Newer sglang also keeps a resolved *string* on the runner (attention
        # backends read it instead of the process-global bag).  The stock body
        # sets it, so we must mirror it when short-circuiting.
        if hasattr(type(self), "kv_cache_dtype_str") or hasattr(self, "kv_cache_dtype_str"):
            self.kv_cache_dtype_str = "float16"
        else:
            try:
                self.kv_cache_dtype_str = "float16"
            except Exception:
                pass
        return
    return original_fn(self)


@plugin_hook(
    "sglang.srt.model_executor.model_runner.ModelRunner.load_model",
    type=HookType.AFTER,
)
def _repair_kernel_bindings(result, self):
    """Re-point stale kernel aliases once every module has been imported.

    sglang imports many kernel symbols with ``from m import f as g`` at module
    import time; a later patch of the defining module does not reach those
    bindings.  By the time the model is loaded all of them exist, so one sweep
    fixes them all.
    """
    try:
        from sglang_kunlun.eb5_adapt.kernels.kernel_ops import repair_imported_bindings

        repair_imported_bindings()
    except Exception:  # noqa: BLE001 - never block model loading
        import logging

        logging.getLogger(__name__).exception(
            "kernel_ops: binding repair failed"
        )


@plugin_hook(
    "sglang.srt.model_executor.model_runner.ModelRunner.__init__",
    type=HookType.AFTER,
)
def _repair_kernel_bindings_early(result, self, *args, **kwargs):
    """Second sweep: attention backends import their kernels during init."""
    try:
        from sglang_kunlun.eb5_adapt.kernels.kernel_ops import repair_imported_bindings

        repair_imported_bindings()
    except Exception:  # noqa: BLE001
        pass
