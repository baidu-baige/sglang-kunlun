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
"""General plugin entry point for Kunlun hooks.

``register_all()`` imports each module that owns ``@plugin_hook(...)``
registrations. The order is intentional: foundational hooks and attention
module redirects must be registered before higher-level subsystem hooks.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

HOOK_MODULES = (
    "sglang_kunlun.hooks.utils.common",
    "sglang_kunlun.hooks.server_args",
    "sglang_kunlun.hooks.arg_groups",
    "sglang_kunlun.hooks.layers",
    "sglang_kunlun.hooks.mem_cache",
    "sglang_kunlun.hooks.model_executor",
    "sglang_kunlun.hooks.distributed",
    "sglang_kunlun.hooks.disaggregation",
    "sglang_kunlun.hooks.entrypoints",
    "sglang_kunlun.hooks.metrics",
    "sglang_kunlun.hooks.function_call",
    "sglang_kunlun.hooks.parser",
    "sglang_kunlun.hooks.jit_kernel",
    "sglang_kunlun.hooks.models",
    "sglang_kunlun.models",
    "sglang_kunlun.hooks.speculative",
    "sglang_kunlun.hooks.constrained",
)

def _preload_kunlun_runtime() -> None:
    """Preload ``torch_xmlir/libxpuapi.so`` with ``RTLD_GLOBAL``.

    ``xspeedgate_ops`` binds ``baidu::xpu::api::*`` symbols that only resolve
    when libxpuapi.so is already in the global symbol scope. Scheduler
    processes are spawned fresh, so shell-level preloads (LD_PRELOAD / a
    one-shot ctypes CDLL in the launcher) do not carry over -- do it here, at
    the top of ``register_all``, in every process that imports the plugin.
    """
    try:
        import ctypes
        import os

        import torch_xmlir

        lib = os.path.join(os.path.dirname(torch_xmlir.__file__), "libxpuapi.so")
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


def register_all() -> None:
    """Import every hook module so decorators register with ``HookRegistry``."""

    _preload_kunlun_runtime()

    # from sglang_kunlun.bootstrap.pre_shim import patch_grammar_backend_default

    # try:
    #     patch_grammar_backend_default()
    # except Exception as exc:
    #     logger.warning("sglang-kunlun: grammar backend default patch failed: %s", exc)

    try:
        from sglang_kunlun.kernels import deep_geem_hook as _deep_geem_hook  # noqa: F401
        from sglang_kunlun.kernels import flashinfer_hook as _flashinfer_hook  # noqa: F401
        from sglang_kunlun.kernels import kernel_ops

        kernel_ops.install()
    except Exception as exc:
        # A broken kernel backend (e.g. xspeedgate_ops built against a newer
        # baidu::xpu::api than the runtime provides) must not take the whole
        # plugin down -- the remaining hooks may still be valid.
        logger.warning("sglang-kunlun: kernel backend install failed: %s", exc)

    registered, failed = [], []
    for module_name in HOOK_MODULES:
        try:
            importlib.import_module(module_name)
            registered.append(module_name)
        except Exception as exc:
            # Same isolation principle: log, skip, keep the rest.
            failed.append(module_name)
            logger.warning(
                "sglang-kunlun: hook module %s failed to import, skipping: %s",
                module_name,
                exc,
            )

    # try:
    #     from sglang_kunlun.kernels.dsv4_torch_shim import patch_compress_old_prefill_plan
    #
    #     patch_compress_old_prefill_plan()
    # except Exception as exc:
    #     logger.warning("sglang-kunlun: compress_old prefill plan patch failed: %s", exc)

    # try:
    #     from sglang_kunlun.hooks.layers.attention.dsv4.compressor import (
    #         patch_compressor_online_state_slot_offset_kwarg,
    #     )
    #
    #     patch_compressor_online_state_slot_offset_kwarg()
    # except Exception as exc:
    #     logger.warning(
    #         "sglang-kunlun: create_paged_compressor_data online_state_slot_offset "
    #         "compat patch failed: %s",
    #         exc,
    #     )

    # try:
    #     from sglang_kunlun.kernels.dsv4_torch_shim import patch_compress_old_forward
    #
    #     patch_compress_old_forward()
    # except Exception as exc:
    #     logger.warning("sglang-kunlun: compress_old compress_forward patch failed: %s", exc)

    # try:
    #     from sglang_kunlun.kernels.dsv4_torch_shim import patch_compress_old_norm_rope
    #
    #     patch_compress_old_norm_rope()
    # except Exception as exc:
    #     logger.warning("sglang-kunlun: compress_old norm_rope patch failed: %s", exc)

    try:
        from sglang_kunlun.hooks.layers.attention.dsv4.compressor import (
            patch_compressor_forward_cuda_alias,
        )

        patch_compressor_forward_cuda_alias()
    except Exception as exc:
        logger.warning("sglang-kunlun: Compressor forward_cuda alias patch failed: %s", exc)

    try:
        from sglang_kunlun.kernels import kernel_ops as _kernel_ops

        triton_ops = len(_kernel_ops.registered_triton_ops())
        jit_ops = len(_kernel_ops.registered_jit_ops())
    except Exception:
        triton_ops = jit_ops = 0

    logger.info(
        "sglang-kunlun: %d/%d hook modules registered (%d skipped), "
        "%d triton ops and %d jit ops installed",
        len(registered),
        len(HOOK_MODULES),
        len(failed),
        triton_ops,
        jit_ops,
    )
