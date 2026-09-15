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
"""Kunlun compatibility hooks for optional flashinfer helper symbols."""

from __future__ import annotations

import sys

from sglang_kunlun.kernels.kernel_ops import register_jit_op


_FLASHINFER_MISSING_SYMBOL_MSG = (
    "flashinfer helper {symbol_name} is not supported by the Kunlun backend"
)


def _unsupported_flashinfer_symbol(symbol_name: str):
    def unsupported(*args, **kwargs):
        raise RuntimeError(
            _FLASHINFER_MISSING_SYMBOL_MSG.format(symbol_name=symbol_name)
        )

    return unsupported


def _ensure_flashinfer_symbol(module_path: str, symbol_name: str) -> None:
    # Do not import flashinfer during Kunlun startup. Its package import
    # initializes tvm_ffi and may synchronously build/load a CUDA extension.
    module = sys.modules.get(module_path)
    if module is None:
        return
    if not hasattr(module, symbol_name):
        setattr(module, symbol_name, _unsupported_flashinfer_symbol(symbol_name))


_ensure_flashinfer_symbol("flashinfer.prefill", "cudnn_batch_prefill_with_kv_cache")
_ensure_flashinfer_symbol("flashinfer.gemm", "mm_M1_16_K7168_N256")


@register_jit_op("flashinfer.prefill", "cudnn_batch_prefill_with_kv_cache")
def cudnn_batch_prefill_with_kv_cache(*args, **kwargs):
    """Raise for unsupported flashinfer batch prefill helper on Kunlun."""
    raise RuntimeError(
        _FLASHINFER_MISSING_SYMBOL_MSG.format(
            symbol_name="cudnn_batch_prefill_with_kv_cache"
        )
    )


@register_jit_op("flashinfer.gemm", "mm_M1_16_K7168_N256")
def mm_M1_16_K7168_N256(*args, **kwargs):
    """Raise for unsupported flashinfer GEMM helper on Kunlun."""
    raise RuntimeError(
        _FLASHINFER_MISSING_SYMBOL_MSG.format(symbol_name="mm_M1_16_K7168_N256")
    )
