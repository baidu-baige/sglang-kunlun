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
"""Kunlun overrides for deep_gemm helper symbols."""

from __future__ import annotations

import importlib
import logging
import sys
import types

import torch

from sglang_kunlun.kernels.kernel_ops import register_jit_op

logger = logging.getLogger(__name__)
_WARNED_PAGED_MQA_METADATA = False


def _get_or_create_deep_gemm_module():
    try:
        return importlib.import_module("deep_gemm")
    except ImportError:
        deep_gemm = types.ModuleType("deep_gemm")
        sys.modules["deep_gemm"] = deep_gemm
        return deep_gemm


def _register_deep_gemm_stub(symbol_name: str):
    def decorator(fn):
        deep_gemm = _get_or_create_deep_gemm_module()
        if not hasattr(deep_gemm, symbol_name):
            setattr(deep_gemm, symbol_name, fn)
        return register_jit_op("deep_gemm", symbol_name)(fn)

    return decorator


@_register_deep_gemm_stub("get_num_sms")
def get_num_sms() -> int:
    """Return the Kunlun SM count used by NSA indexer scheduling."""

    return 32


@_register_deep_gemm_stub("get_paged_mqa_logits_metadata")
def get_paged_mqa_logits_metadata(seqlens=None, *args, **kwargs):
    """Return placeholder paged MQA metadata until Kunlun provides an equivalent."""

    global _WARNED_PAGED_MQA_METADATA
    if not _WARNED_PAGED_MQA_METADATA:
        logger.warning(
            "deep_gemm.get_paged_mqa_logits_metadata is not supported on Kunlun; "
            "returning an empty placeholder tensor"
        )
        _WARNED_PAGED_MQA_METADATA = True
    device = seqlens.device if isinstance(seqlens, torch.Tensor) else None
    return torch.empty(0, dtype=torch.int32, device=device)
