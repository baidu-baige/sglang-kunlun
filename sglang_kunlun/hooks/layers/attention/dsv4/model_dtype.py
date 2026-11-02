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
"""Make DSV4 honor ``--dtype float16``.

Upstream hardcodes ``params_dtype=torch.bfloat16`` on four DSV4 submodules no
matter what ``--dtype`` says: ``Compressor.wkv_gate`` (compressor.py:377),
``C4Indexer.wq_b`` / ``C4Indexer.weights_proj`` (indexer.py:666/674) and
``MQALayer.wo_a`` (deepseek_v4.py:441, live only when
``SGLANG_OPT_FP8_WO_A_GEMM=False``, which is what the Kunlun deployment sets).
Under ``--dtype float16`` that leaves bf16 weights in an otherwise fp16 graph;
``MQALayer``'s ``einsum("tgd,grd->tgr", o, wo_a)`` mixes dtypes outright.

``params_dtype`` is a call argument (``wkv_gate_dtype`` is even a local), so it
cannot be hooked. Instead run the stock ``__init__`` and retype the parameters
it just created: they are still uninitialized ``torch.empty`` at this point, and
weight loading later goes through ``param.data.copy_(loaded_weight)`` which
casts — equivalent to having passed the dtype in.

Deliberately untouched:
  * fp32 params (``Compressor.ape``, the RMSNorm weights) — upstream wants fp32.
  * quantized params (int8 qweight + scales) — only bf16 tensors are converted.
  * the KV cache pack layout (``compressor.py:195`` bf16 pack /
    ``index_buf_accessor.py:78,221`` bf16 asserts) — that is a cache binary
    format, unrelated to parameter dtype.

The indexer projections are behind ``SGLANG_KUNLUN_DSV4_FP16_INDEXER=1``
(default off): the indexer runs at ``|q.k| ~ 127*127*D``, well outside fp16
range, and the deployment already forces ``SGLANG_KUNLUN_INDEXER_FP32=1``.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.hooks.server_args import _FLOAT16_ALIASES

logger = logging.getLogger(__name__)


def _fp16_model_dtype() -> torch.dtype | None:
    """``torch.float16`` when launched with ``--dtype float16``, else ``None``."""
    from sglang.srt.server_args import get_global_server_args

    if get_global_server_args().dtype in _FLOAT16_ALIASES:
        return torch.float16
    return None


def _retype_bf16(module: nn.Module, dtype: torch.dtype, tag: str) -> None:
    """In-place cast every bf16 parameter of ``module`` to ``dtype``."""
    for name, param in module.named_parameters(recurse=True):
        if param.dtype is torch.bfloat16:
            param.data = param.data.to(dtype)
            logger.debug("sglang-kunlun: retyped %s.%s -> %s", tag, name, dtype)


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.compressor.Compressor.__init__",
    type=HookType.AROUND,
)
def compressor_init_kunlun(original_fn, self, *args, **kwargs):
    """Retype ``Compressor.wkv_gate`` from hardcoded bf16 to the model dtype."""
    original_fn(self, *args, **kwargs)
    dtype = _fp16_model_dtype()
    if dtype is not None:
        _retype_bf16(self.wkv_gate, dtype, "Compressor.wkv_gate")


@plugin_hook(
    "sglang.srt.models.deepseek_v4.MQALayer.__init__",
    type=HookType.AROUND,
)
def mqa_layer_init_kunlun(original_fn, self, *args, **kwargs):
    """Retype ``MQALayer.wo_a`` so the wo_a einsum does not mix dtypes."""
    original_fn(self, *args, **kwargs)
    dtype = _fp16_model_dtype()
    if dtype is not None:
        _retype_bf16(self.wo_a, dtype, "MQALayer.wo_a")


@plugin_hook(
    "sglang.srt.layers.attention.dsv4.indexer.C4Indexer.__init__",
    type=HookType.AROUND,
)
def c4_indexer_init_kunlun(original_fn, self, *args, **kwargs):
    """Retype the indexer projections — opt-in, fp16 range is marginal here."""
    original_fn(self, *args, **kwargs)
    if os.environ.get("SGLANG_KUNLUN_DSV4_FP16_INDEXER") != "1":
        return
    dtype = _fp16_model_dtype()
    if dtype is not None:
        _retype_bf16(self.wq_b, dtype, "C4Indexer.wq_b")
        _retype_bf16(self.weights_proj, dtype, "C4Indexer.weights_proj")
