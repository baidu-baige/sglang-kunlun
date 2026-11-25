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
"""Tolerate a Kunlun KV pool under the stock FlashInfer draft backend.

The DSpark draft worker only accepts a fixed set of attention backends and
falls back to ``flashinfer``. ``FlashInferAttnBackend.__init__`` then calls
``token_to_kv_pool.get_kv_cache_quant_method().resolve_attention_access(...)``
unconditionally. Under decode DP attention that getter returns ``None``, so
initialization dies with ``'NoneType' object has no attribute
'resolve_attention_access'``.

A float16 KV pool has no quant method by design and the class already treats a
``None`` access as "no rewrite" (both the access check and the dequant-workspace
flags are guarded). Supply a stub for the duration of the constructor only.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


class _NoQuantMethod(object):
    """Minimal stand-in for a KV-cache quant method that does nothing."""

    name = "none"

    def resolve_attention_access(self, phase, backend):
        """Report that no quantized attention access rewrite is required.

        Args:
            phase: Attention phase, accepted for interface compatibility.
            backend: Attention backend, accepted for interface compatibility.

        Returns:
            None because the underlying float16 KV pool is not quantized.
        """
        return None

    def describe_attention_accesses(self, phase=None):
        """Describe the non-quantized access supported by this stand-in.

        Args:
            phase: Optional attention phase; it does not affect the description.

        Returns:
            A description of the non-quantized float16 KV pool.
        """
        return "none (non-quantized float16 pool)"


@plugin_hook(
    "sglang.srt.layers.attention.flashinfer_backend.FlashInferAttnBackend.__init__",
    type=HookType.AROUND,
)
def flashinfer_init_kunlun(original_fn, self, model_runner, *args, **kwargs):
    """Skip quant-access resolution when the KV pool has no quant method."""
    pool = getattr(model_runner, "token_to_kv_pool", None)
    getter = getattr(pool, "get_kv_cache_quant_method", None)
    if getter is None or getter() is not None:
        return original_fn(self, model_runner, *args, **kwargs)

    logger.warning(
        "sglang-kunlun: FlashInfer backend on a non-quantized Kunlun KV pool; "
        "kv_cache_quant_method is None, skipping quant access resolution",
    )
    pool.get_kv_cache_quant_method = lambda: _NoQuantMethod()
    # With no quant method there is no access kind to validate; the stock check
    # would raise for the None access it just resolved.
    self._check_kv_attention_access = lambda phase, access: None
    try:
        return original_fn(self, model_runner, *args, **kwargs)
    finally:
        del pool.get_kv_cache_quant_method
        self.__dict__.pop("_check_kv_attention_access", None)
