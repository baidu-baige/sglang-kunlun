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
"""Register the ``kunlun`` attention backend factory.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/attention/attention_registry.py
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from sglang.srt.layers.attention.attention_registry import register_attention_backend
except Exception as e:  # pragma: no cover - upstream layout drift
    register_attention_backend = None
    logger.warning("Could not import register_attention_backend: %s", e)


if register_attention_backend is not None:
    @register_attention_backend("kunlun")
    def create_kunlun_attention_backend(runner):
        """Lazy-construct the Kunlun attention backend."""
        from .kunlun_backend import KunlunAttentionBackend

        return KunlunAttentionBackend(runner)

    @register_attention_backend("kunlun_dsv4")
    def create_kunlun_dsv4_attention_backend(runner):
        """Lazy-construct the Kunlun attention backend."""
        from .deepseek_v4_backend import KunlunDeepseekV4AttnBackend

        logger.info("Using KunlunDeepseekV4AttnBackend for dsv4 attention backend (kunlun).")
        return KunlunDeepseekV4AttnBackend(runner)

    logger.info("Registered 'kunlun' 'kunlun_dsv4' attention backend factory")