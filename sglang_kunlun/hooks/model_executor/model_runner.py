# Copyright 2026 Baidu, Inc.
# Licensed under the Apache License, Version 2.0.
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
    """Resolve ``float16`` / ``half`` to ``torch.float16``.

    [kunlun-adapt] 上游实现改从 get_model() config bag 读 dtype，并在解析后
    设置 ``self.kv_cache_dtype_str``（attention backend 直接读取）。float16 不在
    上游白名单内，因此用 bag.override(**kw) 作用域内把 bag 伪装成 bfloat16 走完整解析（上游
    已封印 config 直写），随后覆盖结果为 float16。
    """
    if self.server_args.kv_cache_dtype in ("float16", "half"):
        from sglang.srt.runtime_context import get_model

        bag = get_model()
        with bag.override(kv_cache_dtype="bfloat16"):
            original_fn(self)
        self.kv_cache_dtype = torch.float16
        self.kv_cache_dtype_str = "float16"
        return
    return original_fn(self)
