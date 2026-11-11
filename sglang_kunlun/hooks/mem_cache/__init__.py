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
"""mem_cache hooks."""
from . import common  # noqa: F401
from . import deepseek_v4_memory_pool  # noqa: F401
from . import deepseek_v4_compress_state  # noqa: F401

# AttentionStore (asradix) secondary KV cache. The pool hook imports first: the
# cache hook reaches into it for the pool-sizing stash.
from . import asradix_pool_hook  # noqa: F401
from . import asradix_cache_hook  # noqa: F401
from . import asradix_alloc_hook  # noqa: F401
