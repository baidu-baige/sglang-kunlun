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
"""DSV4 JIT-kernel hooks.

Importing the submodules declares their ``HookRegistry`` registrations; the
actual patching happens via ``HookRegistry.apply_hooks()``.
"""

from . import compress  # noqa: F401
from . import moe  # noqa: F401
from . import gemm  # noqa: F401
from . import attn  # noqa: F401
from . import compress_old  # noqa: F401
