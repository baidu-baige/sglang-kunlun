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
"""sglang-kunlun: OOT SGLang platform plugin for Kunlun (XPU) chips."""

# 以下兼容层随 c11d9c8 一起迁入 eb5_adapt；调用顺序与上游实现保持一致。
from sglang_kunlun.eb5_adapt.bootstrap import compat_aliases as _compat_aliases

_compat_aliases.install()

from sglang_kunlun.eb5_adapt.bootstrap import optional_kernel_stubs as _optional_stubs

_optional_stubs.install_all()

from sglang_kunlun.eb5_adapt.bootstrap import _kunlun_pre_shim

__all__ = ["_kunlun_pre_shim"]
