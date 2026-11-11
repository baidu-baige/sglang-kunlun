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
"""Kunlun-local copies of mem_cache modules that upstream sglang lacks.

The AttentionStore (asradix) secondary KV cache and its two sibling
modules are ported from aiak_sglang 1d855da5ee (luno-5195).
The engine image ships community sglang, whose lineage never carried
them, so the plugin vendors them instead of patching sglang in place.
"""
