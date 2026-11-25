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
"""Keep upstream's zero padding for per-token forward tensors.

Paged allocators reserve page 0 as the dummy write sink. Unified allocators
likewise pin virtual page 0 to physical page 0 and exclude it from allocation.
No scratch allocation, address translation, or padding hook is needed.

Keep this module importable for the model_executor package's hook imports.
"""
