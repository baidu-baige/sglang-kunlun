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
"""Route the flat ``dsv4_attn_metadata_kernels`` dispatch to torch on Kunlun.

``ExpandPrefillCausally`` / ``BuildPageTablePositions`` /
``BuildCausalSwaPageIndices`` each pick the fused triton kernel when
``_inputs_on_cuda(*args)`` is True and the torch reference otherwise. Kunlun XPU
tensors report ``.is_cuda == True`` (CUDA masquerade), so the triton path is
taken and fails to compile on the xpu triton backend
(``PassManager::run failed``). These are new upstream metadata kernels the
plugin does not otherwise replace; forcing the dispatch to False selects their
built-in torch reference implementation, which runs correctly on XPU.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.kernels.ops.attention.dsv4_attn_metadata_kernels._inputs_on_cuda",
    type=HookType.AROUND,
)
def _inputs_on_cuda_kunlun(original_fn, *args, **kwargs):
    return False
