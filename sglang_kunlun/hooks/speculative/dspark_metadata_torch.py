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
"""Register the Kunlun Torch implementation of DSpark SWA metadata.

The replacement preserves the upstream classmethod interface and forwards its
arguments directly to the fixed-shape Torch builder without upstream fallback.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang_kunlun.kernels.dspark_metadata_torch import build_dspark_swa_page_indices


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.dspark_attn_metadata."
    "BuildDsparkSwaPageIndices.execute",
    type=HookType.REPLACE,
)
def build_dspark_swa_page_indices_execute(cls, *args, **kwargs):
    """Forward the metadata execute interface to the Kunlun Torch builder.

    Args:
        cls (type): Upstream operation class supplied by the classmethod
            registry; unused by the Torch implementation.
        *args: Positional arguments forwarded unchanged. The current builder
            accepts keyword-only arguments, so this must be empty.
        **kwargs: Builder inputs: req_to_token, full_to_swa_mapping,
            req_pool_indices_per_request, offsets, invalid, out_loc,
            context_lens, block_size, swa_window and page_index_aligned_size.
            See build_dspark_swa_page_indices for their shapes and constraints.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Contiguous int32 SWA page indices
            shaped [batch * block_size, aligned_width] and per-token lengths
            shaped [batch * block_size], as returned by the builder.

    Raises:
        TypeError: Arguments do not match the keyword-only builder interface
            or a tensor has an unsupported dtype.
        ValueError: Builder settings, shapes or devices are invalid, or eager
            validation detects invalid active indices or context lengths.
    """
    # Registry preserves classmethod. Route directly to the torch implementation.
    return build_dspark_swa_page_indices(*args, **kwargs)
