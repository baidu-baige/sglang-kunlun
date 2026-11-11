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
"""Graph-recordable greedy selection for the Kunlun DSpark Markov head."""

import torch


def sharded_greedy_step(bias, base_local, *, group, vocab_start, gather=None):
    """Select the smallest global ID attaining the maximum, with NaNs first.

    The upstream Triton reduction is not recorded by XMLIR's XPU graph. Its
    output then feeds a recorded Markov embedding with stale/uninitialized IDs.
    Use device torch operations for both reductions and retain integer IDs
    through the optional distributed reduction.
    """
    del gather  # The optional upstream float-bit transport is not needed here.
    assert bias.ndim == base_local.ndim == 2
    assert bias.shape[0] == base_local.shape[0]
    assert bias.shape[1] <= base_local.shape[1]
    rows, width = bias.shape
    invalid = 2147483647
    if width:
        scores = base_local[:, :width].float() + bias.float()
        # XMLIR argmax does not share torch's first-NaN convention. Select NaN
        # IDs explicitly, then reduce finite scores with the native argmax.
        nan = scores.isnan()
        first_nan = torch.where(
            nan, torch.arange(width, dtype=torch.int64, device=bias.device), invalid
        ).amin(dim=-1)
        local_index = torch.where(
            nan.any(dim=-1),
            first_nan,
            scores.masked_fill(nan, -float("inf")).argmax(dim=-1),
        ).unsqueeze(1)
        values = scores.gather(1, local_index).squeeze(1)
        indices = local_index.squeeze(1) + vocab_start
    else:
        values = torch.full((rows,), -float("inf"), device=bias.device)
        indices = torch.full(
            (rows,), invalid, dtype=torch.int64, device=bias.device
        )
    if group.world_size == 1:
        return indices
    values = group.all_gather(values.unsqueeze(0), dim=0)
    indices = group.all_gather(indices.unsqueeze(0), dim=0)
    valid = indices != invalid
    nan = values.isnan() & valid
    maximum = values.masked_fill(~valid | nan, -float("inf")).amax(dim=0)
    winners = torch.where(nan.any(dim=0), nan, valid & (values == maximum))
    return indices.masked_fill(~winners, invalid).amin(dim=0)
