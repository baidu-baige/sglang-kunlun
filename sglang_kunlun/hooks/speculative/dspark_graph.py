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
"""Kunlun DSpark graph options.

1) ``SGLANG_KUNLUN_SKIP_DRAFT_GRAPH=1`` is an explicit fallback that keeps
   the target decode graph and runs the draft step eagerly.
2) The eager-embed branch (``SGLANG_DSPARK_EMBED_IN_GRAPH=0``) hands a 2-D
   ``input_embeds`` to the draft model, while the HC stage path expects
   ``[tokens, hc_mult, dim]`` (``_hc_combine`` indexes ``x[:, 0, :]``).
   ``forward_embed`` performs that expansion on the in-graph path; restore it
   here for the eager path.
3) The folded Markov proposal needs a graph-recordable greedy selector. The
   upstream Triton launches execute eagerly under XMLIR capture, so use torch
   operations that replay with the current logits.
"""

import logging
import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


def _maybe_skip_draft_decode_graph() -> None:
    if os.environ.get("SGLANG_KUNLUN_SKIP_DRAFT_GRAPH") != "1":
        return

    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
        DSparkWorkerV2,
    )

    original = DSparkWorkerV2.init_cuda_graphs
    if getattr(original, "_kunlun_skip_draft_graph", False):
        return

    def init_cuda_graphs(self):
        logger.info(
            "sglang-kunlun: skip DSpark draft decode graph capture "
            "(SGLANG_KUNLUN_SKIP_DRAFT_GRAPH=1); target decode graph stays on"
        )
        # The framework's "no capture" path still initialises the draft
        # runner's eager/prefill runner attributes. Returning without it makes
        # the proposer read decode_cuda_graph_runner/prefill_cuda_graph_runner
        # that were never created.
        self._draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)

    init_cuda_graphs._kunlun_skip_draft_graph = True
    DSparkWorkerV2.init_cuda_graphs = init_cuda_graphs


@plugin_hook(
    "sglang.srt.models.deepseek_v4_dspark."
    "DeepseekV4ForCausalLMDSpark.forward",
    type=HookType.AROUND,
)
def _draft_forward_hc_embed(original_fn, self, *args, **kwargs):
    """Expand 2-D eager ``input_embeds`` into the HC stage layout."""
    def _expand(embeds):
        if (
            embeds is not None
            and embeds.dim() == 2
            and getattr(self, "hc_pre_from_prev_sublayer", False)
        ):
            return embeds.unsqueeze(1).repeat(1, int(self.hc_mult), 1)
        return embeds

    if "input_embeds" in kwargs:
        kwargs["input_embeds"] = _expand(kwargs["input_embeds"])
    elif len(args) >= 4:
        # (input_ids, positions, forward_batch, input_embeds, ...)
        args = args[:3] + (_expand(args[3]),) + args[4:]
    return original_fn(self, *args, **kwargs)


@plugin_hook(
    "sglang.kernels.ops.speculative.dspark.sharded_greedy.sharded_greedy_step",
    type=HookType.REPLACE,
)
def _draft_sharded_greedy(bias, base_local, *, group, vocab_start, gather=None):
    from sglang_kunlun.kernels.dspark_greedy_torch import sharded_greedy_step

    return sharded_greedy_step(
        bias, base_local, group=group, vocab_start=vocab_start, gather=gather
    )


_maybe_skip_draft_decode_graph()
