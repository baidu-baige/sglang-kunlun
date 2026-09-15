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
"""Hooks for ``sglang.srt.speculative.multi_layer_eagle_worker_v2``.

Source: commit e6da00df4f7b583e208aab87dabb60c533b57431

Patch only the eager metadata-init behavior added in the patch version:
- ``MultiLayerEagleDraftWorker._draft_extend_for_decode`` should not skip
  attention backend init in eager mode.
- ``MultiLayerEagleWorkerV2.verify`` should skip attention backend init only
  when running with cuda graph.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.speculative.multi_layer_eagle_worker_v2.MultiLayerEagleDraftWorker._draft_extend_for_decode",
    type=HookType.AROUND,
)
def _draft_extend_for_decode_kunlun(original_fn, self, batch, batch_result):
    original_runner_forward = {
        idx: runner.forward for idx, runner in enumerate(self.draft_runner_list)
    }

    def make_patched_forward(original_forward):
        """Create a patched forward function that forces attention backend init.

        Args:
            original_forward: The original forward function to wrap.

        Returns:
            A patched forward function.
        """
        def patched_forward(*args, **kwargs):
            """Patched forward that ensures skip_attn_backend_init is False."""
            if not kwargs.get("skip_attn_backend_init", False):
                return original_forward(*args, **kwargs)
            kwargs = dict(kwargs)
            kwargs["skip_attn_backend_init"] = False
            return original_forward(*args, **kwargs)

        return patched_forward

    for runner in self.draft_runner_list:
        runner.forward = make_patched_forward(runner.forward)

    try:
        return original_fn(self, batch, batch_result)
    finally:
        for idx, runner in enumerate(self.draft_runner_list):
            runner.forward = original_runner_forward[idx]


@plugin_hook(
    "sglang.srt.speculative.multi_layer_eagle_worker_v2.MultiLayerEagleWorkerV2.verify",
    type=HookType.AROUND,
)
def verify_kunlun(original_fn, self, batch):
    """Patch verify to conditionally skip attention backend init.

    Args:
        original_fn: The original verify function.
        self: The worker instance.
        batch: The batch to verify.

    Returns:
        The result of the original verify function.
    """
    original_forward_batch_generation = self.target_worker.forward_batch_generation

    def patched_forward_batch_generation(*args, **kwargs):
        """Patched forward_batch_generation for verify phase."""
        if kwargs.get("is_verify"):
            kwargs = dict(kwargs)
            kwargs["skip_attn_backend_init"] = kwargs.get(
                "skip_attn_backend_init", False
            ) and bool(
                self.target_worker.model_runner.decode_cuda_graph_runner
                and kwargs.get("forward_batch") is not None
                and self.target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
                    kwargs["forward_batch"]
                )
            )
        return original_forward_batch_generation(*args, **kwargs)

    self.target_worker.forward_batch_generation = patched_forward_batch_generation
    try:
        return original_fn(self, batch)
    finally:
        self.target_worker.forward_batch_generation = original_forward_batch_generation
