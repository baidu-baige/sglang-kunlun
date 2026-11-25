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
"""Keep the ``full`` decode CUDA graph while the engram table stays in host HBM.

The host-table lookup is a CPU gather, which a ``full`` CUDA graph cannot
replay (see ``hooks/utils/engram_precompute.py``).  These hooks move the
hash + gather into a pre-replay pass and leave a pure device-side read in the
captured graph:

* ``DecodeCudaGraphRunner.capture_prepare`` keeps the capture-time
  ``ForwardBatch``.  Its tensor fields are views of the graph's static buffers
  (refreshed by ``load_batch`` each step) and its fixed metadata is exactly
  what the captured Python saw.
* ``FullCudaGraphBackend.capture_one`` binds the persistent buffers and flips
  the plugin into "read the precomputed buffer" mode for the warmups + capture.
* ``FullCudaGraphBackend.replay`` runs the eager precompute right before the
  graph replay, on the same stream.  Doing it in ``replay`` (not ``load_batch``)
  matters: it is after every ``load_batch`` post-hook -- including the scratch
  ``out_cache_loc`` padding in ``hooks.models.deepseek_v4_dspark`` -- so the
  hasher sees the same buffer state the replay does.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.hooks.utils import engram_precompute as ep

logger = logging.getLogger(__name__)

_DECODE_RUNNER = (
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
    "DecodeCudaGraphRunner"
)
_FULL_BACKEND = (
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend."
    "FullCudaGraphBackend"
)
_BREAKABLE_BACKEND = (
    "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend."
    "BreakableCudaGraphBackend"
)


def _is_target_decode_runner(runner) -> bool:
    if runner is None:
        return False
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    if not isinstance(runner, DecodeCudaGraphRunner):
        return False
    return not bool(getattr(runner.model_runner, "is_draft_worker", False))


def _resolve_capture_batch(runner):
    """(capture ForwardBatch, padded tokens) for the replay in flight."""
    if getattr(runner, "ragged_verify_mode", False):
        raise RuntimeError(
            "engram precompute does not support compact ragged verify; "
            "set SGLANG_RAGGED_VERIFY_MODE=static (the default)"
        )
    bs = int(runner.bs)
    num_tokens = bs * int(runner.captured_req_width)
    store = getattr(runner, "_kunlun_engram_capture_fbs", None) or {}
    batch = store.get((bs, num_tokens))
    if batch is None:
        raise RuntimeError(
            "engram precompute: no capture batch for "
            f"(bs, num_tokens)=({bs}, {num_tokens}); captured={sorted(store)}"
        )
    return batch, num_tokens


@plugin_hook(f"{_DECODE_RUNNER}.capture_prepare", type=HookType.AROUND)
def capture_prepare_store_batch(
    original_fn, self, size, stream_idx=None, num_tokens=None
):
    """Prepare capture inputs and retain the batch for Engram precompute.

    Args:
        original_fn: Wrapped capture preparation method.
        self: Decode graph runner receiving the hook.
        size: Batch size selected for capture.
        stream_idx: Optional capture stream index passed to the method.
        num_tokens: Optional padded token count for this capture.

    Returns:
        The original capture preparation result without modification.

    Raises:
        Exception: Propagates errors from the wrapped preparation method.
    """
    result = original_fn(self, size, stream_idx=stream_idx, num_tokens=num_tokens)
    if ep.enabled() and _is_target_decode_runner(self):
        padded = (
            int(num_tokens)
            if num_tokens is not None
            else int(size) * int(self.captured_req_width)
        )
        store = getattr(self, "_kunlun_engram_capture_fbs", None)
        if store is None:
            store = {}
            self._kunlun_engram_capture_fbs = store
        store[(int(size), padded)] = result[0]
    return result


@plugin_hook(f"{_FULL_BACKEND}.capture_one", type=HookType.AROUND)
def capture_one_precompute_mode(
    original_fn,
    self,
    shape_key,
    forward_fn,
    capture_inputs=None,
    post_warmup_hook=None,
):
    """Capture a full graph using persistent Engram embedding buffers.

    Args:
        original_fn: Wrapped full-backend capture method.
        self: Full CUDA graph backend receiving the hook.
        shape_key: Shape identifying the graph to capture.
        forward_fn: Model forward callable passed to the capture method.
        capture_inputs: Optional static inputs for capture.
        post_warmup_hook: Optional callback executed after warmup.

    Returns:
        The result returned by the wrapped capture method.

    Raises:
        RuntimeError: If Engram binding would invalidate persistent buffers.
        Exception: Propagates errors from the wrapped capture method.
    """
    runner = getattr(self, "_cuda_graph_runner", None)
    max_tokens = int(getattr(runner, "max_num_token", 0) or 0) if runner else 0
    if not (
        ep.enabled()
        and _is_target_decode_runner(runner)
        and max_tokens > 0
        and ep.bind(runner.model_runner.model, max_tokens)
    ):
        return original_fn(
            self,
            shape_key,
            forward_fn,
            capture_inputs=capture_inputs,
            post_warmup_hook=post_warmup_hook,
        )
    # The two warmups and the captured forward now read the persistent buffer
    # instead of touching the host table / running the hasher.
    with ep.active_scope():
        return original_fn(
            self,
            shape_key,
            forward_fn,
            capture_inputs=capture_inputs,
            post_warmup_hook=post_warmup_hook,
        )


@plugin_hook(f"{_FULL_BACKEND}.replay", type=HookType.AROUND)
def replay_precompute(original_fn, self, shape_key, static_forward_batch, **kwargs):
    """Refresh Engram embeddings before replaying the selected full graph.

    Args:
        original_fn: Wrapped full-backend replay method.
        self: Full CUDA graph backend receiving the hook.
        shape_key: Shape identifying the graph to replay.
        static_forward_batch: Static batch passed to the wrapped replay.
        **kwargs: Additional keyword arguments forwarded unchanged.

    Returns:
        The result returned by the wrapped replay method.

    Raises:
        RuntimeError: If the capture batch or Engram buffer is unavailable,
            inconsistent, or uses an unsupported ragged-verify layout.
        Exception: Propagates errors from precompute or the wrapped replay.
    """
    runner = getattr(self, "_cuda_graph_runner", None)
    if not (ep.enabled() and ep.is_bound() and _is_target_decode_runner(runner)):
        return original_fn(self, shape_key, static_forward_batch, **kwargs)

    batch, num_tokens = _resolve_capture_batch(runner)
    ep.precompute(batch, num_tokens)
    return original_fn(self, shape_key, static_forward_batch, **kwargs)


@plugin_hook(f"{_BREAKABLE_BACKEND}.capture_one", type=HookType.AROUND)
def reject_breakable_with_host_table(
    original_fn,
    self,
    shape_key,
    forward_fn,
    capture_inputs=None,
    post_warmup_hook=None,
):
    """Fail loudly instead of replaying a frozen engram table.

    The precompute above only serves the ``full`` backend.  Under breakable
    capture the host gather would be recorded into a segment and replayed as
    a constant (the ku doc's failure mode), so refuse the combination.
    """
    runner = getattr(self, "_cuda_graph_runner", None)
    if ep.enabled() and _is_target_decode_runner(runner):
        raise RuntimeError(
            "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 supports the 'full' "
            "decode CUDA graph only; got the breakable backend. Use "
            "--cuda-graph-backend-decode full (or unset the host table)."
        )
    return original_fn(
        self,
        shape_key,
        forward_fn,
        capture_inputs=capture_inputs,
        post_warmup_hook=post_warmup_hook,
    )
