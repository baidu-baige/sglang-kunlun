"""Hooks for ``sglang.srt.speculative.eagle_worker_v2``.

Source: commit e6da00df4f7b583e208aab87dabb60c533b57431

Patch only the eager metadata-init behavior added in the patch version:
- ``EagleDraftWorker._draft_extend_for_decode`` should not skip attention
  backend init in eager mode.
- ``EAGLEWorkerV2.verify`` should skip attention backend init only when
  running with cuda graph.
"""

from __future__ import annotations

import time
import os
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.__init__",
    type=HookType.AFTER,
)
def disable_mtp_index_share_kunlun(result, self, *args, **kwargs):
    """Diagnostic: force the multi-step draft to compute its own DSA top-k.

    0.5.22 dropped ``ps`` from ``EagleDraftWorker.__init__``
    (server_args, gpu_id, ps, nccl_port, target_worker ->
    server_args, gpu_id, nccl_port, target_worker). This hook stays correct
    without a change: it forwards the ctor arguments as ``*args/**kwargs`` and
    never indexes into them, only touching attributes on the constructed
    ``self``.
    """
    if os.environ.get("SGLANG_KUNLUN_DISABLE_MTP_INDEX_SHARE") != "1":
        return result
    if hasattr(self, "index_share_for_mtp_iteration"):
        self.index_share_for_mtp_iteration = False
    if hasattr(self, "seed_dsa_topk_from_draft_extend"):
        self.seed_dsa_topk_from_draft_extend = False
    return result


_DEW_ON = os.environ.get("SGLANG_KUNLUN_PROBE", "0") not in ("", "0")
_DEW_N = 0
_DEW_ACC = {"ms": 0.0}


def _dew_report(ms, bs):
    """Whole-method wrapper timing; prints the first call too, so a silent probe
    can never be mistaken for "the method did not run"."""
    _DEW_N_local = _DEW_ACC
    _DEW_N_local["n"] = _DEW_N_local.get("n", 0) + 1
    _DEW_N_local["ms"] += ms
    if _DEW_N_local["n"] == 1 or _DEW_N_local["n"] % 50 == 0:
        print(
            f"[PROBE_DEW] n={_DEW_N_local['n']} last={ms:.2f}ms "
            f"mean={_DEW_N_local['ms'] / _DEW_N_local['n']:.2f}ms bs={bs}",
            flush=True,
        )


@plugin_hook(
    "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker._draft_extend_for_decode",
    type=HookType.AROUND,
)
def _draft_extend_for_decode_kunlun(
    original_fn, self, batch, batch_result, *, kv_loc_plan=None
):
    # 0.5.22 added the keyword-only ``kv_loc_plan`` (the iteration's KVLocPlan,
    # built by the draft and passed on as ``verify_input.kv_loc_plan``); this
    # wrapper only patches ``skip_attn_backend_init``, so mirror the parameter
    # and hand it to the original untouched.
    original_forward = self.draft_runner.forward

    def patched_forward(*args, **kwargs):
        """Patched forward that ensures skip_attn_backend_init is False."""
        if not kwargs.get("skip_attn_backend_init", False):
            return original_forward(*args, **kwargs)
        kwargs = dict(kwargs)
        kwargs["skip_attn_backend_init"] = False
        return original_forward(*args, **kwargs)

    self.draft_runner.forward = patched_forward
    _t0 = time.perf_counter()
    try:
        return original_fn(self, batch, batch_result, kv_loc_plan=kv_loc_plan)
    finally:
        self.draft_runner.forward = original_forward
        if _DEW_ON:
            try:
                _bs = len(batch.seq_lens)
            except Exception:
                _bs = -1
            _dew_report((time.perf_counter() - _t0) * 1000, _bs)


@plugin_hook(
    "sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2.verify",
    type=HookType.AROUND,
)
def verify_kunlun(
    original_fn, self, batch, pp_proxy_tensors=None, grammar_barrier=None
):
    """Patch verify to conditionally skip attention backend init.

    0.5.22 inserted ``pp_proxy_tensors`` between ``batch`` and
    ``grammar_barrier`` (it is forwarded to the target worker's verify forward
    for PP). Mirror it in the same position and pass it through.

    Args:
        original_fn: The original verify function.
        self: The worker instance.
        batch: The batch to verify.
        pp_proxy_tensors: PP proxy tensors for the target verify forward.

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
        return original_fn(
            self,
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            grammar_barrier=grammar_barrier,
        )
    finally:
        self.target_worker.forward_batch_generation = original_forward_batch_generation
