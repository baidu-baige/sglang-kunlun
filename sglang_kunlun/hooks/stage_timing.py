"""Per-stage host timing for one decode ``run_batch``, reconciled with TPOT.

Why another probe. The two that existed could not be compared with the cell's
TPOT:

* ``_ev_flush``'s CUDA-event spans read 0.0 on this stack --
  ``torch.cuda.Event.elapsed_time`` is not implemented (verified standalone,
  2026-11-18), so the device split has to come from a profiler trace.
* ``[PROBE]``'s host stamps either measure enqueue only (no sync) or serialise
  the step (``SGLANG_KUNLUN_PROBE_SYNC=1``), and in both cases the parts do not
  add up to a step.

This one is deliberately a *host* instrument with one invariant, printed on
every line so it can be checked rather than assumed:

    wall  ==  run_batch  +  outside      (outside = scheduler loop, not us)
    run_batch  ==  sum(top-level segments)  +  resid

``wall`` is the distance between consecutive decode ``run_batch`` entries, i.e.
exactly the quantity the SLO's TPOT is derived from (``TPOT = wall /
tok_per_iter``). So a segment table that sums to ``run_batch`` is on the same
footing as TPOT, and ``resid`` says how much is unattributed.

The segments are nested, which is why they are printed in two groups: the
top-level ones sum to ``run_batch``, and the sub-segments break ``draft`` and
``run_eagle_verify`` down further. Device work lands wherever the host first
blocks on it -- in practice the graph replay and ``_copy_result_to_cpu`` -- so
read a span as "host time attributable to this call", not as device time.

Off unless ``SGLANG_KUNLUN_SEG=1``. ``SGLANG_KUNLUN_SEG_EVERY`` (default 50)
sets the print period.
"""

import os
import time

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_ON = os.environ.get("SGLANG_KUNLUN_SEG", "0") not in ("", "0")
_EVERY = int(os.environ.get("SGLANG_KUNLUN_SEG_EVERY", "50"))
# A decode step is never a second long; a longer gap is the server idling
# between workloads and one such sample ruins the mean.
_WALL_MAX_MS = float(os.environ.get("SGLANG_KUNLUN_SEG_WALL_MAX_MS", "1000"))

_SCHED = "sglang.srt.managers.scheduler.Scheduler"
_OVERLAP = "sglang.srt.managers.overlap_utils"
_METRICS = (
    "sglang.srt.managers.scheduler_components.metrics_reporter."
    "SchedulerMetricsReporter"
)
_EWC = "sglang.srt.speculative.eagle_worker_common"
_EW2 = "sglang.srt.speculative.eagle_worker_v2"
_DRAFT_RUNNER = (
    "sglang.srt.speculative.eagle_draft_cuda_graph_runner."
    "EAGLEDraftCudaGraphRunner"
)
_FULL_CG = (
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend."
    "FullCudaGraphBackend"
)

# Top-level segments must sum to run_batch; the rest nest inside draft and
# run_eagle_verify. Order is the order they run in.
_TOP = (
    "record_scheduler_active",
    "resolve_seq_lens_cpu",
    "resolve_forward_inputs",
    "draft",
    "run_eagle_verify",
    "draft_extend",
    "copy_result_to_cpu",
)
_SUB = (
    "prepare_for_draft",
    "draft_graph_execute",
    "build_eagle_verify_input",
    "eagle_prepare_for_verify",
    "graph_replay",
)

# Segments reported with their per-step call count (see `_report`).
_COUNTED = (
    "refresh_host_mirrors",
    "refresh_host_lengths_from_cpu",
    "init_forward_metadata",
)

_acc = {}
_cnt = {}
_n = 0
_bs_sum = 0
_wall_ms = 0.0
_wall_n = 0
_prev_t0 = None
_in_decode = False


def _add(name, dt_ms):
    if _in_decode:
        _acc[name] = _acc.get(name, 0.0) + dt_ms
        _cnt[name] = _cnt.get(name, 0) + 1


def _timed(name):
    """AROUND wrapper that charges its host time to ``name``."""

    def wrapper(original_fn, *args, **kwargs):
        if not _in_decode:
            return original_fn(*args, **kwargs)
        t = time.perf_counter()
        try:
            return original_fn(*args, **kwargs)
        finally:
            _add(name, (time.perf_counter() - t) * 1000.0)

    return wrapper


def _rank():
    try:
        import torch

        return torch.cuda.current_device()
    except Exception:  # noqa: BLE001 - diagnostics must never break a step
        return -1


def _report():
    n = _n
    if not n:
        return
    wall = _wall_ms / _wall_n if _wall_n else float("nan")
    rb = _acc.get("run_batch", 0.0) / n
    top = {k: _acc.get(k, 0.0) / n for k in _TOP}
    sub = {k: _acc.get(k, 0.0) / n for k in _SUB}
    # Segments whose CALL COUNT is the question, not just their time: the host
    # mirrors are refreshed either from the padded CPU lengths (cheap) or by
    # reading the device back (not), and which one runs is a property of the
    # graph layout, so the per-step call count has to be measured rather than
    # assumed. `n=` is calls per decode step.
    extra = " ".join(
        f"{k}={_acc[k] / n:.2f}(n={_cnt.get(k, 0) / n:.2f})"
        for k in _COUNTED
        if k in _acc
    )
    print(
        "[SEG] r=%d steps=%d bs=%.2f wall=%.2f run_batch=%.2f outside=%.2f "
        "resid=%.2f | %s | %s | %s"
        % (
            _rank(),
            n,
            _bs_sum / n,
            wall,
            rb,
            wall - rb,
            rb - sum(top.values()),
            " ".join(f"{k}={v:.2f}" for k, v in top.items()),
            " ".join(f"{k}={v:.2f}" for k, v in sub.items()),
            extra,
        ),
        flush=True,
    )


@plugin_hook(f"{_SCHED}.run_batch", type=HookType.AROUND)
def seg_run_batch(original_fn, self, batch, *args, **kwargs):
    """Time one decode step and own the step-to-step wall.

    Non-decode batches (prefill, idle) are passed straight through and do not
    contribute to any accumulator: mixing them in is what makes a step wall
    meaningless.
    """
    global _n, _bs_sum, _wall_ms, _wall_n, _prev_t0, _in_decode
    if not _ON:
        return original_fn(self, batch, *args, **kwargs)
    mode = getattr(batch, "forward_mode", None)
    lens = getattr(batch, "seq_lens", None)
    bs = int(lens.shape[0]) if lens is not None else 0
    decode = bool(bs) and mode is not None and mode.is_decode()
    if not decode:
        return original_fn(self, batch, *args, **kwargs)

    t0 = time.perf_counter()
    if _prev_t0 is not None:
        ms = (t0 - _prev_t0) * 1000.0
        if ms <= _WALL_MAX_MS:
            _wall_ms += ms
            _wall_n += 1
    _prev_t0 = t0
    _in_decode = True
    try:
        return original_fn(self, batch, *args, **kwargs)
    finally:
        _in_decode = False
        _acc["run_batch"] = _acc.get("run_batch", 0.0) + (
            time.perf_counter() - t0
        ) * 1000.0
        _bs_sum += bs
        _n += 1
        if _n % _EVERY == 0:
            _report()


# One AROUND per boundary, at exactly the granularity the TPOT work asked for.
# ``plugin_hook`` is a decorator, so each target gets its own named function
# built from ``_timed``; the registry applies REPLACE first, which means these
# wrap whatever the Kunlun hooks installed, not the untouched upstream body.
def _register(path, name):
    plugin_hook(path, type=HookType.AROUND)(_timed(name))


if _ON:
    _register(f"{_METRICS}.record_scheduler_active", "record_scheduler_active")
    _register(f"{_OVERLAP}.FutureMap.resolve_seq_lens_cpu", "resolve_seq_lens_cpu")
    _register(f"{_OVERLAP}.resolve_forward_inputs", "resolve_forward_inputs")
    _register(f"{_EW2}.EagleDraftWorker.draft", "draft")
    _register(f"{_EW2}.EagleDraftWorker._draft_extend_for_decode", "draft_extend")
    _register(f"{_SCHED}._copy_result_to_cpu", "copy_result_to_cpu")
    _register(f"{_EWC}.run_eagle_verify", "run_eagle_verify")
    _register(f"{_EWC}.prepare_for_draft", "prepare_for_draft")
    _register(f"{_EWC}.build_eagle_verify_input", "build_eagle_verify_input")
    _register(f"{_DRAFT_RUNNER}.execute", "draft_graph_execute")
    _register(
        "sglang.srt.speculative.eagle_utils.eagle_prepare_for_verify",
        "eagle_prepare_for_verify",
    )
    _register(f"{_FULL_CG}.replay", "graph_replay")
    # Which host-mirror path the backend actually takes, and how often. The
    # cheap one derives the lengths from the padded CPU input; the other reads
    # seven device tensors back. Measured in isolation at the two operating
    # shapes (`test/test_decode_host_segments.py`): 0.22-0.25 vs 0.37-0.39
    # ms/step, so the split matters only if the expensive one runs.
    _NSA = "sglang_kunlun.hooks.layers.attention.kunlun_nsa_backend.KunlunDSAMetadata"
    _register(f"{_NSA}.refresh_host_mirrors", "refresh_host_mirrors")
    _register(
        f"{_NSA}.refresh_host_lengths_from_cpu", "refresh_host_lengths_from_cpu"
    )
