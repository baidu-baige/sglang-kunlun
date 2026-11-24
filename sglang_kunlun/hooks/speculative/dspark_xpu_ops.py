"""Let the DSPARK ops whose Triton kernels we already replaced run on XPU again.

``dspark_torch_dispatch`` pins six DSPARK modules' ``inputs_on_cuda`` to False so the
unreplaced upstream Triton kernels never launch (they wedge the NOC). That is
module-wide, so it also buries the XPU kernels ``kernels/kernel_ops.py`` registers into
those same modules: ``_window_gather_kernel``, ``_swa_page_indices_kernel``,
``_block_seq_lens_casual_kernel``, ``_softmax_temp_kernel``, ``_mixed_accept_select_kernel``,
``_finalize_accept_lens_kernel`` (plus the two bonus-gather kernels) were dead code, and
the whole speculative step ran the torch reference kept for CPU unit tests. The decode
trace shows the cost: draft segment 31.86 ms/step (9.48x B300), 19.76 ms of it elementwise
across 2306 kernels/step.

This hook re-enables exactly the op classes whose ``triton`` wrapper launches *only*
replaced kernels, by overriding their ``execute`` to call ``cls.triton`` directly.
``inputs_on_cuda`` stays pinned, so any nested ``SomeOtherOp.execute`` inside those
wrappers (e.g. ``CapCorrectLen`` in the accept path, whose ``_cap_correct_len_kernel`` has
no XPU version) still takes the torch path.

Off by default. ERNIE55_DSPARK_XPU_OPS=1 enables every op; a comma-separated list of
class names enables just that subset, so a crash can be bisected per op.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)

#: ``(module, class)`` pairs whose ``*_triton`` wrapper launches only kernels that
#: ``kernels/kernel_ops.py`` replaces, and dispatches no nested op. Verified against
#: upstream 0.5.24. ``ScheduleVerifyLensTopk`` and ``MarkovGreedyStep`` are whole-helper
#: (``register_jit_op``) replacements, so their inner Triton kernels never launch.
_FORCED_XPU_OPS = (
    ("dspark_attn_metadata", "ComputeDsparkWindowGather"),
    ("dspark_attn_metadata", "BuildDsparkSwaPageIndices"),
    ("dspark_attn_metadata", "BuildBlockSeqLensCausal"),
    ("dspark_accept", "SoftmaxTemp"),
    ("dspark_accept", "SelectMixedAccept"),
    ("dspark_accept", "AcceptGreedy"),
    ("dspark_accept", "FinalizeAcceptLens"),
    ("dspark_accept", "CapCorrectLen"),
    ("dspark_verify_window", "ScatterCompactToStrided"),
    ("dspark_verify_window", "BuildCommitInjectLayout"),
    ("dspark_verify_window", "BuildOutTokens"),
    ("dspark_schedule", "ScheduleVerifyLensTopk"),
    ("dspark_draft_model", "MarkovGreedyStep"),
)

#: Deliberately not forced: ``dspark_verify_window.BuildRaggedVerifyWindow`` launches
#: ``_ragged_verify_window_kernel``, which has no Kunlun replacement, and
#: ``dspark_draft_model.CommitKvProj`` / ``markov_sample`` dispatch inline.


def _force_triton(cls) -> None:
    def execute(inner_cls, *args, **kwargs):
        return inner_cls.triton(*args, **kwargs)

    cls.execute = classmethod(execute)


def install() -> None:
    """"0"/unset off, "1"/"all" every op, otherwise a comma-separated class-name"""
    #: subset so a crash can be bisected without editing code.
    want = os.environ.get("ERNIE55_DSPARK_XPU_OPS", "0").strip()
    if want in ("0", ""):
        return
    only = None if want in ("1", "all") else {n.strip() for n in want.split(",") if n.strip()}
    forced = []
    for mod_name, cls_name in _FORCED_XPU_OPS:
        if only is not None and cls_name not in only:
            continue
        try:
            module = importlib.import_module(
                f"sglang.kernels.ops.speculative.dspark.{mod_name}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DSPARK xpu ops: cannot import %s: %r", mod_name, exc)
            continue
        cls = getattr(module, cls_name, None)
        if cls is None or not hasattr(cls, "triton"):
            logger.warning("DSPARK xpu ops: %s.%s missing", mod_name, cls_name)
            continue
        _force_triton(cls)
        forced.append(f"{mod_name}.{cls_name}")
    if only is not None:
        unknown = only - {c for _, c in _FORCED_XPU_OPS}
        if unknown:
            raise RuntimeError(
                "ERNIE55_DSPARK_XPU_OPS names unknown ops: %s" % ",".join(sorted(unknown))
            )
    print("[ERNIE55] dspark xpu ops forced: " + (", ".join(forced) or "none"), flush=True)


install()
