"""Force DSPARK's reference implementations instead of the Triton kernels.

Every op under ``sglang.kernels.ops.speculative.dspark`` dispatches with

    if inputs_on_cuda(*args, **kwargs): return cls.triton(...)
    return cls.torch(...)

and ``inputs_on_cuda`` is just ``tensor.is_cuda``. On Kunlun the XPU runtime
reports ``is_cuda == True`` while Triton is not usable, so the whole speculative
step ran through Triton: the first DSPARK request either aborted the XPU context
(``cuapi_xpu_wait``/``cuapiMemFree`` status=299, ``terminate called recursively``)
or stalled the device queue during warmup.

Pinning each module's ``inputs_on_cuda`` to False routes those ops to their torch
reference, which is the correctness baseline upstream keeps for CPU/unit tests.
``DSV4_DSPARK_TRITON=1`` opts back in for a platform where Triton works.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)

_MODULES = (
    "dspark_attn_metadata",
    "dspark_verify_window",
    "dspark_schedule",
    "dspark_accept",
    # 0.5.19's plugin replaced three Triton kernels of this module
    # (_online_partial_kernel / _online_combine_kernel / _build_step_local_kernel);
    # the 0.5.22 port dropped them, so its ops fell through to Triton and the kernel
    # never retired -- dmesg: `kl3_wait_for_noc_idle`, then the whole TP
    # all_reduce in the embedding hangs.
    "dspark_draft_model",
    "markov_sample",
)


def _inputs_not_on_cuda(*args, **kwargs) -> bool:  # noqa: ARG001
    """Report every input as host-resident so the dispatch takes the torch path.

    Each pinned module gates its kernel on ``inputs_on_cuda(*args)``. Returning
    False unconditionally sends the call to the reference implementation even
    when the tensors really are on device -- that is the point: the Triton
    kernels never launch, so the NOC never wedges.

    Returns:
        Always False.
    """
    return False


def install() -> None:
    """Pin the DSPARK draft modules to their torch reference implementations.

    Replaces each module's ``inputs_on_cuda`` with a function that always says
    False, which makes the dispatcher take the torch path. Idempotent by
    construction: re-assigning the same function is a no-op, and modules that
    do not expose ``inputs_on_cuda`` are skipped.

    ``DSV4_DSPARK_TRITON=1`` opts back into the Triton kernels on a platform
    where they retire correctly.

    Returns:
        None.
    """
    if os.environ.get("DSV4_DSPARK_TRITON") == "1":
        logger.info("DSPARK: keeping the Triton kernels (DSV4_DSPARK_TRITON=1)")
        return
    pinned = []
    for name in _MODULES:
        module = importlib.import_module(
            f"sglang.kernels.ops.speculative.dspark.{name}"
        )
        if hasattr(module, "inputs_on_cuda"):
            module.inputs_on_cuda = _inputs_not_on_cuda
            pinned.append(name)
    logger.info("DSPARK: pinned %d modules to the torch reference", len(pinned))


install()
