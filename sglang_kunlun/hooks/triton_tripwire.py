"""Turn any reach into a Triton kernel into a loud, named error.

On Kunlun a Triton kernel does not fail cleanly: it is queued, never retires,
and the next device-wide operation reports ``kl3_wait_for_noc_idle`` (dmesg) or
``error code=299, wait for noc idle timeout`` while every rank hangs in the
following collective. That costs a machine reboot per discovery.

With this guard installed, the first Triton launch raises immediately with the
module and kernel name, so a missing Kunlun replacement is named instead of
wedging the node. ``DSV4_TRITON_TRIPWIRE=0`` disables it (e.g. to measure what a
kernel would have done).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def install() -> None:
    """Install the guard on ``triton.runtime.jit.JITFunction.run``.

    Idempotent: a marker on the patched callable leaves an already-wrapped run
    alone, so calling this twice is harmless. Skips silently when Triton is
    absent or its layout has drifted -- this is a diagnostic aid and must never
    be the reason a server fails to boot.

    Returns:
        None.
    """
    if os.environ.get("DSV4_TRITON_TRIPWIRE", "1") != "1":
        return
    try:
        from triton.runtime.jit import JITFunction
    except Exception as exc:  # noqa: BLE001 - triton layout drift
        logger.info("Triton tripwire skipped: %s", exc)
        return
    if getattr(JITFunction.run, "_kunlun_tripwire", False):
        return

    original_run = JITFunction.run

    def run(self, *args, **kwargs):
        """Raise instead of launching, naming the kernel that reached the device.

        Replaces ``JITFunction.run`` so the failure happens at the call site,
        before the kernel is queued.

        Raises:
            RuntimeError: always. The message names the module and kernel and
                points at where a Kunlun replacement belongs.
        """
        fn = getattr(self, "fn", None)
        name = getattr(fn, "__name__", "?")
        module = getattr(fn, "__module__", "?")
        raise RuntimeError(
            "Kunlun: Triton kernel "
            f"{module}.{name} reached the device. It has no XPU implementation, "
            "so it would hang the queue (kl3_wait_for_noc_idle). Add a torch "
            "replacement in sglang_kunlun/kernels/kernel_ops.py or route the "
            "op to its reference path."
        )

    run._kunlun_tripwire = True
    JITFunction.run = run
    logger.info("Triton tripwire installed (DSV4_TRITON_TRIPWIRE=0 to disable)")


install()
