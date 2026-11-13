"""Kunlun hooks for ``sglang.srt.disaggregation.mooncake.conn``.

Kunlun RDMA needs MR *start* 2MiB-aligned, not just the length the package
``__init__`` rounds up: EB5's c128 compressed state buffers start at 1MiB
offsets, ``ibv_reg_mr`` still returns success, but the registered range and the
mooncake metadata disagree and transfers die.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from .. import ALIGNMENT_2M

logger = logging.getLogger(__name__)


def _register_region(engine, ptr, length):
    """Register one buffer with a 2MiB-aligned start and length.

    The start is rounded *down* and the clipped prefix added back to the length,
    so the registered range always covers the original buffer; mooncake
    addresses are absolute, so the semantics do not change.
    """
    p0 = int(ptr)
    base = p0 - (p0 % ALIGNMENT_2M)
    span = int(length) + (p0 - base)
    span = (span + ALIGNMENT_2M - 1) // ALIGNMENT_2M * ALIGNMENT_2M
    try:
        rc = engine.batch_register([base], [span])
        return 0 if (rc == 0 or rc is None) else 1
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "kunlun mooncake: register 0x%x len=%d failed: %s", base, span, exc
        )
        return 1


@plugin_hook(
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager."
    "register_buffer_to_engine",
    type=HookType.REPLACE,
)
def register_buffer_to_engine_kunlun(self):
    """Register each buffer on its own instead of as one batch.

    ``batch_register`` is all-or-nothing: one failing ``ibv_reg_mr`` rejects the
    batch, no KV segment reaches the metadata server, and the peer then cannot
    resolve the remote address (``cannot get mmapped addr`` -> ``Failed to send
    kv chunk``). One at a time isolates failures.
    """
    ok = fail = 0
    for ptr, length in self._registerable_regions():
        if _register_region(self.engine, ptr, length) == 0:
            ok += 1
        else:
            fail += 1
    logger.info("kunlun mooncake: per-buffer registration ok=%d fail=%d", ok, fail)
