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

# mooncake's batch_register return codes we have to tell apart (seen in its own
# log via MC_LOG_DIR): -7 = "Transfer Engine does not support overlapped memory region", i.e. the range is already covered by an earlier MR, which is a no-op
# rather than a loss; -202 = ibv_reg_mr "Cannot allocate memory" for the
# 2MiB-expanded range, which the exact-range retry below recovers.
_MC_RC_OVERLAPPED = -7

logger = logging.getLogger(__name__)

# Ranges this process actually got registered, per engine: {id(engine): [(lo, hi)]}.
# Needed because the 2MiB round-down below makes neighbouring host buffers share a
# page, so the second one comes back as "overlapped" (-7). That is only harmless when
# the earlier MR really covers it -- see _covered().
_REGISTERED: dict = {}


def _covered(engine, lo: int, hi: int) -> bool:
    for r_lo, r_hi in _REGISTERED.get(id(engine), ()):
        if r_lo <= lo and hi <= r_hi:
            return True
    return False


def _remember(engine, lo: int, hi: int) -> None:
    _REGISTERED.setdefault(id(engine), []).append((lo, hi))


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
        if rc == 0 or rc is None:
            _remember(engine, base, base + span)
            return 0
        if rc == _MC_RC_OVERLAPPED:
            # KUNLUN_FIX(ERNIE55 1123): do not take -7 on faith. If an earlier MR
            # covers this buffer, it is a real no-op; if it does not, the buffer has
            # no resolvable segment and every peer write into it fails -- which is
            # how greedy draft "broke" the PD aux transfer: it shifts the host-side
            # allocation layout, so a metadata buffer lands in a 2MiB page whose
            # registration was swallowed as overlapped without covering it.
            if _covered(engine, p0, p0 + int(length)):
                return 0
            logger.warning(
                "kunlun mooncake: overlapped rc=-7 for ptr=0x%x len=%d is NOT covered "
                "by an earlier MR; retrying with the exact range",
                p0,
                int(length),
            )
    except Exception as exc:  # noqa: BLE001
        rc = exc
    # KUNLUN_FIX(ERNIE55 1121): when the 2MiB-expanded range is refused, retry with
    # the exact buffer range. A real-PD run on 1121 lost 7-9 regions per rank this
    # way (only an INFO summary was logged, so it was invisible): the lost ones are
    # the low-address host-side aux/state regions, and the prefill side then cannot
    # find a buffer for those addresses -> "Failed to get segment descriptor" plus a
    # full metadata dump on P, and "Failed to get kvcache from prefill instance, it
    # might be dead" on D, for every request whose KV lands there. The 2MiB
    # alignment is a device-memory requirement, so registering host memory with its
    # exact range is safe. Log both attempts: the rc tells us why the aligned one
    # was refused.
    try:
        rc2 = engine.batch_register([p0], [int(length)])
    except Exception as exc:  # noqa: BLE001
        rc2 = exc
    ok2 = rc2 == 0 or rc2 is None or rc2 == _MC_RC_OVERLAPPED
    if ok2:
        _remember(engine, p0, p0 + int(length))
    logger.warning(
        "kunlun mooncake: aligned register base=0x%x span=%d rc=%s -> exact "
        "ptr=0x%x len=%d rc=%s (%s)",
        base,
        span,
        rc,
        p0,
        int(length),
        rc2,
        "recovered" if ok2 else "still failed",
    )
    return 0 if ok2 else 1


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
