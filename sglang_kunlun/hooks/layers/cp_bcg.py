"""Let breakable prefill graphs replay batches that have no prefix hit.

``InterleaveBCGInput.allows_replay`` (upstream
``sglang/srt/layers/cp/bcg/interleave_bcg.py:82``) is

    batch_size > 0 and not contains_mm_inputs and prefix_lens is not None

so a prefill chunk with no cached tokens -- ``prefix_lens is None``, which is
what the scheduler hands over for a cold request -- always falls back to eager.
Measured on the 200-request SLO cell: only 7 % of prefill batches replay the
graph (9 ``cuda graph: True`` against 119 ``False``), and the misses are exactly
the ``#cached-token: 0`` batches. Bucket size is irrelevant (raising it to
16384 changed nothing), and DSV4-flash gets away with it because 0.5.17 routes
prefill CP through a different input type whose ``allows_replay`` has no such
clause.

The clause exists because the replay path reads the CP metadata built from
``prefix_lens``. Rather than dropping the requirement, synthesize the all-zero
list the caller would have produced for a cold batch, so every downstream
consumer still sees a real list of the right length. Gate it behind
``SGLANG_KUNLUN_BCG_ALLOW_COLD_REPLAY=1`` and keep it off by default until the
32-case accuracy gate has passed with it on.
"""

import logging
import os

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_KUNLUN_BCG_ALLOW_COLD_REPLAY", "0") not in (
    "",
    "0",
    "false",
    "False",
)

_installed = False


def _zero_prefix_lens(num_tokens, batch_size):
    """The list a cold batch would carry: no cached prefix on any row."""
    rows = row_count = None
    try:
        import torch

        rows = int(batch_size)
    except Exception:  # pragma: no cover - batch_size is always an int here
        return None
    del num_tokens
    if rows is None or rows <= 0:
        return None
    return [0] * rows


def install() -> None:
    """Wrap ``InterleaveBCGInput.allows_replay`` once, when enabled."""
    global _installed
    if _installed or not _ENABLED:
        return
    try:
        from sglang.srt.layers.cp.bcg import interleave_bcg
    except Exception:
        logger.warning(
            "sglang-kunlun: CP-BCG module missing; cold-prefix replay stays disabled"
        )
        return

    cls = getattr(interleave_bcg, "InterleaveCPBCGInput", None)
    if cls is None or not hasattr(cls, "allows_replay"):
        logger.warning(
            "sglang-kunlun: InterleaveBCGInput.allows_replay not found; "
            "cold-prefix replay stays disabled"
        )
        return

    original = cls.allows_replay
    if getattr(original, "_kunlun_cold_replay", False):
        return

    # Positional signature upstream: (self, batch_size, num_tokens, prefix_lens,
    # contains_mm_inputs). Accept keywords too, since the caller uses neither.
    def allows_replay(self, batch_size, num_tokens, prefix_lens, contains_mm_inputs):
        if prefix_lens is None and not contains_mm_inputs and batch_size > 0:
            prefix_lens = _zero_prefix_lens(num_tokens, batch_size)
        return original(self, batch_size, num_tokens, prefix_lens, contains_mm_inputs)

    allows_replay._kunlun_cold_replay = True
    cls.allows_replay = allows_replay
    _installed = True
    logger.info(
        "sglang-kunlun: breakable prefill graph will replay cold (no-prefix) "
        "batches; prefix_lens=None is materialised as zeros"
    )
