"""Install the shared MTP alignment probes when explicitly requested."""

from __future__ import annotations

import os


if os.environ.get("DSV4_ALIGNMENT_PROBE_DIR"):
    import mtp_alignment_hooks

    def _include_draft_extend(batch):
        mode = str(getattr(batch, "forward_mode", None))
        return mtp_alignment_hooks.has_generation_context() and mode in (
            "ForwardMode.EXTEND",
            "ForwardMode.DRAFT_EXTEND_V2",
        )

    mtp_alignment_hooks._is_extend = _include_draft_extend
    mtp_alignment_hooks.register_target()
