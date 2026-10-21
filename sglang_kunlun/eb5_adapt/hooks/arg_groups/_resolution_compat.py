# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Write ServerArgs fields across both sglang argument-resolution models.

sglang v0.5.x lets a hook simply assign ``server_args.field = value`` during
argument resolution.  The internal fork replaced that with a *declarative*
pipeline: ``ServerArgs.__setattr__`` raises once resolution starts, and a
decision must be recorded through
``sglang.srt.arg_groups.overrides.declare_resolution(server_args, source,
**fields)``, which stores it in the declaration stash that the config bags are
projected from.

``declare()`` below prefers the declarative API when it exists and falls back
to plain assignment, so the same hook body works on either release.
"""

from __future__ import annotations

_SOURCE = "sglang_kunlun"


def declare(server_args, **fields) -> None:
    """Record ``fields`` as resolved values on ``server_args``.

    Tries, in order: the declarative resolution API (internal fork, during
    resolution), the runtime override API (internal fork, after resolution),
    then plain assignment (sglang v0.5.x).
    """
    try:
        from sglang.srt.arg_groups.overrides import declare_resolution

        declare_resolution(server_args, _SOURCE, **fields)
        return
    except Exception:
        pass

    try:
        from sglang.srt.runtime_context import get_context

        get_context().override(_SOURCE, **fields)
        return
    except Exception:
        pass

    for name, value in fields.items():
        setattr(server_args, name, value)
