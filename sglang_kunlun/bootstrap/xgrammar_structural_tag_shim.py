"""Backfill the 0.2.x ``xgrammar.structural_tag`` names on an older xgrammar.

0.5.23's function-call registry is unconditional::

    function_call_parser.py:42   from ...kimik3_detector import KimiK3Detector
    kimik3_detector.py:25        from ...kimik3_structural_tag import (...)
    kimik3_structural_tag.py:4   from xgrammar.structural_tag import (
                                     AnyTokensFormat, ExcludeTokenFormat,
                                     OptionalFormat, StarFormat, TokenFormat, ...)

Those five names arrived in xgrammar 0.2.x. On a host pinned to xgrammar 0.1.30
the import raises ``ImportError: cannot import name 'AnyTokensFormat'`` while
``import sglang`` is still running, so the server never starts -- even though
the names belong to Kimi-K3 tool-call structural tags, which no other model
uses.

Upgrading xgrammar is not always possible: 0.2.x ships a compiled
``libxgrammar_bindings.so`` built for its image's interpreter, so it cannot be
copied from a py3.12 image into a py3.10 one, and these hosts have no index.

So mirror what ``sgl_kernel_stub`` does for ``sgl_kernel``: define the missing
names as pydantic models of the same shape and inject them into the real
``xgrammar.structural_tag`` module before sglang imports it. A model that
actually builds a Kimi-K3 structural tag still gets a real object; if the 0.1.30
grammar compiler cannot consume it, that fails inside xgrammar with its own
error instead of as a missing attribute at import time.

Idempotent, and a no-op when xgrammar already provides the names.
"""

from __future__ import annotations

# The 0.2.x additions ``kimik3_structural_tag`` imports by name.
_REQUIRED = (
    "AnyTokensFormat",
    "ExcludeTokenFormat",
    "OptionalFormat",
    "StarFormat",
    "TokenFormat",
)


def install() -> bool:
    """Add any missing 0.2.x structural-tag names. True when the shim was used."""
    try:
        import xgrammar.structural_tag as st
    except Exception:
        return False

    missing = [name for name in _REQUIRED if not hasattr(st, name)]
    if not missing:
        return False

    try:
        from typing import List, Literal, Optional, Union

        from pydantic import BaseModel
    except Exception:
        return False

    # ``st.Format`` on 0.1.30 is an ``Annotated`` discriminated-union alias,
    # not a class, so it cannot be subclassed. Derive from the same pydantic
    # BaseModel the concrete formats use.
    base = getattr(st, "ConstStringFormat", None)
    base = base.__mro__[1] if isinstance(base, type) else BaseModel
    if not (isinstance(base, type) and issubclass(base, BaseModel)):
        base = BaseModel

    class AnyTokensFormat(base):  # type: ignore[misc, valid-type]
        """Matches any token sequence (xgrammar >= 0.2)."""

        type: Literal["any_tokens"] = "any_tokens"

    class TokenFormat(base):  # type: ignore[misc, valid-type]
        """Matches one specific token (xgrammar >= 0.2)."""

        type: Literal["token"] = "token"
        token: Union[int, str, None] = None

    class ExcludeTokenFormat(base):  # type: ignore[misc, valid-type]
        """Matches any token except the excluded ones (xgrammar >= 0.2)."""

        type: Literal["exclude_token"] = "exclude_token"
        tokens: Optional[List[Union[int, str]]] = None

    class OptionalFormat(base):  # type: ignore[misc, valid-type]
        """Zero-or-one of the wrapped format (xgrammar >= 0.2)."""

        type: Literal["optional"] = "optional"
        element: object = None

    class StarFormat(base):  # type: ignore[misc, valid-type]
        """Zero-or-more of the wrapped format (xgrammar >= 0.2)."""

        type: Literal["star"] = "star"
        element: object = None

    added = {
        "AnyTokensFormat": AnyTokensFormat,
        "TokenFormat": TokenFormat,
        "ExcludeTokenFormat": ExcludeTokenFormat,
        "OptionalFormat": OptionalFormat,
        "StarFormat": StarFormat,
    }
    for name in missing:
        setattr(st, name, added[name])
    return True
