"""Probe site dispatch shared by the debug modules.

Production code carries a single guarded one-liner per probe site::

    if _DEBUG:
        debug_attention.capture("attention.operator_begin", locals())

The registry below maps the site name to a handler that pulls the values it
needs out of the captured frame scope, so the production signature of a probe is
just its local variable names.  Probes that must observe both operator inputs
and outputs store their context under a group name (thread local, because one
frame is always handled by one thread).
"""

from __future__ import annotations

import threading
from typing import Callable, Mapping


class SiteRegistry:
    """Named probe sites of one debug module."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[Mapping[str, object]], None]] = {}
        self._state = threading.local()

    def site(self, name: str):
        """Register the handler of a probe site."""

        def decorator(function):
            if name in self._handlers:
                raise RuntimeError(f"duplicate probe site {name!r}")
            self._handlers[name] = function
            return function

        return decorator

    def capture(self, name: str, scope: Mapping[str, object]) -> None:
        """Run the handler of ``name`` with the captured frame scope."""
        handler = self._handlers.get(name)
        if handler is None:
            raise KeyError(f"unknown probe site {name!r}")
        handler(scope)

    def set_context(self, group: str, context) -> None:
        """Store the context of ``group`` for the current thread."""
        self._contexts()[group] = context

    def get_context(self, group: str):
        """Return the context stored for ``group``, or ``None``."""
        return self._contexts().get(group)

    def _contexts(self) -> dict:
        contexts = getattr(self._state, "contexts", None)
        if contexts is None:
            contexts = {}
            self._state.contexts = contexts
        return contexts

    @property
    def site_names(self) -> tuple[str, ...]:
        """Return the sorted names of every registered probe site."""
        return tuple(sorted(self._handlers))
