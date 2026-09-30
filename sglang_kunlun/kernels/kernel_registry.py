"""Kunlun overrides for the ``sglang.kernels`` registry/selector layer.

Why this exists
---------------
Kimi-K3 resolves a number of kernels through ``sglang.kernels`` instead of a
plain module symbol::

    # sglang/kernels/ops/mamba/__init__.py
    return get_kernel("mamba.causal_conv1d_fwd", KernelBackend.AOT)(...)

``kernel_ops`` cannot cover those: it patches *module symbols*, while these are
looked up by op id at call time. Worse, ``sglang.kernels.selector`` filters specs
by ``PlatformInfo.detect()``, and Kunlun deliberately reports ``device_type ==
"cuda"``, so the CUDA/AOT spec looks eligible and would be loaded.

Registering an extra Kunlun ``KernelSpec`` upstream is not an option either:
``DeviceType`` has no XPU member, so a Kunlun spec would have to claim CUDA and
``select_kernel`` would then abort with "multiple backends usable on device"
(``selector.py:80-84``).

So we override at the resolution boundary: ``get_kernel`` returns the Kunlun
implementation for ops we own, and delegates everything else upstream.

Ordering
--------
``selector.get_kernel`` caches resolved callables (``lru_cache`` on
``_resolve``), so :func:`install` must run before the first kernel call and
clears that cache defensively. It is invoked from ``hooks.registry.register_all``
ahead of ``kernel_ops.install()``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# (op id, backend value or None for "any backend")
OverrideKey = Tuple[str, Optional[str]]


@dataclass(frozen=True)
class KernelOverride:
    """A Kunlun implementation for one ``sglang.kernels`` op."""

    op: str
    backend: Optional[str]
    impl: Callable
    note: str = ""


_OVERRIDES: Dict[OverrideKey, KernelOverride] = {}

# Ops on the Kimi-K3 text-only serving path that are resolved through
# ``sglang.kernels``. Used by the startup self-check so an un-overridden CUDA
# resolution is visible in the log instead of failing deep inside a forward.
KIMI_K3_TRACKED_OPS: Tuple[str, ...] = (
    # KDA causal conv decode (69 layers)
    "mamba.causal_conv1d_update",
    # MoE routing / dispatch
    "moe.moe_align_block_size",
    "moe.topk_softmax",
    # sampling
    "sampling.top_k_renorm_probs",
    "sampling.top_p_renorm_probs",
    # KV cache writes
    "kvcache.reshape_and_cache_flash",
)


def register_kernel_override(
    op: str, backend: Optional[str] = None, *, note: str = ""
) -> Callable:
    """Register a Kunlun implementation for ``op``.

    ``backend=None`` overrides every backend request for that op, which is what
    Kunlun wants: upstream call sites hard-code ``KernelBackend.AOT`` /
    ``.TRITON`` and we replace the implementation regardless.
    """

    def decorator(fn: Callable) -> Callable:
        key: OverrideKey = (op, backend)
        if key in _OVERRIDES:
            raise ValueError(f"kernel override already registered: {key}")
        _OVERRIDES[key] = KernelOverride(op=op, backend=backend, impl=fn, note=note)
        return fn

    return decorator


def registered_overrides() -> Dict[OverrideKey, KernelOverride]:
    """Return a shallow copy of the registered Kunlun kernel overrides."""
    return dict(_OVERRIDES)


def _lookup(op: str, backend) -> Optional[KernelOverride]:
    backend_value = getattr(backend, "value", backend)
    return _OVERRIDES.get((op, backend_value)) or _OVERRIDES.get((op, None))


def install() -> None:
    """Route ``sglang.kernels.selector.get_kernel`` through the Kunlun overrides."""
    try:
        import sglang.kernels.selector as selector
    except Exception:
        logger.warning(
            "sglang.kernels.selector unavailable; kernel registry overrides skipped",
            exc_info=True,
        )
        return

    if getattr(selector, "_kunlun_override_installed", False):
        return

    upstream_get_kernel = selector.get_kernel

    def kunlun_get_kernel(op: str, backend=None) -> Callable:
        override = _lookup(op, backend)
        if override is not None:
            return override.impl
        return upstream_get_kernel(op, backend)

    kunlun_get_kernel.__name__ = "get_kernel"
    kunlun_get_kernel.__doc__ = upstream_get_kernel.__doc__
    kunlun_get_kernel._kunlun_upstream = upstream_get_kernel  # type: ignore[attr-defined]

    # Public wrappers do ``from sglang.kernels.selector import get_kernel``, so the
    # module attribute alone is not enough.
    from sglang_kunlun.kernels.kernel_ops import _replace_imported_bindings

    _replace_imported_bindings(upstream_get_kernel, kunlun_get_kernel)
    selector.get_kernel = kunlun_get_kernel
    selector._kunlun_override_installed = True

    # Anything resolved before we got here would keep pointing at the CUDA impl.
    try:
        selector.clear_cache()
    except Exception:
        logger.warning("selector.clear_cache() failed", exc_info=True)

    logger.info(
        "kernel_registry: installed %d Kunlun kernel override(s): %s",
        len(_OVERRIDES),
        sorted({op for op, _ in _OVERRIDES}),
    )


def resolution_report(ops: Tuple[str, ...] = KIMI_K3_TRACKED_OPS) -> List[str]:
    """Describe how ``ops`` would resolve, without importing any backend.

    Deliberately metadata-only: calling ``KernelSpec.load()`` here would import
    CUDA extensions on a Kunlun host, which is exactly what the shims avoid.
    """
    lines: List[str] = []
    try:
        from sglang.kernels.registry import registry
        from sglang.kernels.spec import PlatformInfo
    except Exception:
        return ["kernel registry unavailable"]

    platform = PlatformInfo.detect()
    for op in ops:
        override = _OVERRIDES.get((op, None))
        if override is None:
            override = next(
                (o for (o_op, _), o in _OVERRIDES.items() if o_op == op), None
            )
        specs = registry.get(op)
        backends = ",".join(sorted(s.backend.value for s in specs)) or "none"
        if override is not None:
            lines.append(f"{op}: kunlun override ({override.note or 'no note'})")
        elif not specs:
            lines.append(f"{op}: NOT REGISTERED upstream")
        else:
            eligible = [s for s in specs if s.is_available(platform)]
            targets = ",".join(sorted(s.backend.value for s in eligible)) or "none"
            lines.append(
                f"{op}: upstream (registered={backends}, "
                f"eligible_on_{platform.device.value}={targets})"
            )
    return lines


_REPORTED = False


def log_resolution_report_once() -> None:
    """Log the Kimi-K3 kernel resolution table once per process."""
    global _REPORTED
    if _REPORTED:
        return
    _REPORTED = True
    for line in resolution_report():
        logger.info("kernel_registry: %s", line)
