"""Stub-injection for ``sgl_kernel`` on Kunlun hosts.

Background
----------
The community ``sgl_kernel`` package's top-level ``__init__.py`` (149 LOC)
eagerly dlopens architecture-specific ``common_ops.<abi>.so`` and pulls in
~120 symbols from many submodules. On a Kunlun host these CUDA shared
libraries are unloadable (``libnvrtc.so.12: cannot open shared object``)
and even if loaded would target the wrong device.

Strategy
--------
Replace ``sgl_kernel`` in ``sys.modules`` with a *self-stubbing* package
**before** ``sglang/__init__.py:4 import sgl_kernel`` runs. The stub
satisfies all import-time names with module-like / callable proxies; any
*runtime* use of an un-hooked symbol raises ``NotImplementedError`` so we
get a loud, attributable failure rather than silent bad math.

Phase 3 hooks (e.g. ``layers/layernorm.py`` REPLACE on ``RMSNorm``)
intercept the call sites that matter; the stub only has to keep
import-time happy and shout if a non-hooked call slips through.

Usage
-----
``install()`` is invoked from ``_kunlun_pre_shim`` (see ``__init__.py``).
Idempotent. No-op if ``sgl_kernel`` already in ``sys.modules``.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class _StubModule(types.ModuleType):
    """A module that lazily auto-creates child submodules / attributes.

    Acts simultaneously as:
      - a Python *package*: ``__path__ = []`` so submodule imports route
        through ``__getattr__`` rather than ``ImportError``;
      - a *callable*: invoking it raises ``NotImplementedError`` with the
        full dotted name, so any kernel call that wasn't replaced by a
        ``@plugin_hook(REPLACE)`` is surfaced loudly;
      - a *namespace*: attribute access creates further ``_StubModule``
        children on demand and registers them in ``sys.modules``.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__: list[str] = []  # makes it a package

    def __getattr__(self, name: str) -> Any:
        # Avoid infinite recursion on dunder lookups Python uses internally.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        full = f"{self.__name__}.{name}"
        sub = sys.modules.get(full)
        if sub is None:
            sub = _StubModule(full)
            sys.modules[full] = sub
        # Cache as a real attribute so subsequent lookups skip __getattr__.
        object.__setattr__(self, name, sub)
        return sub

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{self.__name__}: not implemented in sgl_kernel kunlun stub. "
            "Add a @plugin_hook(REPLACE) in sglang_kunlun for the call site, "
            "or extend the stub with a real Kunlun-backed implementation."
        )


def _define_custom_ops() -> None:
    """Register sgl_kernel custom op schemas so @torch.library.register_fake works."""
    import torch

    lib = torch.library.Library("sgl_kernel", "DEF")
    lib.define(
        "moe_fused_gate(Tensor input, Tensor bias, int num_expert_group, "
        "int topk_group, int topk, int num_fused_shared_experts, "
        "float routed_scaling_factor, bool apply_routed_scaling_factor_on_output) -> (Tensor[])"
    )
    lib.define(
        "kimi_k2_moe_fused_gate(Tensor input, Tensor bias, int topk, bool renormalize, "
        "float routed_scaling_factor, bool apply_routed_scaling_factor_on_output) -> (Tensor[])"
    )
    # Keep lib alive to prevent deregistration
    _define_custom_ops._lib = lib  # type: ignore[attr-defined]


def install() -> None:
    """Install ``sgl_kernel`` and known submodules as stubs.

    Pre-registers the submodules sglang's source tree imports lazily
    (``sgl_kernel.flash_attn``, ``flash_mla``, ``sparse_flash_attn``,
    ``speculative``, ``scalar_type``, ``test_utils``) so that
    ``from sgl_kernel.flash_attn import X`` resolves via
    ``sys.modules`` directly, avoiding any filesystem search that
    might find the real (broken) sgl_kernel package.

    Idempotent: a second call is a no-op if our stub is already
    installed.
    """
    existing = sys.modules.get("sgl_kernel")
    if isinstance(existing, _StubModule):
        return  # already installed

    # If the real sgl_kernel was already imported (it shouldn't be at
    # .pth time), we leave it alone — patching after init is risky and
    # the real package would have already failed import.
    if existing is not None and not isinstance(existing, _StubModule):
        return

    root = _StubModule("sgl_kernel")
    sys.modules["sgl_kernel"] = root

    # Pre-register submodules sglang code imports under
    # ``from sgl_kernel.X import Y`` form. Listing them eagerly is not
    # strictly required — _StubModule.__getattr__ would lazily create
    # them — but pre-registration shields against odd import-system
    # corner cases where Python attempts to file-locate the submodule
    # despite the parent being a stub.
    for sub in (
        "flash_attn",
        "flash_mla",
        "sparse_flash_attn",
        "speculative",
        "scalar_type",
        "test_utils",
        "version",
        "elementwise",
        "attention",
        "allreduce",
        "moe",
        "gemm",
        "quantization",
        "sampling",
        "top_k",
        "memory",
        "kvcacheio",
        "marlin",
        "mamba",
        "hadamard",
        "fused_moe",
        "cutlass_moe",
        "expert_specialization",
        "grammar",
        "spatial",
    ):
        # Trigger lazy registration via attribute access.
        getattr(root, sub)

    # ``version.__version__`` is read at top-level of community
    # sglang_router or similar — provide a benign value.
    sys.modules["sgl_kernel.version"].__version__ = "0.0.0+kunlun-stub"  # type: ignore[attr-defined]

    # Bind real Kunlun implementations for sgl_kernel top-k APIs.
    # fast_topk: pure-Python fallback (no XPU kernel needed), identical to
    #   the mimo-branch reference in sgl-kernel/python/sgl_kernel/top_k.py.
    # fast_topk_v2 / fast_topk_transform_* rely on CUDA kernels
    #   (torch.ops.sgl_kernel.*) only available in the CUDA sgl_kernel build
    #   (NSA path). They are intentionally left as _StubModule callables and
    #   will raise NotImplementedError if called.
    from sglang_kunlun.kernels.sgl_kernel_kunlun.top_k import (
        fast_topk as _fast_topk,
        moe_fused_gate as _moe_fused_gate,
    )

    for module in (root, sys.modules["sgl_kernel.top_k"]):
        module.fast_topk = _fast_topk  # type: ignore[attr-defined]
        module.moe_fused_gate = _moe_fused_gate  # type: ignore[attr-defined]

    # Bind real Kunlun implementations for sgl_kernel sampling APIs used by
    # eagle_info_v2.py:sample() in the non-greedy (top-k/top-p) path.
    from sglang_kunlun.kernels.sgl_kernel_kunlun.sampling import (
        top_k_renorm_prob as _top_k_renorm_prob,
        top_p_renorm_prob as _top_p_renorm_prob,
        tree_speculative_sampling_target_only as _tree_spec_sampling,
    )

    for module in (root, sys.modules["sgl_kernel.sampling"]):
        module.top_k_renorm_prob = _top_k_renorm_prob  # type: ignore[attr-defined]
        module.top_p_renorm_prob = _top_p_renorm_prob  # type: ignore[attr-defined]
        module.tree_speculative_sampling_target_only = _tree_spec_sampling  # type: ignore[attr-defined]

    from sglang_kunlun.kernels.sgl_kernel_kunlun.speculative import (
        build_tree_kernel_efficient as _build_tree_kernel_efficient,
        verify_tree_greedy as _verify_tree_greedy,
    )

    for module in (root, sys.modules["sgl_kernel.speculative"]):
        module.build_tree_kernel_efficient = _build_tree_kernel_efficient  # type: ignore[attr-defined]
        module.verify_tree_greedy = _verify_tree_greedy  # type: ignore[attr-defined]

    from sglang_kunlun.kernels.sgl_kernel_kunlun.elementwise import (
        silu_and_mul as _silu_and_mul,
    )
    from sglang_kunlun.kernels.sgl_kernel_kunlun.layernorm import (
        fused_add_rmsnorm as _fused_add_rmsnorm,
        rmsnorm as _rmsnorm,
    )

    for module in (root, sys.modules["sgl_kernel.elementwise"]):
        module.fused_add_rmsnorm = _fused_add_rmsnorm  # type: ignore[attr-defined]
        module.rmsnorm = _rmsnorm  # type: ignore[attr-defined]
        module.silu_and_mul = _silu_and_mul  # type: ignore[attr-defined]

    # Register custom op schemas so @torch.library.register_fake succeeds.
    _define_custom_ops()
