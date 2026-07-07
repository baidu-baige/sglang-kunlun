"""KunlunSRTPlatform — registered under entry_point ``kunlun``.

Activation contract (see ``sglang.srt.platforms._resolve_platform``):
  * ``activate()`` returns the fully-qualified class name as a string when
    the host machine has Kunlun hardware (i.e. ``torch_xmlir`` importable).
  * Returns ``None`` otherwise so the platform discovery falls back.

Phase 3 (Wave 1) scope:
  * ``KunlunDeviceMixin`` — device identity (``device_name="kunlun"``,
    ``device_type="cuda"``, ``PlatformEnum.OOT``).
  * ``KunlunSRTPlatform.__init__`` — runs ``_kunlun_pre_shim()`` and
    extends ``ATTENTION_BACKEND_CHOICES``.
  * ``apply_server_args_defaults`` — default ``page_size = 128``.
  * ``get_default_attention_backend`` — ``"flashattention"``.
  * ``get_mha_kv_pool_cls`` / ``get_mla_kv_pool_cls`` / ``get_nsa_kv_pool_cls``
    — return Kunlun KV-pool subclasses.
  * ``get_paged_allocator_cls`` — returns ``KunlunPagedTokenToKVPoolAllocator``.
"""

from __future__ import annotations

from sglang.srt.platforms.interface import SRTPlatform

from .device import KunlunDeviceMixin


class KunlunSRTPlatform(KunlunDeviceMixin, SRTPlatform):
    supported_quantization = ["int8"]

    def __init__(self) -> None:
        super().__init__()
        # Must run before any sglang.srt.layers.attention.fla import.
        from sglang_kunlun.bootstrap import _kunlun_pre_shim

        _kunlun_pre_shim()
        self._extend_attention_backend_choices()

    def init_backend(self):
        """Initialize Kunlun backend registrations."""
        super().init_backend()

    @staticmethod
    def _extend_attention_backend_choices() -> None:
        """Register the ``"kunlun"`` attention backend choice.

        Mirrors mimo: ``ATTENTION_BACKEND_CHOICES += ["kunlun"]``.
        """
        try:
            import sglang.srt.server_args as server_args

            choices = getattr(server_args, "ATTENTION_BACKEND_CHOICES", None)
            if choices is not None and "kunlun" not in choices:
                choices.append("kunlun")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply Kunlun-specific defaults.

        Replaces the old ``hooks/server_args.py`` page-size REPLACE hook.
        """
        if getattr(server_args, "page_size", None) is None:
            server_args.page_size = 128

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        return "kunlun"

    def get_draft_prefill_attention_backend_cls(self) -> type | None:
        """Return the Kunlun draft prefill attention backend class."""
        from sglang_kunlun.hooks.layers.attention.kunlun_backend import KunlunAttentionBackend

        return KunlunAttentionBackend

    def get_draft_decode_attention_backend_cls(self) -> type | None:
        """Return the Kunlun draft decode attention backend class."""
        from sglang_kunlun.hooks.layers.attention.kunlun_backend import (
            KunlunFlashAttentionMultiStepBackend,
        )

        return KunlunFlashAttentionMultiStepBackend

    def get_mha_kv_pool_cls(self) -> type:
        from sglang_kunlun.hooks.mem_cache.kunlun_pools import KunlunMHATokenToKVPool

        return KunlunMHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        from sglang_kunlun.hooks.mem_cache.kunlun_pools import KunlunMLATokenToKVPool

        return KunlunMLATokenToKVPool

    def get_dsa_kv_pool_cls(self) -> type:
        from sglang_kunlun.hooks.mem_cache.kunlun_pools import KunlunNSATokenToKVPool

        return KunlunNSATokenToKVPool

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        return DecodeCudaGraphRunner

    def get_paged_allocator_cls(self) -> type:
        from sglang_kunlun.hooks.mem_cache.kunlun_allocator import KunlunPagedTokenToKVPoolAllocator

        return KunlunPagedTokenToKVPoolAllocator

    def get_piecewise_backend_cls(self) -> type:
        from sglang.srt.compilation.backend import SGLangBackend

        return SGLangBackend


def activate() -> str | None:
    """Entry-point activator.

    Returns the fully-qualified class name string for ``KunlunSRTPlatform``
    when ``torch_xmlir`` is importable. Returns ``None`` on non-Kunlun
    machines so the platform discovery treats the plugin as unavailable.
    """
    try:
        import torch_xmlir  # noqa: F401
    except Exception:
        return None
    return "sglang_kunlun.platform:KunlunSRTPlatform"
