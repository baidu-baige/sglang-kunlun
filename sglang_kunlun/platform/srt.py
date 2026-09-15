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
    supported_quantization = ["int8", "w8a8_int8", "compressed-tensors"]

    def get_quantization_config(self, quantization: str):
        """返回昆仑自己的量化配置类。

        上游在 ``layers/quantization/__init__.py`` 里对 out-of-tree 平台会先问
        ``current_platform.get_quantization_config(quantization)``，返回 None 才回落到
        ``QUANTIZATION_METHODS``。不接这个口子的话 compressed-tensors 会走上游实现，
        W4A8/W8A8 的 int8 fused MoE 在那边只有 NPU 分支、直接抛 NotImplementedError。
        """
        if quantization == "compressed-tensors":
            from sglang_kunlun.hooks.layers.quantization.compressed_tensors import (
                KunlunCompressedTensorsConfig,
            )

            return KunlunCompressedTensorsConfig
        return None

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
        """Register the public Kunlun attention backend choices."""
        try:
            import sglang.srt.server_args as server_args

            choices = getattr(server_args, "ATTENTION_BACKEND_CHOICES", None)
            if choices is not None:
                for backend in ("kunlun", "kunlun_compressed"):
                    if backend not in choices:
                        choices.append(backend)
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

        # DeepSeek-V4's upstream defaults run after this hook and select
        # ``dsv4`` unconditionally. Preserve the user's public Kunlun name
        # through that later model-specific adjustment.
        if getattr(server_args, "attention_backend", None) == "kunlun_compressed":
            from sglang.srt.arg_groups import deepseek_v4_hook

            original = deepseek_v4_hook.apply_deepseek_v4_defaults
            if not getattr(original, "_kunlun_compressed", False):
                def apply_deepseek_v4_defaults(args, model_arch):
                    requested = args.attention_backend
                    requested_kv_dtype = args.kv_cache_dtype
                    if requested == "kunlun_compressed" and requested_kv_dtype in ("fp16", "int8"):
                        args.kv_cache_dtype = "bfloat16"
                    original(args, model_arch)
                    if requested == "kunlun_compressed":
                        args.attention_backend = requested
                        args.kv_cache_dtype = requested_kv_dtype
                        if args.prefill_attention_backend in (None, "dsv4"):
                            args.prefill_attention_backend = requested
                        if args.decode_attention_backend in (None, "dsv4"):
                            args.decode_attention_backend = requested

                apply_deepseek_v4_defaults._kunlun_compressed = True
                deepseek_v4_hook.apply_deepseek_v4_defaults = apply_deepseek_v4_defaults

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
