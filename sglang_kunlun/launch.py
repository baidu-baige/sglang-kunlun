"""Explicit launcher for SGLang on Kunlun XPU."""

from __future__ import annotations

import os
import runpy


def main() -> None:
    """Run SGLang launch_server after installing Kunlun bootstrap hooks."""
    os.environ.setdefault("SGLANG_PLATFORM", "kunlun")

    from sglang_kunlun.bootstrap import _kunlun_pre_shim

    _kunlun_pre_shim()
    try:
        import sglang.srt.server_args as server_args

        choices = getattr(server_args, "ATTENTION_BACKEND_CHOICES", None)
        if choices is not None and "kunlun" not in choices:
            choices.append("kunlun")
    except Exception:
        pass
    # Repeat the platform's registration here, because the platform runs it too
    # early to be effective: KunlunSRTPlatform.__init__ fires while
    # sglang.srt.server_args is still executing, past ATTENTION_BACKEND_CHOICES
    # (``server_args.py:171``) but before
    # CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS (``:220``), so the
    # getattr there returns None and the list is silently left alone. By this
    # point the module is fully imported. Idempotent -- it checks membership.
    from sglang_kunlun.platform.srt import KunlunSRTPlatform

    KunlunSRTPlatform._extend_attention_backend_choices()
    import sglang_kunlun.hooks.layers.attention.attention_registry  # noqa: F401

    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
