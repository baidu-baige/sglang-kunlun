# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlun pre-import shim for CUDA-compatible runtime behavior."""

from __future__ import annotations

import threading


_FLOAT16_KV_PATCH_LOCK = threading.Lock()
_GRAMMAR_BACKEND_PATCH_LOCK = threading.Lock()


def _patch_fla_utils_if_loaded() -> None:
    import sys
    import torch

    module = sys.modules.get("sglang.srt.layers.attention.fla.utils")
    if module is None:
        return

    def get_available_device() -> str:
        return "cuda"

    def _check_platform() -> str:
        return "nvidia"

    module.get_available_device = get_available_device
    module._check_platform = _check_platform
    module.device = "cuda"
    module.device_torch_lib = torch.cuda
    module.device_platform = "nvidia"
    module.is_amd = False
    module.is_intel = False
    module.is_nvidia = True
    module.is_intel_alchemist = False
    module.is_nvidia_hopper = False



def _patch_torch_cuda_symm_mem_apis() -> None:
    """Provide no-op stubs for pynccl symmetric-memory CUDA private APIs.

    ``sglang.srt.distributed.device_communicators.pynccl_allocator`` does a
    top-level ``from torch.cuda.memory import (..., _cuda_beginAllocateCurrentThreadToPool,
    _cuda_endAllocateToPool, _cuda_releasePool)``. These private CUDA-graph
    mempool helpers only exist on newer upstream torch (2.6+); the Kunlun
    runtime ships an older torch (2.5.x) that lacks them, so the import crashes
    even though Kunlun never enables pynccl symmetric memory.

    We add no-op stubs so the import succeeds. They are only ever *called* from
    ``SymmetricMemoryContext`` (guarded by ``enable_symm_mem``, which Kunlun
    does not turn on), so these stubs never run on the hot path.
    """
    import torch

    mem = getattr(torch.cuda, "memory", None)
    if mem is None:
        return

    for name in (
        "_cuda_beginAllocateCurrentThreadToPool",
        "_cuda_endAllocateToPool",
        "_cuda_releasePool",
    ):
        if not hasattr(mem, name):

            def _unsupported(*_a, _name=name, **_kw):
                raise NotImplementedError(
                    f"torch.cuda.memory.{_name} is not available on the Kunlun "
                    "runtime; pynccl symmetric memory (enable_symm_mem) is "
                    "unsupported on XPU."
                )

            setattr(mem, name, _unsupported)

    # Mirror onto torch._C so the ``after_2_8_0`` branch in pynccl_allocator
    # (which calls torch._C._cuda_*) also resolves to a stub if reached.
    _C = getattr(torch, "_C", None)
    if _C is not None:
        for name in (
            "_cuda_beginAllocateCurrentThreadToPool",
            "_cuda_beginAllocateToPool",
            "_cuda_endAllocateToPool",
            "_cuda_endAllocateCurrentStreamToPool",
            "_cuda_releasePool",
        ):
            if not hasattr(_C, name):

                def _unsupported_c(*_a, _name=name, **_kw):
                    raise NotImplementedError(
                        f"torch._C.{_name} is not available on the Kunlun runtime."
                    )

                setattr(_C, name, _unsupported_c)


def _patch_flashinfer_optional_symbols() -> None:
    """Inject stubs for optional flashinfer symbols imported at sglang import time.

    Several sglang 0.5.14 modules do *top-level* ``from flashinfer.X import Y``
    inside their ``if _is_cuda:`` block (e.g.
    ``sglang/srt/layers/attention/vision.py`` imports
    ``cudnn_batch_prefill_with_kv_cache``). Because Kunlun masquerades as CUDA
    (``is_cuda()`` -> True), these imports run, but the Kunlun-side flashinfer
    build lacks the symbol, so the import crashes before any plugin hook (which
    is registered later via ``register_all()``) can patch it.

    We pre-inject no-op-raising stubs here so the top-level import succeeds.
    They only raise if actually *called* on the Kunlun path (which doesn't use
    these CUDA-only flashinfer kernels).
    """
    import os
    import sys

    # The Kunlun launcher explicitly disables flashinfer.  Do not even touch
    # its package in that mode: importing it initializes tvm_ffi and may
    # synchronously build/load a CUDA extension.
    if os.environ.get("SGLANG_IS_FLASHINFER_AVAILABLE", "").lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return

    symbols = [
        ("flashinfer.prefill", "cudnn_batch_prefill_with_kv_cache"),
        ("flashinfer.gemm", "mm_M1_16_K7168_N256"),
    ]
    for module_path, symbol_name in symbols:
        # Importing flashinfer here eagerly initializes its JIT/cubin loader
        # (via tvm_ffi). On Kunlun this can block while it probes or builds a
        # CUDA extension. The symbols are optional and the caller already
        # advertises flashinfer as unavailable, so only patch modules that
        # have been loaded by the application.
        module = sys.modules.get(module_path)
        if module is None:
            continue
        if not hasattr(module, symbol_name):

            def _unsupported(*_a, _name=symbol_name, **_kw):
                raise RuntimeError(
                    f"flashinfer.{_name} is not supported by the Kunlun backend"
                )

            setattr(module, symbol_name, _unsupported)


def _rewrite_legacy_cli_argv() -> None:
    """Map legacy CLI flags in ``sys.argv`` to their renamed/split upstream
    equivalents, so stage launch configs keep working after the sglang
    arg-group refactor (flag meaning unchanged). No-op if none are present::

        --cuda-graph-max-bs N        -> --cuda-graph-max-bs-decode N
                                        --cuda-graph-max-bs-prefill N
        --enable-nsa-prefill-context-parallel -> --enable-prefill-cp
        --nsa-prefill-cp-mode round-robin-split -> --cp-strategy interleave
    """
    import sys

    argv = sys.argv[1:]
    out = []
    i, n = 0, len(argv)
    while i < n:
        tok = argv[i]
        if tok == "--cuda-graph-max-bs" and i + 1 < n:
            val = argv[i + 1]
            out += ["--cuda-graph-max-bs-decode", val,
                    "--cuda-graph-max-bs-prefill", val]
            i += 2
        elif tok == "--enable-nsa-prefill-context-parallel":
            out.append("--enable-prefill-cp")
            i += 1
        elif tok == "--nsa-prefill-cp-mode" and i + 1 < n:
            val = argv[i + 1]
            out += ["--cp-strategy",
                    "interleave" if val == "round-robin-split" else val]
            i += 2
        else:
            out.append(tok)
            i += 1
    if out != argv:
        sys.argv[1:] = out


def _widen_kunlun_kv_cache_dtype_choices() -> None:
    """Re-add ``float16`` / ``half`` to the ``--kv-cache-dtype`` argparse
    choices. DeepSeek V4 on Kunlun runs a float16 KV cache, but the upgraded
    sglang arg-group parser only lists bf16/fp8/fp4 variants and rejects
    float16 at parse time. The actual float16 KV handling still lives in the
    arg_groups / model_runner hooks; this only unblocks CLI parsing."""
    import argparse

    if getattr(argparse.ArgumentParser, "_kunlun_kv_dtype_patched", False):
        return
    _orig_add_argument = argparse.ArgumentParser.add_argument

    def add_argument(self, *args, **kwargs):
        if "--kv-cache-dtype" in args and kwargs.get("choices") is not None:
            choices = list(kwargs["choices"])
            for dtype in ("float16", "half"):
                if dtype not in choices:
                    choices.append(dtype)
            kwargs["choices"] = choices
        return _orig_add_argument(self, *args, **kwargs)

    argparse.ArgumentParser.add_argument = add_argument
    argparse.ArgumentParser._kunlun_kv_dtype_patched = True



def _preload_kunlun_cuda_compat() -> None:
    """Preload the Kunlun CUDA-compat ``libcudart.so.11.0`` (RTLD_GLOBAL).

    Mooncake's ``engine.so`` / ``store.so`` (the PD KV transfer engine) are
    linked against ``libcudart.so.11.0``. The Kunlun image ships a compat build
    under ``/usr/local/xpu/so`` but that directory is not on the loader path, so
    ``from mooncake.engine import TransferEngine`` fails with
    ``libcudart.so.11.0: cannot open shared object file``. Loading it globally
    here (before SGLang imports the transfer engine) satisfies the dependency."""
    import ctypes
    import os

    lib = "/usr/local/xpu/so/libcudart.so.11.0"
    if os.path.exists(lib):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _pin_nccl_port_via_argv() -> None:
    """Pin ``nccl_port`` for dp-attention servers by injecting ``--nccl-port``.

    dp-attention startup races the DataParallelController for the ephemeral
    port SGLang picks for ``nccl_port`` (EADDRINUSE on the rank-0 TCPStore). The
    plugin used to pin it after resolution, but the upgraded resolver seals
    post-resolution writes, so pin it as an *input*: pick a free port below the
    ephemeral range (no ephemeral bind can land there) and add it to argv.
    No-op unless dp-attention is on and the operator did not set it."""
    import socket
    import sys

    argv = sys.argv[1:]
    if "--enable-dp-attention" not in argv or "--nccl-port" in argv:
        return
    reserved = set()
    for flag in ("--port", "--disaggregation-bootstrap-port"):
        if flag in argv:
            try:
                reserved.add(int(argv[argv.index(flag) + 1]))
            except (ValueError, IndexError):
                pass
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as fh:
            floor = int(fh.read().split()[0])
    except Exception:
        floor = 32768
    for port in range(floor - 1, 1024, -1):
        if port in reserved:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
            except OSError:
                continue
        sys.argv += ["--nccl-port", str(port)]
        return



def _disable_xpu_unsupported_argv() -> None:
    """Opt out of features that need a PyNCCL communicator, which Kunlun (BKCL,
    no NCCL/NVML) cannot provide. ``enable_tp_lm_head_all_to_all`` auto-enables
    for DSV4 + dp-attention and then aborts CUDA-graph capture with 'requires an
    available PyNCCL communicator'. Setting it as an input keeps the resolver's
    default pass from turning it on."""
    import sys

    argv = sys.argv[1:]
    # Only a sglang entrypoint may have sglang flags appended. This shim is
    # imported by EVERY python process in the image (sitecustomize path), and an
    # unconditional append broke unrelated commands — `python -m venv` in the
    # container died with "unrecognized arguments: --no-enable-tp-lm-head-all-to-all"
    # (2026-11-17, second regression of the same bug).
    looks_like_sglang = (
        any(arg == "--model-path" or arg.startswith("--model-path=")
            for arg in argv)
        or "sglang" in (sys.argv[0] or "")
    )
    if not looks_like_sglang:
        return
    if ("--enable-tp-lm-head-all-to-all" not in argv
            and "--no-enable-tp-lm-head-all-to-all" not in argv):
        sys.argv.append("--no-enable-tp-lm-head-all-to-all")


def _ensure_deep_gemm_module() -> None:
    """Seed an empty ``deep_gemm`` placeholder before SGLang is imported.

    Upgraded sglang ``srt/layers/attention/dsa_backend.py`` runs a bare
    ``import deep_gemm`` at module-import time (under the CUDA-masquerade path),
    which happens while the plugin's own ``kernels.deep_geem_hook`` is still
    being imported -- so the hook cannot install its stub in time and the whole
    ``register_all`` plugin load aborts. Placing an empty module in
    ``sys.modules`` early lets that import resolve; ``deep_geem_hook`` then fills
    in the Kunlun symbols (``get_num_sms`` etc.) on the same module object."""
    import sys
    import types

    if "deep_gemm" in sys.modules:
        return
    try:
        import deep_gemm  # noqa: F401
    except Exception:
        sys.modules["deep_gemm"] = types.ModuleType("deep_gemm")


def _default_bkcl_stream_sync_env() -> None:
    """Default BKCL to ordering its collectives against the compute stream.

    libbkcl does not order collectives against the caller's compute stream, so a
    downstream kernel can read an all-reduce / all-to-all buffer before the
    communication kernel has landed. The symptom is per-request non-determinism:
    the same prompt at temperature=0 returns several different completions, which
    is how it was first measured on DSV4-Flash (deterministic only with these two
    set).

    These are plain runtime environment variables, not server args, so a
    deployment that never exports them silently runs without the ordering. Default
    them on here -- ``setdefault`` keeps an explicit value from the launch script
    winning, which is the opt-out.
    """
    import os

    os.environ.setdefault("BKCL_SYNC_OP", "1")
    os.environ.setdefault("BKCL_FORCE_SYNC", "1")


def _kunlun_pre_shim() -> None:
    """Kunlun pre-import shims applied before SGLang is imported (deep_gemm
    placeholder, CLI/argv normalization, mooncake libcudart preload, torch
    device-capability stubs). Does NOT force triton's target to CUDA."""
    import os
    import sys

    is_kunlun = (
        "torch_xmlir" in sys.modules
        or "xpytorch_import_hook" in sys.modules
        or os.path.exists("/usr/local/xre")
    )
    if not is_kunlun:
        return

    try:
        _default_bkcl_stream_sync_env()
    except Exception:
        pass

    try:
        _ensure_deep_gemm_module()
    except Exception:
        pass

    try:
        _preload_kunlun_cuda_compat()
    except Exception:
        pass

    try:
        _rewrite_legacy_cli_argv()
    except Exception:
        pass

    try:
        _pin_nccl_port_via_argv()
    except Exception:
        pass

    try:
        _disable_xpu_unsupported_argv()
    except Exception:
        pass

    try:
        _widen_kunlun_kv_cache_dtype_choices()
    except Exception:
        pass

    try:
        _patch_torch_cuda_symm_mem_apis()
    except Exception:
        pass

    try:
        _patch_flashinfer_optional_symbols()
    except Exception:
        pass

    try:
        from sglang_kunlun.bootstrap import sgl_kernel_stub
        sgl_kernel_stub.install()
    except Exception:
        pass

    # NOTE: do NOT masquerade triton's target as CUDA. This image's triton
    # ships a real ``xpu`` backend (triton/backends/xpu); forcing
    # target.backend="cuda" makes upstream triton kernels (dsv4 attn
    # metadata, expand_prefill_causally, page_table_positions, ...) abort
    # make_backend with '0 compatible backends for target (cuda)'. Leaving
    # the real xpu target lets them compile on Kunlun.
    try:
        import torch as _torch

        if hasattr(_torch, "cuda") and not getattr(_torch.cuda, "_kunlun_stubbed", False):
            _torch.cuda.get_device_name = lambda *_a, **_kw: "Kunlun-XPU"
            _torch.cuda.get_device_capability = lambda *_a, **_kw: (8, 0)
            _torch.cuda._kunlun_stubbed = True
    except Exception:
        pass

    try:
        _patch_fla_utils_if_loaded()
    except Exception:
        pass


def patch_grammar_backend_default() -> None:
    """Default ``grammar_backend`` to ``"none"`` on Kunlun.

    Upstream moved this default out of ``ServerArgs._handle_grammar_backend``
    (now gone) into the arg-group resolver
    ``sglang.srt.arg_groups.serving_hook.handle_grammar_backend``, which
    declares ``"xgrammar"`` when the field is unset.  Replace that resolver
    step so an unset ``grammar_backend`` still resolves to ``"none"``.  The
    resolver imports the function locally at resolution time (after plugin
    load), so the replacement is picked up.
    """
    from sglang.srt.arg_groups import serving_hook
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

    with _GRAMMAR_BACKEND_PATCH_LOCK:
        if getattr(serving_hook, "_kunlun_grammar_backend_patched", False):
            return

        def handle_grammar_backend(server_args):
            cfg = resolving_view(server_args)
            if cfg.grammar_backend is None:
                declare_resolution(
                    server_args, "_handle_grammar_backend", grammar_backend="none"
                )

        serving_hook.handle_grammar_backend = handle_grammar_backend
        serving_hook._kunlun_grammar_backend_patched = True
