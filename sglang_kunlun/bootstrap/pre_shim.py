"""Kunlun pre-import shim for CUDA-compatible runtime behavior."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _install_import_stub(names: tuple[str, ...]) -> None:
    """Replace ``names`` with empty stub modules in ``sys.modules``."""
    import sys
    import types

    for name in list(sys.modules):
        for root in names:
            if name == root or name.startswith(root + "."):
                del sys.modules[name]
                break

    for name in names:
        stub = types.ModuleType(name)
        stub.__path__ = []  # namespace-like so submodule imports fail cleanly
        stub._kunlun_stub = True
        sys.modules[name] = stub


def _stub_module_if_unloadable(probe: str, names: tuple[str, ...]) -> None:
    """Stub ``names`` when importing ``probe`` fails with a non-ImportError.

    CUDA-only extensions such as CuTeDSL (``cutlass``) and ``deep_gemm`` are
    installed in the Kunlun image but cannot initialize there: they load a CUDA
    toolkit / ``libcudart`` flavour that the XPU driver shim does not provide, and
    they signal that with ``RuntimeError`` rather than ``ImportError``.

    Upstream sglang guards these imports with ``except ImportError`` only, so the
    ``RuntimeError`` escapes and breaks every Kimi-K3 module that transitively
    imports them. Installing an empty stub converts any later
    ``import <name>.<sub>`` into ``ModuleNotFoundError`` (an ``ImportError``),
    which upstream does handle, so those code paths report as unavailable.

    A genuine ``ImportError`` is left alone: upstream already handles it.
    """
    import importlib
    import sys

    if getattr(sys.modules.get(names[0]), "_kunlun_stub", False):
        return

    try:
        importlib.import_module(probe)
        return
    except ImportError:
        return
    except Exception as exc:
        logger.info(
            "Kunlun: %r cannot initialize on this host (%s: %s); installing an "
            "import stub so CUDA-only paths report as unavailable.",
            probe,
            type(exc).__name__,
            str(exc)[:120],
        )

    _install_import_stub(names)


def _disable_flashinfer_version_check() -> None:
    """Stop flashinfer's cubin/python version check from aborting the import.

    ``sglang.srt.layers.quantization.fp4_utils`` does a top-level
    ``from flashinfer import fp4_quantize`` guarded only by ``except ImportError``.
    On Kunlun hosts the installed ``flashinfer-cubin`` and ``flashinfer-python``
    versions do not match, and flashinfer raises ``RuntimeError`` at import time,
    which escapes that guard and breaks every Kimi-K3 module that transitively
    imports the quantization package.

    The FP4/cubin kernels themselves are never used on XPU, so skipping the
    version check is safe and only affects the import path.
    """
    import os

    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


def _patch_fla_utils_if_loaded() -> None:
    import sys
    import torch

    module = sys.modules.get("sglang.kernels.ops.attention.fla.utils")
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


def _patch_optional_tilelang_import() -> None:
    """Convert missing TileLang shared-library failures into optional imports."""
    import builtins
    import functools

    current_import = builtins.__import__
    if getattr(current_import, "_kunlun_optional_tilelang", False):
        return

    @functools.wraps(current_import)
    def optional_tilelang_import(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            return current_import(name, globals, locals, fromlist, level)
        except OSError as error:
            if name == "tilelang" or name.startswith("tilelang."):
                raise ModuleNotFoundError(
                    f"Optional TileLang runtime is unavailable: {error}"
                ) from error
            raise

    optional_tilelang_import._kunlun_optional_tilelang = True
    builtins.__import__ = optional_tilelang_import


def _patch_transformers_optional_modules() -> None:
    """Stub transformers modules that only newer releases provide.

    ``sglang.srt.configs.__init__`` unconditionally imports
    ``muse_glimmer_processing``, which does a top-level
    ``from transformers.image_processing_backends import TorchvisionBackend``.
    That module only exists in transformers >= 4.58, so on an older install the
    *entire* ``sglang.srt.configs`` package fails to import and the server cannot
    start at all -- even though the Kunlun deployment never serves MuseGlimmer.
    A stub module keeps the import working; touching the symbol would fail, which
    is the correct outcome for an unsupported model.
    """
    import sys
    import types

    try:
        import transformers  # noqa: F401
    except Exception:
        return

    if "transformers.image_processing_backends" in sys.modules:
        return
    try:
        import transformers.image_processing_backends  # noqa: F401

        return
    except ImportError:
        pass

    module = types.ModuleType("transformers.image_processing_backends")

    class TorchvisionBackend:  # pragma: no cover - unreachable on Kunlun
        """Placeholder for the transformers >= 4.58 image-processing backend."""

        def __init__(self, *_args, **_kwargs):
            raise NotImplementedError(
                "transformers.image_processing_backends.TorchvisionBackend "
                "requires transformers >= 4.58; the Kunlun deployment stubs it "
                "so that sglang.srt.configs stays importable."
            )

    module.TorchvisionBackend = TorchvisionBackend
    sys.modules["transformers.image_processing_backends"] = module
    setattr(transformers, "image_processing_backends", module)


def _patch_transformers_optional_model_modules() -> None:
    """Stub 只有更新版 transformers 才有的 `transformers.models.*` 包。

    失败形态和 `_patch_transformers_optional_modules` 一样，只是深一层：
    `sglang.srt.configs.glm5_next` 在顶层 `from
    transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig`，
    而 `glm_ocr` 在本机的 transformers 4.57.1 上不存在。不 stub 的话整个
    `sglang.srt.configs` 包不可导入，进而所有经它导入的昆仑 hook 模块全部注册失败——
    哪怕本部署只服务 DeepSeek。stub 类继承 `PretrainedConfig` 以保证导入期可被子类化，
    但实例化时抛错：对本平台不支持的模型来说这是正确行为。
    """
    import sys
    import types

    try:
        import transformers  # noqa: F401
        from transformers.configuration_utils import PretrainedConfig
    except Exception:
        return

    # (package, submodule, symbol names)
    specs = (("glm_ocr", "configuration_glm_ocr", ("GlmOcrVisionConfig",)),)

    for pkg_name, sub_name, symbols in specs:
        full_pkg = f"transformers.models.{pkg_name}"
        full_sub = f"{full_pkg}.{sub_name}"
        if full_sub in sys.modules:
            continue
        try:
            __import__(full_sub)
            continue
        except ImportError:
            pass

        pkg = sys.modules.get(full_pkg) or types.ModuleType(full_pkg)
        pkg.__path__ = []  # mark as package so submodule imports resolve
        sub = types.ModuleType(full_sub)

        for symbol in symbols:

            def _make(symbol_name: str, missing: str = full_pkg):
                class _MissingConfig(PretrainedConfig):
                    """Placeholder for a config class from a newer transformers."""

                    def __init__(self, *_args, **_kwargs):
                        raise NotImplementedError(
                            f"{missing}.{symbol_name} requires a newer "
                            "transformers release; sglang-kunlun stubs it so "
                            "that sglang.srt.configs stays importable."
                        )

                _MissingConfig.__name__ = symbol_name
                _MissingConfig.__qualname__ = symbol_name
                return _MissingConfig

            cls = _make(symbol)
            setattr(sub, symbol, cls)
            setattr(pkg, symbol, cls)

        setattr(pkg, sub_name, sub)
        sys.modules[full_pkg] = pkg
        sys.modules[full_sub] = sub
        setattr(transformers.models, pkg_name, pkg)


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
    import importlib

    symbols = [
        ("flashinfer.prefill", "cudnn_batch_prefill_with_kv_cache"),
        ("flashinfer.gemm", "mm_M1_16_K7168_N256"),
    ]
    for module_path, symbol_name in symbols:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        if not hasattr(module, symbol_name):

            def _unsupported(*_a, _name=symbol_name, **_kw):
                raise RuntimeError(
                    f"flashinfer.{_name} is not supported by the Kunlun backend"
                )

            setattr(module, symbol_name, _unsupported)


def _stub_small_copy_fast_path() -> None:
    """Keep the new 0.5.22 small-copy Triton fast path out of the CUDA-graph fill.

    ``cuda_graph_buffer_registry._grouped_foreach_copy_`` gained a fast path in
    0.5.22::

        if dsts and dsts[0].is_cuda:
            from sglang.kernels.ops.memory.small_copy import try_small_copy
            if try_small_copy(dsts, srcs):
                return

    Two separate problems on Kunlun, and they bite in sequence:

    1. ``small_copy`` declares its kernel with
       ``@triton.jit(do_not_specialize_on_alignment=...)``, a keyword Triton
       3.2+ added. The installed Triton is 3.0.0, so merely *importing* the
       module raises ``TypeError: jit() got an unexpected keyword argument``
       -- on every eager extend batch, which killed the first real forward.
    2. Even with the import fixed, ``try_small_copy``'s only platform gate is
       ``device.type != "cuda"``, which is False here because the Kunlun
       platform reuses the CUDA device type. It would claim the copy and then
       fail at launch ("0 compatible backends for target (cuda)").

    So filtering the Triton kwarg would just move the failure later. Install a
    stub whose ``try_small_copy`` declines instead: declining is the documented
    contract of that call, and the caller falls back to
    ``torch._foreach_copy_`` / ``dst.copy_()``, which is what 0.5.19 always did.
    """
    import sys
    import types

    name = "sglang.kernels.ops.memory.small_copy"
    if name in sys.modules:
        return

    module = types.ModuleType(name)
    module.__doc__ = (
        "Kunlun stub: the upstream small-copy Triton kernel needs Triton 3.2+ "
        "to import and has no Kunlun backend to launch on."
    )

    def try_small_copy(dsts, srcs):  # noqa: ARG001 - contract is "decline"
        return False

    def write_host_ints(dsts, values):
        """Torch-native stand-in for the Triton ``write_host_ints``.

        0.5.23's ``forward_batch_info`` imports ``write_host_ints`` from this
        same module at import time. The upstream kernel streams the host ints
        in as kernel arguments to dodge a copy-engine wait, but its Triton decl
        needs 3.2+ to import (this build ships 3.0.0). Since we fully shadow the
        module to keep ``try_small_copy`` out, we must also supply this symbol;
        do the host->device write in torch -- correct, if not launch-free.
        """
        import torch

        if len(dsts) != len(values):
            raise ValueError(
                f"write_host_ints got {len(dsts)} tensors and {len(values)} value lists"
            )
        for dst, row in zip(dsts, values):
            if (
                dst.dtype not in (torch.int32, torch.int64)
                or dst.ndim != 1
                or not dst.is_contiguous()
                or dst.numel() != len(row)
            ):
                raise ValueError(
                    "write_host_ints needs contiguous 1-D int32/int64 tensors "
                    f"matching the value counts; got {dst.dtype} {tuple(dst.shape)} "
                    f"on {dst.device} for {len(row)} values"
                )
            if len(row):
                dst.copy_(
                    torch.tensor(row, dtype=dst.dtype, device=dst.device),
                    non_blocking=True,
                )

    module.try_small_copy = try_small_copy
    module.write_host_ints = write_host_ints
    module._sglang_kunlun_stub = True
    sys.modules[name] = module


def _provide_copy_pages_without_triton() -> None:
    """Serve ``kvcache.copy_pages`` in torch; its Triton decl needs 3.2+ to import.

    0.5.23 added ``sglang/kernels/ops/kvcache/copy_pages.py`` and
    ``mem_cache/unified_memory_pool.py`` imports ``copy_pages`` from it at module
    import time. The kernel is declared
    ``@triton.jit(do_not_specialize_on_alignment=[...])`` -- a keyword Triton 3.2
    added, while this build ships 3.0.0 -- so the import itself raises
    ``TypeError: jit() got an unexpected keyword argument`` and takes the whole
    plugin load with it (``load_plugins`` -> ``register_all`` ->
    ``hooks.speculative`` -> ``eagle_worker_v2`` -> the NPU graph runner ->
    ``unified_memory_pool``). The 0.5.23 boot died there before printing a single
    scheduler line.

    Unlike ``small_copy``, declining is not an option: the callers
    (``unified_memory_pool`` page defrag/shift) need the copy to happen. But
    upstream already carries a device-agnostic version for its own non-CUDA
    branch::

        env = raw[: num_pages * page_bytes].view(num_pages, page_bytes)
        env[dst_pages] = env[src_pages]

    which is exactly what Kunlun wants (it reports ``is_cuda=True``, so the
    upstream gate would send it to Triton). So pre-register a module whose
    ``copy_pages`` is that torch form, before anything can import the real one.
    Same ``sys.modules`` pre-seeding mechanism as the small_copy stub; no
    monkeypatching of a loaded module.
    """
    import sys
    import types

    name = "sglang.kernels.ops.kvcache.copy_pages"
    if name in sys.modules:
        return

    import torch

    module = types.ModuleType(name)
    module.__doc__ = (
        "Kunlun replacement: upstream's copy_pages kernel needs Triton 3.2+ to "
        "import, and its Triton path has no Kunlun backend to launch on. This is "
        "upstream's own non-CUDA branch, done in torch."
    )

    def copy_pages(
        raw: "torch.Tensor",
        dst_pages: "torch.Tensor",
        src_pages: "torch.Tensor",
        num_pages: int,
        page_bytes: int,
    ) -> None:
        """Copy src -> dst page sets; disjointness is the caller's contract."""
        count = dst_pages.numel()
        assert count == src_pages.numel()
        if count == 0:
            return
        assert raw.dtype == torch.uint8 and raw.is_contiguous()
        env = raw[: num_pages * page_bytes].view(num_pages, page_bytes)
        env[dst_pages] = env[src_pages]

    module.copy_pages = copy_pages
    module._sglang_kunlun_stub = True
    sys.modules[name] = module


def _provide_flash_mla_interface_if_unusable() -> None:
    """Stand in for ``kernels.ops.attention.flash_mla_interface`` when the wheel is absent.

    0.5.23 moved the FlashMLA entry points out of ``sgl_kernel.flash_mla`` into
    ``sglang/kernels/ops/attention/flash_mla_interface.py``, which imports the
    ``sgl-flash-mla`` wheel at module import time. The wheel in this image needs
    ``xtorch_ops``, which the build does not ship::

        flash_mla/flash_mla_interface.py:6  import xtorch_ops
        ModuleNotFoundError: No module named 'xtorch_ops'

    and ``DeepseekV4AttnBackend.__init__`` (``deepseek_v4_backend.py:1263``)
    imports it under a plain ``if is_cuda():`` -- one of the CUDA gates 0.5.23
    added on a patched path. Kunlun's ``is_cuda()`` is True (the platform reuses
    the CUDA device type), so the gate opens and the import kills the scheduler
    before any attention backend is built, even though Kunlun never launches a
    FlashMLA kernel: ``SGLANG_HACK_FLASHMLA_BACKEND=kunlun`` and this plugin's
    own ``kunlun_compressed`` backend serve that path, and the module's own
    docstring says "CUDA only: HIP and XPU keep their own sgl-kernel ops".

    So pre-seed the module with the names upstream reads, each raising if it is
    ever actually invoked -- a wrong answer here must not be silent. Same
    ``sys.modules`` mechanism as the small_copy / copy_pages / deep_gemm shims.
    """
    import importlib.util
    import sys
    import types

    name = "sglang.kernels.ops.attention.flash_mla_interface"
    if name in sys.modules:
        return
    if importlib.util.find_spec("flash_mla") is not None:
        try:
            importlib.import_module("flash_mla")
            return  # the real wheel works; let upstream use it
        except Exception as exc:
            logger.info(
                "Kunlun: flash_mla wheel unusable (%s: %s); stubbing %s",
                type(exc).__name__,
                str(exc)[:120],
                name,
            )

    module = types.ModuleType(name)
    module.__doc__ = (
        "Kunlun stub: the sgl-flash-mla wheel cannot import on this build and "
        "Kunlun serves MLA through its own backend."
    )
    # Upstream reads this as a plain int to size a split-KV decision.
    module.SPLIT_KV_MIN_TOPK = 641

    def _unsupported(_name):
        def fail(*_a, **_kw):
            raise RuntimeError(
                f"flash_mla_interface.{_name} is not available on Kunlun; "
                "the kunlun_compressed backend serves this path"
            )

        fail.__name__ = _name
        return fail

    for fn in (
        "get_mla_metadata",
        "flash_mla_with_kvcache",
        "flash_mla_sparse_fwd",
        "flash_mla_mega_prefill",
        "flash_mla_mega_decode",
    ):
        setattr(module, fn, _unsupported(fn))
    module._sglang_kunlun_stub = True
    sys.modules[name] = module


def _kunlun_pre_shim() -> None:
    """Force triton's active driver to advertise ``backend == "cuda"``."""
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
        _patch_optional_tilelang_import()
    except Exception:
        pass

    # Kimi-K3 pulls the quantization package (and therefore flashinfer +
    # CuTeDSL) in at import time; both shims must run before that happens.
    _disable_flashinfer_version_check()

    for probe, names in (
        ("cutlass", ("cutlass", "nvidia_cutlass_dsl")),
        ("deep_gemm", ("deep_gemm",)),
    ):
        try:
            _stub_module_if_unloadable(probe, names)
        except Exception:
            logger.warning(
                "Kunlun: import stub installation for %r failed", probe, exc_info=True
            )

    # 上面那个 helper 只处理「装了但初始化 RuntimeError」的情况；真的**没装**时
    # 它走 `except ImportError: return`，不装 stub。但上游有**裸 import**：
    #     sglang/srt/layers/attention/dsa_backend.py:97
    #         if is_cuda():
    #             import deep_gemm
    # 昆仑的 is_cuda() 为 True ⇒ deep_gemm 只要不可导入就必炸
    # （实测 D 节点在 tokenizer/HTTP 进程里 `ModuleNotFoundError: No module
    #  named 'deep_gemm'`，整个服务起不来）。这里补一刀兜底。
    try:
        _stub_deep_gemm_if_missing()
    except Exception:
        logger.warning("Kunlun: deep_gemm 兜底 stub 安装失败", exc_info=True)

    try:
        _patch_torch_cuda_symm_mem_apis()
    except Exception:
        pass

    try:
        _patch_flashinfer_optional_symbols()
    except Exception:
        pass

    try:
        _patch_transformers_optional_modules()
    except Exception:
        pass

    try:
        _patch_transformers_optional_model_modules()
    except Exception:
        pass

    try:
        _stub_small_copy_fast_path()
    except Exception:
        logger.warning("Kunlun: small_copy stub installation failed", exc_info=True)

    try:
        _provide_copy_pages_without_triton()
    except Exception:
        logger.warning("Kunlun: copy_pages shim installation failed", exc_info=True)

    try:
        _provide_flash_mla_interface_if_unusable()
    except Exception:
        logger.warning(
            "Kunlun: flash_mla_interface shim installation failed", exc_info=True
        )

    try:
        from sglang_kunlun.bootstrap import sgl_kernel_stub
        sgl_kernel_stub.install()
    except Exception:
        pass

    try:
        import triton.runtime.driver as _driver_cfg
    except Exception:
        return

    _driver_mod = sys.modules.get("triton.runtime.driver")
    if _driver_mod is None or not hasattr(_driver_mod, "_create_driver"):
        return

    class _CudaTarget:
        backend = "cuda"

    def _patch_driver(drv):
        try:
            drv.get_current_target = lambda: _CudaTarget()
        except Exception:
            pass
        return drv

    try:
        if not getattr(_driver_mod, "_kunlun_create_wrapped", False):
            _orig_create = _driver_mod._create_driver

            def _wrapped_create():
                return _patch_driver(_orig_create())

            _driver_mod._create_driver = _wrapped_create
            _driver_mod._kunlun_create_wrapped = True

            try:
                active = _driver_cfg.active
                if active is not None and getattr(active, "_obj", None) is None:
                    active._init_fn = _wrapped_create
            except Exception:
                pass
    except Exception:
        pass

    try:
        active = _driver_cfg.active
        if active is not None:
            active.get_current_target = lambda: _CudaTarget()
            if getattr(active, "_obj", None) is not None:
                _patch_driver(active._obj)
    except Exception:
        pass

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


def _stub_deep_gemm_if_missing() -> None:
    """deep_gemm 完全没安装时也装一个空 stub。

    与 `_stub_module_if_unloadable` 的区别：那个只在「能 import 但抛非 ImportError」
    时装 stub；这里覆盖「`find_spec` 就是 None」的场景，因为上游存在
    `if is_cuda(): import deep_gemm` 这类**没有 try/except 保护**的裸 import，
    在昆仑（is_cuda() 为 True）上会直接抛 ModuleNotFoundError。

    空 stub 的效果：`import deep_gemm` 成功；`from deep_gemm import x` 变成
    ImportError（上游的 except ImportError 能兜住）；`deep_gemm.x` 变成
    AttributeError（响亮失败，不会静默走错路径）。
    脚本里的 `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=True` 会让那些 deep_gemm
    函数本来就不被调用。
    """
    import importlib.util
    import sys

    if getattr(sys.modules.get("deep_gemm"), "_kunlun_stub", False):
        return
    if importlib.util.find_spec("deep_gemm") is not None:
        return
    logger.info(
        "Kunlun: deep_gemm 未安装；安装空 stub 以通过上游 "
        "`if is_cuda(): import deep_gemm` 这类裸 import。"
    )
    _install_import_stub(("deep_gemm",))
