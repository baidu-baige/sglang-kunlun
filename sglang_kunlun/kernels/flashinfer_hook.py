"""Kunlun compatibility hooks for optional flashinfer helper symbols."""

from __future__ import annotations

import importlib

from sglang_kunlun.kernels.kernel_ops import register_jit_op


_FLASHINFER_MISSING_SYMBOL_MSG = (
    "flashinfer helper {symbol_name} is not supported by the Kunlun backend"
)


def _unsupported_flashinfer_symbol(symbol_name: str):
    def unsupported(*args, **kwargs):
        raise RuntimeError(
            _FLASHINFER_MISSING_SYMBOL_MSG.format(symbol_name=symbol_name)
        )

    return unsupported


def _ensure_flashinfer_symbol(module_path: str, symbol_name: str) -> None:
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return
    if not hasattr(module, symbol_name):
        setattr(module, symbol_name, _unsupported_flashinfer_symbol(symbol_name))


_ensure_flashinfer_symbol("flashinfer.prefill", "cudnn_batch_prefill_with_kv_cache")
_ensure_flashinfer_symbol("flashinfer.gemm", "mm_M1_16_K7168_N256")


def _disable_flashinfer_softmax_dispatch() -> None:
    """Force DSpark ``SoftmaxTemp`` off the flashinfer path.

    ``SoftmaxTemp.execute`` prefers ``flashinfer.sampling.softmax`` whenever that
    import succeeded (``_flashinfer_softmax is not None``). The installed wheel
    imports fine, but its OnlineSoftmax kernel is CUDA-only, so on Kunlun the
    launch dies with ``invalid device function``. 0.5.17 handled this by
    hardcoding ``_flashinfer_softmax = None`` in the main repo; the plugin now
    does it instead, which drops dispatch to the triton branch that
    ``kernel_ops`` replaces with the Kunlun ``softmax_temp`` op.
    """
    try:
        module = importlib.import_module(
            "sglang.kernels.ops.speculative.dspark.dspark_accept"
        )
    except ImportError:
        return
    module._flashinfer_softmax = None


_disable_flashinfer_softmax_dispatch()


@register_jit_op("flashinfer.prefill", "cudnn_batch_prefill_with_kv_cache")
def cudnn_batch_prefill_with_kv_cache(*args, **kwargs):
    """Raise for unsupported flashinfer batch prefill helper on Kunlun."""
    raise RuntimeError(
        _FLASHINFER_MISSING_SYMBOL_MSG.format(
            symbol_name="cudnn_batch_prefill_with_kv_cache"
        )
    )


@register_jit_op("flashinfer.gemm", "mm_M1_16_K7168_N256")
def mm_M1_16_K7168_N256(*args, **kwargs):
    """Raise for unsupported flashinfer GEMM helper on Kunlun."""
    raise RuntimeError(
        _FLASHINFER_MISSING_SYMBOL_MSG.format(symbol_name="mm_M1_16_K7168_N256")
    )
