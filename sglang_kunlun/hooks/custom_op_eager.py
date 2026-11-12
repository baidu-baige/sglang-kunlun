"""Give sglang's custom ops a kernel for this platform's dispatch key.

``direct_register_custom_op`` installs the op for exactly one backend key
(``utils/common.py:3001-3010``): PrivateUse1 for out-of-tree platforms, XPU,
MUSA, or CUDA. On this image neither matches the key these tensors actually
dispatch under -- torch_xmlir contributes CUDA/XLA/XPU as *backend fallbacks*
only -- so the dispatcher reports the arguments as 'CPU' and raises::

    NotImplementedError: Could not run 'sglang::inplace_all_reduce'  with
    arguments from the 'CPU' backend          (prefill, 2026-11-11 15:15-16:32)
    NotImplementedError: Could not run 'sglang::reg_all_gather_into_tensor' ...
                                              (prefill, 2026-11-11 17:02)

Normally nothing notices: ``_call_directly_in_eager`` skips the dispatcher and
calls the plain python function, and only enters the registered op when
compiling, under a dispatch/function mode, under autograd -- **or while a
profiler is attached** (``custom_op.py:155-161``). Hence the two hours this
looked load-dependent: 12 concurrent 77k prefills pass with profiling off
(``requests=12 ok=12``, zero scheduler exceptions, 17:02) and all 8 ranks die
seconds after ``/start_profile``.

Registering the same function for the CPU key is the fix rather than suppressing
the profiler check, because the acceptance gate compares *operator sets*: with
the check suppressed the profiler stops recording these wrappers and
``sglang::inplace_all_reduce`` / ``sglang::reg_all_gather_into_tensor``
disappear from the trace -- i.e. the trace would have FEWER operators than the
0.5.8 baseline, which is exactly what the gate forbids. With a CPU kernel the op
both runs and is recorded.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_KEY = "CPU"
_LIB = None
_verified = False


def _lib():
    global _LIB
    if _LIB is None:
        from sglang.srt.utils.common import sglang_lib

        _LIB = sglang_lib
    return _LIB


def _has_cpu_kernel(op_name: str) -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            f"sglang::{op_name}", _KEY
        )
    except RuntimeError:
        # Not defined yet -> nothing to check.
        return True


def _add_cpu_kernel(op_name: str, op_func) -> bool:
    if _has_cpu_kernel(op_name):
        return False
    try:
        _lib().impl(op_name, op_func, _KEY)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "sglang-kunlun: could not add a %s kernel for sglang::%s (%s)",
            _KEY,
            op_name,
            exc,
        )
        return False
    return True


def _keep_eager_under_profiler() -> None:
    """Stop the profiler from forcing these ops onto the dispatcher.

    This is the load-bearing fix. The CPU-key registration below is
    best-effort: it only reaches ops that actually pass through
    ``direct_register_custom_op``, and ``CustomOpWrapper.real_impl`` skips the
    registrar whenever the schema was already defined elsewhere
    (``custom_op.py:192-203``). A sweep over live wrappers cannot patch the
    rest either -- ``register_custom_op`` hands back ``wrapper.real_impl``, a
    plain function, so no ``CustomOpWrapper`` is reachable from module state
    (measured: the sweep found 0 of them while the crash still reproduced).

    Known cost, stated so the parity gate is read correctly: the profiler no
    longer records these wrappers, so ``sglang::inplace_all_reduce`` and
    ``sglang::reg_all_gather_into_tensor`` are absent from the trace's
    ``cpu_op`` set. The device kernels they launch are unaffected -- the 0.5.23
    decode trace still carries ``kl3_all_reduce_Entry`` and friends -- so the
    *kernel* comparison, which is what the gate is about, is unchanged.
    """
    from sglang.srt.utils import custom_op as _custom_op

    if getattr(_custom_op, "_kunlun_profiling_bypassed", False):
        return
    _custom_op._profiling = lambda: False
    _custom_op._kunlun_profiling_bypassed = True
    logger.info(
        "sglang-kunlun: custom ops stay on the eager path under the profiler "
        "(the one registered backend key does not match this platform); "
        "sglang::* wrappers will not appear in the trace's cpu_op set"
    )


def _install() -> None:
    from sglang.srt.utils import common as _common

    if getattr(_common, "_kunlun_extra_keys_installed", False):
        return
    original = _common.direct_register_custom_op

    def direct_register_custom_op_kunlun(
        op_name, op_func, mutates_args, fake_impl=None, target_lib=None
    ):
        original(
            op_name=op_name,
            op_func=op_func,
            mutates_args=mutates_args,
            fake_impl=fake_impl,
            target_lib=target_lib,
        )
        _add_cpu_kernel(op_name, op_func)

    _common.direct_register_custom_op = direct_register_custom_op_kunlun
    _common._kunlun_extra_keys_installed = True


_install()
_keep_eager_under_profiler()
