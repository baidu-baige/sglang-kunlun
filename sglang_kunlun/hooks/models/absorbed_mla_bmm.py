"""Route the absorbed-MLA bmm pair through the vendor ``kunlun_ops.bmm``.

0.5.8 carried a fork-only helper for exactly these two calls
(``aima_sglang/.../models/deepseek_v2.py:267-320`` ``_klx_absorb_bmm``, called at
``:1917`` for ``w_kc`` and ``:2127`` for ``w_vc``). 0.5.23 moved both call sites
into upstream ``deepseek_common/attention_forward_methods/forward_mla.py``
(``:586`` and ``:916-932``) and the fork helper was dropped in the port, so both
absorbed bmms went back to ``aten::bmm``.

What that costs, measured on the 2026-11-12 decode traces
(``/home/zx/script/glm53/traces/{pd_deepep_D,dstack}``, per-step normalized on
``moe_pre_process``/156, 8 ranks). One v-side invocation on 0.5.23, 8 kernels::

    transpose_0213_cluster<float16>       4.0 us  attn_output.transpose(0, 1)
    transpose_021_sdnn_bsp<float16>      25.1 us  w_vc materialization
    cluster_cast_kl3<bfloat16, float16>  27.0 us  w_vc bf16 -> fp16
    transpose_021_sdnn_bsp<float16>      23.2 us  w_vc materialization
    memcpy_1d                            21.5 us  w_vc copy
    xblas::batched_gemm_bsp              16.5 us
    transpose_0213_cluster<float16>       2.9 us  write of the out= view
    memcpy_1d                             2.9 us

0.5.8 ran the same work in three kernels -- ``transpose_0213_cluster`` 3.9 us,
``batched_gemm_bsp`` 15.9 us, ``transpose_0213_cluster`` 2.8 us -- because the
vendor kernel reads the *physical* weight with ``w_trans=True`` and never
materializes the transposed view. The q-side (``:586``) is the same story with an
fp32 round trip instead of a transpose: three ``cluster_cast_kl3`` on the weight
(2.6 + 25.4 + 4.7 us), an fp32 ``batched_gemm_bsp`` (23.0 us against 14.0 us in
0.5.8) and a ``cluster_cast_kl3<float, bfloat16>`` on the result.

Aggregate, 0.5.8 -> 0.5.23:

* ``transpose_021_sdnn_bsp<float16>`` 0.00 -> 162.00/step, 3.894 ms/step, every
  one of them in the 20-30 us band, i.e. all of it weight traffic.
* ``memcpy_1d`` in the 20-25 us band 0.01 -> 80.84/step, 1.708 ms/step.

Both are the weight: 81 layer invocations per step (78 layers + 3 EAGLE draft
replays) x (2 transposes + 1 copy) on the v-side.

Two constraints, quoted from the 0.5.8 helper's own docstring and reproduced
here verbatim in code:

1. ``kunlun_ops.bmm`` ignores the stride of ``x`` / ``out`` and reads/writes a
   dense ``[b, m, k]`` / ``[b, m, n]``. Feeding it a transposed view computes
   silently wrong values (rel err ~10), so ``x`` is made contiguous and ``out``
   is a freshly allocated dense tensor (or the caller's ``out=`` when that one
   is already dense).
2. ``w_kc`` / ``w_vc`` are transposed views over a physically contiguous
   ``[b, n, k]`` buffer (``deepseek_weight_loader.py:779-786``:
   ``w_kc.transpose(1, 2).contiguous().transpose(1, 2)`` and
   ``w_vc.contiguous().transpose(1, 2)``), so the physical tensor goes in with
   ``w_trans=True`` and no weight copy happens at all.

dtype interaction with the fp16 absorbed-weight restore
(``hooks/models/deepseek_v4_precision.py:180-244``): the vendor kernel requires
``w.dtype is x.dtype``, exactly as the 0.5.8 gate did. Activations follow
``--dtype`` (fp16 for this deployment), so this hook only engages once that
restore has put ``w_kc`` / ``w_vc`` in fp16; while they are upstream's hardcoded
bf16 the gate below falls back to ``aten::bmm`` and nothing changes. fp16 is the
dtype that must win: ``kunlun_ops.bmm`` serves bf16/fp16/fp32
(``kunlun_ops/_gemm.py:126-152``) so bf16 would also be native here, but the
xblas JIT cache has no bf16 ``bmm`` instantiation at all, so a bf16 weight would
drag the *activations* back through fp32 in every other consumer. 0.5.8 served
fp16 end to end.

``KLX_MLA_BMM=0`` restores ``aten::bmm`` (same env name and default as
``deepseek_v2.py:203``).
"""

from __future__ import annotations

import contextlib
import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_KLX_MLA_BMM = get_bool_env_var("KLX_MLA_BMM", "true")
try:
    import kunlun_ops as _klx_ops
except Exception:  # pragma: no cover - non-XPU host
    _klx_ops = None
    _KLX_MLA_BMM = False

# The fork restricted itself to the two half types (``deepseek_v2.py:210``);
# fp32 operands mean some other dtype bug upstream, not a path to accelerate.
_KLX_OK_DTYPES = (torch.bfloat16, torch.float16)
_KLX_BMM_SEEN = set()

# Captured at import, before any redirect: the fallbacks and every non-absorbed
# caller must reach the real op even while the redirect is installed.
_ATEN_BMM = torch.bmm

_PREPARE_TARGET = (
    "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla."
    "DeepseekMLAForwardMixin.forward_absorb_prepare"
)
_CORE_TARGET = (
    "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla."
    "DeepseekMLAForwardMixin.forward_absorb_core"
)

try:
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph as _is_in_piecewise_graph,
    )
except ImportError:  # pragma: no cover - upstream renamed the context again

    def _is_in_piecewise_graph() -> bool:
        return False


def _klx_log(x: torch.Tensor, w: torch.Tensor, msg: str) -> None:
    """One line per distinct route, with the variable token count folded out.

    The fork keyed this on the full shapes, which is fine on a decode node where
    ``m`` is the (fixed) batch size.  On the prefill node ``m`` is the chunked
    token count and changes nearly every call, so a shape-keyed cache degenerates
    into one log line per prefill chunk -- measured 200 lines in a single boot.
    ``m`` is also the one dimension that cannot change which kernel is selected
    (the dtype, the weight layout and the stride pattern decide that), so drop it
    from the key and report it as a range instead.
    """
    key = (x.dim(), tuple(x.shape[1:]), tuple(w.shape), str(x.dtype), x.stride()[1:], msg)
    if key in _KLX_BMM_SEEN:
        return
    _KLX_BMM_SEEN.add(key)
    logger.debug(
        "[klx_absorb_bmm] %s | dtype=%s | x=%s stride=%s | w=%s stride=%s",
        msg,
        x.dtype,
        tuple(x.shape),
        x.stride(),
        tuple(w.shape),
        w.stride(),
    )


def _physical_weight(w: torch.Tensor):
    """-> ``(tensor the kernel may read densely, w_trans)``, or None.

    ``w_trans=True`` is the zero-copy case: the ``[b, k, n]`` logical view sits
    on a dense ``[b, n, k]`` buffer, which is precisely the layout
    ``kunlun_ops.bmm`` wants for its transposed operand.
    """
    if w.is_contiguous():
        return w, False
    wt = w.transpose(1, 2)
    if wt.is_contiguous():
        return wt, True
    return None


def _aten_bmm(x: torch.Tensor, w: torch.Tensor, out):
    if out is None:
        return _ATEN_BMM(x, w)
    return _ATEN_BMM(x, w, out=out)


def absorb_bmm(x: torch.Tensor, w: torch.Tensor, out=None):
    """``[b,m,k] @ [b,k,n] -> [b,m,n]`` on the vendor bf16/fp16 native kernel.

    Port of ``aima_sglang/.../models/deepseek_v2.py:267-305``. Every gate below
    is a fallback to ``aten::bmm``, so a shape or layout this kernel does not
    serve costs one python branch, never a wrong answer.
    """
    if (
        _klx_ops is None
        or x.dim() != 3
        or w.dim() != 3
        or x.dtype is not w.dtype
        or x.dtype not in _KLX_OK_DTYPES
        or x.shape[0] != w.shape[0]
        or x.shape[2] != w.shape[1]
        # An idle DP rank decodes 0 tokens; let aten own the degenerate shapes.
        or x.shape[1] == 0
        or w.shape[2] == 0
    ):
        return _aten_bmm(x, w, out)

    layout = _physical_weight(w)
    if layout is None:
        _klx_log(x, w, "fallback: w layout unsupported")
        return _aten_bmm(x, w, out)
    w_arg, w_trans = layout

    shape = (x.shape[0], x.shape[1], w.shape[2])
    if out is not None and (out.dtype is not x.dtype or tuple(out.shape) != shape):
        _klx_log(x, w, "fallback: out dtype/shape mismatch")
        return _aten_bmm(x, w, out)

    # A dense ``out=`` is written in place (``_eager_bmm_attention`` passes one,
    # ``forward_mla.py:1005``); the strided ``out=`` at ``:926`` needs the dense
    # temporary plus one copy, which is the same kernel count 0.5.8 paid for its
    # ``.transpose(0, 1).flatten(1, 2)`` (``deepseek_v2.py:2126-2129``).
    in_place = out is not None and out.is_contiguous()
    res = out if in_place else torch.empty(shape, dtype=x.dtype, device=x.device)
    _klx_log(x, w, "w_trans=%s in_place_out=%s" % (w_trans, in_place))
    _klx_ops.bmm(x.contiguous(), w_arg, res, x_trans=False, w_trans=w_trans)
    if out is None:
        return res
    if not in_place:
        out.copy_(res)
    return out


def _klx_bmm(input: torch.Tensor, mat2: torch.Tensor, *, out=None):
    """``torch.bmm``-compatible entry used only inside the redirect scope."""
    if _tracing():
        return _aten_bmm(input, mat2, out)
    return absorb_bmm(input, mat2, out=out)


def _tracing() -> bool:
    """Keep the vendor call out of dynamo-traced / piecewise-graph regions.

    0.5.8 made the same exception by hand: ``deepseek_v2.py:2111-2117`` keeps
    ``torch.bmm`` for ``is_in_piecewise_cuda_graph()`` because "torch dynamo
    requires out= op was called where output tensor was non-contiguous".
    ``kunlun_ops.bmm`` is a plain pybind call unless ``ENABLE_XINFER_XRAY=1``
    registers it as a custom op, so a traced region would graph-break on it.
    """
    return torch.compiler.is_compiling() or _is_in_piecewise_graph()


@contextlib.contextmanager
def _absorb_bmm_redirect():
    """Swap ``torch.bmm`` for the duration of one absorbed-MLA method.

    The 0.5.8 fork edited the two call sites directly. 0.5.23 has them inside
    400-line upstream methods, and copying those into the plugin would couple it
    to every future edit of the fp8 / DeepGEMM / born-fp8-q / DCP branches
    around them, so the redirect is scoped to the two methods instead: outside
    them ``torch.bmm`` is untouched, and inside them every non-absorbed bmm
    still lands on ``aten::bmm`` through the dtype/layout gate in
    :func:`absorb_bmm`.
    """
    if not _KLX_MLA_BMM or torch.bmm is not _ATEN_BMM or _tracing():
        yield
        return
    torch.bmm = _klx_bmm
    try:
        yield
    finally:
        torch.bmm = _ATEN_BMM


@plugin_hook(_PREPARE_TARGET, type=HookType.AROUND)
def absorb_prepare_kunlun_bmm(original_fn, *args, **kwargs):
    """q-side absorbed bmm (``forward_mla.py:504``/``:586``/``:1005``)."""
    with _absorb_bmm_redirect():
        return original_fn(*args, **kwargs)


@plugin_hook(_CORE_TARGET, type=HookType.AROUND)
def absorb_core_kunlun_bmm(original_fn, *args, **kwargs):
    """v-side absorbed bmm (``forward_mla.py:926``)."""
    with _absorb_bmm_redirect():
        return original_fn(*args, **kwargs)
