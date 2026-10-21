"""Internal online INT8 quantization patch (BF16 checkpoint -> INT8 at load).

Why INT8 and not FP8
--------------------
Kunlun P800 has no FP8 GEMM (no E4M3 tensor-core path), so the online
``blockwise_fp8`` route that EB5 uses on GPU-B is unusable there.  What the
platform does have is a native INT8 GEMM family::

    kunlun_ops.quant2d(x[m, n] bf16, y_int8[m, n], max[m] fp32, force_sdnn=True)
        # symmetric *per-row* quant.  ``max`` is the raw row abs-max (NOT
        # amax/127): dequant is ``q * max / 127``, see
        # ``_xflashinfer_quant.dequant2d_per_token(is_absmax=True)``.

    xpu_kunlun_ops.matmul(x_int8[m, k], w_int8[n, k], out_bf16[m, n],
                          x_trans=False, w_trans=True, bias=fp32[n],
                          x_pc_max fp32[m, 1], w_pc_max fp32[n, 1])
        # dense W8A8: per-token activation scale, per-output-channel weight
        # scale (the same kernel the plugin's ``w8a8_int8`` hooks use).

    kunlun_ops.m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3(
        lhs=(x_int8[m_sum, k], x_scale fp32[m_sum, 1]),
        rhs=(w_int8[E, n, k], w_scale fp32[E, n, 1]),
        out=bf16[m_sum, n], m_indices=counts, lod_force_sdnn=False)
        # MoE experts, per-token x / per-channel w.  ``m_indices`` in the _v3
        # variant is the per-expert *count* and the token rows must already be
        # grouped by expert (which is exactly what DeepEP dispatch produces).

Granularity: measured, not assumed
----------------------------------
``kunlun_ops`` documents a 128x128 block-scale layout for the ``I8_I8``
wrappers, but this build (kunlun_ops 0.1.227) does **not** implement it:

  * ``matmul`` with ``w_pc_max`` shaped ``[n/128, k/128]`` (and every other
    block layout tried: ``[n,k/128]``, ``[n/128,k]``, transposed) returns
    NaN or 20-150% error instead of the 30x-smaller block-quant error;
  * ``m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3`` with
    ``w_scale [E, n/128, k/128]`` returns NaN for every shape tried, while the
    same call with ``[E, n, 1]`` matches to 0.3%.

So the *only* scheme this hardware can run is per-token activations +
per-output-channel weights.  That is what this module implements; the
128x128 helpers below are kept because they are the reference for how much
accuracy a future block-scale kernel would buy (see
``tool/int8_quant_probe.py`` / ``test_eb5_int8_block_quant*.py``):

    scheme (real EB5 weights, m=8)         rel_fro
    per-tensor x + per-tensor w            2.75%
    per-token  x + per-channel w (this)    1.19 - 1.68%
    128-block w + per-token x              1.34 - 1.61%
    128-block w + 128-block x              0.97 - 1.05%

Enable with ``--quantization blockwise_int8`` when ``EB5_INT8_ONLINE=1`` (the
alias only exists under that gate, so the serialized-checkpoint
``blockwise_int8`` path is untouched otherwise), or with the never-shadowing
``--quantization blockwise_int8_online``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import ceil_align, ceil_div, set_weight_attrs
from sglang.srt.utils.common import log_info_on_rank0

logger = logging.getLogger(__name__)

BLOCKWISE_INT8_NAME = "blockwise_int8"
# Alias that cannot collide with the (checkpoint-serialized) upstream
# ``blockwise_int8`` entry; always registered so the method is reachable even
# when EB5_INT8_ONLINE is off.
BLOCKWISE_INT8_ONLINE_NAME = "blockwise_int8_online"


def _flag(name: str, default: str = "false") -> bool:
    """Read a boolean switch from the environment (plugin-side, no fork import)."""

    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}
WEIGHT_BLOCK_SIZE = [128, 128]

# Kernel scale convention: raw abs-max, the GEMM divides by 127 internally.
SCALE_IS_ABSMAX = True
_SCALE_EPS = 1e-8

_LINEAR_LOG_DONE = False
_MOE_LOG_DONE = False


# ---------------------------------------------------------------------------
# Pure-torch quantizers (CPU-capable; also the numeric reference for tests)
# ---------------------------------------------------------------------------


def quantize_weight_perchannel_int8(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel (per-row-of-``[n, k]``) INT8 quantization.

    Returns ``(int8 weight [n, k], fp32 scale [n, 1])``; ``scale`` is the raw
    row abs-max, i.e. dequant is ``q * scale / 127``.
    """
    assert weight.dim() == 2, weight.shape
    w = weight.detach().float()
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(_SCALE_EPS)
    q = torch.round(w * (127.0 / amax)).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), amax.contiguous()


def dequantize_weight_perchannel_int8(
    qweight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    factor = scale if SCALE_IS_ABSMAX else scale * 127.0
    return qweight.float() * (factor / 127.0)


def quantize_act_pertoken_int8_torch(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token INT8 quantization (pure torch)."""
    assert x.dim() == 2, x.shape
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(_SCALE_EPS)
    q = torch.round(xf * (127.0 / amax)).clamp(-127, 127).to(torch.int8)
    return q, amax


def dequantize_act_pertoken_int8(qx: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    factor = scale if SCALE_IS_ABSMAX else scale * 127.0
    return qx.float() * (factor / 127.0)


# -- 128x128 block quantizers ------------------------------------------------
# Not runnable on this kunlun_ops build (see the module docstring); kept as the
# reference for the accuracy a block-scale kernel would give, and used by
# tool/int8_quant_probe.py + test_eb5_int8_block_quant.py.


def _block_view(x: torch.Tensor) -> torch.Tensor:
    n, k = x.shape
    assert n % 128 == 0 and k % 128 == 0, (n, k)
    return x.view(n // 128, 128, k // 128, 128)


def quantize_weight_blockwise_int8(
    weight: torch.Tensor, block_size: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric 128x128 block quantization of a BF16/FP16 weight.

    Returns ``(int8 weight [n, k], fp32 block scale [n/128, k/128])`` with the
    same raw-abs-max scale convention as the kernels.
    """
    block_size = block_size or WEIGHT_BLOCK_SIZE
    assert list(block_size) == WEIGHT_BLOCK_SIZE, block_size
    assert weight.dim() == 2, weight.shape
    w = weight.detach().float()
    n, k = w.shape
    if n % 128 or k % 128:
        pad_n = ceil_align(n, 128) - n
        pad_k = ceil_align(k, 128) - k
        w = torch.nn.functional.pad(w, (0, pad_k, 0, pad_n))
    nb, kb = w.shape[0] // 128, w.shape[1] // 128
    view = _block_view(w)
    amax = view.abs().amax(dim=(1, 3)).clamp_min(_SCALE_EPS)
    scale = amax if SCALE_IS_ABSMAX else amax / 127.0
    q = torch.round(view * (127.0 / amax).view(nb, 1, kb, 1)).clamp(-127, 127)
    q = q.to(torch.int8).reshape(nb * 128, kb * 128)[:n, :k].contiguous()
    return q, scale.contiguous()


def dequantize_weight_blockwise_int8(
    qweight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    q = qweight.float()
    n, k = q.shape
    nb, kb = scale.shape
    view = q.view(nb, 128, kb, 128)
    factor = scale if SCALE_IS_ABSMAX else scale * 127.0
    return (
        (view * factor.view(nb, 1, kb, 1)).reshape(nb * 128, kb * 128)[:n, :k]
        / 127.0
    )


def quantize_act_blockwise_int8_torch(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-(row, 128-K-block) symmetric INT8 quantization, pure torch."""
    assert x.dim() == 2, x.shape
    m, k = x.shape
    if k % block_size:
        x = torch.nn.functional.pad(x, (0, ceil_align(k, block_size) - k))
    k = x.shape[1]
    kb = k // block_size
    xb = x.float().reshape(m * kb, block_size)
    amax = xb.abs().amax(dim=1).clamp_min(_SCALE_EPS)
    q = torch.round(xb * (127.0 / amax).unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    scale = amax if SCALE_IS_ABSMAX else amax / 127.0
    return q.view(m, k), scale.view(m, kb)


def dequantize_act_blockwise_int8(
    qx: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``[m, k]`` int8 + ``[m, kb]`` scale -> ``[m, kb, 128]`` float32."""
    m, k = qx.shape
    kb = scale.shape[1]
    factor = scale if SCALE_IS_ABSMAX else scale * 127.0
    return qx.float().view(m, kb, 128) * (factor / 127.0).unsqueeze(2)


def block_reference_matmul(
    x: torch.Tensor, qweight: torch.Tensor, wscale: torch.Tensor
) -> torch.Tensor:
    """float32 model of a *block-scale* W8A8 INT8 GEMM (for error analysis)."""
    qx, xs = quantize_act_blockwise_int8_torch(x)
    xd = dequantize_act_blockwise_int8(qx, xs)
    nb, kb = wscale.shape
    wd = dequantize_weight_blockwise_int8(qweight, wscale).view(nb * 128, kb, 128)
    return torch.einsum("mkb,nkb->mn", xd, wd)


def perchannel_reference_matmul(
    x: torch.Tensor, qweight: torch.Tensor, wscale: torch.Tensor
) -> torch.Tensor:
    """float32 model of the scheme actually used (per-channel W, per-token A)."""
    qx, xs = quantize_act_pertoken_int8_torch(x)
    return dequantize_act_pertoken_int8(qx, xs) @ dequantize_weight_perchannel_int8(
        qweight, wscale
    ).T


# ---------------------------------------------------------------------------
# Kunlun kernels
# ---------------------------------------------------------------------------


def quantize_act_pertoken_int8(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token INT8 quantize activations with ``kunlun_ops.quant2d``."""
    import kunlun_ops

    assert x.dim() == 2, x.shape
    xc = x if x.is_contiguous() else x.contiguous()
    q = torch.empty(xc.shape, dtype=torch.int8, device=x.device)
    s = torch.empty((xc.shape[0], 1), dtype=torch.float32, device=x.device)
    kunlun_ops.quant2d(xc, q, s, force_sdnn=True)
    return q, s


def int8_pertoken_linear(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense W8A8 INT8 GEMM into ``out`` (per-token x, per-channel w)."""
    import kunlun_ops

    kunlun_ops.xpu_kunlun_ops.matmul(
        x_q,
        weight_q,
        out,
        False,
        True,
        1.0,
        0.0,
        None if bias is None else bias.float(),
        None,
        None,
        x_s,
        weight_s,
    )
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Projections that must stay BF16 even when they are ``LinearBase``:
#   * ``wo_a`` -- consumed directly by the absorb GEMM in
#     ``deepseek_v4._apply_wo_a_bf16_matmul`` (a bmm over the o-group axis that
#     reads ``.weight`` itself); quantizing it would need a batched INT8 kernel.
#   * ``lm_head`` / ``embed_tokens`` -- vocabulary-sized; quantizing them buys
#     little and costs accuracy on the logits.
#   * the DSA indexer / MHC heads -- tiny, and their kernels are fused around
#     bf16 (top-k ordering would also flip on ties at int8 precision).
_ALWAYS_BF16_SUBSTRINGS = [
    "wo_a",
    "lm_head",
    "embed_tokens",
    "indexer",
    "mhc",
]


class InternalBlockwiseInt8Config(QuantizationConfig):
    """Online INT8 for BF16/FP16 checkpoints (per-token A / per-channel W)."""

    def __init__(
        self,
        ignored_layers: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        moe_only: Optional[bool] = None,
    ) -> None:
        super().__init__()
        # Kept for tooling that reads ``weight_block_size`` off a quant config
        # (e.g. shape predicates); the *executed* granularity is per-channel.
        self.weight_block_size = list(WEIGHT_BLOCK_SIZE)
        self.activation_scheme = "dynamic"
        self.blockwise_int8_online = True
        self.is_checkpoint_int8_serialized = False
        self.weight_granularity = "channel"
        ignored = list(ignored_layers or [])
        if moe_only is None:
            moe_only = _flag("EB5_INT8_MOE_ONLY")
        self.moe_only = moe_only
        if moe_only:
            # Everything that is not the routed-expert block stays BF16. Keep
            # the substrings specific enough that ``mlp.experts`` is not hit.
            ignored.extend(
                [
                    "self_attn",
                    "mlp.gate_up_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                    "mlp.shared_experts",
                    "mlp.fc1_latent_proj",
                    "mlp.fc2_latent_proj",
                    "main_proj",
                ]
            )
        self.ignored_layers = list(
            dict.fromkeys(ignored + list(_ALWAYS_BF16_SUBSTRINGS))
        )
        if packed_modules_mapping:
            self.packed_modules_mapping = dict(packed_modules_mapping)

    @classmethod
    def get_name(cls) -> str:
        return BLOCKWISE_INT8_NAME

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Kunlun XPU reports no CUDA capability; the check is skipped there.
        return 0

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "InternalBlockwiseInt8Config":
        config = dict(config)
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        packed = cls.get_from_keys_or(config, ["packed_modules_mapping"], {}) or {}
        return cls(ignored_layers=ignored_layers, packed_modules_mapping=packed)

    # -- layer dispatch -----------------------------------------------------

    def _is_ignored(self, prefix: str) -> bool:
        for pat in self.ignored_layers:
            if pat and pat in prefix:
                return True
        return False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            if self._is_ignored(prefix):
                from sglang.srt.layers.quantization.unquant import (
                    UnquantizedFusedMoEMethod,
                )

                return UnquantizedFusedMoEMethod()
            return Int8MoEMethod(self)

        if isinstance(layer, LinearBase):
            if self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            block_n, block_k = self.weight_block_size
            in_size = getattr(layer, "input_size", None)
            out_size = getattr(layer, "output_size", None)
            if (
                in_size is None
                or out_size is None
                or in_size % block_k
                or out_size % block_n
            ):
                # The INT8 GEMM wants 128-aligned GEMM dims (cheap to honour,
                # and a mis-aligned layer is not worth a separate kernel path).
                logger.info(
                    "blockwise_int8: keeping %s in BF16 (in=%s out=%s not both "
                    "divisible by %s).",
                    prefix,
                    in_size,
                    out_size,
                    self.weight_block_size,
                )
                return UnquantizedLinearMethod()
            method = Int8LinearMethod(self)
            method.prefix = prefix
            return method
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


# ---------------------------------------------------------------------------
# Linear method
# ---------------------------------------------------------------------------


class Int8LinearMethod(LinearMethodBase):
    """W8A8 INT8 linear with online weight quantization at load time."""

    def __init__(self, quant_config: InternalBlockwiseInt8Config):
        self.quant_config = quant_config
        self.prefix = ""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.blockwise_int8_online = True

        # BF16 placeholder, quantized in process_weights_after_loading.  Every
        # dense layer's placeholder is alive until the post-load pass runs, so
        # on the full 43-layer model they add up (8.1 B params = 16.3 GiB/rank
        # at 4 ranks) *on top of* the already-quantized experts -- which is what
        # OOMs a 4-card full-net load.  Staging them on CPU keeps the peak at
        # int8_all + bf16_one_layer, exactly like the MoE path.
        # ``kv_b_proj`` feeds the MLA "absorb" buffers (w_kc/w_vc), which
        # ``post_load_weights`` builds *before* the quant pass runs.  A
        # CPU-resident placeholder there yields CPU absorb buffers, so that one
        # projection always stays on the device (it is 29 MB -- irrelevant for
        # the memory budget).
        stage_on_cpu = _flag("EB5_INT8_LINEAR_CPU_STAGING") and (
            "kv_b_proj" not in self.prefix
        )
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
                device=torch.device("cpu") if stage_on_cpu else None,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        if stage_on_cpu:
            weight._cpu_staged = True
        layer.register_parameter("weight", weight)

        # ``[n, 1]`` per-output-channel scale, filled after loading.  Deliberately
        # a plain Parameter: the kernel only ever reads it, and per-row scales do
        # not fit the column/row sharding helpers (/update_weights_from_tensor
        # therefore does not re-shard it -- see the module docstring).
        scale = Parameter(
            torch.empty(output_size_per_partition, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", scale)
        layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        global _LINEAR_LOG_DONE
        weight = layer.weight.data
        src_dtype = weight.dtype
        if src_dtype == torch.int8:
            return
        if weight.device.type == "cpu":
            # CPU-staged placeholder: bring this one layer over, quantize, and
            # let the BF16 copy die with the local reference.
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weight = weight.to(dev)
            layer.weight.data = weight
        qweight, scale = quantize_weight_perchannel_int8(weight)
        layer.weight = Parameter(qweight, requires_grad=False)
        layer.weight_scale.data = scale
        layer.input_scale = None
        # The INT8 GEMM takes an fp32 bias (same requirement the plugin's
        # w8a8_int8 hook satisfies by rebuilding the parameter in fp32).
        if getattr(layer, "bias", None) is not None and layer.bias.dtype != torch.float32:
            layer.bias.data = layer.bias.data.float()
        if not _LINEAR_LOG_DONE:
            _LINEAR_LOG_DONE = True
            log_info_on_rank0(
                logger,
                "blockwise_int8: online linear quantization executed "
                f"({src_dtype} -> int8, per-output-channel scale); "
                "activations are quantized per token at run time.",
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            raise NotImplementedError(
                "blockwise_int8 does not accept pre-quantized (tuple) inputs; "
                "disable the fused rmsnorm+quant producer for this run."
            )
        if x.shape[0] == 0:
            # DP attention leaves idle ranks with zero tokens; quant2d has no
            # meaningful row scale for an empty batch.
            return torch.empty(
                (0, layer.weight.shape[0]),
                dtype=layer.orig_dtype,
                device=x.device,
            )
        x_q, x_s = quantize_act_pertoken_int8(x)
        out = torch.empty(
            (x_q.shape[0], layer.weight.shape[0]),
            dtype=layer.orig_dtype,
            device=x_q.device,
        )
        if bias is None:
            bias = getattr(layer, "bias", None)
        return int8_pertoken_linear(
            x_q, x_s, layer.weight, layer.weight_scale, out, bias
        )


# ---------------------------------------------------------------------------
# MoE method
# ---------------------------------------------------------------------------


class Int8MoEMethod(FusedMoEMethodBase):
    """Routed experts: online INT8 quantization + grouped INT8 GEMM.

    Weight scales are named/shaped for the kernel and for the ``sglang_kunlun``
    DeepEPMoE hook, which keys off ``w2_weight.dtype == torch.int8`` and reads
    ``w13_weight_scale`` / ``w2_weight_scale`` as ``[E, n, 1]``.  With DeepEP
    that hook then runs the whole expert block (the dispatcher already emits
    per-token INT8 activations), so no plugin change is needed for the MoE.
    """

    def __init__(self, quant_config: InternalBlockwiseInt8Config):
        self.quant_config = quant_config
        self.block_quant = False
        self.weight_block_size = None
        self.moe_runner_config = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # CPU staging keeps the BF16 placeholder off the device: the experts are
        # ~2/3 of the whole model, so an on-device BF16 placeholder cannot
        # coexist with the rest of the weights on a 96 GB card.  post_load.py
        # skips ``_cpu_staged`` tensors when staging for the post-load hook.
        stage_on_cpu = _flag("EB5_INT8_MOE_CPU_STAGING")
        dev = torch.device("cpu") if stage_on_cpu else None

        w13_n = 2 * intermediate_size_per_partition
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, w13_n, hidden_size, dtype=params_dtype, device=dev
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
                device=dev,
            ),
            requires_grad=False,
        )
        if stage_on_cpu:
            w13_weight._cpu_staged = True
            w2_weight._cpu_staged = True
        for name, param in (("w13_weight", w13_weight), ("w2_weight", w2_weight)):
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

        w13_scale = torch.nn.Parameter(
            torch.ones(num_experts, w13_n, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        layer.register_parameter("w2_weight_scale", w2_scale)
        # Aliases for callers/tools that assume the fp8 naming.
        layer.w13_weight_scale_inv = w13_scale
        layer.w2_weight_scale_inv = w2_scale
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.CHANNEL.value
        for param in (w13_scale, w2_scale):
            set_weight_attrs(param, scale_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config) -> None:
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        global _MOE_LOG_DONE
        w13_src = layer.w13_weight.data
        if w13_src.dtype == torch.int8:
            return
        src_dtype = w13_src.dtype
        w2_src = layer.w2_weight.data
        on_cpu = w13_src.device.type == "cpu"
        # ``cuda`` is the device the XPU runtime masquerades as.  Falls back to
        # CPU only when there is no device at all (unit tests), where quantizing
        # on the host is exact and the result is never used for a forward.
        gpu = w13_src.device
        if on_cpu:
            gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_experts = w13_src.shape[0]

        w13_q_all = w13_s_all = w2_q_all = w2_s_all = None
        for e in range(num_experts):
            w13 = w13_src[e]
            w2 = w2_src[e]
            if on_cpu:
                w13 = w13.to(gpu, non_blocking=False)
                w2 = w2.to(gpu, non_blocking=False)
            q13, s13 = quantize_weight_perchannel_int8(w13)
            q2, s2 = quantize_weight_perchannel_int8(w2)
            if w13_q_all is None:
                w13_q_all = torch.empty(
                    w13_src.shape, dtype=torch.int8, device=q13.device
                )
                w2_q_all = torch.empty(w2_src.shape, dtype=torch.int8, device=q2.device)
                w13_s_all = torch.empty(
                    (num_experts, *s13.shape), dtype=torch.float32, device=s13.device
                )
                w2_s_all = torch.empty(
                    (num_experts, *s2.shape), dtype=torch.float32, device=s2.device
                )
            w13_q_all[e].copy_(q13)
            w2_q_all[e].copy_(q2)
            w13_s_all[e].copy_(s13)
            w2_s_all[e].copy_(s2)
            del q13, s13, q2, s2, w13, w2
        del w13_src, w2_src

        # ``.data`` assignment keeps the FusedMoE weight_loader attached; a new
        # Parameter would drop it and silently break /update_weights_from_tensor.
        layer.w13_weight.data = w13_q_all
        layer.w2_weight.data = w2_q_all
        if getattr(layer, "w13_weight_scale", None) is not None:
            layer.w13_weight_scale.data = w13_s_all
            layer.w2_weight_scale.data = w2_s_all
        else:
            layer.w13_weight_scale = Parameter(w13_s_all, requires_grad=False)
            layer.w2_weight_scale = Parameter(w2_s_all, requires_grad=False)
        layer.w13_weight_scale_inv = layer.w13_weight_scale
        layer.w2_weight_scale_inv = layer.w2_weight_scale
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        if not _MOE_LOG_DONE:
            _MOE_LOG_DONE = True
            log_info_on_rank0(
                logger,
                "blockwise_int8: online MoE quantization executed "
                f"({num_experts} experts, {src_dtype} -> int8, "
                f"w13_scale={tuple(w13_s_all.shape)}, "
                f"w2_scale={tuple(w2_s_all.shape)}).",
            )

    # -- dense (non-DeepEP) path -------------------------------------------

    def apply(self, layer: torch.nn.Module, dispatch_output) -> Any:
        """Standard-dispatch W8A8 INT8 MoE (no all-to-all).

        DeepEP runs go through ``sglang_kunlun``'s DeepEPMoE hook instead; this
        path exists so the method is usable (and testable) on a single device.
        """
        import kunlun_ops
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        num_experts = layer.w13_weight.shape[0]
        if num_tokens == 0:
            return StandardCombineInput(hidden_states=x.new_empty(x.shape))

        flat_ids = topk_ids.reshape(-1)
        counts = torch.bincount(flat_ids, minlength=num_experts).to(torch.int32)
        order = torch.argsort(flat_ids, stable=True)
        x_exp = x.index_select(0, order // top_k)

        x_q, x_s = quantize_act_pertoken_int8(x_exp)
        gateup = torch.empty(
            (x_exp.shape[0], layer.w13_weight.shape[1]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        kunlun_ops.m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3(
            (x_q, x_s), (layer.w13_weight, layer.w13_weight_scale), gateup, counts
        )

        act = self._activate(gateup)
        a_q, a_s = quantize_act_pertoken_int8(act)
        down = torch.empty(
            (act.shape[0], layer.w2_weight.shape[1]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        kunlun_ops.m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3(
            (a_q, a_s), (layer.w2_weight, layer.w2_weight_scale), down, counts
        )

        # Scatter back: row r of ``down`` belongs to token r // top_k.
        w = topk_weights.reshape(-1)[order].to(torch.float32).unsqueeze(1)
        out = torch.zeros(
            (num_tokens, down.shape[1]), dtype=torch.float32, device=down.device
        )
        out.index_add_(0, order // top_k, down.float() * w)
        rsf = getattr(self.moe_runner_config, "routed_scaling_factor", None)
        if rsf is not None and rsf != 1.0:
            out.mul_(rsf)
        return StandardCombineInput(hidden_states=out.to(x.dtype))

    def _activate(self, gateup: torch.Tensor) -> torch.Tensor:
        cfg = self.moe_runner_config
        beta = getattr(cfg, "gemm1_alpha", None)
        clamp = getattr(cfg, "gemm1_clamp_limit", None)
        gate, up = gateup.chunk(2, dim=-1)
        if beta is not None:
            # SiTU: beta * tanh(g/beta) * sigmoid(g) * up
            gate = float(beta) * torch.tanh(gate / float(beta)) * torch.sigmoid(gate)
            if clamp is not None and float(clamp) > 0:
                up = up.clamp(-float(clamp), float(clamp))
            return gate * up
        return torch.nn.functional.silu(gate) * up


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_blockwise_int8_patch() -> None:
    """Register the online INT8 method(s).

    ``blockwise_int8_online`` is always available (no name collision with the
    serialized-checkpoint ``blockwise_int8``) so a run can opt in on the command
    line alone; the ``blockwise_int8`` alias is only remapped when
    ``EB5_INT8_ONLINE=1``, which keeps the upstream checkpoint-driven behaviour
    untouched for every other run.
    """
    import sglang.srt.layers.quantization as quantization
    import sglang.srt.server_args as server_args

    quantization.BASE_QUANTIZATION_METHODS[BLOCKWISE_INT8_ONLINE_NAME] = (
        InternalBlockwiseInt8Config
    )
    quantization.QUANTIZATION_METHODS[BLOCKWISE_INT8_ONLINE_NAME] = (
        InternalBlockwiseInt8Config
    )
    if BLOCKWISE_INT8_ONLINE_NAME not in server_args.QUANTIZATION_CHOICES:
        server_args.QUANTIZATION_CHOICES.append(BLOCKWISE_INT8_ONLINE_NAME)
    if hasattr(server_args, "SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES"):
        server_args.SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES = (
            server_args.QUANTIZATION_CHOICES
        )

    if not _flag("EB5_INT8_ONLINE"):
        logger.info(
            "blockwise_int8: EB5_INT8_ONLINE is off; only %s is registered "
            "(the serialized-checkpoint blockwise_int8 path is untouched).",
            BLOCKWISE_INT8_ONLINE_NAME,
        )
        return

    quantization.BASE_QUANTIZATION_METHODS[BLOCKWISE_INT8_NAME] = (
        InternalBlockwiseInt8Config
    )
    quantization.QUANTIZATION_METHODS[BLOCKWISE_INT8_NAME] = (
        InternalBlockwiseInt8Config
    )
    if BLOCKWISE_INT8_NAME not in server_args.QUANTIZATION_CHOICES:
        server_args.QUANTIZATION_CHOICES.append(BLOCKWISE_INT8_NAME)
    server_args.SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES = (
        server_args.QUANTIZATION_CHOICES
    )
    log_info_on_rank0(
        logger,
        "blockwise_int8: EB5_INT8_ONLINE=1, --quantization blockwise_int8 now "
        "means online INT8 (weights per-output-channel quantized at load, "
        "activations per token at run time; 128x128 block scales are not "
        "supported by this kunlun_ops build). scope="
        + ("moe-only" if _flag("EB5_INT8_MOE_ONLY") else "moe+linear")
        + ", moe_cpu_staging="
        + ("on" if _flag("EB5_INT8_MOE_CPU_STAGING") else "off"),
    )


def apply() -> list:
    """适配层入口：注册在线 INT8 量化方法（与其余 ``eb5_*_adapt`` 同签名）。

    必须在 ``prepare_server_args()`` 之前执行 —— ``--quantization`` 的 argparse
    choices 是解析参数时才构建的。插件入口点 ``eb5_kunlun_patch`` 由
    ``load_plugins()`` 在参数解析之前调用，满足该时序。
    """
    register_blockwise_int8_patch()
    return ["sglang.srt.layers.quantization:blockwise_int8(online)"]
