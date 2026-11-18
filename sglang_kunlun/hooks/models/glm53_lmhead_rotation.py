"""Feed the shared lm_head in the basis its rotated weights expect.

`GLM-5.3-W4A8-INT8-Dynamic` stores `lm_head` in a QuaRot-R1 rotated basis:
the weight satisfies ``w_rot = w_fp16 @ D`` with ``D`` the block-diagonal
normalised Hadamard over hidden_size (6144 = 48 x 128). Verified offline:
``cosine(dequant(w_rot), w_fp16 @ D) = 0.99587`` -- quantisation noise, i.e. the
block structure and scaling are exact, and ``cosine(x@D @ dequant(w_rot).T,
x @ w_fp16.T) = 0.99585`` with an 8/8 argmax match.

So a model whose *other* weights are unrotated can still use the int8 lm_head,
as long as the hidden state is rotated on the way in. ``D`` is symmetric and
orthogonal (``H = H^T``), so the direction does not matter.

Gated behind ``SGLANG_KUNLUN_LMHEAD_HADAMARD=1``; the transform is exactly one
block-diagonal matmul per lm_head call.
"""

import logging
import os

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_KUNLUN_LMHEAD_HADAMARD", "0") not in ("", "0", "false", "False")
_BLOCK = int(os.environ.get("SGLANG_KUNLUN_LMHEAD_HADAMARD_BLOCK", "128"))
_cache = {}


def _hadamard(n: int) -> torch.Tensor:
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n**0.5


def _device_matrix(device, dtype, hidden: int) -> torch.Tensor:
    key = (str(device), dtype, hidden)
    D = _cache.get(key)
    if D is None:
        n = hidden // _BLOCK
        if n * _BLOCK != hidden:
            raise ValueError(
                f"hidden_size {hidden} is not a multiple of the rotation block {_BLOCK}"
            )
        D = torch.block_diag(*([_hadamard(_BLOCK)] * n)).to(device=device, dtype=dtype)
        _cache[key] = D
        logger.info(
            "sglang-kunlun: lm_head rotation ready, %d blocks of %d", n, _BLOCK
        )
    return D


@plugin_hook(
    "sglang.srt.models.deepseek_v2.DeepseekV2ForCausalLM.__init__",
    type=HookType.AROUND,
)
def rotate_lm_head_input(original, self, *args, **kwargs):
    """Wrap the instance's `logits_processor` so the head sees the rotated basis.

    Upstream has no `logits_processor` attribute on the class (it is set per
    instance), so the wrap has to happen after construction.
    """
    result = original(self, *args, **kwargs)
    if not _ENABLED:
        return result
    proc = getattr(self, "logits_processor", None)
    if proc is None or getattr(proc, "_kunlun_rotated", False):
        return result
    inner = proc

    def rotated_logits_processor(input_ids, hidden_states, lm_head, forward_batch):
        if isinstance(hidden_states, torch.Tensor) and hidden_states.numel():
            D = _device_matrix(
                hidden_states.device, hidden_states.dtype, hidden_states.shape[-1]
            )
            hidden_states = hidden_states @ D
        return inner(input_ids, hidden_states, lm_head, forward_batch)

    rotated_logits_processor._kunlun_rotated = True
    self.logits_processor = rotated_logits_processor
    return result


# --- int8 lm_head weight substitution ---------------------------------------
#
# The checkpoint that ships the rotated lm_head is on one node only, so the
# weight is extracted to a standalone shard (`SGLANG_KUNLUN_LMHEAD_INT8_FILE`)
# that both nodes mount. Loading it here keeps `--model-path` on the normal
# fp16 checkpoint, so nothing else in the loader changes and a partially
# present model directory can never surprise the decode node.

def _load_int8_lm_head(path):
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        weight = f.get_tensor("lm_head.weight")
        scale = f.get_tensor("lm_head.weight_scale")
    return weight, scale


@plugin_hook(
    "sglang.srt.models.deepseek_v2.DeepseekV2ForCausalLM.load_weights",
    type=HookType.AROUND,
)
def load_int8_lm_head(original, self, weights, *args, **kwargs):
    """Swap the fp16 lm_head for the rotated int8 one, if configured."""
    path = os.environ.get("SGLANG_KUNLUN_LMHEAD_INT8_FILE", "")
    if not path or not os.path.exists(path):
        return original(self, weights, *args, **kwargs)

    head = getattr(self, "lm_head", None)
    others = [(n, t) for n, t in weights if not n.startswith("lm_head.")]
    # Upstream must still see a weight for every parameter it expects; the
    # fp16 tensor is skipped and overwritten below.
    result = original(self, iter(others), *args, **kwargs)
    if head is None:
        return result
    weight, scale = _load_int8_lm_head(path)
    w = head.weight
    packed = weight.to(w.device)
    if os.environ.get("SGLANG_KUNLUN_LMHEAD_INT8_GEMM", "1") not in ("", "0"):
        self.lm_head = Int8RotatedHead(packed, scale.to(w.device).flatten())
        self.lm_head.to(dtype=w.dtype if not w.dtype.is_floating_point else w.dtype)
        logger.info(
            "sglang-kunlun: lm_head -> Int8RotatedHead (%s, block scales)",
            tuple(packed.shape),
        )
        return result
    w.data = packed
    head.weight_scale = scale.to(w.device).float().flatten()
    head._kunlun_int8_rotated = True
    logger.info(
        "sglang-kunlun: lm_head replaced with the rotated int8 shard %s "
        "(%s, scale %s)",
        path,
        tuple(packed.shape),
        tuple(head.weight_scale.shape),
    )
    return result


class Int8RotatedHead(torch.nn.Module):
    """Rotated-basis int8 lm_head, computed with the block-scale int8 GEMM.

    The int8 GEMM (`gemm_I8_I8_bf16_nt`) wants int32 accumulation and block
    scales, while upstream hands the head a float tensor -- so the head owns the
    whole chain: rotate, per-token quantise, GEMM, dequantise. Memory stays at
    the int8 weight (0.95 GB) instead of the fp16 copy (1.90 GB).
    """

    def __init__(self, weight, row_scale, block=128):
        super().__init__()
        import kunlun_ops  # noqa: F401

        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_buffer("row_scale", row_scale.float(), persistent=False)
        self.block = block
        self._block_scale = None

    def _scales(self, k):
        w = self.weight
        v, kk = w.shape
        n_v = (v + self.block - 1) // self.block
        n_k = (kk + self.block - 1) // self.block
        if self._block_scale is None or self._block_scale.shape != (1, n_v, n_k):
            s = self.row_scale.view(1, 1, v)  # per output channel
            pad = n_v * self.block - v
            if pad:
                s = torch.cat([s, s[:, :, :1].expand(1, 1, pad)], dim=2)
            s = s.view(1, n_v, self.block)
            # one k-block-wide slice, replicated across every k block
            s = s.max(dim=2).values.view(1, n_v, 1).expand(1, n_v, n_k).contiguous()
            self._block_scale = s.float()
        return self._block_scale

    def forward(self, x):
        """Run the rotated int8 chain: Hadamard, quantise, int8 GEMM, dequantise."""
        import kunlun_ops

        out_dtype = x.dtype
        # Per-block Hadamard: a dense 6144x6144 matmul is not implemented on this
        # backend (`matmul unimplemented`), and the blocks are independent.
        H = _hadamard(self.block).to(device=x.device, dtype=x.dtype)
        n = x.shape[-1] // self.block
        rows = x.reshape(-1, self.block)  # 2-D matmul only: batched matmul is
        xr = (rows @ H).reshape(*x.shape).contiguous()  # unimplemented here
        m, k = xr.shape
        x8 = torch.empty(m, k, dtype=torch.int8, device=x.device)
        xmax = torch.empty(m, 1, dtype=torch.float32, device=x.device)
        kunlun_ops.quant2d(xr, x8, xmax, force_sdnn=True)
        n_k = (k + self.block - 1) // self.block
        lhs_scale = (xmax.float() / 127.0).expand(m, n_k).contiguous()
        # The vendor reports `int8 matmul requires int32 output tensor` for the
        # bf16 doc'd form, so accumulate in int32 and dequantise ourselves.
        out = torch.empty(m, self.weight.shape[0], dtype=torch.int32, device=x.device)
        kunlun_ops.gemm_I8_I8_bf16_nt(
            (x8, lhs_scale),
            (self.weight.unsqueeze(0), self._scales(k)),
            out,
        )
        logits = out.float() * (xmax.float() / 127.0) * self.row_scale.view(1, -1)
        return logits.to(out_dtype)
