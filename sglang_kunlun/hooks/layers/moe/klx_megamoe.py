"""KLX/Kunlun MegaMoE: hydrax symmetric-memory fused MoE for w4a8 experts."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import get_moe_a2a_backend
def _resolved_args():
    """The batch-shape inputs, from whichever accessor this tree still serves."""
    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args()
    except (ImportError, RuntimeError):
        from sglang.srt.runtime_context import get_server_args

        return get_server_args()

logger = logging.getLogger(__name__)

# hydrax aligns the per-rank capacity up to this (`mega_moe.TOKEN_ALIGNMENT`).
# Unlike the older hydrax builds, the fused kernels are not specialised on the
# capacity -- it only sizes buffers and is validated at runtime -- so any
# aligned value works and the table below is informational only.
MEGAMOE_TOKEN_ALIGNMENT = 384
MEGAMOE_SUPPORTED_CAPACITIES = (
    384,
    768,
    1024,
    1152,
    2048,
    2304,
    4096,
    4224,
    8192,
    8448,
)

MEGAMOE_GATE_UP_BLOCK = 256

MEGAMOE_W4_SCALE = float(os.environ.get("SGLANG_KLX_MEGAMOE_W4_SCALE", "1.0"))

MEGAMOE_FULLY_OVERLAPPED = (
    os.environ.get("SGLANG_KLX_MEGAMOE_FULLY_OVERLAPPED", "1") != "0"
)

MEGAMOE_SHARED_ALT_STREAM = (
    os.environ.get("SGLANG_KLX_MEGAMOE_SHARED_ALT_STREAM", "1") != "0"
)

MEGAMOE_MAX_CAPACITY = int(
    os.environ.get(
        "SGLANG_KLX_MEGAMOE_MAX_CAPACITY", str(max(MEGAMOE_SUPPORTED_CAPACITIES))
    )
)

_SYMM_BUFFERS: dict = {}


def megamoe_capacity(num_tokens: int) -> Optional[int]:
    """Smallest instantiated per-rank capacity that fits `num_tokens`."""
    for cap in MEGAMOE_SUPPORTED_CAPACITIES:
        if num_tokens <= cap:
            return cap
    return None


def _decode_graph_max_bs(server_args) -> int:
    """Largest captured decode batch, across the two argument schemas."""
    cg = getattr(server_args, "cuda_graph_config", None)
    if cg is not None:
        decode = getattr(cg, "decode", None)
        return int(getattr(decode, "max_bs", 0) or 0)
    return int(getattr(server_args, "cuda_graph_max_bs", 0) or 0)


def _dp_size(server_args) -> int:
    """Attention-DP width, which is how the batch is divided per rank."""
    for name in ("attn_dp_size", "dp_size"):
        value = getattr(server_args, name, None)
        if value:
            return int(value)
    return 1


def megamoe_required_tokens_per_rank(server_args) -> int:
    """Largest per-rank token count this serving config can hand the fused op.

    Derived, never configured: the model config only supplies the other
    symmetric-buffer dimensions (experts / hidden / ffn / topk, read off the MoE
    module at runtime), while the token count is bounded by chunked prefill and
    the decode batch. Note `chunked_prefill_size` has already been divided by
    `dp_size` (`ServerArgs._handle_data_parallelism`), and `cuda_graph_max_bs`
    is per scheduler, i.e. per DP rank.
    """
    draft = server_args.speculative_num_draft_tokens or 1
    required = server_args.chunked_prefill_size or 1
    if not server_args.disable_cuda_graph:
        required = max(required, _decode_graph_max_bs(server_args) * draft)
    if server_args.max_running_requests:
        required = max(
            required,
            -(-server_args.max_running_requests // _dp_size(server_args)) * draft,
        )
    return required


def megamoe_capacity_for(server_args) -> int:
    """Per-rank capacity the symmetric workspace is built with.

    This is a *runtime* bound, not a kernel specialisation: the fused op
    validates `num_tokens <= num_max_tokens_per_rank` and sizes its buffers from
    it, so one capacity has to cover the largest shard a single forward produces
    -- there is no chunking. Derived from `--chunked-prefill-size` and the decode
    batch, aligned to what the op expects, and capped by
    `SGLANG_KLX_MEGAMOE_MAX_CAPACITY` (raise it if a prefill shard is bigger than
    the default; the workspace grows linearly with it).
    """
    required = megamoe_required_tokens_per_rank(server_args)
    capacity = min(required, MEGAMOE_MAX_CAPACITY)
    aligned = -(-capacity // MEGAMOE_TOKEN_ALIGNMENT) * MEGAMOE_TOKEN_ALIGNMENT
    return max(MEGAMOE_TOKEN_ALIGNMENT, aligned)


MEGAMOE_WS_BASE_BYTES = 302_845_952
MEGAMOE_WS_BYTES_PER_TOKEN = 1_431_128


def megamoe_xshmem_size(server_args) -> int:
    """Bytes of `XSHMEM_SYMMETRIC_SIZE` the MegaMoE workspace needs."""

    capacity = megamoe_capacity_for(server_args)
    return int((MEGAMOE_WS_BASE_BYTES + MEGAMOE_WS_BYTES_PER_TOKEN * capacity) * 1.15)


def _get_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
):
    """One symmetric buffer per shape, cached for the process lifetime."""
    from hydrax.xaccelerator.mega_moe_fusion import get_symm_buffer

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    buf = _SYMM_BUFFERS.get(key)
    if buf is None:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if get_is_capture_mode():
            # `get_symm_buffer` allocates out of the xshmem heap and exchanges
            # peer pointers, neither of which belongs inside a graph capture.
            # `megamoe_init_symm_buffer` runs at weight load to avoid this.
            raise RuntimeError(
                "megamoe: the symmetric workspace for experts=%d cap=%d topk=%d "
                "hidden=%d ffn=%d is missing during CUDA graph capture; it must "
                "be allocated before capture."
                % (
                    num_experts,
                    num_max_tokens_per_rank,
                    num_topk,
                    hidden,
                    intermediate_hidden,
                )
            )
        buf = get_symm_buffer(
            group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_shared_experts=0,
            use_fp8_dispatch=False,
        )
        _SYMM_BUFFERS[key] = buf
        logger.info(
            "[megamoe] symm buffer ready: experts=%d cap=%d topk=%d hidden=%d ffn=%d",
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
        )
    return buf


def _block_interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    """[E, 2I, ...] laid out as [gate(I); up(I)] -> 256-row block interleaved.

    Only the row (output-channel) dim is permuted, so this works unchanged on
    the int4-packed weight, whose packing runs along the input dim, and on the
    per-channel scales.
    """
    num_experts, two_i = t.shape[0], t.shape[1]
    half = two_i // 2
    gran = MEGAMOE_GATE_UP_BLOCK
    assert half % gran == 0, f"intermediate_size {half} must be a multiple of {gran}"
    rest = t.shape[2:]
    gate = t[:, :half].reshape(num_experts, half // gran, gran, *rest)
    up = t[:, half:].reshape(num_experts, half // gran, gran, *rest)
    return (
        torch.stack([gate, up], dim=2).reshape(num_experts, two_i, *rest).contiguous()
    )


def build_megamoe_expert_weights(layer) -> None:
    """Rewrite the w4a8 expert weights into the hydrax layout once, after loading.

    `KlxCompressedTensorsW4A8MoEMethod` keeps `w13_weight_packed`
    [E_local, 2I, H // 8] int32 (int4 nibbles packed along the input dim, sign
    bits already flipped to two's complement) with `w13_weight_scale`
    [E_local, 2I, 1] fp32 per-channel amax, and `w2_weight_packed`
    [E_local, H, I // 8] with `w2_weight_scale` [E_local, H, 1]. hydrax wants
    the same tensors viewed as int8 (two nibbles per byte, i.e. [E, n, k // 2],
    the view the Kunlun `m_grouped_gemm_I8_I4_*` GEMMs take as well), the
    gate/up rows block-interleaved, and 2-D scales.

    The interleave is done IN PLACE and `megamoe_l1_weights` aliases the model
    parameters: a second copy of w13 would cost ~27 GB/card on GLM-5.3 w4a8
    ep8 (75 MoE layers x 32 experts x 4096 x 6144 int4). That makes the weights
    unusable for the ordinary packed-int4 MoE path, which is fine because
    megamoe is all-or-nothing for the whole run (the MoE forward asserts it
    applies).
    """
    if getattr(layer, "_megamoe_weights_built", False):
        return

    w13 = layer.w13_weight_packed.data
    w13_scale = layer.w13_weight_scale.data
    w13.copy_(_block_interleave_gate_up(w13))
    w13_scale.copy_(_block_interleave_gate_up(w13_scale))

    layer.megamoe_l1_weights = (
        w13.view(torch.int8),
        (w13_scale.squeeze(-1) * MEGAMOE_W4_SCALE).contiguous(),
    )
    layer.megamoe_l2_weights = (
        layer.w2_weight_packed.data.view(torch.int8),
        (layer.w2_weight_scale.data.squeeze(-1) * MEGAMOE_W4_SCALE).contiguous(),
    )
    layer._megamoe_weights_built = True


def megamoe_init_symm_buffer(experts) -> None:
    """Allocate the symmetric workspace and routing scratch before any forward.

    Decode batches replay from a captured graph, and both `get_symm_buffer`
    (xshmem heap allocation plus a peer-pointer exchange) and the filler-id
    table would otherwise be created on the first forward -- which, with no
    warm-up forward before `CudaGraphRunner` captures, is inside a capture.
    Weight loading is collective on every rank and outside any graph, so the
    allocation is anchored there. One buffer covers all MoE layers: the cache
    key is the shape, which is identical for every layer.
    """
    from sglang.srt.distributed.parallel_state import get_tp_group

    config = experts.moe_runner_config
    capacity = megamoe_capacity_for(_resolved_args())
    _get_symm_buffer(
        get_tp_group().device_group,
        num_experts=experts.num_experts,
        num_max_tokens_per_rank=capacity,
        num_topk=config.top_k,
        hidden=config.hidden_size,
        intermediate_hidden=config.intermediate_size_per_partition,
    )

    _filler_topk_ids(
        capacity,
        config.top_k,
        experts.num_experts,
        experts.w13_weight_packed.device,
    )


def should_use_megamoe(moe) -> bool:
    """Whether this MoE module can take the fused path."""
    return get_moe_a2a_backend().is_megamoe() and getattr(
        moe.experts, "_megamoe_weights_built", False
    )


_FILLER_TOPK_IDS: dict = {}


def _filler_topk_ids(num_rows: int, num_topk: int, num_experts: int, device):
    """Valid, load-spread expert ids used to replace masked (-1) routing slots."""
    key = (num_topk, num_experts, str(device))
    filler = _FILLER_TOPK_IDS.get(key)
    if filler is None or filler.shape[0] < num_rows:
        filler = (
            torch.arange(num_rows * num_topk, device=device, dtype=torch.int64)
            % num_experts
        ).view(num_rows, num_topk)
        _FILLER_TOPK_IDS[key] = filler
    return filler[:num_rows]


def _prepare_inputs(hidden_states, topk_ids, topk_weights, num_experts):
    """Quantize + sanitize routing into the tensors the fused op takes.

    Returns `(x, x_sf, topk_idx, topk_weights)` in the layout Hydra's
    `mega_moe_fusion` doc prescribes: per-token int8 `x`, the **raw absmax** as
    a flat `x_sf` (not absmax/127 -- the gemm kernel carries its own
    normalisation), int64 `topk_idx`, float32 `topk_weights`.
    """
    import kunlun_ops

    x = hidden_states.contiguous()
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    absmax = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    kunlun_ops.quant2d(x, q, absmax, force_sdnn=True)
    # Padding / idle rows are all zero, so their amax is 0; the op divides by
    # the scale, so give it a floor instead of a zero.
    absmax = absmax.clamp_min(1e-6)

    valid = topk_ids >= 0
    filler = _filler_topk_ids(
        topk_ids.shape[0], topk_ids.shape[1], num_experts, topk_ids.device
    )
    safe_ids = torch.where(valid, topk_ids.to(torch.int64), filler)
    safe_weights = torch.where(
        valid, topk_weights.float(), torch.zeros_like(topk_weights, dtype=torch.float32)
    )
    return q, absmax.view(-1), safe_ids, safe_weights


def _idle_routing(hidden_states, hidden_size, num_topk):
    """One all-masked row for a rank with no tokens.

    An IDLE rank still has to enter the fused op (it contains cross-rank
    barriers); staging an explicit masked row keeps what gets dispatched
    deterministic instead of whatever the previous iteration left behind.
    """
    zeros = hidden_states.new_zeros((1, hidden_size))
    ids = torch.full((1, num_topk), -1, dtype=torch.int32, device=zeros.device)
    weights = torch.zeros((1, num_topk), dtype=torch.float32, device=zeros.device)
    return zeros, ids, weights


def _top_k_of(moe) -> int:
    """Routed experts per token."""
    config = getattr(moe.topk, "topk_config", None)
    return config.top_k if config is not None else moe.topk.top_k


_MAX_TOKENS_SEEN = 0


def _log_max_tokens(num_tokens: int, capacity: int) -> None:
    """Log the largest shard a fused call has received so far.

    One line per growth, so a run reports the true per-rank prefill shard --
    which is what the capacity has to cover -- without logging every call.
    """
    global _MAX_TOKENS_SEEN
    if num_tokens <= _MAX_TOKENS_SEEN:
        return
    _MAX_TOKENS_SEEN = num_tokens
    logger.info(
        "[megamoe] largest fused call so far: num_tokens=%d (per-rank capacity %d%s)",
        num_tokens,
        capacity,
        "" if num_tokens <= capacity else " -- OVER CAPACITY",
    )


def _run_fused(moe, hidden_states, topk_ids, topk_weights, num_tokens):
    """The fused op, with its symmetric-memory workspace. Model-agnostic."""
    from hydrax.xaccelerator import mega_moe_fusion

    fused_op = (
        mega_moe_fusion.int8_int4_mega_moe_fully_overlapped
        if MEGAMOE_FULLY_OVERLAPPED
        else mega_moe_fusion.int8_int4_mega_moe
    )

    from sglang.srt.distributed.parallel_state import get_tp_group

    hidden_size = hidden_states.shape[-1]
    experts = moe.experts
    config = experts.moe_runner_config
    rows = max(num_tokens, 1)
    # Keyed on the config, not on this batch's tensors, so the buffer is the
    # one `megamoe_init_symm_buffer` allocated before any graph capture.
    assert topk_ids.shape[-1] == config.top_k and hidden_size == config.hidden_size, (
        f"megamoe routing/hidden mismatch: got topk={topk_ids.shape[-1]} "
        f"hidden={hidden_size}, buffer built for topk={config.top_k} "
        f"hidden={config.hidden_size}"
    )
    buf = _get_symm_buffer(
        get_tp_group().device_group,
        num_experts=experts.num_experts,
        num_max_tokens_per_rank=megamoe_capacity_for(_resolved_args()),
        num_topk=config.top_k,
        hidden=config.hidden_size,
        intermediate_hidden=config.intermediate_size_per_partition,
    )
    _log_max_tokens(num_tokens, megamoe_capacity_for(_resolved_args()))
    x, x_sf, safe_ids, safe_weights = _prepare_inputs(
        hidden_states, topk_ids, topk_weights, experts.num_experts
    )

    # Every rank must enter the fused op (it contains cross-rank barriers), so a
    # rank with zero tokens still calls it; one padding row keeps the output
    # pointer non-null.
    y = torch.empty(
        rows, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
    )
    fused_op(
        y,
        x=x,
        x_sf=x_sf,
        topk_idx=safe_ids,
        topk_weights=safe_weights,
        l1_weights=experts.megamoe_l1_weights,
        l2_weights=experts.megamoe_l2_weights,
        sym_buffer=buf,
    )
    return y[:num_tokens].to(hidden_states.dtype)


_ALT_STREAM = None
_ALT_STREAM_LOGGED = False


def _alt_stream_for(moe):
    """The side stream the shared expert runs on while gate+topk run on the main one.

    0.5.23 only builds ``DeepseekV2Model.alt_stream`` for CUDA / MUSA / NPU /
    ROCm, so on this platform it is ``None`` and the shared expert would run
    serially -- the megamoe step then pays it in full instead of hiding it behind
    the routing. One stream per process is enough: the MoE layers use it one at a
    time, and every use joins before returning.
    """
    global _ALT_STREAM, _ALT_STREAM_LOGGED
    stream = getattr(moe, "alt_stream", None)
    if stream is not None:
        return stream
    if not MEGAMOE_SHARED_ALT_STREAM:
        return None
    if _ALT_STREAM is None:
        _ALT_STREAM = torch.cuda.Stream()
    if not _ALT_STREAM_LOGGED:
        _ALT_STREAM_LOGGED = True
        logger.warning(
            "[megamoe] alt_stream was None for this platform; running the shared "
            "expert on a dedicated stream so it overlaps gate+topk."
        )
    return _ALT_STREAM


def _run_shared_experts(moe, hidden_states, gemm_output_zero_allocator=None):
    """Shared expert on the alternate stream, so it overlaps gate + topk.

    Returns `(output, stream_to_join)`: the caller joins the stream before the
    fused op. `(None, None)` when there is no shared expert (fused-shared
    experts) or no tokens.
    """
    if hidden_states.shape[0] == 0:
        return None, None
    alt_stream = _alt_stream_for(moe)
    if alt_stream is None:
        return (
            moe._forward_shared_experts(hidden_states, gemm_output_zero_allocator),
            None,
        )
    alt_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(alt_stream):
        out = moe._forward_shared_experts(hidden_states, gemm_output_zero_allocator)
        out.record_stream(alt_stream)
    return out, alt_stream


def forward_megamoe_deepseek(
    moe,
    hidden_states: torch.Tensor,
    forward_batch=None,
    gemm_output_zero_allocator=None,
):
    """MoE forward for `DeepseekV2MoE` (and the GLM MoE subclasses) on KLX.

    `hidden_states` is this rank's own token shard, and the return value is the
    MoE result for exactly those tokens -- no all-reduce or reduce-scatter
    afterwards.
    """
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo

    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.view(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]

    shared_output, alt_stream = _run_shared_experts(
        moe, hidden_states, gemm_output_zero_allocator
    )

    if num_tokens > 0:
        # 0.5.23's routed gate takes the bump allocator, not the forward batch.
        router_logits = moe.gate(hidden_states, gemm_output_zero_allocator)
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
        )
        topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights
        routed_input = hidden_states
    else:
        routed_input, topk_ids, topk_weights = _idle_routing(
            hidden_states, hidden_dim, _top_k_of(moe)
        )

    if alt_stream is not None:
        # Join BEFORE the fused op. Moving this after it raised the client-side
        # accepted length to 8.56 per iteration -- above the 5 that 4 draft
        # tokens allow -- while the decode role's own counter stayed at 2.85, so
        # that ordering is not shippable until the accounting is explained; it
        # also bought only ~2% throughput.
        torch.cuda.current_stream().wait_stream(alt_stream)

    # One call, whatever the token count: the per-rank capacity only sizes the
    # symmetric workspace, it is a runtime bound the op checks.
    out = _run_fused(moe, routed_input, topk_ids, topk_weights, num_tokens)

    if shared_output is not None:
        out = out + shared_output
    return out.view(-1, hidden_dim)