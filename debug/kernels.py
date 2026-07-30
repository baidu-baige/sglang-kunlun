"""Kernel-level DSV4 probes moved out of ``sglang_kunlun/kernels/kernel_ops.py``.

``probe()`` is the former ``_dsv4_probe`` helper; the remaining functions are
the former inline dump/log blocks.  Environment variables, payload keys and
file names are unchanged.  Call-site metadata (strides, dtypes) is now derived
inside these functions so the production call sites stay allocation free.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Mapping, Optional

import torch

from ._dispatch import SiteRegistry

logger = logging.getLogger(__name__)

# Manually flipped during accuracy runs, exactly as before the refactor.
ENABLE_ACCURACY_DUMPS = False

# Former hard-coded ``/home/zx/debug_dumps`` target of the layer-0 K dump.
LOCAL_DUMP_DIR = os.environ.get("DSV4_LOCAL_DUMP_DIR", "/home/zx/debug_dumps")

_PROBE_COUNTERS: Dict[str, int] = {}
_ONE_SHOT_LOGGED: set[str] = set()
_ONE_SHOT_DUMPED: set[str] = set()


def _one_shot(key: str, store: set) -> bool:
    if key in store:
        return False
    store.add(key)
    return True


def probe(group: str, tensors: Mapping[str, object], **meta) -> None:
    """Persist opt-in DSV4 intermediate tensors without affecting the hot path."""
    output_dir = os.environ.get("DSV4_ACCURACY_DUMP_DIR")
    tensor_values = [
        value for value in tensors.values() if isinstance(value, torch.Tensor)
    ]
    if not output_dir or not tensor_values:
        return
    expected_tokens = int(os.environ.get("DSV4_ACCURACY_DUMP_TOKENS", "0"))
    if expected_tokens and tensor_values[0].shape[0] != expected_tokens:
        return
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    if rank != int(os.environ.get("DSV4_ACCURACY_DUMP_RANK", "0")):
        return
    sample_tokens = int(os.environ.get("DSV4_ACCURACY_DUMP_SAMPLE_TOKENS", "8"))
    tensors = {
        name: value[:sample_tokens]
        if isinstance(value, torch.Tensor) and value.ndim
        else value
        for name, value in tensors.items()
    }
    key = f"{rank}:{group}"
    index = _PROBE_COUNTERS.get(key, 0)
    _PROBE_COUNTERS[key] = index + 1
    if index >= int(os.environ.get("DSV4_ACCURACY_DUMP_LIMIT", "16")):
        return
    payload = {
        "meta": {"rank": rank, "group": group, "call": index, **meta},
        "tensors": {
            name: value.detach().cpu()
            for name, value in tensors.items()
            if isinstance(value, torch.Tensor)
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, os.path.join(output_dir, f"rank{rank}_{group}_{index:04d}.pt"))


def topk(
    scores: torch.Tensor,
    page_indices: torch.Tensor,
    raw_indices: Optional[torch.Tensor],
    page_size: int,
) -> None:
    """Dump the top-k transform inputs and outputs."""
    tensors = {"scores": scores}
    if raw_indices is not None:
        tensors["raw_indices"] = raw_indices
    tensors["page_indices"] = page_indices
    probe("topk", tensors, page_size=page_size)


def write_req_to_token_input(
    req_to_token_ptr: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_tensors: torch.Tensor,
    pre_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    """Dump the request-to-token writer inputs plus their memory layout."""
    if not os.environ.get("DSV4_ACCURACY_DUMP_DIR"):
        return
    probe(
        "write_req_to_token_input",
        {
            "req_pool_indices": req_pool_indices,
            "prefix_tensors": prefix_tensors,
            "pre_lens": pre_lens,
            "seq_lens": seq_lens,
            "extend_lens": extend_lens,
            "out_cache_loc": out_cache_loc,
        },
        req_to_token_stride=tuple(req_to_token_ptr.stride()),
        req_to_token_is_contiguous=req_to_token_ptr.is_contiguous(),
        req_pool_indices_stride=tuple(req_pool_indices.stride()),
        req_pool_indices_is_contiguous=req_pool_indices.is_contiguous(),
        prefix_tensors_stride=tuple(prefix_tensors.stride()),
        prefix_tensors_is_contiguous=prefix_tensors.is_contiguous(),
        pre_lens_stride=tuple(pre_lens.stride()),
        pre_lens_is_contiguous=pre_lens.is_contiguous(),
        seq_lens_stride=tuple(seq_lens.stride()),
        seq_lens_is_contiguous=seq_lens.is_contiguous(),
        extend_lens_stride=tuple(extend_lens.stride()),
        extend_lens_is_contiguous=extend_lens.is_contiguous(),
        out_cache_loc_stride=tuple(out_cache_loc.stride()),
        out_cache_loc_is_contiguous=out_cache_loc.is_contiguous(),
    )


def write_req_to_token_output(
    req_to_token_ptr: torch.Tensor, req_pool_indices: torch.Tensor
) -> None:
    """Dump the rows the request-to-token writer just produced."""
    if not os.environ.get("DSV4_ACCURACY_DUMP_DIR"):
        return
    written_rows = req_to_token_ptr.index_select(
        0, req_pool_indices.to(torch.long)
    )[:, :35328]
    probe("write_req_to_token_output", {"written_rows": written_rows})


def alloc_extend_input(
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    bs_upper: int,
    page_size: int,
) -> None:
    """Dump the alloc-extend kernel inputs plus their dtypes and strides."""
    if not os.environ.get("DSV4_ACCURACY_DUMP_DIR"):
        return
    probe(
        "alloc_extend_input",
        {
            "prefix_lens": prefix_lens,
            "seq_lens": seq_lens,
            "last_loc": last_loc,
            "free_pages": free_pages,
        },
        bs_upper=bs_upper,
        page_size=page_size,
        prefix_lens_dtype=str(prefix_lens.dtype),
        seq_lens_dtype=str(seq_lens.dtype),
        last_loc_dtype=str(last_loc.dtype),
        prefix_lens_stride=tuple(prefix_lens.stride()),
        seq_lens_stride=tuple(seq_lens.stride()),
        last_loc_stride=tuple(last_loc.stride()),
        last_loc_is_contiguous=last_loc.is_contiguous(),
    )


def alloc_extend_output(
    out_indices: torch.Tensor,
    ret_value: torch.Tensor,
    bs_upper: int,
    page_size: int,
) -> None:
    """Dump the alloc-extend kernel outputs."""
    probe(
        "alloc_extend_output",
        {"out_indices": out_indices, "ret_value": ret_value},
        bs_upper=bs_upper,
        page_size=page_size,
    )


def q_fused(
    q_wq_b: torch.Tensor,
    q_rope: torch.Tensor,
    q_hadamard: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Dump the fused Q RoPE/Hadamard stages."""
    if not os.environ.get("DSV4_ACCURACY_DUMP_DIR"):
        return
    probe(
        "q_fused",
        {"q_wq_b": q_wq_b, "q_rope": q_rope, "q_hadamard": q_hadamard},
        positions_shape=tuple(positions.shape),
    )


def q_fused_quant(
    q_int8: torch.Tensor,
    q_scale: torch.Tensor,
    weight: torch.Tensor,
    weights: torch.Tensor,
    weight_scale: float,
) -> None:
    """Dump the fused Q quantization stage."""
    probe(
        "q_fused_quant",
        {
            "q_int8": q_int8,
            "q_scale": q_scale,
            "weights_raw": weight,
            "weights_scaled": weights,
        },
        weight_scale=float(weight_scale),
    )


def log_quant_k_cache_callstack(k_bf16: torch.Tensor) -> None:
    """Log the quant-k-cache replacement inputs once per process."""
    if (
        os.environ.get("DSV4_MTP_PROBE") != "1"
        or os.environ.get("RANK", "0") != "0"
        or not _one_shot("quant_k_cache", _ONE_SHOT_LOGGED)
    ):
        return
    logger.warning(
        "[DSV4_CALLSTACK] quant_k_cache replacement input dtype=%s shape=%s",
        k_bf16.dtype,
        tuple(k_bf16.shape),
    )


def log_set_k_and_s_callstack(
    buf: torch.Tensor, loc: torch.Tensor, pack, page_size: int
) -> None:
    """Log the set_k_and_s replacement inputs once per process."""
    if (
        os.environ.get("DSV4_MTP_PROBE") != "1"
        or os.environ.get("RANK", "0") != "0"
        or not _one_shot("set_k_and_s", _ONE_SHOT_LOGGED)
    ):
        return
    logger.warning(
        "[DSV4_CALLSTACK] set_k_and_s_v4 replacement buf_dtype=%s "
        "buf_shape=%s loc_dtype=%s k_dtype=%s k_shape=%s page_size=%d",
        buf.dtype,
        tuple(buf.shape),
        loc.dtype,
        pack.k_nope_fp8.dtype,
        tuple(pack.k_nope_fp8.shape),
        page_size,
    )


def q_norm_rope_rmsnorm_stage(q_input: torch.Tensor, normalized: torch.Tensor) -> None:
    """Stash the Q RMSNorm stage for the layer-0 attention comparison."""
    if (
        not ENABLE_ACCURACY_DUMPS
        or q_input.shape[0] != 8192
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() != 0
        or not _one_shot("q_norm_rope_rmsnorm", _ONE_SHOT_DUMPED)
    ):
        return
    torch._dsv4_accuracy_q_stages = {
        "q.wq_b_reshaped": q_input.detach().cpu(),
        "q.rmsnorm_output": normalized.detach().cpu(),
    }


def q_norm_rope_rope_stage(
    q_input: torch.Tensor,
    positions: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope: torch.Tensor,
) -> None:
    """Stash the Q RoPE stage for the layer-0 attention comparison."""
    if (
        not ENABLE_ACCURACY_DUMPS
        or q_input.shape[0] != 8192
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() != 0
        or not _one_shot("q_norm_rope_rope", _ONE_SHOT_DUMPED)
    ):
        return
    from sglang_kunlun.kernels.kernel_ops import _dsv4_cos_sin_cache

    stages = getattr(torch, "_dsv4_accuracy_q_stages", {})
    stages.update(
        {
            "q.rope_positions": positions.detach().cpu(),
            "q.rope_freqs": _dsv4_cos_sin_cache(freqs_cis).detach().cpu(),
            "q.rope_output": rope.detach().cpu(),
        }
    )
    torch._dsv4_accuracy_q_stages = stages


def k_norm_rope_layer0_enabled(kv: torch.Tensor, positions: torch.Tensor) -> bool:
    """Return whether the layer-0 K stage dump should run for this call."""
    return (
        ENABLE_ACCURACY_DUMPS
        and kv.shape[0] == 8192
        and positions.numel() == 8192
        and int(positions[0]) == 8192
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and _one_shot("k_norm_rope_layer0", _ONE_SHOT_DUMPED)
    )


def k_norm_rope_layer0_dump(
    kv: torch.Tensor,
    normalized: torch.Tensor,
    rotated: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
) -> None:
    """Dump every layer-0 K stage together with the resulting cache rows."""
    cache_rows = kvcache.view(-1, rotated.shape[-1])[loc.long()]
    torch.save(
        {
            "kv.raw": kv.detach().cpu(),
            "kv.rmsnorm": normalized.detach().cpu(),
            "kv.rope": rotated.detach().cpu(),
            "positions": positions.detach().cpu(),
            "swa_loc": out_loc.detach().cpu(),
            "effective_loc": loc.detach().cpu(),
            "cache_rows": cache_rows.detach().cpu(),
            "page_size": page_size,
            "cache_shape": tuple(kvcache.shape),
        },
        os.path.join(LOCAL_DUMP_DIR, f"layer0_k_stages_0514_pid{os.getpid()}.pt"),
    )


# --------------------------------------------------------------------------
# probe sites
# --------------------------------------------------------------------------

_SITES = SiteRegistry()
capture = _SITES.capture


@_SITES.site("topk.raw")
def _site_topk_raw(scope) -> None:
    topk(
        scope["scores"],
        scope["out_page_indices"],
        scope["out_raw_indices"],
        scope["page_size"],
    )


@_SITES.site("topk.graph_safe")
def _site_topk_graph_safe(scope) -> None:
    topk(scope["scores"], scope["out_page_indices"], None, scope["page_size"])


@_SITES.site("write_req_to_token.input")
def _site_write_req_to_token_input(scope) -> None:
    write_req_to_token_input(
        scope["req_to_token_ptr"],
        scope["req_pool_indices"],
        scope["prefix_tensors"],
        scope["pre_lens"],
        scope["seq_lens"],
        scope["extend_lens"],
        scope["out_cache_loc"],
    )


@_SITES.site("write_req_to_token.output")
def _site_write_req_to_token_output(scope) -> None:
    write_req_to_token_output(scope["req_to_token_ptr"], scope["req_pool_indices"])


@_SITES.site("alloc_extend.input")
def _site_alloc_extend_input(scope) -> None:
    alloc_extend_input(
        scope["prefix_lens"],
        scope["seq_lens"],
        scope["last_loc"],
        scope["free_pages"],
        scope["bs_upper"],
        scope["page_size"],
    )


@_SITES.site("alloc_extend.output")
def _site_alloc_extend_output(scope) -> None:
    alloc_extend_output(
        scope["out_indices"],
        scope["ret_value"],
        scope["bs_upper"],
        scope["page_size"],
    )


@_SITES.site("q_fused.hadamard")
def _site_q_fused_hadamard(scope) -> None:
    q_fused(
        scope["q_wq_b"], scope["q_rope"], scope["q_hadamard"], scope["positions"]
    )


@_SITES.site("q_fused.quant")
def _site_q_fused_quant(scope) -> None:
    q_fused_quant(
        scope["q_int8"],
        scope["q_scale"],
        scope["weight"],
        scope["weights"],
        scope["weight_scale"],
    )


@_SITES.site("quant_k_cache.callstack")
def _site_quant_k_cache_callstack(scope) -> None:
    log_quant_k_cache_callstack(scope["k_bf16"])


@_SITES.site("set_k_and_s.callstack")
def _site_set_k_and_s_callstack(scope) -> None:
    log_set_k_and_s_callstack(
        scope["buf"],
        scope["loc_safe"],
        scope["nope_fp8_rope_bf16_pack"],
        scope["page_size"],
    )


@_SITES.site("q_norm_rope.rmsnorm")
def _site_q_norm_rope_rmsnorm(scope) -> None:
    q_norm_rope_rmsnorm_stage(scope["q_input"], scope["normalized"])


@_SITES.site("q_norm_rope.rope")
def _site_q_norm_rope_rope(scope) -> None:
    q_norm_rope_rope_stage(
        scope["q_input"], scope["positions"], scope["freqs_cis"], scope["rope"]
    )


@_SITES.site("k_norm_rope.layer0")
def _site_k_norm_rope_layer0(scope) -> None:
    if not k_norm_rope_layer0_enabled(scope["kv"], scope["positions"]):
        return
    k_norm_rope_layer0_dump(
        scope["kv"],
        scope["normalized"],
        scope["rotated"],
        scope["positions"],
        scope["out_loc"],
        scope["loc"],
        scope["kvcache"],
        scope["page_size"],
    )
