"""Diagnostic probes for the DSV4 Kunlun compressed-attention backend.

Everything here was previously inlined in
``sglang_kunlun/hooks/layers/attention/kunlun_deepseek_v4_backend.py``. The
production module now keeps one call per probe point and this module owns the
implementation, so no diagnostics ship inside ``sglang_kunlun``.

Contract preserved exactly:

* the same ``DSV4_*`` environment variables gate the same captures
* the same dump file names under ``/home/debug_dumps`` and the directories
  named by ``DSV4_MTP_TENSOR_DUMP_DIR`` / ``DSV4_IFEVAL_MTP_DIAG_DIR`` /
  ``DSV4_ACCURACY_DUMP_DIR``
* the same payload keys, including the ``step{N}_layer{M}.<stage>`` prefixes and
  the ``_mtp_tensor_probe`` / ``_mtp_tensor_probe_contracts`` attribute names

Each entry point re-evaluates its own enable condition instead of relying on a
flag computed in production code, so production carries no debug locals. State
that must survive between a capture point and a later dump point is stashed on
the backend object under a ``_dsv4_`` prefix, never in production locals.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

import torch

from debug.dsv4_probe_bridge import (
    dump_compressed_attention_inputs,
    dump_compressed_attention_outputs,
)

logger = logging.getLogger(__name__)

# Historically a module-level constant in the backend, hardcoded to False. The
# blocks it guards are the one-off full-tensor snapshots used during the initial
# 0.5.8/0.5.14 comparison; flip via the environment instead of editing source.
ACCURACY_DUMP_ENV_VAR = "DSV4_ENABLE_ACCURACY_DUMPS"
ACCURACY_DUMP_DIR = os.environ.get("DSV4_ACCURACY_DUMP_ROOT", "/home/zx/debug_dumps")


def accuracy_dumps_enabled() -> bool:
    """Return whether one-off accuracy tensor dumps are enabled."""
    return os.environ.get(ACCURACY_DUMP_ENV_VAR) == "1"


def log_backend_import(backend_file: str) -> None:
    """Announce which backend file was imported, and by which rank."""
    if os.environ.get("DSV4_C4_ATTN_METADATA_LOG") != "1":
        return
    print(
        "[DSV4_C4_ATTN_BACKEND_IMPORT] "
        f"file={backend_file} pid={os.getpid()} "
        f"TP_RANK={os.environ.get('TP_RANK')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"RANK={os.environ.get('RANK')} "
        f"flag={os.environ.get('DSV4_C4_ATTN_METADATA_LOG')}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Shared helpers (moved verbatim from the backend)
# ---------------------------------------------------------------------------


def _dsv4_ifeval_diag_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _dsv4_ifeval_tensor_summary(tensor: Optional[torch.Tensor]) -> Optional[dict]:
    if not isinstance(tensor, torch.Tensor):
        return None
    flat = tensor.detach().reshape(-1)
    cpu = flat.contiguous().cpu()
    byte_view = cpu.view(torch.uint8)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
        "data_ptr": tensor.data_ptr(),
        "numel": flat.numel(),
        "head": cpu[:16].tolist(),
        "tail": cpu[-16:].tolist() if cpu.numel() else [],
        "sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
    }


def _append_dsv4_ifeval_backend_event(owner, event: dict) -> None:
    dump_dir = os.environ.get("DSV4_IFEVAL_MTP_DIAG_DIR")
    if not dump_dir or _dsv4_ifeval_diag_rank() != 0:
        return
    event_count = getattr(owner, "_dsv4_ifeval_diag_event_count", 0)
    max_events = int(os.environ.get("DSV4_IFEVAL_MTP_DIAG_MAX_EVENTS", "20000"))
    if event_count >= max_events:
        return
    os.makedirs(dump_dir, exist_ok=True)
    event["event_index"] = event_count
    event["pid"] = os.getpid()
    step_id = getattr(owner, "speculative_step_id", -1)
    event["speculative_step_id"] = int(step_id) if step_id is not None else -1
    with open(
        os.path.join(dump_dir, "backend_events_rank0.jsonl"),
        "a",
        encoding="utf-8",
    ) as output:
        output.write(json.dumps(event, separators=(",", ":")) + "\n")
    owner._dsv4_ifeval_diag_event_count = event_count + 1


def _summarize_c4_metadata_tensor(value):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return repr(value)
    flat = value.detach().cpu().reshape(-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "head": flat[:8].tolist(),
        "tail": flat[-8:].tolist() if flat.numel() else [],
    }


def _mtp_tensor_probe_enabled(layer_id: int, num_queries: int) -> bool:
    dump_dir = os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR")
    if not dump_dir or os.environ.get("DSV4_MTP_PROBE_SEQ_LEN"):
        return False
    if (
        num_queries != 1
        and os.environ.get("DSV4_MTP_TENSOR_PROBE_ANY_BATCH") != "1"
    ):
        return False
    probe_layers = {
        int(value)
        for value in os.environ.get(
            "DSV4_MTP_TENSOR_PROBE_LAYERS", "0,21,42"
        ).split(",")
        if value
    }
    if layer_id not in probe_layers:
        return False
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )
    return rank == 0


def _mtp_writer_probe_enabled(layer_id: int) -> bool:
    return _mtp_tensor_probe_enabled(layer_id, 1) and (
        os.environ.get("DSV4_MTP_TENSOR_PROBE_WRITER") == "1"
    )


def _mtp_gather_cache_rows(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return cache.new_empty((*indices.shape, cache.shape[-1]))
    safe_indices = indices.reshape(-1).clamp(min=0, max=cache.shape[0] - 1).long()
    return cache.index_select(0, safe_indices).reshape(*indices.shape, cache.shape[-1])


def _mtp_tensor_stats(tensor: torch.Tensor) -> dict:
    flat = tensor.reshape(-1)
    stats = {
        "numel": flat.numel(),
        "zero_count": int((flat == 0).sum().item()) if flat.numel() else 0,
        "minus_one_count": int((flat == -1).sum().item()) if flat.numel() else 0,
        "negative_count": int((flat < 0).sum().item()) if flat.numel() else 0,
    }
    if flat.numel():
        stats.update(min=float(flat.min().item()), max=float(flat.max().item()))
    if tensor.is_floating_point():
        stats.update(
            nan_count=int(torch.isnan(flat).sum().item()),
            inf_count=int(torch.isinf(flat).sum().item()),
        )
    return stats


def _mtp_probe_matches_seq_lens(seq_lens_cpu, batch_size: int) -> bool:
    if seq_lens_cpu is None:
        return False
    values = torch.as_tensor(seq_lens_cpu)[:batch_size].reshape(-1)
    if not values.numel():
        return False
    expected = int(os.environ.get("DSV4_MTP_PROBE_SEQ_LEN", "-1"))
    if expected < 0:
        return int(values.max()) > 10000
    radius = int(os.environ.get("DSV4_MTP_PROBE_RADIUS", "8"))
    return bool((values - expected).abs().le(radius).any())


def _mtp_probe_layout(tensor: torch.Tensor) -> dict:
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": tuple(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
    }


def dump_previous_mtp_tensor_probe(multistep_backend, forward_batch) -> None:
    """Dump the previous multistep backend tensor probe when selected."""
    dump_dir = os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR")
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )
    if (
        not dump_dir
        or rank != 0
        or not _mtp_probe_matches_seq_lens(
            forward_batch.seq_lens_cpu, forward_batch.batch_size
        )
    ):
        return

    cycle = getattr(multistep_backend, "_mtp_tensor_dump_cycle", 0)
    max_cycles = int(os.environ.get("DSV4_MTP_TENSOR_DUMP_CYCLES", "3"))
    if cycle >= max_cycles:
        return

    tensors = {}
    tensor_metadata = {}
    bs = forward_batch.batch_size

    def record(step: int, name: str, tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        key = f"step{step}.{name}"
        tensors[key] = tensor.detach().cpu()
        tensor_metadata[key] = _mtp_probe_layout(tensor)

    for step, backend in enumerate(multistep_backend.attn_backends):
        metadata = getattr(backend, "forward_metadata", None)
        core = getattr(metadata, "core_attn_metadata", None)
        if core is None:
            core = getattr(metadata, "core_metadata", None)
        indexer = getattr(metadata, "indexer_metadata", None)

        attention_aux = backend._attention_decode_aux.get(bs)
        if attention_aux is not None:
            _, q_lod, _, kv_lens = attention_aux
            record(step, "q_lod", q_lod)
            record(step, "kv_lens", kv_lens)

        page_table = getattr(core, "page_table", None)
        record(step, "page_table", page_table)
        if isinstance(page_table, torch.Tensor) and indexer is not None:
            max_seq_len = int(getattr(indexer, "max_c4_seq_len", 0))
            num_pages = min(
                (max_seq_len + 63) // 64,
                page_table.shape[1],
            )
            record(step, "block_table", page_table[:, :num_pages].to(torch.int32))
        record(step, "c4_seq_lens", getattr(indexer, "c4_seq_lens", None))
        record(
            step,
            "c4_sparse_page_indices",
            getattr(core, "c4_sparse_page_indices", None),
        )
        record(step, "c128_page_indices", getattr(core, "c128_page_indices", None))

    if not tensors:
        logger.warning("[DSV4_MTP_PROBE] no target replay metadata captured")
        return

    os.makedirs(dump_dir, exist_ok=True)
    version = os.environ.get("DSV4_MTP_TENSOR_DUMP_VERSION", "0514")
    path = os.path.join(dump_dir, f"mtp_tensor_{version}_cycle{cycle}_rank0.pt")
    torch.save(
        {
            "version": version,
            "cycle": cycle,
            "probe_seq_len": int(os.environ.get("DSV4_MTP_PROBE_SEQ_LEN", "-1")),
            "seq_lens": list(
                map(
                    int,
                    forward_batch.seq_lens_cpu[: forward_batch.batch_size],
                )
            ),
            "tensors": tensors,
            "tensor_metadata": tensor_metadata,
        },
        path,
    )
    logger.warning("[DSV4_MTP_PROBE] target tensor dump saved path=%s", path)
    multistep_backend._mtp_tensor_dump_cycle = cycle + 1


def _dump_eager_mtp_tensor_probe(backend) -> None:
    if getattr(backend, "speculative_num_steps", 0) > 1:
        return
    dump_dir = os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR")
    probe = getattr(backend, "_mtp_tensor_probe", {})
    if not dump_dir or not probe:
        return
    cycle = getattr(backend, "_mtp_tensor_dump_cycle", 0)
    max_cycles = int(os.environ.get("DSV4_MTP_TENSOR_DUMP_CYCLES", "3"))
    if cycle >= max_cycles:
        return

    tensors = {}
    tensor_metadata = {}
    tensor_stats = {}
    for name, tensor in probe.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        cpu_tensor = tensor.detach().cpu()
        tensors[name] = cpu_tensor
        tensor_metadata[name] = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": tuple(tensor.stride()),
            "device": str(tensor.device),
            "data_ptr": tensor.data_ptr(),
        }
        tensor_stats[name] = _mtp_tensor_stats(cpu_tensor)
    if not tensors:
        return

    os.makedirs(dump_dir, exist_ok=True)
    version = os.environ.get("DSV4_MTP_TENSOR_DUMP_VERSION", "0514")
    path = os.path.join(dump_dir, f"mtp_tensor_{version}_cycle{cycle}_rank0.pt")
    torch.save(
        {
            "version": version,
            "cycle": cycle,
            "tensors": tensors,
            "tensor_metadata": tensor_metadata,
            "tensor_stats": tensor_stats,
            "operator_contracts": getattr(backend, "_mtp_tensor_probe_contracts", {}),
        },
        path,
    )
    logger.warning("[DSV4_MTP_PROBE] target eager tensor dump saved path=%s", path)
    backend._mtp_tensor_dump_cycle = cycle + 1


# ---------------------------------------------------------------------------
# Operator-boundary probes (alias / MTP / prefill)
# ---------------------------------------------------------------------------

BACKEND_FILE = (
    "sglang_kunlun/hooks/layers/attention/kunlun_deepseek_v4_backend.py"
)


def _decode_attention_aliases(scope):
    """Resolve the decode-alias dict and target layer without production help.

    The alias dict is planted on the forward batch by the tensor dump hooks, and
    the target layer comes from the environment, so nothing about this has to be
    computed in the backend.
    """
    aliases = getattr(
        scope["forward_batch"], "_dsv4_decode_attention_aliases", None
    )
    if aliases is None:
        return None
    alias_layer = int(os.environ.get("DSV4_DECODE_ATTENTION_ALIAS_LAYER", "2"))
    if scope["layer"].layer_id != alias_layer:
        return None
    return aliases


def capture_attention_aliases_inputs(backend, scope):
    """Publish the operator inputs into the decode-alias comparison dict."""
    aliases = _decode_attention_aliases(scope)
    if aliases is None:
        return
    operator_only_alias = (
        os.environ.get("DSV4_DECODE_ATTENTION_ALIAS_OPERATOR_ONLY") == "1"
    )
    backend._dsv4_alias_operator_only = operator_only_alias
    q_op = scope["q_op"]
    win_indices_op = scope["win_indices_op"]
    extra_indices_op = scope["extra_indices_op"]
    q_lod_op = scope["q_lod_op"]
    kv_lens_op = scope["kv_lens_op"]
    attn_sink_op = scope["attn_sink_op"]
    core = scope["core"]
    aliases.update(
        {
            "attention_q": q_op.clone() if operator_only_alias else q_op,
            "attention_win_cache": scope["win_cache_op"],
            "attention_win_indices": (
                win_indices_op.clone() if operator_only_alias else win_indices_op
            ),
            "attention_extra_cache": scope["extra_cache_op"],
            "attention_extra_indices": (
                extra_indices_op.clone()
                if operator_only_alias
                else extra_indices_op
            ),
            "attention_q_lod_cpu": scope["q_lod_cpu_op"],
            "attention_q_lod": (
                q_lod_op.clone() if operator_only_alias else q_lod_op
            ),
            "attention_kv_lens_cpu": scope["kv_lens_cpu_op"],
            "attention_kv_lens": (
                kv_lens_op.clone() if operator_only_alias else kv_lens_op
            ),
            "attention_softmax_scale": torch.tensor(
                backend.softmax_scale, dtype=torch.float64
            ),
            "attention_causal": torch.tensor(True),
            "attention_max_window_size": torch.tensor(
                win_indices_op.shape[1], dtype=torch.int64
            ),
            "attention_compress_ratio": torch.tensor(
                scope["effective_ratio"], dtype=torch.int64
            ),
            "attention_compressed_topk": torch.tensor(
                scope["compressed_topk"], dtype=torch.int64
            ),
        }
    )
    if attn_sink_op is not None:
        aliases["attention_sink"] = attn_sink_op
    if not operator_only_alias:
        for name in (
            "raw_out_loc",
            "swa_out_cache_loc",
            "c4_out_loc",
            "page_table",
            "swa_page_indices",
            "swa_topk_lengths",
            "c4_sparse_topk_lengths",
        ):
            value = getattr(core, name, None)
            if isinstance(value, torch.Tensor):
                aliases[f"metadata_{name}"] = value
        aliases.update(
            {
                "metadata_req_pool_indices": scope[
                    "forward_batch"
                ].req_pool_indices,
                "metadata_req_to_token": backend.req_to_token,
            }
        )


def capture_attention_aliases_outputs(backend, scope):
    """Capture compressed-attention outputs in the active alias payload."""
    aliases = _decode_attention_aliases(scope)
    if aliases is None:
        return
    operator_only_alias = getattr(backend, "_dsv4_alias_operator_only", False)
    out_op = scope["out_op"]
    max_logits_op = scope["max_logits_op"]
    lse_op = scope["lse_op"]
    aliases.update(
        {
            "attention_output": (
                out_op.clone() if operator_only_alias else out_op
            ),
            "attention_max_logits": (
                max_logits_op.clone() if operator_only_alias else max_logits_op
            ),
            "attention_lse": (
                lse_op.clone() if operator_only_alias else lse_op
            ),
        }
    )


def _mtp_attention_probe_selected(backend, scope):
    return (
        scope["forward_batch"].forward_mode.is_extend()
        and not torch.cuda.is_current_stream_capturing()
        and _mtp_tensor_probe_enabled(
            getattr(scope["layer"], "layer_id", -1), scope["q_op"].shape[0]
        )
    )


def capture_mtp_attention_inputs(backend, scope):
    """Clone the full MTP replay contract for one extend step."""
    if not _mtp_attention_probe_selected(backend, scope):
        backend._dsv4_mtp_attention_prefix = None
        return
    layer = scope["layer"]
    core = scope["core"]
    forward_batch = scope["forward_batch"]
    win_cache_op = scope["win_cache_op"]
    win_indices_op = scope["win_indices_op"]
    extra_cache_op = scope["extra_cache_op"]
    extra_indices_op = scope["extra_indices_op"]
    attn_sink_op = scope["attn_sink_op"]

    prefix = f"step{backend.speculative_step_id}_layer{layer.layer_id}"
    backend._dsv4_mtp_attention_prefix = prefix
    probe = getattr(backend, "_mtp_tensor_probe", {})
    probe.update(
        {
            f"{prefix}.metadata.input_ids": forward_batch.input_ids.clone(),
            f"{prefix}.metadata.positions": forward_batch.positions.clone(),
            f"{prefix}.metadata.seq_lens": forward_batch.seq_lens.clone(),
            f"{prefix}.metadata.req_pool_indices": forward_batch.req_pool_indices.clone(),
            f"{prefix}.metadata.scheduler_out_cache_loc": forward_batch.out_cache_loc.clone(),
            f"{prefix}.metadata.raw_out_loc": core.raw_out_loc,
            f"{prefix}.metadata.seq_lens_casual": core.seq_lens_casual,
            f"{prefix}.metadata.positions_casual": core.positions_casual,
            f"{prefix}.metadata.page_table": core.page_table,
            f"{prefix}.metadata.swa_page_indices": core.swa_page_indices,
            f"{prefix}.metadata.swa_topk_lengths": core.swa_topk_lengths,
            f"{prefix}.metadata.effective_swa_len": (
                (core.swa_page_indices != 0)
                .to(torch.int32)
                .cumprod(dim=-1)
                .sum(dim=-1)
                .clone()
            ),
            f"{prefix}.attention.q": scope["q_op"].clone(),
            f"{prefix}.attention.win_indices": win_indices_op.clone(),
            f"{prefix}.attention.win_cache_rows": _mtp_gather_cache_rows(
                win_cache_op, win_indices_op
            ).clone(),
            f"{prefix}.attention.extra_indices": extra_indices_op.clone(),
            f"{prefix}.attention.extra_cache_rows": _mtp_gather_cache_rows(
                extra_cache_op, extra_indices_op
            ).clone(),
            f"{prefix}.attention.q_lod_cpu": scope["q_lod_cpu_op"],
            f"{prefix}.attention.q_lod": scope["q_lod_op"].clone(),
            f"{prefix}.attention.kv_lens_cpu": scope["kv_lens_cpu_op"],
            f"{prefix}.attention.kv_lens": scope["kv_lens_op"].clone(),
        }
    )
    for name in (
        "swa_out_cache_loc",
        "c4_out_loc",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c4_sparse_topk_lengths",
        "c4_sparse_page_indices",
        "c4_sparse_raw_indices",
        "c128_out_loc",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
    ):
        tensor = getattr(core, name, None)
        if isinstance(tensor, torch.Tensor):
            probe[f"{prefix}.metadata.{name}"] = tensor
    indexer_metadata = getattr(backend.forward_metadata, "indexer_metadata", None)
    if indexer_metadata is not None:
        for name in (
            "c4_seq_lens",
            "page_table",
            "topk_metadata",
            "deep_gemm_metadata",
        ):
            tensor = getattr(indexer_metadata, name, None)
            if isinstance(tensor, torch.Tensor):
                probe[f"{prefix}.metadata.indexer_{name}"] = tensor
    if attn_sink_op is not None:
        probe[f"{prefix}.attention.attn_sink"] = attn_sink_op
    backend._mtp_tensor_probe = probe
    contracts = getattr(backend, "_mtp_tensor_probe_contracts", {})
    contracts.setdefault(prefix, {}).update(
        {
            "softmax_scale": backend.softmax_scale,
            "causal": True,
            "max_window_size": win_indices_op.shape[1],
            "compress_ratio": scope["effective_ratio"],
            "compressed_topk": scope["compressed_topk"],
            "page_size": backend.page_size,
        }
    )
    backend._mtp_tensor_probe_contracts = contracts


def capture_mtp_attention_outputs(backend, scope):
    """Capture compressed-attention outputs for the active MTP probe."""
    prefix = getattr(backend, "_dsv4_mtp_attention_prefix", None)
    if not prefix:
        return
    backend._dsv4_mtp_attention_prefix = None
    probe = getattr(backend, "_mtp_tensor_probe", {})
    probe[f"{prefix}.attention.output"] = scope["out_op"].clone()
    probe[f"{prefix}.attention.max_logits"] = scope["max_logits_op"].clone()
    probe[f"{prefix}.attention.lse"] = scope["lse_op"].clone()
    backend._mtp_tensor_probe = probe
    _dump_eager_mtp_tensor_probe(backend)


def _prefill_backend_probe_selected(scope):
    return (
        os.environ.get("DSV4_PREFILL_BACKEND_PROBE") == "1"
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and getattr(scope["layer"], "layer_id", -1)
        == int(os.environ.get("DSV4_PREFILL_PROBE_LAYER", "0"))
        and scope["q_op"].shape[0] == 8192
        and int(scope["forward_batch"].positions[0].item()) == 8192
    )


def _prefill_device_probe_prefix_len() -> int:
    return int(os.environ.get("DSV4_PREFILL_PROBE_PREFIX_LEN", "8192"))


def _prefill_device_probe_selected(scope):
    return (
        os.environ.get("DSV4_PREFILL_BACKEND_DEVICE_PROBE") == "1"
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and getattr(scope["layer"], "layer_id", -1)
        == int(os.environ.get("DSV4_PREFILL_PROBE_LAYER", "0"))
        and scope["q_op"].shape[0] == 8192
        and _prefill_device_probe_prefix_len()
        in (scope["forward_batch"].extend_prefix_lens_cpu or [])
    )


def capture_prefill_backend_inputs(backend, scope):
    """Capture the 512-token tail of the repeated layer-0 prefill chunk."""
    if not _prefill_backend_probe_selected(scope):
        backend._dsv4_prefill_backend_probe = None
        return
    layer = scope["layer"]
    core = scope["core"]
    q_op = scope["q_op"]
    win_cache_op = scope["win_cache_op"]
    win_indices_op = scope["win_indices_op"]
    attn_sink_op = scope["attn_sink_op"]

    tail = slice(7680, 8192)
    tail_win_indices = win_indices_op[tail]
    backend._dsv4_prefill_backend_probe = {
        "positions": scope["forward_batch"].positions.detach().cpu(),
        "q_tail": q_op[tail].detach().cpu(),
        "win_indices_tail": tail_win_indices.detach().cpu(),
        "win_cache_rows_tail": _mtp_gather_cache_rows(
            win_cache_op, tail_win_indices
        ).detach().cpu(),
        "q_lod_cpu": scope["q_lod_cpu_op"].detach().cpu(),
        "q_lod": scope["q_lod_op"].detach().cpu(),
        "kv_lens_cpu": scope["kv_lens_cpu_op"].detach().cpu(),
        "kv_lens": scope["kv_lens_op"].detach().cpu(),
        "attn_sink": (
            attn_sink_op.detach().cpu() if attn_sink_op is not None else None
        ),
        "swa_out_cache_loc": (
            getattr(core, "swa_out_cache_loc").detach().cpu()
            if isinstance(getattr(core, "swa_out_cache_loc", None), torch.Tensor)
            else None
        ),
        "raw_out_loc": (
            getattr(core, "raw_out_loc").detach().cpu()
            if isinstance(getattr(core, "raw_out_loc", None), torch.Tensor)
            else None
        ),
        "win_cache_shape": tuple(win_cache_op.shape),
        "win_cache_dtype": str(win_cache_op.dtype),
        "win_indices_shape": tuple(win_indices_op.shape),
        "softmax_scale": backend.softmax_scale,
        "causal": True,
        "max_window_size": win_indices_op.shape[1],
        "compress_ratio": scope["effective_ratio"],
        "compressed_topk": scope["compressed_topk"],
        "page_size": backend.page_size,
    }
    logger.warning(
        "DSV4_COMPRESSED_ATTN_STACK version=0514 backend=%s layer=%s "
        "q_shape=%s win_cache_shape=%s win_indices_shape=%s "
        "win_cache_ptr=%s side_stream=%s",
        f"{type(backend).__module__}.{type(backend).__qualname__}",
        getattr(layer, "layer_id", -1),
        tuple(q_op.shape),
        tuple(win_cache_op.shape),
        tuple(win_indices_op.shape),
        win_cache_op.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
        stack_info=True,
    )


def dump_prefill_backend_probe(backend, scope):
    """Persist the selected prefill backend probe and clear its state."""
    probe = getattr(backend, "_dsv4_prefill_backend_probe", None)
    if not probe:
        return
    backend._dsv4_prefill_backend_probe = None
    tail = slice(7680, 8192)
    probe.update(
        {
            "output_tail": scope["out_op"][tail].detach().cpu(),
            "max_logits_tail": scope["max_logits_op"][tail].detach().cpu(),
            "lse_tail": scope["lse_op"][tail].detach().cpu(),
        }
    )
    torch.save(
        probe,
        f"{ACCURACY_DUMP_DIR}/prefill_backend_0514_repeat_layer0_chunk1_rank0.pt",
    )


def dump_prefill_backend_device_probe(backend, scope):
    """Dump the indices-selected cache rows actually consumed on this rank."""
    if not _prefill_device_probe_selected(scope):
        return
    layer = scope["layer"]
    core = scope["core"]
    q_op = scope["q_op"]
    win_cache_op = scope["win_cache_op"]
    win_indices_op = scope["win_indices_op"]
    extra_cache_op = scope["extra_cache_op"]
    extra_indices_op = scope["extra_indices_op"]
    attn_sink_op = scope["attn_sink_op"]

    extra_unique_indices = torch.unique(extra_indices_op)
    win_unique_indices = torch.unique(win_indices_op)
    payload = {
        "input_ids": scope["forward_batch"].input_ids.clone(),
        "positions": scope["forward_batch"].positions.clone(),
        "q_local": q_op[:, :8].clone(),
        "win_indices": win_indices_op.clone(),
        "win_unique_indices": win_unique_indices,
        "win_unique_cache_rows": win_cache_op.index_select(
            0, win_unique_indices.to(torch.long)
        ).clone(),
        "extra_indices": extra_indices_op.clone(),
        "extra_unique_indices": extra_unique_indices,
        "extra_unique_cache_rows": extra_cache_op.index_select(
            0, extra_unique_indices.to(torch.long)
        ).clone(),
        "output_local": scope["out_op"][:, :8].clone(),
        "max_logits_local": scope["max_logits_op"][:, :8].clone(),
        "lse_local": scope["lse_op"][:, :8].clone(),
        "q_lod_cpu": scope["q_lod_cpu_op"].clone(),
        "q_lod": scope["q_lod_op"].clone(),
        "kv_lens_cpu": scope["kv_lens_cpu_op"].clone(),
        "kv_lens": scope["kv_lens_op"].clone(),
        "attn_sink": (
            attn_sink_op.clone() if attn_sink_op is not None else None
        ),
        "win_cache_shape": tuple(win_cache_op.shape),
        "extra_cache_shape": tuple(extra_cache_op.shape),
        "softmax_scale": backend.softmax_scale,
        "causal": True,
        "max_window_size": win_indices_op.shape[1],
        "compress_ratio": scope["effective_ratio"],
        "compressed_topk": scope["compressed_topk"],
        "page_size": backend.page_size,
        "side_stream": torch.cuda.current_stream().cuda_stream,
    }
    for name in (
        "raw_out_loc",
        "swa_out_cache_loc",
        "c128_out_loc",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
    ):
        tensor = getattr(core, name, None)
        if isinstance(tensor, torch.Tensor):
            payload[name] = tensor.clone()
    logger.warning(
        "DSV4_C128_PREFILL_STACK version=0514 backend=%s layer=%s "
        "q_shape=%s win_indices_shape=%s extra_indices_shape=%s "
        "win_cache_shape=%s extra_cache_shape=%s side_stream=%s",
        f"{type(backend).__module__}.{type(backend).__qualname__}",
        getattr(layer, "layer_id", -1),
        tuple(q_op.shape),
        tuple(win_indices_op.shape),
        tuple(extra_indices_op.shape),
        tuple(win_cache_op.shape),
        tuple(extra_cache_op.shape),
        torch.cuda.current_stream().cuda_stream,
        stack_info=True,
    )
    torch.save(
        {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in payload.items()
        },
        (
            f"{ACCURACY_DUMP_DIR}/prefill_backend_device_0514_layer"
            f"{getattr(layer, 'layer_id', -1)}_prefix"
            f"{_prefill_device_probe_prefix_len()}_rank0.pt"
        ),
    )


def dump_c4_logits_sample(scope):
    """Save three sampled rows of the C4 indexer logits, once per token count."""
    dump_dir = os.environ.get("DSV4_ACCURACY_DUMP_DIR")
    dump_tokens = int(os.environ.get("DSV4_ACCURACY_DUMP_TOKENS", "0"))
    q = scope["q"]
    if not dump_dir or (dump_tokens and q.shape[0] != dump_tokens):
        return
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    if rank != int(os.environ.get("DSV4_ACCURACY_DUMP_RANK", "0")):
        return

    logits = scope["logits"]
    sample_rows = torch.tensor(
        [0, logits.shape[0] // 2, logits.shape[0] - 1],
        dtype=torch.long,
        device=logits.device,
    )
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"rank{rank}_c4_logits_{q.shape[0]}.pt")
    if os.path.exists(path):
        return
    torch.save(
        {
            "sample_rows": sample_rows.detach().cpu(),
            "logits": logits.index_select(0, sample_rows).detach().cpu(),
            "max_seq_len": scope["max_seq_len"],
        },
        path,
    )


# ---------------------------------------------------------------------------
# C4 indexer probes
# ---------------------------------------------------------------------------


def dump_c4_logits(scope):
    """Sample three rows of the C4 logits for the indexer comparison."""
    dump_dir = os.environ.get("DSV4_ACCURACY_DUMP_DIR")
    dump_tokens = int(os.environ.get("DSV4_ACCURACY_DUMP_TOKENS", "0"))
    logits = scope["logits"]
    q = scope["q"]
    if not dump_dir or (dump_tokens and q.shape[0] != dump_tokens):
        return
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    if rank != int(os.environ.get("DSV4_ACCURACY_DUMP_RANK", "0")):
        return
    sample_rows = torch.tensor(
        [0, logits.shape[0] // 2, logits.shape[0] - 1],
        dtype=torch.long,
        device=logits.device,
    )
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"rank{rank}_c4_logits_{q.shape[0]}.pt")
    if os.path.exists(path):
        return
    torch.save(
        {
            "sample_rows": sample_rows.detach().cpu(),
            "logits": logits.index_select(0, sample_rows).detach().cpu(),
            "max_seq_len": scope["max_seq_len"],
        },
        path,
    )


def capture_indexer_operator_probes(backend, scope):
    """Alias, MTP-verify and MTP-replay captures around the C4 indexer call."""
    layer_id = scope["layer_id"]
    q = scope["q"]
    weights = scope["weights"]
    seq_lens = scope["seq_lens"]
    block_table = scope["block_table"]
    k_cache = scope["k_cache"]
    k_scale = scope["k_scale"]
    qlod_cpu = scope["qlod_cpu"]
    qlod_xpu = scope["qlod_xpu"]
    context_lens_cpu = scope["context_lens_cpu"]
    context_lens_xpu = scope["context_lens_xpu"]
    logits = scope["logits"]

    aliases = scope.get("attention_aliases")
    alias_layer = int(os.environ.get("DSV4_DECODE_ATTENTION_ALIAS_LAYER", "2"))
    if aliases is not None and layer_id == alias_layer:
        aliases.update(
            {
                "indexer_operator_q": q,
                "indexer_operator_weight": weights,
                "indexer_operator_seq_lens": seq_lens,
                "indexer_operator_block_table": block_table,
                "indexer_operator_k_cache": k_cache,
                "indexer_operator_k_scale": k_scale,
                "indexer_operator_q_lod_cpu": qlod_cpu,
                "indexer_operator_q_lod": qlod_xpu,
                "indexer_operator_context_lens_cpu": context_lens_cpu,
                "indexer_operator_context_lens": context_lens_xpu,
                "indexer_operator_logits": logits,
            }
        )

    if os.environ.get("DSV4_MTP_VERIFY_LAYER_DUMP") == "1" and scope[
        "is_target_verify"
    ]:
        from sglang.srt.models.deepseek_v4 import _DSV4_DECODE_LAYER_OUTPUTS

        probe_pages = min(140, block_table.shape[1])
        selected_k_cache = k_cache.index_select(
            0,
            block_table[:, :probe_pages]
            .reshape(-1)
            .long()
            .clamp(0, k_cache.shape[0] - 1),
        ).view(
            scope["batch_size"],
            probe_pages,
            scope["block_size"],
            1,
            scope["head_dim"],
        )
        prefix = "layer2_indexer_operator"
        _DSV4_DECODE_LAYER_OUTPUTS.update(
            {
                f"{prefix}_q": q.clone(),
                f"{prefix}_weight": weights.clone(),
                f"{prefix}_seq_lens": seq_lens.clone(),
                f"{prefix}_block_table": block_table.clone(),
                f"{prefix}_k_cache_pages": selected_k_cache.clone(),
                f"{prefix}_k_scale": k_scale[:, :9000].clone(),
                f"{prefix}_q_lod_cpu": qlod_cpu.clone(),
                f"{prefix}_q_lod": qlod_xpu.clone(),
                f"{prefix}_context_lens_cpu": context_lens_cpu.clone(),
                f"{prefix}_context_lens": context_lens_xpu.clone(),
            }
        )

    forward_batch = scope["forward_batch"]
    capture_mtp_probe = (
        forward_batch.forward_mode.is_extend()
        and not torch.cuda.is_current_stream_capturing()
        and _mtp_tensor_probe_enabled(layer_id, scope["q_fp8"].shape[0])
    )
    if not capture_mtp_probe:
        return
    probe_prefix = f"step{backend.speculative_step_id}_layer{layer_id}"
    probe = getattr(backend, "_mtp_tensor_probe", {})
    selected_k_cache = k_cache.index_select(
        0, block_table.reshape(-1).long().clamp(0, k_cache.shape[0] - 1)
    )
    probe.update(
        {
            f"{probe_prefix}.indexer.operator.q": q.clone(),
            f"{probe_prefix}.indexer.operator.weight": weights.clone(),
            f"{probe_prefix}.indexer.operator.seq_lens": seq_lens.clone(),
            f"{probe_prefix}.indexer.operator.page_table": scope[
                "page_table"
            ].clone(),
            f"{probe_prefix}.indexer.operator.block_table": block_table.clone(),
            f"{probe_prefix}.indexer.operator.k_cache_pages": selected_k_cache.clone(),
            f"{probe_prefix}.indexer.operator.k_scale": k_scale.clone(),
            f"{probe_prefix}.indexer.operator.q_lod_cpu": qlod_cpu,
            f"{probe_prefix}.indexer.operator.q_lod": qlod_xpu.clone(),
            f"{probe_prefix}.indexer.operator.context_lens_cpu": context_lens_cpu,
            f"{probe_prefix}.indexer.operator.context_lens": context_lens_xpu.clone(),
            f"{probe_prefix}.indexer.operator.logits": logits.clone(),
        }
    )
    backend._mtp_tensor_probe = probe
    contracts = getattr(backend, "_mtp_tensor_probe_contracts", {})
    contracts.setdefault(probe_prefix, {}).update(
        {
            "indexer_max_context_len": scope["max_seq_len"] * 4,
            "indexer_compress_ratio": 4,
            "indexer_clean_logits": True,
            "indexer_use_xfa_boost": False,
        }
    )
    backend._mtp_tensor_probe_contracts = contracts


def log_multistep_replay_metadata(backend, forward_batch, in_capture: bool) -> None:
    """Log the per-step replayed metadata for the MTP graph investigation."""
    if (
        in_capture
        or os.environ.get("DSV4_MTP_PROBE") != "1"
        or os.environ.get("RANK", "0") != "0"
        or forward_batch.seq_lens_cpu is None
        or int(forward_batch.seq_lens_cpu[: forward_batch.batch_size].max()) <= 10000
        or getattr(backend, "_mtp_probe_replay_calls", 0) >= 3
    ):
        return

    backend._mtp_probe_replay_calls = (
        getattr(backend, "_mtp_probe_replay_calls", 0) + 1
    )

    def _values(tensor, limit=8):
        if tensor is None:
            return None
        return tensor.detach().reshape(-1)[:limit].cpu().tolist()

    def _ptr(tensor):
        return tensor.data_ptr() if tensor is not None else None

    for step, step_backend in enumerate(
        backend.attn_backends[: backend.speculative_num_steps - 1]
    ):
        metadata = step_backend.forward_metadata
        core = getattr(metadata, "core_attn_metadata", None)
        if core is None:
            logger.warning(
                "[DSV4_MTP_PROBE] dsv4_metadata replay=%d step=%d metadata=%s",
                backend._mtp_probe_replay_calls,
                step,
                type(metadata).__name__,
            )
            continue
        logger.warning(
            "[DSV4_MTP_PROBE] dsv4_metadata replay=%d step=%d backend_step=%d "
            "seq_lens=%s seq_lens_cpu=%s positions=%s req_pool_indices=%s "
            "raw_out_loc=%s seq_lens_casual=%s positions_casual=%s "
            "page_table=%s swa_page_indices=%s swa_topk_lengths=%s "
            "c4_out_loc=%s c4_topk_raw=%s c4_topk_clamp1=%s "
            "ptrs=(raw:%s,page:%s,swa:%s,c4loc:%s)",
            backend._mtp_probe_replay_calls,
            step,
            step_backend.speculative_step_id,
            _values(forward_batch.seq_lens),
            _values(forward_batch.seq_lens_cpu),
            _values(getattr(forward_batch, "positions", None)),
            _values(forward_batch.req_pool_indices),
            _values(getattr(core, "raw_out_loc", None), 12),
            _values(getattr(core, "seq_lens_casual", None)),
            _values(getattr(core, "positions_casual", None)),
            _values(getattr(core, "page_table", None), 4),
            _values(getattr(core, "swa_page_indices", None), 8),
            _values(getattr(core, "swa_topk_lengths", None)),
            _values(getattr(core, "c4_out_loc", None), 12),
            _values(getattr(core, "c4_topk_lengths_raw", None)),
            _values(getattr(core, "c4_topk_lengths_clamp1", None)),
            _ptr(getattr(core, "raw_out_loc", None)),
            _ptr(getattr(core, "page_table", None)),
            _ptr(getattr(core, "swa_page_indices", None)),
            _ptr(getattr(core, "c4_out_loc", None)),
        )


def _handle_multistep_replay_metadata(backend, scope):
    forward_batch = scope["forward_batch"]
    dump_previous_mtp_tensor_probe(backend, forward_batch)
    log_multistep_replay_metadata(
        backend, forward_batch, in_capture=scope.get("in_capture", False)
    )


# ---------------------------------------------------------------------------
# Paged allocator probes
# ---------------------------------------------------------------------------

_ALLOC_EXTEND_PROBE_CALL = 0


def capture_alloc_extend_inputs(allocator, scope):
    """Snapshot the alloc_extend arguments before the Kunlun kernel runs."""
    probe_dir = os.environ.get("DSV4_ALLOC_EXTEND_PROBE_DIR")
    if (
        probe_dir is None
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() != 0
    ):
        allocator._dsv4_alloc_extend_probe = None
        return

    prefix_lens = scope["prefix_lens"]
    seq_lens = scope["seq_lens"]
    last_loc = scope["last_loc"]
    allocator._dsv4_alloc_extend_probe = {
        "version": "0514",
        "allocator_id": id(allocator),
        "alloc_fn": scope["alloc_fn"].__name__,
        "page_size": allocator.page_size,
        "extend_num_tokens": scope["extend_num_tokens"],
        "prefix_lens": prefix_lens.detach().cpu(),
        "prefix_lens_cpu": scope["prefix_lens_cpu"].detach().cpu(),
        "prefix_lens_dtype": str(prefix_lens.dtype),
        "prefix_lens_stride": tuple(prefix_lens.stride()),
        "seq_lens": seq_lens.detach().cpu(),
        "seq_lens_cpu": scope["seq_lens_cpu"].detach().cpu(),
        "seq_lens_dtype": str(seq_lens.dtype),
        "seq_lens_stride": tuple(seq_lens.stride()),
        "last_loc": last_loc.detach().cpu(),
        "last_loc_dtype": str(last_loc.dtype),
        "last_loc_shape": tuple(last_loc.shape),
        "last_loc_stride": tuple(last_loc.stride()),
        "last_loc_is_contiguous": last_loc.is_contiguous(),
        "free_pages_before": allocator.free_pages[:256].detach().cpu(),
    }


def dump_alloc_extend_outputs(allocator, scope):
    """Persist allocator outputs for the active extend probe."""
    from pathlib import Path

    global _ALLOC_EXTEND_PROBE_CALL

    probe = getattr(allocator, "_dsv4_alloc_extend_probe", None)
    if not probe:
        return
    allocator._dsv4_alloc_extend_probe = None
    probe_dir = os.environ.get("DSV4_ALLOC_EXTEND_PROBE_DIR")
    probe.update(
        {
            "out_indices": scope["out_indices"].detach().cpu(),
            "merged_value": scope["merged_value"],
            "num_new_pages": scope["num_new_pages"],
        }
    )
    torch.save(
        probe,
        Path(probe_dir) / f"0514-call{_ALLOC_EXTEND_PROBE_CALL:04d}.pt",
    )
    _ALLOC_EXTEND_PROBE_CALL += 1


# ---------------------------------------------------------------------------
# Single-line probe dispatcher
# ---------------------------------------------------------------------------
#
# Production carries exactly one statement per probe point:
#
#     dsv4_probe(self, "forward.alias_inputs", locals())
#
# The scope dict is the caller's ``locals()``. That couples this module to the
# production local *names* listed in REQUIRED_SCOPE below, which is the price of
# keeping the backend free of diagnostics. tools/check_probe_scope.py turns that
# coupling into a gate: every name below must exist in the corresponding
# production function, so a rename fails the build instead of silently
# disabling a probe.

REQUIRED_SCOPE: dict[str, tuple[str, ...]] = {
    "store_cache.pre_store": (
        "layer_id",
        "mapping",
        "raw_loc",
        "scheduler_raw_loc",
        "pack",
        "swa_pool",
        "local_layer_id",
    ),
    "store_cache.post_store": ("swa_pool", "local_layer_id"),
    "forward.pre_normalization": ("layer", "q", "pool", "win_indices"),
    "forward.attention_consume": (
        "layer",
        "forward_batch",
        "core",
        "compress_ratio",
        "win_cache",
        "win_indices",
        "extra_cache",
        "extra_indices",
        "q_lod",
        "kv_lens",
    ),
    "forward.c4_metadata": (
        "layer",
        "forward_batch",
        "core",
        "q_3d",
        "q_lod_cpu",
        "q_lod",
        "kv_lens_cpu",
        "kv_lens",
    ),
    "forward.pre_operator": (
        "layer",
        "forward_batch",
        "core",
        "q_3d",
        "win_cache",
        "win_indices",
        "extra_cache",
        "extra_indices",
        "q_lod_cpu",
        "q_lod",
        "kv_lens_cpu",
        "kv_lens",
        "max_logits",
        "lse",
        "effective_ratio",
        "compressed_topk",
        "attn_sink",
    ),
    "forward.operator_inputs": (
        "layer",
        "forward_batch",
        "core",
        "q_op",
        "win_cache_op",
        "win_indices_op",
        "extra_cache_op",
        "extra_indices_op",
        "q_lod_cpu_op",
        "q_lod_op",
        "kv_lens_cpu_op",
        "kv_lens_op",
        "attn_sink_op",
        "effective_ratio",
        "compressed_topk",
    ),
    "forward.operator_outputs": (
        "layer",
        "forward_batch",
        "core",
        "q_3d",
        "out",
        "q_op",
        "win_cache_op",
        "win_indices_op",
        "extra_cache_op",
        "extra_indices_op",
        "out_op",
        "max_logits_op",
        "lse_op",
        "q_lod_cpu_op",
        "q_lod_op",
        "kv_lens_cpu_op",
        "kv_lens_op",
        "attn_sink_op",
        "effective_ratio",
        "compressed_topk",
    ),
}


def _handle_store_cache_pre_store(backend, scope):
    capture_swa_store_inputs(
        backend,
        layer_id=scope["layer_id"],
        mapping=scope["mapping"],
        raw_loc=scope["raw_loc"],
        scheduler_raw_loc=scope["scheduler_raw_loc"],
        pack=scope["pack"],
        swa_pool=scope["swa_pool"],
        local_layer_id=scope["local_layer_id"],
    )


def _handle_store_cache_post_store(backend, scope):
    capture_swa_store_readback(
        backend,
        swa_pool=scope["swa_pool"],
        local_layer_id=scope["local_layer_id"],
    )


def _handle_forward_pre_normalization(backend, scope):
    log_compressed_attention_prenormalization(
        backend,
        layer=scope["layer"],
        q=scope["q"],
        pool=scope["pool"],
        win_indices=scope["win_indices"],
    )


def _handle_forward_attention_consume(backend, scope):
    capture_ifeval_attention_consume(
        backend,
        layer=scope["layer"],
        forward_batch=scope["forward_batch"],
        core=scope["core"],
        compress_ratio=scope["compress_ratio"],
        win_cache=scope["win_cache"],
        win_indices=scope["win_indices"],
        extra_cache=scope["extra_cache"],
        extra_indices=scope["extra_indices"],
        q_lod=scope["q_lod"],
        kv_lens=scope["kv_lens"],
    )


def _handle_forward_c4_metadata(backend, scope):
    log_c4_attention_metadata(
        layer=scope["layer"],
        forward_batch=scope["forward_batch"],
        core=scope["core"],
        backend_file=BACKEND_FILE,
        q_3d=scope["q_3d"],
        q_lod_cpu=scope["q_lod_cpu"],
        q_lod=scope["q_lod"],
        kv_lens_cpu=scope["kv_lens_cpu"],
        kv_lens=scope["kv_lens"],
    )


def _handle_forward_pre_operator(backend, scope):
    dump_full_attention_snapshot(
        backend,
        layer=scope["layer"],
        q_3d=scope["q_3d"],
        win_cache=scope["win_cache"],
        win_indices=scope["win_indices"],
        extra_cache=scope["extra_cache"],
        extra_indices=scope["extra_indices"],
        q_lod_cpu=scope["q_lod_cpu"],
        q_lod=scope["q_lod"],
        kv_lens_cpu=scope["kv_lens_cpu"],
        kv_lens=scope["kv_lens"],
        softmax_scale=backend.softmax_scale,
        effective_ratio=scope["effective_ratio"],
        compressed_topk=scope["compressed_topk"],
    )
    dump_matched_attention_inputs(
        layer=scope["layer"],
        forward_batch=scope["forward_batch"],
        q_3d=scope["q_3d"],
        win_cache=scope["win_cache"],
        win_indices=scope["win_indices"],
        extra_cache=scope["extra_cache"],
        extra_indices=scope["extra_indices"],
        q_lod_cpu=scope["q_lod_cpu"],
        q_lod=scope["q_lod"],
        kv_lens_cpu=scope["kv_lens_cpu"],
        kv_lens=scope["kv_lens"],
        max_logits=scope["max_logits"],
        lse=scope["lse"],
        softmax_scale=backend.softmax_scale,
        effective_ratio=scope["effective_ratio"],
        compressed_topk=scope["compressed_topk"],
        attn_sink=scope["attn_sink"],
    )
    capture_decode_layer_inputs(
        backend,
        layer=scope["layer"],
        forward_batch=scope["forward_batch"],
        core=scope["core"],
        q_3d=scope["q_3d"],
        win_indices=scope["win_indices"],
        extra_indices=scope["extra_indices"],
        kv_lens=scope["kv_lens"],
    )
    capture_layer42_summary(
        backend,
        layer=scope["layer"],
        forward_batch=scope["forward_batch"],
        q_3d=scope["q_3d"],
        win_cache=scope["win_cache"],
        extra_cache=scope["extra_cache"],
        win_indices=scope["win_indices"],
        extra_indices=scope["extra_indices"],
        kv_lens_cpu=scope["kv_lens_cpu"],
    )


def _handle_forward_operator_inputs(backend, scope):
    capture_attention_aliases_inputs(backend, scope)
    capture_mtp_attention_inputs(backend, scope)
    capture_prefill_backend_inputs(backend, scope)
    dump_compressed_attention_inputs(
        backend,
        q_op=scope["q_op"],
        win_cache_op=scope["win_cache_op"],
        win_indices_op=scope["win_indices_op"],
        extra_cache_op=scope["extra_cache_op"],
        extra_indices_op=scope["extra_indices_op"],
        q_lod_cpu_op=scope["q_lod_cpu_op"],
        q_lod_op=scope["q_lod_op"],
        kv_lens_cpu_op=scope["kv_lens_cpu_op"],
        kv_lens_op=scope["kv_lens_op"],
        attn_sink_op=scope["attn_sink_op"],
        softmax_scale=backend.softmax_scale,
        causal=True,
        effective_ratio=scope["effective_ratio"],
        compressed_topk=scope["compressed_topk"],
    )


def _handle_forward_operator_outputs(backend, scope):
    dump_compressed_attention_outputs(
        backend,
        out_op=scope["out_op"],
        max_logits_op=scope["max_logits_op"],
        lse_op=scope["lse_op"],
    )
    capture_attention_aliases_outputs(backend, scope)
    dump_decode_layer_cache_rows(
        backend,
        forward_batch=scope["forward_batch"],
        win_cache_op=scope["win_cache_op"],
        win_indices_op=scope["win_indices_op"],
        extra_cache_op=scope["extra_cache_op"],
        extra_indices_op=scope["extra_indices_op"],
    )
    dump_prefill_backend_device_probe(backend, scope)
    dump_prefill_backend_probe(backend, scope)
    capture_mtp_attention_outputs(backend, scope)
    dump_decode_layer_attention_output(backend, out=scope["out"])
    dump_matched_attention_output(
        layer=scope["layer"],
        forward_batch=scope["forward_batch"],
        q_3d=scope["q_3d"],
        out=scope["out"],
    )
    dump_layer42_summary(backend, out=scope["out"])


_SITE_HANDLERS = {
    "store_cache.pre_store": _handle_store_cache_pre_store,
    "store_cache.post_store": _handle_store_cache_post_store,
    "forward.pre_normalization": _handle_forward_pre_normalization,
    "forward.attention_consume": _handle_forward_attention_consume,
    "forward.c4_metadata": _handle_forward_c4_metadata,
    "forward.pre_operator": _handle_forward_pre_operator,
    "forward.operator_inputs": _handle_forward_operator_inputs,
    "forward.operator_outputs": _handle_forward_operator_outputs,
    "indexer.c4_logits": lambda backend, scope: dump_c4_logits(scope),
    "indexer.operator_outputs": capture_indexer_operator_probes,
    "multistep.replay_metadata": _handle_multistep_replay_metadata,
    "allocator.alloc_extend_inputs": capture_alloc_extend_inputs,
    "allocator.alloc_extend_outputs": dump_alloc_extend_outputs,
}


def dsv4_probe(backend, site: str, scope: dict) -> None:
    """Single production entry point for every backend probe.

    *scope* is the caller's ``locals()``. Unknown sites and missing names are
    ignored so that a probe can never take down a serving process.
    """
    handler = _SITE_HANDLERS.get(site)
    if handler is None:
        logger.debug("unknown DSV4 probe site: %s", site)
        return
    try:
        handler(backend, scope)
    except KeyError as error:
        logger.warning(
            "DSV4 probe site %s is missing local %s; a production rename "
            "probably broke it",
            site,
            error,
        )



def capture_swa_store_inputs(
    backend,
    *,
    layer_id,
    mapping,
    raw_loc,
    scheduler_raw_loc,
    pack,
    swa_pool,
    local_layer_id,
):
    """Record the SWA writer inputs before the store operator runs."""
    capture_probe = _mtp_writer_probe_enabled(layer_id)
    capture_ifeval_diag = bool(
        os.environ.get("DSV4_IFEVAL_MTP_DIAG_DIR")
        and _dsv4_ifeval_diag_rank() == 0
        and layer_id == 0
    )
    if not (capture_probe or capture_ifeval_diag):
        backend._dsv4_swa_store_state = None
        return

    mapped_loc = mapping.index_select(0, raw_loc.long())
    prefix = None
    if capture_probe:
        prefix = f"step{backend.speculative_step_id}_layer{layer_id}"
        probe = getattr(backend, "_mtp_tensor_probe", {})
        probe.update(
            {
                f"{prefix}.store.scheduler_raw_loc": scheduler_raw_loc,
                f"{prefix}.store.operator_raw_loc": raw_loc,
                f"{prefix}.store.mapped_loc": mapped_loc.clone(),
                f"{prefix}.store.k": pack.k_nope_fp8.clone(),
            }
        )
        backend._mtp_tensor_probe = probe
        contracts = getattr(backend, "_mtp_tensor_probe_contracts", {})
        contracts.setdefault(prefix, {}).update(
            {
                "writer_page_size": swa_pool.page_size,
                "writer_cache_shape": tuple(
                    swa_pool.kv_buffer[local_layer_id].shape
                ),
                "writer_cache_dtype": str(
                    swa_pool.kv_buffer[local_layer_id].dtype
                ),
            }
        )
        backend._mtp_tensor_probe_contracts = contracts
    backend._dsv4_swa_store_state = {
        "capture_probe": capture_probe,
        "capture_ifeval_diag": capture_ifeval_diag,
        "mapped_loc": mapped_loc,
        "prefix": prefix,
        "layer_id": layer_id,
        "scheduler_raw_loc": scheduler_raw_loc,
        "raw_loc": raw_loc,
        "pack": pack,
    }


def capture_swa_store_readback(backend, *, swa_pool, local_layer_id):
    """Read the rows back after the store operator and compare with the write."""
    state = getattr(backend, "_dsv4_swa_store_state", None)
    if not state:
        return
    backend._dsv4_swa_store_state = None

    cache = swa_pool.kv_buffer[local_layer_id].reshape(-1, 512)
    cache_rows = _mtp_gather_cache_rows(cache, state["mapped_loc"]).clone()
    if state["capture_probe"]:
        probe = getattr(backend, "_mtp_tensor_probe", {})
        probe[f"{state['prefix']}.store.cache_rows"] = cache_rows
        backend._mtp_tensor_probe = probe
    if state["capture_ifeval_diag"]:
        packed = state["pack"].k_nope_fp8
        write_matches_readback = bool(
            packed.numel() == cache_rows.numel()
            and torch.equal(packed.reshape(-1), cache_rows.reshape(-1))
        )
        _append_dsv4_ifeval_backend_event(
            backend,
            {
                "kind": "swa_store",
                "layer_id": state["layer_id"],
                "write_matches_readback": write_matches_readback,
                "scheduler_raw_loc": _dsv4_ifeval_tensor_summary(
                    state["scheduler_raw_loc"]
                ),
                "operator_raw_loc": _dsv4_ifeval_tensor_summary(state["raw_loc"]),
                "mapped_loc": _dsv4_ifeval_tensor_summary(state["mapped_loc"]),
                "packed_k": _dsv4_ifeval_tensor_summary(packed),
                "cache_rows": _dsv4_ifeval_tensor_summary(cache_rows),
            },
        )


def dump_full_attention_snapshot(
    backend,
    *,
    layer,
    q_3d,
    win_cache,
    win_indices,
    extra_cache,
    extra_indices,
    q_lod_cpu,
    q_lod,
    kv_lens_cpu,
    kv_lens,
    softmax_scale,
    effective_ratio,
    compressed_topk,
):
    """One-off snapshot of every attention input for the first 8192-token chunk."""
    if not (
        accuracy_dumps_enabled()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and getattr(layer, "layer_id", -1) == 0
        and q_3d.shape[0] == 8192
    ):
        return
    snapshot = dict(getattr(torch, "_dsv4_accuracy_q_stages", {}))
    snapshot.update(
        {
            "meta.pid": os.getpid(),
            "meta.layer_id": getattr(layer, "layer_id", None),
            "meta.backend_type": type(backend).__name__,
            "meta.layer_type": type(layer).__name__,
            "attention.q_after_dtype_cast": q_3d.detach().cpu(),
            "attention.win_kv": win_cache.detach().cpu(),
            "attention.win_indices": win_indices.detach().cpu(),
            "attention.com_kv": extra_cache.detach().cpu(),
            "attention.com_indices": extra_indices.detach().cpu(),
            "attention.qlod_cpu": q_lod_cpu.detach().cpu(),
            "attention.qlod_xpu": q_lod.detach().cpu(),
            "attention.kvseqlen_cpu": kv_lens_cpu.detach().cpu(),
            "attention.kvseqlen_xpu": kv_lens.detach().cpu(),
            "attention.sm_scale": softmax_scale,
            "attention.max_window_size": win_indices.shape[1],
            "attention.compress_ratio": effective_ratio,
            "attention.com_topk": compressed_topk,
        }
    )
    torch.save(
        snapshot,
        f"{ACCURACY_DUMP_DIR}/full_attention_0514_pid{os.getpid()}.pt",
    )


def dump_matched_attention_inputs(
    *,
    layer,
    forward_batch,
    q_3d,
    win_cache,
    win_indices,
    extra_cache,
    extra_indices,
    q_lod_cpu,
    q_lod,
    kv_lens_cpu,
    kv_lens,
    max_logits,
    lse,
    softmax_scale,
    effective_ratio,
    compressed_topk,
    attn_sink,
):
    """Snapshot the exact operator arguments for the second 8192-token chunk."""
    if not _matched_prefill_snapshot_selected(
        layer, forward_batch, q_3d, 0, 8192
    ):
        return
    torch.save(
        {
            "positions": forward_batch.positions.detach().cpu(),
            "q": q_3d.detach().cpu(),
            "win_kv": win_cache.detach().cpu(),
            "win_indices": win_indices.detach().cpu(),
            "com_kv": extra_cache.detach().cpu(),
            "com_indices": extra_indices.detach().cpu(),
            "qlod_cpu": q_lod_cpu.detach().cpu(),
            "qlod_xpu": q_lod.detach().cpu(),
            "kvseqlen_cpu": kv_lens_cpu.detach().cpu(),
            "kvseqlen_xpu": kv_lens.detach().cpu(),
            "max_logits_before": max_logits.detach().cpu(),
            "lse_before": lse.detach().cpu(),
            "softmax_scale": softmax_scale,
            "causal": True,
            "max_window_size": win_indices.shape[1],
            "compress_ratio": effective_ratio,
            "com_topk": compressed_topk,
            "attn_sink": (
                attn_sink.detach().cpu() if attn_sink is not None else None
            ),
        },
        f"{ACCURACY_DUMP_DIR}/matched_attention_0514_pid{os.getpid()}"
        f"_obj{id(layer)}.pt",
    )


def _matched_prefill_snapshot_selected(layer, forward_batch, q_3d, layer_id, position):
    """Shared enable condition for the one-off prefill snapshots."""
    return (
        accuracy_dumps_enabled()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and getattr(layer, "layer_id", -1) == layer_id
        and q_3d.shape[0] == 8192
        and int(forward_batch.positions[0]) == position
    )


def capture_decode_layer_inputs(
    backend,
    *,
    layer,
    forward_batch,
    core,
    q_3d,
    win_indices,
    extra_indices,
    kv_lens,
):
    """Stash the per-layer decode inputs consumed by the decode-alias compare."""
    decode_probe_layers = {
        int(layer_id)
        for layer_id in os.environ.get("DSV4_DECODE_PROBE_LAYERS", "2").split(",")
        if layer_id
    }
    decode_probe_layer = getattr(layer, "layer_id", -1)
    capture_decode_probe = (
        os.environ.get("DSV4_DECODE_LAYER_DUMP") == "1"
        and decode_probe_layer in decode_probe_layers
        and (
            q_3d.shape[0] == 1
            or (
                os.environ.get("DSV4_MTP_VERIFY_LAYER_DUMP") == "1"
                and q_3d.shape[0] == 4
            )
        )
    )
    if not capture_decode_probe:
        backend._dsv4_decode_probe_prefix = None
        return

    from sglang.srt.models.deepseek_v4 import _DSV4_DECODE_LAYER_OUTPUTS

    probe_prefix = f"layer{decode_probe_layer}"
    backend._dsv4_decode_probe_prefix = probe_prefix
    _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_attn_q"] = q_3d.clone()
    _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_win_indices"] = win_indices.clone()
    _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_com_indices"] = extra_indices.clone()
    _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_kv_lens"] = kv_lens.clone()
    for name, value in (
        ("input_ids", forward_batch.input_ids),
        ("positions", forward_batch.positions),
        ("seq_lens", forward_batch.seq_lens),
        ("req_pool_indices", forward_batch.req_pool_indices),
        ("scheduler_out_cache_loc", forward_batch.out_cache_loc),
    ):
        if isinstance(value, torch.Tensor):
            _DSV4_DECODE_LAYER_OUTPUTS[
                f"{probe_prefix}_cache_{name}"
            ] = value.clone()
    for name in (
        "raw_out_loc",
        "swa_out_cache_loc",
        "c4_out_loc",
        "page_table",
        "swa_page_indices",
        "c4_sparse_page_indices",
        "c4_sparse_raw_indices",
    ):
        value = getattr(core, name, None)
        if isinstance(value, torch.Tensor):
            _DSV4_DECODE_LAYER_OUTPUTS[
                f"{probe_prefix}_cache_{name}"
            ] = value.clone()


def capture_layer42_summary(
    backend,
    *,
    layer,
    forward_batch,
    q_3d,
    win_cache,
    extra_cache,
    win_indices,
    extra_indices,
    kv_lens_cpu,
):
    """Count NaNs and index ranges for the layer-42 chunk-3 investigation."""
    if not _matched_prefill_snapshot_selected(
        layer, forward_batch, q_3d, 42, 24576
    ):
        backend._dsv4_layer42_summary = None
        return
    backend._dsv4_layer42_summary = {
        "q_nan": int(torch.isnan(q_3d).sum().item()),
        "win_cache_nan": int(torch.isnan(win_cache).sum().item()),
        "extra_cache_nan": int(torch.isnan(extra_cache).sum().item()),
        "win_indices_min": int(win_indices.min().item()),
        "win_indices_max": int(win_indices.max().item()),
        "extra_indices_min": int(extra_indices.min().item()),
        "extra_indices_max": int(extra_indices.max().item()),
        "kv_lens": kv_lens_cpu.tolist(),
    }


def dump_layer42_summary(backend, *, out):
    """Persist the layer-42 NaN and index summary."""
    summary = getattr(backend, "_dsv4_layer42_summary", None)
    if not summary:
        return
    backend._dsv4_layer42_summary = None
    summary["out_nan"] = int(torch.isnan(out).sum().item())
    torch.save(
        summary,
        f"{ACCURACY_DUMP_DIR}/attention_layer42_summary_0514_pid{os.getpid()}.pt",
    )


def dump_matched_attention_output(*, layer, forward_batch, q_3d, out):
    """Persist attention output for the selected matched prefill chunk."""
    if not _matched_prefill_snapshot_selected(
        layer, forward_batch, q_3d, 0, 8192
    ):
        return
    torch.save(
        out.detach().cpu(),
        f"{ACCURACY_DUMP_DIR}/matched_attention_out_0514_pid{os.getpid()}"
        f"_obj{id(layer)}.pt",
    )


def dump_decode_layer_attention_output(backend, *, out):
    """Store decode attention output in the selected layer probe payload."""
    probe_prefix = getattr(backend, "_dsv4_decode_probe_prefix", None)
    if not probe_prefix:
        return
    from sglang.srt.models.deepseek_v4 import _DSV4_DECODE_LAYER_OUTPUTS

    _DSV4_DECODE_LAYER_OUTPUTS[f"{probe_prefix}_attn_output"] = out.clone()


def dump_decode_layer_cache_rows(
    backend, *, forward_batch, win_cache_op, win_indices_op, extra_cache_op,
    extra_indices_op,
):
    """Store cache rows consumed by the selected decode layer probe."""
    probe_prefix = getattr(backend, "_dsv4_decode_probe_prefix", None)
    if not probe_prefix:
        return
    if os.environ.get("DSV4_DECODE_PROBE_CACHE_ROWS") != "1":
        return
    if forward_batch.seq_lens_cpu is None:
        return
    minimum = int(
        os.environ.get("DSV4_DECODE_PROBE_CACHE_ROWS_MIN_SEQ_LEN", "0")
    )
    if int(forward_batch.seq_lens_cpu[0]) < minimum:
        return

    from sglang.srt.models.deepseek_v4 import _DSV4_DECODE_LAYER_OUTPUTS

    _DSV4_DECODE_LAYER_OUTPUTS[
        f"{probe_prefix}_win_cache_rows"
    ] = _mtp_gather_cache_rows(win_cache_op, win_indices_op)
    _DSV4_DECODE_LAYER_OUTPUTS[
        f"{probe_prefix}_com_cache_rows"
    ] = _mtp_gather_cache_rows(extra_cache_op, extra_indices_op)


# ---------------------------------------------------------------------------
# forward() probes
# ---------------------------------------------------------------------------


def log_compressed_attention_prenormalization(
    backend, *, layer, q, pool, win_indices
):
    """Log the pre-normalization operator arguments once, on rank 0, layer 0."""
    if (
        os.environ.get("DSV4_MTP_PROBE") == "1"
        and os.environ.get("RANK", "0") == "0"
        and getattr(layer, "layer_id", -1) == 0
        and not getattr(backend, "_dsv4_attention_probe_logged", False)
    ):
        logger.warning(
            "[DSV4_CALLSTACK] compressed_attention pre-normalization "
            "q_shape=%s q_dtype=%s win_cache_dtype=%s win_indices_shape=%s "
            "win_indices_min=%s win_indices_max=%s win_indices_head=%s",
            tuple(q.shape),
            q.dtype,
            pool.swa_kv_pool.kv_buffer[0].dtype,
            tuple(win_indices.shape),
            int(win_indices.min().item()) if win_indices.numel() else None,
            int(win_indices.max().item()) if win_indices.numel() else None,
            win_indices[0, :16].detach().cpu().tolist()
            if win_indices.numel()
            else [],
        )
        backend._dsv4_attention_probe_logged = True


def capture_ifeval_attention_consume(
    backend,
    *,
    layer,
    forward_batch,
    core,
    compress_ratio,
    win_cache,
    win_indices,
    extra_cache,
    extra_indices,
    q_lod,
    kv_lens,
):
    """Record what the attention operator consumes, sampling page indices."""
    if not (
        os.environ.get("DSV4_IFEVAL_MTP_DIAG_DIR")
        and _dsv4_ifeval_diag_rank() == 0
        and layer.layer_id == 0
    ):
        return

    def sample_page_indices(indices):
        if not isinstance(indices, torch.Tensor) or indices.ndim < 2:
            return indices
        if indices.shape[-1] <= 8:
            return indices
        return torch.cat((indices[..., :4], indices[..., -4:]), dim=-1)

    sampled_win_indices = sample_page_indices(win_indices)
    sampled_win_cache_rows = _mtp_gather_cache_rows(win_cache, sampled_win_indices)
    sampled_extra_indices = sample_page_indices(extra_indices)
    sampled_extra_cache_rows = (
        _mtp_gather_cache_rows(extra_cache, sampled_extra_indices)
        if isinstance(extra_cache, torch.Tensor)
        and isinstance(sampled_extra_indices, torch.Tensor)
        else None
    )
    _append_dsv4_ifeval_backend_event(
        backend,
        {
            "kind": "attention_consume",
            "layer_id": int(layer.layer_id),
            "compress_ratio": int(compress_ratio),
            "input_ids": _dsv4_ifeval_tensor_summary(
                getattr(forward_batch, "input_ids", None)
            ),
            "positions": _dsv4_ifeval_tensor_summary(
                getattr(forward_batch, "positions", None)
            ),
            "seq_lens": _dsv4_ifeval_tensor_summary(
                getattr(forward_batch, "seq_lens", None)
            ),
            "req_pool_indices": _dsv4_ifeval_tensor_summary(
                getattr(forward_batch, "req_pool_indices", None)
            ),
            "scheduler_out_cache_loc": _dsv4_ifeval_tensor_summary(
                getattr(forward_batch, "out_cache_loc", None)
            ),
            "raw_out_loc": _dsv4_ifeval_tensor_summary(
                getattr(core, "raw_out_loc", None)
            ),
            "page_table": _dsv4_ifeval_tensor_summary(
                getattr(core, "page_table", None)
            ),
            "win_indices": _dsv4_ifeval_tensor_summary(win_indices),
            "sampled_win_indices": _dsv4_ifeval_tensor_summary(
                sampled_win_indices
            ),
            "sampled_win_cache_rows": _dsv4_ifeval_tensor_summary(
                sampled_win_cache_rows
            ),
            "extra_indices": _dsv4_ifeval_tensor_summary(extra_indices),
            "sampled_extra_indices": _dsv4_ifeval_tensor_summary(
                sampled_extra_indices
            ),
            "sampled_extra_cache_rows": _dsv4_ifeval_tensor_summary(
                sampled_extra_cache_rows
            ),
            "q_lod": _dsv4_ifeval_tensor_summary(q_lod),
            "kv_lens": _dsv4_ifeval_tensor_summary(kv_lens),
        },
    )


def log_c4_attention_metadata(
    *,
    layer,
    forward_batch,
    core,
    backend_file,
    q_3d,
    q_lod_cpu,
    q_lod,
    kv_lens_cpu,
    kv_lens,
):
    """Dump the C4 metadata contract for multi-request extend batches."""
    if not (
        os.environ.get("DSV4_C4_ATTN_METADATA_LOG") == "1"
        and forward_batch.batch_size > 1
        and forward_batch.forward_mode.is_extend()
        and layer.layer_id == 0
        and (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
    ):
        return
    logger.warning(
        "[DSV4_C4_ATTN_METADATA] %s",
        json.dumps(
            {
                "backend_file": backend_file,
                "layer_id": layer.layer_id,
                "forward_mode": str(forward_batch.forward_mode),
                "batch_size": forward_batch.batch_size,
                "q": _summarize_c4_metadata_tensor(q_3d),
                "input_ids": _summarize_c4_metadata_tensor(
                    forward_batch.input_ids
                ),
                "positions": _summarize_c4_metadata_tensor(
                    forward_batch.positions
                ),
                "req_pool_indices": _summarize_c4_metadata_tensor(
                    forward_batch.req_pool_indices
                ),
                "seq_lens": _summarize_c4_metadata_tensor(
                    forward_batch.seq_lens
                ),
                "seq_lens_cpu": _summarize_c4_metadata_tensor(
                    forward_batch.seq_lens_cpu
                ),
                "extend_seq_lens_cpu": repr(
                    forward_batch.extend_seq_lens_cpu
                ),
                "out_cache_loc": _summarize_c4_metadata_tensor(
                    forward_batch.out_cache_loc
                ),
                "q_lod_cpu": _summarize_c4_metadata_tensor(q_lod_cpu),
                "q_lod": _summarize_c4_metadata_tensor(q_lod),
                "kv_lens_cpu": _summarize_c4_metadata_tensor(
                    kv_lens_cpu
                ),
                "kv_lens": _summarize_c4_metadata_tensor(kv_lens),
                "raw_out_loc": _summarize_c4_metadata_tensor(
                    getattr(core, "raw_out_loc", None)
                ),
                "page_table": _summarize_c4_metadata_tensor(
                    getattr(core, "page_table", None)
                ),
                "swa_page_indices": _summarize_c4_metadata_tensor(
                    getattr(core, "swa_page_indices", None)
                ),
                "c4_sparse_page_indices": _summarize_c4_metadata_tensor(
                    getattr(core, "c4_sparse_page_indices", None)
                ),
            },
            ensure_ascii=True,
        ),
    )
