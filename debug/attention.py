"""DSV4 attention probes used for precision alignment.

All probes were moved verbatim out of
``sglang_kunlun/hooks/layers/attention/kunlun_deepseek_v4_backend.py`` and
``kunlun_backend.py``.  Activation, environment variables, dump file names and
payload keys are unchanged; only the location of the code moved.

Structure: every probe that has to observe both the operator inputs and the
operator outputs is split into a ``*_begin`` hook returning an opaque context
and a matching ``*_end`` hook.  ``None`` contexts mean "nothing captured".
"""

from __future__ import annotations

import json
import logging
import os
import sys

import torch

from ._dispatch import SiteRegistry
from ._tensors import (
    current_rank,
    eager_tensor_metadata,
    gather_cache_rows,
    ifeval_tensor_summary,
    metadata_tensor_summary,
    tensor_layout,
    tensor_stats,
)

logger = logging.getLogger(__name__)

# The accuracy dumps below were driven by a manually flipped module constant
# rather than an environment variable.  Keep the same default (disabled) so the
# refactor cannot change behaviour; flip it locally when re-running the
# 0514 attention comparisons.
ENABLE_ACCURACY_DUMPS = False

# Directory used by the accuracy dumps that previously hard-coded
# ``/home/zx/debug_dumps``.  The default is unchanged.
LOCAL_DUMP_DIR = os.environ.get("DSV4_LOCAL_DUMP_DIR", "/home/zx/debug_dumps")


def _local_dump_path(name: str) -> str:
    return os.path.join(LOCAL_DUMP_DIR, name)


def _rank0_accuracy_dump_enabled() -> bool:
    return (
        ENABLE_ACCURACY_DUMPS
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
    )


# --------------------------------------------------------------------------
# import banner (DSV4_C4_ATTN_METADATA_LOG)
# --------------------------------------------------------------------------


def log_backend_import(backend_file: str) -> None:
    """Print which attention backend module was imported, per process."""
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


# --------------------------------------------------------------------------
# IFEval MTP diagnosis (DSV4_IFEVAL_MTP_DIAG_DIR)
# --------------------------------------------------------------------------


def _ifeval_diag_active(layer_id: int) -> bool:
    return bool(
        os.environ.get("DSV4_IFEVAL_MTP_DIAG_DIR")
        and current_rank() == 0
        and layer_id == 0
    )


def _append_ifeval_backend_event(owner, event: dict) -> None:
    dump_dir = os.environ.get("DSV4_IFEVAL_MTP_DIAG_DIR")
    if not dump_dir or current_rank() != 0:
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


# --------------------------------------------------------------------------
# MTP tensor probe (DSV4_MTP_TENSOR_DUMP_DIR)
# --------------------------------------------------------------------------


def mtp_tensor_probe_enabled(layer_id: int, num_queries: int) -> bool:
    """Return whether the MTP tensor probe covers this layer and batch."""
    dump_dir = os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR")
    if not dump_dir or os.environ.get("DSV4_MTP_PROBE_SEQ_LEN"):
        return False
    if num_queries != 1 and os.environ.get("DSV4_MTP_TENSOR_PROBE_ANY_BATCH") != "1":
        return False
    probe_layers = {
        int(value)
        for value in os.environ.get("DSV4_MTP_TENSOR_PROBE_LAYERS", "0,21,42").split(",")
        if value
    }
    if layer_id not in probe_layers:
        return False
    return current_rank() == 0


def _mtp_writer_probe_enabled(layer_id: int) -> bool:
    return mtp_tensor_probe_enabled(layer_id, 1) and (
        os.environ.get("DSV4_MTP_TENSOR_PROBE_WRITER") == "1"
    )


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


def _dump_previous_mtp_tensor_probe(multistep_backend, forward_batch) -> None:
    dump_dir = os.environ.get("DSV4_MTP_TENSOR_DUMP_DIR")
    if (
        not dump_dir
        or current_rank() != 0
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
        tensor_metadata[key] = tensor_layout(tensor)

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
            num_pages = min((max_seq_len + 63) // 64, page_table.shape[1])
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
                map(int, forward_batch.seq_lens_cpu[: forward_batch.batch_size])
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
    stats = {}
    for name, tensor in probe.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        cpu_tensor = tensor.detach().cpu()
        tensors[name] = cpu_tensor
        tensor_metadata[name] = eager_tensor_metadata(tensor)
        stats[name] = tensor_stats(cpu_tensor)
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
            "tensor_stats": stats,
            "operator_contracts": getattr(backend, "_mtp_tensor_probe_contracts", {}),
        },
        path,
    )
    logger.warning("[DSV4_MTP_PROBE] target eager tensor dump saved path=%s", path)
    backend._mtp_tensor_dump_cycle = cycle + 1


# --------------------------------------------------------------------------
# SWA cache writer probes (KunlunDeepseekV4AttnBackend.store_cache)
# --------------------------------------------------------------------------


def store_cache_begin(
    backend,
    layer_id: int,
    scheduler_raw_loc: torch.Tensor,
    raw_loc: torch.Tensor,
    mapping: torch.Tensor,
    pack,
    swa_pool,
    local_layer_id: int,
):
    """Capture the SWA writer inputs; returns a context for ``store_cache_end``."""
    capture_probe = _mtp_writer_probe_enabled(layer_id)
    capture_ifeval_diag = _ifeval_diag_active(layer_id)
    if not capture_probe and not capture_ifeval_diag:
        return None

    mapped_loc = mapping.index_select(0, raw_loc.long())
    context = {
        "backend": backend,
        "layer_id": layer_id,
        "capture_probe": capture_probe,
        "capture_ifeval_diag": capture_ifeval_diag,
        "scheduler_raw_loc": scheduler_raw_loc,
        "raw_loc": raw_loc,
        "mapped_loc": mapped_loc,
        "pack": pack,
        "swa_pool": swa_pool,
        "local_layer_id": local_layer_id,
        "prefix": f"step{backend.speculative_step_id}_layer{layer_id}",
    }
    if capture_probe:
        prefix = context["prefix"]
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
                "writer_cache_shape": tuple(swa_pool.kv_buffer[local_layer_id].shape),
                "writer_cache_dtype": str(swa_pool.kv_buffer[local_layer_id].dtype),
            }
        )
        backend._mtp_tensor_probe_contracts = contracts
        context["probe"] = probe
    return context


def store_cache_end(context) -> None:
    """Read the cache rows back and record the writer round-trip."""
    if context is None:
        return
    swa_pool = context["swa_pool"]
    cache = swa_pool.kv_buffer[context["local_layer_id"]].reshape(-1, 512)
    cache_rows = gather_cache_rows(cache, context["mapped_loc"]).clone()
    if context["capture_probe"]:
        context["probe"][f"{context['prefix']}.store.cache_rows"] = cache_rows
    if context["capture_ifeval_diag"]:
        packed = context["pack"].k_nope_fp8
        write_matches_readback = bool(
            packed.numel() == cache_rows.numel()
            and torch.equal(packed.reshape(-1), cache_rows.reshape(-1))
        )
        _append_ifeval_backend_event(
            context["backend"],
            {
                "kind": "swa_store",
                "layer_id": context["layer_id"],
                "write_matches_readback": write_matches_readback,
                "scheduler_raw_loc": ifeval_tensor_summary(
                    context.get("scheduler_raw_loc")
                ),
                "operator_raw_loc": ifeval_tensor_summary(context.get("raw_loc")),
                "mapped_loc": ifeval_tensor_summary(context["mapped_loc"]),
                "packed_k": ifeval_tensor_summary(packed),
                "cache_rows": ifeval_tensor_summary(cache_rows),
            },
        )


# --------------------------------------------------------------------------
# attention forward probes
# --------------------------------------------------------------------------


def log_pre_normalization(backend, layer, q, swa_kv_buffer, win_indices) -> None:
    """Log the compressed-attention inputs once, before index normalization."""
    if (
        os.environ.get("DSV4_MTP_PROBE") != "1"
        or os.environ.get("RANK", "0") != "0"
        or getattr(layer, "layer_id", -1) != 0
        or getattr(backend, "_dsv4_attention_probe_logged", False)
    ):
        return
    logger.warning(
        "[DSV4_CALLSTACK] compressed_attention pre-normalization "
        "q_shape=%s q_dtype=%s win_cache_dtype=%s win_indices_shape=%s "
        "win_indices_min=%s win_indices_max=%s win_indices_head=%s",
        tuple(q.shape),
        q.dtype,
        swa_kv_buffer.dtype,
        tuple(win_indices.shape),
        int(win_indices.min().item()) if win_indices.numel() else None,
        int(win_indices.max().item()) if win_indices.numel() else None,
        win_indices[0, :16].detach().cpu().tolist() if win_indices.numel() else [],
    )
    backend._dsv4_attention_probe_logged = True


def ifeval_attention_consume(
    backend,
    layer,
    compress_ratio: int,
    forward_batch,
    core,
    win_cache,
    win_indices,
    extra_cache,
    extra_indices,
    q_lod,
    kv_lens,
) -> None:
    """Record the tensors the attention operator is about to consume."""
    if not _ifeval_diag_active(int(getattr(layer, "layer_id", -1))):
        return

    def sample_page_indices(indices):
        if not isinstance(indices, torch.Tensor) or indices.ndim < 2:
            return indices
        if indices.shape[-1] <= 8:
            return indices
        return torch.cat((indices[..., :4], indices[..., -4:]), dim=-1)

    sampled_win_indices = sample_page_indices(win_indices)
    sampled_win_cache_rows = gather_cache_rows(win_cache, sampled_win_indices)
    sampled_extra_indices = sample_page_indices(extra_indices)
    sampled_extra_cache_rows = (
        gather_cache_rows(extra_cache, sampled_extra_indices)
        if isinstance(extra_cache, torch.Tensor)
        and isinstance(sampled_extra_indices, torch.Tensor)
        else None
    )
    _append_ifeval_backend_event(
        backend,
        {
            "kind": "attention_consume",
            "layer_id": int(layer.layer_id),
            "compress_ratio": int(compress_ratio),
            "input_ids": ifeval_tensor_summary(
                getattr(forward_batch, "input_ids", None)
            ),
            "positions": ifeval_tensor_summary(
                getattr(forward_batch, "positions", None)
            ),
            "seq_lens": ifeval_tensor_summary(getattr(forward_batch, "seq_lens", None)),
            "req_pool_indices": ifeval_tensor_summary(
                getattr(forward_batch, "req_pool_indices", None)
            ),
            "scheduler_out_cache_loc": ifeval_tensor_summary(
                getattr(forward_batch, "out_cache_loc", None)
            ),
            "raw_out_loc": ifeval_tensor_summary(getattr(core, "raw_out_loc", None)),
            "page_table": ifeval_tensor_summary(getattr(core, "page_table", None)),
            "win_indices": ifeval_tensor_summary(win_indices),
            "sampled_win_indices": ifeval_tensor_summary(sampled_win_indices),
            "sampled_win_cache_rows": ifeval_tensor_summary(sampled_win_cache_rows),
            "extra_indices": ifeval_tensor_summary(extra_indices),
            "sampled_extra_indices": ifeval_tensor_summary(sampled_extra_indices),
            "sampled_extra_cache_rows": ifeval_tensor_summary(sampled_extra_cache_rows),
            "q_lod": ifeval_tensor_summary(q_lod),
            "kv_lens": ifeval_tensor_summary(kv_lens),
        },
    )


def log_dsv4_attn_metadata(
    backend_file: str,
    layer,
    forward_batch,
    core,
    q_3d,
    q_lod_cpu,
    q_lod,
    kv_lens_cpu,
    kv_lens,
) -> None:
    """Log the DSV4 attention metadata of the first multi-request extend batch."""
    if (
        os.environ.get("DSV4_C4_ATTN_METADATA_LOG") != "1"
        or forward_batch.batch_size <= 1
        or not forward_batch.forward_mode.is_extend()
        or layer.layer_id != 0
        or (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
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
                "q": metadata_tensor_summary(q_3d),
                "input_ids": metadata_tensor_summary(forward_batch.input_ids),
                "positions": metadata_tensor_summary(forward_batch.positions),
                "req_pool_indices": metadata_tensor_summary(
                    forward_batch.req_pool_indices
                ),
                "seq_lens": metadata_tensor_summary(forward_batch.seq_lens),
                "seq_lens_cpu": metadata_tensor_summary(forward_batch.seq_lens_cpu),
                "extend_seq_lens_cpu": repr(forward_batch.extend_seq_lens_cpu),
                "out_cache_loc": metadata_tensor_summary(forward_batch.out_cache_loc),
                "q_lod_cpu": metadata_tensor_summary(q_lod_cpu),
                "q_lod": metadata_tensor_summary(q_lod),
                "kv_lens_cpu": metadata_tensor_summary(kv_lens_cpu),
                "kv_lens": metadata_tensor_summary(kv_lens),
                "raw_out_loc": metadata_tensor_summary(
                    getattr(core, "raw_out_loc", None)
                ),
                "page_table": metadata_tensor_summary(getattr(core, "page_table", None)),
                "swa_page_indices": metadata_tensor_summary(
                    getattr(core, "swa_page_indices", None)
                ),
                "c4_sparse_page_indices": metadata_tensor_summary(
                    getattr(core, "c4_sparse_page_indices", None)
                ),
            },
            ensure_ascii=True,
        ),
    )


def _accuracy_dump_attention_inputs(context) -> None:
    """Dump the full attention input set (module-constant gated)."""
    if not _rank0_accuracy_dump_enabled():
        return
    layer = context["layer"]
    if getattr(layer, "layer_id", -1) != 0 or context["q_3d"].shape[0] != 8192:
        return
    snapshot = dict(getattr(torch, "_dsv4_accuracy_q_stages", {}))
    snapshot.update(
        {
            "meta.pid": os.getpid(),
            "meta.layer_id": getattr(layer, "layer_id", None),
            "meta.backend_type": type(context["backend"]).__name__,
            "meta.layer_type": type(layer).__name__,
            "attention.q_after_dtype_cast": context["q_3d"].detach().cpu(),
            "attention.win_kv": context["win_cache"].detach().cpu(),
            "attention.win_indices": context["win_indices"].detach().cpu(),
            "attention.com_kv": context["extra_cache"].detach().cpu(),
            "attention.com_indices": context["extra_indices"].detach().cpu(),
            "attention.qlod_cpu": context["q_lod_cpu"].detach().cpu(),
            "attention.qlod_xpu": context["q_lod"].detach().cpu(),
            "attention.kvseqlen_cpu": context["kv_lens_cpu"].detach().cpu(),
            "attention.kvseqlen_xpu": context["kv_lens"].detach().cpu(),
            "attention.sm_scale": context["softmax_scale"],
            "attention.max_window_size": context["win_indices"].shape[1],
            "attention.compress_ratio": context["effective_ratio"],
            "attention.com_topk": context["compressed_topk"],
        }
    )
    torch.save(
        snapshot,
        _local_dump_path(f"full_attention_0514_pid{os.getpid()}.pt"),
    )

    if int(context["forward_batch"].positions[0]) != 8192:
        return
    torch.save(
        {
            "positions": context["forward_batch"].positions.detach().cpu(),
            "q": context["q_3d"].detach().cpu(),
            "win_kv": context["win_cache"].detach().cpu(),
            "win_indices": context["win_indices"].detach().cpu(),
            "com_kv": context["extra_cache"].detach().cpu(),
            "com_indices": context["extra_indices"].detach().cpu(),
            "qlod_cpu": context["q_lod_cpu"].detach().cpu(),
            "qlod_xpu": context["q_lod"].detach().cpu(),
            "kvseqlen_cpu": context["kv_lens_cpu"].detach().cpu(),
            "kvseqlen_xpu": context["kv_lens"].detach().cpu(),
            "max_logits_before": context["max_logits"].detach().cpu(),
            "lse_before": context["lse"].detach().cpu(),
            "softmax_scale": context["softmax_scale"],
            "causal": True,
            "max_window_size": context["win_indices"].shape[1],
            "compress_ratio": context["effective_ratio"],
            "com_topk": context["compressed_topk"],
            "attn_sink": (
                context["attn_sink"].detach().cpu()
                if context["attn_sink"] is not None
                else None
            ),
        },
        _local_dump_path(
            f"matched_attention_0514_pid{os.getpid()}_obj{id(layer)}.pt"
        ),
    )
    context["dump_matched_output"] = True


def _capture_decode_probe(context) -> None:
    """Copy decode-time attention inputs into the shared decode probe dict."""
    layer_id = int(getattr(context["layer"], "layer_id", -1))
    decode_probe_layers = {
        int(value)
        for value in os.environ.get("DSV4_DECODE_PROBE_LAYERS", "2").split(",")
        if value
    }
    num_queries = context["q_3d"].shape[0]
    if not (
        os.environ.get("DSV4_DECODE_LAYER_DUMP") == "1"
        and layer_id in decode_probe_layers
        and (
            num_queries == 1
            or (
                os.environ.get("DSV4_MTP_VERIFY_LAYER_DUMP") == "1"
                and num_queries == 4
            )
        )
    ):
        return

    from sglang.srt.models.deepseek_v4 import _DSV4_DECODE_LAYER_OUTPUTS

    forward_batch = context["forward_batch"]
    core = context["core"]
    prefix = f"layer{layer_id}"
    context["decode_probe_prefix"] = prefix
    context["decode_probe_outputs"] = _DSV4_DECODE_LAYER_OUTPUTS
    _DSV4_DECODE_LAYER_OUTPUTS[f"{prefix}_attn_q"] = context["q_3d"].clone()
    _DSV4_DECODE_LAYER_OUTPUTS[f"{prefix}_win_indices"] = context[
        "win_indices"
    ].clone()
    _DSV4_DECODE_LAYER_OUTPUTS[f"{prefix}_com_indices"] = context[
        "extra_indices"
    ].clone()
    _DSV4_DECODE_LAYER_OUTPUTS[f"{prefix}_kv_lens"] = context["kv_lens"].clone()
    for name, value in (
        ("input_ids", forward_batch.input_ids),
        ("positions", forward_batch.positions),
        ("seq_lens", forward_batch.seq_lens),
        ("req_pool_indices", forward_batch.req_pool_indices),
        ("scheduler_out_cache_loc", forward_batch.out_cache_loc),
    ):
        if isinstance(value, torch.Tensor):
            _DSV4_DECODE_LAYER_OUTPUTS[f"{prefix}_cache_{name}"] = value.clone()
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
            _DSV4_DECODE_LAYER_OUTPUTS[f"{prefix}_cache_{name}"] = value.clone()


def _capture_layer42_summary(context) -> None:
    layer = context["layer"]
    if (
        not _rank0_accuracy_dump_enabled()
        or getattr(layer, "layer_id", -1) != 42
        or context["q_3d"].shape[0] != 8192
        or int(context["forward_batch"].positions[0]) != 24576
    ):
        return
    context["layer42_summary"] = {
        "q_nan": int(torch.isnan(context["q_3d"]).sum().item()),
        "win_cache_nan": int(torch.isnan(context["win_cache"]).sum().item()),
        "extra_cache_nan": int(torch.isnan(context["extra_cache"]).sum().item()),
        "win_indices_min": int(context["win_indices"].min().item()),
        "win_indices_max": int(context["win_indices"].max().item()),
        "extra_indices_min": int(context["extra_indices"].min().item()),
        "extra_indices_max": int(context["extra_indices"].max().item()),
        "kv_lens": context["kv_lens_cpu"].tolist(),
    }


def _capture_attention_aliases(context) -> None:
    """Alias the operator inputs for the decode operator-replay comparison."""
    forward_batch = context["forward_batch"]
    aliases = getattr(forward_batch, "_dsv4_decode_attention_aliases", None)
    alias_layer = int(os.environ.get("DSV4_DECODE_ATTENTION_ALIAS_LAYER", "2"))
    if aliases is None or context["layer"].layer_id != alias_layer:
        return
    operator_only = os.environ.get("DSV4_DECODE_ATTENTION_ALIAS_OPERATOR_ONLY") == "1"
    context["aliases"] = aliases
    context["operator_only_alias"] = operator_only
    win_indices_op = context["win_indices_op"]
    aliases.update(
        {
            "attention_q": (
                context["q_op"].clone() if operator_only else context["q_op"]
            ),
            "attention_win_cache": context["win_cache_op"],
            "attention_win_indices": (
                win_indices_op.clone() if operator_only else win_indices_op
            ),
            "attention_extra_cache": context["extra_cache_op"],
            "attention_extra_indices": (
                context["extra_indices_op"].clone()
                if operator_only
                else context["extra_indices_op"]
            ),
            "attention_q_lod_cpu": context["q_lod_cpu_op"],
            "attention_q_lod": (
                context["q_lod_op"].clone() if operator_only else context["q_lod_op"]
            ),
            "attention_kv_lens_cpu": context["kv_lens_cpu_op"],
            "attention_kv_lens": (
                context["kv_lens_op"].clone()
                if operator_only
                else context["kv_lens_op"]
            ),
            "attention_softmax_scale": torch.tensor(
                context["softmax_scale"], dtype=torch.float64
            ),
            "attention_causal": torch.tensor(True),
            "attention_max_window_size": torch.tensor(
                win_indices_op.shape[1], dtype=torch.int64
            ),
            "attention_compress_ratio": torch.tensor(
                context["effective_ratio"], dtype=torch.int64
            ),
            "attention_compressed_topk": torch.tensor(
                context["compressed_topk"], dtype=torch.int64
            ),
        }
    )
    if context["attn_sink_op"] is not None:
        aliases["attention_sink"] = context["attn_sink_op"]
    if operator_only:
        return
    core = context["core"]
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
            "metadata_req_pool_indices": forward_batch.req_pool_indices,
            "metadata_req_to_token": context["req_to_token"],
        }
    )


def _capture_mtp_attention_probe(context) -> None:
    backend = context["backend"]
    layer = context["layer"]
    forward_batch = context["forward_batch"]
    if not (
        forward_batch.forward_mode.is_extend()
        and not torch.cuda.is_current_stream_capturing()
        and mtp_tensor_probe_enabled(
            getattr(layer, "layer_id", -1), context["q_op"].shape[0]
        )
    ):
        return
    core = context["core"]
    prefix = f"step{backend.speculative_step_id}_layer{layer.layer_id}"
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
            f"{prefix}.attention.q": context["q_op"].clone(),
            f"{prefix}.attention.win_indices": context["win_indices_op"].clone(),
            f"{prefix}.attention.win_cache_rows": gather_cache_rows(
                context["win_cache_op"], context["win_indices_op"]
            ).clone(),
            f"{prefix}.attention.extra_indices": context["extra_indices_op"].clone(),
            f"{prefix}.attention.extra_cache_rows": gather_cache_rows(
                context["extra_cache_op"], context["extra_indices_op"]
            ).clone(),
            f"{prefix}.attention.q_lod_cpu": context["q_lod_cpu_op"],
            f"{prefix}.attention.q_lod": context["q_lod_op"].clone(),
            f"{prefix}.attention.kv_lens_cpu": context["kv_lens_cpu_op"],
            f"{prefix}.attention.kv_lens": context["kv_lens_op"].clone(),
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
    if context["attn_sink_op"] is not None:
        probe[f"{prefix}.attention.attn_sink"] = context["attn_sink_op"]
    backend._mtp_tensor_probe = probe
    contracts = getattr(backend, "_mtp_tensor_probe_contracts", {})
    contracts.setdefault(prefix, {}).update(
        {
            "softmax_scale": context["softmax_scale"],
            "causal": True,
            "max_window_size": context["win_indices_op"].shape[1],
            "compress_ratio": context["effective_ratio"],
            "compressed_topk": context["compressed_topk"],
            "page_size": context["page_size"],
        }
    )
    backend._mtp_tensor_probe_contracts = contracts
    context["mtp_probe"] = probe
    context["mtp_probe_prefix"] = prefix


def _prefill_probe_layer_matches(layer) -> bool:
    return getattr(layer, "layer_id", -1) == int(
        os.environ.get("DSV4_PREFILL_PROBE_LAYER", "0")
    )


def _capture_prefill_backend_probe(context) -> None:
    """Capture the chunk-1 prefill tail used by the operator replay script."""
    forward_batch = context["forward_batch"]
    if not (
        os.environ.get("DSV4_PREFILL_BACKEND_PROBE") == "1"
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and _prefill_probe_layer_matches(context["layer"])
        and context["q_op"].shape[0] == 8192
        and int(forward_batch.positions[0].item()) == 8192
    ):
        return
    core = context["core"]
    tail = slice(7680, 8192)
    tail_win_indices = context["win_indices_op"][tail]
    context["prefill_backend_probe"] = {
        "positions": forward_batch.positions.detach().cpu(),
        "q_tail": context["q_op"][tail].detach().cpu(),
        "win_indices_tail": tail_win_indices.detach().cpu(),
        "win_cache_rows_tail": gather_cache_rows(
            context["win_cache_op"], tail_win_indices
        )
        .detach()
        .cpu(),
        "q_lod_cpu": context["q_lod_cpu_op"].detach().cpu(),
        "q_lod": context["q_lod_op"].detach().cpu(),
        "kv_lens_cpu": context["kv_lens_cpu_op"].detach().cpu(),
        "kv_lens": context["kv_lens_op"].detach().cpu(),
        "attn_sink": (
            context["attn_sink_op"].detach().cpu()
            if context["attn_sink_op"] is not None
            else None
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
        "win_cache_shape": tuple(context["win_cache_op"].shape),
        "win_cache_dtype": str(context["win_cache_op"].dtype),
        "win_indices_shape": tuple(context["win_indices_op"].shape),
        "softmax_scale": context["softmax_scale"],
        "causal": True,
        "max_window_size": context["win_indices_op"].shape[1],
        "compress_ratio": context["effective_ratio"],
        "compressed_topk": context["compressed_topk"],
        "page_size": context["page_size"],
    }
    backend = context["backend"]
    logger.warning(
        "DSV4_COMPRESSED_ATTN_STACK version=0514 backend=%s layer=%s "
        "q_shape=%s win_cache_shape=%s win_indices_shape=%s "
        "win_cache_ptr=%s side_stream=%s",
        f"{type(backend).__module__}.{type(backend).__qualname__}",
        getattr(context["layer"], "layer_id", -1),
        tuple(context["q_op"].shape),
        tuple(context["win_cache_op"].shape),
        tuple(context["win_indices_op"].shape),
        context["win_cache_op"].data_ptr(),
        torch.cuda.current_stream().cuda_stream,
        stack_info=True,
    )


def _prefill_device_probe_enabled(context) -> bool:
    forward_batch = context["forward_batch"]
    prefix_len = int(os.environ.get("DSV4_PREFILL_PROBE_PREFIX_LEN", "8192"))
    context["prefill_probe_prefix_len"] = prefix_len
    return (
        os.environ.get("DSV4_PREFILL_BACKEND_DEVICE_PROBE") == "1"
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() == 0
        and _prefill_probe_layer_matches(context["layer"])
        and context["q_op"].shape[0] == 8192
        and prefix_len in (forward_batch.extend_prefix_lens_cpu or [])
    )


def operator_begin(
    *,
    backend,
    layer,
    forward_batch,
    core,
    softmax_scale,
    page_size,
    req_to_token,
    effective_ratio,
    compressed_topk,
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
    attn_sink,
    q_op,
    win_cache_op,
    win_indices_op,
    extra_cache_op,
    extra_indices_op,
    out_op,
    max_logits_op,
    lse_op,
    q_lod_cpu_op,
    q_lod_op,
    kv_lens_cpu_op,
    kv_lens_op,
    attn_sink_op,
):
    """Capture every compressed-attention input probe.

    Returns an opaque context to hand to :func:`operator_end`, or ``None`` when
    no probe is active.
    """
    context = {
        "backend": backend,
        "layer": layer,
        "forward_batch": forward_batch,
        "core": core,
        "softmax_scale": softmax_scale,
        "page_size": page_size,
        "req_to_token": req_to_token,
        "effective_ratio": effective_ratio,
        "compressed_topk": compressed_topk,
        "q_3d": q_3d,
        "win_cache": win_cache,
        "win_indices": win_indices,
        "extra_cache": extra_cache,
        "extra_indices": extra_indices,
        "q_lod_cpu": q_lod_cpu,
        "q_lod": q_lod,
        "kv_lens_cpu": kv_lens_cpu,
        "kv_lens": kv_lens,
        "max_logits": max_logits,
        "lse": lse,
        "attn_sink": attn_sink,
        "q_op": q_op,
        "win_cache_op": win_cache_op,
        "win_indices_op": win_indices_op,
        "extra_cache_op": extra_cache_op,
        "extra_indices_op": extra_indices_op,
        "out_op": out_op,
        "max_logits_op": max_logits_op,
        "lse_op": lse_op,
        "q_lod_cpu_op": q_lod_cpu_op,
        "q_lod_op": q_lod_op,
        "kv_lens_cpu_op": kv_lens_cpu_op,
        "kv_lens_op": kv_lens_op,
        "attn_sink_op": attn_sink_op,
        "dump_matched_output": False,
        "layer42_summary": None,
        "decode_probe_prefix": None,
        "prefill_backend_probe": None,
        "aliases": None,
        "mtp_probe": None,
    }
    _accuracy_dump_attention_inputs(context)
    _capture_decode_probe(context)
    _capture_layer42_summary(context)
    _capture_attention_aliases(context)
    _capture_mtp_attention_probe(context)
    _capture_prefill_backend_probe(context)
    context["capture_prefill_device_probe"] = _prefill_device_probe_enabled(context)
    if not any(
        (
            context["dump_matched_output"],
            context["layer42_summary"] is not None,
            context["decode_probe_prefix"] is not None,
            context["prefill_backend_probe"] is not None,
            context["aliases"] is not None,
            context["mtp_probe"] is not None,
            context["capture_prefill_device_probe"],
        )
    ):
        return None
    return context


def _alias_attention_outputs(context) -> None:
    aliases = context["aliases"]
    if aliases is None:
        return
    operator_only = context["operator_only_alias"]
    aliases.update(
        {
            "attention_output": (
                context["out_op"].clone() if operator_only else context["out_op"]
            ),
            "attention_max_logits": (
                context["max_logits_op"].clone()
                if operator_only
                else context["max_logits_op"]
            ),
            "attention_lse": (
                context["lse_op"].clone() if operator_only else context["lse_op"]
            ),
        }
    )


def _decode_probe_cache_rows(context) -> None:
    prefix = context["decode_probe_prefix"]
    forward_batch = context["forward_batch"]
    if (
        prefix is None
        or os.environ.get("DSV4_DECODE_PROBE_CACHE_ROWS") != "1"
        or forward_batch.seq_lens_cpu is None
        or int(forward_batch.seq_lens_cpu[0])
        < int(os.environ.get("DSV4_DECODE_PROBE_CACHE_ROWS_MIN_SEQ_LEN", "0"))
    ):
        return
    outputs = context["decode_probe_outputs"]
    outputs[f"{prefix}_win_cache_rows"] = gather_cache_rows(
        context["win_cache_op"], context["win_indices_op"]
    )
    outputs[f"{prefix}_com_cache_rows"] = gather_cache_rows(
        context["extra_cache_op"], context["extra_indices_op"]
    )


def _dump_prefill_device_probe(context) -> None:
    if not context["capture_prefill_device_probe"]:
        return
    core = context["core"]
    forward_batch = context["forward_batch"]
    backend = context["backend"]
    layer = context["layer"]
    extra_unique_indices = torch.unique(context["extra_indices_op"])
    win_unique_indices = torch.unique(context["win_indices_op"])
    probe = {
        "input_ids": forward_batch.input_ids.clone(),
        "positions": forward_batch.positions.clone(),
        "q_local": context["q_op"][:, :8].clone(),
        "win_indices": context["win_indices_op"].clone(),
        "win_unique_indices": win_unique_indices,
        "win_unique_cache_rows": context["win_cache_op"]
        .index_select(0, win_unique_indices.to(torch.long))
        .clone(),
        "extra_indices": context["extra_indices_op"].clone(),
        "extra_unique_indices": extra_unique_indices,
        "extra_unique_cache_rows": context["extra_cache_op"]
        .index_select(0, extra_unique_indices.to(torch.long))
        .clone(),
        "output_local": context["out_op"][:, :8].clone(),
        "max_logits_local": context["max_logits_op"][:, :8].clone(),
        "lse_local": context["lse_op"][:, :8].clone(),
        "q_lod_cpu": context["q_lod_cpu_op"].clone(),
        "q_lod": context["q_lod_op"].clone(),
        "kv_lens_cpu": context["kv_lens_cpu_op"].clone(),
        "kv_lens": context["kv_lens_op"].clone(),
        "attn_sink": (
            context["attn_sink_op"].clone()
            if context["attn_sink_op"] is not None
            else None
        ),
        "win_cache_shape": tuple(context["win_cache_op"].shape),
        "extra_cache_shape": tuple(context["extra_cache_op"].shape),
        "softmax_scale": context["softmax_scale"],
        "causal": True,
        "max_window_size": context["win_indices_op"].shape[1],
        "compress_ratio": context["effective_ratio"],
        "compressed_topk": context["compressed_topk"],
        "page_size": context["page_size"],
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
            probe[name] = tensor.clone()
    logger.warning(
        "DSV4_C128_PREFILL_STACK version=0514 backend=%s layer=%s "
        "q_shape=%s win_indices_shape=%s extra_indices_shape=%s "
        "win_cache_shape=%s extra_cache_shape=%s side_stream=%s",
        f"{type(backend).__module__}.{type(backend).__qualname__}",
        getattr(layer, "layer_id", -1),
        tuple(context["q_op"].shape),
        tuple(context["win_indices_op"].shape),
        tuple(context["extra_indices_op"].shape),
        tuple(context["win_cache_op"].shape),
        tuple(context["extra_cache_op"].shape),
        torch.cuda.current_stream().cuda_stream,
        stack_info=True,
    )
    torch.save(
        {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in probe.items()
        },
        _local_dump_path(
            "prefill_backend_device_0514_layer"
            f"{getattr(layer, 'layer_id', -1)}_prefix"
            f"{context['prefill_probe_prefix_len']}_rank0.pt"
        ),
    )


def operator_end(context, out: torch.Tensor) -> None:
    """Record the compressed-attention outputs for every active probe."""
    if context is None:
        return
    _alias_attention_outputs(context)
    _decode_probe_cache_rows(context)
    _dump_prefill_device_probe(context)

    prefill_probe = context["prefill_backend_probe"]
    if prefill_probe is not None:
        tail = slice(7680, 8192)
        prefill_probe.update(
            {
                "output_tail": context["out_op"][tail].detach().cpu(),
                "max_logits_tail": context["max_logits_op"][tail].detach().cpu(),
                "lse_tail": context["lse_op"][tail].detach().cpu(),
            }
        )
        torch.save(
            prefill_probe,
            _local_dump_path(
                "prefill_backend_0514_repeat_layer0_chunk1_rank0.pt"
            ),
        )
    if context["mtp_probe"] is not None:
        probe = context["mtp_probe"]
        prefix = context["mtp_probe_prefix"]
        probe[f"{prefix}.attention.output"] = context["out_op"].clone()
        probe[f"{prefix}.attention.max_logits"] = context["max_logits_op"].clone()
        probe[f"{prefix}.attention.lse"] = context["lse_op"].clone()
        _dump_eager_mtp_tensor_probe(context["backend"])
    if context["decode_probe_prefix"] is not None:
        context["decode_probe_outputs"][
            f"{context['decode_probe_prefix']}_attn_output"
        ] = out.clone()
    layer = context["layer"]
    if context["dump_matched_output"]:
        torch.save(
            out.detach().cpu(),
            _local_dump_path(
                f"matched_attention_out_0514_pid{os.getpid()}_obj{id(layer)}.pt"
            ),
        )
    if context["layer42_summary"] is not None:
        summary = context["layer42_summary"]
        summary["out_nan"] = int(torch.isnan(out).sum().item())
        torch.save(
            summary,
            _local_dump_path(f"attention_layer42_summary_0514_pid{os.getpid()}.pt"),
        )


# --------------------------------------------------------------------------
# C4 indexer logits probes (_compute_c4_logits_kunlun)
# --------------------------------------------------------------------------


def c4_logits_begin(backend, c4_indexer, forward_batch, q_fp8):
    """Decide whether the C4 logits probes are active for this call."""
    layer_id = c4_indexer.layer_id
    capture_mtp_probe = (
        forward_batch.forward_mode.is_extend()
        and not torch.cuda.is_current_stream_capturing()
        and mtp_tensor_probe_enabled(layer_id, q_fp8.shape[0])
    )
    return {
        "backend": backend,
        "layer_id": layer_id,
        "capture_mtp_probe": capture_mtp_probe,
        "prefix": f"step{backend.speculative_step_id}_layer{layer_id}",
    }


def c4_prefill_logits_dump(context, logits, num_q_tokens: int, max_seq_len: int) -> None:
    """Persist a three-row sample of the contiguous-prefill C4 logits."""
    del context
    dump_dir = os.environ.get("DSV4_ACCURACY_DUMP_DIR")
    dump_tokens = int(os.environ.get("DSV4_ACCURACY_DUMP_TOKENS", "0"))
    if not dump_dir or (dump_tokens and num_q_tokens != dump_tokens):
        return
    rank = current_rank()
    if rank != int(os.environ.get("DSV4_ACCURACY_DUMP_RANK", "0")):
        return
    sample_rows = torch.tensor(
        [0, logits.shape[0] // 2, logits.shape[0] - 1],
        dtype=torch.long,
        device=logits.device,
    )
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"rank{rank}_c4_logits_{num_q_tokens}.pt")
    if os.path.exists(path):
        return
    torch.save(
        {
            "sample_rows": sample_rows.detach().cpu(),
            "logits": logits.index_select(0, sample_rows).detach().cpu(),
            "max_seq_len": max_seq_len,
        },
        path,
    )


def c4_paged_logits_inputs(
    context,
    *,
    forward_batch,
    is_target_verify: bool,
    q,
    weights,
    seq_lens,
    page_table,
    block_table,
    k_cache,
    k_scale,
    qlod_cpu,
    qlod_xpu,
    context_lens_cpu,
    context_lens_xpu,
    logits,
    batch_size: int,
    block_size: int,
    head_dim: int,
    max_seq_len: int,
) -> None:
    """Alias/record the paged C4 logits operator inputs."""
    layer_id = context["layer_id"]
    aliases = getattr(forward_batch, "_dsv4_decode_attention_aliases", None)
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

    if os.environ.get("DSV4_MTP_VERIFY_LAYER_DUMP") == "1" and is_target_verify:
        from sglang.srt.models.deepseek_v4 import _DSV4_DECODE_LAYER_OUTPUTS

        probe_pages = min(140, block_table.shape[1])
        selected_k_cache = k_cache.index_select(
            0,
            block_table[:, :probe_pages]
            .reshape(-1)
            .long()
            .clamp(0, k_cache.shape[0] - 1),
        ).view(batch_size, probe_pages, block_size, 1, head_dim)
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

    if not context["capture_mtp_probe"]:
        return
    backend = context["backend"]
    prefix = context["prefix"]
    probe = getattr(backend, "_mtp_tensor_probe", {})
    selected_k_cache = k_cache.index_select(
        0, block_table.reshape(-1).long().clamp(0, k_cache.shape[0] - 1)
    )
    probe.update(
        {
            f"{prefix}.indexer.operator.q": q.clone(),
            f"{prefix}.indexer.operator.weight": weights.clone(),
            f"{prefix}.indexer.operator.seq_lens": seq_lens.clone(),
            f"{prefix}.indexer.operator.page_table": page_table.clone(),
            f"{prefix}.indexer.operator.block_table": block_table.clone(),
            f"{prefix}.indexer.operator.k_cache_pages": selected_k_cache.clone(),
            f"{prefix}.indexer.operator.k_scale": k_scale.clone(),
            f"{prefix}.indexer.operator.q_lod_cpu": qlod_cpu,
            f"{prefix}.indexer.operator.q_lod": qlod_xpu.clone(),
            f"{prefix}.indexer.operator.context_lens_cpu": context_lens_cpu,
            f"{prefix}.indexer.operator.context_lens": context_lens_xpu.clone(),
        }
    )
    backend._mtp_tensor_probe = probe
    contracts = getattr(backend, "_mtp_tensor_probe_contracts", {})
    contracts.setdefault(prefix, {}).update(
        {
            "indexer_max_context_len": max_seq_len * 4,
            "indexer_compress_ratio": 4,
            "indexer_clean_logits": True,
            "indexer_use_xfa_boost": False,
        }
    )
    backend._mtp_tensor_probe_contracts = contracts
    context["probe"] = probe


def c4_paged_logits_output(context, logits) -> None:
    """Record the paged C4 logits produced by the operator."""
    if not context["capture_mtp_probe"]:
        return
    context["probe"][f"{context['prefix']}.indexer.operator.logits"] = logits.clone()


# --------------------------------------------------------------------------
# multi-step replay probe
# --------------------------------------------------------------------------


def multistep_replay_probe(multistep_backend, forward_batch, in_capture: bool) -> None:
    """Dump/log the per-step replay metadata of the multi-step backend."""
    if not in_capture:
        _dump_previous_mtp_tensor_probe(multistep_backend, forward_batch)
    replay_calls = getattr(multistep_backend, "_mtp_probe_replay_calls", 0)
    if (
        in_capture
        or os.environ.get("DSV4_MTP_PROBE") != "1"
        or os.environ.get("RANK", "0") != "0"
        or forward_batch.seq_lens_cpu is None
        or int(forward_batch.seq_lens_cpu[: forward_batch.batch_size].max()) <= 10000
        or replay_calls >= 3
    ):
        return

    replay_calls += 1
    multistep_backend._mtp_probe_replay_calls = replay_calls

    def values(tensor, limit=8):
        if tensor is None:
            return None
        return tensor.detach().reshape(-1)[:limit].cpu().tolist()

    def pointer(tensor):
        return tensor.data_ptr() if tensor is not None else None

    steps = multistep_backend.attn_backends[
        : multistep_backend.speculative_num_steps - 1
    ]
    for step, backend in enumerate(steps):
        metadata = backend.forward_metadata
        core = getattr(metadata, "core_attn_metadata", None)
        if core is None:
            logger.warning(
                "[DSV4_MTP_PROBE] dsv4_metadata replay=%d step=%d metadata=%s",
                replay_calls,
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
            replay_calls,
            step,
            backend.speculative_step_id,
            values(forward_batch.seq_lens),
            values(forward_batch.seq_lens_cpu),
            values(getattr(forward_batch, "positions", None)),
            values(forward_batch.req_pool_indices),
            values(getattr(core, "raw_out_loc", None), 12),
            values(getattr(core, "seq_lens_casual", None)),
            values(getattr(core, "positions_casual", None)),
            values(getattr(core, "page_table", None), 4),
            values(getattr(core, "swa_page_indices", None), 8),
            values(getattr(core, "swa_topk_lengths", None)),
            values(getattr(core, "c4_out_loc", None), 12),
            values(getattr(core, "c4_topk_lengths_raw", None)),
            values(getattr(core, "c4_topk_lengths_clamp1", None)),
            pointer(getattr(core, "raw_out_loc", None)),
            pointer(getattr(core, "page_table", None)),
            pointer(getattr(core, "swa_page_indices", None)),
            pointer(getattr(core, "c4_out_loc", None)),
        )


# --------------------------------------------------------------------------
# generic Kunlun attention backend metadata log (kunlun_backend.py)
# --------------------------------------------------------------------------


def log_kunlun_attn_metadata(backend, forward_batch, batch_size: int, metadata) -> None:
    """Log the generic Kunlun backend extend metadata for batches above one."""
    if (
        os.environ.get("DSV4_C4_ATTN_METADATA_LOG") != "1"
        or batch_size <= 1
        or not forward_batch.forward_mode.is_extend()
    ):
        return
    logger.warning(
        "[DSV4_C4_ATTN_METADATA] %s",
        json.dumps(
            {
                "forward_mode": str(forward_batch.forward_mode),
                "batch_size": batch_size,
                "input_ids": metadata_tensor_summary(forward_batch.input_ids),
                "positions": metadata_tensor_summary(forward_batch.positions),
                "req_pool_indices": metadata_tensor_summary(
                    forward_batch.req_pool_indices
                ),
                "seq_lens": metadata_tensor_summary(forward_batch.seq_lens),
                "seq_lens_cpu": metadata_tensor_summary(forward_batch.seq_lens_cpu),
                "extend_seq_lens_cpu": repr(forward_batch.extend_seq_lens_cpu),
                "extend_prefix_lens_cpu": repr(forward_batch.extend_prefix_lens_cpu),
                "out_cache_loc": metadata_tensor_summary(forward_batch.out_cache_loc),
                "page_table": metadata_tensor_summary(metadata.page_table),
                "cu_seqlens_q": metadata_tensor_summary(metadata.cu_seqlens_q),
                "cu_seqlens_k": metadata_tensor_summary(metadata.cu_seqlens_k),
                "cu_seqlens_q_cpu": metadata_tensor_summary(
                    metadata.extra_attn_metadata.cu_seqlens_q_cpu
                ),
                "kv_lod_cpu": metadata_tensor_summary(
                    metadata.extra_attn_metadata.kv_lod_cpu
                ),
                "cum_q_lod_cpu": metadata_tensor_summary(
                    getattr(backend, "cum_q_lod_cpu", None)
                ),
                "cum_kv_lod_cpu": metadata_tensor_summary(
                    getattr(backend, "cum_kv_lod_cpu", None)
                ),
            },
            ensure_ascii=True,
        ),
    )


# --------------------------------------------------------------------------
# probe sites
#
# Production code only names a site and hands over its frame scope; the
# handlers below know which local variables each probe needs.
# --------------------------------------------------------------------------

_SITES = SiteRegistry()
capture = _SITES.capture


def _production_module_file(instance) -> str:
    """Return the file of the production module that owns ``instance``."""
    return getattr(sys.modules[type(instance).__module__], "__file__", "")


@_SITES.site("backend.import")
def _site_backend_import(scope) -> None:
    log_backend_import(scope["__file__"])


@_SITES.site("store_cache.begin")
def _site_store_cache_begin(scope) -> None:
    _SITES.set_context(
        "store_cache",
        store_cache_begin(
            scope["self"],
            scope["layer_id"],
            scope["scheduler_raw_loc"],
            scope["raw_loc"],
            scope["mapping"],
            scope["pack"],
            scope["swa_pool"],
            scope["local_layer_id"],
        ),
    )


@_SITES.site("store_cache.end")
def _site_store_cache_end(scope) -> None:
    del scope
    store_cache_end(_SITES.get_context("store_cache"))


@_SITES.site("attention.pre_normalization")
def _site_attention_pre_normalization(scope) -> None:
    log_pre_normalization(
        scope["self"],
        scope["layer"],
        scope["q"],
        scope["pool"].swa_kv_pool.kv_buffer[0],
        scope["win_indices"],
    )


@_SITES.site("attention.metadata")
def _site_attention_metadata(scope) -> None:
    ifeval_attention_consume(
        scope["self"],
        scope["layer"],
        scope["compress_ratio"],
        scope["forward_batch"],
        scope["core"],
        scope["win_cache"],
        scope["win_indices"],
        scope["extra_cache"],
        scope["extra_indices"],
        scope["q_lod"],
        scope["kv_lens"],
    )
    log_dsv4_attn_metadata(
        _production_module_file(scope["self"]),
        scope["layer"],
        scope["forward_batch"],
        scope["core"],
        scope["q_3d"],
        scope["q_lod_cpu"],
        scope["q_lod"],
        scope["kv_lens_cpu"],
        scope["kv_lens"],
    )


@_SITES.site("attention.operator_begin")
def _site_attention_operator_begin(scope) -> None:
    backend = scope["self"]
    _SITES.set_context(
        "attention_operator",
        operator_begin(
            backend=backend,
            layer=scope["layer"],
            forward_batch=scope["forward_batch"],
            core=scope["core"],
            softmax_scale=backend.softmax_scale,
            page_size=backend.page_size,
            req_to_token=backend.req_to_token,
            effective_ratio=scope["effective_ratio"],
            compressed_topk=scope["compressed_topk"],
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
            attn_sink=scope["attn_sink"],
            q_op=scope["q_op"],
            win_cache_op=scope["win_cache_op"],
            win_indices_op=scope["win_indices_op"],
            extra_cache_op=scope["extra_cache_op"],
            extra_indices_op=scope["extra_indices_op"],
            out_op=scope["out_op"],
            max_logits_op=scope["max_logits_op"],
            lse_op=scope["lse_op"],
            q_lod_cpu_op=scope["q_lod_cpu_op"],
            q_lod_op=scope["q_lod_op"],
            kv_lens_cpu_op=scope["kv_lens_cpu_op"],
            kv_lens_op=scope["kv_lens_op"],
            attn_sink_op=scope["attn_sink_op"],
        ),
    )


@_SITES.site("attention.operator_end")
def _site_attention_operator_end(scope) -> None:
    operator_end(_SITES.get_context("attention_operator"), scope["out"])


@_SITES.site("c4_logits.begin")
def _site_c4_logits_begin(scope) -> None:
    _SITES.set_context(
        "c4_logits",
        c4_logits_begin(
            scope["backend"],
            scope["c4_indexer"],
            scope["forward_batch"],
            scope["q_fp8"],
        ),
    )


@_SITES.site("c4_logits.prefill")
def _site_c4_logits_prefill(scope) -> None:
    c4_prefill_logits_dump(
        _SITES.get_context("c4_logits"),
        scope["logits"],
        scope["q"].shape[0],
        scope["max_seq_len"],
    )


@_SITES.site("c4_logits.paged_inputs")
def _site_c4_logits_paged_inputs(scope) -> None:
    c4_paged_logits_inputs(
        _SITES.get_context("c4_logits"),
        forward_batch=scope["forward_batch"],
        is_target_verify=scope["is_target_verify"],
        q=scope["q"],
        weights=scope["weights"],
        seq_lens=scope["seq_lens"],
        page_table=scope["page_table"],
        block_table=scope["block_table"],
        k_cache=scope["k_cache"],
        k_scale=scope["k_scale"],
        qlod_cpu=scope["qlod_cpu"],
        qlod_xpu=scope["qlod_xpu"],
        context_lens_cpu=scope["context_lens_cpu"],
        context_lens_xpu=scope["context_lens_xpu"],
        logits=scope["logits"],
        batch_size=scope["batch_size"],
        block_size=scope["block_size"],
        head_dim=scope["head_dim"],
        max_seq_len=scope["max_seq_len"],
    )


@_SITES.site("c4_logits.paged_output")
def _site_c4_logits_paged_output(scope) -> None:
    c4_paged_logits_output(_SITES.get_context("c4_logits"), scope["logits"])


@_SITES.site("multistep.replay")
def _site_multistep_replay(scope) -> None:
    multistep_replay_probe(scope["self"], scope["forward_batch"], scope["in_capture"])


@_SITES.site("kunlun_backend.metadata")
def _site_kunlun_backend_metadata(scope) -> None:
    log_kunlun_attn_metadata(
        scope["self"],
        scope["forward_batch"],
        scope["batch_size"],
        scope["metadata"],
    )
