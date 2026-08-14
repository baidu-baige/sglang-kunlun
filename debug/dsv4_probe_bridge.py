"""Bridge between production call sites and the DSV4 precision probes.

Production files keep one call per probe point; the implementation lives here.
Every function degrades to a no-op when the corresponding debug callback is not
installed, so importing this module never changes numerical behaviour.

The invocation contract is unchanged from before the extraction:

* stage probes read ``_dsv4_tensor_dump_stage_callback`` off the module
* W8A8 linear probes read ``_dsv4_tensor_dump_callback`` off the layer
* MoE probes read ``_dsv4_moe_tensor_dump_callback`` off the layer and are
  filtered by ``DSV4_W8A8_MOE_DUMP_ROWS`` / ``DSV4_W8A8_MOE_DUMP_STAGES``
* static W8A8 parameter dumps use ``DSV4_STATIC_W8A8_DUMP_PATH``,
  ``DSV4_STATIC_W8A8_DUMP_MODULE`` and ``DSV4_STATIC_W8A8_DUMP_RANK``
* compressed-attention probes read
  ``_dsv4_tensor_dump_compressed_attention_callback`` off the backend
"""

from __future__ import annotations

import os

import torch


def dump_probe(owner, name: str, value) -> None:
    """Stage probe used by ``MQALayer._dsv4_dump_probe``."""
    callback = getattr(owner, "_dsv4_tensor_dump_stage_callback", None)
    if callback is not None and isinstance(value, torch.Tensor):
        callback(name, value)


def dump_linear_stage(layer, name: str, value) -> None:
    """Per-stage probe for the Kunlun W8A8 linear apply path."""
    callback = getattr(layer, "_dsv4_tensor_dump_callback", None)
    if callback is not None:
        callback(name, value)


def dump_selected_linear_parameters(layer: torch.nn.Module) -> None:
    """Dump parameters for the configured static W8A8 module and rank."""
    output_path = os.getenv("DSV4_STATIC_W8A8_DUMP_PATH")
    selected_module = os.getenv("DSV4_STATIC_W8A8_DUMP_MODULE")
    module_name = getattr(layer, "_dsv4_module_name", None)
    if not output_path or not selected_module or module_name != selected_module:
        return

    from sglang.srt.distributed import get_tensor_model_parallel_rank

    tp_rank = get_tensor_model_parallel_rank()
    selected_rank = int(os.getenv("DSV4_STATIC_W8A8_DUMP_RANK", "0"))
    if tp_rank != selected_rank:
        return

    payload = {"module_name": module_name, "tp_rank": tp_rank}
    for name in ("weight", "weight_scale", "weight_scale_inv"):
        value = getattr(layer, name, None)
        if isinstance(value, torch.Tensor):
            payload[name] = value.detach().cpu()

    temporary_path = f"{output_path}.{os.getpid()}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    print(
        f"[DSV4 static W8A8 dump] module={module_name} rank={tp_rank} "
        f"path={output_path}"
    )


def dump_selected_moe_rows(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> None:
    """Dump configured token rows from a MoE tensor callback."""
    callback = getattr(layer, "_dsv4_moe_tensor_dump_callback", None)
    rows_text = os.getenv("DSV4_W8A8_MOE_DUMP_ROWS")
    stages_text = os.getenv("DSV4_W8A8_MOE_DUMP_STAGES")
    if callback is None or not rows_text or not isinstance(value, torch.Tensor):
        return
    if stages_text and name not in stages_text.split():
        return

    if rows_text.strip().lower() == "all":
        rows = list(range(num_tokens))
    else:
        rows = [int(item) for item in rows_text.split(",") if item]
    rows = [row for row in rows if 0 <= row < num_tokens]
    if not rows or value.ndim == 0:
        return

    if value.shape[0] == num_tokens:
        indices = torch.tensor(rows, dtype=torch.long, device=value.device)
    elif value.shape[0] == num_tokens * top_k:
        indices = torch.tensor(
            [row * top_k + offset for row in rows for offset in range(top_k)],
            dtype=torch.long,
            device=value.device,
        )
    else:
        return
    callback(name, value.index_select(0, indices))


def dump_selected_moe_tensor(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    """Dump a selected MoE tensor when its stage is enabled."""
    callback = getattr(layer, "_dsv4_moe_tensor_dump_callback", None)
    stages_text = os.getenv("DSV4_W8A8_MOE_DUMP_STAGES")
    if callback is None or not isinstance(value, torch.Tensor):
        return
    if stages_text and name not in stages_text.split():
        return
    callback(name, value)


def _local_head_slice(head_total: int) -> slice:
    """TP-local head range, so a rank compares only the heads it consumes."""
    distributed = torch.distributed.is_initialized()
    tp_rank = torch.distributed.get_rank() if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1
    local_head_count = head_total // max(world_size, 1)
    return slice(tp_rank * local_head_count, (tp_rank + 1) * local_head_count)


def dump_compressed_attention_inputs(
    backend,
    *,
    q_op,
    win_cache_op,
    win_indices_op,
    extra_cache_op,
    extra_indices_op,
    q_lod_cpu_op,
    q_lod_op,
    kv_lens_cpu_op,
    kv_lens_op,
    attn_sink_op,
    softmax_scale,
    causal,
    effective_ratio,
    compressed_topk,
) -> None:
    """Capture everything ``compressed_attention`` actually consumes."""
    operator_probe = getattr(
        backend, "_dsv4_tensor_dump_compressed_attention_callback", None
    )
    if operator_probe is None:
        return

    local_head_slice = _local_head_slice(q_op.shape[1])

    def capture_indexed_cache(prefix, cache, indices):
        flat_indices = indices.reshape(-1).to(torch.long)
        valid_indices = flat_indices[
            (flat_indices >= 0) & (flat_indices < cache.shape[0])
        ]
        unique_indices = torch.unique(valid_indices)
        operator_probe(f"input.{prefix}_indices", indices)
        operator_probe(f"input.{prefix}_unique_indices", unique_indices)
        operator_probe(
            f"input.{prefix}_unique_cache_rows",
            cache.index_select(0, unique_indices),
        )

    operator_probe("input.q_local", q_op[:, local_head_slice, :])
    capture_indexed_cache("win", win_cache_op, win_indices_op)
    capture_indexed_cache("extra", extra_cache_op, extra_indices_op)
    operator_probe("input.q_lod_cpu", q_lod_cpu_op)
    operator_probe("input.q_lod", q_lod_op)
    operator_probe("input.kv_lens_cpu", kv_lens_cpu_op)
    operator_probe("input.kv_lens", kv_lens_op)
    if attn_sink_op is not None:
        operator_probe("input.attn_sink_local", attn_sink_op[local_head_slice])
    operator_probe(
        "input.softmax_scale", torch.tensor(softmax_scale, dtype=torch.float64)
    )
    operator_probe("input.causal", torch.tensor(causal, dtype=torch.bool))
    operator_probe(
        "input.max_window_size",
        torch.tensor(win_indices_op.shape[1], dtype=torch.int64),
    )
    operator_probe(
        "input.compress_ratio", torch.tensor(effective_ratio, dtype=torch.int64)
    )
    operator_probe(
        "input.compressed_topk", torch.tensor(compressed_topk, dtype=torch.int64)
    )


def dump_compressed_attention_outputs(
    backend,
    *,
    out_op,
    max_logits_op,
    lse_op,
) -> None:
    """Capture outputs returned by the compressed-attention operator."""
    operator_probe = getattr(
        backend, "_dsv4_tensor_dump_compressed_attention_callback", None
    )
    if operator_probe is None:
        return

    local_head_slice = _local_head_slice(out_op.shape[1])
    operator_probe("output.out_local", out_op[:, local_head_slice, :])
    operator_probe("output.max_logits_local", max_logits_op[:, local_head_slice])
    operator_probe("output.lse_local", lse_op[:, local_head_slice])
