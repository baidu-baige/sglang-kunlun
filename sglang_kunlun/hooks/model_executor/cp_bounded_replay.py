"""Bridge CP tail execution, routing and logits for decoder bounded replay."""
from __future__ import annotations

import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _active_cp_tail(forward_batch):
    """Return the late-layer tail metadata when CP bounded replay is active."""
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if (
        os.environ.get("DSV4_CP_BOUNDED_REPLAY", "1") != "1"
        or not forward_batch.forward_mode.is_extend_without_speculative()
        or not is_cp_active(forward_batch)
    ):
        return None
    backend = get_attn_backend()
    metadata = getattr(backend, "tail_forward_metadata", None)
    tail = getattr(metadata, "late_layer_tail", None)
    return tail if getattr(tail, "cp_metadata", None) is not None else None


@plugin_hook(
    "sglang.srt.layers.cp.utils.cp_gather_after_forward", type=HookType.AROUND
)
def cp_gather_with_tail_metadata(original_fn, x, forward_batch, stream=None):
    """Gather hidden and DSPARK auxiliary rows with the tail's CP layout."""
    tail = _active_cp_tail(forward_batch)
    if tail is None:
        return original_fn(x, forward_batch, stream)
    saved = forward_batch.attn_cp_metadata
    forward_batch.attn_cp_metadata = tail.cp_metadata
    try:
        return original_fn(x, forward_batch, stream)
    finally:
        forward_batch.attn_cp_metadata = saved


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend.enter_late_layer_tail",
    type=HookType.AROUND,
)
def enter_tail_reset_ids(original_fn, self, forward_batch):
    """Clear the cached tail global IDs each time a tail execution begins."""
    self._kunlun_tail_global_ids = None
    return original_fn(self, forward_batch)


@plugin_hook(
    "sglang.srt.models.deepseek_common.moe_transport.MoeRowTransport.__init__",
    type=HookType.AROUND,
)
def tail_moe_global_ids(
    original_fn, self, moe, hidden_states, forward_batch,
    input_ids, input_ids_global, **kwargs,
):
    """Match global routing IDs to the same rank-order gather as the MoE rows.

    0519 skipped tails for image batches and dropped VL bias on text tails.
    Here gathering the actual local IDs preserves both text and image routing,
    so bounded replay and CP stay enabled together for multimodal requests.

    0.5.23 re-ported: `DeepseekV4DecoderLayer._run_moe_ffn_dp_sync` is gone and
    the row transport owns this step. `MoeRowTransport.__init__` is the same
    point in the flow -- it is where `input_ids_global` is both consumed (the
    a2a-scatter branch splits it) and stored -- so the substitution still has to
    happen before the original runs, exactly as it did on the old target.
    """
    from sglang.srt.layers.cp.interleave import attn_cp_interleave_gather
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend

    tail = _active_cp_tail(forward_batch)
    backend = get_attn_backend()
    if (
        tail is not None and input_ids is not None
        and backend.forward_metadata is backend.tail_forward_metadata
        and get_moe_a2a_backend().is_none()
    ):
        cached = getattr(backend, "_kunlun_tail_global_ids", None)
        if cached is None or cached[0] is not input_ids:
            global_ids = attn_cp_interleave_gather(input_ids[:, None]).squeeze(-1)
            backend._kunlun_tail_global_ids = (input_ids, global_ids)
        else:
            global_ids = cached[1]
        input_ids_global = global_ids
    return original_fn(
        self, moe, hidden_states, forward_batch, input_ids, input_ids_global,
        **kwargs,
    )


def _global_tail_token_indices(forward_batch):
    """Compute the global tail token rows and per-sequence lengths."""
    from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import late_layer_tail_layout
    from sglang.srt.layers.attention.deepseek_v4_backend import SWA_WINDOW

    rows, lengths, _ = late_layer_tail_layout(
        extend_lens_cpu=forward_batch.extend_seq_lens_cpu,
        seq_lens_cpu=forward_batch.seq_lens_cpu.tolist(),
        tail_len=SWA_WINDOW, device=forward_batch.out_cache_loc.device,
    )
    return rows, lengths


@plugin_hook(
    "sglang.srt.layers.logits_processor.LogitsProcessor.forward", type=HookType.AROUND
)
def logits_with_tail_rows(
    original_fn, self, input_ids, hidden_states, lm_head, logits_metadata,
    aux_hidden_states=None, **kwargs,
):
    """Compute logits on global tail rows with the tail's sequence layout."""
    from sglang.srt.layers.logits_processor import LogitsMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    tail = (
        _active_cp_tail(logits_metadata)
        if isinstance(logits_metadata, ForwardBatch) else None
    )
    if tail is None:
        return original_fn(
            self, input_ids, hidden_states, lm_head, logits_metadata,
            aux_hidden_states, **kwargs,
        )
    rows, lengths = _global_tail_token_indices(logits_metadata)
    if list(lengths) != list(tail.extend_seq_lens_cpu):
        raise ValueError("CP bounded replay: global and tail lengths disagree")
    if hidden_states.shape[0] != rows.shape[0]:
        raise ValueError("CP bounded replay: gathered hidden rows do not match tail")
    metadata = LogitsMetadata.from_forward_batch(logits_metadata)
    metadata.extend_seq_lens = tail.extend_seq_lens
    metadata.extend_seq_lens_cpu = tail.extend_seq_lens_cpu
    metadata.extend_logprob_start_lens_cpu = tail.extend_seq_lens_cpu
    output = original_fn(
        self, input_ids[rows], hidden_states, lm_head, metadata,
        aux_hidden_states, **kwargs,
    )
    output.hidden_states_token_indices = rows
    return output
