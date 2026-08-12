"""Focused runtime probe for ragged DRAFT_EXTEND_V2 attention consumption."""

from __future__ import annotations

import os

import torch


if os.environ.get("DSV4_RAGGED_DRAFT_PROBE") == "1":
    from mtp_alignment_common import emit, generation_requests
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def _argument(args, kwargs, index, name):
        return args[index] if len(args) > index else kwargs.get(name)

    def _focused_request_row():
        focus_hash = os.environ.get("DSV4_ALIGNMENT_FOCUS_PROMPT_HASH")
        focus_occurrence = int(
            os.environ.get("DSV4_ALIGNMENT_FOCUS_REQUEST_OCCURRENCE", "0")
        )
        matches = [
            index
            for index, request in enumerate(generation_requests())
            if request.get("prompt_hash") == focus_hash
        ]
        return matches[focus_occurrence] if focus_occurrence < len(matches) else None

    def _extend_lengths(forward_batch):
        lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if lengths is None:
            lengths = getattr(forward_batch, "extend_lens", None)
        if lengths is None:
            spec_info = getattr(forward_batch, "spec_info", None)
            lengths = getattr(spec_info, "extend_seq_lens_cpu", None)
        if lengths is None:
            spec_info = getattr(forward_batch, "spec_info", None)
            lengths = getattr(spec_info, "extend_seq_lens_tensor", None)
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.detach().cpu().tolist()
        return [int(value) for value in lengths] if lengths is not None else []

    def _logical_forward_mode(forward_batch):
        actual_mode = getattr(forward_batch, "actual_forward_mode", None)
        return actual_mode if actual_mode is not None else getattr(
            forward_batch, "forward_mode", None
        )

    def _is_draft_extend_v2(forward_batch):
        mode = _logical_forward_mode(forward_batch)
        checker = getattr(mode, "is_draft_extend_v2", None)
        if callable(checker):
            return bool(checker())
        return str(mode) == "ForwardMode.DRAFT_EXTEND_V2"

    def _forward_hook(original_fn, self, *args, **kwargs):
        result = original_fn(self, *args, **kwargs)
        forward_batch = _argument(args, kwargs, 4, "forward_batch")
        layer = _argument(args, kwargs, 3, "layer")
        if forward_batch is None or not _is_draft_extend_v2(forward_batch):
            return result

        layer_id = int(getattr(layer, "layer_id", -1))
        max_layer = int(os.environ.get("DSV4_ALIGNMENT_FOCUS_MAX_LAYER", "0"))
        request_row = _focused_request_row()
        lengths = _extend_lengths(forward_batch)
        if (
            request_row is None
            or layer_id < 0
            or layer_id > max_layer
            or request_row >= len(lengths)
        ):
            return result

        query_row = sum(lengths[: request_row + 1]) - 1
        q = _argument(args, kwargs, 0, "q")
        k = _argument(args, kwargs, 1, "k")
        if not isinstance(q, torch.Tensor) or not 0 <= query_row < q.shape[0]:
            return result

        core = self.forward_metadata.core_attn_metadata
        pool = self.token_to_kv_pool
        win_indices = self._match_queries(core.swa_page_indices, q.shape[0], 0)
        win_indices = win_indices[:, : pool.swa_window_size]
        win_lengths = self._match_queries(core.swa_topk_lengths, q.shape[0], 0)
        win_length = min(int(win_lengths[query_row].item()), win_indices.shape[1])
        selected = win_indices[query_row, :win_length].long()
        valid = selected >= 0
        safe_selected = selected.clamp(min=0)
        cache_dim = pool.swa_kv_pool.kv_cache_total_dim
        win_cache = pool.get_swa_key_buffer_radix(layer_id).reshape(-1, cache_dim)
        cache_rows = win_cache.index_select(0, safe_selected)
        cache_rows = torch.where(valid.unsqueeze(1), cache_rows, torch.zeros_like(cache_rows))

        payload = {
            "request_row": torch.tensor(request_row, dtype=torch.int64),
            "query_row": torch.tensor(query_row, dtype=torch.int64),
            "q": q[query_row],
            "k": k[query_row] if isinstance(k, torch.Tensor) else None,
            "attention_output": result[query_row],
            "extend_lengths": torch.tensor(lengths, dtype=torch.int64),
            "swa_indices": selected,
            "swa_valid": valid,
            "swa_cache_rows": cache_rows,
        }
        for name in (
            "positions_casual",
            "seq_lens_casual",
            "swa_topk_lengths",
            "page_table",
        ):
            value = getattr(core, name, None)
            if isinstance(value, torch.Tensor) and query_row < value.shape[0]:
                payload[name] = value[query_row : query_row + 1]
        emit(
            f"draft_extend_focus.layer{layer_id}.attention_backend.post",
            batch=forward_batch,
            result=payload,
        )
        return result

    HookRegistry.register(
        "sglang_kunlun.hooks.layers.attention.kunlun_deepseek_v4_backend."
        "KunlunDeepseekV4AttnBackend.forward",
        _forward_hook,
        HookType.AROUND,
    )
