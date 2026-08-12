"""Kunlun ragged Draft Extend metadata hooks."""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


def _ragged_extend_lengths(forward_batch, num_queries: int):
    if not forward_batch.forward_mode.is_draft_extend_v2():
        return None

    lengths = forward_batch.extend_seq_lens_cpu
    batch_size = forward_batch.batch_size
    if lengths is None or len(lengths) != batch_size or batch_size <= 0:
        return None

    lengths = [int(length) for length in lengths]
    total = sum(lengths)
    if total > num_queries:
        raise ValueError(
            "Kunlun ragged Draft Extend query count mismatch: "
            f"lengths={lengths}, num_queries={num_queries}"
        )
    if total == num_queries and num_queries % batch_size == 0:
        uniform_length = num_queries // batch_size
        if all(length == uniform_length for length in lengths):
            return None
    return lengths


@plugin_hook(
    "sglang_kunlun.hooks.layers.attention.kunlun_deepseek_v4_backend."
    "KunlunDeepseekV4AttnBackend._build_forward_metadata",
    type=HookType.AROUND,
)
def build_ragged_draft_extend_metadata_kunlun(
    original_fn,
    self,
    forward_batch,
    *,
    max_seq_len_override=None,
    use_prefill_cuda_graph: bool = False,
):
    """Build compact per-token metadata for eager ragged Draft Extend."""
    if not getattr(forward_batch, "_kunlun_ragged_draft_extend", False):
        return original_fn(
            self,
            forward_batch,
            max_seq_len_override=max_seq_len_override,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    from sglang.srt.layers.attention import deepseek_v4_backend as upstream

    logical_forward_mode = upstream._get_logical_forward_mode(forward_batch)
    if not logical_forward_mode.is_draft_extend_v2():
        return original_fn(
            self,
            forward_batch,
            max_seq_len_override=max_seq_len_override,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )
    if use_prefill_cuda_graph:
        raise ValueError("ragged Draft Extend metadata cannot use a fixed-width graph")

    req_pool_indices = forward_batch.req_pool_indices
    seq_lens = forward_batch.seq_lens.to(torch.int32)
    seq_lens_cpu = forward_batch.seq_lens_cpu
    extend_seq_lens = forward_batch.extend_seq_lens
    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
    assert (
        seq_lens_cpu is not None
        and extend_seq_lens is not None
        and extend_seq_lens_cpu is not None
    )
    assert self.req_to_token_pool.req_to_token is self.req_to_token
    assert self.swa_page_size % upstream.SWA_WINDOW == 0 and self.page_size % 128 == 0

    if max_seq_len_override is None:
        max_seq_len_override = getattr(forward_batch, "max_seq_len_override", None)
    max_seq_len = (
        max_seq_len_override
        if max_seq_len_override is not None
        else int(seq_lens_cpu.max().item())
    )
    verify_bs = upstream._get_target_verify_bs(forward_batch)
    self.online_c128_mtp.prepare_forward(
        logical_forward_mode,
        req_pool_indices,
        seq_lens,
        verify_bs=verify_bs,
    )
    return self.init_forward_metadata_prefill(
        max_seq_len=max_seq_len,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu.tolist(),
        out_cache_loc=forward_batch.out_cache_loc,
        num_tokens=sum(extend_seq_lens_cpu),
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_start_loc=forward_batch.extend_start_loc,
        need_compress=False,
        use_prefill_cuda_graph=False,
    )


@plugin_hook(
    "sglang_kunlun.hooks.layers.attention.kunlun_deepseek_v4_backend."
    "KunlunDeepseekV4AttnBackend._make_lod",
    type=HookType.AROUND,
)
def make_ragged_draft_extend_lod_kunlun(
    original_fn, self, forward_batch, num_queries: int, device: torch.device
):
    """Build LoD metadata for non-uniform Draft Extend batches."""
    lengths = _ragged_extend_lengths(forward_batch, num_queries)
    if lengths is None:
        return original_fn(self, forward_batch, num_queries, device)

    batch_size = len(lengths)
    q_lod_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    if batch_size:
        torch.cumsum(
            torch.tensor(lengths, dtype=torch.int32),
            dim=0,
            out=q_lod_cpu[1:],
        )
    kv_lens = forward_batch.seq_lens[:batch_size].to(torch.int32)
    kv_lens_cpu = (
        forward_batch.seq_lens_cpu[:batch_size].to(torch.int32)
        if forward_batch.seq_lens_cpu is not None
        else kv_lens.to("cpu", non_blocking=False)
    )
    # CP alignment pads the token dimension past the accepted draft tokens; give
    # every padding row its own item with a dummy KV length.
    pad_rows = num_queries - int(q_lod_cpu[-1].item())
    if pad_rows:
        q_lod_cpu = torch.cat(
            [
                q_lod_cpu,
                q_lod_cpu[-1] + torch.arange(1, pad_rows + 1, dtype=torch.int32),
            ]
        )
        kv_lens_cpu = torch.cat(
            [kv_lens_cpu, torch.ones(pad_rows, dtype=torch.int32)]
        )
        kv_lens = torch.cat(
            [
                kv_lens,
                torch.ones(pad_rows, dtype=torch.int32, device=kv_lens.device),
            ]
        )
    q_lod = q_lod_cpu.to(device, non_blocking=False)
    return q_lod_cpu, q_lod, kv_lens_cpu, kv_lens
