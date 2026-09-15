"""Hooks for ``sglang.srt.layers.attention.minimax_sparse_backend``。

上游的 ``MiniMaxSparseAttnBackend`` 没有考虑过 TARGET_VERIFY：``prepare_for_v2_verify``
不填 ``extend_*``，``ForwardBatch.init_new`` 对该 mode 也只走 decode 分支，所以
``extend_seq_lens / extend_seq_lens_cpu / extend_prefix_lens`` 全是 ``None``，而 verify
是按 ``is_extend()`` 路由到 ``forward_extend`` 的。这里补齐 verify 需要的 extend 语义与
序列长度上界。
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.attention.minimax_sparse_backend."
    "MiniMaxSparseAttnBackend.init_forward_metadata_out_graph",
    type=HookType.AROUND,
)
def minimax_sparse_init_forward_metadata_out_graph_kunlun(
    original_fn, self, forward_batch, in_capture: bool = False
):
    """修正 verify 的 q/k 长度上界。

    上游对 verify 取 ``_max_seqlen_q=1``（``extend_seq_lens_cpu`` 为 None）、
    ``_max_seqlen_k=seq_lens_cpu.max()``（只有 prefix，缺 verify 宽度）。
    """
    result = original_fn(self, forward_batch, in_capture)
    if forward_batch.forward_mode.is_target_verify():
        verify_width = int(forward_batch.spec_info.draft_token_num)
        self._max_seqlen_q = verify_width
        if in_capture:
            # CUDA graph records Python scalar kernel arguments and temporary
            # tensor shapes. The capture batch uses seq_len=1, so adding the
            # verify width would permanently truncate every replay to a tiny KV
            # range. Capture with a stable upper bound; live LOD tensors still
            # restrict each request to its actual context length.
            self._max_seqlen_k = self.max_context_len
        else:
            self._max_seqlen_k += verify_width
    return result


@plugin_hook(
    "sglang.srt.layers.attention.minimax_sparse_backend."
    "MiniMaxSparseAttnBackend.forward_extend",
    type=HookType.AROUND,
)
def minimax_sparse_forward_extend_kunlun(
    original_fn,
    self,
    q,
    k,
    v,
    layer,
    forward_batch,
    save_kv_cache=True,
    **kwargs,
):
    """按「每请求 draft_token_num 个新 token」补全 verify 的 extend 元数据。

    不补的话上游 ``forward_extend`` 会在 ``extend_seq_lens.device`` 上直接抛异常。
    返回前还原，避免影响 attention 之后的其它消费者。
    """
    if not forward_batch.forward_mode.is_target_verify():
        return original_fn(
            self,
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )

    verify_width = int(forward_batch.spec_info.draft_token_num)
    original_metadata = (
        forward_batch.seq_lens,
        forward_batch.extend_seq_lens,
        forward_batch.extend_seq_lens_cpu,
        forward_batch.extend_prefix_lens,
    )
    prefix_lens = forward_batch.seq_lens
    forward_batch.extend_seq_lens = torch.full(
        (forward_batch.batch_size,),
        verify_width,
        dtype=torch.int32,
        device=prefix_lens.device,
    )
    forward_batch.extend_seq_lens_cpu = [verify_width] * forward_batch.batch_size
    forward_batch.extend_prefix_lens = prefix_lens
    forward_batch.seq_lens = prefix_lens + verify_width
    try:
        return original_fn(
            self,
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )
    finally:
        (
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_seq_lens_cpu,
            forward_batch.extend_prefix_lens,
        ) = original_metadata
