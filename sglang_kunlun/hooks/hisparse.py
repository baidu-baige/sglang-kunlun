"""HiSparse server-argument adjustments for Kunlun.

``validate_hisparse`` documents that its dtype/backend checks "only apply to
the DSA hisparse path" because "DSv4 hisparse handles its own dtype/backend
pairing elsewhere" -- but the guard that skips them is
``is_hip and is_v4_hisparse``, so on Kunlun (``is_hip`` is False, the device
presents as CUDA) DeepSeek-V4 is still forced onto bfloat16/fp8_e4m3 and an
fp16 launch is rejected.

Kunlun serves DeepSeek-V4 with fp16 (see
``hooks/layers/attention/dsv4/dtype_restore.py``), the DSv4 C4 half-cache is a
2-byte layout for either dtype, and ``--kv-cache-dtype bfloat16`` collapses
DSPARK acceptance on this build, so honour the requested dtype for DeepSeek-V4.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.arg_groups.hisparse_hook.validate_hisparse_kv_cache_dtype",
    type=HookType.AROUND,
)
def _allow_requested_kv_dtype_for_dsv4_hisparse(original_fn, server_args):
    """Skip the DSA-only KV-dtype check for DeepSeek-V4 hisparse."""
    from sglang.srt.arg_groups.overrides import model_config_of
    from sglang.srt.configs.model_config import is_deepseek_v4

    if is_deepseek_v4(model_config_of(server_args).hf_config):
        return None
    return original_fn(server_args)


@plugin_hook(
    "sglang.srt.managers.hisparse_coordinator.HiSparseCoordinator.__init__",
    type=HookType.AROUND,
)
def _grow_hisparse_row_buffers(original_fn, self, *args, **kwargs):
    """Size the row-indexed swap-in buffers for speculative verify batches.

    The coordinator sizes ``top_k_device_locs_buffer`` / ``raw_indices_buffer``
    (and the shared-index miss plan) by the request-pool slot count, assuming one
    row per request.  A DSPARK verify forward runs ``bs * num_draft_tokens``
    query rows, each selecting its own top-k, so the row-indexed buffers must
    cover that many rows.
    """
    original_fn(self, *args, **kwargs)

    import torch

    from sglang.srt.runtime_context import get_spec

    draft = int(get_spec().speculative_num_draft_tokens or 1)
    slots = self.req_to_token_pool.req_to_token.shape[0]
    rows = slots * max(1, draft)
    if rows <= self.top_k_device_locs_buffer.shape[0]:
        return

    device = self.device
    top_k = self.top_k
    self.top_k_device_locs_buffer = torch.full(
        (rows, top_k), -1, dtype=torch.int32, device=device
    )
    self.raw_indices_buffer = torch.full(
        (rows, top_k), -1, dtype=torch.int32, device=device
    )
    if getattr(self, "_miss_src", None) is not None:
        self._miss_src = torch.zeros((rows, top_k), dtype=torch.int64, device=device)
        self._miss_dst = torch.zeros((rows, top_k), dtype=torch.int32, device=device)
        self._miss_count = torch.zeros((rows,), dtype=torch.int32, device=device)
