""" model executor hooks """
from . import breakable_cuda_graph  # noqa: F401
from . import capture_align  # noqa: F401
from . import model_runner_kv_cache_mixin  # noqa: F401
from . import forward_batch_info  # noqa: F401
from . import cp_mm_embeds  # noqa: F401
from . import cp_reduce_scatter  # noqa: F401
from . import cp_bounded_replay  # noqa: F401

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.eplb.expert_location._compute_logical_to_all_physical_map",
    type=HookType.AROUND,
)
def compute_logical_to_all_physical_map_kunlun(
    original_fn, physical_to_logical_map: torch.Tensor, **kwargs
):
    """Move the expert-location map's per-element loop to the host.

    Upstream `_compute_logical_to_all_physical_map` uses a double Python loop over the whole
    `[num_layers, num_physical_experts]` table, one `.item()` per element (the comment says
    "rarely called, so we use for loops for maximum clarity"). On N cards it is merely slow;
    on Kunlun every `.item()` is a device sync -- DeepSeek-V4.1 has 40 layers x 384 physical
    experts = 15360 of them, so startup stalls noticeably.

    Upstream logic is unchanged line for line; the tensor is first moved to the host and the
    result is moved back to the original device (upstream constructs it with
    `device=physical_to_logical_map.device`).
    """
    device = physical_to_logical_map.device
    out = original_fn(physical_to_logical_map=physical_to_logical_map.cpu(), **kwargs)
    if isinstance(out, torch.Tensor) and out.device != device:
        out = out.to(device)
    return out
