""" model executor hooks """
from . import model_runner_kv_cache_mixin  # noqa: F401

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.eplb.expert_location._compute_logical_to_all_physical_map",
    type=HookType.AROUND,
)
def compute_logical_to_all_physical_map_kunlun(
    original_fn, physical_to_logical_map: torch.Tensor, **kwargs
):
    """把 expert-location 映射表的逐元素循环放到 host 上跑。

    上游 `_compute_logical_to_all_physical_map` 用双层 Python 循环遍历整张
    `[num_layers, num_physical_experts]` 表，每个元素一次 `.item()`（注释写着
    "rarely called, so we use for loops for maximum clarity"）。在 N 卡上只是慢，
    在昆仑上每次 `.item()` 都是一次设备同步——DeepSeek-V4.1 是 40 层 × 384 个
    physical expert = 15360 次，启动会明显卡顿。

    上游逻辑一行不改，只是先把张量搬到 host，结果再搬回原 device（上游是按
    `device=physical_to_logical_map.device` 构造的）。
    """
    device = physical_to_logical_map.device
    out = original_fn(physical_to_logical_map=physical_to_logical_map.cpu(), **kwargs)
    if isinstance(out, torch.Tensor) and out.device != device:
        out = out.to(device)
    return out
