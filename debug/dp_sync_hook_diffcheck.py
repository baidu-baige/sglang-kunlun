"""Differential check of the forked ``all_gather`` against upstream's own.

Instantiates the real ``MLPSyncBatchInfo`` dataclass, stubs only the two seams
the method reaches outside itself (``exchange_sync_rows`` and the parallel
context's active-rank mask), runs upstream's method and the plugin's hook on
identical inputs, and compares all eleven fields plus their python types.

Covers both collective branches: the gather path and the ``use_all_reduce``
path (on a single-rank gloo group, where all_reduce is identity and the
masked-fallback rows are exercised).

Not part of the shipped test suite: it imports sglang, which the unit tests
deliberately avoid. Run it in the serving container:

    PYTHONPATH=<plugin>:<sglang>/python python debug/dp_sync_hook_diffcheck.py
"""

import os
import sys

import torch

import transformers

if not hasattr(transformers.PretrainedConfig, "__class_validators__"):
    # This box's transformers is newer than sglang's import-time patch expects;
    # unrelated to the hook, and only needed to import sglang at all here.
    transformers.PretrainedConfig.__class_validators__ = []

from sglang.srt.managers.scheduler_components import dp_attn as up

import sglang_kunlun.hooks.managers.scheduler_components.dp_attn as hook

FIELDS = (
    "global_num_tokens",
    "global_num_tokens_for_logprob",
    "can_run_decode_cuda_graph",
    "is_extend_in_batch",
    "can_run_prefill_cuda_graph",
    "prefill_cuda_graph_max_prefix_len",
    "can_run_draft_cuda_graph",
    "any_health_check",
    "global_attention_pairs",
)
class _FakeTpGroup:
    def __init__(self, n):
        self.active_ranks_cpu = torch.ones(n, dtype=torch.int32)
        self.active_ranks = self.active_ranks_cpu


class _FakeParallel:
    def __init__(self, n):
        self.tp_group = _FakeTpGroup(n)


def make_info(dp_size, tp_size, cp_size, seed):
    g = torch.Generator().manual_seed(seed)

    def rnd(hi):
        return int(torch.randint(0, hi, (1,), generator=g).item())

    modes = [int(m.value) for m in up.ForwardMode]
    return up.MLPSyncBatchInfo(
        num_dp_ranks=dp_size,
        tp_size=tp_size,
        cp_size=cp_size,
        num_tokens=rnd(4096),
        num_tokens_for_logprob=rnd(4096),
        can_run_decode_cuda_graph=bool(rnd(2)),
        can_run_draft_cuda_graph=bool(rnd(2)),
        can_run_prefill_cuda_graph=bool(rnd(2)),
        is_extend_in_batch=bool(rnd(2)),
        local_can_run_tbo=bool(rnd(2)),
        local_forward_mode=modes[rnd(len(modes))],
        prefill_cuda_graph_max_prefix_len=rnd(8192),
        is_health_check=bool(rnd(2)),
        attention_pairs=rnd(1 << 20),
    )


def gather_stub(seed):
    """Fill every rank's row with a distinct deterministic pattern."""

    def exchange_sync_rows(global_flat, local_row, group):
        g = torch.Generator().manual_seed(seed)
        width = local_row.numel()
        rows = global_flat.view(-1, width)
        modes = [int(m.value) for m in up.ForwardMode]
        for i in range(rows.shape[0]):
            rows[i] = torch.randint(
                0, 4096, (width,), dtype=local_row.dtype, generator=g
            )
            # Column 5 is a ForwardMode; DPCooperationInfo.create looks it up.
            rows[i, 5] = modes[
                int(torch.randint(0, len(modes), (1,), generator=g).item())
            ]
        rows[0] = local_row

    return exchange_sync_rows
def snapshot(info):
    out = {f: getattr(info, f) for f in FIELDS}
    out["tp0_info_cpu"] = info.tp0_info_cpu
    out["dp_cooperation_info"] = info.dp_cooperation_info
    return out


def run_case(dp_size, tp_size, cp_size, seed, use_all_reduce):
    saved_exchange = up.exchange_sync_rows
    saved_parallel = up.get_parallel
    up.exchange_sync_rows = gather_stub(seed)
    up.get_parallel = lambda n=dp_size * tp_size * cp_size: _FakeParallel(n)
    try:
        group = torch.distributed.group.WORLD if use_all_reduce else None
        ref_info = make_info(dp_size, tp_size, cp_size, seed)
        new_info = make_info(dp_size, tp_size, cp_size, seed)
        hook._UPSTREAM_ALL_GATHER(
            ref_info, device="cpu", group=group, use_all_reduce=use_all_reduce
        )
        hook.all_gather_kunlun(
            new_info, device="cpu", group=group, use_all_reduce=use_all_reduce
        )
    finally:
        up.exchange_sync_rows = saved_exchange
        up.get_parallel = saved_parallel

    ref, new = snapshot(ref_info), snapshot(new_info)
    for name in ref:
        a, b = ref[name], new[name]
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor), name
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert torch.equal(a, b), name
            continue
        assert a == b, f"{name}: {a!r} != {b!r}"
        assert type(a) is type(b), f"{name}: {type(a)} != {type(b)}"
        if isinstance(a, list):
            assert all(type(x) is int for x in b), name
    return ref, new
def main():
    print("fast path enabled:", hook._FAST)
    print("fingerprint ok   :", hook._fingerprint(hook._UPSTREAM_ALL_GATHER))
    assert hook._FAST, "fast path disabled; nothing to compare"

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

    cases = 0
    for metrics in (False, True):
        saved_metrics = up._ENABLE_METRICS_DP_ATTENTION
        up._ENABLE_METRICS_DP_ATTENTION = metrics
        try:
            for dp_size, tp_size, cp_size in ((8, 1, 1), (8, 4, 1), (8, 2, 2), (2, 1, 1)):
                for seed in range(5):
                    for use_all_reduce in (False, True):
                        ref, _ = run_case(
                            dp_size, tp_size, cp_size, seed, use_all_reduce
                        )
                        cases += 1
                print(
                    f"  metrics={metrics} dp={dp_size} tp={tp_size} cp={cp_size} ok, "
                    f"tp0_info_cpu={tuple(ref['tp0_info_cpu'].shape)} "
                    f"contiguous={ref['tp0_info_cpu'].is_contiguous()} "
                    f"coop={ref['dp_cooperation_info']}"
                )
        finally:
            up._ENABLE_METRICS_DP_ATTENTION = saved_metrics
    print(f"{cases} cases, every field equal and same type")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())



