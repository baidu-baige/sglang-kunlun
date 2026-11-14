"""Time the sorted-layout MoE pipeline (0.5.8 production) at the decode shape.

0.5.8 runs `moe_pre_sorted -> quant2d -> moe_fc_v3 -> silu_and_mul -> moe_fc_v3`
(`compressed_tensors_moe.py:2909/2940/2966`), while our plugin runs two
`m_grouped_gemm_I8_I4_bf16_nt_masked_v3` calls. On the traces those are
`moe_bf16_rn_km_nmk` (13.0 ms/step) and `..._kn_nmk` (17.8 ms/step) at identical
call counts (156/step), so this measures both at one fixed decode shape.

The sorted arguments must come from `moe_pre_sorted` -- hand-built
`moe_index`/`lod` are rejected by the kernel (measured: block errors 385/302).

Run: python3 tests/moe/test_moe_fc_v3_pipeline.py
"""

import time

import torch
import torch_xmlir  # noqa: F401
import kunlun_ops

BS, TOP_K, E = 6, 8, 256
HIDDEN, N_UP, N_DOWN = 6144, 4096, 2048
ALIGN, BLOCKS = 128, 12


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000.0 / iters


def build_args():
    x = torch.randn(BS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    topk_index = torch.randint(0, E, (BS, TOP_K), dtype=torch.int32, device="cuda")
    block_statistic = torch.zeros(BLOCKS, E, dtype=torch.int32, device="cuda")
    moe_expand = torch.zeros(BS * TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda")
    moe_index = torch.zeros(BS * TOP_K, dtype=torch.int32, device="cuda")
    expert_m = torch.zeros(E, dtype=torch.int32, device="cuda")
    lod = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    kunlun_ops.moe_pre_sorted(
        x=x, topk_index=topk_index, block_statistic=block_statistic,
        moe_expand=moe_expand, moe_index=moe_index,
        expert_m=expert_m, sorted_tokens_num_lod=lod,
    )
    return moe_expand, moe_index, lod, expert_m


def quant(t):
    q = torch.empty(t.shape, dtype=torch.int8, device=t.device)
    s = torch.empty((t.shape[0], 1), dtype=torch.float32, device=t.device)
    kunlun_ops.quant2d(t, q, s, force_sdnn=True)
    return q, s


def sorted_pipeline():
    moe_expand, idx, lod, _ = build_args()
    x_q, x_s = quant(moe_expand)
    w_up = torch.randint(-128, 127, (E, N_UP, HIDDEN // 2), dtype=torch.int8, device="cuda")
    # WNA16-style per-group scale: [E, N, K/group] (group 128), see
    # compressed_tensors_moe.py:295-305.
    w_up_s = torch.rand(E, N_UP, HIDDEN // ALIGN, dtype=torch.float32, device="cuda")
    gate_up = torch.zeros(BS * TOP_K, N_UP, dtype=torch.bfloat16, device="cuda")
    hidden = torch.zeros(BS * TOP_K, N_DOWN, dtype=torch.bfloat16, device="cuda")
    x_q2, x_s2 = quant(hidden)
    w_dn = torch.randint(-128, 127, (E, HIDDEN, N_DOWN // 2), dtype=torch.int8, device="cuda")
    w_dn_s = torch.rand(E, HIDDEN, N_DOWN // ALIGN, dtype=torch.float32, device="cuda")
    down = torch.zeros(BS * TOP_K, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def up():
        kunlun_ops.moe_fc_v3(
            x=x_q, x_perchannel_max=x_s, weight=w_up, w_perchannel_max=w_up_s,
            sorted_tokens_num_lod=lod, sorted_tokens_idx=idx, moe_topk=TOP_K,
            y=gate_up, use_pack_int4=True, sort_mode=True)

    def dn():
        kunlun_ops.moe_fc_v3(
            x=x_q2, x_perchannel_max=x_s2, weight=w_dn, w_perchannel_max=w_dn_s,
            sorted_tokens_num_lod=lod, sorted_tokens_idx=idx, moe_topk=TOP_K,
            y=down, use_pack_int4=True, sort_mode=True)

    return up, dn


if __name__ == "__main__":
    try:
        up, dn = sorted_pipeline()
        print(f"moe_fc_v3  up   n={N_UP:5d}  {bench(up):7.3f} ms")
        print(f"moe_fc_v3  down n={N_DOWN:5d}  {bench(dn):7.3f} ms")
    except Exception as exc:
        print("moe_fc_v3 pipeline FAIL", str(exc)[:120])
