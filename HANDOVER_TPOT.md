# TPOT 对齐与 MegaMoE 迁移 · 交接文档

状态：2026-11-19。0.5.23 对 0.5.8 的 decode TPOT 已在固定 batch 下追平；MegaMoE
已能启动但收益未达预期。下一步是 P/D 分离压测，然后进入重构。

分支：`glm-5.3-kunlun-0.5.23`（本地）→ `refs/for/kunlun-0.5.23`（CR）。
`origin/kunlun-0.5.23` 目前的 tip 是 `32522be`。

---

## 1. 运行时契约

| 项 | 值 |
|---|---|
| 模型 | `/home/models/GLM-5.3-W4A8-MoE-W8A8-Attn-Dense-Shared` |
| 拓扑 | 1P1D，prefill `10.129.128.250:8000`，decode `10.129.128.144:8000`，router `:8080` |
| 并行 | prefill TP8 + attn CP8 / dp1；decode TP8 + attn DP8 / ep8 |
| 量化 | W4A8，`--kv-cache-dtype int8` |
| 投机 | EAGLE，3 steps，topk1，4 draft tokens |
| 图 | decode full CUDA graph，`--cuda-graph-bs-decode 1..8` |
| 启动脚本 | `script/glm53/0.5.23/{p523.sh,d523.sh}`，驱动 `boot523_1p1d.sh` |
| 0.5.8 基线 | `script/glm53/{p.sh,d.sh}`，驱动 `boot058_1p1d.sh` |

`MAX_RUN` 是全局值，**每个 DP rank 的上限是 `MAX_RUN // attn_dp_size`**。
DP8 下 `MAX_RUN=48` → 每 rank 6；`MAX_RUN=32` → 每 rank 4。

---

## 2. 测量口径（这一节是本轮最大的收获，也是最容易踩的坑）

### 2.1 client 并发 ≠ server batch

`bench_fixed_bs_*.sh` 里的 `K` 是 client 并发 / DP，不是 decode batch。实际 batch
只能取 decode 日志的 `#running-req` 或 `[SEG] bs`。`bs` 在窗口内可以用
`(bs₁·steps₁ − bs₀·steps₀)/(steps₁−steps₀)` 差分（分母是 `_n`，两者一致）。

### 2.2 `[SEG]` 的 `wall` 不能跨窗口差分

`stage_timing.py` 里 `wall = _wall_ms / _wall_n`，而 `_wall_n` 只统计间隔
≤ `SEG_WALL_MAX_MS` 的步，**与 `steps`(`_n`) 不是同一个分母**。用 `steps` 去差分
`wall` 会得到负数或几十毫秒的错值（本轮实际出现过 `steps=0`、`wall=-885`）。
可窗口化的是 `bs` 和各 stage 字段（它们的分母都是 `_n`）。

### 2.3 客户端 TPOT 是可信的，且等于 `(Latency − TTFT)/(OUT−1)`

对每一档都验证过两者完全相等，所以 TPOT 的定义是清楚的。但它**只是单请求速率**，
引用时必须同时给出 accept length 和实际 batch，否则会自相矛盾：
`TPOT × accept_len ≈ step wall`，`step wall × bs ≈ 常数`。

由此产生的两个陷阱：

- 同一个版本在更高并发下 TPOT 会变大（bs 变大），这不是变慢。MegaMoE 的
  "TPOT 从 31.7 涨到 37.1" 就是这个原因（bs 2.02 → 3.52）。
- `Avg Decoded Tok/Iter` 超过 `num_draft_tokens + 1` 时该次运行作废。本轮
  join-after 那次报 8.56（上限 5）而服务端 `accept len` 仍是 2.85，两者矛盾，
  该次所有 TPOT 都不能用。

### 2.4 固定 batch 必须 90% cache，且 bs 上限受 KV 限制

早期固定 batch 用的是 `input=77012 / prefix=77000`，实测 `KV Cache Hit = 99.6%`，
不是要求的 90%，那张对比表已作废。

正确的 90% 口径：`input=77000 / prefix=69300 / suffix=7700`（实测 hit 90.1–93.6%）。

77k 上下文下每请求约 96k KV tokens，KV 池约 405–414k，所以**每 rank 的 bs 上限约 4**。
要跑到 bs=5 必须换更短上下文，当前不允许，因此对比档位是 1/3/4。

### 2.5 profiler 只能用来确认 kernel 是否存在

开启 profiler 会串行化 server，`[SEG]` 出现 `steps=0`、负值，TPOT 出现 53/208/1449 ms。
trace 只用于确认 kernel 名称与调用次数，不用于任何耗时结论。

### 2.6 PD 的负载必须走 router

直连 decode worker 会因为没有 bootstrap room id 而被拒；profiler 可以直连 decode 的
监听地址启动，但负载仍要走 `:8080`。

---

## 3. 已落地的优化（6 个提交，均在 CR 中）

| 提交 | 内容 | 实测效果 |
|---|---|---|
| `c7ca78c` | 对齐 0.5.8 的 DeepEP low-latency dtype 契约：FP16→BF16 只在 dispatch 入口转一次，删掉 dispatch/combine 内重复的 conversion | 固定 bs 下消除主要差距 |
| `623ffb7` | graph DSA metadata 按 token-major `[1]*(bs*K)` 折叠 | 修正 verify 路径语义 |
| `e3f603d` | request-major DSA draft rows 按 `[K]*bs` 折叠 | 修正 draft 路径语义 |
| `87c3639` | 每步长度写入少两次 launch：`cumsum(out=...)` 代替 `cat(zeros, cumsum)` | `_dsa_store_lens_and_cu` 210→155 µs，约 0.17 ms/step |
| `21f1626` | host 镜像刷新改为每个镜像一次 `copy_` | bs2/3 下 0.68→0.39 ms/step |
| `96e9da7` | 两个 SLO 形状的 host 段 bench（测试） | 供后续回归用 |

`32522be`（已在 origin）是 `[SEG]` 探针本身。

关键结论：**固定真实 batch 下 0.5.23 已追平 0.5.8**。

```text
bs≈1（77k / 90% cache / out 256）
  0.5.23  bs 1.36  TPOT 26.54  accept 2.70  KV 90.2%
  0.5.8   bs 1.17  TPOT 26.32  accept 2.54  KV 92.8%
  差 0.22 ms（0.8%），且 0.5.23 的 bs 更高
```

---

## 4. SLO（2 req/s，240 请求，77000 输入 / 69300 前缀 / 600 输出）

```text
                bs     wall    accept  TPOT    out tok/s  TTFT    Latency
deepep         2.02   80.92    2.88    31.74   678.5      42.9s   61.91s
megamoe        3.52  109.72    3.05    37.13   669.6      37.5s   59.75s
```

按每行算 MegaMoE 更快（31.2 vs 40.0 ms/row），TPOT 偏高是因为并发从 2.02 提到 3.52；
端到端吞吐 -1.3%、延迟与 TTFT 都更好。**只有 TPOT 一项变差**。

历史 0.5.8 参考（同一 SLO 形状，未在同一 boot 复测）：TPOT 28.44，accept 2.68，
out 658.78。

---

## 5. MegaMoE 迁移

移植自 0.5.8 的 `b89a38acd` / `b48757625`，代码放在插件里（主仓库
`/home/zx/code/v0.5.23/sglang` 不是 git 仓库，无法单独提交）：

```text
sglang_kunlun/hooks/layers/moe/klx_megamoe.py   hydrax int8_int4 融合算子
sglang_kunlun/hooks/layers/moe/megamoe.py       环境/权重/forward 接线
sglang_kunlun/hooks/layers/moe/fused_moe_triton/layer.py  megamoe → StandardDispatcher
```

启用：`MOE_BACKEND=megamoe`（decode 侧；prefill 目前仍是 deepep）。

必需环境变量（已写进 `d523.sh` / `p523.sh` 顶部，必须在启动前生效）：

```bash
unset MEGA_MOE_USE_SWITCH
unset MEGA_MOE_USE_RELAY MEGA_MOE_RELAY_COMBINE_MODE
unset MEGA_MOE_COMBINE_XLINK_RELAY_FUSION_IBGDA
export MEGA_MOE_DISPATCH_USE_RELAY=0
export MEGA_MOE_COMBINE_USE_RELAY=0
export MEGA_MOE_RELAY_DISPATCH_DEDUP=0
```

`XSHMEM_SYMMETRIC_SIZE` 由插件按 `klx_megamoe.megamoe_xshmem_size()` 在进程内设置
（当前 0.91 GiB / 容量 384），因为它必须早于 bkcl 分配对称堆。

已验证：

- 双端 ready，`Failed to apply hooks = 0`，`not initialized from checkpoints = 0`
- `symm buffer ready: experts=256 cap=384 topk=8 hidden=6144 ffn=2048`
- fused op 确实被调用：`[megamoe] largest fused call: num_tokens=24`
- accept length 正常（服务端 mean 2.85）

未解决：**shared expert 的双流顺序**。`_run_shared_experts` 的 `wait_stream` 放在
fused op 之前（0.5.8 原样），这样只能和 gate+topk 重叠。移到 fused op 之后实测：

```text
                 TPOT    accept   输出吞吐
join-before      36.87   3.12     661.3 tok/s
join-after       38.76   8.56     672.1 tok/s   ← accept 超过上限 5，不可用
```

吞吐只多 1.7%，但 accept 变成不可能值，所以保持 join-before，并在代码注释里写明。

---

## 6. 诊断工具清单

| 工具 | 用途 |
|---|---|
| `stage_timing.py`（`SGLANG_KUNLUN_SEG=1`） | 每步 host 分段，`wall == run_batch + outside` |
| `script/glm53/tools/seg_pair.py` | 把 before/after 累计行算成一个窗口，拒绝 `n<=0` |
| `script/glm53/bench_one_cell.sh` | 单档一次调用，空闲时取快照 |
| `script/glm53/bench_cell.sh` | 运行中每 15s 采样，取步数增量最大的稳态窗口 |
| `script/glm53/bench_slo_cmp.sh` | 240 请求 open-loop SLO |
| `script/glm53/tools/check_argv.py` | 检查反斜杠续行里夹注释导致 argv 被截断 |
| `test/test_decode_host_segments.py` | host 段微基准，bs2/3 两个形状 |
| `compare_masked_moe_dumps.py` | masked GEMM 逐层输入/输出比对 |

---

## 7. 下一步：P / D 分离压测

目标：把 prefill 与 decode 的贡献分开，避免 PD 耦合把结论搅在一起。做法上要点：

1. **固定 P**：decode 侧只发一个形状固定的短请求，测 prefill 的 TTFT 与吞吐随输入长度
   的变化；prefill 侧不要开 `--enable-prefill-cp` 与不开各测一次。
2. **固定 D**：绕过 prefill，用已有的长前缀请求直接进 decode（必须带 bootstrap room id，
   否则被拒），只观察 `[SEG]` 的 `run_batch`/`run_eagle_verify`/`draft`。
3. 两边都必须记录：实际 `bs`、`accept len`、`KV Cache Hit`、`MAX_RUN`、`MAX_RUN // dp`。
4. 判据用 `step wall / row`（每行每步耗时），不要用单请求 TPOT 横向比。

---

## 8. 之后的重构计划（已审计，待执行）

体检：185 个 py 文件 / 53.5k 行 / 113 个环境变量 / 36 个 ≥150 行的函数。

已完成的 Stage 0（本次）：

- `ep_moe/layer.py:841` 的 `self.moe_castte_mode` 是未定义属性，W8A8/FP8 DeepEP 一上就
  `AttributeError`；已改为只看权重 dtype。
- `registry.py` 里 `extra_ops` 被 try-import 之后又无条件 import 一次，与注释
  "absent .so 不是启动失败" 相反；已删掉无条件那次。
- `registry.py` 的 `DEBUG_HOOK_MODULES` 里 `debug.dsv4_mtp_nextn_probes` 不存在，被
  `except ImportError` 静默吞掉；已删。

后续（按风险从低到高）：

1. **纯删除约 1.8k 行**：`ep_moe/layer.py` 8 个不可达 forward（约 760 行）、
   `kernel_ops.py` 零引用函数（约 730 行）、DSA CP in-seq-split 子系统（172 行）、
   `kunlun_backend.py` 两处无条件 `return` 之后的级联块（67 行）。
2. **去重**：`speculative_attention` 30 参数调用 3 份、级联块 4 份、LoD 构造 5 处、
   host 镜像刷新 3 套、`_paged_topk_transform` 两份、per-token int8 量化两处。
3. **拆分 `kernel_ops.py`**（8910 行 / 84 次提交）按 11 个家族拆开，只留注册机制。
4. **诊断收敛**：`_dump_masked_moe_call` 目前每个 masked MoE 层无条件调用 4 次；
   `eagle_worker_v2.py` 的 AROUND 每次调用都换 `draft_runner.forward`；
   这些应改成 probe 开启时才注册 hook。
