# DeepSeek-V4-Flash (Kunlun)

## Overview

This page collects **measured tuning results** for **DeepSeek-V4-Flash** on
**Kunlun P900 (XPU)** with SGLang, served through the `sglang-kunlun` plugin
stack. It is not a deployment recipe: it does not cover model download, image or
launch scripts. Each item below states a knob, what we measured when we changed
it, and the mechanism behind the result.

These are measurements, not a recommended configuration. The absolute values are
specific to our hardware and model, so treat them as starting points and the
mechanisms as the transferable part. See the version note at the end before
quoting any percentage.

## Key Findings

- Per-rank decode batch has a knee around 80
- A measured expert layout is worth a few percent, and how to produce one
- `--chunked-prefill-size` is a per-attention-DP-rank budget
- `--mem-fraction-static` has to be chosen per phase
- `--kv-cache-dtype` has to be passed explicitly

## Tuning Notes

### 1. Per-rank decode batch has a knee around 80

Raising the per-rank decode batch from 32 to 80 gave **+25.7% output tok/s and
-32% J/output-token**. Past the knee the return collapses: 128 added only about
4% more throughput while TPOT worsened by 53%. If you are sizing
`--cuda-graph-max-bs-decode` and `--max-running-requests` for throughput rather
than for single-stream latency, the useful range ends near the knee.

### 2. A measured expert layout is worth a few percent

`--init-expert-location` defaults to `trivial`, which maps logical expert `i` to
physical expert `i` and repeats the same map for every layer. Replacing it with a
layout derived from observed expert load gave **+3-7% prefill input tok/s** on a
PD-disaggregated prefill role (median about 5%, same sign across three rounds).

**Why the accounting has to be per layer.** The expert all-to-all is a collective
within each MoE layer, so every layer waits for its slowest peer. On our workload
the cumulative imbalance over the whole model (max/mean across ranks) was only
**1.203**, while the **per-layer** imbalance was **1.917**: hot experts move
between layers, so summing over layers cancels them out. Judging this by the
cumulative figure leads to the conclusion that it is not worth doing.

**How to produce the input.**

1. Start the server with `--expert-distribution-recorder-mode stat`
   (`--enable-eplb` sets this implicitly).
2. `POST /start_expert_distribution_record`, drive a *representative* load, then
   `POST /stop_expert_distribution_record` and
   `POST /dump_expert_distribution_record`. This writes
   `expert_distribution_recorder_<timestamp>_<rank>.pt`.
3. Restart with `--init-expert-location <that file>`.

The dumped file holds **observed counts, not a layout**. The rebalancing itself
runs at startup, where `init_by_eplb` calls `rebalance_experts`, so the layout you
get is whatever the in-tree algorithm derives from your counts. A 2-D
`[layers, experts]` tensor is accepted as well and is equivalent to the raw 3-D
dump, because the algorithm sums over the pass dimension; pre-summing it only
saves tens of megabytes. Both `.pt` and `.json` are accepted.

**Two traps.**

| Trap | What happens |
| --- | --- |
| Recording needs an explicit `--deepep-mode normal` | With a DeepEP a2a backend, `_SinglePassGatherer.init_new` accepts only `normal` or `low_latency` and raises `NotImplementedError` on `auto`. `normal` also disables CUDA graphs, so the throughput of the recording run is **not** comparable to your baseline: use that run to measure skew, not performance. |
| `--init-expert-location` can change two things, not one | If you have not set `--ep-dispatch-algorithm` explicitly, setting `--init-expert-location` to anything other than `trivial` also resolves that algorithm for you: to `static` when an a2a backend is in use, or to `dynamic` when `--moe-a2a-backend` is `none`. |

That second one is why the number above needs a control arm. We measured the
`static` switch on its own at **-0.7% (within noise)** in a configuration with no
redundant replicas, which is what lets the gain be attributed to the layout rather
than to the algorithm switch. If you reproduce this, measure that arm too.

**Scope.** The counts have to match *your* traffic, since expert load is a
property of the model's gate interacting with your inputs. A layout derived from a
different workload can be worse than `trivial`, so treat a dump as
workload-specific.

### 3. `--chunked-prefill-size` is a per-attention-DP-rank budget

With data-parallel attention enabled, the value you pass is not the global chunk
size: it is divided across attention DP ranks, and the startup log prints the
per-rank number, not the number you typed. Upstream states the same unit in its
own validation message ("tokens per attention DP rank"). So to land on a target
global chunk, pass `target * dp_size`, and keep the per-rank quotient a whole
number of pages.

### 4. `--mem-fraction-static` has to be chosen per phase

On our 4-card configuration (84.90 GB of weights per card) the workable window
was narrow and phase-dependent:

| Value | Result |
| --- | --- |
| below 0.88 | the static pool cannot hold the weights |
| 0.90 | what we used for pure-prefill workloads |
| 0.92 | fine for decode, but a pure-prefill run hit OOM on its first real prefill, in the normal-mode MoE buffer |
| above 0.94 | the DeepEP buffer cannot get memory |

A value that is stable in decode is therefore not automatically safe for a
prefill-heavy deployment.

### 5. Pass `--kv-cache-dtype` explicitly

The default `auto` is rewritten to `fp8_e4m3` by the upstream DeepSeek-V4
override, and this plugin's pack kernel rejects it
(`NotImplementedError: Only support bf16/fp16 by now.`). Pass `bfloat16`,
`float16`, or `half`.

### Version note for the numbers above

The percentages in items 1 and 2 were measured on
`hac_test/aiak-inference-sglang:p800_wangzhihui07_20260817_095734` and have
**not** been re-measured on SGLang 0.5.17 or later. Method: 20 s warmup plus
30 s measurement per cell, throughput taken as an explicit windowed difference
of `generation_tokens_total` in the steady-state middle of the run, power
sampled over exactly the same window, three repeats per cell, `retract` zero
throughout.

One caveat on combining them: enabling the measured expert layout and two-batch
overlap together was measured at +14.7% (95% CI +11.4-18.0%). Both the linear
sum (+11.3%) and the compounded product (+16.6%) fall inside that interval, so
the data cannot distinguish the two. **Do not multiply ratios measured in
different deployments** - quote them separately.
