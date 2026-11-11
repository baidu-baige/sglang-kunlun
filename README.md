![SGLang Kunlun Logo](images/sglang-kunlun-logo-v1.png)

<p align="center">
  <a href="#overview"><b>📖 Overview</b></a> |
  <a href="#quick-start"><b>🚀 Quick Start</b></a> |
  <a href="#installation"><b>📦 Installation</b></a> |
  <a href="#architecture"><b>🧩 Architecture</b></a> |
  <a href="CONTRIBUTING.md"><b>💬 Contributing</b></a>
</p>

<p align="center">
  <img alt="GitHub License" src="https://img.shields.io/github/license/baidu-baige/sglang-kunlun">
  <img alt="GitHub Stars" src="https://img.shields.io/github/stars/baidu-baige/sglang-kunlun">
  <img alt="GitHub Forks" src="https://img.shields.io/github/forks/baidu-baige/sglang-kunlun">
  <img alt="GitHub Issues" src="https://img.shields.io/github/issues/baidu-baige/sglang-kunlun">
  <img alt="Python Version" src="https://img.shields.io/badge/python-%3E%3D3.10-blue">
</p>

---

**SGLang Kunlun** (`sglang-kunlun`) is an out-of-tree (OOT) hardware platform
plugin designed to seamlessly run [SGLang](https://github.com/sgl-project/sglang)
on the **Kunlun XPU**. It registers the Kunlun platform through SGLang entry
points and keeps Kunlun-specific bootstrap, runtime hooks, and kernels outside
the SGLang source tree.

---

## Latest News 🔥

- [2026/11] 🚀 **DeepSeek-V4.1-Flash support on SGLang 0.5.21** — New
  `v0.5.21-dev` branch: EPD three-role serving (encoder-only / language-only
  prefill / disaggregated decode), CP prefill with multimodal inputs, the
  DSA/compressed-attention backend, Engram, mHC and fp4 index-K store. See
  [DeepSeek-V4.1-Flash Support](#deepseek-v41-flash-support-sglang-0521)
- [2026/09] ✨ **Open-source engineering** — Added governance and community
  files (code of conduct, security policy, DCO, issue/PR templates,
  pre-commit, and pytest configuration) aligned with industry-standard
  open-source practices
- [2026/06] 🌟 **Initial release of SGLang Kunlun** — The `sglang-kunlun`
  package was created at version `0.1.0`
- **Current development** — Kunlun platform registration, runtime bootstrap,
  attention, KV-cache, MoE, quantization, speculative decoding, and
  disaggregation hooks are maintained in this repository

---

## Overview

The plugin exposes the following SGLang entry points:

- `sglang.srt.platforms`: registers the `kunlun` platform.
- `sglang.srt.plugins`: registers the Kunlun general hooks.
- `sglang-kunlun-launch`: provides an explicit Kunlun launcher when the
  standard SGLang launcher is not sufficient for a deployment.

The platform activation is conditional: on a machine where `torch_xmlir` is
available, the plugin activates `KunlunSRTPlatform`; otherwise it returns
`None` and does not take over platform discovery.

### ✨ Key Features

- **Out-of-tree integration** — Integrates with SGLang through Python entry
  points and hooks instead of modifying the SGLang source tree.
- **Kunlun bootstrap** — Applies import-order and CUDA-compatible compatibility
  shims needed by the Kunlun runtime.
- **Attention and graph integration** — Provides the Kunlun attention backend,
  speculative-decoding attention paths, and graph-runner integration.
- **KV-cache management** — Provides Kunlun MHA/MLA/DSA KV pools and a paged
  KV-cache allocator.
- **MoE and quantization hooks** — Includes Kunlun paths for fused MoE,
  token dispatch, and INT8/W8A8-related operations.
- **Speculative decoding** — Includes EAGLE, multi-layer EAGLE and DSpark
  draft hooks.
- **DeepSeek-V4.1-Flash** — Dedicated prefill and decode paths: CP prefill with
  multimodal inputs, DSA compressed attention, Engram, mHC, fp4 index-K store,
  and the EPD encoder/language-only role split. See
  [DeepSeek-V4.1-Flash Support](#deepseek-v41-flash-support-sglang-0521).
- **Disaggregation hooks** — Includes integration points for Prefill/Decode
  disaggregation and Mooncake/RDMA-based KV transfer. Validate this path in
  the target runtime before using it in production.
- **OpenAI-compatible serving** — Uses SGLang's OpenAI-compatible server
  interface once the Kunlun platform has been initialized.

---

## Prerequisites

- **Hardware**: Kunlun XPU (the deployment example below targets Kunlun3 P800)
- **OS**: Linux with a compatible Kunlun driver and runtime
- **Software**:
  - Python `>= 3.10`
  - Kunlun-enabled PyTorch with `torch_xmlir` available
  - A SGLang checkout/runtime and `sgl-kernel` checkout compatible with this
    repository. The exact SGLang and runtime versions are deployment-specific;
    keep them aligned with the current branch

The repository's Python packaging metadata is in
[`pyproject.toml`](pyproject.toml), and the test/build dependencies are listed
in [`requirements.txt`](requirements.txt).

---

## Installation 📦

Set `YOUR_PATH` to the workspace containing `sglang-kunlun` and the compatible
SGLang runtime, then install this package in editable mode:

```bash
YOUR_PATH=/path/to/your/workspace
pip install -e "${YOUR_PATH}/sglang-kunlun"
```

### Build the Container Image

The open-source release ships the container build recipe instead of a prebuilt
image. [`dockerfile/`](dockerfile/README.md) contains the Dockerfile, the install
script, the engine patches required by DeepSeek-V4.1 EPD on SGLang 0.5.21, and
the pinned Kunlun runtime versions. The build context must be the parent
directory of this checkout:

```bash
# from the directory that contains sglang-kunlun/
docker build -f sglang-kunlun/dockerfile/Dockerfile \
    -t sglang-kunlun:0.5.21 .
```

The image downloads the community `sglang==0.5.21` wheel from PyPI, verifies its
sha256, applies `dockerfile/patches/*.patch`, installs the runtime versions from
`dockerfile/xpu_requirements.env`, and installs this plugin. See
[`dockerfile/README.md`](dockerfile/README.md) for the source-build route and the
override arguments.

---

## Quick Start 🚀

### Start a DeepSeek-V4.1-Flash EPD Deployment

DeepSeek-V4.1-Flash is served with the EPD topology described in
[DeepSeek-V4.1-Flash Support](#deepseek-v41-flash-support-sglang-0521): an
encoder role, a language-only prefill role and a disaggregated decode role.
Set the model path and the per-role addresses first:

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-W4A8
export MODEL_ID=DeepSeek-V4.1-Flash

export HOST_IP=$(hostname -i)            # prefill pod IP
export PREFILL_IP=$HOST_IP               # encoder registration target
export POD_IP=$(hostname -i)             # decode pod IP
export ENCODER_URLS="http://<enc-ip>:8000 ... http://<enc-ip>:8007"
export IB_DEVICES=mlx5_0,mlx5_1          # RDMA devices for mooncake KV transfer
```

Common runtime variables:

```bash
export SGLANG_PLATFORM=kunlun SGLANG_USE_XPU=1 SGLANG_PLUGINS=kunlun_hooks
export SGLANG_HACK_FLASHMLA_BACKEND=kunlun
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=True
export SGLANG_OPT_USE_JIT_NORM=True SGLANG_OPT_USE_TOPK_V2=False
export SGLANG_TOPK_TRANSFORM_512_TORCH=True
export SGLANG_OPT_USE_FUSED_STORE_CACHE=1 SGLANG_KUNLUN_INDEXER_FP32=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export XMLIR_FORCE_USE_XPU_GRAPH=1 XMLIR_ENABLE_FAST_FC=1
export XPU_USE_FAST_SWIGLU=1 XPU_USE_DEFAULT_CTX=1
export CUDA_GRAPH_OPTIMIZE_STREAM=1
export MOONCAKE_USE_RDMA=TRUE
```

Encoder role (one worker per XPU; the validated pose starts eight workers on
ports 8000-8007):

```bash
python3 -u -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name "$MODEL_ID" \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --tp 1 --dp 1 --dtype float16 \
    --encoder-only --mm-attention-backend sdpa \
    --image-processor-backend pil \
    --encoder-transfer-backend zmq_to_scheduler \
    --encoder-register-urls "http://$PREFILL_IP:8997" \
    --max-running-requests 32
```

Prefill role (8 XPUs, CP interleave, language-only; sends KV over mooncake):

```bash
python3 -u -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name "$MODEL_ID" \
    --host "$HOST_IP" --port 8000 --dist-init-addr "$HOST_IP":7788 \
    --tp 8 --ep 8 --trust-remote-code \
    --dtype float16 --kv-cache-dtype float16 \
    --chunked-prefill-size 32768 --context-length 114688 \
    --max-total-tokens 250000 --max-prefill-tokens 250000 \
    --max-running-requests 4 --mem-fraction-static 0.85 \
    --swa-full-tokens-ratio 0.5 \
    --disable-radix-cache --disable-cuda-graph \
    --attention-backend kunlun_dsv4 \
    --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
    --moe-a2a-backend deepep --deepep-mode normal \
    --enable-prefill-cp --cp-strategy interleave \
    --enable-decoder-swa-bounded-replay \
    --language-only --image-processor-backend pil \
    --encoder-urls $ENCODER_URLS --encoder-bootstrap-port 8997 \
    --disaggregation-mode prefill --disaggregation-bootstrap-port 8200 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device "$IB_DEVICES" \
    --load-balance-method round_robin --enable-metrics
```

Decode role (8 XPUs, disaggregated, low-latency DeepEP with CUDA graphs):

```bash
export XSHMEM_SYMMETRIC_SIZE=536870912

python3 -u -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name "$MODEL_ID" \
    --host "$POD_IP" --port 8000 --trust-remote-code \
    --tp 8 --ep 8 --dtype float16 --kv-cache-dtype float16 \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 8192 --context-length 114688 \
    --max-total-tokens 1048576 --max-prefill-tokens 250000 \
    --max-running-requests 8 --swa-full-tokens-ratio 0.1 \
    --disable-radix-cache --disable-overlap-schedule \
    --cuda-graph-backend-prefill disabled \
    --cuda-graph-max-bs 16 --cuda-graph-bs-decode 8 \
    --attention-backend compressed \
    --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
    --moe-a2a-backend deepep --deepep-mode low_latency \
    --disaggregation-mode decode --disaggregation-bootstrap-port 8200 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device "$IB_DEVICES" --enable-metrics
```

> **Note**: The commands above are the validated 8×Kunlun3 P800 configuration.
> `MODEL_PATH`, IPs, RDMA device names, memory budgets and parallelism must
> match the target machine. The encoder role is not routed: start
> `sglang-router` next to the prefill role and register the prefill and decode
> endpoints with it.

### Send a Request

List the model identifier exposed by the running server before sending a chat
request (the router listens on the prefill side):

```bash
curl http://localhost:8101/v1/models
```

Then send a request using the model identifier returned by that endpoint:

```bash
curl http://localhost:8101/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-id-from-v1-models>",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

---

## DeepSeek-V4.1-Flash Support (SGLang 0.5.21)

The `v0.5.21-dev` branch adds DeepSeek-V4.1-Flash support on top of the
community SGLang 0.5.21 release. The validated deployment is an EPD topology:
one encoder role, one language-only prefill role, and a disaggregated decode
role behind `sglang-router`.

### Validated environment

| Component | Version / setting |
| --- | --- |
| SGLang (core) | `0.5.21` community release (PyPI wheel) |
| PyTorch | `2.9.0+cu129` |
| XSpeedGate operator library | `1.7.0+cbe2814.torch29` |
| Model | DeepSeek-V4.1-Flash, W4A8 checkpoint |
| Model dtype / KV-cache dtype | `float16` / `float16` |

> Upstream SGLang 0.5.21 does not carry DeepSeek-V4.1 EPD support. This
> release is validated with a small set of engine-side patches applied when
> the container image is built: the encoder / language-only split, float16 KV
> cache for `DeepseekV4*`, the language-only + CP guard, the prefill weight
> skip that avoids loading the ViT / aligner / image modules, and a Triton-3.0
> `small_copy` bypass. The code in this branch is the `sglang_kunlun` side of
> that delivery.

### Deployment topology

| Role | Parallelism / flags | Notes |
| --- | --- | --- |
| Encoder (E) | 8 single-card workers in one Pod, ports 8000-8007; `--encoder-only --mm-attention-backend sdpa --image-processor-backend pil` | `pil` avoids the wheel's Rust image extension, which is built against a different torch ABI |
| Prefill (P) | tp8 / ep8 / cp8; `--language-only`; `--chunked-prefill-size 32768` | CP interleave; does not build the ViT / aligner / image modules; sends KV over mooncake |
| Decode (D) | tp8 / ep8; `--deepep-mode low_latency --cuda-graph-bs-decode 8 --max-running-requests 8` | `XSHMEM_SYMMETRIC_SIZE=536870912` |
| Router | `sglang-router 0.3.2` | Start it inside the prefill container on port 8101 and keep its `--prometheus-port` off the engine ports |

### Validation results

| Check | Result |
| --- | --- |
| EPD text / image / encoder-split smoke | Pass (text `5`; image `42`, 189 image tokens) |
| GSM8K canary | 99/100 on the smoke run; the full opencompass sweep is still in progress |
| Prefill, 110K input, 1 output token, chunk 32768 | 32,462.7 tok/s (median of 3 runs, EP-fake dispatch) |
| Decode, 110K input / 1000 output | Not validated: normal / eager DeepEP hits XPU error `-705` at 110K, and the 0.5.21 low-latency + CUDA-graph decode path is still being ported |

Notes:

- The prefill number is a performance ceiling measured with the fake EP-dispatch
  algorithm, which skips expert routing cost; it is not a production dispatch
  number. Same-pose real-dispatch reference: 33,143.6 tok/s.
- The plugin keeps the V1.6 prefill optimization set. DSpark draft hooks are
  present but DSpark is **not** enabled in the validated serving pose.
- The candidate-based indexer path is not ported on 0.5.21: the CP dense
  indexer falls back to the plugin's torch scoring / top-k, which selects an
  exact full-window top-k.
- The decode side currently runs with `--deepep-mode low_latency` and CUDA
  graphs; text and image smoke requests pass, but the 110K / 1000-output
  decode sweep is pending.

---

## Measured Tuning Notes: DeepSeek-V4-Flash on P900

Findings from a parameter sweep on Kunlunxin P900, contributed for reference.
These are measurements, not a recommended configuration: the absolute values are
specific to our hardware and model, so treat them as starting points and the
mechanisms as the transferable part. See the version note at the end of the
section before quoting any percentage.

### 1. Per-rank decode batch has a knee around 80

Raising the per-rank decode batch from 32 to 80 gave **+25.7% output tok/s and
-32% J/output-token**. Past the knee the return collapses: 128 added only about
4% more throughput while TPOT worsened by 53%. If you are sizing
`--cuda-graph-max-bs-decode` and `--max-running-requests` for throughput rather
than for single-stream latency, the useful range ends near the knee.

### 2. A measured expert layout is worth a few percent

`--init-expert-location` defaults to `trivial`. Replacing it with a layout
derived from observed expert load gave **+3-7% prefill input tok/s** (median
about 5%, same sign across three rounds).

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

---

## Compatibility Notes

This repository is a platform plugin rather than a complete SGLang
distribution. Model support depends on the SGLang version, Kunlun runtime,
operator libraries, model weights, and launch configuration being compatible.
The table below records paths explicitly represented in the current code; it
is not a complete compatibility matrix.

| Model or path | Evidence in this repository | Notes |
| --- | --- | --- |
| DeepSeek-V4.1-Flash | `hooks/models/deepseek_v4.py`, `hooks/models/deepseek_v4_vl.py`, the `dsv41_*` layer hooks, the EPD encoder/language-only paths and the DSpark draft hooks | Validated on SGLang 0.5.21 with the EPD topology above (W4A8 weights, float16 KV cache). |
| DeepSeek-V4 | DeepSeek-V4 model and KV-cache compatibility hooks | Runtime acceptance must be verified with the target SGLang/Kunlun versions. |
| MiMo-V2-Flash | General W8A8 INT8 paths | Use only with a matching model/runtime environment. |
| Other SGLang models | General platform, attention, KV-cache, MoE, and quantization hooks | Validate each model family and quantization mode separately. |

---

## Version Matrix

| Component | Version or branch | Source |
| --- | --- | --- |
| `sglang-kunlun` | `0.1.0` | [`pyproject.toml`](pyproject.toml) |
| Release branch | `v0.5.21-dev` | This branch |
| SGLang | `0.5.21` community release | Validated engine for the DeepSeek-V4.1-Flash EPD topology |
| PyTorch | `2.9.0+cu129` | Validated runtime |
| XSpeedGate operator library | `1.7.0+cbe2814.torch29` | Validated operator wheel |

---

## Architecture 🧩

```text
sglang-kunlun/
├── sglang_kunlun/
│   ├── platform/              # Kunlun device and SGLang platform entry point
│   ├── bootstrap/             # Import-order, runtime, and compatibility shims
│   ├── hooks/
│   │   ├── layers/             # Attention, MoE, linear, rotary, quantization and DSV4.1 hooks
│   │   ├── mem_cache/          # Kunlun KV pools, paged allocator and asradix cache hooks
│   │   ├── model_executor/     # Model-runner, KV-cache and CP multimodal integration
│   │   ├── models/             # Model-specific hooks, including DeepSeek-V4.1-Flash
│   │   ├── speculative/        # EAGLE, multi-layer EAGLE and DSpark hooks
│   │   ├── managers/           # Scheduler-side helpers
│   │   ├── disaggregation/     # Prefill/Decode and KV-transfer hooks
│   │   └── entrypoints/        # HTTP server integration hooks
│   ├── kernels/                # Kunlun kernel shims and DSV4 operator replacements
│   ├── models/                 # External model package registration
│   └── launch.py               # Explicit Kunlun launcher
├── test/                       # Unit and contract tests
├── build/                      # Repository packaging scripts
├── pyproject.toml              # Package metadata and SGLang entry points
├── requirements.txt            # Build and test dependencies
├── LICENSE                     # Apache License 2.0
└── NOTICE                      # Copyright and third-party notices
```

---

## Development and Validation

Run the repository tests in an environment with the required Python
dependencies installed:

```bash
python3 -m pytest -q
```

The CI packaging script creates `output/sglang_kunlun.tar.gz`:

```bash
sh build/build.sh
```

For changes affecting runtime behavior, validate on the target Kunlun
environment with the relevant model, quantization mode, parallelism, and
graph/disaggregation settings. A passing unit-test run alone does not establish
end-to-end inference or performance correctness.

---

## Contributing

We welcome contributions from the community! Please read our
[Contributing Guide](CONTRIBUTING.md) before submitting a PR.

### Community

- 📜 [**Code of Conduct**](CODE_OF_CONDUCT.md) — All contributors are expected to follow our code of conduct
- ✍️ [**Developer Certificate of Origin**](DCO) — Please sign off your commits with `git commit -s`; see [MAINTAINERS.md](MAINTAINERS.md) for the maintainer list
- 🔒 [**Security Policy**](SECURITY.md) — Report security vulnerabilities privately

### PR Classification

Use the following prefixes for PR titles:

- `[Platform]` — Platform activation and device integration
- `[Attention]` — Attention backends and KV-cache access
- `[Kernel]` — Kernel shims and operator implementations
- `[MoE]` — MoE, routing, and token-dispatch paths
- `[Quantization]` — Quantization and dequantization paths
- `[Speculative]` — EAGLE or other speculative-decoding paths
- `[PD]` — Prefill/Decode disaggregation and KV transfer
- `[Bugfix]` — Bug fixes
- `[Doc]` — Documentation improvements
- `[Test]` — Tests
- `[CI]` — CI or packaging improvements
- `[Misc]` — Other changes

---

## Star History 🔥

We opened the project on Jun 2026. We love open source and collaboration ❤️

<a href="https://www.star-history.com/?repos=baidu-baige/sglang-kunlun&type=date&legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=baidu-baige/sglang-kunlun&type=date&theme=dark&legend=top-left" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=baidu-baige/sglang-kunlun&type=date&legend=top-left" />
    <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=baidu-baige/sglang-kunlun&type=date&legend=top-left" />
  </picture>
</a>

---

## Sponsors 👋

We sincerely appreciate the [**KunLunXin**](https://www.kunlunxin.com/) team
for their support in providing XPU resources, which enabled efficient model
adaptation debugging, comprehensive end-to-end testing, and broader model
compatibility.

---

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file. See
[`NOTICE`](NOTICE) for copyright attribution and third-party software notices.
