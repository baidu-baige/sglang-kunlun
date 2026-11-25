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
  prefill / disaggregated decode), context-parallel prefill with image
  inputs, and optimized attention, MoE and indexer kernels. See
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
- **Speculative decoding** — Includes EAGLE, multi-layer EAGLE and the
  DeepSeek-V4 draft-model hooks.
- **DeepSeek-V4.1-Flash** — Dedicated prefill and decode paths with
  context-parallel prefill for image inputs, a compressed long-context
  attention backend, optimized MoE / Engram layers, and the EPD
  encoder/language-only role split. See
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

The validated launch scripts are in
[`examples/deepseek-v4.1-epd/`](examples/deepseek-v4.1-epd/):

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-W4A8
export PREFILL_IP=<prefill-ip>
export ENCODER_URLS="http://<encoder-ip>:8000 ... http://<encoder-ip>:8007"

bash examples/deepseek-v4.1-epd/start_encoder.sh   # encoder host
bash examples/deepseek-v4.1-epd/start_prefill.sh   # prefill host
bash examples/deepseek-v4.1-epd/start_decode.sh    # decode host
```

> **Note**: These scripts are the validated 8xKunlun3 P800 configuration and
> must run inside the image built from [`dockerfile/`](dockerfile/README.md).
> IPs, RDMA device names, memory budgets and parallelism must match the target
> machine. Put `sglang-router` in front of the prefill and decode endpoints to
> serve OpenAI-compatible requests.

### Send a Request

List the model identifier exposed by the running server before sending a chat
request (through `sglang-router`, which fronts the prefill and decode
endpoints):

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
| XSpeedGate operator library (`xspeedgate_ops`) | `1.7.0+cbe2814.torch29` |
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
| Decode (D) | tp8 / ep8 / dp8 with DP attention; `--deepep-mode low_latency --mem-fraction-static 0.70 --chunked-prefill-size 2048 --max-running-requests 128 --swa-full-tokens-ratio 0.06` | Same pose as the validated 0.5.21 release |
| Router | `sglang-router 0.3.2` | Start it inside the prefill container on port 8101 and keep its `--prometheus-port` off the engine ports |

### Optimized paths in this release

- **Context-parallel prefill** — splits long prompts across the eight XPUs,
  including image inputs.
- **Compressed / DSA attention** — a faster long-context attention path backed
  by the compressed KV index.
- **Engram and mHC layers** — fused Kunlun kernels for the Engram lookup and
  the mHC normalization.
- **FP4 index-K quantization** — quantizes and stores index-K in FP4 with a
  native kernel instead of the PyTorch fallback.
- **Indexer top-k** — exact full-window top-k selection on Kunlun kernels.
- **Cast and fusion clean-up** — native fp16/bf16 casts plus fused
  hc_mix / hc_combine, removing per-layer copies.

Accuracy and performance results depend on the hardware and model; validate
them on the target machine before using this topology in production.

---

## Compatibility Notes

This repository is a platform plugin rather than a complete SGLang
distribution. Model support depends on the SGLang version, Kunlun runtime,
operator libraries, model weights, and launch configuration being compatible.
The table below records paths explicitly represented in the current code; it
is not a complete compatibility matrix.

| Model or path | Evidence in this repository | Notes |
| --- | --- | --- |
| DeepSeek-V4.1-Flash | DeepSeek-V4.1 model, KV-cache, attention and EPD hooks | Validated on SGLang 0.5.21 with the EPD topology above (W4A8 weights, float16 KV cache). |
| DeepSeek-V4 | DeepSeek-V4 model and KV-cache compatibility hooks | Runtime acceptance must be verified with the target SGLang/Kunlun versions. |
| MiMo-V2-Flash | General W8A8 INT8 paths | Use only with a matching model/runtime environment. |
| Other SGLang models | General platform, attention, KV-cache, MoE, and quantization hooks | Validate each model family and quantization mode separately. |

The HTTP entrypoint also normalizes two client conventions for the
DeepSeek-V4.1 reasoning detector: legacy top-level `thinking` /
`enable_thinking` fields are mapped into `chat_template_kwargs`, and a request
that carries no system message gets the default `You are a helpful assistant.`
system prompt. Set `SGLANG_KUNLUN_DEFAULT_SYSTEM=""` to disable the
system-message injection.

---

## Version Matrix

| Component | Version or branch | Source |
| --- | --- | --- |
| `sglang-kunlun` | `0.5.21` | [`pyproject.toml`](pyproject.toml) |
| Release branch | `v0.5.21-dev` | This branch |
| SGLang | `0.5.21` community release | Validated engine for the DeepSeek-V4.1-Flash EPD topology |
| PyTorch | `2.9.0+cu129` | Validated runtime |
| XSpeedGate operator library (`xspeedgate_ops`) | `1.7.0+cbe2814.torch29` | Validated operator wheel |

---

## Architecture 🧩

```text
sglang-kunlun/
├── sglang_kunlun/
│   ├── platform/              # Kunlun device and SGLang platform entry point
│   ├── bootstrap/             # Import-order, runtime, and compatibility shims
│   ├── hooks/
│   │   ├── layers/             # Attention, MoE, linear, rotary, quantization and DeepSeek-V4 hooks
│   │   ├── mem_cache/          # Kunlun KV pools, paged allocator and cache hooks
│   │   ├── model_executor/     # Model-runner, KV-cache and CP multimodal integration
│   │   ├── models/             # Model-specific hooks, including DeepSeek-V4.1-Flash
│   │   ├── speculative/        # EAGLE, multi-layer EAGLE and DeepSeek-V4 draft hooks
│   │   ├── managers/           # Scheduler-side helpers
│   │   ├── disaggregation/     # Prefill/Decode and KV-transfer hooks
│   │   └── entrypoints/        # HTTP server integration hooks
│   ├── kernels/                # Kunlun kernel shims and DSV4 operator replacements
│   ├── models/                 # External model package registration
│   └── launch.py               # Explicit Kunlun launcher
├── test/                       # Unit and contract tests
├── examples/
│   └── deepseek-v4.1-epd/      # Encoder / prefill / decode launch scripts
├── dockerfile/                 # Container build recipe (Dockerfile, patches, vendor assets)
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
