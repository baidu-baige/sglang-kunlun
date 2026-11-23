# SGLang Kunlun

SGLang Kunlun is an out-of-tree (OOT) hardware platform plugin for running
[SGLang](https://github.com/sgl-project/sglang) on Kunlun XPU hardware. It
registers the Kunlun platform through SGLang entry points and keeps Kunlun-
specific bootstrap, runtime hooks, and kernels outside the SGLang source tree.

[Contributing](CONTRIBUTING.md)

## Latest News

- **2026/11** — The `sglang-kunlun` package is released at version `0.5.21`,
  tracking the community SGLang `0.5.21` engine lineage.
- **2026/11** — Adds DeepSeek-V4.1-Flash (W4A8) support with an encoder /
  language-only prefill / disaggregated decode (EPD) deployment, on top of the
  community SGLang 0.5.21 engine.
- **Current development** — Kunlun platform registration, runtime bootstrap,
  attention, KV-cache, MoE, quantization, speculative decoding, and
  disaggregation hooks are maintained in this repository.

## Overview

The plugin exposes the following SGLang entry points:

- `sglang.srt.platforms`: registers the `kunlun` platform.
- `sglang.srt.plugins`: registers the Kunlun general hooks.
- `sglang-kunlun-launch`: provides an explicit Kunlun launcher when the
  standard SGLang launcher is not sufficient for a deployment.

The platform activation is conditional: on a machine where `torch_xmlir` is
available, the plugin activates `KunlunSRTPlatform`; otherwise it returns
`None` and does not take over platform discovery.

### Key Features

- **Out-of-tree integration** — Integrates with SGLang through Python entry
  points and hooks instead of modifying the SGLang source tree.
- **DeepSeek-V4.1-Flash EPD** — Encoder, language-only prefill, and
  disaggregated decode roles with Mooncake/RDMA KV transfer; validated launch
  scripts live in [`examples/deepseek-v4.1-epd/`](examples/deepseek-v4.1-epd/).
- **Kunlun bootstrap** — Applies import-order and CUDA-compatible compatibility
  shims needed by the Kunlun runtime.
- **Attention and graph integration** — Provides the Kunlun attention backend,
  speculative-decoding attention paths, and graph-runner integration.
- **KV-cache management** — Provides Kunlun MHA/MLA/DSA KV pools and a paged
  KV-cache allocator.
- **MoE and quantization hooks** — Includes Kunlun paths for fused MoE,
  token dispatch, and INT8/W8A8/W4A8-related operations.
- **Speculative decoding** — Includes EAGLE and multi-layer EAGLE hooks.
- **Disaggregation hooks** — Includes integration points for Prefill/Decode
  disaggregation and Mooncake/RDMA-based KV transfer.
- **OpenAI-compatible serving** — Uses SGLang's OpenAI-compatible server
  interface once the Kunlun platform has been initialized.

## Prerequisites

- Kunlun XPU hardware; the deployment below targets Kunlun3 P800 (8 XPUs per
  node for the EPD example).
- Linux with a compatible Kunlun driver and runtime.
- Kunlun-enabled PyTorch with `torch_xmlir` available.
- Python `>= 3.10`.
- A SGLang runtime compatible with this repository; the validated engine is
  the community `sglang==0.5.21` release.

The repository's Python packaging metadata is in
[`pyproject.toml`](pyproject.toml), and the test/build dependencies are listed
in [`requirements.txt`](requirements.txt).

## Installation

For development, install this package in editable mode in an environment that
already has the SGLang runtime:

```bash
pip install -e /path/to/sglang-kunlun
```

## Quick Start

### Build the Container Image

Build from the **parent** directory of this checkout:

```bash
# from the directory that contains sglang-kunlun/
docker build -f sglang-kunlun/dockerfile/Dockerfile -t sglang-kunlun:0.5.21 .
```

The recipe installs the community `sglang==0.5.21` wheel (sha256 verified) and
applies the DeepSeek-V4.1 EPD engine patches. To build the engine from source
instead, pass `SGLANG_ENGINE_SOURCE=source` with a pinned `SGLANG_REF` /
`SGLANG_COMMIT`; see [`dockerfile/README.md`](dockerfile/README.md).

### Start a DeepSeek-V4.1-Flash EPD Deployment

The validated scripts run one per host inside the image above:

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-W4A8
export PREFILL_IP=<prefill-ip>
export ENCODER_URLS="http://<encoder-ip>:8000 ... http://<encoder-ip>:8007"

bash examples/deepseek-v4.1-epd/start_encoder.sh   # encoder host
bash examples/deepseek-v4.1-epd/start_prefill.sh   # prefill host
bash examples/deepseek-v4.1-epd/start_decode.sh    # decode host
```

IPs, RDMA device names, memory budgets, and parallelism must match the target
machine — the scripts expose them as environment variables.

### Route and Check the Endpoint

Put `sglang-router 0.3.2` in front of the prefill and decode endpoints to serve
OpenAI-compatible requests:

```bash
python3 -m sglang_router.launch_router \
  --worker-urls http://<prefill-ip>:8000 http://<decode-ip>:8000 \
  --host 0.0.0.0 --port 8101
```

List the model identifier exposed by the running server before sending a chat
request:

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

## Compatibility Notes

This repository is a platform plugin rather than a complete SGLang
distribution. Model support depends on the SGLang version, Kunlun runtime,
operator libraries, model weights, and launch configuration being compatible.
The table below records paths explicitly represented in the current code; it
is not a complete compatibility matrix.

| Model or path | Evidence in this repository | Notes |
| --- | --- | --- |
| DeepSeek-V4.1-Flash | W4A8 EPD launch scripts in [`examples/deepseek-v4.1-epd/`](examples/deepseek-v4.1-epd/) | Validated on 8xKunlun3 P800; other configurations must be re-validated. |
| DeepSeek-V4 | DeepSeek-V4 model and KV-cache compatibility hooks | Runtime acceptance must be verified with the target SGLang/Kunlun versions. |
| MiMo-V2-Flash | General W8A8 INT8 paths | No dedicated launch example in this branch. |
| Other SGLang models | General platform, attention, KV-cache, MoE, and quantization hooks | Validate each model family and quantization mode separately. |

## Version Matrix

| Component | Version or branch | Source |
| --- | --- | --- |
| `sglang-kunlun` | `0.5.21` | [`pyproject.toml`](pyproject.toml) |
| Release branch | `v0.5.21-dev` | This branch |
| SGLang core | `0.5.21` community release | Validated engine for the DeepSeek-V4.1-Flash EPD topology |
| PyTorch | `2.9.0+cu129` | Validated runtime |
| xspeedgate_ops | `1.7.0+cbe2814.torch29` | Validated operator library |

## Architecture

```text
sglang-kunlun/
├── sglang_kunlun/
│   ├── platform/              # Kunlun device and SGLang platform entry point
│   ├── bootstrap/             # Import-order, runtime, and compatibility shims
│   ├── hooks/
│   │   ├── layers/            # Attention, MoE, linear, rotary, and quantization hooks
│   │   ├── mem_cache/         # Kunlun KV pools and paged allocator
│   │   ├── model_executor/    # Model-runner and KV-cache integration
│   │   ├── models/            # Model-specific hooks, including DeepSeek-V4
│   │   ├── speculative/       # EAGLE and multi-layer EAGLE hooks
│   │   ├── disaggregation/    # Prefill/Decode and KV-transfer hooks
│   │   └── entrypoints/       # HTTP server integration hooks
│   ├── kernels/               # Kunlun kernel shims and operator replacements
│   ├── models/                # External model package registration
│   └── launch.py              # Explicit Kunlun launcher
├── examples/
│   └── deepseek-v4.1-epd/     # Encoder / prefill / decode launch scripts
├── test/                      # Unit and contract tests
├── dockerfile/                # Container build recipe (Dockerfile, patches, vendor assets)
├── pyproject.toml             # Package metadata and SGLang entry points
└── requirements.txt           # Build and test dependencies
```

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

## Contributing

We welcome contributions from the community. Please read the
[Contributing Guide](CONTRIBUTING.md) before submitting a pull request.

Recommended pull-request title prefixes are:

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

## License

This project is licensed under the Apache License 2.0, as declared in
[`pyproject.toml`](pyproject.toml).
