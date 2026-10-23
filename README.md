# SGLang Kunlun

SGLang Kunlun is an out-of-tree (OOT) hardware platform plugin for running
[SGLang](https://github.com/sgl-project/sglang) on Kunlun XPU hardware. It
registers the Kunlun platform through SGLang entry points and keeps Kunlun-
specific bootstrap, runtime hooks, and kernels outside the SGLang source tree.

[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## Latest News

- **2026/06** — The `sglang-kunlun` package was created at version `0.1.0`.
- **Current development** — Kunlun platform registration, runtime bootstrap,
  attention, KV-cache, MoE, quantization, speculative decoding, and
  disaggregation hooks are maintained in this repository.
- **Newly supported features**
  - **BCG (Breakable CUDA Graph)** — breakable CUDA-graph capture for prefill,
    enabled with `--cuda-graph-backend-prefill breakable`,
    `--cuda-graph-max-bs-prefill 8192`, and
    `--cuda-graph-bs-prefill 128 384 512 1024 2048 4096 7168 8192`.
  - **HiSparse** — sparse attention, enabled with `--enable-hisparse` and
    `--hisparse-config '{"top_k":512,"device_buffer_size":4096,"host_to_device_ratio":8}'`.

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
- **Kunlun bootstrap** — Applies import-order and CUDA-compatible compatibility
  shims needed by the Kunlun runtime.
- **Attention and graph integration** — Provides the Kunlun attention backend,
  speculative-decoding attention paths, and graph-runner integration.
- **KV-cache management** — Provides Kunlun MHA/MLA/DSA KV pools and a paged
  KV-cache allocator.
- **MoE and quantization hooks** — Includes Kunlun paths for fused MoE,
  token dispatch, and INT8/W8A8-related operations.
- **Speculative decoding** — Includes EAGLE and multi-layer EAGLE hooks.
- **Disaggregation hooks** — Includes integration points for Prefill/Decode
  disaggregation and Mooncake/RDMA-based KV transfer. Validate this path in
  the target runtime before using it in production.
- **OpenAI-compatible serving** — Uses SGLang's OpenAI-compatible server
  interface once the Kunlun platform has been initialized.

## Prerequisites

- Kunlun XPU hardware; the deployment command below targets Kunlun3 P800.
- Linux with a compatible Kunlun driver and runtime.
- Kunlun-enabled PyTorch with `torch_xmlir` available.
- Python `>= 3.10`.
- A SGLang checkout/runtime and `sgl-kernel` checkout compatible with this
  repository. The exact SGLang and runtime versions are deployment-specific;
  keep them aligned with the current branch.

The repository's Python packaging metadata is in
[`pyproject.toml`](pyproject.toml), and the test/build dependencies are listed
in [`requirements.txt`](requirements.txt).

## Installation

Set `YOUR_PATH` to the workspace containing `sglang-kunlun` and the compatible
SGLang runtime, then install this package in editable mode:

```bash
YOUR_PATH=/path/to/your/workspace
pip install -e "${YOUR_PATH}/sglang-kunlun"
```

## Container Image

For a ready-to-run deployment, build the Kunlun inference image from
[`dockerfile/Dockerfile`](dockerfile/Dockerfile). It is a single self-contained
build file: the Kunlun runtime (`xre`, `xccl`), the Kunlun PyTorch backend
(`xpytorch`/`torch_xmlir`), the operator libraries (`kunlun_ops`,
`xspeedgate_ops`, `deep_ep`, `xmooncake`, `xtriton`, `flash_mla`) and this
plugin are all installed from `ARG`-driven download addresses.

The build context is the parent directory of this repository and must also
contain a SGLang source tree, which the image uses through `PYTHONPATH` rather
than installing it:

```bash
# <ctx>/sglang-kunlun/   this repository
# <ctx>/sglang/          SGLang source tree
docker build -f sglang-kunlun/dockerfile/Dockerfile \
  -t iregistry.baidu-int.com/aicapx/sglang-kunlun:dev .
```

Always flatten the result before publishing it, otherwise the clean-up steps
only become whiteout entries and the image does not shrink. See
[`dockerfile/README.md`](dockerfile/README.md) for the flatten command, the
offline/restricted-network build mode (`REUSE_BASE_PY_ENV`), and — most
importantly — the "why it is written this way" notes that list what must **not**
be removed from the image.

## Quick Start

### Configure the Runtime

The following environment is the Kunlun startup configuration for the current
DeepSeek-V4-Flash-0731 example (Attention W8A8, MoE W4A8 INT8 Dynamic). Replace
the paths and model path for your deployment.

```bash
export DSPARK_MODEL_PATH=/home/models/DeepSeek-V4-Flash-0731-W8A8-Dynamic/
LOG_DIR=./logs_dsv4
mkdir -p $LOG_DIR
DATE=$(date +%Y%m%d%H%M)

export HOST_IP=0.0.0.0
port=30000

unset XPU_DUMMY_EVENT
export PYTHONPATH=/workspace/sglang-kunlun:/workspace/sglang/python${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_SET_CPU_AFFINITY=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export XPU_USE_FAST_SWIGLU=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_ENABLE_FAST_FC=0
export XMLIR_CUDNN_ENABLED=1
export CUDA_GRAPH_OPTIMIZE_STREAM=1
export SGLANG_PLATFORM=kunlun
export SGLANG_USE_XPU=1
export SGLANG_ENABLE_SPEC_V2=0
export SGLANG_SET_CPU_AFFINITY=1
export HYDRAX_SKIP_PROTEUS_BOOTSTRAP=1
export SGLANG_TOOL_STRICT_LEVEL=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_HACK_FLASHMLA_BACKEND=kunlun
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
# Multi-stream, avoids the apply_rotary_emb_triton operator
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export USE_FAST_ALLOC_EXTEND_KUNLUN=False

export SGLANG_OPT_USE_JIT_NORM=True
export SGLANG_OPT_USE_FUSED_STORE_CACHE=0
export SGLANG_OPT_FP8_WO_A_GEMM=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO="torch"
export SGLANG_TOPK_TRANSFORM_512_TORCH=False
export SGLANG_OPT_USE_TOPK_V2=0
# common
export SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_PREP_IN_CUDA_GRAPH=False
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=False
```

### Start the SGLang Server

```bash
nohup \
python3 -u -m sglang.launch_server \
    --model-path $DSPARK_MODEL_PATH \
    --quantization compressed-tensors \
    --port $port    \
    --host $HOST_IP \
    --trust-remote-code \
    --tp 8 \
    --max-running-requests 64 \
    --cuda-graph-max-bs 64 \
    --chunked-prefill-size 16384 \
    --mem-fraction-static 0.6 \
    --context-length 540000  \
    --disable-overlap-schedule \
    --enable-metrics \
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
    --disable-shared-experts-fusion \
    --attention-backend kunlun_compressed \
    --dtype float16 \
    --served-model-name ds-v4 \
    --kv-cache-dtype fp16 \
    --disable-radix-cache \
    --tokenizer-worker-num 4 \
    --speculative-algorithm DSPARK \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --watchdog-timeout 30000 > "$LOG_DIR/log-server-$DATE.log" 2>&1 &

echo "log saved at $LOG_DIR/log-server-$DATE.log"
```

The command above is an example deployment configuration. In particular,
`--model-path`, `--tp`, memory allocation, attention backend, and
speculative-decoding settings must match the model and Kunlun runtime installed
on the target machine.

### Check the OpenAI-Compatible Endpoint

List the model identifier exposed by the running server before sending a chat
request:

```bash
curl http://localhost:30000/v1/models
```

Then send a request using the model identifier returned by that endpoint:

```bash
curl http://localhost:30000/v1/chat/completions \
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
The table below lists the model families and quantization modes currently
supported; it is not a complete compatibility matrix.

| Model | Quantization | Notes |
| --- | --- | --- |
| DeepSeek-V3.2 | `w8a8_int8` | Validate with the target SGLang/Kunlun versions. |
| GLM5, GLM5.1, GLM5.2 | `w8a8_int8` | Validate with the target SGLang/Kunlun versions. |
| GLM5.3 | `w8a8_int8`, `w4a8` | Supports W4A8 in addition to W8A8 INT8. |
| DeepSeek-V4-Flash-0731 | `w8a8_int8` | Current DSPARK launch example (Attention W8A8, MoE W4A8 INT8 Dynamic). |
| MiniMax-M3 | `w8a8_int8` | Validate with the target SGLang/Kunlun versions. |

## Version Matrix

| Component | Version or branch | Source |
| --- | --- | --- |
| `sglang-kunlun` | `0.1.0` | [`pyproject.toml`](pyproject.toml) |
| Development code | `main` | Current repository branch |
| SGLang and Kunlun runtime | Deployment-specific | Must match the current plugin branch |

## Architecture

```text
sglang-kunlun/
├── sglang_kunlun/
│   ├── platform/              # Kunlun device and SGLang platform entry point
│   ├── bootstrap/             # Import-order, runtime, and compatibility shims
│   ├── hooks/
│   │   ├── layers/             # Attention, MoE, linear, rotary, and quantization hooks
│   │   ├── mem_cache/          # Kunlun KV pools and paged allocator
│   │   ├── model_executor/     # Model-runner and KV-cache integration
│   │   ├── models/             # Model-specific hooks, including DeepSeek-V4
│   │   ├── speculative/        # EAGLE and multi-layer EAGLE hooks
│   │   ├── disaggregation/     # Prefill/Decode and KV-transfer hooks
│   │   └── entrypoints/        # HTTP server integration hooks
│   ├── kernels/                # Kunlun kernel shims and operator replacements
│   ├── models/                 # External model package registration
│   └── launch.py               # Explicit Kunlun launcher
├── debug/                      # Debug and alignment helpers
├── dockerfile/                 # Inference image: a single Dockerfile plus notes
├── test/                       # Unit and contract tests
├── third_party/                # SGLang submodule used for reference
├── sitecustomize.py            # Kunlun bootstrap hook loaded from PYTHONPATH
├── pyproject.toml              # Package metadata and SGLang entry points
└── requirements.txt            # Build and test dependencies
```

## Development and Validation

Run the repository tests in an environment with the required Python
dependencies installed:

```bash
python3 -m pytest -q
```

The CI profile in [`ci.yml`](ci.yml) builds the wheel with:

```bash
pip3 install -r requirements.txt && python3 -m build
```

For changes affecting runtime behavior, validate on the target Kunlun
environment with the relevant model, quantization mode, parallelism, and
graph/disaggregation settings. A passing unit-test run alone does not establish
end-to-end inference or performance correctness. Changes under `dockerfile/`
should additionally be validated by building the image, flattening it, and
starting a real service with it.

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

This project is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE)
for the full license text and [`NOTICE`](NOTICE) for copyright attribution and
third-party software notices.
