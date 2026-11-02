# Kunlun P800 SGLang inference image

`Dockerfile.xpu` builds the inference image from `ubuntu:20.04` in 6 stages (os → torch → XPU runtime → Kunlun plugins → sglang layer → cleanup). It does not depend on the Kunlun base image. Every tunable is an `ARG` at the top of the file.

Verified on 2026-10-30 (192.168.32.2): `docker build -f Dockerfile.xpu -t klx-sglang:test .` with no `--build-arg` produces a 13.6 GB image; DeepSeek-V4-Flash-0731 started with the single-node qualification SOP was healthy in 250 s and answered a single request with `--speculative-algorithm DSPARK` and thinking enabled (`reasoning_tokens=97`). Concurrency, long context, function call and PD disaggregation are untested.

## The two repositories

The stack is two **sibling** repositories, neither contains the other:

- `sglang` (`baidu/aicapx/sglang`) — the engine. Only copied to `/workspace/sglang`, **never pip installed**, picked up at runtime through `PYTHONPATH`. Its own `python/pyproject.toml` targets CUDA (`torch==2.11.0`, `flashinfer[cu13]`, `sglang-kernel`) and would replace the Kunlun torch and xtriton, so the image ignores it.
- `sglang-kunlun` (this repo) — the Kunlun plugin, installed with `pip install -e . --no-deps`.

The `PYTHONPATH` order is fixed, plugin before engine, because `sglang_kunlun/bootstrap/pre_shim.py` has to stub `sgl_kernel` and patch triton before sglang is imported:

```bash
export PYTHONPATH=/workspace/sglang-kunlun:/workspace/sglang/python
```

## Build

The build context needs both repositories side by side:

```
<build-context>/
├── Dockerfile.xpu
├── sglang/
└── sglang-kunlun/
```

```bash
docker build -f Dockerfile.xpu -t klx-sglang:test .
```

Seven Kunlun artifact URLs (`XPYTORCH_URL`, `KUNLUN_OPS_URL`, `XRE_URL`, `XCCL_URL`, `DEEPEP_URL`, `MOONCAKE_URL`, `XSPEEDGATE_URL`) have defaults; override them with `--build-arg`. An empty value fails the build at the start of stage 3 / stage 4 instead of at download time.

Behind a restricted network: `uv` bootstraps from `astral.sh` and GitHub, so point `UV_INSTALL_URL` and `UV_PYTHON_INSTALL_MIRROR` at internal mirrors; if Docker Hub is unreachable, set `BASE_IMAGE`.

## Version constraints

Artifact versions and URLs follow the Kunlun release notes ([`_mpWzFJuSciXhX`](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/BeQck0ZK7s/wsEROHhNNf/_mpWzFJuSciXhX)); xspeedgate_ops has its own release log with per-version operator lists ([`kQX1KNQa74ft9E`](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/vMri-fRViV/G4ag4GvOr4/kQX1KNQa74ft9E)).

> The `/versions` file inside a qualified image only logs the first install address and can disagree with what is actually shipped (the SOP image records xspeedgate 1.4.0 but ships 1.6.0). Verify with `pip list`.

**`XPYTORCH_URL` + `KUNLUN_OPS_URL` + `XRE_URL` + `XCCL_URL` are one matched set.** Measured outcomes of mixing them:

- xpytorch `kunlun2jituan/20260806` + kunlun_ops 0.1.227, with XRE 5.0.21.43.6 / 5.19.0.0 / 5.34.0.0 → segfault at `parallel_state.py:359 init_world_group`, before any weight loads
- same, with XRE 5.42.0.0 + XCCL 3.2.8.0 → passes distributed init, then segfaults at `dspark_draft_sampler.py:105` (`exponential_()`) during CUDA graph capture
- xpytorch `kunlun2aiak_output/20260629` + kunlun_ops 0.1.204 + XRE 5.0.21.43.6 + XCCL 3.1.8.1 → works; these are the current defaults

xpytorch is published under two BOS directories, `kunlun2jituan` and `kunlun2aiak_output`. Identical filenames, different packages.

`XSPEEDGATE_URL` must be 1.5.1 or newer — `kernel_ops.py` calls `init_compressed_attn_metadata_v2`, missing in 1.4.0. To check a wheel:

```bash
unzip -q xspeedgate_ops-*.whl -d x && strings x/xspeedgate_ops/*.so | grep -c init_compressed_attn_metadata_v2
```

`PYTHON_VERSION` and `TORCH_VERSION` must match the `cpXXX` / `torchXXX` tags in `XPYTORCH_URL` (`cp310` / `torch290` → `3.10` / `2.9.0`).

## sglang dependencies

Three `ARG` groups, which must stay separate:

- `SGLANG_PIP_PINNED` — with deps, versions locked. `torch` / `torchaudio` / `torchvision` are an ABI-matched triple (`torchaudio 2.11.0` fails with `OSError: Could not load _torchaudio.abi3.so`). Keeping `torch` here stops resolution from replacing it.
- `SGLANG_PIP_PLAIN` — with deps, versions open. Two entries are easy to miss: `ipython` (top-level import in `sglang/utils.py:26`, pulls 9 transitive deps) and `python-multipart` (FastAPI checks it at runtime, so static import scans never find it).
- `SGLANG_PIP_NODEPS` — `--no-deps`, versions locked. `flashinfer-python` must be `0.6.15.post1`: 0.7.x reaches `cute_dsl` and pulls `nvidia-cutlass-dsl 4.8.x`, whose toolkit probe reads the Kunlun libcuda shim as "CUDA 10.2" and aborts on import.

Never add `tilelang` — it drags stock triton in and shadows xtriton, which must be the last wheel installed because `torch==2.9.0` hard-depends on `triton==3.5.0`.

## Runtime notes

- Python lives in `/opt/klx` (uv venv, not conda); `PATH` already points there.
- `PYTHONPATH` / `SGLANG_PLATFORM` / `SGLANG_USE_XPU` are exported by the launch script, not baked into the image — in `ENV` they would make every python process in the container load the `sitecustomize.py` pre-shim.
- `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` is in the image `ENV`. `tvm_ffi` (pulled in by xgrammar / flashinfer) JIT compiles a CUDA dlpack addon on import; there is no nvcc here, so it fails and respawns forever, stalling start-up before any weight loads.
- Launch args: see the repository root `README.md`, or the DeepSeek-V4-Flash SOPs `meg7RU45kTNHtt` (single node) and `0KLAmRAmTbeFQA` (2-node PD).
- The SOP mounts weights as `-v /home/models:/home/models`; adjust the host side per machine (on .2 the weights are in `/ssd3/models`, so use `-v /ssd3/models:/home/models` — the container path and launch command stay unchanged).

## Troubleshooting by crash site

- segfault at `parallel_state.py:359 init_world_group` → the four artifacts above are not a matched set
- segfault at `dspark_draft_sampler.py:105` during graph capture → xpytorch / kunlun_ops are not the qualified pair
- `AttributeError ... 'xspeedgate_ops' object has no attribute` → xspeedgate older than 1.5.1
- process neither exits nor progresses, `ps` gains a `_build_optional_torch_c_dlpack` every few seconds → `TVM_FFI_DISABLE_TORCH_C_DLPACK` is missing
- chained `ModuleNotFoundError` (orjson → IPython → traitlets → ...) → `SGLANG_PIP_PLAIN` was emptied or failed
- `RuntimeError: Form data requires "python-multipart"` → same, that one is a FastAPI runtime check

