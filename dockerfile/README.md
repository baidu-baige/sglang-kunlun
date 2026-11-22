# Container image build (SGLang 0.5.21 / Kunlun XPU)

This directory builds the `sglang-kunlun` inference image: it installs the
Kunlun XPU runtime and operator libraries on top of the base image, installs the
community `sglang==0.5.21` engine, applies the DeepSeek-V4.1 EPD engine patches,
and installs this repository as an out-of-tree plugin.

The open-source release ships this build recipe instead of a prebuilt image.

## Layout

| Path | Purpose |
| --- | --- |
| `Dockerfile` | Image definition. |
| `install.sh` | All install logic. |
| `patches/` | Engine patches applied to the installed `sglang` package (DeepSeek-V4.1 EPD, float16 KV cache, language-only + CP guard, prefill weight skip, Triton-3.0 `small_copy` bypass). |
| `xpu_requirements.env` | Pinned Kunlun runtime component versions. |
| `pip.conf` / `sources.list` | Build-time package mirrors. |
| `vendor/requirements.txt` | Python dependencies of the engine wheel. |
| `vendor/sgl_kernel-*.whl` | Prebuilt `sgl-kernel` wheel the runtime needs. |
| `vendor/common_ops*.so` | Prebuilt `common_ops` module copied into the engine source tree when `SGLANG_ENGINE_SOURCE=source`. |

## Build

`COPY sglang-kunlun /workspace/sglang_kunlun` requires the build context to be
the **parent** directory of this checkout:

```bash
# from the directory that contains sglang-kunlun/
docker build -f sglang-kunlun/dockerfile/Dockerfile \
    -t sglang-kunlun:0.5.21 .
```

By default the build downloads the reviewed community `sglang==0.5.21` wheel
and verifies its sha256. To build the engine from source instead, pass an
explicit commit:

```bash
docker build -f sglang-kunlun/dockerfile/Dockerfile \
    --build-arg SGLANG_ENGINE_SOURCE=source \
    --build-arg SGLANG_REF=v0.5.21 \
    --build-arg SGLANG_COMMIT=<commit> \
    -t sglang-kunlun:0.5.21 .
```

The default `BASE_IMAGE` comes from the Kunlun base-image registry; override it
with `--build-arg BASE_IMAGE=<image>` when building against a different base.

The remaining runtime package versions are read from `xpu_requirements.env`;
override them with the matching `--build-arg` when moving to a newer runtime.
