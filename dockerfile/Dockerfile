ARG BASE_IMAGE=iregistry.baidu-int.com/hac_test/aiak_sgl_ubuntu2204_base:v0.1
FROM ${BASE_IMAGE}

# 推理引擎类型
ARG engine_type="sglang_src"

# Engine source: "wheel" downloads the reviewed community sglang wheel from
# PyPI; "source" clones SGLANG_REPO at SGLANG_REF/SGLANG_COMMIT instead and
# requires an explicit commit.
ARG SGLANG_ENGINE_SOURCE="wheel"
ARG SGLANG_REPO="https://github.com/sgl-project/sglang.git"
ARG SGLANG_REF="v0.5.21"
ARG SGLANG_COMMIT=""
# Optional HTTP(S) proxy used when SGLANG_ENGINE_SOURCE=source clones GitHub.
ARG GITHUB_PROXY=""
ENV GITHUB_PROXY=${GITHUB_PROXY}

ARG XPYTORCH_VERSION=""

ARG XFLASH_MLA_VERSION=""

ARG XDEEPEP_VERSION=""

ARG KUNLUN_OPS_VERSION=""

ARG MOONCAKE_VERSION=""

ARG SGL_KERNEL_VERSION=""

ARG XTRITON_VERSION=""

ARG RUNTIME_VERSION=""

ARG LIB_TMP_VERSION=""

ARG ATTENTION_STORE_VERSION=""

ARG ds_tool_download_addr=""

ARG performance_tool_download_addr=""

RUN rm -rf /workspace/*
RUN mkdir -p /workspace/

# Build from the local plugin tree; the engine is fetched at build time.
COPY sglang-kunlun /workspace/sglang_kunlun

RUN bash /workspace/sglang_kunlun/dockerfile/install.sh \
    --engine_type=${engine_type} \
    --SGLANG_ENGINE_SOURCE="${SGLANG_ENGINE_SOURCE}" \
    --SGLANG_REPO="${SGLANG_REPO}" \
    --SGLANG_REF="${SGLANG_REF}" \
    --SGLANG_COMMIT="${SGLANG_COMMIT}" \
    --performance_tool_download_addr=${performance_tool_download_addr} \
    --ds_tool_download_addr="${ds_tool_download_addr}" \
    --XPYTORCH_VERSION="${XPYTORCH_VERSION}" \
    --XFLASH_MLA_VERSION="${XFLASH_MLA_VERSION}" \
    --XDEEPEP_VERSION="${XDEEPEP_VERSION}" \
    --KUNLUN_OPS_VERSION="${KUNLUN_OPS_VERSION}" \
    --MOONCAKE_VERSION="${MOONCAKE_VERSION}" \
    --SGL_KERNEL_VERSION="${SGL_KERNEL_VERSION}" \
    --XTRITON_VERSION="${XTRITON_VERSION}" \
    --RUNTIME_VERSION="${RUNTIME_VERSION}" \
    --LIB_TMP_VERSION="${LIB_TMP_VERSION}" \
    --ATTENTION_STORE_VERSION="${ATTENTION_STORE_VERSION}"

WORKDIR /workspace


ENV LD_LIBRARY_PATH=/usr/local/xre/so:/usr/local/xccl/so:/usr/local/lib/:${LD_LIBRARY_PATH}
