ARG BASE_IMAGE=iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.42
FROM ${BASE_IMAGE}

# Inference engine type
ARG engine_type="sglang"

# Default: verified vendored Pod engine; source builds require an explicit commit.
ARG SGLANG_ENGINE_SOURCE="wheel"
ARG SGLANG_REPO="https://github.com/sgl-project/sglang.git"
ARG SGLANG_REF="v0.5.21"
ARG SGLANG_COMMIT=""

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

ARG performance_tool_download_addr=""

RUN rm -rf /workspace/*
RUN mkdir -p /workspace/

# Build from the local plugin tree, including the vendored engine wheel.
COPY sglang-kunlun /workspace/sglang_kunlun

RUN bash /workspace/sglang_kunlun/dockerfile/install.sh \
    --engine_type=${engine_type} \
    --SGLANG_ENGINE_SOURCE="${SGLANG_ENGINE_SOURCE}" \
    --SGLANG_REPO="${SGLANG_REPO}" \
    --SGLANG_REF="${SGLANG_REF}" \
    --SGLANG_COMMIT="${SGLANG_COMMIT}" \
    --performance_tool_download_addr=${performance_tool_download_addr} \
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
