#
# Copyright (c) 2026 Baidu Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the sglang-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The build context is the parent directory of this repository and must also
# hold a SGLang source tree, which the image uses through PYTHONPATH:
#
#   <ctx>/sglang-kunlun/   this repository
#   <ctx>/sglang/          SGLang source tree
#
#   docker build -f sglang-kunlun/dockerfile/Dockerfile -t <tag> .
#
# Flatten the image before publishing it (see dockerfile/README.md), and do not
# strip it further without reading the inline notes below -- several seemingly
# unused pieces are load-time hard requirements.

ARG BASE_IMAGE=iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.42
FROM ${BASE_IMAGE}

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# Kunlun artifacts. Keep the versions aligned with the Kunlun release notes:
# https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/BeQck0ZK7s/wsEROHhNNf/_mpWzFJuSciXhX
ARG XRE_ADDR=https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.43.6/peermem/xre-Linux-x86_64-5.0.21.43.6.tar.gz
ARG XCCL_ADDR=https://su.bcebos.com/v1/klx-sdk-release-public/xccl/release/3.1.8.1/xccl_Linux_x86_64_cuda12.tar.gz
ARG XPYTORCH_ADDR=https://klx-sdk-release-public.su.bcebos.com/kunlun2aiak_output/20260629/xpytorch-cp310-torch290-ubuntu2004-x64.run
ARG KUNLUN_OPS_ADDR=https://aihc-private-hcd.bj.bcebos.com/zhouzhengsen/aicapx-1174_dsv4pro/kunlun_ops-0.1.204%2Ba65d931d-cp310-cp310-linux_x86_64.whl
ARG XSPEEDGATE_OPS_ADDR=https://aihc-private-hcd.bj.bcebos.com/xspeedgate_release/torch29/20260812_204558/xspeedgate_ops-1.4.0%2Bcf1c9a0.torch29-cp310-cp310-linux_x86_64.whl
ARG DEEPEP_ADDR=https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/deepep/3.1.8.1/deep_ep-cp310-cp310-linux_x86_64_cuda12.tar.gz
ARG XMOONCAKE_ADDR=https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/xmooncake/20250912_9/output.tar.gz
ARG XTRITON_ADDR=https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/xtriton/20250624/output.tar.gz
ARG XFLASH_MLA_ADDR=https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/flash_mla/20250529/output.tar.gz

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
ARG PIP_INDEX=http://mirrors.baidubce.com/pypi/simple/
ARG PIP_HOST=mirrors.baidubce.com
ARG APT_MIRROR=http://mirrors.baidubce.com/ubuntu/

ENV VENV=/opt/sglang_kunlun \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TERM=xterm-256color \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX} \
    PIP_TRUSTED_HOST=${PIP_HOST}

# Strip what the base image ships for development and keep the minimal runtime
# set. Two things must survive:
#   - /usr/local/cuda-11.7 as an empty directory: the deployment SOP bind-mounts
#     the host's libxpunvidia-ml.so.1 into it, and a missing directory is fatal.
#   - the CUDA toolkit's libcudart.so*, in place: some components dlopen it by
#     absolute path and /etc/ld.so.conf.d registers that directory.
RUN set -eux; \
    for d in /usr/local/cuda-1[0-9]*; do \
        [ -d "$d" ] || continue; \
        keep="$d/targets/x86_64-linux/lib"; \
        rm -rf /tmp/cudart; mkdir -p /tmp/cudart; \
        cp -a "$keep"/libcudart.so* /tmp/cudart/ 2>/dev/null || true; \
        rm -rf "$d"; mkdir -p "$keep"; \
        cp -a /tmp/cudart/libcudart.so* "$keep"/ 2>/dev/null || true; \
        rm -rf /tmp/cudart; \
    done; \
    mkdir -p /usr/local/cuda-11.7; \
    ldconfig; \
    find /usr/lib/locale -mindepth 1 -maxdepth 1 ! -name 'C.UTF-8' -exec rm -rf {} + || true; \
    rm -rf /usr/local/cuda /opt/nvidia /opt/cmake-* /opt/mellanox \
           /usr/lib/jvm /usr/include/boost /usr/lib/llvm-* /usr/lib/debug /usr/lib32 \
           /usr/share/doc /usr/share/man /usr/share/vim /usr/share/mft \
           /root/.tmux /oh-my-tmux /root/.baidu-cx /root/nltk_data \
           /tmp/* /var/tmp/* /var/log/*; \
    printf 'deb %s focal main restricted universe multiverse\ndeb %s focal-updates main restricted universe multiverse\n' \
        "${APT_MIRROR}" "${APT_MIRROR}" > /etc/apt/sources.list; \
    rm -rf /etc/apt/sources.list.d/*; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates tzdata curl wget vim less net-tools lsof pciutils numactl \
        libnuma1 libnuma-dev rdma-core libibverbs1 librdmacm1 ibverbs-providers \
        libgoogle-glog0v5 libarchive13 zlib1g libffi7 libgomp1 gcc g++ make; \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime; echo "${TZ}" > /etc/timezone; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb

RUN sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ~/.bashrc 2>/dev/null; \
    rm -rf /root/miniconda /root/miniconda3 /root/.conda /root/.condarc

# uv only creates the relocatable venv; every install below uses the venv's own
# pip, which honours PIP_INDEX_URL / PIP_NO_CACHE_DIR (uv reads neither).
# --seed is required: the xpytorch installer shells out to pip3.
ENV PATH=${VENV}/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=${VENV}
RUN set -eux; \
    curl -LsSf --retry 5 --retry-delay 5 https://astral.sh/uv/install.sh | sh; \
    uv venv --seed --python 3.10 "${VENV}"; \
    rm -rf /root/.cache/uv

# torch must match the xpytorch build (torch 2.9.0).
RUN set -eux; \
    pip install --index-url "${TORCH_INDEX_URL}" \
        torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0; \
    rm -rf /root/.cache/pip

# SGLang runtime dependencies. Everything SGLang declares in its own
# pyproject [project] dependencies has to be here even when the Kunlun
# inference path never calls it -- those are imported at start-up. The
# training / evaluation / visualisation stack is deliberately left out.
RUN set -eux; \
    pip install \
        transformers==5.3.0 tokenizers==0.22.2 huggingface_hub==1.30.0 hf_transfer hf-xet \
        numpy==1.26.4 scipy==1.15.3 pillow==10.4.0 pandas==2.3.3 pyarrow==22.0.0 \
        scikit-learn==1.7.2 numba==0.62.1 h5py==3.11.0 av==16.0.1 decord2==3.4.0 \
        soundfile==0.13.1 soxr==1.0.0 apache-tvm-ffi==0.1.13.post3 ml_dtypes==0.5.4 \
        torchao==0.9.0 protobuf==6.33.1 compressed-tensors==0.15.0.1 safetensors==0.7.0 \
        xgrammar==0.2.1 llguidance==0.7.30 outlines==0.1.11 outlines_core==0.1.26 \
        interegular==0.3.3 lark==1.2.2 gguf flashinfer-python mistral_common \
        fastapi uvicorn uvloop sse-starlette python-multipart orjson msgspec pyzmq \
        pydantic pydantic-settings psutil setproctitle py-spy prometheus_client \
        requests aiohttp httpx tqdm packaging ninja portalocker einops lxml loguru \
        sentencepiece==0.2.1 tiktoken==0.12.0 partial-json-parser pybase64 xxhash \
        blobfile torch_memory_saver==0.0.9 PyYAML==6.0.2 nvidia-ml-py \
        openai openai-harmony anthropic datasets modelscope timm accelerate \
        ipython easydict distro watchfiles zstandard; \
    rm -rf /root/.cache/pip

# Kunlun runtime: XPU runtime and collective communication library.
RUN set -eux; \
    cd /tmp; \
    curl -fsSL --retry 5 -o xre.tar.gz "${XRE_ADDR}"; \
    tar -xzf xre.tar.gz; \
    mv xre-Linux-x86_64-* /usr/local/xre; \
    cd /usr/local/xre/so && ln -sf libxpucuda.so libcuda.so && ln -sf libxpucuda.so libcuda.so.1; \
    cd /tmp; \
    curl -fsSL --retry 5 -o xccl.tar.gz "${XCCL_ADDR}"; \
    tar -xzf xccl.tar.gz; \
    rm -rf /usr/local/xccl && mv xccl_Linux_x86_64 /usr/local/xccl; \
    rm -rf /tmp/*; \
    { echo "xre=${XRE_ADDR}"; echo "xccl=${XCCL_ADDR}"; } >> /versions

# Kunlun PyTorch backend. The installer also brings in xmlir (torch_xmlir),
# apex, flash_attn, transformer_engine, xformers, proteus, hydrax, torch_xray
# and the xcudart shim libraries.
RUN set -eux; \
    cd /tmp; \
    curl -fsSL --retry 5 -o xpytorch.run "${XPYTORCH_ADDR}"; \
    bash xpytorch.run; \
    rm -rf /tmp/* /root/.cache/pip; \
    echo "xpytorch=${XPYTORCH_ADDR}" >> /versions

# Kunlun operator libraries and KV transfer engine.
RUN set -eux; \
    cd /tmp; \
    for spec in "kunlun_ops=${KUNLUN_OPS_ADDR}" "xspeedgate_ops=${XSPEEDGATE_OPS_ADDR}"; do \
        name="${spec%%=*}"; url="${spec#*=}"; \
        curl -fsSL --retry 5 -o "${name}.whl" "${url}"; \
        pip install --no-deps "${name}.whl"; \
        rm -f "${name}.whl"; \
        echo "${name}=${url}" >> /versions; \
    done; \
    for spec in "deep_ep=${DEEPEP_ADDR}" "xmooncake=${XMOONCAKE_ADDR}" \
                "xtriton=${XTRITON_ADDR}" "flash_mla=${XFLASH_MLA_ADDR}"; do \
        name="${spec%%=*}"; url="${spec#*=}"; \
        curl -fsSL --retry 5 -o pkg.tar.gz "${url}"; \
        mkdir -p unpack && tar -xzf pkg.tar.gz -C unpack; \
        whl="$(find unpack -name '*.whl' | head -n 1)"; test -n "${whl}"; \
        pip install --no-deps "${whl}"; \
        rm -rf unpack pkg.tar.gz; \
        echo "${name}=${url}" >> /versions; \
    done; \
    rm -rf /tmp/* /root/.cache/pip; \
    sp="$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"; \
    for so in libapiinfer.so libinnerinfer.so; do \
        ln -sf "${sp}/kunlun_ops/${so}" "${sp}/torch_xmlir/${so}"; \
    done

# SGLang and this plugin are used from source through PYTHONPATH, so only the
# plugin is installed (for its entry points) and SGLang itself is not.
RUN rm -rf /workspace/* /workspace/.[!.]* 2>/dev/null || true
COPY sglang /workspace/sglang
COPY sglang-kunlun /workspace/sglang-kunlun
RUN set -eux; \
    pip install --no-deps -e /workspace/sglang-kunlun; \
    echo "sglang=$(git -C /workspace/sglang rev-parse --short HEAD 2>/dev/null || echo unknown)" >> /versions; \
    echo "sglang_kunlun=$(git -C /workspace/sglang-kunlun rev-parse --short HEAD 2>/dev/null || echo unknown)" >> /versions; \
    rm -rf /root/.cache/pip

# Only LD_LIBRARY_PATH is baked in.
#
# Do NOT add site-packages/nvidia/cuda_runtime/lib here. torch_xmlir's
# _XMLIRC.so needs libcudart.so.12, and the symbol it wants
# (cudaHostPointerGetAttributes) only exists in the Kunlun shim
# ${VENV}/xcudart/lib/libcudart.so.12.*.kunlun, which xpytorch_import_hook.py
# preloads. Neither the real CUDA toolkit nor the nvidia-cuda-runtime wheel
# exports it, so putting a same-soname NVIDIA library on the search path
# shadows the shim and breaks the import.
#
# Do NOT bake in PYTHONPATH / SGLANG_PLATFORM / SGLANG_USE_XPU either. This
# repository ships a sitecustomize.py that, once PYTHONPATH points at the repo
# root and either variable is set, runs the transformers compatibility patch and
# _kunlun_pre_shim() on every interpreter start -- including pip and the
# --doctor helpers. The deployment SOP exports them right before launch_server.
ENV LD_LIBRARY_PATH=/usr/local/xre/so:/usr/local/xccl/so:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib64

WORKDIR /workspace

# Keep the JIT caches (tvm-ffi / sglang / triton); drop only package manager ones.
RUN set -eux; \
    find /root/.cache -mindepth 1 -maxdepth 1 \
         ! -name 'tvm-ffi' ! -name 'sglang' ! -name 'triton' -exec rm -rf {} + 2>/dev/null || true; \
    find "${VENV}" -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
    find /workspace -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
    rm -rf /tmp/* /var/tmp/* /var/log/*; \
    du -sxh / 2>/dev/null | tail -1

CMD ["/bin/bash"]


