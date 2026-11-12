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
# The image is flattened inside this Dockerfile (the final `FROM scratch` stage),
# so `docker build` already produces the publishable ~17G image -- no separate
# export/import step. Do not strip it further without reading the inline notes
# below -- several seemingly unused pieces are load-time hard requirements.

ARG BASE_IMAGE=iregistry.baidu-int.com/xmlir/xpytorch:torch29-py312-ubuntu20.04-v0.45
FROM ${BASE_IMAGE} AS build

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# Kunlun artifacts. Keep the versions aligned with the Kunlun release notes:
# https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/BeQck0ZK7s/wsEROHhNNf/_mpWzFJuSciXhX
ARG XRE_ADDR=https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.45.0.0/peermem/xre-Linux-x86_64-5.45.0.0.tar.gz
ARG XCCL_ADDR=https://su.bcebos.com/v1/klx-sdk-release-public/xccl/release/3.1.8.1/xccl_Linux_x86_64_cuda12.tar.gz
ARG XDLL_ADDR=https://aihc-private-hcd.bj.bcebos.com/zhangchi/xdll/xdll_ops-1.0.0%2B8bf033d.torch29-cp312-cp312-linux_x86_64.whl
ARG XPYTORCH_ADDR=https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260930/xpytorch-cp312-torch290-ubuntu2004-x64.run
ARG KUNLUN_OPS_ADDR=https://klx-sdk-release-public.su.bcebos.com/shiguangyong/zhiban/1109/kunlun_ops-0.1.277%2B6eb1aa78-cp312-cp312-linux_x86_64.whl
ARG CHEN_OPS_ADDR=https://aihc-private-hcd.bj.bcebos.com/zhz/chen_ops-0.2.0%2Bflashmla.20261028-cp312-cp312-linux_x86_64.whl
ARG XSPEEDGATE_OPS_ADDR=https://aihc-private-hcd.bj.bcebos.com/zhangchi/xsg/xspeedgate_ops-1.6.0%2B78e4024.torch29-cp312-cp312-linux_x86_64.whl
ARG COCOPOD_ADDR=https://aihc-private-hcd.bj.bcebos.com/zhz/cocopod-1.7.0%2Bf98939a.torch29.dev-cp312-cp312-linux_x86_64.whl
ARG DEEPEP_ADDR=https://klx-sdk-release-public.su.bcebos.com/xccl/release/3.2.8.0/deep_ep-cp312-cp312-linux_x86_64_cuda12.tar.gz
ARG XMOONCAKE_ADDR=https://aihc-private-hcd.bj.bcebos.com/tangshiwen/kvcache/mooncake_transfer_engine-0.3.12.post1-cp312-cp312-manylinux_2_31_x86_64.whl
# hydrax (Hydra): MegaMoE 的融合 MoE 算子。xpytorch 的 .run 里自带一个旧版，
# 这里要用 daily 包覆盖它，见下面单独那一步。
ARG HYDRAX_ADDR=https://klx-sdk-release-public.su.bcebos.com/Hydra/daily/20261105/hydrax_py312_torch290.tar.gz
ARG XTRITON_ADDR=https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/xtriton/20250624/output.tar.gz
ARG XFLASH_MLA_ADDR=https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/flash_mla/20250529/output.tar.gz

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
        libgoogle-glog0v5 libarchive13 zlib1g libffi7 libgomp1 gcc g++ make \
        libhiredis0.14 libjsoncpp1 libyaml-cpp0.6 libxxhash0; \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime; echo "${TZ}" > /etc/timezone; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb

# Salvage conda's libstdc++ before miniconda goes away. mooncake's engine.so is
# a manylinux_2_31 build (GCC 11 toolchain) needing CXXABI_1.3.13 /
# GLIBCXX_3.4.29, while focal's system libstdc++6 (10.5) stops at
# GLIBCXX_3.4.28, so `from mooncake.engine import TransferEngine` fails with
#   ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6:
#               version `GLIBCXX_3.4.29' not found
# which SGLang reports as the misleading "Please install mooncake ...".
#
# The base image's conda env ships libstdc++.so.6.0.34 (GLIBCXX_3.4.34,
# CXXABI_1.3.15) and the previous build of this image had that directory first
# on LD_LIBRARY_PATH -- that is why mooncake worked there and breaks here. Copy
# it to /usr/local/lib, which /etc/ld.so.conf.d registers and the baked
# LD_LIBRARY_PATH puts ahead of /usr/lib/x86_64-linux-gnu. libstdc++ is
# backward compatible, so everything else keeps running against it.
RUN set -eux; \
    so="$(ls -1 /root/miniconda/envs/*/lib/libstdc++.so.6.0.* | sort -V | tail -n 1)"; \
    cp -a "${so}" /usr/local/lib/; \
    ln -sfn "$(basename "${so}")" /usr/local/lib/libstdc++.so.6; \
    ldconfig; \
    strings /usr/local/lib/libstdc++.so.6 | grep -q GLIBCXX_3.4.29; \
    echo "libstdcxx=${so}" >> /versions; \
    sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ~/.bashrc 2>/dev/null; \
    rm -rf /root/miniconda /root/miniconda3 /root/.conda /root/.condarc

# uv only creates the relocatable venv; every install below uses the venv's own
# pip, which honours PIP_INDEX_URL / PIP_NO_CACHE_DIR (uv reads neither).
# --seed is required: the xpytorch installer shells out to pip3.
#
# Both downloads in this step are pinned to github.com. Everything Astral ships
# is served by default through releases.astral.sh, which measures ~60 KB/s
# through the outbound proxy against 2.4-12 MB/s for github.com:
#   - UV_INSTALLER_GITHUB_BASE_URL covers the uv tarball (16.8 MB). The
#     installer's default source list is "releases.astral.sh github.com" and it
#     only falls back to the second entry when the first one *fails*, not when
#     it is slow.
#   - UV_PYTHON_INSTALL_MIRROR covers the managed CPython build that
#     `uv venv --python 3.12` downloads (32.9 MB), because step 23 deleted the
#     base image's miniconda and no 3.12 interpreter is left on the image. uv
#     rewrites that URL through its Astral mirror (UV_ASTRAL_MIRROR_URL,
#     default https://releases.astral.sh) unless the mirror is set explicitly.
# Measured: 20s for the whole step with both pinned, 5-10 min without.
ENV PATH=${VENV}/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=${VENV}
ARG UV_INSTALLER_GITHUB_BASE_URL=https://github.com
ARG UV_PYTHON_INSTALL_MIRROR=https://github.com/astral-sh/python-build-standalone/releases/download
RUN set -eux; \
    curl -LsSf --retry 5 --retry-delay 5 https://astral.sh/uv/install.sh \
        | UV_INSTALLER_GITHUB_BASE_URL="${UV_INSTALLER_GITHUB_BASE_URL}" sh; \
    UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR}" \
        uv venv --seed --python 3.12 "${VENV}"; \
    rm -rf /root/.cache/uv

# torch must match the xpytorch build (torch 2.9.0).
# torchaudio MUST come from the same cu129 index as torch: the PyPI mirror
# serves a cu128 build, which imports fine but fails torchaudio's
# _check_cuda_version() against torch 2.9.0+cu129 (12.9 != 12.8).
#
# Downloads are resilience-hardened: the nvidia-* dependency wheels run to
# several hundred MB and the outbound proxy drops long transfers, which killed
# an earlier build mid-download. So:
#   - the internal mirror is the ONLY index (it carries every nvidia-* wheel,
#     served fast from inside the intranet);
#   - torch/torchvision/torchaudio are fetched from pinned direct wheel URLs on
#     the PyTorch cu129 index. Passing them as requirements (instead of using
#     --extra-index-url) avoids pip scanning the huge PyTorch index on every
#     resolution, which is what made Step 26 look "stuck" for minutes;
#   - /root/.cache/pip is a buildkit cache mount, so a failed run resumes from
#     the already-downloaded wheels instead of starting over.
ARG TORCH_WHEEL_BASE=https://download.pytorch.org/whl/cu129
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    PIP_NO_CACHE_DIR=0 pip install --retries 10 --resume-retries 20 --timeout 120 \
        --index-url "${PIP_INDEX}" \
        "${TORCH_WHEEL_BASE}/torch-2.9.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
        "${TORCH_WHEEL_BASE}/torchvision-0.24.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
        "${TORCH_WHEEL_BASE}/torchaudio-2.9.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl"; \
    python -c "import torch, torchaudio; assert torch.version.cuda == '12.9'; print('torch', torch.__version__, '| torchaudio', torchaudio.__version__)"

# SGLang runtime dependencies. Everything SGLang declares in its own
# pyproject [project] dependencies has to be here even when the Kunlun
# inference path never calls it -- those are imported at start-up. The
# training / evaluation / visualisation stack is deliberately left out.
#
# Pins follow the working py3.12 env (python312_torch29_cuda), not the old
# py3.10 image: tokenizers/safetensors/huggingface_hub must match what
# transformers 5.17.0 expects, and the rest are the versions actually verified
# to run with torch 2.9.0+cu129. compressed-tensors stays at 0.15.0.1 on
# purpose -- 0.19.0 declares torch>=2.10 and would drag pip into upgrading the
# Kunlun torch build.
#
# flashinfer-python is pinned to 0.6.15.post1, the version the previous image
# build installed (dockerfile_old/install.sh: `pip install
# flashinfer-python==0.6.15.post1 --no-deps`) and therefore the only one this
# stack has actually run on. The SGLang tree declares a newer one
# (sglang/python/pyproject.toml:42, "flashinfer_python[cu13]==0.6.18"); switch to
# it only after re-verifying, and never with the [cu13] extra, which pulls the
# nvidia-*-cu13 wheels and collides with this image's cu129 torch build.
# Leaving the version unpinned installed 0.7.2, where `import flashinfer` probes
# the CuTe DSL through importlib.util.find_spec("cutlass") and blew up on the
# plugin's cutlass stub -- see the ModuleSpec note in
# sglang_kunlun/bootstrap/pre_shim.py::_install_import_stub.
#
# NOTE: hyperparameter is deliberately NOT in this list. torch_xmlir needs it
# (`from hyperparameter import param_scope`) but adding it here would change
# this RUN and invalidate the ~4h cached dependency layer. It is installed
# right before the xpytorch installer instead (see the next-but-one step),
# because that installer runs `XFLAGS --reset` at the end.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    PIP_NO_CACHE_DIR=0 pip install --retries 10 --timeout 120 \
        transformers==5.17.0 tokenizers==0.23.3 huggingface_hub==1.33.0 hf_transfer hf-xet \
        numpy==1.26.4 scipy==1.17.1 pillow==10.4.0 pandas==2.3.3 pyarrow==23.0.1 \
        scikit-learn==1.8.0 numba==0.64.0 h5py==3.11.0 av==17.0.0 decord2==3.4.0 \
        soundfile==0.13.1 soxr==1.0.0 apache-tvm-ffi==0.1.13.post3 ml_dtypes==0.5.4 \
        torchao==0.9.0 protobuf==6.33.6 compressed-tensors==0.15.0.1 safetensors==0.8.0 \
        xgrammar==0.2.8 llguidance==0.7.30 outlines==0.1.11 outlines_core==0.1.26 \
        interegular==0.3.3 lark==1.2.2 gguf flashinfer-python==0.6.15.post1 mistral_common \
        fastapi uvicorn uvloop sse-starlette python-multipart orjson msgspec pyzmq \
        pydantic pydantic-settings psutil setproctitle py-spy prometheus_client \
        requests aiohttp httpx tqdm packaging ninja portalocker einops lxml loguru \
        sentencepiece==0.2.1 tiktoken==0.12.0 partial-json-parser pybase64 xxhash \
        blobfile torch_memory_saver==0.0.9 PyYAML==6.0.2 nvidia-ml-py \
        openai openai-harmony anthropic datasets modelscope timm accelerate \
        ipython easydict distro watchfiles zstandard

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
    # mooncake's engine.so links libxpucuda.so.1. The launch SOP strips
    # /usr/local/xre/so from LD_LIBRARY_PATH (xre's libxpurt.so.2 is a different
    # generation than torch_xmlir's and segfaults on the first kernel launch) and
    # /etc/ld.so.conf.d does not cover that directory either, so the loader cannot
    # find it there. Keep a copy in /usr/local/lib: it stays on the search path and
    # survives the SOP's bind-mount over /usr/local/xre, which brings the host's
    # xpu tree shipping only libcudart/libxpurt. Note the consequence -- this pins
    # libxpucuda to the XRE_ADDR above even when a newer xre is mounted at runtime.
    cp -a /usr/local/xre/so/libxpucuda.so.*.kunlun /usr/local/lib/; \
    ln -sfn "$(cd /usr/local/lib && ls libxpucuda.so.*.kunlun)" /usr/local/lib/libxpucuda.so.1; \
    ldconfig; \
    rm -rf /tmp/*; \
    { echo "xre=${XRE_ADDR}"; echo "xccl=${XCCL_ADDR}"; } >> /versions

# Kunlun PyTorch backend. The installer also brings in xmlir (torch_xmlir),
# apex, flash_attn, transformer_engine, xformers, proteus, hydrax, torch_xray
# and the xcudart shim libraries.
RUN set -eux; \
    cd /tmp; \
    # torch_xmlir (shipped by the xpytorch .run) imports these at module level;
    # the SGLang deps layer does not provide all of them, and the .run installs
    # its own packages with --no-deps. Install the whole set up front so the
    # installer's final `XFLAGS --reset` cannot fail one module at a time.
    pip install --index-url https://pip.baidu-int.com/simple/ \
        hyperparameter==0.5.6 colorama==0.4.6 matplotlib==3.10.8 openpyxl==3.1.5 \
        prettytable==3.17.0 tabulate==0.9.0 wcwidth==0.6.0 regex==2026.9.29; \
    curl -fsSL --retry 5 -o xpytorch.run "${XPYTORCH_ADDR}"; \
    bash xpytorch.run; \
    rm -rf /tmp/* /root/.cache/pip; \
    echo "xpytorch=${XPYTORCH_ADDR}" >> /versions

# Kunlun operator libraries and KV transfer engine.
RUN set -eux; \
    cd /tmp; \
    # NOTE: XMOONCAKE_ADDR is a *.whl*, not a tarball -- keep it in this
    # URL-install loop. Putting it in the tar loop below makes `tar -xzf`
    # fail with "gzip: stdin has more than one entry".
    for spec in "kunlun_ops=${KUNLUN_OPS_ADDR}" "xspeedgate_ops=${XSPEEDGATE_OPS_ADDR}" \
                "chen_ops=${CHEN_OPS_ADDR}" "cocopod=${COCOPOD_ADDR}" \
                "xdll_ops=${XDLL_ADDR}" "xmooncake=${XMOONCAKE_ADDR}"; do \
        name="${spec%%=*}"; url="${spec#*=}"; \
        # Install straight from the URL: pip derives the real wheel filename.
        # Downloading to "${name}.whl" first makes pip reject the archive as
        # "Invalid wheel filename (wrong number of parts)".
        pip install --no-deps --retries 5 "${url}"; \
        echo "${name}=${url}" >> /versions; \
    done; \
    for spec in "deep_ep=${DEEPEP_ADDR}" \
                "xtriton=${XTRITON_ADDR}" "flash_mla=${XFLASH_MLA_ADDR}"; do \
        name="${spec%%=*}"; url="${spec#*=}"; \
        curl -fsSL --retry 5 -o pkg.tar.gz "${url}"; \
        mkdir -p unpack && tar -xzf pkg.tar.gz -C unpack; \
        whl="$(find unpack -name '*.whl' | head -n 1)"; test -n "${whl}"; \
        # xtriton / flash_mla tarballs ship cp310 wheels (built for the old
        # py3.10 image). The py3.12 venv cannot install them, and the working
        # py312 environment does not have them either -- Kunlun's attention/MLA
        # kernels come from chen_ops / xspeedgate_ops instead. Skip, don't fail.
        case "${whl}" in \
          *cp312*|*py3-none*|*py2.py3*) pip install --no-deps "${whl}"; ;; \
          *) echo "WARN: skip ${whl} (not cp312-compatible)" >&2; ;; \
        esac; \
        rm -rf unpack pkg.tar.gz; \
        echo "${name}=${url}" >> /versions; \
    done; \
    rm -rf /tmp/* /root/.cache/pip; \
    sp="$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"; \
    for so in libapiinfer.so libinnerinfer.so; do \
        ln -sf "${sp}/kunlun_ops/${so}" "${sp}/torch_xmlir/${so}"; \
    done

# hydrax (Hydra) daily：MegaMoE 的融合 MoE 算子（`int8_int4_mega_moe`）。
#
# 必须单独一步、必须 --force-reinstall：上面那个 xpytorch 的 .run 已经装过一个
# hydrax，版本偏旧、没有我们要用的 MegaMoE 入口。普通 `pip install` 看到
# 版本号相同就直接跳过，所以只有 --force-reinstall 才能盖掉。
#
# --no-deps 不能省：--force-reinstall 会连依赖一起强制重装，而 hydrax 声明了
# torch。那会把上面钉死的 torch 2.9.0+cu129（xpytorch 配套的那份）换成 PyPI 的
# 普通构建，整个昆仑后端随之报废。两个参数必须一起给。
#
# 这个 tarball 里只有一个 cp312 的 whl（`hydrax_py312_torch290/hydrax-*.whl`），
# 所以没套上面那个跳过非 cp312 的循环。
RUN set -eux; \
    cd /tmp; \
    curl -fsSL --retry 5 -o hydrax.tar.gz "${HYDRAX_ADDR}"; \
    mkdir -p unpack && tar -xzf hydrax.tar.gz -C unpack; \
    whl="$(find unpack -name 'hydrax-*cp312*.whl' | head -n 1)"; test -n "${whl}"; \
    pip install --no-deps --force-reinstall --retries 5 "${whl}"; \
    python -c "from hydrax.xaccelerator.mega_moe_fusion import int8_int4_mega_moe, get_symm_buffer; import hydrax; print('hydrax', hydrax.__version__)"; \
    rm -rf unpack hydrax.tar.gz /root/.cache/pip; \
    echo "hydrax=${HYDRAX_ADDR}" >> /versions

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
# site-packages/torch_xmlir must stay AHEAD of /usr/local/xccl/so, the way the
# previous image build ordered it (dockerfile_old/Dockerfile:14). Two libraries
# live under both prefixes and the Kunlun components need the torch_xmlir copy:
#   - libbkcl.so: deep_ep_cpp.so wants ras_bridge_mark_deepep(BKCLContext*),
#     which only torch_xmlir's libbkcl exports (XCCL 3.1.8.1 does not). With
#     xccl first, `from deep_ep.buffer_v2 import BufferV2` raises
#     "undefined symbol: _Z22ras_bridge_mark_deepep...", the plugin's
#     try/except leaves Buffer = None, and every decode rank then dies in
#     get_deepep_buffer() with
#     "AttributeError: 'NoneType' object has no attribute
#      get_low_latency_rdma_size_hint".
#   - libxpuapi.so: only torch_xmlir ships it, and xdll_ops links it.
#
# Do NOT reorder this ahead of ${VENV}/xcudart: torch_xmlir carries no
# libcudart, so it cannot shadow the Kunlun shim described below.
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
ENV LD_LIBRARY_PATH=/opt/sglang_kunlun/lib/python3.12/site-packages/torch_xmlir:/usr/local/xre/so:/usr/local/xccl/so:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib64

# tvm_ffi's EnvTensorAllocator needs the torch-c-dlpack addon. Install the
# prebuilt wheel instead of letting tvm_ffi JIT-compile it on first import, the
# way the previous image build did (dockerfile_old/install.sh: `pip install
# torch-c-dlpack-ext --no-deps`, "避免 tvm_ffi 启动时 JIT 编译扩展导致 hang").
#
# tvm_ffi tries `import torch_c_dlpack_ext` first and only falls back to the JIT
# path when that fails, so with this wheel present it never forks a compiler and
# never touches ~/.cache/tvm-ffi. That matters twice over on this image: the JIT
# compile needs CUDA headers, which step 2 deletes along with the toolkit, and a
# fork during `import tvm_ffi` re-enters sitecustomize -> sglang_kunlun ->
# tvm_ffi, which recurses (the launch scripts carry a note about 1191 processes).
ARG TORCH_C_DLPACK_EXT=torch-c-dlpack-ext==0.1.5
RUN set -eux; \
    pip install --no-deps --index-url https://pip.baidu-int.com/simple/ "${TORCH_C_DLPACK_EXT}"; \
    python -c "import torch, torch_c_dlpack_ext; print(torch_c_dlpack_ext.__file__)"; \
    rm -rf /root/.cache/pip; \
    echo "torch_c_dlpack_ext=${TORCH_C_DLPACK_EXT}" >> /versions

WORKDIR /workspace

# Keep the JIT caches (tvm-ffi / sglang / triton); drop only package manager ones.
#
# Also strip two things that only matter during the build:
#   - .git (sglang/.git is 384M): the revision is already recorded in /versions
#     two steps above and nothing at runtime shells out to git;
#   - the second copy of libxdnn_pytorch.so: the xpytorch installer drops the
#     same 230M file both in site-packages/lib/ and in torch_xmlir/, so replace
#     one with a relative symlink instead of shipping it twice;
#   - the two conda lines the base image appends to ~/.bashrc *after* the marked
#     conda block (`. /root/miniconda/...` and `conda activate ...`). Step 23
#     only deletes the marked block, so without this every interactive shell
#     prints "No such file or directory" / "conda: command not found" because
#     miniconda is gone. Done here rather than in Step 23 to keep the layer
#     cache of the expensive pip/xpytorch steps intact.
RUN set -eux; \
    sed -i -e '/\/root\/miniconda/d' -e '/^conda /d' /root/.bashrc 2>/dev/null || true; \
    find /root/.cache -mindepth 1 -maxdepth 1 \
         ! -name 'tvm-ffi' ! -name 'sglang' ! -name 'triton' -exec rm -rf {} + 2>/dev/null || true; \
    find "${VENV}" -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
    find /workspace -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
    rm -rf /workspace/sglang/.git /workspace/sglang-kunlun/.git; \
    sp="$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"; \
    if [ -f "${sp}/lib/libxdnn_pytorch.so" ] && [ -f "${sp}/torch_xmlir/libxdnn_pytorch.so" ]; then \
        ln -sfn ../torch_xmlir/libxdnn_pytorch.so "${sp}/lib/libxdnn_pytorch.so"; \
    fi; \
    rm -rf /tmp/* /var/tmp/* /var/log/*; \
    du -sxh / 2>/dev/null | tail -1

# ── 发布镜像：把构建阶段的可见文件系统收成单层 ──────────────────────────
#
# 等价于发布前手动做 `docker export <容器> | docker import`，但写在 Dockerfile
# 里，一条 docker build 就能出瘦身版：
#   * COPY --from 拷的是构建阶段容器的合并视图，前面 rm -rf 掉的下层文件
#     （base 里的 miniconda 11G、cuda-12.8 8.8G、LLVM/JVM/boost 等，合计约
#     20G）不在视图里，天然不会被拷贝，白障也就进不了镜像 —— 实测
#     43.1G -> ~17G；
#   * 构建阶段（build）本身的 43G 层在构建成功后由 docker builder prune 回收，
#     不影响最终镜像大小。
#
# 跨阶段不继承 ENV/WORKDIR/CMD，下面必须重新声明。尤其 LD_LIBRARY_PATH：
# 漏了会在运行期才报找不到 xre/xccl 的 .so。这份清单要跟上面 ENV 那几处保持一致。
FROM scratch

COPY --from=build / /

ENV VENV=/opt/sglang_kunlun \
    PATH=/opt/sglang_kunlun/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/sglang_kunlun/lib/python3.12/site-packages/torch_xmlir:/usr/local/xre/so:/usr/local/xccl/so:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib64 \
    VIRTUAL_ENV=/opt/sglang_kunlun \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    LANGUAGE=en_US:en \
    TERM=xterm-256color \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=http://mirrors.baidubce.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.baidubce.com

WORKDIR /workspace

CMD ["/bin/bash"]

