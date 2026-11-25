#!/bin/bash

set -exuo pipefail

CURRENT_DIR=$(cd `dirname $0`; pwd)

function usage()
{
	echo -e "bash $0 [args]"
	echo -e ""
    echo -e "Optional Arguments:"
    echo -e "--engine_type=<engine_type>                  engine type, e.g., sglang"
    echo -e "--XPYTORCH_VERSION=<XPYTORCH_VERSION>        XPYTORCH_VERSION"
    echo -e "--XFLASH_MLA_VERSION=<XFLASH_MLA_VERSION>    XFLASH_MLA_VERSION"
    echo -e "--XDEEPEP_VERSION=<XDEEPEP_VERSION>          XDEEPEP_VERSION"
    echo -e "--KUNLUN_OPS_VERSION=<KUNLUN_OPS_VERSION>    KUNLUN_OPS_VERSION"
    echo -e "--MOONCAKE_VERSION=<MOONCAKE_VERSION>        MOONCAKE_VERSION"
    echo -e "--SGL_KERNEL_VERSION=<SGL_KERNEL_VERSION>    SGL_KERNEL_VERSION"
    echo -e "--XTRITON_VERSION=<XTRITON_VERSION>          XTRITON_VERSION"
    echo -e "--ATTENTION_STORE_VERSION=<ATTENTION_STORE_VERSION>  ATTENTION_STORE_VERSION"
    echo -e "--SGLANG_ENGINE_SOURCE=<wheel|source>        default: pinned official wheel (downloaded at build time)"
    echo -e "--SGLANG_REF=<SGLANG_REF>                    SGLANG_REF"
    echo -e "--SGLANG_COMMIT=<SGLANG_COMMIT>              SGLANG_COMMIT, if set, checkout this commit after clone"
    echo -e "--performance_tool_download_addr=<performance_tool_download_addr>    performance_tool_download_addr"
    echo -e ""
    
}

##########################################
# Variables with default values          #
##########################################
engine_type="NA"
SGLANG_REPO="NA"
SGLANG_REF="NA"
SGLANG_COMMIT="NA"
SGLANG_ENGINE_SOURCE="wheel"
SGLANG_EXPECTED_VERSION="0.5.21"
XPYTORCH_VERSION="NA"
XFLASH_MLA_VERSION="NA"
XDEEPEP_VERSION="NA"
KUNLUN_OPS_VERSION="NA"
MOONCAKE_VERSION="NA"
SGL_KERNEL_VERSION="NA"
XTRITON_VERSION="NA"
RUNTIME_VERSION="NA"
LIB_TMP_VERSION="NA"
ATTENTION_STORE_VERSION="NA"
performance_tool_download_addr="NA"
ds_tool_download_addr="NA"

# Parse named arguments.
# Copied from: https://unix.stackexchange.com/a/204927
while [ $# -gt 0 ]; do
    case "$1" in
        --engine_type=*)
            engine_type="${1#*=}"
            ;;
        --SGLANG_REPO=*)
            SGLANG_REPO="${1#*=}"
            ;;
        --SGLANG_REF=*)
            SGLANG_REF="${1#*=}"
            ;;
        --SGLANG_ENGINE_SOURCE=*)
            SGLANG_ENGINE_SOURCE="${1#*=}"
            ;;
        --SGLANG_COMMIT=*)
            SGLANG_COMMIT="${1#*=}"
            ;;
        --performance_tool_download_addr=*)
            performance_tool_download_addr="${1#*=}"
            ;;
        --XPYTORCH_VERSION=*)
            XPYTORCH_VERSION="${1#*=}"
            ;;
        --XFLASH_MLA_VERSION=*)
            XFLASH_MLA_VERSION="${1#*=}"
            ;;
        --XDEEPEP_VERSION=*)
            XDEEPEP_VERSION="${1#*=}"
            ;;
        --KUNLUN_OPS_VERSION=*)
            KUNLUN_OPS_VERSION="${1#*=}"
            ;;
        --MOONCAKE_VERSION=*)
            MOONCAKE_VERSION="${1#*=}"
            ;;
        --SGL_KERNEL_VERSION=*)
            SGL_KERNEL_VERSION="${1#*=}"
            ;;
        --XTRITON_VERSION=*)
            XTRITON_VERSION="${1#*=}"
            ;;
        --ATTENTION_STORE_VERSION=*)
            ATTENTION_STORE_VERSION="${1#*=}"
            ;;
        --RUNTIME_VERSION=*)
            RUNTIME_VERSION="${1#*=}"
            ;;
        --LIB_TMP_VERSION=*)
            LIB_TMP_VERSION="${1#*=}"
            ;;
        --ds_tool_download_addr=*)
            ds_tool_download_addr="${1#*=}"
            ;;
        *)
            printf "***************************\n"
            printf "Error: Invalid argument. $1\n"
            printf "***************************\n"
            usage
            exit 1
    esac
    shift
done

xpu_requirements_dir="/workspace/sglang_kunlun/dockerfile/xpu_requirements.env"
if [[ -f "$xpu_requirements_dir" ]]; then
    set -a
    source "$xpu_requirements_dir"
    set +a
else
    echo "Warning: $xpu_requirements_dir not found, skipping..."
fi

# sglang 引擎来源（社区 github）：优先取 xpu_requirements.env / 命令行参数，缺省回落到内置默认值
DEFAULT_SGLANG_REPO="https://github.com/sgl-project/sglang.git"
DEFAULT_SGLANG_REF="v0.5.21"
if [[ -z "${SGLANG_REPO:-}" || "${SGLANG_REPO}" == "NA" ]]; then
    SGLANG_REPO="${DEFAULT_SGLANG_REPO}"
fi
if [[ -z "${SGLANG_REF:-}" || "${SGLANG_REF}" == "NA" ]]; then
    SGLANG_REF="${DEFAULT_SGLANG_REF}"
fi
############################################################# 基础环境开始 ###########################################################

# 插件仓库（构建上下文），社区引擎源码由 clone_sglang 在构建期 git clone 到 sglang_upstream_dir
plugin_dir="/workspace/sglang_kunlun"
sglang_upstream_dir="/workspace/sglang-upstream"

function install_base_env() {
    rm -rf /etc/apt/sources.list && cp ${plugin_dir}/dockerfile/sources.list /etc/apt/sources.list
    # 设置时区
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    apt-get install net-tools -y
    apt-get install lsof -y
    apt-get install libarchive-dev zlib1g-dev -y
    apt-get install bc tree pwgen nodejs -y
    apt-get install libgoogle-glog-dev -y

    # 安装 cmake
    
    rm -rf /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf /usr/pip.conf
    cp ${plugin_dir}/dockerfile/pip.conf /etc/pip.conf

    # 处理中文乱码
    echo "set fileencodings=utf-8,gbk,utf-16le,cp1252,iso-8859-15,ucs-bom" >> /etc/vim/vimrc
    echo "set termencoding=utf-8" >> /etc/vim/vimrc
    echo "set encoding=utf-8" >> /etc/vim/vimrc
}

function install_sight() {
    # 安装 Nsight Systems（NVIDIA 官方公开下载；2025.1.1 CLI-only amd64）
    wget -qO /workspace/NsightSystems-linux-cli-public-2025.1.1.103-3542797.deb https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/NsightSystems-linux-cli-public-2025.1.1.103-3542797.deb
    dpkg -i /workspace/NsightSystems-linux-cli-public-2025.1.1.103-3542797.deb
    rm -rf /workspace/NsightSystems-linux-cli-public-2025.1.1.103-3542797.deb
}

function fetched_sglang_wheel() {
    local out_dir="${1:?fetched_sglang_wheel requires an output directory}"
    local wheel="${out_dir}/sglang-${SGLANG_EXPECTED_VERSION}-cp310-cp310-manylinux_2_34_x86_64.whl"
    # The official wheel is manylinux_2_34; this image's pip/glibc rejects
    # that tag for installation, but --platform accepts it for download.
    pip3 download --trusted-host mirrors.baidubce.com --no-deps \
        --only-binary=:all: --platform manylinux_2_34_x86_64 \
        --python-version 310 --implementation cp --abi cp310 \
        -d "${out_dir}" "sglang==${SGLANG_EXPECTED_VERSION}" >&2
    printf '%s  %s\n' "0f7202ac852a46e3f478c18cd91448a2277792e532269bb6e2cf987dfe260857" "${wheel}" | sha256sum -c - >&2 || return 1
    printf '%s\n' "${wheel}"
}

function validate_sglang_wheel() {
    python3 -I - "$1" "${SGLANG_EXPECTED_VERSION}" <<'PYCODE'
import email
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    names = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
    if len(names) != 1:
        sys.exit("Expected exactly one wheel METADATA file")
    metadata = email.message_from_bytes(wheel.read(names[0]))
    if metadata["Name"] != "sglang" or metadata["Version"] != sys.argv[2]:
        sys.exit("SGLang wheel does not match the reviewed engine version")
PYCODE
}

function clone_sglang() {
    # 从社区 github 拉取 sglang 引擎源码。
    # Explicit source route only; failure aborts without wheel fallback.
    local repo="${1:-}"
    local ref="${2:-}"
    local commit="${3:-}"
    if [[ -z "${commit}" || "${commit}" == "NA" ]]; then
        echo "Error: source builds require an explicit SGLANG_COMMIT" >&2
        return 1
    fi
    if [[ -z "${repo}" || -z "${ref}" ]]; then
        echo "Error: SGLANG_REPO/SGLANG_REF is empty; cannot clone sglang engine"
        return 1
    fi

    local -a clone_args=(--filter=blob:none)
    if [[ -z "${commit}" || "${commit}" == "NA" ]]; then
        clone_args+=(--branch "${ref}")
    fi

    rm -rf ${sglang_upstream_dir}
    echo "Clone sglang: ${repo} @ ${ref}"
    # Source clone timeout; failure aborts this explicitly requested route.
    local clone_timeout=300
    if ! timeout "${clone_timeout}" git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 \
             clone "${clone_args[@]}" "${repo}" ${sglang_upstream_dir}; then
        echo "Error: failed to clone sglang from github within ${clone_timeout}s"
        rm -rf ${sglang_upstream_dir}
        return 1
    fi

    # 强制 HTTP/1.1：blob 回源大流量传输走 HTTP/2 会报 "curl 16 HTTP2 framing layer"
    git -C ${sglang_upstream_dir} config http.version HTTP/1.1

    if [[ -n "${commit}" && "${commit}" != "NA" ]]; then
        echo "Checking out sglang commit: ${commit}"
        if ! timeout "${clone_timeout}" git -C ${sglang_upstream_dir} checkout ${commit}; then
            echo "Error: failed to checkout sglang commit ${commit} within ${clone_timeout}s"
            return 1
        fi
    fi
    cd /workspace
}

function patch_sglang_disable_rust() {
    # 昆仑镜像无 Rust 工具链；剥离社区版 sglang 的 PyO3 Rust 扩展(sglang.srt.grpc._core)，
    # 以纯 Python 方式构建 wheel（避免 build_rust 报 "can't find Rust compiler"）
    local pyproject="${sglang_upstream_dir}/python/pyproject.toml"
    # 1) 从 build-system.requires 去掉 setuptools-rust
    sed -i -E 's/"setuptools-rust[^"]*",?[[:space:]]*//g' "${pyproject}"
    # 2) 删除 [[tool.setuptools-rust.ext-modules]] 整段（到下一个空行为止）
    sed -i '/\[\[tool.setuptools-rust.ext-modules\]\]/,/^$/d' "${pyproject}"
}

function apply_engine_patches() {
    # Engine-side fixes that are not in the vendored sglang wheel (the wheel is
    # the community upstream build, so its source is not in this repository).
    # Patches live in dockerfile/patches/*.patch as ordinary git diffs against
    # the engine tree and are applied to site-packages with -p2. Each patch is
    # applied in reverse-dry-run first so a re-run is a no-op; a patch that
    # cannot apply cleanly fails the build instead of shipping a half engine.
    local patch_dir="${plugin_dir}/dockerfile/patches"
    [ -d "${patch_dir}" ] || return 0
    local site_packages
    site_packages="$(python3 - <<'PY'
import os
import sglang

print(os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__))))
PY
)"
    local patch_file dry
    for patch_file in "${patch_dir}"/*.patch; do
        [ -f "${patch_file}" ] || continue
        # A forward dry-run decides both cases: it succeeds on an unpatched
        # tree; on an already-patched tree it reports "previously applied" and
        # exits non-zero (--batch never prompts). Anything else is a real
        # context failure and must stop the build.
        if dry="$(patch -p2 -d "${site_packages}" --dry-run --forward --batch \
                < "${patch_file}" 2>&1)"; then
            patch -p2 -d "${site_packages}" --batch < "${patch_file}"
            echo "engine_patch=$(basename "${patch_file}")" >> /versions
            continue
        fi
        if printf '%s' "${dry}" | grep -qE "previously applied|Reversed"; then
            echo "[patch] already applied: $(basename "${patch_file}")"
            continue
        fi
        printf '%s\n' "${dry}" >&2
        echo "[patch] ERROR: $(basename "${patch_file}") does not apply to ${site_packages}" >&2
        exit 1
    done
}

function install_sglang_kunlun_plugin() {
    # 引擎装好后安装 sglang-kunlun OOT 插件
    cd ${plugin_dir}
    pip3 install .
    echo "sglang_kunlun_plugin=${plugin_dir}" >> /versions
    cd /workspace
}

function install_dep() {
    # 用法示例
    # install_dep xpytorch 1.2.3
    # install_dep flash_mla 1.0.0
    # install_dep xtriton 0.9.8

    cd /workspace
    project="$1"
    version="$2"

    # 允许的项目名列表
    allow_projects="xpytorch flash_mla deepep kunlun_ops xmooncake xsgl_kernel xtriton runtime lib_tmp attentionstore"

    # 公共下载前缀
    BASE_URL="https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/${project}/${version}"
    BASE_URL2="https://baidu-kunlun-public.su.bcebos.com/baidu-kunlun-share/${version}"
    if [[ -z "$project" || -z "$version" ]]; then
        echo "Usage: install_dep <project> <version>"
        return 1
    fi

    # 检查 project 是否在允许范围
    found=0
    for p in $allow_projects; do
        if [[ "$p" == "$project" ]]; then
            found=1
            break
        fi
    done
    if [[ $found -eq 0 ]]; then
        echo "Error: project must be one of: $allow_projects"
        return 2
    fi

    if [[ "$project" == "xpytorch" ]]; then
        # # 例子："https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/xpytorch/${XPYTORCH_VERSION}/xpytorch-cp310-torch251-ubuntu2004-x64.run"
        url="https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260806/xpytorch-cp310-torch290-ubuntu2004-x64.run"
        echo "Downloading $url"
        curl -O "$url" || wget "$url"
        chmod +x xpytorch-cp310-torch290-ubuntu2004-x64.run
        bash xpytorch-cp310-torch290-ubuntu2004-x64.run
        rm -rf xpytorch-cp310-torch290-ubuntu2004-x64.run
        echo "${project}_addr=${url}" >> /versions
    elif [[ "$project" == "kunlun_ops" ]]; then
        whl_file="kunlun_ops-0.1.227%2B2b100f96-cp310-cp310-linux_x86_64.whl"
        #url="${BASE_URL2}/${whl_file}"
        url="https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260806/kunlun_ops-0.1.227%2B2b100f96-cp310-cp310-linux_x86_64.whl"
        
        echo "Downloading $url"
        curl -o "$whl_file" "$url" || wget "$url" -O "$whl_file"
        pip3 install "$whl_file" --force-reinstall
        rm "$whl_file"
        echo "${project}_addr=${url}" >> /versions
    elif [[ "$project" == "runtime" ]]; then
        # 下载.43版本xre
        url="https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.43.6/peermem/xre-Linux-x86_64-5.0.21.43.6.tar.gz"
        echo "Downloading $url"
        curl -O "$url" || wget "$url"
        tar -xzvf xre-Linux-x86_64-5.0.21.43.6.tar.gz
        mv xre-Linux-x86_64-5.0.21.43.6 /usr/local/xre
        # 增加软连接
        cd /usr/local/xre/so && ln -s libxpucuda.so libcuda.so && ln -s libxpucuda.so libcuda.so.1 && cd -
        rm -rf xre-Linux-x86_64-5.0.21.43.6*
        echo "${project}_addr=${url}" >> /versions
    elif [[ "$project" == "xsgl_kernel" ]]; then
        whl_file="${plugin_dir}/dockerfile/vendor/sgl_kernel-0.3.21-cp39-abi3-linux_x86_64.whl"
        pip3 install "$whl_file" --force-reinstall
        rm "$whl_file"
    elif [[ "$project" == "lib_tmp" ]]; then
        echo "Add yourself dependencies here"

        # download bkcl
        # url="https://su.bcebos.com/v1/klx-sdk-release-public/xccl/release/3.1.8.1/xccl_Linux_x86_64_cuda12.tar.gz"
        # echo "Downloading $url"
        # curl -O "$url" || wget "$url"
        # tar -xzvf xccl_Linux_x86_64_cuda12.tar.gz
        # rm -rf /usr/local/xccl
        # mv xccl_Linux_x86_64 /usr/local/xccl
        # rm -rf xccl_Linux_x86_64_cuda12.tar.gz
        # echo "${project}_addr=${url}" >> /versions

        # Install deep_ep dependencies
        pip3 install portalocker==3.2.0
        # download xdeepep
        url="https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/deepep/3.1.8.1/deep_ep-cp310-cp310-linux_x86_64_cuda12.tar.gz"
        echo "Downloading $url"
        curl -O "$url" || wget "$url"
        tar -xzvf deep_ep-cp310-cp310-linux_x86_64_cuda12.tar.gz
        whl_file=$(find deep_ep-cp310-cp310-linux_x86_64_cuda12 -name "*.whl" | head -n 1)
        if [[ -z "$whl_file" ]]; then
            echo "No .whl file found in output directory"
            return 3
        fi
        pip3 install "$whl_file" --force-reinstall
        rm -rf deep_ep-cp310-cp310-linux_x86_64*
        echo "${project}_addr=${url}" >> /versions

        # download xdeepgemm
        url="https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/xdeep_gemm/20250818/xdpgm_ubuntu2004_x86_64.tar.gz"
        echo "Downloading $url"
        #curl -O "$url" || wget "$url"
        #tar -xzvf xdpgm_ubuntu2004_x86_64.tar.gz
        #whl_file=$(find xdpgm_ubuntu2004_x86_64 -name "*.whl" | head -n 1)
        #if [[ -z "$whl_file" ]]; then
        #    echo "No .whl file found in output directory"
        #    return 3
        #fi
        #pip3 install "$whl_file" --force-reinstall
        #rm -rf xdpgm_ubuntu2004_x86_64*
        #echo "${project}_addr=${url}" >> /versions

        # Install the delivered XSpeedGate wheel, 1.7.0+cbe2814 (torch29 build).
        # It is the superset of the V1.6 wheel (1.6.0+xsgops20261023f): same fast
        # paths this plugin calls (mqa_logits_sum_heads fp16/round 2,
        # cast_permute01, infer_hc_combine_g4, fp4 index-k quantize/store,
        # engram_gate token kernel) plus the 1.7.0 kernels, and it ships its own
        # cocopod. The torch25 build of the same version is ABI-incompatible with
        # this runtime (undefined symbol torch::autograd::Node::name), so the
        # torch29 directory is mandatory. Do not resolve dependencies: the custom
        # torch/triton runtime must stay put.
        url="https://vllm-ai-models.bj.bcebos.com/aiak_share/20261109/torch29/xspeedgate_ops-1.7.0%2Bcbe2814.torch29-cp310-cp310-linux_x86_64.whl"
        local xsg_dir
        xsg_dir="$(mktemp -d /workspace/xspeedgate.XXXXXX)"
        whl_file="${xsg_dir}/xspeedgate_ops-1.7.0+cbe2814.torch29-cp310-cp310-linux_x86_64.whl"
        wget -O "$whl_file" "$url"
        printf '%s  %s\n' "0ae115bf811c9698b7682c52b8cebca3d7e1b409b11b2c3a9d49a36d55399ad3" "$whl_file" | sha256sum -c -
        pip3 install "$whl_file" --force-reinstall --no-deps
        rm -rf "$xsg_dir"
        echo "xspeedgate_ops_addr=${url}" >> /versions

        # Older delivered wheels shipped xspeedgate_ops only; the plugin still
        # imports cocopod for the torch.ops namespace, so for those wheels
        # cocopod/ is re-extracted from the stock wheel (directory only, no
        # dist-info) so importlib.metadata still reports the delivered version.
        # 1.7.0 ships cocopod itself and this block is a no-op.
        site_pkgs="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
        if [ ! -d "${site_pkgs}/cocopod" ]; then
        stock_url="https://aihc-private-hcd.bj.bcebos.com/xspeedgate_release/torch29/20261015_145018/xspeedgate_ops-1.6.0%2B242f72b.torch29-cp310-cp310-linux_x86_64.whl"
        wget -q -O /tmp/xspeedgate_ops_stock.whl "$stock_url"
        # Pin the stock wheel too: it is only read for cocopod/, but a tampered
        # archive would inject files into site-packages.
        printf '%s  %s\n' "93642ab9baca21859f93fb7058c54774fff7f7576fdab8d6a49811178ce22dad" "/tmp/xspeedgate_ops_stock.whl" | sha256sum -c -
        python3 - /tmp/xspeedgate_ops_stock.whl "$site_pkgs" <<'PY'
import sys
import zipfile

whl, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(whl) as z:
    members = [n for n in z.namelist() if n.startswith("cocopod/")]
    if not members:
        sys.exit("cocopod/ missing from the stock xspeedgate_ops wheel")
    z.extractall(dst, members)
print(f"cocopod: {len(members)} files extracted from the stock wheel")
PY
        rm -f /tmp/xspeedgate_ops_stock.whl
        fi


    elif [[ "$project" == "attentionstore" ]]; then
        whl_file="attentionstore_sdk-${version}-py310-none-linux_x86_64.whl"
        url="https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/as-sdk/${whl_file}"
        echo "Downloading $url"
        curl -O "$url" || wget "$url"
        pip3 install "$whl_file" --force-reinstall
        rm -f "$whl_file"
        echo "${project}_addr=${url}" >> /versions

    else
        # 例子："https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/flash_mla/${XFLASH_MLA_VERSION}/output.tar.gz"
        url="${BASE_URL}/output.tar.gz"
        echo "Downloading $url"
        curl -O "$url" || wget "$url"
        tar -xzvf output.tar.gz
        whl_file=$(find output -name "*.whl" | head -n 1)
        if [[ -z "$whl_file" ]]; then
            echo "No .whl file found in output directory"
            return 3
        fi
        pip3 install "$whl_file" --force-reinstall
        rm -rf output*
        echo "${project}_addr=${url}" >> /versions
    fi
}


function install_sglang_engine() {
    # 先装 sglang 的 Python 依赖（与引擎来源无关，vendored wheel / 源码构建两条路都要）
    # 锁定 xpytorch 的 torch 版本，禁止后续 pip 安装替换/升级 torch，
    # 否则 torch_xmlir/libxpytorch.so 会与被换掉的 libtorch ABI 不匹配 -> import torch 段错误
    TORCH_VER=$(python3 -c "import importlib.metadata as m; print(m.version('torch'))" 2>/dev/null || echo "")
    PIN_ARG=""
    if [ -n "${TORCH_VER}" ]; then
        echo "torch==${TORCH_VER}" > /workspace/torch_pin.txt
        PIN_ARG="-c /workspace/torch_pin.txt"
        echo "pin torch==${TORCH_VER}"
    fi
    local triton_ver
    triton_ver=$(python3 -I -c "import importlib.metadata as m; print(m.version('triton'))" 2>/dev/null || true)
    if [[ -n "${triton_ver}" ]]; then
        [[ -n "${PIN_ARG}" ]] || : > /workspace/torch_pin.txt
        echo "triton==${triton_ver}" >> /workspace/torch_pin.txt
        PIN_ARG="-c /workspace/torch_pin.txt"
    fi
    pip3 install ${PIN_ARG} -r ${plugin_dir}/dockerfile/vendor/requirements.txt
    # 需要 --no-deps 的包放 requirements-no-deps.txt：它们的元数据会要求更新的
    # apache-tvm-ffi，而 XPU 栈锁在 0.1.9（实测 xgrammar 0.2.7 在 0.1.9 上可正常
    # 工作）。当前只有 xgrammar==0.2.7：0.5.21 的 function-call detector 会传
    # parallel_tool_calls，0.2.1 没有该参数，所有 tool-call 请求都会报 TypeError。
    pip3 install --no-deps -r ${plugin_dir}/dockerfile/vendor/requirements-no-deps.txt
    python3 -I - <<'PYCODE'
import inspect

from xgrammar import get_model_structural_tag

if "parallel_tool_calls" not in inspect.signature(get_model_structural_tag).parameters:
    raise SystemExit("xgrammar too old: structural tag lacks parallel_tool_calls")
PYCODE
    # timm / torch_memory_saver / flashinfer 依赖 torch(+torchvision/cuda)，用 --no-deps 单独装，
    # 防止拖入 stock torch/torchvision/cuda 冲掉 xpytorch。
    # flashinfer 本体必须装（插件只补它缺的符号、不提供它），版本对齐社区引擎 pin。
    pip3 install --no-deps timm==1.0.16 torch_memory_saver==0.0.9 flashinfer_python==0.6.12

    # Explicit source builds must match the reviewed wheel version.
    if [[ "${SGLANG_ENGINE_SOURCE}" == "source" ]]; then
        echo "Build sglang from source tree ${sglang_upstream_dir}"
        cd ${sglang_upstream_dir}/python
        rm -rf dist
        python3 -m build --wheel
        local built_wheels=(dist/*.whl)
        [[ "${#built_wheels[@]}" -eq 1 ]] || return 1
        validate_sglang_wheel "${built_wheels[0]}" || return 1
        # --no-deps: 社区版 pyproject 的依赖是 CUDA13 GPU 全家桶(torch==2.11 等)，会冲掉昆仑 xpytorch；
        # 依赖统一由上面的 requirements.txt + install_dep(xpytorch/kunlun_ops…) 提供
        pip3 install --no-deps "${built_wheels[0]}"

        # rm -rf ${sglang_upstream_dir}
        # 拷贝common_ops依赖（源文件由 sglang-kunlun 仓库提供）：
        # 旧版社区 sglang 自带真实的 sgl-kernel/ 目录，需覆盖它的编译产物；
        # 新版社区 sglang 不再 vendor 该目录，插件完全靠 sgl_kernel_stub.py 在运行期打桩替代，目录不存在时跳过拷贝。
        if [[ -d "${sglang_upstream_dir}/sgl-kernel/python/sgl_kernel" ]]; then
            cp ${plugin_dir}/dockerfile/vendor/common_ops.cpython-310-x86_64-linux-gnu.so ${sglang_upstream_dir}/sgl-kernel/python/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so
            rm ${plugin_dir}/dockerfile/vendor/common_ops.cpython-310-x86_64-linux-gnu.so
        else
            echo "sgl-kernel/python/sgl_kernel not found in engine source (community sglang no longer vendors it); skip common_ops copy, relying on sgl_kernel_stub at runtime"
        fi
        # 因上面的目录已经删除，所以后续执行目录需要再存在的目录下执行，不然会遇到莫名其妙的报错
        cd /workspace
        return 0
    fi

    # Default engine is the pinned official wheel, fetched with pip's
    # --platform so the manylinux_2_34 tag is accepted for download only.
    local wheel_dir wheel
    wheel_dir="$(mktemp -d /tmp/sglang-wheel.XXXXXX)"
    wheel="$(fetched_sglang_wheel "${wheel_dir}")" || { rm -rf "${wheel_dir}"; return 1; }
    validate_sglang_wheel "${wheel}" || { rm -rf "${wheel_dir}"; return 1; }
    echo "Install downloaded sglang wheel: ${wheel}"
    if ! pip3 install --no-deps "${wheel}"; then
        # The official 0.5.21 wheel is tagged manylinux_2_34 while this base
        # image's pip/glibc only admits older platform tags. Its payload is
        # pure Python (the only native objects are optional Rust extensions,
        # which the EPD pose avoids), so install the official wheel by
        # extracting it into site-packages instead of rebuilding it.
        echo "pip rejected ${wheel}; extracting the official wheel into site-packages"
        python3 - "${wheel}" <<'PYCODE'
import sys
import sysconfig
import zipfile

site = sysconfig.get_paths()["purelib"]
with zipfile.ZipFile(sys.argv[1]) as wheel:
    wheel.extractall(site)
print("extracted", sys.argv[1], "->", site)
PYCODE
    fi
    python3 -I - "${SGLANG_EXPECTED_VERSION}" <<'PYCODE'
import importlib.metadata
import sys

actual = importlib.metadata.version("sglang")
if actual != sys.argv[1]:
    raise SystemExit(f"Installed sglang version {actual} != {sys.argv[1]}")
print("installed sglang", actual)
PYCODE
    echo "sglang_wheel=${wheel} (downloaded)" >> /versions
    rm -rf "${wheel_dir}"
    cd /workspace
}

function install_performance_tool() {
    local performance_tool_download_addr=$1
    pip3 install ${performance_tool_download_addr}
    echo "performance_tool=${performance_tool_download_addr}" >> /versions
}

function install_ds_tool() {
    local ds_tool_download_addr=$1
    pip3 install ${ds_tool_download_addr}
    echo "ds_tool_download_addr=${ds_tool_download_addr}" >> /versions
}

function file_soft_chain() {
    # mooncake 特殊需求
    ln -s /root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/kunlun_ops/libapiinfer.so /root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch_xmlir/libapiinfer.so
    ln -s /root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/kunlun_ops/libinnerinfer.so /root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/torch_xmlir/libinnerinfer.so
    rm -rf /root/miniconda/envs/python310_torch29_cuda/lib/libffi.so.7
    ln -s /lib/x86_64-linux-gnu/libffi.so.7.1.0 /root/miniconda/envs/python310_torch29_cuda/lib/libffi.so.7
    rm -rf /root/miniconda/envs/python310_torch29_cuda/lib/libffi.so
    ln -s /lib/x86_64-linux-gnu/libffi.so.7.1.0 /root/miniconda/envs/python310_torch29_cuda/lib/libffi.so
}


# 切换到 python310_torch29_cuda conda环境
. /root/miniconda/etc/profile.d/conda.sh
conda env list
conda activate python310_torch29_cuda

install_base_env

if [ "$engine_type" == "sglang" ]; then
    case "${SGLANG_ENGINE_SOURCE}" in
        wheel) : ;;
        source)
            clone_sglang "${SGLANG_REPO}" "${SGLANG_REF}" "${SGLANG_COMMIT}"
            patch_sglang_disable_rust
            ;;
        *) echo "Error: SGLANG_ENGINE_SOURCE must be wheel or source" >&2; exit 1 ;;
    esac
    install_sight
    install_performance_tool ${performance_tool_download_addr}
    install_ds_tool ${ds_tool_download_addr}
    install_dep xpytorch ${XPYTORCH_VERSION}
    install_dep flash_mla ${XFLASH_MLA_VERSION}
    install_dep kunlun_ops ${KUNLUN_OPS_VERSION}
    install_dep xmooncake ${MOONCAKE_VERSION}
    install_dep xsgl_kernel ${SGL_KERNEL_VERSION}
    #install_dep runtime ${RUNTIME_VERSION}
    install_dep lib_tmp ${LIB_TMP_VERSION}
    install_sglang_engine
    install_dep xtriton ${XTRITON_VERSION}
    install_dep attentionstore ${ATTENTION_STORE_VERSION}
    file_soft_chain
    install_sglang_kunlun_plugin
    apply_engine_patches
fi

############################################################# 基础环境结束 ###########################################################
