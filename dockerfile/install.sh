#!/bin/bash

set -exuo pipefail

CURRENT_DIR=$(cd `dirname $0`; pwd)

function usage()
{
	echo -e "bash $0 [args]"
	echo -e ""
    echo -e "Optional Arguments:"
    echo -e "--engine_type=<engine_type>                  engine type, e.g., aiak_sglang"
    echo -e "--XPYTORCH_VERSION=<XPYTORCH_VERSION>        XPYTORCH_VERSION"
    echo -e "--XPYTORCH_DOWNLOAD_ADDR=<XPYTORCH_DOWNLOAD_ADDR>  XPYTORCH_DOWNLOAD_ADDR"
    echo -e "--XFLASH_MLA_VERSION=<XFLASH_MLA_VERSION>    XFLASH_MLA_VERSION"
    echo -e "--XDEEPEP_VERSION=<XDEEPEP_VERSION>          XDEEPEP_VERSION"
    echo -e "--KUNLUN_OPS_VERSION=<KUNLUN_OPS_VERSION>    KUNLUN_OPS_VERSION"
    echo -e "--KUNLUN_OPS_DOWNLOAD_ADDR=<KUNLUN_OPS_DOWNLOAD_ADDR>  KUNLUN_OPS_DOWNLOAD_ADDR"
    echo -e "--MOONCAKE_VERSION=<MOONCAKE_VERSION>        MOONCAKE_VERSION"
    echo -e "--SGL_KERNEL_VERSION=<SGL_KERNEL_VERSION>    SGL_KERNEL_VERSION"
    echo -e "--XSPEEDGATE_OPS_DOWNLOAD_ADDR=<XSPEEDGATE_OPS_DOWNLOAD_ADDR>  XSPEEDGATE_OPS_DOWNLOAD_ADDR"
    echo -e "--ATTENTION_STORE_VERSION=<ATTENTION_STORE_VERSION>  ATTENTION_STORE_VERSION"
    echo -e "--performance_tool_download_addr=<performance_tool_download_addr>    performance_tool_download_addr"
    echo -e "--aiak_ds_tool_download_addr=<aiak_ds_tool_download_addr>    aiak_ds_tool_download_addr"
    echo -e ""
    
}

##########################################
# Variables with default values          #
##########################################
engine_type="NA"
XPYTORCH_VERSION="NA"
XPYTORCH_DOWNLOAD_ADDR="NA"
XFLASH_MLA_VERSION="NA"
XDEEPEP_VERSION="NA"
KUNLUN_OPS_VERSION="NA"
KUNLUN_OPS_DOWNLOAD_ADDR="NA"
MOONCAKE_VERSION="NA"
SGL_KERNEL_VERSION="NA"
RUNTIME_VERSION="NA"
LIB_TMP_VERSION="NA"
XSPEEDGATE_OPS_DOWNLOAD_ADDR="NA"
ATTENTION_STORE_VERSION="NA"
performance_tool_download_addr="NA"
aiak_ds_tool_download_addr="NA"

# Parse named arguments.
# Copied from: https://unix.stackexchange.com/a/204927
while [ $# -gt 0 ]; do
    case "$1" in
        --engine_type=*)
            engine_type="${1#*=}"
            ;;
        --performance_tool_download_addr=*)
            performance_tool_download_addr="${1#*=}"
            ;;
        --XPYTORCH_VERSION=*)
            XPYTORCH_VERSION="${1#*=}"
            ;;
        --XPYTORCH_DOWNLOAD_ADDR=*)
            XPYTORCH_DOWNLOAD_ADDR="${1#*=}"
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
        --KUNLUN_OPS_DOWNLOAD_ADDR=*)
            KUNLUN_OPS_DOWNLOAD_ADDR="${1#*=}"
            ;;
        --MOONCAKE_VERSION=*)
            MOONCAKE_VERSION="${1#*=}"
            ;;
        --SGL_KERNEL_VERSION=*)
            SGL_KERNEL_VERSION="${1#*=}"
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
        --XSPEEDGATE_OPS_DOWNLOAD_ADDR=*)
            XSPEEDGATE_OPS_DOWNLOAD_ADDR="${1#*=}"
            ;;
        --aiak_ds_tool_download_addr=*)
            aiak_ds_tool_download_addr="${1#*=}"
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

xpu_requirements_dir="/workspace/aiak_sglang/dockerfile/xpu_requirements.env"
if [[ -f "$xpu_requirements_dir" ]]; then
    set -a
    source "$xpu_requirements_dir"
    set +a
else
    echo "Warning: $xpu_requirements_dir not found, skipping..."
fi
############################################################# 基础环境开始 ###########################################################

aiak_sglang_dir="/workspace/aiak_sglang"
sglang_kunlun_dir="/workspace/sglang-kunlun"

function install_base_env() {
    rm -rf /etc/apt/sources.list && cp ${sglang_kunlun_dir}/dockerfile/sources.list /etc/apt/sources.list
    # 设置时区
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    apt-get install net-tools -y
    apt-get install lsof -y
    apt-get install libarchive-dev zlib1g-dev -y
    apt-get install bc tree pwgen nodejs -y
    apt-get install libgoogle-glog-dev -y

    # 安装 cmake
    # wget -qO /cmake-3.28.6-linux-x86_64.sh https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak2.0/cmake-3.28.6-linux-x86_64.sh
    # bash /cmake-3.28.6-linux-x86_64.sh --prefix=/usr/local --exclude-subdir && rm -rf /cmake-3.28.6-linux-x86_64.sh
    
    rm -rf /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf /usr/pip.conf
    cp ${sglang_kunlun_dir}/dockerfile/pip.conf /etc/pip.conf

    # 处理中文乱码
    echo "set fileencodings=utf-8,gbk,utf-16le,cp1252,iso-8859-15,ucs-bom" >> /etc/vim/vimrc
    echo "set termencoding=utf-8" >> /etc/vim/vimrc
    echo "set encoding=utf-8" >> /etc/vim/vimrc
}

function install_sight() {
    # 安装 sight
    wget -qO /workspace/NsightSystems-linux-cli-public-2025.1.1.103-3542797.deb https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/NsightSystems-linux-cli-public-2025.1.1.103-3542797.deb
    dpkg -i /workspace/NsightSystems-linux-cli-public-2025.1.1.103-3542797.deb
    rm -rf /workspace/NsightSystems-linux-cli-public-2025.1.1.103-3542797.deb
}

function install_dep() {
    # 用法示例
    # install_dep xpytorch 1.2.3
    # install_dep flash_mla 1.0.0
    # install_dep xtriton 0.9.8

    cd /workspace
    project="$1"

    # 允许的项目名列表
    allow_projects="xpytorch flash_mla deepep kunlun_ops xmooncake xsgl_kernel xtriton runtime lib_tmp attentionstore"

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
        url="${XPYTORCH_DOWNLOAD_ADDR}"
        if [[ -z "$url" || "$url" == "NA" ]]; then
            url="https://klx-sdk-release-public.su.bcebos.com/xpytorch/release/3.6.2.1/xpytorch-cp310-torch290-ubuntu2004-x64.run"
        fi
        run_file="${url##*/}"
        echo "Downloading $url"
        curl -o "$run_file" "$url" || wget "$url" -O "$run_file"
        chmod +x "$run_file"
        bash "$run_file"
        rm -rf "$run_file"
        echo "${project}_addr=${url}" >> /versions
    elif [[ "$project" == "kunlun_ops" ]]; then
        # kunlun_ops
        url="${KUNLUN_OPS_DOWNLOAD_ADDR}"
        if [[ -z "$url" || "$url" == "NA" ]]; then
            url="https://klx-sdk-release-public.su.bcebos.com/kunlun2aiak_output/20260707/kunlun_ops-0.1.205%2B223cef9e-cp310-cp310-linux_x86_64.whl"
        fi
        whl_file="${url##*/}"
        whl_file="${whl_file//%2B/+}"
        echo "Downloading $url"
        curl -o "$whl_file" "$url" || wget "$url" -O "$whl_file"
        pip3 install "$whl_file" --force-reinstall
        rm "$whl_file"
        echo "${project}_addr=${url}" >> /versions
        
        # xspeedgate_ops
        url="${XSPEEDGATE_OPS_DOWNLOAD_ADDR}"
        if [[ -z "$url" || "$url" == "NA" ]]; then
            url="http://aihc-private-hcd.bj.bcebos.com/xspeedgate_release/torch29/20260626_144331/xspeedgate_ops-1.2.0+680bed9.torch29-cp310-cp310-linux_x86_64.whl"
        fi
        whl_file="${url##*/}"
        whl_file="${whl_file//%2B/+}"
        wget -O "$whl_file" "$url"
        pip3 install "$whl_file" --force-reinstall
        rm "$whl_file"
        echo "xspeedgate_ops_addr=${url}" >> /versions

        # # cocopod ops
        # wget -O cocopod-1.3.0+224a318-cp310-cp310-linux_x86_64.whl https://vllm-ai-models.bj.bcebos.com/aiak_share/20260609/torch29/cocopod-1.3.0%2B224a318-cp310-cp310-linux_x86_64.whl
        # pip3 install cocopod-1.3.0+224a318-cp310-cp310-linux_x86_64.whl --force-reinstall
        # rm cocopod-1.3.0+224a318-cp310-cp310-linux_x86_64.whl

    elif [[ "$project" == "runtime" ]]; then
        # 下载.43版本xre
        url="https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.43.6/peermem/xre-Linux-x86_64-5.0.21.43.6.tar.gz"
        # 下载 5.19版本xre
        # url="https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.19.0.0/peermem/xre-Linux-x86_64-5.19.0.0.tar.gz"
        echo "Downloading $url"
        curl -O "$url" || wget "$url"
        tar -xzvf xre-Linux-x86_64-5.0.21.43.6.tar.gz
        mv xre-Linux-x86_64-5.0.21.43.6 /usr/local/xre
        # tar -xzvf xre-Linux-x86_64-5.19.0.0.tar.gz
        # mv xre-Linux-x86_64-5.19.0.0 /usr/local/xre
        # 增加软连接
        cd /usr/local/xre/so && ln -s libxpucuda.so libcuda.so && ln -s libxpucuda.so libcuda.so.1 && cd -
        rm -rf xre-Linux-x86_64-5.0.21.43.6*
        # rm -rf xre-Linux-x86_64-5.19.0.0*
        echo "${project}_addr=${url}" >> /versions

    elif [[ "$project" == "lib_tmp" ]]; then
        echo "Add yourself dependencies here"

        # download bkcl
        url="https://su.bcebos.com/v1/klx-sdk-release-public/xccl/release/3.1.8.1/xccl_Linux_x86_64_cuda12.tar.gz"
        echo "Downloading $url"
        curl -O "$url" || wget "$url"
        tar -xzvf xccl_Linux_x86_64_cuda12.tar.gz
        rm -rf /usr/local/xccl
        mv xccl_Linux_x86_64 /usr/local/xccl
        rm -rf xccl_Linux_x86_64_cuda12.tar.gz
        echo "${project}_addr=${url}" >> /versions

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

     elif [[ "$project" == "xmooncake" ]]; then
        url="https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/xmooncake/20250912_9/output.tar.gz"
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
    
    elif [[ "$project" == "xtriton" ]]; then
        url="https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/xtriton/20250624/output.tar.gz"
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
    
    elif [[ "$project" == "flash_mla" ]]; then
        url="https://su.bcebos.com/v1/klx-sdk-release-public/DS_PD/flash_mla/20250529/output.tar.gz"
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

    elif [[ "$project" == "xsgl_kernel" ]]; then
        url="https://aihc-private-hcd.bj.bcebos.com/LLM/inferenceKit/sgl_kernel-0.3.21-cp39-abi3-linux_x86_64.whl"
        echo "Downloading $url"
        curl -O "$url" || wget "$url"
        whl_file="sgl_kernel-0.3.21-cp39-abi3-linux_x86_64.whl"
        pip3 install "$whl_file" --force-reinstall
        rm -rf sgl_kernel-0.3.21-cp39-abi3*
        echo "${project}_addr=${url}" >> /versions

    else
        echo "Error: project must be one of: $allow_projects"
    fi
}


function install_aiak_sglang() {
    cd ${aiak_sglang_dir}

    # 安装sglang
    pip3 install -r ${aiak_sglang_dir}/requirements.txt
    pip3 install --no-cache-dir compressed_tensors tilelang \
    -i https://pip.baidu-int.com/simple/ --no-deps --trusted-host pip.baidu.com
    cd ${aiak_sglang_dir}/python
    python3 -m build --wheel
    export http_proxy=http://10.63.229.53:8891 && export https_proxy=http://10.63.229.53:8891
    pip3 install dist/*
    unset http_proxy https_proxy

    # rm -rf ${aiak_sglang_dir}
    # 拷贝common_ops依赖 
    cp /workspace/aiak_sglang/common_ops.cpython-310-x86_64-linux-gnu.so /workspace/aiak_sglang/sgl-kernel/python/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so
    rm /workspace/aiak_sglang/common_ops.cpython-310-x86_64-linux-gnu.so
    #/workspace/aiak_sglang/sgl-kernel/python/sgl_kernel/
    # 因上面的目录已经删除，所以后续执行目录需要再存在的目录下执行，不然会遇到莫名其妙的报错
    cd /workspace
}

function install_performance_tool() {
    local performance_tool_download_addr="${1:-}"
    if [[ -z "${performance_tool_download_addr}" ]]; then
        echo "performance_tool_download_addr is empty, skip"
        return 0
    fi
    pip3 install ${performance_tool_download_addr}
    echo "performance_tool=${performance_tool_download_addr}" >> /versions
}

function install_aiak_ds_tool() {
    local aiak_ds_tool_download_addr="${1:-}"
    if [[ -z "${aiak_ds_tool_download_addr}" ]]; then
        echo "aiak_ds_tool_download_addr is empty, skip"
        return 0
    fi
    pip3 install ${aiak_ds_tool_download_addr}
    echo "aiak_ds_tool_download_addr=${aiak_ds_tool_download_addr}" >> /versions
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

function install_sglang_kunlun() {
    cd ${sglang_kunlun_dir}

    # 安装sglang kunlun plugin
    pip3 install -r ${sglang_kunlun_dir}/requirements.txt
    pip3 install -e ${sglang_kunlun_dir} --no-deps
    #pip3 install --no-cache-dir compressed_tensors tilelang \
    #-i https://pip.baidu-int.com/simple/ --no-deps --trusted-host pip.baidu.com
    echo "sglang_kunlun_plugin=${sglang_kunlun_dir}" >> /versions
    cd /workspace
}


# 切换到 python310_torch29_cuda conda环境
. /root/miniconda/etc/profile.d/conda.sh
conda env list
conda activate python310_torch29_cuda

install_base_env
install_sight
install_dep runtime
install_dep lib_tmp
install_dep xpytorch
install_dep flash_mla
install_dep kunlun_ops
install_dep xmooncake
install_dep xtriton
file_soft_chain
 # 框架
install_dep xsgl_kernel
install_aiak_sglang
install_sglang_kunlun

############################################################# 基础环境结束 ###########################################################
