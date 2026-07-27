#!/bin/bash
set -eo pipefail

unset http_proxy https_proxy
workspace=`pwd`

# 确定whl包所在的目录
package_dir="$1"
bos_project="$2"
filename=$(basename "$package_dir")

# bos 上传后的目录地址
bos_bucket="/cce-ai-datasets"
bos_pre_dir="hac_test"
bos_project="${bos_project}"
bos_dir="${bos_bucket}/$bos_pre_dir/$bos_project"

# 判断测试与发布模型的配置文件
bos_cmd="/hac_bcecmd/bcecmd --conf-path /hac_bcecmd/develop_conf/"

function download_bcecmd()
{
  if [ ! -f "/hac_bcecmd/bcecmd" ]; then
    wget -qO ${workspace}/bcecmd.tar.gz https://cce-ai-datasets.bj.bcebos.com/hac_test/package-download/bcecmd.tar.gz
    mkdir -p /hac_bcecmd && tar -zxvf ${workspace}/bcecmd.tar.gz -C /hac_bcecmd
  fi
}

function upload_data()
{
  timestamp=$(date +%s)
  date_version="`date "+%Y%m%d"`_${timestamp}"
  ${bos_cmd} bos cp ${package_dir} bos:${bos_dir}/${date_version}/$filename
  ${bos_cmd} bos ls bos:${bos_dir}
}


function main() {
  # 下载bcecmd
  download_bcecmd
  upload_data
}

main $@
bos_addr=https://cce-ai-datasets.bj.bcebos.com/${bos_pre_dir}/${bos_project}/${date_version}/${filename}
echo "final_bos_addr=$bos_addr"