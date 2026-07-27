# 昆仑芯 SGLang 推理镜像构建

本目录用于构建昆仑 XPU 版本的 `sglang-kunlun` 推理镜像：在昆仑芯官方基础镜像上装好 XPU 运行时与算子库，并以 editable 方式安装本仓库的 SGLang 插件。

## 物料来源

昆仑芯的主要物料产出（xpytorch、kunlun_ops、xre、xccl、deep_ep、xmooncake 等组件的版本与下载地址）来自如下文档。升级版本前请以该文档为准核对：

<https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/BeQck0ZK7s/wsEROHhNNf/_mpWzFJuSciXhX>

## 目录文件

- `Dockerfile`：镜像定义。基础镜像 `iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.42`，将仓库拷贝到 `/workspace/sglang_kunlun` 后执行 `install.sh`。
- `install.sh`：全部安装逻辑，见下文「安装内容」。
- `sources.list` / `pip.conf`：构建期替换的 apt 与 pip 源。
- `enterpoint.sh`：容器内启动 JupyterLab（默认端口 8600，`JUPYTER_TOKEN` 未设置时随机生成）。
- `sync_bos.sh`：把产出包上传到 BOS，用法 `bash sync_bos.sh <package_path> <bos_project>`，结尾输出 `final_bos_addr`。
- `ssh/`：构建期访问内部代码库所需的 ssh 配置。

## 构建

`Dockerfile` 中的 `COPY sglang-kunlun /workspace/sglang_kunlun` 要求构建上下文为**仓库的上一级目录**：

```bash
# 在 sglang-kunlun 的父目录执行
docker build -f sglang-kunlun/dockerfile/Dockerfile -t aicapx_sglang_kunlun:dev .
```

三个物料地址可通过 `--build-arg` 覆盖，不传则使用 `Dockerfile` 中的默认值：

```bash
docker build -f sglang-kunlun/dockerfile/Dockerfile \
  --build-arg XPYTORCH_DOWNLOAD_ADDR=<xpytorch .run 地址> \
  --build-arg KUNLUN_OPS_DOWNLOAD_ADDR=<kunlun_ops .whl 地址> \
  --build-arg XSPEEDGATE_OPS_DOWNLOAD_ADDR=<xspeedgate_ops .whl 地址> \
  -t aicapx_sglang_kunlun:dev .
```

其余组件（xre、xccl、deep_ep、xmooncake、cocopod）的地址目前硬编码在 `install.sh` 的 `install_dep` 中，升级需直接改脚本。

## 安装内容

`install.sh` 在 conda 环境 `python310_torch29_cuda`（Python 3.10 + torch 2.9）中依次执行：

1. `install_base_env`：替换 apt/pip 源，设置时区 Asia/Shanghai，安装 net-tools、lsof、libarchive-dev、pwgen 等基础包。
2. `install_sight`：安装 Nsight Systems CLI。
3. `install_dep xpytorch`：执行 xpytorch `.run` 安装包。
4. `install_dep kunlun_ops`：安装 kunlun_ops whl。
5. `install_dep xspeedgate_ops`：安装 xspeedgate_ops 与 cocopod whl。
6. `install_dep runtime`：安装 xre 5.19.0.0 到 `/usr/local/xre`，并把 `libcuda.so` / `libcuda.so.1` 软链到 `libxpucuda.so`。
7. `install_dep lib_tmp`：安装 xccl 3.1.8.1 到 `/usr/local/xccl`，以及 deep_ep 与 `portalocker`。
8. `install_dep xmooncake`：安装 xmooncake whl。
9. `file_soft_chain`：为 mooncake 补 `torch_xmlir` 下的 `libapiinfer.so` / `libinnerinfer.so` 软链，并修正 `libffi.so.7`。
10. `install_sglang_kunlun`：`pip install -r requirements.txt`，`pip install -e . --no-deps`，另装 `compressed_tensors`、`tilelang`。

`flash_mla`、`xsgl_kernel`、`xtriton`、`attentionstore`、performance tool 相关调用当前在脚本末尾被注释，如需启用请取消对应注释并传入版本参数。

## 运行环境说明

- `LD_LIBRARY_PATH` 已包含 `/usr/local/xre/so`、`/usr/local/xccl/so`、`/usr/local/lib/`。
- 镜像内 `/versions` 记录本次构建实际使用的各组件下载地址，排查版本问题时优先查看该文件。
- 启动服务的环境变量与命令行参考仓库根目录 `README.md` 的「启动」章节。
