# 昆仑芯 SGLang 推理镜像构建

本目录只有一个构建文件 `Dockerfile`。`sync_bos.sh` 与构建无关，是把产出包推到 BOS 的小工具。

## 物料来源

xpytorch、kunlun_ops、xspeedgate_ops、xre、xccl、deep_ep、xmooncake、xtriton、flash_mla 的版本与下载地址以昆仑芯物料文档为准：

<https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/BeQck0ZK7s/wsEROHhNNf/_mpWzFJuSciXhX>

地址全部是 `Dockerfile` 顶部的 `ARG`，升级直接改 `ARG` 默认值或在 `docker build` 时用 `--build-arg` 覆盖。镜像内 `/versions` 记录本次构建实际用到的每个地址。

## 构建

构建上下文是**本仓库的上一级目录**，并且上下文里要同时有 SGLang 源码树：

```
<ctx>/sglang-kunlun/   本仓库
<ctx>/sglang/          SGLang 源码（运行期靠 PYTHONPATH 使用，不做 pip install）
```

```bash
docker build -f sglang-kunlun/dockerfile/Dockerfile \
  -t iregistry.baidu-int.com/aicapx/sglang-kunlun:dev .
```

覆盖物料地址：

```bash
docker build -f sglang-kunlun/dockerfile/Dockerfile \
  --build-arg XPYTORCH_ADDR=<xpytorch .run> \
  --build-arg KUNLUN_OPS_ADDR=<kunlun_ops .whl> \
  --build-arg XSPEEDGATE_OPS_ADDR=<xspeedgate_ops .whl> \
  -t iregistry.baidu-int.com/aicapx/sglang-kunlun:dev .
```

## 压平

构建完必须压平一次再推仓库，否则 `Dockerfile` 里的清理只会变成 whiteout，镜像不会变小。元数据从源镜像的 `Config` 里原样搬，不要手写，避免漏项或写错：

```bash
SRC=sglang-kunlun:dev
DST=iregistry.baidu-int.com/aicapx/sglang-kunlun:dev-flat

mapfile -t ENVS < <(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$SRC" | sed '/^$/d')
ARGS=(); for e in "${ENVS[@]}"; do ARGS+=(-c "ENV $e"); done
ARGS+=(-c 'WORKDIR /workspace' -c 'CMD ["/bin/bash"]')

cid=$(docker create "$SRC" true)
docker export "$cid" | docker import "${ARGS[@]}" - "$DST"
docker rm -f "$cid"
```

发版走 `docker build` + 这一步压平，不要用 `docker commit`。

## 受限网络下怎么构建

`Dockerfile` 是单一线性构建，不带条件分支——镜像内容不随 `--build-arg` 变化，只有下载地址和源会变。外网拉不动时按下面几个 `ARG` 调整，不要改 `Dockerfile` 结构：

- `BASE_IMAGE`：基础镜像先在能联网的机器上 `docker pull` + `docker save`，再 `docker load` 到构建机
- `PIP_INDEX` / `PIP_HOST` / `APT_MIRROR`：指向内网源
- 九个物料 `ARG`（`XRE_ADDR`、`XCCL_ADDR`、`XPYTORCH_ADDR`、`KUNLUN_OPS_ADDR`、`XSPEEDGATE_OPS_ADDR`、`DEEPEP_ADDR`、`XMOONCAKE_ADDR`、`XTRITON_ADDR`、`XFLASH_MLA_ADDR`）：指向内网镜像或构建机上的本地 HTTP 服务

## 运行

启动服务的环境变量与命令行见仓库根 `README.md`，或 DeepSeek-V4-Flash 的准出 SOP：

<https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/vMri-fRViV/G4ag4GvOr4/meg7RU45kTNHtt>

注意 `PYTHONPATH`、`SGLANG_PLATFORM`、`SGLANG_USE_XPU` 由启动脚本 export，镜像里没有固化，原因见 `Dockerfile` 内相应注释。

## 镜像仓库

iregistry.baidu-int.com/aicapx/sglang-kunlun:dev
