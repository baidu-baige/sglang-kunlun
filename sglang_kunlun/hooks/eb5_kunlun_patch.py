# Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""EB5 / EB-Lite 在昆仑上的服务侧适配入口。

**为什么需要它**：EB5 的三个适配模块（`eb5_kunlun_adapt.py` / `eb5_mla_adapt.py` /
`eb5_extra_adapt.py`）放在 :mod:`sglang_kunlun.eb5_adapt` 包内，随插件一起安装；
单测由 pytest 的 `conftest.py` 加载，服务进程则由下面这个插件入口加载。

**怎么挂**：fork 的 `python/pyproject.toml` 已经注册了

    [project.entry-points."sglang.srt.plugins"]
    eb5_internal_runtime_patch = "sglang.srt.internal.models.eb5.runtime_patch:register_eb5_runtime_patch"

本模块在插件里再注册一个同名组的 entry point（见 `pyproject.toml`）：

    eb5_kunlun_patch = "sglang_kunlun.hooks.eb5_kunlun_patch:register_eb5_kunlun_patch"

启动时用 ``SGLANG_PLUGINS`` 白名单把它一起打开（**注意它是白名单，设了就只加载列出的**）：

    export SGLANG_PLUGINS=kunlun,kunlun_hooks,eb5_internal_runtime_patch,eb5_kunlun_patch

适配层默认从已安装的 ``sglang_kunlun.eb5_adapt`` 导入，editable 与 wheel 安装都成立。
``EB5_KUNLUN_ADAPT_DIR`` 仅用于实验：指向一个平铺目录（三个 ``eb5_*_adapt.py``）
时优先从那里加载，方便不改安装包替换适配实现。
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

# 适配层随本包安装（editable / wheel 均可）；这个环境变量只用于实验性覆盖。
ADAPT_DIR_ENV = "EB5_KUNLUN_ADAPT_DIR"


def _load_fork_plugins() -> None:
    """加载 fork 自带、但可能未被插件框架发现的插件。

    fork（`baidu/wenxin/sglang`）注册了三个 ``sglang.srt.plugins`` entry point
    （``eb5_internal_runtime_patch`` / ``internal_hooks`` / ``spec_adaptive``）。
    ``importlib.metadata`` 只能发现**已安装发行版**（dist-info）里的 entry point：
    源码只挂在 PYTHONPATH 上时发现不到，所以这里显式调用作为兜底。

    EB5 依赖其中两个：`internal_hooks`（含 DSV4/EB5 的 `compute_kv_score_bf16`
    等）与 `eb5_internal_runtime_patch`（注册 `eb5_mla_hca` attention backend、
    把通用 arch 改写成 `ErnieMlaHcaForCausalLM`、补 server defaults）。

    如果某个插件已经安装、又在 ``SGLANG_PLUGINS`` 白名单里，``load_plugins()`` 会
    自己加载它，这里跳过以免重复执行（部分 ``register_*`` 没有幂等 guard，例如
    server_args 的 choices 是 ``list.extend``）。``internal_hooks`` 当前不在白名单里，
    所以仍由这里手动加载。
    """
    enabled = {
        item.strip()
        for item in os.environ.get("SGLANG_PLUGINS", "").split(",")
        if item.strip()
    }
    try:
        from importlib.metadata import entry_points

        discoverable = {ep.name for ep in entry_points(group="sglang.srt.plugins")}
    except Exception:  # noqa: BLE001 - 发现失败就全部手动兜底
        discoverable = set()

    def _already_loaded(label: str, mod_path: str) -> bool:
        """兜底判断：fork 的 ``register_*`` 不幂等，重复执行会**重复注册 hook**。

        2026-10-21 实测过一次：``entry_points`` 瞬时报错 → ``discoverable`` 为空
        → 这里又手动跑了一遍 ``register_eb5_runtime_patch``，于是
        ``DSV4PoolConfigurator._get_bytes_per_full_token`` 等目标各挂了 2 个
        AROUND hook —— bytes/token 被重复计费（11,788.8 → 18,878.4），KV pool
        少分配 15 GB、`max_total_num_tokens` 从 3.66M 掉到 2.29M。
        用"它注册过的 hook 目标是否已存在"来判断，与插件加载顺序无关。
        """
        if label != "eb5_internal_runtime_patch":
            return False
        try:
            mod = importlib.import_module(mod_path)
            if getattr(mod, "_kunlun_fallback_done", False):
                return True
        except Exception:  # noqa: BLE001
            pass
        try:
            from sglang.srt.plugins.hook_registry import HookRegistry

            hooks = getattr(HookRegistry, "_hooks", {}) or {}
            return bool(
                hooks.get(
                    "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator."
                    "_build_dsv4_kv_pool"
                )
            )
        except Exception:  # noqa: BLE001
            return False

    for label, mod_path, fn_name in (
        (
            "internal_hooks",
            "sglang.srt.internal.plugins",
            "register_internal_plugins",
        ),
        (
            "eb5_internal_runtime_patch",
            "sglang.srt.internal.models.eb5.runtime_patch",
            "register_eb5_runtime_patch",
        ),
    ):
        if (label in discoverable and label in enabled) or _already_loaded(
            label, mod_path
        ):
            logger.info(
                "eb5_kunlun_patch: fork plugin %s 已由插件框架加载，跳过手动兜底",
                label,
            )
            continue
        try:
            mod = importlib.import_module(mod_path)
            getattr(mod, fn_name)()
            # 本进程内再调 register_eb5_kunlun_patch 时不要重复执行
            setattr(mod, "_kunlun_fallback_done", True)
            logger.warning("eb5_kunlun_patch: loaded fork plugin %s", label)
        except Exception as exc:  # noqa: BLE001 - 单个插件失败不阻断
            logger.warning("eb5_kunlun_patch: fork plugin %s failed: %s", label, exc)


def _import_from_package():
    """从已安装的 ``sglang_kunlun.eb5_adapt`` 导入（editable / wheel 均可）。"""
    try:
        from sglang_kunlun.eb5_adapt import eb5_kunlun_adapt

        return eb5_kunlun_adapt, "sglang_kunlun.eb5_adapt"
    except Exception as exc:  # noqa: BLE001 - 插件加载不能把服务带崩
        logger.warning("eb5_kunlun_patch: cannot import sglang_kunlun.eb5_adapt: %s", exc)
        return None, None


def _load_adapt_module():
    """返回 ``(eb5_kunlun_adapt 模块, 来源描述)``，找不到时返回 ``(None, None)``。

    默认从已安装的 :mod:`sglang_kunlun.eb5_adapt` 导入；设置
    ``EB5_KUNLUN_ADAPT_DIR`` 时优先从该平铺目录导入（实验用）。覆盖目录导入
    失败时回退到包内实现，避免一个过期的环境变量让服务静默丢掉适配层。
    """
    adapt_dir = os.environ.get(ADAPT_DIR_ENV)
    if not adapt_dir:
        return _import_from_package()

    if not os.path.isdir(adapt_dir):
        logger.warning(
            "eb5_kunlun_patch: %s=%s 不是目录；回退到包内适配层",
            ADAPT_DIR_ENV,
            adapt_dir,
        )
        return _import_from_package()
    if adapt_dir not in sys.path:
        sys.path.insert(0, adapt_dir)
    try:
        mod = importlib.import_module("eb5_kunlun_adapt")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "eb5_kunlun_patch: cannot import eb5_kunlun_adapt from %s (%s)；"
            "回退到包内适配层",
            adapt_dir,
            exc,
        )
        return _import_from_package()
    return mod, adapt_dir


def register_eb5_kunlun_patch() -> None:
    """Entry point used by SGLang's plugin loader.

    两步：
    1. 加载 fork 自带、但可能未被框架发现的插件（见 `_load_fork_plugins`）；
    2. 把 EB5 的昆仑适配层（融合算子 → torch 拆解 / KLX 原生算子）应用到当前进程。

    幂等；任何失败只告警，不阻断服务启动。
    """
    if os.environ.get("EB5_KUNLUN_PATCH", "1").lower() in {"0", "false", "no", "off"}:
        logger.info("eb5_kunlun_patch: disabled by EB5_KUNLUN_PATCH")
        return

    _load_fork_plugins()

    mod, origin = _load_adapt_module()
    if mod is None:
        return

    # `eb5_kunlun_adapt.apply()` 内部会依次调用 eb5_mla_adapt / eb5_extra_adapt。
    try:
        patched = mod.apply()
    except Exception as exc:  # noqa: BLE001
        logger.warning("eb5_kunlun_patch: apply() failed: %s", exc, exc_info=True)
        return

    logger.warning(
        "eb5_kunlun_patch: applied %d EB5 Kunlun patches from %s", len(patched), origin
    )
