"""Declare Kunlun-only env fields that upstream SGLang does not define.

Some Kunlun hooks are gated by env flags that existed in the vendored 0.5.17
tree but are absent from upstream ``sglang.srt.environ.Envs`` (for example
``SGLANG_OPT_USE_FUSED_STORE_CACHE``). Reading them through ``envs.<NAME>``
would raise ``AttributeError`` at runtime, so the plugin registers the missing
fields itself instead of patching the main repo.

Fields are only added when absent, so a future upstream release that declares
one of them keeps its own default.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_installed = False


def install() -> None:
    """Attach missing Kunlun env fields to ``sglang.srt.environ.Envs``."""
    global _installed
    if _installed:
        return

    from sglang.srt.environ import EnvBool, Envs

    # name -> field factory, mirroring the 0.5.17 defaults.
    missing = {
        "SGLANG_OPT_USE_FUSED_STORE_CACHE": lambda: EnvBool(True),
        "SGLANG_TOPK_TRANSFORM_512_TORCH": lambda: EnvBool(False),
    }

    added = []
    for name, factory in missing.items():
        if hasattr(Envs, name):
            continue
        field = factory()
        # ``__set_name__`` only fires for class-body assignment, so set the
        # env var name that ``EnvField.get()`` reads explicitly.
        field.name = name
        setattr(Envs, name, field)
        added.append(name)

    if added:
        logger.info(
            "sglang-kunlun: registered %d compat env fields: %s",
            len(added),
            ", ".join(added),
        )

    # --- sglang.srt.utils 符号漂移兜底 ---
    # 本插件多处写 `from sglang.srt.utils import X`，但本 0.5.23 base 把若干工具函数
    # 挪到了子模块。这里在 HOOK_MODULES 导入前把它们回填到 sglang.srt.utils，避免逐
    # 文件改 import（与上面注册缺失 env 字段同一思路：插件自己兜底，不改主仓）。
    import importlib

    import sglang.srt.utils as _u

    _relocated = {
        "set_weight_attrs": "sglang.srt.layers.utils.common",
        "get_num_new_pages": "sglang.srt.mem_cache.utils",
        # 0.5.23 把 add_prefix 挪到了 sglang.srt.layers.utils.common；绝大多数模型文件
        # 已改用新路径，但个别文件（如 models/ernie55.py）仍 from sglang.srt.utils import
        # add_prefix，缺它会导致该模型 import 失败、架构注册不上。
        "add_prefix": "sglang.srt.layers.utils.common",
        # 同理 make_pp_layers 迁到了 sglang.srt.layers.utils.layer_list。
        "make_pp_layers": "sglang.srt.layers.utils.layer_list",
    }
    backfilled = []
    for _name, _mod in _relocated.items():
        if hasattr(_u, _name):
            continue
        try:
            _src = importlib.import_module(_mod)
            setattr(_u, _name, getattr(_src, _name))
            backfilled.append(_name)
        except Exception:
            logger.warning(
                "sglang-kunlun: could not backfill sglang.srt.utils.%s from %s",
                _name,
                _mod,
            )
    if backfilled:
        logger.info(
            "sglang-kunlun: backfilled %d relocated utils onto sglang.srt.utils: %s",
            len(backfilled),
            ", ".join(backfilled),
        )

    _installed = True
