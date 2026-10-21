"""Bootstrap helpers that must run before selected SGLang imports.

先绑定 ``_kunlun_pre_shim``，再安装模块路径别名与可选 kernel 桩：
平台插件可能在 ``compat_aliases.install()`` 期间被重新导入，那时必须已经能
拿到 ``_kunlun_pre_shim``。
"""

from .pre_shim import _kunlun_pre_shim

from . import compat_aliases as _compat_aliases

_compat_aliases.install()

from . import optional_kernel_stubs as _optional_stubs

_optional_stubs.install_all()

__all__ = ["_kunlun_pre_shim"]
