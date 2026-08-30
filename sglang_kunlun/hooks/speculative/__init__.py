"""speculative attention"""
import importlib
import sglang.srt.speculative as upstream_speculative

# HookRegistry resolves dotted targets through parent-module attributes.
upstream_speculative.eagle_worker_v2 = importlib.import_module(
    "sglang.srt.speculative.eagle_worker_v2"
)
# 同理，dspark_precision 挂的是 dspark_components 下的子模块，而
# dspark_components/__init__.py 是空的，不 import 一下 pkgutil.resolve_name 找不到。
_dspark_components = importlib.import_module(
    "sglang.srt.speculative.dspark_components"
)
for _name in ("dspark_worker_v2",):
    setattr(
        _dspark_components,
        _name,
        importlib.import_module(f"sglang.srt.speculative.dspark_components.{_name}"),
    )

from . import draft_utils  # noqa: E402,F401
from . import dspark_draft_backend  # noqa: E402,F401
from . import dspark_precision  # noqa: E402,F401
from . import eagle_worker_v2  # noqa: E402,F401
from . import multi_layer_eagle_worker_v2  # noqa: E402,F401
