"""Operators built from the XSpeedGate tree and shipped with the plugin.

Why a separate package: `wrapper/xspeedgate_ops.cpp` registers with
`TORCH_LIBRARY_FRAGMENT_EXPAND`, so a second .so can add operators to
`torch.ops.xspeedgate_ops` while the installed wheel stays loaded. That is what
keeps `sparse_mixed_attn_int8` -- which exists only as a prebuilt binary in the
installed wheel and has no source in the kernel tree -- from being lost. Never
rebuild or reinstall `xspeedgate_ops` itself to add an operator.

The *package* name must differ from `xspeedgate_ops` (otherwise it shadows the
installed wheel and the process ends up with only these ops), while the .so
*file* name must stay `xspeedgate_ops.cpython-*.so`, because that is what
`PyInit_xspeedgate_ops` is called.

The .so is not in git (`.gitignore` ignores `*.so`); it travels to the decode
node because `boot523_1p1d.sh` tars `sglang_kunlun/` from disk. A fresh checkout
has to rebuild it:

    cd /home/zx/code/klx_kernel/XSpeedGate
    python setup.py --ops dsa_write_page_tables build_ext --inplace
    cp single_op/dsa_write_page_tables/xspeedgate_ops/xspeedgate_ops.cpython-310-*.so \\
       <plugin>/sglang_kunlun/extra_ops/
"""
import logging

logger = logging.getLogger(__name__)

loaded = False
registered: list = []


def _schema_names() -> set:
    import torch

    return {
        s.name
        for s in torch._C._jit_get_all_schemas()
        if s.name.startswith("xspeedgate_ops::")
    }


try:  # pragma: no cover - depends on whether the .so was built on this host
    before = _schema_names()
    from . import xspeedgate_ops  # noqa: F401  (the .so next to this file)

    registered = sorted(_schema_names() - before)
    loaded = True
except ImportError as exc:
    logger.warning(
        "extra_ops.loaded=False (%s); the plugin falls back to the torch "
        "implementations. Rebuild with `python setup.py --ops <stem> "
        "build_ext --inplace` in the XSpeedGate tree if that is unexpected.",
        exc,
    )
else:
    logger.info(
        "extra XSpeedGate operators registered: %s", registered or "none (already present)"
    )
