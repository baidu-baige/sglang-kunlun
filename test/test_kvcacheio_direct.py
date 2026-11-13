"""Unit tests for the XSpeedGate-backed sgl_kernel kvcacheio direct ops.

Two layers, per the op integration rules:

* *wiring*: the wrappers forward the upstream keyword-form calls to
  ``torch.ops.xspeedgate_ops.<op>`` with exactly the same arguments, and
  ``install_into_stub()`` makes the stub namespace resolve to them. This is
  what pool_host/base.py and pool_host/mla.py actually call, so it must hold
  before any boot.
* *behaviour*: on a device, a page-first host buffer and a layer-first device
  buffer round-trip byte-exactly through backup (all-layer) and load
  (per-layer).
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from sglang_kunlun.bootstrap import sgl_kernel_stub
from sglang_kunlun.kernels.sgl_kernel_kunlun import kvcacheio_direct


class _Recorder:
    """Stands in for ``torch.ops.xspeedgate_ops``; records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return call


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(kvcacheio_direct, "_ops", lambda: rec)
    return rec


def _t(value, dtype=torch.int64):
    return torch.tensor([value], dtype=dtype)


class TestWiring:
    def test_all_layer_direct_lf_pf_forwards_upstream_kwargs(self, recorder):
        src = [_t(1)]
        dst = [_t(2)]
        kvcacheio_direct.transfer_kv_all_layer_direct_lf_pf(
            src_ptrs=src,
            dst_ptrs=dst,
            src_indices=_t(3),
            dst_indices=_t(4),
            page_size=1,
        )
        assert recorder.calls == [
            ("transfer_kv_all_layer_direct_lf_pf", (src, dst, _t(3), _t(4), 1), {})
        ] or (
            len(recorder.calls) == 1
            and recorder.calls[0][0] == "transfer_kv_all_layer_direct_lf_pf"
            and recorder.calls[0][1] == (src, dst, _t(3), _t(4), 1)
        )

    def test_per_layer_direct_pf_lf_keeps_layer_id(self, recorder):
        kvcacheio_direct.transfer_kv_per_layer_direct_pf_lf(
            src_ptrs=[_t(1)],
            dst_ptrs=[_t(2)],
            src_indices=_t(3),
            dst_indices=_t(4),
            layer_id=7,
            page_size=1,
        )
        name, args, _ = recorder.calls[0]
        assert name == "transfer_kv_per_layer_direct_pf_lf"
        assert args[4] == 7 and args[5] == 1

    def test_install_into_stub_replaces_the_raising_stubs(self):
        """After install, the dotted lookups upstream uses must not raise."""
        if sys.modules.get("sgl_kernel") is None:
            sgl_kernel_stub.install()
        kvcacheio_direct.install_into_stub()
        for name in kvcacheio_direct.DIRECT_OPS:
            fn = sys.modules["sgl_kernel.kvcacheio"].__dict__.get(name)
            assert callable(fn) and fn is not None, name
            # A plain stub attribute would be a _StubModule and raise on call.
            assert not isinstance(fn, sgl_kernel_stub._StubModule), name


class TestBehaviour:
    def test_round_trip_byte_exact(self):
        """Backup D2H then one-layer load H2D return the same bytes.

        Shapes mirror upstream's allocation exactly: the device buffer is
        [pages, page_size, 1, kv_dim] per layer and the host page-first buffer
        is [pages, layers, 1, page_size * kv_dim]; the call passes page_size=1
        and page-row indices, which is what pool_host/base.py does.
        """
        if not torch.cuda.is_available():
            pytest.skip("no XPU device available")
        if "xspeedgate_ops" not in sys.modules:
            import xspeedgate_ops  # noqa: F401

        pages, layers, kv_dim = 4, 3, 8
        # Device, layer-first: one [pages, 1, 1, kv_dim] buffer per layer.
        dev = torch.randn(layers, pages, 1, 1, kv_dim, device="cuda")
        # Host, page-first: [pages, layers, 1, kv_dim] (page_size collapsed).
        host = torch.zeros(pages, layers, 1, kv_dim, dtype=torch.float32)
        rows = torch.arange(pages, dtype=torch.int64)

        kvcacheio_direct.transfer_kv_all_layer_direct_lf_pf(
            src_ptrs=[dev[l] for l in range(layers)],
            dst_ptrs=[host],
            src_indices=rows,
            dst_indices=rows,
            page_size=1,
        )
        torch.cuda.synchronize()
        # host[p, l] must hold dev[l][p]
        expect_host = dev.permute(1, 0, 2, 3, 4).reshape(pages, layers, 1, kv_dim).cpu()
        assert torch.equal(host, expect_host)

        dev.zero_()
        kvcacheio_direct.transfer_kv_per_layer_direct_pf_lf(
            src_ptrs=[host],
            dst_ptrs=[dev[1]],
            src_indices=rows,
            dst_indices=rows,
            layer_id=1,
            page_size=1,
        )
        torch.cuda.synchronize()
        assert torch.equal(
            dev[1].cpu().reshape(pages, 1, 1, kv_dim),
            host[:, 1].reshape(pages, 1, 1, kv_dim),
        )
        assert torch.count_nonzero(dev[0]) == 0
