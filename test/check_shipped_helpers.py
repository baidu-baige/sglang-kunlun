#!/usr/bin/env python3
"""Correctness gates for the two shipped helpers the host-segment bench targets.

`test_decode_host_segments.py` measures *copies* of the code paths it is
considering, which is what makes it cheap; these call the shipped functions so
an edit cannot drift away from what was measured without failing here.

Covered:

* `_dsa_store_lens_and_cu` -- against the `torch.cat` form it replaced (two
  writes plus `cumsum(out=)` instead of an allocation and a copy);
* `KunlunDSAMetadata.refresh_host_lengths_from_cpu` -- against an independent
  reference derivation, for each of the three forward modes it accepts, at both
  decode operating shapes;
* its layout guard -- a mismatched destination must return False and leave
  every destination untouched, because a partial write would silently mix this
  step's lengths with the previous step's.
"""

from __future__ import annotations

import torch

NEXT_N = 4
TOPK = 2048


class _Mode:
    """The three `ForwardMode` predicates the method branches on."""

    def __init__(self, name: str):
        self._name = name

    def is_decode_or_idle(self) -> bool:
        return self._name == "decode"

    def is_target_verify(self) -> bool:
        return self._name == "verify"

    def is_draft_extend_v2(self) -> bool:
        return self._name == "extend"


def _stub():
    from sglang_kunlun.hooks.layers.attention.kunlun_nsa_backend import (
        KunlunDSAMetadata,
    )

    class _Stub:
        refresh_host_lengths_from_cpu = (
            KunlunDSAMetadata.refresh_host_lengths_from_cpu
        )

    return _Stub


def _reference(seq_cpu: torch.Tensor, mode: str):
    offs = torch.arange(1, NEXT_N + 1, dtype=torch.int32)
    if mode == "decode":
        cache = expanded = seq_cpu
    elif mode == "verify":
        cache = seq_cpu + NEXT_N
        expanded = (seq_cpu[:, None] + offs).reshape(-1)
    else:
        cache = seq_cpu
        expanded = (seq_cpu[:, None] - NEXT_N + offs).clamp(min=0).reshape(-1)
    capped = expanded.clamp(max=TOPK)
    zero = torch.zeros(1, dtype=torch.int32)
    return {
        "cache_seqlens_int32_cpu": cache,
        "cu_seqlens_k_cpu": torch.cat((zero, cache.cumsum(0, dtype=torch.int32))),
        "dsa_seqlens_expanded_cpu": expanded,
        "dsa_cu_seqlens_k_cpu": torch.cat(
            (zero, capped.cumsum(0, dtype=torch.int32))
        ),
    }


def check_real_helpers(device) -> list[str]:
    from sglang_kunlun.kernels.kernel_ops import _dsa_store_lens_and_cu

    out = []
    for bs in (2, 3):
        lengths = torch.randint(1, 5000, (bs,), device=device, dtype=torch.int32)
        got_l = torch.zeros(bs, device=device, dtype=torch.int32)
        got_cu = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        _dsa_store_lens_and_cu(lengths, got_l, got_cu)
        want_cu = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                lengths.cumsum(0, dtype=torch.int32),
            )
        )
        ok = torch.equal(got_l, lengths) and torch.equal(got_cu, want_cu)
        out.append(f"lens/cu shipped helper byte-exact (bs={bs}): {ok}")

    stub_cls = _stub()
    for mode in ("decode", "verify", "extend"):
        for bs in (2, 3):
            rows = bs * NEXT_N
            shapes = {
                "cache_seqlens_int32_cpu": (bs,),
                "cu_seqlens_k_cpu": (bs + 1,),
                "dsa_seqlens_expanded_cpu": (bs,) if mode == "decode" else (rows,),
                "dsa_cu_seqlens_k_cpu": (bs + 1,) if mode == "decode" else (rows + 1,),
            }
            stub = stub_cls()
            for name, shp in shapes.items():
                setattr(stub, name, torch.full(shp, -7, dtype=torch.int32))
            seq_cpu = torch.randint(100, 5000, (bs,), dtype=torch.int32)
            called = stub.refresh_host_lengths_from_cpu(
                seq_cpu, _Mode(mode), NEXT_N, TOPK
            )
            want = _reference(seq_cpu, mode)
            same = called and all(
                torch.equal(getattr(stub, k), v) for k, v in want.items()
            )
            out.append(
                f"host lengths shipped helper matches reference "
                f"({mode}, bs={bs}): {same}"
            )

    # Layout guard: every destination wrong, so nothing may be written.
    stub = stub_cls()
    names = (
        "cache_seqlens_int32_cpu",
        "cu_seqlens_k_cpu",
        "dsa_seqlens_expanded_cpu",
        "dsa_cu_seqlens_k_cpu",
    )
    for name in names:
        setattr(stub, name, torch.zeros((9,), dtype=torch.int32))
    rejected = not stub.refresh_host_lengths_from_cpu(
        torch.randint(1, 100, (2,), dtype=torch.int32), _Mode("verify"), NEXT_N, TOPK
    )
    untouched = all(not getattr(stub, name).any() for name in names)
    out.append(
        f"host lengths rejects a layout mismatch (rejected={rejected}) "
        f"and writes nothing: {untouched}"
    )
    return out
