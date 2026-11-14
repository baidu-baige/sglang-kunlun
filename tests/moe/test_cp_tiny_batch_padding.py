"""Offline check for the micro-batch padding that fixes `deep_ep.cpp:410`.

Two properties have to hold before the padding can go into the dispatch path:

1. every CP rank ends up with at least one row, so the vendor's per-rank token
   counts stay non-negative (the whole point of the change);
2. the repeated rows are row-identical to their originals, so slicing them back
   out after the collective restores exactly the caller's tensor, and the router
   ids they carry are still in range (unlike the existing zero-token pad, which
   sends everything to expert 0).

Run: python3 tests/moe/test_cp_tiny_batch_padding.py
"""

import importlib.util
import pathlib

import torch

# Loaded by path: importing the package pulls in the whole sglang stack, which
# needs a live XPU environment. The helper itself only depends on torch.
_spec = importlib.util.spec_from_file_location(
    "cp_tiny_batch",
    pathlib.Path(__file__).resolve().parents[2]
    / "sglang_kunlun/hooks/layers/moe/cp_tiny_batch.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pad_to_cp, pad_topk_to_cp, unpad = _mod.pad_to_cp, _mod.pad_topk_to_cp, _mod.unpad

CP = 8
HIDDEN = 6144
TOP_K = 8
N_EXPERTS = 256


def case(rows):
    torch.manual_seed(rows)
    hidden = torch.randn(rows, HIDDEN, dtype=torch.float16)
    ids = torch.randint(0, N_EXPERTS, (rows, TOP_K), dtype=torch.int32)
    w = torch.rand(rows, TOP_K, dtype=torch.float32)

    padded, orig = pad_to_cp(hidden, CP)
    ids_p, w_p, orig_ids = pad_topk_to_cp(ids, w, CP)

    assert orig == rows == orig_ids, (orig, rows, orig_ids)
    if rows >= CP:
        assert padded is hidden and ids_p is ids, "large batches must pass through"
        print(f"rows={rows:2d}  pass-through (>= cp_size)")
        return
    assert padded.shape[0] == CP, padded.shape
    assert ids_p.shape[0] == CP and w_p.shape[0] == CP
    assert ids_p.min() >= 0 and ids_p.max() < N_EXPERTS, "topk ids out of range"
    # every row equals one of the originals: the padding is pure repetition
    for r in range(CP):
        src = r % rows
        assert torch.equal(padded[r], hidden[src]), r
        assert torch.equal(ids_p[r], ids[src]), r
    restored = unpad(padded, orig)
    assert torch.equal(restored, hidden), "unpad must return the caller's rows"
    print(f"rows={rows:2d}  padded to {CP}, ids ok, roundtrip exact")


if __name__ == "__main__":
    for r in (1, 2, 3, 7, 8, 16):
        case(r)
    # the failing production shape: 5 sequences, one new token in the chunk
    case(1)
    print("all checks passed")
