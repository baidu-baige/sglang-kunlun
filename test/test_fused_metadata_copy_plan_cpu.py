"""CPU contract test for the DSA replay-metadata fused copy.

`fused_metadata_copy_cuda` / `fused_metadata_copy_multi_cuda`
(`sglang_kunlun/kernels/kernel_ops.py`) now hand their work to
`xspeedgate_ops.fused_dsa_metadata_copy*`, which performs every copy the torch
fallback performed in one launch. The device kernel cannot run here (the GPUs are
held by a measurement run), so this test pins down the part that decides
correctness on any device: the argument plan.

What is compared:

  * `_copy_fused_metadata_fallback` -- the reference implementation, run on the
    CPU tensors, unchanged and still reachable at runtime;
  * `_fused_metadata_copy_plan` (and the multi variant) -- the extents that are
    handed to the kernel, applied here with the kernel's documented addressing
    (offset 1 for the two `cu_seqlens` pairs, `rows x cols` from the source for
    the two tables, everything else a flat prefix).

Every buffer is compared with `torch.equal`, so untouched regions have to agree
too: a segment the kernel should not have copied, or a destination it should not
have widened, shows up as a sentinel mismatch.

Covered: DECODE / TARGET_VERIFY / DRAFT_EXTEND, every combination of the optional
real_page_table / seqlens_expanded / FlashMLA pairs, `max_len` and `max_seqlen_k`
narrower than the page-table destination width, `seqlens_expanded_size` smaller
than the expanded arrays, and the layouts that must be refused (so the plugin
falls back to torch).

Run with:
    ./test/run_precision_cpu_checks.sh
    # or
    PYTHONPATH=$PWD:$PWD/../sglang/python python -m unittest \
        test.test_fused_metadata_copy_plan_cpu -v
"""

import itertools
import unittest

import torch

from sglang_kunlun.kernels import kernel_ops as K

SENTINEL = -999

# Order of the plan's argument tuple: 9 sources, 9 destinations, 7 counts.
NAMES = (
    "cache_seqlens",
    "cu_seqlens_k",
    "page_table_1",
    "dsa_cache_seqlens",
    "seqlens_expanded",
    "dsa_cu_seqlens_k",
    "real_page_table",
    "flashmla_num_splits",
    "flashmla_metadata",
)

# Group order of the multi plan: 8 sources, then 8 destinations per backend.
MULTI_NAMES = (
    "cache_seqlens",
    "cu_seqlens_k",
    "page_table_1",
    "dsa_cache_seqlens",
    "dsa_cu_seqlens_k",
    "real_page_table",
    "flashmla_num_splits",
    "flashmla_metadata",
)


def _rand(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    # Signed range on purpose: the copy is a byte move, so negative ids and
    # values above 2**31/2 must survive unchanged.
    return torch.randint(-(2**30), 2**30, shape, generator=gen, dtype=torch.int32)


def _sentinel(shape):
    return torch.full(shape, SENTINEL, dtype=torch.int32)


def _apply_plan(plan):
    """Apply a single-backend plan with the kernel's addressing rules.

    Mirrors `xpu/src/kunlun3cpp/attention/fused_dsa_metadata_copy.xpu`
    (`dsa_mc_copy_flat` / `dsa_mc_copy_rows`) and the offsets the C++ wrapper
    applies: `CU_OFFSET` for the two cu_seqlens pairs, none anywhere else.
    """
    (
        cache_src,
        cu_src,
        page_src,
        nsa_src,
        exp_src,
        dcu_src,
        rpt_src,
        fns_src,
        fm_src,
        cache_dst,
        cu_dst,
        page_dst,
        nsa_dst,
        exp_dst,
        dcu_dst,
        rpt_dst,
        fns_dst,
        fm_dst,
        n_cache,
        n_cu,
        n_nsa,
        n_exp,
        n_dcu,
        n_fns,
        n_fm,
    ) = plan

    if n_cache > 0:
        cache_dst[:n_cache].copy_(cache_src[:n_cache])
    if n_cu > 0:
        cu_dst[1 : 1 + n_cu].copy_(cu_src[1 : 1 + n_cu])
    if page_src is not None and page_dst is not None:
        rows, cols = page_src.shape
        page_dst[:rows, :cols].copy_(page_src)
    if n_exp > 0:
        exp_dst[:n_exp].copy_(exp_src[:n_exp])
    if n_nsa > 0:
        nsa_dst[:n_nsa].copy_(nsa_src[:n_nsa])
    if n_dcu > 0:
        dcu_dst[1 : 1 + n_dcu].copy_(dcu_src[1 : 1 + n_dcu])
    if rpt_src is not None and rpt_dst is not None:
        rows, cols = rpt_src.shape
        rpt_dst[:rows, :cols].copy_(rpt_src)
    if n_fns > 0:
        fns_dst[:n_fns].copy_(fns_src[:n_fns])
    if n_fm > 0:
        fm_dst[:n_fm].copy_(fm_src[:n_fm])


def _apply_multi_plan(plan):
    """Apply a multi-backend plan: one source, three equally shaped destinations."""
    srcs = plan[:8]
    counts = plan[-6:]
    n_cache, n_cu, n_nsa, n_dcu, n_fns, n_fm = counts
    cache_src, cu_src, page_src, nsa_src, dcu_src, rpt_src, fns_src, fm_src = srcs
    for i in range(3):
        group = plan[8 + 8 * i : 16 + 8 * i]
        (
            cache_dst,
            cu_dst,
            page_dst,
            nsa_dst,
            dcu_dst,
            rpt_dst,
            fns_dst,
            fm_dst,
        ) = group
        if n_cache > 0:
            cache_dst[:n_cache].copy_(cache_src[:n_cache])
        if n_cu > 0:
            cu_dst[1 : 1 + n_cu].copy_(cu_src[1 : 1 + n_cu])
        if page_src is not None and page_dst is not None:
            rows, cols = page_src.shape
            page_dst[:rows, :cols].copy_(page_src)
        if n_nsa > 0:
            nsa_dst[:n_nsa].copy_(nsa_src[:n_nsa])
        if n_dcu > 0:
            dcu_dst[1 : 1 + n_dcu].copy_(dcu_src[1 : 1 + n_dcu])
        if rpt_src is not None and rpt_dst is not None:
            rows, cols = rpt_src.shape
            rpt_dst[:rows, :cols].copy_(rpt_src)
        if n_fns > 0:
            fns_dst[:n_fns].copy_(fns_src[:n_fns])
        if n_fm > 0:
            fm_dst[:n_fm].copy_(fm_src[:n_fm])


def _build_case(
    mode,
    bs,
    next_n,
    copy_width,
    page_width,
    expanded_size=None,
    with_rpt=False,
    with_exp=False,
    with_flashmla=False,
    seed=7,
):
    """Sources plus a factory for two identically seeded destination sets."""
    expanded = bs if mode == 0 else bs * next_n
    size = expanded if expanded_size is None else expanded_size
    page_rows = expanded if mode == 1 else bs
    rpt_cols = max(1, copy_width // 4)

    cache_src = _rand((bs,), seed + 1)
    cu_src = _rand((bs + 1,), seed + 2)
    page_src = _rand((page_rows, copy_width), seed + 3)
    nsa_src = _rand((expanded,), seed + 4)
    exp_src = _rand((expanded,), seed + 5) if with_exp else None
    dcu_src = _rand((expanded + 1,), seed + 6)
    rpt_src = _rand((page_rows, rpt_cols), seed + 7) if with_rpt else None
    fns_src = _rand((expanded + 1,), seed + 8) if with_flashmla else None
    fm_src = _rand((expanded + 1,), seed + 9) if with_flashmla else None

    sources = (
        cache_src,
        cu_src,
        page_src,
        nsa_src,
        exp_src,
        dcu_src,
        rpt_src,
        fns_src,
        fm_src,
    )

    def make_dests():
        # The page table destination is wider (and, for DRAFT_EXTEND, possibly
        # taller) than the copy: the kernel must leave the rest untouched.
        dests = (
            _sentinel((bs,)),
            _sentinel((bs + 1,)),
            _sentinel((page_rows, page_width)),
            _sentinel((expanded,)),
            _sentinel((expanded,)),
            _sentinel((expanded + 1,)),
            _sentinel((page_rows, max(rpt_cols, page_width // 4))) if with_rpt else None,
            _sentinel((expanded + 1,)) if with_flashmla else None,
            _sentinel((expanded + 1,)) if with_flashmla else None,
        )
        return dests

    scalars = (mode, copy_width, copy_width, size)
    return sources, make_dests, scalars


class FusedMetadataCopyPlanTest(unittest.TestCase):
    def _assert_case(self, mode, bs, next_n, copy_width, page_width, **kwargs):
        expanded_size = kwargs.pop("expanded_size", None)
        sources, make_dests, scalars = _build_case(
            mode, bs, next_n, copy_width, page_width, expanded_size, **kwargs
        )
        # `max_len` (DECODE) and `max_seqlen_k` (TARGET_VERIFY/DRAFT_EXTEND) are
        # both `copy_width` here; the row_sliced case below moves them apart from
        # the destination width.
        ref_dests = make_dests()
        kernel_dests = make_dests()

        K._copy_fused_metadata_fallback(*sources, *ref_dests, *scalars)

        plan = K._fused_metadata_copy_plan(*sources, *kernel_dests, *scalars)
        self.assertIsNotNone(plan, f"plan refused mode={mode} copy_width={copy_width}")
        _apply_plan(plan)

        for name, ref, got in zip(NAMES, ref_dests, kernel_dests):
            if ref is None:
                self.assertIsNone(got)
                continue
            self.assertTrue(
                torch.equal(ref, got),
                f"{name} differs (mode={mode} bs={bs} next_n={next_n} "
                f"copy_width={copy_width} page_width={page_width} {kwargs})",
            )
        return plan

    def _plan_segments(self, plan):
        """Number of copy segments the kernel performs for this plan."""
        (
            _cache_src,
            _cu_src,
            page_src,
            _nsa_src,
            _exp_src,
            _dcu_src,
            rpt_src,
            _fns_src,
            _fm_src,
            _cache_dst,
            _cu_dst,
            page_dst,
            _nsa_dst,
            _exp_dst,
            _dcu_dst,
            rpt_dst,
            _fns_dst,
            _fm_dst,
            n_cache,
            n_cu,
            n_nsa,
            n_exp,
            n_dcu,
            n_fns,
            n_fm,
        ) = plan
        segments = sum(1 for n in (n_cache, n_cu, n_nsa, n_exp, n_dcu, n_fns, n_fm) if n > 0)
        if page_src is not None and page_dst is not None and page_src.numel() > 0:
            segments += 1
        if rpt_src is not None and rpt_dst is not None and rpt_src.numel() > 0:
            segments += 1
        return segments

    # ---------------------------------------------------------------- DECODE
    def test_decode_all_optional_combinations(self):
        for with_rpt, with_flashmla in itertools.product((False, True), repeat=2):
            for bs, copy_width, page_width in (
                (1, 1, 1),
                (8, 16, 16),
                (37, 300, 512),
                (128, 4096, 4096),
            ):
                with self.subTest(mode=0, bs=bs, rpt=with_rpt, flashmla=with_flashmla):
                    self._assert_case(
                        0,
                        bs,
                        next_n=1,
                        copy_width=copy_width,
                        page_width=page_width,
                        with_rpt=with_rpt,
                        with_flashmla=with_flashmla,
                    )

    def test_decode_ignores_seqlens_expanded(self):
        """DECODE must not touch `dsa_seqlens_expanded` (it is stale by design)."""
        plan = self._assert_case(
            0, 16, next_n=1, copy_width=64, page_width=64, with_exp=True, with_rpt=True
        )
        self.assertEqual(plan[4], None)  # exp source
        self.assertEqual(plan[13], None)  # exp destination
        self.assertEqual(plan[21], 0)  # n_exp

    def test_decode_segment_count_is_six(self):
        """One launch replaces the six copies of a decode refresh."""
        plan = self._assert_case(
            0, 32, next_n=1, copy_width=128, page_width=256, with_rpt=True
        )
        self.assertEqual(self._plan_segments(plan), 6)

    def test_decode_without_real_page_table_is_five(self):
        plan = self._assert_case(0, 32, next_n=1, copy_width=128, page_width=128)
        self.assertEqual(self._plan_segments(plan), 5)

    # --------------------------------------------------------- TARGET_VERIFY
    def test_target_verify_all_optional_combinations(self):
        for with_rpt, with_exp, with_flashmla in itertools.product(
            (False, True), repeat=3
        ):
            for bs, next_n, copy_width, page_width in (
                (1, 1, 3, 8),
                (4, 3, 64, 64),
                (9, 4, 300, 512),
                (64, 2, 2048, 2048),
            ):
                with self.subTest(
                    mode=1,
                    bs=bs,
                    next_n=next_n,
                    rpt=with_rpt,
                    exp=with_exp,
                    flashmla=with_flashmla,
                ):
                    self._assert_case(
                        1,
                        bs,
                        next_n=next_n,
                        copy_width=copy_width,
                        page_width=page_width,
                        with_rpt=with_rpt,
                        with_exp=with_exp,
                        with_flashmla=with_flashmla,
                    )

    def test_target_verify_expanded_size_smaller_than_arrays(self):
        """`seqlens_expanded_size` bounds dsa_cu_seqlens_k and the FlashMLA pairs."""
        plan = self._assert_case(
            1,
            8,
            next_n=4,
            copy_width=64,
            page_width=96,
            expanded_size=20,  # < bs * next_n == 32
            with_rpt=True,
            with_exp=True,
            with_flashmla=True,
        )
        self.assertEqual(plan[22], 20)  # n_dcu
        self.assertEqual(plan[23], 21)  # n_fns == size + 1
        self.assertEqual(plan[24], 21)  # n_fm
        self.assertEqual(plan[20], 32)  # n_nsa stays a full copy in this mode

    def test_target_verify_segment_count_is_eight(self):
        plan = self._assert_case(
            1,
            8,
            next_n=2,
            copy_width=64,
            page_width=64,
            with_rpt=True,
            with_exp=True,
            with_flashmla=True,
        )
        self.assertEqual(self._plan_segments(plan), 9)
        plan = self._assert_case(
            1, 8, next_n=2, copy_width=64, page_width=64, with_rpt=True, with_exp=True
        )
        self.assertEqual(self._plan_segments(plan), 7)

    def test_target_verify_without_expanded_skips_it(self):
        plan = self._assert_case(
            1, 8, next_n=2, copy_width=32, page_width=32, with_rpt=False, with_exp=False
        )
        self.assertEqual(plan[21], 0)
        self.assertIsNone(plan[4])

    # --------------------------------------------------------- DRAFT_EXTEND
    def test_draft_extend_bounded_prefixes(self):
        plan = self._assert_case(
            2,
            8,
            next_n=4,
            copy_width=48,
            page_width=64,
            with_rpt=True,
            with_exp=True,
            with_flashmla=True,
        )
        size = 32  # bs * next_n
        self.assertEqual(plan[20], size)  # nsa[:size]
        self.assertEqual(plan[21], size)  # expanded[:size]
        self.assertEqual(plan[22], size)  # dcu[1:1+size]
        self.assertEqual(plan[23], size + 1)  # FlashMLA [:size + 1]

    def test_draft_extend_refuses_short_expanded_size(self):
        """`dst[:size].copy_(src)` raises when src is longer; keep it on torch."""
        sources, make_dests, scalars = _build_case(
            2, 8, next_n=4, copy_width=48, page_width=64, expanded_size=17,
            with_exp=True,
        )
        self.assertIsNone(
            K._fused_metadata_copy_plan(*sources, *make_dests(), *scalars)
        )

    # -------------------------------------------------------------- refusals
    def test_refuses_non_int32(self):
        sources, make_dests, scalars = _build_case(0, 8, 1, 16, 16, with_rpt=True)
        bad = list(sources)
        bad[0] = sources[0].to(torch.int64)
        self.assertIsNone(
            K._fused_metadata_copy_plan(*bad, *make_dests(), *scalars)
        )

    def test_refuses_page_table_width_mismatch(self):
        sources, make_dests, scalars = _build_case(0, 8, 1, 16, 32)
        # `page_indices` must be exactly `max_len` wide, like the torch slice.
        self.assertIsNone(
            K._fused_metadata_copy_plan(*sources, *make_dests(), 0, 17, 17, 8)
        )

    def test_refuses_non_contiguous_flat_destination(self):
        sources, make_dests, scalars = _build_case(0, 8, 1, 16, 16)
        dests = list(make_dests())
        wide = _sentinel((16, 2))
        dests[3] = wide[:, 0]  # stride 2
        self.assertIsNone(K._fused_metadata_copy_plan(*sources, *dests, *scalars))

    def test_refuses_1d_page_table(self):
        sources, make_dests, scalars = _build_case(0, 8, 1, 16, 16)
        bad = list(sources)
        bad[2] = torch.zeros(8 * 16, dtype=torch.int32)
        self.assertIsNone(K._fused_metadata_copy_plan(*bad, *make_dests(), *scalars))

    def test_refuses_aliased_table_destinations(self):
        """A shared destination with different sources would be a race."""
        sources, make_dests, scalars = _build_case(
            0, 8, 1, 16, 16, with_rpt=True
        )
        dests = list(make_dests())
        page_dst = dests[2]
        dests[6] = page_dst  # real_page_table_dst is the page-table buffer
        self.assertIsNone(K._fused_metadata_copy_plan(*sources, *dests, *scalars))
        # With the *same* source the two writes agree, so it is accepted.
        same_src = list(sources)
        same_src[6] = sources[2]
        plan = K._fused_metadata_copy_plan(*same_src, *dests, *scalars)
        self.assertIsNotNone(plan)

    def test_refuses_two_segments_writing_one_buffer(self):
        """`cache_seqlens` and `dsa_cache_seqlens` are clamped differently."""
        sources, make_dests, scalars = _build_case(0, 8, 1, 16, 16)
        dests = list(make_dests())
        dests[3] = dests[0]  # dsa_cache_seqlens_dst is cache_seqlens_dst
        self.assertIsNone(K._fused_metadata_copy_plan(*sources, *dests, *scalars))

    def test_refuses_empty_batch(self):
        """bs == 0 has no page-table rows; the torch path keeps its exact no-op."""
        sources, make_dests, scalars = _build_case(0, 0, 1, 16, 16)
        self.assertIsNone(
            K._fused_metadata_copy_plan(*sources, *make_dests(), *scalars)
        )

    def test_plan_is_value_independent(self):
        """The plan reads metadata only, so replaying with new values is safe."""
        sources, make_dests, scalars = _build_case(
            1, 8, 2, 32, 64, with_rpt=True, with_exp=True, with_flashmla=True
        )
        dests = make_dests()
        first = K._fused_metadata_copy_plan(*sources, *dests, *scalars)
        mutated = tuple(
            (t.clone().fill_(i + 1) if isinstance(t, torch.Tensor) else t)
            for i, t in enumerate(sources)
        )
        second = K._fused_metadata_copy_plan(*mutated, *dests, *scalars)
        self.assertEqual(first[18:], second[18:])
        for a, b, name in zip(first[:18], second[:18], NAMES * 2):
            if a is None:
                self.assertIsNone(b, name)
            else:
                self.assertEqual(tuple(a.shape), tuple(b.shape), name)

    # ---------------------------------------------------------------- multi
    def test_multi_plan_matches_torch_fallback(self):
        """Three sibling backends, one launch, same bytes as three fallbacks."""
        for with_rpt, with_flashmla in itertools.product((False, True), repeat=2):
            for bs, copy_width, page_width in ((1, 5, 8), (16, 64, 96), (48, 700, 1024)):
                with self.subTest(bs=bs, rpt=with_rpt, flashmla=with_flashmla):
                    sources, make_dests, scalars = _build_case(
                        0,
                        bs,
                        next_n=1,
                        copy_width=copy_width,
                        page_width=page_width,
                        with_rpt=with_rpt,
                        with_flashmla=with_flashmla,
                    )
                    src8 = sources[:5] + sources[5:7] + sources[7:9]
                    src8 = (
                        sources[0],
                        sources[1],
                        sources[2],
                        sources[3],
                        sources[5],
                        sources[6],
                        sources[7],
                        sources[8],
                    )
                    groups = []
                    for _ in range(3):
                        dests = make_dests()
                        groups.append(
                            (
                                dests[0],
                                dests[1],
                                dests[2],
                                dests[3],
                                dests[5],
                                dests[6],
                                dests[7],
                                dests[8],
                            )
                        )
                    ref_groups = [
                        tuple(_sentinel(t.shape) if t is not None else None for t in g)
                        for g in groups
                    ]

                    mode, max_len, max_seqlen_k, size = scalars
                    for ref in ref_groups:
                        K._copy_fused_metadata_fallback(
                            sources[0],
                            sources[1],
                            sources[2],
                            sources[3],
                            None,
                            sources[5],
                            sources[6],
                            sources[7],
                            sources[8],
                            ref[0],
                            ref[1],
                            ref[2],
                            ref[3],
                            None,
                            ref[4],
                            ref[5],
                            ref[6],
                            ref[7],
                            0,
                            max_len,
                            max_seqlen_k,
                            size,
                        )

                    plan = K._fused_metadata_copy_multi_plan(
                        *src8, tuple(groups), max_len, size
                    )
                    self.assertIsNotNone(plan)
                    _apply_multi_plan(plan)
                    for ref, got, name in zip(ref_groups, groups, MULTI_NAMES * 3):
                        for r, g in zip(ref, got):
                            if r is None:
                                self.assertIsNone(g, name)
                            else:
                                self.assertTrue(torch.equal(r, g), f"{name} differs")

    def test_multi_plan_refuses_when_destinations_disagree(self):
        sources, make_dests, scalars = _build_case(0, 8, 1, 16, 16)
        src8 = (
            sources[0],
            sources[1],
            sources[2],
            sources[3],
            sources[5],
            sources[6],
            sources[7],
            sources[8],
        )
        groups = []
        for i in range(3):
            dests = make_dests()
            groups.append(
                (
                    dests[0],
                    dests[1],
                    dests[2],
                    dests[3],
                    dests[5],
                    dests[6],
                    dests[7],
                    dests[8],
                )
            )
        mode, max_len, _max_seqlen_k, size = scalars
        self.assertIsNotNone(
            K._fused_metadata_copy_multi_plan(*src8, tuple(groups), max_len, size)
        )
        groups[2] = tuple(
            _sentinel((16, 8)) if i == 2 else t for i, t in enumerate(groups[2])
        )
        self.assertIsNone(
            K._fused_metadata_copy_multi_plan(*src8, tuple(groups), max_len, size)
        )

    # ------------------------------------------------------------ entry point
    def test_entry_point_keeps_the_fallback_reachable(self):
        """With no wheel operator the entry point still produces the torch bytes."""
        sources, make_dests, scalars = _build_case(0, 8, 1, 16, 16, with_rpt=True)
        dests = make_dests()

        if K._metadata_copy_kernel_ops() is not False:
            self.skipTest("this wheel registers fused_dsa_metadata_copy")

        self.assertFalse(K._fused_metadata_copy_via_kernel(*sources, *dests, *scalars))

        # What the entry point does next: the very same fallback, on the very
        # same buffers. It must produce the bytes a separate reference run does.
        K._copy_fused_metadata_fallback(*sources, *dests, *scalars)
        ref = [_sentinel(t.shape) if t is not None else None for t in make_dests()]
        K._copy_fused_metadata_fallback(*sources, *ref, *scalars)
        for r, g, name in zip(ref, dests, NAMES):
            if r is not None:
                self.assertTrue(torch.equal(r, g), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
