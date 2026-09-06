"""Tests for the GSplats-specific LOD axis resolvers (``lod/gsplats.py``).

Covers ``resolve_substitutive_axis_gsplats`` (the ``lod_group=`` kwarg) and
``resolve_additive_axis_gsplats`` (the ``additive_lod=`` kwarg) against real
``GSplatData`` — the GSplats peers of ``resolve_additive_axis_points`` /
``resolve_additive_axis_lines`` (tested in ``test_points.py`` / ``test_lines.py``).

The geometry-agnostic kind=lod ``Group`` machinery (builder, validation,
``coverage_fractions``, display-type resolution) is exercised in
``test_lod_group.py``. End-to-end ``add_gsplats_from_data(lod_group=...)``
round-trips live in ``luxar.core.tests.group.test_group``.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimension, Dimensions
from luxar.core.group.lod.group import (
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
    coverage_fractions,
    partitioned_coverage_fractions,
)
from luxar.core.group.lod.gsplats import (
    resolve_additive_axis_gsplats,
    resolve_additive_rungs,
    resolve_substitutive_axis_gsplats,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod import make_additive_lod, make_substitutive_lod
from luxar.io.compiler import LuxarZarrCompiler


def _make_random_gsplat(n: int = 64, ndim: int = 3, seed: int = 0) -> GSplatData:
    """Random anisotropic Gaussian splats with overlap (single substitutive level)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n, ndim)).astype(np.float32) * 1.5
    amplitudes = (np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32)
    tril = ndim * (ndim + 1) // 2
    chol = (rng.standard_normal((n, tril)) * 0.1).astype(np.float32)
    diag_idx = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag_idx] = np.abs(chol[:, diag_idx]) + 0.4
    return GSplatData(centers=centers, amplitudes=amplitudes, cholesky_factors=chol)


@pytest.fixture(scope="module")
def flat() -> GSplatData:
    """A single-substitutive, single-additive dataset (``fit`` output shape)."""
    return _make_random_gsplat(n=64, seed=0)


@pytest.fixture(scope="module")
def pyramid() -> GSplatData:
    """A multi-substitutive pyramid (n_substitutive > 1) for stored-path tests."""
    return make_substitutive_lod(
        _make_random_gsplat(n=64, seed=1), levels=2, device="cpu"
    )


# ────────────────────────────────────────────────────────────────────────
# resolve_substitutive_axis_gsplats — the ``lod_group=`` kwarg
# ────────────────────────────────────────────────────────────────────────


class TestResolveSubstitutiveAxisGsplats:
    def test_none_is_passthrough(self, flat) -> None:
        data, cov = resolve_substitutive_axis_gsplats(flat, None)
        assert data is flat
        assert cov is None

    def test_true_requires_multi_substitutive(self, flat) -> None:
        with pytest.raises(ValueError, match="n_substitutive"):
            resolve_substitutive_axis_gsplats(flat, True)

    def test_true_passes_pyramid_through(self, pyramid) -> None:
        data, cov = resolve_substitutive_axis_gsplats(pyramid, True)
        assert data.n_substitutive == pyramid.n_substitutive
        assert cov is None

    def test_false_collapses_to_finest(self, pyramid) -> None:
        assert pyramid.n_substitutive > 1
        data, _ = resolve_substitutive_axis_gsplats(pyramid, False)
        assert data.n_substitutive == 1

    def test_false_single_level_noop(self, flat) -> None:
        data, _ = resolve_substitutive_axis_gsplats(flat, False)
        assert data is flat

    def test_dict_explicit_coverage_fractions(self, pyramid) -> None:
        data, cov = resolve_substitutive_axis_gsplats(
            pyramid, {"coverage_fractions": [0.0, 0.2, 1.0]}
        )
        assert cov == [0.0, 0.2, 1.0]

    def test_dict_non_monotonic_coverage_fractions_raises(self, pyramid) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            resolve_substitutive_axis_gsplats(
                pyramid, {"coverage_fractions": [0.0, 0.5, 0.1]}
            )

    def test_dict_coverage_fractions_out_of_range_raises(self, pyramid) -> None:
        # The ceiling is MAX_COVERAGE_FRACTION == SCREEN_FILL_DIAGONAL_RATIO /
        # FILL_FACTOR == 4.0 (roughly the metric
        # a screen-filling object produces), not 1.0 — see the constant's docstring.
        with pytest.raises(ValueError, match=r"\[0, 4\]"):
            resolve_substitutive_axis_gsplats(
                pyramid, {"coverage_fractions": [0.0, 4.5]}
            )

    def test_dict_coverage_fractions_above_one_accepted(self, pyramid) -> None:
        # Above the auto-derived 1.0 anchor but within the ceiling: the escape
        # hatch for a level that must hold until the object is LARGER than a
        # half-fitted-axis (e.g. a spatially tiled layer). 4.0 is inclusive.
        _, cov = resolve_substitutive_axis_gsplats(
            pyramid, {"coverage_fractions": [0.0, 1.5]}
        )
        assert cov == [0.0, 1.5]
        _, cov = resolve_substitutive_axis_gsplats(
            pyramid, {"coverage_fractions": [0.0, 4.0]}
        )
        assert cov == [0.0, 4.0]

    def test_dict_empty_coverage_fractions_raises_clean_error(self, pyramid) -> None:
        # Empty explicit list → actionable ValueError, not an IndexError from the
        # [0]/[-1] range check (regression: deep-double-check).
        with pytest.raises(ValueError, match="non-empty"):
            resolve_substitutive_axis_gsplats(pyramid, {"coverage_fractions": []})

    def test_dict_stored_pyramid_rejects_compute_kwargs(self, pyramid) -> None:
        # Compute kwargs on an already-built pyramid (without recompute) must
        # not be silently ignored.
        with pytest.raises(ValueError, match="recompute"):
            resolve_substitutive_axis_gsplats(pyramid, {"levels": 2})

    def test_dict_computes_on_single_level(self, flat) -> None:
        data, _ = resolve_substitutive_axis_gsplats(
            flat, {"levels": 2, "device": "cpu"}
        )
        assert data.n_substitutive >= 2

    def test_invalid_spec_type_raises(self, flat) -> None:
        with pytest.raises(TypeError, match="None, bool, or dict"):
            resolve_substitutive_axis_gsplats(flat, 1.5)  # type: ignore[arg-type]


#: Source sizes for :func:`stacked_with_stored_ladders`. Deliberately UNEQUAL
#: and small-first: an equal-count ladder over equal sources is proportional AND
#: slice-even at once, so it cannot tell a merged per-source ladder apart from a
#: recomputed slice-even one — which is precisely the confusion that let #2485's
#: authored spec go unnoticed.
_STACK_SOURCE_SIZES = (4, 12, 40)


@pytest.fixture(scope="module")
def stacked_with_stored_ladders() -> GSplatData:
    """A stacked dataset whose SOURCES each already carry an additive ladder.

    The shape ``combine_timepoints_to_4d`` produces in
    ``demo_gsplats_4d_nexrad_supercell`` off its precomputed bundle: every
    per-frame archive was written with its own ladder, and
    :meth:`GSplatData.combine_as_new_dimension` merges them rather than dropping
    them (``_concat_additive_levels`` concatenates rung *i* of every source into
    rung *i* of the stack). Amplitudes are scaled ``100 ** source_index`` so the
    small source is also the faintest, as a sparse early radar scan is.
    """
    sources = []
    for index, n in enumerate(_STACK_SOURCE_SIZES):
        source = _make_random_gsplat(n=n, ndim=3, seed=100 + index)
        source = GSplatData(
            centers=source.centers,
            amplitudes=(source.amplitudes * 100.0**index).astype(np.float32),
            cholesky_factors=source.cholesky_factors,
        )
        sources.append(make_additive_lod(source, n_lods=4, method="greedy"))
    return GSplatData.combine_as_new_dimension(sources, sigma=0.0)


def _rung_slice_histogram(data: GSplatData, rung: int) -> list[int]:
    """How many splats of ``rung`` sit at each stacked (time) coordinate."""
    coords = np.asarray(data.substitutive_levels[0].additive_sublods[rung].centers)[
        :, 3
    ]
    return [int((coords == float(i)).sum()) for i in range(len(_STACK_SOURCE_SIZES))]


# ────────────────────────────────────────────────────────────────────────
# resolve_additive_axis_gsplats — the ``additive_lod=`` kwarg
# ────────────────────────────────────────────────────────────────────────


class TestResolveAdditiveAxisGsplats:
    def test_none_is_passthrough(self, flat) -> None:
        assert resolve_additive_axis_gsplats(flat, None) is flat

    def test_true_requires_existing_ladder(self, flat) -> None:
        # flat data has a single additive sub-LOD per substitutive level.
        with pytest.raises(ValueError, match="additive ladder"):
            resolve_additive_axis_gsplats(flat, True)

    def test_true_passes_existing_ladder_through(self, flat) -> None:
        laddered = make_additive_lod(flat, n_lods=3)
        result = resolve_additive_axis_gsplats(laddered, True)
        assert result.n_additive_sublods == 3

    def test_false_flattens_to_single_sublod(self, flat) -> None:
        laddered = make_additive_lod(flat, n_lods=3)
        result = resolve_additive_axis_gsplats(laddered, False)
        assert result.n_additive_sublods == 1

    def test_dict_computes_ladder(self, flat) -> None:
        result = resolve_additive_axis_gsplats(flat, {"n_lods": 3})
        assert result.n_additive_sublods == 3

    def test_dict_breakpoints_pass_through(self, flat) -> None:
        # Cumulative-count breakpoints flow through to make_additive_lod and
        # determine the level count (closes the convenience-path gap).
        result = resolve_additive_axis_gsplats(flat, {"breakpoints": [30, 64]})
        assert result.n_additive_sublods == 2

    def test_dict_counts_clamp_per_substitutive_level(self, pyramid) -> None:
        """REGRESSION: explicit ``counts:`` breakpoints larger than a COARSER
        substitutive level (smaller by K^s) used to abort the whole build with
        'largest breakpoint exceeds N'. They now clamp per level — mirroring the
        CLI per-part sites (recipes/pyramid/gsplat additive)."""
        assert pyramid.n_substitutive > 1
        coarsest = min(
            pyramid.at_substitutive(s).n_splats for s in range(pyramid.n_substitutive)
        )
        # A count that exceeds every coarser level but fits the finest (== the
        # full dataset N — larger would be a typo and abort, see the next test).
        big = pyramid.n_splats
        assert coarsest < big
        result = resolve_additive_axis_gsplats(
            pyramid, {"breakpoints": [1, big]}
        )  # must NOT raise
        # Every level keeps all its splats and gains a (clamped) ladder.
        assert result.n_substitutive == pyramid.n_substitutive
        for s in range(result.n_substitutive):
            lvl = result.at_substitutive(s)
            assert sum(sub.n_splats for sub in lvl.additive_sublods) == lvl.n_splats

    def test_dict_counts_exceeding_whole_group_raise(self, pyramid) -> None:
        """Counts exceeding the WHOLE group (the finest substitutive level) are
        a dataset-scale typo and must still abort loudly — the per-level clamp
        never masks them."""
        with pytest.raises(ValueError, match="exceeds N="):
            resolve_additive_axis_gsplats(
                pyramid, {"breakpoints": [pyramid.n_splats + 1]}
            )

    def test_dict_stream_breakpoints_per_level(self, pyramid) -> None:
        """A ``stream:<c>`` spec sizes each substitutive level's ladder against
        ITS OWN N through the convenience API."""
        result = resolve_additive_axis_gsplats(pyramid, {"breakpoints": "stream:8"})
        for s in range(result.n_substitutive):
            lvl = result.at_substitutive(s)
            incs = [sub.n_splats for sub in lvl.additive_sublods]
            assert sum(incs) == lvl.n_splats
            assert lvl.additive_sublods[0].stats["lod_breakpoints_kind"] == "stream"

    def test_invalid_spec_type_raises(self, flat) -> None:
        with pytest.raises(TypeError, match="None, bool, or dict"):
            resolve_additive_axis_gsplats(flat, 1.5)  # type: ignore[arg-type]

    def test_a_stored_merged_ladder_shadows_the_authored_spec(
        self, stacked_with_stored_ladders
    ) -> None:
        """An authored spec is a NO-OP on a stack whose sources carry ladders.

        #2485's second failure: the NEXRAD supercell's ``additive_lod=`` had been
        inert on its precomputed-bundle path all along, so ``slice_dims=``
        inherited that. The mechanism is documented on the resolver; what this
        pins is the OBSERVABLE — what survives is a per-source equal-count ladder,
        hence exactly PROPORTIONAL per slice (``[1, 3, 10]``, one quarter of each
        source), stamped with the merge's fingerprint (``lod_level`` +
        ``n_sources``) and no ``lod_method``.
        """
        stack = stacked_with_stored_ladders
        assert stack.n_additive_sublods == 4  # the merge, not our ladder

        result = resolve_additive_axis_gsplats(stack, {"n_lods": 4, "slice_dims": [3]})
        assert (
            _rung_slice_histogram(result, 0)
            == [n // 4 for n in _STACK_SOURCE_SIZES]
            == [1, 3, 10]
        )
        stats = result.substitutive_levels[0].additive_sublods[0].stats
        assert "lod_method" not in stats, (
            "the ordering ran after all; the pass-through branch was not taken"
        )
        assert stats["n_sources"] == len(_STACK_SOURCE_SIZES)

    def test_recompute_lets_the_slice_even_ordering_run(
        self, stacked_with_stored_ladders
    ) -> None:
        """``recompute=True`` is what makes ``slice_dims=`` reach the builder.

        Same stack, same spec plus the flag: rung 0 becomes slice-EVEN rather
        than proportional. The 4-splat source is carried WHOLE (it is smaller
        than the equal per-slice budget) and the 40-splat one is capped at the
        same budget, so ``[1, 3, 10]`` becomes ``[4, 5, 5]`` — a rung of the same
        14 splats, redistributed. Under the shadowed spec that smallest source
        got 1.
        """
        result = resolve_additive_axis_gsplats(
            stacked_with_stored_ladders,
            {"n_lods": 4, "slice_dims": [3], "recompute": True},
        )
        assert _rung_slice_histogram(result, 0) == [4, 5, 5]
        assert sum(_rung_slice_histogram(result, 0)) == 14
        stats = result.substitutive_levels[0].additive_sublods[0].stats
        assert stats["lod_method"] == "greedy"
        # Every splat still accounted for, once, across the rebuilt ladder.
        assert [
            sum(counts)
            for counts in zip(
                *(_rung_slice_histogram(result, r) for r in range(4)), strict=True
            )
        ] == list(_STACK_SOURCE_SIZES)

    def test_raw_slice_dim_maps_to_the_stored_scene_dim(self, tmp_path) -> None:
        """The authored raw column and compiled scene-column stamp must agree."""
        data = _make_random_gsplat(n=64, ndim=4, seed=104)
        centers = np.asarray(data.centers).copy()
        centers[:, 0] = np.repeat(np.arange(4, dtype=np.float32), 16)
        data = GSplatData(
            centers=centers,
            amplitudes=data.amplitudes,
            cholesky_factors=data.cholesky_factors,
        )

        out = tmp_path / "slice-dim-frame.luxar.zarr"
        dimensions = Dimensions(
            [
                Dimension("x"),
                Dimension("y"),
                Dimension("z"),
                Dimension("time", display=False, discrete=True),
            ]
        )
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=dimensions)
            scene.add_gsplats_from_data(
                "g",
                data,
                dim_order=["time", "x", "y", "z"],
                additive_lod={"n_lods": 4, "slice_dims": [0]},
            )

        node = zarr.open_group(out, mode="r")["g"]
        assert [list(node[f"additive_{i}"].attrs["slice_dims"]) for i in range(4)] == [
            [3],
            [3],
            [3],
            [3],
        ]

    def test_spatial_method_rejected(self, flat) -> None:
        """B8-G3/[P8]: documents the deliberate cross-geometry asymmetry —
        unlike Points/Lines (which order additive LODs spatially and accept
        ``method='poisson-disk'``/``'spatial-uniform'``), the GSplats additive
        ladder is energy/greedy-based. A spatial method name is forwarded to
        ``make_additive_lod`` and rejected, rather than silently ignored."""
        with pytest.raises(ValueError, match="method must be one of"):
            resolve_additive_axis_gsplats(flat, {"method": "poisson-disk"})


# ────────────────────────────────────────────────────────────────────────
# resolve_additive_rungs — the same vocabulary as an ordering-free COUNT
# ────────────────────────────────────────────────────────────────────────


#: Every spec whose rung count both functions must agree on. ``True`` is here
#: too, but only the LADDERED half of the matrix runs it (see the test).
#:
#: The dicts WITHOUT ``n_lods`` are not padding: the wrapper repeats the
#: resolver's ``n_lods=4`` default by hand (``kwargs.get("n_lods", 4)`` against
#: ``kwargs.setdefault("n_lods", 4)``), and only a spec that leaves the key unset
#: scores the two defaults against each other. Likewise the ``breakpoints``
#: entries, which are the only ones that reach the counts-validation half of the
#: wrapper at all.
_RUNG_PARITY_SPECS = [
    None,
    True,
    False,
    {},
    {"recompute": True},
    {"method": "radial"},
    {"n_lods": 1},
    {"n_lods": 3},
    {"n_lods": 1, "recompute": True},
    {"n_lods": 3, "recompute": True},
    # Explicit cumulative counts that FIT the leaf (64 splats): the in-range
    # path, where the strict validator passes and the clamp has nothing to do.
    {"breakpoints": [16, 32, 48]},
    {"recompute": True, "breakpoints": [16, 32, 48]},
    {"recompute": True, "breakpoints": [64]},
]


class TestResolveAdditiveRungs:
    """``resolve_additive_rungs`` must AGREE with ``resolve_additive_axis_gsplats``.

    The gate in ``gsplats_pipeline.from_io._reject_a_partition_beside_a_stored_ladder``
    refuses a call on the strength of this count, before any data exists, on the
    claim that "the two are one contract stated twice" (#1632). That claim was
    prose only; these tests make it a measurement, by scoring the count against
    the ladder the resolver actually builds for the same spec.
    """

    @pytest.mark.parametrize("spec", _RUNG_PARITY_SPECS)
    @pytest.mark.parametrize("laddered", [False, True])
    def test_the_count_matches_what_the_resolver_builds(self, spec, laddered) -> None:
        data = _make_random_gsplat(n=64, seed=7)
        if laddered:
            data = make_additive_lod(data, n_lods=3)
        stored = data.substitutive_levels[0].n_additive_lods
        assert stored == (3 if laddered else 1)

        if spec is True and not laddered:
            # The one combination with no count to compare: the resolver raises
            # ("requires every substitutive level to already carry an additive
            # ladder") rather than returning data. The query answers 1, which is
            # honest but is a rung count, not a verdict — pinned as its own case
            # below, and documented as out of scope in the docstring.
            pytest.skip("resolve_additive_axis_gsplats raises; nothing to compare")

        built = resolve_additive_axis_gsplats(data, spec)
        assert (
            resolve_additive_rungs(spec, stored_rungs=stored, n_splats=data.n_splats)
            == built.substitutive_levels[0].n_additive_lods
        )

    def test_true_on_an_unladdered_input_counts_the_ladder_that_is_there(self) -> None:
        """The deliberate non-parity case: ``True`` never BUILDS, it only asserts.

        ``resolve_additive_axis_gsplats`` raises on this call; the count is still
        1, which is what the gate needs to hear (there is no ladder, so no
        partition conflict) and lets the resolver report the invalid call itself.
        """
        data = _make_random_gsplat(n=64, seed=8)
        assert resolve_additive_rungs(True, stored_rungs=1, n_splats=data.n_splats) == 1
        with pytest.raises(ValueError, match="additive ladder"):
            resolve_additive_axis_gsplats(data, True)

    @pytest.mark.parametrize(
        "spec",
        [
            {"recompute": True, "breakpoints": [0.3, 1.0]},  # energy fractions
            {"recompute": True, "n_lods": 0},  # malformed
            "stream",  # not None, not a bool, not a dict
            1.5,
        ],
    )
    def test_an_uncountable_spec_is_unknown_rather_than_a_guess(self, spec) -> None:
        """``None`` says "this vocabulary cannot read the spec", never "no ladder".

        Callers must fall back to what they already know — the gate falls back to
        ``stored_rungs``, since an unreadable kwarg cannot be trusted to have
        removed a ladder that is demonstrably in the store.
        """
        assert resolve_additive_rungs(spec, stored_rungs=3, n_splats=64) is None

    def test_an_out_of_range_counts_list_is_unknown_where_the_resolver_raises(
        self,
    ) -> None:
        """STRICT FIRST, CLAMP SECOND — the order the resolver itself uses.

        ``resolve_additive_axis_gsplats`` validates an explicit ``counts:`` list
        against ``substitutive_levels[0].n_splats_total`` BEFORE its per-level
        clamp loop, so a list larger than the data ABORTS rather than shrinking.
        A single grafted leaf is that finest level, so the wrapper must validate
        too: clamping first turned ``[100]`` into ``[64]`` and answered a
        confident ONE rung for a call that cannot run at all — which is how the
        gate came to skip a doomed call and strand a childless wrapper (#1632).
        Both halves are asserted, because "answers None" alone would also pass if
        the resolver had quietly clamped.
        """
        data = _make_random_gsplat(n=64, seed=9)
        spec = {"recompute": True, "breakpoints": [100]}
        with pytest.raises(ValueError, match="largest breakpoint 100 exceeds N=64"):
            resolve_additive_axis_gsplats(data, spec)
        assert resolve_additive_rungs(spec, stored_rungs=1, n_splats=64) is None

    def test_a_coarser_substitutive_level_is_unknown_where_the_resolver_clamps(
        self, pyramid
    ) -> None:
        """A DELIBERATE DIVERGENCE, pinned as such — not parity.

        Every other case in this class scores the two functions as equals. This
        one records where they are documented NOT to agree, and it takes a
        MULTI-substitutive input to see: the rest of the matrix is single-level
        ``_make_random_gsplat`` data, on which a leaf always IS
        ``substitutive_levels[0]`` and the divergence cannot arise.

        ``resolve_additive_axis_gsplats`` validates an explicit ``counts:`` list
        ONCE against the finest level and CLAMPS the coarser ones, so
        ``[n_finest]`` builds a single rung on every level. The query is asked per
        LEAF, so a coarser (smaller) level validates the same list against its own
        N, fails, and answers UNKNOWN where the resolver quietly clamped. Per-leaf
        strict validation is the right rule at the query's live caller — a grafted
        leaf really is the resolver's ``substitutive_levels[0]`` — so the cost is
        paid on the matrix-shaped branch, where it over-refuses a call that would
        have worked, refusing with an empty store rather than stranding anything.
        See the ``resolve_additive_rungs`` docstring's "Where that OVER-REFUSES".
        """
        laddered = resolve_additive_axis_gsplats(pyramid, {"n_lods": 2})
        sizes = [
            laddered.at_substitutive(s).n_splats for s in range(laddered.n_substitutive)
        ]
        assert len(sizes) > 1 and sizes[0] == max(sizes)  # finest is index 0

        spec = {"recompute": True, "breakpoints": [sizes[0]]}
        built = resolve_additive_axis_gsplats(laddered, spec)
        # The resolver builds ONE rung everywhere: the finest level's cut is its
        # own N, and every coarser level's is clamped to its own N.
        assert [
            built.substitutive_levels[s].n_additive_lods
            for s in range(built.n_substitutive)
        ] == [1] * built.n_substitutive

        counts = [
            resolve_additive_rungs(
                spec,
                stored_rungs=laddered.substitutive_levels[s].n_additive_lods,
                n_splats=sizes[s],
            )
            for s in range(laddered.n_substitutive)
        ]
        assert counts[0] == 1  # the finest level agrees
        assert all(c is None for c in counts[1:])  # every coarser one is UNKNOWN


# ────────────────────────────────────────────────────────────────────────
# Partition-bound anchor for add_gsplats_from_data(lod_group=...)
# ────────────────────────────────────────────────────────────────────────


def _write_pyramid(tmp_path, name, *, partitioned, wrap_in_group=False, **lod_group):
    """Write TWO stored pyramids through the scene adder and read their thresholds.

    ``partitioned=True`` hand-builds the ``kind=partition`` wrapper first (the
    shape ``add_gsplats_from_data`` cannot reach on its own), which is what makes
    each ladder a per-tile one. TWO parts on purpose: a one-part
    ``kind=partition`` is the degenerate shape ``partitioned_coverage_fractions``
    documents as 4x too coarse (its "tile" IS the whole object), so it must not be
    the fixture the rule is argued from.

    Returns ``[(coverage, counts), ...]``, one entry per part. The two part sizes
    are deliberately NON-proportional: 256 reduces to 16/64/256 (exact powers of
    ``K``), while 102 reduces to 7/26/102 — the ``ceil`` in the reduction breaks the
    ratio — so the two derived ladders differ numerically. A proportional pair (say
    256 and 128) would derive to the *identical* list, and the fixture could not
    then tell per-part derivation from wrapper-level derivation. See
    ``test_every_partition_bound_ladder_is_fills_screen_anchored``, which asserts
    the two lists differ.
    """
    parts = [
        make_substitutive_lod(
            _make_random_gsplat(n=n, seed=seed), levels=2, device="cpu"
        )
        for n, seed in ((256, 3), (102, 4))
    ]
    out = tmp_path / name
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        wrapper = (
            scene.add_partition_group("tiled", display_type="gsplats", max_elements=256)
            if partitioned
            else scene
        )
        for i, data in enumerate(parts):
            target = (
                wrapper.add_group(f"holder_{i}")
                if (partitioned and wrap_in_group)
                else wrapper
            )
            target.add_gsplats_from_data(f"part_{i}", data, lod_group=lod_group or True)

    root = zarr.open(str(out), mode="r")
    node = root["tiled"] if partitioned else root
    results = []
    for i in range(len(parts)):
        holder = node[f"holder_{i}"] if (partitioned and wrap_in_group) else node
        lod = holder[f"part_{i}"]
        results.append((_lod_child_coverage(lod), _lod_child_counts(lod)))
    return results


def _child_names(lod_group) -> list:
    return sorted(
        (k for k in lod_group.keys() if k.startswith("child_")),
        key=lambda k: int(k.split("_")[1]),
    )


def _lod_child_coverage(lod_group) -> list:
    return [
        float(lod_group[k].attrs["coverage_fraction"]) for k in _child_names(lod_group)
    ]


def _lod_child_counts(lod_group) -> list:
    return [int(lod_group[k].attrs["n_splats"]) for k in _child_names(lod_group)]


class TestPartitionBoundAnchorGsplats:
    """``lod_group=`` under a hand-built ``kind=partition`` takes the tile anchor
    (finest ``PARTITION_FINEST_AREA`` = 1.0 — the tile alone fills the screen).

    The dispatch used to auto-derive the WHOLE-OBJECT ladder unconditionally
    (finest = the whole-object anchor). Two tests here REGRESS without the fix
    (``test_every_partition_bound_ladder_is_fills_screen_anchored``,
    ``test_plain_group_between_partition_and_ladder_still_anchored``); the other two
    are CONTROLS that pass either way and pin what must NOT change. The fixture is
    a real TWO-tile partition and every assertion covers both tiles.
    """

    def test_every_partition_bound_ladder_is_fills_screen_anchored(
        self, tmp_path
    ) -> None:
        parts = _write_pyramid(tmp_path, "tiled.luxar.zarr", partitioned=True)
        assert len(parts) == 2, "must be a real 2-tile partition"
        for i, (cov, counts) in enumerate(parts):
            assert cov == pytest.approx(partitioned_coverage_fractions(counts)), (
                f"part_{i}"
            )
            assert cov[-1] == pytest.approx(PARTITION_FINEST_AREA), f"part_{i}"
        # Screen-occupancy halving is COUNT-INDEPENDENT by design, so both
        # parts derive the same list — the old "non-proportional counts give
        # different thresholds" sensitivity control is deliberately gone. Pin
        # the derived shape instead: halving anchored at fills-screen (area
        # 1.0 — the tile alone fills the screen), ×2 per finer level.
        for i, (cov, counts) in enumerate(parts):
            assert cov == pytest.approx(partitioned_coverage_fractions(counts)), (
                f"part_{i}"
            )
            for a, b in zip(cov[1:-1], cov[2:]):
                assert b == pytest.approx(a * 2.0), (
                    f"part_{i}: expected ×2 area-halving spacing, got {cov}"
                )

    def test_scene_root_still_gets_the_whole_object_anchor(self, tmp_path) -> None:
        """CONTROL (passes pre-fix): the over-trigger guard."""
        parts = _write_pyramid(tmp_path, "root.luxar.zarr", partitioned=False)
        assert len(parts) == 2
        for i, (cov, counts) in enumerate(parts):
            assert cov == pytest.approx(coverage_fractions(counts)), f"part_{i}"
            assert cov[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR), f"part_{i}"

    def test_plain_group_between_partition_and_ladder_still_anchored(
        self, tmp_path
    ) -> None:
        parts = _write_pyramid(
            tmp_path, "tiled.luxar.zarr", partitioned=True, wrap_in_group=True
        )
        assert len(parts) == 2
        for i, (cov, _counts) in enumerate(parts):
            assert cov[-1] == pytest.approx(PARTITION_FINEST_AREA), f"part_{i}"

    def test_explicit_coverage_fractions_still_win_under_a_partition(
        self, tmp_path
    ) -> None:
        """CONTROL (passes pre-fix): an explicit list must keep winning verbatim."""
        explicit = [0.0, 3.52, 4.0]
        parts = _write_pyramid(
            tmp_path,
            "tiled.luxar.zarr",
            partitioned=True,
            coverage_fractions=explicit,
        )
        assert len(parts) == 2
        for i, (cov, _counts) in enumerate(parts):
            assert cov == pytest.approx(explicit), f"part_{i}"


# ────────────────────────────────────────────────────────────────────────
# Acceptance: the hand-built scene path == what `gsplat lod --recipe adaptive`
# derives for the same tree (issue #1411)
# ────────────────────────────────────────────────────────────────────────


def _acceptance_params():
    """Recipe params sized so NOTHING is clamped or randomised away.

    ``levels=2`` with ``K=4`` on 128-splat parts is well inside
    ``_substitutive_for_part``'s ``levels <= log_K(n)`` clamp, so the per-part
    reduction the recipe runs and the one the scene path runs below are the same
    call with the same arguments — which is what makes a count-for-count
    comparison of the two ladders legitimate. Additive ladders and quality stamps
    are off: they change level *contents*, never level *counts*, so leaving them
    on would only slow the test down.
    """
    from luxar.gsplats.lod.recipes import RecipeParams

    return RecipeParams(
        max_elements=128,
        levels=2,
        compression_factor=4,
        additive_ladders=False,
        quality_stamps=False,
        device="cpu",
        seed=0,
    )


def test_hand_built_partition_matches_the_adaptive_recipe(tmp_path) -> None:
    """A hand-built ``kind=partition`` of scene-adder ladders must land on the
    SAME per-tile thresholds the ``adaptive`` recipe derives.

    This is the acceptance criterion of #1411. What it pins is the DISPATCH, not
    the formula: both paths ultimately call the same
    ``partitioned_coverage_fractions`` (and path B re-runs path A's
    ``to_spatial_partition`` / ``make_substitutive_lod``), so this is not two
    independent implementations agreeing. The difference is how each one *chooses*
    that function — ``recipes._substitutive_for_part`` names it outright because it
    *knows* it is building a per-tile ladder, while the scene adder has to *detect*
    the partition from the insertion point. They can only agree if that detection
    works.

    The comparison is count-for-count, not merely shape-for-shape: both sides
    partition the SAME flattened data with the same ``max_elements``/rule and then
    reduce each part with the same ``compression_factor``/``levels``, so the
    per-level splat counts are asserted equal before the thresholds are.
    Run in-process (no CLI subprocess) on a 512-splat CPU-only dataset.
    """
    from luxar.gsplats.lod.recipes import build_adaptive
    from luxar.gsplats.tree import GSplatLodGroup, total_splats

    params = _acceptance_params()
    data = _make_random_gsplat(n=512, seed=7)

    # ---- path A: the library recipe (what `gsplat lod --recipe adaptive` runs).
    recipe_tree = build_adaptive(data, params)
    recipe_counts, recipe_coverage = [], []
    for part in recipe_tree.children:
        assert isinstance(part, GSplatLodGroup), (
            "the adaptive recipe must give every part its own kind=lod group; "
            f"got {type(part).__name__}"
        )
        recipe_counts.append([total_splats(c) for c in part.children])
        recipe_coverage.append(
            [float(c.meta["coverage_fraction"]) for c in part.children]
        )
    assert len(recipe_coverage) > 1  # a real partition, not a single tile

    # ---- path B: hand-built kind=partition + one scene-adder ladder per part.
    # The SAME parts (same flatten → same BSP → same max_elements/rule) reduced
    # the same way, so only the threshold derivation can differ.
    base = data.flattened()
    partition = base.to_spatial_partition(
        max_elements=params.effective_max_elements, rule=params.partition_rule
    )
    out = tmp_path / "handbuilt.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        wrapper = scene.add_partition_group(
            "tiled",
            display_type="gsplats",
            max_elements=params.effective_max_elements,
        )
        for i, part in enumerate(partition.children):
            sub = make_substitutive_lod(
                GSplatData.from_tree(part),
                compression_factor=params.compression_factor,
                levels=params.levels,
                method=params.substitutive_method,
                device=params.device,
                seed=params.seed,
            )
            wrapper.add_gsplats_from_data(f"part_{i}", sub, lod_group=True)

    tiled = zarr.open(str(out), mode="r")["tiled"]
    scene_counts = [
        _lod_child_counts(tiled[f"part_{i}"]) for i in range(len(partition.children))
    ]
    scene_coverage = [
        _lod_child_coverage(tiled[f"part_{i}"]) for i in range(len(partition.children))
    ]

    assert scene_counts == recipe_counts, (
        "the two paths must reduce to identical per-level counts, or the "
        "threshold comparison below is not count-for-count"
    )
    for i, (scene_cov, recipe_cov) in enumerate(zip(scene_coverage, recipe_coverage)):
        assert scene_cov == pytest.approx(recipe_cov), f"part_{i}"
        assert scene_cov[0] == 0.0
        assert scene_cov[-1] == pytest.approx(PARTITION_FINEST_AREA)
