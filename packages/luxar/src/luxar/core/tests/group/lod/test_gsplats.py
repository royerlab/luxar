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

from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.group import (
    MAX_COVERAGE_FRACTION,
    coverage_fractions,
    partitioned_coverage_fractions,
)
from luxar.core.group.lod.gsplats import (
    resolve_additive_axis_gsplats,
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
        # FILL_FACTOR == 4.0 (the metric
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

    def test_spatial_method_rejected(self, flat) -> None:
        """B8-G3/[P8]: documents the deliberate cross-geometry asymmetry —
        unlike Points/Lines (which order additive LODs spatially and accept
        ``method='poisson-disk'``/``'spatial-uniform'``), the GSplats additive
        ladder is energy/greedy-based. A spatial method name is forwarded to
        ``make_additive_lod`` and rejected, rather than silently ignored."""
        with pytest.raises(ValueError, match="method must be one of"):
            resolve_additive_axis_gsplats(flat, {"method": "poisson-disk"})


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
    """``lod_group=`` under a hand-built ``kind=partition`` takes the tile anchor.

    The dispatch used to auto-derive the WHOLE-OBJECT ladder unconditionally
    (finest ``1.0``). Two tests here REGRESS without the fix
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
            assert cov[-1] == pytest.approx(MAX_COVERAGE_FRACTION), f"part_{i}"
        # Each ladder is derived from its OWN counts, not the wrapper's: the two
        # parts are non-proportional (see _write_pyramid), so a wrapper-level
        # derivation would give both the identical list.
        assert parts[0][0] != pytest.approx(parts[1][0]), (
            "the two parts derived identical thresholds, so this fixture cannot "
            f"distinguish per-part from wrapper-level derivation: {parts}"
        )

    def test_scene_root_still_gets_the_whole_object_anchor(self, tmp_path) -> None:
        """CONTROL (passes pre-fix): the over-trigger guard."""
        parts = _write_pyramid(tmp_path, "root.luxar.zarr", partitioned=False)
        assert len(parts) == 2
        for i, (cov, counts) in enumerate(parts):
            assert cov == pytest.approx(coverage_fractions(counts)), f"part_{i}"
            assert cov[-1] == 1.0, f"part_{i}"

    def test_plain_group_between_partition_and_ladder_still_anchored(
        self, tmp_path
    ) -> None:
        parts = _write_pyramid(
            tmp_path, "tiled.luxar.zarr", partitioned=True, wrap_in_group=True
        )
        assert len(parts) == 2
        for i, (cov, _counts) in enumerate(parts):
            assert cov[-1] == pytest.approx(MAX_COVERAGE_FRACTION), f"part_{i}"

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
        assert scene_cov[-1] == pytest.approx(MAX_COVERAGE_FRACTION)
