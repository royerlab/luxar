"""Tests for the geometry-agnostic kind=lod ``Group`` helpers (``lod/group.py``).

These cover the type-neutral machinery shared by Points, Lines, and GSplats:

- The standalone builder (``add_lod_group``): node creation, attr round-trip,
  child enumeration, and ``validate_lod_group()`` failure modes.
- The auto-derivation heuristic (``coverage_fractions``): monotonicity
  and the √-of-ratio scaling.
- Construction-time validation: unsupported selector, negative default_level.
- Display-type resolution (``resolve_display_type`` / ``compute_lod_display_type``),
  including the geometry-agnostic contract that a kind=lod group may hold
  heterogeneous children (finest child determines ``display_type``; no
  homogeneity check, unlike the Partition kind).

The GSplats-specific axis resolvers (``resolve_*_axis_gsplats``) are exercised
in ``test_gsplats.py``; their Points/Lines peers in ``test_points.py`` /
``test_lines.py``. End-to-end ``add_gsplats_from_data(lod_group=/additive_lod=)``
round-trips live in ``luxar.core.tests.group.test_group`` (the broader ``Group``
suite), not here — except the barrier-aware ``coarsen_dims`` end-to-end suite
(``TestCoarsenDimsGsplats``) which lives here alongside the other LOD-group tests,
parallel to the Points/Lines coarsen suites in ``test_substitutive_*.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimension, Dimensions
from luxar.core.group import Group
from luxar.core.group.lod.group import (
    compose_additive_under_substitutive,
    compute_lod_display_type,
    coverage_fractions,
    resolve_display_type,
    resolve_substitutive_axis,
    validate_lod_group,
)
from luxar.io.compiler import LuxarZarrCompiler

# Reusable tiny gsplat primitives (single splat at the origin).
_CENTERS = np.array([[0, 0, 0]], dtype=np.float32)
_CHOL = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────────
# Standalone builder — add_lod_group + child enumeration
# ────────────────────────────────────────────────────────────────────────


class TestAddLodGroup:
    """Round-trip a kind=lod ``Group`` built via the standalone builder."""

    def test_add_lod_group_writes_node_with_defaults(self, tmp_path) -> None:
        """Bare add_lod_group writes type=group + kind=lod with defaults."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            assert isinstance(lod, Group)
            assert lod.attrs["kind"] == "lod"
            assert lod.attrs["selector"] == "coverage"
            assert lod.attrs["default_level"] == 0

        store = zarr.open(str(output_path), mode="r")
        attrs = store["multires"].attrs
        assert attrs["type"] == "group"
        assert attrs["kind"] == "lod"
        assert attrs["selector"] == "coverage"
        assert attrs["default_level"] == 0

    def test_add_lod_group_with_explicit_default_level(self, tmp_path) -> None:
        """The default_level kwarg lands on the zarr attrs."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lod_group("multires", default_level=2)

        store = zarr.open(str(output_path), mode="r")
        assert store["multires"].attrs["default_level"] == 2

    def test_add_lod_group_children_are_subgroups_with_coverage_fraction(
        self, tmp_path
    ) -> None:
        """Children added via inherited add_* methods carry coverage_fraction."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            lod.add_gsplats(
                "coarse",
                centers=_CENTERS,
                amplitudes=1.0,
                cholesky_factors=_CHOL,
                coverage_fraction=0.0,
            )
            lod.add_gsplats(
                "fine",
                centers=_CENTERS,
                amplitudes=1.0,
                cholesky_factors=_CHOL,
                coverage_fraction=1.0,
            )

        store = zarr.open(str(output_path), mode="r")
        grp = store["multires"]
        assert grp.attrs["type"] == "group"
        assert grp.attrs["kind"] == "lod"
        assert sorted(grp.keys()) == ["coarse", "fine"]
        assert grp["coarse"].attrs["type"] == "gsplats"
        assert grp["coarse"].attrs["coverage_fraction"] == 0.0
        assert grp["fine"].attrs["coverage_fraction"] == 1.0

    def test_add_lod_group_can_nest_under_a_group(self, tmp_path) -> None:
        """kind=lod groups can sit beneath a normal Group container."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            g = scene.add_group("scene_root")
            g.add_lod_group("multires")

        store = zarr.open(str(output_path), mode="r")
        nested = store["scene_root"]["multires"]
        assert nested.attrs["type"] == "group"
        assert nested.attrs["kind"] == "lod"


# ────────────────────────────────────────────────────────────────────────
# Construction-time validation
# ────────────────────────────────────────────────────────────────────────


class TestLODGroupValidation:
    """Sad-path checks on the LOD-group builder and ``validate_lod_group``."""

    def test_unknown_selector_rejected(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="selector must be 'coverage'"):
                scene.add_lod_group("multires", selector="distance")

    def test_negative_default_level_rejected(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="default_level must be >= 0"):
                scene.add_lod_group("multires", default_level=-1)

    def test_validate_empty_children_raises(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("empty")
            with pytest.raises(ValueError, match="has no children"):
                validate_lod_group(lod)

    def test_validate_missing_coverage_fraction_raises(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            # Child added without coverage_fraction — validator should catch it.
            lod.add_gsplats(
                "c0", centers=_CENTERS, amplitudes=1.0, cholesky_factors=_CHOL
            )
            with pytest.raises(ValueError, match="missing 'coverage_fraction'"):
                validate_lod_group(lod)

    def test_validate_non_monotonic_raises(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            lod.add_gsplats(
                "c0",
                centers=_CENTERS,
                amplitudes=1.0,
                cholesky_factors=_CHOL,
                coverage_fraction=0.5,
            )
            lod.add_gsplats(
                "c1",
                centers=_CENTERS,
                amplitudes=1.0,
                cholesky_factors=_CHOL,
                coverage_fraction=0.1,  # < previous — invalid
            )
            with pytest.raises(ValueError, match="strictly greater"):
                validate_lod_group(lod)

    def test_validate_passes_for_well_formed_group(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            for i, cf in enumerate([0.0, 0.5, 1.0]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            # Should not raise.
            validate_lod_group(lod)
            assert [float(c.attrs["coverage_fraction"]) for c in lod.children] == [
                0.0,
                0.5,
                1.0,
            ]

    def test_validate_rejects_default_level_out_of_range(self, tmp_path) -> None:
        """``default_level`` past the number of children must fail validation.

        ``__init__`` cannot check this (children are added later); the
        check fires in ``validate_lod_group()``. Without it, a bad value
        sails into the on-disk zarr and only surfaces at viewer load time.
        """
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires", default_level=99)
            for i, cf in enumerate([0.0, 1.0]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            with pytest.raises(ValueError, match="default_level=99"):
                validate_lod_group(lod)


# ────────────────────────────────────────────────────────────────────────
# Auto-derivation heuristic
# ────────────────────────────────────────────────────────────────────────


class TestCoverageFractions:
    """The ``sqrt(N_i / N_finest)`` viewport-relative auto-derivation."""

    def test_coarsest_is_zero(self) -> None:
        assert coverage_fractions([100, 400, 1600])[0] == 0.0

    def test_finest_is_one(self) -> None:
        assert coverage_fractions([100, 400, 1600])[-1] == 1.0

    def test_sqrt_of_ratio_scaling(self) -> None:
        # coverage_i = sqrt(N_i / N_finest): 400/1600 → 0.5, 1600/1600 → 1.0.
        fractions = coverage_fractions([100, 400, 1600])
        assert fractions[1] == pytest.approx(0.5)
        assert fractions[2] == pytest.approx(1.0)

    def test_validated_numeric_example(self) -> None:
        got = coverage_fractions([89894, 359865, 1443108, 5801956, 23368376])
        assert got == pytest.approx(
            [0.0, 0.12409535876820393, 0.24850501137866587, 0.4982794191730524, 1.0]
        )

    def test_two_level_ladder_is_zero_one(self) -> None:
        # Any two-level ladder → [0.0, 1.0] (sqrt(N/N) == 1).
        assert coverage_fractions([100, 400]) == pytest.approx([0.0, 1.0])
        assert coverage_fractions([50, 200]) == pytest.approx([0.0, 1.0])

    def test_strict_monotonicity_enforced(self) -> None:
        # Even when input is non-increasing, output stays monotonic via the
        # relative ÷1.1 downward nudge of the coarser entries (never above 1.0).
        fractions = coverage_fractions([100, 100, 100])
        for i in range(1, len(fractions)):
            assert fractions[i] > fractions[i - 1]
        assert all(0.0 <= f <= 1.0 for f in fractions)

    def test_relative_nudge_is_proportional_and_downward(self) -> None:
        # Equal-count levels separate by ÷1.1 applied to the EARLIER (coarser)
        # entry, so the finest stays anchored at exactly 1.0 and nothing ever
        # exceeds it (the old upward ×1.1 bump produced 1.1 > 1.0 here).
        fractions = coverage_fractions([100, 100, 100])
        # child2 (finest) = sqrt(100/100) = 1.0; child1 = child2 / 1.1.
        assert fractions[1] == pytest.approx(1.0 / 1.1)
        assert fractions[2] == pytest.approx(1.0)

    def test_equal_count_tail_stays_within_unit_interval(self) -> None:
        # Regression: counts [10, 1000, 1000] used to derive [0.0, 1.0, 1.1] —
        # the upward bump violated the [0, 1]/finest==1.0 contract (and the
        # explicit-input validators reject 1.1). Duplicates must resolve by
        # nudging the coarser entry DOWN instead.
        fractions = coverage_fractions([10, 1000, 1000])
        assert fractions[0] == 0.0
        assert fractions[-1] == 1.0
        assert fractions[1] == pytest.approx(1.0 / 1.1)
        assert all(fractions[i] > fractions[i - 1] for i in range(1, len(fractions)))
        assert all(0.0 <= f <= 1.0 for f in fractions)

    def test_non_monotone_counts_capped_at_finest_anchor(self) -> None:
        # A count ladder that DECREASES toward the finest (sqrt ratio > 1 for
        # an intermediate level) is capped back under the 1.0 anchor.
        fractions = coverage_fractions([100, 1000, 10])
        assert fractions == pytest.approx([0.0, 1.0 / 1.1, 1.0])

    def test_derived_values_round_trip_explicit_validator(self) -> None:
        # The derived fractions must be accepted verbatim by the explicit
        # coverage_fractions= validator (strict-ascending, within [0, 1]) —
        # including for the degenerate ladders the guard has to repair.
        for counts in (
            [10, 1000, 1000],  # equal-count tail (used to derive 1.1)
            [100, 100, 100],  # all-equal
            [10, 0, 20],  # zero-count intermediate
            [100, 1000, 10],  # non-monotone (derives > 1 pre-cap)
            [1, 1000, 1_000_000],  # extreme ratio
        ):
            derived = coverage_fractions(counts)
            resolved = resolve_substitutive_axis(
                {"coverage_fractions": derived}, "Points"
            )
            assert resolved is not None
            assert resolved["coverage_fractions"] == pytest.approx(derived), counts

    def test_extreme_count_ratio_monotonic(self) -> None:
        # 1 → 1,000,000 elements must still yield strictly increasing finite
        # fractions (no overflow / non-monotonic artefacts).
        fractions = coverage_fractions([1, 1000, 1_000_000])
        for i in range(1, len(fractions)):
            assert fractions[i] > fractions[i - 1]
        assert all(np.isfinite(f) for f in fractions)  # no NaN

    def test_single_level_is_just_the_floor(self) -> None:
        assert coverage_fractions([42]) == [0.0]

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            coverage_fractions([])

    def test_zero_finest_rejected(self) -> None:
        # An empty finest level can't anchor the ratio denominator; the error
        # is actionable (names the likely cause: an all-non-positive-amplitude
        # reduction that culled every representative).
        with pytest.raises(ValueError, match="(?i)finest.*empty|0 elements"):
            coverage_fractions([10, 0])

    def test_intermediate_zero_count_does_not_raise(self) -> None:
        # A zero-count INTERMEDIATE level derives to a 0 fraction equal to the
        # 0.0 coarsest floor; the downward nudge bottoms out at 0 there, so the
        # guard lifts the zero entry onto a geometric ramp strictly between the
        # 0.0 floor and the next positive threshold instead of raising.
        cf = coverage_fractions([10, 0, 20])
        assert cf[0] == 0.0
        assert cf[-1] == 1.0
        assert all(cf[i] > cf[i - 1] for i in range(1, len(cf))), cf
        assert all(0.0 <= f <= 1.0 for f in cf), cf
        assert all(np.isfinite(f) for f in cf)


# ────────────────────────────────────────────────────────────────────────
# Display-type resolution (shared with the Partition kind)
# ────────────────────────────────────────────────────────────────────────


class TestDisplayTypeResolution:
    """``resolve_display_type`` / ``compute_lod_display_type`` are geometry-agnostic.

    A kind=lod group is intentionally *heterogeneous-tolerant*: children may be
    different geometry types (e.g. a coarse points level, a fine gsplats level),
    and the group's display type is taken from the finest (last) child. There is
    no homogeneity check here — that is the Partition kind's contract, not LOD's.
    """

    def test_resolve_display_type_plain_leaf(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            lod.add_points(
                "pts",
                np.zeros((3, 3), dtype=np.float32),
                coverage_fraction=0.0,
            )
            assert resolve_display_type(lod.children[0]) == "points"

    def test_resolve_display_type_specialized_group_uses_display_type(
        self, tmp_path
    ) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires", display_type="custom_marker")
            assert resolve_display_type(lod) == "custom_marker"

    def test_compute_lod_display_type_uses_finest_child(self, tmp_path) -> None:
        """Heterogeneous children are allowed; the finest (last) child wins."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            # Coarse level is points; fine level is gsplats — heterogeneous.
            lod.add_points(
                "coarse",
                np.zeros((3, 3), dtype=np.float32),
                coverage_fraction=0.0,
            )
            lod.add_gsplats(
                "fine",
                centers=_CENTERS,
                amplitudes=1.0,
                cholesky_factors=_CHOL,
                coverage_fraction=1.0,
            )
            # No homogeneity check fires — the group validates fine...
            validate_lod_group(lod)
            # ...and the display type is the finest child's geometry.
            assert compute_lod_display_type(lod.children) == "gsplats"

    def test_compute_lod_display_type_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty children"):
            compute_lod_display_type([])


# ────────────────────────────────────────────────────────────────────────
# coarsen_dims via add_gsplats_from_data (3rd geometry — parallels Points/Lines)
# ────────────────────────────────────────────────────────────────────────


def _stacked_4d_gsplatdata(n_per=600, n_groups=3, seed=0):
    """Flat GSplatData: dim 0 categorical (0..G-1), dims 1-3 xyz shared."""
    from luxar.gsplats.lift import lift_points_to_gsplats

    rng = np.random.default_rng(seed)
    xyz = rng.normal(0, 5, (n_per, 3)).astype(np.float32)
    parts = [
        np.column_stack([np.full(n_per, g, np.float32), xyz]) for g in range(n_groups)
    ]
    pos = np.vstack(parts).astype(np.float32)
    return lift_points_to_gsplats(pos, np.full(len(pos), 0.5, np.float32))


def _gsplat_coarse_purity(store, name) -> float:
    """Max |fractional offset| of the kind=lod gsplat children's dim-0 centers."""
    grp = store[name]
    worst = 0.0
    for k in grp.keys():
        if not k.startswith("child_"):
            continue
        child = grp[k]
        if "centers" not in list(child.array_keys()):
            continue
        from luxar.encoding import ArrayDecoder

        c0 = ArrayDecoder().decode(child["centers"], grp)[:, 0]
        worst = max(worst, float(np.abs(c0 - np.round(c0)).max()))
    return worst


def _dims_4d():
    return Dimensions(
        [
            Dimension("coloring", categories=["0", "1", "2"], display=False),
            Dimension("x", display=True),
            Dimension("y", display=True),
            Dimension("z", display=True),
        ]
    )


class TestCoarsenDimsGsplats:
    def test_auto_default_groups_by_non_displayed(self, tmp_path) -> None:
        out = tmp_path / "g.luxar.zarr"
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=_dims_4d())
            scene.add_gsplats_from_data(
                "splats",
                _stacked_4d_gsplatdata(),
                lod_group=dict(compression_factor=4, levels=3, device="cpu"),
            )
        store = zarr.open(str(out), mode="r")
        assert _gsplat_coarse_purity(store, "splats") < 1e-4

    def test_all_dims_blends(self, tmp_path) -> None:
        out = tmp_path / "g.luxar.zarr"
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=_dims_4d())
            scene.add_gsplats_from_data(
                "splats",
                _stacked_4d_gsplatdata(n_per=800),
                lod_group=dict(
                    compression_factor=4, levels=3, device="cpu", coarsen_dims="all"
                ),
            )
        store = zarr.open(str(out), mode="r")
        assert _gsplat_coarse_purity(store, "splats") > 0.05

    def test_explicit_indices(self, tmp_path) -> None:
        out = tmp_path / "g.luxar.zarr"
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=_dims_4d())
            scene.add_gsplats_from_data(
                "splats",
                _stacked_4d_gsplatdata(),
                lod_group=dict(
                    compression_factor=4, levels=2, device="cpu", coarsen_dims=[1, 2, 3]
                ),
            )
        store = zarr.open(str(out), mode="r")
        assert _gsplat_coarse_purity(store, "splats") < 1e-4


class TestResolveCoarsenDimsResolver:
    """Direct unit tests for resolve_coarsen_dims (scene -> column indices)."""

    @staticmethod
    def _scene(dims):
        import types

        return types.SimpleNamespace(_dimensions=dims)

    def test_auto_default_groups_by_non_displayed(self):
        from luxar.core.group.lod.group import resolve_coarsen_dims

        r = resolve_coarsen_dims(self._scene(_dims_4d()), 4, None)
        assert r == (1, 2, 3)  # displayed dims; barrier = dim 0

    def test_pure_3d_returns_none(self):
        from luxar.core.group.lod.group import resolve_coarsen_dims

        assert (
            resolve_coarsen_dims(self._scene(Dimensions.default_3d()), 3, None) is None
        )

    def test_all_sentinel_returns_none(self):
        from luxar.core.group.lod.group import resolve_coarsen_dims

        assert resolve_coarsen_dims(self._scene(_dims_4d()), 4, "all") is None

    def test_names_resolve_when_aligned(self):
        from luxar.core.group.lod.group import resolve_coarsen_dims

        assert resolve_coarsen_dims(self._scene(_dims_4d()), 4, ["x", "y", "z"]) == (
            1,
            2,
            3,
        )

    def test_name_on_unaligned_raises(self):
        # n_cols (3) != scene ndim (4): a name could map to the wrong column.
        from luxar.core.group.lod.group import resolve_coarsen_dims

        with pytest.raises(ValueError, match="aligned"):
            resolve_coarsen_dims(self._scene(_dims_4d()), 3, ["x"])

    def test_display_sentinel_on_unaligned_raises(self):
        from luxar.core.group.lod.group import resolve_coarsen_dims

        with pytest.raises(ValueError, match="aligned"):
            resolve_coarsen_dims(self._scene(_dims_4d()), 3, "display")

    def test_out_of_range_index_raises(self):
        from luxar.core.group.lod.group import resolve_coarsen_dims

        with pytest.raises(ValueError, match="out of range"):
            resolve_coarsen_dims(self._scene(_dims_4d()), 4, [9])


class TestComposeAdditiveUnderSubstitutive:
    """The additive-ladder resolver for substitutive levels: a suppressed
    ladder is quiet when it was only the default, but warns when the caller
    explicitly asked for one that cannot be honoured."""

    @staticmethod
    def _resolve(spec):
        # Trivial resolver: echo the spec (a dict) back, None otherwise.
        return spec if isinstance(spec, dict) else None

    def test_default_suppression_is_quiet(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning would fail
            result = compose_additive_under_substitutive(
                None,
                resolve=self._resolve,
                name="node",
                suppress_reason="image_labels is set",
            )
        assert result is None

    def test_explicit_dict_suppression_warns(self):
        with pytest.warns(UserWarning, match="cannot be honoured"):
            result = compose_additive_under_substitutive(
                {"method": "random"},
                resolve=self._resolve,
                name="node",
                suppress_reason="image_labels is set",
            )
        assert result is None

    def test_explicit_true_suppression_warns(self):
        with pytest.warns(UserWarning, match="cannot be honoured"):
            compose_additive_under_substitutive(
                True,
                resolve=self._resolve,
                name="node",
                suppress_reason="line_type='indexed' edges are not preserved",
            )

    def test_opt_out_returns_none_without_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = compose_additive_under_substitutive(
                False,
                resolve=self._resolve,
                name="node",
                suppress_reason="image_labels is set",
            )
        assert result is None
