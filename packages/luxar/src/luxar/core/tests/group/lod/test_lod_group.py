"""Tests for the geometry-agnostic kind=lod ``Group`` helpers (``lod/group.py``).

These cover the type-neutral machinery shared by Points, Lines, and GSplats:

- The standalone builder (``add_lod_group``): node creation, attr round-trip,
  child enumeration, and ``validate_lod_group()`` failure modes.
- The auto-derivation heuristic (``coverage_fractions``): monotonicity
  and the screen-occupancy halving (area anchor 0.5, /2 per coarser level).
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

import math

import numpy as np
import pytest
import zarr

from luxar.conftest import read_ts_number_const, viewer_source
from luxar.core.dimensions import Dimension, Dimensions
from luxar.core.group import Group
from luxar.core.group.lod.group import (
    MAX_COVERAGE_FRACTION,
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
    compose_additive_under_substitutive,
    compute_lod_display_type,
    coverage_fractions,
    derive_coverage_fractions,
    is_partition_bound,
    partitioned_coverage_fractions,
    resolve_display_type,
    resolve_substitutive_axis,
    validate_lod_group,
)
from luxar.io.compiler import LuxarZarrCompiler

# Reusable tiny gsplat primitives (single splat at the origin).
_CENTERS = np.array([[0, 0, 0]], dtype=np.float32)
_CHOL = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)


def _add_single_lod_child(lod: Group) -> None:
    lod.add_gsplats(
        "level_0",
        centers=_CENTERS,
        amplitudes=1.0,
        cholesky_factors=_CHOL,
        coverage_fraction=0.0,
    )


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
            _add_single_lod_child(lod)

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
            lod = scene.add_lod_group("multires", default_level=2)
            _add_single_lod_child(lod)

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
            lod = g.add_lod_group("multires")
            _add_single_lod_child(lod)

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
            with pytest.raises(
                ValueError,
                match=r"selector must be one of \['coverage', 'screen-area'\]",
            ):
                scene.add_lod_group("multires", selector="distance")

    def test_screen_area_selector_accepted(self, tmp_path) -> None:
        """``selector="screen-area"`` (what every DERIVED ladder stamps) must be
        accepted by the hand-built path too, and land on the zarr attrs."""
        output_path = tmp_path / "x.luxar.zarr"
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires", selector="screen-area")
            assert lod.attrs["selector"] == "screen-area"
            _add_single_lod_child(lod)

        store = zarr.open(str(output_path), mode="r")
        assert store["multires"].attrs["selector"] == "screen-area"

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
        # A floor-legal ladder (coarsest 0.0) whose LATER entries descend —
        # otherwise the coarsest-must-be-0.0 check fires first and this test
        # would pass for the wrong reason.
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            for cname, cf in (("c0", 0.0), ("c1", 0.5), ("c2", 0.1)):
                lod.add_gsplats(
                    cname,
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,  # c2 < c1 — invalid
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

    def test_validate_rejects_coverage_fraction_above_the_ceiling(
        self, tmp_path
    ) -> None:
        """A hand-built ladder does not go through the explicit-list resolvers, so
        ``validate_lod_group`` is the only thing enforcing the documented
        ``[0, MAX_COVERAGE_FRACTION]`` bound.

        The bound is a POLICY, not an unreachability fact: a metric above 4.0 is
        perfectly attainable (it just needs the projected diagonal to exceed
        ``SCREEN_FILL_DIAGONAL_RATIO`` times the fitted screen axis, i.e. zoomed
        in past screen-filling). 4.0 is the point
        past which a threshold stops expressing anything a viewport-relative
        selector should encode, so authoring above it is refused rather than
        silently honoured."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            for i, cf in enumerate([0.0, MAX_COVERAGE_FRACTION + 0.5]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            with pytest.raises(ValueError, match=r"must lie in \[0, 4\]"):
                validate_lod_group(lod)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_validate_rejects_non_finite_coverage_fraction(
        self, bad: float, tmp_path
    ) -> None:
        """NaN is the dangerous one: every comparison against it is false, so it
        would pass BOTH the range check and the monotonicity check, then become
        ``prev`` and silence the monotonicity check for the rest of the ladder —
        yielding a store the viewer cannot select from predictably. ±inf would
        likewise satisfy strict ascent."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            # A descending tail AFTER the bad value: with the value accepted the
            # ladder would validate despite 0.5 < 0.9, which is the real hazard.
            for i, cf in enumerate([0.0, bad, 0.9, 0.5]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            with pytest.raises(ValueError, match="not a finite number"):
                validate_lod_group(lod)

    def test_validate_rejects_unknown_selector_attr(self, tmp_path) -> None:
        """A PRESENT selector outside the vocabulary is rejected — node attrs
        are mutable, so a modified/imported group could otherwise pass
        validation and serialize an invalid selector. Only a MISSING selector
        defaults to legacy (the viewer loader's own fallback)."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            for i, cf in enumerate([0.0, 0.5]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            lod.attrs["selector"] = "pixel_size"  # mutate past the builder gate
            with pytest.raises(ValueError, match="must be one of"):
                validate_lod_group(lod)

    def test_validate_rejects_nonzero_coarsest_floor(self, tmp_path) -> None:
        """The coarsest child must be EXACTLY 0.0 — the always-eligible floor
        the format requires. A strictly-ascending ladder like [0.25, 0.5] used
        to validate, leaving no eligible child below 0.25 occupancy (what
        rendered there depended on selector fallback, not the ladder)."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires", selector="screen-area")
            for i, cf in enumerate([0.25, 0.5]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            with pytest.raises(ValueError, match="must be exactly 0.0"):
                validate_lod_group(lod)

    def test_validate_rejects_negative_coverage_fraction(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            for i, cf in enumerate([-0.1, 1.0]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            with pytest.raises(ValueError, match=r"must lie in \[0, 4\]"):
                validate_lod_group(lod)

    def test_validate_accepts_the_partitioned_anchor_at_the_ceiling(
        self, tmp_path
    ) -> None:
        """A derived partition-bound (tile) ladder anchors its finest at
        ``PARTITION_FINEST_AREA`` (1.0 — the tile alone fills the screen), and
        that must validate. Note ``validate_lod_group`` still allows AUTHORED
        lists up to ``MAX_COVERAGE_FRACTION`` (4.0, legacy diagonal units), so
        the derived area ladder sits comfortably inside the bound."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            for i, cf in enumerate(partitioned_coverage_fractions([25, 100, 400])):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            validate_lod_group(lod)  # must not raise
            assert [float(c.attrs["coverage_fraction"]) for c in lod.children] == [
                0.0,
                0.5,
                PARTITION_FINEST_AREA,
            ]

    def test_validate_screen_area_rejects_values_above_the_area_ceiling(
        self, tmp_path
    ) -> None:
        """The ceiling is SELECTOR-DEPENDENT. Under ``selector="screen-area"``
        thresholds are literal screen-area fractions, so anything above the
        fills-screen area (``PARTITION_FINEST_AREA`` = 1.0) is out of contract
        — a value like 2.0 validated fine under the legacy 4.0 bound but would
        write a non-conforming v3.4 store whose level can effectively never be
        held correctly. The legacy-units value in this ladder (2.0) is exactly
        the kind that must now be refused."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires", selector="screen-area")
            for i, cf in enumerate([0.0, 2.0]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            with pytest.raises(
                ValueError, match=r"must lie in \[0, 1\] under selector='screen-area'"
            ):
                validate_lod_group(lod)

    def test_validate_screen_area_accepts_the_fills_screen_boundary(
        self, tmp_path
    ) -> None:
        """Exactly 1.0 — the fills-screen tile anchor every derived partition
        ladder ends on — must validate under ``selector="screen-area"`` (the
        bound is inclusive; an exclusive bound would refuse every derived
        tile ladder)."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires", selector="screen-area")
            for i, cf in enumerate([0.0, 0.5, PARTITION_FINEST_AREA]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            validate_lod_group(lod)  # must not raise

    def test_validate_legacy_selector_keeps_the_diagonal_ceiling(
        self, tmp_path
    ) -> None:
        """The same 2.0 that screen-area refuses stays VALID under the legacy
        ``"coverage"`` selector (and a missing selector attr falls back to it),
        whose authored-list ceiling remains ``MAX_COVERAGE_FRACTION`` = 4.0 —
        proving the two caps genuinely branch on the selector rather than one
        of them having tightened globally."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")  # default selector="coverage"
            for i, cf in enumerate([0.0, 2.0, MAX_COVERAGE_FRACTION]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    coverage_fraction=cf,
                )
            validate_lod_group(lod)  # must not raise


# ────────────────────────────────────────────────────────────────────────
# Auto-derivation heuristic
# ────────────────────────────────────────────────────────────────────────


A = WHOLE_OBJECT_FINEST_ANCHOR  # 0.5 — half the screen AREA


class TestCoverageFractions:
    """SCREEN-OCCUPANCY HALVING: finest anchored at half the screen area (0.5),
    each coarser level halves, coarsest 0.0 — independent of counts."""

    def test_coarsest_is_zero(self) -> None:
        assert coverage_fractions([100, 400, 1600])[0] == 0.0

    def test_finest_is_half_screen_area_anchor(self) -> None:
        # selector="screen-area": a threshold is a literal screen-area
        # fraction, so "half the screen" is simply 0.5.
        assert coverage_fractions([100, 400, 1600])[-1] == pytest.approx(
            WHOLE_OBJECT_FINEST_ANCHOR
        )
        assert WHOLE_OBJECT_FINEST_ANCHOR == 0.5

    def test_area_halving_spacing(self) -> None:
        # One halving of occupied screen AREA per level = /2 on the area
        # fraction: [..., 0.125, 0.25, 0.5].
        fractions = coverage_fractions([100, 400, 1600])
        assert fractions == pytest.approx([0.0, A / 2.0, A])
        assert coverage_fractions([1, 2, 3, 4]) == pytest.approx(
            [0.0, A / 4.0, A / 2.0, A]
        )

    def test_validated_numeric_example(self) -> None:
        # Counts set only the LENGTH: any 5-level ladder derives the same
        # halving thresholds (count-independence is the point of the rule —
        # a count ratio is blind to element size, overlap, and intent).
        got = coverage_fractions([89894, 359865, 1443108, 5801956, 23368376])
        assert got == pytest.approx([0.0, A / 8.0, A / 4.0, A / 2.0, A])

    def test_two_level_ladder_is_zero_anchor(self) -> None:
        # Any two-level ladder → [0.0, 0.5]: coarse below half the screen
        # area, full detail above.
        assert coverage_fractions([100, 400]) == pytest.approx([0.0, A])
        assert coverage_fractions([50, 200]) == pytest.approx([0.0, A])

    def test_strict_monotonicity_by_construction(self) -> None:
        # Halving is strictly ascending regardless of the counts — equal or
        # non-monotone count ladders can no longer produce duplicates.
        for counts in ([100, 100, 100], [100, 1000, 10], [10, 0, 20]):
            fractions = coverage_fractions(counts)
            for i in range(1, len(fractions)):
                assert fractions[i] > fractions[i - 1]
            assert all(0.0 <= f <= A + 1e-9 for f in fractions)

    def test_equal_counts_derive_the_same_halving(self) -> None:
        # Under the count-ratio derivation, equal counts were a degenerate
        # case needing the guard's nudge; under halving they are just another
        # 3-level ladder.
        assert coverage_fractions([100, 100, 100]) == pytest.approx([0.0, A / 2.0, A])

    def test_equal_count_tail_is_not_special(self) -> None:
        # The old count-ratio regression case ([10, 1000, 1000] → 1.1 > anchor)
        # cannot exist under halving: counts do not reach the thresholds.
        assert coverage_fractions([10, 1000, 1000]) == pytest.approx([0.0, A / 2.0, A])

    def test_non_monotone_counts_are_not_special(self) -> None:
        # A count ladder that decreases toward the finest derived above the
        # anchor under sqrt ratios; halving never reads the counts.
        assert coverage_fractions([100, 1000, 10]) == pytest.approx([0.0, A / 2.0, A])

    def test_derived_values_round_trip_explicit_validator(self) -> None:
        # The derived fractions must be accepted verbatim by the explicit
        # coverage_fractions= validator: derived output stays in
        # [0, 0.5], a strict subset of [0, MAX_COVERAGE_FRACTION].
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
        # A zero-count INTERMEDIATE level no longer reaches the thresholds at
        # all (halving is count-independent); only an empty FINEST rejects.
        cf = coverage_fractions([10, 0, 20])
        assert cf == pytest.approx([0.0, A / 2.0, A])
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
                elements=100,
                name="node",
                slices=1,
                suppress_reason="image_labels is set",
            )
        assert result is None

    def test_explicit_dict_suppression_warns(self):
        with pytest.warns(UserWarning, match="cannot be honoured"):
            result = compose_additive_under_substitutive(
                {"method": "random"},
                resolve=self._resolve,
                elements=100,
                name="node",
                slices=1,
                suppress_reason="image_labels is set",
            )
        assert result is None

    def test_suppression_outcome_can_describe_finest_only(self):
        with pytest.warns(UserWarning, match="coarse levels keep their ladder"):
            result = compose_additive_under_substitutive(
                {"method": "random"},
                resolve=self._resolve,
                elements=100,
                name="node",
                slices=1,
                suppress_reason="image_labels is set",
                suppression_outcome=(
                    "the finest level will load all-at-once; coarse levels keep "
                    "their ladder where one applies."
                ),
            )
        assert result is None

    def test_quiet_suppression_uses_custom_outcome(self, capsys):
        result = compose_additive_under_substitutive(
            None,
            resolve=self._resolve,
            elements=100,
            name="node",
            slices=1,
            suppress_reason="image_labels is set",
            suppression_outcome=(
                "the finest level will load all-at-once; coarse levels keep their "
                "ladder where one applies."
            ),
        )

        assert result is None
        assert (
            "'node': streaming ladder skipped (image_labels is set); "
            "the finest level will load all-at-once; coarse levels keep their "
            "ladder where one applies."
        ) in capsys.readouterr().out

    def test_explicit_true_suppression_warns(self):
        with pytest.warns(UserWarning, match="cannot be honoured"):
            compose_additive_under_substitutive(
                True,
                resolve=self._resolve,
                elements=100,
                name="node",
                slices=1,
                suppress_reason="line_type='indexed' edges are not preserved",
            )

    def test_opt_out_returns_none_without_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = compose_additive_under_substitutive(
                False,
                resolve=self._resolve,
                elements=100,
                name="node",
                slices=1,
                suppress_reason="image_labels is set",
            )
        assert result is None


# ────────────────────────────────────────────────────────────────────────
# partitioned_coverage_fractions — the fills-screen anchor for tiled ladders
# ────────────────────────────────────────────────────────────────────────


class TestPartitionedCoverageFractions:
    """A ladder bound to a spatial partition takes the fills-screen anchor.

    ``coverage_fractions`` anchors the finest at half the screen area (0.5).
    That is right for a WHOLE-OBJECT ladder and wrong for a per-tile one (a
    tile projects to a fraction of the object, so every tile would sit on its
    finest level at whole-object framing). The partitioned variant rescales by
    ×2 so the finest lands on ``PARTITION_FINEST_AREA`` = 1.0 — the tile alone
    occupying the whole screen.
    """

    def test_is_the_derived_ladder_scaled_by_the_ceiling(self) -> None:
        # Partition-bound = the whole-object halving re-anchored at
        # fills-screen: ×(PARTITION_FINEST_AREA / WHOLE_OBJECT_FINEST_ANCHOR)
        # = ×2 in area units.
        counts = [25, 100, 400]
        assert coverage_fractions(counts) == pytest.approx([0.0, A / 2.0, A])
        assert partitioned_coverage_fractions(counts) == pytest.approx([0.0, 0.5, 1.0])

    def test_anchors_finest_at_the_ceiling_and_coarsest_at_zero(self) -> None:
        for counts in ([100, 400], [25, 100, 400], [1, 10, 100, 1000]):
            out = partitioned_coverage_fractions(counts)
            assert out[0] == 0.0
            assert out[-1] == pytest.approx(PARTITION_FINEST_AREA)

    def test_scaling_preserves_strict_ascent_and_the_explicit_bound(self) -> None:
        # Including the degenerate ladders the monotonicity guard has to repair —
        # scaling is by a power of two, so ascent and the bound both survive.
        for counts in (
            [10, 1000, 1000],  # equal-count tail
            [100, 100, 100],  # all-equal
            [10, 0, 20],  # zero-count intermediate
            [100, 1000, 10],  # non-monotone (derives > 1 pre-cap)
            [1, 1000, 1_000_000],  # extreme ratio
        ):
            out = partitioned_coverage_fractions(counts)
            assert all(out[i] > out[i - 1] for i in range(1, len(out))), counts
            assert all(0.0 <= f <= PARTITION_FINEST_AREA for f in out), counts

    def test_single_level_is_just_the_floor(self) -> None:
        assert partitioned_coverage_fractions([42]) == [0.0]

    def test_output_round_trips_through_the_explicit_validator(self) -> None:
        # A partition-bound ladder must be expressible as an explicit list:
        # its 1.0 area anchor sits well inside the legacy explicit bound
        # [0, MAX_COVERAGE_FRACTION].
        derived = partitioned_coverage_fractions([25, 100, 400])
        resolved = resolve_substitutive_axis({"coverage_fractions": derived}, "Points")
        assert resolved is not None
        assert resolved["coverage_fractions"] == pytest.approx(derived)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_explicit_list_rejects_non_finite_entries(self, bad: float) -> None:
        """The shared strict-ascent check is the only gate on an explicit list, and
        ``bad <= prev`` is false for NaN — so without an explicit finite test a NaN
        entry would be accepted and would then silence the ascent check for every
        entry after it (here the descending 0.9 → 0.5 tail)."""
        with pytest.raises(ValueError, match="finite"):
            resolve_substitutive_axis(
                {"coverage_fractions": [0.0, bad, 0.9, 0.5]}, "Points"
            )


# ────────────────────────────────────────────────────────────────────────
# is_partition_bound / derive_coverage_fractions — anchor choice by ancestry
# ────────────────────────────────────────────────────────────────────────


class TestIsPartitionBound:
    """Detect a ``kind=partition`` ancestor of a ladder's insertion point.

    The node handed in is the future lod group's PARENT, so the partition wrapper
    itself counts; and anything between the partition and the ladder (a plain
    ``add_group``) leaves the ladder inside one tile, so that counts too.
    """

    def test_the_partition_group_itself(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            part = scene.add_partition_group(
                "tiles", display_type="points", max_elements=1000
            )
            assert is_partition_bound(part) is True

    def test_plain_group_nested_under_the_partition(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            part = scene.add_partition_group(
                "tiles", display_type="points", max_elements=1000
            )
            inner = part.add_group("part_0").add_group("deeper")
            assert is_partition_bound(inner) is True

    def test_lod_group_under_the_partition(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            part = scene.add_partition_group(
                "tiles", display_type="gsplats", max_elements=1000
            )
            assert is_partition_bound(part.add_lod_group("ladder")) is True

    def test_scene_root_is_not_bound(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            assert is_partition_bound(scene) is False

    def test_plain_group_with_no_partition_anywhere(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            assert is_partition_bound(scene.add_group("a").add_group("b")) is False

    def test_partition_nested_under_a_plain_group(self, tmp_path) -> None:
        """The walk is up the whole chain, not just one hop."""
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            outer = scene.add_group("outer")
            part = outer.add_partition_group(
                "tiles", display_type="points", max_elements=1000
            )
            assert is_partition_bound(part) is True
            assert is_partition_bound(outer) is False


class TestDeriveCoverageFractions:
    """Which ladder the adders' single chokepoint returns, per insertion point."""

    COUNTS = [25, 100, 400]

    def test_scene_root_gets_the_whole_object_ladder(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            out = derive_coverage_fractions(self.COUNTS, scene, name="cloud")
        assert out == pytest.approx(coverage_fractions(self.COUNTS))
        assert out[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)

    def test_under_a_partition_gets_the_fills_screen_ladder(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            part = scene.add_partition_group(
                "tiles", display_type="points", max_elements=1000
            )
            out = derive_coverage_fractions(self.COUNTS, part, name="tile_0")
        assert out == pytest.approx(partitioned_coverage_fractions(self.COUNTS))
        assert out[-1] == pytest.approx(PARTITION_FINEST_AREA)

    def test_plain_group_between_partition_and_ladder_still_bound(
        self, tmp_path
    ) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            part = scene.add_partition_group(
                "tiles", display_type="points", max_elements=1000
            )
            inner = part.add_group("group_in_tile")
            out = derive_coverage_fractions(self.COUNTS, inner, name="tile_0")
        assert out[-1] == pytest.approx(PARTITION_FINEST_AREA)

    def test_plain_group_outside_a_partition_is_whole_object(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            out = derive_coverage_fractions(
                self.COUNTS, scene.add_group("plain"), name="cloud"
            )
        assert out[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)


# ────────────────────────────────────────────────────────────────────────
# Cross-language constant lock
# ────────────────────────────────────────────────────────────────────────


def test_max_coverage_fraction_matches_the_viewer_fill_factor() -> None:
    """``MAX_COVERAGE_FRACTION`` must stay ``SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR``.

    The viewer compares every ``coverage_fraction`` against
    ``projectedDiagonalPx / (FILL_FACTOR * fittedAxisPx)`` (issue #1410 moved the
    denominator from the viewport DIAGONAL to the fitted screen AXIS —
    ``min(viewport.width, viewport.height)`` — so the metric stays invariant
    across aspect ratio, not just viewport size). A screen-filling object's
    projected diagonal is no longer simply the fitted axis (that identity only
    held for the old diagonal normalisation); it is
    ``SCREEN_FILL_DIAGONAL_RATIO`` times it — and only APPROXIMATELY, since that
    ratio (``hypot(aspect, 1) / min(aspect, 1)``) is itself aspect-dependent, so
    the metric it yields (the ratio ÷ ``FILL_FACTOR``) measures 2.83 at 1:1,
    4.08 at 16:9 and 5.15 at a real 21:9 panel (2560×1080) — exactly 4.0 only at
    aspect √3 ≈ 1.73; see the ``FILL_FACTOR`` doc in
    ``lod-group-registry.ts``. So a
    screen-filling object produces a metric of *approximately*
    ``SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR`` at the mainstream aspect
    ratios the constant targets — the largest value an authored threshold can
    usefully take. The two constants live in different languages with no
    build-time link, so this test IS the link: it parses the TypeScript
    source. Prose comments on both sides are not enough.
    """
    registry = viewer_source("src/scene/lod-group-registry.ts")
    source = registry.read_text(encoding="utf-8")

    fill_factor = read_ts_number_const(source, "FILL_FACTOR")
    screen_fill_diagonal_ratio = read_ts_number_const(
        source, "SCREEN_FILL_DIAGONAL_RATIO"
    )
    assert fill_factor > 0.0, f"FILL_FACTOR must be positive, read {fill_factor}"
    ref_aspect_ratio = math.hypot(16, 9) / 9
    assert screen_fill_diagonal_ratio == pytest.approx(ref_aspect_ratio, rel=0.03), (
        f"SCREEN_FILL_DIAGONAL_RATIO ({screen_fill_diagonal_ratio}) should stay close "
        f"to the 16:9-reference geometric value it stands for (hypot(16, 9) / 9 = "
        f"{ref_aspect_ratio:.4f}); otherwise it and FILL_FACTOR could compensate for "
        "each other below (e.g. FILL_FACTOR=0.25 + SCREEN_FILL_DIAGONAL_RATIO=1 keeps "
        "the product test green while every viewer switch point silently moves 2x) "
        "and this test would stop being a real cross-language lock."
    )
    expected = screen_fill_diagonal_ratio / fill_factor
    assert MAX_COVERAGE_FRACTION == pytest.approx(expected), (
        f"MAX_COVERAGE_FRACTION ({MAX_COVERAGE_FRACTION}) must equal "
        f"SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR "
        f"({screen_fill_diagonal_ratio} / {fill_factor} = {expected}). These "
        "constants are one decision expressed twice: FILL_FACTOR and "
        "SCREEN_FILL_DIAGONAL_RATIO anchor the viewer's coverage metric and "
        "MAX_COVERAGE_FRACTION bounds what a scene author may write. Change "
        f"BOTH — {registry} and the constant in luxar/core/group/lod/group.py — "
        "or the bound stops meaning 'may be required to fill the screen, at "
        "most'."
    )


# ---------------------------------------------------------------------------
# A default composed ladder must be sized against the RESIDENT SLICE (#2374).
# ---------------------------------------------------------------------------


def test_default_composed_ladder_refuses_to_be_sized_without_a_slice_count():
    """The guard is the SIGNATURE, not a warning.

    This function once took no arguments, so a caller could not supply the one
    fact that decides whether its whole-node download budget is right — and
    three of the four ladder-sizing seams in the tree protected against that
    while this one structurally could not. Requiring the argument is what makes
    the mistake unwriteable rather than merely documented.
    """
    from luxar.core.group.lod.group import default_composed_additive_lod

    with pytest.raises(TypeError):
        default_composed_additive_lod()  # type: ignore[call-arg]


def test_default_composed_ladder_uses_a_resident_share_without_losing_the_ladder():
    from luxar.core.group.lod.group import default_composed_additive_lod
    from luxar.utils.lod_breakpoints import parse_stream_chunk, stream_cuts

    unsliced = parse_stream_chunk(
        default_composed_additive_lod(elements=60_000, slices=1)["counts"]
    )
    sliced = parse_stream_chunk(
        default_composed_additive_lod(elements=60_000, slices=30)["counts"]
    )

    # slices=1 reproduces the historical whole-node value exactly, so an
    # unsliced node's output is unchanged by this plumbing.
    assert unsliced == 39_062
    # A sliced node keeps a useful resident share without scaling the chunk past
    # the node and silently collapsing its ladder to one all-at-once commit.
    assert sliced == unsliced
    assert len(stream_cuts(60_000, sliced)) > 1


def test_default_composed_ladder_rejects_a_share_over_the_commit_ceiling():
    from luxar.core.group.lod.group import default_composed_additive_lod

    with pytest.raises(ValueError, match="12.5% first rung.*900,000-element"):
        default_composed_additive_lod(elements=7_200_001, slices=2)


def test_resident_slice_count_counts_occurring_combinations():
    """Distinct OCCURRING combinations, not the product of per-axis cardinality."""
    from luxar.core.group.lod.group import resident_slice_count

    class _Dims:
        ndim = 3
        displayed = [0, 1]

    class _Scene:
        dimensions = _Dims()

    # Hidden column 2 takes three values; columns 0-1 are displayed and ignored.
    positions = np.array(
        [[0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [0.0, 0.0, 7.0], [2.0, 2.0, 9.0]]
    )
    assert resident_slice_count(_Scene(), positions) == 3


def test_resident_slice_count_treats_unaligned_positions_as_unsliced():
    """1 is the SAFE answer: a scene-dim index is only a centre column when the
    positions align 1:1, so an unaligned node reproduces the historical
    whole-node sizing rather than inventing a divisor from a mis-mapped column.
    """
    from luxar.core.group.lod.group import resident_slice_count

    class _Dims:
        ndim = 4  # scene says 4D...
        displayed = [0, 1]

    class _Scene:
        dimensions = _Dims()

    positions = np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 7.0]])  # ...node is 3-column
    assert resident_slice_count(_Scene(), positions) == 1
    assert resident_slice_count(None, positions) == 1
