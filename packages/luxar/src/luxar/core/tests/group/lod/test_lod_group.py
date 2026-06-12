"""Tests for the geometry-agnostic kind=lod ``Group`` helpers (``lod/group.py``).

These cover the type-neutral machinery shared by Points, Lines, and GSplats:

- The standalone builder (``add_lod_group``): node creation, attr round-trip,
  child enumeration, and ``validate_lod_group()`` failure modes.
- The auto-derivation heuristic (``derive_min_pixel_sizes``): monotonicity
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
suite), not here.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.core.group.lod.group import (
    BASE_PIXEL_SIZE,
    compute_lod_display_type,
    derive_min_pixel_sizes,
    resolve_display_type,
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
            assert lod.attrs["selector"] == "pixel_size"
            assert lod.attrs["default_level"] == 0

        store = zarr.open(str(output_path), mode="r")
        attrs = store["multires"].attrs
        assert attrs["type"] == "group"
        assert attrs["kind"] == "lod"
        assert attrs["selector"] == "pixel_size"
        assert attrs["default_level"] == 0

    def test_add_lod_group_with_explicit_default_level(self, tmp_path) -> None:
        """The default_level kwarg lands on the zarr attrs."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lod_group("multires", default_level=2)

        store = zarr.open(str(output_path), mode="r")
        assert store["multires"].attrs["default_level"] == 2

    def test_add_lod_group_children_are_subgroups_with_min_pixel_size(
        self, tmp_path
    ) -> None:
        """Children added via inherited add_* methods carry min_pixel_size."""
        output_path = tmp_path / "test.luxar.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            lod.add_gsplats(
                "coarse",
                centers=_CENTERS,
                amplitudes=1.0,
                cholesky_factors=_CHOL,
                min_pixel_size=0.0,
            )
            lod.add_gsplats(
                "fine",
                centers=_CENTERS,
                amplitudes=1.0,
                cholesky_factors=_CHOL,
                min_pixel_size=100.0,
            )

        store = zarr.open(str(output_path), mode="r")
        grp = store["multires"]
        assert grp.attrs["type"] == "group"
        assert grp.attrs["kind"] == "lod"
        assert sorted(grp.keys()) == ["coarse", "fine"]
        assert grp["coarse"].attrs["type"] == "gsplats"
        assert grp["coarse"].attrs["min_pixel_size"] == 0.0
        assert grp["fine"].attrs["min_pixel_size"] == 100.0

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
            with pytest.raises(ValueError, match="selector must be 'pixel_size'"):
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

    def test_validate_missing_min_pixel_size_raises(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            # Child added without min_pixel_size — validator should catch it.
            lod.add_gsplats(
                "c0", centers=_CENTERS, amplitudes=1.0, cholesky_factors=_CHOL
            )
            with pytest.raises(ValueError, match="missing 'min_pixel_size'"):
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
                min_pixel_size=50.0,
            )
            lod.add_gsplats(
                "c1",
                centers=_CENTERS,
                amplitudes=1.0,
                cholesky_factors=_CHOL,
                min_pixel_size=10.0,  # < previous — invalid
            )
            with pytest.raises(ValueError, match="strictly greater"):
                validate_lod_group(lod)

    def test_validate_passes_for_well_formed_group(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            for i, mps in enumerate([0.0, 10.0, 50.0]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    min_pixel_size=mps,
                )
            # Should not raise.
            validate_lod_group(lod)
            assert [float(c.attrs["min_pixel_size"]) for c in lod.children] == [
                0.0,
                10.0,
                50.0,
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
            for i, mps in enumerate([0.0, 10.0]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=_CENTERS,
                    amplitudes=1.0,
                    cholesky_factors=_CHOL,
                    min_pixel_size=mps,
                )
            with pytest.raises(ValueError, match="default_level=99"):
                validate_lod_group(lod)


# ────────────────────────────────────────────────────────────────────────
# Auto-derivation heuristic
# ────────────────────────────────────────────────────────────────────────


class TestDeriveMinPixelSizes:
    """The √(N_finer / N_coarsest) auto-derivation."""

    def test_coarsest_is_zero(self) -> None:
        assert derive_min_pixel_sizes([100, 400, 1600])[0] == 0.0

    def test_sqrt_of_ratio_scaling(self) -> None:
        # 4× more elements → 2× the threshold (relative to BASE_PIXEL_SIZE)
        thresholds = derive_min_pixel_sizes([100, 400, 1600])
        assert thresholds[1] == BASE_PIXEL_SIZE * 2.0
        assert thresholds[2] == BASE_PIXEL_SIZE * 4.0

    def test_strict_monotonicity_enforced(self) -> None:
        # Even when input is non-increasing, output stays monotonic via the
        # relative ×1.1 bump (not a fixed +1 px).
        thresholds = derive_min_pixel_sizes([100, 100, 100])
        for i in range(1, len(thresholds)):
            assert thresholds[i] > thresholds[i - 1]

    def test_relative_nudge_is_proportional(self) -> None:
        # Equal-count levels separate by ×1.1, so the bump scales with the
        # threshold magnitude rather than a meaningless fixed pixel.
        thresholds = derive_min_pixel_sizes([100, 100, 100])
        # child1 = bps * sqrt(1) = BASE_PIXEL_SIZE; child2 = child1 * 1.1.
        assert thresholds[1] == BASE_PIXEL_SIZE
        assert thresholds[2] == pytest.approx(BASE_PIXEL_SIZE * 1.1)

    def test_base_pixel_size_override(self) -> None:
        # Overriding the anchor scales every non-zero threshold linearly.
        default = derive_min_pixel_sizes([100, 400, 1600])
        scaled = derive_min_pixel_sizes([100, 400, 1600], base_pixel_size=25.0)
        assert scaled[0] == 0.0
        assert scaled[1] == pytest.approx(default[1] * 2.5)
        assert scaled[2] == pytest.approx(default[2] * 2.5)

    def test_base_pixel_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            derive_min_pixel_sizes([100, 400], base_pixel_size=0.0)

    def test_extreme_count_ratio_monotonic(self) -> None:
        # 1 → 1,000,000 elements must still yield strictly increasing finite
        # thresholds (no overflow / non-monotonic artefacts).
        thresholds = derive_min_pixel_sizes([1, 1000, 1_000_000])
        for i in range(1, len(thresholds)):
            assert thresholds[i] > thresholds[i - 1]
        assert all(t == t for t in thresholds)  # no NaN

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            derive_min_pixel_sizes([])

    def test_zero_coarsest_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1 element"):
            derive_min_pixel_sizes([0, 10])


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
                min_pixel_size=0.0,
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
                min_pixel_size=0.0,
            )
            lod.add_gsplats(
                "fine",
                centers=_CENTERS,
                amplitudes=1.0,
                cholesky_factors=_CHOL,
                min_pixel_size=100.0,
            )
            # No homogeneity check fires — the group validates fine...
            validate_lod_group(lod)
            # ...and the display type is the finest child's geometry.
            assert compute_lod_display_type(lod.children) == "gsplats"

    def test_compute_lod_display_type_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty children"):
            compute_lod_display_type([])
