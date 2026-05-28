"""Tests for the kind=lod ``Group`` (formerly the ``LODGroup`` subclass).

Covers:

- The standalone builder (``add_lod_group``): node creation, attr round-trip,
  child enumeration, ``validate_lod_group()`` failure modes.
- The auto-derivation heuristic (``derive_min_pixel_sizes``): monotonicity
  and the √-of-ratio scaling.
- Constructor / writer-level validation: unsupported selector, negative
  default_level.

Convenience-API resolution (``lod_group=``/``additive_lod=``) is exercised
separately in ``test_group.py`` alongside the rest of the
``add_gsplats_from_data`` surface.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.core.lod import (
    BASE_PIXEL_SIZE,
    derive_min_pixel_sizes,
    validate_lod_group,
)
from luxar.io.compiler import LuxarZarrCompiler

# ────────────────────────────────────────────────────────────────────────
# Standalone builder — add_lod_group + child enumeration
# ────────────────────────────────────────────────────────────────────────


class TestAddLodGroup:
    """Round-trip a kind=lod ``Group`` built via the standalone builder."""

    def test_add_lod_group_writes_node_with_defaults(self, tmp_path) -> None:
        """Bare add_lod_group writes type=group + kind=lod with defaults."""
        output_path = tmp_path / "test.zarr"

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
        output_path = tmp_path / "test.zarr"

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lod_group("multires", default_level=2)

        store = zarr.open(str(output_path), mode="r")
        assert store["multires"].attrs["default_level"] == 2

    def test_add_lod_group_children_are_subgroups_with_min_pixel_size(
        self, tmp_path
    ) -> None:
        """Children added via inherited add_* methods carry min_pixel_size."""
        output_path = tmp_path / "test.zarr"
        centers = np.array([[0, 0, 0]], dtype=np.float32)
        chol = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            lod.add_gsplats(
                "coarse",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=chol,
                min_pixel_size=0.0,
            )
            lod.add_gsplats(
                "fine",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=chol,
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
        output_path = tmp_path / "test.zarr"

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
        with LuxarZarrCompiler(tmp_path / "x.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="selector must be 'pixel_size'"):
                scene.add_lod_group("multires", selector="distance")

    def test_negative_default_level_rejected(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="default_level must be >= 0"):
                scene.add_lod_group("multires", default_level=-1)

    def test_validate_empty_children_raises(self, tmp_path) -> None:
        with LuxarZarrCompiler(tmp_path / "x.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("empty")
            with pytest.raises(ValueError, match="has no children"):
                validate_lod_group(lod)

    def test_validate_missing_min_pixel_size_raises(self, tmp_path) -> None:
        centers = np.array([[0, 0, 0]], dtype=np.float32)
        chol = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "x.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            # Child added without min_pixel_size — validator should catch it.
            lod.add_gsplats(
                "c0", centers=centers, amplitudes=1.0, cholesky_factors=chol
            )
            with pytest.raises(ValueError, match="missing 'min_pixel_size'"):
                validate_lod_group(lod)

    def test_validate_non_monotonic_raises(self, tmp_path) -> None:
        centers = np.array([[0, 0, 0]], dtype=np.float32)
        chol = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "x.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            lod.add_gsplats(
                "c0",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=chol,
                min_pixel_size=50.0,
            )
            lod.add_gsplats(
                "c1",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=chol,
                min_pixel_size=10.0,  # < previous — invalid
            )
            with pytest.raises(ValueError, match="strictly greater"):
                validate_lod_group(lod)

    def test_validate_passes_for_well_formed_group(self, tmp_path) -> None:
        centers = np.array([[0, 0, 0]], dtype=np.float32)
        chol = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "x.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires")
            for i, mps in enumerate([0.0, 10.0, 50.0]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=centers,
                    amplitudes=1.0,
                    cholesky_factors=chol,
                    min_pixel_size=mps,
                )
            # Should not raise.
            validate_lod_group(lod)
            assert [
                float(c.attrs["min_pixel_size"]) for c in lod.children
            ] == [0.0, 10.0, 50.0]

    def test_validate_rejects_default_level_out_of_range(self, tmp_path) -> None:
        """``default_level`` past the number of children must fail validation.

        ``__init__`` cannot check this (children are added later); the
        check fires in ``validate_lod_group()``. Without it, a bad value
        sails into the on-disk zarr and only surfaces at viewer load time.
        """
        centers = np.array([[0, 0, 0]], dtype=np.float32)
        chol = np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32)
        with LuxarZarrCompiler(tmp_path / "x.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("multires", default_level=99)
            for i, mps in enumerate([0.0, 10.0]):
                lod.add_gsplats(
                    f"c{i}",
                    centers=centers,
                    amplitudes=1.0,
                    cholesky_factors=chol,
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
        # 4× more splats → 2× the threshold (relative to BASE_PIXEL_SIZE)
        thresholds = derive_min_pixel_sizes([100, 400, 1600])
        assert thresholds[1] == BASE_PIXEL_SIZE * 2.0
        assert thresholds[2] == BASE_PIXEL_SIZE * 4.0

    def test_strict_monotonicity_enforced(self) -> None:
        # Even when input is non-increasing, output stays monotonic by +1 px bumps.
        thresholds = derive_min_pixel_sizes([100, 100, 100])
        for i in range(1, len(thresholds)):
            assert thresholds[i] > thresholds[i - 1]

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            derive_min_pixel_sizes([])

    def test_zero_coarsest_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1 splat"):
            derive_min_pixel_sizes([0, 10])
