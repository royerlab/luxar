"""Tests for PR δ small refinements:

- compiler-side ``display_type`` back-fill on kind=lod groups.
- ``base_pixel_size`` knob (on ``add_lod_group`` AND ``lod_group=dict()``).
- ``energy:`` breakpoints + ``salience_kind='energy'`` on Points + Lines.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.gsplats import (
    BASE_PIXEL_SIZE,
    derive_min_pixel_sizes,
    resolve_substitutive_axis,
)
from luxar.core.group.lod.lines import (
    _compute_lines_energy,
    make_additive_lod_lines,
    resolve_additive_axis_lines,
)
from luxar.core.group.lod.points import (
    _compute_points_energy,
    _energy_breakpoints_to_counts,
    _perceptual_luminance,
    make_additive_lod_points,
    resolve_additive_axis_points,
)
from luxar.io.compiler import LuxarZarrCompiler

# ────────────────────────────────────────────────────────────────────────
# 1. Compiler-side display_type back-fill
# ────────────────────────────────────────────────────────────────────────


class TestDisplayTypeBackfill:
    def test_explicit_lod_group_gets_display_type_filled(self, tmp_path):
        """User-built lod_group without display_type → compiler fills it."""
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group("hand_built")
            # All children are points → expected display_type is 'points'.
            pos_coarse = np.random.RandomState(0).rand(50, 3).astype(np.float32)
            pos_fine = np.random.RandomState(1).rand(500, 3).astype(np.float32)
            lod.add_points("level_0", pos_coarse, min_pixel_size=0.0)
            lod.add_points("level_1", pos_fine, min_pixel_size=50.0)

        store = zarr.open(str(tmp_path / "t.zarr"), mode="r")
        assert store["hand_built"].attrs["display_type"] == "points"

    def test_explicit_user_display_type_not_overwritten(self, tmp_path):
        """If user authored display_type, the compiler must not touch it."""
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            lod = scene.add_lod_group(
                "with_explicit", display_type="custom_marker"
            )
            pos = np.random.RandomState(0).rand(50, 3).astype(np.float32)
            lod.add_points("level_0", pos, min_pixel_size=0.0)

        store = zarr.open(str(tmp_path / "t.zarr"), mode="r")
        assert (
            store["with_explicit"].attrs["display_type"] == "custom_marker"
        ), "user-authored display_type must not be overwritten"


# ────────────────────────────────────────────────────────────────────────
# 2. base_pixel_size knob
# ────────────────────────────────────────────────────────────────────────


class TestBasePixelSizeKnob:
    def test_derive_uses_default_when_unset(self):
        # Default BASE_PIXEL_SIZE = 10.0; thresholds = [0, 10*sqrt(4)=20].
        thresholds = derive_min_pixel_sizes([100, 400])
        assert thresholds[0] == 0.0
        assert thresholds[1] == pytest.approx(BASE_PIXEL_SIZE * 2.0)

    def test_derive_uses_override(self):
        thresholds = derive_min_pixel_sizes([100, 400], base_pixel_size=20.0)
        # Override = 20 → thresholds = [0, 20*sqrt(4)=40].
        assert thresholds[1] == pytest.approx(40.0)

    def test_derive_rejects_non_positive(self):
        with pytest.raises(ValueError, match="base_pixel_size"):
            derive_min_pixel_sizes([100, 200], base_pixel_size=0.0)
        with pytest.raises(ValueError, match="base_pixel_size"):
            derive_min_pixel_sizes([100, 200], base_pixel_size=-5.0)

    def test_add_lod_group_kwarg_stored_on_attrs(self, tmp_path):
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lod_group("custom_bps", base_pixel_size=25.0)
        store = zarr.open(str(tmp_path / "t.zarr"), mode="r")
        assert store["custom_bps"].attrs["base_pixel_size"] == 25.0

    def test_add_lod_group_rejects_non_positive(self, tmp_path):
        with LuxarZarrCompiler(tmp_path / "t.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="base_pixel_size"):
                scene.add_lod_group("bad", base_pixel_size=0.0)

    def test_resolve_substitutive_axis_passes_through_bps(self):
        # We don't need a real GSplatData — the dict branch with
        # n_substitutive=1 falls through to compute, but base_pixel_size
        # extraction happens before that. Mock minimally.
        class FakeData:
            n_substitutive = 1
            default_substitutive = 0

            def at_substitutive(self, i):
                return self

        data = FakeData()
        # With spec=None → no base_pixel_size returned.
        _, _, bps = resolve_substitutive_axis(data, None)
        assert bps is None

    def test_resolve_substitutive_axis_dict_base_pixel_size(self):
        class FakeData:
            n_substitutive = 2  # stored pyramid; recompute=False path

            def at_substitutive(self, i):
                return self

        data = FakeData()
        _, _, bps = resolve_substitutive_axis(
            data, {"base_pixel_size": 25.0}
        )
        assert bps == 25.0

    def test_resolve_substitutive_axis_rejects_invalid_bps(self):
        class FakeData:
            n_substitutive = 2

            def at_substitutive(self, i):
                return self

        data = FakeData()
        with pytest.raises(ValueError, match="base_pixel_size"):
            resolve_substitutive_axis(data, {"base_pixel_size": 0.0})


# ────────────────────────────────────────────────────────────────────────
# 3a. Points energy: scoring + breakpoints
# ────────────────────────────────────────────────────────────────────────


class TestPointsEnergy:
    def test_perceptual_luminance_rec709(self):
        # Pure red gets the lowest weight per Rec.709.
        c = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        lum = _perceptual_luminance(c)
        np.testing.assert_allclose(lum, [0.2126, 0.7152, 0.0722], atol=1e-6)

    def test_compute_points_energy_with_colors(self):
        radii = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # All-white colors → uniform luminance 1; energy ∝ radii^3.
        colors = np.ones((3, 3), dtype=np.float32)
        energy = _compute_points_energy(3, radii, colors, None)
        # 1 * 1, 1 * 8, 1 * 27
        np.testing.assert_allclose(energy, [1.0, 8.0, 27.0])

    def test_compute_points_energy_falls_back_to_scalars(self):
        radii = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        scalars = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        energy = _compute_points_energy(3, radii, None, scalars)
        np.testing.assert_allclose(energy, [0.5, 1.0, 2.0])

    def test_compute_points_energy_no_colors_no_scalars(self):
        radii = np.array([1.0, 2.0], dtype=np.float32)
        energy = _compute_points_energy(2, radii, None, None)
        # luminance defaults to 1 → energy = radii^3.
        np.testing.assert_allclose(energy, [1.0, 8.0])

    def test_compute_points_energy_clips_negatives(self):
        # Negative luminance (e.g. signed scalars) clipped to 0.
        radii = np.array([1.0, 1.0], dtype=np.float32)
        scalars = np.array([-0.5, 0.5], dtype=np.float32)
        energy = _compute_points_energy(2, radii, None, scalars)
        np.testing.assert_allclose(energy, [0.0, 0.5])

    def test_energy_breakpoints_to_counts_basic(self):
        # 5 elements all equal energy. Fractions [0.4, 1.0] →
        # cumulative counts [2, 5]. (idx where cumsum crosses fraction.)
        energy = np.ones(5)
        perm = np.arange(5, dtype=np.intp)
        counts = _energy_breakpoints_to_counts(energy, perm, [0.4, 1.0])
        assert counts == [2, 5]

    def test_energy_breakpoints_zero_total_falls_back(self):
        # All energies zero → fall back to equal-count cuts.
        energy = np.zeros(10)
        perm = np.arange(10, dtype=np.intp)
        counts = _energy_breakpoints_to_counts(energy, perm, [0.5, 0.9])
        assert counts == [5, 9]

    def test_make_additive_lod_with_energy_breakpoints(self):
        # Sort-by-radius will place largest first; energy fractions
        # then carve cumulative energy.
        rng = np.random.RandomState(0)
        n = 100
        pos = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        radii = rng.uniform(0.1, 1.0, n).astype(np.float32)
        levels = make_additive_lod_points(
            pos,
            radii=radii,
            method="salience",
            counts="energy:0.5,0.9,1.0",
            salience_kind="energy",
        )
        # Total energy preserved across levels.
        total = sum(L.size for L in levels)
        assert total == n
        # At least two levels emitted.
        assert len(levels) >= 2

    def test_salience_kind_energy_outranks_salience_size_for_dark_big(self):
        """Black big sphere outranks bright tiny sphere under
        salience_kind='energy', and vice-versa under 'size'."""
        # 2 points: dark large, bright small.
        pos = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        radii = np.array([5.0, 1.0], dtype=np.float32)  # large, small
        # Make dark/big actually dimmer than bright/small. With radius
        # diff 5×, volume ratio = 125×; need luminance ratio >125 to
        # invert.
        colors = np.array(
            [[0.01, 0.01, 0.01], [1.0, 1.0, 1.0]], dtype=np.float32
        )

        # salience_kind='size' (default): index 0 ranks first (larger radius).
        size_levels = make_additive_lod_points(
            pos, radii=radii, method="salience", counts=[1], salience_kind="size"
        )
        assert size_levels[0][0] == 0

        # salience_kind='energy': low-luminance × big-volume here is
        # 0.01 * 125 = 1.25 vs 1.0 * 1 = 1.0 → big-dim still wins.
        energy_levels = make_additive_lod_points(
            pos,
            radii=radii,
            method="salience",
            counts=[1],
            colors=colors,
            salience_kind="energy",
        )
        # Big-dim has slightly higher energy than small-bright above.
        assert energy_levels[0][0] == 0

    def test_resolver_accepts_energy_breakpoints(self):
        spec = resolve_additive_axis_points(
            {"counts": "energy:0.5,0.9,1.0", "method": "salience"}
        )
        assert spec["counts"] == "energy:0.5,0.9,1.0"

    def test_resolver_accepts_breakpoints_alias(self):
        spec = resolve_additive_axis_points(
            {"breakpoints": "energy:0.5,1.0", "method": "random"}
        )
        assert spec["counts"] == "energy:0.5,1.0"

    def test_resolver_rejects_both_counts_and_breakpoints(self):
        with pytest.raises(ValueError, match="counts.*breakpoints"):
            resolve_additive_axis_points(
                {"counts": [10, 20], "breakpoints": "energy:0.5"}
            )

    def test_resolver_accepts_salience_kind_energy(self):
        spec = resolve_additive_axis_points(
            {"method": "salience", "salience_kind": "energy"}
        )
        assert spec["salience_kind"] == "energy"

    def test_resolver_rejects_invalid_salience_kind(self):
        with pytest.raises(ValueError, match="salience_kind"):
            resolve_additive_axis_points({"salience_kind": "bogus"})


# ────────────────────────────────────────────────────────────────────────
# 3b. Lines energy: scoring + breakpoints
# ────────────────────────────────────────────────────────────────────────


class TestLinesEnergy:
    def test_compute_lines_energy_segments(self):
        # 2 unit-length segments along x, widths 1 and 2 → tube_volume
        # ratio 4×. White colors → mean_luminance = 1.
        v = np.array(
            [[0, 0, 0], [1, 0, 0],
             [0, 1, 0], [1, 1, 0]],
            dtype=np.float32,
        )
        polylines = [
            np.array([0, 1], dtype=np.intp),
            np.array([2, 3], dtype=np.intp),
        ]
        widths = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        energy = _compute_lines_energy(v, polylines, widths, None, None)
        # Segment 0: length 1 × width^2 1 = 1.
        # Segment 1: length 1 × width^2 4 = 4.
        np.testing.assert_allclose(energy, [1.0, 4.0])

    def test_compute_lines_energy_with_colors(self):
        v = np.array(
            [[0, 0, 0], [1, 0, 0],
             [0, 1, 0], [1, 1, 0]],
            dtype=np.float32,
        )
        polylines = [
            np.array([0, 1], dtype=np.intp),
            np.array([2, 3], dtype=np.intp),
        ]
        widths = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        # Polyline 0 bright, polyline 1 dark.
        colors = np.array(
            [[1, 1, 1], [1, 1, 1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
            dtype=np.float32,
        )
        energy = _compute_lines_energy(v, polylines, widths, colors, None)
        # Tube volume is identical (1 × 1 = 1). Brighter polyline wins.
        assert energy[0] > energy[1]

    def test_make_additive_lod_lines_with_energy_breakpoints(self):
        rng = np.random.RandomState(0)
        n_segments = 40
        v = rng.uniform(-5, 5, (2 * n_segments, 3)).astype(np.float32)
        w = rng.uniform(0.05, 0.5, 2 * n_segments).astype(np.float32)
        levels = make_additive_lod_lines(
            v,
            line_type="segments",
            widths=w,
            method="salience",
            counts="energy:0.5,1.0",
            salience_kind="energy",
        )
        # All polylines accounted for.
        total = sum(len(L) for L in levels)
        assert total == n_segments

    def test_resolver_lines_accepts_energy_breakpoints(self):
        spec = resolve_additive_axis_lines(
            {"breakpoints": "energy:0.5,0.9,1.0", "method": "random"}
        )
        assert spec["counts"] == "energy:0.5,0.9,1.0"

    def test_resolver_lines_rejects_invalid_salience_kind(self):
        with pytest.raises(ValueError, match="salience_kind"):
            resolve_additive_axis_lines({"salience_kind": "blue-noise"})
