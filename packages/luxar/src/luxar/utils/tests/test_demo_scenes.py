"""Tests for reusable demo scene generators."""

import tempfile
from pathlib import Path

import numpy as np
import zarr

from luxar.demos.demo_lorenz import lorenz_trajectory
from luxar.encoding import ArrayDecoder
from luxar.utils.scenes import (
    _BYTES_PER_POINT,
    _LARGE_DATASET_POINTS,
    _MAX_RECOMMENDED_POINTS,
    _dataset_size_warning,
    create_lorenz_attractor,
    create_random_spheres,
    create_time_series_demo,
)


def _max_nearest_neighbor_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance from a point of ``a`` to its nearest point in ``b``.

    Comparing two point sets row-by-row needs them in the same order, and the
    compiler reorders points along a Hilbert curve. Matching whole points to
    their nearest counterpart recovers the correspondence without relying on
    order — and unlike a per-axis comparison it keeps each point's three
    coordinates tied together.
    """
    d = np.linalg.norm(
        a[:, None, :].astype(np.float64) - b[None, :, :].astype(np.float64), axis=-1
    )
    return float(d.min(axis=1).max())


class TestCreateLorenzAttractor:
    """Tests for create_lorenz_attractor function."""

    def test_basic_creation(self) -> None:
        """Test basic Lorenz attractor creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "lorenz.luxar.zarr"
            create_lorenz_attractor(store_path, n_points=100)

            # Verify zarr store was created
            assert store_path.exists()

            # Check structure
            store = zarr.open(store_path, mode="r")
            assert "LorenzAttractor" in store
            assert "positions" in store["LorenzAttractor"]

            # Check positions shape
            positions = store["LorenzAttractor"]["positions"][:]
            assert positions.shape[0] == 100
            assert positions.shape[1] == 3  # 3D

    def test_with_seed_reproducibility(self) -> None:
        """Test that seed produces reproducible results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path1 = Path(tmpdir) / "lorenz1.luxar.zarr"
            store_path2 = Path(tmpdir) / "lorenz2.luxar.zarr"

            create_lorenz_attractor(store_path1, n_points=50, seed=42)
            create_lorenz_attractor(store_path2, n_points=50, seed=42)

            store1 = zarr.open(store_path1, mode="r")
            store2 = zarr.open(store_path2, mode="r")

            # Positions should be identical
            pos1 = store1["LorenzAttractor"]["positions"][:]
            pos2 = store2["LorenzAttractor"]["positions"][:]
            np.testing.assert_array_almost_equal(pos1, pos2)

    def test_positions_come_from_the_demo_integrator(self) -> None:
        """The fixture and ``luxar demo run lorenz`` share one integrator.

        Guards the dedup: if either side re-inlines or re-parameterises the
        Lorenz integration, the two stop agreeing and this fails.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "lorenz.luxar.zarr"
            create_lorenz_attractor(store_path, n_points=500, seed=42)

            store = zarr.open(store_path, mode="r")
            written = ArrayDecoder().decode(
                store["LorenzAttractor"]["positions"], store
            )

            # Same trajectory, scaled up for visibility exactly as the builder
            # does. Matched point-to-point (see _max_nearest_neighbor_distance);
            # the tolerance covers the uint16 position quantization, whose step
            # is ~0.006 over this extent. Both directions, so neither set may
            # contain a point the other lacks.
            expected = lorenz_trajectory(500, seed=42) * 100.0 - 50.0
            assert written.shape == expected.shape
            assert _max_nearest_neighbor_distance(written, expected) < 0.05
            assert _max_nearest_neighbor_distance(expected, written) < 0.05

    def test_has_colors_and_radii(self) -> None:
        """Test that colors and radii are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "lorenz.luxar.zarr"
            create_lorenz_attractor(store_path, n_points=100)

            store = zarr.open(store_path, mode="r")
            lorenz_group = store["LorenzAttractor"]

            assert "colors" in lorenz_group
            assert "radii" in lorenz_group

            colors = lorenz_group["colors"][:]
            # Colors are SDR RGB
            assert colors.shape[1] == 3

    def test_different_point_counts(self) -> None:
        """Test creation with different point counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for n in [10, 100, 1000]:
                store_path = Path(tmpdir) / f"lorenz_{n}.luxar.zarr"
                create_lorenz_attractor(store_path, n_points=n)

                store = zarr.open(store_path, mode="r")
                positions = store["LorenzAttractor"]["positions"][:]
                assert positions.shape[0] == n


class TestCreateRandomSpheres:
    """Tests for create_random_spheres function."""

    def test_basic_creation(self) -> None:
        """Test basic random spheres creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "spheres.luxar.zarr"
            create_random_spheres(
                store_path, n_spheres=5, points_per_sphere=50, seed=42
            )

            # Verify zarr store was created
            assert store_path.exists()

            # Check structure - should have sphere_000, sphere_001, etc.
            store = zarr.open(store_path, mode="r")
            for i in range(5):
                sphere_name = f"sphere_{i:03d}"
                assert sphere_name in store, f"Missing {sphere_name}"

    def test_sphere_point_counts(self) -> None:
        """Test that each sphere has correct number of points."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "spheres.luxar.zarr"
            n_spheres = 3
            points_per_sphere = 100

            create_random_spheres(
                store_path,
                n_spheres=n_spheres,
                points_per_sphere=points_per_sphere,
                seed=42,
            )

            store = zarr.open(store_path, mode="r")
            for i in range(n_spheres):
                sphere_name = f"sphere_{i:03d}"
                positions = store[sphere_name]["positions"][:]
                assert positions.shape[0] == points_per_sphere

    def test_seed_reproducibility(self) -> None:
        """Test that seed produces reproducible spheres."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path1 = Path(tmpdir) / "spheres1.luxar.zarr"
            store_path2 = Path(tmpdir) / "spheres2.luxar.zarr"

            create_random_spheres(
                store_path1, n_spheres=3, points_per_sphere=50, seed=123
            )
            create_random_spheres(
                store_path2, n_spheres=3, points_per_sphere=50, seed=123
            )

            store1 = zarr.open(store_path1, mode="r")
            store2 = zarr.open(store_path2, mode="r")

            # First sphere positions should be identical
            pos1 = store1["sphere_000"]["positions"][:]
            pos2 = store2["sphere_000"]["positions"][:]
            np.testing.assert_array_almost_equal(pos1, pos2)

    def test_3d_positions(self) -> None:
        """Test that sphere positions are 3D."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "spheres.luxar.zarr"
            create_random_spheres(
                store_path, n_spheres=1, points_per_sphere=100, seed=42
            )

            store = zarr.open(store_path, mode="r")
            positions = store["sphere_000"]["positions"][:]
            assert positions.shape[1] == 3  # 3D


class TestCreateTimeSeriesDemo:
    """Tests for create_time_series_demo function."""

    def test_basic_creation(self) -> None:
        """Test basic time series demo creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            create_time_series_demo(
                store_path, n_timepoints=5, n_points_per_time=50, seed=42
            )

            # Verify zarr store was created
            assert store_path.exists()

            # Check structure
            store = zarr.open(store_path, mode="r")
            assert "time_series" in store

    def test_4d_positions(self) -> None:
        """Test that positions are 4D (x, y, z, time)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            create_time_series_demo(
                store_path, n_timepoints=5, n_points_per_time=50, seed=42
            )

            store = zarr.open(store_path, mode="r")
            positions = store["time_series"]["positions"][:]
            assert positions.shape[1] == 4  # 4D (x, y, z, time)

    def test_total_point_count(self) -> None:
        """Test that total points = n_timepoints * n_points_per_time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            n_timepoints = 5
            n_points_per_time = 100

            create_time_series_demo(
                store_path,
                n_timepoints=n_timepoints,
                n_points_per_time=n_points_per_time,
                seed=42,
            )

            store = zarr.open(store_path, mode="r")
            positions = store["time_series"]["positions"][:]
            expected_total = n_timepoints * n_points_per_time
            assert positions.shape[0] == expected_total

    def test_time_dimension_values(self) -> None:
        """Test that time dimension has correct integer values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            n_timepoints = 5
            n_points_per_time = 50

            create_time_series_demo(
                store_path,
                n_timepoints=n_timepoints,
                n_points_per_time=n_points_per_time,
                seed=42,
            )

            store = zarr.open(store_path, mode="r")
            # Positions are uint16 per-axis fixed-point under the default AUTO
            # mode — decode (raw levels are meaningless integers) before reading
            # the time column.
            positions = ArrayDecoder().decode(store["time_series"]["positions"], store)

            # Time is the 4th dimension (index 3)
            time_values = positions[:, 3]
            unique_times = np.unique(time_values)

            # Should have exactly n_timepoints unique time values (each distinct
            # input value maps to one quantization code, so the count survives)
            assert len(unique_times) == n_timepoints
            # Time values should be integers 0 to n_timepoints-1, within one
            # quantization step (extent/65535)
            np.testing.assert_allclose(
                sorted(unique_times), list(range(n_timepoints)), atol=1e-3
            )

    def test_seed_reproducibility(self) -> None:
        """Test that seed produces reproducible time series."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path1 = Path(tmpdir) / "ts1.luxar.zarr"
            store_path2 = Path(tmpdir) / "ts2.luxar.zarr"

            create_time_series_demo(
                store_path1, n_timepoints=3, n_points_per_time=50, seed=456
            )
            create_time_series_demo(
                store_path2, n_timepoints=3, n_points_per_time=50, seed=456
            )

            store1 = zarr.open(store_path1, mode="r")
            store2 = zarr.open(store_path2, mode="r")

            pos1 = store1["time_series"]["positions"][:]
            pos2 = store2["time_series"]["positions"][:]
            np.testing.assert_array_almost_equal(pos1, pos2)

    def test_has_colors(self) -> None:
        """Test that colors are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            create_time_series_demo(
                store_path, n_timepoints=5, n_points_per_time=50, seed=42
            )

            store = zarr.open(store_path, mode="r")
            assert "colors" in store["time_series"]

            # Colors are encoded, verify encoding metadata exists
            colors_arr = store["time_series"]["colors"]
            # Should have some encoding - could be LUT, broadcasted, or direct
            # The raw shape depends on encoding - just verify presence
            assert colors_arr.shape[0] > 0

    def test_scene_dimensions(self) -> None:
        """Test that scene has 4D dimensions defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            create_time_series_demo(
                store_path, n_timepoints=5, n_points_per_time=50, seed=42
            )

            store = zarr.open(store_path, mode="r")
            # Check scene dimensions are defined
            assert "scene_dimensions" in store.attrs


class TestEdgeCases:
    """Edge case tests for demo functions."""

    def test_single_point_lorenz(self) -> None:
        """Test Lorenz with single point."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "lorenz.luxar.zarr"
            create_lorenz_attractor(store_path, n_points=1)

            store = zarr.open(store_path, mode="r")
            positions = store["LorenzAttractor"]["positions"][:]
            assert positions.shape[0] == 1

    def test_single_sphere(self) -> None:
        """Test random spheres with single sphere."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "spheres.luxar.zarr"
            create_random_spheres(
                store_path, n_spheres=1, points_per_sphere=10, seed=42
            )

            store = zarr.open(store_path, mode="r")
            assert "sphere_000" in store

    def test_single_timepoint_becomes_two(self) -> None:
        """Test time series with single timepoint gets adjusted to 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "timeseries.luxar.zarr"
            # Requesting 1 timepoint should be adjusted to 2
            create_time_series_demo(
                store_path, n_timepoints=1, n_points_per_time=50, seed=42
            )

            store = zarr.open(store_path, mode="r")
            positions = store["time_series"]["positions"][:]
            # Should have 2 timepoints * 50 points = 100 points
            assert positions.shape[0] == 100


class TestDatasetSizeWarning:
    """Tests for the demo-scene point-count heads-up.

    Relocated from ``typing_utils/tests/test_config.py`` along with the helper
    itself: ``typing_utils.config`` held ~30 public names of which three had a
    production reader, and this was the only one carrying real logic. It lives
    beside its single caller now.
    """

    def test_small_dataset_no_warning(self) -> None:
        """A small demo scene says nothing."""
        assert _dataset_size_warning(1000) is None

    def test_just_below_the_large_threshold_is_quiet(self) -> None:
        """One point below the threshold is still quiet."""
        assert _dataset_size_warning(_LARGE_DATASET_POINTS - 1) is None

    def test_exactly_at_the_large_threshold_is_quiet(self) -> None:
        """The comparison is strictly greater-than, so the boundary is quiet."""
        assert _dataset_size_warning(_LARGE_DATASET_POINTS) is None

    def test_large_dataset_warns_with_the_count(self) -> None:
        """Above the threshold, the message names the count and a size."""
        n_points = _LARGE_DATASET_POINTS + 100
        result = _dataset_size_warning(n_points)
        assert result is not None
        assert "Large dataset" in result
        assert f"{n_points:,}" in result

    def test_very_large_dataset_says_exceeds_recommended(self) -> None:
        """Above the recommended maximum, the wording escalates."""
        n_points = _MAX_RECOMMENDED_POINTS + 100
        result = _dataset_size_warning(n_points)
        assert result is not None
        assert "exceeds recommended maximum" in result
        assert f"{n_points:,}" in result

    def test_exactly_at_the_max_is_still_only_the_large_warning(self) -> None:
        """The recommended-maximum boundary has not been exceeded yet."""
        result = _dataset_size_warning(_MAX_RECOMMENDED_POINTS)
        assert result is not None
        assert "Large dataset" in result

    def test_the_estimated_size_uses_positions_plus_colors(self) -> None:
        """The reported megabytes must be the float32-xyz + uint8-rgb estimate.

        Pinned because the constant replaced an ``estimate_memory_usage()``
        helper that nothing else called; an off-by-one-channel edit here would
        otherwise only show up as a slightly wrong number in demo output.
        """
        assert _BYTES_PER_POINT == 15
        n_points = 2_000_000
        result = _dataset_size_warning(n_points)
        assert result is not None
        expected_mb = n_points * 15 / (1024 * 1024)
        assert f"~{expected_mb:.1f}MB" in result
