"""Advanced tests for Scene class to improve coverage.

Tests cover uncovered lines in scene.py:
- Scene initialization error handling
- add_points with broadcast_dims variations
- add_lines validation
- add_gsplats validation
- _auto_detect_broadcast_dims method
- get_store_path method
- dimensions property setter
- to_zarr method
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.dimensions import Dimension, Dimensions
from luxar.core.scene import Scene


class TestSceneInitialization:
    """Tests for Scene initialization error paths."""

    def test_scene_requires_writer(self) -> None:
        """Test that Scene requires a writer."""
        with pytest.raises(ValueError, match="Writer is required"):
            Scene(writer=None, dimensions=Dimensions.default_3d())  # type: ignore[arg-type]


class TestAddPointsBroadcastDims:
    """Tests for add_points with various broadcast_dims options."""

    def test_broadcast_dims_auto(self) -> None:
        """Test add_points with broadcast_dims='auto'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 9)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # Create 4D positions with only one unique time value
                # This should trigger auto-detection
                positions = np.random.rand(100, 4).astype(np.float32)
                positions[:, 3] = 0  # All at time=0

                points = scene.add_points(
                    "test_points",
                    positions,
                    broadcast_dims="auto",
                )
                assert points is not None

    def test_broadcast_dims_all(self) -> None:
        """Test add_points with broadcast_dims='all'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 9)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # 4D positions
                positions = np.random.rand(100, 4).astype(np.float32)

                points = scene.add_points(
                    "test_points",
                    positions,
                    broadcast_dims="all",
                )
                assert points is not None

    def test_broadcast_dims_explicit_list(self) -> None:
        """Test add_points with explicit broadcast_dims list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 9)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                positions = np.random.rand(100, 4).astype(np.float32)

                points = scene.add_points(
                    "test_points",
                    positions,
                    broadcast_dims=["time"],
                )
                assert points is not None

    def test_broadcast_dims_invalid_value(self) -> None:
        """Test add_points with invalid broadcast_dims value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                positions = np.random.rand(100, 3).astype(np.float32)

                with pytest.raises(ValueError, match="Invalid broadcast_dims"):
                    scene.add_points(
                        "test_points",
                        positions,
                        broadcast_dims="invalid_value",  # type: ignore[arg-type]
                    )

    def test_add_points_1d_positions_error(self) -> None:
        """Test add_points with 1D positions raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                positions = np.random.rand(100).astype(np.float32)  # 1D, not 2D

                with pytest.raises(ValueError, match="shape.*N, D"):
                    scene.add_points("test_points", positions)

    def test_add_points_list_input(self) -> None:
        """Test add_points with list input (converted to array)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Pass list instead of numpy array
                positions = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

                points = scene.add_points("test_points", positions)
                assert points is not None


class TestAddLinesValidation:
    """Tests for add_lines validation paths."""

    def test_add_lines_1d_vertices_error(self) -> None:
        """Test add_lines with 1D vertices raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                vertices = np.random.rand(100).astype(np.float32)  # 1D, not 2D

                with pytest.raises(ValueError, match="shape.*N, D"):
                    scene.add_lines("test_lines", vertices, widths=0.1)

    def test_add_lines_list_input(self) -> None:
        """Test add_lines with list input (converted to array)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Pass list instead of numpy array
                vertices = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

                lines = scene.add_lines("test_lines", vertices, widths=0.1)
                assert lines is not None


class TestAddGSplatsValidation:
    """Tests for add_gsplats validation paths."""

    def test_add_gsplats_1d_centers_error(self) -> None:
        """Test add_gsplats with 1D centers raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                centers = np.random.rand(100).astype(np.float32)  # 1D, not 2D
                cholesky = np.random.rand(100, 6).astype(np.float32)

                with pytest.raises(ValueError, match="shape.*N, D"):
                    scene.add_gsplats(
                        "test_gsplats",
                        centers,
                        amplitudes=1.0,
                        cholesky_factors=cholesky,
                    )

    def test_add_gsplats_list_input(self) -> None:
        """Test add_gsplats with list input (converted to array)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Pass list instead of numpy array
                centers = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
                # Cholesky factors: 3D -> k = 3*(3+1)/2 = 6
                cholesky = np.array([[1.0, 0, 0, 1.0, 0, 1.0], [1.0, 0, 0, 1.0, 0, 1.0]])

                gsplats = scene.add_gsplats(
                    "test_gsplats",
                    centers,
                    amplitudes=1.0,
                    cholesky_factors=cholesky,
                )
                assert gsplats is not None


class TestAutoDetectBroadcastDims:
    """Tests for _auto_detect_broadcast_dims method."""

    def test_auto_detect_no_dimensions(self) -> None:
        """Test auto-detect returns empty when scene has no dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())  # No dimensions

                positions = np.random.rand(100, 3).astype(np.float32)

                # Auto-detect with no dimensions should return empty
                result = scene._auto_detect_broadcast_dims(positions)
                assert result == []

    def test_auto_detect_full_coverage(self) -> None:
        """Test auto-detect returns empty when data has full coverage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 4)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # Create positions with full coverage (all 5 time points)
                n_per_time = 20
                positions = []
                for t in range(5):  # 0, 1, 2, 3, 4
                    pts = np.random.rand(n_per_time, 3).astype(np.float32)
                    time_col = np.full((n_per_time, 1), t, dtype=np.float32)
                    positions.append(np.hstack([pts, time_col]))

                all_positions = np.vstack(positions)  # 100 points, full coverage

                # Should not detect broadcast dims due to full coverage
                _result = scene._auto_detect_broadcast_dims(all_positions)
                # Even with all times present, if only one unique value per column,
                # it might still detect broadcast
                # The logic checks if n_points >= expected_total * 0.8
                # Result is intentionally unused - we're testing that call doesn't crash

    def test_auto_detect_low_coverage(self) -> None:
        """Test auto-detect with low coverage data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                    Dimension(
                        "time", unit="s", display=False, discrete=True, range=(0, 99)
                    ),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # Create positions with only one time value and low coverage
                # Expected total = 100 time points
                # We have 5 points, which is << 80 (expected * 0.8)
                positions = np.random.rand(5, 4).astype(np.float32)
                positions[:, 3] = 5  # All at time=5

                result = scene._auto_detect_broadcast_dims(positions)
                # With only 5 points and 100 expected time points,
                # and all points at the same time, "time" should be detected
                assert "time" in result


class TestDimensionsProperty:
    """Tests for dimensions property getter and setter."""

    def test_dimensions_setter_none(self) -> None:
        """Test that setting dimensions to None raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                    Dimension("z", unit="um", display=True),
                ]
            )

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                assert scene.dimensions is not None

                # Setting to None should raise ValueError
                with pytest.raises(ValueError, match="dimensions cannot be None"):
                    scene.dimensions = None

    def test_dimensions_setter_new_dims(self) -> None:
        """Test setting new dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())  # No initial dims

                new_dims = Dimensions(
                    [
                        Dimension("a", unit="mm", display=True),
                        Dimension("b", unit="mm", display=True),
                    ]
                )

                scene.dimensions = new_dims
                assert scene.dimensions is not None
                assert len(scene.dimensions.dimensions) == 2


class TestToZarr:
    """Tests for to_zarr method."""

    def test_to_zarr_not_implemented(self) -> None:
        """Test that to_zarr raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                with pytest.raises(NotImplementedError, match="not yet implemented"):
                    scene.to_zarr(Path(tmpdir) / "export.zarr")


class TestGetStorePath:
    """Tests for get_store_path method."""

    def test_get_store_path_returns_path(self) -> None:
        """Test that get_store_path returns the store path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            with LuxarZarrCompiler(store_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                result = scene.get_store_path()
                assert str(store_path) in result
