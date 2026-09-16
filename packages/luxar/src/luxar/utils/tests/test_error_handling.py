"""Comprehensive error handling tests for Luxar."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from luxar import Dimension, Dimensions, LuxarZarrCompiler


class TestDimensionErrorHandling:
    """Test error handling in dimensions module."""

    def test_invalid_range(self) -> None:
        """Test invalid dimension ranges."""
        # Min >= max
        with pytest.raises(ValueError, match="min must be less than max"):
            Dimension("x", range=(10, 5))

        # Wrong tuple length
        with pytest.raises(ValueError, match="Range must be a tuple"):
            Dimension("x", range=(1, 2, 3))

    def test_invalid_step(self) -> None:
        """Test invalid step sizes."""
        with pytest.raises(ValueError, match="Step size must be positive"):
            Dimension("x", step=-1)

        with pytest.raises(ValueError, match="Step size must be positive"):
            Dimension("x", step=0)

    def test_invalid_scale(self) -> None:
        """Test invalid scale values."""
        with pytest.raises(ValueError, match="Scale must be positive"):
            Dimension("x", scale=0)

        with pytest.raises(ValueError, match="Scale must be positive"):
            Dimension("x", scale=-2)

    def test_duplicate_dimension_names(self) -> None:
        """Test that duplicate dimension names are rejected."""
        with pytest.raises(ValueError, match="must be unique"):
            Dimensions([Dimension("x"), Dimension("y"), Dimension("x")])  # Duplicate

    def test_too_many_displayed_dimensions(self) -> None:
        """Test that only 3 dimensions can be displayed."""
        with pytest.raises(ValueError, match="Maximum 3 dimensions"):
            Dimensions(
                [
                    Dimension("a", display=True),
                    Dimension("b", display=True),
                    Dimension("c", display=True),
                    Dimension("d", display=True),  # 4th displayed
                ]
            )

    def test_no_displayed_dimensions(self) -> None:
        """Test that at least one dimension must be displayed."""
        with pytest.raises(ValueError, match="At least one dimension"):
            Dimensions([Dimension("x", display=False), Dimension("y", display=False)])

    def test_get_nonexistent_dimension(self) -> None:
        """Test accessing non-existent dimensions."""
        dims = Dimensions([Dimension("x"), Dimension("y")])

        # get_dimension returns None
        assert dims.get_dimension("z") is None

        # get_index raises ValueError
        with pytest.raises(ValueError, match="not found"):
            dims.get_index("z")

    def test_invalid_positions_shape(self) -> None:
        """Test position validation with wrong shapes."""
        dims = Dimensions([Dimension("x"), Dimension("y")])

        # 1D array
        with pytest.raises(ValueError, match="must be a 2D array"):
            dims.validate_positions(np.array([1, 2, 3]))

        # 3D array
        with pytest.raises(ValueError, match="must be a 2D array"):
            dims.validate_positions(np.zeros((10, 2, 3)))

    def test_position_dimension_mismatch(self) -> None:
        """Test positions with wrong number of dimensions."""
        dims = Dimensions([Dimension("x"), Dimension("y")])

        # 3 dimensions when expecting 2
        positions = np.zeros((10, 3))
        with pytest.raises(ValueError, match="has 3 dimensions.*has 2 dimensions"):
            dims.validate_positions(positions)

    def test_positions_outside_range(self) -> None:
        """Test positions outside defined ranges."""
        dims = Dimensions(
            [Dimension("x", range=(-10, 10)), Dimension("y", range=(-5, 5))]
        )

        # X coordinate too large
        positions = np.array([[15, 0]])
        with pytest.raises(ValueError, match="outside range"):
            dims.validate_positions(positions)

        # Y coordinate too small
        positions = np.array([[0, -10]])
        with pytest.raises(ValueError, match="outside range"):
            dims.validate_positions(positions)

    def test_from_positions_invalid_input(self) -> None:
        """Test from_positions with invalid inputs."""
        # Not a 2D array
        with pytest.raises(ValueError, match="must be 2D array"):
            Dimensions.from_positions(np.array([1, 2, 3]))

        # Wrong number of names
        positions = np.zeros((10, 3))
        with pytest.raises(ValueError, match="Got 2 names for 3 dimensions"):
            Dimensions.from_positions(positions, names=["x", "y"])


class TestSceneErrorHandling:
    """Test error handling in Scene class."""

    def test_add_points_invalid_positions(self) -> None:
        """Test adding points with invalid position arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with LuxarZarrCompiler(Path(tmpdir) / "test.luxar.zarr") as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Not array-like
                with pytest.raises(ValueError, match="Positions must have shape"):
                    scene.add_points("bad", "not an array")

                # 1D array
                with pytest.raises(ValueError, match="Positions must have shape"):
                    scene.add_points("bad", np.array([1, 2, 3]))

                # 3D array
                with pytest.raises(ValueError, match="Positions must have shape"):
                    scene.add_points("bad", np.zeros((10, 10, 3)))

    def test_add_points_dimension_validation(self) -> None:
        """Test that dimension mismatches are caught at write time.

        The API validates dimensions when adding data to scenes, catching
        mismatches early with helpful error messages.
        """
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            dims = Dimensions(
                [Dimension("x", range=(-10, 10)), Dimension("y", range=(-10, 10))]
            )
            with LuxarZarrCompiler(Path(tmpdir) / "test.luxar.zarr") as compiler:
                scene = compiler.create_scene(dimensions=dims)

                # 2D points - matches scene dimensions: should succeed
                good_positions = np.random.uniform(-5, 5, (100, 2)).astype(np.float32)
                scene.add_points("2d_points", good_positions)

                # 3D points - mismatches scene dimensions: should fail
                positions_3d = np.random.randn(100, 3).astype(np.float32)
                with pytest.raises(ValueError, match="Dimension mismatch"):
                    scene.add_points("3d_points", positions_3d)

                # Points outside range - warns but succeeds
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    out_of_range = good_positions.copy()
                    out_of_range[0, 0] = 20  # x > 10
                    scene.add_points("out_of_range", out_of_range)
                    # Should produce a warning about out-of-range values
                    range_warnings = [
                        x for x in w if "outside declared range" in str(x.message)
                    ]
                    assert len(range_warnings) >= 1

            # Verify matching points were written, mismatched was not
            import zarr

            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="r")
            assert "2d_points" in store
            assert "3d_points" not in store  # Dimension mismatch prevented writing
            assert "out_of_range" in store

    def test_scene_initialization_errors(self) -> None:
        """Test scene initialization error cases."""
        # Invalid units in Dimensions
        from luxar.typing_utils.enums import PhysicalUnit

        with pytest.raises(ValueError, match="Invalid"):
            PhysicalUnit.validate("invalid_unit")
        assert PhysicalUnit.validate("second").value == "s"

        # Invalid path (simulate permission error)
        # This is platform-specific, so we'll skip for now
        pass


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dimensions(self) -> None:
        """Test empty dimension list."""
        # Empty dimensions are actually allowed
        dims = Dimensions([])
        assert dims.ndim == 0
        assert dims.names == []
        assert dims.displayed == []

    def test_single_dimension(self) -> None:
        """Test single dimension edge case."""
        dims = Dimensions([Dimension("x")])
        assert dims.ndim == 1
        assert dims.displayed == [0]
        assert dims.non_displayed == []

    def test_large_dimension_count(self) -> None:
        """Test handling of many dimensions."""
        # Create 20 dimensions
        dims_list = []
        for i in range(20):
            display = i < 3  # Only first 3 displayed
            dims_list.append(Dimension(f"dim{i}", display=display))

        dims = Dimensions(dims_list)
        assert dims.ndim == 20
        assert len(dims.displayed) == 3
        assert len(dims.non_displayed) == 17

    def test_extreme_values(self) -> None:
        """Test extreme parameter values."""
        # Very large range
        dim = Dimension("x", range=(-1e10, 1e10))
        assert dim.get_step() == 2e8  # 1% of range

        # Very small step
        dim = Dimension("x", step=1e-10)
        assert dim.step == 1e-10

        # Very large scale
        dim = Dimension("x", scale=1e6)
        assert dim.scale == 1e6

    def test_unicode_dimension_names(self) -> None:
        """Test Unicode characters in dimension names."""
        dims = Dimensions(
            [
                Dimension("θ"),  # Greek theta
                Dimension("λ"),  # Greek lambda
                Dimension("时间"),  # Chinese for "time"
            ]
        )
        assert dims.get_dimension("θ") is not None
        assert dims.get_index("λ") == 1

    def test_concurrent_scene_access(self) -> None:
        """Test that new compiler overwrites existing data (mode='w' behavior)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            # Create and finalize first scene
            with LuxarZarrCompiler(zarr_path) as compiler:
                scene1 = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene1.add_points("points1", np.random.randn(100, 3).astype(np.float32))

            # Verify first scene data exists
            import zarr

            store = zarr.open_group(zarr_path, mode="r")
            assert "points1" in store
            # `.shape[0]`, not `len(...)`: zarr 2's Array defined __len__, zarr 3's
            # does not, so `len(array)` raises TypeError.
            points1_count = store["points1/positions"].shape[0]
            assert points1_count == 100

            # Create second scene at same path - overwrites
            with LuxarZarrCompiler(zarr_path) as compiler:
                scene2 = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene2.add_points("points2", np.random.randn(50, 3).astype(np.float32))

            # Verify second scene overwrote first
            store = zarr.open_group(zarr_path, mode="r")
            assert "points2" in store
            assert "points1" not in store  # First scene data should be gone


class TestRecoveryStrategies:
    """Test graceful error recovery."""

    def test_partial_scene_recovery(self) -> None:
        """Test that validation errors don't corrupt the scene."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Add valid data
                scene.add_points("valid", np.random.randn(100, 3).astype(np.float32))

                # Try to add invalid data - validation should catch it
                from luxar.validation import ValidationError

                with pytest.raises((ValueError, ValidationError)):
                    # Wrong shape - 1D array instead of 2D
                    scene.add_points("invalid", np.array([1, 2, 3]))

                # Add more valid data after the error (matching dimensions)
                scene.add_points(
                    "also_valid", np.random.randn(50, 3).astype(np.float32)
                )

            # Verify valid data is preserved and invalid was never written
            import zarr

            store = zarr.open_group(zarr_path, mode="r")
            assert "valid" in store
            assert "also_valid" in store
            assert "invalid" not in store

    def test_dimension_validation_enforced(self) -> None:
        """Test that scene dimensions are validated when adding data.

        When a scene has dimensions defined, data must match those dimensions.
        Mismatched dimensions raise ValueError with helpful error messages.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(zarr_path) as compiler:
                # Create scene with 3D dimensions
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())

                # Add 3D points - matches scene dimensions: should succeed
                positions_3d = np.random.randn(50, 3).astype(np.float32)
                scene.add_points("points_3d", positions_3d)

                # Add 5D points - mismatches scene dimensions: should fail
                positions_5d = np.random.randn(100, 5).astype(np.float32)
                with pytest.raises(ValueError, match="Dimension mismatch"):
                    scene.add_points("points_5d", positions_5d)

                # Add 7D points - also mismatches: should fail
                positions_7d = np.random.randn(25, 7).astype(np.float32)
                with pytest.raises(ValueError, match="Dimension mismatch"):
                    scene.add_points("points_7d", positions_7d)

                # Add 2D points - also mismatches: should fail
                positions_2d = np.random.randn(30, 2).astype(np.float32)
                with pytest.raises(ValueError, match="Dimension mismatch"):
                    scene.add_points("points_2d", positions_2d)

            # Verify only matching points were written
            import zarr

            store = zarr.open_group(zarr_path, mode="r")
            assert store["points_3d/positions"].shape == (50, 3)
            assert "points_5d" not in store
            assert "points_7d" not in store
            assert "points_2d" not in store
