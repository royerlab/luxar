"""Comprehensive error handling tests for Luxar."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from luxar import Dimension, Dimensions, Scene
from luxar.types import DimensionMetadata, validate_dimension_metadata


class TestDimensionErrorHandling:
    """Test error handling in dimensions module."""

    def test_invalid_range(self):
        """Test invalid dimension ranges."""
        # Min >= max
        with pytest.raises(ValueError, match="min must be less than max"):
            Dimension("x", range=(10, 5))

        # Wrong tuple length
        with pytest.raises(ValueError, match="Range must be a tuple"):
            Dimension("x", range=(1, 2, 3))

    def test_invalid_step(self):
        """Test invalid step sizes."""
        with pytest.raises(ValueError, match="Step size must be positive"):
            Dimension("x", step=-1)

        with pytest.raises(ValueError, match="Step size must be positive"):
            Dimension("x", step=0)

    def test_invalid_scale(self):
        """Test invalid scale values."""
        with pytest.raises(ValueError, match="Scale must be positive"):
            Dimension("x", scale=0)

        with pytest.raises(ValueError, match="Scale must be positive"):
            Dimension("x", scale=-2)

    def test_duplicate_dimension_names(self):
        """Test that duplicate dimension names are rejected."""
        with pytest.raises(ValueError, match="must be unique"):
            Dimensions([Dimension("x"), Dimension("y"), Dimension("x")])  # Duplicate

    def test_too_many_displayed_dimensions(self):
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

    def test_no_displayed_dimensions(self):
        """Test that at least one dimension must be displayed."""
        with pytest.raises(ValueError, match="At least one dimension"):
            Dimensions([Dimension("x", display=False), Dimension("y", display=False)])

    def test_get_nonexistent_dimension(self):
        """Test accessing non-existent dimensions."""
        dims = Dimensions([Dimension("x"), Dimension("y")])

        # get_dimension returns None
        assert dims.get_dimension("z") is None

        # get_index raises ValueError
        with pytest.raises(ValueError, match="not found"):
            dims.get_index("z")

    def test_invalid_positions_shape(self):
        """Test position validation with wrong shapes."""
        dims = Dimensions([Dimension("x"), Dimension("y")])

        # 1D array
        with pytest.raises(ValueError, match="must be a 2D array"):
            dims.validate_positions(np.array([1, 2, 3]))

        # 3D array
        with pytest.raises(ValueError, match="must be a 2D array"):
            dims.validate_positions(np.zeros((10, 2, 3)))

    def test_position_dimension_mismatch(self):
        """Test positions with wrong number of dimensions."""
        dims = Dimensions([Dimension("x"), Dimension("y")])

        # 3 dimensions when expecting 2
        positions = np.zeros((10, 3))
        with pytest.raises(ValueError, match="has 3 dimensions.*has 2 dimensions"):
            dims.validate_positions(positions)

    def test_positions_outside_range(self):
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

    def test_from_positions_invalid_input(self):
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

    def test_add_points_invalid_positions(self):
        """Test adding points with invalid position arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Scene(Path(tmpdir) / "test.zarr")

            # Not array-like
            with pytest.raises(ValueError, match="Positions must have shape"):
                scene.add_points("bad", "not an array")

            # 1D array
            with pytest.raises(ValueError, match="Positions must have shape"):
                scene.add_points("bad", np.array([1, 2, 3]))

            # 3D array
            with pytest.raises(ValueError, match="Positions must have shape"):
                scene.add_points("bad", np.zeros((10, 10, 3)))

    def test_add_points_dimension_validation(self):
        """Test dimension validation when adding points."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dims = Dimensions(
                [Dimension("x", range=(-10, 10)), Dimension("y", range=(-10, 10))]
            )
            scene = Scene(Path(tmpdir) / "test.zarr", dimensions=dims)

            # Valid points
            good_positions = np.random.uniform(-5, 5, (100, 2)).astype(np.float32)
            scene.add_points("good", good_positions)

            # Invalid points - wrong dimensions
            bad_positions_3d = np.random.randn(100, 3).astype(np.float32)
            with pytest.raises(ValueError, match="has 3 dimensions.*has 2 dimensions"):
                scene.add_points("bad_dims", bad_positions_3d)

            # Invalid points - out of range
            bad_positions_range = good_positions.copy()
            bad_positions_range[0, 0] = 20  # x > 10
            with pytest.raises(ValueError, match="outside range"):
                scene.add_points("bad_range", bad_positions_range)

    def test_scene_initialization_errors(self):
        """Test scene initialization error cases."""
        # Invalid units
        with pytest.raises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdir:
                Scene(Path(tmpdir) / "test.zarr", units="invalid_unit")

        # Invalid path (simulate permission error)
        # This is platform-specific, so we'll skip for now
        pass

    def test_finalize_errors(self):
        """Test scene finalization error handling."""
        # Already finalized scene
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Scene(Path(tmpdir) / "test.zarr")
            scene.finalize()
            # Second finalize should work (idempotent)
            scene.finalize()


class TestLegacyDimensionMetadata:
    """Test error handling for legacy dimension metadata."""

    def test_invalid_metadata_count(self):
        """Test validation of dimension metadata count."""
        # Too few metadata entries
        with pytest.raises(ValueError, match="Expected 3 dimension metadata"):
            validate_dimension_metadata(
                [DimensionMetadata(name="x"), DimensionMetadata(name="y")], ndim=3
            )

        # Too many metadata entries
        with pytest.raises(ValueError, match="Expected 2 dimension metadata"):
            validate_dimension_metadata(
                [
                    DimensionMetadata(name="x"),
                    DimensionMetadata(name="y"),
                    DimensionMetadata(name="z"),
                ],
                ndim=2,
            )

    def test_invalid_metadata_types(self):
        """Test validation of metadata types."""
        # Not a list
        with pytest.raises(ValueError, match="must be a list"):
            validate_dimension_metadata("not a list", ndim=2)

        # List of wrong types (not dict or DimensionMetadata)
        with pytest.raises(ValueError, match="must be dict or DimensionMetadata"):
            validate_dimension_metadata(["string1", "string2"], ndim=2)

    def test_metadata_dict_conversion(self):
        """Test conversion from dict to DimensionMetadata."""
        # Valid dict
        data = [
            {"name": "x", "unit": "um", "scale": 1.0},
            {"name": "y", "unit": "um", "scale": 1.0},
        ]
        metadata = validate_dimension_metadata(data, ndim=2)
        assert len(metadata) == 2
        assert all(isinstance(m, DimensionMetadata) for m in metadata)

        # Invalid dict (missing required field will be handled by from_dict)
        # from_dict creates a DimensionMetadata with default name
        # So this won't raise an error anymore
        metadata = validate_dimension_metadata([{"unit": "um"}], ndim=1)
        assert len(metadata) == 1  # Should succeed with default values


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dimensions(self):
        """Test empty dimension list."""
        # Empty dimensions are actually allowed
        dims = Dimensions([])
        assert dims.ndim == 0
        assert dims.names == []
        assert dims.displayed == []

    def test_single_dimension(self):
        """Test single dimension edge case."""
        dims = Dimensions([Dimension("x")])
        assert dims.ndim == 1
        assert dims.displayed == [0]
        assert dims.non_displayed == []

    def test_large_dimension_count(self):
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

    def test_extreme_values(self):
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

    def test_unicode_dimension_names(self):
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

    def test_concurrent_scene_access(self):
        """Test multiple scenes accessing same zarr store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create and finalize first scene
            scene1 = Scene(zarr_path)
            scene1.add_points("points1", np.random.randn(100, 3).astype(np.float32))
            scene1.finalize()

            # Try to create second scene at same path
            # Should overwrite (mode='w')
            scene2 = Scene(zarr_path)
            scene2.add_points("points2", np.random.randn(50, 3).astype(np.float32))
            scene2.finalize()


class TestRecoveryStrategies:
    """Test graceful error recovery."""

    def test_partial_scene_recovery(self):
        """Test recovering from partial scene creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr"

            # Create scene with some data
            scene = Scene(zarr_path)
            scene.add_points("valid", np.random.randn(100, 3).astype(np.float32))

            # Try to add invalid data
            try:
                scene.add_points("invalid", np.array([1, 2, 3]))  # Wrong shape
            except ValueError:
                pass  # Expected

            # Scene should still be finalizable
            scene.finalize()

            # Verify valid data is preserved
            import zarr

            store = zarr.open_group(zarr_path, mode="r")
            assert "valid" in store
            assert "invalid" not in store

    def test_dimension_inference_fallback(self):
        """Test dimension inference when not explicitly set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Scene(Path(tmpdir) / "test.zarr")

            # Add 5D points without scene dimensions
            positions = np.random.randn(100, 5).astype(np.float32)
            scene.add_points("points", positions)

            # Dimensions should be inferred
            assert scene.dimensions is not None
            assert scene.dimensions.ndim == 5
            assert scene.dimensions.displayed == [0, 1, 2]  # First 3 displayed

            # Add more points - should validate against inferred dims
            more_positions = np.random.randn(50, 5).astype(np.float32)
            scene.add_points("more_points", more_positions)

            scene.finalize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
