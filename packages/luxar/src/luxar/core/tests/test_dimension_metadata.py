"""
Test dimension functionality (current Dimension class).
"""

import numpy as np
import pytest

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.validation.types import validate_positions


class TestDimensionValidation:
    """Test dimension validation functions."""

    def test_validate_positions_nd(self):
        """Test validating nD positions."""
        # 2D positions
        pos_2d = np.random.rand(100, 2).astype(np.float32)
        validated = validate_positions(pos_2d)
        assert validated.shape == (100, 2)
        assert validated.dtype == np.float32

        # 3D positions
        pos_3d = np.random.rand(50, 3).astype(np.float32)
        validated = validate_positions(pos_3d)
        assert validated.shape == (50, 3)

        # 5D positions
        pos_5d = np.random.rand(20, 5).astype(np.float32)
        validated = validate_positions(pos_5d)
        assert validated.shape == (20, 5)

        # Test with expected dimensions
        validated = validate_positions(pos_3d, ndim=3)
        assert validated.shape == (50, 3)

        # Test dimension mismatch
        with pytest.raises(ValueError, match="Expected 2 dimensions, got 3"):
            validate_positions(pos_3d, ndim=2)

    def test_validate_positions_errors(self):
        """Test position validation errors."""
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="must have shape"):
            validate_positions(np.array([1, 2, 3]))

        # No dimensions
        with pytest.raises(ValueError, match="at least 1 dimension"):
            validate_positions(np.zeros((10, 0)))


class TestSceneDimensionMetadata:
    """Test dimension metadata in Scene."""

    def test_scene_dimension_metadata(self, tmp_path):
        """Test setting and getting scene-level dimensions."""
        import zarr

        from luxar import Dimension, Dimensions

        # Test scene without dimensions
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()
            # Initially None
            assert scene.dimensions is None

        # Test scene with dimensions
        dims = Dimensions(
            [
                Dimension("x", unit="um", scale=0.5, display=True),
                Dimension("y", unit="um", scale=0.5, display=True),
                Dimension("z", unit="um", scale=1.0, display=True),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test_with_dims.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Retrieve dimensions
            retrieved = scene.dimensions
            assert retrieved is not None
            assert len(retrieved.dimensions) == 3
            assert retrieved.dimensions[0].name == "x"
            assert retrieved.dimensions[1].unit == "um"
            assert retrieved.dimensions[2].scale == 1.0

            # Check stored in zarr attrs (new format)
            store = zarr.open_group(tmp_path / "test_with_dims.zarr", mode="r")
            assert "scene_dimensions" in store.attrs

    def test_scene_dimension_persistence(self, tmp_path):
        """Test dimension metadata persists through save/load."""
        zarr_path = tmp_path / "persist.zarr"

        # Create scene with metadata using new API
        dims = Dimensions(
            [
                Dimension("t", unit="ms", range=(0, 100), step=2.0),
                Dimension("x", unit="px"),
                Dimension("y", unit="px"),
            ]
        )

        with LuxarZarrCompiler(zarr_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add some points
            positions = np.random.rand(100, 3).astype(np.float32)
            scene.add_points("test", positions)

        # Load in new scene
        import zarr

        root = zarr.open_group(zarr_path, mode="r")

        # Check metadata persisted with new format
        assert "scene_dimensions" in root.attrs
        dims_dict = root.attrs["scene_dimensions"]
        assert len(dims_dict["dimensions"]) == 3
        assert dims_dict["dimensions"][0]["name"] == "t"
        assert dims_dict["dimensions"][0]["unit"] == "ms"
        assert dims_dict["dimensions"][0]["range"] == [0, 100]

    # Test removed: Dimension validation was removed in new flexible API
    # The dimension_metadata parameter in add_points is deprecated

    # Test removed: Dimension inheritance was removed in new flexible API
    # Points no longer inherit dimension metadata from the scene

    # Test removed: Mixed dimensionality is fully supported in new flexible API
    # Each points can have any dimensionality independent of others
