"""
Test dimension metadata functionality.
"""

import numpy as np
import pytest

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.types import (
    DimensionMetadata,
    validate_dimension_metadata,
    validate_positions,
)


class TestDimensionMetadata:
    """Test DimensionMetadata class."""

    def test_dimension_metadata_creation(self):
        """Test creating dimension metadata."""
        dim = DimensionMetadata(name="x", unit="um", scale=0.5, range=(-10.0, 10.0))
        assert dim.name == "x"
        assert dim.unit == "um"
        assert dim.scale == 0.5
        assert dim.range == (-10.0, 10.0)

    def test_dimension_metadata_defaults(self):
        """Test dimension metadata defaults."""
        dim = DimensionMetadata()
        assert dim.name == ""
        assert dim.unit == ""
        assert dim.scale == 1.0
        assert dim.range is None

    def test_to_dict(self):
        """Test converting to dictionary."""
        dim = DimensionMetadata(name="time", unit="ms", scale=2.0, range=(0.0, 100.0))
        d = dim.to_dict()
        assert d == {"name": "time", "unit": "ms", "scale": 2.0, "range": [0.0, 100.0]}

        # Test without range
        dim2 = DimensionMetadata(name="x", unit="px")
        d2 = dim2.to_dict()
        assert d2 == {"name": "x", "unit": "px", "scale": 1.0}

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"name": "z", "unit": "nm", "scale": 0.1, "range": [-50.0, 50.0]}
        dim = DimensionMetadata.from_dict(data)
        assert dim.name == "z"
        assert dim.unit == "nm"
        assert dim.scale == 0.1
        assert dim.range == (-50.0, 50.0)

        # Test with missing fields
        data2 = {"name": "y"}
        dim2 = DimensionMetadata.from_dict(data2)
        assert dim2.name == "y"
        assert dim2.unit == ""
        assert dim2.scale == 1.0
        assert dim2.range is None


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

    def test_validate_dimension_metadata(self):
        """Test validating dimension metadata list."""
        # Valid metadata
        metadata = [
            DimensionMetadata(name="x", unit="um"),
            DimensionMetadata(name="y", unit="um"),
            DimensionMetadata(name="z", unit="um"),
        ]
        validated = validate_dimension_metadata(metadata, 3)
        assert len(validated) == 3
        assert all(isinstance(m, DimensionMetadata) for m in validated)

        # From dictionaries
        dict_metadata = [
            {"name": "x", "unit": "px"},
            {"name": "y", "unit": "px"},
        ]
        validated = validate_dimension_metadata(dict_metadata, 2)
        assert len(validated) == 2
        assert validated[0].name == "x"
        assert validated[1].unit == "px"

        # Wrong count
        with pytest.raises(
            ValueError, match="Expected 3 dimension metadata entries, got 2"
        ):
            validate_dimension_metadata(metadata[:2], 3)

        # Invalid type
        with pytest.raises(ValueError, match="must be dict or DimensionMetadata"):
            validate_dimension_metadata(["x", "y", "z"], 3)


class TestSceneDimensionMetadata:
    """Test dimension metadata in Scene."""

    def test_scene_dimension_metadata(self, tmp_path):
        """Test setting and getting dimension metadata on scene."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Initially None
            assert scene.dimension_metadata is None

            # Set metadata
            metadata = [
                DimensionMetadata(name="x", unit="um", scale=0.5),
                DimensionMetadata(name="y", unit="um", scale=0.5),
                DimensionMetadata(name="z", unit="um", scale=1.0),
            ]
            scene.dimension_metadata = metadata

            # Retrieve
            retrieved = scene.dimension_metadata
            assert len(retrieved) == 3
            assert retrieved[0].name == "x"
            assert retrieved[1].unit == "um"
            assert retrieved[2].scale == 1.0

            # Check stored in attrs
            assert "dimension_metadata" in scene.attrs
            assert len(scene.attrs["dimension_metadata"]) == 3

    def test_scene_dimension_persistence(self, tmp_path):
        """Test dimension metadata persists through save/load."""
        zarr_path = tmp_path / "persist.zarr"

        # Create scene with metadata using new API
        dims = Dimensions([
            Dimension("t", unit="ms", range=(0, 100), step=2.0),
            Dimension("x", unit="px"),
            Dimension("y", unit="px"),
        ])

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

    @pytest.mark.skip(reason="Dimension validation removed in new flexible API")
    def test_add_points_with_dimension_metadata(self, tmp_path):
        """Test adding points with dimension metadata."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

        # 5D points
        positions = np.random.rand(50, 5).astype(np.float32)
        metadata = [
            DimensionMetadata(name="t", unit="s"),
            DimensionMetadata(name="x", unit="um"),
            DimensionMetadata(name="y", unit="um"),
            DimensionMetadata(name="z", unit="um"),
            DimensionMetadata(name="c", unit="au"),
        ]

        points = scene.add_points("5d_points", positions, dimension_metadata=metadata)

        # Check scene metadata was set
        assert scene.dimension_metadata is not None
        assert len(scene.dimension_metadata) == 5
        assert scene.dimension_metadata[0].name == "t"
        assert scene.dimension_metadata[4].name == "c"

        # Check points attributes
        assert "dimension_metadata" in points.attrs
        assert len(points.attrs["dimension_metadata"]) == 5

    @pytest.mark.skip(reason="Dimension inheritance removed in new flexible API")
    def test_add_points_inherit_scene_metadata(self, tmp_path):
        """Test points inherit scene dimension metadata."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Set scene metadata
            scene.dimension_metadata = [
                DimensionMetadata(name="x", unit="mm"),
                DimensionMetadata(name="y", unit="mm"),
                DimensionMetadata(name="z", unit="mm"),
            ]

            # Add 3D points without specifying metadata
            positions = np.random.rand(30, 3).astype(np.float32)
            points = scene.add_points("inherit_test", positions)

            # Should inherit from scene
            assert "dimension_metadata" in points.attrs
            assert points.attrs["dimension_metadata"][0]["name"] == "x"
            assert points.attrs["dimension_metadata"][1]["unit"] == "mm"

    @pytest.mark.skip(reason="Mixed dimensionality is now allowed in new flexible API")
    def test_mixed_dimensionality(self, tmp_path):
        """Test scene with mixed dimensionality points."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Add 3D points
            pos_3d = np.random.rand(20, 3).astype(np.float32)
            metadata_3d = [
                DimensionMetadata(name="x", unit="um"),
                DimensionMetadata(name="y", unit="um"),
                DimensionMetadata(name="z", unit="um"),
            ]
            scene.add_points("points_3d", pos_3d, dimension_metadata=metadata_3d)

            # Add 2D points (should work independently)
            pos_2d = np.random.rand(30, 2).astype(np.float32)
            metadata_2d = [
                DimensionMetadata(name="x", unit="px"),
                DimensionMetadata(name="y", unit="px"),
            ]
            points_2d = scene.add_points(
                "points_2d", pos_2d, dimension_metadata=metadata_2d
            )

            # Each should have its own metadata
            assert len(points_2d.attrs["dimension_metadata"]) == 2
