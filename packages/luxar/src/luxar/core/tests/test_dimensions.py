"""Tests for scene-level dimensions."""

import numpy as np
import pytest

from luxar.dimensions import Dimension, Dimensions


class TestDimension:
    """Test the Dimension class."""

    def test_dimension_creation(self) -> None:
        """Test creating dimensions with various parameters."""
        # Basic dimension
        dim = Dimension("x", unit="um")
        assert dim.name == "x"
        assert dim.unit == "um"
        assert dim.display is True  # Default
        assert dim.scale == 1.0

        # Full specification
        dim = Dimension(
            "time",
            unit="s",
            range=(0, 10),
            step=0.5,
            display=False,
            discrete=True,
            cyclic=False,
            scale=2.0,
            description="Time dimension",
        )
        assert dim.name == "time"
        assert dim.range == (0, 10)
        assert dim.step == 0.5
        assert dim.display is False
        assert dim.discrete is True
        assert dim.description == "Time dimension"

    def test_dimension_validation(self) -> None:
        """Test dimension parameter validation."""
        # Invalid range
        with pytest.raises(ValueError, match="Range must be a tuple"):
            Dimension("x", range=[0, 1, 2])

        with pytest.raises(ValueError, match="min must be less than max"):
            Dimension("x", range=(10, 5))

        # Invalid step
        with pytest.raises(ValueError, match="Step size must be positive"):
            Dimension("x", step=-1)

        # Invalid scale
        with pytest.raises(ValueError, match="Scale must be positive"):
            Dimension("x", scale=0)

    def test_get_step(self) -> None:
        """Test automatic step calculation."""
        # Explicit step
        dim = Dimension("x", step=0.25)
        assert dim.get_step() == 0.25

        # Discrete dimension
        dim = Dimension("channel", discrete=True)
        assert dim.get_step() == 1.0

        # Auto-calculated from range
        dim = Dimension("y", range=(0, 100))
        assert dim.get_step() == 1.0  # 1% of range

        # Default fallback
        dim = Dimension("z")
        assert dim.get_step() == 0.1

    def test_dimension_serialization(self) -> None:
        """Test to_dict and from_dict."""
        dim = Dimension(
            "time", unit="ms", range=(0, 1000), step=10, display=False, discrete=True
        )

        # Serialize
        data = dim.to_dict()
        assert data["name"] == "time"
        assert data["unit"] == "ms"
        assert data["range"] == [0, 1000]
        assert data["step"] == 10
        assert data["display"] is False

        # Deserialize
        dim2 = Dimension.from_dict(data)
        assert dim2.name == dim.name
        assert dim2.unit == dim.unit
        assert dim2.range == dim.range
        assert dim2.step == dim.step
        assert dim2.display == dim.display


class TestDimensions:
    """Test the Dimensions container class."""

    def test_dimensions_creation(self) -> None:
        """Test creating dimension collections."""
        dims = Dimensions([Dimension("x"), Dimension("y"), Dimension("z")])
        assert dims.ndim == 3
        assert dims.names == ["x", "y", "z"]
        assert dims.displayed == [0, 1, 2]
        assert dims.non_displayed == []

    def test_dimension_validation(self) -> None:
        """Test dimensions validation rules."""
        # Duplicate names
        with pytest.raises(ValueError, match="must be unique"):
            Dimensions([Dimension("x"), Dimension("x")])  # Duplicate

        # Too many displayed
        with pytest.raises(ValueError, match="Maximum 3 dimensions"):
            Dimensions(
                [
                    Dimension("x", display=True),
                    Dimension("y", display=True),
                    Dimension("z", display=True),
                    Dimension("t", display=True),  # 4th displayed
                ]
            )

        # None displayed
        with pytest.raises(ValueError, match="At least one dimension"):
            Dimensions([Dimension("x", display=False), Dimension("y", display=False)])

    def test_dimension_access(self) -> None:
        """Test accessing dimensions."""
        dims = Dimensions(
            [
                Dimension("time", unit="s", display=False),
                Dimension("x", unit="px"),
                Dimension("y", unit="px"),
            ]
        )

        # Get by name
        time_dim = dims.get_dimension("time")
        assert time_dim is not None
        assert time_dim.unit == "s"

        # Get index
        assert dims.get_index("time") == 0
        assert dims.get_index("y") == 2

        # Non-existent
        assert dims.get_dimension("z") is None
        with pytest.raises(ValueError, match="not found"):
            dims.get_index("z")

    def test_position_validation(self) -> None:
        """Test validating positions against dimensions."""
        dims = Dimensions(
            [Dimension("x", range=(-10, 10)), Dimension("y", range=(-5, 5))]
        )

        # Valid positions
        positions = np.array([[0, 0], [5, 2], [-5, -2]])
        dims.validate_positions(positions)  # Should not raise

        # Wrong shape
        with pytest.raises(ValueError, match="must be a 2D array"):
            dims.validate_positions(np.array([1, 2, 3]))

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="has 3 dimensions"):
            dims.validate_positions(np.array([[1, 2, 3]]))

        # Out of range
        positions = np.array([[15, 0]])  # x=15 > 10
        with pytest.raises(ValueError, match="outside range"):
            dims.validate_positions(positions)

    def test_dimensions_serialization(self) -> None:
        """Test serialization of dimension collections."""
        dims = Dimensions(
            [
                Dimension("t", unit="s", display=False),
                Dimension("x", unit="m"),
                Dimension("y", unit="m"),
            ]
        )

        # Serialize
        data = dims.to_dict()
        assert len(data["dimensions"]) == 3
        assert data["dimensions"][0]["name"] == "t"

        # Deserialize
        dims2 = Dimensions.from_dict(data)
        assert dims2.ndim == 3
        assert dims2.names == ["t", "x", "y"]
        assert dims2.displayed == [1, 2]

    def test_convenience_constructors(self) -> None:
        """Test convenience constructor methods."""
        # 2D
        dims = Dimensions.default_2d()
        assert dims.ndim == 2
        assert dims.names == ["x", "y"]
        assert len(dims.displayed) == 2

        # 3D
        dims = Dimensions.default_3d()
        assert dims.ndim == 3
        assert dims.names == ["x", "y", "z"]
        assert len(dims.displayed) == 3

        # Time series
        dims = Dimensions.default_timeseries(n_timepoints=100)
        assert dims.ndim == 4
        assert dims.names == ["t", "x", "y", "z"]
        assert dims.displayed == [1, 2, 3]  # t is hidden
        assert dims.dimensions[0].discrete is True

        # Multichannel
        dims = Dimensions.default_multichannel(n_channels=5)
        assert dims.ndim == 4
        assert dims.names == ["c", "x", "y", "z"]
        assert dims.dimensions[0].range == (0, 4)

    def test_from_positions(self) -> None:
        """Test inferring dimensions from positions."""
        # 3D positions
        positions = np.random.randn(100, 3)
        dims = Dimensions.from_positions(positions)
        assert dims.ndim == 3
        assert dims.names == ["x", "y", "z"]
        assert all(d.range is None for d in dims.dimensions)  # No ranges set

        # 5D with custom names
        positions = np.random.randn(50, 5)
        dims = Dimensions.from_positions(positions, names=["t", "x", "y", "z", "c"])
        assert dims.names == ["t", "x", "y", "z", "c"]
        assert dims.displayed == [0, 1, 2]  # First 3 dimensions are displayed

        # Invalid inputs
        with pytest.raises(ValueError, match="must be 2D array"):
            Dimensions.from_positions(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="Got 2 names for 3 dimensions"):
            Dimensions.from_positions(np.random.randn(10, 3), names=["x", "y"])


class TestSceneIntegration:
    """Test dimensions integration with Scene."""

    # Test removed: Scene dimension validation was removed in new flexible API
    # The Scene class now accepts any dimension values without validation

    def test_scene_dimension_persistence(self, tmp_path) -> None:
        """Test dimensions are saved and loaded correctly."""
        import zarr

        from luxar import LuxarZarrCompiler

        # Create scene with dimensions
        dims = Dimensions.default_timeseries()
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            compiler.create_scene(dimensions=dims)
            # Context manager handles finalization

        # Load and check
        root = zarr.open_group(tmp_path / "test.zarr", mode="r")
        assert "scene_dimensions" in root.attrs

        # Recreate from saved data
        loaded_dims = Dimensions.from_dict(root.attrs["scene_dimensions"])
        assert loaded_dims.ndim == 4
        assert loaded_dims.names == ["t", "x", "y", "z"]
