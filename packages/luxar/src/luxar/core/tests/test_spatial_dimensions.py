"""Tests for spatial dimension functionality in Luxar.

This module tests the spatial flag implementation for dimensions,
which controls whether points extend through a dimension spatially.
"""

import numpy as np
import pytest

from luxar import Dimension, Dimensions


class TestSpatialDimensions:
    """Test spatial flag functionality in Dimension and Dimensions classes."""

    def test_spatial_flag_auto_determination(self) -> None:
        """Test automatic spatial flag determination based on dimension properties."""
        # Displayed dimension should be spatial
        dim1 = Dimension("x", display=True)
        assert dim1.spatial is True

        # Discrete dimension should not be spatial
        dim2 = Dimension("time", discrete=True, display=False)
        assert dim2.spatial is False

        # Non-displayed continuous dimension defaults to non-spatial
        dim3 = Dimension("depth", display=False)
        assert dim3.spatial is False

        # Explicit override to make non-displayed dimension spatial
        dim4 = Dimension("depth", display=False, spatial=True)
        assert dim4.spatial is True

    def test_displayed_dimensions_always_spatial(self) -> None:
        """Test that displayed dimensions are always spatial by default."""
        # Displayed dimensions auto-determine to spatial
        dim_y = Dimension("y", display=True)
        assert dim_y.spatial is True

        dim_z = Dimension("z", display=True, discrete=False)
        assert dim_z.spatial is True

        # Note: Explicitly setting spatial=False is allowed but not recommended for displayed dims
        dim_x = Dimension("x", display=True, spatial=False)
        assert dim_x.spatial is False  # Explicit override is respected

    def test_discrete_spatial_validation(self) -> None:
        """Test that discrete dimensions cannot be forced spatial unless displayed."""
        # This should raise ValueError - discrete non-displayed can't be spatial
        with pytest.raises(ValueError, match="cannot be both discrete and spatial"):
            Dimension("time", discrete=True, spatial=True, display=False)

        # This should be OK - displayed discrete can be spatial
        dim = Dimension("category", discrete=True, spatial=True, display=True)
        assert dim.spatial is True
        assert dim.discrete is True

    def test_non_spatial_auto_correction_to_discrete(self) -> None:
        """Test that non-spatial, non-displayed dimensions are auto-corrected to discrete."""
        import warnings

        # Non-spatial, non-displayed, continuous should auto-correct to discrete with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dim = Dimension("depth", display=False, spatial=False, discrete=False)

            # Check that it was auto-corrected
            assert dim.discrete is True
            assert dim.spatial is False

            # Check that a warning was issued
            assert len(w) == 1
            assert "must be discrete" in str(w[0].message)
            assert "Setting discrete=True automatically" in str(w[0].message)

        # Explicitly setting discrete=True should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dim2 = Dimension("time", display=False, spatial=False, discrete=True)
            assert dim2.discrete is True
            assert len(w) == 0  # No warning

    def test_dimensions_spatial_extend_dims_property(self) -> None:
        """Test the spatial_extend_dims property returns correct flags."""
        dims = Dimensions(
            [
                Dimension("x", display=True),  # spatial=True (displayed)
                Dimension("y", display=True),  # spatial=True (displayed)
                Dimension("z", display=True),  # spatial=True (displayed)
                Dimension("depth", display=False, spatial=True),  # explicit spatial
                Dimension(
                    "time", display=False, discrete=True
                ),  # spatial=False (discrete)
                Dimension(
                    "channel", display=False, spatial=False
                ),  # explicit non-spatial
            ]
        )

        expected = [True, True, True, True, False, False]
        assert dims.spatial_extend_dims == expected

    def test_dimension_serialization_with_spatial_flag(self) -> None:
        """Test that spatial flag is preserved through serialization."""
        # Create dimension with explicit spatial flag
        dim = Dimension("depth", display=False, spatial=True, unit="μm")

        # Serialize to dict
        dim_dict = dim.to_dict()
        assert dim_dict["spatial"] is True

        # Deserialize from dict
        dim_restored = Dimension.from_dict(dim_dict)
        assert dim_restored.spatial is True
        assert dim_restored.display is False
        assert dim_restored.unit == "μm"

    def test_dimensions_serialization_with_spatial_flags(self) -> None:
        """Test that Dimensions preserves spatial flags through serialization."""
        dims = Dimensions(
            [
                Dimension("x", display=True),
                Dimension("y", display=True),
                Dimension("z", display=False, spatial=True),
                Dimension("time", display=False, discrete=True),
            ]
        )

        # Serialize
        dims_dict = dims.to_dict()

        # Deserialize
        dims_restored = Dimensions.from_dict(dims_dict)

        # Check spatial flags are preserved
        assert dims_restored.spatial_extend_dims == [True, True, True, False]

    def test_mixed_spatial_non_spatial_dimensions(self) -> None:
        """Test complex scenarios with mixed spatial and non-spatial dimensions."""
        # Microscopy example: xyz spatial, time discrete, depth spatial, channel discrete
        dims = Dimensions(
            [
                Dimension("x", unit="μm", display=True),  # spatial
                Dimension("y", unit="μm", display=True),  # spatial
                Dimension("z", unit="μm", display=True),  # spatial
                Dimension(
                    "time", unit="s", display=False, discrete=True
                ),  # non-spatial
                Dimension("depth", unit="μm", display=False, spatial=True),  # spatial
                Dimension("channel", display=False, discrete=True),  # non-spatial
            ]
        )

        assert dims.spatial_extend_dims == [True, True, True, False, True, False]
        assert dims.displayed == [0, 1, 2]
        assert dims.non_displayed == [3, 4, 5]

    def test_edge_cases(self) -> None:
        """Test edge cases for spatial dimension handling."""
        # All dimensions displayed (common 3D case)
        dims_3d = Dimensions(
            [
                Dimension("x", display=True),
                Dimension("y", display=True),
                Dimension("z", display=True),
            ]
        )
        assert dims_3d.spatial_extend_dims == [True, True, True]

        # Single dimension
        dims_1d = Dimensions([Dimension("x", display=True)])
        assert dims_1d.spatial_extend_dims == [True]

        # No spatial dimensions (all discrete non-displayed)
        dims_categorical = Dimensions(
            [
                Dimension("view", display=True),  # Still spatial because displayed
                Dimension("category", display=False, discrete=True),
                Dimension("label", display=False, discrete=True),
            ]
        )
        assert dims_categorical.spatial_extend_dims == [True, False, False]

    def test_spatial_flag_none_handling(self) -> None:
        """Test that None spatial flag is properly auto-determined."""
        # Explicitly set spatial=None
        dim1 = Dimension("x", display=True, spatial=None)
        assert dim1.spatial is True  # Auto-determined as True

        dim2 = Dimension("time", display=False, discrete=True, spatial=None)
        assert dim2.spatial is False  # Auto-determined as False

        dim3 = Dimension("depth", display=False, spatial=None)
        assert dim3.spatial is False  # Defaults to False for non-displayed continuous


class TestCompilerIntegration:
    """Test that the compiler correctly extracts and stores spatial metadata."""

    def test_compiler_stores_spatial_metadata(self, tmp_path) -> None:
        """LuxarZarrCompiler stores each dimension's spatial flag inside the
        scene_dimensions attribute."""
        import zarr

        from luxar import Dimension, Dimensions
        from luxar.io import LuxarZarrCompiler

        # Create scene with mixed spatial/non-spatial dimensions
        dimensions = Dimensions(
            [
                Dimension("x", display=True),  # spatial
                Dimension("y", display=True),  # spatial
                Dimension("z", display=True),  # spatial
                Dimension("depth", display=False, spatial=True),  # explicit spatial
                Dimension("time", display=False, discrete=True),  # non-spatial
            ]
        )

        # Add some test points
        positions = np.random.randn(100, 5).astype(np.float32)

        # Compile to zarr
        output_path = tmp_path / "test.luxar.zarr"

        from luxar import Scene

        # Create a scene with the dimensions
        with LuxarZarrCompiler(output_path) as compiler:
            scene = Scene(dimensions=dimensions, writer=compiler)
            scene.add_points("test_points", positions=positions)

        # Read back and check spatial metadata
        store = zarr.open_group(str(output_path), mode="r")

        # Check scene dimensions have spatial flags
        scene_dims = store.attrs["scene_dimensions"]["dimensions"]
        assert scene_dims[0]["spatial"] is True  # x
        assert scene_dims[1]["spatial"] is True  # y
        assert scene_dims[2]["spatial"] is True  # z
        assert scene_dims[3]["spatial"] is True  # depth
        assert scene_dims[4]["spatial"] is False  # time
