"""Test rendering attributes for Node class."""

import pytest

from luxar.scene import Scene


class TestNodeRenderingAttributes:
    """Test Node rendering properties (opacity, gamma, blending_mode)."""

    def test_default_rendering_attributes(self, tmp_path):
        """Test that nodes have correct default rendering attributes."""
        scene = Scene(tmp_path / "test.zarr")
        node = scene.add_group("test_node")

        assert node.opacity == 1.0
        assert node.gamma == 1.0
        assert node.blending_mode == "additive"

    def test_opacity_getter_setter(self, tmp_path):
        """Test opacity property getter and setter."""
        scene = Scene(tmp_path / "test.zarr")
        node = scene.add_group("test_node")

        # Test setting valid opacity
        node.opacity = 0.5
        assert node.opacity == 0.5
        assert node.attrs["opacity"] == 0.5

        # Test edge cases
        node.opacity = 0.0
        assert node.opacity == 0.0

        node.opacity = 1.0
        assert node.opacity == 1.0

        # Test type conversion
        node.opacity = 0.7
        assert node.opacity == 0.7

        node.opacity = "0.3"
        assert node.opacity == 0.3

    def test_opacity_validation(self, tmp_path):
        """Test opacity validation."""
        scene = Scene(tmp_path / "test.zarr")
        node = scene.add_group("test_node")

        # Test invalid values
        with pytest.raises(ValueError, match="Opacity must be between 0.0 and 1.0"):
            node.opacity = -0.1

        with pytest.raises(ValueError, match="Opacity must be between 0.0 and 1.0"):
            node.opacity = 1.5

        with pytest.raises(TypeError, match="Opacity must be convertible to float"):
            node.opacity = "invalid"

    def test_gamma_getter_setter(self, tmp_path):
        """Test gamma property getter and setter."""
        scene = Scene(tmp_path / "test.zarr")
        node = scene.add_group("test_node")

        # Test setting valid gamma
        node.gamma = 1.5
        assert node.gamma == 1.5
        assert node.attrs["gamma"] == 1.5

        # Test edge cases
        node.gamma = 0.2
        assert node.gamma == 0.2

        node.gamma = 2.0
        assert node.gamma == 2.0

        # Test type conversion
        node.gamma = 1
        assert node.gamma == 1.0

        node.gamma = "1.8"
        assert node.gamma == 1.8

    def test_gamma_validation(self, tmp_path):
        """Test gamma validation."""
        scene = Scene(tmp_path / "test.zarr")
        node = scene.add_group("test_node")

        # Test invalid values
        with pytest.raises(ValueError, match="Gamma must be between 0.2 and 2.0"):
            node.gamma = 0.1

        with pytest.raises(ValueError, match="Gamma must be between 0.2 and 2.0"):
            node.gamma = 2.5

        with pytest.raises(TypeError, match="Gamma must be convertible to float"):
            node.gamma = "not_a_number"

    def test_blending_mode_getter_setter(self, tmp_path):
        """Test blending_mode property getter and setter."""
        scene = Scene(tmp_path / "test.zarr")
        node = scene.add_group("test_node")

        # Test setting valid blending modes
        for mode in ["normal", "additive", "subtractive", "minimum", "maximum"]:
            node.blending_mode = mode
            assert node.blending_mode == mode
            assert node.attrs["blending_mode"] == mode

    def test_blending_mode_validation(self, tmp_path):
        """Test blending mode validation."""
        scene = Scene(tmp_path / "test.zarr")
        node = scene.add_group("test_node")

        # Test invalid values
        with pytest.raises(ValueError, match="Invalid blending mode"):
            node.blending_mode = "invalid"

        with pytest.raises(ValueError, match="Invalid blending mode"):
            node.blending_mode = "overlay"

        with pytest.raises(TypeError, match="Blending mode must be a string"):
            node.blending_mode = 123

    def test_rendering_attributes_persistence(self, tmp_path):
        """Test that rendering attributes are persisted to zarr attrs."""
        import zarr

        store_path = tmp_path / "test.zarr"

        # Create scene and set attributes
        scene = Scene(store_path)
        node = scene.add_group("test_node")
        node.opacity = 0.7
        node.gamma = 1.5
        node.blending_mode = "subtractive"

        # Check that attributes are written to zarr
        store = zarr.open_group(store_path, mode="r")
        test_node_attrs = store["test_node"].attrs
        assert test_node_attrs.get("opacity") == 0.7
        assert test_node_attrs.get("gamma") == 1.5
        assert test_node_attrs.get("blending_mode") == "subtractive"

    def test_rendering_attributes_in_add_points(self, tmp_path):
        """Test setting rendering attributes when adding points."""
        import numpy as np

        scene = Scene(tmp_path / "test.zarr")
        positions = np.random.randn(100, 3).astype(np.float32)

        # Add points with custom rendering attributes
        points = scene.add_points(
            "test_points", positions, opacity=0.5, gamma=1.2, blending_mode="normal"
        )

        assert points.opacity == 0.5
        assert points.gamma == 1.2
        assert points.blending_mode == "normal"

    def test_rendering_attributes_inheritance(self, tmp_path):
        """Test that child nodes can access parent rendering attributes."""
        scene = Scene(tmp_path / "test.zarr")

        # Create parent with custom attributes
        parent = scene.add_group("parent", opacity=0.3, blending_mode="subtractive")
        assert parent.opacity == 0.3
        assert parent.blending_mode == "subtractive"

        # Create child without specifying attributes
        child = parent.add_group("child")
        # Child should have its own default values (not inherit from parent in Python)
        assert child.opacity == 1.0
        assert child.blending_mode == "additive"

        # But parent attributes are stored for TypeScript inheritance
        assert parent.attrs["opacity"] == 0.3
        assert parent.attrs["blending_mode"] == "subtractive"
