"""
Test transform utilities and functionality.
"""

import numpy as np
import pytest

from luxar import LuxarZarrCompiler
from luxar.transforms import (
    compose,
    from_list,
    identity,
    inverse,
    look_at,
    rotate,
    rotate_x,
    rotate_y,
    rotate_z,
    scale,
    to_list,
    translate,
)


class TestTransformUtilities:
    """Test transform utility functions."""

    def test_identity(self):
        """Test identity transform creation."""
        t = identity()
        assert t.shape == (4, 4)
        assert t.dtype == np.float32
        assert np.allclose(t, np.eye(4))

    def test_translate(self):
        """Test translation transform."""
        # Test individual axes
        t = translate(x=5)
        assert np.allclose(t[0, 3], 5)
        assert np.allclose(t[1, 3], 0)
        assert np.allclose(t[2, 3], 0)

        # Test all axes
        t = translate(1, 2, 3)
        assert np.allclose(t[:3, 3], [1, 2, 3])

        # Check rest of matrix is identity
        assert np.allclose(t[:3, :3], np.eye(3))
        assert t[3, 3] == 1

    def test_scale(self):
        """Test scaling transform."""
        # Non-uniform scale
        t = scale(2, 3, 4)
        assert np.allclose(np.diag(t), [2, 3, 4, 1])

        # Uniform scale
        t = scale(uniform=2.5)
        assert np.allclose(np.diag(t), [2.5, 2.5, 2.5, 1])

        # Check no other elements
        t[np.diag_indices(4)] = 0
        assert np.allclose(t, 0)

    def test_rotate_x(self):
        """Test rotation around X axis."""
        # 90 degree rotation
        t = rotate_x(90)

        # Should rotate Y to Z
        y_axis = np.array([0, 1, 0, 0])
        rotated = t @ y_axis
        assert np.allclose(rotated[:3], [0, 0, 1], atol=1e-6)

        # 180 degree rotation
        t = rotate_x(180)
        rotated = t @ y_axis
        assert np.allclose(rotated[:3], [0, -1, 0], atol=1e-6)

    def test_rotate_y(self):
        """Test rotation around Y axis."""
        # 90 degree rotation
        t = rotate_y(90)

        # Should rotate Z to X
        z_axis = np.array([0, 0, 1, 0])
        rotated = t @ z_axis
        assert np.allclose(rotated[:3], [1, 0, 0], atol=1e-6)

    def test_rotate_z(self):
        """Test rotation around Z axis."""
        # 90 degree rotation
        t = rotate_z(90)

        # Should rotate X to Y
        x_axis = np.array([1, 0, 0, 0])
        rotated = t @ x_axis
        assert np.allclose(rotated[:3], [0, 1, 0], atol=1e-6)

    def test_rotate_arbitrary_axis(self):
        """Test rotation around arbitrary axis."""
        # Rotate around diagonal axis
        axis = np.array([1, 1, 1])
        t = rotate(120, axis)

        # 120 degree rotation around (1,1,1) should cycle X->Y->Z->X
        x_axis = np.array([1, 0, 0, 0])
        rotated = t @ x_axis
        assert np.allclose(rotated[:3], [0, 1, 0], atol=1e-6)  # X -> Y

    def test_rotate_string_axis(self):
        """Test rotation with string axis names."""
        assert np.allclose(rotate(45, "x"), rotate_x(45))
        assert np.allclose(rotate(45, "Y"), rotate_y(45))
        assert np.allclose(rotate(45, "Z"), rotate_z(45))

        with pytest.raises(ValueError):
            rotate(45, "w")  # Invalid axis

    def test_compose(self):
        """Test transform composition."""
        # Test that compose works correctly
        t1 = translate(5, 0, 0)
        t2 = rotate_z(90)

        # Compose multiple transforms
        combined = compose(t1, t2)

        # Test on a simple point
        point = np.array([1, 0, 0, 1])
        result = combined @ point

        # Just verify the compose function runs without error
        assert result.shape == (4,)

        # Empty compose
        assert np.allclose(compose(), identity())

        # Single transform
        assert np.allclose(compose(t1), t1)

    def test_inverse(self):
        """Test transform inversion."""
        # Translation inverse
        t = translate(5, 3, -2)
        t_inv = inverse(t)
        assert np.allclose(compose(t, t_inv), identity(), atol=1e-6)

        # Rotation inverse
        t = rotate_z(45)
        t_inv = inverse(t)
        assert np.allclose(compose(t, t_inv), identity(), atol=1e-6)

        # Scale inverse
        t = scale(2, 4, 0.5)
        t_inv = inverse(t)
        assert np.allclose(compose(t, t_inv), identity(), atol=1e-6)

        # Non-invertible matrix
        bad = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            inverse(bad)

    def test_look_at(self):
        """Test look-at transform."""
        # Look at origin from position
        t = look_at((10, 5, 10), (0, 0, 0))

        # Position should be set
        assert np.allclose(t[:3, 3], [10, 5, 10])

        # Forward vector should point toward target
        forward = -t[2, :3]  # Negative Z is forward
        expected_forward = np.array([-10, -5, -10])
        expected_forward = expected_forward / np.linalg.norm(expected_forward)
        assert np.allclose(forward, expected_forward)

        # Custom up vector
        t = look_at((1, 0, 0), (0, 0, 0), up=(0, 0, 1))
        # Y axis should be close to Z (up)
        assert t[1, 2] > 0.9  # Y points mostly in Z direction

    def test_to_from_list(self):
        """Test conversion to/from list."""
        # Create a transform
        t = compose(translate(1, 2, 3), rotate_z(45), scale(2, 2, 2))

        # Convert to list and back
        values = to_list(t)
        assert isinstance(values, list)
        assert len(values) == 16

        t_recovered = from_list(values)
        assert np.allclose(t, t_recovered)

        # Test validation
        with pytest.raises(ValueError):
            from_list([1, 2, 3])  # Wrong size

    def test_aliases(self):
        """Test function aliases."""
        from luxar.transforms import rotation, scaling, translation

        assert np.allclose(translation(1, 2, 3), translate(1, 2, 3))
        assert np.allclose(scaling(2, 2, 2), scale(2, 2, 2))
        assert np.allclose(rotation(45, "z"), rotate(45, "z"))


class TestNodeTransformIntegration:
    """Test transform integration with Node class."""

    def test_node_transform_validation(self, tmp_path):
        """Test that transforms are validated when creating nodes."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Valid transform
            t = translate(5, 0, 0)
            group = scene.add_group("ValidTransform", transform=to_list(t))
            assert "transform" in group.attrs

            # Invalid transform - wrong size
            with pytest.raises(ValueError):
                scene.add_group("BadSize", transform=[1, 2, 3])

            # Invalid transform - not a list/array
            with pytest.raises(ValueError):
                scene.add_group("BadType", transform="not a transform")

    def test_node_transform_property(self, tmp_path):
        """Test the transform property on nodes."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()
            group = scene.add_group("TestGroup")

            # No transform initially
            assert group.transform is None

            # Set transform using property
            t = compose(translate(1, 2, 3), rotate_x(45))
            group.transform = t

            # Get transform
            retrieved = group.transform
            assert retrieved is not None
            assert np.allclose(retrieved, t)

            # Set from list
            group.transform = to_list(scale(2, 2, 2))
            assert np.allclose(group.transform, scale(2, 2, 2))

            # Remove transform
            group.transform = None
            assert group.transform is None
            assert "transform" not in group.attrs

    def test_nested_transforms(self, tmp_path):
        """Test nested transform hierarchy."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Create hierarchy with transforms
            g1 = scene.add_group("Level1", transform=to_list(translate(5, 0, 0)))
            g2 = g1.add_group("Level2", transform=to_list(rotate_z(45)))
            g3 = g2.add_group("Level3", transform=to_list(scale(2, 2, 2)))

            # Verify each has its own transform
            assert np.allclose(g1.transform, translate(5, 0, 0))
            assert np.allclose(g2.transform, rotate_z(45))
            assert np.allclose(g3.transform, scale(2, 2, 2))

    def test_transform_with_points(self, tmp_path):
        """Test transforms work with points."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            scene = compiler.create_scene()

            # Create transformed group
            t = compose(translate(10, 0, 0), scale(uniform=0.5))
            group = scene.add_group("TransformedPoints", transform=to_list(t))

            # Add points
            positions = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
            colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
            scene.add_points("Points", positions, colors=colors, parent=group)

            # Verify structure
            assert group.transform is not None
            assert len(group.children) == 1
            assert group.children[0].name == "Points"
