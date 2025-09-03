import numpy as np
import pytest
import zarr

from luxar import LuxarZarrCompiler
from luxar.typing_utils.datatypes import DataTypeConfig, DataTypeMode


def test_bad_positions_shape(tmp_path):
    store = tmp_path / "bad.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene()
        with pytest.raises(ValueError, match="Positions must"):
            scene.add_points("Broken", np.ones((3,)), parent=scene)


def test_mismatched_colors(tmp_path):
    store = tmp_path / "bad2.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene()
        pos = np.ones((10, 3), np.float32)
        col = np.ones((5, 3), np.uint8)
        with pytest.raises(ValueError, match="Colors must"):
            scene.add_points("Nope", pos, col, parent=scene)


def test_valid_radii(tmp_path):
    """Test that valid radii are accepted and stored correctly."""
    store = tmp_path / "radii_test.zarr"
    # Use PRECISION mode to preserve float32 for test consistency
    dtype_config = DataTypeConfig(mode=DataTypeMode.PRECISION)
    with LuxarZarrCompiler(store, dtype_config=dtype_config) as compiler:
        compiler.create_scene()

        positions = np.random.randn(100, 3).astype(np.float32)
        radii = np.random.uniform(0.1, 2.0, 100).astype(np.float32)

        compiler.write_points("test", positions, radii=radii)

    # Verify radii were stored
    root = zarr.open_group(store, mode="r")
    assert "test/radii" in root
    stored_radii = root["test/radii"][:]
    np.testing.assert_array_equal(stored_radii, radii)


def test_negative_radii(tmp_path):
    """Radii with negative values should fail."""
    store = tmp_path / "negative_radii.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene()

        positions = np.random.randn(100, 3).astype(np.float32)
        radii = np.random.uniform(-1.0, 1.0, 100).astype(np.float32)  # Some negative

        # Our new validation provides more helpful error messages
        from luxar.validation import ValidationError

        with pytest.raises(ValidationError, match="Radii must be positive"):
            compiler.write_points("test", positions, radii=radii)


def test_mismatched_radii(tmp_path):
    """Radii with wrong number of points should fail."""
    store = tmp_path / "mismatched_radii.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene()

        positions = np.random.randn(100, 3).astype(np.float32)
        radii = np.random.uniform(0.1, 2.0, 50).astype(np.float32)  # Wrong N

        # Our new validation provides more helpful error messages
        from luxar.validation import ValidationError

        with pytest.raises(ValidationError, match="doesn't match number of points"):
            compiler.write_points("test", positions, radii=radii)


def test_wrong_shape_radii(tmp_path):
    """Radii with wrong dimensions should fail."""
    store = tmp_path / "wrong_shape_radii.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene()

        positions = np.random.randn(100, 3).astype(np.float32)
        radii = np.random.uniform(0.1, 2.0, (100, 2)).astype(np.float32)  # Wrong shape

        # Our new validation provides more helpful error messages
        from luxar.validation import ValidationError

        with pytest.raises(ValidationError, match="Expected 1D array"):
            compiler.write_points("test", positions, radii=radii)


def test_valid_sharpness(tmp_path):
    """Test that valid sharpness values are accepted and stored correctly."""
    store = tmp_path / "sharpness_test.zarr"
    # Use PRECISION mode to preserve float32 for test consistency
    dtype_config = DataTypeConfig(mode=DataTypeMode.PRECISION)
    with LuxarZarrCompiler(store, dtype_config=dtype_config) as compiler:
        compiler.create_scene()

        positions = np.random.randn(100, 3).astype(np.float32)
        sharpness = np.random.uniform(0.5, 10.0, 100).astype(np.float32)

        compiler.write_points("test", positions, sharpness=sharpness)

    # Verify sharpness was stored
    root = zarr.open_group(store, mode="r")
    assert "test/sharpness" in root
    stored_sharpness = root["test/sharpness"][:]
    np.testing.assert_array_equal(stored_sharpness, sharpness)


def test_negative_sharpness(tmp_path):
    """Sharpness with negative values should fail."""
    store = tmp_path / "negative_sharpness.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene()

        positions = np.random.randn(100, 3).astype(np.float32)
        sharpness = np.random.uniform(-1.0, 1.0, 100).astype(
            np.float32
        )  # Some negative

        # Our new validation provides more helpful error messages
        from luxar.validation import ValidationError

        with pytest.raises(ValidationError, match="Sharpness must be positive"):
            compiler.write_points("test", positions, sharpness=sharpness)


def test_mismatched_sharpness(tmp_path):
    """Sharpness with wrong number of points should fail."""
    store = tmp_path / "mismatched_sharpness.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene()

        positions = np.random.randn(100, 3).astype(np.float32)
        sharpness = np.random.uniform(0.5, 10.0, 50).astype(np.float32)  # Wrong N

        # Our new validation provides more helpful error messages
        from luxar.validation import ValidationError

        with pytest.raises(ValidationError, match="doesn't match"):
            compiler.write_points("test", positions, sharpness=sharpness)


def test_wrong_shape_sharpness(tmp_path):
    """Sharpness with wrong dimensions should fail."""
    store = tmp_path / "wrong_shape_sharpness.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene()

        positions = np.random.randn(100, 3).astype(np.float32)
        sharpness = np.random.uniform(0.5, 10.0, (100, 2)).astype(
            np.float32
        )  # Wrong shape

        # Our new validation provides more helpful error messages
        from luxar.validation import ValidationError

        with pytest.raises(ValidationError, match="Expected 1D array"):
            compiler.write_points("test", positions, sharpness=sharpness)


def test_sharpness_warning(tmp_path):
    """Test that out-of-range sharpness values trigger a warning."""
    store = tmp_path / "sharpness_warning.zarr"
    with LuxarZarrCompiler(store) as compiler:
        compiler.create_scene()

        positions = np.random.randn(100, 3).astype(np.float32)
        # Mix of values including out-of-range
        sharpness = np.array([0.3, 2.0, 15.0] * 33 + [5.0]).astype(
            np.float32
        )  # 100 values

        with pytest.warns(UserWarning, match="Using extreme sharpness values"):
            compiler.write_points("test", positions, sharpness=sharpness)
