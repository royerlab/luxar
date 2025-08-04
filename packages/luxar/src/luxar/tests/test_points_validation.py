import numpy as np
import pytest

from luxar import Scene


def test_bad_positions_shape(tmp_path):
    store = tmp_path / "bad.zarr"
    scene = Scene(store)
    with pytest.raises(ValueError, match="Positions must"):
        scene.add_points("Broken", np.ones((3,)), parent=scene)


def test_mismatched_colors(tmp_path):
    store = tmp_path / "bad2.zarr"
    scene = Scene(store)
    pos = np.ones((10, 3), np.float32)
    col = np.ones((5, 3), np.uint8)
    with pytest.raises(ValueError, match="Colors must"):
        scene.add_points("Nope", pos, col, parent=scene)
