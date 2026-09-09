"""Structure-aware points readback from compiled scenes."""

from pathlib import Path

import numpy as np
import pytest

from luxar._zarr_compat import consolidate, open_group
from luxar.core.dimensions import Dimensions
from luxar.io.compiler import LuxarZarrCompiler
from luxar.io.reader import LuxarScene


def _expected_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(6, dtype=np.float32)
    positions = np.column_stack((indices, indices * 2, indices * -1))
    colors = np.column_stack(
        (indices / 5, indices[::-1] / 5, np.full(6, 0.5, dtype=np.float32))
    )
    return positions, colors, indices + 1, indices / 5


def _write_structured_points_scene(path: Path) -> None:
    positions, colors, radii, sharpness = _expected_points()
    with LuxarZarrCompiler(path) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        compiler.write_points(
            "cloud/child_0",
            np.array([[99.0, 99.0, 99.0]], dtype=np.float32),
        )
        compiler.write_points(
            "cloud/child_1/part_0/additive_0",
            positions[:2],
            colors=colors[:2],
            radii=radii[:2],
            sharpness=sharpness[:2],
        )
        compiler.write_points(
            "cloud/child_1/part_0/additive_1",
            positions[2:4],
            colors=colors[2:4],
            radii=radii[2:4],
            sharpness=sharpness[2:4],
        )
        compiler.write_points(
            "cloud/child_1/part_1",
            positions[4:],
            colors=colors[4:],
            radii=radii[4:],
            sharpness=sharpness[4:],
        )

    root = open_group(path, mode="a")
    root["cloud"].attrs.update({"type": "group", "kind": "lod"})
    root["cloud"]["child_1"].attrs.update({"type": "group", "kind": "partition"})
    consolidate(root)


def test_flatten_reconstructs_the_finest_points_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_path = tmp_path / "structured.luxar.zarr"
    _write_structured_points_scene(scene_path)

    scene = LuxarScene.load(scene_path)
    data = scene.get_points("cloud", flatten=True)
    assert data.colors is not None
    assert data.radii is not None
    assert data.sharpness is not None

    decode = scene._decoder.decode

    def reject_positions(array, root):
        if array.name.endswith("/positions"):
            pytest.fail("single-field read must not decode shared positions")
        return decode(array, root)

    monkeypatch.setattr(scene._decoder, "decode", reject_positions)
    colors_only = scene.get_point_array("cloud", "colors", flatten=True)
    assert colors_only is not None

    order = np.argsort(data.positions[:, 0])
    expected_positions, expected_colors, expected_radii, expected_sharpness = (
        _expected_points()
    )
    np.testing.assert_allclose(data.positions[order], expected_positions, atol=1e-6)
    np.testing.assert_allclose(data.colors[order], expected_colors, atol=2.0 / 255)
    np.testing.assert_allclose(colors_only[order], expected_colors, atol=2.0 / 255)
    np.testing.assert_allclose(data.radii[order], expected_radii, atol=0.03)
    np.testing.assert_allclose(
        data.sharpness[order], expected_sharpness, atol=2.0 / 255
    )
    assert data.chunk_bounds is not None
    assert data.metadata["type"] == "points"
    assert data.metadata["kind"] == "lod"
    assert data.metadata["n_points"] == 6


def test_flatten_rejects_inconsistent_optional_arrays(tmp_path: Path) -> None:
    scene_path = tmp_path / "structured.luxar.zarr"
    _write_structured_points_scene(scene_path)
    root = open_group(scene_path, mode="a")
    del root["cloud"]["child_1"]["part_0"]["additive_1"]["colors"]
    consolidate(root)

    with pytest.raises(ValueError, match="inconsistent 'colors'"):
        LuxarScene.load(scene_path).get_points("cloud", flatten=True)
