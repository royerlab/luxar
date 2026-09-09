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
            "cloud/lod_coarse",
            np.array([[99.0, 99.0, 99.0]], dtype=np.float32),
        )
        compiler.write_points(
            "cloud/lod_fine/part_0/additive_0",
            positions[:2],
            colors=colors[:2],
            radii=radii[:2],
            sharpness=sharpness[:2],
        )
        compiler.write_points(
            "cloud/lod_fine/part_0/additive_1",
            positions[2:4],
            colors=colors[2:4],
            radii=radii[2:4],
            sharpness=sharpness[2:4],
        )
        compiler.write_points(
            "cloud/lod_fine/part_1",
            positions[4:],
            colors=colors[4:],
            radii=radii[4:],
            sharpness=sharpness[4:],
        )

    root = open_group(path, mode="a")
    root["cloud"].attrs.update({"type": "group", "kind": "lod"})
    root["cloud"]["lod_coarse"].attrs["child_index"] = 0
    root["cloud"]["lod_fine"].attrs.update(
        {"type": "group", "kind": "partition", "child_index": 1}
    )
    root["cloud"]["lod_fine"]["part_0"].attrs["type"] = "group"
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

    expected_positions, expected_colors, expected_radii, expected_sharpness = (
        _expected_points()
    )
    np.testing.assert_allclose(data.positions, expected_positions, atol=1e-6)
    np.testing.assert_allclose(data.colors, expected_colors, atol=2.0 / 255)
    np.testing.assert_allclose(colors_only, expected_colors, atol=2.0 / 255)
    np.testing.assert_allclose(data.radii, expected_radii, atol=0.03)
    np.testing.assert_allclose(data.sharpness, expected_sharpness, atol=2.0 / 255)
    assert data.chunk_bounds is None
    assert data.metadata["type"] == "points"
    assert "kind" not in data.metadata
    assert data.metadata["n_points"] == 6

    with pytest.raises(ValueError, match="not available for flattened points"):
        scene.get_point_array("cloud", "chunk_bounds", flatten=True)


def test_flatten_rejects_inconsistent_optional_arrays(tmp_path: Path) -> None:
    scene_path = tmp_path / "structured.luxar.zarr"
    _write_structured_points_scene(scene_path)
    root = open_group(scene_path, mode="a")
    del root["cloud"]["lod_fine"]["part_0"]["additive_1"]["colors"]
    consolidate(root)

    with pytest.raises(ValueError, match="inconsistent 'colors'"):
        LuxarScene.load(scene_path).get_points("cloud", flatten=True)


def test_flatten_rejects_incomplete_additive_ladder(tmp_path: Path) -> None:
    scene_path = tmp_path / "incomplete.luxar.zarr"
    positions, _, _, _ = _expected_points()
    with LuxarZarrCompiler(scene_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "cloud",
            positions,
            additive_lod={"n_lods": 3, "method": "random", "seed": 0},
        )

    root = open_group(scene_path, mode="a")
    del root["cloud"]["additive_1"]
    consolidate(root)

    scene = LuxarScene.load(scene_path)
    with pytest.raises(ValueError, match="has 2 additive increments but declares 3"):
        scene.get_points("cloud", flatten=True)
    with pytest.raises(ValueError, match="has 2 additive increments but declares 3"):
        scene.get_point_array("cloud", "positions", flatten=True)


@pytest.mark.parametrize("damage", ["delete", "untyped"])
def test_flatten_rejects_incomplete_partition(tmp_path: Path, damage: str) -> None:
    scene_path = tmp_path / "incomplete-partition.luxar.zarr"
    positions, _, _, _ = _expected_points()
    with LuxarZarrCompiler(scene_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("cloud", positions, partition={"max_elements": 2})

    root = open_group(scene_path, mode="a")
    if damage == "delete":
        del root["cloud"]["part_0"]
    else:
        del root["cloud"]["part_0"].attrs["type"]
    consolidate(root)

    with pytest.raises(ValueError, match="partition parts but BSP tree declares"):
        LuxarScene.load(scene_path).get_points("cloud", flatten=True)


def test_flatten_validates_stored_point_count(tmp_path: Path) -> None:
    scene_path = tmp_path / "wrong-count.luxar.zarr"
    positions, _, _, _ = _expected_points()
    with LuxarZarrCompiler(scene_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "cloud",
            positions,
            additive_lod={"n_lods": 2, "method": "random", "seed": 0},
        )

    root = open_group(scene_path, mode="a")
    root["cloud"].attrs["n_points"] = len(positions) + 1
    consolidate(root)

    with pytest.raises(ValueError, match="has 6 decoded positions but declares 7"):
        LuxarScene.load(scene_path).get_points("cloud", flatten=True)


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({"type": "group", "kind": "lod"}, "empty LOD structure"),
        ({"type": "group", "kind": "partition"}, "empty partition"),
        ({"type": "group"}, "is not a points node"),
    ],
)
def test_flatten_rejects_invalid_structures(
    tmp_path: Path, attrs: dict[str, str], message: str
) -> None:
    scene_path = tmp_path / "invalid.luxar.zarr"
    with LuxarZarrCompiler(scene_path) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())

    root = open_group(scene_path, mode="a")
    root.require_group("cloud").attrs.update(attrs)
    consolidate(root)

    with pytest.raises(ValueError, match=message):
        LuxarScene.load(scene_path).get_points("cloud", flatten=True)


@pytest.mark.parametrize("transform_name", ["transform", "nd_transform"])
def test_flatten_rejects_descendant_transforms(
    tmp_path: Path, transform_name: str
) -> None:
    scene_path = tmp_path / "structured.luxar.zarr"
    _write_structured_points_scene(scene_path)
    root = open_group(scene_path, mode="a")
    value = (
        np.eye(4).tolist() if transform_name == "transform" else {"T": {"offset": 1}}
    )
    root["cloud"]["lod_fine"]["part_1"].attrs[transform_name] = value
    consolidate(root)

    with pytest.raises(ValueError, match=rf"descendant .* carries {transform_name}"):
        LuxarScene.load(scene_path).get_points("cloud", flatten=True)


def test_get_point_array_preserves_flat_leaf_behavior(tmp_path: Path) -> None:
    scene_path = tmp_path / "flat.luxar.zarr"
    positions, colors, _, _ = _expected_points()
    with LuxarZarrCompiler(scene_path) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        compiler.write_points("cloud", positions, colors=colors)

    scene = LuxarScene.load(scene_path)
    decoded = scene.get_point_array("cloud", "positions")
    assert decoded is not None
    np.testing.assert_allclose(decoded, scene.get_points("cloud").positions, atol=1e-6)
    with pytest.raises(ValueError, match="Unknown points array 'normals'"):
        scene.get_point_array("cloud", "normals")

    with pytest.raises(ValueError, match="is not a points node"):
        scene.get_points("cloud/positions", flatten=True)


def test_flatten_round_trips_the_add_points_authoring_path(tmp_path: Path) -> None:
    scene_path = tmp_path / "authored.luxar.zarr"
    positions, colors, radii, sharpness = _expected_points()
    with LuxarZarrCompiler(scene_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "cloud",
            positions,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            partition={"max_elements": 2},
            substitutive_lod={"levels": 1, "device": "cpu", "seed": 0},
            additive_lod={"n_lods": 2, "method": "random", "seed": 0},
        )

    data = LuxarScene.load(scene_path).get_points("cloud", flatten=True)
    assert data.colors is not None
    assert data.radii is not None
    assert data.sharpness is not None
    order = np.argsort(data.positions[:, 0])
    np.testing.assert_allclose(data.positions[order], positions, atol=1e-6)
    np.testing.assert_allclose(data.colors[order], colors, atol=2.0 / 255)
    np.testing.assert_allclose(data.radii[order], radii, atol=0.03)
    np.testing.assert_allclose(data.sharpness[order], sharpness, atol=2.0 / 255)
