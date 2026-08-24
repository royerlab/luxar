"""Regression coverage for optional peak-UMAP link metadata."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import zarr

from luxar.demos.demo_human_multiome_peak_umap import create_human_scene
from luxar.demos.demo_mouse_multiome_peak_umap import create_mouse_scene
from luxar.demos.demo_zebrahub_multiome_peak_umap import create_zebrahub_scene


def _geometry_groups(node: zarr.Group) -> list[zarr.Group]:
    groups = [node]
    for name in sorted(node.group_keys()):
        groups.extend(_geometry_groups(node[name]))
    return groups


def _decode_keys(node: zarr.Group) -> list[str]:
    offsets = np.asarray(node["key_offsets"][:]).astype(int)
    data = bytes(np.asarray(node["key_bytes"][:]).tobytes())
    return [
        data[offsets[i] : offsets[i + 1]].decode("utf-8")
        for i in range(len(offsets) - 1)
    ]


@pytest.mark.parametrize(
    "builder",
    [create_human_scene, create_mouse_scene, create_zebrahub_scene],
)
@pytest.mark.parametrize("category_maps", [None, {"celltype": []}])
def test_scene_omits_links_without_celltype_categories(
    tmp_path: Path,
    builder: Callable[[Path, np.ndarray, dict, dict | None], int],
    category_maps: dict | None,
) -> None:
    coordinates = np.column_stack((np.arange(20), np.zeros(20), np.zeros(20))).astype(
        np.float32
    )
    attributes = {"celltype": np.zeros(20, dtype=np.int32)}
    output_path = tmp_path / f"{builder.__name__}.luxar.zarr"

    assert builder(output_path, coordinates, attributes, category_maps) == 20

    root = zarr.open_group(str(output_path), mode="r")["Cells"]
    for node in _geometry_groups(root):
        assert "link" not in node.attrs
        assert "copy" not in node.attrs
        assert "has_keys" not in node.attrs
        assert "key_offsets" not in node


@pytest.mark.parametrize(
    "builder",
    [create_human_scene, create_mouse_scene, create_zebrahub_scene],
)
def test_scene_serializes_valid_celltype_links_and_suppresses_missing_codes(
    tmp_path: Path,
    builder: Callable[[Path, np.ndarray, dict, dict | None], int],
) -> None:
    codes = np.array([0, 1, 2, -1, 7, 1], dtype=np.int32)
    coordinates = np.column_stack(
        (np.arange(len(codes)), np.zeros(len(codes)), np.zeros(len(codes)))
    ).astype(np.float32)
    attributes = {"celltype": codes}
    category_maps = {"celltype": ["Tcell", "Bcell", "Monocyte"]}
    output_path = tmp_path / f"{builder.__name__}.luxar.zarr"

    assert builder(output_path, coordinates, attributes, category_maps) == len(codes)

    root = zarr.open_group(str(output_path), mode="r")["Cells"]
    node = next(
        node for node in _geometry_groups(root) if dict(node.attrs).get("has_keys")
    )
    assert node.attrs["link"] == "https://www.ebi.ac.uk/ols4/search?q={hover_key}"
    assert node.attrs["copy"] == "{hover_key}"
    assert node.attrs["has_keys"] is True
    assert _decode_keys(node) == ["Tcell", "Bcell", "Monocyte", "", "", "Bcell"]
