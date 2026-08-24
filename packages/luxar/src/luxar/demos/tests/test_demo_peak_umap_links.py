"""Regression coverage for optional peak-UMAP link metadata."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import zarr

from luxar.demos.demo_human_multiome_peak_umap import create_human_scene
from luxar.demos.demo_mouse_multiome_peak_umap import create_mouse_scene
from luxar.demos.demo_zebrahub_multiome_peak_umap import create_zebrahub_scene


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

    node = zarr.open_group(str(output_path), mode="r")["Cells"]
    assert "link" not in node.attrs
    assert "copy" not in node.attrs
    assert "has_keys" not in node.attrs
    assert "key_offsets" not in node
