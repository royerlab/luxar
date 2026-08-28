"""Serialization tests for the Cosmicflows galaxy destination."""

from __future__ import annotations

import numpy as np
import zarr

from luxar.demos.demo_cosmicflows_laniakea import (
    PRESETS,
    BasinLineData,
    GalaxyData,
    write_laniakea_scene,
)


def _decode_strings(node, channel: str) -> list[str]:
    offsets = np.asarray(node[f"{channel}_offsets"][:]).astype(int)
    data = bytes(np.asarray(node[f"{channel}_bytes"][:]).tobytes())
    return [
        data[offsets[i] : offsets[i + 1]].decode("utf-8")
        for i in range(len(offsets) - 1)
    ]


def test_galaxy_keys_stay_aligned_with_basin_labels(tmp_path) -> None:
    galaxies = GalaxyData(
        positions=np.array([[0, 0, 0], [10, 20, 30]], dtype=np.float32),
        basin_ids=np.array([1, 7], dtype=np.int16),
        radii=np.ones(2, dtype=np.float32),
        colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        pgc=np.array([4, 12345], dtype=np.int64),
    )
    output = tmp_path / "cosmicflows.luxar.zarr"

    write_laniakea_scene(output, galaxies, [], "preview", PRESETS["preview"])

    root = zarr.open_group(str(output), mode="r")
    galaxy_nodes = [
        root[name]
        for name in root.keys()
        if hasattr(root[name], "attrs")
        and dict(root[name].attrs).get("link")
        == "https://ned.ipac.caltech.edu/byname?objname={hover_key}"
    ]
    assert len(galaxy_nodes) == 1
    node = galaxy_nodes[0]
    attrs = dict(node.attrs)
    assert attrs["has_keys"] is True
    assert _decode_strings(node, "key") == ["PGC4", "PGC12345"]
    assert _decode_strings(node, "label") == ["Basin 1", "Basin 7"]


def test_basin_streamlines_have_substitutive_lod(tmp_path) -> None:
    galaxies = GalaxyData(
        positions=np.array([[0, 0, 0]], dtype=np.float32),
        basin_ids=np.array([1], dtype=np.int16),
        radii=np.ones(1, dtype=np.float32),
        colors=np.ones((1, 3), dtype=np.float32),
        pgc=np.array([4], dtype=np.int64),
    )
    vertices = np.column_stack(
        [
            np.arange(65, dtype=np.float32),
            np.zeros(65, dtype=np.float32),
            np.zeros(65, dtype=np.float32),
        ]
    )
    segments = np.column_stack(
        [np.arange(64, dtype=np.uint32), np.arange(1, 65, dtype=np.uint32)]
    )
    basin = BasinLineData(
        basin_id=1,
        vertices=vertices,
        segments=segments,
        streamline_count=1,
    )
    output = tmp_path / "cosmicflows_lod.luxar.zarr"

    write_laniakea_scene(output, galaxies, [basin], "preview", PRESETS["preview"])

    root = zarr.open_group(str(output), mode="r")
    ladder = root["Basin 1 streamlines"]
    assert ladder.attrs["kind"] == "lod"
    assert ladder.attrs["display_type"] == "lines"
    assert ladder["child_0"].attrs["type"] == "gsplats"
    finest = ladder[f"child_{len(list(ladder.group_keys())) - 1}"]
    assert finest.attrs["type"] == "lines"
    assert finest.attrs["n_segments"] == 64
