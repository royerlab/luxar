"""Serialization tests for the Cosmicflows galaxy destination."""

from __future__ import annotations

import numpy as np
import zarr

from luxar.demos.demo_cosmicflows_laniakea import (
    PRESETS,
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
