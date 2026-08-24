"""Interaction metadata for the Cosmicflows/Laniakea demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from luxar.demos.demo_cosmicflows_laniakea import (
    PRESETS,
    GalaxyData,
    write_laniakea_scene,
)


def _decode_strings(node: zarr.Group, prefix: str) -> list[str]:
    offsets = np.asarray(node[f"{prefix}_offsets"][:], dtype=int)
    data = bytes(np.asarray(node[f"{prefix}_bytes"][:]).tobytes())
    return [
        data[offsets[i] : offsets[i + 1]].decode("utf-8")
        for i in range(len(offsets) - 1)
    ]


def test_galaxy_links_serialize_row_aligned_pgc_keys(tmp_path: Path) -> None:
    galaxies = GalaxyData(
        positions=np.array([[10, 20, 30], [-10, -20, -30]], dtype=np.float32),
        basin_ids=np.array([1, 7], dtype=np.int32),
        radii=np.ones(2, dtype=np.float32),
        colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        pgc=np.array([123, 987], dtype=np.int64),
    )
    output = tmp_path / "cosmicflows.luxar.zarr"

    write_laniakea_scene(output, galaxies, [], "preview", PRESETS["preview"])

    root = zarr.open_group(str(output), mode="r")
    node = root["CF4 galaxies (55,486 in plus-minus 500 Mpc cube)"]
    assert node.attrs["link"] == (
        "https://ned.ipac.caltech.edu/byname?objname={hover_key}"
    )
    assert node.attrs["has_keys"] is True
    labels = _decode_strings(node, "label")
    keys = _decode_strings(node, "key")
    assert set(zip(labels, keys, strict=True)) == {
        ("Basin 1", "PGC123"),
        ("Basin 7", "PGC987"),
    }
