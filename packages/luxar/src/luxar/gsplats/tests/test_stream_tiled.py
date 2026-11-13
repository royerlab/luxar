"""Store-writing uniform tiled merge invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from luxar._zarr_compat import open_group
from luxar.gsplats.fit_tiled_gsplats import merge_tile_results
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
from luxar.gsplats.stream_tiled import stream_tile_results
from luxar.gsplats.tiling import compute_tile_specs


def _tile(index: int) -> GSplatData:
    return GSplatData(
        centers=np.array([[float(index * 3 + 1), 2.0]], dtype=np.float32),
        amplitudes=np.array([0.5 + index], dtype=np.float32),
        cholesky_factors=np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
        stats={
            "image_min": 0.0,
            "image_max": 2.0,
            "intensity_range": 2.0,
            "norm_range": [0.0, 2.0],
            "voxel_size": [1.0, 1.0],
            "floor": 0.0,
            "tile_amplitude_pruned": index,
        },
    )


def test_streamed_tile_parts_and_root_match_in_memory_merge(tmp_path: Path) -> None:
    shape = (4, 10)
    specs = compute_tile_specs(shape, 4, 1)
    assert len(specs) == 3
    tiles = [_tile(i) for i in range(len(specs))]
    memory = merge_tile_results(
        tiles,
        volume_shape=shape,
        tile_size=4,
        overlap=1,
        num_tiles=len(specs),
        progressive=False,
        cull_retention=None,
        elapsed=0.0,
        verbose=False,
        partition=True,
    )
    memory_path = tmp_path / "memory.gsplats.zarr"
    fitting, config, provenance, pipeline = split_fitting_info(memory.meta["fit_stats"])
    write_gsplats_tree(
        memory_path,
        memory,
        fitting_info=fitting,
        fitting_config=config,
        provenance_info=provenance,
        pipeline_info=pipeline,
    )
    streamed_path = tmp_path / "streamed.gsplats.zarr"
    count = stream_tile_results(
        streamed_path,
        lambda i: tiles[i],
        num_tiles=len(specs),
        volume_shape=shape,
        tile_size=4,
        overlap=1,
        progressive=False,
        cull_retention=None,
        elapsed=0.0,
        source_dtype="float32",
        applied_floor=0.0,
        tile_amplitude_floor=1e-5,
        verbose=False,
    )
    assert count == 3
    memory_root = open_group(memory_path, mode="r")
    stream_root = open_group(streamed_path, mode="r")
    assert stream_root.attrs["kind"] == memory_root.attrs["kind"] == "partition"
    assert stream_root.attrs["bsp_tree"] == memory_root.attrs["bsp_tree"]
    for i in range(3):
        a = memory_root[f"part_{i}"]
        b = stream_root[f"part_{i}"]
        for name in ("centers", "amplitudes"):
            np.testing.assert_array_equal(b[name][:], a[name][:])
    fitting_stats = stream_root["fitting"].attrs
    assert fitting_stats["n_splats"] == 3
    assert fitting_stats["source_dtype"] == "float32"
    assert fitting_stats["voxel_size"] == [1.0, 1.0]
    assert stream_root["pipeline"].attrs["tile_amplitude_pruned"] == 3
