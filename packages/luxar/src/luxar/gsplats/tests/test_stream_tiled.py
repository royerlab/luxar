"""Store-writing uniform tiled merge invariants."""

from __future__ import annotations

import gc
import tracemalloc
from pathlib import Path

import numpy as np

from luxar._zarr_compat import open_group
from luxar.gsplats.batch.stream_tiled import stream_tile_results
from luxar.gsplats.fit_tiled_gsplats import (
    fit_tiled,
    fit_tiled_to_store,
    merge_tile_results,
    stamp_tile_amplitude_floor,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import split_fitting_info, write_gsplats_tree
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
    volume = np.ones(shape, dtype=np.float32)
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
        source_dtype="float32",
        source_itemsize=4,
        applied_floor=0.0,
        volume=volume,
        device="cpu",
    )
    stamp_tile_amplitude_floor(memory, 1e-5, 3)
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
        source_itemsize=4,
        applied_floor=0.0,
        tile_amplitude_floor=1e-5,
        volume=volume,
        device="cpu",
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
    assert fitting_stats["source_shape"] == memory_root["fitting"].attrs["source_shape"]
    assert fitting_stats["voxel_size"] == [1.0, 1.0]
    assert stream_root["pipeline"].attrs["tile_amplitude_pruned"] == 3
    assert (
        stream_root["pipeline"].attrs["floor"] == memory_root["pipeline"].attrs["floor"]
    )
    for key in ("image_min", "image_max", "intensity_range"):
        assert stream_root["pipeline"].attrs[key] == memory_root["pipeline"].attrs[key]
    for key in ("mse", "ssim", "foreground_fraction"):
        assert np.isclose(
            fitting_stats[key], memory_root["fitting"].attrs[key], rtol=0, atol=1e-6
        )


def test_streamed_merge_peak_does_not_follow_tile_count(tmp_path: Path) -> None:
    def peak_for(n_tiles: int) -> int:
        def load_tile(index: int) -> GSplatData:
            return GSplatData(
                centers=np.full((20_000, 2), (index * 4 + 1), dtype=np.float32),
                amplitudes=np.ones(20_000, dtype=np.float32),
                cholesky_factors=np.tile(
                    np.array([[1.0, 0.0, 1.0]], dtype=np.float32), (20_000, 1)
                ),
            )

        gc.collect()
        tracemalloc.start()
        try:
            stream_tile_results(
                tmp_path / f"{n_tiles}.gsplats.zarr",
                load_tile,
                num_tiles=n_tiles,
                volume_shape=(4, n_tiles * 4),
                tile_size=4,
                overlap=0,
                progressive=False,
                cull_retention=None,
                elapsed=0.0,
                verbose=False,
            )
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    few = peak_for(4)
    many = peak_for(24)
    assert many < few + 8_000_000, (few, many)


def test_empty_slot_keeps_floor_count_and_prunes_split_tree(tmp_path: Path) -> None:
    empty = GSplatData(
        centers=np.zeros((0, 2), dtype=np.float32),
        amplitudes=np.zeros(0, dtype=np.float32),
        cholesky_factors=np.zeros((0, 3), dtype=np.float32),
        stats={"tile_amplitude_pruned": 7},
    )
    tiles = [_tile(0), _tile(1), empty, _tile(3)]
    path = tmp_path / "empty.gsplats.zarr"

    count = stream_tile_results(
        path,
        lambda i: tiles[i],
        num_tiles=4,
        volume_shape=(4, 16),
        tile_size=4,
        overlap=0,
        progressive=False,
        cull_retention=None,
        elapsed=0.0,
        tile_amplitude_floor=1e-5,
        verbose=False,
    )

    root = open_group(path, mode="r")
    assert count == root["fitting"].attrs["n_splats"] == 3
    assert root["pipeline"].attrs["tile_amplitude_pruned"] == 11
    assert [r["slot_id"] for r in root["fitting"].attrs["part_provenance"]] == [
        "tile_0",
        "tile_1",
        "tile_3",
    ]
    assert "part_3" not in root


def test_single_tile_keeps_bare_leaf_contract(tmp_path: Path) -> None:
    path = tmp_path / "one.gsplats.zarr"
    count = stream_tile_results(
        path,
        lambda _: _tile(0),
        num_tiles=1,
        volume_shape=(4, 4),
        tile_size=4,
        overlap=0,
        progressive=False,
        cull_retention=None,
        elapsed=0.0,
        verbose=False,
    )
    root = open_group(path, mode="r")
    assert count == 1
    assert root.attrs.get("kind") != "partition"
    assert root["fitting"].attrs["n_splats"] == 1


def test_sequential_store_matches_fitted_tree_without_staging_loss(
    tmp_path: Path,
) -> None:
    y, x = np.mgrid[:12, :20]
    volume = np.exp(-((y - 6) ** 2 / 16 + (x - 10) ** 2 / 49)).astype(np.float32)
    kwargs = dict(
        tile_size=12,
        overlap=4,
        seeds=8,
        n_iters=2,
        device="cpu",
        floor="none",
        cull_retention=None,
        verbose=False,
    )
    memory = fit_tiled(volume, partition=True, **kwargs)
    memory_path = tmp_path / "memory-fit.gsplats.zarr"
    fitting, config, provenance, pipeline = split_fitting_info(memory.meta["fit_stats"])
    write_gsplats_tree(
        memory_path,
        memory,
        fitting_info=fitting,
        fitting_config=config,
        provenance_info=provenance,
        pipeline_info=pipeline,
    )
    stored_path = tmp_path / "stored-fit.gsplats.zarr"
    fit_tiled_to_store(volume, stored_path, **kwargs)
    a = open_group(memory_path, mode="r")
    b = open_group(stored_path, mode="r")

    for key in ("n_splats", "mse", "psnr_db", "ssim", "foreground_psnr_db"):
        assert a["fitting"].attrs[key] == b["fitting"].attrs[key]
    for i in range(2):
        for name in ("centers", "amplitudes"):
            np.testing.assert_array_equal(
                a[f"part_{i}/{name}"][:], b[f"part_{i}/{name}"][:]
            )
