"""Partition-store coverage for ``luxar gsplat compare``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition


def _data(n_splats: int, *, seed: int, image_min: float = 0.0) -> GSplatData:
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(3.0, 13.0, (n_splats, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n_splats).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (n_splats, 1),
        ),
        stats={"image_min": image_min},
    )


def _compare(store: Path, reference: Path, output_json: Path | None = None):
    args = [
        "gsplat",
        "compare",
        str(store),
        str(reference),
        "--device",
        "cpu",
    ]
    if output_json is not None:
        args.extend(["--output-json", str(output_json)])
    return CliRunner().invoke(app, args)


def test_partition_metrics_match_the_same_flat_splats(tmp_path: Path) -> None:
    """Tree materialization must preserve the scientific comparison result."""
    data = _data(40, seed=0)
    flat_path = tmp_path / "flat.gsplats.zarr"
    partition_path = tmp_path / "partition.gsplats.zarr"
    data.save(flat_path, include_fitting_info=True)
    write_gsplats_tree(
        partition_path,
        data.to_spatial_partition(max_elements=10),
        ordering="none",
        pipeline_info=data.stats,
    )

    reference_path = tmp_path / "reference.npy"
    np.save(
        reference_path,
        np.random.default_rng(1).random((16, 16, 16), dtype=np.float32),
    )
    flat_json = tmp_path / "flat.json"
    partition_json = tmp_path / "partition.json"

    flat_result = _compare(flat_path, reference_path, flat_json)
    partition_result = _compare(partition_path, reference_path, partition_json)
    assert flat_result.exit_code == 0, flat_result.output
    assert partition_result.exit_code == 0, partition_result.output

    flat_payload = json.loads(flat_json.read_text())
    partition_payload = json.loads(partition_json.read_text())
    metadata_keys = {
        "gsplats",
        "reference",
        "n_splats",
        "ndim",
        "shape",
        "compression_ratio",
    }
    flat_metrics = {
        key: value for key, value in flat_payload.items() if key not in metadata_keys
    }
    partition_metrics = {
        key: value
        for key, value in partition_payload.items()
        if key not in metadata_keys
    }
    assert partition_metrics == pytest.approx(flat_metrics, rel=1e-3)


def test_nested_lod_selection_and_whole_store_ratio_are_reported(
    tmp_path: Path,
) -> None:
    """Users can see that compare scores finest leaves but sizes the full store."""
    groups = [
        GSplatLodGroup(
            children=[_data(6, seed=seed).tree, _data(12, seed=seed + 1).tree]
        )
        for seed in (10, 20)
    ]
    store = tmp_path / "partition_lods.gsplats.zarr"
    write_gsplats_tree(
        store,
        GSplatPartition(children=groups, max_elements=12),
        ordering="none",
        pipeline_info={"image_min": 0.0},
    )
    reference = tmp_path / "reference.npy"
    np.save(reference, np.random.default_rng(2).random((16, 16, 16), dtype=np.float32))

    result = _compare(store, reference)

    assert result.exit_code == 0, result.output
    assert "Loaded 24 splats (3D)" in result.output
    assert "Materialized 2 default-rendered leaf/leaves" in result.output
    assert "Skipped 12 splats in coarse LOD levels" in result.output
    assert "compression ratio covers the whole store" in result.output
