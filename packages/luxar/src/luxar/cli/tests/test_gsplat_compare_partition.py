"""Partition-store coverage for ``luxar gsplat compare``."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.gsplat_ops.inspect_commands import _load_gsplats_for_comparison
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import write_gsplats_tree


def test_compare_partition_uses_all_default_rendered_leaves(tmp_path: Path) -> None:
    """The default tiled output shape is scored without a flatten pre-pass."""
    centers = np.array(
        [
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0],
            [9.0, 10.0, 11.0],
            [11.0, 12.0, 13.0],
        ],
        dtype=np.float32,
    )
    data = GSplatData(
        centers=centers,
        amplitudes=np.linspace(0.2, 1.0, len(centers), dtype=np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (len(centers), 1),
        ),
    )
    partition_path = tmp_path / "partition.gsplats.zarr"
    write_gsplats_tree(
        partition_path,
        data.to_spatial_partition(max_elements=2),
        ordering="none",
        pipeline_info={"image_min": 0.0},
    )
    reference_path = tmp_path / "reference.npy"
    np.save(
        reference_path, np.random.default_rng(0).random((16, 16, 16), dtype=np.float32)
    )

    loaded = _load_gsplats_for_comparison(partition_path)
    assert loaded.n_splats == data.n_splats
    assert loaded.stats["image_min"] == 0.0

    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "compare",
            str(partition_path),
            str(reference_path),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, f"compare failed: {result.stdout}"
    assert "Loaded 5 splats (3D)" in result.stdout
