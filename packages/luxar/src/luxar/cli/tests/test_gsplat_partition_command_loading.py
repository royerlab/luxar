"""Partition-loading contracts for flat and read-only gsplat commands."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    n_splats = 6
    cholesky = np.zeros((n_splats, 6), dtype=np.float32)
    cholesky[:, [0, 2, 5]] = 0.5
    data = GSplatData(
        centers=np.array(
            [
                [2, 2, 2],
                [3, 2, 2],
                [4, 2, 2],
                [8, 8, 8],
                [9, 8, 8],
                [10, 8, 8],
            ],
            dtype=np.float32,
        ),
        amplitudes=np.linspace(0.2, 1.0, n_splats, dtype=np.float32),
        cholesky_factors=cholesky,
    )

    partition_path = tmp_path / "partition.gsplats.zarr"
    flat_path = tmp_path / "flat.gsplats.zarr"
    target_path = tmp_path / "target.npy"
    write_gsplats_tree(
        partition_path,
        data.to_spatial_partition(max_elements=n_splats // 2),
    )
    data.save(flat_path)
    np.save(target_path, np.zeros((12, 12, 12), dtype=np.float32))
    return partition_path, flat_path, target_path


def test_render_partition_matches_sum_of_default_leaves(tmp_path: Path) -> None:
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.tree import iter_default_leaves

    partition_path, _, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "render.npy"

    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "render",
            str(partition_path),
            str(output_path),
            "--shape",
            "12,12,12",
            "--device",
            "cpu",
            "--truncate",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    actual = np.load(output_path)
    node, _ = load_gsplat_node(partition_path)
    expected = sum(
        (
            GSplatData.from_tree(leaf).render_to_volume(
                shape=(12, 12, 12),
                device="cpu",
                truncate=3,
            )
            for leaf in iter_default_leaves(node)
        ),
        np.zeros((12, 12, 12), dtype=np.float32),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert "Loaded 6 splats (3D)" in result.output


@pytest.mark.parametrize(
    ("case", "command_name"),
    [
        ("cull", "cull"),
        ("cull-target", "cull"),
        ("filter", "filter"),
        ("slice", "slice"),
        ("decimate", "decimate"),
        ("merge", "merge"),
    ],
)
def test_flat_output_commands_reject_partition_without_traceback(
    tmp_path: Path,
    case: str,
    command_name: str,
) -> None:
    partition_path, flat_path, target_path = _write_inputs(tmp_path)
    output_path = tmp_path / f"{case}.gsplats.zarr"

    args = {
        "cull": [
            "gsplat",
            "cull",
            str(partition_path),
            str(output_path),
        ],
        "cull-target": [
            "gsplat",
            "cull",
            str(partition_path),
            str(output_path),
            "--target",
            str(target_path),
            "--device",
            "cpu",
        ],
        "filter": [
            "gsplat",
            "filter",
            str(partition_path),
            str(output_path),
            "--amplitude-min",
            "0",
        ],
        "slice": [
            "gsplat",
            "slice",
            str(partition_path),
            str(output_path),
            "0:12,0:12,0:12",
        ],
        "decimate": [
            "gsplat",
            "decimate",
            str(partition_path),
            str(output_path),
            "--target",
            "2",
            "--device",
            "cpu",
        ],
        "merge": [
            "gsplat",
            "merge",
            str(partition_path),
            str(flat_path),
            "--output",
            str(output_path),
        ],
    }[case]

    result = CliRunner().invoke(app, args)
    combined_output = result.output + (result.stderr or "")

    assert result.exit_code == 1
    assert (
        f"`luxar gsplat {command_name}` requires a matrix-shaped input"
        in combined_output
    )
    assert "`luxar gsplat flatten` first" in combined_output
    assert "not matrix-shaped" not in combined_output
    assert "Traceback" not in combined_output
    assert not output_path.exists()
