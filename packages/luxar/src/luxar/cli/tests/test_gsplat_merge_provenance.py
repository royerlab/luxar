"""Merge collapses input fit provenance into one honest summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.gsplats.gsplat_data import GSplatData


def _write_fit(path: Path, *, seed: int, stats: dict[str, Any]) -> None:
    rng = np.random.default_rng(seed)
    n_splats = 3
    cholesky = np.zeros((n_splats, 6), dtype=np.float32)
    cholesky[:, [0, 2, 5]] = 0.5
    GSplatData(
        centers=rng.uniform(0, 8, size=(n_splats, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.25, 1, size=n_splats).astype(np.float32),
        cholesky_factors=cholesky,
        stats=stats,
    ).save(path)


def _merge(tmp_path: Path, stats: list[dict[str, Any]], *extra: str) -> dict[str, Any]:
    inputs = []
    for index, fitting in enumerate(stats):
        path = tmp_path / f"input-{index}.gsplats.zarr"
        _write_fit(path, seed=index, stats=fitting)
        inputs.append(path)

    output = tmp_path / "merged.gsplats.zarr"
    result = CliRunner().invoke(
        app,
        ["gsplat", "merge", *map(str, inputs), "-o", str(output), *extra],
    )
    assert result.exit_code == 0, result.output
    return GSplatData.load(output, include_stats=True).stats


def _info(path: Path) -> str:
    result = CliRunner().invoke(app, ["gsplat", "info", str(path), "--no-histograms"])
    assert result.exit_code == 0, result.output
    return result.output


def _nested_provenance(
    *, source_shape: list[int], source_dtype: str, source_bytes: int
) -> list[dict[str, Any]]:
    return [
        {
            "coordinate": coordinate,
            "fit_reference": {"kind": "acquisition"},
            "fitting": {
                "source_shape": source_shape,
                "source_dtype": source_dtype,
                "source_bytes": source_bytes,
                "source_voxels": source_bytes // 2,
                "time_seconds": time_seconds,
            },
        }
        for coordinate, time_seconds in ((0.0, 1.25), (1.0, 2.75))
    ]


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("--as-dimension",),
        ("--channel-colors", "#ff0000,#00ff00"),
    ],
)
def test_merge_replaces_input_provenance_with_one_summary(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    provenance = _nested_provenance(
        source_shape=[4, 8, 8, 8], source_dtype="uint16", source_bytes=4096
    )

    stats = _merge(
        tmp_path,
        [
            {"part_provenance": provenance},
            {"part_provenance": provenance},
        ],
        *extra,
    )

    assert stats["part_provenance"] == [
        {
            "part_count": 2,
            "fitting": {
                "source_shape": [4, 8, 8, 8],
                "source_dtype": "uint16",
                "source_bytes": 16_384,
                "source_voxels": 8192,
                "time_seconds": 8.0,
            },
        }
    ]
    assert "time_seconds" not in stats


def test_merge_summary_omits_partial_and_disputed_fields(tmp_path: Path) -> None:
    stats = _merge(
        tmp_path,
        [
            {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_bytes": 1024,
                "source_voxels": 512,
                "time_seconds": 1.5,
            },
            {
                "source_shape": [9, 8, 8],
                "source_dtype": "float32",
                "source_voxels": 576,
                "time_seconds": 2.5,
            },
        ],
    )

    assert stats["part_provenance"] == [
        {
            "part_count": 2,
            "fitting": {
                "source_voxels": 1088,
                "time_seconds": 4.0,
            },
        }
    ]


def test_info_reports_the_collapsed_input_count(tmp_path: Path) -> None:
    inputs = []
    for index in range(2):
        path = tmp_path / f"info-input-{index}.gsplats.zarr"
        _write_fit(path, seed=index, stats={})
        inputs.append(path)
    output = tmp_path / "info-merged.gsplats.zarr"
    result = CliRunner().invoke(
        app,
        ["gsplat", "merge", *map(str, inputs), "-o", str(output)],
    )
    assert result.exit_code == 0, result.output

    assert "part_provenance: 2 parts" in _info(output)
