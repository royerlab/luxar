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


def _merge_result(
    tmp_path: Path, stats: list[dict[str, Any]], *extra: str
) -> tuple[dict[str, Any], str]:
    inputs = []
    for index, fitting in enumerate(stats):
        path = tmp_path / f"result-input-{index}.gsplats.zarr"
        _write_fit(path, seed=index, stats=fitting)
        inputs.append(path)

    output = tmp_path / "result-merged.gsplats.zarr"
    result = CliRunner().invoke(
        app,
        ["gsplat", "merge", *map(str, inputs), "-o", str(output), *extra],
    )
    assert result.exit_code == 0, result.output
    return GSplatData.load(output, include_stats=True).stats, result.output


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
                "source_stored_bytes": source_bytes // 2,
                "source_voxels": source_bytes // 2,
                "source_declared": True,
                "time_seconds": time_seconds,
            },
        }
        for coordinate, time_seconds in ((0.0, 1.25), (1.0, 2.75))
    ]


@pytest.mark.parametrize(
    "extra",
    [
        (),
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
            "part_count": 4,
            "fitting": {
                "source_shape": [2, 2, 4, 8, 8, 8],
                "source_dtype": "uint16",
                "source_bytes": 16_384,
                "source_stored_bytes": 8192,
                "source_voxels": 8192,
                "source_declared": True,
                "time_seconds": 8.0,
            },
        }
    ]


def test_merge_as_dimension_keeps_addressable_input_provenance(tmp_path: Path) -> None:
    stats = _merge(
        tmp_path,
        [
            {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_bytes": 1024,
                "source_voxels": 512,
                "psnr_db": 31.0,
                "time_seconds": 1.5,
            },
            {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_bytes": 1024,
                "source_voxels": 512,
                "psnr_db": 32.0,
                "time_seconds": 2.5,
            },
        ],
        "--as-dimension",
        "--values",
        "10,20",
    )

    provenance = stats["part_provenance"]
    assert [record["coordinate"] for record in provenance] == [10.0, 20.0]
    assert [record["fitting"]["psnr_db"] for record in provenance] == [31.0, 32.0]
    assert [record["fitting"]["time_seconds"] for record in provenance] == [1.5, 2.5]
    for record in provenance:
        assert "source_shape" not in record["fitting"]
        assert "source_dtype" not in record["fitting"]
        assert "source_bytes" not in record["fitting"]
        assert "source_voxels" not in record["fitting"]
    assert stats["source_shape"] == [2, 8, 8, 8]
    assert stats["source_dtype"] == "uint16"
    assert stats["source_bytes"] == 2048
    assert stats["source_voxels"] == 1024
    assert stats["time_seconds"] == 4.0


def test_merge_preserves_agreed_normalization_and_total_time(tmp_path: Path) -> None:
    normalization = {
        "floor": 10.0,
        "image_min": 0.0,
        "image_max": 100.0,
        "intensity_range": 100.0,
    }
    stats = _merge(
        tmp_path,
        [
            {**normalization, "time_seconds": 1.5},
            {**normalization, "time_seconds": 2.5},
        ],
    )

    assert {key: stats[key] for key in normalization} == normalization
    assert stats["time_seconds"] == 4.0


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


def test_merge_summary_reports_carried_and_dropped_fields(tmp_path: Path) -> None:
    _, output = _merge_result(
        tmp_path,
        [
            {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_bytes": 1024,
                "source_voxels": 512,
            },
            {
                "source_shape": [9, 8, 8],
                "source_dtype": "float32",
                "source_voxels": 576,
            },
        ],
    )

    assert "Collapsed fit provenance for 2 parts" in output
    assert "carried: source_voxels" in output
    assert (
        "dropped as disputed/incomplete: source_bytes, source_dtype, source_shape"
        in output
    )


def test_provenance_free_merge_does_not_invent_a_summary(tmp_path: Path) -> None:
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

    stats = GSplatData.load(output, include_stats=True).stats
    assert "part_provenance" not in stats
    assert "part_provenance" not in _info(output)
