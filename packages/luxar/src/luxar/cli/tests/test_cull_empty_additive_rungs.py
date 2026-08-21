"""CLI regressions for culling additive ladders with emptied rungs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel


def _rung(seed: int, amplitudes: np.ndarray, level: int) -> AdditiveSubLOD:
    rng = np.random.default_rng(seed)
    count = amplitudes.size
    cholesky = np.zeros((count, 6), dtype=np.float32)
    cholesky[:, [0, 2, 5]] = 1.5
    return AdditiveSubLOD(
        centers=rng.uniform(0, 32, (count, 3)).astype(np.float32),
        amplitudes=amplitudes.astype(np.float32),
        cholesky_factors=cholesky,
        stats={
            "lod_method": "greedy",
            "lod_level": level,
            "lod_breakpoints_kind": "equal-count",
            "lod_n_splats": count,
            "lod_cumulative_n": (level + 1) * count,
        },
    )


def _level(seed: int, *, level_index: int) -> SubstitutiveLevel:
    summary = {
        "lod_method": "greedy",
        "lod_n_lods": 3,
        "lod_breakpoints_kind": "equal-count",
        "lod_cutpoints": [100, 200, 300],
    }
    return SubstitutiveLevel(
        additive_sublods=[
            _rung(seed, np.linspace(1.0, 0.8, 100), 0),
            _rung(seed + 1, np.linspace(0.79, 0.6, 100), 1),
            _rung(seed + 2, np.linspace(0.59, 0.2, 100), 2),
        ],
        compression_factor=4**level_index,
        level_index=level_index,
        stats=summary,
    )


@pytest.mark.parametrize("n_levels", [1, 2])
def test_cull_cli_writes_stream_and_levels_after_rung_empties(
    tmp_path: Path, n_levels: int
) -> None:
    source = tmp_path / "source.gsplats.zarr"
    output = tmp_path / "culled.gsplats.zarr"
    GSplatData.from_substitutive_levels(
        [_level(10 * index, level_index=index) for index in range(n_levels)]
    ).save(source)

    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "cull",
            str(source),
            str(output),
            "-m",
            "cumulative",
            "-r",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    loaded = GSplatData.load(output)
    assert loaded.n_substitutive == n_levels
    for level in loaded.substitutive_levels:
        counts = [lod.n_splats for lod in level.additive_sublods]
        assert counts
        assert all(count > 0 for count in counts)
        assert level.stats["lod_n_lods"] == len(counts)
        assert level.stats["lod_cutpoints"] == list(np.cumsum(counts))
        assert [lod.stats["lod_level"] for lod in level.additive_sublods] == list(
            range(len(counts))
        )
        assert [lod.stats["lod_n_splats"] for lod in level.additive_sublods] == counts
        assert [
            lod.stats["lod_cumulative_n"] for lod in level.additive_sublods
        ] == list(np.cumsum(counts))
