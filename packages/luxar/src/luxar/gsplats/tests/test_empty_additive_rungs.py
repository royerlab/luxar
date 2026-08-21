"""Empty additive rungs are removed after count-changing rewrites."""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
from luxar.validation import ValidationError


def _rung(identifier: float, amplitude: float, level: int) -> AdditiveSubLOD:
    centers = np.array([[identifier, 0.0, 0.0]], dtype=np.float32)
    cholesky = np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return AdditiveSubLOD(
        centers=centers,
        amplitudes=np.array([amplitude], dtype=np.float32),
        cholesky_factors=cholesky,
        stats={
            "lod_method": "greedy",
            "lod_level": level,
            "lod_breakpoints_kind": "equal-count",
            "lod_n_splats": 1,
            "lod_cumulative_n": level + 1,
        },
    )


def test_cull_prunes_empty_middle_rung_and_rederives_structure(tmp_path) -> None:
    summary = {
        "lod_method": "greedy",
        "lod_n_lods": 3,
        "lod_breakpoints_kind": "equal-count",
        "lod_cutpoints": [1, 2, 3],
    }
    source = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[
                    _rung(0.0, 10.0, 0),
                    _rung(1.0, 0.1, 1),
                    _rung(2.0, 9.0, 2),
                ],
                stats=dict(summary),
            )
        ],
        stats=dict(summary),
    )

    result = source.cull(method="cumulative", retention=0.99)

    assert [lod.n_splats for lod in result.additive_sublods] == [1, 1]
    assert [float(lod.centers[0, 0]) for lod in result.additive_sublods] == [0.0, 2.0]
    assert [lod.stats["lod_level"] for lod in result.additive_sublods] == [0, 1]
    assert [lod.stats["lod_n_splats"] for lod in result.additive_sublods] == [1, 1]
    assert [lod.stats["lod_cumulative_n"] for lod in result.additive_sublods] == [1, 2]
    assert result.substitutive_levels[0].stats["lod_n_lods"] == 2
    assert result.substitutive_levels[0].stats["lod_cutpoints"] == [1, 2]
    assert result.stats["lod_n_lods"] == 2
    assert result.stats["lod_cutpoints"] == [1, 2]

    output = tmp_path / "culled.gsplats.zarr"
    result.save(output)
    loaded = GSplatData.load(output)
    assert [lod.n_splats for lod in loaded.additive_sublods] == [1, 1]
    assert loaded.substitutive_levels[0].stats["lod_n_lods"] == 2
    assert loaded.substitutive_levels[0].stats["lod_cutpoints"] == [1, 2]


def test_pruning_drops_stale_quality_stamps_without_inventing_rung_keys() -> None:
    first = _rung(0.0, 10.0, 0)
    first = AdditiveSubLOD(
        centers=first.centers,
        amplitudes=first.amplitudes,
        cholesky_factors=first.cholesky_factors,
        stats={"energy_fraction_cum": 0.5, "label": "keep"},
    )
    source = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[first, _rung(1.0, 0.1, 1)],
                compression_factor=4,
                parent_method="greedy",
                level_index=2,
                stats={
                    "lod_n_lods": 2,
                    "lod_cutpoints": [1, 2],
                    "reference_energy": 10.0,
                    "quality": 30.0,
                    "n_splats_total": 2,
                    "label": "keep",
                },
            )
        ],
        stats={"reference_energy": 10.0, "label": "keep"},
    )

    result = source.cull(method="cumulative", retention=0.99)

    assert result.n_additive_sublods == 1
    assert result.substitutive_levels[0].compression_factor == 4
    assert result.substitutive_levels[0].parent_method == "greedy"
    assert result.substitutive_levels[0].level_index == 2
    assert result.additive_sublods[0].stats == {"label": "keep"}
    assert result.substitutive_levels[0].stats == {
        "lod_n_lods": 1,
        "lod_cutpoints": [1],
        "label": "keep",
    }
    assert result.stats["label"] == "keep"
    assert "reference_energy" not in result.stats


def test_multi_level_pruning_refreshes_root_summary_from_finest_level() -> None:
    summary = {"lod_n_lods": 2, "lod_cutpoints": [1, 2]}
    source = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[_rung(0.0, 10.0, 0), _rung(1.0, 0.1, 1)],
                stats=dict(summary),
            ),
            SubstitutiveLevel(
                additive_sublods=[_rung(2.0, 10.0, 0), _rung(3.0, 0.1, 1)],
                compression_factor=4,
                level_index=1,
                stats=dict(summary),
            ),
        ],
        stats=dict(summary),
    )

    result = source.cull(method="cumulative", retention=0.99)

    assert [level.n_additive_lods for level in result.substitutive_levels] == [1, 1]
    assert result.stats["lod_n_lods"] == 1
    assert result.stats["lod_cutpoints"] == [1]


def test_all_empty_ladder_keeps_writer_validation_path(tmp_path) -> None:
    source = GSplatData.from_additive_sublods([_rung(0.0, 1.0, 0), _rung(1.0, 1.0, 1)])

    result = source.filter(np.zeros(source.n_splats, dtype=bool))

    assert [lod.n_splats for lod in result.additive_sublods] == [0, 0]
    with pytest.raises(ValidationError, match="positions: Cannot write empty points"):
        result.save(tmp_path / "empty.gsplats.zarr")
