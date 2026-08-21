"""Empty additive rungs are removed after count-changing rewrites."""

from __future__ import annotations

import numpy as np

from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel


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
