"""Empty additive rungs are removed after count-changing rewrites."""

from __future__ import annotations

import json

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


def _wide_rung(
    identifier: float, amplitudes: list[float], level: int
) -> AdditiveSubLOD:
    count = len(amplitudes)
    centers = np.zeros((count, 3), dtype=np.float32)
    centers[:, 0] = identifier
    cholesky = np.tile(
        np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (count, 1),
    )
    return AdditiveSubLOD(
        centers=centers,
        amplitudes=np.asarray(amplitudes, dtype=np.float32),
        cholesky_factors=cholesky,
        stats={
            "lod_level": level,
            "lod_n_splats": count,
            "lod_cumulative_n": (level + 1) * count,
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


def test_pruning_refreshes_the_batch_merge_spelling_of_the_rung_count() -> None:
    """The same ladder summary, one spelling over (#1600 review).

    ``batch-fit merge --recipe stream`` publishes it WITHOUT the ``lod_`` prefix
    (``_recipe_pipeline_info``), and culling or filtering such a store prunes
    rungs: the prefixed half self-healed through ``_refresh_ladder_summary``
    while ``n_lods: 3`` rode through a few keys away, still describing the
    ladder that is gone.

    The trio's other two must NOT move. ``method`` is the additive ORDERING,
    which dropping an empty rung does not change; ``breakpoints`` is the build
    SPEC that was requested, and this rewrite built no new ladder from another
    one (unlike ``gsplat additive``, which drops it).

    The block comes from the producer rather than a literal, so a knob added
    there cannot quietly go unhandled — minus ``per_part``, which claims a
    partition this single-ladder matrix is not.
    """
    from luxar.gsplats.batch.merge_orchestrator import _recipe_pipeline_info
    from luxar.gsplats.lod.recipes import RecipeParams

    info = _recipe_pipeline_info(
        "stream",
        RecipeParams(n_lods=3, additive_method="mass", breakpoints="equal-count"),
    )
    assert info and info["n_lods"] == 3
    summary = {
        **{key: value for key, value in info.items() if key != "per_part"},
        "lod_n_lods": 3,
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
                stats={"lod_n_lods": 3, "lod_cutpoints": [1, 2, 3]},
            )
        ],
        stats=dict(summary),
    )

    result = source.cull(method="cumulative", retention=0.99)

    assert [lod.n_splats for lod in result.additive_sublods] == [1, 1]
    assert result.stats["lod_n_lods"] == 2, "the prefixed half regressed"
    assert result.stats["n_lods"] == 2
    assert result.stats["method"] == "mass"
    assert result.stats["breakpoints"] == "equal-count"
    assert summary["n_lods"] == 3, "the caller's dict was edited"


def test_pruning_recomputes_authored_ladder_stamps() -> None:
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
    assert result.additive_sublods[0].stats == {
        "energy_fraction_cum": 1.0,
        "lod_n_splats": 1,
        "lod_cumulative_n": 1,
        "label": "keep",
    }
    level_stats = result.substitutive_levels[0].stats
    assert level_stats["lod_n_lods"] == 1
    assert level_stats["lod_cutpoints"] == [1]
    assert level_stats["n_splats_total"] == 1
    assert level_stats["reference_energy"] != 10.0
    assert np.isfinite(level_stats["reference_energy"])
    assert "quality" not in level_stats
    assert level_stats["label"] == "keep"
    assert result.stats["label"] == "keep"
    assert "reference_energy" not in result.stats


def test_multi_level_pruning_refreshes_root_summary_from_finest_level() -> None:
    summary = {
        "lod_n_lods": 2,
        "lod_cutpoints": [1, 2],
        "reference_energy": 99.0,
        "quality": 30.0,
        "n_splats_total": 2,
    }
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
    for level in result.substitutive_levels:
        assert level.stats["lod_n_lods"] == 1
        assert level.stats["lod_cutpoints"] == [1]
        assert level.stats["reference_energy"] != 99.0
        assert np.isfinite(level.stats["reference_energy"])
        assert "quality" not in level.stats
        assert level.stats["n_splats_total"] == 1
    assert result.stats["lod_n_lods"] == 1
    assert result.stats["lod_cutpoints"] == [1]
    assert "reference_energy" not in result.stats
    assert "quality" not in result.stats
    assert "n_splats_total" not in result.stats


@pytest.mark.parametrize(
    ("selector", "expected_cutpoints"),
    [(1, [2]), (99, [2]), (-1, [2, 4])],
)
def test_multi_level_root_tracks_selected_level_and_any_count_change(
    selector: int, expected_cutpoints: list[int]
) -> None:
    summary = {
        "lod_n_lods": 2,
        "lod_cutpoints": [2, 4],
        "lod_substitutive_level": selector,
        "reference_energy": 99.0,
        "quality": 30.0,
        "n_splats_total": 4,
    }
    source = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[
                    _wide_rung(0.0, [10.0, 9.0], 0),
                    _wide_rung(1.0, [8.0, 7.0], 1),
                ],
                stats=dict(summary),
            ),
            SubstitutiveLevel(
                additive_sublods=[
                    _wide_rung(2.0, [10.0, 9.0], 0),
                    _wide_rung(3.0, [0.2, 0.1], 1),
                ],
                compression_factor=4,
                level_index=1,
                stats=dict(summary),
            ),
        ],
        stats=dict(summary),
    )

    def _drop_coarse_tail(level: GSplatData) -> GSplatData:
        if level.substitutive_levels[0].level_index == 0:
            return level.translate(np.zeros(3, dtype=np.float32))
        return level.filter(np.array([True, True, False, False]))

    result = source._map_substitutive(_drop_coarse_tail)

    assert [
        [lod.n_splats for lod in level.additive_sublods]
        for level in result.substitutive_levels
    ] == [[2, 2], [2]]
    assert result.stats["lod_substitutive_level"] == selector
    assert result.stats["lod_n_lods"] == len(expected_cutpoints)
    assert result.stats["lod_cutpoints"] == expected_cutpoints
    assert "reference_energy" not in result.stats
    assert "quality" not in result.stats
    assert "n_splats_total" not in result.stats


def test_count_change_refreshes_ladder_stamps_without_pruning() -> None:
    summary = {
        "lod_n_lods": 2,
        "lod_cutpoints": [2, 4],
        "reference_energy": 99.0,
        "quality": 30.0,
        "n_splats_total": 4,
        "label": "keep",
    }
    source = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(
                additive_sublods=[
                    _wide_rung(0.0, [10.0, 9.0], 0),
                    _wide_rung(1.0, [8.0, 7.0], 1),
                ],
                compression_factor=4,
                parent_method="greedy",
                level_index=2,
                stats=dict(summary),
            )
        ],
        stats=dict(summary),
    )

    result = source.cull(method="cumulative", retention=0.79)

    assert [lod.n_splats for lod in result.additive_sublods] == [2, 1]
    assert [lod.stats["lod_level"] for lod in result.additive_sublods] == [0, 1]
    assert [lod.stats["lod_n_splats"] for lod in result.additive_sublods] == [2, 1]
    assert [lod.stats["lod_cumulative_n"] for lod in result.additive_sublods] == [2, 3]
    level = result.substitutive_levels[0]
    assert level.compression_factor == 4
    assert level.parent_method == "greedy"
    assert level.level_index == 2
    assert level.stats["lod_n_lods"] == 2
    assert level.stats["lod_cutpoints"] == [2, 3]
    assert level.stats["n_splats_total"] == 3
    assert level.stats["reference_energy"] != 99.0
    assert np.isfinite(level.stats["reference_energy"])
    assert "quality" not in level.stats
    assert level.stats["label"] == "keep"
    assert result.stats == {
        "lod_n_lods": 2,
        "lod_cutpoints": [2, 3],
        "label": "keep",
        "culled": True,
        "culling_method": "cumulative",
        "n_original": 4,
        "n_culled": 1,
        "amplitude_retention": pytest.approx(27.0 / 34.0),
    }
    assert all(type(value) is int for value in result.stats["lod_cutpoints"])
    json.dumps(result.stats)


def test_all_empty_ladder_keeps_writer_validation_path(tmp_path) -> None:
    source = GSplatData.from_additive_sublods([_rung(0.0, 1.0, 0), _rung(1.0, 1.0, 1)])

    result = source.filter(np.zeros(source.n_splats, dtype=bool))

    assert [lod.n_splats for lod in result.additive_sublods] == [0, 0]
    with pytest.raises(ValidationError, match="positions: Cannot write empty points"):
        result.save(tmp_path / "empty.gsplats.zarr")
