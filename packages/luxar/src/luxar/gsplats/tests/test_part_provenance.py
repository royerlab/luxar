"""Stacked fits retain per-coordinate provenance without inventing one score."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import split_fitting_info
from luxar.gsplats.lod import RecipeParams, build_recipe
from luxar.gsplats.merged_quality import (
    collect_part_provenance,
    summarize_part_provenance,
)

from ._gsplat_data_helpers import _make_3d_gsplat


def _fit(seed: int, **stats: object) -> GSplatData:
    fit = _make_3d_gsplat(n=2, seed=seed)
    fit.stats.update(stats)
    return fit


def test_stacked_fit_provenance_is_rooted_in_fitting_and_describes_the_source() -> None:
    fits = [
        _fit(
            1,
            psnr_db=41.5,
            foreground_psnr_db=28.25,
            source_shape=[3, 4, 5],
            source_declared=True,
            source_dtype="uint16",
            source_voxels=60,
            source_bytes=120,
            source_stored_bytes=80,
        ),
        _fit(
            2,
            psnr_db=43.0,
            foreground_psnr_db=29.75,
            source_shape=[3, 4, 5],
            source_declared=True,
            source_dtype="uint16",
            source_voxels=60,
            source_bytes=120,
            source_stored_bytes=75,
        ),
    ]
    reference = {
        "kind": "preprocessed",
        "note": "connected-component filter, minimum 2 voxels",
    }

    provenance = collect_part_provenance(
        fits,
        values=[0.0, 5.0],
        fit_reference=reference,
    )
    stacked = GSplatData.combine_as_new_dimension(
        fits,
        values=[0.0, 5.0],
        sigma=0.0,
        part_provenance=provenance,
    )

    assert [part["coordinate"] for part in stacked.stats["part_provenance"]] == [
        0.0,
        5.0,
    ]
    assert stacked.stats["part_provenance"][0]["fit_reference"] == reference
    assert stacked.stats["part_provenance"][1]["fitting"]["psnr_db"] == 43.0
    assert stacked.stats["source_shape"] == [2, 3, 4, 5]
    assert stacked.stats["source_declared"] is True
    assert stacked.stats["source_dtype"] == "uint16"
    assert stacked.stats["source_voxels"] == 120
    assert stacked.stats["source_bytes"] == 240
    assert stacked.stats["source_stored_bytes"] == 155
    for part in stacked.stats["part_provenance"]:
        assert "source_shape" not in part["fitting"]
        assert "source_declared" not in part["fitting"]
        assert "source_dtype" not in part["fitting"]
        assert "source_voxels" not in part["fitting"]
        assert "source_bytes" not in part["fitting"]
    assert stacked.stats["part_provenance"][0]["fitting"]["source_stored_bytes"] == 80
    assert stacked.stats["part_provenance"][1]["fitting"]["source_stored_bytes"] == 75

    fitting, _config, _provenance, pipeline = split_fitting_info(stacked.stats)
    assert fitting is not None
    assert fitting["part_provenance"] == stacked.stats["part_provenance"]
    assert "part_provenance" not in (pipeline or {})


def test_non_finite_part_metrics_are_omitted_without_dropping_the_part() -> None:
    fits = [
        _fit(1, psnr_db=math.nan, foreground_fraction=0.0),
        _fit(2, psnr_db=math.inf, foreground_fraction=0.25),
    ]

    provenance = collect_part_provenance(
        fits,
        values=[0, 1],
        fit_reference={"kind": "synthetic"},
    )

    assert len(provenance) == 2
    assert "psnr_db" not in provenance[0]["fitting"]
    assert "psnr_db" not in provenance[1]["fitting"]
    assert provenance[0]["fitting"]["foreground_fraction"] == 0.0
    assert provenance[1]["fitting"]["foreground_fraction"] == 0.25


def test_unknown_reference_and_nested_provenance_are_preserved() -> None:
    component = collect_part_provenance(
        [_fit(1, psnr_db=41.5), _fit(2)],
        values=[0, 1],
        fit_reference=None,
    )
    stack = _fit(3, part_provenance=component)

    provenance = collect_part_provenance(
        [stack],
        values=[7],
        fit_reference=None,
    )

    assert "fit_reference" not in provenance[0]
    assert provenance[0]["fitting"]["part_provenance"] == component

    composed = _fit(4, part_provenance=provenance)
    reduced = composed.filter_by(amplitude_min=0.5)
    nested = reduced.stats["part_provenance"][0]["fitting"]["part_provenance"]
    assert "psnr_db" not in nested[0]["fitting"]
    assert nested[1]["fitting"] == {}


def test_part_summary_distinguishes_shared_and_independent_sources() -> None:
    provenance = [
        {
            "coordinate": 0.0,
            "fitting": {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_voxels": 512,
                "source_bytes": 1024,
                "time_seconds": 1.5,
            },
        },
        {
            "coordinate": 1.0,
            "fitting": {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_voxels": 512,
                "source_bytes": 1024,
                "time_seconds": 2.5,
            },
        },
    ]

    independent = summarize_part_provenance(provenance)
    shared = summarize_part_provenance(provenance, shared_source=True)

    assert independent == [
        {
            "part_count": 2,
            "fitting": {
                "source_shape": [2, 8, 8, 8],
                "source_dtype": "uint16",
                "source_voxels": 1024,
                "source_bytes": 2048,
                "time_seconds": 4.0,
            },
        }
    ]
    assert shared == [
        {
            "part_count": 2,
            "fitting": {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_voxels": 512,
                "source_bytes": 1024,
                "time_seconds": 4.0,
            },
        }
    ]


def test_shared_part_summary_keeps_dtype_when_shapes_disagree() -> None:
    provenance = [
        {
            "coordinate": 0.0,
            "fitting": {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_voxels": 512,
            },
        },
        {
            "coordinate": 1.0,
            "fitting": {
                "source_shape": [4, 8, 8],
                "source_dtype": "uint16",
                "source_voxels": 256,
            },
        },
    ]

    assert summarize_part_provenance(provenance, shared_source=True) == [
        {
            "part_count": 2,
            "fitting": {
                "source_dtype": "uint16",
                "source_voxels": 768,
            },
        }
    ]


def test_shared_part_summary_keeps_source_identity_without_voxel_total() -> None:
    provenance = [
        {
            "coordinate": coordinate,
            "fitting": {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_declared": True,
            },
        }
        for coordinate in (0.0, 1.0)
    ]

    assert summarize_part_provenance(provenance, shared_source=True) == [
        {
            "part_count": 2,
            "fitting": {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_declared": True,
            },
        }
    ]


def test_part_summary_counts_nested_collapsed_inputs() -> None:
    provenance = [
        {
            "coordinate": 0.0,
            "fitting": {
                "part_provenance": [
                    {
                        "part_count": 2,
                        "fitting": {
                            "source_shape": [2, 8, 8, 8],
                            "source_dtype": "uint16",
                            "source_voxels": 1024,
                            "source_bytes": 2048,
                            "time_seconds": 3.0,
                        },
                    }
                ]
            },
        },
        {
            "coordinate": 1.0,
            "fitting": {
                "source_shape": [8, 8, 8],
                "source_dtype": "uint16",
                "source_voxels": 512,
                "source_bytes": 1024,
                "time_seconds": 4.0,
            },
        },
    ]

    assert summarize_part_provenance(provenance) == [
        {
            "part_count": 3,
            "fitting": {
                "source_dtype": "uint16",
                "source_voxels": 1536,
                "source_bytes": 3072,
                "time_seconds": 7.0,
            },
        }
    ]


def test_part_provenance_validates_reference_and_stack_cardinality() -> None:
    fits = [_fit(1), _fit(2)]
    with pytest.raises(ValueError, match="fit_reference.kind"):
        collect_part_provenance(
            fits,
            values=[0, 1],
            fit_reference={"kind": "unknown"},
        )
    with pytest.raises(ValueError, match="fit_reference.note"):
        collect_part_provenance(
            fits,
            values=[0, 1],
            fit_reference={"kind": "preprocessed", "note": "  "},
        )
    with pytest.raises(ValueError, match="part_provenance.*2 datasets"):
        GSplatData.combine_as_new_dimension(
            fits,
            part_provenance=[{"coordinate": 0, "fitting": {}}],
        )

    valid = {"coordinate": 1.0, "fitting": {}}
    with pytest.raises(TypeError, match=r"part_provenance\[0\] must be a dict"):
        GSplatData.combine_as_new_dimension(
            fits,
            values=[0.0, 1.0],
            part_provenance=cast(Any, [None, valid]),
        )
    with pytest.raises(ValueError, match=r"part_provenance\[0\]\.coordinate"):
        GSplatData.combine_as_new_dimension(
            fits,
            values=[0.0, 1.0],
            part_provenance=[{"coordinate": 2.0, "fitting": {}}, valid],
        )
    with pytest.raises(ValueError, match=r"part_provenance\[0\]\.fitting"):
        GSplatData.combine_as_new_dimension(
            fits,
            values=[0.0, 1.0],
            part_provenance=cast(
                Any,
                [{"coordinate": 0.0, "fitting": []}, valid],
            ),
        )


def test_source_summary_requires_complete_agreement() -> None:
    fits = [
        _fit(
            1,
            source_shape=[3, 4, 5],
            source_declared=True,
            source_dtype="uint16",
            source_voxels=60,
            source_bytes=120,
            source_stored_bytes=80,
        ),
        _fit(
            2,
            source_shape=[6, 4, 5],
            source_declared=True,
            source_dtype="uint16",
            source_voxels=120,
            source_bytes=240,
            source_stored_bytes=160,
        ),
    ]
    provenance = collect_part_provenance(
        fits,
        values=[0, 1],
        fit_reference={"kind": "preprocessed"},
    )

    stacked = GSplatData.combine_as_new_dimension(
        fits,
        values=[0, 1],
        part_provenance=provenance,
    )

    assert "source_shape" not in stacked.stats
    assert "source_declared" not in stacked.stats
    assert stacked.stats["source_dtype"] == "uint16"
    assert "source_voxels" not in stacked.stats
    assert "source_bytes" not in stacked.stats
    assert "source_stored_bytes" not in stacked.stats


@pytest.mark.parametrize(
    ("disputed_key", "disputed_value", "unanimous_key", "unanimous_value"),
    [
        ("source_dtype", "uint8", "source_declared", True),
        ("source_declared", False, "source_dtype", "uint16"),
    ],
)
def test_source_summary_omits_only_disputed_source_fields(
    disputed_key: str,
    disputed_value: Any,
    unanimous_key: str,
    unanimous_value: Any,
) -> None:
    source_stats = {
        "source_shape": [3, 4, 5],
        "source_declared": True,
        "source_dtype": "uint16",
        "source_voxels": 60,
        "source_bytes": 120,
        "source_stored_bytes": 80,
    }
    second_source_stats = {**source_stats, disputed_key: disputed_value}
    fits = [_fit(1, **source_stats), _fit(2, **second_source_stats)]
    provenance = collect_part_provenance(
        fits,
        values=[0, 1],
        fit_reference={"kind": "preprocessed"},
    )

    stacked = GSplatData.combine_as_new_dimension(
        fits,
        values=[0, 1],
        part_provenance=provenance,
    )

    assert stacked.stats["source_shape"] == [2, 3, 4, 5]
    assert disputed_key not in stacked.stats
    assert stacked.stats[unanimous_key] == unanimous_value
    assert stacked.stats["source_voxels"] == 120
    assert stacked.stats["source_bytes"] == 240
    assert stacked.stats["source_stored_bytes"] == 160


def test_part_provenance_survives_lod_and_save_load(tmp_path: Path) -> None:
    fits = [
        _fit(
            1,
            source_shape=[8, 8, 8],
            source_declared=True,
            source_dtype="uint16",
            source_voxels=512,
            source_bytes=1024,
            source_stored_bytes=700,
        ),
        _fit(
            2,
            source_shape=[8, 8, 8],
            source_declared=True,
            source_dtype="uint16",
            source_voxels=512,
            source_bytes=1024,
            source_stored_bytes=650,
        ),
    ]
    provenance = collect_part_provenance(
        fits,
        values=[0.0, 1.0],
        fit_reference={"kind": "acquisition"},
    )
    stacked = GSplatData.combine_as_new_dimension(
        fits,
        values=[0.0, 1.0],
        part_provenance=provenance,
    )

    ladder = build_recipe(
        stacked,
        "levels",
        RecipeParams(
            compression_factor=2,
            levels=2,
            coarsen_dims=[0, 1, 2],
            device="cpu",
        ),
    )
    path = tmp_path / "stack.gsplats.zarr"
    ladder.save(path, include_fitting_info=True)
    loaded = GSplatData.load(path, include_stats=True)

    assert loaded.stats["part_provenance"] == stacked.stats["part_provenance"]
    assert loaded.stats["source_shape"] == [2, 8, 8, 8]
    assert loaded.stats["source_declared"] is True
    assert loaded.stats["source_voxels"] == 1024
    assert loaded.stats["source_bytes"] == 2048
    assert loaded.stats["source_stored_bytes"] == 1350
