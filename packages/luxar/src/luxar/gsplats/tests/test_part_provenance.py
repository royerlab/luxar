"""Stacked fits retain per-coordinate provenance without inventing one score."""

from __future__ import annotations

import math

import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import split_fitting_info
from luxar.gsplats.merged_quality import collect_part_provenance

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
            source_dtype="uint16",
            source_bytes=120,
        ),
        _fit(
            2,
            psnr_db=43.0,
            foreground_psnr_db=29.75,
            source_shape=[3, 4, 5],
            source_dtype="uint16",
            source_bytes=120,
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
    assert stacked.stats["source_dtype"] == "uint16"
    assert stacked.stats["source_bytes"] == 240

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


def test_part_provenance_validates_reference_and_stack_cardinality() -> None:
    fits = [_fit(1), _fit(2)]
    with pytest.raises(ValueError, match="fit_reference.kind"):
        collect_part_provenance(
            fits,
            values=[0, 1],
            fit_reference={"kind": "unknown"},
        )
    with pytest.raises(ValueError, match="part_provenance.*2 datasets"):
        GSplatData.combine_as_new_dimension(
            fits,
            part_provenance=[{"coordinate": 0, "fitting": {}}],
        )
