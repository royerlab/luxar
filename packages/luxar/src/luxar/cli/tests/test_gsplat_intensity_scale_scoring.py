"""`compare`, `render` and `cull --target` undo a recorded `intensity_scale`.

`transform --scale-intensity k` multiplies every amplitude by `k` and records the
cumulative factor, while `image_min` stays a raw source level. A scorer that puts
the raw reference on the fit's basis must also take the factor back out, or a
pure display rescale reads as reconstruction error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli.main import app
from luxar.cli.tests._testing import normalized_cli_output
from luxar.encoding import EncodingMode
from luxar.gsplats.fit_basis import stored_intensity_scale
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.scoring_grid import amplitudes_in_source_units

PEDESTAL = 100.0
SHAPE = (16, 16, 16)

#: Keys that describe the run, not the score.
_RUN_KEYS = {"gsplats", "reference", "compression_ratio", "intensity_scale"}


def _model(n: int = 30) -> GSplatData:
    rng = np.random.default_rng(0)
    return GSplatData(
        centers=rng.uniform(3.0, 13.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.3, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1.3, 0.2, 1.1, -0.1, 0.1, 1.2], (n, 1)).astype(
            np.float32
        ),
        stats={
            "image_min": PEDESTAL,
            "floor": PEDESTAL,
            "image_max": PEDESTAL + 1000.0,
            "intensity_range": 1000.0,
            "source_shape": list(SHAPE),
        },
    )


def _invoke(*args: str):
    result = CliRunner().invoke(app, ["gsplat", *args])
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture(params=["flat", "partition"])
def store_and_reference(tmp_path: Path, request) -> tuple[Path, Path]:
    """An unedited store, plus the raw volume it reproduces up to a little noise."""
    data = _model()
    rng = np.random.default_rng(1)
    volume = data.render_to_volume(shape=SHAPE, device="cpu") + PEDESTAL
    volume = (volume + rng.normal(0.0, 0.01, SHAPE)).astype(np.float32)
    ref = tmp_path / "ref.npy"
    np.save(ref, volume)
    store = tmp_path / "fit.gsplats.zarr"
    if request.param == "partition":
        write_gsplats_tree(
            store,
            data.to_spatial_partition(max_elements=8),
            ordering="none",
            pipeline_info=data.stats,
            encoding_mode=EncodingMode.PRECISION,
        )
    else:
        data.save(
            store, include_fitting_info=True, encoding_mode=EncodingMode.PRECISION
        )
    return store, ref


def _scaled(store: Path, factor: str) -> Path:
    out = store.with_name(f"scaled_{factor}.gsplats.zarr")
    _invoke(
        "transform",
        str(store),
        str(out),
        "--scale-intensity",
        factor,
        "-e",
        "precision",
    )
    return out


def _compare(store: Path, ref: Path) -> tuple[dict, str]:
    out = store.with_suffix(".json")
    result = _invoke(
        "compare", str(store), str(ref), "--device", "cpu", "--output-json", str(out)
    )
    return json.loads(out.read_text()), result.output


@pytest.mark.parametrize("factor", ["2", "0.3"])
def test_compare_scores_a_rescaled_store_like_the_original(
    store_and_reference, factor: str
) -> None:
    store, ref = store_and_reference
    scaled = _scaled(store, factor)
    _, stats = load_gsplat_node(scaled, include_stats=True)
    assert stats["intensity_scale"] == pytest.approx(float(factor))

    before, _ = _compare(store, ref)
    after, output = _compare(scaled, ref)
    assert "intensity_scale=" in output
    assert after["intensity_scale"] == pytest.approx(float(factor))
    assert before["psnr_robust_db"] > 40.0, "the fixture should score well"
    for key, value in before.items():
        if key in _RUN_KEYS or not isinstance(value, float):
            continue
        assert after[key] == pytest.approx(value, rel=1e-4, abs=1e-9), key


def test_an_unedited_store_is_scored_exactly_as_before(store_and_reference) -> None:
    """Absent (or 1.0) is the identity: no rescale, no extra JSON key, no line.

    (The transform rewrites the store, which can reorder the splats and so the
    float summation; the code-path identity is pinned by
    ``test_identity_scale_returns_the_data_itself``.)
    """
    store, ref = store_and_reference
    noop = _scaled(store, "1.0")
    _, stats = load_gsplat_node(noop, include_stats=True)
    assert "intensity_scale" not in stats

    before, out_before = _compare(store, ref)
    after, out_after = _compare(noop, ref)
    assert "intensity_scale" not in before and "intensity_scale" not in after
    assert "intensity_scale=" not in out_before + out_after
    for key, value in before.items():
        if key in {"gsplats", "reference", "compression_ratio"}:
            continue
        assert after[key] == pytest.approx(value, rel=1e-6, abs=1e-12), key


def test_render_is_in_source_units_after_a_rescale(store_and_reference) -> None:
    store, _ref = store_and_reference
    scaled = _scaled(store, "2")
    base, scaled_out = store.with_suffix(".npy"), scaled.with_suffix(".npy")
    _invoke("render", str(store), str(base), "--device", "cpu")
    result = _invoke("render", str(scaled), str(scaled_out), "--device", "cpu")
    assert "intensity_scale=2" in result.output
    np.testing.assert_allclose(np.load(scaled_out), np.load(base), rtol=1e-5)


def test_cull_error_budget_ignores_a_rescale(tmp_path: Path) -> None:
    """The budget is a percentile of the residual, so a target in the wrong units
    inflates it with the rescale itself and culls on that instead of on fit
    error. Moving the target into the stored units gives the unedited result."""
    data = _model(60)
    rng = np.random.default_rng(2)
    volume = data.render_to_volume(shape=SHAPE, device="cpu") + PEDESTAL
    volume = (volume + rng.normal(0.0, 0.01, SHAPE)).astype(np.float32)
    ref = tmp_path / "ref.npy"
    np.save(ref, volume)
    store = tmp_path / "fit.gsplats.zarr"
    data.save(store, include_fitting_info=True, encoding_mode=EncodingMode.PRECISION)
    scaled = _scaled(store, "4")

    counts = []
    for src in (store, scaled):
        out = src.with_name(f"culled_{src.name}")
        result = _invoke(
            "cull", str(src), str(out), "--target", str(ref), "--device", "cpu"
        )
        counts.append(GSplatData.load(out).n_splats)
        if src is scaled:
            assert "intensity_scale=4" in result.output
            amps = GSplatData.load(out).amplitudes
    assert counts[0] == counts[1]
    # The culled store keeps the edit: its amplitudes are still the scaled ones.
    kept = GSplatData.load(tmp_path / f"culled_{store.name}").amplitudes
    np.testing.assert_allclose(np.sort(amps), np.sort(kept) * 4.0, rtol=1e-5)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 1.0), (2.5, 2.5), (0.0, 1.0), (-2.0, 1.0), (float("nan"), 1.0)],
)
def test_stored_intensity_scale_treats_nonsense_as_no_edit(value, expected) -> None:
    stats = {} if value is None else {"intensity_scale": value}
    assert stored_intensity_scale(stats) == expected
    assert stored_intensity_scale(None) == 1.0
    assert stored_intensity_scale({"intensity_scale": True}) == 1.0


def test_identity_scale_returns_the_data_itself() -> None:
    data = _model(4)
    assert amplitudes_in_source_units(data, 1.0) is data
    halved = amplitudes_in_source_units(data, 2.0)
    np.testing.assert_allclose(halved.amplitudes, data.amplitudes / 2.0)
    np.testing.assert_array_equal(halved.centers, data.centers)


def test_compare_scores_a_normalized_store_like_the_original(
    store_and_reference,
) -> None:
    """``--normalize-intensity`` is one exact global factor too; after a prior
    ``--scale-intensity`` the two compose into one recorded product."""
    store, ref = store_and_reference
    scaled = _scaled(store, "2")
    normalized = scaled.with_name("normalized.gsplats.zarr")
    _invoke(
        "transform",
        str(scaled),
        str(normalized),
        "--normalize-intensity",
        "1.0",
        "-e",
        "precision",
    )
    _, stats = load_gsplat_node(normalized, include_stats=True)
    assert stats["intensity_scale"] != pytest.approx(2.0)

    before, _ = _compare(store, ref)
    after, _ = _compare(normalized, ref)
    for key, value in before.items():
        if key in _RUN_KEYS or not isinstance(value, float):
            continue
        assert after[key] == pytest.approx(value, rel=1e-4, abs=1e-9), key


def test_a_round_trip_rescale_is_not_labelled_as_an_edit(store_and_reference) -> None:
    store, _ref = store_and_reference
    there = _scaled(store, "49")
    back = there.with_name("back.gsplats.zarr")
    _invoke(
        "transform",
        str(there),
        str(back),
        "--scale-intensity",
        str(1.0 / 49.0),
        "-e",
        "precision",
    )
    _, stats = load_gsplat_node(back, include_stats=True)
    assert "intensity_scale" not in stats


@pytest.mark.parametrize("flag", ["--scale-intensity", "--normalize-intensity"])
@pytest.mark.parametrize("bad", ["0", "-2", "nan", "inf"])
def test_a_non_positive_intensity_factor_is_a_usage_error(
    store_and_reference, flag: str, bad: str
) -> None:
    store, _ref = store_and_reference
    out = store.with_name("bad.gsplats.zarr")
    result = CliRunner().invoke(
        app, ["gsplat", "transform", str(store), str(out), flag, bad]
    )
    assert result.exit_code == 2, result.output
    assert "positive" in normalized_cli_output(result)
    assert not out.exists()


def test_compare_warns_when_an_edit_dropped_the_amplitude_basis(
    tmp_path: Path,
) -> None:
    """A clamp is not one global factor, so nothing can be taken back out: the
    store keeps ``image_min`` but loses ``intensity_range``, and says so."""
    data = _model()
    ref = tmp_path / "ref.npy"
    np.save(ref, data.render_to_volume(shape=SHAPE, device="cpu") + PEDESTAL)
    store = tmp_path / "fit.gsplats.zarr"
    data.save(store, include_fitting_info=True)
    clamped = data.clamp_intensity(max=float(np.median(data.amplitudes)))
    assert "intensity_range" not in clamped.stats
    assert "image_min" in clamped.stats
    edited = store.with_name("clamped.gsplats.zarr")
    clamped.save(edited, include_fitting_info=True)
    _, output = _compare(edited, ref)
    assert "no amplitude basis" in output
    _, clean = _compare(store, ref)
    assert "no amplitude basis" not in clean


def test_reference_in_amplitude_units_dtype_and_ownership() -> None:
    from luxar.gsplats.fit_basis import reference_in_amplitude_units

    ints = np.arange(8, dtype=np.uint16)
    out = reference_in_amplitude_units(ints, 2.0)
    assert out.dtype == np.float32 and float(out[-1]) == 14.0

    floats = np.ones(4, dtype=np.float32)
    assert reference_in_amplitude_units(floats, 1.0) is floats
    copied = reference_in_amplitude_units(floats, 3.0)
    assert copied is not floats and float(floats[0]) == 1.0
    in_place = reference_in_amplitude_units(floats, 3.0, in_place=True)
    assert in_place is floats and float(floats[0]) == 3.0
