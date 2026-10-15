"""`gsplat compare` must score on the fit's basis, and validate an override (#1173).

A render is background-relative; a raw reference is not. These pin that the
command reconciles them from the store, that an explicit override is honoured, and
that a nonsensical override is refused rather than silently scoring the wrong
thing.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli.main import app
from luxar.cli.tests._testing import normalized_cli_output
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import write_gsplats_tree

PEDESTAL = 600.0


@pytest.fixture(params=["flat", "partition"])
def fitted_store_and_reference(tmp_path, request):
    """A store advertising a known basis, plus the raw volume it was fitted from."""
    volume = np.full((10, 10, 10), PEDESTAL, dtype=np.float32)
    volume[3:7, 3:7, 3:7] += 1200.0
    ref = tmp_path / "ref.npy"
    np.save(ref, volume)

    rng = np.random.default_rng(0)
    n = 10
    data = GSplatData(
        centers=rng.uniform(3.0, 7.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.3, 0.9, n).astype(np.float32),
        cholesky_factors=np.tile([1.2, 0, 1.2, 0, 0, 1.2], (n, 1)).astype(np.float32),
        stats={"image_min": PEDESTAL, "floor": PEDESTAL},
    )
    store = tmp_path / "fit.gsplats.zarr"
    if request.param == "partition":
        write_gsplats_tree(
            store,
            data.to_spatial_partition(max_elements=4),
            ordering="none",
            pipeline_info=data.stats,
        )
    else:
        data.save(store, include_fitting_info=True)
    return store, ref


def _run(*args: str):
    return CliRunner().invoke(app, ["gsplat", "compare", *args])


def test_the_stored_basis_is_read_and_reported(fitted_store_and_reference) -> None:
    """Silence here would leave a reader unable to tell which basis a score is on."""
    store, ref = fitted_store_and_reference
    result = _run(str(store), str(ref))
    assert result.exit_code == 0, result.output
    assert "image_min=600" in result.output
    assert "from dataset" in result.output


def test_an_explicit_override_wins_and_says_so(fitted_store_and_reference) -> None:
    """For a store fitted before the level was persisted."""
    store, ref = fitted_store_and_reference
    result = _run(str(store), str(ref), "--image-min", "100")
    assert result.exit_code == 0, result.output
    assert "image_min=100" in result.output
    assert "from --image-min" in result.output


@pytest.mark.parametrize("bad", ["-50", "nan", "inf"])
def test_a_nonsensical_override_is_refused(fitted_store_and_reference, bad) -> None:
    """A level is what the fit SUBTRACTED, so it cannot be negative or
    non-finite. A negative one shifts the reference the wrong way, which would
    score against the wrong thing silently — and typing `-50` for `50` is the
    obvious way to get there. Validated to the same standard as a stored level,
    which `fit_image_min` already refuses.

    Rejected as a `typer.BadParameter`, so this is a usage error (exit 2) rather
    than a runtime failure — the value is wrong on the command line, before any
    volume is read.
    """
    store, ref = fitted_store_and_reference
    result = _run(str(store), str(ref), "--image-min", bad)
    output = normalized_cli_output(result)
    assert result.exit_code == 2, output
    assert "finite, non-negative level" in output


def test_a_store_with_no_basis_warns_rather_than_pretending(tmp_path) -> None:
    """Absence is ordinary — a merge drops keys its inputs disagree on — so the
    command must say the score is on the raw volume instead of implying otherwise.
    """
    volume = np.full((8, 8, 8), 300.0, dtype=np.float32)
    volume[2:5, 2:5, 2:5] += 900.0
    ref = tmp_path / "ref.npy"
    np.save(ref, volume)

    rng = np.random.default_rng(1)
    data = GSplatData(
        centers=rng.uniform(2.0, 5.0, (8, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.3, 0.9, 8).astype(np.float32),
        cholesky_factors=np.tile([1.0, 0, 1.0, 0, 0, 1.0], (8, 1)).astype(np.float32),
    )
    store = tmp_path / "nobasis.gsplats.zarr"
    data.save(store)

    result = _run(str(store), str(ref))
    assert result.exit_code == 0, result.output
    assert "image_min" in result.output  # the hint names the missing stat


# ---------------------------------------------------------------------------
# Scoring grid, peaks, foreground, input selection (WP5)
# ---------------------------------------------------------------------------

#: Keys that describe the run, not the score.
_RUN_KEYS = {"gsplats", "reference", "compression_ratio", "voxel_size"}


def _voxel_model(n: int = 30, seed: int = 0) -> GSplatData:
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(3.0, 13.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.3, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1.3, 0.2, 1.1, -0.1, 0.1, 1.2], (n, 1)).astype(
            np.float32
        ),
        stats={"image_min": 0.0},
    )


def _physical(data: GSplatData, vs: tuple[float, ...], stamp: bool) -> GSplatData:
    """The fitter's physical conversion (fitting/results.py), plus its stamp."""
    tril = np.concatenate([[vs[i]] * (i + 1) for i in range(3)])
    stats = dict(data.stats)
    if stamp:
        stats["voxel_size"] = list(vs)
    return GSplatData(
        centers=(data.centers * np.asarray(vs)).astype(np.float32),
        amplitudes=data.amplitudes,
        cholesky_factors=(data.cholesky_factors * tril).astype(np.float32),
        stats=stats,
    )


def _save(data: GSplatData, path) -> None:
    from luxar.encoding import EncodingMode

    data.save(path, encoding_mode=EncodingMode.PRECISION, include_fitting_info=True)


def _json_compare(store, ref, out, *extra: str) -> dict:
    import json

    result = _run(str(store), str(ref), "--device", "cpu", "-j", str(out), *extra)
    assert result.exit_code == 0, result.output
    return json.loads(out.read_text())


def _scores(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _RUN_KEYS}


@pytest.fixture
def blob_reference(tmp_path):
    rng = np.random.default_rng(5)
    volume = rng.normal(0.0, 0.02, (16, 16, 16)).astype(np.float32)
    volume[4:12, 4:12, 4:12] += 0.8
    ref = tmp_path / "ref.npy"
    np.save(ref, volume)
    return ref, volume


def test_a_physical_fit_scores_like_the_same_model_in_voxels(
    tmp_path, blob_reference
) -> None:
    """The stamp puts physical-unit splats back on the reference's grid."""
    ref, _ = blob_reference
    vox = _voxel_model()
    _save(vox, tmp_path / "vox.gsplats.zarr")
    _save(_physical(vox, (2.0, 0.4, 0.25), True), tmp_path / "phys.gsplats.zarr")

    by_voxel = _json_compare(tmp_path / "vox.gsplats.zarr", ref, tmp_path / "a.json")
    by_phys = _json_compare(tmp_path / "phys.gsplats.zarr", ref, tmp_path / "b.json")

    assert by_phys["voxel_size"] == [2.0, 0.4, 0.25]
    assert by_voxel["voxel_size"] is None
    assert _scores(by_phys) == pytest.approx(_scores(by_voxel), rel=1e-4, abs=1e-6)


def test_an_unstamped_physical_store_warns_and_accepts_voxel_size(
    tmp_path, blob_reference
) -> None:
    ref, _ = blob_reference
    vox = _voxel_model()
    _save(vox, tmp_path / "vox.gsplats.zarr")
    store = tmp_path / "phys.gsplats.zarr"
    _save(_physical(vox, (2.0, 0.4, 0.25), False), store)

    unmapped = _run(str(store), str(ref), "--device", "cpu")
    assert unmapped.exit_code == 0, unmapped.output
    assert "fall inside the voxel grid" in unmapped.output
    assert "--voxel-size" in unmapped.output

    by_voxel = _json_compare(tmp_path / "vox.gsplats.zarr", ref, tmp_path / "a.json")
    overridden = _json_compare(
        store, ref, tmp_path / "b.json", "--voxel-size", "2,0.4,0.25"
    )
    assert _scores(overridden) == pytest.approx(_scores(by_voxel), rel=1e-4, abs=1e-6)


def test_a_bad_voxel_size_is_a_usage_error(tmp_path, blob_reference) -> None:
    ref, _ = blob_reference
    _save(_voxel_model(), tmp_path / "vox.gsplats.zarr")
    result = _run(str(tmp_path / "vox.gsplats.zarr"), str(ref), "--voxel-size", "1,2")
    assert result.exit_code == 2, result.output


def test_json_carries_both_peaks_with_clear_keys(tmp_path, blob_reference) -> None:
    ref, volume = blob_reference
    _save(_voxel_model(), tmp_path / "vox.gsplats.zarr")
    payload = _json_compare(tmp_path / "vox.gsplats.zarr", ref, tmp_path / "m.json")
    assert payload["psnr_headline"] == "psnr_robust_db"
    assert payload["robust_peak_quantile"] == 0.9999
    for key in (
        "psnr_robust_db",
        "psnr_db",
        "foreground_psnr_robust_db",
        "foreground_psnr_db",
        "foreground_ssim",
        "robust_data_range",
        "data_range",
    ):
        assert key in payload
    assert payload["data_range"] == pytest.approx(float(volume.max() - volume.min()))
    assert payload["robust_data_range"] <= payload["data_range"]
    assert payload["foreground_method"] == "otsu_smoothed"


def test_mask_and_threshold_override_the_foreground(tmp_path, blob_reference) -> None:
    ref, volume = blob_reference
    store = tmp_path / "vox.gsplats.zarr"
    _save(_voxel_model(), store)
    mask = np.zeros_like(volume, dtype=np.uint8)
    mask[:8] = 1
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)

    masked = _json_compare(store, ref, tmp_path / "m.json", "--mask", str(mask_path))
    assert masked["foreground_method"] == "mask"
    assert masked["foreground_fraction"] == pytest.approx(0.5)

    thresholded = _json_compare(
        store, ref, tmp_path / "t.json", "--fg-threshold", "0.4"
    )
    assert thresholded["foreground_method"] == "threshold"
    assert thresholded["foreground_threshold_raw"] == 0.4
    assert thresholded["foreground_fraction"] == pytest.approx(
        float((volume > 0.4).mean())
    )

    np.save(mask_path, mask[:4])
    wrong = _run(str(store), str(ref), "--mask", str(mask_path))
    assert wrong.exit_code == 2, wrong.output


def test_fg_threshold_is_in_the_references_raw_units(fitted_store_and_reference):
    """A user reads intensities off the raw stack; the basis shift is ours."""
    import json

    store, ref = fitted_store_and_reference
    out = ref.parent / "t.json"
    result = _run(str(store), str(ref), "--fg-threshold", "1000", "-j", str(out))
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    # PEDESTAL 600 + 1200 inside a 4^3 cube of a 10^3 volume.
    assert payload["foreground_fraction"] == pytest.approx(64 / 1000)
    assert payload["foreground_threshold"] == pytest.approx(1000 - PEDESTAL)


def test_axes_and_array_key_select_the_fitted_channel(tmp_path, blob_reference):
    ref, volume = blob_reference
    store = tmp_path / "vox.gsplats.zarr"
    _save(_voxel_model(), store)
    stack = np.stack([np.zeros_like(volume), volume])  # (c, z, y, x)
    npz = tmp_path / "stack.npz"
    np.savez(npz, other=np.zeros(3), vol=stack)

    direct = _json_compare(store, ref, tmp_path / "a.json")
    selected = _json_compare(
        store,
        npz,
        tmp_path / "b.json",
        "--array-key",
        "vol",
        "--axes",
        "c,z,y,x",
        "--channel",
        "1",
    )
    assert _scores(selected) == pytest.approx(_scores(direct), rel=1e-6, abs=1e-9)


def test_a_tiff_channel_without_axes_is_refused(tmp_path, blob_reference) -> None:
    """It used to be silently ignored: every channel scored as extra z."""
    import tifffile

    _, volume = blob_reference
    store = tmp_path / "vox.gsplats.zarr"
    _save(_voxel_model(), store)
    tif = tmp_path / "stack.tif"
    tifffile.imwrite(tif, np.stack([volume, volume]))

    result = _run(str(store), str(tif), "--channel", "1")
    output = normalized_cli_output(result)
    assert result.exit_code == 1, output
    assert "channel=1" in output and "--axes" in output

    ok = _run(str(store), str(tif), "--channel", "1", "--axes", "c,z,y,x")
    assert ok.exit_code == 0, ok.output


@pytest.mark.parametrize("command", ["fit", "cal"])
def test_fit_and_cal_refuse_an_inapplicable_tiff_index(tmp_path, command) -> None:
    import tifffile

    tif = tmp_path / "stack.tif"
    tifffile.imwrite(tif, np.zeros((2, 8, 8, 8), np.float32))
    out = tmp_path / ("out.gsplats.zarr" if command == "fit" else "cal.json")
    result = CliRunner().invoke(
        app, ["gsplat", command, str(tif), str(out), "--timepoint", "1"]
    )
    output = normalized_cli_output(result)
    assert result.exit_code == 1, output
    assert "timepoint=1" in output


def test_at_fit_resolution_scores_on_the_decimated_grid(tmp_path) -> None:
    """A --downscale fit's splats live on the source grid; this undoes that."""
    from luxar.gsplats.fitting.downscale import downscale_volume

    rng = np.random.default_rng(9)
    volume = rng.normal(0.0, 0.02, (20, 20, 20)).astype(np.float32)
    volume[6:14, 6:14, 6:14] += 0.8
    ref = tmp_path / "ref.npy"
    np.save(ref, volume)

    down = _voxel_model(seed=2)  # a model on the (10, 10, 10) fit grid
    down.centers[:] = np.clip(down.centers * 0.6, 0.0, 9.0)
    rescaled = _physical(down, (2.0, 2.0, 2.0), stamp=False)  # = rescale_centers
    rescaled.stats.update({"source_shape": [20, 20, 20], "fitted_shape": [10, 10, 10]})
    store = tmp_path / "ds.gsplats.zarr"
    _save(rescaled, store)

    at_fit = _json_compare(store, ref, tmp_path / "f.json", "--at-fit-resolution")
    assert at_fit["shape"] == [10, 10, 10]
    assert at_fit["scored_grid"] == "fit"
    assert at_fit["downscale_factors"] == [2, 2, 2]

    # Equivalent by hand: the fit-grid model against the decimated reference.
    np.save(tmp_path / "ref_down.npy", downscale_volume(volume, (2, 2, 2)))
    _save(down, tmp_path / "down.gsplats.zarr")
    by_hand = _json_compare(
        tmp_path / "down.gsplats.zarr", tmp_path / "ref_down.npy", tmp_path / "h.json"
    )
    run_keys = {"scored_grid", "downscale_factors", "n_splats"}
    a = {k: v for k, v in _scores(at_fit).items() if k not in run_keys}
    b = {k: v for k, v in _scores(by_hand).items() if k not in run_keys}
    assert a == pytest.approx(b, rel=1e-4, abs=1e-6)

    full = _json_compare(store, ref, tmp_path / "s.json")
    assert full["shape"] == [20, 20, 20] and full["scored_grid"] == "source"


def test_quiet_is_silent(fitted_store_and_reference) -> None:
    store, ref = fitted_store_and_reference
    out = ref.parent / "q.json"
    result = _run(str(store), str(ref), "-q", "-j", str(out))
    assert result.exit_code == 0
    assert result.output.strip() == ""
    assert out.exists()


def test_a_transformed_and_saved_store_keeps_its_basis(
    fitted_store_and_reference,
) -> None:
    """`GSplatData.load` → edit → `save` in Python must not lose image_min."""
    store, ref = fitted_store_and_reference
    if not (store / "fitting").exists() and not (store / "pipeline").exists():
        pytest.skip("partition fixture variant carries its basis differently")
    try:
        data = GSplatData.load(store)
    except ValueError:
        pytest.skip("partition stores are not matrix-shaped")
    assert data.stats.get("image_min") == pytest.approx(PEDESTAL)
    edited = store.parent / "edited.gsplats.zarr"
    data.translate(np.zeros(3)).save(edited)

    result = _run(str(edited), str(ref))
    assert result.exit_code == 0, result.output
    assert "image_min=600" in result.output and "from dataset" in result.output


def test_physical_load_save_keeps_scoring_grid(tmp_path, blob_reference) -> None:
    ref, _ = blob_reference
    source = tmp_path / "physical.gsplats.zarr"
    saved = tmp_path / "saved.gsplats.zarr"
    _save(_physical(_voxel_model(), (2.0, 0.5, 0.5), True), source)
    before = _json_compare(source, ref, tmp_path / "before.json")

    from luxar.encoding import EncodingMode

    GSplatData.load(source).save(saved, encoding_mode=EncodingMode.PRECISION)
    after = _json_compare(saved, ref, tmp_path / "after.json")

    assert GSplatData.load(saved).stats["voxel_size"] == [2.0, 0.5, 0.5]
    assert after["psnr_robust_db"] == pytest.approx(before["psnr_robust_db"])


def test_a_hot_pixel_moves_neither_the_headline_peak_nor_the_foreground(
    tmp_path,
) -> None:
    """Plain Otsu over [min, max] put every real voxel below the threshold once
    one voxel sat at 60000; the robust peak and the capped histogram ignore it."""
    # p99.99 can only exclude a voxel once there are >= 10,000 of them (a 16^3
    # volume's nearest-rank p99.99 IS its max), hence a 24x32x32 reference.
    rng = np.random.default_rng(5)
    volume = rng.normal(0.0, 0.02, (24, 32, 32)).astype(np.float32)
    volume[4:12, 4:12, 4:12] += 0.8
    ref = tmp_path / "ref32.npy"
    np.save(ref, volume)
    store = tmp_path / "vox.gsplats.zarr"
    _save(_voxel_model(), store)
    clean = _json_compare(store, ref, tmp_path / "c.json")

    hot = volume.copy()
    hot[0, 0, 0] = 1000.0
    hot_ref = tmp_path / "hot.npy"
    np.save(hot_ref, hot)
    spiked = _json_compare(store, hot_ref, tmp_path / "h.json")

    # The spike takes one of the top ranks, shifting p99.99 by one order
    # statistic — not to 1000.
    assert spiked["robust_data_range"] == pytest.approx(
        clean["robust_data_range"], rel=0.01
    )
    assert spiked["data_range"] > 100 * clean["data_range"]
    # The blurred spike itself is foreground (its tent footprint, <= 27 voxels);
    # before the capped histogram the whole blob dropped out (fraction ~0).
    assert spiked["foreground_fraction"] == pytest.approx(
        clean["foreground_fraction"], abs=27 / volume.size
    )


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (("--shape", "8,8,8"), "does not match reference shape"),
        (("--axes", "c,z,y,x"), "Dimension mismatch"),
    ],
)
def test_quiet_still_reports_errors(fitted_store_and_reference, extra, needle):
    """-q silences narration, never the reason a run failed."""
    store, ref = fitted_store_and_reference
    np.save(ref.parent / "r4.npy", np.zeros((2, 10, 10, 10), np.float32))
    target = ref if extra[0] == "--shape" else ref.parent / "r4.npy"
    if extra[0] == "--axes":
        extra = ("--axes", "t,c,y,x")  # drops two axes: a 2-D reference
    result = _run(str(store), str(target), "-q", *extra)
    output = normalized_cli_output(result)
    assert result.exit_code != 0
    assert needle in output, output


def test_quiet_reports_a_missing_reader(fitted_store_and_reference, monkeypatch):
    """The ImportError branch (no tifffile) used to aprint + exit silently."""
    import sys

    store, _ = fitted_store_and_reference
    tif = store.parent / "ref.tif"
    tif.write_bytes(b"not read")
    monkeypatch.setitem(sys.modules, "tifffile", None)
    result = _run(str(store), str(tif), "-q")
    output = normalized_cli_output(result)
    assert result.exit_code == 1
    assert "tifffile" in output, output
