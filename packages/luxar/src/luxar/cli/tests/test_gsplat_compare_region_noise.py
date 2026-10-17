"""``gsplat compare`` on a ``--region`` crop fit, and its noise-floor reference.

A crop fit keeps full-volume centers and records ``fitting/source_region``.
Scored against the FULL source volume it used to (a) warn "crammed into a
corner?" because the splats fill only the region, and (b) take its default
foreground mask over the whole volume, so the foreground PSNR disagreed with the
fit-time stamp by up to 6 dB. It now scores the fitted region by default.

The noise reference answers "is ~20 dB foreground PSNR good?": the PSNR a
noise-free model would score against the noisy reference, from the noise sigma
measured in the foreground.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli.gsplat_ops.compare_ops import (
    model_error_over_noise,
    noise_reference,
    noise_verdict,
)
from luxar.cli.main import app
from luxar.cli.tests._testing import normalized_cli_output
from luxar.cli.tests.test_gsplat_render_grid import _crop_fit_store
from luxar.gsplats.calibration.noise_floor import masked_noise_sigma

_FULL_SHAPE = (24, 26, 22)
_REGION = ((5, 19), (7, 21), (3, 17))


def _full_reference(tmp_path: Path, crop: np.ndarray) -> Path:
    """The crop's render pasted into a bigger volume holding a structure the fit
    never saw (outside the region)."""
    full = np.zeros(_FULL_SHAPE, np.float32)
    full[tuple(slice(a, b) for a, b in _REGION)] = crop
    full[20:23, 1:5, 18:21] = float(crop.max())  # outside the region
    path = tmp_path / "full_ref.npy"
    np.save(path, full)
    return path


def _compare(store: Path, ref: Path, out: Path, *extra: str) -> tuple[str, dict]:
    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "compare",
            str(store),
            str(ref),
            "--device",
            "cpu",
            "-j",
            str(out),
            *extra,
        ],
    )
    output = normalized_cli_output(result)
    assert result.exit_code == 0, output
    return output, json.loads(out.read_text())


def test_a_crop_fit_is_scored_on_its_region_of_the_full_volume(tmp_path: Path) -> None:
    store, crop_model = _crop_fit_store(tmp_path, physical=False)
    crop = crop_model.render_to_volume(shape=(14, 14, 14), device="cpu")
    ref = _full_reference(tmp_path, crop)

    output, record = _compare(store, ref, tmp_path / "m.json")
    assert "fitted region 5:19,7:21,3:17" in output
    assert "crammed" not in output
    assert record["scored_region"] == [list(span) for span in _REGION]
    assert record["shape"] == [14, 14, 14]
    assert record["max_abs_error"] < 1e-3  # a perfect fit of what it saw
    assert record["fitted_region"] == record["scored_region"]
    assert "of the fitted region" in output  # compression vs the region


def test_whole_volume_scores_everything_without_a_corner_warning(
    tmp_path: Path,
) -> None:
    store, crop_model = _crop_fit_store(tmp_path, physical=False)
    crop = crop_model.render_to_volume(shape=(14, 14, 14), device="cpu")
    ref = _full_reference(tmp_path, crop)

    output, record = _compare(store, ref, tmp_path / "w.json", "--whole-volume")
    assert record["scored_region"] is None
    assert record["fitted_region"] == [list(span) for span in _REGION]
    assert record["whole_volume"] is True
    assert record["shape"] == list(_FULL_SHAPE)
    # The structure outside the region is error now...
    assert record["max_abs_error"] > 0.5 * float(crop.max())
    # ...but the splats filling only their region are no "corner" frame mismatch.
    assert "crammed" not in output
    assert "counts as error" in output


def test_a_full_volume_mask_is_cropped_with_the_reference(tmp_path: Path) -> None:
    store, crop_model = _crop_fit_store(tmp_path, physical=False)
    crop = crop_model.render_to_volume(shape=(14, 14, 14), device="cpu")
    ref = _full_reference(tmp_path, crop)
    mask = np.zeros(_FULL_SHAPE, np.uint8)
    mask[8:14, 10:16, 6:12] = 1  # inside the region
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)

    _, record = _compare(store, ref, tmp_path / "k.json", "--mask", str(mask_path))
    assert record["foreground_method"] == "mask"
    assert record["foreground_fraction"] == pytest.approx(6**3 / 14**3)


def test_corner_warning_still_fires_for_a_real_frame_mismatch(tmp_path: Path) -> None:
    """Region-awareness must not hide the unstamped-physical-store symptom."""
    from luxar.cli.gsplat_ops.compare_ops import warn_if_outside_grid
    from luxar.utils.verbosity import verbosity

    rng = np.random.default_rng(0)
    centers = rng.uniform(0, 4, (50, 3)).astype(np.float32)  # a 4^3 corner
    lines: list[str] = []
    import luxar.cli.gsplat_ops.compare_ops as ops

    original = ops.aprint
    ops.aprint = lambda msg="": lines.append(str(msg))  # type: ignore[assignment]
    try:
        with verbosity("normal"):
            warn_if_outside_grid(centers, (40, 40, 40))
            warn_if_outside_grid(centers, (40, 40, 40), extent_shape=(5, 5, 5))
    finally:
        ops.aprint = original  # type: ignore[assignment]
    assert len(lines) == 1 and "crammed" in lines[0]


# ---------------------------------------------------------------------------
# Noise reference
# ---------------------------------------------------------------------------


def _noisy_blob(sigma: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    zz, yy, xx = np.mgrid[:20, :64, :64].astype(np.float32)
    clean = 400.0 * np.exp(
        -((zz - 10) ** 2 / 40 + (yy - 32) ** 2 / 300 + (xx - 32) ** 2 / 300)
    )
    noise = np.random.default_rng(seed).normal(0.0, sigma, clean.shape)
    return clean.astype(np.float32), (clean + noise).astype(np.float32)


def test_masked_noise_sigma_recovers_the_foreground_noise() -> None:
    clean, noisy = _noisy_blob(sigma=12.0)
    mask = clean > 100.0
    assert masked_noise_sigma(noisy, mask) == pytest.approx(12.0, rel=0.08)
    # Measured where the mask is: a noise-free background does not dilute it.
    hetero = clean + np.where(mask, noisy - clean, 0.0)
    assert masked_noise_sigma(hetero, mask) == pytest.approx(12.0, rel=0.08)
    assert math.isnan(masked_noise_sigma(noisy, np.zeros_like(mask)))
    assert math.isnan(masked_noise_sigma(np.zeros(5), np.ones(5, bool)))


def test_noise_ceiling_is_the_score_of_the_clean_signal() -> None:
    """A noise-free model's foreground PSNR against the noisy data IS the ceiling."""
    import torch

    from luxar.gsplats.metrics import compute_quality_metrics

    clean, noisy = _noisy_blob(sigma=12.0, seed=1)
    noisy = np.clip(noisy, 0.0, None)
    metrics = compute_quality_metrics(torch.from_numpy(clean), torch.from_numpy(noisy))
    from luxar.gsplats.metrics import default_foreground

    field, _ = default_foreground(torch.from_numpy(noisy))
    mask = (field > metrics["foreground_threshold"]).numpy()
    ref = noise_reference(noisy, mask, metrics)
    assert ref["foreground_psnr_noise_ceiling_db"] == pytest.approx(
        metrics["foreground_psnr_robust_db"], abs=0.7
    )
    assert "at the noise floor" in ref["noise_verdict"]


def test_noise_verdict_bands() -> None:
    assert model_error_over_noise(-3.0103) == pytest.approx(1.0, rel=1e-3)
    assert model_error_over_noise(0.5) is None
    above = noise_verdict(2.0, 25.0)
    assert above.startswith("above the estimated ceiling (2.0 dB): ambiguous")
    assert "sub-voxel" in above and "reproducing noise" in above
    # Far above is likelier a wrong ceiling than a fit: no verdict at all.
    assert noise_verdict(8.0, 25.0).startswith("undetermined: 8.0 dB above")
    for gap in (1.5, 2.9, 3.5, 20.0):  # never a definite "fitting noise" call
        assert "ABOVE" not in noise_verdict(gap, 25.0)
    assert "at the noise floor" in noise_verdict(0.4, 25.0)
    assert "at the noise floor" in noise_verdict(-0.9, 25.0)
    assert "near the noise floor" in noise_verdict(-2.5, 25.0)
    below = noise_verdict(-6.0, 25.0)
    assert "6.0 dB below the noise ceiling" in below and "signal left" in below
    assert "nearly noise-free" in noise_verdict(-6.0, 75.0)


def _spots(sigma_blob: tuple[float, ...], seed: int = 0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    shape = (24, 96, 96)
    v = np.zeros(shape, np.float32)
    v[tuple(rng.integers(0, n, 200) for n in shape)] = 1.0
    v = gaussian_filter(v, sigma_blob)
    return (v / v.max()).astype(np.float32)


@pytest.mark.parametrize(
    ("blob", "noise", "reliable"),
    [
        ((2.0, 6.0, 6.0), 0.02, True),  # smooth specimen: the band is noise
        ((1.5, 3.0, 3.0), 0.1, True),
        ((1.0, 1.5, 1.5), 0.02, False),  # spots at the PSF scale
        ((0.7, 0.8, 0.8), 0.02, False),  # sub-voxel puncta
    ],
)
def test_voxel_scale_structure_is_flagged_not_counted_as_noise(
    blob: tuple[float, ...], noise: float, reliable: bool
) -> None:
    from luxar.gsplats.calibration.noise_floor import estimate_masked_noise

    clean = _spots(blob)
    noisy = clean + np.random.default_rng(1).normal(0.0, noise, clean.shape)
    est = estimate_masked_noise(noisy.astype(np.float32), clean > 0.1)
    assert est.reliable is reliable, est
    if reliable:
        assert est.sigma == pytest.approx(noise, rel=0.15)


def test_unreliable_ceiling_gives_no_verdict() -> None:
    import torch

    from luxar.gsplats.metrics import compute_quality_metrics, default_foreground

    clean = _spots((0.7, 0.8, 0.8))
    noisy = np.clip(
        clean + np.random.default_rng(2).normal(0.0, 0.02, clean.shape), 0.0, None
    ).astype(np.float32)
    metrics = compute_quality_metrics(torch.from_numpy(clean), torch.from_numpy(noisy))
    field, thr = default_foreground(torch.from_numpy(noisy))
    ref = noise_reference(noisy, (field > thr).numpy(), metrics)
    assert ref["noise_ceiling_reliable"] is False
    assert ref["noise_verdict"].startswith("undetermined: the finest band is not white")


def test_shot_noise_sigma_is_the_rms_not_the_median() -> None:
    """Poisson noise grows with the signal; the MSE averages its variance."""
    from luxar.gsplats.calibration.noise_floor import estimate_masked_noise

    clean = _spots((2.0, 6.0, 6.0)) * 50.0 + 2.0
    noisy = np.random.default_rng(4).poisson(clean).astype(np.float32)
    mask = clean > 7.0
    true_rms = float(np.sqrt(np.mean((noisy - clean)[mask] ** 2)))
    est = estimate_masked_noise(noisy, mask)
    assert est.sigma == pytest.approx(true_rms, rel=0.06)
    assert est.sigma_median < est.sigma


def test_quantised_data_does_not_read_as_noise_free() -> None:
    from luxar.gsplats.calibration.noise_floor import estimate_masked_noise

    rng = np.random.default_rng(6)
    v = (rng.random((8, 64, 64)) < 0.1).astype(np.float32)  # MAD of HH is 0
    est = estimate_masked_noise(v, np.ones(v.shape, bool))
    assert est.sigma > 0.0


def test_compare_reports_the_noise_reference(tmp_path: Path) -> None:
    """A model equal to the clean signal, scored against signal + noise."""
    from luxar.encoding import EncodingMode
    from luxar.gsplats.gsplat_data import GSplatData

    model = GSplatData(
        centers=np.array([[10.0, 16.0, 16.0], [10.0, 20.0, 12.0]], np.float32),
        amplitudes=np.array([1.0, 0.7], np.float32),
        cholesky_factors=np.tile([3.0, 0, 5.0, 0, 0, 5.0], (2, 1)).astype(np.float32),
        stats={"source_shape": [20, 32, 32], "image_min": 0.0},
    )
    store = tmp_path / "smooth.gsplats.zarr"
    model.save(store, encoding_mode=EncodingMode.PRECISION, include_fitting_info=True)
    clean = GSplatData.load(store).render_to_volume(shape=(20, 32, 32), device="cpu")
    noisy = clean + np.random.default_rng(3).normal(0.0, 0.08, clean.shape)
    ref = tmp_path / "noisy.npy"
    np.save(ref, noisy.astype(np.float32))

    output, record = _compare(store, ref, tmp_path / "n.json")
    assert "Noise ceiling:" in output and "Verdict:" in output
    assert record["noise_sigma"] == pytest.approx(0.08, rel=0.1)
    assert record["noise_ceiling_reliable"] is True
    assert record["noise_verdict"].startswith("at the noise floor")
    assert abs(record["foreground_psnr_vs_noise_ceiling_db"]) < 1.0
