"""A fit's own reported metrics must be scored on the basis it reconstructs (#1173).

Deliberately NOT marked `slow`, despite running real CPU fits (~6.5 s + ~1.8 s).
CI runs `-m 'not slow'`, and these are the only coverage of the primary site, so
marking them put the one test that justifies the change outside the gate: with the
marker on, replacing `reference_on_fit_basis` in `results.py` with the identity
left the whole non-slow fitting suite green. Keep them collected.

The site that mattered most: `psnr_db`/`ssim`/`mse` are persisted into the store
and printed by `gsplat info`, so a floor-penalised score is not a cosmetic
problem — it is a published number that inverts comparisons between fits.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from luxar.gsplats import fit_gaussian_splats  # noqa: E402


def _volume_with_pedestal(pedestal: float) -> np.ndarray:
    """A few bright blobs sitting on a constant pedestal."""
    v = np.full((24, 24, 24), pedestal, dtype=np.float32)
    for z, y, x in ((6, 6, 6), (12, 14, 10), (17, 8, 16)):
        v[z - 2 : z + 3, y - 2 : y + 3, x - 2 : x + 3] += 2000.0
    return v


def _fit_psnr(volume: np.ndarray, floor) -> float:
    res = fit_gaussian_splats(
        volume,
        seeds=200,
        floor=floor,
        n_iters=40,
        device="cpu",
        verbose=False,
        enable_dynamic_ops=False,
        cull_retention=1.0,
    )
    stats = res.stats or {}
    if "psnr_db" not in stats:
        pytest.skip("this build reports no quality metrics for the fit path")
    return float(stats["psnr_db"])


def test_a_floored_fit_is_not_penalised_for_the_pedestal_it_removed() -> None:
    """The inversion this fix exists to remove.

    Scored against a raw reference, a fit that correctly refuses to reconstruct a
    large pedestal is charged for it, so raising the floor *lowers* the reported
    PSNR on identical data. On the fit's own basis that coupling is gone: two
    fits of the same signal over different pedestals score comparably.
    """
    small = _fit_psnr(_volume_with_pedestal(50.0), 50.0)
    large = _fit_psnr(_volume_with_pedestal(4000.0), 4000.0)

    # Both fits face the same signal above their own floor, so their scores must
    # be in the same ballpark. Pre-fix the 4000-count pedestal was counted as
    # error and dragged `large` down by tens of dB.
    assert abs(small - large) < 12.0, (
        f"pedestal still leaking into the score: PSNR {small:.1f} dB over a "
        f"50-count pedestal vs {large:.1f} dB over a 4000-count one"
    )


def test_reported_metrics_use_the_recorded_image_min() -> None:
    """The score must be consistent with the level the store advertises, so a
    reader can reproduce it."""
    volume = _volume_with_pedestal(600.0)
    res = fit_gaussian_splats(
        volume,
        seeds=200,
        floor=600.0,
        n_iters=40,
        device="cpu",
        verbose=False,
        enable_dynamic_ops=False,
        cull_retention=1.0,
    )
    stats = res.stats or {}
    assert stats.get("image_min") == pytest.approx(600.0)
    if "psnr_db" not in stats:
        pytest.skip("this build reports no quality metrics for the fit path")
    # A raw-reference score on this volume would be dominated by the 600-count
    # pedestal: SNR ~ 20*log10(range/600) caps it near 20 dB. Clearing that bar
    # shows the reference was shifted.
    assert float(stats["psnr_db"]) > 22.0
