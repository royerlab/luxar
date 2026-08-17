"""A tiled fit scores the archive it actually ships.

Every tile scores itself, but tiles overlap and are Hann-apodized, so their
errors do not compose into the merged one — and the merged result is what gets
written. Before this, a tiled archive carried no ``psnr_db`` at all, which is
the one number a published dataset is expected to state.

The load-bearing test here is the coordinate-frame equivalence: with
``output_space="real"`` the merged splats live in physical coordinates and have
to be mapped back onto the tile grid before they can be rendered against it. An
inverse that is wrong per-axis produces a plausible-looking but badly wrong
number, so it is checked against the same fit scored with no conversion at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from luxar.gsplats.fit_tiled_gsplats import (
    _QUALITY_PEAK_VOLUMES,
    fit_tiled,
)

#: Anisotropic on purpose: an isotropic spacing would hide a per-axis error in
#: the Cholesky row scaling, which is applied row by row.
VOXEL_SIZE = (3.0, 1.0, 1.0)

_QUALITY_KEYS = ("psnr_db", "mse", "ssim", "foreground_psnr_db", "foreground_fraction")


@pytest.fixture(scope="module")
def volume() -> np.ndarray:
    """Small structured volume — big enough that tiling actually tiles."""
    shape = (24, 48, 48)
    zz, yy, xx = np.meshgrid(
        *(np.arange(s, dtype=np.float32) for s in shape), indexing="ij"
    )
    vol = np.zeros(shape, dtype=np.float32)
    for cz, cy, cx in ((7, 12, 12), (16, 34, 30), (12, 24, 24)):
        vol += np.exp(
            -(((zz - cz) / 2.0) ** 2 + ((yy - cy) / 3.0) ** 2 + ((xx - cx) / 3.0) ** 2)
        )
    return vol


def _fit(volume: np.ndarray, **kwargs: Any) -> Any:
    opts: dict[str, Any] = dict(
        tile_size=24,
        overlap=6,
        seeds=60,
        n_iters=25,
        device="cpu",
        verbose=False,
    )
    opts.update(kwargs)
    return fit_tiled(volume, **opts)


def test_a_tiled_fit_records_merged_quality(
    volume: np.ndarray, capsys: pytest.CaptureFixture
) -> None:
    """The regression: the merged archive used to carry no PSNR at all."""
    stats = _fit(volume).stats
    missing = [k for k in _QUALITY_KEYS if k not in stats]
    assert not missing, f"the merged tiled result lost {missing}"
    assert np.isfinite(stats["psnr_db"]) and stats["psnr_db"] > 0
    # The unscored notice is unconditional on its own branch, so nothing else
    # would catch a change that printed it next to a store carrying `psnr_db`.
    assert "No merged quality metrics" not in capsys.readouterr().out


def test_the_merged_score_survives_the_physical_coordinate_round_trip(
    volume: np.ndarray,
) -> None:
    """A real-space tiled fit must score the same as one that never converted.

    This is what catches a wrong inverse in ``_to_voxel_frame``: rendering
    physical centers on a voxel grid, or undoing the Cholesky scaling on the
    wrong axis, would collapse the PSNR rather than merely nudge it.
    """
    plain = _fit(volume).stats
    real = _fit(volume, voxel_size=VOXEL_SIZE, output_space="real").stats
    assert real["psnr_db"] == pytest.approx(plain["psnr_db"], abs=0.5), (
        f"physical-coordinate scoring diverged: {real['psnr_db']:.2f} dB vs "
        f"{plain['psnr_db']:.2f} dB — the voxel-frame inverse is wrong"
    )
    assert real["foreground_fraction"] == pytest.approx(
        plain["foreground_fraction"], abs=1e-6
    )


def test_the_returned_splats_stay_in_physical_coordinates(
    volume: np.ndarray,
) -> None:
    """Scoring copies the mixture; it must not rewrite what the caller gets."""
    real = _fit(volume, voxel_size=VOXEL_SIZE, output_space="real")
    plain = _fit(volume)
    # z is scaled x3, so the physical extent must clearly exceed the voxel one.
    assert real.centers[:, 0].max() > plain.centers[:, 0].max() * 2


def test_the_memory_budget_skips_rather_than_thrashes(
    volume: np.ndarray, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Above the budget the score is skipped — and says so even when quiet.

    ``verbose=False`` here on purpose: a scripted fit is exactly where an
    unscored archive would otherwise appear with no explanation anywhere, since
    nothing about the skip reaches the store.
    """
    monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", "0.0000001")
    stats = _fit(volume).stats
    assert "psnr_db" not in stats
    out = capsys.readouterr().out
    assert "Merged quality metrics skipped" in out
    assert "gsplat compare" in out


def test_the_peak_estimate_covers_ssims_intermediates() -> None:
    """The budget must count the SSIM peak, not just the two volumes it scores.

    SSIM holds ~6 convolution intermediates live on top of the reconstruction
    and the reference, and its own tiled fallback only engages on CUDA. Since
    scoring runs before the archive is written, an under-count is not a missing
    metric — it is a thrash or an OOM kill that loses the finished fit.
    """
    from luxar.gsplats.metrics import _SSIM_PEAK_TENSOR_COUNT

    assert _QUALITY_PEAK_VOLUMES >= _SSIM_PEAK_TENSOR_COUNT


def test_an_unparseable_budget_override_does_not_lose_the_fit(
    volume: np.ndarray, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A typo in the override falls back to the default instead of raising.

    This runs after every tile has been fitted, so a bad environment variable
    must not be what takes the run down.
    """
    monkeypatch.setenv("LUXAR_TILED_QUALITY_MAX_GB", "24 GiB")
    stats = _fit(volume).stats
    assert np.isfinite(stats["psnr_db"])
    assert "LUXAR_TILED_QUALITY_MAX_GB" in capsys.readouterr().out


def test_a_partition_is_left_alone_but_says_so(
    volume: np.ndarray, capsys: pytest.CaptureFixture
) -> None:
    """A partition merge has no fit-stats dict to stamp, so it is not scored.

    Asserted rather than assumed: the scoring call sits right after the merge,
    and a partition returns a node with no ``stats`` dict to write into. The
    absence is announced, so the missing PSNR is a stated limitation rather than
    a hole the user has to discover in the store. ``verbose=False`` is
    deliberate: a quiet scripted fit is exactly where the unexplained gap would
    otherwise appear, since nothing about the omission reaches the store.
    """
    node = _fit(volume, partition=True, verbose=False)
    assert not hasattr(node, "stats") or "psnr_db" not in getattr(node, "stats", {})
    out = capsys.readouterr().out
    assert "No merged quality metrics" in out
    assert "gsplat compare" in out
    # `compare` cannot load a `kind=partition` store directly
    # (``GSplatData.load`` raises "not matrix-shaped"), so a recourse that omits
    # the flatten step tracebacks on the tiled default.
    assert "gsplat flatten" in out
