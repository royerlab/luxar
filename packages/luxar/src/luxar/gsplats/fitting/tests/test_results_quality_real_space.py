"""A fit in physical coordinates still gets scored.

``output_space="real"`` is the DEFAULT, so any fit handed a ``voxel_size``
returned physical coordinates — and the round-trip metrics were skipped
wholesale, because the splats no longer sat on ``config.V``'s grid. Ten demos
pass a ``voxel_size``, so ten cached archives shipped with no ``psnr_db`` and,
once it existed, no ``foreground_psnr_db`` either: precisely the numbers a
published dataset is expected to state.

The conversion is a pure per-axis scale of centers and Cholesky rows, with
amplitudes untouched, so the same mixture can be scored on the voxel grid. These
tests pin that it IS scored, and that the score does not depend on which
coordinate system the caller asked the result back in.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats import fit_gaussian_splats

#: Anisotropic on purpose — an isotropic voxel_size would hide a per-axis
#: scaling bug in the Cholesky rows, which are scaled row-by-row.
VOXEL_SIZE = (4.0, 1.0, 1.0)

_QUALITY_KEYS = (
    "psnr_db",
    "mse",
    "ssim",
    "foreground_psnr_db",
    "foreground_threshold",
    "foreground_fraction",
)


def _blobs(shape: tuple[int, int, int] = (16, 24, 24)) -> np.ndarray:
    """A small volume with structure and background, so PSNR is meaningful."""
    zz, yy, xx = np.meshgrid(
        *(np.arange(s, dtype=np.float32) for s in shape), indexing="ij"
    )
    vol = np.zeros(shape, dtype=np.float32)
    for cz, cy, cx in ((5, 8, 8), (10, 16, 15)):
        vol += np.exp(
            -(((zz - cz) / 1.5) ** 2 + ((yy - cy) / 2.5) ** 2 + ((xx - cx) / 2.5) ** 2)
        )
    return vol


def _fit(volume: np.ndarray, **kwargs: object):
    return fit_gaussian_splats(
        volume, seeds=40, n_iters=30, device="cpu", verbose=False, **kwargs
    )


def test_a_real_space_fit_records_its_quality_metrics() -> None:
    """The regression itself: voxel_size used to mean "no metrics at all"."""
    stats = _fit(_blobs(), voxel_size=np.array(VOXEL_SIZE)).stats
    missing = [k for k in _QUALITY_KEYS if k not in stats]
    assert not missing, (
        f"a fit with voxel_size lost {missing}. The metrics must be scored on "
        "the pre-conversion (voxel-space) arrays, not skipped."
    )
    assert np.isfinite(stats["psnr_db"]) and stats["psnr_db"] > 0


def test_the_score_does_not_depend_on_the_output_coordinate_space() -> None:
    """Same fit, same grid, same number — the conversion is only a rescale.

    Were the metrics ever computed against the PHYSICAL centers on the voxel
    grid, this would diverge wildly rather than agree to the optimiser's own
    determinism.
    """
    volume = _blobs()
    real = _fit(volume, voxel_size=np.array(VOXEL_SIZE)).stats
    voxel = _fit(volume, voxel_size=np.array(VOXEL_SIZE), output_space="voxel").stats
    for key in ("psnr_db", "foreground_psnr_db", "foreground_fraction"):
        assert real[key] == pytest.approx(voxel[key], rel=1e-4, abs=1e-6), (
            f"{key} differs between output spaces: {real[key]} vs {voxel[key]}"
        )


def test_a_fit_without_a_voxel_size_is_unaffected() -> None:
    """The path that always worked keeps working (no voxel_size, no conversion)."""
    stats = _fit(_blobs()).stats
    assert all(k in stats for k in _QUALITY_KEYS)


def test_the_physical_result_is_still_returned_in_physical_coordinates() -> None:
    """Scoring a voxel-space copy must not leak back into what the caller gets."""
    volume = _blobs()
    real = _fit(volume, voxel_size=np.array(VOXEL_SIZE))
    voxel = _fit(volume, voxel_size=np.array(VOXEL_SIZE), output_space="voxel")
    # The z axis is scaled x4, so the physical extent must exceed the voxel one.
    assert real.centers[:, 0].max() > voxel.centers[:, 0].max() * 2, (
        "the returned centers look like voxel indices — the physical conversion "
        "was lost when the scoring copy was introduced"
    )
