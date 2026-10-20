"""Spot (punctum) fidelity: a report-only check on the calibrated K*.

The held-out PSNR that picks K* averages over every masked voxel. On sparse,
diffraction-limited puncta (FISH spots) the puncta centres are ~0.01% of those
voxels, so background and diffuse signal can dominate the curve: it comes out
flat and K* lands far below what the spots need.

This module detects spots once on the floor-subtracted reference and scores
each grid point by the median ratio of the rendered peak to the reference
peak at those spots. ``cal`` records that curve and warns when K* renders the
spots clearly worse than a larger K did; it never moves K*.

The warning compares renders with renders. The reference peak carries its own
noise and a detected maximum is biased up, so even a perfect render of a dim
spot scores below 1 against it (0.61 at a 2-count Poisson peak, 0.83 at 3).
That bias is the same for every grid point, so the ratio at K* is normalised
by the best ratio the sweep reached and the bias cancels.

Measured with ``cal`` on EASI-FISH level-0 crops (64x192x192), K* reached
27%, 78% and 93% of the sweep's best spot fidelity on the three sparse crops
(104-310 spots) and 99% on a dense FISH crop. On an Allen hiPSC DNA crop the
494 chromatin maxima reached 91%, so the warning can also fire on textured
nuclei; it is advice about small bright structures, not a verdict.

The detector is a local maximum (5-voxel window) of the volume blurred with
``sigma = 1`` voxel that rises more than six robust sigmas above the median.
Both peak values are taken on the blurred volumes, the rendered one as its
3-voxel neighbourhood maximum, so a sub-voxel offset does not count as a miss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: Gaussian blur (voxels) applied before detecting and before comparing peaks.
SPOT_BLUR_SIGMA = 1.0
#: Detection threshold: median + this many robust sigmas of the blurred volume.
SPOT_N_SIGMA = 6.0
#: Fewer detected spots than this and the volume is not treated as punctate.
SPOT_MIN_COUNT = 20
#: Warn when K* reaches less than this share of the sweep's best spot fidelity.
#: The three sparse EASI-FISH crops sat at 0.27-0.93, a dense FISH crop at 0.99.
SPOT_UNDERFIT_SHARE = 0.95
#: The K the warning suggests is the first grid point within this share of the best.
SPOT_TARGET_SHARE = 0.98
#: Half-width of the patch a rendered peak is measured in: the blur's support
#: (scipy's ``truncate=4``) plus the 3-voxel maximum window.
_PATCH_HALF = int(4.0 * SPOT_BLUR_SIGMA + 0.5) + 1


@dataclass(frozen=True)
class SpotReference:
    """Detected spot positions and their blurred reference peaks."""

    indices: tuple[np.ndarray, ...]
    peaks: np.ndarray

    @property
    def n(self) -> int:
        """Number of detected spots."""
        return int(self.peaks.size)


def _robust_sigma(values: np.ndarray) -> tuple[float, float]:
    """``(median, 1.4826 * MAD)`` of ``values``."""
    med = float(np.median(values))
    return med, 1.4826 * float(np.median(np.abs(values - med)))


def detect_spots(V: np.ndarray) -> Optional[SpotReference]:
    """Detect bright punctate maxima in a floor-subtracted volume.

    Returns ``None`` when the volume has no measurable spread to set a
    threshold from. After a floor clip most voxels of a sparse volume can be
    exactly zero, which makes the MAD zero; the spread is then measured over
    the non-zero voxels instead.
    """
    from scipy import ndimage as ndi

    blurred = ndi.gaussian_filter(np.asarray(V, dtype=np.float32), SPOT_BLUR_SIGMA)
    med, sigma = _robust_sigma(blurred)
    if not sigma > 0:
        positive = blurred[blurred > 0]
        if positive.size == 0:
            return None
        med, sigma = _robust_sigma(positive)
        if not sigma > 0:
            return None
    maxima = (blurred == ndi.maximum_filter(blurred, size=5)) & (
        blurred > med + SPOT_N_SIGMA * sigma
    )
    for axis, size in enumerate(maxima.shape):
        if size > 4:  # a maximum on the border has an unseen side
            index = [slice(None)] * maxima.ndim
            for edge in (slice(0, 2), slice(size - 2, size)):
                index[axis] = edge
                maxima[tuple(index)] = False
    indices = np.nonzero(maxima)
    return SpotReference(indices=indices, peaks=blurred[indices])


def _reflect(index: np.ndarray, size: int) -> np.ndarray:
    """scipy's ``mode="reflect"`` for indices at most one size out of range."""
    index = np.where(index < 0, -index - 1, index)
    return np.where(index >= size, 2 * size - index - 1, index)


def _patch_indices(spots: SpotReference, shape: tuple[int, ...]) -> tuple:
    """Broadcast index arrays gathering a reflect-padded patch around each spot."""
    offsets = np.arange(-_PATCH_HALF, _PATCH_HALF + 1)
    ndim = len(shape)
    out = []
    for axis, size in enumerate(shape):
        idx = _reflect(spots.indices[axis][:, None] + offsets[None, :], size)
        out.append(
            idx.reshape(
                (-1,) + (1,) * axis + (offsets.size,) + (1,) * (ndim - axis - 1)
            )
        )
    return tuple(out)


def spot_peak_ratio(rendered: Any, spots: SpotReference) -> float:
    """Median of rendered over reference peak at the detected spots (``nan`` if none).

    ``rendered`` is a numpy array or a torch tensor on any device. Only a small
    patch around each spot is gathered (on the tensor's device) and blurred, so
    the cost scales with the spot count, not the volume; the values equal a
    whole-volume blur because each patch carries the blur's full support with
    the same reflected border.
    """
    if spots.n == 0:
        return float("nan")
    from scipy import ndimage as ndi

    shape = tuple(int(n) for n in rendered.shape)
    if min(shape) <= _PATCH_HALF:  # a single reflection would not cover it
        blurred = ndi.gaussian_filter(_to_numpy(rendered), SPOT_BLUR_SIGMA)
        local_max = ndi.maximum_filter(blurred, size=3)[spots.indices]
    else:
        index = _patch_indices(spots, shape)
        if not isinstance(rendered, np.ndarray):
            import torch

            index = tuple(torch.from_numpy(i).to(rendered.device) for i in index)
        patches = ndi.gaussian_filter(
            _to_numpy(rendered[index]),
            (0.0,) + (SPOT_BLUR_SIGMA,) * len(shape),
        )
        centre = (slice(None),) + (slice(_PATCH_HALF - 1, _PATCH_HALF + 2),) * len(
            shape
        )
        local_max = patches[centre].reshape(spots.n, -1).max(axis=1)
    ratio = local_max / np.maximum(spots.peaks, np.finfo(np.float32).tiny)
    return float(np.median(ratio))


def _to_numpy(values: Any) -> np.ndarray:
    """A float32 numpy copy of an array or a (possibly device) torch tensor."""
    if not isinstance(values, np.ndarray):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def spot_fidelity_warning(
    k_values: Sequence[int],
    ratios: Sequence[float],
    k_star: int,
    n_spots: Optional[int],
    k_skipped: Sequence[int] = (),
    *,
    k_effective: Optional[Sequence[int]] = None,
    crop_scoped: bool = False,
    whole_volume: Optional[Mapping[str, Any]] = None,
    density: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """The ``cal`` warning when K* under-renders detected puncta, else ``None``.

    Silent unless at least :data:`SPOT_MIN_COUNT` spots were detected, a larger
    K rendered them better, and K* reached less than
    :data:`SPOT_UNDERFIT_SHARE` of the best ratio of the sweep (the reference
    noise bias cancels in that share). The suggested K is the first grid point
    within :data:`SPOT_TARGET_SHARE` of the best; when that is the largest K
    fitted, the curve may still be rising, so the message also says how to
    measure further (``--no-early-stop`` when early stopping skipped points in
    ``k_skipped``, else ``--k-max``).

    ``crop_scoped``: the sweep ran on a ``--region`` / ``--auto-region`` crop, so
    its K are crop budgets while ``fit --seeds`` takes a whole-volume one. The
    suggestion is then restated with the planner's own transfer: with
    ``whole_volume`` (``cal.json``'s ``whole_volume_seeds``) as
    ``seeds * eff(K) / k_star_crop_seeds``, ``eff`` being ``k_effective`` (the
    post-cull counts the planner transfers); otherwise as the
    ``fit --tiling content`` density flags from ``density``
    (``splat_density``).
    """
    if not n_spots or n_spots < SPOT_MIN_COUNT or len(ratios) != len(k_values):
        return None
    by_k = {
        int(k): float(r)
        for k, r in zip(k_values, ratios, strict=True)
        if r is not None and math.isfinite(float(r)) and float(r) >= 0
    }
    at_star = by_k.get(int(k_star))
    if at_star is None:
        return None
    best_k = max(by_k, key=lambda k: by_k[k])
    best = by_k[best_k]
    if best <= 0:
        return None
    share = at_star / best
    if best_k <= int(k_star) or share >= SPOT_UNDERFIT_SHARE:
        return None
    k_spot = min(k for k, r in by_k.items() if r >= SPOT_TARGET_SHARE * best)
    eff = dict(zip((int(k) for k in k_values), k_effective or k_values, strict=True))
    advice = _scoped_advice(
        k_spot, int(eff[k_spot]), crop_scoped, whole_volume, density
    )
    if k_spot == max(by_k):
        rerun = (
            f"rerun with --no-early-stop (early stopping skipped K up to "
            f"{max(k_skipped):,})"
            if k_skipped
            else "extend --k-max"
        )
        advice += f"; the spot curve was still rising there, so {rerun} to see where it levels off"
    return (
        f"small bright structures: K*={int(k_star):,} renders the {n_spots:,} "
        f"detected spots (local maxima) at {share:.0%} of the best spot fidelity "
        f"the sweep reached (K={best_k:,}). Held-out PSNR weighs every voxel, so "
        "on sparse puncta or foci the background can dominate it and K* can "
        "under-seed them; if they matter, " + advice + "."
    )


def _scoped_advice(
    k_spot: int,
    eff_spot: int,
    crop_scoped: bool,
    whole_volume: Optional[Mapping[str, Any]],
    density: Optional[Mapping[str, Any]],
) -> str:
    """The "fit at least K" advice, restated for the whole volume when scoped."""
    if not crop_scoped:
        return f"fit at least --seeds {k_spot:,}"
    if whole_volume and whole_volume.get("k_star_crop_seeds"):
        whole = int(
            round(
                float(whole_volume["seeds"])
                * eff_spot
                / float(whole_volume["k_star_crop_seeds"])
            )
        )
        return (
            f"fit at least --seeds {whole:,} for the whole volume (crop K={k_spot:,} "
            "restated with the same density transfer as K*)"
        )
    flags = f"--k-star-ref {eff_spot:,}".replace(",", "")
    if density and density.get("n_features_reference"):
        flags += f" --n-features-ref {int(density['n_features_reference'])}"
        threshold = density.get("feature_threshold")
        if threshold is not None and math.isfinite(float(threshold)):
            flags += f" --feature-threshold {float(threshold):g}"
    return (
        f"crop K={k_spot:,} is a crop budget: transfer it with "
        f"`fit --tiling content {flags}`, not `fit --seeds`"
    )
