"""Post-fit speck pruning (``min_extent``) shared by every fitter.

Splat relocation reseeds a moved splat as an isotropic Gaussian of
``relocation_reseed_sigma_vox`` (0.5 voxel by default). A share of those never
grow out of it and end the fit as sub-voxel "specks": a bright dot no wider
than a voxel, which renders as glitter and carries almost none of the
reconstruction. Pruning every splat whose geometric-mean SPATIAL sigma is below
``min_extent`` fitted-grid voxels removes them; the held-out change measured on
nine fits was within -0.07 dB (+0.40 dB on a Tribolium light-sheet stack). See
the fitting README for the table. The single-pass, tiled, content and
progressive fitters all use these definitions.

Guard: under the default ``--floor auto`` the pruned specks carry 0.5-0.75% of
the fitted brightness. After a high floor (``p95``) the same threshold removed
4.3-15%, i.e. real structure fitted small. So when pruning would remove more
than :data:`DEFAULT_MIN_EXTENT_MAX_MASS_SHARE` of the mass, the fit (tile, box)
keeps its splats and says so (``stats["min_extent_skipped"] = True``).

Mass here is an axis-aligned upper bound on the Gaussian integral up to a
constant: ``amplitude * prod(marginal spatial sigmas)``. It equals the integral
for axis-aligned splats.

The "spatial" axes are those with non-zero variance. A STACKED time/channel
axis (sigma 0 for every splat) is ignored, but in a native nD fit where time
is itself a fitted axis, its sigma (in time steps) counts in the geometric mean
like any other axis.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData

#: Default ``min_extent`` (fitted-grid voxels). Specks sit at sigma 0.45-0.55
#: under the default 0.5-voxel reseed, so 0.6 clears them without reaching real
#: sub-voxel-but-resolved structure (which the fit sizes >= ~0.7 voxel).
DEFAULT_MIN_EXTENT_VOX = 0.6

#: Pruning is skipped when it would remove more than this share of the fitted
#: brightness (config/YAML key ``min_extent_max_mass_share``).
DEFAULT_MIN_EXTENT_MAX_MASS_SHARE = 0.02

def validate_min_extent(min_extent: "float | None") -> float:
    """Normalize ``min_extent``: ``None``/``0`` disable (-> ``0.0``)."""
    if min_extent is None:
        return 0.0
    if isinstance(min_extent, bool):
        raise ValueError("min_extent must be a number of voxels, not a bool")
    value = float(min_extent)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"min_extent must be a finite number of voxels >= 0 (0 disables), "
            f"got {min_extent!r}"
        )
    return value


def validate_max_mass_share(share: "float | None") -> float:
    """Validate ``min_extent_max_mass_share`` (a fraction in ``(0, 1]``)."""
    if share is None:
        return DEFAULT_MIN_EXTENT_MAX_MASS_SHARE
    value = float(share)
    if isinstance(share, bool) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(
            f"min_extent_max_mass_share must be a fraction in (0, 1], got {share!r}"
        )
    return value


def validate_reseed_sigma(sigma: "float | None") -> "float | None":
    """Validate an explicit relocation reseed sigma (voxels); ``None`` keeps 0.5."""
    if sigma is None:
        return None
    if isinstance(sigma, bool):
        raise ValueError("relocation_reseed_sigma_vox must be a number, not a bool")
    value = float(sigma)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"relocation_reseed_sigma_vox must be a finite number of voxels > 0, "
            f"got {sigma!r}"
        )
    return value


def fitted_voxel_unit(
    ndim: int,
    *,
    voxel_size: "Sequence[float] | float | np.ndarray | None" = None,
    output_space: str = "real",
    downscale: "int | Sequence[int] | None" = None,
) -> np.ndarray:
    """Per-axis size of one FITTED-grid voxel in the result's output units.

    ``finalize_results`` rescales a downscaled fit back to input voxels and
    then, for ``output_space="real"``, to physical units; this undoes both, so
    a threshold in voxels means what the optimizer saw.
    """
    unit = np.ones(ndim, dtype=np.float64)
    if output_space == "real" and voxel_size is not None:
        unit *= np.broadcast_to(np.asarray(voxel_size, dtype=np.float64), (ndim,))
    if downscale is not None:
        unit *= np.broadcast_to(np.asarray(downscale, dtype=np.float64), (ndim,))
    return unit


def _spatial_sigmas(data: "GSplatData", unit: np.ndarray) -> np.ndarray:
    """Marginal sigmas in ``unit`` over the non-degenerate (spatial) axes."""
    from luxar.gsplats.utils.spatial_axes import spatial_axes_from_max_sigma

    sig = data.marginal_sigmas() / unit
    axes = spatial_axes_from_max_sigma(sig.max(axis=0))
    result: np.ndarray = np.clip(sig[:, axes], 1e-12, None)
    return result


def spatial_extent_vox(data: "GSplatData", unit: np.ndarray) -> np.ndarray:
    """Per-splat geometric-mean SPATIAL sigma in fitted voxels.

    :meth:`GSplatData.scale` on the voxel-unit view: a stacked time/channel
    axis (zero variance for every splat) is ignored rather than collapsing the
    mean to zero.
    """
    if data.n_splats == 0:
        return np.empty(0, dtype=np.float64)
    sig = _spatial_sigmas(data, unit)
    result: np.ndarray = np.exp(np.mean(np.log(sig), axis=1))
    return result


def integrated_mass(data: "GSplatData") -> np.ndarray:
    """Axis-aligned upper bound on each splat's Gaussian integral (up to a constant)."""
    if data.n_splats == 0:
        return np.empty(0, dtype=np.float64)
    sig = _spatial_sigmas(data, np.ones(data.ndim, dtype=np.float64))
    amps = np.asarray(data.amplitudes, dtype=np.float64)
    result: np.ndarray = amps * np.prod(sig, axis=1)
    return result


def min_extent_keep_mask(
    data: "GSplatData", min_extent_vox: float, unit: np.ndarray
) -> np.ndarray:
    """Boolean mask of splats whose spatial extent reaches ``min_extent_vox``."""
    if min_extent_vox <= 0 or data.n_splats == 0:
        return np.ones(data.n_splats, dtype=bool)
    result: np.ndarray = spatial_extent_vox(data, unit) >= min_extent_vox
    return result


def stamp_min_extent(
    stats: dict[str, Any],
    *,
    min_extent_vox: float,
    removed: int,
    removed_mass: float,
    input_mass: float,
    skipped: bool = False,
) -> None:
    """Record one pruning pass in ``stats`` (``skipped``: the guard kept them)."""
    stats["min_extent_skipped"] = bool(skipped)
    stats["min_extent_vox"] = float(min_extent_vox)
    stats["min_extent_removed"] = int(removed)
    stats["min_extent_input_mass"] = float(input_mass)
    stats["min_extent_removed_mass_share"] = (
        float(removed_mass / input_mass) if input_mass > 0 else 0.0
    )


def aggregate_min_extent_stats(
    sources: Sequence[dict[str, Any]], target: dict[str, Any]
) -> None:
    """Sum per-tile/box pruning records onto a merge root.

    Mass shares are combined through each source's input mass, so the merged
    share is the share of the WHOLE fit's pre-pruning brightness.
    ``min_extent_skipped`` is True when the guard kept any part's splats.
    """
    recorded = [s for s in sources if s.get("min_extent_removed") is not None]
    if not recorded:
        return
    removed = sum(int(s["min_extent_removed"]) for s in recorded)
    input_mass = sum(float(s.get("min_extent_input_mass") or 0.0) for s in recorded)
    removed_mass = sum(
        float(s.get("min_extent_removed_mass_share") or 0.0)
        * float(s.get("min_extent_input_mass") or 0.0)
        for s in recorded
    )
    thresholds = {s.get("min_extent_vox") for s in recorded}
    shared = next(iter(thresholds)) if len(thresholds) == 1 else None
    stamp_min_extent(
        target,
        min_extent_vox=float(shared) if shared is not None else 0.0,
        removed=removed,
        removed_mass=removed_mass,
        input_mass=input_mass,
        skipped=any(bool(s.get("min_extent_skipped")) for s in recorded),
    )
    if shared is None:
        target["min_extent_vox"] = None


def format_min_extent_line(stats: dict[str, Any]) -> Optional[str]:
    """One summary line for a pruning record, or ``None`` when absent."""
    removed = stats.get("min_extent_removed")
    if removed is None:
        return None
    share = float(stats.get("min_extent_removed_mass_share") or 0.0)
    threshold = stats.get("min_extent_vox")
    at = f" < {threshold:g} voxel" if threshold else ""
    kept = (
        " (the mass guard kept some parts' splats)"
        if stats.get("min_extent_skipped")
        else ""
    )
    return (
        f"Speck pruning (spatial sigma{at}): removed {int(removed):,} splats "
        f"carrying {100.0 * share:.2f}% of the fitted brightness{kept}"
    )


def skipped_note(removed: int, share: float, max_share: float) -> str:
    """The one-line note printed when the mass guard keeps the splats."""
    return (
        f"min-extent: would remove {removed:,} splats carrying {100.0 * share:.1f}% "
        f"of the brightness (>{100.0 * max_share:g}%); kept them — typical after a "
        "high --floor"
    )
