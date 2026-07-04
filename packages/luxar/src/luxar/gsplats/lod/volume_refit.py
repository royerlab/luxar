"""Volume re-fit of a coarse LOD level (``refine="volume"``).

The third quality rung for substitutive levels, above the moment-matched merge
and the mixture-space ``refine="l2"`` pass: warm-start a **full Gaussian-splat
fit against the source volume** from the merge output. Unlike ``l2`` — whose
target is the *fine mixture* and therefore inherits the fine fit's own error —
this optimizes the true render-fidelity objective at the coarse budget.
Benchmarked on real microscopy (skimage cells3d nuclei): +5–6 dB
full-res and +10–12 dB at viewing scale over the merge, with unchanged splat
count and lower cross-level drift than a cold fit.

This module is a thin orchestration layer at ``GSplatData`` altitude: the heavy
lifting (rasterizer, Adam, schedulers) is entirely
:func:`~luxar.gsplats.fit_gsplats.fit_gaussian_splats` with a ``GSplatData``
warm-start seed. It deliberately does NOT live in ``_substitutive/`` — that
subpackage's contract is silent tensor kernels that know nothing about
``GSplatData``.

Safety: the returned splats are **never worse than the seed** — both the seed
and the re-fit candidate are rendered to the volume's grid and the lower-MSE
one wins. Two additional guards keep the ladder coherent:

- **Mass pinning** (``conserve_mass``, default on): the re-fit's amplitudes are
  rescaled so its rendered DC equals the seed's — the seed's mass was already
  pinned to the fine chain's by the substitutive ``conserve_mass`` step, so
  without this the (volume-accurate) re-fit reintroduces the cross-level
  brightness pop that step exists to prevent.
- **Frame checks** (both directions): a seed whose center bounding box falls
  clearly outside the volume's voxel index range (an ENLARGED physical frame,
  e.g. ``fit --voxel-size 4`` or ``gsplat transform --scale``) skips the re-fit
  up front; a SHRUNK frame (physical units below 1 per voxel — the common
  sub-micron microscopy case) fits inside that box, so it is caught *after*
  the fit by the relocation check: a fit that wholesale moved/rescaled the
  splats is a frame mismatch, and the seed is returned with a warning. In
  either case a voxel-frame re-fit would have *won* the MSE guard while being
  misplaced relative to the rest of the ladder.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from luxar.gsplats.gsplat_data import GSplatData

__all__ = ["VolumeRefitConfig", "volume_refine_splats"]


@dataclass(frozen=True)
class VolumeRefitConfig:
    """Knobs for the volume re-fit of one coarse level.

    Only ``iters`` and ``conserve_mass`` are user-exposed (via
    ``--refine-iters`` and the ladder-wide ``--conserve-mass`` flag); the rest
    are fixed operating constants, not a tuning surface. The benchmark showed
    the warm-started fit near-converged by 150–300 iterations.
    """

    iters: int = 300
    #: Adam learning rate — ``fit_gaussian_splats``' default, which the
    #: benchmark used unchanged.
    lr: float = 0.01
    #: Loss plateau patience before the fit stops early (warm starts sit close
    #: to a minimum, so a short fuse saves most of the budget on easy levels).
    early_stop_patience: int = 50
    #: Render both seed and candidate to the volume grid and keep the lower-MSE
    #: one. Disable only in tests probing the raw fit path.
    never_worse: bool = True
    #: Rescale the re-fit's amplitudes so its rendered DC equals the seed's
    #: (whose mass the substitutive ``conserve_mass`` step already pinned to
    #: the fine chain's). Keeps brightness constant across LOD switches; the
    #: re-fit otherwise tracks the volume's true DC, which the finest level may
    #: under-explain — a visible pop. Follows the ladder's ``conserve_mass``.
    conserve_mass: bool = True
    #: Frame-mismatch heuristic: skip the re-fit (returning the seed) when more
    #: than this fraction of seed centers lie outside the volume's voxel index
    #: range, padded by this fraction of each extent. Catches physical-unit or
    #: transform-scaled coordinate frames that the MSE guard cannot.
    frame_tolerance: float = 0.5


def _render(data: GSplatData, volume: np.ndarray, device: Optional[str]) -> np.ndarray:
    return np.asarray(
        data.render_to_volume(shape=volume.shape, device=device), dtype=np.float32
    )


def _mse(rendered: np.ndarray, volume: np.ndarray) -> float:
    return float(np.mean((rendered - volume) ** 2))


def _frame_mismatch(seed: GSplatData, volume: np.ndarray, tolerance: float) -> bool:
    """True when the seed's centers are clearly not in the volume's voxel frame.

    Catches the ENLARGED/translated direction (centers escaping the padded
    voxel index box). A SHRUNK frame (physical units below 1 per voxel, e.g.
    ``fit --voxel-size 0.25``) fits inside the box and is caught after the fit
    by :func:`_relocated` instead — the two checks are complementary.
    """
    if seed.n_splats == 0:
        return False
    centers = np.asarray(seed.centers, dtype=np.float64)
    shape = np.asarray(volume.shape, dtype=np.float64)
    pad = tolerance * shape
    inside = np.all((centers >= -pad) & (centers <= shape + pad), axis=1)
    return bool(np.mean(inside) < (1.0 - tolerance))


def _median_splat_sigma(data: GSplatData) -> float:
    """Median per-splat size (mean |Cholesky diagonal| per splat) — the scale a
    splat can legitimately move within (its own footprint)."""
    chol = np.asarray(data.cholesky_factors, dtype=np.float64)
    ndim = np.asarray(data.centers).shape[1]
    diag_idx = [i * (i + 3) // 2 for i in range(ndim)]
    return float(np.median(np.abs(chol[:, diag_idx]).mean(axis=1)))


def _relocated(seed: GSplatData, refit: GSplatData, volume: np.ndarray) -> bool:
    """True when the fit wholesale RELOCATED the splats — the signature of a
    coordinate-frame mismatch the bbox check cannot see (a shrunk, rotated, or
    axis-permuted frame sits inside the voxel box; the fit drags the splats to
    the voxel-frame signal, and the relocated result *wins* the MSE guard
    while being misplaced relative to the rest of the ladder).

    The fit is identity-preserving (no cull / dynamic ops / re-sort), so rows
    correspond 1:1 and PER-SPLAT displacement is the direct measure: a
    warm-started re-fit on a matching frame moves each splat a little (the
    benchmark measured mean drift ~1.4 % of the extent), while any frame
    mismatch — shrink, shift, rotation, mirror, axis swap — moves the typical
    splat by a large fraction of the cloud's own spread. (Aggregate moments
    such as center-of-mass + RMS are provably blind to isometries about the
    center of mass and to amplitude-only changes, so they are NOT used.)

    The threshold grants slack for the seed's spread, the volume scale, and
    each splat's own footprint (a coarse splat correcting within its own σ is
    optimization, not relocation).
    """
    if seed.n_splats == 0 or refit.n_splats == 0 or refit.n_splats != seed.n_splats:
        return False
    c_s = np.asarray(seed.centers, dtype=np.float64)
    c_r = np.asarray(refit.centers, dtype=np.float64)
    disp = np.linalg.norm(c_r - c_s, axis=1)
    med_disp = float(np.median(disp))
    if not np.isfinite(med_disp):
        return True  # a diverged fit (NaN/Inf centers) is never a valid level
    com = c_s.mean(axis=0)
    rms = float(np.sqrt(((c_s - com) ** 2).sum(axis=1).mean()))
    diag = float(np.linalg.norm(np.asarray(volume.shape, dtype=np.float64)))
    allowed = max(0.5 * rms, 0.05 * diag, _median_splat_sigma(seed))
    return med_disp > allowed


def volume_refine_splats(
    seed: GSplatData,
    volume: np.ndarray,
    *,
    config: VolumeRefitConfig,
    device: Optional[str] = None,
) -> Tuple[GSplatData, Dict[str, Any]]:
    """Warm-start re-fit ``seed`` against ``volume``; keep whichever is closer.

    Parameters
    ----------
    seed : GSplatData
        One coarse level's merge output (a flat splat set). Its centers must be
        in the volume's voxel coordinate frame — the frame ``fit_gaussian_splats``
        emits, so any level derived from a fit of this volume qualifies. A seed
        whose bounding box clearly disagrees with that frame is returned
        untouched (see ``VolumeRefitConfig.frame_tolerance``).
    volume : np.ndarray
        The source volume, full resolution (fitting a blurred/downscaled proxy
        was benchmarked and rejected — it discards positional detail the merge
        seed inherits from the sharp fine fit).
    config : VolumeRefitConfig
        Operating constants; ``config.iters`` is the one quality/time knob and
        ``config.conserve_mass`` follows the ladder-wide setting.
    device : str, optional
        Torch device for the fit and the guard renders (``None`` = auto).

    Returns
    -------
    (GSplatData, dict)
        The refined level (or the untouched ``seed`` when it renders closer to
        the volume, or on a frame mismatch) and a flat, JSON-safe stats dict:
        ``mse_seed``, ``mse_refit``, ``improved``, ``seed_won``,
        ``mass_pinned``, ``mass_scale``, ``frame_mismatch``, ``n_seed``,
        ``n_refit``, ``iters``, ``wall_s``.
    """
    # Lazy import: fit_gsplats pulls in torch/fitting stacks; keep module
    # import cheap for consumers that never refine.
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    t0 = time.time()
    stats: Dict[str, Any] = {
        "n_seed": int(seed.n_splats),
        "n_refit": int(seed.n_splats),
        "iters": int(config.iters),
        "frame_mismatch": False,
        "mass_pinned": False,
        "mass_scale": 1.0,
        "improved": False,
        "seed_won": True,
    }
    if _frame_mismatch(seed, volume, config.frame_tolerance):
        # A voxel-frame re-fit of a non-voxel-frame seed would WIN the MSE
        # guard while landing in the wrong place relative to the ladder's
        # other levels — refuse rather than corrupt.
        warnings.warn(
            "volume_refine_splats: the seed's center bounding box lies "
            "outside the volume's voxel index range — the splats look like a "
            "different coordinate frame (physical units / transform-scaled?). "
            "Skipping the re-fit for this level; pass a volume in the splats' "
            "frame to enable it.",
            RuntimeWarning,
            stacklevel=2,
        )
        stats["frame_mismatch"] = True
        stats["wall_s"] = float(time.time() - t0)
        return seed, stats

    refit = fit_gaussian_splats(
        volume,
        seeds=seed,
        n_iters=config.iters,
        lr=config.lr,
        early_stop_patience=config.early_stop_patience,
        device=device,
        verbose=False,
        # Render/guard consistency: evaluate the candidate with the SAME
        # Gaussian truncation the seed (and the rest of the ladder) uses.
        truncate=seed.truncation_radius,
        # Identity-preserving: no cull and no splat relocation/birth/death, so
        # the output rows correspond 1:1 to the seed's and colors carry over.
        cull_retention=1.0,
        enable_dynamic_ops=False,
        sort_splats_enabled=False,
    )
    # Rebuild with the seed's colors (the fit never produces colors) and its
    # truncation_radius (the ladder's — NOT the fit default), keeping the
    # level a drop-in replacement.
    refit = GSplatData(
        centers=refit.centers,
        amplitudes=refit.amplitudes,
        cholesky_factors=refit.cholesky_factors,
        colors=(
            seed.colors.copy()
            if seed.colors is not None and refit.n_splats == seed.n_splats
            else None
        ),
        truncation_radius=seed.truncation_radius,
    )
    stats["n_refit"] = int(refit.n_splats)

    if _relocated(seed, refit, volume):
        # The fit dragged the splats wholesale to a different place/scale/
        # orientation — the frame-mismatch signature for the shrunk/rotated/
        # axis-swapped frames that fit inside the bbox check (the relocated
        # result would WIN the MSE guard while being misplaced relative to the
        # ladder's other levels).
        warnings.warn(
            "volume_refine_splats: the re-fit relocated the splats wholesale "
            "(the median per-splat displacement exceeds the seed's own "
            "spread/footprint) — the seed's coordinate frame does not match "
            "the volume's voxel frame. Discarding the re-fit and keeping the "
            "seed; pass a volume in the splats' frame to enable it.",
            RuntimeWarning,
            stacklevel=2,
        )
        stats["frame_mismatch"] = True
        stats["n_refit"] = int(seed.n_splats)
        stats["wall_s"] = float(time.time() - t0)
        return seed, stats

    r_seed = _render(seed, volume, device)
    r_refit = _render(refit, volume, device)

    if config.conserve_mass:
        # Pin the re-fit's rendered DC to the seed's (itself pinned to the fine
        # chain's by the substitutive conserve_mass step): the free fit tracks
        # the VOLUME's DC, which the finest level may under-explain — stored
        # unpinned, that renders as a brightness pop at the LOD switch
        # (measured ~14 % on the benchmark volume). Rendering is linear in
        # amplitudes, so one scalar on both is exact.
        dc_seed, dc_refit = float(r_seed.sum()), float(r_refit.sum())
        if dc_refit > 0.0 and dc_seed > 0.0:
            scale = dc_seed / dc_refit
            refit = GSplatData(
                centers=refit.centers,
                amplitudes=(refit.amplitudes * scale).astype(np.float32),
                cholesky_factors=refit.cholesky_factors,
                colors=refit.colors,
                truncation_radius=refit.truncation_radius,
            )
            r_refit = r_refit * scale
            stats["mass_pinned"] = True
            stats["mass_scale"] = float(scale)

    if config.never_worse:
        mse_seed = _mse(r_seed, volume)
        mse_refit = _mse(r_refit, volume)
        stats["mse_seed"] = mse_seed
        stats["mse_refit"] = mse_refit
        stats["improved"] = bool(mse_refit < mse_seed)
        stats["seed_won"] = bool(mse_refit >= mse_seed)
        result = refit if mse_refit < mse_seed else seed
        if stats["seed_won"]:
            # The STORED artifact is the untouched seed — pinning happened
            # only to the discarded candidate; the stats must describe what
            # was kept.
            stats["mass_pinned"] = False
            stats["mass_scale"] = 1.0
    else:
        stats["improved"] = True
        stats["seed_won"] = False
        result = refit
    stats["wall_s"] = float(time.time() - t0)
    return result, stats
