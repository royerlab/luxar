"""
Result finalization for Gaussian splat fitting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    import torch

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.fit_basis import reference_on_fit_basis
from luxar.gsplats.fitting.config import (
    FitConfig,
    OptimizationResults,
    PreprocessedData,
)
from luxar.gsplats.fitting.initialization import (
    apply_sigma_min_diag_floor,
    resolve_fit_initial_sigma_diag,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils.trils import pack_tril

_SCALE_DIAGNOSTIC_TOLERANCE_VOX = 0.01

# ANSI 256-color codes: red → yellow → green → cyan gradient
_GRADIENT_CODES = [
    196,
    196,
    202,
    208,
    214,
    220,
    226,
    190,
    154,
    118,
    82,
    46,
    48,
    50,
    51,
    45,
]
_RESET = "\033[0m"
_DIM = "\033[2m"


def _print_amplitude_histogram(
    amps: "torch.Tensor",
    n_bins: int = 16,
) -> None:
    """Print a colorful ASCII histogram of amplitude distribution.

    Parameters
    ----------
    amps : torch.Tensor or np.ndarray
        Amplitude values (normalized scale).
    n_bins : int
        Number of histogram bins.
    """
    # Convert to numpy
    if hasattr(amps, "cpu"):
        amps_np = amps.detach().cpu().numpy().ravel()
    else:
        amps_np = np.asarray(amps).ravel()

    if len(amps_np) == 0:
        aprint("No amplitudes to display")
        return

    total = len(amps_np)
    min_val = float(amps_np.min())
    max_val = float(amps_np.max())
    median_val = float(np.median(amps_np))
    mean_val = float(np.mean(amps_np))

    # Build histogram bins (nudge upper edge so max value falls inside last bin)
    eps = max(abs(max_val) * 1e-8, 1e-15)
    bin_edges = np.linspace(min_val, max_val + eps, n_bins + 1)
    counts, _ = np.histogram(amps_np, bins=bin_edges)
    max_count = int(counts.max()) if counts.max() > 0 else 1
    count_width = len(str(max_count))
    bar_width = 30

    with asection("📊 Amplitude Distribution (normalized scale)"):
        aprint(
            f"n={total}  "
            f"range: [{min_val:.6f}, {max_val:.6f}]  "
            f"median: {median_val:.6f}  "
            f"mean: {mean_val:.6f}"
        )
        for i in range(n_bins):
            lo = bin_edges[i]
            count = int(counts[i])
            pct = 100.0 * count / total

            # Bar with color gradient
            bar_len = round(bar_width * count / max_count) if max_count > 0 else 0
            cidx = min(i * len(_GRADIENT_CODES) // n_bins, len(_GRADIENT_CODES) - 1)
            color = f"\033[38;5;{_GRADIENT_CODES[cidx]}m"
            bar = color + "█" * bar_len + _RESET
            pad = " " * (bar_width - bar_len)

            aprint(f"  {lo:9.6f} ┤{bar}{pad} {count:>{count_width}} ({pct:5.1f}%)")

        aprint(f"  {max_val:9.6f} ┘")


def _apply_voxel_footprint_correction(
    Ls: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Inflate covariances: Sigma_new = L @ L^T + sigma^2 * I_d

    Works for any dimension d:
    - np.eye(d) creates d-dimensional identity
    - Batch matmul handles any (N, d, d) shape
    - np.linalg.cholesky works for any dimension

    Parameters
    ----------
    Ls : np.ndarray, shape (N, d, d)
        Cholesky factors (lower triangular matrices)
    sigma : float
        Standard deviation in voxel units to add (internally squared to get variance)

    Returns
    -------
    np.ndarray, shape (N, d, d)
        New Cholesky factors L_new where L_new @ L_new^T = Sigma_new
    """
    N, d, _ = Ls.shape  # d detected automatically from input
    variance = sigma * sigma  # Convert sigma to variance
    Sigma = Ls @ Ls.transpose(0, 2, 1)  # (N, d, d) batch matmul
    Sigma_corrected = Sigma + variance * np.eye(d, dtype=Ls.dtype)  # d-dim identity
    return np.linalg.cholesky(Sigma_corrected)  # Works for any d


def _clip_to_bounds(
    centers: np.ndarray,
    Ls: np.ndarray,
    shape: Sequence[int],
    truncate: float,
) -> np.ndarray:
    """
    Scale down L rows so no splat extends beyond volume bounds.

    For each splat k in dimension i, ensures:
        truncate * sqrt(Sigma_ii) <= min(center_ki, shape_i - 1 - center_ki)

    where Sigma_ii = sum_j(L[k,i,j]^2) is the marginal variance along axis i.

    This preserves the splat's orientation (ratios within each row of L) but
    scales it down to fit within the volume bounds.

    Parameters
    ----------
    centers : np.ndarray, shape (N, d)
        Splat center positions in voxel coordinates.
    Ls : np.ndarray, shape (N, d, d)
        Lower-triangular Cholesky factors.
    shape : Sequence[int]
        Volume shape (d elements).
    truncate : float
        Truncation radius (same as used during rendering).

    Returns
    -------
    np.ndarray, shape (N, d, d)
        Clipped Cholesky factors.
    """
    shape_arr = np.array(shape, dtype=np.float32)  # (d,)
    dist_to_edge = np.maximum(
        np.minimum(centers, shape_arr - 1.0 - centers), 0.0
    )  # (N, d)

    max_sigma_sq = (dist_to_edge / truncate) ** 2  # (N, d)

    # Actual sigma_sq per dimension: Sigma_ii = sum_j(L[i,j]^2)
    actual_sigma_sq = np.sum(Ls * Ls, axis=2)  # (N, d)

    # Scale factor per row: min(1, sqrt(max_allowed / actual))
    eps = 1e-12
    ratio = max_sigma_sq / np.maximum(actual_sigma_sq, eps)
    scale = np.sqrt(np.minimum(ratio, 1.0))  # (N, d)

    # Scale each row of L: Ls_clipped[k,i,j] = Ls[k,i,j] * scale[k,i]
    clipped: np.ndarray = Ls * scale[:, :, np.newaxis]  # (N,d,d) * (N,d,1) -> broadcast
    return clipped


def _near_sigma_statistics(
    marginal_sigmas: np.ndarray,
    target_sigma: np.ndarray,
) -> tuple[int, float]:
    """Count splats whose marginal sigma matches the target on every axis."""
    if len(marginal_sigmas) == 0:
        return 0, 0.0
    near = np.all(
        np.abs(marginal_sigmas - target_sigma) <= _SCALE_DIAGNOSTIC_TOLERANCE_VOX,
        axis=1,
    )
    count = int(np.count_nonzero(near))
    return count, count / len(marginal_sigmas)


def _fit_diagnostic_stats(
    optimization_results: OptimizationResults,
    config: FitConfig,
    preprocessed_data: PreprocessedData,
    Ls: np.ndarray,
) -> dict[str, Any]:
    """Build reproducibility and candidate-scale diagnostics for one fit."""
    marginal_sigmas = np.sqrt(np.einsum("nij,nij->ni", Ls, Ls, optimize=True))
    relocation_sigma = np.full(
        preprocessed_data.d, config.dynamic_config.init_sigma_vox, dtype=np.float32
    )
    relocation_count, relocation_fraction = _near_sigma_statistics(
        marginal_sigmas, relocation_sigma
    )
    relocation_statistics = optimization_results.relocation_statistics
    stats: dict[str, Any] = {
        "configured_iterations": config.n_iters,
        "dynamic_ops_enabled": config.enable_dynamic_ops,
        "dynamic_ops_step_every": config.dynamic_config.step_every,
        "dynamic_ops_k_max_residuals": config.dynamic_config.k_max_residuals,
        "dynamic_ops_relocation_events": relocation_statistics["total_relocations"],
        "dynamic_ops_unique_splats_relocated": relocation_statistics["unique_splats"],
        "scale_diagnostic_tolerance_vox": _SCALE_DIAGNOSTIC_TOLERANCE_VOX,
        "fit_init_sigma_vox": (
            config.init_sigma_vox if preprocessed_data.init_L is None else None
        ),
        "relocation_init_sigma_vox": config.dynamic_config.init_sigma_vox,
        "splats_near_relocation_init_sigma_count": relocation_count,
        "splats_near_relocation_init_sigma_fraction": relocation_fraction,
    }

    sigma_min = (
        np.asarray(config.sigma_min_diag, dtype=np.float32)
        if config.sigma_min_diag is not None
        else None
    )
    candidates_match_dimensions = sigma_min is None or sigma_min.shape == (
        preprocessed_data.d,
    )
    fit_sigma: Optional[np.ndarray] = None
    initial_marginal_sigmas: Optional[np.ndarray] = None
    if candidates_match_dimensions:
        if preprocessed_data.init_L is None:
            fit_sigma = resolve_fit_initial_sigma_diag(config, preprocessed_data)
        else:
            initial_Ls = preprocessed_data.init_L.astype(np.float32, copy=True)
            if sigma_min is not None:
                clamped_diagonal = apply_sigma_min_diag_floor(
                    np.diagonal(initial_Ls, axis1=1, axis2=2), sigma_min
                )
                for axis in range(preprocessed_data.d):
                    initial_Ls[:, axis, axis] = clamped_diagonal[:, axis]
            initial_marginal_sigmas = np.sqrt(
                np.einsum("nij,nij->ni", initial_Ls, initial_Ls, optimize=True)
            )
            if len(initial_marginal_sigmas) > 0 and np.allclose(
                initial_marginal_sigmas,
                initial_marginal_sigmas[0],
                rtol=1e-6,
                atol=1e-6,
            ):
                fit_sigma = initial_marginal_sigmas[0]

    if fit_sigma is not None:
        fit_count, fit_fraction = _near_sigma_statistics(marginal_sigmas, fit_sigma)
        stats.update(
            {
                "fit_init_sigma_diag_vox": fit_sigma.tolist(),
                "splats_near_fit_init_sigma_count": fit_count,
                "splats_near_fit_init_sigma_fraction": fit_fraction,
            }
        )
    elif initial_marginal_sigmas is not None:
        has_initial_splats = len(initial_marginal_sigmas) > 0
        stats.update(
            {
                "fit_init_sigma_diag_vox": None,
                "fit_init_marginal_sigma_diag_vox_min": (
                    initial_marginal_sigmas.min(axis=0).tolist()
                    if has_initial_splats
                    else None
                ),
                "fit_init_marginal_sigma_diag_vox_median": (
                    np.median(initial_marginal_sigmas, axis=0).tolist()
                    if has_initial_splats
                    else None
                ),
                "fit_init_marginal_sigma_diag_vox_max": (
                    initial_marginal_sigmas.max(axis=0).tolist()
                    if has_initial_splats
                    else None
                ),
                "splats_near_fit_init_sigma_count": None,
                "splats_near_fit_init_sigma_fraction": None,
            }
        )

    if sigma_min is not None and candidates_match_dimensions:
        sigma_min_count, sigma_min_fraction = _near_sigma_statistics(
            marginal_sigmas, sigma_min
        )
        stats.update(
            {
                "sigma_min_diag_vox": sigma_min.tolist(),
                "splats_near_sigma_min_count": sigma_min_count,
                "splats_near_sigma_min_fraction": sigma_min_fraction,
            }
        )
    return stats


#: Fraction of the normalized [0, 1] intensity range a voxel must EXCEED to count
#: as occupied. A bare ``> 0`` test measures noise, not sparsity: ``V_normalized``
#: is ``clip((V - image_min) / range, 0, 1)`` and ``image_min`` is a low
#: percentile -- or, under ``floor=auto``, the estimated background MODE, i.e. the
#: background's own centre -- so roughly half of a noisy background survives it.
#: Measured on a volume that is 2.2% signal: 58% "occupancy", a number that flatly
#: contradicts the compression ratio it is printed beside. One percent of the
#: dynamic range sits far above camera read noise (a few 1e-3 of range on a 16-bit
#: acquisition) and far below real structure.
#:
#: The threshold is deliberately relative to the fit's OWN normalization, so
#: ``occupancy`` describes what the fit actually had to represent: with
#: ``floor=none`` an unsuppressed pedestal counts, because the optimiser did spend
#: splats on it.
_OCCUPANCY_THRESHOLD = 0.01

#: Voxels per counting block in :func:`_occupied_fraction` (16M -> a 16 MB bool
#: temporary, whatever the volume's size).
_OCCUPANCY_BLOCK_VOXELS = 1 << 24


def _occupied_fraction(Vn: np.ndarray, fitted_voxels: int) -> float:
    """Fraction of ``Vn`` above :data:`_OCCUPANCY_THRESHOLD` of its range.

    Counted in blocks along the first axis rather than through a whole-volume
    ``Vn > t`` mask: that mask is another full-size allocation, on top of the
    three copies of the volume already resident at this point. Basic slicing is a
    view, so the temporary is bounded by the block size whatever the volume's.
    """
    rows = max(1, _OCCUPANCY_BLOCK_VOXELS // max(1, int(np.prod(Vn.shape[1:]))))
    occupied = 0
    for start in range(0, len(Vn), rows):
        block = Vn[start : start + rows]
        occupied += int(np.count_nonzero(block > _OCCUPANCY_THRESHOLD))
    return float(occupied / fitted_voxels)


#: Source-grid stamps that describe the VOLUME and so belong to a whole fit,
#: however many times the fitter was invoked to produce it.
#:
#: Deliberately excludes ``voxels_per_splat``: that one is a ratio against the
#: splat count of the invocation that produced it, so a multi-pass fitter
#: copying it verbatim would report the first pass's density for the whole
#: result. It has to be recomputed against the final count.
SOURCE_GRID_VOLUME_KEYS = (
    "source_shape",
    "source_dtype",
    "source_voxels",
    "source_bytes",
    "source_stored_bytes",
    "source_declared",
    "fitted_shape",
    "fitted_voxels",
    "occupancy",
)


def lift_source_grid_stats(dest: dict[str, Any], passes: "Sequence[Any]") -> None:
    """Copy the source-grid stamps from a multi-pass fit's FIRST pass onto ``dest``.

    Every pass of a progressive fit sees the same volume (later ones fit its
    residual), so the first pass's record of that volume describes the fit as a
    whole. Left in the per-pass stats it never reaches ``_FITTING_INFO_KEYS``, and
    the dataset cannot say what it is a representation of.

    ``passes`` are the accumulated sub-LODs, in order; an empty list is a no-op.
    """
    if not passes:
        return
    first = getattr(passes[0], "stats", None) or {}
    for key in SOURCE_GRID_VOLUME_KEYS:
        if key in first:
            dest[key] = first[key]


def lift_normalization_stats(
    dest: dict[str, Any],
    passes: "Sequence[Any]",
    applied_floor: "float | None",
) -> None:
    """Record a multi-pass fit's normalization provenance on ``dest`` (#1175).

    A progressive fit subtracts the pedestal from the volume ONCE up front and
    then runs every pass with ``floor="none"``, so no pass's own stats knows the
    level — the whole fit used to ship no record of the background it removed.
    ``applied_floor`` is that up-front level (``None`` when suppression was
    disabled or refused).

    The bounds come from the FIRST pass only: it is the one that sees the volume
    itself, while later passes normalize their own residual by its own extent,
    so no single ``intensity_range`` describes them all. They were measured on
    the already-subtracted array, so the level is added back — the block is in
    the input volume's own units on every writer path, matching the single-pass
    fitter where ``image_min`` IS the applied level.

    ``floor`` is the effective baseline the single-pass fitter would record: the
    greater of the resolved floor and the configured low normalization endpoint.
    Pass 0 receives the same bounds shifted onto the already-subtracted basis,
    so adding that baseline back yields the same ``image_min``, ``image_max`` and
    ``intensity_range`` as a flat fit.
    """
    dest["floor"] = applied_floor
    if not passes:
        return
    first = getattr(passes[0], "stats", None) or {}
    shift = float(applied_floor) if applied_floor is not None else 0.0
    for key in ("image_min", "image_max"):
        if key in first:
            dest[key] = float(first[key]) + shift
    if "intensity_range" in first:
        dest["intensity_range"] = float(first["intensity_range"])


def stamp_voxels_per_splat(stats: dict[str, Any], n_splats: int) -> None:
    """Quote density against the splats actually DELIVERED.

    Called after any post-fit cull rather than beside the other source-grid
    stamps: the pre-cull count would overstate how much of the volume each
    surviving splat stands for, and it is the surviving ones that ship. A no-op
    without a fitted grid to divide, or with nothing left to divide by.
    """
    fitted_voxels = stats.get("fitted_voxels")
    if fitted_voxels and n_splats:
        stats["voxels_per_splat"] = float(fitted_voxels / n_splats)


def _source_grid_stats(
    config: FitConfig, preprocessed_data: PreprocessedData, n_splats: int
) -> dict[str, Any]:
    """Record the volume the splats represent, so compression is computable later.

    A fitted ``.gsplats.zarr`` records its own byte size but nothing about what it
    is a representation of, which makes "how much did this compress?" unanswerable
    from the artifact. It is not answerable from the producing script either: the
    fitted grid is derived at run time from downscale factors and from isotropic
    resampling of the voxel spacing, so it is not a constant anyone can read off.

    Two grids are kept separate on purpose:

    ``source_*``
        the array handed to the fitter, in its original dtype -- the honest
        denominator for a compression ratio.
    ``fitted_*``
        the grid actually optimised against, after any downscaling. Equal to the
        source grid when no downscaling happened.

    ``occupancy`` is the fraction of fitted voxels carrying signal (see
    :func:`_occupied_fraction`). Sparse microscopy volumes are typically >99%
    empty, and a compression ratio means something quite different at 0.03%
    occupancy than at 50%, so the ratio should never be quoted without it.
    """
    out: dict[str, Any] = {}
    V = getattr(config, "V", None)
    if V is not None and hasattr(V, "shape"):
        # A DECLARED source grid wins over the array's own. Most producers
        # preprocess before fitting — a demo that downscales a 5D OME-Zarr
        # channel to 128^3 hands the fitter something that is no longer the
        # acquisition, so measuring `V` would quote the ratio against the
        # working copy. `source_declared` is recorded alongside so a reader can
        # tell a measured grid from a stated one; an unmarked declaration would
        # be indistinguishable from a measurement, which is the whole risk of
        # letting callers name their own denominator.
        declared = getattr(config, "source_shape", None)
        shape = [int(x) for x in (declared if declared else V.shape)]
        out["source_shape"] = shape
        voxels = int(np.prod(shape)) if shape else 0
        out["source_voxels"] = voxels
        if declared:
            out["source_declared"] = True
        stored = getattr(config, "source_stored_bytes", None)
        if stored:
            # What the acquisition OCCUPIES, beside what it decodes to. `info`
            # quotes both ratios: against raw voxels the splats look best, and
            # against the stored file is what a reader downloading it compares.
            out["source_stored_bytes"] = int(stored)
        # `config.V` has already been cast to float32, so its own dtype/nbytes
        # would describe the fitter's working copy rather than the caller's
        # array. Use what validation captured before the cast, and fall back to
        # the cast array only when that is unavailable.
        dtype = getattr(config, "source_dtype", None) or str(getattr(V, "dtype", ""))
        itemsize = getattr(config, "source_itemsize", None)
        out["source_dtype"] = dtype
        if itemsize:
            out["source_bytes"] = voxels * int(itemsize)
        elif dtype == str(getattr(V, "dtype", "")) and getattr(V, "nbytes", None):
            # No captured item size: quote the working array's own bytes ONLY
            # while the recorded dtype IS that array's dtype. A dtype name numpy
            # could not size (recorded verbatim, on purpose) would otherwise be
            # paired with float32 byte counts, and `gsplat info` would turn that
            # pair into a compression ratio inflated by the cast. No bytes is
            # better than bytes measured on a different type — `info` already
            # stays silent when the source size is unknown.
            #
            # UNREACHABLE on every current path, and deliberately left rather
            # than deleted: `config.V` is always the post-cast float32 array, so
            # this needs `dtype == "float32"`, whose item size is never missing.
            # Should that cast ever move, note this branch measures `V` — which
            # a DECLARED `source_shape` makes the wrong array, not merely the
            # wrong type. Guard on `declared` here if it becomes live; it is not
            # guarded today because no test could prove the guard works.
            out["source_bytes"] = int(V.nbytes)

    Vn = getattr(preprocessed_data, "V_normalized", None)
    if Vn is not None and hasattr(Vn, "shape"):
        out["fitted_shape"] = [int(x) for x in Vn.shape]
        fitted_voxels = int(np.prod(Vn.shape)) if Vn.ndim else 0
        out["fitted_voxels"] = fitted_voxels
        if fitted_voxels:
            out["occupancy"] = _occupied_fraction(Vn, fitted_voxels)
        if n_splats:
            out["voxels_per_splat"] = float(fitted_voxels / n_splats)
    return out


def finalize_results(
    optimization_results: OptimizationResults,
    config: FitConfig,
    preprocessed_data: PreprocessedData,
) -> GSplatData:
    """
    Finalize optimization results and return as GSplatData.

    Parameters
    ----------
    optimization_results : OptimizationResults
        Results from optimization loop
    config : FitConfig
        Configuration used for fitting
    preprocessed_data : PreprocessedData
        Preprocessed data with normalization metadata

    Returns
    -------
    GSplatData
        Dataclass containing centers, amplitudes, cholesky_factors, and stats
    """
    amps_dev = optimization_results.amps  # still on device, normalized scale
    centers_dev = optimization_results.centers
    Ls_dev = optimization_results.Ls

    if config.verbose:
        _print_amplitude_histogram(amps_dev)

    # Transfer to CPU + numpy
    centers_np = centers_dev.cpu().numpy()
    Ls_np = Ls_dev.cpu().numpy()
    amps_np = amps_dev.cpu().numpy()
    diagnostic_stats = _fit_diagnostic_stats(
        optimization_results, config, preprocessed_data, Ls_np
    )

    # Rescale amplitudes to original intensity range.
    # NOTE: image_min (incl. any subtracted background floor) is intentionally
    # NOT added back — output amplitudes are background-relative by design
    # (background -> 0), which is what the viewer wants. See PreprocessedData.floor.
    amps_np = amps_np * preprocessed_data.intensity_range
    if config.verbose:
        aprint(
            f"Rescaled amplitudes to original intensity range (factor: {preprocessed_data.intensity_range:.4f})"
        )

    # Clip splats to volume bounds if enabled (before voxel footprint correction)
    # When downscaling is active, splats are still in downscaled coords here,
    # so use the downscaled shape for clipping.
    if config.clip_to_bounds:
        clip_shape = config.V.shape
        if preprocessed_data.downscale_factors is not None:
            clip_shape = tuple(
                -(-s // f)  # ceil division: equivalent to math.ceil(s / f)
                for s, f in zip(config.V.shape, preprocessed_data.downscale_factors)
            )
        Ls_np = _clip_to_bounds(centers_np, Ls_np, clip_shape, config.truncate)
        if config.verbose:
            aprint(f"Clipped splats to volume bounds (truncate={config.truncate:.1f})")

    # Apply voxel footprint correction if enabled
    if config.voxel_footprint_correction:
        # Check for numeric types (int/float) but exclude bool (which is a subclass of int)
        if isinstance(
            config.voxel_footprint_correction, (int, float)
        ) and not isinstance(config.voxel_footprint_correction, bool):
            sigma = float(config.voxel_footprint_correction)
        else:
            # Default: 1-voxel box footprint has sigma = sqrt(1/12) ≈ 0.289 voxels
            sigma = np.sqrt(1.0 / 12.0)
        Ls_np = _apply_voxel_footprint_correction(Ls_np, sigma)
        if config.verbose:
            aprint(f"Applied voxel footprint correction (sigma={sigma:.4f} voxels)")

    # Pack Cholesky factors (without sharpness)
    cholesky_packed = pack_tril(Ls_np)

    # Rescale from downscaled coords to original coords (before voxel_size conversion)
    if preprocessed_data.downscale_factors is not None:
        from luxar.gsplats.fitting.downscale import (
            rescale_centers,
            rescale_cholesky_packed,
        )

        factors = preprocessed_data.downscale_factors
        centers_np = rescale_centers(centers_np, factors)
        cholesky_packed = rescale_cholesky_packed(cholesky_packed, factors)
        if config.verbose:
            aprint(
                f"Rescaled splats to original coordinates (downscale factors={factors})"
            )

    # Convert to physical coordinates if requested. The pre-conversion arrays
    # are kept: they are the only ones that live on `config.V`'s grid, and the
    # quality metrics below have to render against exactly that grid.
    voxel_space_arrays: Optional[tuple[np.ndarray, np.ndarray]] = None
    if config.output_space == "real" and config.voxel_size is not None:
        voxel_space_arrays = (centers_np, cholesky_packed)
        vs = config.voxel_size  # (d,)
        d = centers_np.shape[1] if len(centers_np) > 0 else config.V.ndim
        # Scale centers: voxel indices → physical coordinates
        centers_np = centers_np * vs  # (N, d) * (d,)
        # Scale packed Cholesky: row i has (i+1) elements, each scaled by vs[i]
        # L_phys[i,j] = voxel_size[i] * L_vox[i,j]
        tril_scales = np.concatenate([[vs[i]] * (i + 1) for i in range(d)])
        cholesky_packed = cholesky_packed * tril_scales  # (N, tril) * (tril,)
        if config.verbose:
            aprint(
                f"Converted output to physical coordinates (voxel_size={vs.tolist()})"
            )

    # Compute statistics reflecting best state (not final state)
    stats: dict[str, Any] = {
        "time_seconds": optimization_results.end_time - optimization_results.start_time,
        "iterations": optimization_results.actual_iters,
        "best_iteration": optimization_results.best_iteration,  # Iteration that achieved best quality
        "final_loss": optimization_results.best_loss,
        "final_max_abs_error": optimization_results.best_max_abs_error,
        "final_rel_l2": optimization_results.best_rel_l2,
        "converged": optimization_results.converged_early,
        "early_stopped": optimization_results.early_stopped,
        "n_splats": len(amps_np),
        # Normalization metadata (recorded for inspection/reproducibility).
        # `floor` is the background level subtracted before fitting (None if
        # floor suppression was disabled); it is NOT added back to amplitudes.
        "image_min": preprocessed_data.image_min,
        "image_max": preprocessed_data.image_max,
        "intensity_range": preprocessed_data.intensity_range,
        "floor": preprocessed_data.floor,
        **diagnostic_stats,
        # What the splats are a representation OF. Without this, a stored
        # .gsplats.zarr cannot say how much it compressed: the source grid is
        # nowhere on disk, and it is not recoverable from the demo either,
        # because the fitted grid is computed at run time (downscale factors,
        # isotropic resampling from voxel spacing). Two grids, kept apart on
        # purpose -- `source_*` is the array handed to the fitter, `fitted_*` is
        # what it actually optimised against after any downscaling.
        **_source_grid_stats(config, preprocessed_data, len(amps_np)),
    }

    # Store movie frames in stats for later display (don't show here to avoid timing issues)
    if (
        config.napari_movie
        and optimization_results.movie_frames is not None
        and len(optimization_results.movie_frames["reconstruction"]) > 0
    ):
        stats["movie_frames"] = optimization_results.movie_frames
        stats["movie_shape"] = config.V.shape
    else:
        stats["movie_frames"] = None

    result = GSplatData(
        centers=centers_np.astype(np.float32),
        amplitudes=amps_np.astype(np.float32),
        cholesky_factors=cholesky_packed.astype(np.float32),
        stats=stats,
        truncation_radius=config.truncate,
    )

    # Compute round-trip quality metrics (PSNR, SSIM, MSE).
    #
    # `result` may be in PHYSICAL coordinates, which do not match
    # `config.V.shape`. This used to skip the metrics entirely — and since
    # `output_space="real"` is the DEFAULT, every fit that passed a voxel_size
    # (ten of the demos) silently produced an archive with no PSNR at all, which
    # is the one number a published dataset most needs. The conversion is a pure
    # per-axis scale of centers and Cholesky rows with amplitudes untouched, so
    # the pre-conversion arrays describe the SAME mixture on `config.V`'s own
    # grid: score that copy instead of giving up.
    if voxel_space_arrays is None:
        scored = result
    else:
        scored = GSplatData(
            centers=voxel_space_arrays[0].astype(np.float32),
            amplitudes=amps_np.astype(np.float32),
            cholesky_factors=voxel_space_arrays[1].astype(np.float32),
            truncation_radius=config.truncate,
        )
    try:
        import torch

        from luxar.gsplats.metrics import compute_quality_metrics
        from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

        with torch.no_grad():
            device = str(preprocessed_data.V_tensor.device)
            rendered = render_to_volume_tensor(
                scored,
                shape=config.V.shape,
                device=device,
                truncate=config.truncate,
            )
            # Score against the basis the fit actually reconstructs. The render
            # is background-relative (``V - image_min``, never ``V``), so a raw
            # reference charges the fit for the pedestal it deliberately did not
            # represent — and the penalty grows with the floor, which inverts
            # comparisons: a floor-suppressed fit of the SAME data scores worse
            # than an unfloored one while being the better representation.
            # `data_range` and `rel_l2` follow the reference, so shifting it here
            # fixes their denominators too (#1173).
            # `copy=False`, not a bare `astype`: `config.V` is already float32
            # (see the note above), and `astype` copies even when the dtype
            # matches — a second full-volume allocation on top of the shift's, on
            # a volume that can be gigabytes. This keeps the float64 guard (a
            # float64 reference makes the metrics raise against a float32 render)
            # without paying for it in the common case.
            ref = torch.from_numpy(
                reference_on_fit_basis(config.V, preprocessed_data.image_min).astype(
                    np.float32, copy=False
                )
            ).to(rendered.device)
            quality = compute_quality_metrics(rendered, ref)
            del rendered, ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stats["mse"] = quality["mse"]
        stats["psnr_db"] = quality["psnr_db"]
        stats["ssim"] = quality["ssim"]
        # Foreground PSNR is the honest score on sparse volumes, where the
        # global figure is mostly a report on reconstructed emptiness.
        stats["foreground_psnr_db"] = quality["foreground_psnr_db"]
        stats["foreground_threshold"] = quality["foreground_threshold"]
        stats["foreground_fraction"] = quality["foreground_fraction"]
        if config.verbose:
            aprint(
                f"Quality: PSNR={quality['psnr_db']:.1f} dB, "
                f"foreground PSNR={quality['foreground_psnr_db']:.1f} dB "
                f"(over {quality['foreground_fraction'] * 100:.2f}% of voxels), "
                f"SSIM={quality['ssim']:.4f}, MSE={quality['mse']:.2e}"
            )
    except Exception as exc:
        if config.verbose:
            aprint(f"Note: post-fit quality metrics skipped ({exc})")

    return result
