"""
Multi-Scale Gaussian Splat Fitting (Experimental)

Fits Gaussian splats using multi-scale decomposition. This approach decomposes an
image into multiple frequency bands and fits splats independently on each band.

WARNING: Testing has shown that this approach does NOT provide speed or quality
benefits over direct fitting with `fit_gaussian_splats()`. The decomposition
overhead and the need to fit multiple scales often results in slower total time
and comparable or worse reconstruction quality. Use `fit_gaussian_splats()` for
production workloads.

This module is preserved for research/experimentation purposes.
"""

import time
import warnings
from typing import Any, List, Optional, Union

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.multiscale import decompose_image

# Minimum dimension size for a scale to be meaningful
_MIN_SCALE_DIM = 8  # Minimum 8 pixels per dimension after downsampling

# Minimum seeds per scale as fraction of voxels (1% = quite low but non-zero)
_MIN_SEEDS_VOXEL_FRACTION = 0.01


def _compute_floats_per_splat(ndim: int) -> int:
    """Compute number of floats needed to represent one Gaussian splat."""
    # center (d) + cholesky (d*(d+1)/2) + amplitude (1)
    return ndim + ndim * (ndim + 1) // 2 + 1


def _compression_ratio_to_total_seeds(
    compression_ratio: float,
    total_voxels: int,
    ndim: int,
) -> int:
    """Convert compression ratio to total target seed count."""
    floats_per_splat = _compute_floats_per_splat(ndim)
    target = int(compression_ratio * total_voxels / floats_per_splat)
    return max(1, target)


def _distribute_seeds_by_maxima(
    seeds: Union[int, float],
    maxima_per_scale: List[int],
    scales_list: List[np.ndarray],
    scale_factors: List[int],
    ndim: int,
    verbose: bool = False,
) -> List[int]:
    """
    Distribute seeds across scales using sqrt-weighted local maxima counts.

    Uses sqrt() to flatten the distribution, giving coarse scales a fairer share
    while still allocating more seeds to fine scales with more structure.

    Supports both absolute seed count (int) and compression ratio (float).

    Parameters
    ----------
    seeds : int or float
        Total number of seeds (int) or compression ratio (float 0-1).
        For compression ratio, the denominator is the sum of all scale voxels.
    maxima_per_scale : List[int]
        Number of local maxima detected at each scale.
    scales_list : List[np.ndarray]
        Decomposed scale images.
    scale_factors : List[int]
        Scale factors (1, 2, 4, etc.).
    ndim : int
        Number of dimensions.
    verbose : bool
        Print distribution information.

    Returns
    -------
    List[int]
        Number of seeds allocated to each scale.
    """
    from arbol import aprint

    n_scales = len(scales_list)

    # Compute total voxels across all scales
    total_voxels = sum(s.size for s in scales_list)

    # Handle compression ratio (float)
    if isinstance(seeds, float):
        if not (0 < seeds <= 1.0):
            raise ValueError(f"Compression ratio must be in (0, 1.0], got {seeds}")
        compression_ratio = seeds
        total_seeds = _compression_ratio_to_total_seeds(
            compression_ratio, total_voxels, ndim
        )
        if verbose:
            floats_per_splat = _compute_floats_per_splat(ndim)
            aprint(
                f"Ratio {compression_ratio:.1%} → {total_seeds} seeds "
                f"({floats_per_splat} floats/splat in {ndim}D)"
            )
    else:
        total_seeds = seeds

    # Handle edge cases
    if not maxima_per_scale or len(maxima_per_scale) != n_scales:
        # Fallback: distribute equally
        per_scale = max(1, total_seeds // n_scales)
        return [per_scale] * n_scales

    # Calculate minimum seeds per scale (based on voxel count)
    min_seeds_per_scale = []
    for scale_img in scales_list:
        min_seeds = max(5, int(scale_img.size * _MIN_SEEDS_VOXEL_FRACTION))
        min_seeds_per_scale.append(min_seeds)

    # Apply sqrt() to flatten the distribution - this makes seed allocation more uniform
    # while still giving more seeds to scales with more structure (maxima).
    # sqrt() is a moderate compression; log1p() would be more aggressive.
    weights = [np.sqrt(max(1, n)) for n in maxima_per_scale]
    total_weight = sum(weights)

    # Distribute seeds proportionally to flattened weights, respecting minimums
    seeds_per_scale = []

    for i, weight in enumerate(weights):
        # Proportional allocation based on sqrt-compressed weights
        proportion = weight / total_weight if total_weight > 0 else 1 / n_scales
        allocated = int(round(total_seeds * proportion))

        # Enforce minimum
        allocated = max(allocated, min_seeds_per_scale[i])

        seeds_per_scale.append(allocated)

    # Adjust for rounding: add/remove from largest allocation
    total_allocated = sum(seeds_per_scale)
    if total_allocated != total_seeds:
        diff = total_seeds - total_allocated
        # Find scale with most seeds and adjust
        max_idx = seeds_per_scale.index(max(seeds_per_scale))
        seeds_per_scale[max_idx] = max(
            min_seeds_per_scale[max_idx], seeds_per_scale[max_idx] + diff
        )

    # Print nice distribution table
    if verbose:
        _print_distribution_table(
            maxima_per_scale,
            seeds_per_scale,
            scale_factors,
            scales_list,
            total_seeds,
        )

    return seeds_per_scale


def _print_distribution_table(
    maxima_per_scale: List[int],
    seeds_per_scale: List[int],
    scale_factors: List[int],
    scales_list: List[np.ndarray],
    total_seeds: int,
) -> None:
    """Print a visual table of maxima and seed distribution."""
    from arbol import aprint

    total_maxima = sum(maxima_per_scale)

    aprint("=" * 70)
    aprint("Seed Distribution (sqrt-weighted by maxima, fine → coarse):")
    aprint("=" * 70)

    for i, (scale, n_maxima, n_seeds) in enumerate(
        zip(scale_factors, maxima_per_scale, seeds_per_scale)
    ):
        maxima_pct = n_maxima / total_maxima * 100 if total_maxima > 0 else 0
        seeds_pct = n_seeds / total_seeds * 100 if total_seeds > 0 else 0
        bar_length = int(seeds_pct / 2)  # Bar shows seed allocation
        bar = "█" * bar_length

        shape_str = "x".join(str(s) for s in scales_list[i].shape)
        aprint(
            f"Scale {scale:2d}x: {n_maxima:5d} maxima ({maxima_pct:4.1f}%) "
            f"→ {bar:<20} {n_seeds:4d} seeds ({seeds_pct:4.1f}%) [{shape_str}]"
        )

    aprint(
        f"Total: {total_maxima} maxima → {sum(seeds_per_scale)} seeds (sqrt-weighted)"
    )
    aprint("=" * 70)


def fit_multiscale_gaussian_splats(
    V: np.ndarray,
    seeds: Optional[Union[int, float]] = None,
    scales: Optional[List[int]] = None,
    base_init_sigma: float = 1.5,
    n_iters_decomp: int = 1000,
    n_iters_per_scale: int = 500,
    lr: float = 0.01,
    verbose: bool = False,
    napari_movie: bool = False,
    movie_every: int = 1,
    movie_max_frames: Optional[int] = None,
    visualize_per_scale: bool = False,
    return_intermediate: bool = False,
    **fit_kwargs: Any,
) -> GSplatData:
    """
    Fit Gaussian splats using multi-scale decomposition (EXPERIMENTAL).

    .. warning::
        Testing has shown this approach does NOT provide speed or quality benefits
        over direct fitting. Use ``fit_gaussian_splats()`` instead for production.

    This function decomposes the input image into multiple scales, fits Gaussian splats
    independently on each scale, scales parameters back to full resolution, and combines
    all splats together. The decomposition overhead typically negates any theoretical
    speedup from fitting on smaller images.

    Parameters
    ----------
    V : np.ndarray
        Input image of shape (d1, d2, ..., dn). Must be non-empty and finite.
    seeds : int or float, optional
        Controls the number of splats to fit:
        - **int**: Total target number of seeds across all scales.
        - **float** (0 < seeds <= 1.0): Compression ratio. Specifies the fraction
          of data size to use for splat storage (e.g., 0.1 = 10% compression).
        - **None**: Auto-seeding is used for each scale independently.

        Seeds are distributed across scales using sqrt-weighted local maxima counts.
        This flattens the distribution so coarse scales get a fairer share while
        fine scales (with more maxima) still receive more seeds.
    scales : List[int], optional
        Scale factors for decomposition, e.g., [1, 2, 4, 8].
        Larger values = coarser scales with fewer voxels.
        Default: [1, 2, 4, 8] for most applications.
    base_init_sigma : float, default=1.5
        Base initial sigma in voxels. Will be multiplied by scale_factor
        for each scale to create scale-appropriate Gaussians.
    n_iters_decomp : int, default=1000
        Number of iterations for multi-scale decomposition.
    n_iters_per_scale : int, default=500
        Number of iterations for fitting Gaussians on each scale.
    lr : float, default=0.01
        Learning rate passed to fit_gaussian_splats().
    verbose : bool, default=False
        Enable verbose logging of decomposition and fitting progress.
    napari_movie : bool, default=False
        Enable recording and display of optimization progress for animation.
        When True, shows napari movie viewer for both decomposition AND each
        per-scale Gaussian fitting (close each viewer to continue to next).
    movie_every : int, default=1
        Record a movie frame every N iterations.
    movie_max_frames : Optional[int], default=None
        Maximum number of frames to store. If None, no limit (can use lots of memory).
    visualize_per_scale : bool, default=False
        Enable per-scale visualization showing splat locations and reconstructions.
        If True, returns per-scale visualizations in stats['per_scale_visualizations'].
    return_intermediate : bool, default=False
        Return intermediate per-scale results for detailed analysis and visualization.
        If True, stats['intermediate'] contains a list of dicts, one per scale:
        - 'scale_factor': The scale factor (1, 2, 4, etc.)
        - 'target': The decomposed scale target image (V_scale)
        - 'splats': GSplatData fitted to this scale (in downsampled coordinates)
        - 'splats_full_res': GSplatData scaled to full resolution
        - 'reconstruction': Reconstruction of V_scale using fitted splats
        - 'residual': Residual (target - reconstruction) at this scale
    **fit_kwargs
        Additional arguments passed to fit_gaussian_splats() for each scale.
        Supports all parameters: loss_type, asymmetric_penalty, l1_amp,
        dynamic_ops, max_abs_error, etc.

    Returns
    -------
    GSplatData
        Dataclass containing combined results from all scales:
        - centers: np.ndarray, shape (N_total, d) - Center positions at full resolution
        - amplitudes: np.ndarray, shape (N_total,) - Combined amplitudes
        - cholesky_factors: np.ndarray, shape (N_total, d*(d+1)//2) - At full res
        - stats: Dict[str, Any] - Combined statistics including:
          * decomposition_stats: Stats from decompose_image()
          * per_scale_stats: List of stats from each scale's fitting
          * n_splats_per_scale: Number of splats fitted per scale
          * total_splats: Total number of splats across all scales
          * computational_speedup: Theoretical voxel-based speedup (often not realized)
          * total_time_seconds: Total wall-clock time
          * decomposition_time_seconds: Time for decomposition
          * fitting_time_seconds: Time for per-scale fitting
          * per_scale_visualizations: Viz data (only if visualize_per_scale)
          * intermediate: Per-scale GSplatData (only if return_intermediate)
          * scale_images: Decomposed images (only if return_intermediate)

        Geometric params (centers, Cholesky) are scaled to full resolution.

    Notes
    -----
    - **Performance**: Despite theoretical voxel reduction at coarse scales,
      the decomposition overhead and need to fit multiple scales typically
      results in slower overall time than direct fitting. Benchmark before
      assuming speedup.
    - Coarse scales capture large structures, fine scales capture details
    - Amplitudes do not scale (see Parameter Scaling Rules)
    - Uses existing fit_gaussian_splats() as building block

    Examples
    --------
    >>> # Simple 2D example
    >>> result = fit_multiscale_gaussian_splats(
    ...     image_2d,
    ...     scales=[1, 2, 4],
    ...     n_iters_per_scale=300
    ... )
    >>> print(f"Fitted {len(result.amplitudes)} splats")

    >>> # 3D volume with custom parameters
    >>> result = fit_multiscale_gaussian_splats(
    ...     volume_3d,
    ...     scales=[1, 2, 4, 8, 16],
    ...     base_init_sigma=2.0,
    ...     n_iters_decomp=2000,
    ...     n_iters_per_scale=500,
    ...     lr=0.02,
    ...     loss_type='l1',
    ...     max_abs_error=0.01,
    ...     verbose=True
    ... )
    >>> print(f"Centers shape: {result.centers.shape}")
    """
    # Warn users that this approach doesn't provide expected benefits
    warnings.warn(
        "fit_multiscale_gaussian_splats() is experimental and does NOT provide "
        "speed or quality benefits over fit_gaussian_splats(). "
        "Consider using fit_gaussian_splats() instead.",
        UserWarning,
        stacklevel=2,
    )

    # Default scales
    if scales is None:
        scales = [1, 2, 4, 8]

    # Sort scales to ensure consistent order (fine to coarse)
    scales = sorted(scales)

    # Validate inputs
    if not isinstance(V, np.ndarray):
        raise TypeError(f"V must be a numpy array, got {type(V)}")
    if V.size == 0:
        raise ValueError("V must be non-empty")
    if not np.all(np.isfinite(V)):
        raise ValueError("V must contain only finite values")
    if len(scales) == 0:
        raise ValueError("scales must contain at least one scale factor")
    if any(s < 1 for s in scales):
        raise ValueError("All scale factors must be >= 1")

    # Validate that scales are appropriate for image size (Issue #8)
    for scale in scales:
        downsampled_shape = tuple(max(1, s // scale) for s in V.shape)
        if any(dim < _MIN_SCALE_DIM for dim in downsampled_shape):
            raise ValueError(
                f"Scale {scale} → dims {downsampled_shape} too small. "
                f"Min is {_MIN_SCALE_DIM}px. Use smaller scale factors."
            )

    if base_init_sigma <= 0:
        raise ValueError(f"base_init_sigma must be positive, got {base_init_sigma}")
    if n_iters_decomp < 0:
        raise ValueError(f"n_iters_decomp must be non-negative, got {n_iters_decomp}")
    if n_iters_per_scale < 0:
        raise ValueError(
            f"n_iters_per_scale must be non-negative, got {n_iters_per_scale}"
        )
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")

    start_time = time.time()
    d = V.ndim

    # Step 1: Multi-Scale Decomposition
    if verbose:
        with asection("Multi-Scale Decomposition"):
            aprint(f"Decomposing image into {len(scales)} scales: {scales}")
            aprint(f"Image shape: {V.shape}")
            aprint(f"Number of iterations: {n_iters_decomp}")

    scales_list, decomp_stats = decompose_image(
        V,
        scales=scales,
        n_iters=n_iters_decomp,
        verbose=verbose,
        napari_movie=napari_movie,
        movie_every=movie_every,
        movie_max_frames=movie_max_frames,
    )

    if verbose:
        decomp_time = decomp_stats.get("time_seconds", 0)
        err = decomp_stats.get("final_error", 0)
        aprint(f"Decomposition: {decomp_time:.2f}s, error: {err:.6e}")
        energy_dist = decomp_stats.get("energy_distribution", [])
        aprint(f"Energy distribution: {' → '.join([f'{e:.1%}' for e in energy_dist])}")

    # Compute seeds per scale if total seeds specified
    seeds_per_scale: Optional[List[int]] = None
    if seeds is not None:
        seeds_per_scale = _distribute_seeds_by_maxima(
            seeds=seeds,
            maxima_per_scale=decomp_stats.get("maxima_per_scale", []),
            scales_list=scales_list,
            scale_factors=scales,
            ndim=d,
            verbose=verbose,
        )

    # Step 2: Independent Fitting Per Scale
    all_params = []
    all_amps = []
    per_scale_stats: list[dict[str, Any]] = []
    per_scale_visualizations: Optional[list] = [] if visualize_per_scale else None
    intermediate_results: Optional[list] = [] if return_intermediate else None

    if verbose:
        with asection("Fitting Gaussian Splats Per Scale"):
            aprint(
                f"Base init_sigma: {base_init_sigma} voxels (scales with scale factor)"
            )
            aprint(f"Iterations per scale: {n_iters_per_scale}")
            aprint(f"Learning rate: {lr}")

    for scale_idx, (scale_factor, V_scale) in enumerate(zip(scales, scales_list)):
        # Adjust init_sigma for scale. Scale init_sigma by scale_factor so
        # when splats are scaled back to full resolution, they have appropriate
        # coverage. This ensures consistent physical coverage after upscaling.
        init_sigma_scaled = base_init_sigma * scale_factor

        if verbose:
            with asection(f"Scale {scale_factor}x (index {scale_idx})"):
                aprint(f"Scale shape: {V_scale.shape}")
                n_voxels = V_scale.size
                voxel_reduction = (V.size / n_voxels) if n_voxels > 0 else 1
                aprint(f"Voxels: {n_voxels:,} ({voxel_reduction:.1f}× reduction)")
                aprint(f"Init sigma: {init_sigma_scaled:.2f} voxels")

        # Fit Gaussians on this scale with error handling (Issue #13)
        scale_start = time.time()

        # Get per-scale seed count if specified
        scale_seeds = seeds_per_scale[scale_idx] if seeds_per_scale else None

        try:
            result = fit_gaussian_splats(
                V_scale,
                seeds=scale_seeds,
                init_sigma_vox=init_sigma_scaled,
                n_iters=n_iters_per_scale,
                lr=lr,
                verbose=verbose,
                napari_movie=napari_movie,
                movie_every=movie_every,
                movie_max_frames=movie_max_frames,
                **fit_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Fitting failed on scale {scale_factor}× (shape {V_scale.shape})"
            ) from e

        scale_time = time.time() - scale_start

        # Extract components from result
        centers_scale = result.centers
        amps = result.amplitudes
        chol_scale = result.cholesky_factors
        stats_scale = result.stats

        if verbose:
            n_splats = len(amps)
            final_error = stats_scale.get("final_error", 0)
            aprint(f"Fitted {n_splats} splats in {scale_time:.2f}s")
            aprint(f"Final error: {final_error:.6e}")

        # Combine geometric parameters (centers and cholesky factors)
        params_geom = np.column_stack([centers_scale, chol_scale])

        # Scale geometric parameters back to full resolution
        if scale_factor > 1:
            # Scale centers: downsampled → full res. Downsampled pixel (i,j)
            # covers [i*s:(i+1)*s], centered at i*s + (s-1)/2.
            offset = (scale_factor - 1) / 2.0
            params_geom[:, :d] = params_geom[:, :d] * scale_factor + offset

            # Scale Cholesky: L_full = scale * L_down. Since Σ = L @ L.T,
            # this gives Σ_full = k² * Σ_down, so σ_full = k * σ_down.
            params_geom[:, d:] *= scale_factor

            # Amplitudes do NOT get scaled. Rationale:
            # The decomposition uses area-averaging for downsampling, which preserves
            # average intensity (not total energy). Fitted amplitudes in downsampled
            # coordinates represent the same intensity levels as in full resolution.
            # When we scale the Gaussian's sigma (coverage), the amplitude represents
            # the same peak intensity, which is correct for reconstruction.

        all_params.append(params_geom)
        all_amps.append(amps)

        # Store per-scale statistics
        per_scale_stats.append(
            {
                "scale_factor": int(scale_factor),
                "scale_shape": tuple(V_scale.shape),
                "n_voxels": int(V_scale.size),
                "n_splats": int(len(amps)),
                "final_error": float(stats_scale.get("final_error", 0)),
                "time_seconds": float(scale_time),
            }
        )

        # Store intermediate results if requested
        if return_intermediate and intermediate_results is not None:
            truncate = fit_kwargs.get("truncate", 3.0)

            # GSplatData in downsampled (scale) coordinates
            splats_scale = GSplatData(
                centers=centers_scale.copy(),
                amplitudes=amps.copy(),
                cholesky_factors=chol_scale.copy(),
                stats=stats_scale,
            )

            # GSplatData scaled to full resolution
            centers_full_res = params_geom[:, :d].copy()
            chol_full_res = params_geom[:, d:].copy()
            splats_full_res = GSplatData(
                centers=centers_full_res,
                amplitudes=amps.copy(),
                cholesky_factors=chol_full_res,
                stats={},
            )

            # Render reconstruction at scale resolution (in downsampled space)
            recon_scale = render_gaussians_numpy(
                V_scale.shape, splats_scale, truncate=truncate
            )
            residual_scale = V_scale - recon_scale

            intermediate_results.append(
                {
                    "scale_factor": int(scale_factor),
                    "target": V_scale.copy(),
                    "splats": splats_scale,
                    "splats_full_res": splats_full_res,
                    "reconstruction": recon_scale,
                    "residual": residual_scale,
                }
            )

        # Store per-scale visualization data if requested
        if visualize_per_scale and per_scale_visualizations is not None:
            # Render reconstruction at full resolution using only this scale's splats
            # Get truncate parameter from fit_kwargs if present
            truncate = fit_kwargs.get("truncate", 3.0)

            # Extract centers and cholesky factors from scaled params_geom
            centers_full_res = params_geom[:, :d]
            chol_full_res = params_geom[:, d:]

            # Create result object for rendering
            result_vis = GSplatData(
                centers=centers_full_res,
                amplitudes=amps,
                cholesky_factors=chol_full_res,
                stats={},  # Empty stats for visualization
            )

            # Render
            recon_full_res = render_gaussians_numpy(
                V.shape, result_vis, truncate=truncate
            )
            residual_full_res = V - recon_full_res

            per_scale_visualizations.append(
                {
                    "scale_factor": int(scale_factor),
                    "scale_shape": tuple(V_scale.shape),
                    "n_splats": int(len(amps)),
                    "original_scale": V_scale.copy(),  # Downsampled image
                    "centers": centers_full_res.copy(),  # Centers at full res
                    "reconstruction": recon_full_res.copy(),  # Full res recon
                    "residual": residual_full_res.copy(),  # Full res residual
                    "error_mse": float(np.mean(residual_full_res**2)),
                    "error_max_abs": float(np.abs(residual_full_res).max()),
                }
            )

            if verbose:
                error_mse = np.mean(residual_full_res**2)
                error_max_abs = np.abs(residual_full_res).max()
                aprint(f"Scale recon: MSE={error_mse:.6e}, Max={error_max_abs:.6f}")

    # Step 3: Combination
    if verbose:
        with asection("Combining Results"):
            total_splats = sum(len(p) for p in all_params)
            aprint(f"Total splats across all scales: {total_splats:,}")

    # Combine all scales (Issue #9: Fix empty array handling)
    if len(all_params) > 0:
        # Filter out empty arrays
        valid_params = [p for p in all_params if len(p) > 0]
        valid_amps = [a for a in all_amps if len(a) > 0]

        if len(valid_params) > 0:
            params_combined = np.vstack(valid_params)
            amps_combined = np.concatenate(valid_amps)
        else:
            # No splats at all - return empty arrays with correct shape
            expected_cols = d + d * (d + 1) // 2
            params_combined = np.zeros((0, expected_cols), dtype=np.float32)
            amps_combined = np.zeros(0, dtype=np.float32)
    else:
        # No scales processed - return empty arrays with correct shape
        expected_cols = d + d * (d + 1) // 2
        params_combined = np.zeros((0, expected_cols), dtype=np.float32)
        amps_combined = np.zeros(0, dtype=np.float32)

    # Compute final statistics
    total_time = time.time() - start_time
    decomp_time = decomp_stats.get("time_seconds", 0)
    fitting_time = total_time - decomp_time

    # Estimate computational speedup (Issue #7: Improved calculation)
    # This is a theoretical voxel-count-based speedup, NOT wall-clock time
    # Formula: baseline_voxel_cost / actual_voxel_cost
    # Baseline: fitting at full resolution for all scales: len(scales) * V.size
    # Actual: sum of downsampled volumes processed: sum(V_scale.size)
    baseline_voxel_cost = len(scales) * V.size
    actual_voxel_cost = sum(int(stat["n_voxels"]) for stat in per_scale_stats)
    computational_speedup = (
        baseline_voxel_cost / actual_voxel_cost if actual_voxel_cost > 0 else 1.0
    )

    stats = {
        "decomposition_stats": decomp_stats,
        "per_scale_stats": per_scale_stats,
        "n_splats_per_scale": [stat["n_splats"] for stat in per_scale_stats],
        "total_splats": int(len(params_combined)),
        "computational_speedup": float(computational_speedup),
        "total_time_seconds": float(total_time),
        "decomposition_time_seconds": float(decomp_time),
        "fitting_time_seconds": float(fitting_time),
    }

    # Add per-scale visualizations if requested
    if visualize_per_scale and per_scale_visualizations is not None:
        stats["per_scale_visualizations"] = per_scale_visualizations

    # Add intermediate results if requested
    if return_intermediate and intermediate_results is not None:
        stats["intermediate"] = intermediate_results
        # Also include the decomposed scale images for reference
        stats["scale_images"] = [s.copy() for s in scales_list]

    if verbose:
        aprint(f"\nMulti-scale fitting complete in {total_time:.2f}s")
        aprint(f"  Decomposition: {decomp_time:.2f}s")
        aprint(f"  Fitting: {fitting_time:.2f}s")
        aprint(f"Total splats: {stats['total_splats']:,}")
        aprint(f"Theoretical voxel speedup: {computational_speedup:.1f}×")
        aprint(f"Splats per scale: {stats['n_splats_per_scale']}")

    # Unpack params_combined into separate components
    centers_final = params_combined[:, :d]
    cholesky_final = params_combined[:, d:]

    return GSplatData(
        centers=centers_final,
        amplitudes=amps_combined,
        cholesky_factors=cholesky_final,
        stats=stats,
    )
