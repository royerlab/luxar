"""
Multi-Scale Gaussian Splat Fitting

Fits Gaussian splats using multi-scale decomposition for computational efficiency.
Uses existing fit_gaussian_splats() as a building block.
"""

import time
from typing import List, Optional

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fit_result import GaussianSplatResult
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
from luxar.gsplats.multiscale import decompose_image

# Minimum dimension size for a scale to be meaningful
_MIN_SCALE_DIM = 8  # Minimum 8 pixels per dimension after downsampling


def fit_multiscale_gaussian_splats(
    V: np.ndarray,
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
    **fit_kwargs,
) -> GaussianSplatResult:
    """
    Fit Gaussian splats using multi-scale decomposition for computational efficiency.

    This function decomposes the input image into multiple scales, fits Gaussian splats
    independently on each scale (coarse scales have fewer voxels → faster), scales
    parameters back to full resolution, and combines all splats together.

    Parameters
    ----------
    V : np.ndarray
        Input image of shape (d1, d2, ..., dn). Must be non-empty and finite.
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
        Fewer iterations needed per scale due to fewer voxels.
    lr : float, default=0.01
        Learning rate passed to fit_gaussian_splats().
    verbose : bool, default=False
        Enable verbose logging of decomposition and fitting progress.
    napari_movie : bool, default=False
        Enable recording of decomposition optimization progress for animation.
    movie_every : int, default=1
        Record a movie frame every N iterations during decomposition.
    movie_max_frames : Optional[int], default=None
        Maximum number of frames to store. If None, no limit (can use lots of memory).
    visualize_per_scale : bool, default=False
        Enable per-scale visualization showing splat locations and reconstructions.
        If True, returns per-scale visualizations in stats['per_scale_visualizations'].
    **fit_kwargs
        Additional arguments passed to fit_gaussian_splats() for each scale.
        Supports all parameters: loss_type, asymmetric_penalty, l1_amp,
        dynamic_ops, max_abs_error, etc.

    Returns
    -------
    GaussianSplatResult
        Dataclass containing combined results from all scales:
        - centers: np.ndarray, shape (N_total, d) - Center positions at full resolution
        - amplitudes: np.ndarray, shape (N_total,) - Combined amplitudes from all scales
        - cholesky_factors: np.ndarray, shape (N_total, d*(d+1)//2) - Cholesky factors at full resolution
        - sharpnesses: np.ndarray, shape (N_total,) - Per-splat sharpness values (not scaled)
        - stats: Dict[str, Any] - Combined statistics including:
          * decomposition_stats: Stats from decompose_image()
          * per_scale_stats: List of stats from each scale's fitting
          * n_splats_per_scale: Number of splats fitted per scale
          * total_splats: Total number of splats across all scales
          * computational_speedup: Theoretical voxel-based speedup
          * total_time_seconds: Total wall-clock time
          * decomposition_time_seconds: Time for decomposition
          * fitting_time_seconds: Time for per-scale fitting
          * per_scale_visualizations: Visualization data (only if visualize_per_scale=True)

        All geometric parameters (centers, Cholesky factors) are scaled to full resolution.
        Sharpness values are dimensionless and not scaled.

    Notes
    -----
    - Computational speedup: Scale factor r reduces voxel count by (1/r)^d
      * 2D with scale 8: 64× fewer pixels
      * 3D with scale 8: 512× fewer voxels
    - Coarse scales capture large structures efficiently
    - Fine scales capture details at full resolution
    - Amplitudes and sharpness values do not scale (see Parameter Scaling Rules)
    - Uses existing fit_gaussian_splats() as building block (thin wrapper)

    Examples
    --------
    >>> # Simple 2D example
    >>> params, amps, stats = fit_multiscale_gaussian_splats(
    ...     image_2d,
    ...     scales=[1, 2, 4],
    ...     n_iters_per_scale=300
    ... )

    >>> # 3D volume with custom parameters
    >>> params, amps, stats = fit_multiscale_gaussian_splats(
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
    """
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
                f"Scale {scale} results in too small dimensions {downsampled_shape} "
                f"for image shape {V.shape}. Minimum dimension is {_MIN_SCALE_DIM} pixels. "
                f"Consider using smaller scale factors."
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
        aprint(
            f"Decomposition complete in {decomp_time:.2f}s, final error: {decomp_stats.get('final_error', 0):.6e}"
        )
        energy_dist = decomp_stats.get("energy_distribution", [])
        aprint(f"Energy distribution: {' → '.join([f'{e:.1%}' for e in energy_dist])}")

    # Step 2: Independent Fitting Per Scale
    all_params = []
    all_amps = []
    all_sharpness = []
    per_scale_stats = []
    per_scale_visualizations = [] if visualize_per_scale else None

    if verbose:
        with asection("Fitting Gaussian Splats Per Scale"):
            aprint(
                f"Base init_sigma: {base_init_sigma} voxels (scales with scale factor)"
            )
            aprint(f"Iterations per scale: {n_iters_per_scale}")
            aprint(f"Learning rate: {lr}")

    for scale_idx, (scale_factor, V_scale) in enumerate(zip(scales, scales_list)):
        # Adjust init_sigma for scale
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
        try:
            result = fit_gaussian_splats(
                V_scale,
                init_sigma_vox=init_sigma_scaled,
                n_iters=n_iters_per_scale,
                lr=lr,
                verbose=verbose,
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
        sharpness_scale = result.sharpnesses
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
            # Scale centers (first d columns)
            params_geom[:, :d] *= scale_factor
            # Scale Cholesky factors (columns d to end)
            params_geom[:, d:] *= scale_factor
            # Sharpness does NOT get scaled - it's dimensionless

        all_params.append(params_geom)
        all_amps.append(amps)
        all_sharpness.append(sharpness_scale)

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

        # Store per-scale visualization data if requested
        if visualize_per_scale:
            # Render reconstruction at full resolution using only this scale's splats
            # Get truncate parameter from fit_kwargs if present
            truncate = fit_kwargs.get("truncate", 3.0)

            # Extract centers and cholesky factors from scaled params_geom
            centers_full_res = params_geom[:, :d]
            chol_full_res = params_geom[:, d:]

            # Create result object for rendering
            result_vis = GaussianSplatResult(
                centers=centers_full_res,
                amplitudes=amps,
                cholesky_factors=chol_full_res,
                sharpnesses=sharpness_scale,
                stats={}  # Empty stats for visualization
            )

            # Render
            recon_full_res = render_gaussians_numpy(V.shape, result_vis, truncate=truncate)
            residual_full_res = V - recon_full_res

            per_scale_visualizations.append(
                {
                    "scale_factor": int(scale_factor),
                    "scale_shape": tuple(V_scale.shape),
                    "n_splats": int(len(amps)),
                    "original_scale": V_scale.copy(),  # Original downsampled image
                    "centers": centers_full_res.copy(),  # Splat centers at full resolution
                    "reconstruction": recon_full_res.copy(),  # Reconstruction at full resolution
                    "residual": residual_full_res.copy(),  # Residual at full resolution
                    "error_mse": float(np.mean(residual_full_res**2)),
                    "error_max_abs": float(np.abs(residual_full_res).max()),
                }
            )

            if verbose:
                error_mse = np.mean(residual_full_res**2)
                error_max_abs = np.abs(residual_full_res).max()
                aprint(
                    f"Scale reconstruction: MSE={error_mse:.6e}, Max abs error={error_max_abs:.6f}"
                )

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
        valid_sharpness = [s for s in all_sharpness if len(s) > 0]

        if len(valid_params) > 0:
            params_combined = np.vstack(valid_params)
            amps_combined = np.concatenate(valid_amps)
            sharpness_combined = np.concatenate(valid_sharpness)
            # Add sharpness column to create final params array
            params_final = np.column_stack([params_combined, sharpness_combined])
        else:
            # No splats at all - return empty arrays with correct shape
            expected_cols = d + d * (d + 1) // 2 + 1  # +1 for sharpness
            params_final = np.zeros((0, expected_cols), dtype=np.float32)
            amps_combined = np.zeros(0, dtype=np.float32)
    else:
        # No scales processed - return empty arrays with correct shape
        expected_cols = d + d * (d + 1) // 2 + 1  # +1 for sharpness
        params_final = np.zeros((0, expected_cols), dtype=np.float32)
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
    actual_voxel_cost = sum(stat["n_voxels"] for stat in per_scale_stats)
    computational_speedup = (
        baseline_voxel_cost / actual_voxel_cost if actual_voxel_cost > 0 else 1.0
    )

    stats = {
        "decomposition_stats": decomp_stats,
        "per_scale_stats": per_scale_stats,
        "n_splats_per_scale": [stat["n_splats"] for stat in per_scale_stats],
        "total_splats": int(len(params_final)),
        "computational_speedup": float(computational_speedup),
        "total_time_seconds": float(total_time),
        "decomposition_time_seconds": float(decomp_time),
        "fitting_time_seconds": float(fitting_time),
    }

    # Add per-scale visualizations if requested
    if visualize_per_scale and per_scale_visualizations is not None:
        stats["per_scale_visualizations"] = per_scale_visualizations

    if verbose:
        aprint(f"\nMulti-scale fitting complete in {total_time:.2f}s")
        aprint(f"  Decomposition: {decomp_time:.2f}s")
        aprint(f"  Fitting: {fitting_time:.2f}s")
        aprint(f"Total splats: {stats['total_splats']:,}")
        aprint(f"Theoretical voxel speedup: {computational_speedup:.1f}×")
        aprint(f"Splats per scale: {stats['n_splats_per_scale']}")

    # Unpack params_final into separate components
    centers_final = params_final[:, :d]
    cholesky_final = params_final[:, d:-1]  # Everything between centers and sharpness
    sharpnesses_final = params_final[:, -1]

    return GaussianSplatResult(
        centers=centers_final,
        amplitudes=amps_combined,
        cholesky_factors=cholesky_final,
        sharpnesses=sharpnesses_final,
        stats=stats,
    )
