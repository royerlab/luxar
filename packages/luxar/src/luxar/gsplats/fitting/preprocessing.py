"""
Data preprocessing for Gaussian splat fitting.

Handles normalization, seed generation, and gradient dilution compensation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from arbol import aprint, asection
from scipy.spatial import distance

from luxar.gsplats.fitting.config import FitConfig, PreprocessedData
from luxar.gsplats.utils.device import resolve_torch_device

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


def _compute_floats_per_splat(ndim: int) -> int:
    """
    Compute number of floats needed to represent one Gaussian splat.

    Each splat requires:
    - d floats for center position
    - d*(d+1)/2 floats for Cholesky factor (lower triangular)
    - 1 float for amplitude

    Parameters
    ----------
    ndim : int
        Number of dimensions

    Returns
    -------
    int
        Number of floats per splat
    """
    # center (d) + cholesky (d*(d+1)/2) + amplitude (1)
    return ndim + ndim * (ndim + 1) // 2 + 1


def _compression_ratio_to_target_count(
    ratio: float,
    shape: tuple[int, ...],
) -> int:
    """
    Convert compression ratio to target seed count.

    The compression ratio is defined as:
        ratio = (n_splats * floats_per_splat) / total_voxels

    This function inverts that to compute the target number of splats:
        n_splats = ratio * total_voxels / floats_per_splat

    Parameters
    ----------
    ratio : float
        Compression ratio (splat floats / image floats), in range (0, 1.0]
    shape : tuple[int, ...]
        Volume shape

    Returns
    -------
    int
        Target number of seeds (at least 1)

    Examples
    --------
    >>> _compression_ratio_to_target_count(0.1, (100, 100))  # 2D
    142  # 0.1 * 10000 / 7 = 142.8 -> 142

    >>> _compression_ratio_to_target_count(0.1, (64, 64, 64))  # 3D
    2383  # 0.1 * 262144 / 11 = 2383.1 -> 2383
    """
    ndim = len(shape)
    total_voxels = int(np.prod(shape))
    floats_per_splat = _compute_floats_per_splat(ndim)

    target = int(ratio * total_voxels / floats_per_splat)
    return max(1, target)  # At least 1 seed


@dataclass
class _InitContext:
    """Mutable context for tracking pre-initialized parameters during seed generation.

    This avoids mutating the input FitConfig object.
    """

    init_L: Optional[np.ndarray] = None
    init_amps: Optional[np.ndarray] = None


def preprocess_data(config: FitConfig) -> PreprocessedData:
    """
    Preprocess input data for optimization.

    Performs normalization, seed generation, and gradient dilution compensation.

    Parameters
    ----------
    config : FitConfig
        Configuration containing input data and parameters (not mutated)

    Returns
    -------
    PreprocessedData
        Preprocessed data ready for optimization
    """
    from luxar.gsplats.gsplat_data import GSplatData

    V = config.V.copy()  # Work with a copy

    # Validate input: NaN/Inf causes silent failures in normalization and fitting
    if np.any(np.isnan(V)):
        nan_count = int(np.sum(np.isnan(V)))
        raise ValueError(
            f"Input volume contains {nan_count} NaN value(s). "
            f"Clean the data before fitting (e.g., np.nan_to_num(V))."
        )
    if np.any(np.isinf(V)):
        inf_count = int(np.sum(np.isinf(V)))
        raise ValueError(
            f"Input volume contains {inf_count} Inf value(s). "
            f"Clean the data before fitting (e.g., np.nan_to_num(V))."
        )

    # Downscale volume if requested (before seed generation and normalization)
    downscale_factors = config.downscale
    if downscale_factors is not None:
        from luxar.gsplats.fitting.downscale import downscale_volume

        original_shape = V.shape
        V = downscale_volume(V, downscale_factors)
        if config.verbose:
            aprint(
                f"Downscaled volume: {original_shape} → {V.shape} "
                f"(factors={downscale_factors})"
            )

    seeds = config.seeds
    seed_kwargs = config.seed_kwargs or {}  # Default to empty dict if None

    # Add device to seed_kwargs if not already present (for GPU acceleration)
    if "device" not in seed_kwargs:
        # Convert torch.device to string for seed generation
        seed_kwargs["device"] = str(config.device)

    # Create mutable context for init parameters (avoids mutating config)
    init_ctx = _InitContext(
        init_L=config.init_L.copy() if config.init_L is not None else None,
        init_amps=config.init_amps.copy() if config.init_amps is not None else None,
    )

    # Handle GSplatData seeds specially
    if isinstance(seeds, GSplatData):
        seed_centers = seeds.centers.copy()
        # Extract pre-initialized parameters from GSplatData
        _extract_gsplatdata_init(init_ctx, seeds)
        # Rescale seed centers to downscaled coordinates if downscaling is active
        if downscale_factors is not None:
            scale = np.array([1.0 / f for f in downscale_factors], dtype=np.float32)
            seed_centers = seed_centers * scale
            # Also rescale pre-initialized Cholesky factors (L[i,j] /= factor[i])
            if init_ctx.init_L is not None:
                for i, f in enumerate(downscale_factors):
                    init_ctx.init_L[:, i, :] /= f
            if config.verbose:
                aprint(
                    f"Rescaled GSplatData seeds to downscaled coordinates "
                    f"(factors={downscale_factors})"
                )
        if config.verbose:
            aprint(
                f"Using GSplatData seeds: {len(seed_centers)} splats with pre-initialized parameters"
            )
    elif seeds is None:
        # Auto-generate using specified method
        with asection(f"Generating seeds using '{config.seed_method}' method"):
            seed_centers = _generate_seeds(
                V,
                None,
                config.seed_method,
                config.verbose,
                init_ctx=init_ctx,
                **seed_kwargs,
            )
    elif isinstance(seeds, int):
        # User-specified exact count (seed_method still applies)
        with asection(f"Generating seeds using '{config.seed_method}' method"):
            seed_centers = _generate_seeds(
                V,
                seeds,
                config.seed_method,
                config.verbose,
                init_ctx=init_ctx,
                **seed_kwargs,
            )
    elif isinstance(seeds, float):
        # User-specified compression ratio → compute target seed count
        # Compression ratio = (n_splats * floats_per_splat) / total_voxels
        target_count = _compression_ratio_to_target_count(seeds, V.shape)
        if config.verbose:
            floats_per_splat = _compute_floats_per_splat(V.ndim)
            aprint(
                f"Compression ratio {seeds:.3f} → target {target_count} seeds "
                f"({floats_per_splat} floats/splat in {V.ndim}D)"
            )
        with asection(f"Generating seeds using '{config.seed_method}' method"):
            seed_centers = _generate_seeds(
                V,
                target_count,
                config.seed_method,
                config.verbose,
                init_ctx=init_ctx,
                **seed_kwargs,
            )
    else:
        # User-provided array of seed centers
        seed_centers = (
            seeds.copy() if isinstance(seeds, np.ndarray) else np.array(seeds)
        )
        # Rescale seed centers to downscaled coordinates if downscaling is active
        if downscale_factors is not None:
            scale = np.array([1.0 / f for f in downscale_factors], dtype=np.float32)
            seed_centers = seed_centers * scale
            # Also rescale pre-initialized Cholesky factors if provided via config
            if init_ctx.init_L is not None:
                for i, f in enumerate(downscale_factors):
                    init_ctx.init_L[:, i, :] /= f
            if config.verbose:
                aprint(
                    f"Rescaled explicit seed centers to downscaled coordinates "
                    f"(factors={downscale_factors})"
                )

    # Normalize input data
    with asection("Normalizing input data"):
        V_normalized, image_min, image_max, intensity_range, applied_floor = (
            _normalize_data(V, config.norm_percentile, config.verbose, config.floor)
        )

    # Rescale pre-initialized amplitudes to match normalized image scale
    # The seeding methods extract amplitudes from the original image, but
    # optimization works on the normalized [0, 1] image. Without this rescaling,
    # amp_max constraints would be on the wrong scale.
    if init_ctx.init_amps is not None:
        init_ctx.init_amps = np.clip(
            (init_ctx.init_amps - image_min) / intensity_range, 0.0, 1.0
        )
        if config.verbose:
            aprint(
                f"Rescaled init_amps to normalized range: "
                f"[{init_ctx.init_amps.min():.4f}, {init_ctx.init_amps.max():.4f}]"
            )

    # Set auto-convergence threshold
    max_abs_error = _set_convergence_threshold(config.max_abs_error, config.verbose)

    # Get dimensions
    d = V.ndim
    N = int(seed_centers.shape[0])

    # Compute L1 regularization values as fractions of the learning rate
    # This ensures regularization pressure scales proportionally with optimization strength
    # Use config values if provided, otherwise calculate defaults
    l1_amp = config.l1_amp
    if l1_amp is None:
        l1_amp = 0.1 * config.lr  # 10% of LR for amplitude sparsity

    l1_diag = config.l1_diag
    if l1_diag is None:
        l1_diag = 0.01 * config.lr  # 1% of LR for mild shape regularization

    # Move to device
    V_tensor = torch.tensor(V_normalized, dtype=torch.float32, device=config.device)

    # Log L1 regularization settings
    if config.verbose:
        amp_pct = l1_amp / config.lr * 100 if config.lr > 0 else 0
        diag_pct = l1_diag / config.lr * 100 if config.lr > 0 else 0
        aprint(
            f"L1 regularization: amplitude={l1_amp:.4f} ({amp_pct:.0f}% of LR), "
            f"diagonal={l1_diag:.5f} ({diag_pct:.1f}% of LR)"
        )

    return PreprocessedData(
        V_normalized=V_normalized,
        V_tensor=V_tensor,
        seed_centers=seed_centers,
        image_min=image_min,
        image_max=image_max,
        intensity_range=intensity_range,
        floor=applied_floor,
        d=d,
        N=N,
        max_abs_error=max_abs_error,
        rel_l2_target=config.rel_l2_target,
        l1_amp=l1_amp,
        l1_diag=l1_diag,
        init_L=init_ctx.init_L,
        init_amps=init_ctx.init_amps,
        downscale_factors=downscale_factors,
    )


def _generate_seeds(
    V: np.ndarray,
    target_count: int | None,
    seed_method: str,
    verbose: bool,
    init_ctx: _InitContext | None = None,
    **seed_kwargs: Any,
) -> np.ndarray:
    """
    Generate seed centers using specified detection method(s).

    Parameters
    ----------
    V : np.ndarray
        Input image/volume
    target_count : int | None
        Target number of seeds to generate. If specified:
        - If more seeds detected: subsample to exact count (keep highest intensity)
        - If fewer seeds detected: adaptively lower thresholds to find more
        - As last resort: add grid-based seeds to reach target
        If None, uses all generated seeds.
    seed_method : str
        Seed generation method: "decomposition", "grid", "edges", "auto",
        or comma-separated combinations (e.g., "decomposition,edges")
    verbose : bool
        Whether to print progress
    init_ctx : _InitContext | None
        Optional context to populate with GSplatData initialization parameters
    **seed_kwargs
        Additional parameters routed to seed generation methods

    Returns
    -------
    np.ndarray
        Generated seed centers (N, ndim)
    """
    from luxar.gsplats.seeds import generate_seeds

    # Generate seeds using specified method (returns GSplatData)
    # Pass target_count so intelligent seeding methods get proper budget allocation:
    # - With target_seeds: 50% decomposition, 30% edges, 20% grid
    # - Without: Auto estimates ~100 seeds, rest filled by grid fallback
    if target_count is not None:
        seeds_result = generate_seeds(
            V,
            method=seed_method,
            target_seeds=target_count,
            verbose=verbose,
            **seed_kwargs,
        )
    else:
        seeds_result = generate_seeds(
            V, method=seed_method, verbose=verbose, **seed_kwargs
        )

    # Extract centers from the GSplatData returned by generate_seeds
    seed_centers = seeds_result.centers
    # If init_ctx provided, extract pre-initialized parameters
    if init_ctx is not None:
        _extract_gsplatdata_init(init_ctx, seeds_result)
        if verbose:
            aprint("Using scale-informed initialization from seeding method")

    # Log initial generation
    if verbose:
        actual_proportion = len(seed_centers) / V.size * 100
        aprint(
            f"Generated {len(seed_centers)} seed centers using '{seed_method}' method "
            f"({actual_proportion:.3f}% of voxels)"
        )

    # Handle target count if specified
    if target_count is not None:
        # Skip subsampling if within 5% of target (expensive farthest-first selection)
        tolerance = 0.05
        if len(seed_centers) > target_count * (1 + tolerance):
            # More than needed: subsample with spatial diversity + intensity weighting
            idx = np.clip(
                np.round(seed_centers).astype(int),
                0,
                np.array(V.shape) - 1,
            )
            intensities = V[tuple(idx.T)]

            # Need indices when we have pre-initialized arrays to slice
            has_init_arrays = init_ctx is not None and init_ctx.init_L is not None

            if has_init_arrays:
                assert init_ctx is not None
                assert init_ctx.init_L is not None
                subsample_with_idx = _subsample_seeds_spatially_diverse(
                    seed_centers,
                    intensities,
                    target_count,
                    verbose,
                    return_indices=True,
                )
                assert isinstance(subsample_with_idx, tuple)
                seed_centers, selected_indices = subsample_with_idx
                # Slice the pre-initialized arrays to match subsampled seeds
                init_ctx.init_L = init_ctx.init_L[selected_indices]
                if init_ctx.init_amps is not None:
                    init_ctx.init_amps = init_ctx.init_amps[selected_indices]
            else:
                result = _subsample_seeds_spatially_diverse(
                    seed_centers, intensities, target_count, verbose
                )
                assert isinstance(result, np.ndarray)
                seed_centers = result

            if verbose:
                actual_proportion = len(seed_centers) / V.size * 100
                aprint(
                    f"Subsampled to {len(seed_centers)} seeds "
                    f"(spatial diversity + intensity)"
                )

        elif len(seed_centers) < target_count:
            # Not enough seeds: add grid-based fallback seeds to reach target_count.
            # Preserve scale-informed initialization for original seeds and
            # generate appropriate init_L for the new grid fallback seeds.
            n_original = len(seed_centers)
            seed_centers, grid_spacing = _ensure_minimum_seeds(
                V, target_count, seed_centers, seed_method, verbose, **seed_kwargs
            )
            n_added = len(seed_centers) - n_original

            # Extend init_L/init_amps for the new grid seeds
            if init_ctx is not None and n_added > 0:
                ndim = V.ndim
                _extend_init_arrays_for_grid_seeds(
                    init_ctx,
                    n_added,
                    ndim,
                    grid_spacing,
                    V,
                    seed_centers,
                    n_original,
                    verbose,
                )

    return seed_centers


def _subsample_seeds_spatially_diverse(
    seeds: np.ndarray,
    intensities: np.ndarray,
    target_count: int,
    verbose: bool,
    return_indices: bool = False,
    smart_subsample_threshold: int = 10000,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Subsample seeds to exact count with spatial diversity and intensity weighting.

    Uses farthest-first selection among high-quality candidates to ensure
    both good spatial coverage and high-intensity seeds.

    For large target_counts (>10k), uses fast random subsampling to avoid O(n²) slowness.

    Algorithm:
    1. Filter to keep only seeds above intensity threshold (50th percentile)
    2. Start with highest intensity seed
    3. Iteratively select seed that is FARTHEST from already-selected seeds
    4. Repeat until target_count reached

    Parameters
    ----------
    seeds : np.ndarray, shape (N, ndim)
        Candidate seed positions
    intensities : np.ndarray, shape (N,)
        Intensity values at each seed location
    target_count : int
        Exact number of seeds to select
    verbose : bool
        Whether to print progress
    return_indices : bool, default=False
        If True, also return the original indices of selected seeds

    Returns
    -------
    np.ndarray, shape (target_count, ndim) or tuple
        Selected seeds with spatial diversity and high intensity.
        If return_indices=True, returns (seeds, original_indices).
    """
    from luxar.utils.spatial_hash import BatchedSpatialHashGrid

    n_available = len(seeds)
    if n_available <= target_count:
        if return_indices:
            return seeds, np.arange(n_available)
        return seeds  # Return all if not enough

    # Determine intensity threshold to filter low-quality seeds
    # Use percentile that ensures we have enough candidates for selection
    # Calculate percentile that would keep at least target_count seeds
    min_percentile = max(0.0, 100.0 * (1.0 - target_count / n_available))
    # Use at least 30th percentile for quality, but relax if needed for count
    intensity_percentile = min(50.0, min_percentile)

    intensity_threshold = np.percentile(intensities, intensity_percentile)

    # Filter to candidates above threshold
    valid_mask = intensities >= intensity_threshold
    valid_indices = np.where(valid_mask)[0]  # Track original indices
    valid_seeds = seeds[valid_mask]
    valid_intensities = intensities[valid_mask]

    # Sanity check - should always have enough now
    if len(valid_seeds) < target_count:
        # Edge case: use all seeds if filtering still removed too many
        # (can happen with many tied intensity values at percentile boundary)
        valid_seeds = seeds
        valid_intensities = intensities
        valid_indices = np.arange(len(seeds))  # All original indices

    # For very large target counts, use fast random sampling instead of farthest-first
    # Farthest-first is O(n²) and only matters for small selections
    if target_count >= smart_subsample_threshold:
        # Fast path: random weighted sampling for large selections
        # Sort by intensity and take top candidates with some randomness
        sort_idx = np.argsort(valid_intensities)[::-1]  # Descending

        # Take top 120% of target, then randomly select exact target from those
        n_candidates = min(len(valid_seeds), int(target_count * 1.2))
        top_candidates = sort_idx[:n_candidates]

        # Random selection from top candidates
        rng = np.random.default_rng(seed=42)
        selected_from_candidates = rng.choice(
            top_candidates, size=target_count, replace=False
        )

        result: np.ndarray = valid_seeds[selected_from_candidates]
        original_indices: np.ndarray = valid_indices[selected_from_candidates]

        if verbose:
            aprint(
                f"Used fast random subsampling for large target_count={target_count}"
            )

        if return_indices:
            return result, original_indices
        return result

    # Farthest-first selection with intensity priority (for smaller selections)
    # Start with highest intensity seed
    first_idx: int = int(np.argmax(valid_intensities))
    selected_indices_list: list[int] = [first_idx]
    selected_list: list[np.ndarray] = [valid_seeds[first_idx]]

    # Build remaining candidates
    remaining_indices = list(range(len(valid_seeds)))
    remaining_indices.remove(first_idx)

    # Iteratively select farthest seed using batch distance computation
    # For medium selections (1000-10000), use GPU if available for
    # substantial speedup (often orders of magnitude, GPU-dependent)
    use_gpu = target_count > 1000

    if use_gpu:
        # CUDA-only path; this routine has not been validated on MPS, so we
        # opt out of Metal and fall back to CPU when CUDA is unavailable.
        # (`torch` and `resolve_torch_device` are imported at module top, so
        #  a missing-torch ImportError surfaces at module load — no inline
        #  try/except needed here.)
        device = resolve_torch_device(use_metal=False)
        use_gpu = device.type == "cuda"

    if use_gpu:
        # GPU-accelerated farthest-first selection
        valid_seeds_gpu = torch.tensor(valid_seeds, device=device, dtype=torch.float32)
        selected_mask = torch.zeros(len(valid_seeds), dtype=torch.bool, device=device)
        selected_mask[first_idx] = True

        for _ in range(target_count - 1):
            # Get remaining and selected coordinates
            remaining_mask = ~selected_mask
            remaining_coords = valid_seeds_gpu[remaining_mask]
            selected_coords = valid_seeds_gpu[selected_mask]

            # Compute pairwise distances on GPU
            pairwise_dists = torch.cdist(remaining_coords, selected_coords)

            # Find farthest point
            min_dists = pairwise_dists.min(dim=1).values
            farthest_in_remaining = min_dists.argmax()

            # Map back to global index
            remaining_indices_gpu = torch.where(remaining_mask)[0]
            farthest_global = remaining_indices_gpu[farthest_in_remaining]

            # Update selection
            selected_mask[farthest_global] = True

        # Extract final selection
        selected_indices_final: np.ndarray = torch.where(selected_mask)[0].cpu().numpy()
        selected_arr = valid_seeds[selected_indices_final]
    else:
        # CPU fallback: original algorithm
        # Complexity: O(n² d) - slow for large selections
        while len(selected_list) < target_count and remaining_indices:
            # Compute pairwise distances between remaining and selected seeds
            remaining_coords = valid_seeds[remaining_indices]
            selected_coords_arr = np.array(selected_list)

            # cdist computes all pairwise distances at once: (n_remaining, n_selected)
            pairwise_dists_cpu = distance.cdist(remaining_coords, selected_coords_arr)

            # For each remaining point, find distance to nearest selected point
            min_dists_to_selected = pairwise_dists_cpu.min(axis=1)

            # Select point with maximum minimum distance (farthest from any selected)
            farthest_idx_in_remaining = int(np.argmax(min_dists_to_selected))
            farthest_idx_global = remaining_indices[farthest_idx_in_remaining]

            # Add to selection
            selected_list.append(valid_seeds[farthest_idx_global])
            selected_indices_list.append(farthest_idx_global)
            remaining_indices.remove(farthest_idx_global)

        selected_indices_final = np.array(selected_indices_list)
        selected_arr = valid_seeds[selected_indices_final]

    result = np.array(selected_arr)
    # Map selected_indices (within valid_seeds) back to original indices
    original_indices = valid_indices[selected_indices_final]

    if verbose and len(result) == target_count:
        # Calculate spatial distribution metric (average nearest-neighbor distance)
        if len(result) > 1:
            # Pick cell_size as a uniform-density estimate of the typical
            # NN distance (bbox volume / N)^(1/D). Shell expansion handles
            # outliers; correctness doesn't depend on a tight choice.
            bbox = result.max(axis=0) - result.min(axis=0)
            volume = float(np.prod(np.maximum(bbox, 1e-9)))
            cell_size = max(
                (volume / max(len(result), 1)) ** (1.0 / result.shape[1]), 1.0
            )
            grid = BatchedSpatialHashGrid.from_points(
                result, cell_size=cell_size, device="auto"
            )
            nn_distances, _ = grid.query_knn(result, k=2)
            nn_dist_arr = np.asarray(nn_distances)
            avg_spacing = float(np.mean(nn_dist_arr[:, 1]))  # nearest neighbor dist
            aprint(
                f"Spatial diversity: avg nearest-neighbor distance = "
                f"{avg_spacing:.1f} voxels"
            )

    if return_indices:
        return result, original_indices
    return result


def _ensure_minimum_seeds(
    V: np.ndarray,
    target_count: int,
    initial_seeds: np.ndarray,
    seed_method: str,
    verbose: bool,
    **seed_kwargs: Any,
) -> tuple[np.ndarray, float]:
    """
    Ensure minimum seed count by adding grid-based seeds.

    Strategy:
    1. KEEP initial seeds (don't discard!)
    2. ADD grid-based seeds to reach target
    3. Subsample to exact target_count using spatial diversity

    Parameters
    ----------
    V : np.ndarray
        Input image/volume
    target_count : int
        Target number of seeds needed
    initial_seeds : np.ndarray
        Seeds already found (MUST be preserved!)
    seed_method : str
        Seed generation method (for logging only)
    verbose : bool
        Whether to print progress
    **seed_kwargs
        Seed generation parameters (unused, kept for compatibility)

    Returns
    -------
    tuple[np.ndarray, float]
        Seed centers (exactly target_count) and the grid spacing used for fallback seeds
    """
    # Start with initial seeds - NEVER discard these!
    current_seeds = initial_seeds
    grid_spacing = 0.0  # Will be updated if grid seeds are added

    if verbose:
        aprint(
            f"Method '{seed_method}' generated {len(current_seeds)} seeds, "
            f"need {target_count} - will add grid fallback"
        )

    # If still not enough, add grid-based seeds
    if len(current_seeds) < target_count:
        if verbose:
            aprint(
                f"Still need {target_count - len(current_seeds)} seeds, "
                "adding grid-based fallback"
            )
        current_seeds, grid_spacing = _add_grid_fallback_seeds(
            V, target_count, current_seeds, verbose
        )

    # Subsample to exact count using spatial diversity
    if len(current_seeds) > target_count:
        idx = np.clip(
            np.round(current_seeds).astype(int),
            0,
            np.array(V.shape) - 1,
        )
        intensities = V[tuple(idx.T)]
        subsample_result = _subsample_seeds_spatially_diverse(
            current_seeds, intensities, target_count, verbose
        )
        assert isinstance(subsample_result, np.ndarray)
        current_seeds = subsample_result

        if verbose:
            aprint(
                f"Subsampled to {target_count} seeds (spatial diversity + intensity)"
            )

    return current_seeds, grid_spacing


def _add_grid_fallback_seeds(
    V: np.ndarray,
    target_count: int,
    existing_seeds: np.ndarray,
    verbose: bool,
) -> tuple[np.ndarray, float]:
    """
    Add grid-based seeds to reach target count.

    Places seeds on a uniform grid, avoiding regions near existing seeds.

    Parameters
    ----------
    V : np.ndarray
        Input image/volume
    target_count : int
        Target number of seeds
    existing_seeds : np.ndarray
        Existing seed locations
    verbose : bool
        Whether to print progress

    Returns
    -------
    tuple[np.ndarray, float]
        Combined seeds (existing + grid-based) and the grid spacing used
    """
    needed = target_count - len(existing_seeds)
    if needed <= 0:
        return existing_seeds, 0.0  # No grid added, spacing irrelevant

    ndim = V.ndim
    shape = np.array(V.shape)

    # Calculate grid spacing to generate enough points
    # Target more points than needed to account for filtering
    volume = np.prod(shape)
    # Use more aggressive multiplier (3-5x) to ensure enough after filtering
    target_grid_points = needed * 4
    spacing = int(np.ceil((volume / target_grid_points) ** (1.0 / ndim)))
    spacing = max(spacing, 1)  # Allow minimum spacing of 1 (dense grid)

    # Generate grid points
    grid_coords_list: list[tuple[Any, ...]] = []
    ranges = [np.arange(spacing // 2, s, spacing) for s in shape]

    import itertools

    for coords in itertools.product(*ranges):
        grid_coords_list.append(coords)

    grid_coords: np.ndarray = np.array(grid_coords_list, dtype=float)

    # Remove grid points too close to existing seeds (if any exist)
    # But be less aggressive about filtering to ensure we get enough
    if len(existing_seeds) > 0:
        from luxar.utils.spatial_hash import BatchedSpatialHashGrid

        # Use smaller min_distance to be more permissive
        min_distance = max(1.0, spacing * 0.3)  # 30% of spacing, min 1 voxel
        # Cell size must accommodate the query radius; pick generously so
        # the 3^D shell finds the nearest existing seed in one pass.
        cell_size = max(spacing, min_distance * 2.0)
        grid = BatchedSpatialHashGrid.from_points(
            np.asarray(existing_seeds, dtype=np.float32),
            cell_size=cell_size,
            device="auto",
        )
        distances, _ = grid.query_knn(grid_coords.astype(np.float32), k=1)
        grid_coords = grid_coords[distances[:, 0] > min_distance]

    # Track the final spacing used (for init_L generation)
    final_spacing = float(spacing)

    # If we still don't have enough grid points after filtering,
    # generate a denser grid without filtering
    if len(grid_coords) < needed:
        if verbose:
            aprint(
                f"Grid filtering left only {len(grid_coords)} points, "
                f"generating denser unfiltered grid"
            )
        # Dense grid without filtering
        spacing_dense = max(1, int((volume / (needed * 2)) ** (1.0 / ndim)))
        final_spacing = float(spacing_dense)  # Update to denser spacing
        ranges_dense = [np.arange(0, s, spacing_dense) for s in shape]
        grid_coords_dense: list[tuple[Any, ...]] = []
        for coords in itertools.product(*ranges_dense):
            grid_coords_dense.append(coords)
        grid_coords = np.array(grid_coords_dense, dtype=float)

    # Sort by intensity and take top N
    if len(grid_coords) > 0:
        idx = np.clip(np.round(grid_coords).astype(int), 0, shape - 1)
        intensities = V[tuple(idx.T)]
        sorted_indices = np.argsort(intensities)[::-1]

        # Take exactly 'needed' to reach target (or all if fewer available)
        take = min(needed, len(grid_coords))
        grid_coords = grid_coords[sorted_indices[:take]]

        if verbose:
            aprint(f"Added {len(grid_coords)} grid-based fallback seeds")

        # Combine with existing
        return np.vstack([existing_seeds, grid_coords]), final_spacing
    else:
        if verbose:
            aprint("Warning: Could not add grid seeds, using existing only")
        return existing_seeds, 0.0


def _resolve_floor(V: np.ndarray, floor: "str | float | None") -> "float | None":
    """Resolve a ``floor`` spec to a concrete background level, or ``None``.

    ``None`` means "no explicit floor" — the caller keeps its default
    ``image_min`` (hard ``min``/``norm_percentile``). Accepted forms:

    - ``"auto"`` → histogram-mode estimate (see :func:`estimate_floor`).
    - ``"pN"`` (e.g. ``"p10"``) → the Nth intensity percentile.
    - ``"none"`` / ``"0"`` / ``0`` / ``None`` → disabled (returns ``None``).
    - ``float`` / numeric string → that fixed intensity value.
    """
    if floor is None:
        return None
    if isinstance(floor, str):
        f = floor.strip().lower()
        if f in ("none", ""):
            return None
        if f == "auto":
            from luxar.gsplats.calibration import estimate_floor

            return float(estimate_floor(V, method="mode"))
        if f.startswith("p"):
            pct = float(f[1:])
            return float(np.percentile(V, pct))
        value = float(f)  # numeric string
    else:
        value = float(floor)
    if value == 0.0:
        return None
    return value


def _normalize_data(
    V: np.ndarray,
    norm_percentile: float,
    verbose: bool,
    floor: "str | float | None" = None,
) -> tuple[np.ndarray, float, float, float, "float | None"]:
    """Normalize input data to [0, 1] range.

    ``floor`` (see :func:`_resolve_floor`) overrides how ``image_min`` is
    chosen: an explicit background level raises ``image_min`` so the pedestal
    is clipped to 0 by the existing ``np.clip((V - image_min) / range, 0, 1)``.
    ``norm_percentile`` still governs ``image_max`` (bright-outlier clipping),
    so the two are orthogonal.
    """
    # Configurable normalization - store parameters for intensity rescaling
    if norm_percentile == 0.0:
        # Full range normalization
        image_min: float = float(np.min(V))
        image_max: float = float(np.max(V))
        if verbose:
            aprint("Normalization: full min-max range")
    else:
        # Percentile-based robust normalization
        image_min = float(np.percentile(V, norm_percentile))
        image_max = float(np.percentile(V, 100.0 - norm_percentile))
        if verbose:
            aprint(
                f"Normalization: {norm_percentile:.1f}%-"
                f"{100.0 - norm_percentile:.1f}% percentile range"
            )

    # Background floor suppression: raise image_min to the resolved floor.
    resolved_floor = _resolve_floor(V, floor)
    applied_floor: "float | None" = None
    if resolved_floor is not None:
        if resolved_floor >= image_max:
            # A floor at/above the brightest voxel would erase all signal
            # (empty [0,1] range). Refuse it and keep the default image_min.
            if verbose:
                aprint(
                    f"Warning: floor {resolved_floor:.6g} >= image max "
                    f"{image_max:.6g}; ignoring (would erase all signal)"
                )
        else:
            # Only ever RAISE image_min (never below the percentile-based value
            # chosen above): the floor is orthogonal to norm_percentile's low-end
            # clipping. Clamp into [image_min, image_max) so the range stays
            # strictly positive. (When norm_percentile==0, image_min == min(V),
            # so this reduces to max(resolved_floor, min(V)) as before.)
            image_min = float(max(resolved_floor, image_min))
            applied_floor = image_min
            if verbose:
                aprint(
                    f"Floor suppression: subtracting background level {image_min:.6g}"
                )

    intensity_range = image_max - image_min

    if np.abs(intensity_range) < 1e-12:
        V = np.full_like(V, 0.5, dtype=np.float32)
        intensity_range = 1.0  # Avoid division by zero in rescaling
        if verbose:
            aprint("Warning: Input image is nearly uniform")
    else:
        V = np.clip((V - image_min) / intensity_range, 0.0, 1.0)

    return V, image_min, image_max, intensity_range, applied_floor


def _set_convergence_threshold(max_abs_error: float | None, verbose: bool) -> float:
    """Set convergence threshold with sensible default."""
    # Auto-convergence threshold: set sensible default if not provided
    if max_abs_error is None:
        max_abs_error = 0.01  # 1% of normalized [0,1] dynamic range
        if verbose:
            aprint(
                f"Auto-convergence threshold: {max_abs_error:.3f} "
                f"(1% of normalized range)"
            )
    else:
        if verbose:
            aprint(f"Convergence threshold: {max_abs_error:.6f} (user-specified)")

    return max_abs_error


def _extract_gsplatdata_init(init_ctx: _InitContext, gsplat_data: GSplatData) -> None:
    """
    Extract pre-initialized parameters from GSplatData.

    Populates init_ctx.init_L and init_ctx.init_amps from a GSplatData object. This enables using moment pursuit results
    or loaded splats as initialization for gradient descent refinement.

    Parameters
    ----------
    init_ctx : _InitContext
        Context to update with pre-initialized parameters.
    gsplat_data : GSplatData
        Source of initialization data.
    """
    from luxar.gsplats.utils.trils import unpack_tril

    ndim = gsplat_data.centers.shape[1]

    # Extract Cholesky factors (convert from packed to matrix form)
    init_ctx.init_L = unpack_tril(gsplat_data.cholesky_factors, ndim)

    # Extract amplitudes
    init_ctx.init_amps = gsplat_data.amplitudes.copy()


def _extend_init_arrays_for_grid_seeds(
    init_ctx: _InitContext,
    n_added: int,
    ndim: int,
    grid_spacing: float,
    V: np.ndarray,
    seed_centers: np.ndarray,
    n_original: int,
    verbose: bool,
) -> None:
    """
    Extend init_L/init_amps arrays for grid fallback seeds.

    When we add grid-based fallback seeds, we need to generate appropriate
    initialization arrays for them while preserving the original seeds' values.

    Parameters
    ----------
    init_ctx : _InitContext
        Context with init arrays to extend.
    n_added : int
        Number of grid seeds added.
    ndim : int
        Number of dimensions.
    grid_spacing : float
        Grid spacing used for fallback seeds.
    V : np.ndarray
        Input image/volume (for sampling amplitudes).
    seed_centers : np.ndarray
        All seed centers (original + added).
    n_original : int
        Number of original seeds (before adding grid fallback).
    verbose : bool
        Whether to print progress.
    """
    from scipy import ndimage as ndi

    # Grid sigma = spacing/2 for coverage (same as seed_from_grid)
    grid_sigma = max(1.0, grid_spacing / 2.0) if grid_spacing > 0 else 1.0

    if verbose:
        aprint(
            f"Extending init arrays for {n_added} grid fallback seeds "
            f"(σ={grid_sigma:.1f} from spacing={grid_spacing:.1f})"
        )

    # Extend init_L
    if init_ctx.init_L is not None:
        # Create isotropic L for grid seeds with σ = grid_sigma
        grid_L = np.zeros((n_added, ndim, ndim), dtype=np.float32)
        for i in range(ndim):
            grid_L[:, i, i] = grid_sigma
        init_ctx.init_L = np.concatenate([init_ctx.init_L, grid_L], axis=0)
    else:
        # No original init_L, create for all seeds
        # Original seeds get σ=1.0 (no scale info), grid seeds get σ=grid_sigma
        all_L = np.zeros((n_original + n_added, ndim, ndim), dtype=np.float32)
        for i in range(ndim):
            all_L[:n_original, i, i] = 1.0  # Original seeds: σ=1.0
            all_L[n_original:, i, i] = grid_sigma  # Grid seeds: σ=grid_sigma
        init_ctx.init_L = all_L

    # Extend init_amps - sample from image at grid seed locations
    grid_centers = seed_centers[n_original:]
    coords_for_interp = grid_centers.T
    grid_amps = (
        ndi.map_coordinates(V, coords_for_interp, order=1, mode="nearest") * 0.9
    ).astype(np.float32)

    if init_ctx.init_amps is not None:
        init_ctx.init_amps = np.concatenate([init_ctx.init_amps, grid_amps], axis=0)
    else:
        # No original init_amps, sample for all
        original_centers = seed_centers[:n_original]
        original_coords = original_centers.T
        original_amps = (
            ndi.map_coordinates(V, original_coords, order=1, mode="nearest") * 0.9
        ).astype(np.float32)
        init_ctx.init_amps = np.concatenate([original_amps, grid_amps], axis=0)
