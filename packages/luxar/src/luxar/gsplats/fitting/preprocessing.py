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
    from luxar.gsplats.calibration import FloorEstimate
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
    # Which intensity convention ``init_amps`` is expressed in. There are two,
    # and they differ by exactly the background floor:
    #
    # * False (default) — RAW-IMAGE-SAMPLED: amplitudes read off the original
    #   (un-normalized, un-floored) volume, as every seeding method produces
    #   them. Rescaling to the optimizer's [0, 1] scale is
    #   ``(a - image_min) / intensity_range``.
    # * True — BACKGROUND-RELATIVE: amplitudes that already have the pedestal
    #   removed, as a previous fit's output does. ``finalize_results`` multiplies
    #   the normalized amplitudes by ``intensity_range`` and deliberately does
    #   NOT add ``image_min`` back (see ``results.py``), so rescaling is
    #   ``a / intensity_range`` — subtracting ``image_min`` again would remove
    #   the floor a SECOND time and zero every sub-floor seed (#1172).
    #
    # Set by the CALL SITES. ``_extract_gsplatdata_init`` cannot tell which kind
    # of ``GSplatData`` it was handed, and neither can ``preprocess_data``: the
    # ``seeds=GSplatData`` door carries BOTH a previous fit's output (background-
    # relative) and ``generate_seeds()`` output (raw — the documented explicit-
    # seeding workflow), and a bare GSplatData records no provenance. Only the
    # caller knows, so that branch reads ``config.seed_amps_background_relative``.
    # The seeding path inside ``_generate_seeds`` is unambiguous (it samples the
    # still-raw volume itself) and pins the flag to False.
    init_amps_background_relative: bool = False


def _rescale_init_amps(
    init_ctx: _InitContext,
    image_min: float,
    intensity_range: float,
    verbose: bool,
) -> None:
    """Rescale pre-initialized amplitudes to the normalized image scale, in place.

    Optimization works on the normalized [0, 1] image; without this rescaling
    ``amp_max`` constraints would be on the wrong scale. WHICH rescaling applies
    depends on the amplitude convention (see
    ``_InitContext.init_amps_background_relative``): a warm start from a previous
    fit already has the pedestal removed, so subtracting ``image_min`` again
    would remove the floor twice and zero every sub-floor seed (#1172); a
    raw-image-sampled array still carries it.

    Residual approximation (out of scope): a re-fit resolves its OWN
    ``intensity_range``, which need not be byte-identical to the one the seed was
    produced under, so the warm start is exact only when both fits resolve the
    same normalization.

    A no-op when there are no pre-initialized amplitudes.
    """
    if init_ctx.init_amps is None:
        return

    if init_ctx.init_amps_background_relative:
        init_ctx.init_amps = np.clip(init_ctx.init_amps / intensity_range, 0.0, 1.0)
    else:
        init_ctx.init_amps = np.clip(
            (init_ctx.init_amps - image_min) / intensity_range, 0.0, 1.0
        )

    if verbose:
        # min()/max() have no identity on an empty array, so report the count.
        if init_ctx.init_amps.size == 0:
            aprint("Rescaled init_amps to normalized range: 0 seeds")
        else:
            aprint(
                f"Rescaled init_amps to normalized range: "
                f"[{init_ctx.init_amps.min():.4f}, {init_ctx.init_amps.max():.4f}]"
            )


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
    # ``config.init_amps`` follows the raw-image-sampled convention (see
    # FitConfig.init_amps), hence the default background_relative=False.
    init_ctx = _InitContext(
        init_L=config.init_L.copy() if config.init_L is not None else None,
        init_amps=config.init_amps.copy() if config.init_amps is not None else None,
    )

    # Handle GSplatData seeds specially
    if isinstance(seeds, GSplatData):
        seed_centers = seeds.centers.copy()
        # Extract pre-initialized parameters from GSplatData
        _extract_gsplatdata_init(init_ctx, seeds)
        # Which amplitude convention those extracted amplitudes are in is the
        # CALLER's declaration: this door carries both a previous fit's output
        # (background-relative — see `results.py`) and `generate_seeds()` output
        # (raw-image-sampled), and a bare GSplatData records no provenance.
        # Overwrites whatever convention config.init_amps had — the extractor
        # replaced the array.
        init_ctx.init_amps_background_relative = config.seed_amps_background_relative
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
        (
            V_normalized,
            image_min,
            image_max,
            intensity_range,
            applied_floor,
            floor_strategy,
        ) = _normalize_data_with_strategy(
            V,
            config.norm_percentile,
            config.verbose,
            config.floor,
            config.norm_range,
        )

    # Rescale pre-initialized amplitudes to match normalized image scale
    # (convention-dependent — see _rescale_init_amps).
    _rescale_init_amps(init_ctx, image_min, intensity_range, config.verbose)

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
        floor_strategy=floor_strategy,
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
    # - With target_seeds: `auto` splits it 60% edges / 40% grid (_auto_combine)
    # - Without: `auto` invents its own budget first —
    #   max(100, prod(shape)**(1/ndim) / 2), capped at 10k — then splits that 60/40
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
        # generate_seeds() sampled these amplitudes off the volume BEFORE
        # normalization (V is still raw here), so they carry the pedestal and the
        # `- image_min` rescaling is the correct one. Same convention as the grid
        # fallback amplitudes appended by _extend_init_arrays_for_grid_seeds
        # below, which is why concatenating them stays provenance-consistent.
        init_ctx.init_amps_background_relative = False
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
    from luxar.gsplats.spatial_hash import BatchedSpatialHashGrid

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


def _regular_grid_coords(ranges: list[np.ndarray], ndim: int) -> np.ndarray:
    """Cartesian product of per-axis ranges as an ``(N, ndim)`` float array.

    Always ``(N, ndim)``, including when N is 0. That is the whole point: an
    empty ``itertools.product`` fed to ``np.array`` collapses to shape ``(0,)``,
    and the spatial-hash query rejects that with
    "query must have shape (Q, 3); got (0,)" rather than treating it as an empty
    point set. It is reached whenever ``spacing // 2`` lands past the end of any
    axis — easy on an anisotropic tile, where a spacing derived from the total
    volume can exceed the short axis outright — and it took down a whole tiled
    fit 29 minutes in, on the last tile, after all the real work was done.
    """
    import itertools

    coords = np.array(list(itertools.product(*ranges)), dtype=float)
    return coords if coords.size else np.empty((0, ndim), dtype=float)


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
    ranges = [np.arange(spacing // 2, s, spacing) for s in shape]

    grid_coords: np.ndarray = _regular_grid_coords(ranges, ndim)

    # Remove grid points too close to existing seeds (if any exist)
    # But be less aggressive about filtering to ensure we get enough
    # (skip entirely when there is nothing to filter — querying an empty set is
    # both wasteful and, historically, fatal).
    if len(existing_seeds) > 0 and len(grid_coords) > 0:
        from luxar.gsplats.spatial_hash import BatchedSpatialHashGrid

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
        grid_coords = _regular_grid_coords(ranges_dense, ndim)

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


def _resolve_floor_result(
    V: np.ndarray, floor: "str | float | None"
) -> "FloorEstimate | None":
    """Resolve a floor spec while retaining estimator provenance."""
    from luxar.gsplats.calibration import FloorEstimate

    if floor is None:
        return None
    if isinstance(floor, str):
        f = floor.strip().lower()
        if f in ("none", ""):
            return None
        if f in ("auto", "specimen"):
            from luxar.gsplats.calibration import estimate_floor_result

            method = "mode" if f == "auto" else "specimen"
            return estimate_floor_result(V, method=method)
        if f.startswith("p"):
            pct = float(f[1:])
            V = np.asarray(V)
            values = V[V != 0.0] if np.any(V != 0.0) else V
            return FloorEstimate(float(np.percentile(values, pct)), "percentile")
        value = float(f)
    else:
        value = float(floor)
    if value == 0.0:
        return None
    return FloorEstimate(value, "numeric")


def _resolve_floor(V: np.ndarray, floor: "str | float | None") -> "float | None":
    """Resolve a ``floor`` spec to a concrete background level, or ``None``.

    ``None`` means "no explicit floor" — the caller keeps its default
    ``image_min`` (hard ``min``/``norm_percentile``). Accepted forms:

    - ``"auto"`` → histogram-mode estimate (see :func:`estimate_floor`).
    - ``"specimen"`` → compact upper background mode, or ``auto`` fallback.
    - ``"pN"`` (e.g. ``"p10"``) → the Nth percentile of non-zero intensities.
    - ``"none"`` / ``"0"`` / ``0`` / ``None`` → disabled (returns ``None``).
    - ``float`` / numeric string → that fixed intensity value.
    """
    result = _resolve_floor_result(V, floor)
    return None if result is None else float(result.level)


# Sampling budget for resolve_volume_floor: the floor level is estimated from
# at most this many voxels (~128 MB as float32), drawn as a bounded number of
# evenly spaced contiguous slabs along the volume's LONGEST axis. If one full
# cross-section is already too large, the slab is deterministically cropped
# along the remaining axes. These contiguous reads are cheap on chunked zarr
# stores, unlike a stride which touches essentially every chunk.
FLOOR_SAMPLE_BUDGET_VOXELS = 32_000_000
# Maximum number of evenly spaced contiguous sample blocks along the sampled axis.
_FLOOR_SAMPLE_BLOCKS = 32


def _centered_base_slices(
    shape: tuple[int, ...], block_shape: list[int], axis: int
) -> list[slice]:
    """Center-crop slices for every axis except ``axis`` (the sampled one)."""
    base_slices: list[slice] = [slice(None)] * len(shape)
    for i, (full_len, sample_len) in enumerate(zip(shape, block_shape, strict=True)):
        if i == axis or sample_len == full_len:
            continue
        start = (full_len - sample_len) // 2
        base_slices[i] = slice(start, start + sample_len)
    return base_slices


def _sample_volume_for_floor(volume: Any, budget: int) -> "np.ndarray | None":
    """Read a bounded, deterministic sample of ``volume`` as flat float32.

    Samples evenly spaced contiguous slab blocks along the **longest** axis
    (ties -> lowest index), so a small leading axis — e.g. an unsqueezed
    ``(1, Z, Y, X)`` store — cannot defeat the budget the way hard-coded
    axis-0 slabs would. If one complete slab exceeds the budget, its remaining
    axes are recursively center-cropped, longest first, until one slab fits.
    When the budget allows only one block, it is placed at the middle of the
    sampled axis (the first slab is systematically biased). The sample is a
    pure function of ``volume.shape`` and ``budget``. Returns ``None`` for an
    empty volume.
    """
    if budget < 1:
        raise ValueError("floor sample budget must be at least 1 voxel")

    shape = tuple(int(s) for s in volume.shape)
    total = 1
    for s in shape:
        total *= s
    if total == 0:
        return None
    if total <= budget:
        return np.asarray(volume[...], dtype=np.float32).ravel()

    axis = shape.index(max(shape))  # longest axis; ties -> lowest index

    # Start with one full cross-section perpendicular to `axis`. If that alone
    # exceeds the budget, center-crop the longest remaining dimensions until
    # the cross-section fits. Recursive cropping is needed for high-dimensional
    # shapes where reducing only the second-longest axis to one is insufficient.
    sample_shape = list(shape)
    sample_shape[axis] = 1
    slab_voxels = total // shape[axis]
    while slab_voxels > budget:
        crop_axis = max(
            (i for i, length in enumerate(sample_shape) if i != axis and length > 1),
            key=lambda i: (sample_shape[i], -i),
        )
        other_voxels = slab_voxels // sample_shape[crop_axis]
        sample_shape[crop_axis] = max(1, budget // other_voxels)
        slab_voxels = other_voxels * sample_shape[crop_axis]

    n_slabs = min(shape[axis], max(1, budget // slab_voxels))
    n_blocks = min(_FLOOR_SAMPLE_BLOCKS, n_slabs)
    block_len = n_slabs // n_blocks
    span = shape[axis] - block_len
    if n_blocks == 1:
        # A single block is read from the MIDDLE of the axis: the first slab
        # of a stack is systematically atypical (vignetting, empty leading
        # planes, axial intensity gradients).
        starts = [span // 2]
    else:
        starts = sorted(
            {int(round(span * i / (n_blocks - 1))) for i in range(n_blocks)}
        )

    base_slices = _centered_base_slices(shape, sample_shape, axis)

    samples = []
    for start in starts:
        region = base_slices.copy()
        region[axis] = slice(start, start + block_len)
        samples.append(np.asarray(volume[tuple(region)], dtype=np.float32).ravel())
    return np.concatenate(samples)


# Smallest span a resolved normalization range may report. Reached only when a
# subtracted floor sits at or above the sampled top, i.e. the sample says the
# whole volume is pedestal; callers treat a span this small as "no usable shared
# scale" rather than as a real range (see `_tile_norm_range`).
NORM_RANGE_MIN_SPAN = 1e-12


def _norm_range_has_usable_span(norm_range: tuple[float, float]) -> bool:
    """Whether a resolved shared range is safe to forward to another fit."""
    lo, hi = float(norm_range[0]), float(norm_range[1])
    span = hi - lo
    return (
        np.isfinite(lo)
        and np.isfinite(hi)
        and np.isfinite(span)
        and span > NORM_RANGE_MIN_SPAN
    )


def resolve_volume_norm_range(
    volume: Any,
    norm_percentile: float,
    *,
    subtract: float | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Resolve the normalization range against a whole volume.

    The intensity-scale counterpart of :func:`resolve_volume_floor`, and it
    exists for the same reason. A tiled fit hands each worker one tile; if the
    tile is normalized by its OWN min/max then each tile is stretched to fill
    [0, 1] by a different factor. Output amplitudes are rescaled by that same
    factor afterwards, so the *physical* amplitude of a linear fit largely
    cancels out — what does NOT cancel is everything the optimiser expresses
    as an absolute quantity in the normalized range: the convergence tolerance
    (``max_abs_error``, 1% of it by default), seeding and culling thresholds,
    and any ``amp_max``. A dim tile is therefore resolved to a much finer
    physical accuracy than a bright one, and the two tiles' splats are not
    mutually comparable. Sharing one range makes a tiled fit behave like the
    whole-volume fit it is meant to approximate.

    The flip side is deliberate: a tile far dimmer than the volume maximum is
    now held to the same ABSOLUTE tolerance as the rest of the volume, so it
    converges earlier instead of resolving its own noise at full contrast.

    Parameters
    ----------
    volume : np.ndarray or zarr.Array
        Full volume (may be a lazy zarr array; only a bounded sample is read,
        via the same budget and block layout as :func:`resolve_volume_floor`).
    norm_percentile : float
        0 for full min-max; otherwise the low/high percentile pair, exactly as
        :func:`_normalize_data` interprets it.
    subtract : float, optional
        A level already subtracted from the tile before fitting (the resolved
        floor). The returned range is shifted to match, since the fit sees
        post-subtraction data. Clamped at 0 like the tile's own clip.
    verbose : bool, default False
        Print the resolved range via arbol.

    Returns
    -------
    tuple[float, float]
        ``(image_min, image_max)`` to hand to every tile of this volume.

    Notes
    -----
    Determinism matters as much as it does for the floor: the sample is a pure
    function of ``volume.shape`` and the fixed budget, so independent workers
    (``--tile k/M``, ``-j N``) resolve the SAME range for the volume they are
    HANDED, without coordinating. Batch-fit instead resolves one range across
    its bounded plan-time ``(t, c)`` samples, records it in the manifest, and
    forwards it to every task, so spatial and temporal children share the same
    normalization scale.
    """
    sample = _sample_volume_for_floor(volume, int(FLOOR_SAMPLE_BUDGET_VOXELS))
    if sample is None or sample.size == 0:
        return (0.0, 1.0)
    if norm_percentile == 0.0:
        lo, hi = float(np.min(sample)), float(np.max(sample))
    else:
        lo = float(np.percentile(sample, norm_percentile))
        hi = float(np.percentile(sample, 100.0 - norm_percentile))
    if subtract is not None:
        lo = max(0.0, lo - float(subtract))
        hi = max(lo + NORM_RANGE_MIN_SPAN, hi - float(subtract))
    if verbose:
        aprint(f"Whole-volume normalization range: [{lo:.6g}, {hi:.6g}]")
    return (lo, hi)


def resolve_volume_norm_range_denoised(
    volume: Any,
    norm_percentile: float,
    *,
    denoise_h: float | None,
    denoise_params: dict[str, Any] | None,
    subtract: float | None = None,
    probe_cache: dict[str, Any] | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Resolve the shared normalization range on the data tiles will fit.

    With denoising disabled this is exactly :func:`resolve_volume_norm_range`.
    Otherwise, when the whole volume fits the bounded probe budget, the raw
    whole-volume range is shifted by the denoise-induced
    endpoint change measured on the deterministic shape-preserving probe used
    for floor correction.  When the volume fits the probe budget, the probe is
    the whole volume and the result exactly matches resolving after a full
    denoise, as the non-tiled path does. Above that budget the raw range is kept:
    a bounded max-shift did not converge in measurement and is not worth an NLM pass.
    """
    raw_lo, raw_hi = resolve_volume_norm_range(volume, norm_percentile)
    if (
        denoise_h is None
        or denoise_params is None
        or not _volume_fits_probe_budget(volume, DENOISE_PROBE_BUDGET_VOXELS)
    ):
        lo, hi = raw_lo, raw_hi
    else:
        try:
            probe = _denoise_probe_arrays(
                volume, float(denoise_h), denoise_params, probe_cache
            )
            if probe is None:
                lo, hi = raw_lo, raw_hi
            else:
                raw_probe, denoised_probe = probe
                if norm_percentile == 0.0:
                    probe_raw_lo = float(np.min(raw_probe))
                    probe_raw_hi = float(np.max(raw_probe))
                    probe_denoised_lo = float(np.min(denoised_probe))
                    probe_denoised_hi = float(np.max(denoised_probe))
                else:
                    probe_raw_lo = float(np.percentile(raw_probe, norm_percentile))
                    probe_raw_hi = float(
                        np.percentile(raw_probe, 100.0 - norm_percentile)
                    )
                    probe_denoised_lo = float(
                        np.percentile(denoised_probe, norm_percentile)
                    )
                    probe_denoised_hi = float(
                        np.percentile(denoised_probe, 100.0 - norm_percentile)
                    )
                lo = probe_denoised_lo + (raw_lo - probe_raw_lo)
                hi = probe_denoised_hi + (raw_hi - probe_raw_hi)
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = raw_lo, raw_hi
        except Exception as exc:
            aprint(
                "Denoised normalization-range probe failed "
                f"({exc}); keeping the raw-basis range."
            )
            lo, hi = raw_lo, raw_hi

    if subtract is not None:
        lo = max(0.0, lo - float(subtract))
        hi = max(lo + NORM_RANGE_MIN_SPAN, hi - float(subtract))
    if verbose:
        aprint(f"Whole-volume normalization range: [{lo:.6g}, {hi:.6g}]")
    return (lo, hi)


def _floor_level_and_sample_max(
    volume: Any,
    floor: "str | float | None",
    *,
    guard_numeric: bool = False,
    sample_budget: int | None = None,
) -> "tuple[float | None, float | None, str | None]":
    """The resolved whole-volume floor level AND the sampled max it was judged on.

    The shared body of :func:`resolve_volume_floor` (which drops the max) and
    :func:`resolve_volume_floor_denoised` (which re-uses it to guard its own
    corrected level on the SAME basis, without a second bounded read). Every
    message, guard and return value is the public function's — see there.

    Returns ``(level, sample_max, strategy)``. ``sample_max`` is ``None`` whenever no
    sample was drawn (a spec that needs no data, or an empty volume) or the spec
    resolved to "nothing to subtract" before the guard was reached. A non-``None``
    level with a ``None`` max therefore identifies exactly one case — the
    read-free numeric short-circuit — which is how :func:`resolve_volume_floor`
    keeps that path SILENT, as it was before this body was split out.
    """
    if floor is None:
        return None, None, None
    needs_data = guard_numeric
    if isinstance(floor, str):
        f = floor.strip().lower()
        if f in ("none", ""):
            return None, None, None
        # One predicate for "measured ON the volume", so this function and the
        # denoise-basis regime rule below can never disagree about a spec.
        if _floor_spec_is_volume_derived(floor):
            needs_data = True
    if not needs_data:
        # Numeric spec: echo the constant back — never sample the volume.
        # 0 disables; a negative level is legitimate (see Notes).
        result = _resolve_floor_result(np.empty(0, dtype=np.float32), floor)
        return (
            None if result is None else float(result.level),
            None,
            None,
        )

    budget = FLOOR_SAMPLE_BUDGET_VOXELS if sample_budget is None else sample_budget
    sample = _sample_volume_for_floor(volume, int(budget))
    if sample is None:
        return None, None, None

    result = _resolve_floor_result(sample, floor)
    if result is None:
        return None, None, None
    resolved = float(result.level)
    sample_max = float(sample.max())
    if resolved >= sample_max:
        aprint(
            f"Warning: floor {resolved:.6g} >= sampled volume max "
            f"{sample_max:.6g}; ignoring (would erase all signal)."
        )
        return None, sample_max, None
    strategy = (
        result.strategy
        if isinstance(floor, str) and floor.strip().lower() == "specimen"
        else None
    )
    return resolved, sample_max, strategy


def resolve_volume_floor(
    volume: Any,
    floor: "str | float | None",
    *,
    guard_numeric: bool = False,
    sample_budget: "int | None" = None,
    verbose: bool = False,
) -> "float | None":
    """Resolve a ``floor`` spec against a whole volume, without loading it all.

    The whole-volume counterpart of :func:`_resolve_floor` for tiled fitting:
    the returned level is a property of the *volume*, never of any tile, so
    independent workers (``--tile k/M``, ``-j N``, batch-fit) all subtract one
    identical pedestal.

    Parameters
    ----------
    volume : np.ndarray or zarr.Array
        Full volume (may be a lazy zarr array; only a bounded sample is read).
    floor : str, float, or None
        Floor spec (see :func:`_resolve_floor`). A numeric spec (float or
        numeric string) short-circuits and is echoed back without touching
        the volume — unless ``guard_numeric`` is set; ``"none"``/``None``/``0``
        return ``None``.
    guard_numeric : bool, default False
        Also apply the "floor >= max would erase all signal" guard to a
        numeric spec (one bounded sample read). Pass ``True`` where a
        USER-supplied spec is first turned into a level; leave ``False`` for
        levels already resolved and guarded upstream (e.g. the concrete level
        the parent hands each tile worker), preserving the read-free
        short-circuit.
    sample_budget : int, optional
        Override the bounded sample voxel budget. ``None`` uses
        :data:`FLOOR_SAMPLE_BUDGET_VOXELS`.
    verbose : bool, default False
        Print the resolved level via arbol.

    Returns
    -------
    float or None
        The concrete background level to subtract, or ``None`` (disabled,
        nothing to subtract, or the guard below refused the level).

    Notes
    -----
    - **Memory bound**: at most :data:`FLOOR_SAMPLE_BUDGET_VOXELS` voxels are
      sampled, as evenly spaced contiguous slab blocks along the volume's
      longest axis. If one full cross-section exceeds the budget, it is
      deterministically center-cropped along the remaining axes until it fits.
      A volume within the budget is read whole.
    - **Determinism**: the sample is a pure function of ``volume.shape`` and
      the fixed budget, so two independent processes given the same volume
      and spec always resolve the same level.
    - A **negative** resolved level (dark-frame-corrected / deconvolved data
      with a negative background) is returned like any other: floor
      suppression means "put the background at 0", so a background sitting at
      ``-2`` is shifted up by ``V - (-2)`` — exactly what the non-tiled
      path's ``image_min = max(resolved_floor, image_min)`` does when
      ``resolved_floor`` is negative.
    - The "floor >= max would erase all signal" guard is applied against the
      **sampled** max: such a level is refused with an ``aprint`` warning and
      ``None`` is returned. For numeric specs the guard runs only with
      ``guard_numeric=True``.
    """
    resolved, sample_max, _ = _floor_level_and_sample_max(
        volume,
        floor,
        guard_numeric=guard_numeric,
        sample_budget=sample_budget,
    )
    if resolved is None:
        return None
    # A read-free numeric short-circuit (level resolved, no sample drawn) stays
    # SILENT: nothing was measured against the volume, so there is nothing to
    # report about it. This mirrors the early `return` the numeric branch had
    # before the body moved into `_floor_level_and_sample_max`, keeping the log
    # surface identical for every input class.
    if verbose and sample_max is not None:
        aprint(f"Resolved whole-volume background floor: {resolved:.6g}")
    return float(resolved)


def resolve_volume_floor_with_strategy(
    volume: Any,
    floor: "str | float | None",
    *,
    guard_numeric: bool = False,
    sample_budget: "int | None" = None,
    verbose: bool = False,
) -> "tuple[float | None, str | None]":
    """Resolve specimen level and branch from the same bounded sample."""
    resolved, sample_max, strategy = _floor_level_and_sample_max(
        volume,
        floor,
        guard_numeric=guard_numeric,
        sample_budget=sample_budget,
    )
    if verbose and resolved is not None and sample_max is not None:
        aprint(f"Resolved whole-volume background floor: {resolved:.6g}")
    return resolved, strategy


# Voxel budget for the denoise-correction probe of
# :func:`resolve_volume_floor_denoised`: at most this many voxels are denoised a
# second time to measure the shift denoising induces on the floor estimate.
# Deliberately ~16x smaller than FLOOR_SAMPLE_BUDGET_VOXELS, because this sample
# is not merely READ but run through NLM. A volume within the budget is probed
# WHOLE (as one block), which is what makes the corrected level exactly equal to
# the denoised-whole-volume estimate on small volumes.
DENOISE_PROBE_BUDGET_VOXELS = 2_000_000
# Number of evenly spaced probe blocks along the volume's longest axis. Three
# blocks (start / middle / end) span axial gradients without turning the probe
# into a second full denoise pass.
_DENOISE_PROBE_BLOCKS = 3


def _floor_spec_is_volume_derived(floor: "str | float | None") -> bool:
    """Whether resolving this ``floor`` spec has to look at the data.

    ``"auto"``, ``"specimen"`` and ``"pNN"`` are measured ON the volume; everything else
    (``None``, ``"none"``, a number, a numeric string) is a user absolute that
    no measurement may move. Mirrors the spec branching of
    :func:`resolve_volume_floor`.
    """
    if not isinstance(floor, str):
        return False
    f = floor.strip().lower()
    return f in ("auto", "specimen") or f.startswith("p")


def _floor_spec_is_percentile(floor: "str | float | None") -> bool:
    """Whether this ``floor`` spec is a ``pNN`` percentile.

    The one volume-derived spec whose denoise-induced shift survives being
    measured on a bounded crop — see :func:`resolve_volume_floor_denoised`'s
    Notes for the measurements that decide this.
    """
    return isinstance(floor, str) and floor.strip().lower().startswith("p")


def _volume_fits_probe_budget(volume: Any, budget: int) -> bool:
    """Whether the denoise probe of this volume IS the whole volume.

    ``<=`` and not ``<``, matching :func:`_sample_blocks_for_denoise_probe`
    exactly: a volume of exactly ``budget`` voxels is returned as one whole
    block, so the correction measured on it is the denoised-whole-volume
    estimate. Reads nothing — ``volume.shape`` is enough, which is what lets the
    caller skip the probe (and its NLM pass) entirely.
    """
    total = 1
    for s in volume.shape:
        total *= int(s)
    return total <= budget


def _cubic_block_shape(shape: tuple[int, ...], per_block: int) -> list[int]:
    """Shrink ``shape`` toward a cube until one block fits ``per_block`` voxels.

    Halves the longest dimension (ties -> lowest index) repeatedly, so the probe
    block keeps real neighbourhood context along EVERY axis — a single plane
    would be meaningless input for 3D NLM. Stops early if nothing can shrink
    further (every dimension already 1).
    """
    block_shape = list(shape)
    while True:
        voxels = 1
        for s in block_shape:
            voxels *= s
        if voxels <= per_block:
            return block_shape
        longest = max(range(len(block_shape)), key=lambda i: (block_shape[i], -i))
        if block_shape[longest] <= 1:
            return block_shape
        block_shape[longest] = max(1, block_shape[longest] // 2)


def _sample_blocks_for_denoise_probe(
    volume: Any,
    budget: int,
    n_blocks: int = _DENOISE_PROBE_BLOCKS,
) -> "list[np.ndarray] | None":
    """Read a bounded, deterministic, SHAPE-PRESERVING probe of ``volume``.

    Unlike :func:`_sample_volume_for_floor` (which returns one flat array,
    because a floor estimator only needs values), this returns whole nD blocks:
    the probe is handed to 3D NLM, which needs real neighbourhood context along
    every axis. Hence the blocks are cropped toward a roughly cubic shape —
    repeatedly halving the longest dimension — rather than being thin slabs.

    Up to ``n_blocks`` evenly spaced contiguous blocks are read along the
    volume's **longest** axis (ties -> lowest index), center-cropped in the
    other axes; contiguous reads are cheap on chunked zarr stores. The sample is
    a pure function of ``volume.shape``, ``budget`` and ``n_blocks``, so
    independent workers (``--tile k/M``, ``-j N``) probe identically without
    coordinating.

    A volume within ``budget`` is returned as ONE block containing the whole
    volume. Returns ``None`` for an empty volume.
    """
    if budget < 1:
        raise ValueError("denoise probe budget must be at least 1 voxel")

    shape = tuple(int(s) for s in volume.shape)
    total = 1
    for s in shape:
        total *= s
    if total == 0:
        return None
    if total <= budget:
        return [np.asarray(volume[...], dtype=np.float32)]

    axis = shape.index(max(shape))  # longest axis; ties -> lowest index
    per_block = max(1, budget // max(1, n_blocks))

    block_shape = _cubic_block_shape(shape, per_block)
    block_len = block_shape[axis]
    span = shape[axis] - block_len
    n_used = max(1, min(n_blocks, shape[axis] // max(1, block_len)))
    if n_used == 1:
        # A single block comes from the MIDDLE of the axis: the first slab of a
        # stack is systematically atypical (vignetting, empty leading planes).
        starts = [span // 2]
    else:
        starts = sorted({int(round(span * i / (n_used - 1))) for i in range(n_used)})

    base_slices = _centered_base_slices(shape, block_shape, axis)
    blocks = []
    for start in starts:
        region = base_slices.copy()
        region[axis] = slice(start, start + block_len)
        blocks.append(np.asarray(volume[tuple(region)], dtype=np.float32))
    return blocks


def _denoise_probe_arrays(
    volume: Any,
    denoise_h: float,
    denoise_params: "dict[str, Any]",
    probe_cache: "dict[str, Any] | None" = None,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Read and denoise the bounded probe once, optionally caching both arrays."""
    if probe_cache is not None and "raw" in probe_cache:
        return probe_cache["raw"], probe_cache["denoised"]

    from luxar.gsplats.preprocessing.denoise_pipeline import denoise_volume_array

    blocks = _sample_blocks_for_denoise_probe(volume, int(DENOISE_PROBE_BUDGET_VOXELS))
    if not blocks:
        return None
    denoised = [
        denoise_volume_array(block, h=float(denoise_h), **denoise_params)
        for block in blocks
    ]
    raw_flat = np.concatenate([block.ravel() for block in blocks])
    denoised_flat = np.concatenate(
        [np.asarray(block, dtype=np.float32).ravel() for block in denoised]
    )
    # Check the OUTPUT, not just the input: estimate_floor on NaN-bearing data
    # can return a finite, plausible ~0.001 and silently disable suppression.
    if not bool(np.isfinite(denoised_flat).all()):
        raise ValueError("denoise probe produced non-finite values")
    if probe_cache is not None:
        probe_cache["raw"] = raw_flat
        probe_cache["denoised"] = denoised_flat
    return raw_flat, denoised_flat


def _denoise_probe_correction(
    volume: Any,
    floor: "str | float | None",
    level_raw: float,
    denoise_h: float,
    denoise_params: "dict[str, Any]",
    probe_cache: "dict[str, Any] | None" = None,
) -> "tuple[float, float, int, str | None] | None":
    """Shift ``level_raw`` onto the denoised basis with a bounded probe.

    Returns ``(level, delta, probe_voxels, strategy)``, or ``None`` when there is
    nothing trustworthy to correct WITH — an unreadable probe, a raising
    denoiser, or a degenerate or non-finite estimate. Every ``None`` says why out
    loud, and the caller then keeps ``level_raw``: this function degrades, it
    never raises.

    The whole probe pipeline — the READ, the denoise pass and both estimator
    calls — is inside one ``try``, and the denoised probe is checked for
    finiteness before either estimator runs. Checking the probe OUTPUT rather than
    the resulting level is the point: a NaN does not reliably propagate to the
    level, so a level test would miss the worst case. Measured, one NaN in the
    probe makes ``auto`` return 0.0009765625 — finite, plausible, and floor
    suppression effectively off for the whole run; a NaN ``pNN`` level instead
    makes every tile all-NaN and the fit dies later blaming the input data; and a
    ``-inf`` raises out of ``np.histogram``. With the probe finite the only way
    out is a non-finite ``level_raw``, which needs a non-finite INPUT volume — a
    ``pNN`` spec over a NaN-bearing volume already resolves to NaN in
    :func:`resolve_volume_floor`, tiled or not, so there is nothing here for a
    second test to improve. Today's kernels emit none of this; it is insurance.
    """
    try:
        probe = _denoise_probe_arrays(
            volume, float(denoise_h), denoise_params, probe_cache
        )
        if probe is None:
            # Defensive only: an empty volume has no floor sample either, so
            # `level_raw` never got this far. Kept so a future sampler change
            # cannot turn "no probe" into a TypeError mid-fit.
            aprint(
                "Note: denoise floor probe found no data; keeping the raw-basis "
                f"background level {level_raw:.6g}."
            )
            return None
        raw_flat, denoised_flat = probe
        raw_result = _resolve_floor_result(raw_flat, floor)
        denoised_result = _resolve_floor_result(denoised_flat, floor)
    except Exception as exc:  # noqa: BLE001 - never let a probe break a fit
        aprint(
            f"Note: denoise floor probe failed ({type(exc).__name__}: {exc}); "
            f"keeping the raw-basis background level {level_raw:.6g}."
        )
        return None

    if raw_result is None or denoised_result is None:
        # Defensive only: a volume-derived spec ("auto"/"pNN") always resolves to
        # a number on a non-empty array, and an empty one already returned above.
        aprint(
            "Note: denoise floor probe is degenerate (no floor estimable); "
            f"keeping the raw-basis background level {level_raw:.6g}."
        )
        return None

    probe_raw = float(raw_result.level)
    probe_denoised = float(denoised_result.level)
    delta = probe_denoised - probe_raw
    if delta == 0.0:
        # Nothing to correct (denoising left this estimator's answer alone, or
        # the probe is constant). No note: the two bases agree. Returning here
        # rather than computing `level` keeps `level_raw` BIT-exact — the
        # reconstruction below is only exactly the identity for a `delta` of 0
        # when no rounding creeps into the two additions.
        strategy = (
            denoised_result.strategy
            if isinstance(floor, str) and floor.strip().lower() == "specimen"
            else None
        )
        return float(level_raw), 0.0, int(raw_flat.size), strategy

    # Algebraically ``level_raw + delta``, grouped so that the whole-volume-probe
    # case is EXACT: there ``probe_raw == level_raw`` bit for bit (same
    # estimator, same values), the offset is a hard 0.0, and the result is the
    # denoised-whole-volume estimate itself rather than a rounded reconstruction.
    level = probe_denoised + (float(level_raw) - probe_raw)
    strategy = (
        denoised_result.strategy
        if isinstance(floor, str) and floor.strip().lower() == "specimen"
        else None
    )
    return level, delta, int(raw_flat.size), strategy


def resolve_volume_floor_denoised_with_strategy(
    volume: Any,
    floor: "str | float | None",
    *,
    denoise_h: "float | None" = None,
    denoise_params: "dict[str, Any] | None" = None,
    guard_numeric: bool = False,
    sample_budget: int | None = None,
    probe_cache: dict[str, Any] | None = None,
    verbose: bool = False,
) -> "tuple[float | None, str | None]":
    """Resolve a denoised-basis floor and its specimen estimator branch.

    The tiled paths denoise each tile and then subtract a global level, while
    the non-tiled path denoises the whole volume and estimates the level from
    THAT. Denoising collapses the noise tail and shifts the histogram mode, so
    resolving on the raw volume and subtracting from denoised tiles removes a
    measurably different pedestal than ``--tiling none`` does on the same input
    (#1178). Estimating on denoised data is the better default — the mode
    estimator is more reliable once the tail is collapsed — so this function
    keeps :func:`resolve_volume_floor`'s whole-volume basis (one global level,
    the #1174 invariant) and applies the denoise-induced CORRECTION measured on
    a small bounded probe, **wherever that shift can actually be measured**: on a
    volume within the probe budget always, and above it only for a ``pNN`` spec.
    See the Notes for the two regimes and the measurements behind them.

    Parameters
    ----------
    volume : np.ndarray or zarr.Array
        Full volume (may be lazy; only bounded samples are read).
    floor : str, float, or None
        Floor spec, exactly as :func:`resolve_volume_floor` interprets it. Only
        a VOLUME-DERIVED spec (``"auto"`` / ``"pNN"``) is ever corrected; a
        numeric spec or ``"none"`` is a user absolute and passes through
        untouched. Above the probe budget only ``"pNN"`` is corrected.
    denoise_h : float, optional
        NLM filtering strength the tiles will be denoised with. ``None``
        (denoise off) delegates to :func:`resolve_volume_floor` verbatim.
    denoise_params : dict, optional
        The remaining ``denoise_volume_array`` keyword arguments
        (``patch_size``, ``search_distance``, ``backend``, ``device``,
        ``use_2d``, ``norm_range``), passed **verbatim** so the probe is
        smoothed exactly as the tiles are. ``None`` delegates like
        ``denoise_h=None``.
    guard_numeric : bool, default False
        Forwarded to :func:`resolve_volume_floor` (see there).
    sample_budget : int, optional
        Override the bounded raw floor-sample voxel budget. ``None`` uses
        :data:`FLOOR_SAMPLE_BUDGET_VOXELS`.
    verbose : bool, default False
        Print the raw level, the correction and the final level. Forwarded to
        :func:`resolve_volume_floor` on the paths that delegate to it.

    Returns
    -------
    float or None
        The concrete level every tile should subtract from its DENOISED data,
        or ``None`` (disabled, or a guard refused the level).

    Notes
    -----
    - **Two regimes, one measured rule.** The correction is applied where it is
      demonstrably right, and not applied where it is not:

      1. The probe covers the WHOLE volume (``total <=``
         :data:`DENOISE_PROBE_BUDGET_VOXELS`). The corrected level then IS the
         denoised-whole-volume estimate, bit for bit — the same estimator over
         the same values — so it is applied for any volume-derived spec. This is
         what makes tiled/non-tiled parity exact on small volumes.
      2. Above the budget the probe is a handful of cubic centre crops, and
         whether its shift transfers depends on the ESTIMATOR. A ``pNN``
         percentile shift does; the ``auto`` histogram-mode shift does not, and
         is therefore not applied at all — the raw-basis level is kept (exactly
         the pre-#1178 behaviour) and one note says so. No probe is denoised in
         that case, so the skip costs nothing.

    - **What was measured** (synthetic 24x64x64 stacks with a known pedestal,
      six background families x six seeds, production denoise params including
      the whole-volume ``norm_range``, probe at 4.7% of the volume; error =
      ``|level - reference|`` against the reference ``--tiling none`` computes,
      ``_resolve_floor(denoise_whole(volume), spec)``):

      * ``pNN`` (``p10``): mean error 4.386 raw -> **1.408** corrected, closer on
        28/36 volumes, and the worst family (Poisson) goes from a mean 11.656 to
        1.644 (worst single volume 2.091). Applied.
      * ``auto``: mean error 1.309 raw -> 0.830 corrected, but closer on only
        21/36 volumes — the sign is close to a coin flip. It wins big on the two
        families whose true shift is large (gamma-skewed and masked pedestals,
        ~2.6 -> ~0.6) and loses on the four whose true shift is ~0.2-0.6 (flat
        Gaussian 0.248 -> 0.473, vignetted 0.649 -> 1.149), because a
        crop-measured mode shift carries ~1 unit of noise regardless. Two
        independent reviewers measured the same aggregate as net WORSE on their
        volumes. Not applied above the budget.

    - **The deciding measurement is the probe-size sweep** (12 volumes, probe at
      2.3 / 4.7 / 18.8 / 37.5% of the volume). ``p10``'s corrected error falls
      monotonically — 1.03, 0.98, 0.59, 0.38 units, closer than raw on 9/12 then
      12/12 — so the percentile shift is a real property of the data that a bigger
      probe measures better. ``auto``'s does not move: 0.91, 0.75, 0.72, 0.90,
      closer than raw on 8/12 even with 37.5% of the volume in the probe,
      and its WORST case gets worse (2.6 -> 4.3). The mode shift is a property of
      the LOCAL background level, which varies spatially, so no affordable probe
      converges on it — and a real light-sheet stack sits at ~0.03%, far below
      anything measured here.

    - **Cost**: one extra denoise pass over at most
      :data:`DENOISE_PROBE_BUDGET_VOXELS` voxels per CALL, and none at all for
      ``auto`` above the budget (regime 2 is decided from ``volume.shape``, before
      anything is read). That is once per resolution, not once per tile — but
      every worker resolves its own level, so a ``-j N`` run or an ``M``-way
      ``--tile k/M`` fleet pays it once per worker, and for a volume within the
      probe budget the probe IS the whole volume (M whole-volume denoise passes
      for M workers).
    - **Determinism**: the probe is a pure function of ``volume.shape`` and the
      budget, so independent workers (``--tile k/M``, ``-j N``) that share a
      volume, ``h`` and params all reach the same corrected level — provided
      they also share a denoise BACKEND. ``backend="auto"`` resolves to skimage
      on a CPU-only host and to the torch/CUDA kernel on a GPU host, and the two
      do NOT agree closely enough for the estimators to be indifferent: on four
      synthetic pedestals their outputs differed by a mean of ~0.34-0.48 and by up
      to 26-34 INTENSITY units at individual voxels, and the resolved ``auto``
      level came out different in 4 of 4 configurations (by 0.001-0.043 units; a
      reviewer measured 0.005-0.085 on other data). A level difference across
      backends is therefore the norm, not a corner case: pin
      ``--denoise-backend`` for a fleet spanning heterogeneous hosts.
    - The "level >= sampled max would erase all signal" guard is re-applied to
      the corrected level against the **raw floor sample's** max — the same
      basis, and the same bounded read, :func:`resolve_volume_floor` judges on.
      The probe's own denoised max is deliberately NOT used, and the reason is
      NOT that it would catch less: NLM shrinks the range, so the denoised max is
      a strictly TIGHTER bound and would veto a SUPERSET of levels (measured on a
      light-sheet crop: raw max 288.5 vs denoised max 263.0, and a level between
      the two erases every denoised tile while passing the raw-max guard). It is
      not used because a centre-cropped, smoothed probe may legitimately see no
      signal at all — a masked or zero-padded middle — and a veto there would
      silently drop floor suppression for a whole run, which is worse than the
      level being a little generous. The cost of that choice is the gap: a level
      between the denoised and raw maxima is not caught. A background-mode level
      does not land there in practice.
    - **Degrades, never crashes**: an unreadable probe, a failing
      ``denoise_volume_array`` (no torch, an unavailable backend, a raising
      kernel), a degenerate estimate or a probe carrying non-finite values each
      print an honest note and return the RAW-basis level, i.e. exactly today's
      behaviour. See :func:`_denoise_probe_correction`.
    """
    if (
        denoise_h is None
        or denoise_params is None
        or not _floor_spec_is_volume_derived(floor)
    ):
        # Denoise off, or a user absolute no measurement may move: identical to
        # the pre-#1178 behaviour, with no probe and no extra read.
        return resolve_volume_floor_with_strategy(
            volume,
            floor,
            guard_numeric=guard_numeric,
            sample_budget=sample_budget,
            verbose=verbose,
        )

    # REGIME 2, decided from `volume.shape` alone — before any read, and in
    # particular before any NLM pass. A bounded crop cannot measure the
    # histogram-mode shift (see Notes for the numbers), so `auto` keeps the raw
    # basis and says so instead of pretending otherwise.
    if not _volume_fits_probe_budget(
        volume, int(DENOISE_PROBE_BUDGET_VOXELS)
    ) and not _floor_spec_is_percentile(floor):
        level, strategy = resolve_volume_floor_with_strategy(
            volume,
            floor,
            guard_numeric=guard_numeric,
            sample_budget=sample_budget,
            verbose=verbose,
        )
        if level is not None:
            aprint(
                f"Note: --floor {floor} keeps its RAW-basis level {level:.6g}. This "
                "volume is larger than the denoise probe budget, and the "
                "histogram-mode shift denoising induces is not measurable on a "
                "bounded sample of it — measured, applying it was as likely to "
                "move the level away from the non-tiled estimate as toward it. "
                "For a level resolved on the denoised data, use an explicit "
                "--floor pNN (whose shift does transfer) or a fit that is not "
                "tiled."
            )
        return level, strategy

    # The sampled max comes back with the level so the guard below can judge the
    # CORRECTED level on the very same basis, without a second bounded read.
    level_raw, raw_sample_max, raw_strategy = _floor_level_and_sample_max(
        volume,
        floor,
        guard_numeric=guard_numeric,
        sample_budget=sample_budget,
    )
    if level_raw is None:
        # Disabled, or refused by the "erases all signal" guard — nothing to
        # correct.
        return None, None

    corrected = _denoise_probe_correction(
        volume,
        floor,
        float(level_raw),
        float(denoise_h),
        denoise_params,
        probe_cache,
    )
    if corrected is None:
        # Nothing to correct, or nothing trustworthy to correct with — the helper
        # has already explained itself where that was worth saying.
        return level_raw, raw_strategy
    level, delta, probe_voxels, strategy = corrected
    if delta == 0.0:
        return level, strategy

    # Re-guard on the RAW SAMPLED max, the basis `resolve_volume_floor` uses (see
    # Notes): the probe's denoised max is a tighter bound but an unreliable one,
    # because a centre-cropped probe may see no signal at all.
    # `raw_sample_max` is never None here (a volume-derived spec always samples).
    if raw_sample_max is not None and level >= raw_sample_max:
        aprint(
            f"Warning: denoised-basis floor {level:.6g} >= sampled volume max "
            f"{raw_sample_max:.6g}; ignoring (would erase all signal)."
        )
        return None, None
    if verbose:
        aprint(
            f"Resolved whole-volume background floor on the DENOISED basis: "
            f"{level:.6g} (raw {level_raw:.6g} {delta:+.6g} from a "
            f"{probe_voxels:,}-voxel denoise probe)"
        )
    return level, strategy


def resolve_volume_floor_denoised(
    volume: Any,
    floor: "str | float | None",
    *,
    denoise_h: "float | None" = None,
    denoise_params: "dict[str, Any] | None" = None,
    guard_numeric: bool = False,
    sample_budget: int | None = None,
    probe_cache: dict[str, Any] | None = None,
    verbose: bool = False,
) -> "float | None":
    """Resolve a denoised-basis floor while discarding provenance."""
    level, _ = resolve_volume_floor_denoised_with_strategy(
        volume,
        floor,
        denoise_h=denoise_h,
        denoise_params=denoise_params,
        guard_numeric=guard_numeric,
        sample_budget=sample_budget,
        probe_cache=probe_cache,
        verbose=verbose,
    )
    return level


def _resolve_norm_bounds(
    V: np.ndarray,
    norm_percentile: float,
    verbose: bool,
    norm_range: "tuple[float, float] | None",
) -> tuple[float, float]:
    """The ``(image_min, image_max)`` normalization will use, before any floor.

    A supplied ``norm_range`` wins outright (tiled fitting resolves one against
    the whole volume); otherwise the pair comes from THIS array, either its
    extremes (``norm_percentile == 0``) or a symmetric percentile pair.
    """
    if norm_range is not None:
        image_min, image_max = float(norm_range[0]), float(norm_range[1])
        if verbose:
            aprint(
                f"Normalization: whole-volume range [{image_min:.6g}, "
                f"{image_max:.6g}] (supplied, not derived from this array)"
            )
        return image_min, image_max
    if norm_percentile == 0.0:
        # Full range normalization
        if verbose:
            aprint("Normalization: full min-max range")
        return float(np.min(V)), float(np.max(V))
    # Percentile-based robust normalization
    if verbose:
        aprint(
            f"Normalization: {norm_percentile:.1f}%-"
            f"{100.0 - norm_percentile:.1f}% percentile range"
        )
    return (
        float(np.percentile(V, norm_percentile)),
        float(np.percentile(V, 100.0 - norm_percentile)),
    )


def _resolve_applied_norm_bounds(
    V: np.ndarray,
    norm_percentile: float,
    verbose: bool,
    floor: "str | float | None" = None,
    norm_range: "tuple[float, float] | None" = None,
) -> tuple[float, float, "float | None"]:
    """Resolve the normalization bounds and effective floor for one fit."""
    image_min, image_max, applied_floor, _ = _resolve_applied_norm_bounds_with_strategy(
        V, norm_percentile, verbose, floor, norm_range
    )
    return image_min, image_max, applied_floor


def _resolve_applied_norm_bounds_with_strategy(
    V: np.ndarray,
    norm_percentile: float,
    verbose: bool,
    floor: "str | float | None" = None,
    norm_range: "tuple[float, float] | None" = None,
) -> tuple[float, float, "float | None", "str | None"]:
    """Resolve normalization bounds, floor, and specimen provenance."""
    # Configurable normalization - store parameters for intensity rescaling
    image_min, image_max = _resolve_norm_bounds(V, norm_percentile, verbose, norm_range)

    # Background floor suppression: raise image_min to the resolved floor.
    floor_result = _resolve_floor_result(V, floor)
    resolved_floor = None if floor_result is None else float(floor_result.level)
    applied_floor: "float | None" = None
    floor_strategy: "str | None" = None
    if resolved_floor is not None:
        guard_max = image_max if norm_range is not None else float(np.max(V))
        if resolved_floor >= guard_max:
            # A floor at/above the normalization ceiling would leave no usable
            # range. Refuse it and keep the default image_min.
            if verbose:
                aprint(
                    f"Warning: floor {resolved_floor:.6g} >= image max "
                    f"{guard_max:.6g}; ignoring (would erase all signal)"
                )
        else:
            # Only ever RAISE image_min (never below the percentile-based value
            # chosen above): the floor is orthogonal to norm_percentile's low-end
            # clipping. Clamp into [image_min, image_max) so the range stays
            # strictly positive. (When norm_percentile==0, image_min == min(V),
            # so this reduces to max(resolved_floor, min(V)) as before.)
            image_min = float(max(resolved_floor, image_min))
            if image_min >= image_max:
                image_max = float(np.max(V))
                if verbose:
                    aprint(
                        f"Normalization: expanding high endpoint to data max "
                        f"{image_max:.6g} so floor {image_min:.6g} preserves signal"
                    )
            applied_floor = image_min
            if isinstance(floor, str) and floor.strip().lower() == "specimen":
                assert floor_result is not None
                floor_strategy = floor_result.strategy
            if verbose:
                aprint(
                    f"Floor suppression: subtracting background level {image_min:.6g}"
                )

    return image_min, image_max, applied_floor, floor_strategy


def _normalize_data(
    V: np.ndarray,
    norm_percentile: float,
    verbose: bool,
    floor: "str | float | None" = None,
    norm_range: "tuple[float, float] | None" = None,
) -> tuple[np.ndarray, float, float, float, "float | None"]:
    """Normalize input data to [0, 1] range.

    ``floor`` (see :func:`_resolve_floor`) overrides how ``image_min`` is
    chosen: an explicit background level raises ``image_min`` so the pedestal
    is clipped to 0 by the existing ``np.clip((V - image_min) / range, 0, 1)``.
    ``norm_percentile`` still governs ``image_max`` (bright-outlier clipping),
    so the two are normally orthogonal. If the floor overtakes a
    percentile-derived high endpoint, that endpoint expands to the data maximum,
    dropping bright-outlier clipping to preserve usable signal.

    ``norm_range`` supplies ``(image_min, image_max)`` outright, bypassing
    ``norm_percentile``'s derivation from ``V``. Tiled fitting passes a range
    resolved against the WHOLE volume so that every tile maps a given physical
    intensity to the same normalized value, and is therefore held to the same
    absolute convergence tolerance and thresholds (see
    :func:`resolve_volume_norm_range`). Because such a range is estimated from
    a bounded sample, a value above ``image_max`` is real signal rather than an
    outlier and is left unclipped when ``norm_percentile == 0``.
    """
    normalized = _normalize_data_with_strategy(
        V, norm_percentile, verbose, floor, norm_range
    )
    return normalized[:5]


def _normalize_data_with_strategy(
    V: np.ndarray,
    norm_percentile: float,
    verbose: bool,
    floor: "str | float | None" = None,
    norm_range: "tuple[float, float] | None" = None,
) -> tuple[np.ndarray, float, float, float, "float | None", "str | None"]:
    """Normalize data while retaining specimen-floor provenance."""
    image_min, image_max, applied_floor, floor_strategy = (
        _resolve_applied_norm_bounds_with_strategy(
            V, norm_percentile, verbose, floor, norm_range
        )
    )

    intensity_range = image_max - image_min

    if np.abs(intensity_range) < 1e-12:
        V = np.full_like(V, 0.5, dtype=np.float32)
        intensity_range = 1.0  # Avoid division by zero in rescaling
        if verbose:
            aprint("Warning: Input image is nearly uniform")
    else:
        # A SUPPLIED full-range (``norm_percentile == 0``) range is the whole
        # volume's extremes ESTIMATED from a bounded sample, so this array can
        # legitimately hold a voxel brighter than it. Clipping there would
        # flatten exactly the brightest structure — something the per-array
        # path never does, since that array's own max is its ceiling by
        # construction. Keep the shared scale, drop the ceiling. A percentile
        # range asked for bright-outlier clipping, so its ceiling stays.
        ceiling = None if (norm_range is not None and norm_percentile == 0.0) else 1.0
        V = np.clip((V - image_min) / intensity_range, 0.0, ceiling)

    return V, image_min, image_max, intensity_range, applied_floor, floor_strategy


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

    Both arrays are OVERWRITTEN wholesale, so the caller must also set
    ``init_ctx.init_amps_background_relative`` to the convention of
    ``gsplat_data.amplitudes`` — this function cannot know it (a caller's warm
    start is background-relative, ``generate_seeds`` output is raw-sampled).

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
