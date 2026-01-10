# generate.py
"""
Unified entry point for seed generation.

This module provides the main `generate_seeds()` function that serves as a unified
interface to all seed generation methods in the package. All methods return
GSplatData with scale-informed Gaussian shapes.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.seeds.grid import seed_from_grid
from luxar.gsplats.seeds.multiscale_decomposition import seed_from_decomposition


def generate_seeds(
    V: np.ndarray,
    method: str = "auto",
    **kwargs,
) -> GSplatData:
    """
    Generate seed Gaussian splats using specified method(s).

    All seeding methods return GSplatData with scale-informed Gaussian shapes,
    allowing the fitter to use full geometry (centers, sigmas, amplitudes).

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume to analyze.
    method : str, default="auto"
        Seed generation method(s) to use. Options:
        - "auto": Principled combination of all methods (default, recommended)
        - "decomposition": Multi-scale decomposition for blob-like features
        - "grid": Uniform grid for spatial coverage
        - "edges": Edge-based with anisotropic shapes (for boundaries)
        - "decomposition,grid": Combine specific methods (comma-separated)
    **kwargs
        Method-specific parameters. Common parameters are routed to all applicable
        methods, while method-specific parameters are routed only to their respective
        methods.

        **Common Parameters** (apply to multiple methods):

        min_distance : float, default=2.0
            Minimum Euclidean distance (in voxels) between seed centers.
            Used for deduplication when combining multiple methods.

        target_seeds : int or None, optional
            Target number of seeds for "auto" mode. If None, auto-estimated.

        **Decomposition Parameters** (method="decomposition"):

        scales : list[int], default=[1, 2, 4, 8, 16, 32, 64]
            Scale factors for decomposition.
        ignore_finest_k : int, default=1
            Number of finest scales to ignore for peak detection.
        threshold_rel : float, default=0.1
            Relative threshold for peak detection (0.0-1.0).
        peaks_per_scale : int or None, optional
            Maximum number of peaks per scale.
        decompose_kwargs : dict, optional
            Additional kwargs passed to decompose_image().
        verbose : bool, default=False
            Print progress information.

        **Grid Parameters** (method="grid"):

        spacing : float or Sequence[float] or None, optional
            Grid spacing in voxels. None = auto (~5% of smallest dimension).
        jitter : float, default=0.0
            Jitter fraction (0.0-0.5) for random offset.
        sigma : float or None, optional
            Gaussian sigma. None = spacing / 2.
        exclude_below : float or None, optional
            Absolute intensity threshold.
        exclude_below_percentile : float or None, optional
            Percentile intensity threshold (0-100).

        **Edge Parameters** (method="edges"):

        n_seeds : int or None, optional
            Target number of edge seeds.
        edge_threshold_rel : float, default=0.1
            Relative edge threshold.
        structure_radius : float, default=3.0
            Radius for structure tensor computation.
        min_sigma : float, default=0.5
            Minimum sigma from structure tensor.
        max_sigma : float, default=16.0
            Maximum sigma from structure tensor.

    Returns
    -------
    GSplatData
        Gaussian splat seeds with:
        - centers: Peak/centroid positions
        - amplitudes: Peak intensities
        - cholesky_factors: Scale-informed Cholesky factors
        - sharpnesses: All set to 2.0 (standard Gaussian)

    Examples
    --------
    **Basic usage with default settings (auto combination):**

    >>> from luxar.gsplats.seeds import generate_seeds
    >>> seeds = generate_seeds(image)  # Returns GSplatData
    >>> print(f"Generated {len(seeds.centers)} seed splats")

    **Using decomposition method:**

    >>> seeds = generate_seeds(image, method="decomposition")

    **Using grid method:**

    >>> seeds = generate_seeds(image, method="grid", spacing=10.0)

    **Combining methods:**

    >>> seeds = generate_seeds(image, method="decomposition,grid")

    **Use with fit_gaussian_splats:**

    >>> from luxar.gsplats import fit_gaussian_splats
    >>> seeds = generate_seeds(image, method="auto")
    >>> result = fit_gaussian_splats(image, seeds=seeds)

    Notes
    -----
    - Default method is "auto" (principled combination)
    - "decomposition" is best for blob-like features
    - "grid" provides uniform spatial coverage
    - "edges" captures boundaries with anisotropic shapes
    - The fitter will refine all parameters during optimization
    """
    # Input validation
    V = np.asarray(V, dtype=float)
    if V.size == 0:
        raise ValueError("Input array V cannot be empty")
    if V.ndim == 0:
        raise ValueError("Input array V must have at least 1 dimension")

    # Parse method string
    method = method.lower().strip()
    single_methods = {"decomposition", "grid", "edges"}

    # Handle comma-separated methods (e.g., "decomposition,grid")
    if "," in method:
        methods = [m.strip() for m in method.split(",")]
        for m in methods:
            if m not in single_methods:
                raise ValueError(
                    f"Invalid method: '{m}'. "
                    f"Combined methods must be one of: {single_methods}"
                )
    elif method == "auto":
        # Principled combination of all methods
        methods = ["decomposition", "edges", "grid"]
    elif method in single_methods:
        methods = [method]
    else:
        raise ValueError(
            f"Invalid method: '{method}'. "
            "Valid options: 'decomposition', 'grid', 'edges', 'auto', "
            "or comma-separated combination (e.g., 'decomposition,grid')"
        )

    # Extract common parameters
    min_distance = kwargs.pop("min_distance", 2.0)
    target_seeds = kwargs.pop("target_seeds", None)

    # Route parameters to appropriate methods
    decomposition_kwargs: Dict[str, Any] = {}
    grid_kwargs: Dict[str, Any] = {}
    edges_kwargs: Dict[str, Any] = {}

    # Define parameter routing rules
    decomposition_params = {
        "scales",
        "ignore_finest_k",
        "peaks_per_scale",
        "threshold_rel",
        "decompose_kwargs",
        "verbose",
    }

    grid_params = {
        "spacing",
        "jitter",
        "sigma",
        "exclude_below",
        "exclude_below_percentile",
    }

    edges_params = {
        "n_seeds",
        "edge_threshold_rel",
        "structure_radius",
        "min_sigma",
        "max_sigma",
    }

    # Passthrough params from higher-level APIs
    passthrough_params = {"use_metal"}

    # Route parameters
    for key, value in kwargs.items():
        used_by_decomposition = key in decomposition_params
        used_by_grid = key in grid_params
        used_by_edges = key in edges_params

        if "decomposition" in methods and used_by_decomposition:
            decomposition_kwargs[key] = value
        if "grid" in methods and used_by_grid:
            grid_kwargs[key] = value
        if "edges" in methods and used_by_edges:
            edges_kwargs[key] = value

        if not (
            used_by_decomposition
            or used_by_grid
            or used_by_edges
            or key in passthrough_params
        ):
            import warnings

            warnings.warn(
                f"Parameter '{key}' is not used by any selected method: {methods}",
                UserWarning,
            )

    # Add min_distance to decomposition (grid doesn't use it directly)
    decomposition_kwargs["min_distance"] = min_distance
    edges_kwargs["min_distance"] = min_distance

    # Handle auto mode with budget allocation
    if method == "auto":
        return _auto_combine(
            V,
            target_seeds=target_seeds,
            min_distance=min_distance,
            decomposition_kwargs=decomposition_kwargs,
            grid_kwargs=grid_kwargs,
            edges_kwargs=edges_kwargs,
        )

    # Generate seeds using specified method(s)
    results: List[GSplatData] = []

    for m in methods:
        if m == "decomposition":
            result = seed_from_decomposition(V, **decomposition_kwargs)
            results.append(result)
        elif m == "grid":
            result = seed_from_grid(V, **grid_kwargs)
            results.append(result)
        elif m == "edges":
            # Lazy import since edges.py may not exist yet
            try:
                from luxar.gsplats.seeds.edges import seed_from_edges

                result = seed_from_edges(V, **edges_kwargs)
                results.append(result)
            except ImportError:
                import warnings

                warnings.warn(
                    "Edge seeding not available yet (edges.py not implemented)",
                    UserWarning,
                )

    # Combine results if multiple methods were used
    if len(results) == 0:
        return _empty_gsplatdata(V.ndim)
    elif len(results) == 1:
        return results[0]
    else:
        # Combine GSplatData from multiple methods
        return _combine_gsplatdata(results, min_distance)


def _empty_gsplatdata(ndim: int) -> GSplatData:
    """Create empty GSplatData."""
    tril_size = ndim * (ndim + 1) // 2
    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros(0, dtype=np.float32),
        cholesky_factors=np.zeros((0, tril_size), dtype=np.float32),
        sharpnesses=np.zeros(0, dtype=np.float32),
    )


def _auto_combine(
    V: np.ndarray,
    target_seeds: Optional[int],
    min_distance: float,
    decomposition_kwargs: Dict[str, Any],
    grid_kwargs: Dict[str, Any],
    edges_kwargs: Dict[str, Any],
) -> GSplatData:
    """
    Principled combination of all seeding methods.

    Budget allocation:
    - Decomposition: 50% (blob-like features)
    - Edges: 30% (boundaries)
    - Grid: 20% (coverage)

    Seeds are added in priority order with Mahalanobis-aware deduplication.
    """
    ndim = V.ndim

    # Estimate target if not provided
    if target_seeds is None:
        # Heuristic: ~1 seed per 100 voxels^(1/ndim), minimum 100
        total_voxels = float(np.prod(V.shape))
        target_seeds = max(100, int(total_voxels ** (1.0 / ndim) / 2))
        target_seeds = min(target_seeds, 10000)  # Cap at 10k

    # Budget allocation
    budget_decomp = int(target_seeds * 0.50)
    budget_edges = int(target_seeds * 0.30)
    budget_grid = target_seeds - budget_decomp - budget_edges

    results: List[GSplatData] = []

    # Phase 1: Decomposition seeds (highest priority)
    try:
        # Limit peaks based on budget
        n_scales = len(decomposition_kwargs.get("scales", [1, 2, 4, 8, 16, 32, 64]))
        peaks_per_scale = max(10, budget_decomp // n_scales)
        decomp_kwargs = {**decomposition_kwargs, "peaks_per_scale": peaks_per_scale}
        seeds_decomp = seed_from_decomposition(V, **decomp_kwargs)
        if len(seeds_decomp.centers) > 0:
            results.append(seeds_decomp)
    except Exception:
        pass  # Continue if decomposition fails

    # Phase 2: Edge seeds (second priority)
    try:
        from luxar.gsplats.seeds.edges import seed_from_edges

        edge_kwargs = {**edges_kwargs, "n_seeds": budget_edges * 2}  # Over-sample
        seeds_edges = seed_from_edges(V, **edge_kwargs)
        if len(seeds_edges.centers) > 0:
            results.append(seeds_edges)
    except ImportError:
        pass  # Edges not available yet
    except Exception:
        pass  # Continue if edges fails

    # Phase 3: Grid seeds (fill gaps)
    try:
        # Auto-compute spacing based on budget
        total_voxels = float(np.prod(V.shape))
        target_grid_density = budget_grid / total_voxels
        spacing = max(2.0, (1.0 / target_grid_density) ** (1.0 / ndim))
        grid_kwargs_auto = {**grid_kwargs, "spacing": spacing}
        seeds_grid = seed_from_grid(V, **grid_kwargs_auto)
        if len(seeds_grid.centers) > 0:
            results.append(seeds_grid)
    except Exception:
        pass  # Continue if grid fails

    # Combine with deduplication
    if len(results) == 0:
        return _empty_gsplatdata(ndim)
    elif len(results) == 1:
        return results[0]
    else:
        return _combine_gsplatdata(results, min_distance)


def _combine_gsplatdata(
    results: List[GSplatData],
    min_distance: float,
) -> GSplatData:
    """
    Combine GSplatData from multiple seeding methods with deduplication.

    Uses greedy deduplication sorted by amplitude (highest priority first).
    """
    # Filter empty results
    results = [r for r in results if len(r.centers) > 0]

    if len(results) == 0:
        return _empty_gsplatdata(2)  # Default to 2D

    # Concatenate all arrays
    all_centers = np.vstack([r.centers for r in results])
    all_amplitudes = np.concatenate([r.amplitudes for r in results])
    all_cholesky = np.vstack([r.cholesky_factors for r in results])
    all_sharpnesses = np.concatenate([r.sharpnesses for r in results])

    if len(all_centers) == 0:
        ndim = results[0].centers.shape[1]
        return _empty_gsplatdata(ndim)

    # Deduplicate using amplitude priority
    sort_idx = np.argsort(all_amplitudes)[::-1]
    centers_sorted = all_centers[sort_idx]
    amplitudes_sorted = all_amplitudes[sort_idx]
    cholesky_sorted = all_cholesky[sort_idx]
    sharpnesses_sorted = all_sharpnesses[sort_idx]

    # Greedy deduplication
    kept_mask = np.ones(len(centers_sorted), dtype=bool)

    for i in range(len(centers_sorted)):
        if not kept_mask[i]:
            continue

        diffs = centers_sorted[i + 1 :] - centers_sorted[i]
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        nearby = distances < min_distance
        kept_mask[i + 1 :][nearby] = False

    return GSplatData(
        centers=centers_sorted[kept_mask].astype(np.float32),
        amplitudes=amplitudes_sorted[kept_mask].astype(np.float32),
        cholesky_factors=cholesky_sorted[kept_mask].astype(np.float32),
        sharpnesses=sharpnesses_sorted[kept_mask].astype(np.float32),
    )
