# generate.py
"""
Unified entry point for seed generation.

This module provides the main `generate_seeds()` function that serves as a unified
interface to all seed generation methods in the package. All methods return
GSplatData with scale-informed Gaussian shapes.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.seeds.grid import seed_from_grid
from luxar.gsplats.seeds.multiscale_decomposition import seed_from_decomposition
from luxar.gsplats.seeds.peaks import seed_from_peaks
from luxar.utils.arbol_warnings import arbol_warnings


@arbol_warnings()
def generate_seeds(
    V: np.ndarray,
    method: str = "auto",
    **kwargs: Any,
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
        - "auto": Fast edges + grid combination (default, recommended)
        - "decomposition": Multi-scale decomposition for blob-like features (slow)
        - "grid": Uniform grid for spatial coverage
        - "edges": Edge-based boundary detection with Sobel gradients
        - "peaks": Local maxima after Gaussian blur (ideal for sparse residuals)
        - "decomposition,edges,grid": Include all methods (comma-separated)
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

        device : str or None, optional
            PyTorch device for GPU acceleration. Options:
            - None (default): CPU using scipy.ndimage
            - 'cpu': Force CPU
            - 'cuda': NVIDIA GPU (if available)
            - 'mps': Apple Metal (if available)
            - 'auto': Auto-detect best device

            GPU acceleration provides substantial speedup for large volumes (>100³) — often orders of magnitude depending on GPU and problem size.
            Applied to all selected seeding methods.

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

    Returns
    -------
    GSplatData
        Gaussian splat seeds with:
        - centers: Peak/centroid positions
        - amplitudes: Peak intensities
        - cholesky_factors: Scale-informed Cholesky factors
        - Standard Gaussian profile (no sharpness parameter)

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
    - Default method is "auto" (fast edges + grid combination)
    - "decomposition" is best for blob-like features but slow
    - "grid" provides uniform spatial coverage
    - "edges" captures boundaries with isotropic shapes (orientation learned during fitting)
    - Use "decomposition,edges,grid" to include all methods
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
    single_methods = {"decomposition", "grid", "edges", "peaks"}

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
        # Fast combination: edges + grid (decomposition is slow)
        methods = ["edges", "grid"]
    elif method in single_methods:
        methods = [method]
    else:
        raise ValueError(
            f"Invalid method: '{method}'. "
            "Valid options: 'decomposition', 'grid', 'edges', 'peaks', 'auto', "
            "or comma-separated combination (e.g., 'decomposition,grid')"
        )

    # Extract common parameters
    min_distance = kwargs.pop("min_distance", 2.0)
    target_seeds = kwargs.pop("target_seeds", None)
    verbose = kwargs.get("verbose", False)  # Don't pop - methods may use it
    device = kwargs.get("device", None)  # Don't pop - methods may use it

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
        "device",
    }

    grid_params = {
        "spacing",
        "jitter",
        "sigma",
        "exclude_below",
        "exclude_below_percentile",
        "device",
    }

    edges_params = {
        "n_seeds",
        "edge_threshold_rel",
        "device",
    }

    peaks_params = {
        "n_seeds",
        "init_sigma",
        "device",
    }

    # Passthrough params from higher-level APIs (used by fitter, not seeding)
    passthrough_params = {
        "use_metal",
        "use_fp16",
        "n_iterations",
        "lr",
        "verbose",
        "device",
    }

    peaks_kwargs: Dict[str, Any] = {}

    # Route parameters
    for key, value in kwargs.items():
        used_by_decomposition = key in decomposition_params
        used_by_grid = key in grid_params
        used_by_edges = key in edges_params
        used_by_peaks = key in peaks_params

        if "decomposition" in methods and used_by_decomposition:
            decomposition_kwargs[key] = value
        if "grid" in methods and used_by_grid:
            grid_kwargs[key] = value
        if "edges" in methods and used_by_edges:
            edges_kwargs[key] = value
        if "peaks" in methods and used_by_peaks:
            peaks_kwargs[key] = value

        if not (
            used_by_decomposition
            or used_by_grid
            or used_by_edges
            or used_by_peaks
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
            verbose=verbose,
            device=device,
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
        elif m == "peaks":
            # Pass target_seeds as n_seeds if not already set
            if "n_seeds" not in peaks_kwargs and target_seeds is not None:
                peaks_kwargs["n_seeds"] = target_seeds
            result = seed_from_peaks(V, **peaks_kwargs)
            results.append(result)

    # Combine results if multiple methods were used
    if len(results) == 0:
        return _empty_gsplatdata(V.ndim)
    elif len(results) == 1:
        return results[0]
    else:
        # Combine GSplatData from multiple methods
        return _combine_gsplatdata(results, min_distance, device=device, ndim=V.ndim)


def _empty_gsplatdata(ndim: int) -> GSplatData:
    """Create empty GSplatData."""
    tril_size = ndim * (ndim + 1) // 2
    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros(0, dtype=np.float32),
        cholesky_factors=np.zeros((0, tril_size), dtype=np.float32),
    )


def _auto_combine(
    V: np.ndarray,
    target_seeds: Optional[int],
    min_distance: float,
    decomposition_kwargs: Dict[str, Any],
    grid_kwargs: Dict[str, Any],
    edges_kwargs: Dict[str, Any],
    verbose: bool = False,
    device: Optional[str] = None,
) -> GSplatData:
    """
    Fast combination of edges + grid seeding methods.

    Budget allocation:
    - Edges: 60% (boundaries and structure)
    - Grid: 40% (coverage)

    Seeds are added in priority order with deduplication.
    Note: Decomposition (multiscale) is excluded by default for speed.
    Use method="decomposition,edges,grid" to include it explicitly.
    """
    ndim = V.ndim

    # Estimate target if not provided
    if target_seeds is None:
        # Heuristic: ~1 seed per 100 voxels^(1/ndim), minimum 100
        total_voxels = float(np.prod(V.shape))
        target_seeds = max(100, int(total_voxels ** (1.0 / ndim) / 2))
        target_seeds = min(target_seeds, 10000)  # Cap at 10k

    # Budget allocation (edges + grid only)
    budget_edges = int(target_seeds * 0.60)
    budget_grid = target_seeds - budget_edges

    if verbose:
        aprint(
            f"Target: {target_seeds} seeds (edges: {budget_edges}, grid: {budget_grid})"
        )

    results: List[GSplatData] = []

    # Edge seeds first (highest priority - captures structure)
    try:
        from luxar.gsplats.seeds.edges import seed_from_edges

        with asection(f"Edge detection ({budget_edges} seeds)"):
            edge_kwargs = {**edges_kwargs, "n_seeds": budget_edges}
            seeds_edges = seed_from_edges(V, **edge_kwargs)
            if len(seeds_edges.centers) > 0:
                results.append(seeds_edges)
                if verbose:
                    aprint(f"✓ Generated {len(seeds_edges.centers)} edge seeds")
    except ImportError:
        pass  # Edges not available yet
    except (ValueError, RuntimeError) as e:
        import warnings

        warnings.warn(f"Edge seeding failed: {e}", UserWarning, stacklevel=2)

    # Grid seeds fill coverage gaps after edge seeding.
    try:
        # Auto-compute spacing based on budget
        total_voxels = float(np.prod(V.shape))
        target_grid_density = budget_grid / total_voxels
        spacing = max(2.0, (1.0 / target_grid_density) ** (1.0 / ndim))

        with asection(f"Grid generation (spacing={spacing:.1f})"):
            grid_kwargs_auto = {**grid_kwargs, "spacing": spacing}
            seeds_grid = seed_from_grid(V, **grid_kwargs_auto)
            if len(seeds_grid.centers) > 0:
                results.append(seeds_grid)
                if verbose:
                    aprint(f"✓ Generated {len(seeds_grid.centers)} grid seeds")
    except (ValueError, RuntimeError) as e:
        import warnings

        warnings.warn(f"Grid seeding failed: {e}", UserWarning, stacklevel=2)

    # Combine with deduplication
    if len(results) == 0:
        return _empty_gsplatdata(ndim)
    elif len(results) == 1:
        return results[0]
    else:
        total_before = sum(len(r.centers) for r in results)
        with asection(f"Deduplication ({total_before} → target)"):
            result = _combine_gsplatdata(
                results, min_distance, device=device, ndim=ndim
            )
            if verbose:
                aprint(f"✓ Final: {len(result.centers)} seeds after deduplication")
            return result


def _combine_gsplatdata(
    results: List[GSplatData],
    min_distance: float,
    device: Optional[str] = None,
    ndim: int = 2,
) -> GSplatData:
    """
    Combine GSplatData from multiple seeding methods with deduplication.

    Uses greedy deduplication sorted by amplitude (highest priority first).
    """
    # Filter empty results
    results = [r for r in results if len(r.centers) > 0]

    if len(results) == 0:
        return _empty_gsplatdata(ndim)

    # Concatenate all arrays
    all_centers = np.vstack([r.centers for r in results])
    all_amplitudes = np.concatenate([r.amplitudes for r in results])
    all_cholesky = np.vstack([r.cholesky_factors for r in results])

    if len(all_centers) == 0:
        ndim = results[0].centers.shape[1]
        return _empty_gsplatdata(ndim)

    # Deduplicate using KD-tree acceleration with O(1) index tracking
    # This avoids O(M×N) coordinate matching that was a major bottleneck
    from luxar.gsplats.seeds.utils import dedupe_farthest_first

    # dedupe_farthest_first now returns (coords, indices) tuple
    # Indices provide direct O(1) lookup into original arrays
    deduped_centers, kept_indices = dedupe_farthest_first(
        all_centers,
        min_distance=min_distance,
        intensities=all_amplitudes,
        device=device,
    )

    return GSplatData(
        centers=all_centers[kept_indices].astype(np.float32),
        amplitudes=all_amplitudes[kept_indices].astype(np.float32),
        cholesky_factors=all_cholesky[kept_indices].astype(np.float32),
    )
