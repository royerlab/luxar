# generate.py
"""
Unified entry point for seed generation.

This module provides the main `generate_seeds()` function that serves as a unified
interface to all seed generation methods in the package.
"""

from typing import Any, Dict

import numpy as np

from luxar.gsplats.seeds.multiscale_decomposition import (
    find_seeds_multiscale_decomposition,
)
from luxar.gsplats.seeds.multiscale_gaussian import find_seeds_multiscale_gaussian
from luxar.gsplats.seeds.utils import combine_seeds


def generate_seeds(
    V: np.ndarray,
    method: str = "both",
    **kwargs,
) -> np.ndarray:
    """
    Generate seed locations for Gaussian splat fitting using specified method(s).

    This is the primary entry point for seed generation, providing a unified interface
    to all available methods. It supports single methods or combinations thereof.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume to analyze.
    method : str, default="both"
        Seed generation method(s) to use. Options:
        - "gaussian": Multiscale Gaussian-blurred peak detection only
        - "decomposition": Multi-scale decomposition method only
        - "both" or "decomposition,gaussian": Decomposition first, then Gaussian
        - "gaussian,decomposition": Gaussian first, then decomposition
    **kwargs
        Method-specific parameters. Common parameters are routed to all applicable
        methods, while method-specific parameters are routed only to their respective
        methods.

        **Common Parameters** (apply to both methods):

        min_distance : float, default=2.0
            Minimum Euclidean distance (in voxels) between seed centers.
            Used for deduplication when combining multiple methods.

        **Multiscale Gaussian Parameters** (method="gaussian"):

        spacing : Sequence[float], optional
            Physical spacing between voxels along each axis.
        scales : Sequence[float], default=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
            Standard deviations (in voxels) for Gaussian filtering.
        peaks_per_scale : int or None, optional
            Maximum number of peaks per scale. None = unlimited.
        percentile_thresh : float, default=75.0
            Intensity percentile threshold (0-100) for peak detection.
        apply_clahe : bool, default=True
            Whether to apply CLAHE preprocessing.
        clahe_tile_size : int, default=32
            Tile size for CLAHE preprocessing.
        clahe_clip_limit : float, default=16.0
            Contrast limiting factor for CLAHE.
        clahe_nbins : int, default=256
            Number of histogram bins for CLAHE.

        **Decomposition Parameters** (method="decomposition"):

        scales : list[int], default=[1, 2, 4, 8, 16, 32, 64]
            Scale factors for decomposition (note: integers, not floats).
        ignore_finest_k : int, default=1
            Number of finest scales to ignore for peak detection.
        threshold_rel : float, default=0.1
            Relative threshold for peak detection (0.0-1.0).
        decompose_kwargs : dict, optional
            Additional kwargs passed to decompose_image().
        verbose : bool, default=False
            Print progress information.

    Returns
    -------
    np.ndarray
        Array of shape (N, ndim) containing seed center coordinates in voxel units.
        Coordinates may be sub-voxel due to centroid refinement.

    Examples
    --------
    **Basic usage with default settings (both methods):**

    >>> from luxar.gsplats.seeds import generate_seeds
    >>> seeds = generate_seeds(image)

    **Using only Gaussian method:**

    >>> seeds = generate_seeds(image, method="gaussian")

    **Using only decomposition method:**

    >>> seeds = generate_seeds(image, method="decomposition")

    **Custom parameters for Gaussian method:**

    >>> seeds = generate_seeds(
    ...     image,
    ...     method="gaussian",
    ...     scales=[2.0, 4.0, 8.0],
    ...     percentile_thresh=80.0,
    ...     min_distance=3.0,
    ... )

    **Custom parameters for decomposition method:**

    >>> seeds = generate_seeds(
    ...     image,
    ...     method="decomposition",
    ...     scales=[1, 2, 4, 8],
    ...     ignore_finest_k=2,
    ...     threshold_rel=0.2,
    ...     min_distance=3.0,
    ... )

    **Combining both methods with custom parameters:**

    >>> seeds = generate_seeds(
    ...     image,
    ...     method="both",
    ...     scales=[2.0, 4.0, 8.0],  # Will be used by Gaussian (floats)
    ...     ignore_finest_k=1,  # Will be used by decomposition
    ...     min_distance=3.0,  # Used for combining results
    ... )

    **Specific method order:**

    >>> # Gaussian first, then decomposition
    >>> seeds = generate_seeds(image, method="gaussian,decomposition")

    Notes
    -----
    - Parameter routing is automatic based on method names
    - When using "both", decomposition runs first, then Gaussian
    - Results are combined using spatial deduplication (farthest-first)
    - The `min_distance` parameter controls final deduplication
    - For Gaussian method, `scales` should be floats (sigma values)
    - For decomposition method, `scales` should be integers (downsample factors)
    - Both methods can use `peaks_per_scale` to limit peaks per scale

    See Also
    --------
    find_seeds_multiscale_gaussian : Direct Gaussian method access
    find_seeds_multiscale_decomposition : Direct decomposition method access
    """
    # Input validation
    V = np.asarray(V, dtype=float)
    if V.size == 0:
        raise ValueError("Input array V cannot be empty")
    if V.ndim == 0:
        raise ValueError("Input array V must have at least 1 dimension")

    # Parse method string
    method = method.lower().strip()
    valid_methods = {"gaussian", "decomposition", "both"}

    # Handle comma-separated methods
    if "," in method:
        methods = [m.strip() for m in method.split(",")]
        if len(methods) != 2:
            raise ValueError(
                f"Invalid method string: '{method}'. "
                "Expected single method or two comma-separated methods."
            )
        for m in methods:
            if m not in {"gaussian", "decomposition"}:
                raise ValueError(
                    f"Invalid method: '{m}'. Valid methods: 'gaussian', 'decomposition'"
                )
    elif method == "both":
        methods = ["decomposition", "gaussian"]  # Default order
    elif method in valid_methods:
        methods = [method]
    else:
        raise ValueError(
            f"Invalid method: '{method}'. "
            "Valid options: 'gaussian', 'decomposition', 'both', "
            "'gaussian,decomposition', 'decomposition,gaussian'"
        )

    # Extract min_distance for combining (default: 2.0)
    min_distance = kwargs.pop("min_distance", 2.0)

    # Route parameters to appropriate methods
    gaussian_kwargs: Dict[str, Any] = {}
    decomposition_kwargs: Dict[str, Any] = {}

    # Define parameter routing rules
    gaussian_params = {
        "spacing",
        "scales",  # Note: Gaussian uses float scales
        "peaks_per_scale",
        "percentile_thresh",
        "apply_clahe",
        "clahe_tile_size",
        "clahe_clip_limit",
        "clahe_nbins",
    }

    decomposition_params = {
        "scales",  # Note: Decomposition uses int scales
        "ignore_finest_k",
        "peaks_per_scale",
        "threshold_rel",
        "decompose_kwargs",
        "verbose",
    }

    # Route parameters based on which methods are being used
    for key, value in kwargs.items():
        # Check which method(s) use this parameter
        used_by_gaussian = key in gaussian_params
        used_by_decomposition = key in decomposition_params

        # Route to appropriate method(s)
        if "gaussian" in methods and used_by_gaussian:
            gaussian_kwargs[key] = value
        if "decomposition" in methods and used_by_decomposition:
            decomposition_kwargs[key] = value

        # Warn about unused parameters (not consumed by any selected method)
        if not (used_by_gaussian or used_by_decomposition):
            import warnings

            warnings.warn(
                f"Parameter '{key}' is not used by any selected method: {methods}",
                UserWarning,
            )

    # Generate seeds using specified method(s)
    seed_arrays = []

    for m in methods:
        if m == "gaussian":
            seeds = find_seeds_multiscale_gaussian(V, **gaussian_kwargs)
            seed_arrays.append(seeds)
        elif m == "decomposition":
            seeds = find_seeds_multiscale_decomposition(V, **decomposition_kwargs)
            seed_arrays.append(seeds)

    # Combine results if multiple methods were used
    if len(seed_arrays) == 0:
        # Should never happen due to validation above, but handle gracefully
        return np.zeros((0, V.ndim), dtype=float)
    elif len(seed_arrays) == 1:
        # Single method - return directly (no need to combine)
        return seed_arrays[0]
    else:
        # Multiple methods - combine with deduplication
        combined = combine_seeds(*seed_arrays, min_distance=min_distance)
        return combined
