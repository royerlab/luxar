# generate.py
"""
Unified entry point for seed generation.

This module provides the main `generate_seeds()` function that serves as a unified
interface to all seed generation methods in the package. All methods now return
GSplatData with scale-informed Gaussian shapes.
"""

from typing import Any, Dict

import numpy as np

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.seeds.multiscale_decomposition import seed_from_decomposition
from luxar.gsplats.seeds.multiscale_gaussian import seed_from_gaussian


def generate_seeds(
    V: np.ndarray,
    method: str = "decomposition",
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
    method : str, default="decomposition"
        Seed generation method(s) to use. Options:
        - "gaussian": Multiscale Gaussian-blurred peak detection
        - "decomposition": Multi-scale decomposition method (default, recommended)
        - "moments": Moment-based with full covariance estimation
        - "all": All methods (decomposition + gaussian + moments) combined
        - "decomposition,gaussian": Combine two specific methods (comma-separated)
    **kwargs
        Method-specific parameters. Common parameters are routed to all applicable
        methods, while method-specific parameters are routed only to their respective
        methods.

        **Common Parameters** (apply to multiple methods):

        min_distance : float, default=2.0
            Minimum Euclidean distance (in voxels) between seed centers.
            Used for deduplication when combining multiple methods.

        **Multiscale Gaussian Parameters** (method="gaussian"):

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

        **Moment Parameters** (method="moments"):

        scales : Sequence[int], default=(1, 2, 4, 8)
            Scale factors for decomposition.
        nms_radius_vox : float, default=2.0
            Non-maximum suppression radius.
        peak_threshold_rel : float, default=0.1
            Relative intensity threshold.
        moment_radius_scale : float, default=1.5
            Moment integration radius as multiple of scale.
        min_eigenvalue : float, default=0.25
            Minimum covariance eigenvalue.
        max_eigenvalue : float, default=64.0
            Maximum covariance eigenvalue.
        verbose : bool, default=True
            Print progress information.

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
    **Basic usage with default settings (decomposition):**

    >>> from luxar.gsplats.seeds import generate_seeds
    >>> seeds = generate_seeds(image)  # Returns GSplatData
    >>> aprint(f"Generated {len(seeds.centers)} seed splats")

    **Using Gaussian method:**

    >>> seeds = generate_seeds(image, method="gaussian")

    **Using moment-based method (full covariance):**

    >>> seeds = generate_seeds(image, method="moments")

    **Custom parameters for Gaussian method:**

    >>> seeds = generate_seeds(
    ...     image,
    ...     method="gaussian",
    ...     scales=[2.0, 4.0, 8.0],
    ...     percentile_thresh=80.0,
    ...     min_distance=3.0,
    ... )

    **Combining all methods:**

    >>> seeds = generate_seeds(image, method="all")

    **Use with fit_gaussian_splats:**

    >>> from luxar.gsplats import fit_gaussian_splats
    >>> seeds = generate_seeds(image, method="decomposition")
    >>> result = fit_gaussian_splats(image, seeds=seeds)

    Notes
    -----
    - Default method is "decomposition" (most principled scale separation)
    - For anisotropic features, use method="moments" for full covariance
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
    single_methods = {"gaussian", "decomposition", "moments"}
    valid_methods = single_methods | {"all", "both"}

    # Handle comma-separated methods (e.g., "gaussian,decomposition")
    if "," in method:
        methods = [m.strip() for m in method.split(",")]
        for m in methods:
            if m not in single_methods:
                raise ValueError(
                    f"Invalid method: '{m}'. "
                    f"Combined methods must be one of: {single_methods}"
                )
    elif method == "all":
        # Run all seeding methods for maximum coverage
        methods = ["decomposition", "gaussian", "moments"]
    elif method == "both":
        # Legacy alias: "both" = decomposition + gaussian (not moments)
        methods = ["decomposition", "gaussian"]
    elif method in single_methods:
        methods = [method]
    else:
        raise ValueError(
            f"Invalid method: '{method}'. "
            "Valid options: 'gaussian', 'decomposition', 'moments', 'both', 'all', "
            "or comma-separated combination (e.g., 'gaussian,decomposition')"
        )

    # Extract min_distance for combining
    min_distance = kwargs.pop("min_distance", 2.0)

    # Route parameters to appropriate methods
    gaussian_kwargs: Dict[str, Any] = {}
    decomposition_kwargs: Dict[str, Any] = {}
    moment_kwargs: Dict[str, Any] = {}

    # Define parameter routing rules
    gaussian_params = {
        "scales",
        "peaks_per_scale",
        "percentile_thresh",
        "apply_clahe",
        "clahe_tile_size",
        "clahe_clip_limit",
        "clahe_nbins",
    }

    decomposition_params = {
        "scales",
        "ignore_finest_k",
        "peaks_per_scale",
        "threshold_rel",
        "decompose_kwargs",
        "verbose",
    }

    moment_params = {
        "scales",
        "decomp_n_iters",
        "decomp_lr",
        "decomp_energy_weight",
        "nms_radius_vox",
        "peak_threshold_rel",
        "max_peaks_per_scale",
        "skip_finest_scales",
        "moment_radius_scale",
        "min_eigenvalue",
        "max_eigenvalue",
        "device",
        "verbose",
    }

    # Passthrough params from higher-level APIs
    passthrough_params = {"use_metal"}

    # Route parameters
    for key, value in kwargs.items():
        used_by_gaussian = key in gaussian_params
        used_by_decomposition = key in decomposition_params
        used_by_moment = key in moment_params

        if "gaussian" in methods and used_by_gaussian:
            gaussian_kwargs[key] = value
        if "decomposition" in methods and used_by_decomposition:
            decomposition_kwargs[key] = value
        if "moments" in methods and used_by_moment:
            moment_kwargs[key] = value

        if not (
            used_by_gaussian
            or used_by_decomposition
            or used_by_moment
            or key in passthrough_params
        ):
            import warnings

            warnings.warn(
                f"Parameter '{key}' is not used by any selected method: {methods}",
                UserWarning,
            )

    # Add min_distance to individual method kwargs
    gaussian_kwargs["min_distance"] = min_distance
    decomposition_kwargs["min_distance"] = min_distance

    # Generate seeds using specified method(s)
    results = []

    for m in methods:
        if m == "gaussian":
            result = seed_from_gaussian(V, **gaussian_kwargs)
            results.append(result)
        elif m == "decomposition":
            result = seed_from_decomposition(V, **decomposition_kwargs)
            results.append(result)
        elif m == "moments":
            from luxar.gsplats.seeds.moment_seeding import seed_from_moments

            result = seed_from_moments(V, **moment_kwargs)
            results.append(result)

    # Combine results if multiple methods were used
    if len(results) == 0:
        return GSplatData(
            centers=np.zeros((0, V.ndim), dtype=np.float32),
            amplitudes=np.zeros(0, dtype=np.float32),
            cholesky_factors=np.zeros(
                (0, V.ndim * (V.ndim + 1) // 2), dtype=np.float32
            ),
            sharpnesses=np.zeros(0, dtype=np.float32),
        )
    elif len(results) == 1:
        return results[0]
    else:
        # Combine GSplatData from multiple methods
        return _combine_gsplatdata(results, min_distance)


def _combine_gsplatdata(
    results: list,
    min_distance: float,
) -> GSplatData:
    """
    Combine GSplatData from multiple seeding methods with deduplication.
    """
    # Concatenate all arrays
    all_centers = np.vstack([r.centers for r in results if len(r.centers) > 0])
    all_amplitudes = np.concatenate(
        [r.amplitudes for r in results if len(r.amplitudes) > 0]
    )
    all_cholesky = np.vstack(
        [r.cholesky_factors for r in results if len(r.cholesky_factors) > 0]
    )
    all_sharpnesses = np.concatenate(
        [r.sharpnesses for r in results if len(r.sharpnesses) > 0]
    )

    if len(all_centers) == 0:
        ndim = results[0].centers.shape[1] if results else 2
        return GSplatData(
            centers=np.zeros((0, ndim), dtype=np.float32),
            amplitudes=np.zeros(0, dtype=np.float32),
            cholesky_factors=np.zeros((0, ndim * (ndim + 1) // 2), dtype=np.float32),
            sharpnesses=np.zeros(0, dtype=np.float32),
        )

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
        centers=centers_sorted[kept_mask],
        amplitudes=amplitudes_sorted[kept_mask],
        cholesky_factors=cholesky_sorted[kept_mask],
        sharpnesses=sharpnesses_sorted[kept_mask],
    )
