# decomposition.py
"""
Decomposition-based candidate generation for Gaussian splatting.

This module provides scale-hierarchical detection via multi-scale image
decomposition, offering principled scale separation and noise suppression.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from arbol import aprint

from luxar.gsplats.candidates.utils import dedupe_farthest_first, local_maxima


def find_candidates_from_decomposition(
    V: np.ndarray,
    scales: List[int] = [1, 2, 4, 8, 16, 32, 64],
    ignore_finest_k: int = 1,
    peaks_per_scale: Optional[int] = None,
    min_distance: float = 2.0,
    threshold_rel: float = 0.1,
    decompose_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate candidate Gaussian splat locations using multi-scale decomposition.

    This method decomposes the input image into multiple scales using `decompose_image()`,
    finds local maxima in each scale (excluding the finest k scales), and returns their
    positions as candidate splat centers. This approach provides principled scale
    separation and sparse representation compared to overcomplete methods.

    The key advantage is that energy is explicitly distributed across scales through
    optimization, creating a natural hierarchy from coarse to fine structure.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume. Shape: (s_0, s_1, ..., s_{n-1}).
    scales : List[int], optional
        Scale factors for decomposition. Default: [1, 2, 4, 8, 16, 32, 64].
        Scale 1 = full resolution, scale 2 = half resolution, etc.
    ignore_finest_k : int, optional
        Number of finest scales to ignore for peak detection. Default: 1.
        Setting k=1 ignores the full-resolution scale to suppress noise.
        If k >= len(scales), a warning is issued and k is set to len(scales)-1.
    peaks_per_scale : int or None, optional
        Maximum number of peaks to extract per scale. If None, extract all peaks
        above threshold. Default: None (unlimited).
    min_distance : float, optional
        Minimum Euclidean distance between candidates (in voxels). Default: 2.0.
        Closer candidates are deduplicated using farthest-first selection,
        keeping the higher-energy peak.
    threshold_rel : float, optional
        Relative threshold for peak detection (0.0 to 1.0). Default: 0.1.
        Peaks must be at least threshold_rel * max_intensity_in_scale to be considered.
    decompose_kwargs : dict or None, optional
        Additional keyword arguments passed to decompose_image().
        Common options:
        - n_iters: optimization iterations (default 500)
        - energy_weight: hierarchical energy penalty (default 0.01)
        - loss_type: "l1" (default), "mse", or "poisson"
        - lr: learning rate (default 0.01)
    verbose : bool, optional
        Print progress information. Default: False.

    Returns
    -------
    candidates : np.ndarray, shape (N, ndim)
        Candidate center coordinates in voxel units (float).
        Sorted by energy (descending).

    Notes
    -----
    - Ignoring the finest k scales (default k=1) suppresses noise and overfitting
    - Scales are processed from coarse to fine; coarser scales contribute first
    - Candidates are deduplicated spatially using farthest-first with min_distance
    - Peak positions are mapped from scale resolution to full resolution
    - Energy-based sorting ensures high-quality candidates are prioritized

    Examples
    --------
    >>> from skimage import data
    >>> import numpy as np
    >>> from luxar.gsplats.candidates import find_candidates_from_decomposition
    >>>
    >>> # Load example image
    >>> image = data.cell().astype(np.float32)
    >>>
    >>> # Generate candidates (ignore finest scale to suppress noise)
    >>> candidates = find_candidates_from_decomposition(
    ...     image,
    ...     scales=[1, 2, 4, 8],
    ...     ignore_finest_k=1,
    ...     min_distance=3.0,
    ...     verbose=True
    ... )
    >>> print(f"Generated {len(candidates)} candidate locations")

    See Also
    --------
    decompose_image : Multi-scale image decomposition
    find_candidates_multiscale_gaussian : Alternative multiscale Gaussian candidate generation
    """
    # Lazy import to avoid circular dependency
    from luxar.gsplats.multiscale.decompose import decompose_image

    # Input validation
    V = np.asarray(V, dtype=float)
    if V.size == 0:
        raise ValueError("Input array V cannot be empty")
    if V.ndim == 0:
        raise ValueError("Input array V must have at least 1 dimension")

    ndim = V.ndim

    # Validate scales
    if not scales or len(scales) == 0:
        raise ValueError("scales must be a non-empty list")
    if any(s <= 0 for s in scales):
        raise ValueError("All scale values must be positive")

    # Validate ignore_finest_k
    if ignore_finest_k < 0:
        raise ValueError("ignore_finest_k must be non-negative")
    if ignore_finest_k >= len(scales):
        import warnings

        warnings.warn(
            f"ignore_finest_k={ignore_finest_k} >= len(scales)={len(scales)}. "
            f"Using ignore_finest_k={len(scales) - 1} instead."
        )
        ignore_finest_k = len(scales) - 1

    # Validate other parameters
    if peaks_per_scale is not None and peaks_per_scale <= 0:
        raise ValueError("peaks_per_scale must be positive when specified")
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")
    if not (0 <= threshold_rel <= 1):
        raise ValueError("threshold_rel must be between 0 and 1")

    # Prepare decompose_image kwargs
    decompose_kwargs = decompose_kwargs or {}

    # Set verbose in decompose_kwargs if not already set
    if "verbose" not in decompose_kwargs:
        decompose_kwargs["verbose"] = verbose

    if verbose:
        aprint(f"[Decomposition Candidates] Input shape: {V.shape}, ndim: {ndim}")
        aprint(f"[Decomposition Candidates] Scales: {scales}")
        aprint(f"[Decomposition Candidates] Ignoring finest {ignore_finest_k} scale(s)")

    # Step 1: Decompose image into multiple scales
    if verbose:
        aprint("[Decomposition Candidates] Running decompose_image...")

    scale_images, stats = decompose_image(V, scales=scales, **decompose_kwargs)

    if verbose:
        aprint(
            f"[Decomposition Candidates] Decomposition complete. "
            f"Converged: {stats.get('converged', False)}, "
            f"Iterations: {stats.get('actual_iters', 'N/A')}"
        )

    # Step 2: Find local maxima in each scale (excluding finest k)
    all_candidates = []
    all_energies = []

    # Determine which scales to process (skip finest k)
    scales_to_process = list(range(ignore_finest_k, len(scales)))

    if verbose:
        aprint(
            f"[Decomposition Candidates] Processing {len(scales_to_process)} "
            f"scale(s) for peak detection"
        )

    # Process scales from coarse to fine (reverse order, excluding ignored finest)
    for scale_idx in reversed(scales_to_process):
        scale_factor = scales[scale_idx]
        scale_img = scale_images[scale_idx]

        if verbose:
            aprint(
                f"[Decomposition Candidates]   Scale {scale_factor}: "
                f"shape {scale_img.shape}, "
                f"range [{scale_img.min():.3f}, {scale_img.max():.3f}]"
            )

        # Compute threshold for this scale
        max_intensity = scale_img.max()
        if max_intensity <= 0:
            if verbose:
                aprint("[Decomposition Candidates]     Skipping (max intensity = 0)")
            continue

        threshold = threshold_rel * max_intensity

        # Find local maxima in scale image
        # Use radius = 1 for finest resolution in scale image (3x3x... neighborhood)
        radius = 1
        peaks = local_maxima(
            scale_img, radius=radius, thresh=threshold, top_k=peaks_per_scale
        )

        if len(peaks) == 0:
            if verbose:
                aprint(
                    f"[Decomposition Candidates]     No peaks found "
                    f"(threshold={threshold:.3f})"
                )
            continue

        if verbose:
            aprint(
                f"[Decomposition Candidates]     Found {len(peaks)} peak(s) "
                f"(threshold={threshold:.3f})"
            )

        # Map peak coordinates to full resolution
        # Peak at position (i, j, ...) in scale image corresponds to
        # position (i*scale_factor + scale_factor/2, j*scale_factor + scale_factor/2, ...)
        # in full resolution image
        candidates_full_res = peaks.astype(float) * scale_factor + scale_factor / 2.0

        # Get energy (intensity) at each peak location
        energies = scale_img[tuple(peaks.T)]

        all_candidates.append(candidates_full_res)
        all_energies.append(energies)

    # Step 3: Combine all candidates
    if len(all_candidates) == 0:
        if verbose:
            aprint("[Decomposition Candidates] No candidates found across all scales")
        return np.zeros((0, ndim), dtype=float)

    candidates = np.vstack(all_candidates)
    energies = np.concatenate(all_energies)

    if verbose:
        aprint(
            f"[Decomposition Candidates] Total candidates before deduplication: "
            f"{len(candidates)}"
        )

    # Step 4: Deduplicate spatially close candidates
    # Use farthest-first selection with energy priority
    candidates_dedup = dedupe_farthest_first(
        candidates, min_distance=min_distance, intensities=energies
    )

    if verbose:
        aprint(
            f"[Decomposition Candidates] Candidates after deduplication: "
            f"{len(candidates_dedup)}"
        )

    # Step 5: Return candidates (already sorted by energy from dedupe_farthest_first)
    return candidates_dedup
