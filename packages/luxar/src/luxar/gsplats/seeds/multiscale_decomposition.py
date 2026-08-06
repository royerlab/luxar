# multiscale_decomposition.py
"""
Decomposition-based seed generation for Gaussian splatting.

This module provides scale-hierarchical detection via multi-scale image
decomposition. Returns GSplatData with scale-informed Gaussian shapes
where sigma = scale_factor for each detected seed.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from arbol import aprint

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.seeds.utils import (
    SEED_AMPLITUDE_SCALE,
    local_maxima,
    sigmas_to_cholesky_isotropic,
)


def seed_from_decomposition(
    V: np.ndarray,
    scales: Optional[List[int]] = None,
    ignore_finest_k: int = 1,
    peaks_per_scale: Optional[int] = None,
    min_distance: float = 2.0,
    threshold_rel: float = 0.1,
    decompose_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    device: Optional[str] = None,
) -> GSplatData:
    """
    Generate seed Gaussian splats using multi-scale decomposition.

    This method decomposes the input image into multiple scales using
    `decompose_image()`, finds local maxima in each scale (excluding the
    finest k scales), and returns
    GSplatData with isotropic Gaussians where sigma = scale_factor.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume. Shape: (s_0, s_1, ..., s_{n-1}).
    scales : List[int], default=[1, 2, 4, 8, 16, 32, 64]
        Scale factors for decomposition.
        Scale 1 = full resolution, scale 2 = half resolution, etc.
        Each seed's sigma is set to the scale at which it was detected.
    ignore_finest_k : int, default=1
        Number of finest scales to ignore for peak detection.
        Setting k=1 ignores the full-resolution scale to suppress noise.
    peaks_per_scale : int or None, optional
        Maximum number of peaks to extract per scale. If None, extract all.
    min_distance : float, default=2.0
        Minimum Euclidean distance between seeds (in voxels).
    threshold_rel : float, default=0.1
        Relative threshold for peak detection (0.0 to 1.0).
    decompose_kwargs : dict or None, optional
        Additional keyword arguments passed to decompose_image().
    verbose : bool, default=False
        Print progress information.
    device : str, optional
        PyTorch device for GPU acceleration. Options:
        - None (default): CPU using scipy.ndimage
        - 'cpu': Force CPU
        - 'cuda': NVIDIA GPU (if available)
        - 'mps': Apple Metal (if available)
        - 'auto': Auto-detect best device

        Forwarded to ``decompose_image()``; GPU acceleration provides
        substantial speedup for the decomposition on large volumes (>100³),
        with the magnitude depending on GPU and problem size. Peak detection
        itself always runs on the CPU.

    Returns
    -------
    GSplatData
        Gaussian splat seeds with:
        - centers: Peak positions in full resolution coordinates
        - amplitudes: Peak intensities
        - cholesky_factors: Isotropic Cholesky factors where sigma = scale_factor
        - Standard Gaussian profile (no sharpness parameter)

    Notes
    -----
    The sigma for each seed equals the decomposition scale_factor at which
    it was detected. Features at scale=4 will have sigma=4 voxels.
    """
    # Lazy import to avoid circular dependency
    from luxar.gsplats.multiscale.decompose import decompose_image

    # Default scales
    if scales is None:
        scales = [1, 2, 4, 8, 16, 32, 64]

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
    if "verbose" not in decompose_kwargs:
        decompose_kwargs["verbose"] = verbose
    if "device" not in decompose_kwargs and device is not None:
        decompose_kwargs["device"] = device

    if verbose:
        aprint(f"[Decomposition Seeds] Input shape: {V.shape}, ndim: {ndim}")
        aprint(f"[Decomposition Seeds] Scales: {scales}")
        aprint(f"[Decomposition Seeds] Ignoring finest {ignore_finest_k} scale(s)")
        if device is not None:
            aprint(f"[Decomposition Seeds] Using device: {device}")

    # Step 1: Decompose image into multiple scales
    if verbose:
        aprint("[Decomposition Seeds] Running decompose_image...")

    scale_images, stats = decompose_image(V, scales=scales, **decompose_kwargs)
    actual_scales = stats.get("scales", scales)

    if verbose:
        aprint(
            f"[Decomposition Seeds] Decomposition complete. "
            f"Converged: {stats.get('converged', False)}"
        )

    # Step 2: Find local maxima in each scale (excluding finest k)
    all_seeds: List[np.ndarray] = []
    all_scales_detected: List[np.ndarray] = []
    all_energies: List[np.ndarray] = []

    scales_to_process = list(range(ignore_finest_k, len(actual_scales)))

    if verbose:
        aprint(f"[Decomposition Seeds] Processing {len(scales_to_process)} scale(s)")

    # Process scales from coarse to fine
    for scale_idx in reversed(scales_to_process):
        scale_factor = actual_scales[scale_idx]
        scale_img = scale_images[scale_idx]

        if verbose:
            aprint(
                f"[Decomposition Seeds]   Scale {scale_factor}: shape {scale_img.shape}"
            )

        # Compute threshold for this scale
        max_intensity = scale_img.max()
        if max_intensity <= 0:
            if verbose:
                aprint("[Decomposition Seeds]     Skipping (max intensity = 0)")
            continue

        threshold = threshold_rel * max_intensity

        # Find local maxima in scale image
        radius = 1
        peaks = local_maxima(
            scale_img, radius=radius, thresh=threshold, top_k=peaks_per_scale
        )

        if len(peaks) == 0:
            if verbose:
                aprint("[Decomposition Seeds]     No peaks found")
            continue

        if verbose:
            aprint(f"[Decomposition Seeds]     Found {len(peaks)} peak(s)")

        # Map peak coordinates to full resolution
        seeds_full_res = peaks.astype(float) * scale_factor + scale_factor / 2.0

        # Get energy (intensity) at each peak location
        energies = scale_img[tuple(peaks.T)]

        all_seeds.append(seeds_full_res)
        all_scales_detected.append(np.full(len(peaks), float(scale_factor)))
        all_energies.append(energies)

    # Step 3: Combine all seeds
    if len(all_seeds) == 0:
        if verbose:
            aprint("[Decomposition Seeds] No seeds found across all scales")
        return GSplatData(
            centers=np.zeros((0, ndim), dtype=np.float32),
            amplitudes=np.zeros(0, dtype=np.float32),
            cholesky_factors=np.zeros((0, ndim * (ndim + 1) // 2), dtype=np.float32),
        )

    seeds = np.vstack(all_seeds)
    seed_scales = np.concatenate(all_scales_detected)
    energies = np.concatenate(all_energies)

    if verbose:
        aprint(f"[Decomposition Seeds] Total seeds before dedup: {len(seeds)}")

    # Step 4: Deduplicate spatially close seeds
    seeds, seed_scales, energies = _dedupe_with_scales_and_energies(
        seeds, seed_scales, energies, min_distance
    )

    if verbose:
        aprint(f"[Decomposition Seeds] Seeds after dedup: {len(seeds)}")

    # Get amplitudes from original image, scaled down to avoid initial over-prediction
    seeds_int = np.clip(np.round(seeds).astype(int), 0, np.array(V.shape) - 1)
    amplitudes = V[tuple(seeds_int.T)].astype(np.float32) * SEED_AMPLITUDE_SCALE

    # Build Cholesky factors from scales (sigma = scale_factor)
    cholesky_factors = sigmas_to_cholesky_isotropic(
        seed_scales.astype(np.float32), ndim
    )

    return GSplatData(
        centers=seeds.astype(np.float32),
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
    )


def _dedupe_with_scales_and_energies(
    coords: np.ndarray,
    scales: np.ndarray,
    energies: np.ndarray,
    min_distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deduplicate seeds while preserving scale and energy information.

    Uses energy-based priority for selection.
    """
    if len(coords) == 0:
        return coords, scales, energies

    # Sort by energy (highest first)
    sort_idx = np.argsort(energies)[::-1]
    coords_sorted = coords[sort_idx]
    scales_sorted = scales[sort_idx]
    energies_sorted = energies[sort_idx]

    # Greedy deduplication
    kept_mask = np.ones(len(coords_sorted), dtype=bool)

    for i in range(len(coords_sorted)):
        if not kept_mask[i]:
            continue

        # Mark nearby seeds as rejected
        diffs = coords_sorted[i + 1 :] - coords_sorted[i]
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        nearby = distances < min_distance
        kept_mask[i + 1 :][nearby] = False

    return (
        coords_sorted[kept_mask],
        scales_sorted[kept_mask],
        energies_sorted[kept_mask],
    )
