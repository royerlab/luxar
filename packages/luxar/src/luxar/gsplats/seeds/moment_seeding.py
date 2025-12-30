# moment_seeding.py
"""
Moment-based seed generation for Gaussian splatting.

This module provides seed generation using image moments to estimate
full covariance matrices (not just isotropic). Unlike fit_moment_pursuit,
this does NOT use NNLS - amplitudes are simply peak intensities.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from arbol import aprint, asection

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.utils.trils import pack_tril


def seed_from_moments(
    V: np.ndarray,
    *,
    # Scale decomposition
    scales: Sequence[int] = (1, 2, 4, 8),
    decomp_n_iters: int = 200,
    decomp_lr: float = 0.01,
    decomp_energy_weight: float = 0.01,
    # Peak detection
    nms_radius_vox: float = 2.0,
    peak_threshold_rel: float = 0.1,
    max_peaks_per_scale: Optional[int] = None,
    skip_finest_scales: int = 1,
    # Moment computation
    moment_radius_scale: float = 1.5,
    min_eigenvalue: float = 0.25,
    max_eigenvalue: float = 64.0,
    # Output
    device: Optional[str] = None,
    verbose: bool = True,
) -> GSplatData:
    """
    Seed Gaussian splats using image moments for full covariance estimation.

    This method provides the richest initialization by computing actual
    covariance matrices from local image moments, capturing anisotropic
    (elliptical) features. Unlike seed_from_gaussian and seed_from_decomposition
    which use isotropic (spherical) Gaussians.

    **Key difference from fit_moment_pursuit**: No NNLS amplitude solving.
    Amplitudes are simply peak intensities - much faster, and the subsequent
    fit_gaussian_splats will optimize amplitudes anyway.

    Parameters
    ----------
    V : np.ndarray
        Input image/volume to fit (nD supported).

    Scale Decomposition Parameters
    ------------------------------
    scales : Sequence[int], default=(1, 2, 4, 8)
        Scale factors for decomposition.
    decomp_n_iters : int, default=200
        Decomposition optimization iterations.
    decomp_lr : float, default=0.01
        Decomposition learning rate.
    decomp_energy_weight : float, default=0.01
        Energy penalty weight in decomposition.

    Peak Detection Parameters
    -------------------------
    nms_radius_vox : float, default=2.0
        Non-maximum suppression radius.
    peak_threshold_rel : float, default=0.1
        Relative intensity threshold (0-1).
    max_peaks_per_scale : int, optional
        Maximum peaks per scale.
    skip_finest_scales : int, default=1
        Number of finest scales to skip (suppress noise).

    Moment Computation Parameters
    -----------------------------
    moment_radius_scale : float, default=1.5
        Moment integration radius as multiple of scale.
    min_eigenvalue : float, default=0.25
        Minimum covariance eigenvalue.
    max_eigenvalue : float, default=64.0
        Maximum covariance eigenvalue.

    Returns
    -------
    GSplatData
        Gaussian splat seeds with:
        - centers: Centroid-refined peak positions
        - amplitudes: Peak intensities (no NNLS)
        - cholesky_factors: Moment-derived Cholesky factors (full covariance)
        - sharpnesses: All set to 2.0 (standard Gaussian)

    Notes
    -----
    This is the most accurate seeding method for anisotropic features,
    but also the slowest due to moment computation. For most cases,
    seed_from_decomposition provides a good balance.
    """
    import time

    start_time = time.time()

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    ndim = V.ndim
    shape = V.shape

    # Handle empty/zero images
    if V.max() <= 0:
        if verbose:
            aprint("Input image is empty (max <= 0), returning empty result")
        return GSplatData(
            centers=np.zeros((0, ndim), dtype=np.float32),
            amplitudes=np.zeros(0, dtype=np.float32),
            cholesky_factors=np.zeros((0, ndim * (ndim + 1) // 2), dtype=np.float32),
            sharpnesses=np.zeros(0, dtype=np.float32),
        )

    if verbose:
        aprint(f"Moment-based seeding on {ndim}D image {shape}")
        aprint(f"Scales: {list(scales)}, Device: {device}")

    # Phase 1: Multi-scale decomposition
    with asection("Phase 1: Multi-scale decomposition") if verbose else nullcontext():
        from luxar.gsplats.multiscale.decompose import decompose_image

        scale_list = list(scales)
        scales_result, decomp_stats = decompose_image(
            V,
            scales=scale_list,
            n_iters=decomp_n_iters,
            lr=decomp_lr,
            energy_weight=decomp_energy_weight,
            verbose=verbose,
            device=device,
        )

        if verbose:
            aprint(f"Decomposition complete: {len(scales_result)} scale components")

    # Phase 2: Per-scale peak finding and moment computation
    all_centers = []
    all_covariances = []
    all_masses = []

    with asection("Phase 2: Peak finding and moment computation") if verbose else nullcontext():
        for scale_idx, (scale_factor, V_scale) in enumerate(zip(scale_list, scales_result)):
            # Skip finest scales
            if scale_idx < skip_finest_scales:
                if verbose:
                    aprint(f"Scale {scale_factor}x: skipped (finest)")
                continue

            if verbose:
                aprint(f"Scale {scale_factor}x: processing...")

            # NMS radius handling
            min_dim = min(V_scale.shape)
            effective_nms_radius = max(nms_radius_vox, 2.0 if min_dim < 16 else 1.0)

            peaks = _find_peaks_in_scale(
                V_scale,
                nms_radius_vox=effective_nms_radius,
                threshold_rel=peak_threshold_rel,
                max_peaks=max_peaks_per_scale,
                device=device,
            )

            if len(peaks) == 0:
                if verbose:
                    aprint("  No peaks found")
                continue

            if verbose:
                aprint(f"  Found {len(peaks)} peaks")

            # Compute moments at peaks
            moment_radius = moment_radius_scale
            centers, covariances, masses = _compute_local_moments(
                V_scale,
                peaks,
                radius=moment_radius,
                min_eigenvalue=min_eigenvalue / (scale_factor**2),
                max_eigenvalue=max_eigenvalue / (scale_factor**2),
            )

            # Filter zero-mass peaks
            valid_mask = masses > 1e-10
            if not np.any(valid_mask):
                continue

            centers = centers[valid_mask]
            covariances = covariances[valid_mask]
            masses = masses[valid_mask]

            # Scale back to full resolution
            centers_full = centers * scale_factor
            covariances_full = covariances * (scale_factor**2)

            all_centers.append(centers_full)
            all_covariances.append(covariances_full)
            all_masses.append(masses)

            if verbose:
                aprint(f"  Kept {len(centers)} valid splats")

    # Phase 3: Combine all scales
    if not all_centers:
        if verbose:
            aprint("No peaks found in any scale - returning empty result")
        return GSplatData(
            centers=np.zeros((0, ndim), dtype=np.float32),
            amplitudes=np.zeros(0, dtype=np.float32),
            cholesky_factors=np.zeros((0, ndim * (ndim + 1) // 2), dtype=np.float32),
            sharpnesses=np.zeros(0, dtype=np.float32),
        )

    centers = np.concatenate(all_centers, axis=0)
    covariances = np.concatenate(all_covariances, axis=0)
    masses = np.concatenate(all_masses, axis=0)

    N_before_dedup = len(centers)
    if verbose:
        aprint(f"Total splats from all scales (before dedup): {N_before_dedup}")

    # Deduplicate
    centers, covariances, masses = _deduplicate_centers(
        centers, covariances, masses, min_distance=nms_radius_vox
    )

    N = len(centers)
    if verbose and N < N_before_dedup:
        aprint(f"After deduplication: {N} splats")

    # Convert covariances to Cholesky factors
    cholesky_factors = _covariance_to_cholesky(covariances)

    # Get amplitudes from original image (no NNLS!)
    centers_int = np.clip(np.round(centers).astype(int), 0, np.array(V.shape) - 1)
    amplitudes = V[tuple(centers_int.T)].astype(np.float32)

    # Standard sharpness
    sharpnesses = np.full(N, 2.0, dtype=np.float32)

    end_time = time.time()
    if verbose:
        aprint(f"Moment seeding complete: {N} splats in {end_time - start_time:.2f}s")

    return GSplatData(
        centers=centers.astype(np.float32),
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        sharpnesses=sharpnesses,
    )


def _find_peaks_in_scale(
    V_scale: np.ndarray,
    nms_radius_vox: float,
    threshold_rel: float,
    max_peaks: Optional[int],
    device: str,
) -> np.ndarray:
    """Find local maxima in a scale component using NMS."""
    from luxar.gsplats.fitting.dynamic_ops.peak_finding import (
        _find_residual_peaks_global,
    )

    V_tensor = torch.from_numpy(V_scale).float().to(device)

    threshold = threshold_rel * V_tensor.max().item()
    V_thresholded = V_tensor.clone()
    V_thresholded[V_thresholded < threshold] = 0

    k_max = max_peaks if max_peaks is not None else 10000

    try:
        peak_tuples = _find_residual_peaks_global(
            V_thresholded, k_max_residuals=k_max, nms_radius_vox=nms_radius_vox
        )
    except Exception:
        peak_tuples = _simple_peak_finding(V_thresholded, nms_radius_vox)

    if not peak_tuples:
        return np.zeros((0, V_scale.ndim), dtype=np.float32)

    return np.array(peak_tuples, dtype=np.float32)


def _simple_peak_finding(
    V: torch.Tensor, nms_radius: float
) -> List[Tuple[int, ...]]:
    """Simple fallback peak finding using max pooling."""
    import torch.nn.functional as F

    ndim = V.ndim
    kernel_size = int(2 * nms_radius + 1) | 1

    pad = kernel_size // 2
    V_padded = V.unsqueeze(0).unsqueeze(0)

    if ndim == 2:
        pooled = F.max_pool2d(V_padded, kernel_size, stride=1, padding=pad)
    elif ndim == 3:
        pooled = F.max_pool3d(V_padded, kernel_size, stride=1, padding=pad)
    else:
        return []

    pooled = pooled.squeeze(0).squeeze(0)
    peaks_mask = (V == pooled) & (V > 0)
    peak_indices = torch.nonzero(peaks_mask, as_tuple=False)

    return [tuple(p.tolist()) for p in peak_indices]


def _compute_local_moments(
    image: np.ndarray,
    centers: np.ndarray,
    radius: float,
    *,
    min_eigenvalue: float = 0.25,
    max_eigenvalue: float = 16.0,
    use_gaussian_weighting: bool = True,
    regularization: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute image moments at peak locations.

    Returns (refined_centers, covariances, masses).
    """
    ndim = image.ndim
    N = len(centers)
    int_radius = int(np.ceil(radius))

    refined_centers = np.zeros((N, ndim), dtype=np.float32)
    covariances = np.zeros((N, ndim, ndim), dtype=np.float32)
    masses = np.zeros(N, dtype=np.float32)

    # Pre-compute offset grid
    window_size = 2 * int_radius + 1
    offsets_1d = np.arange(-int_radius, int_radius + 1, dtype=np.float32)
    offset_grids = np.meshgrid(*([offsets_1d] * ndim), indexing="ij")
    offsets = np.stack([g.ravel() for g in offset_grids], axis=1)

    # Gaussian weights
    if use_gaussian_weighting:
        dist_sq = (offsets**2).sum(axis=1)
        gauss_weights = np.exp(-dist_sq / (2 * radius**2))
    else:
        gauss_weights = np.ones(len(offsets), dtype=np.float32)

    for i, center in enumerate(centers):
        center_int = np.round(center).astype(int)

        # Build slices with boundary handling
        slices = []
        valid_mask = np.ones(len(offsets), dtype=bool)

        for d in range(ndim):
            lo = center_int[d] - int_radius
            hi = center_int[d] + int_radius + 1

            lo_clamp = max(0, lo)
            hi_clamp = min(image.shape[d], hi)

            slices.append(slice(lo_clamp, hi_clamp))

            # Track valid offset indices
            offset_lo = lo_clamp - lo
            offset_hi = window_size - (hi - hi_clamp)
            offset_indices = np.arange(window_size)
            valid_offset_indices = offset_indices[offset_lo:offset_hi]
            dim_valid = np.zeros(window_size, dtype=bool)
            dim_valid[valid_offset_indices] = True
            shape = [1] * ndim
            shape[d] = window_size
            dim_valid_full = np.tile(
                dim_valid.reshape(shape), [window_size if j != d else 1 for j in range(ndim)]
            ).ravel()
            valid_mask &= dim_valid_full

        patch = image[tuple(slices)]

        if patch.size == 0:
            refined_centers[i] = center
            covariances[i] = np.eye(ndim) * min_eigenvalue
            masses[i] = 0.0
            continue

        patch_flat = patch.ravel()
        valid_offsets = offsets[valid_mask][: len(patch_flat)]
        valid_weights = gauss_weights[valid_mask][: len(patch_flat)]

        weighted_intensities = patch_flat * valid_weights
        mass = weighted_intensities.sum()
        masses[i] = mass

        if mass < 1e-10:
            refined_centers[i] = center
            covariances[i] = np.eye(ndim) * min_eigenvalue
            continue

        # Centroid
        centroid_offset = (valid_offsets * weighted_intensities[:, None]).sum(axis=0) / mass
        refined_centers[i] = center_int + centroid_offset

        # Covariance
        centered_offsets = valid_offsets - centroid_offset
        cov = np.zeros((ndim, ndim), dtype=np.float64)
        for p in range(len(patch_flat)):
            outer = np.outer(centered_offsets[p], centered_offsets[p])
            cov += outer * weighted_intensities[p]
        cov /= mass

        # Regularize and clamp eigenvalues
        cov += np.eye(ndim) * regularization
        cov = _clamp_eigenvalues(cov, min_eigenvalue, max_eigenvalue)
        covariances[i] = cov.astype(np.float32)

    return refined_centers, covariances, masses


def _clamp_eigenvalues(cov: np.ndarray, min_ev: float, max_ev: float) -> np.ndarray:
    """Clamp eigenvalues of covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_clamped = np.clip(eigvals, min_ev, max_ev)
    return eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T


def _covariance_to_cholesky(
    covariances: np.ndarray, regularization: float = 1e-6
) -> np.ndarray:
    """Convert covariance matrices to packed Cholesky factors."""
    from scipy.linalg import cholesky

    N, d, _ = covariances.shape
    L_matrices = np.zeros((N, d, d), dtype=np.float32)

    for i in range(N):
        cov = covariances[i] + np.eye(d) * regularization
        try:
            L_matrices[i] = cholesky(cov, lower=True)
        except np.linalg.LinAlgError:
            # Fallback: eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, regularization)
            L_matrices[i] = eigvecs @ np.diag(np.sqrt(eigvals))

    return pack_tril(L_matrices)


def _deduplicate_centers(
    centers: np.ndarray,
    covariances: np.ndarray,
    masses: np.ndarray,
    min_distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove duplicate centers within min_distance."""
    if len(centers) == 0:
        return centers, covariances, masses

    # Sort by mass descending
    order = np.argsort(-masses)

    kept_indices = []
    kept_centers = []

    for idx in order:
        center = centers[idx]

        if len(kept_centers) > 0:
            kept_array = np.array(kept_centers)
            distances = np.linalg.norm(kept_array - center, axis=1)
            if np.any(distances < min_distance):
                continue

        kept_indices.append(idx)
        kept_centers.append(center)

    kept_indices = np.array(kept_indices)

    return (
        centers[kept_indices],
        covariances[kept_indices],
        masses[kept_indices],
    )
