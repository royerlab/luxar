# peaks.py
"""Intensity-weighted seeding for sparse residuals.

Seeds are placed at non-zero voxels, sampled with probability proportional
to intensity. Every seed lands on actual signal — no seeds wasted on
zero-background. The default ``init_sigma`` is derived from expected
inter-seed spacing so seeds can cover their allotted neighbourhood.

Designed for progressive fitting where residuals are sparse (mostly zero
with scattered structures of varying shape — peaks, plateaus, edges).

All heavy computation is in PyTorch for GPU acceleration.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from arbol import aprint, asection

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.seeds.gpu_ops import _get_device, should_use_gpu
from luxar.gsplats.seeds.utils import SEED_AMPLITUDE_SCALE, sigmas_to_cholesky_isotropic


def seed_from_peaks(
    V: np.ndarray,
    n_seeds: Optional[int] = None,
    init_sigma: Optional[float] = None,
    device: Optional[str] = None,
) -> GSplatData:
    """Generate seeds at non-zero voxels, weighted by intensity.

    Samples ``n_seeds`` locations from the non-zero voxels of V, with
    probability proportional to voxel intensity.  Brighter voxels are
    more likely to receive a seed.  Unless overridden, ``init_sigma`` is
    auto-scaled to roughly half the expected inter-seed spacing (with a
    floor of 1.5 voxels) so that splats have enough support to generate
    useful gradients without massively overlapping.

    This method is ideal for sparse residuals in progressive fitting
    where the signal has varying shape (peaks, plateaus, edges) and
    peak-detection would miss non-extremal structures.

    Parameters
    ----------
    V : np.ndarray
        Input volume (any dimensionality). Zero voxels are ignored.
    n_seeds : int, optional
        Number of seeds to generate. If None or larger than the number
        of non-zero voxels, returns one seed per non-zero voxel.
    init_sigma : float, optional
        Initial Gaussian sigma for seed splats. If None, auto-scaled based
        on expected inter-seed spacing: ``(non_zero_voxels / n_seeds)^(1/d) / 2``.
        This ensures splats are large enough for gradients but don't
        massively overlap and overshoot.
    device : str, optional
        PyTorch device ('cuda', 'mps', 'cpu', or None for auto).

    Returns
    -------
    GSplatData
        Seeds with centers at sampled non-zero voxels, amplitudes from
        V, and isotropic Cholesky factors at ``init_sigma``.
    """
    ndim = V.ndim

    if init_sigma is None:
        # Auto-scale based on expected inter-seed spacing in the signal
        # region.  Use the "significant" voxels (above median of non-zero
        # values) as the effective volume — the diffuse background inflates
        # the non-zero count but isn't where splats should focus.
        nonzero_vals_np = V[V > 0]
        if len(nonzero_vals_np) > 0:
            median_val = float(np.median(nonzero_vals_np))
            n_significant = max(1, int((V > median_val).sum()))
        else:
            n_significant = max(1, int(np.prod(V.shape)))
        n_target = n_seeds if n_seeds is not None else n_significant
        spacing = (n_significant / max(1, n_target)) ** (1.0 / ndim)
        init_sigma = max(1.5, spacing / 2.0)

    # Resolve device (None → "auto" for GPU auto-detection)
    effective_device = device if device is not None else "auto"
    use_gpu = should_use_gpu(V, effective_device)
    dev = _get_device(effective_device) if use_gpu else torch.device("cpu")

    with asection(f"Peak seeding ({ndim}D)"):
        V_tensor = torch.from_numpy(V.astype(np.float32)).to(dev)

        # Find non-zero voxels
        nonzero_mask = V_tensor > 0
        nonzero_coords = torch.nonzero(nonzero_mask, as_tuple=False)  # (M, ndim)
        n_nonzero = len(nonzero_coords)

        if n_nonzero == 0:
            aprint("No non-zero voxels found")
            from luxar.gsplats.utils.trils import tril_size

            return GSplatData(
                centers=np.zeros((0, ndim), dtype=np.float32),
                amplitudes=np.zeros(0, dtype=np.float32),
                cholesky_factors=np.zeros((0, tril_size(ndim)), dtype=np.float32),
            )

        # Get intensities at non-zero locations
        intensities = V_tensor[tuple(nonzero_coords.T)]  # (M,)

        # Determine how many seeds
        if n_seeds is None or n_seeds >= n_nonzero:
            # Use all non-zero voxels
            selected_coords = nonzero_coords
            selected_intensities = intensities
        else:
            # Sample n_seeds locations weighted by intensity.
            # torch.multinomial has a 2^24 category limit — if we exceed it,
            # pre-filter to the top candidates by intensity first.
            max_categories = 2**24 - 1
            if n_nonzero > max_categories:
                # Keep top max_categories by intensity (fast top-k on GPU)
                _, topk_idx = torch.topk(intensities, max_categories)
                nonzero_coords = nonzero_coords[topk_idx]
                intensities = intensities[topk_idx]

            probs = intensities / intensities.sum()
            indices = torch.multinomial(probs, n_seeds, replacement=False)
            selected_coords = nonzero_coords[indices]
            selected_intensities = intensities[indices]

        # Sort by intensity (brightest first)
        sorted_idx = torch.argsort(selected_intensities, descending=True)
        selected_coords = selected_coords[sorted_idx]
        selected_intensities = selected_intensities[sorted_idx]

        # Transfer to CPU
        centers = selected_coords.cpu().numpy().astype(np.float32)
        amplitudes = selected_intensities.cpu().numpy().astype(np.float32)
        amplitudes = amplitudes * SEED_AMPLITUDE_SCALE

        # Isotropic Cholesky at init_sigma (auto-scaled to ~half the
        # expected inter-seed spacing unless overridden)
        cholesky = sigmas_to_cholesky_isotropic(
            np.full(len(centers), init_sigma, dtype=np.float32), ndim
        )

        aprint(
            f"Sampled {len(centers)} seeds from {n_nonzero:,} non-zero voxels "
            f"(init_sigma={init_sigma:.3f})"
        )

    return GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky,
    )
