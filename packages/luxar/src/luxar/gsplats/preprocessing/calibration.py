"""
Noise2Self (J-invariant) calibration for Non-Local Means h parameter.

Implements GPU-accelerated grid search over h values using the J-invariant
trick: mask each pixel from its own denoising estimate, then measure
self-consistency to find the optimal h without a clean reference image.
"""

from __future__ import annotations

import itertools
from typing import Sequence, Union

import numpy as np
import torch


def _generate_j_invariant_masks(
    shape: tuple[int, ...],
    stride: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Generate a set of complementary J-invariant masks.

    For stride *s*, produces *s^ndim* boolean masks that tile the volume in a
    checkerboard-like pattern.  Each mask selects the pixels that are held out
    in one phase of the cross-validation.

    Returns a list of boolean tensors of the given *shape*.
    """
    ndim = len(shape)
    masks: list[torch.Tensor] = []

    # Generate all offset combinations for the stride grid
    offsets = [range(stride)] * ndim

    for combo in itertools.product(*offsets):
        # Create grid indices for this offset
        grids = []
        for dim_idx in range(ndim):
            idx = torch.arange(combo[dim_idx], shape[dim_idx], stride, device=device)
            grids.append(idx)

        # Build the boolean mask via meshgrid + scatter
        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        if ndim == 2:
            gy, gx = torch.meshgrid(grids[0], grids[1], indexing="ij")
            mask[gy, gx] = True
        elif ndim == 3:
            gz, gy, gx = torch.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
            mask[gz, gy, gx] = True
        else:
            # Fallback for arbitrary ndim
            mg = torch.meshgrid(*grids, indexing="ij")
            mask[tuple(mg)] = True

        masks.append(mask)

    return masks


def _interpolate_masked(
    volume: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Replace masked pixels with the mean of their unmasked neighbours.

    Simple nearest-neighbour interpolation: for each masked pixel, average
    the immediate face-connected neighbours that are *not* masked.  This
    removes the center pixel's contribution from its own denoising estimate
    (the core J-invariant requirement).
    """
    filled = volume.clone()
    ndim = volume.ndim

    # Sum of unmasked neighbours and count
    neighbour_sum = torch.zeros_like(volume)
    neighbour_count = torch.zeros_like(volume)

    for dim in range(ndim):
        for shift in (-1, 1):
            shifted = torch.roll(volume, shifts=shift, dims=dim)
            shifted_mask = torch.roll(mask, shifts=shift, dims=dim)
            # Only count neighbours that are not masked
            valid = ~shifted_mask
            neighbour_sum += shifted * valid
            neighbour_count += valid.float()

    # Replace masked pixels with neighbour average
    safe_count = neighbour_count.clamp(min=1.0)
    filled[mask] = (neighbour_sum / safe_count)[mask]
    return filled


def calibrate_nlm_h(
    volume: torch.Tensor,
    h_range: Sequence[float] | None = None,
    patch_size: int = 3,
    search_distance: int = 5,
    *,
    stride: int = 2,
    backend: str = "auto",
    device: Union[str, torch.device, None] = None,
    use_2d_slice: bool = True,
    slice_index: int | None = None,
) -> float:
    """Find optimal NLM filtering strength *h* via Noise2Self cross-validation.

    The J-invariant method masks subsets of pixels, denoises without their
    contribution, then measures how well the denoised values predict the
    held-out originals.  The *h* with the lowest mean squared error wins.

    Parameters
    ----------
    volume : torch.Tensor
        2D ``(H, W)`` or 3D ``(D, H, W)`` input tensor.
    h_range : sequence of float, optional
        Candidate *h* values.  Default: ``arange(0.005, 0.08, 0.005)``.
    patch_size : int
        Comparison patch side length (odd).
    search_distance : int
        Search window half-size.
    stride : int
        Stride for J-invariant mask grid.  Smaller stride = more masks =
        more accurate but slower.  Default 2 (4 masks in 2D, 8 in 3D).
    backend : str
        Backend for the inner ``denoise_nlm`` calls.
    device : str or torch.device, optional
        Target device for calibration computation.
    use_2d_slice : bool
        If *True* and volume is 3D, calibrate on a single 2D slice for
        speed (matching the typical microscopy workflow). For volumes
        of other dimensionality (1D, 2D, or 4D+), this flag is silently
        ignored and calibration runs on the full volume — graceful
        fallback rather than an error, since the slice optimization is
        only meaningful for the 3D microscopy case.
    slice_index : int, optional
        Which z-slice to use when ``use_2d_slice=True`` and the volume
        is 3D. Default: middle slice. Ignored for non-3D volumes.

    Returns
    -------
    float
        Optimal *h* parameter.
    """
    from .nlm_core import denoise_nlm

    if h_range is None:
        h_range = np.arange(0.005, 0.08, 0.005).tolist()

    if device is not None:
        volume = volume.to(torch.device(device))

    # For 3D volumes, calibrate on a 2D slice for speed. For 1D/2D/4D+
    # volumes the slice-trick doesn't apply, so we silently calibrate on
    # the full volume — see the docstring for `use_2d_slice` above.
    if use_2d_slice and volume.ndim == 3:
        z_idx = slice_index if slice_index is not None else volume.shape[0] // 2
        cal_data = volume[z_idx].clone()
    else:
        cal_data = volume.clone()

    cal_data = cal_data.float()

    masks = _generate_j_invariant_masks(cal_data.shape, stride, device=cal_data.device)

    best_h = h_range[0]
    best_mse = float("inf")

    for h_val in h_range:
        total_mse = 0.0

        for mask in masks:
            # Replace masked pixels with neighbour interpolation
            masked_input = _interpolate_masked(cal_data, mask)

            # Denoise the masked input
            denoised = denoise_nlm(
                masked_input,
                h=h_val,
                patch_size=patch_size,
                search_distance=search_distance,
                backend=backend,
            )

            # Measure prediction error only on held-out pixels
            diff = (denoised[mask] - cal_data[mask]) ** 2
            total_mse += diff.mean().item()

        total_mse /= len(masks)

        if total_mse < best_mse:
            best_mse = total_mse
            best_h = h_val

    return float(best_h)
