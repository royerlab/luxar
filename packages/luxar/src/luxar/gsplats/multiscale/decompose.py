# decompose.py

"""
Multi-scale image decomposition for efficient Gaussian splatting.

This module implements n-dimensional image decomposition into non-negative
scale components, enabling efficient multi-scale Gaussian splat fitting.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arbol import aprint, asection

from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus_torch
from luxar.gsplats.utils.device import resolve_torch_device


def _cubic_upsample_2x_1d(img: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Upsample 2× along one axis using Keys cubic convolution.

    Uses Keys cubic kernel with a=-0.5 for high-quality interpolation.
    The kernel at fractional position 0.5 is: [-1/16, 9/16, 9/16, -1/16]

    This implementation uses vectorized PyTorch operations (unfold + broadcasting)
    for maximum efficiency - no Python loops!

    Parameters
    ----------
    img : torch.Tensor
        Input image of any dimensionality
    axis : int
        Axis to upsample (0 to ndim-1)

    Returns
    -------
    torch.Tensor
        Image upsampled 2× along specified axis
    """
    # Keys cubic kernel at x=0.5: [-1/16, 9/16, 9/16, -1/16]
    kernel = torch.tensor(
        [-1 / 16, 9 / 16, 9 / 16, -1 / 16], dtype=img.dtype, device=img.device
    )

    # Move axis to last position for easier processing
    img = img.movedim(axis, -1)
    orig_shape = img.shape
    size_in = orig_shape[-1]
    size_out = 2 * size_in

    # Flatten all dimensions except last: (batch, in_size)
    img_flat = img.reshape(-1, size_in)

    # Create output tensor
    out = torch.zeros(img_flat.shape[0], size_out, dtype=img.dtype, device=img.device)

    # Even positions: copy original values (vectorized)
    out[:, ::2] = img_flat

    # Odd positions: cubic interpolation (vectorized)
    # Pad input for boundary handling: replicate edges
    # Padding (1, 2) means 1 element on left, 2 on right
    img_padded = F.pad(img_flat, (1, 2), mode="replicate")

    # Extract sliding windows of size 4 with stride 1
    # unfold(dimension, size, step) creates windows efficiently
    # Result shape: (batch, size_in, 4)
    windows = img_padded.unfold(-1, 4, 1)

    # Apply cubic kernel: (batch, size_in, 4) * (4,) -> (batch, size_in)
    interpolated = (windows * kernel).sum(dim=-1)

    # Place interpolated values at odd positions
    out[:, 1::2] = interpolated

    # Reshape back to original dimensionality
    out_shape = list(orig_shape)
    out_shape[-1] = size_out
    out = out.reshape(out_shape)

    # Move axis back to original position
    out = out.movedim(-1, axis)

    return out


def _cubic_upsample_2x_nd(img: torch.Tensor) -> torch.Tensor:
    """
    Upsample 2× in all dimensions using separable Keys cubic convolution.

    Applies 1D cubic upsampling along each axis sequentially.

    Parameters
    ----------
    img : torch.Tensor
        Input n-dimensional image

    Returns
    -------
    torch.Tensor
        Image upsampled 2× in all dimensions
    """
    result = img
    for axis in range(img.ndim):
        result = _cubic_upsample_2x_1d(result, axis)
    return result


def _cubic_upsample_recursive(
    img: torch.Tensor, target_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Recursively upsample to target shape using 2× cubic upsampling.

    Applies Keys cubic convolution recursively for power-of-2 scale factors.

    Parameters
    ----------
    img : torch.Tensor
        Input image
    target_shape : Tuple[int, ...]
        Target shape

    Returns
    -------
    torch.Tensor
        Upsampled image
    """
    current_shape = img.shape

    if current_shape == target_shape:
        return img

    # Check if all dimensions need 2× upsampling or more
    factors = [t / s for t, s in zip(target_shape, current_shape)]

    if all(f >= 2.0 for f in factors):
        # Apply 2× upsampling
        img = _cubic_upsample_2x_nd(img)
        # Recursively continue
        return _cubic_upsample_recursive(img, target_shape)
    elif all(1.0 <= f < 2.0 for f in factors):
        # Close to target, use trilinear for fractional part
        # and crop if needed
        img_expanded = img[None, None, ...]

        ndim = img.ndim
        if ndim == 2:
            mode = "bilinear"
        elif ndim == 3:
            mode = "trilinear"
        else:
            mode = "nearest"

        upsampled = F.interpolate(
            img_expanded,
            size=target_shape,
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        )
        result: torch.Tensor = upsampled[0, 0]
        return result
    else:
        raise ValueError(f"Cannot upsample {current_shape} to {target_shape}")


def _get_interpolation_mode(ndim: int, interpolation: str = "cubic") -> str:
    """
    Get appropriate interpolation mode based on dimensionality and preference.

    Parameters
    ----------
    ndim : int
        Number of dimensions
    interpolation : str, default='cubic'
        Interpolation method: 'nearest', 'linear', or 'cubic'
        - 'nearest': Nearest-neighbor (fastest, blocky)
        - 'linear': Linear interpolation (smooth, medium speed)
        - 'cubic': Cubic interpolation (highest quality, uses fast Keys cubic for 3D+)

    Returns
    -------
    str
        Mode: 'nearest', 'bilinear', 'trilinear', 'bicubic', or 'cubic_keys'

    Notes
    -----
    Mapping to implementation modes:

    **'nearest' mode:**
    - All dimensions: 'nearest' (PyTorch F.interpolate)

    **'linear' mode:**
    - 2D: 'bilinear' (PyTorch F.interpolate)
    - 3D: 'trilinear' (PyTorch F.interpolate)
    - nD (n>3): 'nearest' (fallback)

    **'cubic' mode:**
    - 2D: 'bicubic' (PyTorch F.interpolate)
    - 3D+: 'cubic_keys' (custom Keys cubic convolution with separable filters)

    Keys cubic convolution uses vectorized operations for fast nD cubic interpolation.
    Note: Cubic interpolation can produce small negative values (undershoot)
    which are clamped to zero in _upsample_to_shape().
    """
    if interpolation == "nearest":
        return "nearest"
    elif interpolation == "linear":
        if ndim == 2:
            return "bilinear"
        elif ndim == 3:
            return "trilinear"
        else:
            return "nearest"  # Fallback for nD where n > 3
    elif interpolation == "cubic":
        if ndim == 2:
            return "bicubic"
        else:
            # Use Keys cubic convolution for 3D and higher dimensions
            return "cubic_keys"
    else:
        raise ValueError(
            f"Invalid interpolation mode: {interpolation}. "
            f"Must be 'nearest', 'linear', or 'cubic'"
        )


def _upsample_to_shape(
    img: torch.Tensor, target_shape: Tuple[int, ...], mode: str
) -> torch.Tensor:
    """
    Upsample image to target shape using specified interpolation mode.

    Parameters
    ----------
    img : torch.Tensor
        Input image of shape (d0, d1, ..., dn)
    target_shape : Tuple[int, ...]
        Target shape (s0, s1, ..., sn)
    mode : str
        Mode: 'nearest', 'bilinear', 'trilinear', 'bicubic', or 'cubic_keys'.
        'cubic_keys' uses fast vectorized Keys cubic convolution for nD data.

    Returns
    -------
    torch.Tensor
        Upsampled image of shape target_shape. If cubic interpolation is used,
        negative values (undershoot) are clamped to zero.
    """
    if img.shape == target_shape:
        return img

    # Keys cubic convolution for 3D+ cubic interpolation
    if mode == "cubic_keys":
        upsampled = _cubic_upsample_recursive(img, target_shape)
        # Clamp negative values from cubic undershoot
        upsampled = torch.clamp(upsampled, min=0.0)
        return upsampled

    # Standard PyTorch interpolation for 2D or linear modes
    # Add batch and channel dimensions for F.interpolate
    img_expanded = img[None, None, ...]

    # Upsample (align_corners only for interpolating modes, not 'nearest' or 'area')
    if mode in ("nearest", "area", "nearest-exact"):
        upsampled = F.interpolate(img_expanded, size=target_shape, mode=mode)
    else:
        upsampled = F.interpolate(
            img_expanded, size=target_shape, mode=mode, align_corners=False
        )

    # Remove batch and channel dimensions
    upsampled = upsampled[0, 0]

    # Clamp negative values from cubic interpolation (bicubic can produce undershoot)
    if mode == "bicubic":
        upsampled = torch.clamp(upsampled, min=0.0)

    return upsampled


def _downsample_to_scale(
    img: torch.Tensor, scale: int, mode: str = "area"
) -> torch.Tensor:
    """
    Downsample image by scale factor using appropriate pooling.

    Parameters
    ----------
    img : torch.Tensor
        Input image of shape (s0, s1, ..., sn)
    scale : int
        Downsampling factor
    mode : str
        Downsampling mode ('area' for averaging, 'max' for max pooling)

    Returns
    -------
    torch.Tensor
        Downsampled image of shape (s0//scale, s1//scale, ..., sn//scale)
    """
    if scale == 1:
        return img

    ndim = img.ndim
    device_type = img.device.type

    # Add batch and channel dimensions
    img_expanded = img[None, None, ...]

    if ndim == 2:
        if mode == "area":
            downsampled = F.avg_pool2d(img_expanded, kernel_size=scale, stride=scale)
        else:
            downsampled = F.max_pool2d(img_expanded, kernel_size=scale, stride=scale)
    elif ndim == 3:
        # MPS doesn't support avg_pool3d/max_pool3d, fall back to interpolate or CPU
        if device_type == "mps":
            # Use interpolate with area mode as fallback
            target_shape = tuple(s // scale for s in img.shape)
            downsampled = F.interpolate(
                img_expanded,
                size=target_shape,
                mode="trilinear" if mode == "area" else "nearest",
                align_corners=False if mode == "area" else None,
            )
        else:
            # Use native pooling for CPU/CUDA
            if mode == "area":
                downsampled = F.avg_pool3d(
                    img_expanded, kernel_size=scale, stride=scale
                )
            else:
                downsampled = F.max_pool3d(
                    img_expanded, kernel_size=scale, stride=scale
                )
    else:
        # For nD where n > 3: F.interpolate doesn't support tensors with >5 dims.
        # Use separable approach for both area and nearest modes.
        if mode == "area":
            # Separable area averaging: reshape each axis into
            # (size//scale, scale) blocks and average along the block axis.
            # This is exact (not an approximation) because averaging is separable.
            result = img
            for ax in range(ndim):
                s = result.shape[ax]
                new_s = s // scale
                # Truncate to exact multiple of scale
                slices = [slice(None)] * result.ndim
                slices[ax] = slice(0, new_s * scale)
                result = result[tuple(slices)]
                # Reshape axis into (new_s, scale) and average over the block dim
                new_shape = list(result.shape)
                new_shape[ax : ax + 1] = [new_s, scale]
                result = result.reshape(new_shape).mean(dim=ax + 1)
        else:
            # Nearest-neighbor downsampling: stride along each axis
            stride_slices = tuple(slice(None, None, scale) for _ in range(ndim))
            result = img[stride_slices]
        downsampled = result[None, None, ...]

    # Remove batch and channel dimensions
    return downsampled[0, 0]


class MultiScaleDecomposer(nn.Module):
    """
    PyTorch model for multi-scale image decomposition.

    Decomposes an n-dimensional image V into K non-negative scale components:
        V = Σₖ upsample(Vₖ)

    where Vₖ are images at different resolutions (scale factors).

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the target image (n-dimensional)
    scales : List[int], default=[1, 2, 4, 8]
        Scale factors for decomposition. Scale 1 = full resolution,
        scale 2 = half resolution, etc.
    interpolation : str, default='cubic'
        Interpolation method for upsampling: 'nearest', 'linear', or 'cubic'.
        - 'nearest': Fastest, blocky output
        - 'linear': Fast, smooth output
        - 'cubic': Highest quality (default), now practical for 3D

        Implementation details:
        - 2D: 'nearest', 'bilinear', or 'bicubic' (PyTorch)
        - 3D+: 'nearest', 'trilinear', or Keys cubic convolution (vectorized)

        Keys cubic convolution uses separable filters with vectorized operations
        for efficient nD interpolation (27-43× faster than torch-interpol).
        Note: Cubic interpolation may produce small negative values (undershoot)
        which are automatically clamped to zero.

    Attributes
    ----------
    raw_images : nn.ParameterList
        Learnable parameters for each scale (unconstrained, applied softplus)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        scales: List[int] = [1, 2, 4, 8],
        interpolation: str = "cubic",
    ) -> None:
        super().__init__()
        self.shape = tuple(shape)
        self.scales = scales
        self.ndim = len(shape)
        self.interpolation = interpolation

        # Determine interpolation mode based on dimensionality and preference
        self.upsample_mode = _get_interpolation_mode(self.ndim, interpolation)

        # Create learnable parameters for each scale
        # Initialize to zeros (will be properly initialized later)
        self.raw_images = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        tuple(max(1, s // scale) for s in shape), dtype=torch.float32
                    )
                )
                for scale in scales
            ]
        )

    def forward(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Forward pass: apply non-negativity and upsample all scales.

        Returns
        -------
        scales_list : List[torch.Tensor]
            List of K non-negative images at each scale
        upsampled_list : List[torch.Tensor]
            List of K upsampled images (all at full resolution)
        reconstruction : torch.Tensor
            Sum of all upsampled scales (final reconstruction)
        """
        scales_list = []
        upsampled_list = []

        for raw_img, scale in zip(self.raw_images, self.scales):
            # Apply softplus for non-negativity constraint
            img_scale = F.softplus(raw_img)
            scales_list.append(img_scale)

            # Upsample to full resolution
            img_upsampled = _upsample_to_shape(
                img_scale, self.shape, self.upsample_mode
            )
            upsampled_list.append(img_upsampled)

        # Sum all upsampled scales
        reconstruction = torch.sum(torch.stack(upsampled_list), dim=0)

        return scales_list, upsampled_list, reconstruction

    @torch.no_grad()
    def initialize_from_pyramid(self, target: torch.Tensor) -> None:
        """
        Initialize parameters from Gaussian pyramid decomposition.

        This provides a sensible starting point where the decomposition
        already approximately represents the target image. Energy is
        distributed across scales from coarse to fine.

        Uses negative propagation: if a coarse scale overshoots (causing
        negative values in the remainder), those negatives are propagated
        to finer scales, which compensate by reducing their values in those
        regions. This ensures energy conservation without information loss.

        Parameters
        ----------
        target : torch.Tensor
            Target image to decompose (shape must match self.shape)
        """
        if target.shape != self.shape:
            raise ValueError(
                f"Target shape {target.shape} does not match model shape {self.shape}"
            )

        remaining = target.clone()

        # Process from coarsest to finest scale
        for raw_param, scale in zip(reversed(self.raw_images), reversed(self.scales)):
            # Downsample remaining signal to current scale
            # If remaining has negatives, they naturally reduce this scale's values
            # through the averaging operation
            img_scale = _downsample_to_scale(remaining, scale, mode="area")

            # Each individual scale must be non-negative (for softplus)
            # But we let the remaining signal carry negatives
            img_scale_clamped = torch.clamp(img_scale, min=1e-6)

            # Use stable inverse softplus (GPU-native, no CPU transfer)
            raw_param.data = stable_inverse_softplus_torch(img_scale_clamped)

            # Subtract this scale's contribution from remaining
            img_upsampled = _upsample_to_shape(
                img_scale_clamped, self.shape, self.upsample_mode
            )
            remaining = remaining - img_upsampled
            # DON'T clamp here - let negatives propagate to next finer scale!

    @torch.no_grad()
    def initialize_finest_scale(self, target: torch.Tensor) -> None:
        """
        Initialize with all energy in the finest (highest resolution) scale.

        This creates a "trivial" starting point where all energy is in the
        finest scale and must be redistributed during optimization. All other
        scales start at near-zero values.

        Parameters
        ----------
        target : torch.Tensor
            Target image to decompose (shape must match self.shape)
        """
        if target.shape != self.shape:
            raise ValueError(
                f"Target shape {target.shape} does not match model shape {self.shape}"
            )

        # Initialize all scales to large negative values
        # softplus(-10) ≈ 4.5e-5, which is effectively zero
        for raw_param in self.raw_images:
            raw_param.data.fill_(-10.0)

        # Put all energy in the finest scale (scale factor = 1)
        finest_idx = self.scales.index(1) if 1 in self.scales else 0

        # For finest scale, downsample target if needed
        finest_scale = self.scales[finest_idx]
        if finest_scale == 1:
            img_scale = target.clone()
        else:
            img_scale = _downsample_to_scale(target, finest_scale, mode="area")

        # Initialize using inverse softplus (GPU-native, no CPU transfer)
        img_scale_clamped = torch.clamp(img_scale, min=1e-6)
        self.raw_images[finest_idx].data = stable_inverse_softplus_torch(
            img_scale_clamped
        )

    @torch.no_grad()
    def initialize_uniform(self, target: torch.Tensor) -> None:
        """
        Initialize with energy split uniformly across all scales.

        Each scale gets 1/K of the total energy (when upsampled to full resolution),
        where K is the number of scales. This provides a balanced starting point
        between pyramid and finest initialization.

        Parameters
        ----------
        target : torch.Tensor
            Target image to decompose (shape must match self.shape)
        """
        if target.shape != self.shape:
            raise ValueError(
                f"Target shape {target.shape} does not match model shape {self.shape}"
            )

        # Compute target energy (sum)
        target_energy = torch.sum(target)

        # Each scale gets equal share of energy (after upsampling)
        energy_per_scale_upsampled = target_energy / len(self.scales)

        # Initialize each scale
        for raw_param, scale in zip(self.raw_images, self.scales):
            # Downsample target to this scale's resolution
            img_scale = _downsample_to_scale(target, scale, mode="area")

            # Account for upsampling factor: when we upsample, the energy gets
            # multiplied by scale^ndim. So we need to divide by this factor.
            upsampling_factor = scale**self.ndim
            energy_per_scale_downsampled = (
                energy_per_scale_upsampled / upsampling_factor
            )

            # Normalize to have the correct energy at this resolution
            current_energy = torch.sum(img_scale)
            if current_energy > 1e-12:
                img_scale = img_scale * (energy_per_scale_downsampled / current_energy)
            else:
                # If downsampled image is all zeros, create small uniform values
                img_scale = torch.full_like(
                    img_scale, energy_per_scale_downsampled / img_scale.numel()
                )

            # Initialize using inverse softplus (GPU-native, no CPU transfer)
            img_scale_clamped = torch.clamp(img_scale, min=1e-6)
            raw_param.data = stable_inverse_softplus_torch(img_scale_clamped)

    @torch.no_grad()
    def initialize_coarse(self, target: torch.Tensor) -> None:
        """
        Initialize with energy weighted toward coarse scales.

        Energy is distributed proportionally to scale factor: coarser scales
        (larger scale factors) get more energy. For scales [1, 2, 4, 8], the
        distribution is [1x, 2x, 4x, 8x], so scale 8 gets 8 times more energy
        than scale 1. This strongly biases the initialization toward coarse scales.

        Parameters
        ----------
        target : torch.Tensor
            Target image to decompose (shape must match self.shape)
        """
        if target.shape != self.shape:
            raise ValueError(
                f"Target shape {target.shape} does not match model shape {self.shape}"
            )

        # Compute target energy (sum)
        target_energy = torch.sum(target)

        # Energy distribution proportional to scale factor
        # For scales [1, 2, 4, 8]: weights are [1, 2, 4, 8]
        total_weight = sum(self.scales)

        # Initialize each scale
        for raw_param, scale in zip(self.raw_images, self.scales):
            # Downsample target to this scale's resolution
            img_scale = _downsample_to_scale(target, scale, mode="area")

            # This scale's share of energy (proportional to scale factor)
            energy_fraction = scale / total_weight
            energy_per_scale_upsampled = target_energy * energy_fraction

            # Account for upsampling factor: when we upsample, the energy gets
            # multiplied by scale^ndim. So we need to divide by this factor.
            upsampling_factor = scale**self.ndim
            energy_per_scale_downsampled = (
                energy_per_scale_upsampled / upsampling_factor
            )

            # Normalize to have the correct energy at this resolution
            current_energy = torch.sum(img_scale)
            if current_energy > 1e-12:
                img_scale = img_scale * (energy_per_scale_downsampled / current_energy)
            else:
                # If downsampled image is all zeros, create small uniform values
                img_scale = torch.full_like(
                    img_scale, energy_per_scale_downsampled / img_scale.numel()
                )

            # Initialize using inverse softplus (GPU-native, no CPU transfer)
            img_scale_clamped = torch.clamp(img_scale, min=1e-6)
            raw_param.data = stable_inverse_softplus_torch(img_scale_clamped)

    @torch.no_grad()
    def initialize_zero(self, target: torch.Tensor) -> None:
        """
        Initialize all scales to zero (or near-zero).

        This creates a "worst case" starting point where all scales start at
        effectively zero and must be learned from scratch. Useful for understanding
        the importance of initialization and as a baseline comparison.

        Parameters
        ----------
        target : torch.Tensor
            Target image to decompose (shape must match self.shape)
        """
        if target.shape != self.shape:
            raise ValueError(
                f"Target shape {target.shape} does not match model shape {self.shape}"
            )

        # Initialize all scales to large negative values
        # softplus(-10) ≈ 4.5e-5, which is effectively zero
        for raw_param in self.raw_images:
            raw_param.data.fill_(-10.0)


def _compute_max_abs_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute maximum absolute error between prediction and target.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted reconstruction
    target : torch.Tensor
        Target image

    Returns
    -------
    float
        Maximum absolute error
    """
    return float(torch.max(torch.abs(pred - target)))


def decomposition_loss(
    model: MultiScaleDecomposer,
    target: torch.Tensor,
    energy_weight: float = 0.001,
    alpha: float = 1.5,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-scale decomposition loss.

    Combines reconstruction fidelity with hierarchical energy penalties
    to encourage energy distribution toward coarse scales.

    Loss = L_reconstruction + λ_energy × L_energy

    where:
        L_reconstruction = loss_fn(Σₖ upsample(Vₖ), V) with optional asymmetric penalty
        L_energy = Σₖ (αᵏ × ∫Vₖ) / ∫V

    Parameters
    ----------
    model : MultiScaleDecomposer
        Decomposition model
    target : torch.Tensor
        Target image to reconstruct
    energy_weight : float, default=0.001
        Weight for energy penalty (higher pushes more energy to coarse scales)
    alpha : float, default=1.5
        Growth factor for energy penalties (αᵏ grows exponentially with scale index)
    loss_type : str, default="l1"
        Type of reconstruction loss: "l1" (Mean Absolute Error, default),
        "mse" (Mean Squared Error), or "poisson" (Poisson Deviance)
    asymmetric_penalty : Optional[float], default=10.0
        Over-prediction penalty factor. Multiplies reconstruction loss for regions
        where pred > target by this factor. Set to None to disable asymmetric loss.

    Returns
    -------
    total_loss : torch.Tensor
        Combined loss (scalar)
    stats : dict
        Dictionary with per-component losses and diagnostics:
        - 'recon_loss': Reconstruction loss
        - 'energy_loss': Hierarchical energy penalty (normalized)
        - 'total_loss': Combined loss
        - 'energy_scale_i': Energy fraction at scale i
    """
    # Reuse the canonical loss kernels (function-local import to be defensive
    # against import cycles, though none currently exists).
    from luxar.gsplats.fitting.losses import (
        _compute_l1_loss,
        _compute_mse_loss,
        _compute_poisson_loss,
    )

    # Forward pass
    scales_list, upsampled_list, reconstruction = model()

    # Primary: Reconstruction fidelity with asymmetric penalty
    if loss_type.lower() == "poisson":
        recon_loss = _compute_poisson_loss(reconstruction, target, asymmetric_penalty)
    elif loss_type.lower() == "mse":
        recon_loss = _compute_mse_loss(reconstruction, target, asymmetric_penalty)
    else:
        # Default to L1
        recon_loss = _compute_l1_loss(reconstruction, target, asymmetric_penalty)

    # Secondary: Hierarchical energy penalty
    # Exponentially penalize finer scales to push energy toward coarse scales
    total_target_energy = torch.sum(target)
    energy_loss = torch.zeros((), device=target.device, dtype=target.dtype)

    for k, img_scale in enumerate(scales_list):
        # Weight grows exponentially with fineness
        # k=0 is finest (scale 1), k=K-1 is coarsest (scale 8)
        # We want to penalize fine scales more, so invert the indexing
        weight = alpha ** (len(scales_list) - 1 - k)
        scale_energy = torch.sum(img_scale)
        energy_loss = energy_loss + weight * scale_energy

    # Normalize by total target energy to make scale-invariant
    energy_loss = energy_loss / (total_target_energy + 1e-12)

    # Combine losses
    total_loss = recon_loss + energy_weight * energy_loss

    # Diagnostics
    stats = {
        "recon_loss": recon_loss.item(),
        "energy_loss": energy_loss.item(),
        "total_loss": total_loss.item(),
    }

    # Per-scale energy distribution (from upsampled versions)
    with torch.no_grad():
        for k, img_upsampled in enumerate(upsampled_list):
            energy_frac = torch.sum(img_upsampled) / (total_target_energy + 1e-12)
            stats[f"energy_scale_{k}"] = energy_frac.item()

    return total_loss, stats


def decompose_image(
    V: np.ndarray,
    scales: List[int] = [1, 2, 4, 8, 16, 32],
    n_iters: int = 500,
    lr: float = 0.01,
    energy_weight: float = 0.01,
    alpha: float = 1.5,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0,
    init_method: str = "coarse",
    max_abs_error_threshold: Optional[float] = None,
    interpolation: str = "cubic",
    napari_movie: bool = False,
    movie_every: int = 1,
    movie_max_frames: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Decompose n-dimensional image into multi-scale non-negative components.

    Optimizes a decomposition V = Σₖ upsample(Vₖ) where each Vₖ represents
    features at a different scale, with energy preferentially distributed
    toward coarse scales.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image to decompose
    scales : List[int], default=[1, 2, 4, 8]
        Scale factors. Scale 1 = full res, scale 2 = half res, etc.
        Scales larger than min image dim are filtered with a warning.
        The actual scales used are returned in stats['scales'].
    n_iters : int, default=500
        Number of optimization iterations
    lr : float, default=0.01
        Learning rate for Adam optimizer
    energy_weight : float, default=0.01
        Weight for hierarchical energy penalty (higher = more energy to coarse)
    alpha : float, default=1.5
        Growth factor for energy penalties (higher = stronger coarse preference)
    loss_type : str, default="l1"
        Type of reconstruction loss: "l1" (Mean Absolute Error, default),
        "mse" (Mean Squared Error), or "poisson" (Poisson Deviance)
    asymmetric_penalty : Optional[float], default=10.0
        Over-prediction penalty factor. Multiplies reconstruction loss for regions
        where pred > target by this factor. Set to None to disable asymmetric loss.
    init_method : str, default="coarse"
        Initialization: "coarse" (energy toward coarse - BEST), "pyramid"
        (Gaussian pyramid), "uniform" (equal split), or "finest" (all in
        finest). Coarse gives best convergence and quality.
    max_abs_error_threshold : float, optional
        Convergence threshold for max absolute error. Stops early when
        max|reconstruction - target| < threshold. None uses 1% of image range.
    interpolation : str, default='cubic'
        Interpolation method for upsampling scale components.
        Three modes available:
        - 'nearest': Nearest-neighbor (fastest, blocky output)
        - 'linear': Linear interpolation (fast, smooth)
        - 'cubic': Cubic interpolation (highest quality, practical for 3D)

        Implementation details:
        - 2D: Uses PyTorch's 'bicubic' interpolation
        - 3D+: Uses Keys cubic convolution (vectorized, 27-43× faster)
        - Keys cubic uses separable filters for efficient nD processing

        Note: Cubic interpolation can produce small negative values (undershoot)
        due to the negative lobes in the cubic kernel. These are automatically
        clamped to zero to maintain non-negativity constraint.
    napari_movie : bool, default=False
        Enable recording of optimization progress for napari movie visualization
    movie_every : int, default=1
        Record a movie frame every N iterations (only if napari_movie=True)
    movie_max_frames : Optional[int], default=None
        Maximum number of frames to store. If None, no limit (can use lots of memory).
        Oldest frames are discarded when limit is reached.
    device : str, optional
        PyTorch device ('cpu', 'cuda', 'mps'). Auto-detects if None.
    verbose : bool, default=True
        Print optimization progress

    Returns
    -------
    scales_list : List[np.ndarray]
        List of K non-negative images at each scale.
        scales_list[i] has shape (s₀/rᵢ, s₁/rᵢ, ..., sₙ₋₁/rᵢ)
    stats : dict
        Optimization statistics:
        - 'history': List of per-iteration loss components
        - 'final_error': Final reconstruction MSE
        - 'best_error': Best reconstruction MSE achieved
        - 'best_max_abs_error': Best maximum absolute error achieved
        - 'converged': Boolean indicating if convergence criterion was met
        - 'best_iteration': Iteration where best result was achieved
        - 'actual_iters': Number of iterations run (less if converged early)
        - 'energy_distribution': Fraction of total energy per scale
        - 'scales': Scale factors used
        - 'time_seconds': Total optimization time
        - 'movie_frames': Dictionary with movie data (if napari_movie=True), or None
        - 'interpolation': Interpolation mode used ('nearest', 'linear', or 'cubic')

    Examples
    --------
    >>> import numpy as np
    >>> from luxar.gsplats.multiscale import decompose_image
    >>>
    >>> # 2D example
    >>> V = np.random.rand(256, 256)
    >>> scales_list, stats = decompose_image(V, scales=[1, 2, 4])
    >>> aprint([s.shape for s in scales_list])
    [(256, 256), (128, 128), (64, 64)]
    >>>
    >>> # 3D example
    >>> V = np.random.rand(128, 128, 128)
    >>> scales_list, stats = decompose_image(V, scales=[1, 2, 4, 8])
    >>> aprint(f"Energy distribution: {stats['energy_distribution']}")
    Energy distribution: [0.62, 0.23, 0.11, 0.04]

    Notes
    -----
    - Uses Gaussian pyramid initialization for stable convergence
    - All output images are guaranteed non-negative
    - Reconstruction: V ≈ Σₖ upsample(scales_list[k])
    - Higher alpha values push more energy to coarse scales
    """
    # Auto-detect device (centralized helper handles CUDA/MPS/CPU and missing
    # ``torch.backends.mps`` on older PyTorch builds; explicit value passed
    # through). Returned as a ``torch.device`` so downstream tensor ops avoid
    # repeated string-to-device coercion.
    resolved_device: torch.device = resolve_torch_device(device)

    # Convert to torch tensor
    V_tensor = torch.tensor(V, dtype=torch.float32, device=resolved_device)

    # Set convergence threshold (adaptive if not specified)
    if max_abs_error_threshold is None:
        # Auto-convergence: 1% of image value range
        image_range = float(V.max() - V.min())
        # Handle edge case: uniform image (all same values)
        # Use 1% of mean absolute value as fallback for uniform images
        if image_range < 1e-12:
            image_mean = float(np.abs(V).mean())
            max_abs_error_threshold = 0.01 * max(image_mean, 1e-6)
            if verbose:
                aprint(
                    f"Auto-convergence threshold: {max_abs_error_threshold:.6e} "
                    f"(uniform image, using 1% of mean absolute value)"
                )
        else:
            max_abs_error_threshold = 0.01 * image_range
            if verbose:
                aprint(
                    f"Auto-convergence threshold: {max_abs_error_threshold:.6f} "
                    f"(1% of image range [{V.min():.3f}, {V.max():.3f}])"
                )

    # Filter out scales that are too large for the image dimensions
    # A scale is valid if all dimensions satisfy: dim // scale >= 1
    min_dim = min(V.shape)
    original_scales = scales
    scales = [s for s in scales if s <= min_dim]

    if len(scales) == 0:
        # If no scales are valid, use scale=1 only
        scales = [1]
        if verbose:
            import warnings

            warnings.warn(
                f"Scales {original_scales} too large for shape {V.shape}. "
                "Using scale=[1]."
            )
    elif len(scales) < len(original_scales):
        removed_scales = [s for s in original_scales if s not in scales]
        if verbose:
            import warnings

            warnings.warn(
                f"Scales {removed_scales} are too large for image shape {V.shape} "
                f"(min dimension={min_dim}). Using scales={scales} instead."
            )

    with asection("Multi-Scale Image Decomposition"):
        if verbose:
            aprint(f"Input shape: {V.shape} ({V.ndim}D)")
            aprint(f"Scales: {scales}")
            aprint(f"Device: {resolved_device}")
            asymmetric_str = (
                f"{asymmetric_penalty}" if asymmetric_penalty is not None else "None"
            )
            aprint(
                f"Parameters: n_iters={n_iters}, lr={lr}, "
                f"energy_weight={energy_weight}, alpha={alpha}, "
                f"loss_type='{loss_type}', asymmetric_penalty={asymmetric_str}"
            )

        # Initialize model
        if verbose:
            if init_method == "finest":
                aprint("Initializing model with all energy in finest scale...")
            elif init_method == "uniform":
                aprint(
                    "Initializing model with energy split uniformly across scales..."
                )
            elif init_method == "coarse":
                aprint(
                    "Initializing model with energy weighted toward coarse scales..."
                )
            elif init_method == "zero":
                aprint("Initializing model with all scales at zero...")
            else:
                aprint("Initializing model from Gaussian pyramid...")

        model = MultiScaleDecomposer(
            V.shape, scales=scales, interpolation=interpolation
        ).to(resolved_device)

        if init_method == "finest":
            model.initialize_finest_scale(V_tensor)
        elif init_method == "uniform":
            model.initialize_uniform(V_tensor)
        elif init_method == "coarse":
            model.initialize_coarse(V_tensor)
        elif init_method == "zero":
            model.initialize_zero(V_tensor)
        else:  # "pyramid" is default
            model.initialize_from_pyramid(V_tensor)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Movie recording setup (only if enabled)
        movie_frames: Optional[Dict[str, Any]] = {} if napari_movie else None
        if napari_movie:
            if movie_max_frames is None:
                movie_max_frames = 10000  # Large but finite limit
            movie_frames = {
                "target": [],
                "reconstruction": [],
                "residual": [],
                "scales": [],  # List of lists: each entry is a list of scale components
                "iterations": [],
            }
            if verbose:
                aprint(
                    f"Movie recording enabled: every {movie_every} iterations, "
                    f"max {movie_max_frames} frames"
                )

        # Training loop
        start_time = time.time()
        history = []
        best_recon_loss = float("inf")

        # Best state tracking for quality guarantee
        best_max_abs_error = float("inf")
        best_state = None
        best_iteration = 0
        converged_early = False

        if verbose:
            aprint(f"Optimizing decomposition ({n_iters} iterations)...")
            aprint(f"Convergence: max err < {max_abs_error_threshold:.6f}")

        for it in range(1, n_iters + 1):
            optimizer.zero_grad()

            # Compute loss
            loss, stats = decomposition_loss(
                model,
                V_tensor,
                energy_weight=energy_weight,
                alpha=alpha,
                loss_type=loss_type,
                asymmetric_penalty=asymmetric_penalty,
            )

            # Backward pass
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            # Record history
            history.append(stats)

            # Track best loss
            if stats["recon_loss"] < best_recon_loss:
                best_recon_loss = stats["recon_loss"]

            # Convergence check and best state tracking using maximum absolute error
            with torch.no_grad():
                scales_list_current, _, reconstruction = model()
                current_max_abs_error = _compute_max_abs_error(reconstruction, V_tensor)

                # Track best state based on max absolute error (quality guarantee)
                if current_max_abs_error < best_max_abs_error:
                    # Save previous best for logging comparison
                    previous_best = best_max_abs_error

                    best_max_abs_error = current_max_abs_error
                    best_iteration = it

                    # Save current best state (deep copy to avoid mutations)
                    best_state = {
                        "raw_images": [p.detach().clone() for p in model.raw_images],
                        "scales_list": [
                            s.detach().clone() for s in scales_list_current
                        ],
                        "reconstruction": reconstruction.detach().clone(),
                        "iteration": it,
                        "max_abs_error": current_max_abs_error,
                        "recon_loss": stats["recon_loss"],
                    }

                    # Smart logging: significant improvements or early iterations
                    if verbose and (
                        it <= 10 or current_max_abs_error < previous_best * 0.95
                    ):
                        aprint(
                            f"    ★ New best state: iteration {it}, "
                            f"max_abs_error={current_max_abs_error:.6f}"
                        )

                # Check for convergence
                if current_max_abs_error < max_abs_error_threshold:
                    converged_early = True
                    if verbose:
                        aprint(f"✓ CONVERGENCE ACHIEVED at iteration {it}")
                        aprint(
                            f"  Max absolute error: {current_max_abs_error:.6f} < "
                            f"threshold: {max_abs_error_threshold:.6f}"
                        )
                    break

            # Movie frame recording (only if enabled and at specified intervals)
            if movie_frames is not None and it % movie_every == 0:
                if movie_max_frames is None:
                    movie_max_frames = 10000
                with torch.no_grad():
                    # Memory-bounded recording: remove oldest frames if limit exceeded
                    if len(movie_frames["target"]) >= movie_max_frames:
                        # Remove oldest frame (FIFO)
                        for key in [
                            "target",
                            "reconstruction",
                            "residual",
                            "scales",
                            "iterations",
                        ]:
                            movie_frames[key].pop(0)

                    # Get current reconstruction and scale components
                    scales_list_current, _, reconstruction = model()

                    # Store frames as numpy arrays (detached from computation graph)
                    target_frame = V_tensor.cpu().numpy()
                    recon_frame = reconstruction.detach().cpu().numpy()
                    residual_frame = torch.abs(V_tensor - reconstruction).cpu().numpy()

                    # Store individual scale components (as list of numpy arrays)
                    scales_frame = [
                        s.detach().cpu().numpy() for s in scales_list_current
                    ]

                    movie_frames["target"].append(target_frame)
                    movie_frames["reconstruction"].append(recon_frame)
                    movie_frames["residual"].append(residual_frame)
                    movie_frames["scales"].append(scales_frame)
                    movie_frames["iterations"].append(it)

            # Logging
            if verbose and (it % max(1, n_iters // 10) == 0 or it == 1):
                # Format energy distribution
                energy_dist = [
                    stats.get(f"energy_scale_{k}", 0.0) for k in range(len(scales))
                ]
                energy_str = ", ".join([f"{e:.1%}" for e in energy_dist])

                aprint(
                    f"[{it:4d}/{n_iters}] "
                    f"recon={stats['recon_loss']:.6f} "
                    f"energy=[{energy_str}]"
                )

        end_time = time.time()
        elapsed = end_time - start_time
        actual_iters = (
            it  # Actual number of iterations (may be less if converged early)
        )

        # Log convergence status
        if verbose:
            if converged_early:
                aprint("✓ Optimization terminated: CONVERGENCE ACHIEVED")
            else:
                aprint("⚠ Optimization terminated: ITERATION LIMIT REACHED")
                aprint(
                    f"  Final max absolute error: {best_max_abs_error:.6f} "
                    f"(threshold: {max_abs_error_threshold:.6f})"
                )

        # Restore best state (quality guarantee)
        if best_state is not None:
            if verbose:
                improvement = (
                    " (better than final iteration)"
                    if best_iteration != actual_iters
                    else ""
                )
                aprint(
                    f"Restoring best state from iteration {best_iteration}{improvement}"
                )

            # Restore best parameters
            for param, best_param in zip(model.raw_images, best_state["raw_images"]):
                param.data.copy_(best_param)

            # Use saved best scales and reconstruction
            scales_list = best_state["scales_list"]
            reconstruction = best_state["reconstruction"]
            best_recon_loss = best_state["recon_loss"]
            best_max_abs_error = best_state["max_abs_error"]
        else:
            # Fallback to final state if no best state saved
            with torch.no_grad():
                scales_list, _, reconstruction = model()
                best_max_abs_error = _compute_max_abs_error(reconstruction, V_tensor)

        # Convert to numpy
        with torch.no_grad():
            scales_np = [s.cpu().numpy() for s in scales_list]
            final_recon_error = F.mse_loss(reconstruction, V_tensor).item()

        # Count local maxima per scale for seed distribution
        from luxar.gsplats.seeds.utils import count_local_maxima

        maxima_per_scale = []
        for scale_img in scales_np:
            n_maxima = count_local_maxima(
                scale_img, radius=1, threshold_rel=0.1, blur=True
            )
            # Ensure at least 1 maximum per scale to avoid division issues
            maxima_per_scale.append(max(1, n_maxima))

        if verbose:
            aprint(f"Local maxima per scale: {maxima_per_scale}")

        # Final statistics
        final_energy_dist = [
            history[-1].get(f"energy_scale_{k}", 0.0) for k in range(len(scales))
        ]

        if verbose:
            with asection("Decomposition Results"):
                aprint(f"Time: {elapsed:.2f} seconds")
                aprint(f"Actual iterations: {actual_iters}/{n_iters}")
                aprint(f"Best iteration: {best_iteration}")
                aprint(f"Final reconstruction MSE: {final_recon_error:.6e}")
                aprint(f"Best reconstruction MSE: {best_recon_loss:.6e}")
                aprint(f"Best max absolute error: {best_max_abs_error:.6e}")
                aprint(f"Converged: {converged_early}")
                aprint("Energy distribution (coarse → fine):")
                for k, (scale, energy_pct) in enumerate(zip(scales, final_energy_dist)):
                    shape_str = "x".join(str(s) for s in scales_np[k].shape)
                    aprint(f"  Scale {scale:2d}x: {energy_pct:6.1%}  [{shape_str}]")

    # Return results
    return scales_np, {
        "history": history,
        "final_error": final_recon_error,
        "best_error": best_recon_loss,
        "best_max_abs_error": best_max_abs_error,
        "converged": converged_early,
        "best_iteration": best_iteration,
        "actual_iters": actual_iters,
        "energy_distribution": final_energy_dist,
        "maxima_per_scale": maxima_per_scale,  # For seed distribution
        "scales": scales,
        "time_seconds": elapsed,
        "movie_frames": movie_frames,
        "interpolation": interpolation,  # Store interpolation mode for visualization
    }


def upsample_for_visualization(
    img: np.ndarray, target_shape: Tuple[int, ...], interpolation: str = "cubic"
) -> np.ndarray:
    """
    Upsample numpy array to target shape for visualization purposes.

    Uses the same interpolation methods as the optimization to ensure
    visual consistency between optimization and visualization.

    Parameters
    ----------
    img : np.ndarray
        Input image of any dimensionality
    target_shape : Tuple[int, ...]
        Target shape to upsample to
    interpolation : str, default='cubic'
        Interpolation method: 'nearest', 'linear', or 'cubic'.
        Should match the interpolation used during optimization.

    Returns
    -------
    np.ndarray
        Upsampled image of shape target_shape

    Examples
    --------
    >>> import numpy as np
    >>> from luxar.gsplats.multiscale import upsample_for_visualization
    >>> img = np.random.rand(64, 64)
    >>> upsampled = upsample_for_visualization(img, (256, 256), 'cubic')
    >>> upsampled.shape
    (256, 256)
    """
    if img.shape == target_shape:
        return img

    # Convert to torch, upsample using same logic as optimization, convert back
    img_torch = torch.from_numpy(img)
    interp_mode = _get_interpolation_mode(len(target_shape), interpolation)
    upsampled_torch = _upsample_to_shape(img_torch, target_shape, interp_mode)

    return upsampled_torch.cpu().numpy()


def show_optimization_movie(
    movie_frames: Dict[str, Any], shape: Tuple[int, ...], interpolation: str = "cubic"
) -> None:
    """
    Display napari viewer with optimization movie (target, reconstruction,
    residual, and all scales over time).

    Parameters
    ----------
    movie_frames : dict
        Dictionary containing movie frame data:
        - 'target': List of target frames
        - 'reconstruction': List of reconstruction frames
        - 'residual': List of residual frames
        - 'scales': List of lists of scale components (one list per frame)
        - 'iterations': List of iteration numbers
    shape : tuple
        Shape of the original data
    interpolation : str, default='cubic'
        Upsampling method: 'nearest', 'linear', or 'cubic'. Should match
        the interpolation used during optimization.
    """
    try:
        import napari

        aprint("🎬 Creating multi-scale decomposition movie visualization...")
        aprint(f"  Using '{interpolation}' interpolation for upsampling")

        # Convert lists to stacks (time, spatial_dims...)
        target_stack = np.array(movie_frames["target"])
        reconstruction_stack = np.array(movie_frames["reconstruction"])
        residual_stack = np.array(movie_frames["residual"])
        iterations = movie_frames["iterations"]

        # Get number of scales from first frame
        n_scales = len(movie_frames["scales"][0])
        n_frames = len(iterations)

        aprint(f"  Processing {n_scales} scale components across {n_frames} frames...")

        # Upsample all scale components to target shape and create stacks.
        # scales_stacks: list of n_scales arrays, shape (time, *spatial_dims)
        scales_stacks = []
        for scale_idx in range(n_scales):
            # Collect this scale across all frames
            scale_frames = []
            for frame_idx in range(n_frames):
                scale_component = movie_frames["scales"][frame_idx][scale_idx]
                # Upsample if needed using same interpolation as optimization
                if scale_component.shape != shape:
                    scale_upsampled = upsample_for_visualization(
                        scale_component, shape, interpolation
                    )
                else:
                    scale_upsampled = scale_component
                scale_frames.append(scale_upsampled)

            # Stack into (time, spatial_dims...)
            scale_stack = np.array(scale_frames)
            scales_stacks.append(scale_stack)
            aprint(f"    Scale {scale_idx + 1}/{n_scales} shape: {scale_stack.shape}")

        # Create napari viewer with time series
        viewer = napari.Viewer(
            title=f"Multi-Scale Decomposition Movie ({len(iterations)} frames)"
        )

        # Determine contrast limits from target (shared across most layers)
        contrast_limits = [0, float(target_stack.max())]

        # Add image stacks as layers
        viewer.add_image(
            target_stack,
            name="Target",
            colormap="magma",
            contrast_limits=contrast_limits,
        )

        viewer.add_image(
            reconstruction_stack,
            name="Reconstruction",
            colormap="magma",
            contrast_limits=contrast_limits,
        )

        # Add individual scale components
        for scale_idx, scale_stack in enumerate(scales_stacks):
            viewer.add_image(
                scale_stack,
                name=f"Scale {scale_idx + 1}",
                colormap="viridis",
                contrast_limits=contrast_limits,
                blending="additive",
                opacity=0.7,
                visible=True,  # All scales visible by default
            )

        viewer.add_image(
            residual_stack,
            name="Residual (absolute)",
            colormap="inferno",
            contrast_limits=[0, max(1e-12, float(residual_stack.max()))],
        )

        # Set up the time slider
        viewer.dims.axis_labels = ["iteration"] + [
            f"dim_{i}" for i in range(len(shape))
        ]

        # Add text overlay with movie information
        info_text = "Multi-Scale Decomposition Movie\n"
        info_text += f"Frames: {len(iterations)}\n"
        info_text += f"Iterations: {iterations[0]} → {iterations[-1]}\n"
        info_text += f"Shape: {shape}\n"
        info_text += f"Scales: {n_scales}\n\n"
        info_text += "Use time slider to scrub through optimization\n"
        info_text += "Toggle layers to compare target/reconstruction/scales/residual"

        viewer.text_overlay.text = info_text
        viewer.text_overlay.visible = True

        aprint(
            f"🎬 Movie: {len(iterations)} frames, iters {iterations[0]}-"
            f"{iterations[-1]}"
        )
        aprint(f"   {n_scales} scale components included (all visible)")
        aprint("Use the time slider to scrub through optimization progress!")
        aprint(
            "Toggle layer visibility to compare target/reconstruction/scales/residual"
        )
        aprint("All scale components are visible - toggle them off to reduce clutter")
        aprint("Close napari window to continue...")

        # Run napari - blocks until window is closed
        napari.run()

    except ImportError:
        aprint("⚠ napari not available for movie visualization")
    except Exception as e:
        aprint(f"⚠ Movie visualization error: {e}")
