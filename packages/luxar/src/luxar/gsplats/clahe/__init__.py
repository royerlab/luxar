"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) for nD volumes.

This subpackage provides a PyTorch-based implementation of CLAHE that works
on arbitrary-dimensional tensors. CLAHE is particularly useful for:
- Enhancing local contrast in images/volumes with varying background
- Preprocessing for feature detection in heterogeneous data
- Creating perceptually-balanced sampling distributions

Key Features:
- nD support: works on 1D, 2D, 3D, and higher-dimensional tensors
- Contrast limiting: prevents noise amplification in uniform regions
- Tile-based processing: adapts to local intensity distributions
- PyTorch native: GPU-accelerated, differentiable operations

Example:
    >>> import torch
    >>> from luxar.gsplats.clahe import apply_clahe
    >>>
    >>> # 2D image with varying background
    >>> image = torch.randn(256, 256)
    >>>
    >>> # Apply CLAHE with default parameters
    >>> enhanced = apply_clahe(image, tile_size=16, clip_limit=2.0)
    >>>
    >>> # Result has locally-equalized contrast
    >>> assert enhanced.shape == image.shape
"""

from .clahe_core import apply_clahe, compute_clahe_sampling_probabilities

__all__ = ["apply_clahe", "compute_clahe_sampling_probabilities"]
