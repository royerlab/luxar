"""
Preprocessing for Gaussian splat fitting.

This subpackage provides GPU-accelerated preprocessing operations for
scientific volumes, with a focus on denoising before Gaussian splat fitting.
Noisy data wastes splats on background artifacts; denoising first leads to
more efficient, higher-quality fits.

Key Features:
- Non-Local Means (NLM) denoising for 2D images and 3D volumes
- Three-tier backend: skimage (CPU reference), PyTorch (GPU), CUDA (maximum perf)
- Automatic backend selection based on device and availability
- Noise2Self (J-invariant) calibration for automatic h parameter selection

Example:
    >>> import torch
    >>> from luxar.gsplats.preprocessing import denoise_nlm, calibrate_nlm_h
    >>>
    >>> volume = torch.randn(64, 128, 128)  # noisy 3D volume
    >>>
    >>> # Auto-calibrate denoising strength
    >>> h = calibrate_nlm_h(volume, device='cuda')
    >>>
    >>> # Denoise with best available backend
    >>> denoised = denoise_nlm(volume, h=h, device='cuda')
"""

from __future__ import annotations

from .calibration import calibrate_nlm_h
from .nlm_core import denoise_nlm

__all__ = ["denoise_nlm", "calibrate_nlm_h"]
