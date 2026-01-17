"""GSplats rendering module.

This module provides high-performance rendering functions for Gaussian splats,
with automatic backend selection (CUDA, MPS, CPU).
"""

from luxar.gsplats.rendering.volume_rendering import render_to_volume

__all__ = ["render_to_volume"]
