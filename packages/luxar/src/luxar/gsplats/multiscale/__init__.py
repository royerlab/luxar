"""
Multi-scale image decomposition for efficient Gaussian splatting.

This package provides tools for decomposing n-dimensional images into
non-negative multi-scale components, enabling efficient hierarchical
Gaussian splat fitting.

Main Functions
--------------
decompose_image : Decompose image into multi-scale components
MultiScaleDecomposer : PyTorch model for decomposition
decomposition_loss : Loss function for optimization

Examples
--------
>>> from luxar.gsplats.multiscale import decompose_image
>>> import numpy as np
>>>
>>> # Decompose a 2D image
>>> V = np.random.rand(256, 256)
>>> scales_list, stats = decompose_image(V, scales=[1, 2, 4])
>>>
>>> # Use scale components
>>> V_full = scales_list[0]      # Full resolution
>>> V_half = scales_list[1]      # Half resolution
>>> V_quarter = scales_list[2]   # Quarter resolution
"""

from luxar.gsplats.multiscale.decompose import (
    MultiScaleDecomposer,
    decompose_image,
    decomposition_loss,
    show_optimization_movie,
)

__all__ = [
    "decompose_image",
    "MultiScaleDecomposer",
    "decomposition_loss",
    "show_optimization_movie",
]
