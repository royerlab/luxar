"""Volume-rendering adapter mixin for ``GSplatData``."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from .base import _GSplatDataOps

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


class RenderMixin(_GSplatDataOps):
    """``GSplatData.render_to_volume`` — GPU-accelerated volume rendering."""

    def render_to_volume(
        self,
        shape: tuple[int, ...],
        device: str | None = None,
        truncate: float | None = None,
        intensity_floor: float = 1e-5,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """Render Gaussian splats to a volume using GPU-accelerated rendering.

        This is a convenience method that automatically selects the fastest available
        backend (CUDA, MPS, or CPU) and uses the optimized PyTorch renderer.

        Parameters
        ----------
        shape : tuple[int, ...]
            Output volume shape (e.g., (128, 128, 128) for 3D).
        device : str, optional
            Device to use for rendering. If None, auto-detects the best device.
            Options: "cuda", "mps", "cpu".
        truncate : float, optional
            Truncation radius in standard deviations. Gaussians are evaluated within
            this radius from their centers. Defaults to ``self.truncation_radius``.
        intensity_floor : float, default=1e-5
            Minimum intensity threshold for amplitude-aware culling. Splats with
            contributions below this threshold are culled early for performance.
        chunk_size : int, optional
            Chunk size for memory management when processing large volumes. If None,
            automatically calculated based on available memory.

        Returns
        -------
        np.ndarray
            Rendered volume with the specified shape.

        Examples
        --------
        >>> # Render to 128³ volume
        >>> volume = gsplat_data.render_to_volume(shape=(128, 128, 128))
        >>>
        >>> # Force CPU rendering
        >>> volume = gsplat_data.render_to_volume(shape=(128, 128, 128), device="cpu")
        >>>
        >>> # Use larger truncation radius
        >>> volume = gsplat_data.render_to_volume(shape=(128, 128, 128), truncate=4.0)

        Notes
        -----
        - For 8K splats on 128³ volume: substantially faster than NumPy
          implementation (often orders of magnitude on GPU; varies by hardware)
        - Automatically chunks large volumes to prevent out-of-memory errors
        - Uses specialized fast paths for 2D/3D rendering
        """
        if truncate is None:
            truncate = self.truncation_radius

        from luxar.gsplats.rendering.volume_rendering import render_to_volume

        return render_to_volume(
            cast("GSplatData", self),
            shape=tuple(shape),
            device=device,
            truncate=truncate,
            intensity_floor=intensity_floor,
            chunk_size=chunk_size,
        )
