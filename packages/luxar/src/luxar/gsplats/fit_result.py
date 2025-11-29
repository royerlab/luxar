"""Gaussian Splat fitting result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from luxar.encoding import EncodingMode


@dataclass
class GaussianSplatResult:
    """Results from Gaussian splat fitting.

    Attributes
    ----------
    centers : np.ndarray, shape (N, d)
        Splat center positions in voxel coordinates.
    amplitudes : np.ndarray, shape (N,)
        Non-negative splat amplitudes, rescaled to original image intensity range.
    cholesky_factors : np.ndarray, shape (N, d*(d+1)//2)
        Packed lower-triangular Cholesky factors (L) where Σ = L @ L.T.
        The packing order follows: [L00, L10, L11, L20, L21, L22, ...]
    sharpnesses : np.ndarray, shape (N,)
        Per-splat sharpness values (s=2.0 is standard Gaussian).
        Lower values (s<2) create heavy-tailed Gaussians.
        Higher values (s>2) create sharper, more compact splats.
    stats : Dict[str, Any]
        Optimization statistics including:
        - time_seconds: Total optimization time
        - iterations: Number of iterations completed
        - converged: Whether convergence criteria were met
        - sharpness_min/max/mean/std/median: Sharpness statistics
        - best_iteration: Iteration where best state was found
        - final_max_abs_error: Maximum absolute error in best state
        - movie_frames: Optional optimization movie frames (if napari_movie=True)
    """

    centers: np.ndarray
    amplitudes: np.ndarray
    cholesky_factors: np.ndarray
    sharpnesses: np.ndarray
    stats: Dict[str, Any]

    def save(
        self,
        path: str | Path,
        ordering: Literal["morton", "hilbert", "none"] = "hilbert",
        encoding_mode: Optional["EncodingMode"] = None,
        positive_scalar_encoding: Literal["linear", "log"] = "linear",
        include_fitting_info: bool = True,
        include_provenance: bool = False,
        description: Optional[str] = None,
    ) -> None:
        """Save splats to .gsplats.zarr format.

        Args:
            path: Output path (should end with .gsplats.zarr)
            ordering: Spatial ordering method ("morton", "hilbert", or "none")
            encoding_mode: Encoding mode (AUTO, PRECISION, or MEMORY), defaults to AUTO
            positive_scalar_encoding: Encoding for amplitudes ("linear" or "log")
            include_fitting_info: Whether to include fitting statistics
            include_provenance: Whether to include provenance info from stats
            description: Optional user description

        Example:
            >>> result = fit_gaussian_splats(image, n_iters=1000)
            >>> result.save("fitted.gsplats.zarr", encoding_mode=EncodingMode.MEMORY)
        """
        from luxar.encoding import EncodingMode
        from luxar.gsplats.io.save_gsplats import save_gsplats

        # Use AUTO as default
        if encoding_mode is None:
            encoding_mode = EncodingMode.AUTO

        # Extract fitting info from stats
        fitting_info = None
        fitting_config = None
        provenance_info = None

        if include_fitting_info and self.stats:
            # Extract common fitting fields
            fitting_info = {
                k: v
                for k, v in self.stats.items()
                if k
                in [
                    "time_seconds",
                    "iterations",
                    "converged",
                    "fitter_name",
                    "fitter_version",
                    "timestamp",
                ]
            }

            # Extract fitting config if present
            if "config" in self.stats:
                fitting_config = self.stats["config"]

        if include_provenance and self.stats:
            if "provenance" in self.stats:
                provenance_info = self.stats["provenance"]

        # Call save function
        save_gsplats(
            path=path,
            centers=self.centers,
            amplitudes=self.amplitudes,
            cholesky_factors=self.cholesky_factors,
            sharpnesses=self.sharpnesses,
            ordering=ordering,
            encoding_mode=encoding_mode,
            positive_scalar_encoding=positive_scalar_encoding,
            fitting_info=fitting_info,
            fitting_config=fitting_config,
            provenance_info=provenance_info,
            description=description,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        include_stats: bool = False,
    ) -> "GaussianSplatResult":
        """Load splats from .gsplats.zarr format.

        Args:
            path: Path to .gsplats.zarr directory
            include_stats: Whether to include fitting/provenance metadata

        Returns:
            GaussianSplatResult with decoded arrays

        Example:
            >>> result = GaussianSplatResult.load("fitted.gsplats.zarr")
            >>> print(result.centers.shape)
        """
        from luxar.gsplats.io.load_gsplats import load_gsplats

        return load_gsplats(path, include_stats=include_stats)
