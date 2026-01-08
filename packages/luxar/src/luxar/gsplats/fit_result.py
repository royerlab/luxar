"""Gaussian Splat data container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from luxar.encoding import EncodingMode


@dataclass
class GSplatData:
    """Container for Gaussian splat data.

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
    colors : Optional[np.ndarray], shape (N, 3)
        Optional RGB colors per splat. Can be uint8 [0, 255] for SDR or
        float32 for HDR. None if colors are not present.
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
    colors: Optional[np.ndarray] = None
    stats: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize stats to empty dict if None."""
        if self.stats is None:
            self.stats = {}

    def save(
        self,
        path: str | Path,
        ordering: Literal["morton", "hilbert", "none"] = "hilbert",
        encoding_mode: Optional["EncodingMode"] = None,
        positive_scalar_encoding: Literal["linear", "log"] = "linear",
        color_mode: Optional[Literal["sdr", "hdr"]] = None,
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
            color_mode: Required if colors are float32 ("sdr" or "hdr")
            include_fitting_info: Whether to include fitting statistics
            include_provenance: Whether to include provenance info from stats
            description: Optional user description

        Example:
            >>> result = fit_gaussian_splats(image, n_iters=1000)
            >>> result.save("fitted.gsplats.zarr", encoding_mode=EncodingMode.MEMORY)
            >>> # With colors
            >>> result_with_colors.save("colored.gsplats.zarr", color_mode="sdr")
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
            colors=self.colors,
            sharpnesses=self.sharpnesses,
            ordering=ordering,
            encoding_mode=encoding_mode,
            color_mode=color_mode,
            positive_scalar_encoding=positive_scalar_encoding,
            fitting_info=fitting_info,
            fitting_config=fitting_config,
            provenance_info=provenance_info,
            description=description,
        )

    def translate(self, offset: np.ndarray) -> "GSplatData":
        """Translate all splat centers by an offset vector.

        Args:
            offset: Translation vector (shape: (d,) where d is spatial dimensions)

        Returns:
            New GSplatData with translated centers (all other data unchanged)

        Example:
            >>> # Shift all splats by [10, 20, 30]
            >>> translated = data.translate(np.array([10, 20, 30]))
        """
        return GSplatData(
            centers=self.centers + offset,
            amplitudes=self.amplitudes.copy(),
            cholesky_factors=self.cholesky_factors.copy(),
            sharpnesses=self.sharpnesses.copy(),
            colors=self.colors.copy() if self.colors is not None else None,
            stats=self.stats.copy() if self.stats else {},
        )

    def center_at_centroid(self) -> "GSplatData":
        """Center the splats at their center of mass (amplitude-weighted centroid).

        The centroid is computed as the amplitude-weighted average of splat centers,
        which corresponds to the center of mass of the represented density.

        Returns:
            New GSplatData centered at origin (centroid at [0, 0, ...])

        Example:
            >>> # Center splats at origin for easier viewing
            >>> centered = data.center_at_centroid()
            >>> aprint(centered.centers.mean(axis=0))  # Should be close to [0, 0, 0]
        """
        # Compute amplitude-weighted centroid
        total_amplitude = self.amplitudes.sum()
        if total_amplitude > 0:
            centroid = (self.centers.T @ self.amplitudes) / total_amplitude
        else:
            centroid = self.centers.mean(axis=0)

        # Translate to center at origin
        return self.translate(-centroid)

    def scale_intensity(self, factor: float) -> "GSplatData":
        """Scale all splat amplitudes by a multiplicative factor.

        This effectively brightens (factor > 1) or dims (factor < 1) the
        entire representation.

        Args:
            factor: Multiplicative scaling factor for amplitudes

        Returns:
            New GSplatData with scaled amplitudes

        Example:
            >>> # Reduce brightness by 10x
            >>> dimmed = data.scale_intensity(0.1)
            >>> # Brighten by 2x
            >>> brightened = data.scale_intensity(2.0)
        """
        return GSplatData(
            centers=self.centers.copy(),
            amplitudes=self.amplitudes * factor,
            cholesky_factors=self.cholesky_factors.copy(),
            sharpnesses=self.sharpnesses.copy(),
            colors=self.colors.copy() if self.colors is not None else None,
            stats=self.stats.copy() if self.stats else {},
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        include_stats: bool = False,
    ) -> "GSplatData":
        """Load splats from .gsplats.zarr format.

        Args:
            path: Path to .gsplats.zarr directory
            include_stats: Whether to include fitting/provenance metadata

        Returns:
            GSplatData with decoded arrays

        Example:
            >>> data = GSplatData.load("fitted.gsplats.zarr")
            >>> aprint(data.centers.shape)
        """
        from luxar.gsplats.io.load_gsplats import load_gsplats

        return load_gsplats(path, include_stats=include_stats)
