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
        - final_rel_l2: Relative L2 error in best state
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

    def __repr__(self) -> str:
        """Summary representation (avoids dumping full arrays)."""
        n = len(self.amplitudes)
        ndim = self.centers.shape[1] if n > 0 else 0
        if n > 0:
            amp_range = f"[{float(self.amplitudes.min()):.4g}, {float(self.amplitudes.max()):.4g}]"
            sharp_range = f"[{float(self.sharpnesses.min()):.4g}, {float(self.sharpnesses.max()):.4g}]"
        else:
            amp_range = "[]"
            sharp_range = "[]"
        colors = "yes" if self.colors is not None else "no"
        return (
            f"GSplatData({n:,} splats, {ndim}D, "
            f"amplitudes={amp_range}, sharpness={sharp_range}, colors={colors})"
        )

    @property
    def sharpness(self) -> np.ndarray:
        """Alias for sharpnesses (matches Scene.add_gsplats parameter name)."""
        return self.sharpnesses

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
        compress: Optional[Literal["zip", "tar.gz"]] = None,
    ) -> None:
        """Save splats to .gsplats.zarr format.

        Args:
            path: Output path (should end with .gsplats.zarr or .gsplats.zarr.zip/.tar.gz if compress is used)
            ordering: Spatial ordering method ("morton", "hilbert", or "none")
            encoding_mode: Encoding mode (AUTO, PRECISION, or MEMORY), defaults to AUTO
            positive_scalar_encoding: Encoding for amplitudes ("linear" or "log")
            color_mode: Required if colors are float32 ("sdr" or "hdr")
            include_fitting_info: Whether to include fitting statistics
            include_provenance: Whether to include provenance info from stats
            description: Optional user description
            compress: Optional compression format ("zip" or "tar.gz"). Creates compressed archive.

        Example:
            >>> result = fit_gaussian_splats(image, n_iters=1000)
            >>> result.save("fitted.gsplats.zarr", encoding_mode=EncodingMode.MEMORY)
            >>> # With colors
            >>> result_with_colors.save("colored.gsplats.zarr", color_mode="sdr")
            >>> # With compression for storage/git-lfs
            >>> result.save("fitted.gsplats.zarr.zip", compress="zip")
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
            # Extract common fitting fields (including pruning stats)
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
                    "pruned",
                    "pruning_method",
                    "n_original",
                    "n_removed",
                    "amplitude_retention",
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
            compress=compress,
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
            centers=self.centers + offset,  # NEW array
            amplitudes=self.amplitudes,  # REFERENCE (no copy needed)
            cholesky_factors=self.cholesky_factors,  # REFERENCE
            sharpnesses=self.sharpnesses,  # REFERENCE
            colors=self.colors,  # REFERENCE (None-safe)
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
            centers=self.centers,  # REFERENCE (no copy needed)
            amplitudes=self.amplitudes * factor,  # NEW array
            cholesky_factors=self.cholesky_factors,  # REFERENCE
            sharpnesses=self.sharpnesses,  # REFERENCE
            colors=self.colors,  # REFERENCE (None-safe)
            stats=self.stats.copy() if self.stats else {},
        )

    def prune(
        self,
        method: Literal[
            "cumulative", "amplitude_percentile", "combined"
        ] = "cumulative",
        target_retention: float = 0.95,
        amplitude_percentile: float = 5.0,
        volume_percentile: float = 95.0,
    ) -> "GSplatData":
        """Prune low-impact splats to reduce file size while preserving quality.

        Removes splats that contribute minimally to the reconstruction. This is
        useful for reducing file size, memory usage, and rendering cost.

        Args:
            method: Pruning strategy:
                - "cumulative": Keep top splats that contribute target_retention of total amplitude
                - "amplitude_percentile": Remove bottom amplitude_percentile by amplitude
                - "combined": Remove splats with (low amplitude OR large volume outliers)
            target_retention: For "cumulative": fraction of amplitude to retain (0.0-1.0)
            amplitude_percentile: For "amplitude_percentile"/"combined": bottom percentile to remove (0-100)
            volume_percentile: For "combined": remove splats above this volume percentile (0-100)

        Returns:
            New GSplatData with pruned splats

        Examples:
            >>> # Recommended: Keep 95% of amplitude (removes ~80-90% of splats)
            >>> pruned = data.prune(method="cumulative", target_retention=0.95)
            >>>
            >>> # More aggressive: Keep 90% of amplitude
            >>> pruned = data.prune(method="cumulative", target_retention=0.90)
            >>>
            >>> # Remove bottom 10% by amplitude
            >>> pruned = data.prune(method="amplitude_percentile", amplitude_percentile=10)
            >>>
            >>> # Remove artifacts (low amp OR large volume)
            >>> pruned = data.prune(method="combined", amplitude_percentile=5, volume_percentile=95)

        Notes:
            The "cumulative" method is recommended as it provides a quality guarantee
            (e.g., "retain 95% of signal") and automatically determines the optimal threshold.
        """
        N_original = len(self.amplitudes)

        # Validate method
        valid_methods = ("cumulative", "amplitude_percentile", "combined")
        if method not in valid_methods:
            raise ValueError(f"Unknown pruning method: {method}")

        # Short-circuit for empty data
        if N_original == 0:
            pruned_stats = self.stats.copy() if self.stats else {}
            pruned_stats.update(
                {
                    "pruned": True,
                    "pruning_method": method,
                    "n_original": 0,
                    "n_removed": 0,
                }
            )
            return GSplatData(
                centers=self.centers.copy(),
                amplitudes=self.amplitudes.copy(),
                cholesky_factors=self.cholesky_factors.copy(),
                sharpnesses=self.sharpnesses.copy(),
                colors=self.colors.copy() if self.colors is not None else None,
                stats=pruned_stats,
            )

        # Compute volumes for combined method
        if method == "combined":
            # Unpack diagonal elements from Cholesky factors
            # For 3D: cholesky is (N, 6) packed as [L00, L10, L11, L20, L21, L22]
            ndim = self.centers.shape[1]
            chol_size = self.cholesky_factors.shape[1]
            expected_size = ndim * (ndim + 1) // 2

            if chol_size != expected_size:
                raise ValueError(
                    f"Cholesky factors have unexpected shape: {self.cholesky_factors.shape}"
                )

            # Extract diagonal elements for volume computation
            # For nD: positions are at [0, 2, 5, 9, 14, ...] = cumsum([1,2,3,4,5,...])
            diag_indices = np.cumsum(np.arange(1, ndim + 1)) - 1
            diag_elements = self.cholesky_factors[:, diag_indices]

            # Volume ∝ det(Σ)^(1/2) = |det(L)| = |product of diagonal elements|
            det_L = np.prod(diag_elements, axis=1)
            det_Sigma = det_L**2
            volumes = np.abs(det_Sigma) ** (1 / ndim)  # Take nth root for nD

        # Select pruning strategy
        if method == "cumulative":
            # Sort by amplitude (descending) and find cutoff
            sorted_indices = np.argsort(self.amplitudes)[::-1]
            sorted_amps = self.amplitudes[sorted_indices]
            cumsum_amps = np.cumsum(sorted_amps)
            total_amp = cumsum_amps[-1]
            if total_amp == 0:
                # All amplitudes are zero - keep all if any retention requested
                mask = (
                    np.ones(N_original, dtype=bool)
                    if target_retention > 0
                    else np.zeros(N_original, dtype=bool)
                )
            else:
                cumsum_norm = cumsum_amps / total_amp

                # Find where we reach target retention
                n_keep = np.searchsorted(cumsum_norm, target_retention) + 1
                n_keep = min(n_keep, N_original)  # Safety check

                # Create mask for splats to keep
                keep_indices = sorted_indices[:n_keep]
                mask = np.zeros(N_original, dtype=bool)
                mask[keep_indices] = True

        elif method == "amplitude_percentile":
            # Remove bottom percentile
            threshold = np.percentile(self.amplitudes, amplitude_percentile)
            mask = self.amplitudes >= threshold

        elif method == "combined":
            # Remove if (low amplitude OR large volume)
            amp_threshold = np.percentile(self.amplitudes, amplitude_percentile)
            vol_threshold = np.percentile(volumes, volume_percentile)
            mask = (self.amplitudes >= amp_threshold) & (volumes <= vol_threshold)

        else:
            raise ValueError(f"Unknown pruning method: {method}")

        # Apply mask
        pruned_centers = self.centers[mask]
        pruned_cholesky = self.cholesky_factors[mask]
        pruned_amplitudes = self.amplitudes[mask]
        pruned_sharpnesses = self.sharpnesses[mask]
        pruned_colors = self.colors[mask] if self.colors is not None else None

        # Update stats
        pruned_stats = self.stats.copy() if self.stats else {}
        pruned_stats["pruned"] = True
        pruned_stats["pruning_method"] = method
        pruned_stats["n_original"] = N_original
        pruned_stats["n_removed"] = N_original - len(pruned_amplitudes)
        total_amp = np.sum(self.amplitudes)
        pruned_stats["amplitude_retention"] = (
            float(np.sum(pruned_amplitudes) / total_amp) if total_amp > 0 else 1.0
        )

        return GSplatData(
            centers=pruned_centers,
            amplitudes=pruned_amplitudes,
            cholesky_factors=pruned_cholesky,
            sharpnesses=pruned_sharpnesses,
            colors=pruned_colors,
            stats=pruned_stats,
        )

    def render_to_volume(
        self,
        shape: tuple[int, ...],
        device: str | None = None,
        truncate: float = 3.0,
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
        truncate : float, default=3.0
            Truncation radius in standard deviations. Gaussians are evaluated within
            this radius from their centers.
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
        - For 8K splats on 128³ volume: ~100-1000x faster than NumPy implementation
        - Automatically chunks large volumes to prevent out-of-memory errors
        - Uses specialized fast paths for 2D/3D rendering
        """
        from luxar.gsplats.rendering.volume_rendering import render_to_volume

        return render_to_volume(
            self,
            shape=tuple(shape),
            device=device,
            truncate=truncate,
            intensity_floor=intensity_floor,
            chunk_size=chunk_size,
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

    @classmethod
    def merge_with_channel_colors(
        cls,
        gsplats_per_channel: list["GSplatData"],
        channel_colors: list[tuple[float, float, float]],
    ) -> "GSplatData":
        """Merge multiple GSplatData objects, assigning a fixed color per channel.

        This is useful for multi-channel visualization where each channel was
        fitted separately and should be displayed with a distinct color.

        Args:
            gsplats_per_channel: List of GSplatData objects, one per channel.
                All must have the same dimensionality.
            channel_colors: List of RGB color tuples (one per channel).
                Each tuple should have values in [0, 1] range, e.g., (1.0, 0.0, 0.5).

        Returns:
            New GSplatData with all splats merged and colors assigned.

        Raises:
            ValueError: If lists have different lengths or dimensionalities don't match.

        Example:
            >>> # Fit each channel separately
            >>> gsplats_ch0 = fit_gaussian_splats(volume_ch0, ...)
            >>> gsplats_ch1 = fit_gaussian_splats(volume_ch1, ...)
            >>>
            >>> # Merge with magenta for ch0, cyan for ch1
            >>> merged = GSplatData.merge_with_channel_colors(
            ...     [gsplats_ch0, gsplats_ch1],
            ...     channel_colors=[(1.0, 0.0, 0.5), (0.0, 1.0, 0.5)],
            ... )
            >>>
            >>> # Add to scene
            >>> scene.add_gsplats_from_data("multichannel", merged)
        """
        if len(gsplats_per_channel) != len(channel_colors):
            raise ValueError(
                f"Number of GSplatData objects ({len(gsplats_per_channel)}) must match "
                f"number of colors ({len(channel_colors)})"
            )

        if len(gsplats_per_channel) == 0:
            raise ValueError("At least one GSplatData object is required")

        # Validate all have same dimensionality
        ndim = gsplats_per_channel[0].centers.shape[1]
        for i, gsplat in enumerate(gsplats_per_channel[1:], start=1):
            if gsplat.centers.shape[1] != ndim:
                raise ValueError(
                    f"Dimensionality mismatch: channel 0 has {ndim}D, "
                    f"channel {i} has {gsplat.centers.shape[1]}D"
                )

        # Concatenate all arrays
        all_centers = np.concatenate([g.centers for g in gsplats_per_channel], axis=0)
        all_amplitudes = np.concatenate(
            [g.amplitudes for g in gsplats_per_channel], axis=0
        )
        all_cholesky = np.concatenate(
            [g.cholesky_factors for g in gsplats_per_channel], axis=0
        )
        all_sharpnesses = np.concatenate(
            [g.sharpnesses for g in gsplats_per_channel], axis=0
        )

        # Build colors array: each splat gets the color of its source channel
        color_arrays = []
        for gsplat, color in zip(gsplats_per_channel, channel_colors):
            n_splats = len(gsplat.amplitudes)
            # Create (N, 3) array filled with channel color
            channel_color_array = np.tile(
                np.array(color, dtype=np.float32), (n_splats, 1)
            )
            color_arrays.append(channel_color_array)

        all_colors = np.concatenate(color_arrays, axis=0)

        # Merge stats (basic aggregation)
        merged_stats: Dict[str, Any] = {
            "merged_from_channels": len(gsplats_per_channel),
            "splats_per_channel": [len(g.amplitudes) for g in gsplats_per_channel],
        }

        # Sum time if available
        total_time = sum(g.stats.get("time_seconds", 0) for g in gsplats_per_channel)
        if total_time > 0:
            merged_stats["time_seconds"] = total_time

        return cls(
            centers=all_centers,
            amplitudes=all_amplitudes,
            cholesky_factors=all_cholesky,
            sharpnesses=all_sharpnesses,
            colors=all_colors,
            stats=merged_stats,
        )
