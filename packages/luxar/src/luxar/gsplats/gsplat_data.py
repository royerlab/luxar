"""Gaussian Splat data container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from luxar.encoding import EncodingMode


@dataclass(eq=False)
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
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shape consistency."""
        n = self.centers.shape[0]
        if self.amplitudes.shape != (n,):
            raise ValueError(
                f"Amplitudes shape {self.amplitudes.shape} doesn't match "
                f"centers count ({n},)"
            )
        if self.sharpnesses.shape != (n,):
            raise ValueError(
                f"Sharpnesses shape {self.sharpnesses.shape} doesn't match "
                f"centers count ({n},)"
            )
        if self.colors is not None and self.colors.shape[0] != n:
            raise ValueError(
                f"Colors count {self.colors.shape[0]} doesn't match centers count {n}"
            )
        if self.centers.ndim >= 2:
            from luxar.gsplats.utils.trils import validate_cholesky_shape

            validate_cholesky_shape(
                self.cholesky_factors,
                ndim=self.centers.shape[1],
                n_splats=n,
                allow_uniform=False,
            )

    @property
    def n_splats(self) -> int:
        """Number of splats."""
        return self.centers.shape[0]

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.centers.shape[1] if self.centers.ndim >= 2 else 0

    def __len__(self) -> int:
        """Return number of splats."""
        return self.n_splats

    def __repr__(self) -> str:
        """Summary representation (avoids dumping full arrays)."""
        n = self.n_splats
        ndim = self.ndim
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

    # ── Computed properties ─────────────────────────────────

    def _cholesky_diag_elements(self) -> np.ndarray:
        """Extract diagonal elements from packed Cholesky factors.

        Returns shape (N, d) where result[i, j] = L_i[j, j].
        """
        ndim = self.ndim
        diag_indices = np.cumsum(np.arange(1, ndim + 1)) - 1
        return self.cholesky_factors[:, diag_indices]

    def volumes(self) -> np.ndarray:
        """Per-splat characteristic length: det(Σ)^(1/d).

        This is the geometric mean of the eigenvalues (not a true volume).
        For lower-triangular L: det(L) = product of diagonal elements,
        det(Sigma) = det(L)^2.

        Returns:
            shape (N,) float array.
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.float64)
        diag = self._cholesky_diag_elements()
        det_L = np.prod(diag, axis=1)
        return np.abs(det_L**2) ** (1.0 / self.ndim)

    def masses(self) -> np.ndarray:
        """Per-splat mass: amplitude * volume.

        Returns:
            shape (N,) float array.
        """
        return self.amplitudes * self.volumes()

    def marginal_sigmas(self) -> np.ndarray:
        """Per-dimension standard deviation: sqrt(Sigma_ii).

        For lower-triangular L: Sigma[i,i] = sum_j L[i,j]^2.

        Returns:
            shape (N, d) float array.
        """
        if self.n_splats == 0:
            return np.empty((0, self.ndim), dtype=np.float64)
        from luxar.gsplats.utils.trils import unpack_tril

        L = unpack_tril(self.cholesky_factors.astype(np.float64), self.ndim)
        return np.sqrt(np.sum(L**2, axis=2))

    def eccentricities(self) -> np.ndarray:
        """Per-splat eccentricity: max marginal sigma / min marginal sigma.

        1.0 = isotropic. Higher values = more elongated.

        Returns:
            shape (N,) float array. Returns 1.0 for degenerate splats.
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.float64)
        sigmas = self.marginal_sigmas()
        min_s = sigmas.min(axis=1)
        max_s = sigmas.max(axis=1)
        result = np.ones(self.n_splats, dtype=np.float64)
        nonzero = min_s > 0
        result[nonzero] = max_s[nonzero] / min_s[nonzero]
        return result

    # ── Filtering ───────────────────────────────────────────

    def filter(self, mask: np.ndarray) -> "GSplatData":
        """Return new GSplatData with only the splats where mask is True.

        Args:
            mask: Boolean array of shape (N,).

        Returns:
            New GSplatData with filtered arrays.

        Example:
            >>> filtered = data.filter(data.volumes() < 100)
            >>> filtered = data.filter((data.amplitudes > 0.1) & (data.eccentricities() < 5))
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_splats,):
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match splat count ({self.n_splats},)"
            )
        return GSplatData(
            centers=self.centers[mask],
            amplitudes=self.amplitudes[mask],
            cholesky_factors=self.cholesky_factors[mask],
            sharpnesses=self.sharpnesses[mask],
            colors=self.colors[mask] if self.colors is not None else None,
            stats=dict(self.stats),
        )

    @staticmethod
    def _resolve_threshold(
        val: float | None,
        normalized: bool,
        dataset_values: np.ndarray,
    ) -> float | None:
        """Map a threshold from [0,1] normalized range to absolute if needed."""
        if val is None:
            return None
        if normalized:
            dmin, dmax = float(dataset_values.min()), float(dataset_values.max())
            return dmin + val * (dmax - dmin)
        return val

    def filter_by(
        self,
        *,
        bbox: list[tuple[float, float]] | None = None,
        volume_min: float | None = None,
        volume_max: float | None = None,
        volume_normalized: bool = False,
        amplitude_min: float | None = None,
        amplitude_max: float | None = None,
        amplitude_normalized: bool = False,
        eccentricity_min: float | None = None,
        eccentricity_max: float | None = None,
        sharpness_min: float | None = None,
        sharpness_max: float | None = None,
        mass_min: float | None = None,
        mass_max: float | None = None,
        mass_normalized: bool = False,
        sigma_axis: int | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        truncate: float = 3.0,
    ) -> "GSplatData":
        """Filter splats by multiple criteria (AND logic).

        All criteria are optional. Only specified criteria are applied.
        Multiple criteria combine with AND — a splat must satisfy all
        active criteria to be kept.

        Args:
            bbox: Bounding box per dimension as [(min0, max0), (min1, max1), ...].
                  Length must equal ndim. Filters by center position.
            volume_min: Minimum volume (characteristic length * truncate).
            volume_max: Maximum volume.
            volume_normalized: If True, interpret volume thresholds as 0-1
                mapped to the dataset's [min, max] volume range.
            amplitude_min: Minimum amplitude.
            amplitude_max: Maximum amplitude.
            amplitude_normalized: If True, interpret amplitude thresholds as 0-1
                mapped to the dataset's [min, max] amplitude range.
            eccentricity_min: Minimum eccentricity (1.0 = isotropic).
            eccentricity_max: Maximum eccentricity.
            sharpness_min: Minimum sharpness value.
            sharpness_max: Maximum sharpness value.
            mass_min: Minimum mass (amplitude * volume).
            mass_max: Maximum mass.
            mass_normalized: If True, interpret mass thresholds as 0-1
                mapped to the dataset's [min, max] mass range.
            sigma_axis: Axis index for per-axis sigma filtering.
            sigma_min: Minimum marginal sigma on sigma_axis.
            sigma_max: Maximum marginal sigma on sigma_axis.
            truncate: Sigma truncation factor for volume computation (default 3.0).

        Returns:
            New GSplatData with only splats that pass all criteria.

        Raises:
            ValueError: If bbox length doesn't match ndim, sigma_axis is out
                of range, or sigma_min/sigma_max given without sigma_axis.

        Examples:
            >>> # Keep splats with amplitude >= 0.1 and eccentricity <= 5
            >>> filtered = data.filter_by(amplitude_min=0.1, eccentricity_max=5.0)
            >>>
            >>> # Spatial crop to a bounding box (3D)
            >>> filtered = data.filter_by(bbox=[(0, 50), (0, 50), (0, 50)])
            >>>
            >>> # Remove top 10% largest volumes (normalized)
            >>> filtered = data.filter_by(volume_max=0.9, volume_normalized=True)
        """
        # Short-circuit for empty data
        if self.n_splats == 0:
            result = self.filter(np.ones(0, dtype=bool))
            result.stats.update(
                {
                    "filtered": True,
                    "filter_criteria": {},
                    "n_original": 0,
                    "n_removed": 0,
                    "truncate": truncate,
                }
            )
            return result

        # Validate sigma_axis usage
        if (sigma_min is not None or sigma_max is not None) and sigma_axis is None:
            raise ValueError("sigma_min/sigma_max require sigma_axis to be specified")
        if sigma_axis is not None and not (0 <= sigma_axis < self.ndim):
            raise ValueError(
                f"sigma_axis={sigma_axis} out of range for {self.ndim}D data"
            )

        mask = np.ones(self.n_splats, dtype=bool)
        criteria: dict[str, object] = {}

        # -- Bounding box (center position)
        if bbox is not None:
            if len(bbox) != self.ndim:
                raise ValueError(
                    f"bbox has {len(bbox)} dimensions, expected {self.ndim}"
                )
            criteria["bbox"] = bbox
            for i, (lo, hi) in enumerate(bbox):
                mask &= (self.centers[:, i] >= lo) & (self.centers[:, i] <= hi)

        # -- Volume (characteristic length * truncate)
        if volume_min is not None or volume_max is not None:
            vols = self.volumes() * truncate
            vmin = self._resolve_threshold(volume_min, volume_normalized, vols)
            vmax = self._resolve_threshold(volume_max, volume_normalized, vols)
            if vmin is not None:
                mask &= vols >= vmin
                criteria["volume_min"] = vmin
            if vmax is not None:
                mask &= vols <= vmax
                criteria["volume_max"] = vmax
            if volume_normalized:
                criteria["volume_normalized"] = True

        # -- Amplitude
        if amplitude_min is not None or amplitude_max is not None:
            amps = self.amplitudes
            amin = self._resolve_threshold(amplitude_min, amplitude_normalized, amps)
            amax = self._resolve_threshold(amplitude_max, amplitude_normalized, amps)
            if amin is not None:
                mask &= amps >= amin
                criteria["amplitude_min"] = amin
            if amax is not None:
                mask &= amps <= amax
                criteria["amplitude_max"] = amax
            if amplitude_normalized:
                criteria["amplitude_normalized"] = True

        # -- Eccentricity
        if eccentricity_min is not None or eccentricity_max is not None:
            ecc = self.eccentricities()
            if eccentricity_min is not None:
                mask &= ecc >= eccentricity_min
                criteria["eccentricity_min"] = eccentricity_min
            if eccentricity_max is not None:
                mask &= ecc <= eccentricity_max
                criteria["eccentricity_max"] = eccentricity_max

        # -- Sharpness
        if sharpness_min is not None or sharpness_max is not None:
            sharp = self.sharpnesses
            if sharpness_min is not None:
                mask &= sharp >= sharpness_min
                criteria["sharpness_min"] = sharpness_min
            if sharpness_max is not None:
                mask &= sharp <= sharpness_max
                criteria["sharpness_max"] = sharpness_max

        # -- Mass (amplitude * volume)
        if mass_min is not None or mass_max is not None:
            m = self.masses()
            mmin = self._resolve_threshold(mass_min, mass_normalized, m)
            mmax = self._resolve_threshold(mass_max, mass_normalized, m)
            if mmin is not None:
                mask &= m >= mmin
                criteria["mass_min"] = mmin
            if mmax is not None:
                mask &= m <= mmax
                criteria["mass_max"] = mmax
            if mass_normalized:
                criteria["mass_normalized"] = True

        # -- Per-axis sigma
        if sigma_axis is not None and (sigma_min is not None or sigma_max is not None):
            sigmas = self.marginal_sigmas()[:, sigma_axis]
            criteria["sigma_axis"] = sigma_axis
            if sigma_min is not None:
                mask &= sigmas >= sigma_min
                criteria["sigma_min"] = sigma_min
            if sigma_max is not None:
                mask &= sigmas <= sigma_max
                criteria["sigma_max"] = sigma_max

        # Apply mask
        result = self.filter(mask)
        result.stats.update(
            {
                "filtered": True,
                "filter_criteria": criteria,
                "n_original": self.n_splats,
                "n_removed": self.n_splats - result.n_splats,
                "truncate": truncate,
            }
        )
        return result

    def slice_by(self, slices: list[slice]) -> "GSplatData":
        """Slice splats by coordinate ranges per dimension (numpy-style).

        Each slice specifies a [start, stop] range for that dimension's center
        coordinate. ``None`` in start/stop means unbounded.

        Args:
            slices: One slice per dimension. ``slice(lo, hi)`` keeps splats
                with center in [lo, hi]. ``slice(None, None)`` keeps all.

        Returns:
            New GSplatData with only splats inside all ranges.

        Raises:
            ValueError: If number of slices doesn't match ndim.

        Examples:
            >>> # Keep x in [0,50], all y, z in [10,90]
            >>> sliced = data.slice_by([slice(0, 50), slice(None, None), slice(10, 90)])
            >>>
            >>> # Open-ended: x >= 50
            >>> sliced = data.slice_by([slice(50, None), slice(None, None), slice(None, None)])
        """
        if len(slices) != self.ndim:
            raise ValueError(f"Expected {self.ndim} slices, got {len(slices)}")
        bbox = []
        for s in slices:
            lo = float(s.start) if s.start is not None else float("-inf")
            hi = float(s.stop) if s.stop is not None else float("inf")
            bbox.append((lo, hi))
        return self.filter_by(bbox=bbox)

    # ── Combine / Split / Embed ─────────────────────────────

    @classmethod
    def concatenate(cls, datasets: list["GSplatData"]) -> "GSplatData":
        """Concatenate multiple GSplatData objects into one.

        All datasets must have the same dimensionality.

        Colors: if all have colors, concatenate; if all None, None;
        if mixed, fill missing with white (1,1,1).

        Args:
            datasets: List of GSplatData (same ndim required).

        Returns:
            New GSplatData with all splats concatenated.
        """
        if len(datasets) == 0:
            raise ValueError("At least one GSplatData is required")

        # Filter out empty datasets to avoid shape mismatch in np.concatenate
        non_empty = [d for d in datasets if d.n_splats > 0]
        if len(non_empty) == 0:
            return datasets[0]  # All empty: return first as-is

        ndim = non_empty[0].ndim
        for i, ds in enumerate(non_empty[1:], start=1):
            if ds.ndim != ndim:
                raise ValueError(
                    f"Dimensionality mismatch: dataset 0 has {ndim}D, "
                    f"dataset {i} has {ds.ndim}D"
                )

        all_centers = np.concatenate([d.centers for d in non_empty], axis=0)
        all_amplitudes = np.concatenate([d.amplitudes for d in non_empty])
        all_cholesky = np.concatenate([d.cholesky_factors for d in non_empty], axis=0)
        all_sharpnesses = np.concatenate([d.sharpnesses for d in non_empty])

        has_colors = [d.colors is not None for d in non_empty]
        if all(has_colors):
            all_colors = np.concatenate([d.colors for d in non_empty], axis=0)
        elif not any(has_colors):
            all_colors = None
        else:
            parts = []
            for d in non_empty:
                if d.colors is not None:
                    parts.append(d.colors)
                else:
                    parts.append(np.ones((d.n_splats, 3), dtype=np.float32))
            all_colors = np.concatenate(parts, axis=0)

        merged_stats: Dict[str, Any] = {
            "concatenated_from": len(datasets),
            "splats_per_source": [d.n_splats for d in datasets],
        }
        total_time = sum(d.stats.get("time_seconds", 0) for d in non_empty)
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

    @classmethod
    def combine_as_new_dimension(
        cls,
        datasets: "list[GSplatData]",
        values: "np.ndarray | list[float] | None" = None,
        sigma: float = 0.0,
    ) -> "GSplatData":
        """Combine datasets by embedding each into a new dimension, then concatenating.

        Each dataset is promoted from D-dimensional to (D+1)-dimensional by
        appending a coordinate in the new dimension, then all are concatenated
        into a single dataset.

        This is useful for combining per-timepoint 3D fits into a single 4D
        dataset, per-slice 2D fits into 3D, or any similar stacking operation.

        Args:
            datasets: List of GSplatData, all with the same ndim.
            values: Coordinate for each dataset in the new dimension.
                If None, uses 0.0, 1.0, 2.0, ... (one per dataset).
                If scalar-per-dataset, all splats in that dataset get the same
                coordinate.  Can also be a list of per-splat arrays if different
                splats within a dataset need different coordinates.
            sigma: Standard deviation in the new dimension.
                Use 0.0 for discrete dimensions (e.g., time frames) where
                splats should not extend across the new axis.
                Use a positive value for continuous dimensions where splats
                should have Gaussian extent.

        Returns:
            Single GSplatData with ndim+1 dimensions containing all splats.

        Raises:
            ValueError: If datasets is empty, lengths mismatch, or ndims differ.

        Example:
            >>> # Combine 3D timepoints into 4D
            >>> combined = GSplatData.combine_as_new_dimension(
            ...     [t0_3d, t1_3d, t2_3d], sigma=0.0
            ... )
            >>> combined.ndim  # 4
            >>> combined.n_splats  # sum of all timepoints
        """
        if not datasets:
            raise ValueError("At least one GSplatData is required")

        if values is None:
            values = [float(i) for i in range(len(datasets))]
        elif hasattr(values, "__len__"):
            values = list(values)
        else:
            raise TypeError(
                f"values must be a list/array or None, got {type(values).__name__}"
            )

        if len(values) != len(datasets):
            raise ValueError(
                f"Number of values ({len(values)}) must match "
                f"number of datasets ({len(datasets)})"
            )

        embedded = [
            ds.embed_dimension(val, sigma=sigma) for ds, val in zip(datasets, values)
        ]
        return cls.concatenate(embedded)

    def split(self, n_or_indices: "int | list[int] | np.ndarray") -> "list[GSplatData]":
        """Split into multiple GSplatData objects.

        Args:
            n_or_indices: If int, split into n roughly equal parts.
                If list/array of ints, split at those indices.

        Returns:
            List of GSplatData objects.
        """
        indices = np.arange(self.n_splats)
        if isinstance(n_or_indices, int):
            groups = np.array_split(indices, n_or_indices)
        else:
            groups = np.split(indices, n_or_indices)

        from luxar.gsplats.utils.trils import tril_size

        results = []
        for idx in groups:
            if len(idx) == 0:
                d = self.ndim
                k = tril_size(d) if d > 0 else 0
                results.append(
                    GSplatData(
                        centers=np.empty((0, d), dtype=self.centers.dtype),
                        amplitudes=np.empty(0, dtype=self.amplitudes.dtype),
                        cholesky_factors=np.empty(
                            (0, k), dtype=self.cholesky_factors.dtype
                        ),
                        sharpnesses=np.empty(0, dtype=self.sharpnesses.dtype),
                        colors=np.empty((0, 3), dtype=np.float32)
                        if self.colors is not None
                        else None,
                        stats=dict(self.stats),
                    )
                )
            else:
                results.append(
                    GSplatData(
                        centers=self.centers[idx],
                        amplitudes=self.amplitudes[idx],
                        cholesky_factors=self.cholesky_factors[idx],
                        sharpnesses=self.sharpnesses[idx],
                        colors=self.colors[idx] if self.colors is not None else None,
                        stats=dict(self.stats),
                    )
                )
        return results

    def embed_dimension(
        self,
        values: "np.ndarray | float",
        sigma: float = 0.0,
    ) -> "GSplatData":
        """Add a new dimension to the splat data.

        Appends a column to centers and embeds Cholesky factors into
        the higher-dimensional space.

        Args:
            values: Coordinate for the new dimension. Scalar (same for all)
                or (N,) array (per-splat).
            sigma: Standard deviation in the new dimension (default 0.0
                for discrete dimensions like time).

        Returns:
            New GSplatData with ndim+1 dimensions.

        Example:
            >>> data_4d = data_3d.embed_dimension(5.0, sigma=0.0)
            >>> data_4d = data_3d.embed_dimension(time_values, sigma=0.5)
        """
        from luxar.gsplats.utils.trils import embed_cholesky_packed

        n = self.n_splats
        d = self.ndim

        if np.isscalar(values):
            new_col = np.full((n, 1), values, dtype=self.centers.dtype)
        else:
            values = np.asarray(values, dtype=self.centers.dtype)
            if values.shape != (n,):
                raise ValueError(
                    f"values shape {values.shape} doesn't match splat count ({n},)"
                )
            new_col = values.reshape(n, 1)

        new_centers = np.concatenate([self.centers, new_col], axis=1)
        new_cholesky = embed_cholesky_packed(
            self.cholesky_factors,
            d_src=d,
            d_dst=d + 1,
            dim_mapping=list(range(d)),
            fill_sigma={d: sigma},
        )

        return GSplatData(
            centers=new_centers,
            amplitudes=self.amplitudes,
            cholesky_factors=new_cholesky,
            sharpnesses=self.sharpnesses,
            colors=self.colors,
            stats=dict(self.stats),
        )

    # ── Geometric transforms ────────────────────────────────

    def transform(self, matrix: np.ndarray) -> "GSplatData":
        """Apply affine transformation to all splats.

        Transforms centers and covariance matrices. Amplitudes, sharpnesses,
        and colors are unchanged.

        Args:
            matrix: Either (d, d) for linear-only transform or
                (d+1, d+1) for full affine (last row must be [0..0, 1]).

        Returns:
            New GSplatData with transformed geometry.

        Raises:
            ValueError: If matrix shape is invalid.
            np.linalg.LinAlgError: If transform produces non-positive-definite covariance.

        Example:
            >>> scaled = data.transform(np.eye(3) * 2.0)
            >>> M = np.eye(4); M[:3, 3] = [10, 20, 30]
            >>> transformed = data.transform(M)
        """
        from luxar.gsplats.utils.trils import pack_tril, unpack_tril

        matrix = np.asarray(matrix, dtype=np.float64)
        d = self.ndim

        if matrix.shape == (d, d):
            A = matrix
            t = np.zeros(d, dtype=np.float64)
        elif matrix.shape == (d + 1, d + 1):
            A = matrix[:d, :d]
            t = matrix[:d, d]
            expected = np.zeros(d + 1, dtype=np.float64)
            expected[-1] = 1.0
            if not np.allclose(matrix[d, :], expected):
                raise ValueError(
                    f"Last row of (d+1)x(d+1) matrix must be [0...0, 1], "
                    f"got {matrix[d, :]}"
                )
        else:
            raise ValueError(
                f"Matrix shape must be ({d},{d}) or ({d + 1},{d + 1}), got {matrix.shape}"
            )

        if self.n_splats == 0:
            return GSplatData(
                centers=self.centers.copy(),
                amplitudes=self.amplitudes,
                cholesky_factors=self.cholesky_factors.copy(),
                sharpnesses=self.sharpnesses,
                colors=self.colors,
                stats=dict(self.stats),
            )

        new_centers = (self.centers.astype(np.float64) @ A.T + t).astype(
            self.centers.dtype
        )

        L = unpack_tril(self.cholesky_factors.astype(np.float64), d)
        Sigma = L @ np.swapaxes(L, -2, -1)
        Sigma_new = A @ Sigma @ A.T
        L_new = np.linalg.cholesky(Sigma_new)
        new_cholesky = pack_tril(L_new).astype(self.cholesky_factors.dtype)

        return GSplatData(
            centers=new_centers,
            amplitudes=self.amplitudes,
            cholesky_factors=new_cholesky,
            sharpnesses=self.sharpnesses,
            colors=self.colors,
            stats=dict(self.stats),
        )

    # ── Intensity transforms ────────────────────────────────

    def affine_intensity(self, scale: float = 1.0, offset: float = 0.0) -> "GSplatData":
        """Apply affine transform to amplitudes: new_amp = scale * amp + offset.

        Args:
            scale: Multiplicative factor.
            offset: Additive offset.

        Returns:
            New GSplatData with transformed amplitudes.
        """
        return GSplatData(
            centers=self.centers,
            amplitudes=self.amplitudes * scale + offset,
            cholesky_factors=self.cholesky_factors,
            sharpnesses=self.sharpnesses,
            colors=self.colors,
            stats=dict(self.stats),
        )

    def normalize_intensity(self, target_max: float = 1.0) -> "GSplatData":
        """Normalize amplitudes so the maximum equals target_max.

        Args:
            target_max: Desired maximum amplitude (default 1.0).

        Returns:
            New GSplatData. Returns copy if all amplitudes are zero.
        """
        current_max = float(self.amplitudes.max()) if self.n_splats > 0 else 0.0
        if current_max == 0:
            return GSplatData(
                centers=self.centers,
                amplitudes=self.amplitudes.copy(),
                cholesky_factors=self.cholesky_factors,
                sharpnesses=self.sharpnesses,
                colors=self.colors,
                stats=dict(self.stats),
            )
        return self.scale_intensity(target_max / current_max)

    def clamp_intensity(
        self,
        min: "float | None" = None,
        max: "float | None" = None,
    ) -> "GSplatData":
        """Clamp amplitudes to a range.

        Args:
            min: Lower bound (None = no lower bound).
            max: Upper bound (None = no upper bound).

        Returns:
            New GSplatData with clamped amplitudes.
        """
        new_amps = self.amplitudes.copy()
        if min is not None:
            new_amps = np.maximum(new_amps, min)
        if max is not None:
            new_amps = np.minimum(new_amps, max)
        return GSplatData(
            centers=self.centers,
            amplitudes=new_amps,
            cholesky_factors=self.cholesky_factors,
            sharpnesses=self.sharpnesses,
            colors=self.colors,
            stats=dict(self.stats),
        )

    # ── I/O ─────────────────────────────────────────────────

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
            # Extract common fitting fields (including quality metrics and pruning stats)
            fitting_info = {
                k: v
                for k, v in self.stats.items()
                if k
                in [
                    "time_seconds",
                    "iterations",
                    "converged",
                    "early_stopped",
                    "best_iteration",
                    "final_loss",
                    "final_max_abs_error",
                    "final_rel_l2",
                    "n_splats",
                    "n_splats_before_culling",
                    "n_culled",
                    "fitter_name",
                    "fitter_version",
                    "timestamp",
                    "pruned",
                    "pruning_method",
                    "n_original",
                    "n_removed",
                    "amplitude_retention",
                    "filtered",
                    "filter_criteria",
                    "truncate",
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
            stats=dict(self.stats),
        )

    def center_at_centroid(self) -> "GSplatData":
        """Center the splats at their center of mass (amplitude-weighted centroid).

        The centroid is computed as the amplitude-weighted average of splat centers,
        which corresponds to the center of mass of the represented density.

        Returns:
            New GSplatData centered at origin (amplitude-weighted centroid at [0, 0, ...])

        Example:
            >>> # Center splats at origin for easier viewing
            >>> centered = data.center_at_centroid()
            >>> # Amplitude-weighted centroid is now at origin
            >>> centroid = (centered.centers.T @ centered.amplitudes) / centered.amplitudes.sum()
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
            stats=dict(self.stats),
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
        N_original = self.n_splats

        # Validate method
        valid_methods = ("cumulative", "amplitude_percentile", "combined")
        if method not in valid_methods:
            raise ValueError(f"Unknown pruning method: {method}")

        # Short-circuit for empty data
        if N_original == 0:
            result = self.filter(np.ones(0, dtype=bool))
            result.stats.update(
                {
                    "pruned": True,
                    "pruning_method": method,
                    "n_original": 0,
                    "n_removed": 0,
                }
            )
            return result

        # Compute mask based on pruning strategy
        mask = np.ones(N_original, dtype=bool)

        if method == "cumulative":
            sorted_indices = np.argsort(self.amplitudes)[::-1]
            sorted_amps = self.amplitudes[sorted_indices]
            cumsum_amps = np.cumsum(sorted_amps)
            total_amp = cumsum_amps[-1]
            if total_amp == 0:
                mask = (
                    np.ones(N_original, dtype=bool)
                    if target_retention > 0
                    else np.zeros(N_original, dtype=bool)
                )
            else:
                cumsum_norm = cumsum_amps / total_amp
                n_keep = np.searchsorted(cumsum_norm, target_retention) + 1
                n_keep = min(n_keep, N_original)
                keep_indices = sorted_indices[:n_keep]
                mask = np.zeros(N_original, dtype=bool)
                mask[keep_indices] = True

        elif method == "amplitude_percentile":
            threshold = np.percentile(self.amplitudes, amplitude_percentile)
            mask = self.amplitudes >= threshold

        elif method == "combined":
            vols = self.volumes()
            amp_threshold = np.percentile(self.amplitudes, amplitude_percentile)
            vol_threshold = np.percentile(vols, volume_percentile)
            mask = (self.amplitudes >= amp_threshold) & (vols <= vol_threshold)

        # Apply mask via filter()
        result = self.filter(mask)

        # Update stats with pruning metadata
        total_amp = np.sum(self.amplitudes)
        result.stats.update(
            {
                "pruned": True,
                "pruning_method": method,
                "n_original": N_original,
                "n_removed": N_original - result.n_splats,
                "amplitude_retention": (
                    float(np.sum(result.amplitudes) / total_amp)
                    if total_amp > 0
                    else 1.0
                ),
            }
        )

        return result

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
        ndim = gsplats_per_channel[0].ndim
        for i, gsplat in enumerate(gsplats_per_channel[1:], start=1):
            if gsplat.ndim != ndim:
                raise ValueError(
                    f"Dimensionality mismatch: channel 0 has {ndim}D, "
                    f"channel {i} has {gsplat.ndim}D"
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
