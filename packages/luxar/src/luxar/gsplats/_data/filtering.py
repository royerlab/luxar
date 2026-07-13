"""Attribute/geometry filtering mixin for ``GSplatData``."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .base import _GSplatDataOps

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


class FilteringMixin(_GSplatDataOps):
    """``filter`` / ``filter_by`` / ``slice_by`` and the threshold resolver."""

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
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_splats,):
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match splat count ({self.n_splats},)"
            )

        # A raw boolean mask is sized to the default substitutive level, so it
        # cannot be applied per-level — coarser substitutive levels are dropped.
        # Warn loudly (never silent) and point at the criteria-based ops, which
        # DO preserve the full pyramid (see filter_by / cull).
        if self.n_substitutive > 1:
            warnings.warn(
                "filter(mask) keeps only the default substitutive level "
                f"(n_substitutive={self.n_substitutive}); coarser levels are "
                "dropped. Use filter_by(...) / cull(...) to filter every "
                "substitutive level and preserve the pyramid.",
                UserWarning,
                stacklevel=2,
            )

        # Multi-LOD path: split mask across LODs
        if self.n_additive_sublods > 1:

            def _filter_lod(lod: AdditiveSubLOD, offset: int, n: int) -> AdditiveSubLOD:
                lod_mask = mask[offset : offset + n]
                return AdditiveSubLOD(
                    centers=lod.centers[lod_mask],
                    amplitudes=lod.amplitudes[lod_mask],
                    cholesky_factors=lod.cholesky_factors[lod_mask],
                    colors=lod.colors[lod_mask] if lod.colors is not None else None,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )

            return self._map_additive(_filter_lod)

        return GSplatData(
            centers=self.centers[mask],
            amplitudes=self.amplitudes[mask],
            cholesky_factors=self.cholesky_factors[mask],
            colors=self.colors[mask] if self.colors is not None else None,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    @staticmethod
    def _resolve_threshold(
        val: float | None,
        normalized: bool,
        dataset_values: np.ndarray,
        percentile: bool = False,
    ) -> float | None:
        """Resolve a threshold to an absolute value.

        - ``percentile``: ``val`` in [0,100] → the ``val``-th percentile of
          ``dataset_values`` (robust on heavy-tailed attributes; preferred over
          ``normalized``).
        - ``normalized``: ``val`` in [0,1] → linear map onto [min, max].
        - otherwise: ``val`` is already absolute.
        """
        if val is None:
            return None
        if percentile:
            return float(np.percentile(dataset_values, val))
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
        volume_percentile: bool = False,
        scale_min: float | None = None,
        scale_max: float | None = None,
        scale_normalized: bool = False,
        scale_percentile: bool = False,
        amplitude_min: float | None = None,
        amplitude_max: float | None = None,
        amplitude_normalized: bool = False,
        amplitude_percentile: bool = False,
        eccentricity_min: float | None = None,
        eccentricity_max: float | None = None,
        eccentricity_percentile: bool = False,
        mass_min: float | None = None,
        mass_max: float | None = None,
        mass_normalized: bool = False,
        mass_percentile: bool = False,
        sigma_axis: int | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        sigma_percentile: bool = False,
        isolation_max: float | None = None,
        isolation_percentile: bool = False,
        min_neighbors: int | None = None,
        neighbor_radius: float | None = None,
        spatial_dims: Sequence[int] | None = None,
        truncate: float | None = None,
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
            mass_min: Minimum mass (amplitude * volume).
            mass_max: Maximum mass.
            mass_normalized: If True, interpret mass thresholds as 0-1
                mapped to the dataset's [min, max] mass range.
            sigma_axis: Axis index for per-axis sigma filtering.
            sigma_min: Minimum marginal sigma on sigma_axis.
            sigma_max: Maximum marginal sigma on sigma_axis.
            scale_min/scale_max: Characteristic size (geometric-mean marginal
                sigma over the spatial/``spatial_dims`` axes; see ``scale()``).
                The recommended "remove large diffuse background" knob — cleaner
                than ``volume`` on nD timelapses.
            isolation_max: Remove splats whose nearest-neighbour distance (over
                the spatial axes, grouped by the non-spatial axes) EXCEEDS this
                — i.e. spatially isolated noise splats.
            min_neighbors / neighbor_radius: Remove splats with fewer than
                ``min_neighbors`` other splats within ``neighbor_radius``.
            spatial_dims: Override the axes used for scale / eccentricity /
                isolation (default: auto-detected non-degenerate axes).
            *_percentile: For volume/scale/amplitude/mass/sigma/eccentricity/
                isolation — interpret the corresponding min/max as a percentile
                in [0,100] of that attribute (robust on heavy-tailed data).
            truncate: Sigma truncation factor for volume computation.
                Defaults to ``self.truncation_radius``.

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
        if truncate is None:
            truncate = self.truncation_radius

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
        # Local-density filter needs both knobs (mirrors the sigma_axis rule).
        if (min_neighbors is None) != (neighbor_radius is None):
            raise ValueError(
                "min_neighbors and neighbor_radius must be specified together"
            )

        # Multi-substitutive: apply the SAME criteria to every substitutive
        # level and rebuild the pyramid (decision 6) rather than silently
        # collapsing to the default level. Each level is filtered through the
        # single-substitutive path below (a per-level view); thresholds with
        # *_normalized resolve per-level (each level to its own range).
        if self.n_substitutive > 1:
            out = self._map_substitutive(
                lambda lvl: lvl.filter_by(
                    bbox=bbox,
                    volume_min=volume_min,
                    volume_max=volume_max,
                    volume_normalized=volume_normalized,
                    volume_percentile=volume_percentile,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    scale_normalized=scale_normalized,
                    scale_percentile=scale_percentile,
                    amplitude_min=amplitude_min,
                    amplitude_max=amplitude_max,
                    amplitude_normalized=amplitude_normalized,
                    amplitude_percentile=amplitude_percentile,
                    eccentricity_min=eccentricity_min,
                    eccentricity_max=eccentricity_max,
                    eccentricity_percentile=eccentricity_percentile,
                    mass_min=mass_min,
                    mass_max=mass_max,
                    mass_normalized=mass_normalized,
                    mass_percentile=mass_percentile,
                    sigma_axis=sigma_axis,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    sigma_percentile=sigma_percentile,
                    isolation_max=isolation_max,
                    isolation_percentile=isolation_percentile,
                    min_neighbors=min_neighbors,
                    neighbor_radius=neighbor_radius,
                    spatial_dims=spatial_dims,
                    truncate=truncate,
                )
            )
            out.stats.update(
                {
                    "filtered": True,
                    "n_original": self.n_splats,
                    "n_removed": self.n_splats - out.n_splats,
                    "truncate": truncate,
                }
            )
            return out

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
            vmin = self._resolve_threshold(
                volume_min, volume_normalized, vols, volume_percentile
            )
            vmax = self._resolve_threshold(
                volume_max, volume_normalized, vols, volume_percentile
            )
            if vmin is not None:
                mask &= vols >= vmin
                criteria["volume_min"] = vmin
            if vmax is not None:
                mask &= vols <= vmax
                criteria["volume_max"] = vmax
            if volume_normalized:
                criteria["volume_normalized"] = True

        # -- Scale (geometric-mean marginal sigma over the spatial axes)
        if scale_min is not None or scale_max is not None:
            scl = self.scale(axes=spatial_dims)
            smin = self._resolve_threshold(
                scale_min, scale_normalized, scl, scale_percentile
            )
            smax = self._resolve_threshold(
                scale_max, scale_normalized, scl, scale_percentile
            )
            if smin is not None:
                mask &= scl >= smin
                criteria["scale_min"] = smin
            if smax is not None:
                mask &= scl <= smax
                criteria["scale_max"] = smax

        # -- Amplitude
        if amplitude_min is not None or amplitude_max is not None:
            amps = self.amplitudes
            amin = self._resolve_threshold(
                amplitude_min, amplitude_normalized, amps, amplitude_percentile
            )
            amax = self._resolve_threshold(
                amplitude_max, amplitude_normalized, amps, amplitude_percentile
            )
            if amin is not None:
                mask &= amps >= amin
                criteria["amplitude_min"] = amin
            if amax is not None:
                mask &= amps <= amax
                criteria["amplitude_max"] = amax
            if amplitude_normalized:
                criteria["amplitude_normalized"] = True

        # -- Eccentricity (spatial isotropy; auto-ignores degenerate axes)
        if eccentricity_min is not None or eccentricity_max is not None:
            ecc = self.eccentricities(axes=spatial_dims)
            emin = self._resolve_threshold(
                eccentricity_min, False, ecc, eccentricity_percentile
            )
            emax = self._resolve_threshold(
                eccentricity_max, False, ecc, eccentricity_percentile
            )
            if emin is not None:
                mask &= ecc >= emin
                criteria["eccentricity_min"] = emin
            if emax is not None:
                mask &= ecc <= emax
                criteria["eccentricity_max"] = emax

        # -- Mass (amplitude * volume)
        if mass_min is not None or mass_max is not None:
            m = self.masses()
            mmin = self._resolve_threshold(
                mass_min, mass_normalized, m, mass_percentile
            )
            mmax = self._resolve_threshold(
                mass_max, mass_normalized, m, mass_percentile
            )
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
            smn = self._resolve_threshold(sigma_min, False, sigmas, sigma_percentile)
            smx = self._resolve_threshold(sigma_max, False, sigmas, sigma_percentile)
            if smn is not None:
                mask &= sigmas >= smn
                criteria["sigma_min"] = smn
            if smx is not None:
                mask &= sigmas <= smx
                criteria["sigma_max"] = smx

        # -- Isolation (remove spatially-isolated noise splats)
        if isolation_max is not None:
            nn = self.nearest_neighbor_distances(spatial_axes=spatial_dims)
            finite = nn[np.isfinite(nn)]
            if isolation_percentile and finite.size == 0:
                # Every splat is an isolated singleton (no finite NN distance);
                # a percentile is undefined → drop them all.
                mask &= False
                criteria["isolation_max"] = "all-isolated"
            else:
                imax = self._resolve_threshold(
                    isolation_max, False, finite, isolation_percentile
                )
                if imax is not None:
                    # +inf (no neighbour) always exceeds the threshold → removed.
                    mask &= nn <= imax
                    criteria["isolation_max"] = imax

        # -- Local density (keep only well-supported splats)
        if min_neighbors is not None and neighbor_radius is not None:
            counts = self.neighbor_counts(neighbor_radius, spatial_axes=spatial_dims)
            mask &= counts >= int(min_neighbors)
            criteria["min_neighbors"] = int(min_neighbors)
            criteria["neighbor_radius"] = float(neighbor_radius)

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
