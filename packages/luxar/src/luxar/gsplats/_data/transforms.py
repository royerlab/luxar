"""Geometric-transform mixin for ``GSplatData``.

Affine transforms, translation and centroid centering, plus the two
structure-preserving map helpers (``_map_substitutive`` / ``_map_additive``)
every per-level op in the sibling mixins rebuilds its ladder through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

import numpy as np

# center_at_centroid shifts only the spatial axes (#487).
from luxar.gsplats.utils.spatial_axes import spatial_only_shift

from .base import _GSplatDataOps

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import (
        AdditiveSubLOD,
        GSplatData,
        SubstitutiveLevel,
    )


class TransformsMixin(_GSplatDataOps):
    """``transform`` / ``translate`` / ``center_at_centroid`` and the per-level
    ``_map_substitutive`` / ``_map_additive`` rebuild helpers."""

    def _map_substitutive(
        self, fn: "Callable[[GSplatData], GSplatData]"
    ) -> "GSplatData":
        """Apply a single-level transform to EVERY substitutive level, rebuild.

        ``fn`` maps a single-substitutive-level view (``n_substitutive == 1``)
        to a transformed single-level ``GSplatData``; per-level metadata
        (compression_factor / parent_method / level_index / stats) is preserved.
        Mirrors :meth:`filter_by`'s per-level rebuild so spatial
        and intensity ops never silently collapse the substitutive LOD ladder
        to the finest level. Callers guard with ``if self.n_substitutive > 1``.
        """
        from luxar.gsplats.gsplat_data import GSplatData, SubstitutiveLevel

        new_levels: List["SubstitutiveLevel"] = []
        for s, src in enumerate(self.substitutive_levels):
            out = fn(self._view_of_level(src))
            new_levels.append(
                SubstitutiveLevel(
                    additive_sublods=out.substitutive_levels[0].additive_sublods,
                    compression_factor=src.compression_factor,
                    parent_method=src.parent_method,
                    level_index=src.level_index,
                    stats=dict(src.stats),
                )
            )
        return GSplatData.from_substitutive_levels(new_levels, stats=dict(self.stats))

    def _map_additive(
        self, fn: "Callable[[AdditiveSubLOD, int, int], AdditiveSubLOD]"
    ) -> "GSplatData":
        """Apply a per-sub-LOD transform to EVERY additive sub-LOD, rebuild.

        ``fn`` receives ``(lod, offset, n)`` — the sub-LOD, its start offset
        into the flattened finest-leaf arrays, and its splat count — and returns
        a replacement :class:`AdditiveSubLOD` (which may change N, ndim, or
        array widths). The additive-dimension sibling of :meth:`_map_substitutive`;
        callers guard the multi-sub-LOD branch with
        ``if self.n_additive_sublods > 1``.
        """
        from luxar.gsplats.gsplat_data import GSplatData

        new_lods: List["AdditiveSubLOD"] = []
        offset = 0
        for lod in self.additive_sublods:
            n = lod.n_splats
            new_lods.append(fn(lod, offset, n))
            offset += n
        return GSplatData.from_additive_sublods(new_lods, stats=dict(self.stats))

    def transform(self, matrix: np.ndarray) -> "GSplatData":
        """Apply affine transformation to all splats.

        Transforms centers and covariance matrices. Amplitudes and colors
        are unchanged.

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
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData
        from luxar.gsplats.utils.trils import pack_tril, unpack_tril

        # Multi-substitutive: transform every level and rebuild the pyramid
        # (mirrors filter_by/cull) rather than collapsing to the finest level.
        if self.n_substitutive > 1:
            return self._map_substitutive(lambda lvl: lvl.transform(matrix))

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
            if self.n_additive_sublods > 1:
                return GSplatData.from_additive_sublods(
                    [
                        AdditiveSubLOD(
                            centers=lod.centers.copy(),
                            amplitudes=lod.amplitudes,
                            cholesky_factors=lod.cholesky_factors.copy(),
                            colors=lod.colors,
                            stats=dict(lod.stats),
                            truncation_radius=lod.truncation_radius,
                        )
                        for lod in self.additive_sublods
                    ],
                    stats=dict(self.stats),
                )
            return GSplatData(
                centers=self.centers.copy(),
                amplitudes=self.amplitudes,
                cholesky_factors=self.cholesky_factors.copy(),
                colors=self.colors,
                stats=dict(self.stats),
                truncation_radius=self.truncation_radius,
            )

        # Precompute cholesky transform (shared between single/multi-LOD paths)
        is_diagonal = np.count_nonzero(A - np.diag(np.diagonal(A))) == 0
        if is_diagonal:
            diag = np.diagonal(A)
            if np.any(diag <= 0):
                raise ValueError(f"Diagonal scale factors must be positive, got {diag}")
            tril_scales = np.concatenate([[diag[i]] * (i + 1) for i in range(d)])

        def _transform_cholesky(chol: np.ndarray) -> np.ndarray:
            if is_diagonal:
                return np.asarray(chol * tril_scales.astype(chol.dtype))
            L = unpack_tril(chol.astype(np.float64), d)
            Sigma = L @ np.swapaxes(L, -2, -1)
            Sigma_new = A @ Sigma @ A.T
            L_new = np.linalg.cholesky(Sigma_new)
            return pack_tril(L_new).astype(chol.dtype)

        # Multi-LOD path: transform each LOD independently
        if self.n_additive_sublods > 1:

            def _transform_lod(
                lod: "AdditiveSubLOD", offset: int, n: int
            ) -> "AdditiveSubLOD":
                lod_centers = (lod.centers.astype(np.float64) @ A.T + t).astype(
                    lod.centers.dtype
                )
                return AdditiveSubLOD(
                    centers=lod_centers,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=_transform_cholesky(lod.cholesky_factors),
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )

            return self._map_additive(_transform_lod)

        # Single-LOD fast path
        new_centers = (self.centers.astype(np.float64) @ A.T + t).astype(
            self.centers.dtype
        )
        new_cholesky = _transform_cholesky(self.cholesky_factors)

        return GSplatData(
            centers=new_centers,
            amplitudes=self.amplitudes,
            cholesky_factors=new_cholesky,
            colors=self.colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
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
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        # Multi-substitutive: translate every level and rebuild the pyramid.
        if self.n_substitutive > 1:
            return self._map_substitutive(lambda lvl: lvl.translate(offset))

        # Multi-LOD path: translate each LOD independently
        if self.n_additive_sublods > 1:
            return self._map_additive(
                lambda lod, offset_, n: AdditiveSubLOD(
                    centers=lod.centers + offset,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod.cholesky_factors,
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )

        return GSplatData(
            centers=self.centers + offset,
            amplitudes=self.amplitudes,
            cholesky_factors=self.cholesky_factors,
            colors=self.colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
        )

    def center_at_centroid(self) -> "GSplatData":
        """Center the splats at their center of mass (amplitude-weighted centroid).

        The centroid is the amplitude-weighted average of splat centers (the
        center of mass of the represented density). Only the **spatial**
        (non-degenerate) axes are re-origined: a zero-variance categorical axis
        (a per-timepoint time axis, a channel axis) keeps its original
        coordinates, because centering it would push integer timepoints to
        fractional offsets and misalign the viewer's slice navigator. For pure
        spatial data (no degenerate axis) every axis is centered, as before.

        Returns:
            New GSplatData with its spatial centroid at the origin.

        Example:
            >>> # Center splats at origin for easier viewing
            >>> centered = data.center_at_centroid()
        """
        # Empty data: nothing to center. Return a structure-preserving copy
        # (translate by zero) rather than computing mean() of an empty array,
        # which would emit a spurious "Mean of empty slice" RuntimeWarning.
        if self.n_splats == 0:
            return self.translate(np.zeros(self.ndim, dtype=np.float64))

        # Compute amplitude-weighted centroid
        total_amplitude = self.amplitudes.sum()
        if total_amplitude > 0:
            centroid = (self.centers.T @ self.amplitudes) / total_amplitude
        else:
            centroid = self.centers.mean(axis=0)

        # Shift only the spatial (non-degenerate) axes; leave categorical axes
        # (zero covariance extent — e.g. a stacked-time axis) at their
        # coordinates. Mirrors scale()/eccentricities()/isolation grouping.
        shift = spatial_only_shift(centroid, self._nondegenerate_axes())
        return self.translate(-shift)
