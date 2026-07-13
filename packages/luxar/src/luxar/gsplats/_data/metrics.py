"""Shared computed metrics mixin for splat array containers.

``_SplatArrayMixin`` is inherited by BOTH ``AdditiveSubLOD`` and ``GSplatData``
(via ``_GSplatDataOps``). It reads only ``centers``/``amplitudes``/
``cholesky_factors`` plus ``ndim``/``n_splats``/``truncation_radius`` and is
otherwise self-contained (all heavy imports are lazy in-method, keeping module
load torch-free)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from luxar.gsplats.utils.spatial_axes import (
    SPATIAL_SIGMA_EPS,
    spatial_axes_from_max_sigma,
)


class _SplatArrayMixin:
    """Shared computed properties for splat array containers.

    Requires the implementing class to have:
    - ``centers``: np.ndarray of shape (N, d)
    - ``amplitudes``: np.ndarray of shape (N,)
    - ``cholesky_factors``: np.ndarray of shape (N, d*(d+1)//2)
    """

    centers: np.ndarray
    amplitudes: np.ndarray
    cholesky_factors: np.ndarray

    @property
    def n_splats(self) -> int:
        """Number of splats."""
        return int(self.centers.shape[0])

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return int(self.centers.shape[1]) if self.centers.ndim >= 2 else 0

    def __len__(self) -> int:
        """Return number of splats."""
        return self.n_splats

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
        result: np.ndarray = np.abs(det_L**2) ** (1.0 / self.ndim)
        return result

    def masses(self) -> np.ndarray:
        """Per-splat mass: amplitude * volume.

        Returns:
            shape (N,) float array.
        """
        result: np.ndarray = self.amplitudes * self.volumes()
        return result

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
        result: np.ndarray = np.sqrt(np.sum(L**2, axis=2))
        return result

    def _nondegenerate_axes(self, eps: float = SPATIAL_SIGMA_EPS) -> np.ndarray:
        """Axes that carry real extent (max marginal sigma across splats > eps).

        A per-timepoint categorical / time axis (built with ``sigma=0``) has
        ~zero variance for *every* splat and is dropped, so scale / eccentricity
        become spatial-by-default on nD timelapses. Falls back to all axes if
        that would leave nothing (e.g. all-degenerate or empty data).
        """
        if self.n_splats == 0:
            return np.arange(self.ndim)
        return spatial_axes_from_max_sigma(self.marginal_sigmas().max(axis=0), eps)

    def _resolve_axes(self, axes: Optional[Sequence[int]]) -> np.ndarray:
        """Normalise an ``axes`` argument: ``None`` → auto non-degenerate axes."""
        if axes is None:
            return self._nondegenerate_axes()
        return np.asarray(list(axes), dtype=int)

    def scale(self, axes: Optional[Sequence[int]] = None) -> np.ndarray:
        """Per-splat characteristic size (world units): geometric mean of the
        marginal sigmas over ``axes``.

        Unlike ``volumes()`` (``det(Σ)^(1/d)`` over ALL dims, which collapses on
        a zero-variance time axis), ``scale`` defaults to the auto-detected
        non-degenerate (spatial) axes, so it is the meaningful "size" metric for
        nD timelapses. Large scale = diffuse / low-frequency (background).

        Returns:
            shape (N,) float array.
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.float64)
        ax = self._resolve_axes(axes)
        sig = np.clip(self.marginal_sigmas()[:, ax], 1e-12, None)
        result: np.ndarray = np.exp(np.mean(np.log(sig), axis=1))
        return result

    def eccentricities(self, axes: Optional[Sequence[int]] = None) -> np.ndarray:
        """Per-splat eccentricity: max marginal sigma / min marginal sigma.

        1.0 = isotropic. Higher values = more elongated. By default the ratio is
        taken over the auto-detected non-degenerate (spatial) axes — for pure 3D
        data this is all axes (unchanged), but on a timelapse it ignores the
        ~zero-variance time axis (which would otherwise force the degenerate
        1.0 fallback for every splat).

        Returns:
            shape (N,) float array. Returns 1.0 for degenerate splats.
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.float64)
        ax = self._resolve_axes(axes)
        sigmas = self.marginal_sigmas()[:, ax]
        min_s = sigmas.min(axis=1)
        max_s = sigmas.max(axis=1)
        result = np.ones(self.n_splats, dtype=np.float64)
        nonzero = min_s > 0
        result[nonzero] = max_s[nonzero] / min_s[nonzero]
        return result

    def principal_radii(self, anisotropy: bool = True) -> np.ndarray:
        """Per-splat element radius (world units) at the truncation boundary.

        Used by ``gsplat filter`` (eccentricity / volume). The Gaussian is
        truncated at ``truncation_radius`` sigmas, so the radius is
        ``truncation_radius * semi_axis``.

        - ``anisotropy=True`` → the largest principal semi-axis
          ``sqrt(lambda_max(Sigma))`` (worst-case projected radius;
          orientation-independent — the splat's biggest reach in any direction).
        - ``anisotropy=False`` → the isotropic-equivalent geometric-mean semi-axis
          ``det(Sigma)^(1/2d)`` (== ``sqrt(volumes())``).

        Returns:
            shape (N,) float array.
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.float64)
        trunc = float(getattr(self, "truncation_radius", 3.0))
        if anisotropy:
            from luxar.gsplats.utils.trils import unpack_tril

            chol = self.cholesky_factors.astype(np.float64)
            ell = unpack_tril(chol, self.ndim)
            sigma = ell @ np.swapaxes(ell, -2, -1)
            # eigvalsh returns ascending eigenvalues; the last is lambda_max.
            lam_max = np.linalg.eigvalsh(sigma)[:, -1]
            semi = np.sqrt(np.clip(lam_max, 0.0, None))
        else:
            semi = np.sqrt(self.volumes())
        result: np.ndarray = trunc * semi
        return result

    def _grouped_spatial(
        self,
        spatial_axes: Optional[Sequence[int]],
        group_axes: Optional[Sequence[int]],
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Split centers into spatial coords + integer group ids.

        Neighbour queries run *within* a group so splats at different
        timepoints/channels are never neighbours. By default ``spatial_axes`` =
        auto non-degenerate axes and ``group_axes`` = the complement (the
        near-constant categorical/time axes). Returns
        ``(spatial (N, ds) float64, group_ids (N,) int64)``.
        """
        ax = self._resolve_axes(spatial_axes)
        if group_axes is None:
            grp = np.array(
                [d for d in range(self.ndim) if d not in set(ax.tolist())], dtype=int
            )
        else:
            grp = np.asarray(list(group_axes), dtype=int)
        spatial = self.centers[:, ax].astype(np.float64)
        if grp.size == 0:
            group_ids = np.zeros(self.n_splats, dtype=np.int64)
        else:
            # Round categorical coords to collapse float noise, then map unique
            # rows → contiguous ids.
            keys = np.round(self.centers[:, grp].astype(np.float64), 6)
            _, group_ids = np.unique(keys, axis=0, return_inverse=True)
        return spatial, group_ids.astype(np.int64)

    def _cell_size_hint(self, spatial: np.ndarray) -> float:
        """A ~1-point-per-cell grid cell size heuristic for the spatial coords."""
        n, ds = spatial.shape
        if n <= 1:
            return 1.0
        extent = float(np.max(spatial.max(axis=0) - spatial.min(axis=0)))
        if extent <= 0:
            return 1.0
        return float(max(extent / max(1.0, n ** (1.0 / max(ds, 1))), 1e-6))

    def nearest_neighbor_distances(
        self,
        spatial_axes: Optional[Sequence[int]] = None,
        group_axes: Optional[Sequence[int]] = None,
        k: int = 1,
    ) -> np.ndarray:
        """Distance from each splat to its ``k``-th nearest neighbour.

        Computed over the spatial axes and grouped by the non-spatial axes (so a
        timelapse's timepoints never count as neighbours). Large distance =
        spatially isolated (a noise-splat signature). Returns shape ``(N,)``;
        ``+inf`` where a group has ``<= k`` splats (no neighbour exists).
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.float64)
        from luxar.utils.spatial_hash import BatchedSpatialHashGrid

        spatial, group_ids = self._grouped_spatial(spatial_axes, group_axes)
        out = np.full(self.n_splats, np.inf, dtype=np.float64)
        for gid in np.unique(group_ids):
            idx = np.flatnonzero(group_ids == gid)
            if idx.size <= k:
                continue  # no k-th neighbour in this group → stays +inf
            pts = spatial[idx]
            grid = BatchedSpatialHashGrid.from_points(
                pts, cell_size=self._cell_size_hint(pts), device="auto"
            )
            dists, _ = grid.query_knn(pts, k=k + 1)  # column 0 is self
            out[idx] = dists[:, k]
        return out

    def neighbor_counts(
        self,
        radius: float,
        spatial_axes: Optional[Sequence[int]] = None,
        group_axes: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Number of OTHER splats within ``radius`` (Euclidean, spatial axes),
        grouped by the non-spatial axes. Returns shape ``(N,)`` int64. Low count
        = spatially isolated.
        """
        if self.n_splats == 0:
            return np.empty(0, dtype=np.int64)
        from luxar.utils.spatial_hash import BatchedSpatialHashGrid

        spatial, group_ids = self._grouped_spatial(spatial_axes, group_axes)
        out = np.zeros(self.n_splats, dtype=np.int64)
        for gid in np.unique(group_ids):
            idx = np.flatnonzero(group_ids == gid)
            if idx.size == 0:
                continue
            pts = spatial[idx]
            # query_radius requires radius <= cell_size.
            grid = BatchedSpatialHashGrid.from_points(
                pts, cell_size=float(radius), device="auto"
            )
            neigh = grid.query_radius(pts, radius=float(radius))
            # each list entry includes self → subtract 1.
            out[idx] = np.array([max(len(n) - 1, 0) for n in neigh], dtype=np.int64)
        return out
