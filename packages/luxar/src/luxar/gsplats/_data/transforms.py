"""Geometric-transform mixin for ``GSplatData``.

Affine transforms, translation and centroid centering, plus the two
structure-preserving map helpers (``_map_substitutive`` / ``_map_additive``)
every per-level op in the sibling mixins rebuilds its ladder through.
"""

from __future__ import annotations

from dataclasses import replace
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


_PRUNED_LEVEL_STATS = ("reference_energy", "quality", "n_splats_total")


def _prune_empty_additive_sublods(
    lods: "List[AdditiveSubLOD]",
    *,
    refresh_structural: bool,
) -> "tuple[List[AdditiveSubLOD], list[int] | None]":
    """Drop zero-width rungs and refresh surviving structural stamps.

    A wholly empty leaf keeps its existing rung so the normal empty-dataset
    validation remains responsible for rejecting an unwritable result.  When
    at least one rung survives, empty rungs carry no prefix information and are
    removed while the remaining rungs keep their original order.  Structural
    stamps are refreshed when rung widths or ladder shape changed; pruning
    additionally drops the per-rung energy fraction because its prefix ladder
    changed shape.
    """
    nonempty = [lod for lod in lods if lod.n_splats > 0]
    if not nonempty:
        return lods, None

    pruned = len(nonempty) != len(lods)
    authored_rung_keys = {
        key
        for lod in nonempty
        for key in ("lod_level", "lod_n_splats", "lod_cumulative_n")
        if key in lod.stats
    }
    cumulative = 0
    cutpoints: list[int] = []
    refreshed: List["AdditiveSubLOD"] = []
    for level, lod in enumerate(nonempty):
        cumulative += int(lod.n_splats)
        cutpoints.append(cumulative)
        stats = dict(lod.stats)
        if pruned:
            stats.pop("energy_fraction_cum", None)
        if refresh_structural and "lod_level" in authored_rung_keys:
            stats["lod_level"] = level
        if refresh_structural and "lod_n_splats" in authored_rung_keys:
            stats["lod_n_splats"] = int(lod.n_splats)
        if refresh_structural and "lod_cumulative_n" in authored_rung_keys:
            stats["lod_cumulative_n"] = cumulative
        refreshed.append(replace(lod, stats=stats))
    return refreshed, cutpoints


def _refresh_ladder_summary(stats: dict, cutpoints: list[int]) -> dict:
    """Refresh authored ladder-size metadata after empty-rung pruning.

    BOTH spellings of the rung count, because both describe the ladder the
    rebuild just changed: ``lod_n_lods`` / ``lod_cutpoints`` (stamped by
    ``make_additive_lod``) and the un-prefixed ``n_lods``
    (:func:`~luxar.gsplats.batch.merge_orchestrator._recipe_pipeline_info`, what
    ``batch-fit merge --recipe stream`` publishes for the same ladder). A ``cull``
    / ``filter`` that empties a rung refreshed the first and left the second
    asserting the pre-prune count a few keys away (#1600 review). No ``lod_kind``
    gate is needed for the count the way :func:`~luxar.gsplats.lod.restamp
    ._recipe_ladder_keys` needs one for ``method``: that producer's ``stream``
    branch is the only site in the codebase that stamps ``n_lods`` at all.

    PRESENT KEYS ONLY — the refresh corrects a claim, it never starts making one.

    The un-prefixed count's two siblings are deliberately left alone:

    * ``method`` is the additive ORDERING, which dropping empty rungs does not
      change — the ``lod_*`` half leaves ``lod_method`` untouched here for
      exactly the same reason.
    * ``breakpoints`` is the build SPEC that was REQUESTED (``"stream:14000"``,
      ``"counts:5,15,40"``): provenance of how the ladder was built, not a
      measurement of what it now holds, and this rewrite built no new ladder
      from a different spec — it pruned the one that spec produced. (Contrast
      :func:`~luxar.gsplats.lod.restamp._refresh_recipe_ladder`, where a
      ``gsplat additive`` DOES replace the ladder with one built from other
      knobs, so the old spec is dropped.) Its prefixed twin
      ``lod_breakpoints_kind`` survives this same rewrite untouched.

    A single root count is always well defined here: both callers hold ONE
    ladder, and the ``per_part`` shape that would make it ambiguous is a
    ``kind=partition``, which ``GSplatData`` refuses to load at all.
    """
    refreshed = dict(stats)
    if "lod_n_lods" in refreshed:
        refreshed["lod_n_lods"] = len(cutpoints)
    if "lod_cutpoints" in refreshed:
        refreshed["lod_cutpoints"] = list(cutpoints)
    if "n_lods" in refreshed:
        refreshed["n_lods"] = len(cutpoints)
    return refreshed


def _stats_after_ladder_rebuild(
    stats: dict, cutpoints: list[int], *, count_changed: bool
) -> dict:
    """Refresh structural stamps and remove count-invalidated measurements."""
    refreshed = _refresh_ladder_summary(stats, cutpoints)
    if count_changed:
        for key in _PRUNED_LEVEL_STATS:
            refreshed.pop(key, None)
    return refreshed


class TransformsMixin(_GSplatDataOps):
    """``transform`` / ``translate`` / ``center_at_centroid`` and the per-level
    ``_map_substitutive`` / ``_map_additive`` rebuild helpers."""

    def _map_substitutive(
        self, fn: "Callable[[GSplatData], GSplatData]"
    ) -> "GSplatData":
        """Apply a single-level transform to EVERY substitutive level, rebuild.

        ``fn`` maps a single-substitutive-level view (``n_substitutive == 1``)
        to a transformed single-level ``GSplatData``; per-level ancestry is
        preserved while structural stats are refreshed and count-invalidated
        measurements are removed.
        Mirrors :meth:`filter_by`'s per-level rebuild so spatial
        and intensity ops never silently collapse the substitutive LOD ladder
        to the finest level. Callers guard with ``if self.n_substitutive > 1``.
        """
        from luxar.gsplats.gsplat_data import GSplatData, SubstitutiveLevel

        new_levels: List["SubstitutiveLevel"] = []
        any_count_changed = False
        for src in self.substitutive_levels:
            out = fn(self._view_of_level(src))
            out_level = out.substitutive_levels[0]
            count_changed = sum(
                lod.n_splats for lod in out_level.additive_sublods
            ) != sum(lod.n_splats for lod in src.additive_sublods)
            any_count_changed = any_count_changed or count_changed
            cutpoints = [
                int(c)
                for c in np.cumsum([lod.n_splats for lod in out_level.additive_sublods])
            ]
            new_levels.append(
                SubstitutiveLevel(
                    additive_sublods=out_level.additive_sublods,
                    compression_factor=src.compression_factor,
                    parent_method=src.parent_method,
                    level_index=src.level_index,
                    stats=_stats_after_ladder_rebuild(
                        src.stats,
                        cutpoints,
                        count_changed=count_changed,
                    ),
                )
            )
        summary_level = int(self.stats.get("lod_substitutive_level", 0))
        summary_level = min(max(summary_level, 0), len(new_levels) - 1)
        root_cutpoints = [
            int(c)
            for c in np.cumsum(
                [lod.n_splats for lod in new_levels[summary_level].additive_sublods]
            )
        ]
        return GSplatData.from_substitutive_levels(
            new_levels,
            stats=_stats_after_ladder_rebuild(
                self.stats,
                root_cutpoints,
                count_changed=any_count_changed,
            ),
        )

    def _map_additive(
        self, fn: "Callable[[AdditiveSubLOD, int, int], AdditiveSubLOD]"
    ) -> "GSplatData":
        """Apply a per-sub-LOD transform to EVERY additive sub-LOD, rebuild.

        ``fn`` receives ``(lod, offset, n)`` — the sub-LOD, its start offset
        into the flattened finest-leaf arrays, and its splat count — and returns
        a replacement :class:`AdditiveSubLOD` (which may change N, ndim, or
        array widths). Empty rungs are removed when another rung survives. The
        additive-dimension sibling of :meth:`_map_substitutive`.
        """
        from luxar.gsplats.gsplat_data import GSplatData, SubstitutiveLevel

        new_lods: List["AdditiveSubLOD"] = []
        offset = 0
        for lod in self.additive_sublods:
            n = lod.n_splats
            new_lods.append(fn(lod, offset, n))
            offset += n
        source_level = self.substitutive_levels[0]
        structure_changed = len(new_lods) != len(source_level.additive_sublods) or any(
            new_lod.n_splats != old_lod.n_splats
            for new_lod, old_lod in zip(new_lods, source_level.additive_sublods)
        )
        new_lods, cutpoints = _prune_empty_additive_sublods(
            new_lods,
            refresh_structural=structure_changed,
        )
        count_changed = sum(lod.n_splats for lod in new_lods) != sum(
            lod.n_splats for lod in source_level.additive_sublods
        )
        level_stats = dict(source_level.stats)
        root_stats = dict(self.stats)
        if cutpoints is not None:
            level_stats = _stats_after_ladder_rebuild(
                level_stats, cutpoints, count_changed=count_changed
            )
            root_stats = _stats_after_ladder_rebuild(
                root_stats, cutpoints, count_changed=count_changed
            )
        return GSplatData.from_substitutive_levels(
            [
                SubstitutiveLevel(
                    additive_sublods=new_lods,
                    compression_factor=source_level.compression_factor,
                    parent_method=source_level.parent_method,
                    level_index=source_level.level_index,
                    stats=level_stats,
                )
            ],
            stats=root_stats,
        )

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
        from luxar.gsplats.gsplat_data import AdditiveSubLOD
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
            return self._map_additive(
                lambda lod, offset, n: AdditiveSubLOD(
                    centers=lod.centers.copy(),
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod.cholesky_factors.copy(),
                    colors=lod.colors,
                    label_ids=lod.label_ids,
                    label_vocabulary=lod.label_vocabulary,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )

        # Precompute cholesky transform.
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
                label_ids=lod.label_ids,
                label_vocabulary=lod.label_vocabulary,
                stats=dict(lod.stats),
                truncation_radius=lod.truncation_radius,
            )

        return self._map_additive(_transform_lod)

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
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        # Multi-substitutive: translate every level and rebuild the pyramid.
        if self.n_substitutive > 1:
            return self._map_substitutive(lambda lvl: lvl.translate(offset))

        return self._map_additive(
            lambda lod, offset_, n: AdditiveSubLOD(
                centers=lod.centers + offset,
                amplitudes=lod.amplitudes,
                cholesky_factors=lod.cholesky_factors,
                colors=lod.colors,
                label_ids=lod.label_ids,
                label_vocabulary=lod.label_vocabulary,
                stats=dict(lod.stats),
                truncation_radius=lod.truncation_radius,
            )
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
