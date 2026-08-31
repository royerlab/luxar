"""Intensity / color mixin for ``GSplatData``.

Amplitude (brightness) edits — affine, normalize, clamp, scale, per-splat
reweighting and the soft scale-band filter — plus color replacement. All are
ladder-preserving: a pyramid is rebuilt level by level, never collapsed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Sequence, cast

import numpy as np

from .base import _GSplatDataOps
from .filtering import _stats_after_content_change

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


def amplitudes_changed(before: np.ndarray, after: np.ndarray) -> bool:
    """Whether an amplitude edit actually moved anything.

    Rewriting amplitudes invalidates the inherited measured scores — PSNR/MSE
    are absolute-error metrics, so a global ``x0.5`` changes them outright — but
    several call sites pass through unchanged values on purpose
    (``normalize_intensity`` on an all-zero dataset scales by 1.0 solely to
    preserve the pyramid, ``soft_scale_filter`` with no cutoff), and those must
    keep the stamp like any other no-op rewrite.

    Public (re-exported from :mod:`luxar.gsplats.gsplat_data`) because the CLI's
    node-tree path needs the SAME predicate: it scrubs the root ``fitting/`` group
    itself, and gating that on the flag's mere presence made ``transform
    --scale-intensity 1.0`` destroy a partition's scores while the flat path kept
    them.
    """
    return before.shape != after.shape or not bool(np.array_equal(before, after))


class IntensityMixin(_GSplatDataOps):
    """Amplitude and color edits — ``scale_intensity`` and friends."""

    def _with_new_amplitudes(self, new_amplitudes: np.ndarray) -> "GSplatData":
        """Return a new GSplatData with replaced amplitudes, preserving LODs.

        The chokepoint for every amplitude edit on a single-substitutive view,
        so the measured reconstruction scores are dropped here (see
        :data:`~luxar.gsplats._data.filtering._CONTENT_SCOPED_STATS_KEYS`): a
        rescaled, clamped or soft-attenuated splat set renders different values
        than the one the fit scored.
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        changed = amplitudes_changed(self.amplitudes, new_amplitudes)

        return _stats_after_content_change(
            self._map_additive(
                lambda lod, offset, n: AdditiveSubLOD(
                    centers=lod.centers,
                    amplitudes=new_amplitudes[offset : offset + n],
                    cholesky_factors=lod.cholesky_factors,
                    colors=lod.colors,
                    label_ids=lod.label_ids,
                    label_vocabulary=lod.label_vocabulary,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            ),
            changed=changed,
            source=self,
        )

    def _map_amplitudes_per_level(
        self, fn: "Callable[[GSplatData], GSplatData]"
    ) -> "GSplatData":
        """``_map_substitutive`` for an amplitude edit — pyramid AND stats.

        The per-level recursion scrubs each level through
        :meth:`_with_new_amplitudes`, but ``_map_substitutive`` rebuilds the
        TOP-level stats from ``dict(self.stats)`` (the dict ``gsplat info``
        reads ``psnr_db`` from), so it needs the same pass. Mirrors what
        ``filter_by`` / ``cull`` do on their own multi-substitutive branch.
        """
        out = self._map_substitutive(fn)
        before_levels = self.substitutive_levels
        after_levels = out.substitutive_levels
        return _stats_after_content_change(
            out,
            changed=len(before_levels) != len(after_levels)
            or any(
                len(old.additive_sublods) != len(new.additive_sublods)
                or any(
                    amplitudes_changed(old_lod.amplitudes, new_lod.amplitudes)
                    for old_lod, new_lod in zip(
                        old.additive_sublods, new.additive_sublods
                    )
                )
                for old, new in zip(before_levels, after_levels)
            ),
            source=self,
        )

    def with_colors(self, colors: "np.ndarray | tuple[float, ...]") -> "GSplatData":
        """Return a new GSplatData with replaced colors, preserving LODs.

        Args:
            colors: Either an (N, 3) RGB / (N, 4) RGBA array of per-splat
                colors, or a single (r, g, b) / (r, g, b, a) tuple/array to
                broadcast to all splats. The alpha channel is per-splat
                opacity in [0, 1].

        Returns:
            New GSplatData with the specified colors.
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        arr = (
            colors
            if isinstance(colors, np.ndarray)
            else np.asarray(colors, dtype=np.float32)
        )
        is_broadcast = arr.ndim == 1 and arr.shape in ((3,), (4,))

        # Multi-substitutive: rebuild the pyramid. A single (r, g, b) broadcasts
        # cleanly to every level; an explicit per-splat array cannot (each level
        # has a different splat count), so reject it rather than silently
        # collapse the ladder to the finest level.
        if self.n_substitutive > 1:
            if not is_broadcast:
                raise ValueError(
                    "with_colors with an explicit per-splat color array is not "
                    "supported on a multi-substitutive pyramid (each level has a "
                    "different splat count). Pass a single (r, g, b) to broadcast "
                    "across all levels, or operate per level via at_substitutive()."
                )
            rgb = arr.astype(np.float32)
            return self._map_substitutive(lambda lvl: lvl.with_colors(rgb))

        if is_broadcast:
            # Broadcast single color to all splats
            colors = np.tile(arr.astype(np.float32), (self.n_splats, 1))
        else:
            colors = arr
        if colors.shape not in ((self.n_splats, 3), (self.n_splats, 4)):
            raise ValueError(
                f"colors shape {colors.shape} doesn't match "
                f"({self.n_splats}, 3) or ({self.n_splats}, 4)"
            )
        return self._map_additive(
            lambda lod, offset, n: AdditiveSubLOD(
                centers=lod.centers,
                amplitudes=lod.amplitudes,
                cholesky_factors=lod.cholesky_factors,
                colors=colors[offset : offset + n],
                label_ids=lod.label_ids,
                label_vocabulary=lod.label_vocabulary,
                stats=dict(lod.stats),
                truncation_radius=lod.truncation_radius,
            )
        )

    def with_label_ids(
        self,
        label_ids: "np.ndarray | Sequence[int]",
        label_vocabulary: "dict[int, str]",
    ) -> "GSplatData":
        """Attach exact categorical ids to a flat/additive gsplat dataset."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, validate_label_channel

        label_ids = np.asarray(label_ids)
        if self.n_substitutive > 1:
            raise ValueError(
                "with_label_ids is not supported on a multi-substitutive pyramid: "
                "coarse splats merge multiple fine class ids. Attach labels before "
                "building an additive ladder, not to substitutive levels."
            )
        vocabulary = validate_label_channel(label_ids, label_vocabulary, self.n_splats)
        return self._map_additive(
            lambda lod, offset, n: AdditiveSubLOD(
                centers=lod.centers,
                amplitudes=lod.amplitudes,
                cholesky_factors=lod.cholesky_factors,
                colors=lod.colors,
                label_ids=label_ids[offset : offset + n],
                label_vocabulary=vocabulary,
                stats=dict(lod.stats),
                truncation_radius=lod.truncation_radius,
            )
        )

    def without_label_ids(self) -> "GSplatData":
        """Remove categorical ids and vocabulary while preserving all LODs."""
        if self.n_substitutive > 1:
            return self._map_substitutive(lambda level: level.without_label_ids())
        return self._map_additive(
            lambda lod, _offset, _n: replace(lod, label_ids=None, label_vocabulary=None)
        )

    def affine_intensity(self, scale: float = 1.0, offset: float = 0.0) -> "GSplatData":
        """Apply affine transform to amplitudes: new_amp = scale * amp + offset.

        Args:
            scale: Multiplicative factor.
            offset: Additive offset.

        Returns:
            New GSplatData with transformed amplitudes.
        """
        if self.n_substitutive > 1:
            return self._map_amplitudes_per_level(
                lambda lvl: lvl.affine_intensity(scale, offset)
            )
        return self._with_new_amplitudes(self.amplitudes * scale + offset)

    def normalize_intensity(self, target_max: float = 1.0) -> "GSplatData":
        """Normalize amplitudes so the maximum equals target_max.

        Args:
            target_max: Desired maximum amplitude (default 1.0).

        Returns:
            New GSplatData. Returns copy if all amplitudes are zero.
        """
        # A single global factor (from the finest level's max) is applied
        # uniformly to all substitutive levels via scale_intensity — which is
        # itself pyramid-preserving — so the ladder is kept and levels stay
        # consistently scaled (a per-level normalization would shift them apart).
        current_max = float(self.amplitudes.max()) if self.n_splats > 0 else 0.0
        if current_max == 0:
            return self.scale_intensity(1.0)  # no-op, but preserves the pyramid
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
        if self.n_substitutive > 1:
            return self._map_amplitudes_per_level(
                lambda lvl: lvl.clamp_intensity(min, max)
            )
        new_amps = self.amplitudes.copy()
        if min is not None:
            new_amps = np.maximum(new_amps, min)
        if max is not None:
            new_amps = np.minimum(new_amps, max)
        return self._with_new_amplitudes(new_amps)

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
        if self.n_substitutive > 1:
            return self._map_amplitudes_per_level(
                lambda lvl: lvl.scale_intensity(factor)
            )
        return self._with_new_amplitudes(self.amplitudes * factor)

    def reweight_amplitude(self, multiplier: np.ndarray) -> "GSplatData":
        """Return a copy with per-splat amplitudes multiplied by ``multiplier``.

        The per-splat counterpart of ``scale_intensity`` (which is scalar-only).
        ``multiplier`` must be shape ``(n_splats,)`` and operates on this
        (matrix / default-level) view; it preserves the additive ladder. A
        global multiplier is not meaningful across substitutive levels — callers
        with a pyramid should reweight per-level (see ``soft_scale_filter``).
        """
        multiplier = np.asarray(multiplier, dtype=np.float64)
        if multiplier.shape != (self.n_splats,):
            raise ValueError(
                f"multiplier shape {multiplier.shape} != ({self.n_splats},)"
            )
        return self._with_new_amplitudes(self.amplitudes * multiplier)

    def soft_scale_filter(
        self,
        *,
        highpass: float | None = None,
        lowpass: float | None = None,
        width: float = 1.0,
        spatial_dims: Sequence[int] | None = None,
    ) -> "GSplatData":
        """Soft "frequency" filter: attenuate amplitude by a smooth function of
        each splat's characteristic ``scale()`` — a gentler alternative to a hard
        scale cut (no popping, splat count unchanged).

        - ``highpass``: suppress splats with scale ABOVE the cutoff (removes
          large diffuse / low-frequency background). Multiplier → 0 for very
          large scales, → 1 for small.
        - ``lowpass``: suppress splats with scale BELOW the cutoff (removes fine
          detail / high-frequency). Multiplier → 0 for very small scales, → 1
          for large.

        Both may be combined (a band-pass). ``width`` is the transition softness
        in octaves (log2 scale); larger = gentler roll-off.

        The cutoff is in the same world units as ``scale()``.
        """
        if self.n_splats == 0 or (highpass is None and lowpass is None):
            return cast("GSplatData", self)
        if self.n_substitutive > 1:
            # Reweight each substitutive level against its OWN scale distribution.
            return self._map_amplitudes_per_level(
                lambda lvl: lvl.soft_scale_filter(
                    highpass=highpass,
                    lowpass=lowpass,
                    width=width,
                    spatial_dims=spatial_dims,
                )
            )
        scl = np.clip(self.scale(axes=spatial_dims), 1e-12, None)
        w = max(float(width), 1e-6)
        mult = np.ones(self.n_splats, dtype=np.float64)
        # Smoothstep in log2(scale) space, spanning ±width octaves about cutoff.

        def _smoothstep(t: np.ndarray) -> np.ndarray:
            t = np.clip(t, 0.0, 1.0)
            out: np.ndarray = t * t * (3.0 - 2.0 * t)
            return out

        if highpass is not None:
            # 1 (keep) for scale <= cutoff, ramping to 0 above.
            t = (np.log2(scl) - np.log2(float(highpass))) / w + 0.5
            mult *= 1.0 - _smoothstep(t)
        if lowpass is not None:
            # 1 (keep) for scale >= cutoff, ramping to 0 below.
            t = (np.log2(scl) - np.log2(float(lowpass))) / w + 0.5
            mult *= _smoothstep(t)
        return self.reweight_amplitude(mult)
