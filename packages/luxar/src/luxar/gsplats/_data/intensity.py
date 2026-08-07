"""Intensity / color mixin for ``GSplatData``.

Amplitude (brightness) edits — affine, normalize, clamp, scale, per-splat
reweighting and the soft scale-band filter — plus color replacement. All are
ladder-preserving: a pyramid is rebuilt level by level, never collapsed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, cast

import numpy as np

from .base import _GSplatDataOps

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


class IntensityMixin(_GSplatDataOps):
    """Amplitude and color edits — ``scale_intensity`` and friends."""

    def _with_new_amplitudes(self, new_amplitudes: np.ndarray) -> "GSplatData":
        """Return a new GSplatData with replaced amplitudes, preserving LODs."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        if self.n_additive_sublods > 1:
            return self._map_additive(
                lambda lod, offset, n: AdditiveSubLOD(
                    centers=lod.centers,
                    amplitudes=new_amplitudes[offset : offset + n],
                    cholesky_factors=lod.cholesky_factors,
                    colors=lod.colors,
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )
        return GSplatData(
            centers=self.centers,
            amplitudes=new_amplitudes,
            cholesky_factors=self.cholesky_factors,
            colors=self.colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
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
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

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
        if self.n_additive_sublods > 1:
            return self._map_additive(
                lambda lod, offset, n: AdditiveSubLOD(
                    centers=lod.centers,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod.cholesky_factors,
                    colors=colors[offset : offset + n],
                    stats=dict(lod.stats),
                    truncation_radius=lod.truncation_radius,
                )
            )
        return GSplatData(
            centers=self.centers,
            amplitudes=self.amplitudes,
            cholesky_factors=self.cholesky_factors,
            colors=colors,
            stats=dict(self.stats),
            truncation_radius=self.truncation_radius,
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
            return self._map_substitutive(
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
            return self._map_substitutive(lambda lvl: lvl.clamp_intensity(min, max))
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
            return self._map_substitutive(lambda lvl: lvl.scale_intensity(factor))
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
            return self._map_substitutive(
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
