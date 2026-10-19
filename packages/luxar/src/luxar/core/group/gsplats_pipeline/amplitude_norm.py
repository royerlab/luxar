"""Amplitude normalisation at scene insertion.

WHY THIS EXISTS — the rule, and the measurement behind it
=========================================================

A fitted ``.gsplats.zarr`` stores amplitudes in **raw source units**: the
fitter multiplies its ``[0, 1]`` working copy back out by the volume's own
intensity range (``gsplats/fitting/results.py``), so an archive fitted from a
uint16 detector stack carries amplitudes in detector counts — hundreds or
thousands.

**Those units cannot reach the screen unscaled, and the colormap window cannot
scale them.** The stored amplitude drives two things in the shader, and the
display window only reaches one:

1. the **colormap LUT index** —
   ``t = clamp((A - uScalarMin) * uScalarScale, 0, 1)``. This is what
   ``amplitude_data_range`` (and the Layers panel window) feeds. It is clamped
   to ``[0, 1]`` and selects a *colour*; it can never scale brightness.
2. **emitted radiance and, under ``volumetric``, optical depth** —
   ``vAmplitude2D = aAmplitude * ...`` and
   ``tau = uAbsorption * uOpacity * intensity`` with ``intensity ∝
   vAmplitude2D``. Nothing windows these. They are **linear in the raw stored
   amplitude**.

So an amplitude of 800 emits 1000x the radiance of 0.8 and produces 1000x the
optical depth. Under ``volumetric`` that saturates ``1 - exp(-tau)`` to 1.0 and
the specimen becomes an opaque shell; under the additive modes it blows out to
white. ``opacity`` would have to carry a ~1/500 factor on a ``[0, 1]`` control;
the per-node ``display_gain`` render attr can carry it (unbounded — see "The
data-unit alternative" below, what ``gsplat convert`` uses). Leaving the
factor on ``opacity`` is unauthorable in the Layers panel, and the reason so
much shipped appearance is a magic tiny constant.

Hence, for the Python adders: **normalise at insertion** (or pass
``display_gain`` with ``normalize_amplitudes=False``). By the time a node is in
a scene its rendered amplitudes should sit in ``[0, ~1]``, which puts ``opacity`` and
``absorption`` back in their natural ranges and makes appearance portable
between datasets.

The data-unit alternative: ``display_gain``
-------------------------------------------

Normalising REWRITES the data, so the Layers panel then shows rescaled numbers
rather than the data's own units. ``luxar gsplat convert`` therefore no longer
normalises: it keeps the amplitudes in source data units and stamps the same
factor as a per-node ``display_gain`` attr instead, which the viewer multiplies
into exactly the two amplitude-linear terms above (emission and optical depth)
and NOT into the colormap lookup — so the render is the one normalising would
have produced (``LUXAR_ZARR_FORMAT.md``, "Amplitude units and display gain").
The Python ``add_gsplats_*`` default is unchanged (``normalize_amplitudes=True``);
pass ``normalize_amplitudes=False, display_gain=...`` to opt in.

Design decisions, and why
-------------------------

* **Robust reference, not the maximum.** The reference is the pooled 99.9th
  percentile, not ``max``. One hot splat — a cosmic ray, a saturated pixel, an
  outlier a refit happens to produce — would otherwise compress the whole scene
  toward black and change the factor on every refit. ~0.1% of splats are left
  above 1.0 for the authored display window to clip. This matches what
  ``amplitude_data_range`` already uses as its upper bound
  (``io/_compiler/gsplat_assembly.py``), so the window and the normalisation
  agree instead of drifting.
  (Contrast :meth:`~luxar.gsplats.gsplat_data.GSplatData.normalize_intensity`,
  which targets ``max`` and is therefore not portable across datasets.)

* **Pure scale, no offset.** Amplitudes are background-relative and
  non-negative; subtracting a floor here would make the dimmest splats
  invisible and would not commute with the mass-conserving coarsening the LOD
  builders rely on. A single multiplicative factor does.

* **ONE factor for the whole structure.** Every substitutive level and every
  additive rung is scaled by the same number. A per-level or per-rung factor
  would rescale the rungs against each other, so each streaming prefix and each
  LOD level would render at a different exposure — the same failure
  ``io/_compiler/finalize/amplitude_window.py`` exists to prevent for windows.
  The reference pools every spatial partition part, but follows only the finest
  child through each substitutive LOD group: coarse merged representatives carry
  combined mass and must not darken the finest view users inspect up close.
  Consequently, a child inserted directly into a ``kind=lod`` or
  ``kind=partition`` group defaults to no normalisation: scaling it independently
  would destroy the sibling-relative exposure that the enclosing structure owns.

* **``auto`` is a no-op on data that is already in range.** Many datasets are
  fitted from volumes already normalised to ``[0, 1]`` and then dimmed by a
  hand-tuned constant. Rescaling those would double-brighten them. So ``auto``
  only acts when the reference exceeds 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from luxar.gsplats.amplitude_reference import (
    AMPLITUDE_REFERENCE_PERCENTILE as AMPLITUDE_REFERENCE_PERCENTILE,
)
from luxar.gsplats.amplitude_reference import (
    IN_RANGE_TOLERANCE as IN_RANGE_TOLERANCE,
)
from luxar.gsplats.amplitude_reference import (
    automatic_factor,
)
from luxar.gsplats.amplitude_reference import (
    sample_reference as _sample_reference,
)
from luxar.gsplats.exposure_attrs import (
    NORMALIZATION_MODE_ATTR as NORMALIZATION_MODE_ATTR,
)
from luxar.gsplats.exposure_attrs import (
    PROJECTED_SUM_AXIS_ATTR as PROJECTED_SUM_AXIS_ATTR,
)
from luxar.gsplats.exposure_attrs import (
    PROJECTED_SUM_P999_ATTR as PROJECTED_SUM_P999_ATTR,
)
from luxar.gsplats.exposure_attrs import (
    PROJECTED_SUM_TARGET_ATTR as PROJECTED_SUM_TARGET_ATTR,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import GSplatNode

#: Node attr recording the factor applied, so an authored window, an
#: ``amplitude_mass`` stamp or a later refit can be reconciled with the values
#: actually stored. Absent when nothing was scaled.
NORMALIZATION_FACTOR_ATTR = "amplitude_normalization_factor"

#: Render attrs of the data-unit path (``LUXAR_ZARR_FORMAT.md``): the per-node
#: brightness factor the viewer applies to the stored amplitudes, the label of
#: the stored amplitudes' unit, and the level subtracted before fitting (in the
#: same unit; display-only). Unlike the provenance attrs above, the viewer READS
#: all three.
DISPLAY_GAIN_ATTR = "display_gain"
AMPLITUDE_UNIT_ATTR = "amplitude_unit"
AMPLITUDE_OFFSET_ATTR = "amplitude_offset"
#: ``[0, image_max - image_min]``: the SOURCE's intensity span above the offset,
#: in the amplitude unit. A reference the Layers panel shows beside the splat
#: window — a splat amplitude is one Gaussian's PEAK, and overlapping Gaussians
#: sum to the voxel value, so the peak window sits well below the data's range.
SOURCE_DATA_RANGE_ATTR = "source_data_range"


@dataclass(frozen=True)
class AmplitudeScale:
    """An EXACT raw-to-stored amplitude factor, applied unconditionally.

    Unlike a float ``normalize_amplitudes`` target (which divides by the
    measured reference, so the factor carries the percentile's rounding), this
    applies — and stamps as ``amplitude_normalization_factor`` — exactly
    ``factor``, so a reader can divide it back out bit-for-bit. A factor of 1.0
    is a no-op and stamps nothing.
    """

    factor: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.factor) and self.factor > 0.0):
            raise ValueError(
                f"AmplitudeScale factor must be positive and finite, got {self.factor!r}"
            )


#: Accepted spellings of the ``normalize_amplitudes`` argument.
#: ``True``/``"auto"`` -> scale to 1.0 only when the reference exceeds 1.0;
#: ``False``/``None`` -> leave amplitudes alone;
#: ``float`` -> scale the reference to exactly that value, unconditionally;
#: :class:`AmplitudeScale` -> multiply by exactly that factor.
NormalizeSpec = Union[bool, str, float, AmplitudeScale, None]


def pooled_amplitudes_from_node(node: "GSplatNode") -> np.ndarray:
    """Reference amplitudes for a node tree, pooled into one array.

    Pools every partition part so adjacent tiles share one exposure, but follows
    only the finest child of each substitutive LOD group. Coarse levels contain
    merged representatives with systematically larger amplitudes; including them
    would under-expose the finest/default view and make this path disagree with
    :func:`normalize_gsplat_data`. All additive sub-LODs of each selected finest
    leaf participate because they form one prefix-sum representation.
    """
    from luxar.gsplats.tree import iter_default_leaves

    chunks = [
        np.asarray(sub.amplitudes, dtype=np.float64)
        for leaf in iter_default_leaves(node)
        for sub in leaf.additive_sublods
    ]
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)


def resolve_factor(spec: NormalizeSpec, reference: float) -> Optional[float]:
    """Turn a ``normalize_amplitudes`` argument into a factor, or ``None``.

    ``None`` means "do not touch the amplitudes" — either because the caller
    opted out, or because ``auto`` found the data already in range, or because
    there is no usable reference (an empty or all-zero set).
    """
    if spec is None or spec is False:
        return None
    if isinstance(spec, AmplitudeScale):
        return None if spec.factor == 1.0 else float(spec.factor)

    if spec is True or (isinstance(spec, str) and spec.lower() == "auto"):
        target = 1.0
        automatic = True
    else:
        if isinstance(spec, bool):  # pragma: no cover - False handled above
            return None
        if isinstance(spec, str):
            raise ValueError(
                "normalize_amplitudes must be True/False/'auto' or a number, "
                f"got {spec!r}"
            )
        target = float(spec)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError(
                "normalize_amplitudes target must be a positive finite number, "
                f"got {target!r}"
            )
        automatic = False

    if automatic:
        return automatic_factor(reference)
    if not (reference > 0.0) or not np.isfinite(reference):
        # All-zero or degenerate amplitudes: any factor is meaningless, and
        # scaling by one would only invalidate the fit's measured stamps.
        return None
    return target / reference


def normalize_gsplat_data(
    result: "GSplatData", spec: NormalizeSpec
) -> Tuple["GSplatData", Optional[float]]:
    """Normalise a ``GSplatData``'s amplitudes; return it and the factor used.

    Rebuilds the ladder with one factor across every substitutive level, then
    drops top-level and per-rung reconstruction scores, rescales each level's
    quadratic ``reference_energy``, and drops its source-volume ``refine_stats``.
    Scale-invariant ``quality`` stays valid. This deliberately avoids the general
    intensity-edit restamp path: scene insertion is core API and must not import
    the SciPy-backed LOD builders. See #2229 and #2230 for that dependency boundary.
    """
    amps = np.asarray(result.amplitudes, dtype=np.float64).ravel()
    factor = resolve_factor(spec, _sample_reference(amps))
    if factor is None:
        return result, None

    from luxar.gsplats._data.filtering import scrub_measured_stats
    from luxar.gsplats.gsplat_data import AdditiveSubLOD

    def scale_level(level: "GSplatData") -> "GSplatData":
        return level._map_additive(
            lambda lod, _offset, _count: AdditiveSubLOD(
                centers=lod.centers,
                amplitudes=lod.amplitudes * factor,
                cholesky_factors=lod.cholesky_factors,
                colors=lod.colors,
                label_ids=lod.label_ids,
                label_vocabulary=lod.label_vocabulary,
                stats=dict(lod.stats),
                truncation_radius=lod.truncation_radius,
            )
        )

    scaled = (
        result._map_substitutive(scale_level)
        if result.n_substitutive > 1
        else scale_level(result)
    )
    scrub_measured_stats(scaled)
    from luxar.gsplats.tree import iter_leaves

    for leaf in iter_leaves(scaled.tree):
        _scale_energy_stats(leaf.meta.get("stats"), factor)
    return scaled, factor


def normalize_node_in_place(node: "GSplatNode", spec: NormalizeSpec) -> Optional[float]:
    """Normalise a node TREE's amplitudes in place; return the factor used.

    The in-place rewrite is deliberate. The additive sub-LOD containers are
    frozen, and rebuilding one through the ``additive_sublods=`` constructor
    drops the authored per-rung metadata the fast path carries straight off disk
    (``coverage_fraction``, the energy stamps the viewer's LOD upgrades read).
    The rewrite scales only amplitude values, rescales their energy stamps, and
    removes refinement stats that no longer describe the scaled data.

    This is the ``kind=partition`` / nested-tree counterpart of
    :func:`normalize_gsplat_data`. It exists because those trees never become a
    ``GSplatData`` — they are grafted node for node — and so had no
    normalisation path at all, which is exactly the shape of the largest and
    rawest datasets.
    """
    from luxar.gsplats.tree import iter_leaves

    pooled = pooled_amplitudes_from_node(node)
    factor = resolve_factor(spec, _sample_reference(pooled))
    if factor is None:
        return None

    scale = np.float32(factor)
    for leaf in iter_leaves(node):
        for sub in leaf.additive_sublods:
            amps = sub.amplitudes
            if not amps.flags.writeable:
                amps = np.array(amps, copy=True)
                object.__setattr__(sub, "amplitudes", amps)
            np.multiply(amps, scale, out=amps, casting="unsafe")
        _scale_energy_stats(leaf.meta.get("stats"), factor)
    return factor


def _scale_energy_stats(stats: object, factor: float) -> None:
    """Keep amplitude-dependent level stamps consistent after a global scale."""
    if not isinstance(stats, dict):
        return
    if "reference_energy" in stats:
        stats["reference_energy"] = float(stats["reference_energy"]) * factor**2
    stats.pop("refine_stats", None)


def stamp_factor(attrs: dict, factor: Optional[float]) -> None:
    """Record the applied factor on the node attrs, when there was one."""
    if factor is not None:
        attrs[NORMALIZATION_FACTOR_ATTR] = float(factor)
