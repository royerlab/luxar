"""Amplitude normalisation at scene insertion.

WHY THIS EXISTS — the rule, and the measurement behind it
=========================================================

A fitted ``.gsplats.zarr`` stores amplitudes in **raw source units**: the
fitter multiplies its ``[0, 1]`` working copy back out by the volume's own
intensity range (``gsplats/fitting/results.py``), so an archive fitted from a
uint16 detector stack carries amplitudes in detector counts — hundreds or
thousands.

**Those units cannot reach the screen unchanged, and no viewer control can
rescue them.** The stored amplitude drives two things in the shader, and the
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
white. The only remaining lever is ``opacity``, which then has to carry a
~1/500 factor on a ``[0, 1]`` control — unauthorable in the Layers panel, and
the reason so much shipped appearance is a magic tiny constant.

Hence: **normalise at insertion, not at display.** By the time a node is in a
scene its amplitudes should sit in ``[0, ~1]``, which puts ``opacity`` and
``absorption`` back in their natural ranges and makes appearance portable
between datasets.

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

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import GSplatNode

#: Upper reference percentile mapped to the normalisation target.
#: Kept equal to the percentile ``amplitude_data_range`` uses for its upper
#: bound so the LUT window and the normalisation cannot drift apart.
AMPLITUDE_REFERENCE_PERCENTILE = 99.9

#: How far above 1.0 the reference may sit and still count as "already in
#: range". Not cosmetic: amplitudes are stored float32 and the reference is an
#: interpolated percentile, so a set normalised to exactly 1.0 comes back a few
#: ULPs ABOVE it (measured: 1.0000000465). Without this, ``auto`` fires a second
#: time with a factor of ~1.0 — harmless numerically, but it stamps a
#: meaningless factor and makes the operation non-idempotent, so re-inserting an
#: already-normalised node would no longer be a no-op.
IN_RANGE_TOLERANCE = 1e-6

#: Node attr recording the factor applied, so an authored window, an
#: ``amplitude_mass`` stamp or a later refit can be reconciled with the values
#: actually stored. Absent when nothing was scaled.
NORMALIZATION_FACTOR_ATTR = "amplitude_normalization_factor"

#: Accepted spellings of the ``normalize_amplitudes`` argument.
#: ``True``/``"auto"`` -> scale to 1.0 only when the reference exceeds 1.0;
#: ``False``/``None`` -> leave amplitudes alone;
#: ``float`` -> scale the reference to exactly that value, unconditionally.
NormalizeSpec = Union[bool, str, float, None]


def _sample_reference(values: np.ndarray) -> float:
    """Pooled robust upper reference for a 1-D amplitude sample."""
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, AMPLITUDE_REFERENCE_PERCENTILE))


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

    if not (reference > 0.0) or not np.isfinite(reference):
        # All-zero or degenerate amplitudes: any factor is meaningless, and
        # scaling by one would only invalidate the fit's measured stamps.
        return None
    if automatic and reference <= 1.0 + IN_RANGE_TOLERANCE:
        return None  # already in range — see module docstring
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
