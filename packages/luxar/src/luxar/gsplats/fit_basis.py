"""Putting a raw reference volume onto a fit's background-relative basis.

A fit reconstructs ``V - image_min``, never ``V``: amplitudes are rescaled back by
``intensity_range`` on the way out, but ``image_min`` — which absorbs any
subtracted background floor — is deliberately never added back (see
``fitting/results.py`` and ``docs/specs/GSPLATS_ZARR_FORMAT.md``). So a fitted
dataset and a raw volume live in two different bases, and every command that
hands one to the other has to convert or it is comparing different quantities.

That conversion is one clip-and-subtract, and it was previously open-coded
nowhere — which is why several call sites simply skipped it (#1173, #1177). It
lives here so the sites agree, and so the "the store does not say" case is
handled once instead of five times.

The functions are deliberately dumb: no I/O, no zarr, no torch. Callers supply
the stats mapping they already loaded.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "fit_image_min",
    "reference_in_amplitude_units",
    "reference_on_fit_basis",
    "rescale_intensity_basis",
    "source_intensity_span",
    "stored_intensity_scale",
    "MISSING_BASIS_HINT",
]

#: What to tell a user when the store predates the stat. Shared so `compare`,
#: `cull` and the LOD re-fit all say the same thing.
MISSING_BASIS_HINT = (
    "the dataset records no normalization basis (image_min), so the reference "
    "cannot be put on the fit's scale; scores/decisions below are computed "
    "against the RAW volume and will be penalised by any background the fit "
    "removed. Re-fit to record it, or pass the level explicitly."
)


def fit_image_min(stats: Optional[Mapping[str, Any]]) -> Optional[float]:
    """The level a fit subtracted, or ``None`` when the store does not record it.

    Prefers ``image_min`` over ``floor``, and the distinction matters: ``floor``
    is only the *requested* background suppression, while ``image_min`` is the
    level normalization actually subtracted — the two coincide when a floor is
    active and diverge when it is not (``image_min`` is then the data minimum, or
    a percentile endpoint). ``image_min`` is always the right quantity for
    reversing the shift; ``floor`` is a defensive fallback for incomplete or
    externally-authored metadata.

    Returns ``None`` rather than ``0.0`` for "not recorded", so callers can tell
    "this fit removed nothing" apart from "this fit did not say", and warn about
    the second. A non-finite or negative value is treated as not recorded: both
    would corrupt the shift, and neither is a level any fit legitimately applies.

    Note that ``0.0`` is a real answer, not a degenerate one: per the format spec
    a TILED fit pins every tile at ``image_min = 0`` on purpose, so that no
    second per-tile constant is subtracted twice across an overlap band.

    "Not recorded" is likewise ordinary rather than exceptional —
    ``agreed_normalization_stats`` drops any key the merged inputs disagree on,
    so a merge of independently fitted volumes says nothing here by design.

    The keys are a subset of ``NORMALIZATION_STATS_KEYS``
    (``gsplats/io/save_gsplats.py``), which is the canonical spelling; they are
    named literally rather than imported to keep this module free of the IO layer.
    """
    if not stats:
        return None
    for key in ("image_min", "floor"):
        value = stats.get(key)
        # `bool` before `float`: it is an `int` subclass, so `float(True)` is 1.0.
        # On a normalised store, where levels are ~0.04, a level of 1.0 clips the
        # ENTIRE reference to zero — a silently meaningless score rather than an
        # error. No fit writes a boolean here, so reject it as not-recorded.
        if value is None or isinstance(value, bool):
            continue
        try:
            level = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(level) or level < 0.0:
            continue
        return level
    return None


def reference_on_fit_basis(
    volume: np.ndarray, image_min: Optional[float], *, copy: bool = True
) -> np.ndarray:
    """Shift ``volume`` onto the basis a fit with this ``image_min`` reconstructs.

    The exact inverse of the shift ``_normalize_data`` applies, clipped at zero
    the same way. ``image_min`` of ``None`` or ``0.0`` skips the shift, so a
    caller can pass whatever it resolved without branching.

    Clipping is not cosmetic: sub-floor voxels were clipped to 0 going in, so
    leaving them negative here would score the fit for failing to reproduce
    values it was never shown.

    **Dtype:** a floating input keeps its own dtype; an INTEGER input becomes
    float32. Both matter. Subtracting a python float from an integer array
    promotes to float64 under NEP 50 — four times the memory of a ``uint16``
    reference, and a float64 result then makes ``compute_quality_metrics`` raise
    ``expected scalar type Double but found Float`` against a float32 render.
    Today every caller comes through ``load_volume``, which already returns
    float32, so this is a guard on the contract rather than a live bug — but the
    conversion is centralised here precisely so the next caller cannot trip it.

    **copy=False** shifts a writeable floating ``volume`` IN PLACE and returns
    it — for a caller that owns a freshly loaded reference and cannot afford the
    two full-size temporaries (difference, then clip) the copying path costs.
    """
    array = np.asarray(volume)
    if array.dtype.kind != "f":
        array = array.astype(np.float32)
    if image_min is None or image_min == 0.0:
        return array
    # Subtract in the array's OWN dtype so float32 stays float32 (a python float
    # is weak under NEP 50, but being explicit survives future promotion rules).
    # Annotated because `dtype.type(...)` is untyped, which would otherwise make
    # the whole expression — and this function's return — `Any`.
    level = array.dtype.type(image_min)
    if not copy and array.flags.writeable and array is volume:
        np.subtract(array, level, out=array)
        np.maximum(array, array.dtype.type(0.0), out=array)
        return array
    shifted: np.ndarray = np.clip(array - level, 0.0, None)
    return shifted


def rescale_intensity_basis(stats: Optional[dict], factor: Optional[float]) -> None:
    """Keep a store's normalisation basis truthful after an amplitude edit.

    ``factor`` is the exact multiplier applied to every amplitude
    (``transform --scale-intensity`` or ``--normalize-intensity``):
    ``intensity_range`` scales with it, so
    ``amplitude / intensity_range`` still recovers the fit's working value.
    ``image_min`` / ``image_max`` remain raw source levels; ``intensity_scale``
    records the cumulative multiplier so a scene cannot label edited amplitudes
    as source counts.
    ``None`` means the edit was not one global factor (an offset, a clamp, a
    per-splat reweight): the amplitude-unit keys are DROPPED, so a later
    ``gsplat convert`` labels the store ``a.u.`` instead of claiming counts.
    Mutates ``stats`` in place; a ``None``/empty mapping is a no-op.
    """
    if not stats:
        return
    if factor is None:
        stats.pop("intensity_range", None)
        stats.pop("image_max", None)
        stats.pop("intensity_scale", None)
        return
    if "intensity_range" in stats:
        try:
            stats["intensity_range"] = float(stats["intensity_range"]) * float(factor)
        except (TypeError, ValueError):
            stats.pop("intensity_range")
    scale = float(stats.get("intensity_scale", 1.0)) * float(factor)
    # A round trip (x49 then x1/49) lands within an ulp of 1, not on it; leaving
    # that residue would label a no-op store as edited.
    if math.isclose(scale, 1.0, rel_tol=1e-12):
        stats.pop("intensity_scale", None)
    else:
        stats["intensity_scale"] = scale


def stored_intensity_scale(stats: Optional[Mapping[str, Any]]) -> float:
    """The cumulative amplitude multiplier a store records, or ``1.0``.

    ``transform --scale-intensity k`` multiplies every amplitude by ``k`` and
    records the running product as ``intensity_scale``
    (:func:`rescale_intensity_basis`), while ``image_min`` stays in raw source
    units. A scorer that puts a raw reference on the fit's basis must therefore
    also divide the render by this factor (or multiply the reference by it), or
    a pure display rescale reads as reconstruction error. Absent, non-numeric,
    non-finite or non-positive values mean "no recorded edit": ``1.0``.
    """
    if not stats:
        return 1.0
    value = stats.get("intensity_scale", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def reference_in_amplitude_units(
    volume: np.ndarray, scale: float, *, in_place: bool = False
) -> np.ndarray:
    """``volume`` multiplied by a store's ``intensity_scale``, into its amplitude units.

    For a consumer that keeps the store's EDITED amplitudes (``cull --target``,
    the LOD volume re-fit) and so moves the reference instead of the splats.
    ``scale == 1.0`` returns ``volume`` untouched. An integer input becomes
    float32 (never float64); ``in_place=True`` multiplies a writeable floating
    ``volume`` in place, for a caller that owns a fresh copy.
    """
    if scale == 1.0:
        return volume
    array = np.asarray(volume)
    if array.dtype.kind != "f":
        promoted: np.ndarray = np.multiply(array, np.float32(scale), dtype=np.float32)
        return promoted
    factor = array.dtype.type(scale)
    if in_place and array.flags.writeable:
        np.multiply(array, factor, out=array)
        return array
    scaled: np.ndarray = np.multiply(array, factor)
    return scaled


def source_intensity_span(stats: Optional[Mapping[str, Any]]) -> Optional[float]:
    """The source's intensity span above the subtracted level, or ``None``.

    ``intensity_range`` (what the fit divided by, scaled with any amplitude edit;
    initially exactly ``image_max - image_min``), else ``image_max - image_min``.
    This is the extent of the data in amplitude units, unlike the per-splat
    peak window. A scaled store cannot fall back to raw source bounds.
    """
    if not stats:
        return None
    candidates = []
    rng = stats.get("intensity_range")
    if isinstance(rng, (int, float)) and not isinstance(rng, bool):
        candidates.append(float(rng))
    if stats.get("intensity_scale", 1.0) == 1.0:
        lo, hi = stats.get("image_min"), stats.get("image_max")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            candidates.append(float(hi) - float(lo))
    for span in candidates:
        if np.isfinite(span) and span > 0.0:
            return span
    return None
