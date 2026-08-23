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

from typing import Any, Mapping, Optional

import numpy as np

__all__ = ["fit_image_min", "reference_on_fit_basis", "MISSING_BASIS_HINT"]

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
    volume: np.ndarray, image_min: Optional[float]
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
    shifted: np.ndarray = np.clip(array - array.dtype.type(image_min), 0.0, None)
    return shifted
