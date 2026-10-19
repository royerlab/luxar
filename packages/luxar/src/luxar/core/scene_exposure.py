"""Opening-view exposure for ``luxar gsplat convert`` (auto exposure).

The scene keeps the fit's amplitudes in SOURCE DATA UNITS — a fitted
``.gsplats.zarr`` already stores them that way (``fitting/results.py`` multiplies
the ``[0, 1]`` working copy back out by ``intensity_range``), so the Layers panel
can read in the data's own units. Brightness is a separate per-node ``display_gain``
(``LUXAR_ZARR_FORMAT.md``, "Amplitude units and display gain"), which the viewer
multiplies into every amplitude-linear term of the draw but NOT into the
colormap lookup — exactly what pre-multiplying the stored amplitudes used to do,
without rewriting the data.

Per-splat normalisation maps the p99.9 SPLAT to 1.0, but additive blending sums
every splat along the view ray, so a dense fit still opens as a saturated white
blob. Auto exposure (the convert default) predicts the opening view's per-pixel
sum with :func:`luxar.gsplats.exposure.plan_projected_exposure` and dims — never
brightens — so its p99.9 lands at ``--target-brightness`` (default by tone
mapping: 2.0 filmic, 0.9 clipping), unless a side view (orbit 90°) or another
timepoint would then exceed ``SIDE_CAP`` (4) times that, in which case those set
the factor. The gain is ``per-splat normalisation × that factor``, stamped with
``amplitude_normalization_mode`` and the prediction beside it.

``--no-auto-exposure`` stamps the per-splat normalisation alone as the gain (the
legacy look, attr for attr in the render); ``--scale-intensity S`` makes the gain
exactly ``S``. The ``.gsplats.zarr`` itself is never touched: ``gsplat render`` /
``compare`` / ``export`` read it, not the scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..gsplats.amplitude_reference import sample_reference as _sample_reference
from .group.gsplats_pipeline.amplitude_norm import (
    NORMALIZATION_MODE_ATTR,
    PROJECTED_SUM_AXIS_ATTR,
    PROJECTED_SUM_P999_ATTR,
    PROJECTED_SUM_TARGET_ATTR,
    NormalizeSpec,
    resolve_factor,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..gsplats.exposure import ExposurePlan, HiddenAxis
    from ..gsplats.tree import GSplatNode
    from .dimensions import Dimensions
    from .scene_orientation import ConvertOrientation
    from .viewer_config import CameraConfig

__all__ = [
    "ConvertExposure",
    "auto_exposure_attrs",
    "predict_volumetric_ray_mass",
    "resolve_convert_exposure",
    "volumetric_depth_line",
    "volumetric_gain_line",
]


@dataclass(frozen=True)
class ConvertExposure:
    """What convert hands ``add_gsplats_*`` for the amplitudes.

    Attributes:
        gain: The ``display_gain`` (brightness relative to data units).
        attrs: ``display_gain``, the unit labels and the provenance attrs to
            stamp on the gsplats node.
        line: One narration line, or ``None``.
        normalize: The ``normalize_amplitudes`` argument — always ``False``:
            the amplitudes stay in data units.
    """

    gain: float
    attrs: Dict[str, Any] = field(default_factory=dict)
    line: Optional[str] = None
    normalize: NormalizeSpec = False


def opening_view_columns(
    orientation: "ConvertOrientation",
) -> Tuple[List[int], Optional[int], Optional[str]]:
    """:func:`luxar.gsplats.exposure.opening_view_columns` for convert's layout."""
    from ..gsplats.exposure import opening_view_columns as _columns

    return _columns(orientation.dimensions, orientation.dim_order, orientation.camera)


def opening_hidden_axes(
    orientation: "ConvertOrientation", centers: Any
) -> List["HiddenAxis"]:
    """:func:`luxar.gsplats.exposure.opening_hidden_axes` for convert's layout."""
    from ..gsplats.exposure import opening_hidden_axes as _hidden

    return _hidden(orientation.dimensions, centers, orientation.dim_order)


def _gain_attrs(
    gain: float, mode: str, stats: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """The public :func:`~luxar.gsplats.exposure.data_unit_display_attrs` set
    (gain + unit labels — what a Python caller gets too) + the provenance mode."""
    from ..gsplats.exposure import data_unit_display_attrs

    return {**data_unit_display_attrs(stats, gain=gain), NORMALIZATION_MODE_ATTR: mode}


def _exposure(
    gain: float,
    mode: str,
    units: Optional[Mapping[str, Any]],
    line: Optional[str] = None,
    **extra: Any,
) -> ConvertExposure:
    """A :class:`ConvertExposure` carrying ``gain`` (and its stamp attrs);
    ``units`` is the store's root stats, which decide the unit labels."""
    attrs = _gain_attrs(gain, mode, units)
    attrs.update(extra)
    return ConvertExposure(gain=float(gain), attrs=attrs, line=line)


def _manual(
    scale_intensity: float, units: Optional[Mapping[str, Any]]
) -> ConvertExposure:
    """``--scale-intensity``: gain exactly ``S``, recorded as mode ``manual``."""
    return _exposure(
        scale_intensity,
        "manual",
        units,
        f"Exposure: manual gain ×{scale_intensity:g} (--display-gain); "
        "auto exposure off",
    )


#: Opening-view projected p99.9, in per-splat-normalised units (the brightest
#: splat peaks at ~1), above which the brightest rays stack so many splats that
#: additive blending shows an integrated haze rather than interior structure.
#: Measured on real fits: sparse cells/nuclei (A549, Allen DNA at 128k splats,
#: h2afva) sit at 0.6-10; dense nuclear stains (cells3d nuclei, a full Allen
#: FOV) at 28-31, the volumes testers described as a featureless glow.
DENSE_PROJECTED_SUM = 20.0


def dense_view_hint(predicted_before: float) -> Optional[str]:
    """One line pointing a dense volume at volumetric blending, else ``None``."""
    if predicted_before < DENSE_PROJECTED_SUM:
        return None
    return (
        f"Dense along the view: the brightest rays stack ~{predicted_before:.0f} "
        "splat peaks, so additive blending shows an integrated glow rather than "
        "interior detail. For interior structure try --blending-mode volumetric "
        "(convert then prints a starting --absorption)."
    )


def _exposure_line(plan: "ExposurePlan", per_splat: float = 1.0) -> str:
    """The one narration line, naming the STAMPED display gain (``per_splat ×
    plan.factor``) and the side-view cap only when it binds."""
    gain = f"display gain {per_splat * plan.factor:.2g}"
    dimming = f"×{plan.factor:.2g} below per-splat normalisation"
    if plan.predicted_before == 0:
        if plan.factor >= 1.0:
            return (
                "Exposure: nothing visible at the opening position; "
                f"{gain} (per-splat normalisation)"
            )
        return (
            "Exposure: nothing visible at the opening position; "
            f"{gain} ({dimming}), capped by {plan.cap_source}"
        )
    if plan.factor >= 1.0:
        return (
            f"Exposure: default view already within range (projected p99.9 "
            f"{plan.predicted_before:.2g} ≤ {plan.target:g}); {gain} "
            "(per-splat normalisation)"
        )
    reason = "the default view isn't saturated"
    detail = f"projected p99.9 {plan.predicted_before:.2g} → {plan.predicted_after:.2g}"
    if plan.side_capped and plan.side_p999 is not None:
        reason += f", capped by {plan.cap_source}"
        detail += f", {plan.cap_source} {plan.side_p999 * plan.factor:.2g}"
    return (
        f"Exposure: {gain} ({dimming}) so {reason} "
        f"({detail}; --no-auto-exposure to disable)"
    )


def resolve_convert_exposure(
    node: "GSplatNode",
    orientation: "ConvertOrientation",
    *,
    auto_exposure: bool,
    target: Optional[float],
    scale_intensity: Optional[float],
    colormap: Optional[str],
    opacity: float,
    blending_mode: str,
    tone_mapping: Optional[str] = None,
    stats: Optional[Mapping[str, Any]] = None,
) -> ConvertExposure:
    """Decide the ``display_gain`` convert stamps (see the module docstring).

    ``target=None`` picks the tone-mapping default
    (:func:`~luxar.gsplats.exposure.default_target_brightness`). ``stats`` is the
    store's root stats, which decide the unit labels.
    """
    from ..gsplats.exposure import (
        SUMMING_BLENDING_MODES,
        collect_default_splats,
        default_target_brightness,
        opening_truncation_radius,
        plan_projected_exposure,
        stamped_amplitude_window,
    )

    units = stats or {}
    if scale_intensity is not None:
        return _manual(scale_intensity, units)
    centers, chol, amps, colors = collect_default_splats(node)
    # The legacy per-splat normalisation, as a gain: p99.9 splat → 1.0 when
    # the data sit above 1.0, else 1 (``normalize_amplitudes=True``'s rule).
    per_splat = resolve_factor(True, _sample_reference(amps)) or 1.0
    if not auto_exposure:
        return _exposure(per_splat, "per_splat", units)
    if blending_mode not in SUMMING_BLENDING_MODES:
        return _exposure(
            per_splat,
            "per_splat",
            units,
            f"Exposure: auto exposure applies to additive/luminous blending; "
            f"'{blending_mode}' keeps per-splat normalization",
        )
    if target is None:
        target = default_target_brightness(tone_mapping)
    columns, axis_col, axis_name = opening_view_columns(orientation)
    try:
        plan = plan_projected_exposure(
            centers,
            chol,
            amps * per_splat,
            colors,
            view_axis=axis_col,
            display_dims=columns,
            hidden=opening_hidden_axes(orientation, centers),
            colormap=colormap,
            opacity=opacity,
            data_range=stamped_amplitude_window(node, per_splat),
            target=target,
            trunc=opening_truncation_radius(node),
        )
    except ValueError as exc:
        return _exposure(
            per_splat,
            "per_splat",
            units,
            f"Exposure: could not predict the opening view ({exc}); "
            "using per-splat normalization",
        )
    extra: Dict[str, Any] = {
        PROJECTED_SUM_P999_ATTR: float(plan.predicted_after),
        PROJECTED_SUM_TARGET_ATTR: float(target),
    }
    if axis_name is not None:
        extra[PROJECTED_SUM_AXIS_ATTR] = axis_name
    exposure = _exposure(
        per_splat * plan.factor,
        "projected",
        units,
        _exposure_line(plan, per_splat),
        **extra,
    )
    hint = dense_view_hint(plan.predicted_before)
    if hint is not None and exposure.line:
        exposure = replace(exposure, line=f"{exposure.line}\n{hint}")
    return exposure


def predict_volumetric_ray_mass(
    node: "GSplatNode",
    orientation: "ConvertOrientation",
    *,
    gain: float,
    opacity: float,
) -> Optional[float]:
    """Opening-view p99.9 of the per-ray density sum ``volumetric`` absorbs by.

    The shader's optical depth is ``tau = absorption * opacity * intensity``,
    where ``intensity`` is the sum-projected ray mass of the GAINED amplitudes —
    the additive render's sum before any colour or colormap gain. That is the
    projected-sum predictor with every splat's colour gain set to 1, so the
    returned value ``S`` gives ``tau ~= absorption * S`` on the brightest rays.
    ``None`` when nothing can be predicted (empty node, unsupported dims).
    """

    from ..gsplats.exposure import (
        collect_default_splats,
        opening_truncation_radius,
        plan_projected_exposure,
    )

    centers, chol, amps, _colors = collect_default_splats(node)
    if np.asarray(amps).size == 0:
        return None
    columns, axis_col, _axis_name = opening_view_columns(orientation)
    try:
        plan = plan_projected_exposure(
            centers,
            chol,
            np.asarray(amps, dtype=np.float64) * gain,
            np.ones((np.asarray(amps).size, 3)),
            view_axis=axis_col,
            display_dims=columns,
            hidden=opening_hidden_axes(orientation, centers),
            opacity=opacity,
            trunc=opening_truncation_radius(node),
        )
    except ValueError:
        return None
    return float(plan.predicted_before) if plan.predicted_before > 0 else None


def volumetric_depth_line(ray_mass: float, absorption: float) -> str:
    """Narrate the brightest rays' optical depth and the κ for ~one depth there."""
    import math

    tau = absorption * ray_mass
    return (
        f"Volumetric: brightest rays (opening-view p99.9) carry density "
        f"{ray_mass:.3g}, so --absorption {absorption:g} gives optical depth "
        f"~{tau:.3g} there ({100 * math.exp(-tau):.0f}% transmitted); "
        f"--absorption {1.0 / ray_mass:.2g} would give ~1 (37%)"
    )


def volumetric_gain_line(gain: float, ray_mass: float) -> str:
    """Suggest additive auto exposure's ``--display-gain`` for a volumetric scene.

    ``ray_mass`` is the brightest rays' density at ``gain``; the kappa named
    gives about one optical depth there.
    """
    return (
        f"Volumetric: no auto exposure, so emission keeps per-splat "
        f"normalization and a small --absorption opens saturated. Additive auto "
        f"exposure would use --display-gain {gain:.2g} (an absolute gain, not a "
        f"multiplier); with it the brightest rays carry density {ray_mass:.3g} "
        f"and --absorption {1.0 / ray_mass:.2g} gives ~1 optical depth"
    )


def auto_exposure_attrs(
    source: Any,
    *,
    dim_order: Optional[Sequence[str]] = None,
    dimensions: Optional["Dimensions"] = None,
    camera: Optional["CameraConfig"] = None,
    colormap: Optional[str] = None,
    opacity: float = 1.0,
    blending_mode: str = "additive",
    tone_mapping: Optional[str] = None,
    target: Optional[float] = None,
    stats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """``luxar gsplat convert``'s auto exposure, as kwargs for the Python adders.

    Returns ``normalize_amplitudes=False`` plus the ``display_gain``, unit labels
    and provenance attrs convert would stamp, computed by convert's own
    resolver (:func:`resolve_convert_exposure`), so
    the two cannot disagree. ``add_gsplats_from_file`` already applies this by
    default to a fitted store; call it for a scene you lay out yourself.

    The layout the exposure is predicted for:

    * ``dimensions=None`` (default): convert's own scene for the store, from
      :func:`luxar.core.scene_orientation.resolve_convert_orientation`
      (the store's axis names, the y-down XY camera; a partition keeps column
      order). ``dim_order`` then NAMES the columns as ``--dim-order`` does and
      ``camera`` replaces convert's opening camera. Any column count works.
    * ``dimensions=`` given: that scene, with the columns mapped by ``dim_order``
      and the opening view ``camera`` (``None`` = the viewer default).

    Pass the ``.gsplats.zarr`` PATH: it works for every store shape and reads
    the root stats that decide the unit labels. An in-memory
    :class:`GSplatData` or :class:`~luxar.gsplats.tree.GSplatNode` gives the
    same gain, but its labels fall back to ``a.u.`` / ``data units`` unless
    ``stats=`` supplies the root stats.

    The gain only ever DIMS below per-splat normalisation, one node at a time.
    Under a non-summing blending mode it is the per-splat normalisation alone,
    which is also the fallback (with a warning) when the opening view cannot be
    predicted, e.g. a hidden scene dimension with no center column.

    Raises:
        ValueError: A malformed ``dim_order``.
    """
    from dataclasses import replace

    from .scene_orientation import ConvertOrientation

    node, source_stats = _exposure_source(source)
    if stats is None:
        stats = source_stats
    if dimensions is None:
        orientation = _store_orientation(node, stats, dim_order)
        if camera is not None:
            orientation = replace(orientation, camera=camera)
    else:
        orientation = ConvertOrientation(
            dimensions, list(dim_order) if dim_order else None, camera, ""
        )
    exposure = resolve_convert_exposure(
        node,
        orientation,
        auto_exposure=True,
        target=target,
        scale_intensity=None,
        colormap=colormap,
        opacity=opacity,
        blending_mode=blending_mode,
        tone_mapping=tone_mapping,
        stats=stats,
    )
    if exposure.line and exposure.line.startswith("Exposure: could not predict"):
        import warnings

        warnings.warn(exposure.line, stacklevel=2)
    return {"normalize_amplitudes": exposure.normalize, **exposure.attrs}


def _store_orientation(
    node: "GSplatNode",
    stats: Mapping[str, Any],
    dim_order: Optional[Sequence[str]],
) -> Any:
    """Convert's orientation for ``node`` (a matrix-shaped tree is remapped)."""
    from .scene_defaults import source_extent
    from .scene_orientation import resolve_convert_orientation

    extent, remap = source_extent(node, stats)
    return resolve_convert_orientation(
        extent,
        (node.meta or {}).get("dimension_metadata"),
        dim_order=",".join(dim_order) if dim_order else None,
        fitting_info=stats,
        remap=remap,
    )


def _exposure_source(source: Any) -> Tuple["GSplatNode", Mapping[str, Any]]:
    """``(node, root stats)`` from a path, a :class:`GSplatNode` or a ``GSplatData``."""
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    if isinstance(source, GSplatData):
        return source.tree, dict(source.stats or {})
    if isinstance(source, (GSplatLeaf, GSplatLodGroup, GSplatPartition)):
        meta = source.meta or {}
        return source, dict(meta.get("fit_stats") or meta.get("stats") or {})
    from luxar.gsplats.io.load_gsplats import load_gsplat_node

    node, stats = load_gsplat_node(source, include_stats=True)
    return node, stats or {}
