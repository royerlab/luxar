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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from ...core.group.gsplats_pipeline.amplitude_norm import (
    AMPLITUDE_OFFSET_ATTR,
    AMPLITUDE_UNIT_ATTR,
    DISPLAY_GAIN_ATTR,
    NORMALIZATION_MODE_ATTR,
    PROJECTED_SUM_AXIS_ATTR,
    PROJECTED_SUM_P999_ATTR,
    PROJECTED_SUM_TARGET_ATTR,
    SOURCE_DATA_RANGE_ATTR,
    NormalizeSpec,
    _sample_reference,
    resolve_factor,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...gsplats.exposure import ExposurePlan, HiddenAxis
    from ...gsplats.tree import GSplatNode
    from .scene_orientation import ConvertOrientation

__all__ = ["ConvertExposure", "resolve_convert_exposure"]


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


def _column_of(name: str, orientation: "ConvertOrientation") -> int:
    """Center column holding the scene dimension ``name``."""
    if orientation.dim_order is not None:
        return list(orientation.dim_order).index(name)
    return orientation.dimensions.names.index(name)


def _world_index(letter: str, displayed: List[str]) -> int:
    """Displayed position of the dimension named ``letter`` (as the camera does)."""
    lowered = [name.lower() for name in displayed]
    return lowered.index(letter) if letter in lowered else "xyz".index(letter)


def opening_view_columns(
    orientation: "ConvertOrientation",
) -> Tuple[List[int], Optional[int], Optional[str]]:
    """Displayed center columns, the view-axis column and its scene name.

    The camera's ``view_axis`` (``xy`` faces the XY plane, looking along the
    third displayed axis); without one, the viewer's default looks along the
    third displayed dimension. A planar scene has no view axis.
    """
    dims = orientation.dimensions
    displayed = [dims.dimensions[i].name for i in dims.displayed]
    columns = [_column_of(name, orientation) for name in displayed]
    if len(displayed) != 3:
        return columns, None, None
    camera = orientation.camera
    depth = 2
    if camera is not None and camera.view_axis is not None:
        plane = {_world_index(c, displayed) for c in camera.view_axis}
        depth = ({0, 1, 2} - plane).pop()
    return columns, columns[depth], displayed[depth]


def opening_hidden_axes(
    orientation: "ConvertOrientation", centers: Any
) -> List["HiddenAxis"]:
    """Each hidden scene dimension as a :class:`HiddenAxis` at its opening position.

    The viewer's ``SceneDimsManager.defaultPosition``: a discrete (or
    categorical) axis opens at its range minimum, a continuous one at its centre.
    """
    import numpy as np

    from ...gsplats.exposure import HiddenAxis

    axes: List[HiddenAxis] = []
    for dim in orientation.dimensions.dimensions:
        if dim.display:
            continue
        column = _column_of(dim.name, orientation)
        values = np.asarray(centers[:, column], dtype=np.float64)
        if dim.range is not None:
            lo, hi = float(dim.range[0]), float(dim.range[1])
        else:
            lo, hi = float(values.min()), float(values.max())
        discrete = bool(dim.discrete or dim.categories)
        step = float(dim.step or 1.0)
        position = lo if discrete else 0.5 * (lo + hi)
        axes.append(HiddenAxis(column, position, discrete, step, (lo, hi)))
    return axes


def _leaf_truncation(node: "GSplatNode") -> float:
    """The truncation radius the opening leaves render with."""
    from ...gsplats.tree import iter_default_leaves
    from ...typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

    for leaf in iter_default_leaves(node):
        for sub in leaf.additive_sublods:
            radius = getattr(sub, "truncation_radius", None)
            if radius:
                return float(radius)
    return DEFAULT_TRUNCATION_RADIUS


def _gain_attrs(gain: float, mode: str, units: Mapping[str, Any]) -> Dict[str, Any]:
    """``display_gain`` + the unit labels + the provenance mode."""
    return {DISPLAY_GAIN_ATTR: float(gain), NORMALIZATION_MODE_ATTR: mode, **units}


def _exposure(
    gain: float,
    mode: str,
    units: Mapping[str, Any],
    line: Optional[str] = None,
    **extra: Any,
) -> ConvertExposure:
    """A :class:`ConvertExposure` carrying ``gain`` (and its stamp attrs)."""
    attrs = _gain_attrs(gain, mode, units)
    attrs.update(extra)
    return ConvertExposure(gain=float(gain), attrs=attrs, line=line)


def amplitude_unit_attrs(stats: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``amplitude_unit`` (+ ``amplitude_offset``) for a store's amplitudes.

    A fit records its normalisation basis (``intensity_range``), and its
    amplitudes are then in source data units above the subtracted level
    (``image_min``, see :mod:`luxar.gsplats.fit_basis`): ``counts`` for an
    integer source, ``data units`` otherwise, ``above background`` when a level
    was subtracted, with that level as ``amplitude_offset``. A store recording
    no basis — an imported classical splat file, a merge of disagreeing fits —
    gets ``a.u.`` and no offset: nothing says what its numbers mean.

    With a basis, ``source_data_range`` records ``[0, intensity_range]`` in
    amplitude units. After intensity scaling the amplitudes and range are in
    arbitrary units; the raw source floor is then omitted from the scene.
    """
    import numpy as np

    from ...gsplats.fit_basis import fit_image_min, source_intensity_span

    stats = stats or {}
    if stats.get("intensity_range") is None:
        return {AMPLITUDE_UNIT_ATTR: "a.u."}
    try:
        integer = np.dtype(stats.get("source_dtype")).kind in "iu"
    except (TypeError, ValueError):
        integer = False
    scaled = stats.get("intensity_scale", 1.0) != 1.0
    unit = "a.u." if scaled else ("counts" if integer else "data units")
    level = fit_image_min(stats)
    attrs: Dict[str, Any] = {}
    if not scaled and level is not None and level > 0.0:
        unit += " above background"
        attrs[AMPLITUDE_OFFSET_ATTR] = float(level)
    attrs[AMPLITUDE_UNIT_ATTR] = unit
    span = source_intensity_span(stats)
    if span is not None:
        attrs[SOURCE_DATA_RANGE_ATTR] = [0.0, float(span)]
    return attrs


def _manual(scale_intensity: float, units: Mapping[str, Any]) -> ConvertExposure:
    """``--scale-intensity``: gain exactly ``S``, recorded as mode ``manual``."""
    return _exposure(
        scale_intensity,
        "manual",
        units,
        f"Exposure: manual gain ×{scale_intensity:g} (--scale-intensity); "
        "auto exposure off",
    )


def _exposure_line(plan: "ExposurePlan") -> str:
    """The one narration line; names the side-view cap only when it binds."""
    if plan.predicted_before == 0:
        if plan.factor >= 1.0:
            return "Exposure: nothing visible at the opening position; brightness unchanged"
        return (
            "Exposure: nothing visible at the opening position; "
            f"scaled splat brightness ×{plan.factor:.2g}, capped by {plan.cap_source}"
        )
    if plan.factor >= 1.0:
        return (
            f"Exposure: default view already within range (projected p99.9 "
            f"{plan.predicted_before:.2g} ≤ {plan.target:g}); brightness unchanged"
        )
    reason = "the default view isn't saturated"
    detail = f"projected p99.9 {plan.predicted_before:.2g} → {plan.predicted_after:.2g}"
    if plan.side_capped and plan.side_p999 is not None:
        reason += f", capped by {plan.cap_source}"
        detail += f", {plan.cap_source} {plan.side_p999 * plan.factor:.2g}"
    return (
        f"Exposure: scaled splat brightness ×{plan.factor:.2g} so {reason} "
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
    from ...gsplats.exposure import (
        SUMMING_BLENDING_MODES,
        collect_default_splats,
        default_target_brightness,
        plan_projected_exposure,
        stamped_amplitude_window,
    )

    units = amplitude_unit_attrs(stats)
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
            trunc=_leaf_truncation(node),
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
    return _exposure(
        per_splat * plan.factor, "projected", units, _exposure_line(plan), **extra
    )
