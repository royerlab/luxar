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
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from ...core.group.gsplats_pipeline.amplitude_norm import (
    NORMALIZATION_MODE_ATTR,
    PROJECTED_SUM_AXIS_ATTR,
    PROJECTED_SUM_P999_ATTR,
    PROJECTED_SUM_TARGET_ATTR,
    NormalizeSpec,
    resolve_factor,
)
from ...gsplats.amplitude_reference import sample_reference as _sample_reference

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...gsplats.exposure import ExposurePlan, HiddenAxis
    from ...gsplats.tree import GSplatNode
    from .scene_orientation import ConvertOrientation

__all__ = [
    "ConvertExposure",
    "predict_volumetric_ray_mass",
    "resolve_convert_exposure",
    "volumetric_depth_line",
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


def _gain_attrs(
    gain: float, mode: str, stats: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """The public :func:`~luxar.gsplats.exposure.data_unit_display_attrs` set
    (gain + unit labels — what a Python caller gets too) + the provenance mode."""
    from ...gsplats.exposure import data_unit_display_attrs

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
    from ...gsplats.exposure import (
        SUMMING_BLENDING_MODES,
        collect_default_splats,
        default_target_brightness,
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
    import numpy as np

    from ...gsplats.exposure import collect_default_splats, plan_projected_exposure

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
            trunc=_leaf_truncation(node),
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
