"""Opening-view exposure for ``luxar gsplat convert`` (auto exposure).

Per-splat amplitude normalisation maps the p99.9 SPLAT to 1.0, but additive
blending sums every splat along the view ray, so a dense fit still opens as a
saturated white blob. Auto exposure (the convert default) predicts the opening
view's per-pixel sum with :func:`luxar.gsplats.exposure.plan_projected_exposure`
and dims the amplitudes — never brightens — so its p99.9 lands at
``--target-brightness`` (default by tone mapping: 2.0 filmic, 0.9 clipping),
unless a side view (orbit 90°) or another timepoint would then exceed
``SIDE_CAP`` (4) times that, in which case those set the factor.
The whole raw-to-stored factor is applied as ONE exact :class:`AmplitudeScale`
and stamped as ``amplitude_normalization_factor``, with
``amplitude_normalization_mode`` and the prediction beside it, so the stored
amplitudes divide back to the fit's own values. The ``.gsplats.zarr`` itself is
never touched: ``gsplat render`` / ``compare`` / ``export`` read it, not the scene.

``--no-auto-exposure`` is the legacy per-splat normalisation, attr for attr;
``--scale-intensity S`` replaces both with an exact manual factor ``S``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ...core.group.gsplats_pipeline.amplitude_norm import (
    NORMALIZATION_FACTOR_ATTR,
    NORMALIZATION_MODE_ATTR,
    PROJECTED_SUM_AXIS_ATTR,
    PROJECTED_SUM_P999_ATTR,
    PROJECTED_SUM_TARGET_ATTR,
    AmplitudeScale,
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
        normalize: The ``normalize_amplitudes`` argument.
        attrs: Provenance attrs to stamp on the gsplats node(s).
        line: One narration line, or ``None``.
    """

    normalize: NormalizeSpec
    attrs: Dict[str, Any] = field(default_factory=dict)
    line: Optional[str] = None


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


def _manual(scale_intensity: float) -> ConvertExposure:
    """``--scale-intensity``: exactly ``raw × S``, recorded as mode ``manual``."""
    return ConvertExposure(
        normalize=AmplitudeScale(scale_intensity),
        attrs={
            NORMALIZATION_FACTOR_ATTR: float(scale_intensity),
            NORMALIZATION_MODE_ATTR: "manual",
        },
        line=f"Exposure: manual ×{scale_intensity:g} (--scale-intensity); "
        "auto exposure off",
    )


def _exposure_line(plan: "ExposurePlan") -> str:
    """The one narration line; names the side-view cap only when it binds."""
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
) -> ConvertExposure:
    """Decide the amplitude scaling convert applies (see the module docstring).

    ``target=None`` picks the tone-mapping default
    (:func:`~luxar.gsplats.exposure.default_target_brightness`).
    """
    from ...gsplats.exposure import (
        SUMMING_BLENDING_MODES,
        collect_default_splats,
        default_target_brightness,
        plan_projected_exposure,
        stamped_amplitude_window,
    )

    if scale_intensity is not None:
        return _manual(scale_intensity)
    if not auto_exposure:
        return ConvertExposure(normalize=True)
    if blending_mode not in SUMMING_BLENDING_MODES:
        return ConvertExposure(
            normalize=True,
            line=f"Exposure: auto exposure applies to additive/luminous blending; "
            f"'{blending_mode}' keeps per-splat normalization",
        )
    if target is None:
        target = default_target_brightness(tone_mapping)
    columns, axis_col, axis_name = opening_view_columns(orientation)
    centers, chol, amps, colors = collect_default_splats(node)
    per_splat = resolve_factor(True, _sample_reference(amps)) or 1.0
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
        return ConvertExposure(
            normalize=True,
            line=f"Exposure: could not predict the opening view ({exc}); "
            "using per-splat normalization",
        )
    total = per_splat * plan.factor
    attrs: Dict[str, Any] = {
        NORMALIZATION_FACTOR_ATTR: float(total),
        NORMALIZATION_MODE_ATTR: "projected",
        PROJECTED_SUM_P999_ATTR: float(plan.predicted_after),
        PROJECTED_SUM_TARGET_ATTR: float(target),
    }
    if axis_name is not None:
        attrs[PROJECTED_SUM_AXIS_ATTR] = axis_name
    line = _exposure_line(plan)
    return ConvertExposure(normalize=AmplitudeScale(total), attrs=attrs, line=line)
