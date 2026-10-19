"""``add_gsplats_from_file``'s default for a FITTED store: what ``convert`` does.

A bare ``scene.add_gsplats_from_file(name, "fit.gsplats.zarr")`` used to keep
whatever scene it was given (``Dimensions.default_3d()`` opens a ``z, y, x`` fit
edge-on), normalise each splat (a dense fit opens saturated under ``additive``)
and stay out of the Layers panel, while ``luxar gsplat convert`` on the same
store looked right. With ``fitted_defaults=None`` (the default) a fitted store now
gets convert's choices, resolved by
:func:`luxar.gsplats.scene_defaults.resolve_fitted_scene`, the function convert
itself calls.

**Which stores.** A store is fitted when it names its columns
(:func:`luxar.gsplats.scene_orientation.fitted_column_names`): a complete root
``dimension_metadata``, or fitting provenance whose ``source_shape`` has the
centers' rank. A classical-splat import (``.ply`` / ``.splat`` / ``.spz``) or a
hand-built store has neither and keeps the old behaviour. ``fitted_defaults=True``
applies the defaults to any ``.gsplats.zarr`` store (a classical import never
gets them); ``False`` turns them off.

**Node level** (only what the caller did not pass; an explicit argument wins):

* ``dim_order``: convert's column mapping, ONLY when this call lays out a fresh
  scene (below) or the scene's dimensions are the ones an earlier fitted add laid
  out (a second channel). On any other scene the columns keep the positional
  mapping they always had, so existing geometry and transforms authored for it
  stay where they were, and one line suggests the ``dim_order`` to pass.
  Passing ``dim_order`` / ``fill`` / ``fill_sigma`` / ``transform`` /
  ``nd_transform`` counts as choosing the orientation yourself.
* Exposure: ``normalize_amplitudes=False`` with convert's auto ``display_gain``
  and unit labels, predicted for the scene the node lands in, under the tone
  mapping the scene has AT ADD TIME (setting ``tone_mapping`` later does not
  re-expose). Skipped as a group when the caller passes ``normalize_amplitudes``
  or any of :data:`EXPOSURE_ARGS` (a window or a gain authored against the old,
  normalised amplitudes would be read in data units).
* ``layer=True``, and ``blending_mode`` / ``opacity`` / ``absorption`` written
  explicitly with convert's defaults.
* The amplitude encoding: 8-bit for a uint8/int8 source, as convert configures
  the compiler, unless the compiler was given ``gsplat_amplitude_bits``.

Inside a ``kind=lod`` / ``kind=partition`` group the defaults do not apply
automatically (siblings there share one exposure).

**Scene level** (dimensions and the y-down XY camera) only when it is
unambiguous: the node goes on the root scene, the scene's dimensions equal
``Dimensions.default_3d()`` (a scene EXPLICITLY created with
``Dimensions.default_3d()`` counts as fresh: the two cannot be told apart), it has
no camera, nothing else in it is geometry, and the caller made no orientation
choice (above). Otherwise the scene is left alone and one line says what to pass
to ``create_scene``.

**Cost.** The store is loaded once and handed to the writer, but resolving runs
convert's projected-exposure prediction (a rasterised sum over the default
leaves), which is noticeable on a very large store; ``fitted_defaults=False``
skips it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from arbol import aprint

from ....typing_utils.constants import DEFAULT_BLENDING_MODE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...dimensions import Dimensions
    from ...node import Node
    from ...scene import Scene
    from ...viewer_config import ViewerConfig
    from ..group import Group

__all__ = [
    "EXPOSURE_ARGS",
    "FITTED_LAYOUT_ATTR",
    "FittedAdderDefaults",
    "ORIENTATION_ARGS",
    "resolve_adder_defaults",
]

#: Caller kwargs that author the amplitude scale or a window on it. Any of them
#: (or an explicit ``normalize_amplitudes``) keeps the old exposure path.
EXPOSURE_ARGS = frozenset(
    {
        "display_gain",
        "amplitude_unit",
        "amplitude_offset",
        "source_data_range",
        "intensity",
        "offset",
    }
)

#: Node attrs that place or orient the node: passing any of them (or
#: ``dim_order`` / ``fill`` / ``fill_sigma``) is an orientation choice.
ORIENTATION_ARGS = frozenset({"transform", "nd_transform"})

#: Node attrs convert writes explicitly, filled when the caller did not.
_COMPOSITING_DEFAULTS = ("blending_mode", "opacity", "absorption", "layer")

#: Classical-splat suffixes the adder imports on the fly (never fitted).
_CLASSICAL_SUFFIXES = (".ply", ".splat", ".spz")


@dataclass
class FittedAdderDefaults:
    """What the adder applies for a fitted store.

    Attributes:
        dim_order: The ``dim_order`` to pass on (the caller's, or convert's).
        normalize_amplitudes: ``False`` when convert's exposure applies;
            ``None`` leaves the caller's value (or the unset default) alone.
        attrs: The node attrs, the caller's plus the filled defaults.
        amplitude_bits: The compiler tier to use while writing, or ``None``.
        dimensions: Scene dimensions to set before the add, or ``None``.
        viewer_config: Scene viewer config to set before the add, or ``None``.
        loaded: The ``(node, stats)`` already read, handed to the writer so the
            store is not loaded twice.
        applied: ``False`` when nothing applies (the store is not fitted) and
            only ``loaded`` is meaningful.
    """

    dim_order: Optional[List[str]]
    normalize_amplitudes: Optional[bool]
    attrs: Dict[str, Any] = field(default_factory=dict)
    amplitude_bits: Optional[Literal[8, 16]] = None
    dimensions: "Optional[Dimensions]" = None
    viewer_config: "Optional[ViewerConfig]" = None
    loaded: Optional[Tuple[Any, Dict[str, Any]]] = None
    applied: bool = True


#: Private scene attribute recording the dimensions a fitted add laid out, so a
#: second channel on the same scene knows it may use convert's ``dim_order``.
FITTED_LAYOUT_ATTR = "_fitted_layout_dims"


def _scene_is_fresh(scene: "Scene") -> bool:
    """Default 3-D dimensions, no camera, no geometry: safe to lay out."""
    from ...dimensions import Dimensions

    if scene.dimensions.to_dict() != Dimensions.default_3d().to_dict():
        return False
    config = scene.viewer_config
    if config is not None and config.camera is not None:
        return False
    return not scene._has_authored_geometry()


def _scene_viewer_config(scene: "Scene", camera: Any) -> "Optional[ViewerConfig]":
    """The scene's viewer config with ``camera`` set (``None`` if no camera)."""
    from ...viewer_config import ViewerConfig

    if camera is None:
        return None
    config = scene.viewer_config
    return replace(config, camera=camera) if config else ViewerConfig(camera=camera)


def _laid_out_by_a_fitted_add(scene: "Scene") -> bool:
    """Whether the scene's dimensions are the ones an earlier fitted add set."""
    recorded = getattr(scene, FITTED_LAYOUT_ATTR, None)
    return recorded is not None and recorded == scene.dimensions.to_dict()


def _existing_layout(
    scene: "Scene", conv: Any, dim_order: Optional[List[str]], remap: bool
) -> Any:
    """The layout of a scene that is NOT re-laid out: its dims and camera.

    Convert's ``dim_order`` is used only on a scene an earlier fitted add laid
    out, and only when the store can be remapped and its names are exactly the
    scene's dimension names. Any other scene keeps the positional mapping.
    """
    from ....gsplats.scene_orientation import ConvertOrientation

    order = dim_order
    if (
        order is None
        and remap
        and conv.dim_order is not None
        and _laid_out_by_a_fitted_add(scene)
        and sorted(conv.dim_order) == sorted(scene.dimensions.names)
    ):
        order = list(conv.dim_order)
    config = scene.viewer_config
    camera = config.camera if config is not None else None
    return ConvertOrientation(scene.dimensions, order, camera, "")


def _string_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def resolve_adder_defaults(
    group: "Group",
    *,
    path: Union[str, Path],
    parent: "Optional[Node]",
    fitted_defaults: Optional[bool],
    dim_order: Optional[List[str]],
    orientation_given: bool,
    normalize_given: bool,
    flatten: bool,
    attrs: Dict[str, Any],
) -> Optional[FittedAdderDefaults]:
    """Convert's defaults for this ``add_gsplats_from_file`` call, or ``None``.

    ``None`` means "apply nothing": the defaults are off, the file is missing
    (the adder reports that itself), a classical import, not fitted, or the
    target is a ``kind=lod`` / ``kind=partition`` group under the automatic
    default.
    """
    if fitted_defaults is False:
        return None
    path = Path(path)
    if not path.exists() or path.suffix.lower() in _CLASSICAL_SUFFIXES:
        return None
    target = parent or group
    if fitted_defaults is None and target.attrs.get("kind") in ("lod", "partition"):
        return None
    orientation_given = orientation_given or bool(ORIENTATION_ARGS & attrs.keys())
    return _resolve_for_store(
        group,
        path,
        target,
        fitted_defaults=fitted_defaults,
        dim_order=dim_order,
        orientation_given=orientation_given,
        normalize_given=normalize_given,
        flatten=flatten,
        attrs=attrs,
    )


def _resolve_for_store(
    group: "Group",
    path: Path,
    target: "Node",
    *,
    fitted_defaults: Optional[bool],
    dim_order: Optional[List[str]],
    orientation_given: bool,
    normalize_given: bool,
    flatten: bool,
    attrs: Dict[str, Any],
) -> Optional[FittedAdderDefaults]:
    """Load the store and resolve; ``None`` when it is not fitted (auto)."""
    from ....gsplats.io.load_gsplats import load_gsplat_node
    from ....gsplats.scene_defaults import resolve_fitted_scene, source_extent
    from ....gsplats.scene_orientation import (
        fitted_column_names,
        resolve_convert_orientation,
    )

    node, stats = load_gsplat_node(path, include_stats=True)
    try:
        extent, remap = source_extent(node, stats, flatten=flatten)
    except ValueError:
        # An empty tree: the adder words that fault itself.
        return FittedAdderDefaults(
            dim_order, None, dict(attrs), loaded=(node, stats), applied=False
        )
    metadata = (node.meta or {}).get("dimension_metadata")
    names = fitted_column_names(int(extent.shape[1]), metadata, stats)
    if names is None and fitted_defaults is None:
        return FittedAdderDefaults(
            dim_order, None, dict(attrs), loaded=(node, stats), applied=False
        )

    scene = group._find_scene()
    conv = resolve_convert_orientation(
        extent,
        metadata,
        fitting_info=stats,
        remap=remap,
    )
    lay_out = (not orientation_given) and target is scene and _scene_is_fresh(scene)
    layout = conv if lay_out else _existing_layout(scene, conv, dim_order, remap)
    exposure_given = normalize_given or bool(EXPOSURE_ARGS & attrs.keys())
    config = scene.viewer_config
    resolved = resolve_fitted_scene(
        node,
        extent,
        stats=stats,
        remap=remap,
        colormap=_string_or_none(attrs.get("colormap")),
        tone_mapping=config.tone_mapping if config is not None else None,
        blending_mode=attrs.get("blending_mode", DEFAULT_BLENDING_MODE),
        opacity=float(attrs.get("opacity", 1.0)),
        absorption=float(attrs.get("absorption", 1.0)),
        layer=bool(attrs.get("layer", True)),
        auto_exposure=not exposure_given,
        orientation=layout,
    )
    defaults = _assemble(
        scene,
        resolved,
        conv,
        lay_out=lay_out,
        remap=remap,
        dim_order=dim_order,
        orientation_given=orientation_given,
        exposure_given=exposure_given,
        attrs=attrs,
    )
    defaults.loaded = (node, stats)
    return defaults


def _assemble(
    scene: "Scene",
    resolved: Any,
    conv: Any,
    *,
    lay_out: bool,
    remap: bool,
    dim_order: Optional[List[str]],
    orientation_given: bool,
    exposure_given: bool,
    attrs: Dict[str, Any],
) -> FittedAdderDefaults:
    """Merge the resolved defaults under the caller's kwargs and narrate."""
    merged = dict(attrs)
    for key in _COMPOSITING_DEFAULTS:
        merged.setdefault(key, resolved.attrs[key])
    normalize: Optional[bool] = None
    if not exposure_given:
        normalize = False
        merged.update(resolved.exposure.attrs)
    order = dim_order if orientation_given else resolved.orientation.dim_order
    if not remap:
        order = dim_order
    defaults = FittedAdderDefaults(
        dim_order=list(order) if order is not None else None,
        normalize_amplitudes=normalize,
        attrs=merged,
        amplitude_bits=resolved.amplitude_bits
        if resolved.amplitude_bits == 8
        else None,
    )
    if lay_out:
        defaults.dimensions = conv.dimensions
        defaults.viewer_config = _scene_viewer_config(scene, conv.camera)
    _narrate(
        scene,
        defaults,
        conv,
        lay_out=lay_out,
        filled_order=not orientation_given and defaults.dim_order is not None,
        filled=[k for k in _COMPOSITING_DEFAULTS if k not in attrs],
        exposure_given=exposure_given,
    )
    return defaults


def _differs_from(scene: "Scene", conv: Any) -> bool:
    """Whether convert would lay the scene out differently (names or camera)."""
    config = scene.viewer_config
    camera = config.camera if config is not None else None
    return bool(
        scene.dimensions.names != conv.dimensions.names or camera != conv.camera
    )


def _narrate(
    scene: "Scene",
    defaults: FittedAdderDefaults,
    conv: Any,
    *,
    lay_out: bool,
    filled_order: bool,
    filled: List[str],
    exposure_given: bool,
) -> None:
    """One line for what the defaults FILLED, one for a scene left alone."""
    parts = []
    if lay_out:
        parts.append(f"scene dims {conv.dimensions.names}")
        if defaults.viewer_config is not None:
            parts.append("y-down XY camera")
    if filled_order:
        parts.append(f"dim_order {defaults.dim_order}")
    if not exposure_given:
        parts.append(f"display gain {defaults.attrs['display_gain']:.3g}")
    if defaults.amplitude_bits == 8:
        parts.append("8-bit amplitudes")
    parts.extend(f"{key}={defaults.attrs[key]!r}" for key in filled)
    aprint(
        "Fitted store: applied gsplat convert's defaults ("
        + ", ".join(parts)
        + "); fitted_defaults=False keeps the plain adder."
    )
    if lay_out:
        return
    if not filled_order and defaults.dim_order is None and conv.dim_order:
        aprint(
            "Columns kept their positional mapping on this scene; pass "
            f"dim_order={list(conv.dim_order)} for convert's."
        )
    if _differs_from(scene, conv):
        aprint(
            "Scene dims/camera left as created; for convert's layout pass "
            "fitted_scene_defaults(path).dimensions / .viewer_config to "
            "create_scene (luxar.gsplats.scene_defaults)."
        )
