"""The scene ``luxar gsplat convert`` builds for a fitted store, for Python callers.

``convert`` does more than graft a ``.gsplats.zarr`` into a scene. It names the
center columns (a fit stores them in array order, ``z, y, x``) and maps them onto
``x, y, z`` scene dimensions, opens the camera on the XY plane with y pointing
down (the Fiji/napari view), auto-exposes a ``display_gain`` so a dense fit does
not open as a saturated white blob, lists the node in the Layers panel, and picks
the amplitude encoding from the source dtype.

:func:`resolve_fitted_scene` is the single source of truth for all of it.
``convert`` resolves its scene through it, and so does
``Group.add_gsplats_from_file``'s default for a fitted store
(``fitted_defaults=None``; see
:mod:`luxar.core.group.gsplats_pipeline.fitted_defaults`), so a bare
``scene.add_gsplats_from_file(name, path)`` opens the way ``convert --no-center``
does. :func:`fitted_scene_defaults` returns the same scene as values (the
scene's ``dimensions`` and ``viewer_config``, the compiler's
``gsplat_amplitude_bits`` and the adder kwargs in ``attrs``) for a caller who
lays the scene out by hand.

Example:
    >>> from luxar import LuxarZarrCompiler
    >>> from luxar.core.scene_defaults import fitted_scene_defaults
    >>> fit = "fit.gsplats.zarr"
    >>> defaults = fitted_scene_defaults(fit, colormap="plasma", tone_mapping="ACES")
    >>> with LuxarZarrCompiler(
    ...     "scene.luxar.zarr", gsplat_amplitude_bits=defaults.amplitude_bits
    ... ) as compiler:
    ...     scene = compiler.create_scene(
    ...         dimensions=defaults.dimensions,
    ...         viewer_config=defaults.viewer_config,
    ...     )
    ...     scene.add_gsplats_from_file("embryo", fit, **defaults.attrs)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ..typing_utils.constants import DEFAULT_BLENDING_MODE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..gsplats.tree import GSplatNode
    from .dimensions import Dimensions
    from .scene_exposure import ConvertExposure
    from .scene_orientation import ConvertOrientation
    from .viewer_config import ViewerConfig

__all__ = [
    "FittedScene",
    "fitted_scene_defaults",
    "load_fitted_source",
    "resolve_fitted_scene",
    "source_extent",
]


@dataclass(frozen=True)
class FittedScene:
    """The scene ``gsplat convert`` would build, split into its two halves.

    Attributes:
        dimensions: Scene dimensions for ``compiler.create_scene``.
        viewer_config: Scene ``ViewerConfig`` (tone mapping and the opening
            camera), or ``None`` for the viewer defaults.
        attrs: Keyword arguments for ``add_gsplats_from_file`` (convert passes
            the same set to ``add_gsplats_from_data``): ``dim_order`` when the
            columns are remapped, ``layer``, ``blending_mode``, ``opacity``,
            ``absorption``, ``colormap`` when given, ``normalize_amplitudes=False``
            and the ``display_gain`` plus unit labels.
        orientation: How the columns were laid into the scene.
        exposure: The display gain decision (``exposure.line`` narrates it).
        amplitude_bits: The compiler's ``gsplat_amplitude_bits`` (8 for a
            uint8/int8 source, else 16), as ``convert`` sets it.
    """

    dimensions: "Dimensions"
    viewer_config: "Optional[ViewerConfig]"
    attrs: Dict[str, Any]
    orientation: "ConvertOrientation"
    exposure: "ConvertExposure"
    amplitude_bits: Literal[8, 16] = 16


def _dim_order_spec(dim_order: Union[str, Sequence[str], None]) -> Optional[str]:
    """``["z", "y", "x"]`` or ``"z,y,x"`` as the comma-separated spec."""
    if dim_order is None or isinstance(dim_order, str):
        return dim_order
    return ",".join(dim_order)


def _viewer_config(
    orientation: "ConvertOrientation", tone_mapping: Optional[str]
) -> "Optional[ViewerConfig]":
    """Tone mapping and/or the orientation's camera, else ``None``."""
    from .viewer_config import ViewerConfig

    if tone_mapping is None and orientation.camera is None:
        return None
    return ViewerConfig(tone_mapping=tone_mapping, camera=orientation.camera)


def resolve_fitted_scene(
    node: "GSplatNode",
    extent: np.ndarray,
    *,
    stats: Optional[Mapping[str, Any]] = None,
    dimension_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    remap: bool = True,
    dim_order: Union[str, Sequence[str], None] = None,
    view: Optional[str] = None,
    colormap: Optional[str] = None,
    tone_mapping: Optional[str] = None,
    blending_mode: str = DEFAULT_BLENDING_MODE,
    opacity: float = 1.0,
    absorption: float = 1.0,
    layer: bool = True,
    auto_exposure: bool = True,
    target_brightness: Optional[float] = None,
    display_gain: Optional[float] = None,
    orientation: "Optional[ConvertOrientation]" = None,
) -> FittedScene:
    """Resolve ``convert``'s scene for an already-loaded node tree.

    Args:
        node: The node tree whose default leaves are exposed.
        extent: Its center coordinates (or a two-row min/max box), ``(N, D)``.
        stats: The store's root stats (unit labels, fitting provenance).
        dimension_metadata: The store's root ``dimension_metadata`` (axis
            names); ``None`` reads it from ``node.meta``.
        remap: ``True`` for a matrix-shaped tree (columns are mapped onto
            ``x, y, z``-ordered scene dims through ``dim_order``); ``False`` for
            a grafted partition / nested tree, which cannot be remapped.
        dim_order: Names for the center columns (``--dim-order``); ``None``
            reads the store's axis metadata.
        view: Opening face (``xy`` / ``xz`` / ``yz``), ``None`` = ``xy`` for an
            image-ordered fit.
        colormap: Colormap for scalar amplitudes (also weighs the exposure).
        tone_mapping: Scene tone mapping (also picks the exposure target).
        blending_mode: Node blending mode.
        opacity: Node opacity.
        absorption: Volumetric kappa.
        layer: List the node in the Layers panel.
        auto_exposure: Dim for the predicted opening-view sum (summing modes).
        target_brightness: Auto-exposure target; ``None`` = tone-mapping default.
        display_gain: An absolute gain that replaces auto exposure.
        orientation: A layout already decided (an existing scene's dimensions
            and camera); ``None`` resolves convert's from the store.

    Raises:
        ValueError: On a malformed ``dim_order`` or ``view``.
    """
    from ..gsplats.io.save_gsplats import resolve_amplitude_bits
    from .scene_exposure import resolve_convert_exposure
    from .scene_orientation import resolve_convert_orientation

    stats = dict(stats or {})
    if dimension_metadata is None:
        dimension_metadata = (getattr(node, "meta", None) or {}).get(
            "dimension_metadata"
        )
    if orientation is None:
        orientation = resolve_convert_orientation(
            extent,
            dimension_metadata,
            dim_order=_dim_order_spec(dim_order),
            view=view,
            fitting_info=stats,
            remap=remap,
        )
    exposure = resolve_convert_exposure(
        node,
        orientation,
        auto_exposure=auto_exposure,
        target=target_brightness,
        scale_intensity=display_gain,
        colormap=colormap,
        opacity=opacity,
        blending_mode=blending_mode,
        tone_mapping=tone_mapping,
        stats=stats,
    )
    attrs: Dict[str, Any] = {
        "normalize_amplitudes": exposure.normalize,
        "opacity": opacity,
        "absorption": absorption,
        "blending_mode": blending_mode,
        "layer": layer,
        **exposure.attrs,
    }
    if colormap is not None:
        attrs["colormap"] = colormap
    if orientation.dim_order is not None:
        attrs["dim_order"] = list(orientation.dim_order)
    return FittedScene(
        dimensions=orientation.dimensions,
        viewer_config=_viewer_config(orientation, tone_mapping),
        attrs=attrs,
        orientation=orientation,
        exposure=exposure,
        amplitude_bits=resolve_amplitude_bits(
            "auto", source_dtype=_source_dtype(stats)
        ),
    )


def _source_dtype(stats: Mapping[str, Any]) -> Optional[str]:
    """The recorded source dtype when it is a string (what the adder reads)."""
    value = stats.get("source_dtype")
    return value if isinstance(value, str) else None


def source_extent(
    node: "GSplatNode",
    stats: Optional[Mapping[str, Any]] = None,
    *,
    flatten: bool = False,
) -> Tuple[np.ndarray, bool]:
    """``(extent, remap)`` of a loaded node tree, as ``convert`` lays it out.

    ``remap`` is whether the columns can be mapped through ``dim_order``: a
    matrix-shaped tree (or any tree with ``flatten=True``) is re-authored
    through ``GSplatData`` and can, and ``extent`` is its centers; a partition /
    nested tree is grafted and cannot, and ``extent`` is its center bounds.

    Raises:
        ValueError: When a grafted node tree holds no splats.
    """
    from ..gsplats.gsplat_data import GSplatData
    from ..gsplats.tree import center_bounds, is_matrix_shaped

    if flatten:
        flat = GSplatData.from_default_selection(node, stats=dict(stats or {}))
        return flat.flattened().centers, True
    if is_matrix_shaped(node):
        return GSplatData.from_tree(node).centers, True
    bounds = center_bounds(node)
    if bounds is None:
        raise ValueError("the node tree holds no splats")
    return np.array([bounds[0], bounds[1]], dtype=np.float32), False


def load_fitted_source(
    path: Union[str, Path], *, flatten: bool = False
) -> Tuple["GSplatNode", Dict[str, Any], np.ndarray, bool]:
    """``(node, stats, extent, remap)`` for :func:`resolve_fitted_scene`.

    Raises:
        ValueError: When a grafted node tree holds no splats.
    """
    from ..gsplats.io.load_gsplats import load_gsplat_node

    node, stats = load_gsplat_node(path, include_stats=True)
    try:
        extent, remap = source_extent(node, stats, flatten=flatten)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    return node, stats, extent, remap


def fitted_scene_defaults(path: Union[str, Path], **options: Any) -> FittedScene:
    """``gsplat convert``'s scene for the store at ``path`` (without centering).

    Loads the store and calls :func:`resolve_fitted_scene` with ``options``
    (``colormap``, ``tone_mapping``, ``blending_mode``, ``dim_order``, ``view``,
    ``display_gain``, ...). The result matches ``luxar gsplat convert
    --no-center`` with the same flags: pass ``amplitude_bits`` to
    ``LuxarZarrCompiler(gsplat_amplitude_bits=...)``, ``dimensions`` /
    ``viewer_config`` to ``compiler.create_scene`` and ``**attrs`` to
    ``add_gsplats_from_file`` for the same ``path``. A partition / nested store
    is grafted as is, so ``attrs`` then carries no ``dim_order``.

    The store is read here and again by the adder: a second full load, which
    costs as much as the first on a large store. (The adder's own fitted
    default reads it twice too, once to resolve and once to write.)

    Raises:
        ValueError: A malformed option, or a grafted tree with no splats.
    """
    node, stats, extent, remap = load_fitted_source(path)
    return resolve_fitted_scene(node, extent, stats=stats, remap=remap, **options)
