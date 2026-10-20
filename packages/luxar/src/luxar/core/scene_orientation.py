"""Scene orientation for ``luxar gsplat convert``: image convention by default.

A fit of an image volume stores its center columns in ARRAY order — ``(z, y, x)``
for a 3-D stack, ``(y, x)`` for a plane. Mapped onto the scene positionally, the
slow ``z`` axis lands on world X: the stack opens edge-on, and because world +Y
points up while image rows count DOWN, it is also mirrored against Fiji/napari.

:func:`resolve_convert_orientation` fixes both without touching geometry:

* **Axes.** The columns are named (``--dim-order`` first, then the stored
  ``dimension_metadata``, then — only for a store with fitting provenance whose
  ``fitting/source_shape`` has the centers' rank — the array-order assumption;
  anything else keeps the positional mapping) and, on the matrix-shaped
  path, mapped onto scene dimensions ordered ``x, y, z`` through ``dim_order`` —
  the same mechanism the demos use (Cholesky factors are permuted with the
  centers). A grafted partition/nested tree cannot be remapped
  (``add_gsplats_from_file`` refuses ``dim_order`` there), so its scene
  dimensions keep column order, named honestly. When the names include the
  spatial axes, they also decide which dims are DISPLAYED (x, y, z shown; every
  other axis hidden and discrete), wherever those columns sit.
* **The y flip is a CAMERA choice, not a negative scale.** The scene gets
  ``CameraConfig(view_axis="xy", up=-e_y)``: the viewer faces the XY plane with
  x to the right and y pointing DOWN, looking along +z — exactly the Fiji/napari
  view, and still a proper rotation of the data. A negative-scale node transform
  would also flip y, but it mirrors the geometry: handedness inverts (a mesh's
  winding would turn inside out, every pick/scale-bar readout and the scene
  bounds would report ``-y``, and a ``Home``/``F`` re-frame or a ``--view xz``
  would inherit the mirror). The camera leaves every coordinate, unit and bound
  exactly as fitted.

``view_axis`` letters are scene dimension NAMES (the viewer resolves them against
the displayed dims), so the same preset works whether the columns were remapped
or kept in column order. ``--view xz|yz`` picks another face.

``--dim-order`` NAMES the columns; it does not reorder data. ``--dim-order x,y,z``
on a ``(z, y, x)`` fit is therefore the LEGACY POSITIONAL MODE: it labels the z
column "x" (exactly the pre-image-convention mapping, so the stack opens edge-on
again) and, x preceding y, applies no flip. Use it to reproduce old scenes or for
data genuinely stored as x, y, z.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from ..gsplats.io.save_gsplats import fit_axis_dimension_metadata
from .dimension_inference import build_dimensions_from_data, infer_discrete_step
from .dimensions import Dimension, Dimensions
from .viewer_config import VALID_VIEW_AXES, CameraConfig

__all__ = [
    "ConvertOrientation",
    "fitted_column_names",
    "parse_dim_order",
    "resolve_convert_orientation",
]

#: Canonical display order of the spatial scene axes.
_CANONICAL_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class ConvertOrientation:
    """How ``gsplat convert`` lays the fitted columns into the scene.

    Attributes:
        dimensions: Scene dimensions (ranges/units carried with their columns).
        dim_order: Column → scene-dimension names for ``add_gsplats_*``, or
            ``None`` when the columns already sit in scene order.
        camera: Opening camera (axis preset + up), or ``None`` for the viewer
            default.
        summary: One line describing the decision, for the CLI narration.
    """

    dimensions: Dimensions
    dim_order: Optional[List[str]]
    camera: Optional[CameraConfig]
    summary: str


def parse_dim_order(spec: Optional[str], ndim: int) -> Optional[List[str]]:
    """Parse ``--dim-order z,y,x`` into one name per center column.

    Raises:
        ValueError: On a count that does not match ``ndim``, an empty name, or a
            duplicate.
    """
    if spec is None:
        return None
    names = [part.strip() for part in spec.split(",")]
    if any(not name for name in names):
        raise ValueError(f"--dim-order has an empty name: {spec!r}")
    if len(names) != ndim:
        raise ValueError(
            f"--dim-order names {len(names)} columns but the dataset has {ndim}"
        )
    if len({name.lower() for name in names}) != len(names):
        raise ValueError(f"--dim-order names must be unique: {spec!r}")
    return names


def _metadata_names(
    metadata: Optional[Sequence[Mapping[str, Any]]], ndim: int
) -> Optional[List[str]]:
    """Column names from stored ``dimension_metadata`` when it is complete."""
    if not isinstance(metadata, (list, tuple)) or len(metadata) != ndim:
        return None
    names = [
        entry.get("name") if isinstance(entry, Mapping) else None for entry in metadata
    ]
    if not all(isinstance(name, str) and name for name in names):
        return None
    return [str(name) for name in names]


def _column_names(
    ndim: int,
    metadata: Optional[Sequence[Mapping[str, Any]]],
    explicit: Optional[List[str]],
    fitting_info: Optional[Mapping[str, Any]],
) -> Optional[List[str]]:
    """Name each center column: ``--dim-order`` > stored metadata > array order.

    The array-order assumption covers stores fitted before fits were stamped, and
    ONLY those: it needs fitting provenance whose ``source_shape`` has the
    centers' rank (:func:`fit_axis_dimension_metadata`). Geometric x, y, z data
    written as ``.gsplats.zarr`` (a classical-splat import, a lifted point cloud,
    a hand-built store) carries no such grid and keeps the positional mapping.
    """
    if explicit is not None:
        return explicit
    names = _metadata_names(metadata, ndim)
    if names is not None:
        return names
    fitted = fit_axis_dimension_metadata(dict(fitting_info or {}), ndim)
    return [str(entry["name"]) for entry in fitted] if fitted else None


def fitted_column_names(
    ndim: int,
    dimension_metadata: Optional[Sequence[Mapping[str, Any]]],
    fitting_info: Optional[Mapping[str, Any]],
) -> Optional[List[str]]:
    """The store's own column names, or ``None`` for an unnamed store.

    A store is FITTED (in the sense ``add_gsplats_from_file``'s convert defaults
    use) exactly when this is not ``None``: a complete root
    ``dimension_metadata``, or fitting provenance whose ``source_shape`` has the
    centers' rank. A classical-splat import or a hand-built store has neither.
    """
    return _column_names(ndim, dimension_metadata, None, fitting_info)


def _canonical_rank(name: str) -> int:
    lowered = name.lower()
    return (
        _CANONICAL_AXES.index(lowered)
        if lowered in _CANONICAL_AXES
        else len(_CANONICAL_AXES)
    )


def _scene_order(columns: List[Dimension]) -> List[int]:
    """Column indices in scene order: displayed x, y, z first, then the rest."""
    displayed = [i for i, dim in enumerate(columns) if dim.display]
    hidden = [i for i, dim in enumerate(columns) if not dim.display]
    displayed.sort(key=lambda i: _canonical_rank(columns[i].name))
    return displayed + hidden


def _is_image_order(names: List[str]) -> bool:
    """Whether the y column precedes the x column — the array (row, column) order.

    That is what makes the y flip right: an image's rows count down. ``x, y, z``
    (the geometric order) is not image order, which is how ``--dim-order x,y,z``
    opts out of the flip.
    """
    lowered = [name.lower() for name in names]
    if "x" not in lowered or "y" not in lowered:
        return False
    return lowered.index("y") < lowered.index("x")


def _world_axis(letter: str, displayed_names: List[str]) -> int:
    """World axis (0/1/2) the dimension named ``letter`` is drawn along."""
    lowered = [name.lower() for name in displayed_names]
    if letter in lowered:
        return lowered.index(letter)
    return _CANONICAL_AXES.index(letter)


def _camera(
    view: Optional[str], image: bool, displayed_names: List[str]
) -> Optional[CameraConfig]:
    """Opening camera for ``view`` (default ``xy`` for an image), else ``None``."""
    view = view or ("xy" if image else None)
    if view is None:
        return None
    if view not in VALID_VIEW_AXES:
        raise ValueError(f"--view must be one of {VALID_VIEW_AXES}, got {view!r}")
    vertical = _world_axis(view[1], displayed_names)
    horizontal = _world_axis(view[0], displayed_names)
    if max(vertical, horizontal) >= len(displayed_names):
        raise ValueError(
            f"--view {view} needs a displayed '{view[0]}' and '{view[1]}' "
            f"dimension; the scene displays {displayed_names}"
        )
    if horizontal == vertical:
        raise ValueError(
            f"--view {view}: '{view[0]}' and '{view[1]}' resolve to the same "
            f"displayed axis among {displayed_names}"
        )
    up = [0.0, 0.0, 0.0]
    up[vertical] = -1.0 if image else 1.0
    return CameraConfig(up=(up[0], up[1], up[2]), view_axis=view)


def _names_spatial(names: List[str]) -> bool:
    """Whether the names carry at least x and y — enough to choose the display."""
    lowered = {name.lower() for name in names}
    return {"x", "y"} <= lowered


def _renamed(
    columns: Dimensions, names: List[str], extent: np.ndarray
) -> List[Dimension]:
    """Rename each column's dimension; with spatial names, re-derive display.

    ``build_dimensions_from_data`` displays the first three COLUMNS, which is
    wrong for a hidden axis that comes first (``t, z, y, x``): x would be hidden
    and t drawn. When the names include x and y, the display follows the names —
    x, y, z shown, every other axis hidden and discrete with its inferred step.
    """
    spatial = _names_spatial(names)
    renamed = []
    for i, (dim, name) in enumerate(zip(columns.dimensions, names, strict=True)):
        values = {**dim.to_dict(), "name": name}
        if spatial and name.lower() in _CANONICAL_AXES:
            values.update(display=True, discrete=False)
        elif spatial:
            step = infer_discrete_step(extent[:, i])
            values.update(display=False, discrete=True, spatial=False, step=step)
        renamed.append(Dimension.from_dict(values))
    return renamed


def resolve_convert_orientation(
    extent: np.ndarray,
    dimension_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    dim_order: Optional[str] = None,
    view: Optional[str] = None,
    fitting_info: Optional[Mapping[str, Any]] = None,
    remap: bool = True,
) -> ConvertOrientation:
    """Resolve scene dimensions, ``dim_order`` and the opening camera.

    Args:
        extent: Center coordinates (or a two-row min/max box) — ``(N, D)``.
        dimension_metadata: The store's ``dimension_metadata`` root attr.
        dim_order: ``--dim-order`` spec naming each column, e.g. ``"z,y,x"``.
        view: ``--view`` axis preset (``xy``/``xz``/``yz``); ``None`` = ``xy``
            for image-ordered columns, else the viewer default.
        fitting_info: The store's fitting statistics (``fitting/`` attrs);
            a ``source_shape`` of the centers' rank licenses the array-order
            default for an unnamed store.
        remap: Map columns onto x, y, z-ordered scene dims via ``dim_order``.
            ``False`` for a grafted partition/nested tree, which cannot be
            remapped: the scene dims then keep column order.

    Raises:
        ValueError: On a malformed ``--dim-order`` or ``--view``.
    """
    ndim = int(extent.shape[1])
    explicit = parse_dim_order(dim_order, ndim)
    columns = build_dimensions_from_data(extent, dimension_metadata)
    names = _column_names(ndim, dimension_metadata, explicit, fitting_info)
    if names is None:
        displayed = [columns.dimensions[i].name for i in columns.displayed]
        camera = _camera(view, False, displayed)
        return ConvertOrientation(
            columns, None, camera, "Columns mapped positionally (no axis names)"
        )

    renamed = _renamed(columns, names, extent)
    order = _scene_order(renamed) if remap else list(range(ndim))
    scene_dims = Dimensions([renamed[i] for i in order])
    mapping = names if order != list(range(ndim)) else None
    image = _is_image_order(names)
    displayed = [scene_dims.dimensions[i].name for i in scene_dims.displayed]
    camera = _camera(view, image, displayed)
    summary = f"Columns {names} -> scene dims {scene_dims.names}" + (
        f" (image convention: y down, facing {camera.view_axis if camera else 'xy'})"
        if image
        else ""
    )
    return ConvertOrientation(scene_dims, mapping, camera, summary)
