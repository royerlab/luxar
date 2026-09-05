"""Node-authoring blocks shared byte-for-byte by the geometry writers.

These are the small, geometry-agnostic steps that :meth:`~luxar.io.compiler.\
LuxarZarrCompiler.write_points`, :meth:`~luxar.io.compiler.LuxarZarrCompiler.\
write_lines` and :meth:`~luxar.io.compiler.LuxarZarrCompiler.write_mesh` perform
identically: normalizing the ``transform`` / ``nd_transform`` attrs and stamping
the default compositing attributes. Both are pure in-place mutators of the
caller's ``attrs`` dict so the on-disk output is unchanged from the inlined
versions.

Also home to the per-geometry ``*_RESERVED_ATTRS`` sets — the writer-authoritative
keys a caller must not supply.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import zarr
from numpy.typing import NDArray

from ...core.dimensions import Dimensions
from ...core.group.compositing import (
    IDENTITY_COMPOSITING_ATTRS,
    WRITER_STAMPED_APPEARANCE_DEFAULTS,
)
from ...typing_utils.constants import (
    ELEMENT_TEXELS_PER_ELEMENT,
    max_elements_per_node,
)


class ElementCapacityWarning(UserWarning):
    """A node may exceed the viewer's per-node element-texture capacity."""


def validate_node_path(path: str) -> str:
    """Validate a writer node path and return it with the leading ``/`` stripped.

    The compiler-side half of the node-name chokepoint (see
    :func:`~luxar.validation.base.validate_node_name`): every ``/``-separated
    segment of the path must be a valid node name. In particular an EMPTY path
    (or an empty segment, e.g. from ``add_points("", ...)``) resolves to the
    zarr ROOT group via ``require_group("")`` — writing a geometry node there
    stamps ``type='points'`` onto the scene root and makes the store
    unloadable — and dot-prefixed segments collide with zarr's reserved
    metadata keys (``.zgroup``/``.zattrs``/``.zarray``/``.zmetadata``).

    The exact single-segment root path ``overlays`` is likewise rejected: it is
    reserved for screen-space overlay metadata (written internally as
    ``overlays/<name>``), so a raw geometry writer must not claim it. Only the
    bare root name collides; ``overlays/<name>`` and nested paths like
    ``geometry/overlays`` are allowed.

    Raises:
        ValidationError: If the path or any of its segments is invalid.
    """
    from ...validation.base import ValidationError, validate_node_name

    if not isinstance(path, str):
        raise ValidationError(
            f"node path: Expected a string, got {type(path).__name__}"
        )
    stripped = path.lstrip("/")
    if not stripped.strip():
        raise ValidationError(
            f"node path: Path must not be empty (got {path!r}). An empty path "
            "resolves to the zarr ROOT group and would overwrite the scene root.",
            "Provide a non-empty node name/path",
        )
    if stripped == "overlays":
        raise ValidationError(
            "node path: Top-level node path 'overlays' is reserved for "
            "screen-space overlay metadata. Write internal overlays below "
            "'overlays/<name>' or choose a different user node name.",
            "Choose a different top-level node name",
        )
    for segment in stripped.split("/"):
        validate_node_name(segment, context=f"node path {path!r}")
    return stripped


def prepare_transform_attrs(attrs: Dict[str, Any], store: zarr.Group) -> None:
    """Normalize ``transform`` / ``nd_transform`` attrs in place.

    ``transform`` is round-tripped through :func:`~luxar.core.transforms.\
    prepare_transform_for_zarr` (NumPy→THREE.js column-major); ``nd_transform``
    is validated against the scene dimensions read from ``store.attrs``. No-op
    when neither key is present.
    """
    if "transform" in attrs:
        from ...core.transforms import prepare_transform_for_zarr

        attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])

    if "nd_transform" in attrs:
        from ...validation.nd_transforms import validate_nd_transform

        dims = None
        if "scene_dimensions" in store.attrs:
            dims = Dimensions.from_dict(store.attrs["scene_dimensions"])
        attrs["nd_transform"] = validate_nd_transform(attrs["nd_transform"], dims)


def record_forwarded_sort_order(
    metadata: Dict[str, Any],
    requested: bool,
    sort_order: Optional[NDArray[np.integer]],
) -> None:
    """Record this node's spatial permutation in its metadata, on request.

    Backs the private ``_return_sort_order`` forwarding flag on ``write_points``
    / ``write_lines``: the multi-LOD parent writers need each level's spatial
    permutation to build the ladder's union label CSR, and the permutation is
    never persisted on disk. OPT-IN so the flat write path does not park a big
    index array in the compiler's metadata cache.

    Args:
        metadata: The writer's metadata dict, mutated in place.
        requested: The popped ``_return_sort_order`` flag.
        sort_order: The node's permutation, or ``None`` for no spatial
            reordering (identity).
    """
    if requested:
        metadata["sort_order"] = sort_order


def apply_default_render_attrs(attrs: Dict[str, Any]) -> None:
    """Stamp the default compositing attrs (opacity/absorption/gamma/intensity/
    offset) in place, only where the caller did not supply them.

    ``blending_mode`` is deliberately NOT stamped: unlike these identity-valued
    attrs (multiplicative/additive no-ops under the viewer's hierarchical
    composition), a stamped blending default would OVERRIDE an ancestor-set
    mode under the viewer's nearest-setter-wins rule. An unset leaf inherits
    from the nearest ancestor; the viewer defaults to ``additive`` when no
    ancestor sets it.

    Mirrors the GSplat defaults in :func:`~luxar.io._compiler.gsplat_assembly.\
    apply_gsplat_group_attrs` (which additionally defaults ``truncation_radius``)
    — literally, not by coincidence: both read their values from
    :data:`~luxar.core.group.compositing.WRITER_STAMPED_APPEARANCE_DEFAULTS`,
    which is also what the READER consults to tell a manufactured identity from
    an authored one (the ``gsplat merge`` agreement rule).
    """
    for key in IDENTITY_COMPOSITING_ATTRS:
        if key not in attrs:
            attrs[key] = WRITER_STAMPED_APPEARANCE_DEFAULTS[key]


def warn_if_over_element_cap(
    geometry_type: str,
    count: int,
    node_path: str,
    *,
    enabled: bool = True,
) -> bool:
    """Warn when a node holds more elements than one node can render (#1957).

    The viewer packs per-element render data into an element texture and
    CLAMPS a node that overflows it — ``clampElementCapacity`` drops the tail
    with a single console warning and no other signal. Geometry is stored in
    Hilbert order, so the lost tail is one spatially CONTIGUOUS lobe: the
    symptom is a clean-edged wedge of missing geometry, which reads as a data
    or masking bug rather than a capacity limit. #1957 lost the entire North
    Atlantic to a 2.3% overflow this way.

    The cap depends on the viewer's GPU, so the only bound an author can rely
    on is the 4096-class floor in
    :func:`~luxar.typing_utils.constants.max_elements_per_node`. Warning here
    puts the diagnosis at the point where it is cheap to act on — while the
    data is being authored — instead of leaving it to whoever opens the scene
    on a smaller GPU months later.

    A warning, not an error: a node above the floor still renders whole on a
    16384-class GPU, so refusing to write it would reject data that works.

    Args:
        geometry_type: A key of ``ELEMENT_TEXELS_PER_ELEMENT`` ("points",
            "lines", "gsplats"). Any other type is a no-op.
        count: Elements in this node — segments for lines, points for points,
            splats for gsplats.
        node_path: The node's path, for the message.
        enabled: False to suppress a redundant child warning when its parent
            checks the aggregate count.

    Warns:
        ElementCapacityWarning: If a supported geometry node exceeds the
            conservative viewer capacity floor.

    Returns:
        True if a warning was emitted.
    """
    if not enabled or geometry_type not in ELEMENT_TEXELS_PER_ELEMENT:
        return False
    cap = max_elements_per_node(geometry_type)
    if count <= cap:
        return False
    noun = {"points": "points", "lines": "segments", "gsplats": "splats"}[geometry_type]
    remedy = (
        "Run `luxar gsplat lod --recipe tiles`, or use "
        "partition=dict(max_elements=...) from the scene API"
        if geometry_type == "gsplats"
        else "Split it with partition=dict(max_elements=...)"
    )
    message = (
        f"'{node_path}' holds {count:,} {noun}, above the {cap:,} a single "
        f"{geometry_type} node can render on a 4096-class GPU. If the whole "
        f"node is committed at once, such a GPU can silently drop the tail — "
        f"and because elements are stored in Hilbert "
        f"order, that tail is one contiguous region, so it looks like a "
        f"clean-edged hole in the data (#1957). An nD node sliced on a "
        f"non-displayed dimension commits only its current slice. {remedy} "
        f"to render everywhere."
    )
    warnings.warn(message, ElementCapacityWarning, stacklevel=1)
    return True
