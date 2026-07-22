"""Node-authoring blocks shared byte-for-byte by the Points and Lines writers.

These are the small, geometry-agnostic steps that :meth:`~luxar.io.compiler.\
LuxarZarrCompiler.write_points` and :meth:`~luxar.io.compiler.LuxarZarrCompiler.\
write_lines` performed identically: normalizing the ``transform`` /
``nd_transform`` attrs and stamping the default compositing attributes. Both are
pure in-place mutators of the caller's ``attrs`` dict so the on-disk output is
unchanged from the inlined versions.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet

import zarr

from ...core.dimensions import Dimensions

# Writer-authoritative attrs each geometry writer stamps unconditionally.
# User-supplied values for these keys are rejected in the fail-fast gate:
# letting them through would either silently lose the user's value (the stamp
# wins on disk) or blow up post-write with an accidental TypeError when the
# Node object is constructed (``type=`` collides with the Node constructor).
# The ``ordering*`` stamps are deliberately NOT reserved: ``ordering=`` is an
# accepted (stamped-over) call pattern in existing code and tests.
POINTS_RESERVED_ATTRS: FrozenSet[str] = frozenset(
    {
        "type",
        "n_points",
        "ndim",
        "has_colors",
        "has_radii",
        "has_sharpness",
        "has_scalars",
        "has_labels",
        "position_bounds",
        "max_radius",
    }
)
LINES_RESERVED_ATTRS: FrozenSet[str] = frozenset(
    {
        "type",
        "n_vertices",
        "n_segments",
        "ndim",
        "original_line_type",
        "has_colors",
        "has_sharpness",
        "has_scalars",
        "has_labels",
        "max_width",
        "position_bounds",
    }
)
GSPLATS_RESERVED_ATTRS: FrozenSet[str] = frozenset(
    {
        "type",
        "n_splats",
        "ndim",
        "has_colors",
        "has_labels",
        "amplitude_range",
        "amplitude_data_range",
        "center_bounds",
        "position_bounds",
    }
)


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
    for segment in stripped.split("/"):
        validate_node_name(segment, context=f"node path {path!r}")
    return stripped


def validate_broadcast_color(colors: Any, context: str = "colors") -> None:
    """Validate a uniform broadcast color (list/tuple) BEFORE any zarr write.

    The encoder broadcasts a color list/tuple to all elements; a wrong-length
    or non-numeric tuple used to die inside the encoder AFTER the positions
    were already written. Shared by the Points and Lines writers' fail-fast
    gates (per the three-geometry symmetry rule).

    Raises:
        ValidationError: If the broadcast color is not a finite numeric
            RGB (3) or RGBA (4) sequence.
    """
    from ...validation.base import ValidationError

    if len(colors) not in (3, 4):
        raise ValidationError(
            f"{context}: Uniform color must have 3 (RGB) or 4 (RGBA) "
            f"components, got {len(colors)}",
            "Pass e.g. colors=(1.0, 0.0, 0.0) for a uniform red",
        )
    import numpy as np

    for i, component in enumerate(colors):
        # np.floating/np.integer included: np.float32 does NOT subclass
        # Python float (np.float64 does), and float32 tuple components — e.g.
        # tuple(color_array[i]) — are a legitimate caller pattern.
        if not isinstance(component, (int, float, np.integer, np.floating)) or not np.isfinite(
            component
        ):
            raise ValidationError(
                f"{context}: Uniform color component {i} must be a finite "
                f"number, got {component!r}",
                "Use finite numeric RGB(A) components",
            )


def validate_scalars_preflight(
    scalars: Any, n_elements: int, context: str = "scalars"
) -> None:
    """Validate colormap scalars (array or broadcast scalar) BEFORE any write.

    The scalars dataset writer runs late in the pipeline, so a wrong-length or
    NaN scalars input used to leave a partial node behind. Shared by the
    Points and Lines writers' fail-fast gates.

    Raises:
        ValidationError: If scalars are not finite numeric values with one
            entry per element (or a single broadcast value).
    """
    import numpy as np

    from ...validation.base import (
        ValidationError,
        _validate_numeric_finite_values,
    )

    if isinstance(scalars, (int, float)):
        if not np.isfinite(scalars):
            raise ValidationError(
                f"{context}: Scalar value must be finite. Got {scalars}",
                "Provide a finite scalar value",
            )
        return

    if not isinstance(scalars, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array or scalar, "
            f"got {type(scalars).__name__}",
            "Convert to numpy array: np.array(scalars)",
        )

    if scalars.ndim != 1 or (
        scalars.shape[0] != n_elements and scalars.shape[0] != 1
    ):
        raise ValidationError(
            f"{context}: Expected 1D array with {n_elements} values or 1 "
            f"(broadcast), got shape {scalars.shape}",
            f"Provide exactly {n_elements} scalar values or a single value",
        )

    _validate_numeric_finite_values(scalars, context)


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
    apply_gsplat_group_attrs` (which additionally defaults ``truncation_radius``).
    """
    for key, default in (
        ("opacity", 1.0),
        ("absorption", 1.0),
        ("gamma", 1.0),
        ("intensity", 1.0),
        ("offset", 0.0),
    ):
        if key not in attrs:
            attrs[key] = default


def validate_render_attrs(
    attrs: Dict[str, Any],
    reserved_attrs: FrozenSet[str] = frozenset(),
) -> None:
    """Validate render attrs that would corrupt a node if written unchecked.

    Called as the FIRST step of every geometry writer — before the zarr group
    is created — so an invalid value fails the write without leaving a partial
    node on disk. Covers every pure attr validator (no store access needed):
    blending_mode / absorption / opacity / gamma / intensity / offset / layer /
    visible / colormap. The values are validated only (not converted) — the
    writer stores the caller's attrs unchanged.

    Args:
        attrs: The node attrs dict to validate.
        reserved_attrs: Writer-stamped keys the caller must not supply (see
            ``POINTS_RESERVED_ATTRS`` / ``LINES_RESERVED_ATTRS`` /
            ``GSPLATS_RESERVED_ATTRS``). A collision fails the write up front
            instead of being silently overwritten by the writer's stamps (or
            exploding post-write in the Node constructor).
    """
    if reserved_attrs:
        collisions = sorted(reserved_attrs & attrs.keys())
        if collisions:
            raise ValueError(
                f"Attribute(s) {collisions} are reserved: the writer stamps "
                f"them authoritatively (type, element counts, presence flags, "
                f"bounds, ...). Remove them from the node attrs."
            )

    if "blending_mode" in attrs:
        from ...validation.types import validate_blending_mode

        validate_blending_mode(attrs["blending_mode"])

    if "absorption" in attrs:
        from ...validation.types import validate_absorption

        validate_absorption(attrs["absorption"])

    if "opacity" in attrs:
        from ...validation.types import validate_opacity

        validate_opacity(attrs["opacity"])

    if "gamma" in attrs:
        from ...validation.types import validate_gamma

        validate_gamma(attrs["gamma"])

    if "intensity" in attrs:
        from ...validation.types import validate_intensity

        validate_intensity(attrs["intensity"])

    if "offset" in attrs:
        from ...validation.types import validate_offset

        validate_offset(attrs["offset"])

    if "layer" in attrs:
        from ...validation.types import validate_layer

        validate_layer(attrs["layer"])

    if "visible" in attrs:
        from ...validation.types import validate_visible

        validate_visible(attrs["visible"])

    if "colormap" in attrs and attrs["colormap"] is not None:
        from ...validation.types import validate_colormap

        validate_colormap(attrs["colormap"])
