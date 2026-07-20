"""Node-authoring blocks shared byte-for-byte by the Points and Lines writers.

These are the small, geometry-agnostic steps that :meth:`~luxar.io.compiler.\
LuxarZarrCompiler.write_points` and :meth:`~luxar.io.compiler.LuxarZarrCompiler.\
write_lines` performed identically: normalizing the ``transform`` /
``nd_transform`` attrs and stamping the default compositing attributes. Both are
pure in-place mutators of the caller's ``attrs`` dict so the on-disk output is
unchanged from the inlined versions.
"""

from __future__ import annotations

from typing import Any, Dict

import zarr

from ...core.dimensions import Dimensions


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


def validate_render_attrs(attrs: Dict[str, Any]) -> None:
    """Validate render attrs that would corrupt a node if written unchecked.

    Called as the FIRST step of every geometry writer — before the zarr group
    is created — so an invalid value fails the write without leaving a partial
    node on disk.
    """
    if "blending_mode" in attrs:
        from ...validation.types import validate_blending_mode

        validate_blending_mode(attrs["blending_mode"])

    if "absorption" in attrs:
        from ...validation.types import validate_absorption

        validate_absorption(attrs["absorption"])
