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
    """Stamp the default compositing attrs (opacity/gamma/intensity/offset/\
    blending_mode) in place, only where the caller did not supply them.

    Mirrors the GSplat defaults in :func:`~luxar.io._compiler.gsplat_assembly.\
    apply_gsplat_group_attrs` (which additionally defaults ``truncation_radius``).
    """
    for key, default in (
        ("opacity", 1.0),
        ("gamma", 1.0),
        ("intensity", 1.0),
        ("offset", 0.0),
        ("blending_mode", "additive"),
    ):
        if key not in attrs:
            attrs[key] = default
