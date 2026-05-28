"""Overlay write internals.

Called by the overlay adders to (a) generate or validate the overlay
name and (b) persist the overlay's attrs + optional image bytes into
the zarr store.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ...overlay import Overlay

if TYPE_CHECKING:
    from ..scene import Scene


def next_overlay_name(scene: "Scene", name: Optional[str]) -> str:
    """Generate or validate an overlay name."""
    if name is None:
        name = f"overlay_{scene._overlay_counter}"
        scene._overlay_counter += 1
    else:
        if "/" in name:
            raise ValueError(f"Overlay name cannot contain '/': got '{name}'")
    # Check for duplicate names
    existing_names = {o.name for o in scene._overlays}
    if name in existing_names:
        raise ValueError(
            f"Overlay name '{name}' already exists. Use a unique name."
        )
    return name


def write_overlay(
    scene: "Scene",
    name: str,
    overlay_type: str,
    position: Tuple[float, float],
    attrs: Dict[str, Any],
    image_data: Optional[bytes] = None,
    image_filename: Optional[str] = None,
) -> Overlay:
    """Write overlay metadata (and optional image) to the zarr store.

    Creates an ``overlays/{name}`` group with metadata in ``.zattrs``.
    For image overlays, writes the image file directly to the zarr directory.
    """
    overlay_path = f"overlays/{name}"

    # Write group with all overlay attributes
    if scene._writer is not None:
        scene._writer.write_group(overlay_path, **attrs)

        # Write raw image file if provided
        if image_data is not None and image_filename is not None:
            store_path = Path(scene._writer.store_path)
            image_dir = store_path / "overlays" / name
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / image_filename
            image_path.write_bytes(image_data)

    overlay = Overlay(
        name=name,
        overlay_type=overlay_type,
        position=position,
        attrs=attrs,
    )
    scene._overlays.append(overlay)
    return overlay
