"""Overlay data class for screen-space annotations in Luxar scenes.

Overlays are lightweight metadata containers representing text, image, video,
or HTML annotations positioned in normalized screen coordinates over the viewer canvas.
They are NOT part of the 3D scene graph — they exist at the Scene level only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class Overlay:
    """Metadata container for a single screen-space overlay.

    Returned by Scene.add_text(), Scene.add_image(), Scene.add_video(), and
    Scene.add_html() for optional inspection. Overlays are immediately written
    to zarr and not kept in memory beyond this lightweight descriptor.

    Attributes:
        name: Unique name of the overlay (auto-generated or user-specified)
        overlay_type: One of 'overlay_text', 'overlay_image', 'overlay_video',
            'overlay_html'
        position: (x, y) in normalized screen coordinates [0, 1], top-left origin
        attrs: All overlay attributes as written to zarr .zattrs
    """

    name: str
    overlay_type: str
    position: Tuple[float, float]
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Compact ``<Overlay ...>`` summary: name, type, 2-dp screen position."""
        return (
            f"<Overlay '{self.name}' type={self.overlay_type} "
            f"pos=({self.position[0]:.2f}, {self.position[1]:.2f})>"
        )
