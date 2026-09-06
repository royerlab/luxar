"""luxar.sound – Defines the Sound node: an audio clip placed in a Luxar scene."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from arbol import aprint

from ..typing_utils.enums import NodeType
from .datanode import DataNode

if TYPE_CHECKING:
    from ..io.writer import ZarrWriterProtocol


class Sound(DataNode):
    """Sound node: an opaque MP3/AAC clip, optionally positioned in nD.

    Like the geometry nodes this is a lightweight metadata container — the clip
    and the optional ``positions`` array are written to Zarr immediately by the
    writer and never kept in memory. A sound node is *heard* rather than drawn:
    it has no appearance attrs, contributes nothing to the scene bounds, and is
    audible only where the slab rule on its hidden-dimension coordinates says so
    (``docs/guides/specs/SOUND_SPEC.md`` §3).

    Sounds are created by :meth:`luxar.Group.add_sound` and should not be
    instantiated directly by users.

    Args:
        name: Name of the sound node
        metadata: Metadata dictionary about the written clip
        parent: Parent node in hierarchy
        writer: Writer interface for progressive writing
        **attrs: Compositing attributes (``layer`` / ``visible`` / ``transform``
            / ``nd_transform``)
    """

    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional["DataNode"] = None,
        writer: Optional["ZarrWriterProtocol"] = None,
        **attrs: Any,
    ) -> None:
        """Initialize a Sound node.

        Args:
            name: Name of the sound node
            metadata: Metadata dictionary about the written clip
            parent: Parent node in the scene graph
            writer: Writer interface for progressive writing
            **attrs: Additional attributes for the node
        """
        super().__init__(
            name,
            parent=parent,
            writer=writer,
            type=NodeType.SOUND.value,
            metadata=metadata,
            **attrs,
        )

        if metadata:
            where = (
                f"{metadata.get('n_positions', 0)} position(s)"
                if metadata.get("spatial")
                else "non-spatial"
            )
            aprint(
                f"✓ Sound node '{name}' created ({metadata.get('format', '?')}, "
                f"{metadata.get('trigger', '?')}, bus {metadata.get('bus', '?')}, "
                f"{where})."
            )

    @property
    def n_elements(self) -> int:
        """Number of source positions (0 for a non-spatial clip)."""
        return int(self._metadata.get("n_positions", 0))

    @property
    def n_positions(self) -> int:
        """Number of rows in the ``positions`` array (0 when absent)."""
        return self.n_elements

    @property
    def spatial(self) -> bool:
        """True when the clip plays through a panner at its position(s)."""
        return bool(self._metadata.get("spatial", False))

    @property
    def trigger(self) -> str:
        """Playback trigger (``continuous`` or ``once``)."""
        return str(self._metadata.get("trigger", "continuous"))

    @property
    def bus(self) -> str:
        """Mixer bus the clip is routed to (``ambient`` / ``voice`` / ``effects``)."""
        return str(self._metadata.get("bus", "ambient"))

    @property
    def attach_to(self) -> Optional[str]:
        """Name of the node whose bounding-box centre the source follows, if any."""
        value = self._metadata.get("attach_to")
        return str(value) if isinstance(value, str) else None

    @property
    def format(self) -> str:
        """Clip codec as sniffed by the writer (``mp3`` or ``aac``)."""
        return str(self._metadata.get("format", ""))

    @property
    def audio_file(self) -> str:
        """Filename of the clip inside the node group (``audio.mp3`` / ``audio.m4a``)."""
        return str(self._metadata.get("audio_file", ""))

    @property
    def duration_ms(self) -> Optional[float]:
        """Clip duration in ms when the writer could measure it, else ``None``."""
        value = self._metadata.get("duration_ms")
        return None if value is None else float(value)
