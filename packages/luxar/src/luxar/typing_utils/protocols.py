"""Protocols for structural (duck) typing.

Both protocols here have a real consumer, and that is the bar for living in this
module — an unused ``Protocol`` type-checks nothing, so it cannot rot loudly:

- :class:`CompressorProtocol` is one arm of ``luxar.encoding.compression``'s
  ``CompressorLike`` union, which annotates every compressor parameter in the
  writing path.
- :class:`NodeProtocol` is the element type of ``Node.walk()``'s yielded pairs
  (``core.node.node`` casts to it), and so the type external code annotates
  against when it walks a scene graph.

Six further names were removed in the Phase 4 cleanup because nothing anywhere
in the repo referenced them: ``SceneProtocol``, ``PointsProtocol`` and the
``NodeT`` / ``NumericT`` / ``ArrayT`` / ``ZarrDataT`` type variables. The two
protocols in particular described a ``Scene``/``Points`` API that had drifted
from the real one (no ``add_lines``, ``add_gsplats`` or ``add_mesh``), which is
the specific hazard of an interface no implementation is checked against.

Validation functions, dataclasses, and type guards live in
``luxar.validation.types``. For simple type aliases see ``aliases.py``, for
enums and literal types ``enums.py``, for constants ``constants.py``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol

from .aliases import GroupAttrs, SceneHierarchy

# =============================================================================
# Protocol Definitions
# =============================================================================


class CompressorProtocol(Protocol):
    """Protocol for Zarr compressor objects."""

    def encode(self, buf: Any) -> bytes:
        """Encode data buffer."""
        ...

    def decode(self, buf: bytes, out: Optional[Any] = None) -> Any:
        """Decode data buffer."""
        ...


class NodeProtocol(Protocol):
    """Protocol for scene graph nodes."""

    name: str
    children: List[NodeProtocol]
    parent: Optional[NodeProtocol]

    @property
    def attrs(self) -> GroupAttrs:
        """Node attributes."""
        ...

    def add_group(self, name: str, **attrs: Any) -> NodeProtocol:
        """Add a child group node."""
        ...

    def walk(self, depth: int = 0) -> SceneHierarchy:
        """Walk the node hierarchy depth-first."""
        ...
