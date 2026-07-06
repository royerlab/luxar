"""Protocols and generic type variables for type checking.

This module contains:
- Protocol definitions for type checking
- Generic type variables

Validation functions, dataclasses, and type guards have moved to
``luxar.validation.types``.

For simple type aliases, see aliases.py.
For enums and literal types, see enums.py.
For constants, see constants.py.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Protocol,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray

# Import type aliases from aliases module
from .aliases import (
    ColorArray,
    GroupAttrs,
    PathLike,
    PositionArray,
    SceneHierarchy,
)

if TYPE_CHECKING:
    from ..encoding.compression import CompressorLike

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


class PointsProtocol(Protocol):
    """Protocol for points data containers."""

    def __init__(
        self,
        name: str,
        positions: PositionArray,
        colors: Optional[ColorArray] = None,
        parent: Optional[NodeProtocol] = None,
        *,
        chunk_size: int = 32_768,
        compressor: "CompressorLike" = None,
        **attrs: Any,
    ) -> None:
        """Initialize points object."""
        ...


class SceneProtocol(Protocol):
    """Protocol for scene containers."""

    def add_group(self, name: str, **attrs: Any) -> NodeProtocol:
        """Add a group to the scene."""
        ...

    def add_points(
        self,
        name: str,
        positions: PositionArray,
        colors: Optional[ColorArray] = None,
        parent: Optional[NodeProtocol] = None,
        **attrs: Any,
    ) -> PointsProtocol:
        """Add points to the scene."""
        ...

    def finalize(self) -> None:
        """Finalize the scene."""
        ...

    def get_store_path(self) -> PathLike:
        """Get the scene store path."""
        ...


# =============================================================================
# Generic Type Variables and Constraints
# =============================================================================

# Generic node type
NodeT = TypeVar("NodeT", bound=NodeProtocol)

# Generic numeric array type
NumericT = TypeVar("NumericT", bound=np.generic)
ArrayT = TypeVar("ArrayT", bound=NDArray[Any])

# Zarr-compatible data types
ZarrDataT = TypeVar("ZarrDataT", np.float32, np.uint8, np.int32, np.int64, np.float64)

# =============================================================================
# Validation Functions - Moved to validation.types module
# =============================================================================
# Note: Validation functions are now in luxar.validation.types
# Import from there directly instead of from protocols
