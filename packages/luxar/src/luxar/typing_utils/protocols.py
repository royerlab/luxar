"""Protocols, dataclasses, validation functions, and type guards.

This module contains:
- Protocol definitions for type checking
- Dataclasses for structured data
- Validation functions for runtime type checking
- Type guards for conditional type narrowing
- Generic type variables

For simple type aliases, see aliases.py.
For enums and literal types, see enums.py.
For constants, see constants.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

# Python 3.10+ compatibility
if sys.version_info >= (3, 10):
    pass
else:
    pass

from ..typing_utils.constants import GAMMA_MAX, GAMMA_MIN, OPACITY_MAX, OPACITY_MIN
from ..typing_utils.enums import BlendingMode, NodeType, PhysicalUnit

# Import type aliases from aliases module
from .aliases import (
    ColorArray,
    GroupAttrs,
    PathLike,
    PositionArray,
    SceneHierarchy,
    TransformMatrix,
)

# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class DimensionMetadata:
    """Metadata for a single dimension in nD data.

    Attributes:
        name: Name of the dimension (e.g., "x", "y", "z", "time", "channel")
        unit: Physical unit of the dimension (e.g., "um", "ms", "nm")
        scale: Scale factor for the dimension (default: 1.0)
        range: Optional (min, max) bounds for this dimension
    """

    name: str = ""
    unit: str = ""
    scale: float = 1.0
    range: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {"name": self.name, "unit": self.unit, "scale": self.scale}
        if self.range is not None:
            data["range"] = list(self.range)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DimensionMetadata:
        """Create from dictionary."""
        range_val = data.get("range")
        if range_val is not None:
            range_val = tuple(range_val)
        return cls(
            name=data.get("name", ""),
            unit=data.get("unit", ""),
            scale=data.get("scale", 1.0),
            range=range_val,
        )


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


class ZarrGroupProtocol(Protocol):
    """Protocol for Zarr group objects."""

    @property
    def attrs(self) -> GroupAttrs:
        """Group attributes."""
        ...

    @property
    def basename(self) -> Optional[str]:
        """Base name of the group."""
        ...

    @property
    def store(self) -> Any:
        """Zarr store backing this group."""
        ...

    def require_group(self, name: str) -> ZarrGroupProtocol:
        """Require a subgroup."""
        ...

    def create_dataset(
        self,
        name: str,
        *,
        data: Optional[NDArray[Any]] = None,
        chunks: Optional[Union[int, Tuple[int, ...]]] = None,
        compressor: Optional[CompressorProtocol] = None,
        dtype: Optional[Any] = None,
        overwrite: bool = False,
    ) -> Any:
        """Create a dataset in this group."""
        ...

    def group_keys(self) -> List[str]:
        """Get list of subgroup keys."""
        ...

    def __getitem__(self, key: str) -> Union[ZarrGroupProtocol, Any]:
        """Get subgroup or dataset by key."""
        ...

    def __contains__(self, key: str) -> bool:
        """Check if key exists in group."""
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


class PointCloudProtocol(Protocol):
    """Protocol for point cloud data containers."""

    def __init__(
        self,
        name: str,
        positions: PositionArray,
        colors: Optional[ColorArray] = None,
        parent: Optional[NodeProtocol] = None,
        *,
        chunk_size: int = 32_768,
        compressor: Optional[CompressorProtocol] = None,
        **attrs: Any,
    ) -> None:
        """Initialize point cloud."""
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
    ) -> PointCloudProtocol:
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
# Validation Functions
# =============================================================================


def validate_positions(positions: Any, ndim: Optional[int] = None) -> PositionArray:
    """Validate and convert positions array to correct type.

    Args:
        positions: Input array to validate
        ndim: Expected number of dimensions (optional). If None, any dimensionality is accepted.

    Returns:
        Validated positions array with shape (N, D)

    Raises:
        ValueError: If positions are invalid shape or type
    """
    if not isinstance(positions, np.ndarray):
        raise ValueError("Positions must be a numpy array")

    if positions.ndim != 2:
        raise ValueError(
            f"Positions must have shape (N, D), got shape {positions.shape}"
        )

    if positions.shape[1] < 1:
        raise ValueError(
            f"Positions must have at least 1 dimension, got {positions.shape[1]}"
        )

    if ndim is not None and positions.shape[1] != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got {positions.shape[1]}")

    return positions.astype(np.float32, copy=False)


def validate_colors(
    colors: Any, n_points: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Validate and convert colors array to HDR float32 format.

    Args:
        colors: Input array to validate
        n_points: Expected number of points

    Returns:
        Validated colors array in HDR float32 format

    Raises:
        ValueError: If colors are invalid shape or type
    """
    if not isinstance(colors, np.ndarray):
        raise ValueError("Colors must be a numpy array")

    if colors.shape != (n_points, 3):
        raise ValueError(f"Colors must have shape ({n_points}, 3)")

    # Support HDR colors - use float32 for full HDR range
    # Colors can be any positive value (0.0 to infinity) for HDR emission
    return colors.astype(np.float32, copy=False)


def validate_radii(radii: Any, n_points: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Validate and convert radii array to correct type.

    Args:
        radii: Input array to validate
        n_points: Expected number of points
    Returns:
        Validated radii array
    Raises:
        ValueError: If radii are invalid shape or type
    """
    if not isinstance(radii, np.ndarray):
        raise ValueError("Radii must be a numpy array")
    if radii.ndim != 1:
        raise ValueError("Radii must have shape (N,)")
    if radii.shape[0] != n_points:
        raise ValueError(f"Radii shape {radii.shape} doesn't match positions")
    # Ensure all radii are positive
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive values")
    return radii.astype(np.float32, copy=False)


def validate_sharpness(
    sharpness: Any, n_points: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Validate and convert sharpness array to correct type.

    Sharpness controls the falloff profile of points, from soft (low values) to sharp (high values).
    Must be positive float32 values with shape (N,) where N is the number of points.
    Typical range is 0.5 to 10.0.

    Args:
        sharpness: Input sharpness array to validate
        n_points: Expected number of points

    Returns:
        Validated sharpness array as float32

    Raises:
        ValueError: If sharpness values are invalid
    """
    if not isinstance(sharpness, np.ndarray):
        raise ValueError("Sharpness must be a numpy array")

    if sharpness.ndim != 1:
        raise ValueError("Sharpness must have shape (N,)")

    if sharpness.shape[0] != n_points:
        raise ValueError(f"Sharpness shape {sharpness.shape} doesn't match positions")

    if np.any(sharpness <= 0):
        raise ValueError("All sharpness values must be positive")

    # Warn if values are outside typical range
    if np.any(sharpness < 0.5) or np.any(sharpness > 10.0):
        import warnings

        warnings.warn(
            "Sharpness values outside typical range [0.5, 10.0] detected. "
            "Very low values (<0.5) create uniform disks, very high values (>10) create hard edges."
        )

    return sharpness.astype(np.float32, copy=False)


def validate_transform(transform: Any) -> TransformMatrix:
    """Validate and convert transform matrix to correct type.

    Args:
        transform: Input transform matrix

    Returns:
        Validated 4x4 transform matrix

    Raises:
        ValueError: If transform is invalid shape or type
    """
    if not isinstance(transform, np.ndarray):
        raise ValueError("Transform must be a numpy array")

    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix")

    return transform.astype(np.float32, copy=False)


def validate_node_type(node_type: str) -> NodeType:
    """Validate node type string.

    Args:
        node_type: Input node type string

    Returns:
        Validated node type

    Raises:
        ValueError: If node type is invalid
    """
    valid_types = (NodeType.POINTS.value, NodeType.GROUP.value, NodeType.SCENE.value)
    if node_type not in valid_types:
        raise ValueError(
            f"Invalid node type '{node_type}'. Must be one of {valid_types}"
        )
    # Cast is safe after validation
    from typing import cast

    return cast(NodeType, node_type)


def validate_physical_unit(unit: str) -> PhysicalUnit:
    """Validate physical unit string.

    Args:
        unit: Input unit string

    Returns:
        Validated physical unit

    Raises:
        ValueError: If unit is invalid
    """
    valid_units = (
        PhysicalUnit.NANOMETER.value,
        PhysicalUnit.MICROMETER.value,
        PhysicalUnit.MILLIMETER.value,
        PhysicalUnit.CENTIMETER.value,
        PhysicalUnit.METER.value,
        PhysicalUnit.METRE.value,
        "meter",  # Not in enum but needed for compatibility
        PhysicalUnit.KILOMETER.value,
        PhysicalUnit.INCH.value,
        PhysicalUnit.FOOT.value,
        PhysicalUnit.PIXEL.value,
        PhysicalUnit.ASTRONOMICAL_UNIT.value,
    )
    if unit not in valid_units:
        raise ValueError(f"Invalid unit '{unit}'. Must be one of {valid_units}")
    # Cast is safe after validation
    from typing import cast

    return cast(PhysicalUnit, unit)


def validate_opacity(opacity: Any) -> float:
    """Validate and convert opacity value.

    Args:
        opacity: Value to validate as opacity (0.0 to 1.0)

    Returns:
        Valid opacity as float

    Raises:
        ValueError: If opacity is not a valid float between 0 and 1
        TypeError: If opacity cannot be converted to float
    """
    try:
        opacity_float = float(opacity)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Opacity must be convertible to float, got {type(opacity).__name__}"
        ) from e

    if not OPACITY_MIN <= opacity_float <= OPACITY_MAX:
        raise ValueError(
            f"Opacity must be between {OPACITY_MIN} and {OPACITY_MAX}, got {opacity_float}"
        )

    return opacity_float


def validate_gamma(gamma: Any) -> float:
    """Validate and convert gamma value.

    Args:
        gamma: Value to validate as gamma (0.2 to 2.0)

    Returns:
        Valid gamma as float

    Raises:
        ValueError: If gamma is not within valid range
        TypeError: If gamma cannot be converted to float
    """
    try:
        gamma_float = float(gamma)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Gamma must be convertible to float, got {type(gamma).__name__}"
        ) from e

    if not GAMMA_MIN <= gamma_float <= GAMMA_MAX:
        raise ValueError(
            f"Gamma must be between {GAMMA_MIN} and {GAMMA_MAX}, got {gamma_float}"
        )

    return gamma_float


def validate_blending_mode(mode: Any) -> BlendingMode:
    """Validate blending mode string.

    Args:
        mode: Blending mode to validate

    Returns:
        Valid blending mode

    Raises:
        ValueError: If mode is not a valid blending mode
        TypeError: If mode is not a string
    """
    if not isinstance(mode, str):
        raise TypeError(f"Blending mode must be a string, got {type(mode).__name__}")

    valid_modes = {"normal", "additive"}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid blending mode '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}"
        )

    return cast(BlendingMode, mode)


def validate_dimension_metadata(
    metadata: List[Any], ndim: int
) -> List[DimensionMetadata]:
    """Validate dimension metadata list.

    Args:
        metadata: List of dimension metadata (dicts or DimensionMetadata objects)
        ndim: Expected number of dimensions

    Returns:
        List of validated DimensionMetadata objects

    Raises:
        ValueError: If metadata is invalid
    """
    if not isinstance(metadata, list):
        raise ValueError("Dimension metadata must be a list")

    if len(metadata) != ndim:
        raise ValueError(
            f"Expected {ndim} dimension metadata entries, got {len(metadata)}"
        )

    validated = []
    for i, item in enumerate(metadata):
        if isinstance(item, DimensionMetadata):
            validated.append(item)
        elif isinstance(item, dict):
            validated.append(DimensionMetadata.from_dict(item))
        else:
            raise ValueError(
                f"Dimension metadata entry {i} must be dict or DimensionMetadata"
            )

    return validated


# =============================================================================
# Type Guards
# =============================================================================


def is_position_array(obj: Any) -> bool:
    """Check if object is a valid position array."""
    try:
        validate_positions(obj)
        return True
    except (ValueError, TypeError):
        return False


def is_color_array(obj: Any, n_points: int) -> bool:
    """Check if object is a valid color array."""
    try:
        validate_colors(obj, n_points)
        return True
    except (ValueError, TypeError):
        return False


def is_transform_matrix(obj: Any) -> bool:
    """Check if object is a valid transform matrix."""
    try:
        validate_transform(obj)
        return True
    except (ValueError, TypeError):
        return False
