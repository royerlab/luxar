"""Scene-level dimension definitions for Luxar.

This module provides classes for defining the coordinate system of a scene,
including dimension names, units, ranges, and navigation properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Dimension:
    """Definition of a single dimension in a scene.

    Attributes:
        name: Identifier for the dimension (e.g., "x", "time", "channel")
        unit: Physical unit (e.g., "um", "s", "px")
        range: Optional min/max values as tuple
        step: Default step size for navigation (None = auto-calculate)
        display: Whether dimension should be displayed (max 3 can be True)
        discrete: Whether dimension has discrete values (for channels, indices)
        cyclic: Whether dimension wraps around (for angles)
        scale: Physical scale factor (default 1.0)
        description: Optional human-readable description
    """

    name: str
    unit: str = ""
    range: Optional[Tuple[float, float]] = None
    step: Optional[float] = None
    display: bool = True
    discrete: bool = False
    cyclic: bool = False
    scale: float = 1.0
    description: str = ""

    def __post_init__(self):
        """Validate dimension parameters."""
        if self.range is not None:
            if len(self.range) != 2:
                raise ValueError("Range must be a tuple of (min, max)")
            if self.range[0] >= self.range[1]:
                raise ValueError(
                    f"Invalid range {self.range}: min must be less than max"
                )

        if self.step is not None and self.step <= 0:
            raise ValueError(f"Step size must be positive, got {self.step}")

        if self.scale <= 0:
            raise ValueError(f"Scale must be positive, got {self.scale}")

    def get_step(self) -> float:
        """Get the step size for navigation.

        Returns auto-calculated step if not explicitly set.
        """
        if self.step is not None:
            return self.step

        # Auto-calculate step size
        if self.discrete:
            return 1.0

        if self.range is not None:
            # Use 1% of range as default step
            return (self.range[1] - self.range[0]) * 0.01

        # Default fallback
        return 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "unit": self.unit,
            "range": list(self.range) if self.range else None,
            "step": self.step,
            "display": self.display,
            "discrete": self.discrete,
            "cyclic": self.cyclic,
            "scale": self.scale,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dimension:
        """Create from dictionary."""
        range_val = data.get("range")
        if range_val is not None:
            range_val = tuple(range_val)

        return cls(
            name=data["name"],
            unit=data.get("unit", ""),
            range=range_val,
            step=data.get("step"),
            display=data.get("display", True),
            discrete=data.get("discrete", False),
            cyclic=data.get("cyclic", False),
            scale=data.get("scale", 1.0),
            description=data.get("description", ""),
        )


@dataclass
class Dimensions:
    """Complete dimension specification for a scene.

    Defines the coordinate system including dimension order,
    properties, and display configuration.
    """

    dimensions: List[Dimension] = field(default_factory=list)

    def __post_init__(self):
        """Validate dimensions configuration."""
        # Check for duplicate names
        names = [d.name for d in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError("Dimension names must be unique")

        # Check display count
        display_count = sum(1 for d in self.dimensions if d.display)
        if display_count > 3:
            raise ValueError(
                f"Maximum 3 dimensions can be displayed, got {display_count}"
            )

        # Ensure at least one dimension is displayed if any exist
        if self.dimensions and display_count == 0:
            raise ValueError("At least one dimension must be displayed")

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.dimensions)

    @property
    def names(self) -> List[str]:
        """List of dimension names in order."""
        return [d.name for d in self.dimensions]

    @property
    def displayed(self) -> List[int]:
        """Indices of displayed dimensions."""
        return [i for i, d in enumerate(self.dimensions) if d.display]

    @property
    def non_displayed(self) -> List[int]:
        """Indices of non-displayed dimensions."""
        return [i for i, d in enumerate(self.dimensions) if not d.display]

    def get_dimension(self, name: str) -> Optional[Dimension]:
        """Get dimension by name."""
        for dim in self.dimensions:
            if dim.name == name:
                return dim
        return None

    def get_index(self, name: str) -> int:
        """Get dimension index by name."""
        for i, dim in enumerate(self.dimensions):
            if dim.name == name:
                return i
        raise ValueError(f"Dimension '{name}' not found")

    def validate_positions(self, positions: np.ndarray, name: str = "positions"):
        """Validate that positions array matches scene dimensions.

        Args:
            positions: Array to validate
            name: Name for error messages

        Raises:
            ValueError: If positions don't match scene dimensions
        """
        if positions.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got shape {positions.shape}")

        if positions.shape[1] != self.ndim:
            raise ValueError(
                f"{name} has {positions.shape[1]} dimensions, "
                f"but scene has {self.ndim} dimensions"
            )

        # Check ranges if specified
        for i, dim in enumerate(self.dimensions):
            if dim.range is not None:
                col = positions[:, i]
                min_val, max_val = col.min(), col.max()
                if min_val < dim.range[0] or max_val > dim.range[1]:
                    raise ValueError(
                        f"{name} dimension '{dim.name}' has values "
                        f"[{min_val:.2f}, {max_val:.2f}] outside "
                        f"range {dim.range}"
                    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"dimensions": [d.to_dict() for d in self.dimensions]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dimensions:
        """Create from dictionary."""
        dims = [Dimension.from_dict(d) for d in data["dimensions"]]
        return cls(dimensions=dims)

    # --- Convenience constructors ---

    @classmethod
    def default_2d(cls) -> Dimensions:
        """Create default 2D dimensions (x, y)."""
        return cls(
            [
                Dimension("x", unit="px", display=True),
                Dimension("y", unit="px", display=True),
            ]
        )

    @classmethod
    def default_3d(cls) -> Dimensions:
        """Create default 3D dimensions (x, y, z)."""
        return cls(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

    @classmethod
    def default_timeseries(
        cls, n_timepoints: int = 100, time_unit: str = "s"
    ) -> Dimensions:
        """Create default time series dimensions (t, x, y, z)."""
        return cls(
            [
                Dimension(
                    "t",
                    unit=time_unit,
                    range=(0, n_timepoints - 1),
                    step=1.0,
                    display=False,
                    discrete=True,
                ),
                Dimension("x", unit="px", display=True),
                Dimension("y", unit="px", display=True),
                Dimension("z", unit="px", display=True),
            ]
        )

    @classmethod
    def default_multichannel(cls, n_channels: int = 3) -> Dimensions:
        """Create default multichannel dimensions (c, x, y, z)."""
        return cls(
            [
                Dimension(
                    "c",
                    unit="ch",
                    range=(0, n_channels - 1),
                    step=1.0,
                    display=False,
                    discrete=True,
                ),
                Dimension("x", unit="px", display=True),
                Dimension("y", unit="px", display=True),
                Dimension("z", unit="px", display=True),
            ]
        )

    @classmethod
    def from_positions(
        cls, positions: np.ndarray, names: Optional[List[str]] = None
    ) -> Dimensions:
        """Infer dimensions from positions array.

        Args:
            positions: Positions array
            names: Optional dimension names

        Returns:
            Inferred dimensions
        """
        if positions.ndim != 2:
            raise ValueError(f"Positions must be 2D array, got shape {positions.shape}")

        ndim = positions.shape[1]

        # Default names
        if names is None:
            if ndim <= 3:
                names = ["x", "y", "z"][:ndim]
            else:
                names = ["x", "y", "z"] + [f"dim{i}" for i in range(3, ndim)]

        if len(names) != ndim:
            raise ValueError(f"Got {len(names)} names for {ndim} dimensions")

        # Create dimensions
        dimensions = []
        for i, name in enumerate(names):
            # First 3 dimensions are displayed
            display = i < 3

            # Don't set range when inferring - too restrictive
            dimensions.append(Dimension(name=name, display=display))

        return cls(dimensions)
