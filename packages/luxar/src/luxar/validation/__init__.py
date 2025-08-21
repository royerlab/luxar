"""Validation functions for Luxar data structures."""

from .base import (
    ValidationError,
    validate_colors_for_writing,
    validate_positions_for_writing,
    validate_radii_for_writing,
    validate_sharpness_for_writing,
)
from .nd import (
    DimensionalCoverageError,
    broadcast_to_all_slices,
    validate_dimensional_coverage,
)

__all__ = [
    # Exception
    "ValidationError",
    # From base
    "validate_colors_for_writing",
    "validate_positions_for_writing",
    "validate_radii_for_writing",
    "validate_sharpness_for_writing",
    # From nd
    "DimensionalCoverageError",
    "broadcast_to_all_slices",
    "validate_dimensional_coverage",
]
