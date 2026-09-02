"""Validation functions for Luxar data structures.

This package contains:
- types.py: Basic type validation and type guards
- base.py: Detailed validation with helpful error messages (for writing)
- nd.py: nD dimensional coverage validation
"""

from .base import (
    ValidationError,
    validate_cholesky_for_writing,
    validate_colors_for_writing,
    validate_labels_for_writing,
    validate_node_name,
    validate_positions_for_writing,
    validate_radii_for_writing,
    validate_sharpness_for_writing,
    validate_widths_for_writing,
    validate_zarr_attributes,
)
from .category_validation import validate_categories
from .nd import (
    DimensionalCoverageError,
    broadcast_to_all_slices,
    validate_dimensional_coverage,
)
from .types import (
    is_color_array,
    is_position_array,
    is_transform_matrix,
    validate_absorption,
    validate_blending_mode,
    validate_category_indices,
    validate_colors,
    validate_depth_level,
    validate_gamma,
    validate_line_join,
    validate_node_type,
    validate_opacity,
    validate_physical_unit,
    validate_positions,
    validate_radii,
    validate_sharpness,
    validate_transform,
    validate_truncation_radius,
)

__all__ = [
    # Exceptions
    "ValidationError",
    "DimensionalCoverageError",
    # From base (detailed validation for writing)
    "validate_cholesky_for_writing",
    "validate_colors_for_writing",
    "validate_labels_for_writing",
    "validate_node_name",
    "validate_positions_for_writing",
    "validate_radii_for_writing",
    "validate_sharpness_for_writing",
    "validate_widths_for_writing",
    "validate_zarr_attributes",
    # From types (basic validation and type guards)
    "validate_positions",
    "validate_colors",
    "validate_radii",
    "validate_sharpness",
    "validate_transform",
    "validate_node_type",
    "validate_physical_unit",
    "validate_opacity",
    "validate_truncation_radius",
    "validate_gamma",
    "validate_absorption",
    "validate_blending_mode",
    "validate_depth_level",
    "validate_line_join",
    "validate_categories",
    "validate_category_indices",
    "is_position_array",
    "is_color_array",
    "is_transform_matrix",
    # From nd (dimensional validation)
    "broadcast_to_all_slices",
    "validate_dimensional_coverage",
]
