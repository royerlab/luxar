"""Validation for nD dimensional consistency in scenes.

This module provides validation to ensure all point groups in a scene
have consistent coverage of non-displayed dimensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Annotation only, in two signatures. `validation` sits BELOW `core`: core
    # imports validation (and now imports `validation.writing` at module level,
    # since A1-03 moved the shared pre-write gates there), so a runtime import
    # back into core is the one edge that would reinstate a cycle.
    from ..core.dimensions import Dimensions


class DimensionalCoverageError(ValueError):
    """Error raised when point groups have inconsistent dimensional coverage."""

    def __init__(
        self,
        message: str,
        group_name: str,
        missing_coverage: Optional[Dict[str, Set[float]]] = None,
    ) -> None:
        """Initialize dimensional coverage error.

        Args:
            message: Error message
            group_name: Name of the group with coverage issues
            missing_coverage: Dict mapping dimension names to missing values
        """
        full_message = message
        if missing_coverage:
            full_message += "\n\nMissing coverage:"
            for dim_name, values in missing_coverage.items():
                full_message += f"\n  {dim_name}: {sorted(values)}"

        full_message += "\n\n💡 Suggestions:"
        full_message += "\n  1. Provide points for all time/channel combinations"
        full_message += "\n  2. Use broadcast_to_all_slices() helper function"
        full_message += "\n  3. Mark this group as 'static=True' (future feature)"

        super().__init__(full_message)
        self.group_name = group_name
        self.missing_coverage = missing_coverage


def validate_dimensional_coverage(
    scene_dimensions: Dimensions,
    point_groups: Dict[str, NDArray[np.float32]],
) -> None:
    """Validate that all point groups have consistent dimensional coverage.

    For non-displayed dimensions (like Time and Channel), ensures that either:
    1. All groups have points at the same set of dimension values
    2. Groups are properly marked for handling (future: static flag)

    Args:
        scene_dimensions: Scene dimension specifications
        point_groups: Dict mapping group names to position arrays

    Raises:
        DimensionalCoverageError: If groups have inconsistent coverage
    """
    # Find non-displayed dimensions
    non_displayed_dims = [
        (i, dim) for i, dim in enumerate(scene_dimensions.dimensions) if not dim.display
    ]

    if not non_displayed_dims:
        # No non-displayed dimensions, nothing to validate
        return

    # Collect unique values for each non-displayed dimension per group
    coverage_by_group: Dict[str, Dict[int, Set[float]]] = {}

    for group_name, positions in point_groups.items():
        coverage_by_group[group_name] = {}

        for dim_idx, dim in non_displayed_dims:
            if dim_idx < positions.shape[1]:
                # Get unique values for this dimension
                unique_values = set(np.unique(positions[:, dim_idx]))
                coverage_by_group[group_name][dim_idx] = unique_values

    # Handle empty groups
    if not point_groups:
        return

    # Find the reference coverage (from the first or largest group)
    reference_group = max(point_groups.keys(), key=lambda g: len(point_groups[g]))
    reference_coverage = coverage_by_group[reference_group]

    # Check each group against reference
    for group_name, coverage in coverage_by_group.items():
        if group_name == reference_group:
            continue

        for dim_idx, dim in non_displayed_dims:
            if dim_idx not in coverage:
                continue

            ref_values = reference_coverage.get(dim_idx, set())
            group_values = coverage[dim_idx]

            if ref_values != group_values:
                missing = ref_values - group_values

                if missing:
                    raise DimensionalCoverageError(
                        f"Group '{group_name}' has incomplete coverage of "
                        f"dimension '{dim.name}' (index {dim_idx}).\n"
                        f"Expected values: {sorted(ref_values)}\n"
                        f"Found values: {sorted(group_values)}",
                        group_name,
                        {dim.name: missing},
                    )


def broadcast_to_all_slices(
    positions: NDArray[np.float32],
    colors: Optional[NDArray[np.float32]],
    radii: Optional[NDArray[np.float32]],
    scene_dimensions: Dimensions,
) -> Tuple[
    NDArray[np.float32], Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]
]:
    """Broadcast points to cover all non-displayed dimension values.

    Helper function that replicates points across all combinations of
    non-displayed dimensions (e.g., Time and Channel).

    Args:
        positions: Original position array
        colors: Original colors array (optional)
        radii: Original radii array (optional)
        scene_dimensions: Scene dimension specifications

    Returns:
        Tuple of (broadcasted_positions, broadcasted_colors, broadcasted_radii)
    """
    # Find non-displayed dimensions
    non_displayed_dims = [
        (i, dim) for i, dim in enumerate(scene_dimensions.dimensions) if not dim.display
    ]

    if not non_displayed_dims:
        # No broadcasting needed
        return positions, colors, radii

    # Get unique values for each non-displayed dimension
    dim_values = []
    for dim_idx, dim in non_displayed_dims:
        if dim.range and dim.discrete:
            # Use discrete values within range
            values = np.arange(dim.range[0], dim.range[1] + 1, dim.step or 1)
        else:
            # Use existing unique values from positions
            if dim_idx < positions.shape[1]:
                values = np.unique(positions[:, dim_idx])
            else:
                values = np.array([0])  # Default single value
        dim_values.append(values)

    # Calculate total number of slices
    n_slices = np.prod([len(v) for v in dim_values])
    n_points = positions.shape[0]

    # Create broadcasted arrays
    new_positions = np.zeros(
        (n_points * n_slices, positions.shape[1]), dtype=np.float32
    )
    new_colors = (
        None if colors is None else np.zeros((n_points * n_slices, 3), dtype=np.float32)
    )
    new_radii = (
        None if radii is None else np.zeros(n_points * n_slices, dtype=np.float32)
    )

    # Generate all combinations and replicate data
    idx = 0
    for slice_values in np.ndindex(*[len(v) for v in dim_values]):
        start_idx = idx * n_points
        end_idx = (idx + 1) * n_points

        # Copy positions
        new_positions[start_idx:end_idx] = positions

        # Set non-displayed dimension values
        for i, (dim_idx, _) in enumerate(non_displayed_dims):
            new_positions[start_idx:end_idx, dim_idx] = dim_values[i][slice_values[i]]

        # Copy colors and radii if present
        if colors is not None and new_colors is not None:
            new_colors[start_idx:end_idx] = colors
        if radii is not None and new_radii is not None:
            new_radii[start_idx:end_idx] = radii

        idx += 1

    return new_positions, new_colors, new_radii
