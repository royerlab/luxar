"""Point spatial index implementation for efficient nD point queries.

This module provides functionality to build and query point-specific spatial indices for
nD points, enabling efficient range queries and lazy loading. This implementation is
specifically designed for point data and will be complemented by other spatial index
types for lines, meshes, and volumes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from arbol import aprint
from numpy.typing import NDArray

from ..typing_utils.constants import (
    SPATIAL_INDEX_FALLBACK_CELLS,
    SPATIAL_INDEX_MAX_CELLS_DISCRETE,
    SPATIAL_INDEX_MAX_CELLS_SINGLE_DIM,
    SPATIAL_INDEX_MIN_CELLS_SINGLE_DIM,
    SPATIAL_INDEX_MULTI_DIM_MAX_CELLS_PER_DIM,
    SPATIAL_INDEX_MULTI_DIM_MAX_TARGET,
    SPATIAL_INDEX_MULTI_DIM_MIN_TARGET,
    SPATIAL_INDEX_MULTI_DIM_POINTS_DIVISOR,
    SPATIAL_INDEX_SINGLE_DIM_POINTS_DIVISOR,
)


def decode_cell_id(cell_id: int, grid_shape: NDArray[np.uint32]) -> List[int]:
    """Convert linear cell ID back to nD grid coordinates.

    This is the inverse of the linearization in build_spatial_index().
    Extracts grid coordinates from a linear cell ID using row-major order.

    Example for grid_shape (2, 3, 4):
        - ID 0 → (0, 0, 0)
        - ID 1 → (0, 0, 1)
        - ID 4 → (0, 1, 0)
        - ID 12 → (1, 0, 0)

    Args:
        cell_id: Linear cell ID
        grid_shape: Shape of the grid in each dimension

    Returns:
        List of grid coordinates in original dimension order
    """
    coords = []
    # Process dimensions in reverse order (innermost first)
    for dim_size in reversed(grid_shape):
        # Extract coordinate for this dimension using modulo
        coords.append(int(cell_id % dim_size))
        # Divide to process next dimension
        cell_id //= dim_size
    # Reverse to get back to original dimension order
    return list(reversed(coords))


def build_spatial_index(
    positions: NDArray[np.float32],
    grid_shape: Optional[NDArray[np.uint32]] = None,
    min_points_per_cell: int = 100,
    max_points_per_cell: int = 1000,
    displayed_dims: Optional[List[int]] = None,
    discrete_dims: Optional[List[int]] = None,
    spatial_extend_dims: Optional[List[bool]] = None,
    max_radius: float = 0.1,
) -> Dict[str, Union[NDArray, List[int], int]]:
    """Build spatial index for nD points, indexing only non-displayed dimensions.

    This function reorders points based on spatial locality using a regular grid
    partitioning. Points in the same grid cell are stored contiguously.
    Only non-displayed dimensions are indexed for efficient slicing.

    Args:
        positions: Point positions of shape (N, D)
        grid_shape: Number of cells per dimension, auto-determined if None
        min_points_per_cell: Minimum target points per cell
        max_points_per_cell: Maximum target points per cell
        displayed_dims: Indices of displayed dimensions (max 3). If None, first 3 dims are displayed.
        discrete_dims: Indices of discrete dimensions (e.g., time, categories). If None, all dims are continuous.
        spatial_extend_dims: List of booleans indicating which dimensions points extend through spatially.
        max_radius: Maximum radius of points for determining grid cell size in spatial dimensions.

    Returns:
        Dictionary containing:
            - occupied_cells: nD coordinates of occupied grid cells (only for indexed dims)
            - cell_ranges: Start and end indices for points in each cell
            - sorted_positions: Reordered positions (all dimensions)
            - sort_order: Indices mapping original to sorted positions
            - grid_origin: Minimum coordinate per dimension (only for indexed dims)
            - cell_size: Size of each cell per dimension (only for indexed dims)
            - grid_shape: Final grid shape used (only for indexed dims)
            - indexed_dimensions: List of dimension indices that are indexed
            - full_dimensions: Total number of dimensions in original data
            - displayed_dimensions: Indices of displayed dimensions
    """
    n_points, n_dims = positions.shape

    # Determine displayed dimensions
    if displayed_dims is None:
        # Default: first 3 dimensions are displayed
        displayed_dims = list(range(min(3, n_dims)))

    # Calculate non-displayed dimensions
    non_displayed_dims = [d for d in range(n_dims) if d not in displayed_dims]

    # If all dimensions are displayed, skip spatial index
    if len(non_displayed_dims) == 0:
        aprint(f"  ⚠️ All {n_dims} dimensions are displayed - spatial index not needed")
        return {
            "occupied_cells": np.array([], dtype=np.uint32).reshape(0, 0),
            "cell_ranges": np.array([], dtype=np.uint64).reshape(0, 2),
            "sorted_positions": positions,
            "sort_order": np.arange(n_points, dtype=np.int64),
            "grid_origin": np.array([], dtype=np.float32),
            "cell_size": np.array([], dtype=np.float32),
            "grid_shape": np.array([], dtype=np.uint32),
            "indexed_dimensions": [],
            "full_dimensions": n_dims,
            "displayed_dimensions": displayed_dims,
            "total_points": n_points,
        }

    # Extract only non-displayed dimensions for indexing
    indexed_positions = positions[:, non_displayed_dims]
    n_indexed_dims = len(non_displayed_dims)

    aprint(
        f"  📊 Building spatial index on {n_indexed_dims} non-displayed dimensions out of {n_dims} total"
    )
    aprint(f"     Displayed dims: {displayed_dims}, Indexed dims: {non_displayed_dims}")

    n_points, _ = indexed_positions.shape

    # Handle empty arrays
    if n_points == 0:
        if grid_shape is None:
            grid_shape = np.array([2] * n_indexed_dims, dtype=np.uint32)
        else:
            grid_shape = np.array(grid_shape, dtype=np.uint32)

        return {
            "occupied_cells": np.array([], dtype=np.uint32).reshape(0, n_indexed_dims),
            "cell_ranges": np.array([], dtype=np.uint64).reshape(0, 2),
            "sorted_positions": positions,
            "sort_order": np.array([], dtype=np.int64),
            "grid_origin": np.zeros(n_indexed_dims, dtype=np.float32),
            "cell_size": np.ones(n_indexed_dims, dtype=np.float32),
            "grid_shape": grid_shape,
            "indexed_dimensions": non_displayed_dims,
            "full_dimensions": n_dims,
            "displayed_dimensions": displayed_dims,
            "total_points": n_points,
        }

    # Auto-determine grid shape if not provided
    if grid_shape is None:
        grid_shape_list = []

        # Determine which dimensions are discrete
        discrete_dims_set = set(discrete_dims) if discrete_dims else set()

        for idx, dim_idx in enumerate(non_displayed_dims):
            if dim_idx in discrete_dims_set:
                # For discrete dimensions, use one cell per unique value
                unique_vals = len(np.unique(indexed_positions[:, idx]))
                # Cap to prevent memory issues
                cells = min(SPATIAL_INDEX_MAX_CELLS_DISCRETE, unique_vals)
                grid_shape_list.append(cells)
                aprint(
                    f"  📊 Auto grid: dimension {dim_idx} is discrete with {unique_vals} unique values → {cells} cells"
                )
            else:
                # For continuous dimensions, use adaptive grid based on data
                # Aim for reasonable number of cells
                if n_indexed_dims == 1:
                    # Single indexed dimension - can use more cells
                    cells = min(
                        SPATIAL_INDEX_MAX_CELLS_SINGLE_DIM,
                        max(
                            SPATIAL_INDEX_MIN_CELLS_SINGLE_DIM,
                            int(np.sqrt(n_points / SPATIAL_INDEX_SINGLE_DIM_POINTS_DIVISOR)),
                        ),
                    )
                else:
                    # Multiple indexed dimensions - use fewer cells per dimension
                    # to keep total index size manageable
                    target_cells = max(
                        SPATIAL_INDEX_MULTI_DIM_MIN_TARGET,
                        min(
                            SPATIAL_INDEX_MULTI_DIM_MAX_TARGET,
                            n_points // SPATIAL_INDEX_MULTI_DIM_POINTS_DIVISOR,
                        ),
                    )
                    cells = int(np.power(target_cells, 1.0 / n_indexed_dims))
                    cells = max(
                        SPATIAL_INDEX_FALLBACK_CELLS,
                        min(SPATIAL_INDEX_MULTI_DIM_MAX_CELLS_PER_DIM, cells),
                    )  # Clamp to [2, 20]
                grid_shape_list.append(cells)
                aprint(
                    f"  📊 Auto grid: dimension {dim_idx} is continuous → {cells} cells"
                )

        grid_shape = np.array(grid_shape_list, dtype=np.uint32)
    else:
        grid_shape = np.array(grid_shape, dtype=np.uint32)

    # Calculate grid bounds using only indexed dimensions
    min_coords = np.min(indexed_positions, axis=0)
    max_coords = np.max(indexed_positions, axis=0)

    # Add small epsilon to avoid edge cases
    epsilon = 1e-6
    grid_origin = min_coords - epsilon
    grid_range = (max_coords - min_coords) + 2 * epsilon
    cell_size = grid_range / grid_shape

    # Adjust cell size for spatial dimensions to account for point extension
    if spatial_extend_dims is not None:
        for idx, dim_idx in enumerate(non_displayed_dims):
            if dim_idx < len(spatial_extend_dims) and spatial_extend_dims[dim_idx]:
                # For spatial dimensions, ensure cells are at least 2*max_radius
                # This ensures we catch all potentially visible points during queries
                min_cell_size = 2 * max_radius
                if cell_size[idx] < min_cell_size:
                    aprint(
                        f"  📏 Adjusting cell size for spatial dimension {dim_idx}: "
                        f"{cell_size[idx]:.3f} → {min_cell_size:.3f}"
                    )
                    cell_size[idx] = min_cell_size

    # Assign points to grid cells based on indexed dimensions only
    grid_indices = np.floor((indexed_positions - grid_origin) / cell_size).astype(
        np.uint32
    )
    grid_indices = np.clip(grid_indices, 0, grid_shape - 1)

    # Convert nD grid indices to linear cell IDs
    # This implements row-major linearization: cell_id = i0 * stride0 + i1 * stride1 + ...
    # where stride_d = product of all grid dimensions after d
    # Example for 3D grid (2, 3, 4):
    #   - Cell (0,0,0) → ID 0
    #   - Cell (0,0,1) → ID 1
    #   - Cell (0,1,0) → ID 4 (skip 4 cells in last dimension)
    #   - Cell (1,0,0) → ID 12 (skip 3*4 cells in middle dimensions)
    cell_ids = np.zeros(n_points, dtype=np.uint64)
    stride = 1  # Start with innermost dimension (stride = 1)
    # Process dimensions from last to first (row-major order)
    for d in range(n_indexed_dims - 1, -1, -1):
        cell_ids += grid_indices[:, d] * stride
        stride *= grid_shape[d]  # Update stride for next dimension

    # Sort points by cell ID for spatial locality
    sort_order = np.argsort(cell_ids)
    sorted_positions = positions[sort_order]
    sorted_cell_ids = cell_ids[sort_order]

    # Build sparse index
    occupied_cells = []
    cell_ranges = []

    if n_points > 0:
        current_cell = sorted_cell_ids[0]
        start_idx = 0

        for i in range(1, n_points):
            if sorted_cell_ids[i] != current_cell:
                # Record the completed cell
                grid_coords = decode_cell_id(current_cell, grid_shape)
                occupied_cells.append(grid_coords)
                cell_ranges.append([start_idx, i])

                # Start new cell
                current_cell = sorted_cell_ids[i]
                start_idx = i

        # Record final cell
        grid_coords = decode_cell_id(current_cell, grid_shape)
        occupied_cells.append(grid_coords)
        cell_ranges.append([start_idx, n_points])

    result = {
        "occupied_cells": np.array(occupied_cells, dtype=np.uint32),
        "cell_ranges": np.array(cell_ranges, dtype=np.uint64),
        "sorted_positions": sorted_positions,
        "sort_order": sort_order,
        "grid_origin": grid_origin.astype(np.float32),
        "cell_size": cell_size.astype(np.float32),
        "grid_shape": grid_shape,
        "indexed_dimensions": non_displayed_dims,
        "full_dimensions": n_dims,
        "displayed_dimensions": displayed_dims,
        "total_points": n_points,
    }

    # Add spatial metadata if provided
    if spatial_extend_dims is not None:
        result["spatial_extend_dims"] = spatial_extend_dims
        result["max_radius"] = max_radius

    return result


def query_spatial_index(
    index_metadata: Dict[str, Any],
    occupied_cells: NDArray[np.uint32],
    cell_ranges: NDArray[np.uint64],
    slice_pos: NDArray[np.float32],
    tolerance: NDArray[np.float32],
) -> List[Tuple[int, int]]:
    """Query spatial index for points within tolerance of a slice position.

    Args:
        index_metadata: Index metadata containing grid_shape, grid_origin, cell_size
        occupied_cells: nD coordinates of occupied grid cells
        cell_ranges: Start and end indices for points in each cell
        slice_pos: Current slice position in nD space
        tolerance: Tolerance (radius) for each dimension

    Returns:
        List of (start, end) index pairs for point ranges to load
    """
    grid_shape = np.array(index_metadata["grid_shape"])
    grid_origin = np.array(index_metadata["grid_origin"])
    cell_size = np.array(index_metadata["cell_size"])

    # Calculate grid range to check
    min_grid = np.floor((slice_pos - tolerance - grid_origin) / cell_size).astype(int)
    max_grid = np.ceil((slice_pos + tolerance - grid_origin) / cell_size).astype(int)

    # Clamp to valid grid bounds
    min_grid = np.maximum(0, min_grid)
    max_grid = np.minimum(grid_shape - 1, max_grid)

    # Find matching occupied cells
    point_ranges = []
    for i, cell_coords in enumerate(occupied_cells):
        # Check if cell is within query range
        if np.all(cell_coords >= min_grid) and np.all(cell_coords <= max_grid):
            start, end = cell_ranges[i]
            point_ranges.append((int(start), int(end)))

    return point_ranges


def apply_sort_order(
    array: Optional[NDArray], sort_order: NDArray[np.int64]
) -> Optional[NDArray]:
    """Apply sort order to an array.

    Args:
        array: Array to sort, or None
        sort_order: Indices for sorting

    Returns:
        Sorted array or None if input was None
    """
    if array is None:
        return None
    return array[sort_order]


def validate_spatial_index(
    index_data: Dict[str, NDArray], n_points: int, n_dims: int
) -> None:
    """Validate spatial index consistency.

    This ensures that the spatial index correctly covers all points and
    maintains proper data structure invariants.

    Args:
        index_data: Spatial index data dictionary
        n_points: Expected number of points
        n_dims: Expected number of dimensions (full, not just indexed)

    Raises:
        ValueError: If validation fails
    """
    # Check required keys
    required_keys = {
        "occupied_cells",
        "cell_ranges",
        "sorted_positions",
        "sort_order",
        "grid_origin",
        "cell_size",
        "grid_shape",
    }
    missing_keys = required_keys - set(index_data.keys())
    if missing_keys:
        raise ValueError(f"Spatial index missing required keys: {missing_keys}")

    # Validate sorted positions shape (still contains all dimensions)
    if index_data["sorted_positions"].shape != (n_points, n_dims):
        raise ValueError(
            f"Sorted positions shape {index_data['sorted_positions'].shape} "
            f"doesn't match expected ({n_points}, {n_dims})"
        )

    # Validate sort order length
    if len(index_data["sort_order"]) != n_points:
        raise ValueError(
            f"Sort order length {len(index_data['sort_order'])} "
            f"doesn't match n_points {n_points}"
        )

    # Validate that all points are covered by cell ranges
    if len(index_data["cell_ranges"]) > 0:
        total_points = sum(int(end - start) for start, end in index_data["cell_ranges"])
        if total_points != n_points:
            raise ValueError(
                f"Cell ranges cover {total_points} points but expected {n_points}"
            )

        # Check that ranges are non-overlapping and sorted
        prev_end = 0
        for i, (start, end) in enumerate(index_data["cell_ranges"]):
            if start != prev_end:
                raise ValueError(
                    f"Gap or overlap in cell ranges at index {i}: "
                    f"previous end={prev_end}, current start={start}"
                )
            if end <= start:
                raise ValueError(
                    f"Invalid range at index {i}: start={start}, end={end}"
                )
            prev_end = end

        # Final range should end at n_points
        if prev_end != n_points:
            raise ValueError(
                f"Last cell range ends at {prev_end} but should be {n_points}"
            )

    # Validate occupied cells shape
    n_occupied = len(index_data["occupied_cells"])
    if n_occupied != len(index_data["cell_ranges"]):
        raise ValueError(
            f"Number of occupied cells ({n_occupied}) doesn't match "
            f"number of cell ranges ({len(index_data['cell_ranges'])})"
        )

    # For the new spatial index, occupied cells only have indexed dimensions
    # Get the number of indexed dimensions
    n_indexed_dims = len(index_data.get("indexed_dimensions", []))

    # If no dimensions are indexed (all displayed), occupied_cells should be empty
    if n_indexed_dims == 0:
        if n_occupied > 0:
            raise ValueError(
                f"Expected no occupied cells when all dimensions are displayed, but got {n_occupied}"
            )
    elif n_occupied > 0:
        # Check occupied cells shape matches indexed dimensions
        if index_data["occupied_cells"].shape[1] != n_indexed_dims:
            raise ValueError(
                f"Occupied cells have {index_data['occupied_cells'].shape[1]} dimensions "
                f"but expected {n_indexed_dims} indexed dimensions"
            )

    # Validate grid parameters (for indexed dimensions only)
    if len(index_data["grid_shape"]) != n_indexed_dims:
        raise ValueError(
            f"Grid shape has {len(index_data['grid_shape'])} dimensions "
            f"but expected {n_indexed_dims} indexed dimensions"
        )

    if len(index_data["grid_origin"]) != n_indexed_dims:
        raise ValueError(
            f"Grid origin has {len(index_data['grid_origin'])} dimensions "
            f"but expected {n_indexed_dims} indexed dimensions"
        )

    if len(index_data["cell_size"]) != n_indexed_dims:
        raise ValueError(
            f"Cell size has {len(index_data['cell_size'])} dimensions "
            f"but expected {n_indexed_dims} indexed dimensions"
        )
