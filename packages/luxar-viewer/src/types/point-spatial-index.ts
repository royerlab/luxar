/**
 * Type definitions for point spatial index structures.
 *
 * The point spatial index uses a regular grid partitioning of nD space,
 * stored as a sparse representation containing only occupied cells.
 * This implementation is specifically designed for point data.
 */

/**
 * Metadata for a point spatial index stored in zarr attributes
 */
export interface PointSpatialIndexMetadata {
  /** Number of cells per dimension (only indexed dims) */
  grid_shape: number[];

  /** Minimum coordinate per dimension (only indexed dims) */
  grid_origin: number[];

  /** Size of each cell per dimension (only indexed dims) */
  cell_size: number[];

  /** Number of occupied cells */
  num_occupied: number;

  /** Total possible cells */
  total_cells: number;

  /** Total number of points in the dataset */
  total_points: number;

  /** Number of dimensions indexed (non-displayed) */
  dimensions: number;

  /** Total number of dimensions in original data */
  full_dimensions: number;

  /** Which dimension indices are indexed (e.g., [3] for time in xyzt) */
  indexed_dimensions: number[];

  /** Which dimension indices are displayed (e.g., [0,1,2] for xyz) */
  displayed_dimensions: number[];

  /** Version of index builder */
  build_version: string;

  /** Maximum points in any single cell */
  max_points_per_cell?: number;
}

/**
 * Point spatial index data loaded from zarr arrays
 */
export interface PointSpatialIndex {
  /** Index metadata */
  metadata: PointSpatialIndexMetadata;

  /** Shape: [numOccupied * D], flattened nD coordinates */
  occupiedCells: Uint32Array;

  /** Shape: [numOccupied * 2], flattened [start, end] ranges */
  cellRanges: BigUint64Array;
}

/**
 * Query result containing point ranges to load
 */
export interface PointRange {
  /** Starting index (inclusive) */
  start: number;

  /** Ending index (exclusive) */
  end: number;
}
