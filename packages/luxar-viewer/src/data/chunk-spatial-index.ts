/**
 * Chunk-Based Spatial Index for Efficient nD Point Queries
 *
 * This module implements spatial queries using Morton/Hilbert-ordered chunks
 * with bounding boxes. This replaces the grid-based spatial index approach.
 *
 * Design:
 * - Python orders points using Morton/Hilbert space-filling curves
 * - Points are divided into chunks (default ~10K points each)
 * - Each chunk has a bounding box in nD space
 * - Queries test chunk bounding boxes for intersection
 * - Only matching chunks are loaded from zarr
 *
 * Benefits:
 * - Simple: No grid discretization needed
 * - Fast: Linear scan of ~100-1000 chunks
 * - Memory efficient: Only store bounding boxes
 * - Correct: Uses what Python already provides
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';
import { log, Modules } from '../utils/log';

/** Point range [start, end) for loading data */
export interface PointRange {
  start: number;
  end: number;
}

/** Chunk-based spatial index structure */
export interface ChunkSpatialIndex {
  /** Metadata from node attributes */
  metadata: {
    /** Ordering algorithm: 'morton' or 'hilbert' */
    ordering: 'morton' | 'hilbert';
    /** Dimensions using space-filling curve ordering (spatial dims) */
    ordering_dims: number[];
    /** Dimensions using lexicographic ordering (discrete/slice dims) */
    slice_dims: number[];
    /** Bits per dimension for space-filling curve encoding */
    ordering_bits_per_dim: number;
    /** Points per chunk */
    chunk_size: number;
    /** Total points in dataset */
    total_points: number;
    /** Total number of chunks */
    total_chunks: number;
    /** Full dimensionality */
    ndim: number;
  };

  /** Chunk bounding boxes: shape (num_chunks, ndim, 2) flattened */
  chunkBounds: Float32Array;
}

/**
 * Load chunk-based spatial index from zarr arrays
 *
 * Reads chunk_bounds array and metadata from node attributes.
 * Returns null if chunk_bounds not found (3D datasets without spatial ordering).
 */
export async function loadChunkSpatialIndex(
  zarrLocation: zarr.Location<zarr.Readable>,
  nodeAttrs: any
): Promise<ChunkSpatialIndex | null> {
  try {
    // Try to load chunk_bounds array
    const boundsLoc = zarrLocation.resolve('chunk_bounds');
    const boundsArray = await zarr.open(boundsLoc, { kind: 'array' });
    const boundsData = await get(boundsArray);

    // Extract shape information
    const [numChunks, ndim, _two] = boundsArray.shape;

    // Validate shape: third dimension must be 2 (min, max)
    if (_two !== 2) {
      log.error(
        Modules.SPATIAL_INDEX,
        `Invalid chunk_bounds shape: expected [..., 2], got [..., ${_two}]`
      );
      return null;
    }

    // Validate: array length should match expected size
    const expectedLength = numChunks * ndim * 2;
    const actualLength = (boundsData.data as ArrayLike<number>).length;
    if (actualLength !== expectedLength) {
      log.warning(
        Modules.SPATIAL_INDEX,
        `Chunk bounds array length mismatch: expected ${expectedLength} (${numChunks}×${ndim}×2), got ${actualLength}`
      );
    }

    // Validate: check if ndim matches position array dimensions (if available)
    const positionDims = nodeAttrs.n_dims || nodeAttrs.ndim;
    if (positionDims !== undefined && positionDims !== ndim) {
      log.warning(
        Modules.SPATIAL_INDEX,
        `Dimensionality mismatch: chunk_bounds has ${ndim}D but node attributes indicate ${positionDims}D`
      );
    }

    // Validate: ordering_dims + slice_dims should cover all dimensions
    // Support both new (ordering_dims) and legacy (morton_dims) field names
    const orderingDims = nodeAttrs.ordering_dims || nodeAttrs.morton_dims || [];
    const sliceDims = nodeAttrs.slice_dims || [];
    const allDims = new Set([...orderingDims, ...sliceDims]);
    if (allDims.size > 0 && allDims.size !== ndim) {
      log.warning(
        Modules.SPATIAL_INDEX,
        `Dimension coverage mismatch: ordering_dims[${orderingDims.length}] + slice_dims[${sliceDims.length}] = ${allDims.size}, but ndim=${ndim}`
      );
    }

    // Create chunk index
    // Support both new (ordering_*) and legacy (morton_*) field names for backward compatibility
    const chunkIndex: ChunkSpatialIndex = {
      metadata: {
        ordering: nodeAttrs.ordering || 'hilbert',
        ordering_dims: orderingDims,
        slice_dims: sliceDims,
        ordering_bits_per_dim:
          nodeAttrs.ordering_bits_per_dim || nodeAttrs.morton_bits_per_dim || 21,
        chunk_size: nodeAttrs.chunk_size || 0,
        total_points: nodeAttrs.n_points || 0,
        total_chunks: numChunks,
        ndim: ndim,
      },
      chunkBounds: new Float32Array(boundsData.data as ArrayBuffer | ArrayLike<number>),
    };

    log.info(
      Modules.SPATIAL_INDEX,
      `Loaded chunk spatial index: ${numChunks} chunks, ${ndim}D, ${chunkIndex.metadata.total_points} points`
    );

    return chunkIndex;
  } catch (error: any) {
    // No chunk_bounds - this is expected for 3D datasets without spatial ordering
    if (
      error.message?.includes('404') ||
      error.message?.includes('Not Found') ||
      error.message?.includes('Node not found')
    ) {
      log.info(Modules.SPATIAL_INDEX, 'No chunk_bounds found - dataset has no spatial indexing');
    } else {
      log.warning(Modules.SPATIAL_INDEX, `Could not load chunk_bounds: ${error.message}`);
    }
    return null;
  }
}

/**
 * Query chunks for a given nD view state
 *
 * Returns indices of chunks whose bounding boxes intersect the query region.
 *
 * @param index - Chunk spatial index
 * @param slicePosition - Position in nD space
 * @param tolerance - Tolerance/radius in each dimension
 * @returns Array of chunk indices to load
 */
export function queryChunksForView(
  index: ChunkSpatialIndex,
  slicePosition: number[],
  tolerance: number[]
): number[] {
  const { total_chunks, ndim } = index.metadata;
  const { chunkBounds } = index;

  const matchingChunks: number[] = [];

  // Test each chunk for intersection with query box
  for (let chunkIdx = 0; chunkIdx < total_chunks; chunkIdx++) {
    let intersects = true;

    // Check intersection in each dimension
    for (let d = 0; d < ndim; d++) {
      // Get chunk bounding box for this dimension
      // Layout: chunkBounds[chunkIdx, d, 0/1] flattened row-major
      const offset = chunkIdx * ndim * 2 + d * 2;
      const chunkMin = chunkBounds[offset];
      const chunkMax = chunkBounds[offset + 1];

      // Get query box for this dimension
      const queryMin = slicePosition[d] - tolerance[d];
      const queryMax = slicePosition[d] + tolerance[d];

      // Test for intersection: boxes DON'T overlap if:
      // - chunk max < query min (chunk entirely before query)
      // - chunk min > query max (chunk entirely after query)
      if (chunkMax < queryMin || chunkMin > queryMax) {
        intersects = false;
        break; // No need to check other dimensions
      }
    }

    if (intersects) {
      matchingChunks.push(chunkIdx);
    }
  }

  log.query(
    Modules.SPATIAL_INDEX,
    `Found ${matchingChunks.length}/${total_chunks} chunks matching query`
  );

  return matchingChunks;
}

/**
 * Convert chunk indices to point ranges for data loading
 *
 * @param chunkIndices - Array of chunk indices from query
 * @param chunkSize - Points per chunk
 * @param totalPoints - Total points in dataset (for last chunk)
 * @returns Array of point ranges {start, end}
 */
export function chunkIndicesToRanges(
  chunkIndices: number[],
  chunkSize: number,
  totalPoints: number
): PointRange[] {
  return chunkIndices.map((chunkIdx) => {
    const start = chunkIdx * chunkSize;
    const end = Math.min((chunkIdx + 1) * chunkSize, totalPoints);
    return { start, end };
  });
}

/**
 * Merge overlapping or adjacent point ranges
 *
 * Sorts ranges by start position and merges any that overlap or are adjacent.
 * This reduces the number of zarr chunk loads needed.
 */
export function mergePointRanges(ranges: PointRange[]): PointRange[] {
  if (ranges.length === 0) return [];

  // Sort by start position
  const sorted = [...ranges].sort((a, b) => a.start - b.start);

  const merged: PointRange[] = [];
  let current = sorted[0];

  for (let i = 1; i < sorted.length; i++) {
    const next = sorted[i];

    // Merge if overlapping or adjacent
    if (next.start <= current.end) {
      current = {
        start: current.start,
        end: Math.max(current.end, next.end),
      };
    } else {
      merged.push(current);
      current = next;
    }
  }

  merged.push(current);

  log.info(
    Modules.SPATIAL_INDEX,
    `Merged ${ranges.length} ranges → ${merged.length} continuous ranges`
  );

  return merged;
}
