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
import type { PointRange } from './data-loader-types';

export type { PointRange };

/** Chunk-based spatial index structure */
interface ChunkSpatialIndexNodeAttrs {
  ordering?: 'morton' | 'hilbert' | 'none';
  ordering_dims?: number[];
  slice_dims?: number[];
  ordering_bits_per_dim?: number;
  chunk_size?: number;
  n_points?: number;
  n_dims?: number;
  ndim?: number;
}

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
 * Load chunk-based spatial index from Zarr arrays.
 *
 * Reads the `chunk_bounds` array (shape: num_chunks × ndim × 2) and extracts
 * spatial ordering metadata from node attributes. This enables efficient nD
 * queries by testing chunk bounding boxes instead of loading all points.
 *
 * The spatial index is optional - 3D datasets without Morton/Hilbert ordering
 * won't have `chunk_bounds` and will return null (graceful fallback to full load).
 *
 * Validation performed:
 * - chunk_bounds shape must be [..., 2] for [min, max]
 * - Array length must match expected size (num_chunks × ndim × 2)
 * - Dimensionality must match position array metadata
 * - ordering_dims + slice_dims must cover all dimensions
 *
 * @param zarrLocation - Zarr location of the node containing chunk_bounds array.
 *                       Typically a Points or Lines node location.
 *
 * @param nodeAttrs - Node attributes (from .zattrs) containing:
 *                    - ordering: 'morton' or 'hilbert' (default: 'hilbert')
 *                    - ordering_dims: Dimensions using space-filling curve
 *                    - slice_dims: Dimensions using lexicographic ordering
 *                    - ordering_bits_per_dim: Bits per dimension for encoding (default: 21)
 *                    - chunk_size: Points per chunk
 *                    - n_points: Total points in dataset
 *                    - n_dims / ndim: Total dimensionality
 *
 * @returns Promise resolving to ChunkSpatialIndex with metadata and bounds,
 *          or null if chunk_bounds doesn't exist (expected for 3D datasets).
 *          Null is NOT an error - it indicates graceful fallback to full loading.
 *
 * @throws {Error} If chunk_bounds exists but has invalid shape
 * @throws {Error} If chunk_bounds array cannot be decoded
 *
 * @example
 * ```typescript
 * // Load spatial index for nD points node
 * const nodeLoc = rootLoc.resolve('cells/points');
 * const nodeGroup = await zarr.open(nodeLoc, { kind: 'group' });
 * const index = await loadChunkSpatialIndex(nodeLoc, nodeGroup.attrs);
 *
 * if (index) {
 *   console.log(`Loaded ${index.metadata.total_chunks} chunks`);
 *   // Use queryChunksForView for efficient queries
 * } else {
 *   console.log('No spatial index - loading full dataset');
 *   // Fall back to loading all points
 * }
 * ```
 *
 * @example
 * ```typescript
 * // Handle 404 gracefully (3D datasets)
 * const index = await loadChunkSpatialIndex(nodeLoc, attrs);
 * // index === null for 3D datasets without spatial ordering
 * // This is expected and handled by loader
 * ```
 *
 * @see {@link queryChunksForView} for querying the loaded index
 * @see SPECIFICATIONS.md - Section 2.1 for chunk spatial index format
 * @see luxar/io/SPECIFICATIONS.md - for Python encoding details
 */
export async function loadChunkSpatialIndex(
  zarrLocation: zarr.Location<zarr.Readable>,
  nodeAttrs: ChunkSpatialIndexNodeAttrs
): Promise<ChunkSpatialIndex | null> {
  // Skip network probe if spatial ordering is disabled or absent.
  // This matches the guard in gsplats-chunk-spatial-index.ts and
  // lines-chunk-spatial-index.ts, avoiding 3-4 HTTP 404s per node
  // for datasets without spatial indexing.
  if (!nodeAttrs.ordering || nodeAttrs.ordering === 'none') {
    log.info(Modules.SPATIAL_INDEX, 'No spatial ordering — skipping chunk_bounds probe');
    return null;
  }

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
    const orderingDims = nodeAttrs.ordering_dims || [];
    const sliceDims = nodeAttrs.slice_dims || [];
    const allDims = new Set([...orderingDims, ...sliceDims]);
    if (allDims.size > 0 && allDims.size !== ndim) {
      log.warning(
        Modules.SPATIAL_INDEX,
        `Dimension coverage mismatch: ordering_dims[${orderingDims.length}] + slice_dims[${sliceDims.length}] = ${allDims.size}, but ndim=${ndim}`
      );
    }

    // Create chunk index
    const ordering = nodeAttrs.ordering === 'morton' ? 'morton' : 'hilbert';
    const chunkIndex: ChunkSpatialIndex = {
      metadata: {
        ordering,
        ordering_dims: orderingDims,
        slice_dims: sliceDims,
        ordering_bits_per_dim: nodeAttrs.ordering_bits_per_dim || 21,
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
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    // No chunk_bounds - this is expected for 3D datasets without spatial ordering
    if (
      message.includes('404') ||
      message.includes('Not Found') ||
      message.includes('Node not found')
    ) {
      log.info(Modules.SPATIAL_INDEX, 'No chunk_bounds found - dataset has no spatial indexing');
    } else {
      log.warning(Modules.SPATIAL_INDEX, `Could not load chunk_bounds: ${message}`);
    }
    return null;
  }
}

/**
 * Query chunks whose bounding boxes intersect the view region.
 *
 * Uses AABB (axis-aligned bounding box) intersection test in nD space.
 * A chunk intersects if its bounds overlap the query box in ALL dimensions.
 * This is the core spatial query operation for efficient nD point loading.
 *
 * Algorithm:
 * 1. For each chunk, test intersection in all dimensions
 * 2. Chunk intersects if: chunkMax >= queryMin AND chunkMin <= queryMax (for every dimension)
 * 3. Early exit on first non-intersecting dimension (optimization)
 * 4. Return indices of all matching chunks
 *
 * @param index - Chunk spatial index containing:
 *                - metadata: Chunk configuration (total_chunks, ndim, etc.)
 *                - chunkBounds: Float32Array of bounding boxes, shape (num_chunks, ndim, 2) flattened
 *                  Layout: [chunk0_dim0_min, chunk0_dim0_max, chunk0_dim1_min, ...]
 *
 * @param slicePosition - Current position in nD space, one value per dimension.
 *                        Array length must equal index.metadata.ndim.
 *                        Example: [0, 2.5, 1, 0, 0] for 5D dataset
 *
 * @param tolerance - Search radius per dimension in world units.
 *                    Array length must equal index.metadata.ndim.
 *                    For displayed dimensions, typically 0 (not used in query).
 *                    For slice dimensions, defines "slice thickness".
 *                    Example: [0, 1.0, 0, 0, 0] = ±1.0 units in dimension 1
 *
 * @returns Array of chunk indices that intersect the query region.
 *          Indices are in range [0, total_chunks).
 *          Returns empty array if no chunks intersect.
 *          Typical result size: 1-20 chunks for well-distributed data.
 *
 * @example
 * ```typescript
 * // Query chunks at position [0, 5.0, 0] with tolerance [0, 2.0, 0]
 * const chunks = queryChunksForView(
 *   spatialIndex,
 *   [0, 5.0, 0],
 *   [0, 2.0, 0]
 * );
 * console.log(`Found ${chunks.length} intersecting chunks`);
 * // Typical output: "Found 3 intersecting chunks"
 * ```
 *
 * @example
 * ```typescript
 * // Query all chunks (infinite tolerance)
 * const tolerance = Array(ndim).fill(Infinity);
 * const allChunks = queryChunksForView(index, slicePosition, tolerance);
 * console.log(`Total chunks: ${allChunks.length}`);
 * ```
 *
 * @remarks Performance: O(total_chunks × ndim), typically 500-5000 comparisons for standard datasets.
 *              Fast due to early exit optimization and linear memory access pattern.
 *
 * @see {@link loadChunkSpatialIndex} for index creation
 * @see SPECIFICATIONS.md - Section 2.2 for AABB intersection algorithm
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
