/**
 * Dual spatial index loading for Lines.
 *
 * Lines have two independent spatial orderings:
 * - Vertices: Ordered in D-dimensional space using space-filling curves
 * - Segments: Ordered in D-space with bounds including line width
 *
 * The segment bounds already include line width extent, so spatial queries
 * for non-displayed dimensions use tolerance = 0.
 *
 * @module data/lines-chunk-spatial-index
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';
import type { LinesMetadata, LinesChunkSpatialIndex, SegmentRange } from '../types/lines';
import { log, Modules } from '../utils/log';

/**
 * Load the dual spatial index for a Lines node.
 *
 * Reads both vertex_chunk_bounds and segment_chunk_bounds arrays from zarr.
 * Returns null if no spatial ordering is enabled.
 *
 * @param location - Zarr location of the lines group
 * @param attrs - Lines metadata from .zattrs
 * @returns LinesChunkSpatialIndex or null if no spatial ordering
 */
export async function loadLinesChunkSpatialIndex(
  location: zarr.Location<zarr.Readable>,
  attrs: LinesMetadata
): Promise<LinesChunkSpatialIndex | null> {
  // Check if spatial ordering is enabled
  if (attrs.ordering === 'none' || !attrs.vertex_ordering || !attrs.segment_ordering) {
    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `Lines node has no spatial ordering (ordering=${attrs.ordering})`
    );
    return null;
  }

  try {
    // Load vertex chunk bounds
    const vertexBoundsArray = await zarr.open(location.resolve('vertex_chunk_bounds'), {
      kind: 'array',
    });
    const vertexBoundsData = await get(vertexBoundsArray);
    const vertexChunkBounds = new Float32Array(
      vertexBoundsData.data as ArrayBuffer | ArrayLike<number>
    );

    // Load segment chunk bounds
    const segmentBoundsArray = await zarr.open(location.resolve('segment_chunk_bounds'), {
      kind: 'array',
    });
    const segmentBoundsData = await get(segmentBoundsArray);
    const segmentChunkBounds = new Float32Array(
      segmentBoundsData.data as ArrayBuffer | ArrayLike<number>
    );

    // Compute chunk counts from metadata
    const vertexChunkCount = Math.ceil(attrs.n_vertices / attrs.vertex_ordering.chunk_size);
    const segmentChunkCount = Math.ceil(attrs.n_segments / attrs.segment_ordering.chunk_size);

    // Validate bounds array sizes
    const expectedVertexSize = vertexChunkCount * attrs.ndim * 2;
    const expectedSegmentSize = segmentChunkCount * attrs.ndim * 2;

    if (vertexChunkBounds.length !== expectedVertexSize) {
      log.warning(
        Modules.SPATIAL_INDEX_LOADER,
        `Vertex bounds size mismatch: got ${vertexChunkBounds.length}, expected ${expectedVertexSize}`
      );
    }

    if (segmentChunkBounds.length !== expectedSegmentSize) {
      log.warning(
        Modules.SPATIAL_INDEX_LOADER,
        `Segment bounds size mismatch: got ${segmentChunkBounds.length}, expected ${expectedSegmentSize}`
      );
    }

    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `Loaded Lines spatial index: ${vertexChunkCount} vertex chunks, ${segmentChunkCount} segment chunks`
    );

    return {
      metadata: attrs,
      vertexChunkBounds,
      segmentChunkBounds,
      vertexChunkCount,
      segmentChunkCount,
    };
  } catch (error: any) {
    // No spatial index - expected for datasets without ordering
    if (
      error.message?.includes('404') ||
      error.message?.includes('Not Found') ||
      error.message?.includes('Node not found')
    ) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        'No chunk bounds found - Lines dataset has no spatial indexing'
      );
    } else {
      log.error(Modules.SPATIAL_INDEX_LOADER, 'Failed to load Lines spatial index:', error);
    }
    return null;
  }
}

/**
 * Query segment chunks that intersect the view region.
 *
 * IMPORTANT: segment_chunk_bounds already include line width extent.
 * Tolerance for spatial (non-displayed) dimensions should be 0.
 *
 * @param index - Lines spatial index
 * @param slicePosition - Current nD position
 * @param tolerance - Per-dimension tolerance (1e10 for displayed, 0.5 for discrete, 0 for spatial)
 * @returns Array of segment chunk indices
 */
export function querySegmentChunksForView(
  index: LinesChunkSpatialIndex,
  slicePosition: number[],
  tolerance: number[]
): number[] {
  const { segmentChunkBounds, segmentChunkCount } = index;
  const ndim = index.metadata.ndim;
  const matchingChunks: number[] = [];

  for (let chunk = 0; chunk < segmentChunkCount; chunk++) {
    let intersects = true;

    for (let dim = 0; dim < ndim; dim++) {
      // Layout: segmentChunkBounds[chunk, dim, 0/1] flattened row-major
      const offset = chunk * ndim * 2 + dim * 2;
      const chunkMin = segmentChunkBounds[offset];
      const chunkMax = segmentChunkBounds[offset + 1];

      const queryMin = slicePosition[dim] - tolerance[dim];
      const queryMax = slicePosition[dim] + tolerance[dim];

      // AABB intersection test: no overlap if completely before or after
      if (chunkMax < queryMin || chunkMin > queryMax) {
        intersects = false;
        break;
      }
    }

    if (intersects) {
      matchingChunks.push(chunk);
    }
  }

  log.query(
    Modules.SPATIAL_INDEX_LOADER,
    `Lines: Found ${matchingChunks.length}/${segmentChunkCount} segment chunks matching query`
  );

  return matchingChunks;
}

/**
 * Query vertex chunks that intersect the view region.
 *
 * Used for direct vertex loading when segment-based loading is not needed.
 *
 * @param index - Lines spatial index
 * @param slicePosition - Current nD position
 * @param tolerance - Per-dimension tolerance
 * @returns Array of vertex chunk indices
 */
export function queryVertexChunksForView(
  index: LinesChunkSpatialIndex,
  slicePosition: number[],
  tolerance: number[]
): number[] {
  const { vertexChunkBounds, vertexChunkCount } = index;
  const ndim = index.metadata.ndim;
  const matchingChunks: number[] = [];

  for (let chunk = 0; chunk < vertexChunkCount; chunk++) {
    let intersects = true;

    for (let dim = 0; dim < ndim; dim++) {
      const offset = chunk * ndim * 2 + dim * 2;
      const chunkMin = vertexChunkBounds[offset];
      const chunkMax = vertexChunkBounds[offset + 1];

      const queryMin = slicePosition[dim] - tolerance[dim];
      const queryMax = slicePosition[dim] + tolerance[dim];

      if (chunkMax < queryMin || chunkMin > queryMax) {
        intersects = false;
        break;
      }
    }

    if (intersects) {
      matchingChunks.push(chunk);
    }
  }

  log.query(
    Modules.SPATIAL_INDEX_LOADER,
    `Lines: Found ${matchingChunks.length}/${vertexChunkCount} vertex chunks matching query`
  );

  return matchingChunks;
}

/**
 * Compute tolerance array for Lines queries.
 *
 * CRITICAL: segment_chunk_bounds already include line width.
 * - Displayed dimensions: infinite tolerance (want all segments in view)
 * - Discrete dimensions: 0.5 (half-unit for integer matching)
 * - Spatial dimensions: 0 (bounds already include width!)
 *
 * @param sceneDimensions - Dimension metadata array
 * @param displayDims - Indices of displayed dimensions
 * @returns Tolerance array for spatial queries
 */
export function computeLinesTolerance(
  sceneDimensions: Array<{ discrete?: boolean; step?: number }>,
  displayDims: number[]
): number[] {
  return sceneDimensions.map((dim, idx) => {
    if (displayDims.includes(idx)) {
      // Displayed dimensions: infinite tolerance (want all segments in view)
      return 1e10;
    }
    if (dim.discrete) {
      // Discrete dimensions: use half of step size if available
      // This matches the encoder's padding strategy for tight bounds
      if (dim.step !== undefined && dim.step !== null) {
        return dim.step / 2;
      }
      // Fallback for backwards compatibility
      return 0.5;
    }
    // Spatial dimensions: zero - bounds already include width!
    return 0;
  });
}

/**
 * Convert segment chunk indices to segment ranges.
 *
 * @param chunkIndices - Array of chunk indices
 * @param chunkSize - Segments per chunk
 * @param totalSegments - Total segments in dataset
 * @returns Array of segment ranges
 */
export function segmentChunkIndicesToRanges(
  chunkIndices: number[],
  chunkSize: number,
  totalSegments: number
): SegmentRange[] {
  return chunkIndices.map((chunkIdx) => {
    const start = chunkIdx * chunkSize;
    const end = Math.min((chunkIdx + 1) * chunkSize, totalSegments);
    return { start, end };
  });
}

/**
 * Convert vertex chunk indices to vertex ranges.
 *
 * @param chunkIndices - Array of chunk indices
 * @param chunkSize - Vertices per chunk
 * @param totalVertices - Total vertices in dataset
 * @returns Array of vertex ranges
 */
export function vertexChunkIndicesToRanges(
  chunkIndices: number[],
  chunkSize: number,
  totalVertices: number
): SegmentRange[] {
  return chunkIndices.map((chunkIdx) => {
    const start = chunkIdx * chunkSize;
    const end = Math.min((chunkIdx + 1) * chunkSize, totalVertices);
    return { start, end };
  });
}

/**
 * Merge overlapping or adjacent ranges.
 *
 * Sorts ranges by start position and merges any that overlap or are adjacent.
 * This reduces the number of zarr chunk loads needed.
 *
 * @param ranges - Array of segment/vertex ranges
 * @returns Merged array of ranges
 */
export function mergeRanges(ranges: SegmentRange[]): SegmentRange[] {
  if (ranges.length === 0) return [];

  // Sort by start position
  const sorted = [...ranges].sort((a, b) => a.start - b.start);

  const merged: SegmentRange[] = [];
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

  if (merged.length !== ranges.length) {
    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `Merged ${ranges.length} ranges → ${merged.length} continuous ranges`
    );
  }

  return merged;
}

/**
 * Compute which vertex chunks contain a set of vertex indices.
 *
 * Given a sorted list of vertex indices, determine which chunks need to be loaded.
 *
 * @param vertexIndices - Sorted array of vertex indices
 * @param chunkSize - Vertices per chunk
 * @returns Array of chunk indices
 */
export function computeVertexChunksForIndices(
  vertexIndices: number[],
  chunkSize: number
): number[] {
  if (vertexIndices.length === 0) return [];

  const chunkSet = new Set<number>();
  for (const idx of vertexIndices) {
    chunkSet.add(Math.floor(idx / chunkSize));
  }

  return Array.from(chunkSet).sort((a, b) => a - b);
}

/**
 * Compute continuous vertex ranges from a sorted list of indices.
 *
 * Identifies runs of consecutive indices and groups them into ranges
 * for efficient batch loading.
 *
 * @param sortedIndices - Sorted array of vertex indices
 * @returns Array of ranges covering all indices
 */
export function computeVertexRangesFromIndices(sortedIndices: number[]): SegmentRange[] {
  if (sortedIndices.length === 0) return [];

  const ranges: SegmentRange[] = [];
  let rangeStart = sortedIndices[0];
  let rangeEnd = sortedIndices[0] + 1;

  for (let i = 1; i < sortedIndices.length; i++) {
    const idx = sortedIndices[i];

    if (idx === rangeEnd) {
      // Consecutive - extend current range
      rangeEnd++;
    } else {
      // Gap - save current range and start new one
      ranges.push({ start: rangeStart, end: rangeEnd });
      rangeStart = idx;
      rangeEnd = idx + 1;
    }
  }

  // Don't forget the last range
  ranges.push({ start: rangeStart, end: rangeEnd });

  return ranges;
}
