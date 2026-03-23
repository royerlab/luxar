/**
 * Spatial index loading and querying for GSplats.
 *
 * GSplats use space-filling curves (Morton or Hilbert) for chunk-based loading.
 * Each chunk has a bounding box that includes splat extents (based on Cholesky factors).
 *
 * @module data/gsplats-chunk-spatial-index
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';
import type {
  GSplatsMetadata,
  GSplatsChunkSpatialIndex,
  SplatRange,
  GSplatsViewState,
} from '../types/gsplats';
import type { DimensionMetadata } from '../types/dims';
import { log, Modules } from '../utils/log';

/**
 * Load the spatial index for a GSplats node.
 *
 * Reads chunk_bounds array from zarr if spatial ordering is enabled.
 * Returns null if no spatial ordering is enabled.
 *
 * @param location - Zarr location of the gsplats group
 * @param attrs - GSplats metadata from .zattrs
 * @returns GSplatsChunkSpatialIndex or null if no spatial ordering
 */
export async function loadGSplatsChunkSpatialIndex(
  location: zarr.Location<zarr.Readable>,
  attrs: GSplatsMetadata
): Promise<GSplatsChunkSpatialIndex | null> {
  // Check if spatial ordering is enabled
  if (attrs.ordering === 'none') {
    log.info(
      Modules.GSPLATS_SPATIAL_INDEX_LOADER,
      `GSplats node has no spatial ordering (ordering=${attrs.ordering})`
    );
    return null;
  }

  try {
    // Load chunk bounds
    const boundsArray = await zarr.open(location.resolve('chunk_bounds'), {
      kind: 'array',
    });
    const boundsData = await get(boundsArray);
    const chunkBounds = new Float32Array(boundsData.data as ArrayBuffer | ArrayLike<number>);

    // Compute chunk count from metadata
    const chunkCount = Math.ceil(attrs.n_splats / attrs.chunk_size);

    // Validate bounds array size
    const expectedSize = chunkCount * attrs.ndim * 2;
    if (chunkBounds.length !== expectedSize) {
      log.warning(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        `GSplats bounds size mismatch: got ${chunkBounds.length}, expected ${expectedSize}`
      );
    }

    log.info(
      Modules.GSPLATS_SPATIAL_INDEX_LOADER,
      `Loaded GSplats spatial index: ${chunkCount} chunks, ${attrs.ndim}D`
    );

    return {
      metadata: attrs,
      chunkBounds,
      chunkCount,
    };
  } catch (error: unknown) {
    // No spatial index - expected for datasets without ordering
    const errorMessage = error instanceof Error ? error.message : String(error);
    if (
      errorMessage?.includes('404') ||
      errorMessage?.includes('Not Found') ||
      errorMessage?.includes('Node not found')
    ) {
      log.info(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        'No chunk bounds found - GSplats dataset has no spatial indexing'
      );
    } else {
      log.error(Modules.GSPLATS_SPATIAL_INDEX_LOADER, 'Failed to load GSplats spatial index:', error);
    }
    return null;
  }
}

/**
 * Query splat chunks that intersect the view region.
 *
 * @param index - GSplats spatial index
 * @param slicePosition - Current nD position
 * @param tolerance - Per-dimension tolerance (1e10 for displayed, 0 for hidden spatial)
 * @returns Array of chunk indices that may contain visible splats
 */
export function queryGSplatsChunksForView(
  index: GSplatsChunkSpatialIndex,
  slicePosition: number[],
  tolerance: number[]
): number[] {
  const { chunkBounds, chunkCount } = index;
  const ndim = index.metadata.ndim;
  const matchingChunks: number[] = [];

  for (let chunk = 0; chunk < chunkCount; chunk++) {
    let intersects = true;

    for (let dim = 0; dim < ndim; dim++) {
      // Layout: chunkBounds[chunk, dim, 0/1] flattened row-major
      const offset = chunk * ndim * 2 + dim * 2;
      const chunkMin = chunkBounds[offset];
      const chunkMax = chunkBounds[offset + 1];

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

  return matchingChunks;
}

/**
 * Convert chunk indices to splat ranges.
 *
 * @param chunkIndices - Array of chunk indices
 * @param chunkSize - Number of splats per chunk
 * @param totalSplats - Total number of splats in dataset
 * @returns Array of splat ranges
 */
export function chunkIndicesToSplatRanges(
  chunkIndices: number[],
  chunkSize: number,
  totalSplats: number
): SplatRange[] {
  return chunkIndices.map((chunk) => ({
    start: chunk * chunkSize,
    end: Math.min((chunk + 1) * chunkSize, totalSplats),
  }));
}

/**
 * Merge overlapping or adjacent splat ranges.
 *
 * @param ranges - Array of splat ranges (may overlap or be adjacent)
 * @returns Array of merged, non-overlapping ranges sorted by start
 */
export function mergeRanges(ranges: SplatRange[]): SplatRange[] {
  if (ranges.length === 0) return [];

  // Sort by start index
  const sorted = [...ranges].sort((a, b) => a.start - b.start);

  const merged: SplatRange[] = [sorted[0]];

  for (let i = 1; i < sorted.length; i++) {
    const current = sorted[i];
    const last = merged[merged.length - 1];

    // Merge if overlapping or adjacent
    if (current.start <= last.end) {
      last.end = Math.max(last.end, current.end);
    } else {
      merged.push(current);
    }
  }

  return merged;
}

/**
 * Compute tolerance for gsplats queries based on view state.
 *
 * For displayed dimensions: tolerance = 1e10 (include everything)
 * For hidden dimensions: tolerance based on Gaussian extent
 *
 * @param dimensions - Dimension metadata from scene
 * @param displayDims - Which dimensions are being displayed
 * @param defaultTolerance - Default tolerance for hidden dimensions (default: 3.0 = 3 sigma)
 * @returns Per-dimension tolerance array
 */
export function computeGSplatsTolerance(
  dimensions: DimensionMetadata[] | undefined,
  displayDims: number[],
  defaultTolerance: number = 3.0
): number[] {
  const ndim = dimensions?.length ?? displayDims.length;
  const tolerance = new Array(ndim).fill(0);

  for (let dim = 0; dim < ndim; dim++) {
    if (displayDims.includes(dim)) {
      // Displayed dimensions: include everything
      tolerance[dim] = 1e10;
    } else {
      // Hidden dimensions: check if discrete first
      const dimMeta = dimensions?.[dim];

      if (dimMeta?.discrete) {
        // Discrete dimensions need exact matching (0.5 tolerance for floating point safety)
        tolerance[dim] = 0.5;
      } else if (dimMeta?.step) {
        // Continuous dimensions: use step size as tolerance
        tolerance[dim] = dimMeta.step * defaultTolerance;
      } else {
        tolerance[dim] = defaultTolerance;
      }
    }
  }

  return tolerance;
}

/**
 * Compute tolerance from GSplatsViewState.
 *
 * Always computes fresh tolerance for gsplats spatial queries.
 * Note: viewState.tolerance (from ViewStateManager) is not used because it sets
 * displayed dimensions to 0, but gsplats need 1e10 for displayed dims to load
 * all visible data. We always compute fresh using computeGSplatsTolerance.
 *
 * @param viewState - Current view state
 * @returns Per-dimension tolerance array
 */
export function computeToleranceFromViewState(viewState: GSplatsViewState): number[] {
  // Always compute fresh tolerance for gsplats - don't use viewState.tolerance
  // because it's designed for points slicing, not gsplats spatial queries
  return computeGSplatsTolerance(viewState.dimensions, viewState.displayDims);
}
