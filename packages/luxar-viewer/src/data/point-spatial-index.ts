/**
 * Point spatial index functionality for efficient nD point queries.
 *
 * The point spatial index uses a regular grid partitioning of nD space,
 * stored as a sparse representation containing only occupied cells.
 * This implementation is specifically designed for point data.
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';
import { PointSpatialIndexMetadata, PointSpatialIndex, PointRange } from '../types/point-spatial-index';
import { log, Modules } from '../utils/log';
import { config } from '../config';

// Export types
export type { PointSpatialIndexMetadata, PointSpatialIndex, PointRange };

/**
 * Load point spatial index from zarr group or location
 */
export async function loadPointSpatialIndex(groupOrLocation: any): Promise<PointSpatialIndex | null> {
  try {
    // Handle both zarr group and location objects
    let indexGroup;

    // If it's a location, try to open it as a group
    if (groupOrLocation.resolve && !groupOrLocation.members) {
      // It's a location, need to check for spatial_index
      try {
        indexGroup = await groupOrLocation.resolve('spatial_index');
      } catch {
        log.info(Modules.SPATIAL_INDEX, 'No spatial index found, using linear scanning');
        return null;
      }
    } else if (groupOrLocation.members) {
      // It's a group, check if it has spatial_index
      if (!groupOrLocation.members.has('spatial_index')) {
        log.info(Modules.SPATIAL_INDEX, 'No spatial index found, using linear scanning');
        return null;
      }
      indexGroup = await groupOrLocation.resolve('spatial_index');
    } else {
      // Unknown type, try to resolve directly
      try {
        indexGroup = await groupOrLocation.resolve('spatial_index');
      } catch {
        log.info(Modules.SPATIAL_INDEX, 'No spatial index found, using linear scanning');
        return null;
      }
    }

    // Try to open the index group to access its attributes
    let indexGroupOpened;
    try {
      indexGroupOpened = await zarr.open(indexGroup, { kind: 'group' });
    } catch (openError: any) {
      // This is expected when no spatial index exists - not an error
      if (openError.message?.includes('404') || openError.message?.includes('Not Found')) {
        log.info(
          Modules.SPATIAL_INDEX,
          'No spatial index exists for this dataset (this is normal for datasets without spatial indexing)'
        );
      } else {
        log.info(
          Modules.SPATIAL_INDEX,
          'Could not open spatial index group:',
          openError.message || openError
        );
      }
      return null;
    }

    // Load metadata from attributes - validated structure from Python
    // The attrs object from zarr doesn't have strong typing, so we cast to unknown first
    const metadata = indexGroupOpened.attrs as unknown as PointSpatialIndexMetadata;

    // Check if all dimensions are displayed (no spatial index needed)
    if (!metadata.indexed_dimensions || metadata.indexed_dimensions.length === 0) {
      log.info(Modules.SPATIAL_INDEX, 'All dimensions are displayed - no spatial index needed');
      // Return a valid but empty index
      return {
        metadata: {
          ...metadata,
          grid_shape: [],
          grid_origin: [],
          cell_size: [],
          num_occupied: 0,
          dimensions: 0,
        },
        occupiedCells: new Uint32Array(0),
        cellRanges: new BigUint64Array(0),
      };
    }

    if (!metadata.grid_shape || !metadata.grid_origin || !metadata.cell_size) {
      log.warning(Modules.SPATIAL_INDEX, 'Invalid spatial index metadata');
      return null;
    }

    // Load occupied cells array
    const occupiedCellsArray = await zarr.open(indexGroup.resolve('occupied_cells'), {
      kind: 'array',
    });
    const occupiedCellsData = await get(occupiedCellsArray);

    // Load cell ranges array
    const cellRangesArray = await zarr.open(indexGroup.resolve('cell_ranges'), { kind: 'array' });
    const cellRangesData = await get(cellRangesArray);

    log.info(
      Modules.SPATIAL_INDEX,
      `Loaded spatial index: ${metadata.num_occupied} occupied cells`
    );
    log.info(Modules.SPATIAL_INDEX, `  Grid shape: [${metadata.grid_shape.join(', ')}]`);
    log.info(
      Modules.SPATIAL_INDEX,
      `  Cell size: [${metadata.cell_size.map((v) => v.toFixed(2)).join(', ')}]`
    );

    return {
      metadata,
      occupiedCells: new Uint32Array(occupiedCellsData.data as ArrayBuffer),
      cellRanges: new BigUint64Array(cellRangesData.data as ArrayBuffer),
    };
  } catch (error) {
    log.error(Modules.SPATIAL_INDEX, 'Failed to load spatial index:', error);
    return null;
  }
}

/**
 * Query point spatial index for points within tolerance of a slice position
 */
export function queryPointSpatialIndex(
  index: PointSpatialIndex,
  slicePos: number[],
  tolerance: number[]
): PointRange[] {
  const { metadata, occupiedCells, cellRanges } = index;
  const D = metadata.dimensions; // Number of indexed dimensions

  // Validate input dimensions
  const fullDim = metadata.full_dimensions || D;
  if (slicePos.length < fullDim || tolerance.length < fullDim) {
    log.warning(
      Modules.SPATIAL_INDEX,
      `Dimension mismatch: expected ${fullDim} dimensions, got slicePos=${slicePos.length}, tolerance=${tolerance.length}`
    );
  }

  // If no dimensions are indexed (all displayed), return all points
  if (D === 0 || !metadata.indexed_dimensions || metadata.indexed_dimensions.length === 0) {
    log.info(Modules.SPATIAL_INDEX, 'No indexed dimensions - returning all points');
    // Use total_points if available, otherwise fall back to a large number
    const totalPoints = metadata.total_points || Number.MAX_SAFE_INTEGER;
    return [{ start: 0, end: totalPoints }];
  }

  // Extract only the indexed dimensions from slicePos and tolerance
  const indexedSlicePos = new Float32Array(D);
  const indexedTolerance = new Float32Array(D);

  for (let i = 0; i < D; i++) {
    const fullDimIdx = metadata.indexed_dimensions[i];
    indexedSlicePos[i] = slicePos[fullDimIdx] ?? 0; // Default to 0 if undefined
    indexedTolerance[i] = tolerance[fullDimIdx] ?? config.dataLoading.spatial.defaultTolerance; // Default tolerance
  }

  // Debug logging for indexed values
  log.info(
    Modules.SPATIAL_INDEX,
    `  Indexed slice position: [${Array.from(indexedSlicePos)
      .map((p) => p.toFixed(2))
      .join(', ')}]`
  );
  log.info(
    Modules.SPATIAL_INDEX,
    `  Indexed tolerance: [${Array.from(indexedTolerance)
      .map((t) => t.toFixed(3))
      .join(', ')}]`
  );

  // Calculate grid bounds to query (only for indexed dimensions)
  const minGrid = new Uint32Array(D);
  const maxGrid = new Uint32Array(D);

  for (let d = 0; d < D; d++) {
    const sliceVal = indexedSlicePos[d];
    const tolVal = indexedTolerance[d];

    if (tolVal === 0) {
      // For exact matching (discrete dimensions), find the single cell
      const cellIdx = Math.floor((sliceVal - metadata.grid_origin[d]) / metadata.cell_size[d]);
      minGrid[d] = Math.max(0, Math.min(metadata.grid_shape[d] - 1, cellIdx));
      maxGrid[d] = minGrid[d]; // Same cell for exact match
    } else {
      // For continuous dimensions with tolerance
      const minVal = sliceVal - tolVal;
      const maxVal = sliceVal + tolVal;

      minGrid[d] = Math.max(
        0,
        Math.floor((minVal - metadata.grid_origin[d]) / metadata.cell_size[d])
      );
      maxGrid[d] = Math.min(
        metadata.grid_shape[d] - 1,
        Math.ceil((maxVal - metadata.grid_origin[d]) / metadata.cell_size[d])
      );
    }
  }

  // Check each occupied cell
  const ranges: PointRange[] = [];
  const numOccupied = metadata.num_occupied;

  for (let i = 0; i < numOccupied; i++) {
    let inRange = true;

    // Check if cell is within query bounds
    for (let d = 0; d < D; d++) {
      const cellCoord = occupiedCells[i * D + d];
      if (cellCoord < minGrid[d] || cellCoord > maxGrid[d]) {
        inRange = false;
        break;
      }
    }

    if (inRange) {
      const start = Number(cellRanges[i * 2]);
      const end = Number(cellRanges[i * 2 + 1]);
      ranges.push({ start, end });
    }
  }

  // Log query results for debugging
  const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
  log.info(
    Modules.SPATIAL_INDEX,
    `Spatial query found ${ranges.length} cells with ${totalPoints} points`
  );

  // Debug: Show grid bounds that were queried
  log.info(Modules.SPATIAL_INDEX, 'Query grid bounds (indexed dimensions only):');
  for (let d = 0; d < D; d++) {
    const fullDimIdx = metadata.indexed_dimensions ? metadata.indexed_dimensions[d] : d;
    log.info(
      Modules.SPATIAL_INDEX,
      `  Indexed dim ${d} (full dim ${fullDimIdx}): cells [${minGrid[d]}, ${maxGrid[d]}] of [0, ${metadata.grid_shape[d] - 1}]`
    );
  }

  return ranges;
}

/**
 * Merge overlapping or adjacent point ranges for efficient loading
 */
export function mergePointRanges(ranges: PointRange[]): PointRange[] {
  if (ranges.length === 0) return [];

  // Sort by start index
  const sorted = [...ranges].sort((a, b) => a.start - b.start);

  const merged: PointRange[] = [];
  let current = sorted[0];

  for (let i = 1; i < sorted.length; i++) {
    const next = sorted[i];

    // Check if ranges overlap or are adjacent
    if (next.start <= current.end) {
      // Merge ranges
      current = {
        start: current.start,
        end: Math.max(current.end, next.end),
      };
    } else {
      // Add current range and start new one
      merged.push(current);
      current = next;
    }
  }

  merged.push(current);
  return merged;
}

/**
 * Calculate which zarr chunks need to be loaded for given point ranges
 */
export function calculateChunksToLoad(ranges: PointRange[], chunkSize: number): Set<number> {
  const chunks = new Set<number>();

  for (const range of ranges) {
    const startChunk = Math.floor(range.start / chunkSize);
    const endChunk = Math.floor((range.end - 1) / chunkSize);

    for (let chunk = startChunk; chunk <= endChunk; chunk++) {
      chunks.add(chunk);
    }
  }

  return chunks;
}

/**
 * Estimate memory usage for loading point ranges
 */
export function estimateMemoryUsage(ranges: PointRange[], bytesPerPoint: number): number {
  const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
  return totalPoints * bytesPerPoint;
}

/**
 * Create a debug summary of the point spatial index
 */
export function debugPointSpatialIndex(index: PointSpatialIndex): string {
  const { metadata } = index;
  const cellsPerDim = metadata.grid_shape.join('×');
  const originStr = metadata.grid_origin.map((v) => v.toFixed(2)).join(', ');
  const cellSizeStr = metadata.cell_size.map((v) => v.toFixed(2)).join('×');

  return (
    `Spatial Index: ${cellsPerDim} grid, ${metadata.num_occupied}/${metadata.total_cells} occupied, ` +
    `origin: [${originStr}], cell size: [${cellSizeStr}]`
  );
}
