import type { ViewState } from '../../data-loader-types';
import type { LoadRange } from '../base-types';
import type { ChunkSpatialIndex } from './spatial-query-builder';

/** Minimal first-axis layout needed to locate zarr chunk boundaries. */
export interface FirstAxisChunkLayout {
  shape: readonly number[];
  chunks: readonly number[];
}

/** Return the sole moved hidden dimension, or null for no/multi-axis motion. */
function movedHiddenDimension(current: ViewState, predicted: ViewState): number | null {
  let moved: number | null = null;
  for (let dim = 0; dim < current.slicePosition.length; dim++) {
    if (current.displayDims.includes(dim)) continue;
    if (current.slicePosition[dim] === predicted.slicePosition[dim]) continue;
    if (moved !== null) return null;
    moved = dim;
  }
  return moved;
}

/** Map an element row to the hidden-axis bound of its spatial-index atom. */
function boundaryPosition(
  index: ChunkSpatialIndex,
  row: number,
  dimension: number,
  forward: boolean
): number | null {
  const atomRows = index.metadata.chunk_size;
  if (!atomRows || atomRows <= 0) return null;
  const atom = Math.floor(row / atomRows);
  if (atom < 0 || atom >= index.chunkCount) return null;
  const offset = atom * index.metadata.ndim * 2 + dimension * 2;
  const position = index.chunkBounds[offset + (forward ? 0 : 1)];
  return Number.isFinite(position) ? position : null;
}

/** Return the first row in the adjacent zarr chunk in playback direction. */
function nextBoundaryRow(edge: number, chunkRows: number, forward: boolean): number {
  return forward
    ? Math.ceil(edge / chunkRows) * chunkRows
    : Math.floor(edge / chunkRows) * chunkRows - 1;
}

/** Select the leading visible row edge in playback direction. */
function rangeEdge(ranges: readonly LoadRange[], forward: boolean): number {
  return forward
    ? Math.max(...ranges.map((range) => range.end))
    : Math.min(...ranges.map((range) => range.start));
}

/** Validate and return the row count and first-axis chunk size. */
function usableFirstAxisLayout(array: FirstAxisChunkLayout): [number, number] | null {
  // `chunks[0]` is the zarr chunk-grid shape. For zarr-3 sharded arrays it is
  // the shard size, not the byte-range-readable inner chunk size, so this
  // planner must not be reused for sharded stores without inner-grid metadata.
  const rowCount = array.shape[0] ?? 0;
  const chunkRows = array.chunks[0] ?? 0;
  return rowCount > 0 && chunkRows > 0 && chunkRows < rowCount ? [rowCount, chunkRows] : null;
}

/** Test whether a candidate coordinate reaches or passes the predicted slice. */
function reachesPrediction(position: number, predictedPosition: number, forward: boolean): boolean {
  return forward ? position >= predictedPosition : position <= predictedPosition;
}

/** Inputs for one array's next-boundary search. */
interface BoundarySearch {
  array: FirstAxisChunkLayout;
  index: ChunkSpatialIndex;
  edge: number;
  dimension: number;
  predictedPosition: number;
  forward: boolean;
}

/** Find the first future index atom that enters the array's adjacent zarr chunk. */
function nextArrayBoundaryPosition(search: BoundarySearch): number | null {
  const layout = usableFirstAxisLayout(search.array);
  if (!layout) return null;
  const [rowCount, chunkRows] = layout;

  let row = nextBoundaryRow(search.edge, chunkRows, search.forward);
  while (row >= 0 && row < rowCount) {
    const position = boundaryPosition(search.index, row, search.dimension, search.forward);
    if (
      position !== null &&
      reachesPrediction(position, search.predictedPosition, search.forward)
    ) {
      return position;
    }
    row += search.forward ? chunkRows : -chunkRows;
  }
  return null;
}

/** Find the nearest qualifying boundary across the loader's arrays. */
function nearestArrayBoundaryPosition(
  arrays: readonly FirstAxisChunkLayout[],
  search: Omit<BoundarySearch, 'array'>
): number | null {
  let nearest: number | null = null;
  for (const array of arrays) {
    const position = nextArrayBoundaryPosition({ ...search, array });
    if (
      position !== null &&
      (nearest === null ||
        Math.abs(position - search.predictedPosition) <
          Math.abs(nearest - search.predictedPosition))
    ) {
      nearest = position;
    }
  }
  return nearest;
}

/**
 * Plan the predicted slice plus the nearest next zarr chunk boundary across all arrays.
 *
 * The index atom bounds translate first-axis row boundaries back into hidden-axis
 * coordinates. Multi-axis motion and unusable metadata retain the one-step plan.
 */
export function planChunkBoundaryViewStates(
  current: ViewState,
  predicted: ViewState,
  ranges: readonly LoadRange[],
  index: ChunkSpatialIndex,
  arrays: readonly FirstAxisChunkLayout[]
): ViewState[] {
  const dimension = movedHiddenDimension(current, predicted);
  if (dimension === null || ranges.length === 0) return [predicted];

  const delta = predicted.slicePosition[dimension] - current.slicePosition[dimension];
  if (!Number.isFinite(delta) || delta === 0) return [predicted];
  const forward = delta > 0;
  const edge = rangeEdge(ranges, forward);
  const predictedPosition = predicted.slicePosition[dimension];
  const nearestBoundary = nearestArrayBoundaryPosition(arrays, {
    index,
    edge,
    dimension,
    predictedPosition,
    forward,
  });

  const boundaryPositions =
    nearestBoundary === null || nearestBoundary === predictedPosition ? [] : [nearestBoundary];
  return [predictedPosition, ...boundaryPositions].map((position) => {
    const slicePosition = [...predicted.slicePosition];
    slicePosition[dimension] = position;
    return { ...predicted, slicePosition };
  });
}
