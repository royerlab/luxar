import type { ViewState } from '../../data-loader-types';
import type { LoadRange } from '../base-types';
import type { ChunkSpatialIndex } from './spatial-query-builder';

export interface FirstAxisChunkLayout {
  shape: readonly number[];
  chunks: readonly number[];
}

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

function nextBoundaryRow(edge: number, chunkRows: number, forward: boolean): number {
  return forward
    ? Math.ceil(edge / chunkRows) * chunkRows
    : Math.floor(edge / chunkRows) * chunkRows - 1;
}

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
  const edge = forward
    ? Math.max(...ranges.map((range) => range.end))
    : Math.min(...ranges.map((range) => range.start));
  const positions = new Set<number>([predicted.slicePosition[dimension]]);

  for (const array of arrays) {
    const rowCount = array.shape[0] ?? 0;
    const chunkRows = array.chunks[0] ?? 0;
    if (rowCount <= 0 || chunkRows <= 0 || chunkRows >= rowCount) continue;

    let row = nextBoundaryRow(edge, chunkRows, forward);
    while (row >= 0 && row < rowCount) {
      const position = boundaryPosition(index, row, dimension, forward);
      if (
        position !== null &&
        (forward
          ? position > current.slicePosition[dimension]
          : position < current.slicePosition[dimension])
      ) {
        positions.add(position);
        break;
      }
      row += forward ? chunkRows : -chunkRows;
    }
  }

  return [...positions]
    .sort(
      (left, right) =>
        Math.abs(left - current.slicePosition[dimension]) -
        Math.abs(right - current.slicePosition[dimension])
    )
    .map((position) => {
      const slicePosition = [...predicted.slicePosition];
      slicePosition[dimension] = position;
      return { ...predicted, slicePosition };
    });
}
