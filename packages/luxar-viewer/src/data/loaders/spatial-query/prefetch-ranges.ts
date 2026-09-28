/**
 * Shared cache-warming read for the Points / Lines / GSplats spatial-index
 * loaders' `prefetchChunks`.
 *
 * For every array, computes the DISTINCT chunk coordinates the first-axis
 * `ranges` touch (all chunks over the trailing axes) and warms each one once
 * via the L0 proxy's `warmChunk`: the chunk is fetched (L1/L2) and decoded into
 * L0, and nothing else happens — no zarrita `get()`, so no output selection is
 * allocated and filled only to be discarded, and no read of the owning
 * loader's demand-load hooks (its per-update abort signal and residency probe
 * stay untouched by prefetch traffic). An array not wrapped by L0 falls back to
 * a bare `getChunk` (fetch + decode, still no output assembly).
 *
 * Deliberately NOT a `RangeLoader.loadDirectTyped` call. Prefetch warms FUTURE
 * frames. Callers may supply their own speculative-work signal (for example a
 * ladder lookahead cancelled by a view change); it is intentionally distinct
 * from the active demand update's signal. Miss decodes count under
 * `decode.count.prefetch` unless that signal is tagged with another origin
 * (e.g. `'lookahead'`).
 *
 * @module data/loaders/spatial-query/prefetch-ranges
 */

import type * as zarr from '../../zarr';
import type { LoadRange } from '../base-types';
import { warmChunk } from '../../../cache/decompressed-chunk-cache/warm-chunk';

/**
 * Chunk coordinates covering rows `[range.start, range.end)` of an array with
 * `shape` / `chunks` — every chunk along the trailing axes. The range is
 * clamped to the array; an empty result means nothing to read.
 */
export function chunkCoordsForRange(
  shape: readonly number[],
  chunks: readonly number[],
  range: LoadRange
): number[][] {
  if (shape.length === 0) return [[]];
  const start = Math.max(0, range.start);
  const end = Math.min(range.end, shape[0]);
  if (end <= start) return [];
  let coords: number[][] = [];
  for (let c = Math.floor(start / chunks[0]); c <= Math.floor((end - 1) / chunks[0]); c++) {
    coords.push([c]);
  }
  for (let axis = 1; axis < shape.length; axis++) {
    const count = Math.ceil(shape[axis] / chunks[axis]);
    const next: number[][] = [];
    for (const prefix of coords) {
      for (let c = 0; c < count; c++) next.push([...prefix, c]);
    }
    coords = next;
  }
  return coords;
}

/** The distinct chunk coordinates `ranges` touch on one array. */
function distinctChunkCoords(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  ranges: ReadonlyArray<LoadRange>
): number[][] {
  const distinct = new Map<string, number[]>();
  for (const range of ranges) {
    for (const coords of chunkCoordsForRange(array.shape, array.chunks, range)) {
      distinct.set(coords.join(','), coords);
    }
  }
  return [...distinct.values()];
}

/**
 * Warm the cache for `ranges` across every array in `arrays`.
 *
 * @param arrays - The (normally L0-wrapped) attribute arrays to warm. Callers
 *   filter out absent optional arrays before passing.
 * @param ranges - First-axis ranges (`{ start, end }`); each covers every chunk
 *   over the trailing axes.
 * @param signal - Optional owner signal for cancelling speculative reads.
 */
export async function prefetchRangesIntoCache(
  arrays: ReadonlyArray<zarr.Array<zarr.DataType, zarr.Readable>>,
  ranges: ReadonlyArray<LoadRange>,
  signal?: AbortSignal
): Promise<void> {
  const warms: Promise<void>[] = [];
  for (const array of arrays) {
    for (const coords of distinctChunkCoords(array, ranges)) {
      warms.push(warmChunk(array, coords, { signal }));
    }
  }
  await Promise.all(warms);
}
