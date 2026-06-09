/**
 * Shared cache-warming read for the Points / Lines / GSplats spatial-index
 * loaders' `prefetchChunks`.
 *
 * Fires a zarr `get()` for every (array × range) pair and discards the result:
 * the read populates the L0 / L1 / L2 caches as a side-effect so the next
 * demand `updateView()` is a fast cache hit, with no full-size output buffers
 * allocated only to be thrown away.
 *
 * Deliberately NOT a `RangeLoader.loadDirectTyped` call. Prefetch warms FUTURE
 * frames, so — unlike a demand read — it allocates no typed output and carries
 * no per-update abort signal (a demand-side supersede must not cancel warming;
 * the demand reader owns that signal). Keeping it separate is what makes the
 * three loaders' prefetch paths a single shared helper instead of three copies.
 *
 * @module data/loaders/spatial-query/prefetch-ranges
 */

import * as zarr from '../../zarr';
import { get } from '../../zarr';
import { firstAxisRangeSlice } from './range-loader/encoding-types';
import type { LoadRange } from '../base-types';

/**
 * Warm the cache for `ranges` across every array in `arrays`.
 *
 * @param arrays - The (already cache-wrapped) attribute arrays to warm. Callers
 *   filter out absent optional arrays before passing.
 * @param ranges - First-axis ranges (`{ start, end }`) to fetch on each array;
 *   `firstAxisRangeSlice` extends each to a full slice over the trailing axes.
 */
export async function prefetchRangesIntoCache(
  arrays: ReadonlyArray<zarr.Array<zarr.DataType, zarr.Readable>>,
  ranges: ReadonlyArray<LoadRange>
): Promise<void> {
  const fetches: Promise<unknown>[] = [];
  for (const array of arrays) {
    const shape = array.shape;
    for (const range of ranges) {
      fetches.push(get(array, firstAxisRangeSlice(shape, range)));
    }
  }
  await Promise.all(fetches);
}
