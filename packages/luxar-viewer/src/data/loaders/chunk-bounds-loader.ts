/**
 * Shared chunk-bounds zarr probe used by Points, Lines, and GSplats loaders.
 *
 * All three geometry types expose a `chunk_bounds` (or `vertex_chunk_bounds`
 * / `segment_chunk_bounds`) array of shape `(numChunks, ndim, 2)` storing
 * per-chunk min/max bounds along each dimension. The fetch logic — open the
 * zarr array, decode the data, soft-fallback on 404 / Not Found — is the
 * same for every caller, so it lives here.
 *
 * Loader-specific shape validation, dimensionality reconciliation, and
 * metadata wrapping stay in the individual loaders, which receive the raw
 * `Float32Array` plus the original zarr shape from this helper.
 *
 * @module data/loaders/chunk-bounds-loader
 */

import * as zarr from '../zarr';
import { readArray } from '../zarr';
import { log } from '../../utils/log';

/**
 * Result of a successful chunk-bounds zarr probe.
 *
 * `data` is the flattened `(numChunks * ndim * 2)` Float32Array produced
 * by reading the zarr array. `shape` is the original zarr shape (typically
 * `[numChunks, ndim, 2]`) so callers can run their own shape sanity checks.
 */
export interface ChunkBoundsArray {
  data: Float32Array;
  shape: readonly number[];
}

/** Heuristic for "the zarr array does not exist on the server / file system". */
function isNotFoundError(message: string): boolean {
  return (
    message.includes('404') || message.includes('Not Found') || message.includes('Node not found')
  );
}

/**
 * Open a chunk-bounds-style zarr array under ``location`` and return its
 * contents as a Float32Array.
 *
 * Returns ``null`` for the expected "array missing" case — datasets without
 * spatial ordering legitimately omit the array — so callers can fall back
 * to loading all elements without inspecting the error path. Other failures
 * (corrupt zarr, network errors) also return ``null`` after a warning so
 * the loader never blocks the scene on a degraded spatial-index probe.
 *
 * @param location  zarrita location of the parent group (the loader's node).
 * @param arrayName Zarr array key under ``location`` (``"chunk_bounds"``,
 *                  ``"vertex_chunk_bounds"``, ``"segment_chunk_bounds"``).
 * @param logModule Log-module label used by the loader (geometry-specific).
 * @param notFoundMessage Friendly message logged when the array is absent.
 */
export async function fetchChunkBoundsArray(
  location: zarr.Location<zarr.Readable>,
  arrayName: string,
  logModule: string,
  notFoundMessage: string
): Promise<ChunkBoundsArray | null> {
  try {
    const boundsArray = await zarr.open(location.resolve(arrayName), { kind: 'array' });
    const boundsData = await readArray(boundsArray);
    const data = new Float32Array(boundsData.data as ArrayBuffer | ArrayLike<number>);
    return { data, shape: Array.from(boundsArray.shape) };
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    if (isNotFoundError(message)) {
      log.info(logModule, notFoundMessage);
      return null;
    }
    log.warning(logModule, `Could not load ${arrayName}: ${message}`);
    return null;
  }
}
