/**
 * GSplats-specific chunk-index probe + array-bounds registration.
 *
 * Mirrors `points/chunk-index-loader.ts` and `lines/chunk-index-loader.ts`
 * with the gsplats single-bounds shape: a chunk lookup carries the
 * single `chunk_bounds` blob covering all splats, since the segment/
 * vertex split lines requires has no equivalent here.
 *
 * Wraps the shared `fetchChunkBoundsArray` helper with the gsplats-only
 * validation rule (chunk count reconciled against
 * `Math.ceil(n_splats / chunk_size)`, taking the smaller value when the
 * metadata and array disagree) and folds the returned `Float32Array`
 * into the canonical `ChunkSpatialIndex` shape that `SpatialQueryBuilder`
 * consumes.
 *
 * Also re-exposes the prefetcher's `registerArrayBounds` call as a
 * tiny path-prefixing helper so the gsplats facade doesn't need to
 * know how the path is normalized — same shape as the points/lines
 * equivalents.
 *
 * @module data/gsplats/chunk-index-loader
 */

import * as zarr from 'zarrita';
import { log, Modules } from '../../utils/log';
import { fetchChunkBoundsArray } from '../loaders/chunk-bounds-loader';
import { type ChunkSpatialIndex } from '../loaders';
import type { ChunkPrefetcher } from '../../cache';
import type { GSplatsMetadata } from '../../types/gsplats';

/**
 * Probe `chunk_bounds` under `zarrLocation` and assemble a
 * `ChunkSpatialIndex`. Returns `null` for the soft-fail cases the
 * gsplats facade handles inline:
 *   - `attrs.ordering === 'none'` — no spatial ordering, full load
 *     fallback;
 *   - the `chunk_bounds` array is missing on disk — full load
 *     fallback (legacy datasets without spatial indexing).
 *
 * Logs a warning (does not raise) on chunk-count mismatches between
 * `Math.ceil(n_splats / chunk_size)` and the array's element-count
 * implied count, taking the smaller of the two as the truth.
 */
export async function loadGSplatsChunkIndex(
  zarrLocation: zarr.Location<zarr.Readable>,
  attrs: GSplatsMetadata
): Promise<ChunkSpatialIndex | null> {
  if (attrs.ordering === 'none') {
    log.info(
      Modules.GSPLATS_SPATIAL_INDEX_LOADER,
      `GSplats node has no spatial ordering (ordering=${attrs.ordering})`
    );
    return null;
  }

  const result = await fetchChunkBoundsArray(
    zarrLocation,
    'chunk_bounds',
    Modules.GSPLATS_SPATIAL_INDEX_LOADER,
    'No chunk bounds found - GSplats dataset has no spatial indexing'
  );
  if (!result) return null;

  const chunkBounds = result.data;

  // Reconcile expected vs actual chunk count and use the smaller value.
  let chunkCount = Math.ceil(attrs.n_splats / attrs.chunk_size);
  const actualChunks = Math.floor(chunkBounds.length / (attrs.ndim * 2));
  if (chunkCount !== actualChunks) {
    log.warning(
      Modules.GSPLATS_SPATIAL_INDEX_LOADER,
      `GSplats chunk count mismatch: metadata implies ${chunkCount} chunks, ` +
        `but chunkBounds array has ${actualChunks} chunks — using min`
    );
    chunkCount = Math.min(chunkCount, actualChunks);
  }

  return {
    chunkBounds,
    chunkCount,
    metadata: { ndim: attrs.ndim, chunk_size: attrs.chunk_size },
  };
}

/**
 * Register a child array's shape with the prefetcher so subsequent
 * range fetches that exceed the array's bounds can be short-circuited
 * (no spurious 404s). No-op when no prefetcher is wired up.
 *
 * The path normalization mirrors the original inline call site: the
 * leading `/` (if present) is stripped from the node path, then
 * `arrayName` is appended.
 */
export function registerGSplatsArrayBounds(
  prefetcher: ChunkPrefetcher | null | undefined,
  nodePath: string,
  arrayName: string,
  array: zarr.Array<zarr.DataType, zarr.Readable>
): void {
  if (!prefetcher) return;
  const trimmedPath = nodePath.startsWith('/') ? nodePath.slice(1) : nodePath;
  prefetcher.registerArrayBounds(`${trimmedPath}/${arrayName}`, array.shape, array.chunks);
}
