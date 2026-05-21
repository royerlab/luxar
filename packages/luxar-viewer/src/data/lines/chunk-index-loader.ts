/**
 * Lines-specific chunk-index probe + array-bounds registration.
 *
 * Mirrors `points/chunk-index-loader.ts` but with the dual-bounds
 * shape lines requires: every chunk lookup carries *both* a vertex-
 * chunk-bounds blob and a segment-chunk-bounds blob, since a query
 * scan on the segment side still needs vertex-side metadata for byte
 * accounting and for re-enabling vertex-driven prefetching.
 *
 * Wraps the shared `fetchChunkBoundsArray` helper with the lines-only
 * validation rules (size mismatches against `n_vertices ×
 * vertex_ordering.chunk_size` and the segment-side equivalent) and
 * folds the two `Float32Array`s into the `LinesDualChunkIndex` shape
 * that `SpatialQueryBuilder` consumes via its segment side.
 *
 * Also re-exposes the prefetcher's `registerArrayBounds` call as a
 * tiny path-prefixing helper so the lines loader doesn't need to know
 * how the path is normalized — same shape as the points equivalent
 * in `points/chunk-index-loader.ts`.
 *
 * The lines-specific helper `computeVertexRangesFromIndices` (run
 * after a query returns segment indices, to coalesce the unique
 * referenced vertex indices into contiguous runs) lives here too —
 * it operates on per-segment vertex indices, not on chunk bounds, but
 * conceptually belongs to the lines spatial-index path.
 *
 * @module data/lines/chunk-index-loader
 */

import * as zarr from '../zarr';
import { log, Modules } from '../../utils/log';
import { fetchChunkBoundsArray } from '../loaders/chunk-bounds-loader';
import { type ChunkSpatialIndex } from '../loaders';
import type { ChunkPrefetcher } from '../../cache/chunk-prefetcher';
import type { LinesMetadata, SegmentRange } from '../../types/lines';

/**
 * Internal index shape: the lines loader carries both vertex and
 * segment chunk bounds. Only the segment side feeds `SpatialQueryBuilder`;
 * the vertex side is kept so the loader can compute byte counts and
 * (in the future) re-enable vertex-driven prefetching.
 */
export interface LinesDualChunkIndex {
  segmentIndex: ChunkSpatialIndex;
  vertexChunkBounds: Float32Array;
  vertexChunkCount: number;
}

/**
 * Probe both `vertex_chunk_bounds` and `segment_chunk_bounds` under
 * `zarrLocation` and assemble a `LinesDualChunkIndex`. Returns `null`
 * for the soft-fail cases the lines loader handles inline:
 *   - `attrs.ordering === 'none'` — no spatial ordering, full load
 *     fallback;
 *   - `attrs.vertex_ordering` or `attrs.segment_ordering` missing —
 *     metadata incomplete, full load fallback;
 *   - either `chunk_bounds` array is missing on disk — full load
 *     fallback (datasets without spatial indexing).
 *
 * Logs warnings (does not raise) on size mismatches between the
 * blob's element count and the expected `chunkCount × ndim × 2`.
 */
export async function loadLinesDualChunkIndex(
  zarrLocation: zarr.Location<zarr.Readable>,
  attrs: LinesMetadata
): Promise<LinesDualChunkIndex | null> {
  if (attrs.ordering === 'none' || !attrs.vertex_ordering || !attrs.segment_ordering) {
    log.info(
      Modules.LINES_LOADER,
      `Lines node has no spatial ordering (ordering=${attrs.ordering})`
    );
    return null;
  }

  const vertexResult = await fetchChunkBoundsArray(
    zarrLocation,
    'vertex_chunk_bounds',
    Modules.LINES_LOADER,
    'No chunk bounds found - Lines dataset has no spatial indexing'
  );
  if (!vertexResult) return null;

  const segmentResult = await fetchChunkBoundsArray(
    zarrLocation,
    'segment_chunk_bounds',
    Modules.LINES_LOADER,
    'No chunk bounds found - Lines dataset has no spatial indexing'
  );
  if (!segmentResult) return null;

  const vertexChunkBounds = vertexResult.data;
  const segmentChunkBounds = segmentResult.data;

  const vertexChunkCount = Math.ceil(attrs.n_vertices / attrs.vertex_ordering.chunk_size);
  const segmentChunkCount = Math.ceil(attrs.n_segments / attrs.segment_ordering.chunk_size);

  const expectedVertexSize = vertexChunkCount * attrs.ndim * 2;
  const expectedSegmentSize = segmentChunkCount * attrs.ndim * 2;
  if (vertexChunkBounds.length !== expectedVertexSize) {
    log.warning(
      Modules.LINES_LOADER,
      `Vertex bounds size mismatch: got ${vertexChunkBounds.length}, expected ${expectedVertexSize}`
    );
  }
  if (segmentChunkBounds.length !== expectedSegmentSize) {
    log.warning(
      Modules.LINES_LOADER,
      `Segment bounds size mismatch: got ${segmentChunkBounds.length}, expected ${expectedSegmentSize}`
    );
  }

  return {
    segmentIndex: {
      chunkBounds: segmentChunkBounds,
      chunkCount: segmentChunkCount,
      metadata: { ndim: attrs.ndim, chunk_size: attrs.segment_ordering.chunk_size },
    },
    vertexChunkBounds,
    vertexChunkCount,
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
export function registerLinesArrayBounds(
  prefetcher: ChunkPrefetcher | null | undefined,
  nodePath: string,
  arrayName: string,
  array: zarr.Array<zarr.DataType, zarr.Readable>
): void {
  if (!prefetcher) return;
  const trimmedPath = nodePath.startsWith('/') ? nodePath.slice(1) : nodePath;
  prefetcher.registerArrayBounds(`${trimmedPath}/${arrayName}`, array.shape, array.chunks);
}

/**
 * Compute contiguous vertex ranges from a sorted list of vertex indices.
 *
 * Used after loading segment data: each segment references two vertex
 * indices, and we batch the unique sorted indices into runs of consecutive
 * integers so zarr loading touches the minimum number of chunks.
 *
 * This is genuinely lines-specific (operates on per-segment vertex indices,
 * not on chunk bounds) and is therefore not in the canonical
 * `loaders/spatial-query-builder` API.
 */
export function computeVertexRangesFromIndices(sortedIndices: number[]): SegmentRange[] {
  if (sortedIndices.length === 0) return [];

  const ranges: SegmentRange[] = [];
  let rangeStart = sortedIndices[0];
  let rangeEnd = sortedIndices[0] + 1;

  for (let i = 1; i < sortedIndices.length; i++) {
    const idx = sortedIndices[i];
    if (idx === rangeEnd) {
      rangeEnd++;
    } else {
      ranges.push({ start: rangeStart, end: rangeEnd });
      rangeStart = idx;
      rangeEnd = idx + 1;
    }
  }
  ranges.push({ start: rangeStart, end: rangeEnd });

  return ranges;
}
