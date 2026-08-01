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
 * The lines-specific vertex-index helpers live here too —
 * `sortedUniqueVertexIndices`, `computeVertexRangesFromIndices`, and
 * `remapSegmentIndices` (run after a query returns segment indices, to
 * coalesce the unique referenced vertex indices into contiguous runs
 * and remap them to local buffer positions). They operate on
 * per-segment vertex indices, not on chunk bounds, but conceptually
 * belong to the lines spatial-index path.
 *
 * @module data/lines/chunk-index-loader
 */

import * as zarr from '../zarr';
import { log, Modules } from '../../utils/log';
import { fetchChunkBoundsArray, type ChunkSpatialIndex } from '../loaders';
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
 * Sorted, de-duplicated vertex indices from raw per-segment index data.
 *
 * Uses a typed-array sort + single-pass dedupe rather than a JS `Set`,
 * which V8 caps at 2^24 (16,777,216) entries; the next `.add` throws
 * `RangeError: Set maximum size exceeded`. A lines node referencing more
 * than 2^24 unique vertex indices therefore silently failed to load
 * (issue #1049). Numeric ascending; the returned view aliases a fresh
 * copy, so the caller's `segmentData` is not mutated.
 */
export function sortedUniqueVertexIndices(segmentData: Uint32Array): Uint32Array {
  if (segmentData.length === 0) return new Uint32Array(0);
  const sorted = segmentData.slice(); // copy; typed-array .sort() is numeric ascending
  sorted.sort();
  // In-place dedupe: at step i we compare sorted[i] against sorted[i-1]
  // BEFORE writing to sorted[n], and n <= i always holds, so we never
  // clobber an element we have not yet read.
  let n = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (i === 0 || sorted[i] !== sorted[i - 1]) sorted[n++] = sorted[i];
  }
  return sorted.subarray(0, n);
}

/**
 * Compute contiguous vertex ranges from a sorted list of vertex indices.
 *
 * Used after loading segment data: each segment references two vertex
 * indices, and we batch the unique sorted indices into runs of consecutive
 * integers so zarr loading touches the minimum number of chunks.
 *
 * Accepts any `ArrayLike<number>` (a plain `number[]` or the `Uint32Array`
 * returned by {@link sortedUniqueVertexIndices}); it only reads `.length`
 * and integer indices.
 *
 * This is genuinely lines-specific (operates on per-segment vertex indices,
 * not on chunk bounds) and is therefore not in the canonical
 * `loaders/spatial-query-builder` API.
 */
export function computeVertexRangesFromIndices(sortedIndices: ArrayLike<number>): SegmentRange[] {
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

/**
 * Remap each global vertex index in `segmentData` to its local position in the
 * concatenated, loaded vertex buffer, writing results into `out` (must have
 * length >= segmentData.length). `mergedVertexRanges` must be sorted ascending
 * and pairwise-disjoint, as produced by `mergeRanges(computeVertexRangesFromIndices(...))`.
 * Returns the total local vertex count (the sum of the range widths).
 *
 * Uses a prefix-offset table + binary search over the ranges rather than a
 * global->local `Map`: V8 caps a `Map` at 2^24 entries and throws, so a lines
 * node with more than 2^24 unique vertex indices otherwise failed to load at
 * the remap stage even after the dedupe was fixed (issue #1049). Throws if a
 * segment references an index outside the loaded ranges.
 */
export function remapSegmentIndices(
  segmentData: Uint32Array,
  mergedVertexRanges: readonly { start: number; end: number }[],
  out: Uint32Array
): number {
  const nRanges = mergedVertexRanges.length;
  const offsets = new Array<number>(nRanges);
  let total = 0;
  for (let j = 0; j < nRanges; j++) {
    offsets[j] = total;
    total += mergedVertexRanges[j].end - mergedVertexRanges[j].start;
  }
  for (let i = 0; i < segmentData.length; i++) {
    const g = segmentData[i];
    // binary search for the range [start, end) containing g
    let lo = 0;
    let hi = nRanges - 1;
    let found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const r = mergedVertexRanges[mid];
      if (g < r.start) hi = mid - 1;
      else if (g >= r.end) lo = mid + 1;
      else {
        found = mid;
        break;
      }
    }
    if (found === -1) {
      throw new Error(`Vertex index ${g} not found in loaded data`);
    }
    out[i] = offsets[found] + (g - mergedVertexRanges[found].start);
  }
  return total;
}
