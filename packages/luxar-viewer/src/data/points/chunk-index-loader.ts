/**
 * Points-specific chunk-index probe + array-bounds registration.
 *
 * Wraps the shared `fetchChunkBoundsArray` helper with the
 * Points-only validation rules (shape `(N, ndim, 2)`, dim coverage
 * reconciliation against `ordering_dims` ∪ `slice_dims`) and folds
 * the returned `Float32Array` into a `PointsChunkIndex` shape that
 * downstream callers (`SpatialQueryBuilder`, the points loader's
 * metric/log code) consume.
 *
 * Also re-exposes the prefetcher's `registerArrayBounds` call as a
 * tiny path-prefixing helper so the loader doesn't need to know how
 * the path is normalized.
 *
 * Extracted from `data/points-spatial-index-loader.ts` so each branch
 * (no-ordering short-circuit, shape-mismatch warning, dim-coverage
 * warning, success) can be unit-tested without a real zarr store.
 *
 * @module data/point-loader/chunk-index-loader
 */

import type { OrderingMethodName } from '../../types/format-contract';
import * as zarr from '../zarr';
import { log, Modules } from '../../utils/log';
import { fetchChunkBoundsArray } from '../loaders';
import type { ChunkPrefetcher } from '../../cache/chunk-prefetcher';

/** Subset of points-node attributes the chunk-index probe needs. */
export interface PointsNodeAttrsForIndex {
  ordering?: OrderingMethodName;
  ordering_dims?: number[];
  slice_dims?: number[];
  ordering_bits_per_dim?: number;
  chunk_size?: number;
  n_points?: number;
  ndim?: number;
}

/**
 * Points-specific chunk spatial index, extending the canonical
 * `{chunkBounds, chunkCount, metadata: {ndim, chunk_size}}` shape with
 * the descriptive ordering metadata the points loader logs and forwards
 * to `SpatialQueryBuilder`.
 */
export interface PointsChunkIndex {
  chunkBounds: Float32Array;
  chunkCount: number;
  metadata: {
    ordering: Exclude<OrderingMethodName, 'none'>;
    ordering_dims: number[];
    slice_dims: number[];
    ordering_bits_per_dim: number;
    chunk_size: number;
    total_points: number;
    total_chunks: number;
    ndim: number;
  };
}

/**
 * Probe `chunk_bounds` under `zarrLocation` and assemble a
 * `PointsChunkIndex`. Returns `null` for the two soft-fail cases
 * the loader already handled inline:
 *   - the node has no spatial ordering (`ordering` is undefined or
 *     `'none'`) — the dataset is 3D-without-Morton/Hilbert and the
 *     loader should fall back to "load all points";
 *   - the `chunk_bounds` array is missing or the shape's last dim
 *     isn't 2 — same fallback path.
 *
 * Logs warnings (does not raise) on length mismatches and on
 * dimension-coverage mismatches between `ordering_dims ∪ slice_dims`
 * and the array's `ndim`.
 */
export async function loadPointsChunkIndex(
  zarrLocation: zarr.Location<zarr.Readable>,
  nodeAttrs: PointsNodeAttrsForIndex
): Promise<PointsChunkIndex | null> {
  if (!nodeAttrs.ordering || nodeAttrs.ordering === 'none') {
    log.info(Modules.SPATIAL_INDEX, 'No spatial ordering — skipping chunk_bounds probe');
    return null;
  }

  const result = await fetchChunkBoundsArray(
    zarrLocation,
    'chunk_bounds',
    Modules.SPATIAL_INDEX,
    'No chunk_bounds found - dataset has no spatial indexing'
  );
  if (!result) return null;

  const [numChunks, ndim, lastDim] = result.shape;
  if (lastDim !== 2) {
    log.error(
      Modules.SPATIAL_INDEX,
      `Invalid chunk_bounds shape: expected [..., 2], got [..., ${lastDim}]`
    );
    return null;
  }

  const expectedLength = numChunks * ndim * 2;
  if (result.data.length !== expectedLength) {
    log.warning(
      Modules.SPATIAL_INDEX,
      `Chunk bounds array length mismatch: expected ${expectedLength} (${numChunks}×${ndim}×2), got ${result.data.length}`
    );
  }

  const positionDims = nodeAttrs.ndim;
  if (positionDims !== undefined && positionDims !== ndim) {
    log.warning(
      Modules.SPATIAL_INDEX,
      `Dimensionality mismatch: chunk_bounds has ${ndim}D but node attributes indicate ${positionDims}D`
    );
  }

  const orderingDims = nodeAttrs.ordering_dims ?? [];
  const sliceDims = nodeAttrs.slice_dims ?? [];
  const allDims = new Set([...orderingDims, ...sliceDims]);
  if (allDims.size > 0 && allDims.size !== ndim) {
    log.warning(
      Modules.SPATIAL_INDEX,
      `Dimension coverage mismatch: ordering_dims[${orderingDims.length}] + slice_dims[${sliceDims.length}] = ${allDims.size}, but ndim=${ndim}`
    );
  }

  const ordering = nodeAttrs.ordering === 'morton' ? 'morton' : 'hilbert';
  return {
    chunkBounds: result.data,
    chunkCount: numChunks,
    metadata: {
      ordering,
      ordering_dims: orderingDims,
      slice_dims: sliceDims,
      ordering_bits_per_dim: nodeAttrs.ordering_bits_per_dim ?? 21,
      chunk_size: nodeAttrs.chunk_size ?? 0,
      total_points: nodeAttrs.n_points ?? 0,
      total_chunks: numChunks,
      ndim,
    },
  };
}

/**
 * Register a Points child array's shape with the prefetcher so
 * subsequent range fetches that exceed the array's bounds can be
 * short-circuited (no spurious 404s). No-op when no prefetcher is
 * wired up.
 *
 * The path normalization mirrors the original inline call site: the
 * leading `/` (if present) is stripped from the node path, then
 * `arrayName` is appended.
 *
 * Mirrors `registerLinesArrayBounds` / `registerGSplatsArrayBounds`
 * (one helper per node type) so the three callers read uniformly.
 * The underlying generic primitive is
 * `ChunkPrefetcher.registerArrayBounds`, which keeps its bare name —
 * the prefetcher doesn't care which node type asked.
 */
export function registerPointsArrayBounds(
  prefetcher: ChunkPrefetcher | null | undefined,
  nodePath: string,
  arrayName: string,
  array: zarr.Array<zarr.DataType, zarr.Readable>
): void {
  if (!prefetcher) return;
  const trimmedPath = nodePath.startsWith('/') ? nodePath.slice(1) : nodePath;
  prefetcher.registerArrayBounds(`${trimmedPath}/${arrayName}`, array.shape, array.chunks);
}
