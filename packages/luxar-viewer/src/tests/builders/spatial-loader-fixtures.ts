/**
 * Shared test fixtures for the three spatial-index loader test files —
 * `data/points/spatial-index-loader.test.ts`,
 * `data/lines/spatial-index-loader.test.ts`,
 * `data/gsplats/spatial-index-loader.test.ts`.
 *
 * Per the cross-geometry symmetry rule: each geometry's
 * loader has the same `zarr.Location` and `SpatialQueryBuilder` mock
 * boilerplate. The pieces that work cleanly through hoisting live
 * here; the `vi.mock(...)` calls themselves still belong in each test
 * file (vi.mock paths resolve relative to the calling file).
 */

import { vi, type Mock } from 'vitest';

/**
 * Build a stub `zarr.Location`. `resolve(path)` returns a string, so
 * test-side `zarr.open` mocks can match by path substring (e.g.
 * `path.includes('vertices')`).
 */
export function makeMockZarrLocation(prefix = 'mock'): { resolve: Mock } {
  return {
    resolve: vi.fn().mockImplementation((path: string) => `${prefix}://${path}`),
  };
}

/**
 * Default chunk-bounds zarr array shape for points / gsplats: one bounds
 * pair (min,max) per chunk, per dimension. `[chunks, ndim, 2]`.
 */
export function makeChunkBoundsArray(chunkCount: number, ndim: number) {
  return {
    shape: [chunkCount, ndim, 2],
    dtype: 'float32',
    attrs: {},
  };
}

/**
 * Mock zarr array descriptor — the minimal shape `zarr.open` returns
 * that the loaders care about (shape + dtype + optional attrs).
 */
export function makeMockZarrArray(shape: number[], dtype = 'float32', attrs: object = {}) {
  return { shape, dtype, attrs };
}

/**
 * Build a `Float32Array` of zero-filled chunk-bounds data. Used as the
 * `data` field of `zarr.get`'s response when probing chunk_bounds. All
 * chunks therefore intersect every query region.
 */
export function makeChunkBoundsBuffer(chunkCount: number, ndim: number): Float32Array {
  return new Float32Array(chunkCount * ndim * 2);
}

/**
 * Default ranges returned by the mocked SpatialQueryBuilder.execute()
 * — two contiguous-ish ranges so the loader's range-merge / accumulator
 * logic gets exercised.
 */
export const DEFAULT_VISIBLE_RANGES = [
  { start: 0, end: 100 },
  { start: 200, end: 300 },
] as const;
