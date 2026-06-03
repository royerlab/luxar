/**
 * Unit tests for the gsplats-specific chunk-index probe + array-bounds
 * registration helper.
 *
 * Audit G8 / G1-G2 (viewer-data-cache-workers-wasm): mirrors the
 * existing `points/chunk-index-loader.test.ts` parity. Lines has its
 * own dual-index test file separately. GSplats was the asymmetry.
 *
 * The shared `fetchChunkBoundsArray` is mocked so tests can drive the
 * gsplats validation paths (no-ordering short-circuit, chunk-count
 * reconciliation warning) in isolation.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../../../../data/loaders/chunk-bounds-loader', () => ({
  fetchChunkBoundsArray: vi.fn(),
}));

import { fetchChunkBoundsArray } from '../../../../data/loaders';
import {
  loadGSplatsChunkIndex,
  registerGSplatsArrayBounds,
} from '../../../../data/gsplats/chunk-index-loader';
import { log } from '../../../../utils/log';
import type { ChunkPrefetcher } from '../../../../cache/chunk-prefetcher';
import type { GSplatsMetadata } from '../../../../types/gsplats';

const mockFetchChunkBounds = vi.mocked(fetchChunkBoundsArray);

function makeLocation(): import('zarrita').Location<import('zarrita').Readable> {
  return {
    resolve: vi.fn(),
  } as unknown as import('zarrita').Location<import('zarrita').Readable>;
}

function makeAttrs(overrides: Partial<GSplatsMetadata> = {}): GSplatsMetadata {
  return {
    ordering: 'morton',
    n_splats: 100,
    chunk_size: 10,
    ndim: 3,
    ...overrides,
  } as GSplatsMetadata;
}

describe('loadGSplatsChunkIndex', () => {
  let infoSpy: ReturnType<typeof vi.spyOn>;
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.restoreAllMocks();
    mockFetchChunkBounds.mockReset();
    infoSpy = vi.spyOn(log, 'info').mockImplementation(() => {});
    warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
  });

  it('returns null without probing when ordering is "none"', async () => {
    const result = await loadGSplatsChunkIndex(makeLocation(), makeAttrs({ ordering: 'none' }));
    expect(result).toBeNull();
    expect(mockFetchChunkBounds).not.toHaveBeenCalled();
    expect(infoSpy).toHaveBeenCalled();
  });

  it('returns null when fetchChunkBoundsArray returns null (array missing)', async () => {
    mockFetchChunkBounds.mockResolvedValueOnce(null);
    const result = await loadGSplatsChunkIndex(makeLocation(), makeAttrs());
    expect(result).toBeNull();
  });

  it('returns the assembled ChunkSpatialIndex on a clean morton probe', async () => {
    // 10 chunks × 3 dims × 2 (min/max) = 60 floats.
    const data = new Float32Array(60);
    mockFetchChunkBounds.mockResolvedValueOnce({ data, shape: [10, 3, 2] });

    const attrs = makeAttrs({ n_splats: 100, chunk_size: 10, ndim: 3 });
    const result = await loadGSplatsChunkIndex(makeLocation(), attrs);

    expect(result).not.toBeNull();
    expect(result!.chunkBounds).toBe(data);
    expect(result!.chunkCount).toBe(10);
    expect(result!.metadata.ndim).toBe(3);
    expect(result!.metadata.chunk_size).toBe(10);
  });

  it('logs WARNING when metadata implies a different chunk count than the array', async () => {
    // Metadata implies 10 chunks (100 / 10), but array implies 5 chunks
    // (30 floats / (3 × 2) = 5). loader takes the smaller value.
    const data = new Float32Array(30);
    mockFetchChunkBounds.mockResolvedValueOnce({ data, shape: [5, 3, 2] });

    const attrs = makeAttrs({ n_splats: 100, chunk_size: 10, ndim: 3 });
    const result = await loadGSplatsChunkIndex(makeLocation(), attrs);

    expect(result).not.toBeNull();
    expect(result!.chunkCount).toBe(5); // min(10, 5) = 5
    expect(warnSpy).toHaveBeenCalledWith(
      expect.anything(),
      expect.stringContaining('chunk count mismatch')
    );
  });

  it('reconciles to actualChunks when metadata is smaller than array reports', async () => {
    // Metadata: 1 splat / 10 = 1 chunk. Array: 5 chunks.
    const data = new Float32Array(30);
    mockFetchChunkBounds.mockResolvedValueOnce({ data, shape: [5, 3, 2] });

    const attrs = makeAttrs({ n_splats: 1, chunk_size: 10, ndim: 3 });
    const result = await loadGSplatsChunkIndex(makeLocation(), attrs);

    expect(result).not.toBeNull();
    expect(result!.chunkCount).toBe(1); // min(1, 5) = 1
    expect(warnSpy).toHaveBeenCalled();
  });
});

describe('registerGSplatsArrayBounds', () => {
  function makePrefetcher(): {
    prefetcher: ChunkPrefetcher;
    register: ReturnType<typeof vi.fn>;
  } {
    const register = vi.fn();
    const prefetcher = {
      registerArrayBounds: register,
    } as unknown as ChunkPrefetcher;
    return { prefetcher, register };
  }

  function makeArray(
    shape: number[],
    chunks: number[]
  ): import('zarrita').Array<import('zarrita').DataType, import('zarrita').Readable> {
    return { shape, chunks } as unknown as import('zarrita').Array<
      import('zarrita').DataType,
      import('zarrita').Readable
    >;
  }

  it('is a no-op when prefetcher is null', () => {
    expect(() =>
      registerGSplatsArrayBounds(null, '/group/sub', 'centers', makeArray([100, 3], [50, 3]))
    ).not.toThrow();
  });

  it('is a no-op when prefetcher is undefined', () => {
    expect(() =>
      registerGSplatsArrayBounds(undefined, 'group/sub', 'centers', makeArray([100, 3], [50, 3]))
    ).not.toThrow();
  });

  it('strips leading slash from node path before joining with arrayName', () => {
    const { prefetcher, register } = makePrefetcher();
    registerGSplatsArrayBounds(prefetcher, '/group/sub', 'centers', makeArray([100, 3], [50, 3]));
    expect(register).toHaveBeenCalledWith('group/sub/centers', [100, 3], [50, 3]);
  });

  it('keeps a path that has no leading slash unchanged', () => {
    const { prefetcher, register } = makePrefetcher();
    registerGSplatsArrayBounds(
      prefetcher,
      'group/sub',
      'cholesky_factors',
      makeArray([100, 6], [25, 6])
    );
    expect(register).toHaveBeenCalledWith('group/sub/cholesky_factors', [100, 6], [25, 6]);
  });
});
