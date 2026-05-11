/**
 * Unit tests for the points-specific chunk-index probe + array-bounds
 * registration helper.
 *
 * The shared `fetchChunkBoundsArray` is mocked so tests can drive the
 * Points-only validation paths (no-ordering short-circuit, last-dim-not-2,
 * length mismatch warning, dim-coverage warning) and the
 * `morton`/`hilbert` ordering branch in isolation.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../../../../data/loaders/chunk-bounds-loader', () => ({
  fetchChunkBoundsArray: vi.fn(),
}));

import { fetchChunkBoundsArray } from '../../../../data/loaders/chunk-bounds-loader';
import {
  loadPointsChunkIndex,
  registerPointsArrayBounds,
  type PointsNodeAttrsForIndex,
} from '../../../../data/points/chunk-index-loader';
import { log } from '../../../../utils/log';
import type { ChunkPrefetcher } from '../../../../cache';

const mockFetchChunkBounds = vi.mocked(fetchChunkBoundsArray);

function makeLocation(): import('zarrita').Location<import('zarrita').Readable> {
  return {
    resolve: vi.fn(),
  } as unknown as import('zarrita').Location<import('zarrita').Readable>;
}

describe('loadPointsChunkIndex', () => {
  let infoSpy: ReturnType<typeof vi.spyOn>;
  let warnSpy: ReturnType<typeof vi.spyOn>;
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.restoreAllMocks();
    mockFetchChunkBounds.mockReset();
    infoSpy = vi.spyOn(log, 'info').mockImplementation(() => {});
    warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    errorSpy = vi.spyOn(log, 'error').mockImplementation(() => {});
  });

  it('returns null without probing when ordering is missing', async () => {
    const result = await loadPointsChunkIndex(makeLocation(), {});
    expect(result).toBeNull();
    expect(mockFetchChunkBounds).not.toHaveBeenCalled();
    expect(infoSpy).toHaveBeenCalled();
  });

  it('returns null without probing when ordering is "none"', async () => {
    const result = await loadPointsChunkIndex(makeLocation(), { ordering: 'none' });
    expect(result).toBeNull();
    expect(mockFetchChunkBounds).not.toHaveBeenCalled();
  });

  it('returns null when fetchChunkBoundsArray returns null (array missing)', async () => {
    mockFetchChunkBounds.mockResolvedValueOnce(null);
    const result = await loadPointsChunkIndex(makeLocation(), { ordering: 'morton' });
    expect(result).toBeNull();
  });

  it('returns null + ERROR when last dimension of shape is not 2', async () => {
    mockFetchChunkBounds.mockResolvedValueOnce({
      data: new Float32Array(8),
      shape: [2, 2, 3], // last dim is 3, not 2
    });
    const result = await loadPointsChunkIndex(makeLocation(), { ordering: 'morton' });
    expect(result).toBeNull();
    expect(errorSpy).toHaveBeenCalledWith(
      expect.anything(),
      expect.stringContaining('Invalid chunk_bounds shape')
    );
  });

  it('logs WARNING but still succeeds when data length does not match shape', async () => {
    // shape = [1, 3, 2] → expected length 6, but supply 4
    mockFetchChunkBounds.mockResolvedValueOnce({
      data: new Float32Array(4),
      shape: [1, 3, 2],
    });
    const result = await loadPointsChunkIndex(makeLocation(), { ordering: 'morton' });
    expect(result).not.toBeNull();
    expect(warnSpy).toHaveBeenCalledWith(
      expect.anything(),
      expect.stringContaining('length mismatch')
    );
  });

  it('logs WARNING when nodeAttrs.ndim disagrees with chunk_bounds ndim', async () => {
    mockFetchChunkBounds.mockResolvedValueOnce({
      data: new Float32Array(6),
      shape: [1, 3, 2], // ndim=3
    });
    const result = await loadPointsChunkIndex(makeLocation(), {
      ordering: 'morton',
      ndim: 5, // mismatch
    });
    expect(result).not.toBeNull();
    expect(warnSpy).toHaveBeenCalledWith(
      expect.anything(),
      expect.stringContaining('Dimensionality mismatch')
    );
  });

  it('logs WARNING on dim-coverage mismatch (ordering ∪ slice ≠ ndim)', async () => {
    mockFetchChunkBounds.mockResolvedValueOnce({
      data: new Float32Array(6),
      shape: [1, 3, 2], // ndim=3
    });
    const result = await loadPointsChunkIndex(makeLocation(), {
      ordering: 'morton',
      ordering_dims: [0],
      slice_dims: [1], // covers 2 of 3 dims
    });
    expect(result).not.toBeNull();
    expect(warnSpy).toHaveBeenCalledWith(
      expect.anything(),
      expect.stringContaining('Dimension coverage mismatch')
    );
  });

  it('returns the assembled PointsChunkIndex on a clean morton probe', async () => {
    const data = new Float32Array([0, 1, 2, 3, 4, 5]);
    mockFetchChunkBounds.mockResolvedValueOnce({ data, shape: [1, 3, 2] });
    const attrs: PointsNodeAttrsForIndex = {
      ordering: 'morton',
      ordering_dims: [0, 1, 2],
      slice_dims: [],
      ordering_bits_per_dim: 16,
      chunk_size: 100,
      n_points: 1000,
      ndim: 3,
    };

    const result = await loadPointsChunkIndex(makeLocation(), attrs);
    expect(result).toEqual({
      chunkBounds: data,
      chunkCount: 1,
      metadata: {
        ordering: 'morton',
        ordering_dims: [0, 1, 2],
        slice_dims: [],
        ordering_bits_per_dim: 16,
        chunk_size: 100,
        total_points: 1000,
        total_chunks: 1,
        ndim: 3,
      },
    });
  });

  it('maps ordering=hilbert through to the metadata', async () => {
    mockFetchChunkBounds.mockResolvedValueOnce({
      data: new Float32Array(6),
      shape: [1, 3, 2],
    });
    const result = await loadPointsChunkIndex(makeLocation(), { ordering: 'hilbert' });
    expect(result?.metadata.ordering).toBe('hilbert');
  });

  it('uses sensible defaults when ordering_bits_per_dim, chunk_size, n_points are absent', async () => {
    mockFetchChunkBounds.mockResolvedValueOnce({
      data: new Float32Array(6),
      shape: [1, 3, 2],
    });
    const result = await loadPointsChunkIndex(makeLocation(), { ordering: 'morton' });
    expect(result?.metadata.ordering_bits_per_dim).toBe(21);
    expect(result?.metadata.chunk_size).toBe(0);
    expect(result?.metadata.total_points).toBe(0);
  });
});

describe('registerPointsArrayBounds', () => {
  function makePrefetcher(): {
    prefetcher: ChunkPrefetcher;
    register: ReturnType<typeof vi.fn>;
  } {
    const register = vi.fn();
    const prefetcher = {
      // The underlying prefetcher primitive keeps its bare name —
      // it's the generic method, not the per-type wrapper.
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
      registerPointsArrayBounds(null, '/group/sub', 'positions', makeArray([100, 3], [50, 3]))
    ).not.toThrow();
  });

  it('is a no-op when prefetcher is undefined', () => {
    expect(() =>
      registerPointsArrayBounds(undefined, 'group/sub', 'positions', makeArray([100, 3], [50, 3]))
    ).not.toThrow();
  });

  it('strips leading slash from node path before joining with arrayName', () => {
    const { prefetcher, register } = makePrefetcher();
    registerPointsArrayBounds(
      prefetcher,
      '/group/sub',
      'positions',
      makeArray([100, 3], [50, 3])
    );
    expect(register).toHaveBeenCalledWith('group/sub/positions', [100, 3], [50, 3]);
  });

  it('keeps a path that has no leading slash unchanged', () => {
    const { prefetcher, register } = makePrefetcher();
    registerPointsArrayBounds(prefetcher, 'group/sub', 'colors', makeArray([100, 4], [25, 4]));
    expect(register).toHaveBeenCalledWith('group/sub/colors', [100, 4], [25, 4]);
  });
});
