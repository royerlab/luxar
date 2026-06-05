/**
 * Unit tests for fetchChunkBoundsArray — the shared zarr probe used by
 * Points / Lines / GSplats spatial-index loaders.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('zarrita', async () => {
  const actual = await vi.importActual<typeof import('zarrita')>('zarrita');
  return {
    ...actual,
    open: vi.fn(),
    get: vi.fn(),
  };
});

import { open as zarrOpen, get as zarrGet } from 'zarrita';
import { fetchChunkBoundsArray } from '../../../../data/loaders';
import { log, Modules } from '../../../../utils/log';

const mockOpen = vi.mocked(zarrOpen);
const mockGet = vi.mocked(zarrGet);

function makeLocation(): import('zarrita').Location<import('zarrita').Readable> {
  return {
    resolve: vi.fn((name: string) => ({ name })),
  } as unknown as import('zarrita').Location<import('zarrita').Readable>;
}

describe('fetchChunkBoundsArray', () => {
  let infoSpy: ReturnType<typeof vi.spyOn>;
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.restoreAllMocks();
    mockOpen.mockReset();
    mockGet.mockReset();
    infoSpy = vi.spyOn(log, 'info').mockImplementation(() => {});
    warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
  });

  it('returns the decoded data + shape on success', async () => {
    const buffer = new Float32Array([1, 2, 3, 4, 5, 6]);
    mockOpen.mockResolvedValueOnce({ shape: [1, 3, 2] } as never);
    mockGet.mockResolvedValueOnce({ data: buffer } as never);

    const result = await fetchChunkBoundsArray(
      makeLocation(),
      'chunk_bounds',
      Modules.SPATIAL_INDEX_LOADER,
      'No bounds available'
    );

    expect(result).not.toBeNull();
    expect(Array.from(result!.data)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(result!.shape).toEqual([1, 3, 2]);
    // Combined shape: catches a null/missing field that the separate
    // (non-null-asserted) checks above would skip over.
    expect(result).toEqual({
      data: new Float32Array([1, 2, 3, 4, 5, 6]),
      shape: [1, 3, 2],
    });
  });

  it('returns null + INFO log on a 404 error', async () => {
    mockOpen.mockRejectedValueOnce(new Error('HTTP 404 not available'));

    const result = await fetchChunkBoundsArray(
      makeLocation(),
      'chunk_bounds',
      Modules.SPATIAL_INDEX_LOADER,
      'No bounds available'
    );

    expect(result).toBeNull();
    expect(infoSpy).toHaveBeenCalledWith(Modules.SPATIAL_INDEX_LOADER, 'No bounds available');
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('returns null + INFO log on a "Not Found" error message', async () => {
    mockOpen.mockRejectedValueOnce(new Error('Not Found: chunk_bounds'));

    const result = await fetchChunkBoundsArray(
      makeLocation(),
      'chunk_bounds',
      Modules.SPATIAL_INDEX_LOADER,
      'No bounds'
    );

    expect(result).toBeNull();
    expect(infoSpy).toHaveBeenCalled();
  });

  it('returns null + INFO log on a "Node not found" zarrita-style error', async () => {
    mockOpen.mockRejectedValueOnce(new Error('Node not found'));

    const result = await fetchChunkBoundsArray(
      makeLocation(),
      'segment_chunk_bounds',
      Modules.SPATIAL_INDEX_LOADER,
      'No segments'
    );

    expect(result).toBeNull();
    expect(infoSpy).toHaveBeenCalled();
  });

  it('returns null + WARN log on an unexpected error', async () => {
    mockOpen.mockRejectedValueOnce(new Error('connection reset'));

    const result = await fetchChunkBoundsArray(
      makeLocation(),
      'vertex_chunk_bounds',
      Modules.SPATIAL_INDEX_LOADER,
      'No vertex bounds'
    );

    expect(result).toBeNull();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const args = warnSpy.mock.calls[0];
    expect(args[1]).toContain('vertex_chunk_bounds');
    expect(args[1]).toContain('connection reset');
    // Info log should NOT have fired for the unexpected error path
    expect(infoSpy).not.toHaveBeenCalled();
  });

  it('coerces a non-Error rejection (string/object) into a useful warning', async () => {
    mockOpen.mockRejectedValueOnce('plain-string-error' as unknown as Error);

    const result = await fetchChunkBoundsArray(
      makeLocation(),
      'chunk_bounds',
      Modules.SPATIAL_INDEX_LOADER,
      'No bounds'
    );

    expect(result).toBeNull();
    expect(warnSpy).toHaveBeenCalled();
    expect(warnSpy.mock.calls[0][1]).toContain('plain-string-error');
  });
});
