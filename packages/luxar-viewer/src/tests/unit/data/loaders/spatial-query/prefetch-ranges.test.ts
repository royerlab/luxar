/**
 * Unit tests for the shared prefetch cache-warming helper.
 *
 * `prefetchRangesIntoCache` fires one `get()` per (array × range), discards the
 * result, and awaits all — warming L0/L1/L2 without allocating output. `zarrita`
 * is mocked so we can count and inspect the issued reads.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { prefetchRangesIntoCache } from '../../../../../data/loaders';

vi.mock('zarrita', async () => {
  const actual = await vi.importActual('zarrita');
  return {
    ...actual,
    get: vi.fn(),
    slice: (start: number | null, end?: number | null) => ({ start, end: end ?? null }),
  };
});
import { get as zarrGet } from 'zarrita';
const mockGet = vi.mocked(zarrGet);

describe('prefetchRangesIntoCache', () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockGet.mockResolvedValue({ data: new Uint8Array() } as never);
  });

  it('fires one get() per (array × range) and awaits all', async () => {
    const arr2d = { shape: [10, 3] } as never;
    const arr1d = { shape: [10] } as never;
    const ranges = [
      { start: 0, end: 2 },
      { start: 5, end: 6 },
    ];

    await prefetchRangesIntoCache([arr2d, arr1d], ranges);

    // 2 arrays × 2 ranges = 4 reads.
    expect(mockGet).toHaveBeenCalledTimes(4);
  });

  it('builds a first-axis slice with full slices over trailing axes', async () => {
    const arr2d = { shape: [10, 3] } as never;
    await prefetchRangesIntoCache([arr2d], [{ start: 2, end: 5 }]);

    // firstAxisRangeSlice([10,3], {2,5}) → [slice(2,5), slice(null)].
    expect(mockGet.mock.calls[0][1]).toEqual([
      { start: 2, end: 5 },
      { start: null, end: null },
    ]);
  });

  it('forwards a caller-owned abort signal to every speculative read', async () => {
    const controller = new AbortController();
    await prefetchRangesIntoCache(
      [{ shape: [10] } as never],
      [{ start: 2, end: 5 }],
      controller.signal
    );

    expect(mockGet.mock.calls[0][2]).toEqual({ signal: controller.signal });
  });

  it('issues no reads when there are no arrays or no ranges', async () => {
    await prefetchRangesIntoCache([], [{ start: 0, end: 1 }]);
    expect(mockGet).not.toHaveBeenCalled();

    await prefetchRangesIntoCache([{ shape: [10] } as never], []);
    expect(mockGet).not.toHaveBeenCalled();
  });
});
