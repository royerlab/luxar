/**
 * Unit tests for `types/dims.ts::initializeDims`.
 *
 * Two HIGH-5 regressions are pinned here:
 *   1. numPoints=0 used to divide-by-zero and throw a misleading
 *      "0 elements for 0 points" error.
 *   2. metadata shorter than ndim used to be silently truncated by the
 *      `for (i < ndim && i < metadata.length)` loop and then fall through
 *      to the "no display dims" fallback — masking authoring bugs.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { initializeDims, type DimensionMetadata } from '../../../types/dims';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('initializeDims', () => {
  it('handles numPoints=0 without throwing (empty point cloud)', () => {
    // HIGH-5 (a) regression: empty datasets are legitimate. Old code
    // computed `0 / 0 = NaN`, failed Number.isInteger, and threw
    // "Invalid positions array: 0 elements for 0 points".
    expect(() => initializeDims(0, 0)).not.toThrow();

    const dims = initializeDims(0, 0);
    expect(dims.ndim).toBe(3); // default when no metadata
    expect(dims.currentStep).toEqual([0, 0, 0]);
    expect(dims.displayed).toEqual([]); // nothing to display
    expect(dims.metadata).toEqual([]);
  });

  it('handles numPoints=0 with metadata (inferring ndim from metadata.length)', () => {
    const metadata: DimensionMetadata[] = [
      { name: 't', unit: 's', scale: 1 },
      { name: 'z', unit: 'um', scale: 1 },
      { name: 'y', unit: 'um', scale: 1 },
      { name: 'x', unit: 'um', scale: 1 },
    ];
    const dims = initializeDims(0, 0, metadata);
    expect(dims.ndim).toBe(4);
    expect(dims.currentStep).toEqual([0, 0, 0, 0]);
    expect(dims.displayed).toEqual([]);
    expect(dims.metadata).toBe(metadata);
  });

  it('pads short metadata with defaults and warns instead of silently truncating', () => {
    // HIGH-5 (b) regression: ndim=4 but only 2 metadata entries used to
    // silently fall through to the spatial-default fallback. The fix
    // pads metadata to length=ndim and emits a warning.
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const metadata: DimensionMetadata[] = [
      { name: 't', unit: 's', scale: 1, display: false },
      { name: 'c', unit: '', scale: 1, display: false },
    ];
    // 4 points × 4 dims = 16 elements
    const dims = initializeDims(4, 16, metadata);

    expect(dims.ndim).toBe(4);
    expect(dims.metadata).toHaveLength(4);
    expect(dims.metadata?.[0].name).toBe('t');
    expect(dims.metadata?.[1].name).toBe('c');
    // Padded entries get sensible defaults so the slice navigator works.
    expect(dims.metadata?.[2].name).toBe('dim2');
    expect(dims.metadata?.[3].name).toBe('dim3');

    expect(warnSpy).toHaveBeenCalledTimes(1);
    const msg = warnSpy.mock.calls[0]?.[0] as string;
    expect(msg).toContain('Dims');
    expect(msg).toMatch(/2 entries but ndim=4/);
  });

  it('still throws for non-integer ndim (genuinely invalid)', () => {
    // 7 elements / 2 points = 3.5 → not an integer
    expect(() => initializeDims(2, 7)).toThrow(/Invalid positions array/);
  });

  it('respects display flags when metadata is complete', () => {
    const metadata: DimensionMetadata[] = [
      { name: 't', unit: 's', scale: 1, display: false },
      { name: 'z', unit: 'um', scale: 1, display: true },
      { name: 'y', unit: 'um', scale: 1, display: true },
      { name: 'x', unit: 'um', scale: 1, display: true },
    ];
    const dims = initializeDims(2, 8, metadata);
    expect(dims.displayed).toEqual([1, 2, 3]);
  });

  it('defaults to last 3 dims when no metadata is provided', () => {
    const dims = initializeDims(2, 8); // 4D
    expect(dims.ndim).toBe(4);
    expect(dims.displayed).toEqual([1, 2, 3]);
  });
});
