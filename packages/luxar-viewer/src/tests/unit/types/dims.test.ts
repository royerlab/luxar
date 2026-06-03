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
import { initializeDims, getDimensionRanges, type DimensionMetadata } from '../../../types/dims';

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

  it('pads short metadata with named defaults (behavior, no logging coupling)', () => {
    // types.md C1[P1][P2] fix: split this test in two. The original
    // bundled the **behaviour** (padded metadata entries) with a
    // **logging side-effect** (`console.warn` spy). The behaviour
    // contract is independent of how the warning is delivered — a
    // future refactor that routes through `log.warning` or a structured
    // logger should not break the behaviour test.
    //
    // HIGH-5 (b) regression: ndim=4 but only 2 metadata entries used to
    // silently fall through to the spatial-default fallback. The fix
    // pads metadata to length=ndim with sensible defaults.
    const metadata: DimensionMetadata[] = [
      { name: 't', unit: 's', scale: 1, display: false },
      { name: 'c', unit: '', scale: 1, display: false },
    ];
    // Suppress the warning so the test output stays clean — but DON'T
    // assert on it; that's the next test's job.
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      // 4 points × 4 dims = 16 elements
      const dims = initializeDims(4, 16, metadata);

      expect(dims.ndim).toBe(4);
      expect(dims.metadata).toHaveLength(4);
      expect(dims.metadata?.[0].name).toBe('t');
      expect(dims.metadata?.[1].name).toBe('c');
      // Padded entries get sensible defaults so the slice navigator works.
      expect(dims.metadata?.[2].name).toBe('dim2');
      expect(dims.metadata?.[3].name).toBe('dim3');
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('emits a warning when short metadata is padded (logging side-effect — IMPLEMENTATION DETAIL)', () => {
    // types.md C1[P1][P2] note: this test couples to "log.warning goes
    // through console.warn". A future refactor to a structured logger
    // would break this test even though the production behaviour is
    // still correct. The behaviour itself is verified above; this test
    // is a documented side-effect pin so a missing warning is caught,
    // not a behaviour contract.
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const metadata: DimensionMetadata[] = [
        { name: 't', unit: 's', scale: 1, display: false },
        { name: 'c', unit: '', scale: 1, display: false },
      ];
      initializeDims(4, 16, metadata);

      expect(warnSpy).toHaveBeenCalledTimes(1);
      const msg = warnSpy.mock.calls[0]?.[0] as string;
      expect(msg).toContain('Dims');
      expect(msg).toMatch(/2 entries but ndim=4/);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('still throws for non-integer ndim (genuinely invalid)', () => {
    // 7 elements / 2 points = 3.5 → not an integer
    expect(() => initializeDims(2, 7)).toThrow(/Invalid positions array/);
  });

  // [types.md OOS-5] Defensive numPoints guard. The parameter is typed
  // `number` but TypeScript has no non-negative-integer refinement, so
  // a buggy caller could pass a negative or fractional value.
  // Pre-guard, `-6 / -3 = 2` would pass Number.isInteger and the
  // downstream `new Array(ndim)` would throw RangeError far from the
  // actual bug. Catch at the boundary with a clear message.
  it('throws for negative numPoints (defensive)', () => {
    expect(() => initializeDims(-3, 9)).toThrow(/Invalid numPoints: -3/);
  });

  it('throws for negative numPoints even when totalElements is also negative (no silent positive ndim)', () => {
    // Pre-guard, -6/-3 = 2 would pass Number.isInteger and proceed.
    expect(() => initializeDims(-3, -6)).toThrow(/Invalid numPoints: -3/);
  });

  it('throws for non-integer numPoints', () => {
    expect(() => initializeDims(2.5, 10)).toThrow(/Invalid numPoints: 2\.5/);
  });

  it('throws for NaN numPoints', () => {
    expect(() => initializeDims(Number.NaN, 9)).toThrow(/Invalid numPoints: NaN/);
  });

  it('throws for Infinity numPoints', () => {
    expect(() => initializeDims(Number.POSITIVE_INFINITY, 9)).toThrow(
      /Invalid numPoints: Infinity/
    );
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
    // types.md C2[P2] strengthening: prior assertions only covered ndim
    // and displayed. Pin the missing observable contracts so a regression
    // initialising metadata=[] (instead of preserving undefined) or
    // sizing currentStep wrongly would surface.
    const dims = initializeDims(2, 8); // 4D
    expect(dims.ndim).toBe(4);
    expect(dims.displayed).toEqual([1, 2, 3]);
    // currentStep has one slot per dim, all zero on init.
    expect(dims.currentStep).toHaveLength(4);
    expect(Array.from(dims.currentStep)).toEqual([0, 0, 0, 0]);
    // metadata is preserved as-passed; no metadata arg → undefined,
    // NOT an empty array. A mutation that returned `metadata: []` would
    // pass the previous test but fail this one.
    expect(dims.metadata).toBeUndefined();
  });

  // OOS-1 (round-2 audit): the inverse of HIGH-5 (b) — metadata LONGER than
  // ndim. Previously the `i < effectiveMetadata.length` clamp in the
  // display-determination loop silently truncated extra entries. Now we
  // emit a warning so authoring bugs surface.
  it('warns when metadata is longer than ndim (extra entries ignored)', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    // ndim=2 (4 elements / 2 points) but 4 metadata entries
    const metadata: DimensionMetadata[] = [
      { name: 'x', unit: 'um', scale: 1, display: true },
      { name: 'y', unit: 'um', scale: 1, display: true },
      { name: 'extra1', unit: '', scale: 1 },
      { name: 'extra2', unit: '', scale: 1 },
    ];
    const dims = initializeDims(2, 4, metadata);

    expect(dims.ndim).toBe(2);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const msg = warnSpy.mock.calls[0]?.[0] as string;
    expect(msg).toContain('Dims');
    expect(msg).toMatch(/4 entries but ndim=2/);
    expect(msg).toMatch(/extra entries past index 1 will be ignored/);
  });
});

describe('getDimensionRanges', () => {
  it('returns [min, max] per dim for a non-empty point cloud', () => {
    // 3 points × 2 dims: [(1, 10), (2, 20), (3, 30)]
    const positions = new Float32Array([1, 10, 2, 20, 3, 30]);
    const ranges = getDimensionRanges(positions, 2, 3);
    expect(ranges).toEqual([
      [1, 3],
      [10, 30],
    ]);
  });

  // OOS-2 (round-2 audit): on an empty point cloud, the old code returned
  // [Infinity, -Infinity] per dim (the init values). Downstream consumers
  // (camera bounds, slider ranges) were not robust to that. The fix
  // returns [0, 0] placeholders.
  it('returns [0, 0] placeholders for an empty point cloud (no Infinity leak)', () => {
    const ranges = getDimensionRanges(new Float32Array(0), 3, 0);
    expect(ranges).toEqual([
      [0, 0],
      [0, 0],
      [0, 0],
    ]);
    // All values are finite — a downstream `bounds.expandByPoint(...)` will
    // not poison itself with Infinity.
    for (const [lo, hi] of ranges) {
      expect(Number.isFinite(lo)).toBe(true);
      expect(Number.isFinite(hi)).toBe(true);
    }
  });

  it('returns an empty array for ndim=0 on an empty point cloud', () => {
    const ranges = getDimensionRanges(new Float32Array(0), 0, 0);
    expect(ranges).toEqual([]);
  });
});
