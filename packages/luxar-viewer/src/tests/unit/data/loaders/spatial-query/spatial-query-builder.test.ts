import { describe, it, expect, test, vi } from 'vitest';
import * as fc from 'fast-check';
import {
  SpatialQueryBuilder,
  buildQueryPosition,
  chunkIndicesToRanges,
  mergeRanges,
  shouldExtendVisibility,
  createLoadAllRange,
  executeSpatialQuery,
  type ChunkSpatialIndex,
  type BaseViewState,
} from '../../../../../data/loaders';
import type { LoadRange } from '../../../../../data/loaders';
import type { DimensionMetadata } from '../../../../../types/dims';
import * as toleranceComputer from '../../../../../data/loaders/spatial-query/tolerance-computer';

// ============================================================================
// buildQueryPosition
// ============================================================================

describe('buildQueryPosition', () => {
  it('builds position array padded to ndim', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [1.0, 2.0, 3.0],
      tolerance: [],
    };

    const position = buildQueryPosition(viewState, 5);

    expect(position).toEqual([1.0, 2.0, 3.0, 0, 0]);
  });

  it('truncates when slicePosition is longer than ndim', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [1.0, 2.0, 3.0, 4.0, 5.0],
      tolerance: [],
    };

    const position = buildQueryPosition(viewState, 3);

    expect(position).toEqual([1.0, 2.0, 3.0]);
  });

  it('handles short slicePosition by padding with zero', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [1.0],
      tolerance: [],
    };

    const position = buildQueryPosition(viewState, 3);

    expect(position).toEqual([1.0, 0, 0]);
  });
});

// ============================================================================
// chunkIndicesToRanges
// ============================================================================

describe('chunkIndicesToRanges', () => {
  it('converts chunk indices to ranges, clipping the last chunk', () => {
    const ranges = chunkIndicesToRanges([0, 2, 3], 1000, 3500);

    expect(ranges).toEqual([
      { start: 0, end: 1000 },
      { start: 2000, end: 3000 },
      { start: 3000, end: 3500 }, // clipped to totalElements
    ]);
  });

  it('handles a single chunk', () => {
    expect(chunkIndicesToRanges([1], 1000, 2000)).toEqual([{ start: 1000, end: 2000 }]);
  });

  it('handles empty input', () => {
    expect(chunkIndicesToRanges([], 1000, 5000)).toEqual([]);
  });
});

// ============================================================================
// mergeRanges
// ============================================================================

describe('mergeRanges', () => {
  it('merges overlapping ranges', () => {
    expect(
      mergeRanges([
        { start: 0, end: 100 },
        { start: 50, end: 150 },
      ])
    ).toEqual([{ start: 0, end: 150 }]);
  });

  it('merges adjacent ranges', () => {
    expect(
      mergeRanges([
        { start: 0, end: 100 },
        { start: 100, end: 200 },
      ])
    ).toEqual([{ start: 0, end: 200 }]);
  });

  it('keeps non-overlapping ranges separate', () => {
    expect(
      mergeRanges([
        { start: 0, end: 100 },
        { start: 200, end: 300 },
      ])
    ).toEqual([
      { start: 0, end: 100 },
      { start: 200, end: 300 },
    ]);
  });

  it('handles unsorted input', () => {
    expect(
      mergeRanges([
        { start: 200, end: 300 },
        { start: 0, end: 100 },
        { start: 50, end: 150 },
      ])
    ).toEqual([
      { start: 0, end: 150 },
      { start: 200, end: 300 },
    ]);
  });

  it('handles empty input', () => {
    expect(mergeRanges([])).toEqual([]);
  });
});

// [data.md/H2][P12] mergeRanges — algebraic invariants.
//   - Idempotence: merge(merge(rs)) === merge(rs).
//   - Permutation invariance: shuffling input doesn't change the result.
//   - Coverage invariance: every position covered by `rs` is covered by
//     `mergeRanges(rs)` and vice versa (union of intervals is preserved).
//   - Non-overlap output: in the result, no two adjacent ranges overlap or touch.
//   - Sorted output: results come out start-ascending.
describe('mergeRanges — algebraic invariants (data.md H2)', () => {
  // Generator: arbitrary [start, end) range with start < end, bounded.
  const rangeArb: fc.Arbitrary<LoadRange> = fc
    .tuple(fc.integer({ min: 0, max: 1000 }), fc.integer({ min: 1, max: 200 }))
    .map(([start, len]) => ({ start, end: start + len }));

  const rangesArb = fc.array(rangeArb, { minLength: 0, maxLength: 50 });

  test('idempotence: merge(merge(rs)) === merge(rs)', () => {
    fc.assert(
      fc.property(rangesArb, (rs) => {
        const once = mergeRanges(rs);
        const twice = mergeRanges(once);
        expect(twice).toEqual(once);
      })
    );
  });

  test('permutation invariance: shuffling the input does not change the result', () => {
    fc.assert(
      fc.property(
        rangesArb,
        fc.func(fc.integer({ min: 0, max: 1000 })) as fc.Arbitrary<() => number>,
        (rs, hash) => {
          const a = mergeRanges(rs);
          // Stable shuffle via random comparator backed by `hash`.
          const shuffled = [...rs].sort(() => hash() - hash());
          const b = mergeRanges(shuffled);
          expect(b).toEqual(a);
        }
      )
    );
  });

  test('output is sorted ascending with no overlapping or touching pairs', () => {
    fc.assert(
      fc.property(rangesArb, (rs) => {
        const merged = mergeRanges(rs);
        for (let i = 1; i < merged.length; i++) {
          // strictly start-ascending
          expect(merged[i].start).toBeGreaterThan(merged[i - 1].start);
          // adjacent ranges in the output do NOT touch or overlap —
          // mergeRanges fuses any two ranges with `next.start <= prev.end`.
          expect(merged[i].start).toBeGreaterThan(merged[i - 1].end);
        }
      })
    );
  });

  test('coverage invariance: union of intervals is preserved', () => {
    // Sample positions covered by input vs output and verify they agree
    // exactly. The position-set is bounded so we can enumerate it cheaply.
    fc.assert(
      fc.property(rangesArb, (rs) => {
        const covers = (ranges: LoadRange[], x: number): boolean =>
          ranges.some((r) => r.start <= x && x < r.end);
        const merged = mergeRanges(rs);
        for (let x = 0; x <= 1200; x++) {
          expect(covers(rs, x)).toBe(covers(merged, x));
        }
      })
    );
  });

  test('empty in, empty out (identity)', () => {
    expect(mergeRanges([])).toEqual([]);
  });
});

// ============================================================================
// shouldExtendVisibility
// ============================================================================

describe('shouldExtendVisibility', () => {
  const dims: DimensionMetadata[] = [
    { name: 'x', unit: 'um', scale: 1 },
    { name: 'y', unit: 'um', scale: 1 },
    { name: 'z', unit: 'um', scale: 1 },
    { name: 't', unit: 's', scale: 1 },
  ];

  it('returns true when an extend_to_all dim is currently hidden', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 5],
      tolerance: [],
      dimensions: dims,
    };
    expect(shouldExtendVisibility(['t'], viewState)).toBe(true);
  });

  it('returns false when the extend_to_all dim is currently displayed', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 3],
      slicePosition: [0, 0, 0, 5],
      tolerance: [],
      dimensions: dims,
    };
    expect(shouldExtendVisibility(['t'], viewState)).toBe(false);
  });

  it('returns false when extendDims is empty or undefined', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 5],
      tolerance: [],
      dimensions: dims,
    };
    expect(shouldExtendVisibility([], viewState)).toBe(false);
    expect(shouldExtendVisibility(undefined, viewState)).toBe(false);
  });

  it('returns false when viewState lacks dimension metadata', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 5],
      tolerance: [],
    };
    expect(shouldExtendVisibility(['t'], viewState)).toBe(false);
  });
});

// ============================================================================
// createLoadAllRange
// ============================================================================

describe('createLoadAllRange', () => {
  it('produces a single range covering all elements', () => {
    expect(createLoadAllRange(10000)).toEqual([{ start: 0, end: 10000 }]);
  });
});

// ============================================================================
// executeSpatialQuery
// ============================================================================

describe('executeSpatialQuery', () => {
  it('returns matching chunk indices for an intersecting query', () => {
    // Two chunks in 3D: [0,50]³ and [50,100]³
    const chunkBounds = new Float32Array([0, 50, 0, 50, 0, 50, 50, 100, 50, 100, 50, 100]);

    const result = executeSpatialQuery({
      chunkBounds,
      queryPosition: [25, 25, 25],
      queryTolerance: [10, 10, 10],
      numChunks: 2,
      ndim: 3,
    });

    expect(result).toEqual([0]);
  });

  it('returns empty when no chunks intersect', () => {
    const chunkBounds = new Float32Array([0, 50, 0, 50, 0, 50]);

    const result = executeSpatialQuery({
      chunkBounds,
      queryPosition: [200, 200, 200],
      queryTolerance: [10, 10, 10],
      numChunks: 1,
      ndim: 3,
    });

    expect(result).toEqual([]);
  });

  it('matches all chunks with infinite tolerance', () => {
    // 3 chunks in 4D, last one far from origin
    const chunkBounds = new Float32Array([
      0, 10, 0, 10, 0, 10, 0, 10, 10, 20, 10, 20, 10, 20, 10, 20, 90, 100, 90, 100, 90, 100, 90,
      100,
    ]);

    const result = executeSpatialQuery({
      chunkBounds,
      queryPosition: [0, 0, 0, 0],
      queryTolerance: [1e10, 1e10, 1e10, 1e10],
      numChunks: 3,
      ndim: 4,
    });

    expect(result).toEqual([0, 1, 2]);
  });

  it('returns synchronously (not a Promise)', () => {
    const chunkBounds = new Float32Array([0, 100, 0, 100, 0, 100]);
    const result = executeSpatialQuery({
      chunkBounds,
      queryPosition: [50, 50, 50],
      queryTolerance: [1e10, 1e10, 1e10],
      numChunks: 1,
      ndim: 3,
    });
    expect(Array.isArray(result)).toBe(true);
    expect(result).not.toBeInstanceOf(Promise);
  });
});

// ============================================================================
// SpatialQueryBuilder
// ============================================================================

describe('SpatialQueryBuilder', () => {
  function makeIndex(numChunks: number, ndim: number, chunkSize = 1000): ChunkSpatialIndex {
    return {
      chunkBounds: new Float32Array(numChunks * ndim * 2),
      chunkCount: numChunks,
      metadata: { ndim, chunk_size: chunkSize },
    };
  }

  // data.md W10 audit acknowledgment [P3]: the two `vi.spyOn(toleranceComputer,
  // 'computeTolerance')` tests below intentionally test implementation
  // coupling (that the builder routes through this helper) rather than
  // pure behavior. The behavior — that the same tolerance is ultimately
  // applied to the chunk-bound test — is covered indirectly by the
  // `chunk overlap` and `pre-computed tolerance path` blocks lower in
  // this file. Keeping the spy tests but noting the trade-off: a
  // refactor that inlined `computeTolerance` into the builder would
  // break these (false positive) without changing behavior. Acceptable
  // because the helper is a public seam for the geometry-typed tolerance
  // model and we want to detect accidental decoupling.
  describe('geometry-aware path', () => {
    it('delegates tolerance to computeTolerance with the given geometryType', async () => {
      const index = makeIndex(1, 3);
      // Chunk 0 covers [0,100]³
      index.chunkBounds.set([0, 100, 0, 100, 0, 100], 0);

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [50, 50, 50],
        tolerance: [],
      };

      const spy = vi.spyOn(toleranceComputer, 'computeTolerance');

      await new SpatialQueryBuilder(index, viewState, {
        geometryType: 'gsplats',
        totalElements: 1000,
      }).execute();

      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy).toHaveBeenCalledWith(
        'gsplats',
        viewState.displayDims,
        index.metadata.ndim,
        viewState.dimensions,
        undefined
      );

      spy.mockRestore();
    });

    it('forwards toleranceOptions to computeTolerance', async () => {
      const index = makeIndex(1, 3);
      index.chunkBounds.set([0, 100, 0, 100, 0, 100], 0);
      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [50, 50, 50],
        tolerance: [],
      };

      const spy = vi.spyOn(toleranceComputer, 'computeTolerance');

      await new SpatialQueryBuilder(index, viewState, {
        geometryType: 'points',
        toleranceOptions: { maxRadius: 5.0, spatialExtendDims: [true, true, true] },
        totalElements: 1000,
      }).execute();

      expect(spy).toHaveBeenCalledWith('points', [0, 1, 2], 3, undefined, {
        maxRadius: 5.0,
        spatialExtendDims: [true, true, true],
      });

      spy.mockRestore();
    });
  });

  describe('pre-computed tolerance path', () => {
    it('uses caller-supplied tolerance verbatim, bypassing computeTolerance', async () => {
      const index = makeIndex(2, 3);
      // Chunk 0: [0,50]³
      index.chunkBounds.set([0, 50, 0, 50, 0, 50], 0);
      // Chunk 1: [60,100]³ (gap from 50–60)
      index.chunkBounds.set([60, 100, 60, 100, 60, 100], 6);

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [25, 25, 25],
        tolerance: [],
      };

      const spy = vi.spyOn(toleranceComputer, 'computeTolerance');

      // Tight tolerance — only chunk 0 should match
      const ranges = await new SpatialQueryBuilder(index, viewState, {
        tolerance: [10, 10, 10],
        totalElements: 2000,
      }).execute();

      expect(spy).not.toHaveBeenCalled();
      expect(ranges).toEqual([{ start: 0, end: 1000 }]);

      spy.mockRestore();
    });
  });

  describe('extend-to-all short-circuit', () => {
    it('returns single load-all range when an extendDim is hidden', async () => {
      const index = makeIndex(2, 4);
      const dims: DimensionMetadata[] = [
        { name: 'x', unit: 'um', scale: 1 },
        { name: 'y', unit: 'um', scale: 1 },
        { name: 'z', unit: 'um', scale: 1 },
        { name: 't', unit: 's', scale: 1 },
      ];

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [],
        dimensions: dims,
      };

      const ranges = await new SpatialQueryBuilder(index, viewState, {
        geometryType: 'lines',
        totalElements: 5000,
        extendDims: ['t'],
      }).execute();

      expect(ranges).toEqual([{ start: 0, end: 5000 }]);
    });

    it('runs the normal query when no extendDim is hidden', async () => {
      const index = makeIndex(1, 3);
      index.chunkBounds.set([0, 100, 0, 100, 0, 100], 0);
      const dims: DimensionMetadata[] = [
        { name: 'x', unit: 'um', scale: 1 },
        { name: 'y', unit: 'um', scale: 1 },
        { name: 'z', unit: 'um', scale: 1 },
      ];

      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [50, 50, 50],
        tolerance: [],
        dimensions: dims,
      };

      const ranges = await new SpatialQueryBuilder(index, viewState, {
        tolerance: [1e10, 1e10, 1e10],
        totalElements: 1000,
        extendDims: ['t'],
      }).execute();

      expect(ranges).toEqual([{ start: 0, end: 1000 }]);
    });
  });

  describe('chunkSize fallback', () => {
    it('uses options.chunkSize when index.metadata.chunk_size is missing', async () => {
      const index: ChunkSpatialIndex = {
        chunkBounds: new Float32Array([0, 100, 0, 100, 0, 100]),
        chunkCount: 1,
        metadata: { ndim: 3 },
      };
      const viewState: BaseViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [50, 50, 50],
        tolerance: [],
      };

      const ranges = await new SpatialQueryBuilder(index, viewState, {
        tolerance: [1e10, 1e10, 1e10],
        totalElements: 500,
        chunkSize: 250,
      }).execute();

      expect(ranges).toEqual([{ start: 0, end: 250 }]);
    });
  });
});
