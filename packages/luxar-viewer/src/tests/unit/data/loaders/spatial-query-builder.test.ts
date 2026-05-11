import { describe, it, expect, vi } from 'vitest';
import {
  SpatialQueryBuilder,
  buildQueryPosition,
  chunkIndicesToRanges,
  mergeRanges,
  shouldExtendVisibility,
  createLoadAllRange,
  executeSpatialQuery,
  type ChunkSpatialIndex,
} from '../../../../data/loaders/spatial-query-builder';
import type { BaseViewState } from '../../../../data/loaders/base-types';
import type { DimensionMetadata } from '../../../../types/dims';
import * as toleranceComputer from '../../../../data/utils/tolerance-computer';

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
