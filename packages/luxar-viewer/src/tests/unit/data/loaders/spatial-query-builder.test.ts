import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  SpatialQueryBuilder,
  computeQueryTolerance,
  buildQueryPosition,
  chunkIndicesToRanges,
  mergeRanges,
  shouldExtendVisibility,
  createLoadAllRange,
  DISPLAYED_DIM_TOLERANCE,
} from '../../../../data/loaders/spatial-query-builder';
import type { BaseViewState } from '../../../../data/loaders/base-types';
import type { DimensionMetadata } from '../../../../types/dims';

// Mock worker pool
vi.mock('../../../../workers/worker-pool', () => ({
  getWorkerPool: () => ({
    getWorker: () =>
      Promise.resolve({
        querySpatialIndex: vi.fn().mockResolvedValue(new Uint32Array([0, 1, 2])),
      }),
  }),
}));

// Mock config
vi.mock('../../../../config', () => ({
  config: {
    dataLoading: {
      performance: {
        useWebWorkers: false, // Start with main thread for easier testing
      },
    },
  },
}));

describe('computeQueryTolerance', () => {
  it('should set infinite tolerance for displayed dimensions', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [],
    };

    const tolerance = computeQueryTolerance(viewState, 3);

    expect(tolerance[0]).toBe(DISPLAYED_DIM_TOLERANCE);
    expect(tolerance[1]).toBe(DISPLAYED_DIM_TOLERANCE);
    expect(tolerance[2]).toBe(DISPLAYED_DIM_TOLERANCE);
  });

  it('should use step-based tolerance for hidden dimensions with step metadata', () => {
    const dimensions: DimensionMetadata[] = [
      { name: 'x', unit: 'um', scale: 1, step: 0.5 },
      { name: 'y', unit: 'um', scale: 1, step: 0.5 },
      { name: 'z', unit: 'um', scale: 1, step: 0.5 },
      { name: 't', unit: 's', scale: 1, step: 1.0 },
    ];

    const viewState: BaseViewState = {
      displayDims: [0, 1, 2], // Display x, y, z; hide t
      slicePosition: [0, 0, 0, 5],
      tolerance: [],
      dimensions,
    };

    const tolerance = computeQueryTolerance(viewState, 4, { stepMultiplier: 1.0 });

    expect(tolerance[0]).toBe(DISPLAYED_DIM_TOLERANCE); // Displayed
    expect(tolerance[1]).toBe(DISPLAYED_DIM_TOLERANCE); // Displayed
    expect(tolerance[2]).toBe(DISPLAYED_DIM_TOLERANCE); // Displayed
    expect(tolerance[3]).toBe(1.0); // Hidden: step * 1.0
  });

  it('should use explicit tolerance from viewState for hidden dimensions', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 5],
      tolerance: [0, 0, 0, 2.5],
    };

    const tolerance = computeQueryTolerance(viewState, 4);

    expect(tolerance[3]).toBe(2.5);
  });

  it('should use maxRadius as fallback for hidden dimensions', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 5],
      tolerance: [],
    };

    const tolerance = computeQueryTolerance(viewState, 4, { maxRadius: 5.0 });

    expect(tolerance[3]).toBe(5.0);
  });

  it('should apply stepMultiplier correctly', () => {
    const dimensions: DimensionMetadata[] = [
      { name: 'x', unit: 'um', scale: 1, step: 0.5 },
      { name: 't', unit: 's', scale: 1, step: 1.0 },
    ];

    const viewState: BaseViewState = {
      displayDims: [0],
      slicePosition: [0, 5],
      tolerance: [],
      dimensions,
    };

    const tolerance = computeQueryTolerance(viewState, 2, { stepMultiplier: 2.0 });

    expect(tolerance[1]).toBe(2.0); // step (1.0) * multiplier (2.0)
  });
});

describe('buildQueryPosition', () => {
  it('should build position array of correct length', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [1.0, 2.0, 3.0],
      tolerance: [],
    };

    const position = buildQueryPosition(viewState, 5);

    expect(position).toHaveLength(5);
    expect(position[0]).toBe(1.0);
    expect(position[1]).toBe(2.0);
    expect(position[2]).toBe(3.0);
    expect(position[3]).toBe(0);
    expect(position[4]).toBe(0);
  });

  it('should handle short slicePosition', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [1.0],
      tolerance: [],
    };

    const position = buildQueryPosition(viewState, 3);

    expect(position).toHaveLength(3);
    expect(position[0]).toBe(1.0);
    expect(position[1]).toBe(0);
    expect(position[2]).toBe(0);
  });
});

describe('chunkIndicesToRanges', () => {
  it('should convert chunk indices to ranges', () => {
    const ranges = chunkIndicesToRanges([0, 2, 3], 1000, 3500);

    expect(ranges).toEqual([
      { start: 0, end: 1000 },
      { start: 2000, end: 3000 },
      { start: 3000, end: 3500 }, // Last chunk is partial
    ]);
  });

  it('should handle single chunk', () => {
    const ranges = chunkIndicesToRanges([1], 1000, 2000);

    expect(ranges).toEqual([{ start: 1000, end: 2000 }]);
  });

  it('should handle empty input', () => {
    const ranges = chunkIndicesToRanges([], 1000, 5000);

    expect(ranges).toEqual([]);
  });
});

describe('mergeRanges', () => {
  it('should merge overlapping ranges', () => {
    const ranges = [
      { start: 0, end: 100 },
      { start: 50, end: 150 },
    ];

    const merged = mergeRanges(ranges);

    expect(merged).toEqual([{ start: 0, end: 150 }]);
  });

  it('should merge adjacent ranges', () => {
    const ranges = [
      { start: 0, end: 100 },
      { start: 100, end: 200 },
    ];

    const merged = mergeRanges(ranges);

    expect(merged).toEqual([{ start: 0, end: 200 }]);
  });

  it('should not merge non-overlapping ranges', () => {
    const ranges = [
      { start: 0, end: 100 },
      { start: 200, end: 300 },
    ];

    const merged = mergeRanges(ranges);

    expect(merged).toEqual([
      { start: 0, end: 100 },
      { start: 200, end: 300 },
    ]);
  });

  it('should handle unsorted input', () => {
    const ranges = [
      { start: 200, end: 300 },
      { start: 0, end: 100 },
      { start: 50, end: 150 },
    ];

    const merged = mergeRanges(ranges);

    expect(merged).toEqual([
      { start: 0, end: 150 },
      { start: 200, end: 300 },
    ]);
  });

  it('should handle empty input', () => {
    expect(mergeRanges([])).toEqual([]);
  });
});

describe('shouldExtendVisibility', () => {
  const dimMetadata: DimensionMetadata[] = [
    { name: 'x', unit: 'um', scale: 1 },
    { name: 'y', unit: 'um', scale: 1 },
    { name: 'z', unit: 'um', scale: 1 },
    { name: 't', unit: 's', scale: 1 },
  ];

  it('should return true when extending across hidden dimension', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2], // x, y, z displayed; t hidden
      slicePosition: [0, 0, 0, 5],
      tolerance: [],
    };

    expect(shouldExtendVisibility(['t'], viewState, dimMetadata)).toBe(true);
  });

  it('should return false when extend_to_all dimension is displayed', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 3], // x, y, t displayed; z hidden
      slicePosition: [0, 0, 0, 5],
      tolerance: [],
    };

    expect(shouldExtendVisibility(['t'], viewState, dimMetadata)).toBe(false);
  });

  it('should return false when extendDims is empty', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [],
    };

    expect(shouldExtendVisibility([], viewState, dimMetadata)).toBe(false);
  });

  it('should return false when extendDims is undefined', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [],
    };

    expect(shouldExtendVisibility(undefined, viewState, dimMetadata)).toBe(false);
  });

  it('should return false when dimMetadata is missing', () => {
    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [],
    };

    expect(shouldExtendVisibility(['t'], viewState, undefined)).toBe(false);
    expect(shouldExtendVisibility(['t'], viewState, [])).toBe(false);
  });
});

describe('createLoadAllRange', () => {
  it('should create single range covering all elements', () => {
    const ranges = createLoadAllRange(10000);

    expect(ranges).toEqual([{ start: 0, end: 10000 }]);
  });
});

describe('SpatialQueryBuilder', () => {
  const createMockIndex = (numChunks: number, ndim: number) => ({
    chunkBounds: new Float32Array(numChunks * ndim * 2),
    chunkCount: numChunks,
    metadata: { ndim, chunk_size: 1000 },
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should build and execute query', async () => {
    // Create a simple 3D index with 2 chunks
    const index = createMockIndex(2, 3);
    // Chunk 0: [0,50] x [0,50] x [0,50]
    index.chunkBounds.set([0, 50, 0, 50, 0, 50], 0);
    // Chunk 1: [50,100] x [50,100] x [50,100]
    index.chunkBounds.set([50, 100, 50, 100, 50, 100], 6);

    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [25, 25, 25], // Should hit chunk 0
      tolerance: [10, 10, 10],
    };

    const builder = new SpatialQueryBuilder(index, viewState, 2000, 1000);
    const ranges = await builder.execute();

    // With infinite tolerance for displayed dims, both chunks should match
    expect(ranges.length).toBeGreaterThan(0);
  });

  it('should return all elements when extending visibility', async () => {
    const index = createMockIndex(2, 4);
    const dimMetadata: DimensionMetadata[] = [
      { name: 'x', unit: 'um', scale: 1 },
      { name: 'y', unit: 'um', scale: 1 },
      { name: 'z', unit: 'um', scale: 1 },
      { name: 't', unit: 's', scale: 1 },
    ];

    const viewState: BaseViewState = {
      displayDims: [0, 1, 2], // t is hidden
      slicePosition: [0, 0, 0, 5],
      tolerance: [],
      dimensions: dimMetadata,
    };

    const builder = new SpatialQueryBuilder(index, viewState, 5000, 1000);
    const ranges = await builder.withExtendToAll(['t']).execute();

    // Should return single range covering all elements
    expect(ranges).toEqual([{ start: 0, end: 5000 }]);
  });

  it('should support fluent configuration', async () => {
    const index = createMockIndex(1, 3);
    index.chunkBounds.set([0, 100, 0, 100, 0, 100], 0);

    const viewState: BaseViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [50, 50, 50],
      tolerance: [],
    };

    const builder = new SpatialQueryBuilder(index, viewState, 1000, 1000);
    const ranges = await builder
      .withMaxRadius(10)
      .withDefaultTolerance(5)
      .withStepMultiplier(2.0)
      .execute();

    expect(ranges.length).toBeGreaterThan(0);
  });
});
