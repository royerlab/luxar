import { describe, it, expect } from 'vitest';
import {
  queryGSplatsChunksForView,
  chunkIndicesToSplatRanges,
  mergeRanges,
  computeGSplatsTolerance,
  computeToleranceFromViewState,
} from '../../../data/gsplats-chunk-spatial-index';
import type { GSplatsChunkSpatialIndex, GSplatsViewState, SplatRange } from '../../../types/gsplats';

describe('queryGSplatsChunksForView', () => {
  it('should find chunks intersecting query region', () => {
    const index: GSplatsChunkSpatialIndex = {
      metadata: {
        type: 'gsplats',
        n_splats: 2000,
        ndim: 3,
        has_colors: false,
        has_sharpness: false,
        chunk_size: 1000,
        amplitude_range: { min: 0, max: 1 },
        sharpness_bounds: { min: 2, max: 2 },
        center_bounds: { min: [0, 0, 0], max: [100, 100, 100] },
        ordering: 'morton',
      },
      // Two chunks: [0-50] and [50-100] in each dimension
      chunkBounds: new Float32Array([
        // Chunk 0: [0,50] x [0,50] x [0,50]
        0, 50, 0, 50, 0, 50,
        // Chunk 1: [50,100] x [50,100] x [50,100]
        50, 100, 50, 100, 50, 100,
      ]),
      chunkCount: 2,
    };

    // Query at center [25, 25, 25] - should hit chunk 0
    const result1 = queryGSplatsChunksForView(index, [25, 25, 25], [10, 10, 10]);
    expect(result1).toEqual([0]);

    // Query at center [75, 75, 75] - should hit chunk 1
    const result2 = queryGSplatsChunksForView(index, [75, 75, 75], [10, 10, 10]);
    expect(result2).toEqual([1]);

    // Query at center [50, 50, 50] - should hit both chunks
    const result3 = queryGSplatsChunksForView(index, [50, 50, 50], [10, 10, 10]);
    expect(result3).toEqual([0, 1]);
  });

  it('should return empty array when no chunks intersect', () => {
    const index: GSplatsChunkSpatialIndex = {
      metadata: {
        type: 'gsplats',
        n_splats: 1000,
        ndim: 3,
        has_colors: false,
        has_sharpness: false,
        chunk_size: 1000,
        amplitude_range: { min: 0, max: 1 },
        sharpness_bounds: { min: 2, max: 2 },
        center_bounds: { min: [0, 0, 0], max: [50, 50, 50] },
        ordering: 'morton',
      },
      chunkBounds: new Float32Array([0, 50, 0, 50, 0, 50]),
      chunkCount: 1,
    };

    // Query far outside chunk bounds
    const result = queryGSplatsChunksForView(index, [200, 200, 200], [10, 10, 10]);
    expect(result).toEqual([]);
  });

  it('should use infinite tolerance for displayed dimensions', () => {
    const index: GSplatsChunkSpatialIndex = {
      metadata: {
        type: 'gsplats',
        n_splats: 1000,
        ndim: 3,
        has_colors: false,
        has_sharpness: false,
        chunk_size: 1000,
        amplitude_range: { min: 0, max: 1 },
        sharpness_bounds: { min: 2, max: 2 },
        center_bounds: { min: [0, 0, 0], max: [100, 100, 100] },
        ordering: 'morton',
      },
      chunkBounds: new Float32Array([0, 100, 0, 100, 0, 100]),
      chunkCount: 1,
    };

    // With infinite tolerance, any position should hit the chunk
    const result = queryGSplatsChunksForView(index, [1000, 1000, 1000], [1e10, 1e10, 1e10]);
    expect(result).toEqual([0]);
  });
});

describe('chunkIndicesToSplatRanges', () => {
  it('should convert chunk indices to splat ranges', () => {
    const result = chunkIndicesToSplatRanges([0, 2, 3], 1000, 3500);
    expect(result).toEqual([
      { start: 0, end: 1000 },
      { start: 2000, end: 3000 },
      { start: 3000, end: 3500 }, // Last chunk is partial
    ]);
  });

  it('should handle single chunk', () => {
    const result = chunkIndicesToSplatRanges([1], 1000, 2000);
    expect(result).toEqual([{ start: 1000, end: 2000 }]);
  });

  it('should handle empty input', () => {
    const result = chunkIndicesToSplatRanges([], 1000, 5000);
    expect(result).toEqual([]);
  });
});

describe('mergeRanges', () => {
  it('should merge overlapping ranges', () => {
    const ranges: SplatRange[] = [
      { start: 0, end: 100 },
      { start: 50, end: 150 },
    ];
    const result = mergeRanges(ranges);
    expect(result).toEqual([{ start: 0, end: 150 }]);
  });

  it('should merge adjacent ranges', () => {
    const ranges: SplatRange[] = [
      { start: 0, end: 100 },
      { start: 100, end: 200 },
    ];
    const result = mergeRanges(ranges);
    expect(result).toEqual([{ start: 0, end: 200 }]);
  });

  it('should not merge non-overlapping ranges', () => {
    const ranges: SplatRange[] = [
      { start: 0, end: 100 },
      { start: 200, end: 300 },
    ];
    const result = mergeRanges(ranges);
    expect(result).toEqual([
      { start: 0, end: 100 },
      { start: 200, end: 300 },
    ]);
  });

  it('should handle unsorted input', () => {
    const ranges: SplatRange[] = [
      { start: 200, end: 300 },
      { start: 0, end: 100 },
      { start: 50, end: 150 },
    ];
    const result = mergeRanges(ranges);
    expect(result).toEqual([
      { start: 0, end: 150 },
      { start: 200, end: 300 },
    ]);
  });

  it('should handle empty input', () => {
    const result = mergeRanges([]);
    expect(result).toEqual([]);
  });
});

describe('computeGSplatsTolerance', () => {
  it('should set infinite tolerance for displayed dimensions', () => {
    const result = computeGSplatsTolerance(undefined, [0, 1, 2]);
    expect(result[0]).toBe(1e10);
    expect(result[1]).toBe(1e10);
    expect(result[2]).toBe(1e10);
  });

  it('should use step-based tolerance for hidden dimensions', () => {
    const dimensions = [
      { name: 'x', unit: 'um', scale: 1, step: 0.5 },
      { name: 'y', unit: 'um', scale: 1, step: 0.5 },
      { name: 'z', unit: 'um', scale: 1, step: 0.5 },
      { name: 't', unit: 's', scale: 1, step: 1.0 },
    ];
    const displayDims = [0, 1, 2]; // Display x, y, z; hide t

    const result = computeGSplatsTolerance(dimensions, displayDims);

    expect(result[0]).toBe(1e10); // Displayed
    expect(result[1]).toBe(1e10); // Displayed
    expect(result[2]).toBe(1e10); // Displayed
    expect(result[3]).toBe(3.0); // Hidden: 1.0 * 3.0 (default tolerance)
  });

  it('should use custom default tolerance', () => {
    const dimensions = [
      { name: 'x', unit: 'um', scale: 1 },
      { name: 'y', unit: 'um', scale: 1 },
      { name: 'z', unit: 'um', scale: 1 },
      { name: 't', unit: 's', scale: 1 },
    ];
    const result = computeGSplatsTolerance(dimensions, [0, 1, 2], 5.0);
    expect(result[3]).toBe(5.0);
  });
});

describe('computeToleranceFromViewState', () => {
  it('should use provided tolerance if available', () => {
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 2, 3],
    };
    const result = computeToleranceFromViewState(viewState);
    expect(result).toEqual([1, 2, 3]);
  });

  it('should compute tolerance from dimensions if not provided', () => {
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [],
      dimensions: [
        { name: 'x', unit: 'um', scale: 1 },
        { name: 'y', unit: 'um', scale: 1 },
        { name: 'z', unit: 'um', scale: 1 },
      ],
    };
    const result = computeToleranceFromViewState(viewState);
    expect(result[0]).toBe(1e10);
    expect(result[1]).toBe(1e10);
    expect(result[2]).toBe(1e10);
  });
});
