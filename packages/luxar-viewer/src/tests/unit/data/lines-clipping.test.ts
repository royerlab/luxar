import { describe, it, expect } from 'vitest';
import {
  clipSegmentToSlice,
  buildInstanceBuffers,
  buildInstanceBuffersWASM,
  lerp,
  lerpVec3,
  distance3D,
} from '../../../data/lines-spatial-index-loader';
import type { LoadedLinesData, ProcessedLinesData } from '../../../types/lines';

describe('clipSegmentToSlice', () => {
  // Default 3D display setup: display dims [0, 1, 2] (XYZ)
  const displayDims3D = [0, 1, 2];
  const slicePos3D = [0, 0, 0];
  const tolerance3D = [1e10, 1e10, 1e10]; // All dimensions visible in 3D

  // 4D setup: display dims [0, 1, 2], slice on dim 3
  const displayDims4D = [0, 1, 2];
  const slicePos4D = [0, 0, 0, 5]; // At dim3 = 5
  const tolerance4D = [1e10, 1e10, 1e10, 0.5]; // 0.5 tolerance on dim3

  describe('Case A: Both endpoints IN slice', () => {
    it('should return full segment when both endpoints within tolerance', () => {
      const p1 = [0, 0, 0, 5.2]; // Within +-0.5 of slicePos4D[3]=5
      const p2 = [10, 10, 10, 4.8];

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0);
      expect(result.t2).toBe(1);
      expect(result.p1).toEqual([0, 0, 0]); // Projected to 3D
      expect(result.p2).toEqual([10, 10, 10]);
    });

    it('should handle 3D data with all dimensions visible', () => {
      const p1 = [1, 2, 3];
      const p2 = [4, 5, 6];

      const result = clipSegmentToSlice(p1, p2, slicePos3D, tolerance3D, displayDims3D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0);
      expect(result.t2).toBe(1);
      expect(result.p1).toEqual([1, 2, 3]);
      expect(result.p2).toEqual([4, 5, 6]);
    });
  });

  describe('Case B: P1 IN, P2 OUT', () => {
    it('should clip P2 to slice boundary', () => {
      const p1 = [0, 0, 0, 5]; // IN (dim3 = 5, within 5+-0.5)
      const p2 = [10, 10, 10, 10]; // OUT (dim3 = 10, outside 5+-0.5)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0); // P1 unchanged
      expect(result.t2).toBeLessThan(1); // P2 clipped

      // t where dim3 crosses 5.5 (upper boundary)
      // p1[3] + t*(p2[3]-p1[3]) = 5.5
      // 5 + t*(10-5) = 5.5
      // t = 0.1
      expect(result.t2).toBeCloseTo(0.1, 5);

      // Interpolated 3D position at t=0.1
      expect(result.p2[0]).toBeCloseTo(1, 5); // 0 + 0.1*10
      expect(result.p2[1]).toBeCloseTo(1, 5);
      expect(result.p2[2]).toBeCloseTo(1, 5);
    });
  });

  describe('Case C: P1 OUT, P2 IN', () => {
    it('should clip P1 to slice boundary', () => {
      const p1 = [0, 0, 0, 0]; // OUT (dim3 = 0, outside 5+-0.5)
      const p2 = [10, 10, 10, 5]; // IN (dim3 = 5, within 5+-0.5)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBeGreaterThan(0); // P1 clipped
      expect(result.t2).toBe(1); // P2 unchanged

      // t where dim3 crosses 4.5 (lower boundary)
      // 0 + t*(5-0) = 4.5
      // t = 0.9
      expect(result.t1).toBeCloseTo(0.9, 5);

      // Interpolated 3D position at t=0.9
      expect(result.p1[0]).toBeCloseTo(9, 5);
      expect(result.p1[1]).toBeCloseTo(9, 5);
      expect(result.p1[2]).toBeCloseTo(9, 5);
    });
  });

  describe('Case D: Both OUT, opposite sides', () => {
    it('should clip both endpoints when segment crosses slice', () => {
      const p1 = [0, 0, 0, 0]; // OUT below (dim3 = 0)
      const p2 = [10, 10, 10, 10]; // OUT above (dim3 = 10)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBeGreaterThan(0); // P1 clipped
      expect(result.t2).toBeLessThan(1); // P2 clipped

      // t1 where dim3 crosses 4.5
      // 0 + t*(10-0) = 4.5 -> t = 0.45
      expect(result.t1).toBeCloseTo(0.45, 5);

      // t2 where dim3 crosses 5.5
      // 0 + t*(10-0) = 5.5 -> t = 0.55
      expect(result.t2).toBeCloseTo(0.55, 5);
    });
  });

  describe('Case E: Both OUT, same side', () => {
    it('should return invisible when both endpoints below slice', () => {
      const p1 = [0, 0, 0, 0]; // OUT (dim3 = 0)
      const p2 = [10, 10, 10, 2]; // OUT (dim3 = 2, still below 4.5)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(false);
    });

    it('should return invisible when both endpoints above slice', () => {
      const p1 = [0, 0, 0, 8]; // OUT (dim3 = 8)
      const p2 = [10, 10, 10, 10]; // OUT (dim3 = 10, both above 5.5)

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(false);
    });
  });

  describe('Edge cases', () => {
    it('should handle segment exactly on slice boundary', () => {
      const p1 = [0, 0, 0, 4.5]; // ON lower boundary
      const p2 = [10, 10, 10, 5.5]; // ON upper boundary

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBeCloseTo(0, 5);
      expect(result.t2).toBeCloseTo(1, 5);
    });

    it('should handle segment parallel to slice dimension', () => {
      const p1 = [0, 0, 0, 5]; // All dim3 = 5
      const p2 = [10, 10, 10, 5];

      const result = clipSegmentToSlice(p1, p2, slicePos4D, tolerance4D, displayDims4D);

      expect(result.visible).toBe(true);
      expect(result.t1).toBe(0);
      expect(result.t2).toBe(1);
    });

    it('should handle 2D display (padding to 3D)', () => {
      const displayDims2D = [0, 1]; // Only X, Y
      const p1 = [0, 0, 5]; // Z = 5, but we're slicing on it
      const p2 = [10, 10, 5];
      const slicePos = [0, 0, 5];
      const tolerance = [1e10, 1e10, 0.5]; // Slicing on Z

      const result = clipSegmentToSlice(p1, p2, slicePos, tolerance, displayDims2D);

      expect(result.visible).toBe(true);
      expect(result.p1).toEqual([0, 0, 0]); // 2D padded to 3D
      expect(result.p2).toEqual([10, 10, 0]);
    });

    it('should handle multiple slice dimensions', () => {
      // 5D case: display [0,1,2], slice on [3,4]
      const displayDims5D = [0, 1, 2];
      const slicePos5D = [0, 0, 0, 5, 10];
      const tolerance5D = [1e10, 1e10, 1e10, 0.5, 1.0];

      // Segment inside both slice dimensions
      const p1 = [0, 0, 0, 5.2, 10.5]; // Both within tolerance
      const p2 = [10, 10, 10, 4.8, 9.5];

      const result = clipSegmentToSlice(p1, p2, slicePos5D, tolerance5D, displayDims5D);

      expect(result.visible).toBe(true);
      expect(result.p1).toEqual([0, 0, 0]);
      expect(result.p2).toEqual([10, 10, 10]);
    });

    it('should clip in multiple dimensions', () => {
      // Segment that crosses slice in both non-display dimensions
      const displayDims5D = [0, 1, 2];
      const slicePos5D = [0, 0, 0, 5, 10];
      const tolerance5D = [1e10, 1e10, 1e10, 0.5, 1.0];

      const p1 = [0, 0, 0, 4, 8]; // Both outside
      const p2 = [10, 10, 10, 6, 12]; // Both outside

      const result = clipSegmentToSlice(p1, p2, slicePos5D, tolerance5D, displayDims5D);

      // Both cross through valid ranges - should be visible
      expect(result.visible).toBe(true);
      expect(result.t1).toBeGreaterThan(0);
      expect(result.t2).toBeLessThan(1);
    });
  });
});

describe('lerp', () => {
  it('should interpolate between values', () => {
    expect(lerp(0, 10, 0)).toBe(0);
    expect(lerp(0, 10, 1)).toBe(10);
    expect(lerp(0, 10, 0.5)).toBe(5);
    expect(lerp(0, 10, 0.25)).toBe(2.5);
  });

  it('should handle negative values', () => {
    expect(lerp(-10, 10, 0.5)).toBe(0);
    expect(lerp(-10, -5, 0.5)).toBe(-7.5);
  });
});

describe('lerpVec3', () => {
  it('should interpolate between 3D vectors', () => {
    const a = [0, 0, 0];
    const b = [10, 20, 30];

    expect(lerpVec3(a, b, 0)).toEqual([0, 0, 0]);
    expect(lerpVec3(a, b, 1)).toEqual([10, 20, 30]);
    expect(lerpVec3(a, b, 0.5)).toEqual([5, 10, 15]);
  });
});

describe('distance3D', () => {
  it('should calculate Euclidean distance', () => {
    expect(distance3D([0, 0, 0], [1, 0, 0])).toBe(1);
    expect(distance3D([0, 0, 0], [3, 4, 0])).toBe(5); // 3-4-5 triangle
    expect(distance3D([0, 0, 0], [1, 1, 1])).toBeCloseTo(Math.sqrt(3));
  });

  it('should handle negative coordinates', () => {
    expect(distance3D([-1, 0, 0], [1, 0, 0])).toBe(2);
    expect(distance3D([0, -2, 0], [0, 2, 0])).toBe(4);
  });
});

describe('buildInstanceBuffers', () => {
  it('should transform loaded data to GPU-ready format', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([
        0,
        0,
        0, // v0
        10,
        10,
        10, // v1
        20,
        20,
        20, // v2
      ]),
      segments: new Uint32Array([0, 1, 1, 2]), // Two segments
      widths: new Float32Array([0.1, 0.2, 0.3]),
      colors: new Float32Array([
        1,
        0,
        0, // Red
        0,
        1,
        0, // Green
        0,
        0,
        1, // Blue
      ]),
      sharpness: new Float32Array([0.5, 0.8, 1.0]),
      segmentCount: 2,
      vertexCount: 3,
      ndim: 3,
    };

    const result = buildInstanceBuffers(
      loadedData,
      [0, 0, 0], // slice position
      [1e10, 1e10, 1e10], // tolerance (all visible)
      [0, 1, 2] // display dims
    );

    expect(result.segmentCount).toBe(2);

    // First segment: v0 → v1
    expect(Array.from(result.startPositions.slice(0, 3))).toEqual([0, 0, 0]);
    expect(Array.from(result.endPositions.slice(0, 3))).toEqual([10, 10, 10]);
    expect(result.startWidths[0]).toBeCloseTo(0.1, 5);
    expect(result.endWidths[0]).toBeCloseTo(0.2, 5);
    expect(result.startSharpness[0]).toBeCloseTo(0.5, 5);
    expect(result.endSharpness[0]).toBeCloseTo(0.8, 5);
    expect(Array.from(result.startColors.slice(0, 3))).toEqual([1, 0, 0]);
    expect(Array.from(result.endColors.slice(0, 3))).toEqual([0, 1, 0]);

    // Segment length for first segment
    const expectedLength = Math.sqrt(10 * 10 * 3);
    expect(result.segmentLengths[0]).toBeCloseTo(expectedLength);

    // No clipping
    expect(result.startClipped[0]).toBe(0);
    expect(result.endClipped[0]).toBe(0);
  });

  it('should handle clipped segments', () => {
    // 4D segment that crosses the slice
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([
        0,
        0,
        0,
        0, // v0: dim3 = 0
        10,
        10,
        10,
        10, // v1: dim3 = 10
      ]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.3]),
      colors: new Float32Array([1, 0, 0, 0, 0, 1]),
      sharpness: new Float32Array([0.0, 1.0]),
      segmentCount: 1,
      vertexCount: 2,
      ndim: 4,
    };

    const result = buildInstanceBuffers(
      loadedData,
      [0, 0, 0, 5], // Slice at dim3 = 5
      [1e10, 1e10, 1e10, 0.5], // Tolerance 0.5 on dim3
      [0, 1, 2]
    );

    expect(result.segmentCount).toBe(1);

    // Both endpoints should be clipped
    expect(result.startClipped[0]).toBe(1);
    expect(result.endClipped[0]).toBe(1);

    // Widths should be interpolated
    // t1 = 0.45, t2 = 0.55
    // startWidth = lerp(0.1, 0.3, 0.45) = 0.19
    // endWidth = lerp(0.1, 0.3, 0.55) = 0.21
    expect(result.startWidths[0]).toBeCloseTo(0.19, 2);
    expect(result.endWidths[0]).toBeCloseTo(0.21, 2);

    // Sharpness interpolated
    // startSharpness = lerp(0.0, 1.0, 0.45) = 0.45
    // endSharpness = lerp(0.0, 1.0, 0.55) = 0.55
    expect(result.startSharpness[0]).toBeCloseTo(0.45, 2);
    expect(result.endSharpness[0]).toBeCloseTo(0.55, 2);
  });

  it('should filter out invisible segments', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([
        0,
        0,
        0,
        0, // v0: dim3 = 0 (outside)
        10,
        10,
        10,
        2, // v1: dim3 = 2 (outside, same side)
        20,
        20,
        20,
        5, // v2: dim3 = 5 (inside)
      ]),
      segments: new Uint32Array([
        0,
        1, // Invisible (both below)
        1,
        2, // Visible (crosses slice)
      ]),
      widths: new Float32Array([0.1, 0.1, 0.1]),
      colors: null,
      sharpness: null,
      segmentCount: 2,
      vertexCount: 3,
      ndim: 4,
    };

    const result = buildInstanceBuffers(
      loadedData,
      [0, 0, 0, 5],
      [1e10, 1e10, 1e10, 0.5],
      [0, 1, 2]
    );

    // Only one segment should remain
    expect(result.segmentCount).toBe(1);
  });

  it('should use default colors when colors is null', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      colors: null, // No colors
      sharpness: null,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const result = buildInstanceBuffers(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    // Default white color
    expect(Array.from(result.startColors.slice(0, 3))).toEqual([1, 1, 1]);
    expect(Array.from(result.endColors.slice(0, 3))).toEqual([1, 1, 1]);
  });

  it('should use default sharpness when sharpness is null', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      colors: null,
      sharpness: null, // No sharpness
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const result = buildInstanceBuffers(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    // Default sharpness 1.0
    expect(result.startSharpness[0]).toBe(1.0);
    expect(result.endSharpness[0]).toBe(1.0);
  });

  it('should handle empty input', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array(0),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      colors: null,
      sharpness: null,
      segmentCount: 0,
      vertexCount: 0,
      ndim: 3,
    };

    const result = buildInstanceBuffers(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    expect(result.segmentCount).toBe(0);
    expect(result.startPositions.length).toBe(0);
  });
});

// ============================================================================
// buildInstanceBuffersWASM - WASM vs TypeScript Comparison Tests
// ============================================================================

describe('buildInstanceBuffersWASM vs TypeScript', () => {
  /**
   * Helper to compare two ProcessedLinesData results
   */
  function compareResults(wasm: ProcessedLinesData, ts: ProcessedLinesData, tolerance = 1e-5): void {
    expect(wasm.segmentCount).toBe(ts.segmentCount);

    const count = wasm.segmentCount;
    if (count === 0) return;

    // Compare positions
    for (let i = 0; i < count * 3; i++) {
      expect(Math.abs(wasm.startPositions[i] - ts.startPositions[i])).toBeLessThan(tolerance);
      expect(Math.abs(wasm.endPositions[i] - ts.endPositions[i])).toBeLessThan(tolerance);
    }

    // Compare widths
    for (let i = 0; i < count; i++) {
      expect(Math.abs(wasm.startWidths[i] - ts.startWidths[i])).toBeLessThan(tolerance);
      expect(Math.abs(wasm.endWidths[i] - ts.endWidths[i])).toBeLessThan(tolerance);
    }

    // Compare colors
    for (let i = 0; i < count * 3; i++) {
      expect(Math.abs(wasm.startColors[i] - ts.startColors[i])).toBeLessThan(tolerance);
      expect(Math.abs(wasm.endColors[i] - ts.endColors[i])).toBeLessThan(tolerance);
    }

    // Compare sharpness
    for (let i = 0; i < count; i++) {
      expect(Math.abs(wasm.startSharpness[i] - ts.startSharpness[i])).toBeLessThan(tolerance);
      expect(Math.abs(wasm.endSharpness[i] - ts.endSharpness[i])).toBeLessThan(tolerance);
    }

    // Compare segment lengths
    for (let i = 0; i < count; i++) {
      expect(Math.abs(wasm.segmentLengths[i] - ts.segmentLengths[i])).toBeLessThan(tolerance);
    }

    // Compare clipped flags
    for (let i = 0; i < count; i++) {
      expect(wasm.startClipped[i]).toBe(ts.startClipped[i]);
      expect(wasm.endClipped[i]).toBe(ts.endClipped[i]);
    }
  }

  it('should match TypeScript for simple 3D data', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([0, 0, 0, 10, 10, 10, 20, 20, 20]),
      segments: new Uint32Array([0, 1, 1, 2]),
      widths: new Float32Array([0.1, 0.2, 0.3]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      sharpness: new Float32Array([0.5, 0.8, 1.0]),
      segmentCount: 2,
      vertexCount: 3,
      ndim: 3,
    };

    const tsResult = buildInstanceBuffers(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);
    const wasmResult = buildInstanceBuffersWASM(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    compareResults(wasmResult, tsResult);
  });

  it('should match TypeScript for 4D data with clipping', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([
        0, 0, 0, 0, // v0
        10, 10, 10, 10, // v1
        20, 20, 20, 5, // v2 (in slice)
      ]),
      segments: new Uint32Array([0, 1, 1, 2]),
      widths: new Float32Array([0.1, 0.2, 0.3]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      sharpness: new Float32Array([0.5, 0.8, 1.0]),
      segmentCount: 2,
      vertexCount: 3,
      ndim: 4,
    };

    const slicePos = [0, 0, 0, 5];
    const tolerance = [1e10, 1e10, 1e10, 0.5];
    const displayDims = [0, 1, 2];

    const tsResult = buildInstanceBuffers(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = buildInstanceBuffersWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(wasmResult, tsResult);
  });

  it('should match TypeScript for 5D data with multiple slices', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([
        0, 0, 0, 5, 10, // v0
        10, 10, 10, 5, 10, // v1
        20, 20, 20, 6, 11, // v2
      ]),
      segments: new Uint32Array([0, 1, 1, 2]),
      widths: new Float32Array([0.1, 0.2, 0.3]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      sharpness: new Float32Array([0.5, 0.8, 1.0]),
      segmentCount: 2,
      vertexCount: 3,
      ndim: 5,
    };

    const slicePos = [0, 0, 0, 5, 10];
    const tolerance = [1e10, 1e10, 1e10, 0.5, 1.0];
    const displayDims = [0, 1, 2];

    const tsResult = buildInstanceBuffers(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = buildInstanceBuffersWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(wasmResult, tsResult);
  });

  it('should match TypeScript for null color attributes', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.2]),
      colors: null, // No colors provided
      sharpness: new Float32Array([0.5, 0.8]),
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const tsResult = buildInstanceBuffers(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);
    const wasmResult = buildInstanceBuffersWASM(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    compareResults(wasmResult, tsResult);
  });

  it('should match TypeScript for null sharpness attributes', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.2]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      sharpness: null, // No sharpness provided
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const tsResult = buildInstanceBuffers(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);
    const wasmResult = buildInstanceBuffersWASM(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    compareResults(wasmResult, tsResult);
  });

  it('should match TypeScript for all 5 clipping cases', () => {
    // Test data covering all cases: A, B, C, D, E
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([
        0, 0, 0, 5, // v0: IN (Case A endpoint)
        10, 10, 10, 5, // v1: IN (Case A endpoint)
        20, 20, 20, 10, // v2: OUT above (Case B endpoint)
        30, 30, 30, 0, // v3: OUT below (Case C endpoint)
        40, 40, 40, 15, // v4: OUT far above (Case E endpoint)
        50, 50, 50, 20, // v5: OUT far above (Case E endpoint)
      ]),
      segments: new Uint32Array([
        0, 1, // Case A: both in
        1, 2, // Case B: p1 in, p2 out
        3, 1, // Case C: p1 out, p2 in
        3, 2, // Case D: both out, opposite sides
        4, 5, // Case E: both out, same side
      ]),
      widths: new Float32Array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
      colors: new Float32Array([
        1, 0, 0, // Red
        0, 1, 0, // Green
        0, 0, 1, // Blue
        1, 1, 0, // Yellow
        1, 0, 1, // Magenta
        0, 1, 1, // Cyan
      ]),
      sharpness: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
      segmentCount: 5,
      vertexCount: 6,
      ndim: 4,
    };

    const slicePos = [0, 0, 0, 5];
    const tolerance = [1e10, 1e10, 1e10, 0.5];
    const displayDims = [0, 1, 2];

    const tsResult = buildInstanceBuffers(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = buildInstanceBuffersWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(wasmResult, tsResult);
  });

  it('should match TypeScript for stress test with 100+ segments', () => {
    const numVertices = 200;
    const numSegments = 100;

    // Generate test data
    const vertices = new Float32Array(numVertices * 4);
    const segments = new Uint32Array(numSegments * 2);
    const widths = new Float32Array(numVertices);
    const colors = new Float32Array(numVertices * 3);
    const sharpness = new Float32Array(numVertices);

    for (let i = 0; i < numVertices; i++) {
      vertices[i * 4 + 0] = Math.sin(i * 0.1) * 10;
      vertices[i * 4 + 1] = Math.cos(i * 0.1) * 10;
      vertices[i * 4 + 2] = i * 0.5;
      vertices[i * 4 + 3] = Math.sin(i * 0.05) * 5 + 5; // Varies around slice position
      widths[i] = 0.1 + (i % 10) * 0.01;
      colors[i * 3 + 0] = Math.sin(i * 0.2);
      colors[i * 3 + 1] = Math.cos(i * 0.2);
      colors[i * 3 + 2] = Math.sin(i * 0.3);
      sharpness[i] = 0.5 + (i % 5) * 0.1;
    }

    for (let i = 0; i < numSegments; i++) {
      segments[i * 2] = i * 2;
      segments[i * 2 + 1] = i * 2 + 1;
    }

    const loadedData: LoadedLinesData = {
      vertices,
      segments,
      widths,
      colors,
      sharpness,
      segmentCount: numSegments,
      vertexCount: numVertices,
      ndim: 4,
    };

    const slicePos = [0, 0, 0, 5];
    const tolerance = [1e10, 1e10, 1e10, 1.0];
    const displayDims = [0, 1, 2];

    const tsResult = buildInstanceBuffers(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = buildInstanceBuffersWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(wasmResult, tsResult);
  });

  it('should match TypeScript for 6D data', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([
        0, 0, 0, 5, 10, 15, // v0
        10, 10, 10, 5, 10, 15, // v1
      ]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.2]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      sharpness: new Float32Array([0.5, 0.8]),
      segmentCount: 1,
      vertexCount: 2,
      ndim: 6,
    };

    const slicePos = [0, 0, 0, 5, 10, 15];
    const tolerance = [1e10, 1e10, 1e10, 0.5, 1.0, 2.0];
    const displayDims = [0, 1, 2];

    const tsResult = buildInstanceBuffers(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = buildInstanceBuffersWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(wasmResult, tsResult);
  });

  it('should match TypeScript when all segments are filtered', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([
        0, 0, 0, 0, // v0: far from slice
        10, 10, 10, 0, // v1: far from slice
      ]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.2]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      sharpness: new Float32Array([0.5, 0.8]),
      segmentCount: 1,
      vertexCount: 2,
      ndim: 4,
    };

    const slicePos = [0, 0, 0, 10]; // Slice far from vertices
    const tolerance = [1e10, 1e10, 1e10, 0.5];
    const displayDims = [0, 1, 2];

    const tsResult = buildInstanceBuffers(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = buildInstanceBuffersWASM(loadedData, slicePos, tolerance, displayDims);

    expect(wasmResult.segmentCount).toBe(0);
    expect(tsResult.segmentCount).toBe(0);
    compareResults(wasmResult, tsResult);
  });

  it('should match TypeScript for mixed visibility (some visible, some not)', () => {
    const loadedData: LoadedLinesData = {
      vertices: new Float32Array([
        0, 0, 0, 5, // v0: in
        10, 10, 10, 5, // v1: in
        20, 20, 20, 0, // v2: out
        30, 30, 30, 0, // v3: out
        40, 40, 40, 5, // v4: in
      ]),
      segments: new Uint32Array([
        0, 1, // Visible
        2, 3, // Invisible
        1, 4, // Visible
      ]),
      widths: new Float32Array([0.1, 0.1, 0.1, 0.1, 0.1]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]),
      sharpness: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]),
      segmentCount: 3,
      vertexCount: 5,
      ndim: 4,
    };

    const slicePos = [0, 0, 0, 5];
    const tolerance = [1e10, 1e10, 1e10, 0.5];
    const displayDims = [0, 1, 2];

    const tsResult = buildInstanceBuffers(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = buildInstanceBuffersWASM(loadedData, slicePos, tolerance, displayDims);

    expect(wasmResult.segmentCount).toBe(2); // Only 2 visible
    expect(tsResult.segmentCount).toBe(2);
    compareResults(wasmResult, tsResult);
  });
});
