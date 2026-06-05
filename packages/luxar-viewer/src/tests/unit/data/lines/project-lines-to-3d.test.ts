// [data.md/O2][P10] Split from `lines-clipping.test.ts`: projectLinesTo3D
// behavioral coverage + createEmptyLinesData constructor. WASM-parity
// suite lives in `project-lines-wasm-parity.test.ts`; clipSegmentToSlice
// + small math helpers live in `clip-segment-to-slice.test.ts`.
import { describe, it, expect } from 'vitest';
import { projectLinesTo3D, createEmptyLinesData } from '../../../../data/lines/projection';
import type { LinesMetadata, LoadedLinesData } from '../../../../types/lines';

describe('projectLinesTo3D', () => {
  it('should transform loaded data to GPU-ready format', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([
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
      scalars: undefined,
      segmentCount: 2,
      vertexCount: 3,
      ndim: 3,
    };

    const result = projectLinesTo3D(
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
    expect(result.segmentLengths[0]).toBeCloseTo(expectedLength, 5);

    // No clipping
    expect(result.startClipped[0]).toBe(0);
    expect(result.endClipped[0]).toBe(0);
  });

  it('should handle clipped segments', () => {
    // 4D segment that crosses the slice
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([
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
      scalars: undefined,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 4,
    };

    const result = projectLinesTo3D(
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
      positions: new Float32Array([
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
      scalars: undefined,
      segmentCount: 2,
      vertexCount: 3,
      ndim: 4,
    };

    const result = projectLinesTo3D(loadedData, [0, 0, 0, 5], [1e10, 1e10, 1e10, 0.5], [0, 1, 2]);

    // Only one segment should remain
    expect(result.segmentCount).toBe(1);
  });

  it('should use default colors when colors is null', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      colors: null, // No colors
      sharpness: null,
      scalars: undefined,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const result = projectLinesTo3D(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    // Default white color
    expect(Array.from(result.startColors.slice(0, 3))).toEqual([1, 1, 1]);
    expect(Array.from(result.endColors.slice(0, 3))).toEqual([1, 1, 1]);
  });

  it('should use default sharpness when sharpness is null', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      colors: null,
      sharpness: null, // No sharpness
      scalars: undefined,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const result = projectLinesTo3D(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    // Default sharpness 1.0
    expect(result.startSharpness[0]).toBe(1.0);
    expect(result.endSharpness[0]).toBe(1.0);
  });

  it('should handle empty input', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array(0),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      colors: null,
      sharpness: null,
      scalars: undefined,
      segmentCount: 0,
      vertexCount: 0,
      ndim: 3,
    };

    const result = projectLinesTo3D(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    expect(result.segmentCount).toBe(0);
    expect(result.startPositions.length).toBe(0);
  });

  it('normalizes Uint8 colors to [0, 1] before lerp', () => {
    // Without normalization, Uint8 values [0, 255] flow straight into
    // the lerp and end up in startColors/endColors as floats in
    // [0, 255], which the shader interprets as vastly oversaturated
    // colors.
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.2]),
      colors: new Uint8Array([255, 0, 0, 0, 255, 0]),
      sharpness: null,
      scalars: undefined,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const result = projectLinesTo3D(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    expect(result.segmentCount).toBe(1);
    expect(Array.from(result.startColors.slice(0, 3))).toEqual([1, 0, 0]);
    expect(Array.from(result.endColors.slice(0, 3))).toEqual([0, 1, 0]);
  });

  it('normalizes Uint16 colors to [0, 1] before lerp', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.2]),
      colors: new Uint16Array([65535, 0, 0, 0, 65535, 0]),
      sharpness: null,
      scalars: undefined,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const result = projectLinesTo3D(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    expect(result.segmentCount).toBe(1);
    expect(result.startColors[0]).toBeCloseTo(1, 5);
    expect(result.startColors[1]).toBeCloseTo(0, 5);
    expect(result.endColors[1]).toBeCloseTo(1, 5);
  });

  it('Float32 colors pass through unchanged (no double-scaling)', () => {
    // Use values exactly representable in Float32 to allow strict equality.
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.2]),
      colors: new Float32Array([0.5, 0.25, 0.75, 0.125, 0.875, 0.5]),
      sharpness: null,
      scalars: undefined,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const result = projectLinesTo3D(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    expect(result.segmentCount).toBe(1);
    expect(Array.from(result.startColors.slice(0, 3))).toEqual([0.5, 0.25, 0.75]);
    expect(Array.from(result.endColors.slice(0, 3))).toEqual([0.125, 0.875, 0.5]);
  });
});

// ============================================================================
// [P5] BOUNDARY: orphaned scalars. `projectLinesTo3D` recomputes
// `vertexCount = floor(positions.length / ndim)`. When scalars are present
// but `scalars.length !== vertexCount` the scalar branch is suppressed
// (fail-closed) — the output omits startScalars/endScalars entirely. With an
// empty positions buffer vertexCount === 0, so any non-empty scalar array is
// an orphan and gets dropped; segmentCount is 0.
// ============================================================================

describe('projectLinesTo3D — orphaned scalars (vertexCount 0)', () => {
  it('suppresses scalars and emits segmentCount 0 when scalars exist but there are no vertices', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array(0), // vertexCount → 0
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      colors: null,
      sharpness: null,
      // 3 orphaned scalars with zero vertices — length mismatch (3 !== 0).
      scalars: new Float32Array([0.1, 0.5, 0.9]),
      segmentCount: 0,
      vertexCount: 0,
      ndim: 3,
    };

    const result = projectLinesTo3D(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    // Scalar projection suppressed: the optional fields are omitted (undefined).
    expect(result.startScalars).toBeUndefined();
    expect(result.endScalars).toBeUndefined();
    expect(result.segmentCount).toBe(0);
  });

  it('suppresses scalars when count mismatches a non-empty vertex set (1 scalar for 2 vertices)', () => {
    // Sibling case with real vertices to confirm the suppression is driven by
    // the length check, not just the empty buffer. 2 vertices, 1 scalar.
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 10, 10, 10]), // 2 vertices
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.2]),
      colors: null,
      sharpness: null,
      scalars: new Float32Array([0.5]), // 1 scalar for 2 vertices → mismatch
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const result = projectLinesTo3D(loadedData, [0, 0, 0], [1e10, 1e10, 1e10], [0, 1, 2]);

    expect(result.startScalars).toBeUndefined();
    expect(result.endScalars).toBeUndefined();
    // The segment itself still renders (only the scalar branch is dropped).
    expect(result.segmentCount).toBe(1);
  });
});

describe('createEmptyLinesData', () => {
  function makeAttrs(overrides: Partial<LinesMetadata> = {}): LinesMetadata {
    return {
      type: 'lines',
      ndim: 3,
      n_vertices: 0,
      n_segments: 0,
      ...overrides,
    } as LinesMetadata;
  }

  it('returns zero-length typed arrays with the canonical "no visible lines" shape', () => {
    const data = createEmptyLinesData(makeAttrs());

    expect(data.positions).toBeInstanceOf(Float32Array);
    expect(data.positions.length).toBe(0);
    expect(data.segments).toBeInstanceOf(Uint32Array);
    expect(data.segments.length).toBe(0);
    expect(data.widths).toBeInstanceOf(Float32Array);
    expect(data.widths.length).toBe(0);
    expect(data.segmentCount).toBe(0);
    expect(data.vertexCount).toBe(0);
  });

  it('omits optional colors and sharpness arrays', () => {
    const data = createEmptyLinesData(makeAttrs());

    expect(data.colors).toBeNull();
    expect(data.sharpness).toBeNull();
  });

  it('forwards ndim from the metadata', () => {
    const data3 = createEmptyLinesData(makeAttrs({ ndim: 3 }));
    expect(data3.ndim).toBe(3);

    const data5 = createEmptyLinesData(makeAttrs({ ndim: 5 }));
    expect(data5.ndim).toBe(5);

    const data10 = createEmptyLinesData(makeAttrs({ ndim: 10 }));
    expect(data10.ndim).toBe(10);
  });

  it('returns fresh arrays on each call (no shared buffer state)', () => {
    const a = createEmptyLinesData(makeAttrs());
    const b = createEmptyLinesData(makeAttrs());
    expect(a.positions).not.toBe(b.positions);
    expect(a.segments).not.toBe(b.segments);
    expect(a.widths).not.toBe(b.widths);
  });
});
