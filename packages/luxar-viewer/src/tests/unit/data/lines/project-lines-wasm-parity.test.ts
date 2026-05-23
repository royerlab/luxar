// [data.md/O2][P10] Split from `lines-clipping.test.ts`: WASM vs TypeScript
// parity for projectLinesTo3D. Concrete projectLinesTo3D coverage +
// createEmptyLinesData constructor live in `project-lines-to-3d.test.ts`;
// clipSegmentToSlice + small math helpers live in `clip-segment-to-slice.test.ts`.
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { projectLinesTo3D, projectLinesTo3DWASM } from '../../../../data/lines/projection';
import type { LoadedLinesData } from '../../../../types/lines';

// Check if WASM files exist for comparison tests
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const wasmJsPath = join(__dirname, '../../../../../public/wasm/luxar_wasm.js');
const wasmBinaryPath = join(__dirname, '../../../../../public/wasm/luxar_wasm_bg.wasm');
const wasmFilesExist = existsSync(wasmJsPath) && existsSync(wasmBinaryPath);

// ============================================================================
// WASM vs TypeScript Comparison Tests
// ============================================================================

describe.skipIf(!wasmFilesExist)('projectLinesTo3DWASM vs projectLinesTo3D', () => {
  // Initialize WASM module before tests
  beforeAll(async () => {
    if (!wasmFilesExist) return;

    try {
      // Load WASM binary
      const wasmBinary = readFileSync(wasmBinaryPath);
      // Dynamic import for ES module
      const wasm = await import(wasmJsPath);
      wasm.initSync({ module: wasmBinary });
    } catch (error) {
      console.error('Failed to load WASM module:', error);
    }
  });

  /**
   * Helper to compare ProcessedLinesData from both implementations
   */
  function compareResults(
    tsResult: ReturnType<typeof projectLinesTo3D>,
    wasmResult: ReturnType<typeof projectLinesTo3DWASM>,
    tolerance = 1e-5
  ) {
    // Segment counts must match
    expect(wasmResult.segmentCount).toBe(tsResult.segmentCount);

    // Compare all arrays with tolerance for floating point
    const compareArrays = (
      name: string,
      ts: Float32Array | Uint8Array,
      wasm: Float32Array | Uint8Array
    ) => {
      expect(wasm.length).toBe(ts.length);
      for (let i = 0; i < ts.length; i++) {
        if (Math.abs(ts[i] - wasm[i]) > tolerance) {
          throw new Error(`${name}[${i}] differs: TS=${ts[i]}, WASM=${wasm[i]}`);
        }
      }
    };

    compareArrays('startPositions', tsResult.startPositions, wasmResult.startPositions);
    compareArrays('endPositions', tsResult.endPositions, wasmResult.endPositions);
    compareArrays('startColors', tsResult.startColors, wasmResult.startColors);
    compareArrays('endColors', tsResult.endColors, wasmResult.endColors);
    compareArrays('startWidths', tsResult.startWidths, wasmResult.startWidths);
    compareArrays('endWidths', tsResult.endWidths, wasmResult.endWidths);
    compareArrays('startSharpness', tsResult.startSharpness, wasmResult.startSharpness);
    compareArrays('endSharpness', tsResult.endSharpness, wasmResult.endSharpness);
    compareArrays('segmentLengths', tsResult.segmentLengths, wasmResult.segmentLengths);
    compareArrays('startClipped', tsResult.startClipped, wasmResult.startClipped);
    compareArrays('endClipped', tsResult.endClipped, wasmResult.endClipped);
  }

  it('should produce identical results for simple 3D data', () => {
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
      segments: new Uint32Array([0, 1, 1, 2]),
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

    const slicePos = [0, 0, 0];
    const tolerance = [1e10, 1e10, 1e10];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);
  });

  it('should produce identical results for 4D data with clipping', () => {
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

    const slicePos = [0, 0, 0, 5];
    const tolerance = [1e10, 1e10, 1e10, 0.5];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);
  });

  it('should produce identical results when filtering invisible segments', () => {
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

    const slicePos = [0, 0, 0, 5];
    const tolerance = [1e10, 1e10, 1e10, 0.5];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);
    expect(tsResult.segmentCount).toBe(1); // Only one visible
  });

  it('should produce identical results with null colors (default white)', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      colors: null,
      sharpness: null,
      scalars: undefined,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const slicePos = [0, 0, 0];
    const tolerance = [1e10, 1e10, 1e10];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);
    // Verify default white
    expect(Array.from(wasmResult.startColors.slice(0, 3))).toEqual([1, 1, 1]);
  });

  it('should produce identical results with null sharpness (default 1.0)', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 10, 10, 10]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
      sharpness: null,
      scalars: undefined,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const slicePos = [0, 0, 0];
    const tolerance = [1e10, 1e10, 1e10];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);
    // Verify default sharpness
    expect(wasmResult.startSharpness[0]).toBe(1.0);
  });

  it('should produce identical results with empty input', () => {
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

    const slicePos = [0, 0, 0];
    const tolerance = [1e10, 1e10, 1e10];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);
    expect(wasmResult.segmentCount).toBe(0);
  });

  it('should produce identical results for 5D data with multiple slice dimensions', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([
        0,
        0,
        0,
        5,
        10, // v0: in slice
        10,
        10,
        10,
        5,
        10, // v1: in slice
        20,
        20,
        20,
        0,
        20, // v2: out of slice
      ]),
      segments: new Uint32Array([
        0,
        1, // Visible (both in)
        1,
        2, // Clips on both dim3 and dim4
      ]),
      widths: new Float32Array([0.1, 0.2, 0.3]),
      colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      sharpness: new Float32Array([0.5, 0.7, 0.9]),
      scalars: undefined,
      segmentCount: 2,
      vertexCount: 3,
      ndim: 5,
    };

    const slicePos = [0, 0, 0, 5, 10];
    const tolerance = [1e10, 1e10, 1e10, 0.5, 1.0];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);
  });

  it('should produce identical results for many segments (stress test)', () => {
    // Generate 100 segments
    const numSegments = 100;
    const numVertices = numSegments + 1;

    const positions = new Float32Array(numVertices * 4);
    const segments = new Uint32Array(numSegments * 2);
    const widths = new Float32Array(numVertices);
    const colors = new Float32Array(numVertices * 3);
    const sharpness = new Float32Array(numVertices);

    for (let i = 0; i < numVertices; i++) {
      // Create a spiral path in 3D with varying dim3
      const angle = (i / numVertices) * Math.PI * 4;
      positions[i * 4] = Math.cos(angle) * 10;
      positions[i * 4 + 1] = Math.sin(angle) * 10;
      positions[i * 4 + 2] = i;
      positions[i * 4 + 3] = Math.sin(angle * 0.5) * 2 + 5; // Oscillates around 5

      widths[i] = 0.1 + (i / numVertices) * 0.2;
      colors[i * 3] = i / numVertices;
      colors[i * 3 + 1] = 1 - i / numVertices;
      colors[i * 3 + 2] = 0.5;
      sharpness[i] = 0.5 + (i / numVertices) * 0.5;
    }

    for (let i = 0; i < numSegments; i++) {
      segments[i * 2] = i;
      segments[i * 2 + 1] = i + 1;
    }

    const loadedData: LoadedLinesData = {
      positions,
      segments,
      widths,
      colors,
      sharpness,
      scalars: undefined,
      segmentCount: numSegments,
      vertexCount: numVertices,
      ndim: 4,
    };

    const slicePos = [0, 0, 50, 5];
    const tolerance = [1e10, 1e10, 1e10, 1.0];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);
  });

  it('should produce identical results with both endpoints clipped (Case D)', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([
        0,
        0,
        0,
        0, // v0: dim3 = 0 (below)
        10,
        10,
        10,
        10, // v1: dim3 = 10 (above)
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

    const slicePos = [0, 0, 0, 5];
    const tolerance = [1e10, 1e10, 1e10, 0.5];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);

    // Both endpoints should be clipped
    expect(wasmResult.startClipped[0]).toBe(1);
    expect(wasmResult.endClipped[0]).toBe(1);
  });

  it('should produce identical results with all segments invisible (Case E)', () => {
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([
        0,
        0,
        0,
        0, // v0: dim3 = 0
        10,
        10,
        10,
        1, // v1: dim3 = 1 (both below slice at 5)
      ]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      colors: null,
      sharpness: null,
      scalars: undefined,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 4,
    };

    const slicePos = [0, 0, 0, 5];
    const tolerance = [1e10, 1e10, 1e10, 0.5];
    const displayDims = [0, 1, 2];

    const tsResult = projectLinesTo3D(loadedData, slicePos, tolerance, displayDims);
    const wasmResult = projectLinesTo3DWASM(loadedData, slicePos, tolerance, displayDims);

    compareResults(tsResult, wasmResult);
    expect(wasmResult.segmentCount).toBe(0);
  });
});
