/**
 * Tests for `src/wasm/typescript/projection.ts` (nD → 3D projection).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D7).
 */

import { describe, it, expect } from 'vitest';
import { extract_3d_positions } from '../../../../wasm/typescript';

describe('projection: extract_3d_positions', () => {
  it('should extract 3D positions from 5D data', () => {
    // 5D data: 2 points
    const positionsNd = new Float32Array([
      1.0,
      2.0,
      3.0,
      4.0,
      5.0, // point 0
      6.0,
      7.0,
      8.0,
      9.0,
      10.0, // point 1
    ]);
    const displayDims = new Uint32Array([0, 2, 4]); // X=dim0, Y=dim2, Z=dim4
    const output = new Float32Array(6);

    extract_3d_positions(positionsNd, displayDims, 5, 2, output);

    expect(output[0]).toBe(1.0); // point0.x = dim0
    expect(output[1]).toBe(3.0); // point0.y = dim2
    expect(output[2]).toBe(5.0); // point0.z = dim4
    expect(output[3]).toBe(6.0); // point1.x = dim0
    expect(output[4]).toBe(8.0); // point1.y = dim2
    expect(output[5]).toBe(10.0); // point1.z = dim4
  });

  it('should handle 2D display (z=0)', () => {
    // 4D data displayed as 2D
    const positionsNd = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const displayDims = new Uint32Array([0, 1]); // Only X and Y
    const output = new Float32Array(3);

    extract_3d_positions(positionsNd, displayDims, 4, 1, output);

    expect(output[0]).toBe(1.0); // x
    expect(output[1]).toBe(2.0); // y
    expect(output[2]).toBe(0.0); // z = 0 (default)
  });

  it('should handle reordered dimensions', () => {
    const positionsNd = new Float32Array([10.0, 20.0, 30.0]);
    const displayDims = new Uint32Array([2, 0, 1]); // Z->X, X->Y, Y->Z
    const output = new Float32Array(3);

    extract_3d_positions(positionsNd, displayDims, 3, 1, output);

    expect(output[0]).toBe(30.0); // x = dim2
    expect(output[1]).toBe(10.0); // y = dim0
    expect(output[2]).toBe(20.0); // z = dim1
  });
});
