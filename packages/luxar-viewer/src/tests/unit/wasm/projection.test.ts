/**
 * Unit tests for the TypeScript fallback of the WASM projection routines.
 *
 * These run when the WASM module isn't available (or loaded asynchronously
 * the call site couldn't wait for); the fallbacks must behave identically
 * to the WASM implementations. Pure math on typed arrays — no mocks.
 */

import { describe, it, expect } from 'vitest';
import { extract_3d_positions } from '../../../wasm/typescript/projection';

describe('extract_3d_positions', () => {
  it('selects the displayed dimensions from each nD position', () => {
    // 3 points in 4D; show dims [0, 1, 2].
    const positionsNd = new Float32Array([1, 2, 3, 99, 4, 5, 6, 99, 7, 8, 9, 99]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const output = new Float32Array(9);
    extract_3d_positions(positionsNd, displayDims, 4, 3, output);
    expect(Array.from(output)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('reorders dimensions when displayDims is non-canonical', () => {
    const positionsNd = new Float32Array([1, 2, 3, 4]);
    const displayDims = new Uint32Array([3, 0, 2]);
    const output = new Float32Array(3);
    extract_3d_positions(positionsNd, displayDims, 4, 1, output);
    expect(Array.from(output)).toEqual([4, 1, 3]);
  });

  it('zero-fills the unused output axes when fewer than 3 displayDims', () => {
    const positionsNd = new Float32Array([10, 20, 30]);
    const displayDims = new Uint32Array([0, 1]); // 2D output
    const output = new Float32Array(3);
    extract_3d_positions(positionsNd, displayDims, 3, 1, output);
    expect(output[0]).toBe(10);
    expect(output[1]).toBe(20);
    expect(output[2]).toBe(0);
  });

  it('caps at 3 displayDims even when more are supplied', () => {
    // ndim=5, displayDims has 5 entries — only first 3 should be used.
    const positionsNd = new Float32Array([1, 2, 3, 4, 5]);
    const displayDims = new Uint32Array([0, 1, 2, 3, 4]);
    const output = new Float32Array(3);
    extract_3d_positions(positionsNd, displayDims, 5, 1, output);
    expect(Array.from(output)).toEqual([1, 2, 3]);
  });

  it('handles numPoints=0 by leaving output untouched (no iteration)', () => {
    const output = new Float32Array(3).fill(42);
    extract_3d_positions(new Float32Array(0), new Uint32Array([0, 1, 2]), 3, 0, output);
    expect(Array.from(output)).toEqual([42, 42, 42]);
  });
});

// [wasm.md/G8][P5] extract_3d_positions displayDims.length=0 and =1 cases.
// The source caps numDisplayDims at min(displayDims.length, 3) and
// zero-fills the rest; the boundary cases are unverified above.
describe('extract_3d_positions — displayDims boundary lengths', () => {
  it('displayDims.length=0 zero-fills all 3 output slots', () => {
    const positionsNd = new Float32Array([10, 20, 30]);
    const displayDims = new Uint32Array([]); // no displayed dims
    const output = new Float32Array(3).fill(42);
    extract_3d_positions(positionsNd, displayDims, 3, 1, output);
    expect(Array.from(output)).toEqual([0, 0, 0]);
  });

  it('displayDims.length=1 puts that dim at output[0] and zero-fills rest', () => {
    // ndim=4, single displayed dim selecting index 2 → output [d2, 0, 0].
    const positionsNd = new Float32Array([100, 200, 300, 400]);
    const displayDims = new Uint32Array([2]);
    const output = new Float32Array(3).fill(42);
    extract_3d_positions(positionsNd, displayDims, 4, 1, output);
    expect(Array.from(output)).toEqual([300, 0, 0]);
  });
});
