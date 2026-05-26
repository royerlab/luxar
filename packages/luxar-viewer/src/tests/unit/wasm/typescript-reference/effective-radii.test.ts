/**
 * Tests for `src/wasm/typescript/effective-radii.ts` (nD hypersphere slicing).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D5).
 */

import { describe, it, expect } from 'vitest';
import { calculate_effective_radii } from '../../../../wasm/typescript';

describe('effective_radii: calculate_effective_radii', () => {
  it('should preserve radii when no hidden dimensions', () => {
    // 3D case: all dimensions displayed
    const positions = new Float32Array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    const radii = new Float32Array([1.0, 0.5]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0]);
    const spatialExtend = new Uint8Array([1, 1, 1]);
    const output = new Float32Array(2);

    const visible = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePos,
      spatialExtend,
      3,
      2,
      output
    );

    expect(visible).toBe(2);
    expect(output[0]).toBeCloseTo(1.0, 5);
    expect(output[1]).toBeCloseTo(0.5, 5);
  });

  it('should compute correct effective radius with hidden spatial dims', () => {
    // 4D: point at distance 0.6 from slice in hidden dim
    // R=1.0, D=0.6, R_eff = sqrt(1 - 0.36) = 0.8
    const positions = new Float32Array([0.0, 0.0, 0.0, 0.6]);
    const radii = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0]);
    const spatialExtend = new Uint8Array([1, 1, 1, 1]);
    const output = new Float32Array(1);

    const visible = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePos,
      spatialExtend,
      4,
      1,
      output
    );

    expect(visible).toBe(1);
    expect(output[0]).toBeCloseTo(0.8, 5);
  });

  it('should return zero when point is outside radius', () => {
    // 5D: point at distance 1.0 from slice (sqrt(0.6² + 0.8²) = 1.0)
    const positions = new Float32Array([0.0, 0.0, 0.0, 0.6, 0.8]);
    const radii = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0, 0.0]);
    const spatialExtend = new Uint8Array([1, 1, 1, 1, 1]);
    const output = new Float32Array(1);

    const visible = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePos,
      spatialExtend,
      5,
      1,
      output
    );

    expect(visible).toBe(0);
    expect(output[0]).toBeCloseTo(0, 5);
  });

  it('should filter discrete dimension mismatches', () => {
    // 4D: dim 3 is discrete (not spatial)
    const positions = new Float32Array([
      0.0,
      0.0,
      0.0,
      0.0, // matches
      0.0,
      0.0,
      0.0,
      5.0, // doesn't match
    ]);
    const radii = new Float32Array([1.0, 1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0]);
    const spatialExtend = new Uint8Array([1, 1, 1, 0]); // dim 3 is discrete
    const output = new Float32Array(2);

    const visible = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePos,
      spatialExtend,
      4,
      2,
      output
    );

    expect(visible).toBe(1);
    expect(output[0]).toBeCloseTo(1.0, 5);
    expect(output[1]).toBeCloseTo(0, 5);
  });

  it('should handle large point counts', () => {
    const numPoints = 10000;
    const ndim = 4;

    const positions = new Float32Array(numPoints * ndim);
    const radii = new Float32Array(numPoints).fill(1.0);

    // Place points at varying distances in dim 3
    for (let i = 0; i < numPoints; i++) {
      positions[i * ndim + 3] = (i / numPoints) * 2; // 0 to 2
    }

    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0]);
    const spatialExtend = new Uint8Array([1, 1, 1, 1]);
    const output = new Float32Array(numPoints);

    const visible = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePos,
      spatialExtend,
      ndim,
      numPoints,
      output
    );

    // Points with distance < 1 should be visible
    expect(visible).toBe(5000); // First half have distance < 1
  });

  it('should handle multiple hidden spatial dimensions', () => {
    // 6D: display dims 0,1,2; hidden spatial dims 3,4,5
    // Point at (0,0,0, 0.3, 0.4, 0) - distance = 0.5
    // R=1.0, D=0.5, R_eff = sqrt(1 - 0.25) = sqrt(0.75) ≈ 0.866
    const positions = new Float32Array([0.0, 0.0, 0.0, 0.3, 0.4, 0.0]);
    const radii = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    const spatialExtend = new Uint8Array([1, 1, 1, 1, 1, 1]);
    const output = new Float32Array(1);

    const visible = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePos,
      spatialExtend,
      6,
      1,
      output
    );

    expect(visible).toBe(1);
    expect(output[0]).toBeCloseTo(Math.sqrt(0.75), 5);
  });

  it('MED-18: dims beyond spatialExtendDims.length default to spatial (documented fallback)', () => {
    // The docstring pins the contract: dims `d >= spatialExtendDims.length`
    // are treated as spatial (contribute to Pythagorean distance) rather
    // than as discrete (which would silently apply the 0.5 exact-match
    // tolerance). This test makes the documented default observable.
    //
    // Setup: 5D, but spatialExtendDims has only 3 entries (covers dims 0-2).
    // Dims 3 and 4 are "missing" — must default to spatial.
    // Display dims = [0,1,2]; point at (0,0,0, 0.3, 0.4) on hidden dims.
    // Spatial default: D² = 0.09 + 0.16 = 0.25, R_eff = sqrt(0.75).
    // Discrete-default would treat dims 3 & 4 as exact-match: |0.3|, |0.4|
    // both <= 0.5 → match, no distance accumulated → R_eff = 1.0.
    // The two paths yield clearly different outputs.
    const positions = new Float32Array([0.0, 0.0, 0.0, 0.3, 0.4]);
    const radii = new Float32Array([1.0]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0, 0.0]);
    // Only 3 entries — dims 3 and 4 are beyond the array length.
    const spatialExtend = new Uint8Array([1, 1, 1]);
    const output = new Float32Array(1);

    const visible = calculate_effective_radii(
      positions,
      radii,
      displayDims,
      slicePos,
      spatialExtend,
      5, // ndim = 5, but spatialExtend.length = 3
      1,
      output
    );

    expect(visible).toBe(1);
    // Spatial default path: sqrt(1 - 0.25) = sqrt(0.75) ≈ 0.866.
    // (Discrete default would yield 1.0.)
    expect(output[0]).toBeCloseTo(Math.sqrt(0.75), 5);
  });
});
