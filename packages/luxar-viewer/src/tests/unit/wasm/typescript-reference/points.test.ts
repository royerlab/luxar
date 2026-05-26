/**
 * Tests for `src/wasm/typescript/points.ts` (hypersphere visibility).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D3).
 */

import { describe, it, expect } from 'vitest';
import { compute_nd_visibility_points } from '../../../../wasm/typescript';

describe('points: compute_nd_visibility_points', () => {
  it('should filter visible points in 5D', () => {
    const ndim = 5;
    const numPoints = 4;

    const positions = new Float32Array([
      0.0,
      0.0,
      0.0,
      0.0,
      0.0, // Point 0: at origin
      1.0,
      1.0,
      1.0,
      0.5,
      0.5, // Point 1: within tolerance in hidden dims
      0.0,
      0.0,
      0.0,
      5.0,
      5.0, // Point 2: far in hidden dims
      0.0,
      0.0,
      0.0,
      0.8,
      0.0, // Point 3: borderline
    ]);

    const radii = new Float32Array([1.0, 1.0, 1.0, 1.0]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0, 0.0]);
    const tolerance = new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0]);

    const output = new Uint8Array(numPoints);
    const visibleCount = compute_nd_visibility_points(
      positions,
      radii,
      slicePos,
      tolerance,
      ndim,
      numPoints,
      output
    );

    expect(output[0]).toBe(1); // visible
    expect(output[1]).toBe(1); // visible
    expect(output[2]).toBe(0); // hidden
    expect(visibleCount).toBeGreaterThanOrEqual(2);
  });

  it('should handle 3D case (all visible)', () => {
    const output = new Uint8Array(2);
    const count = compute_nd_visibility_points(
      new Float32Array([0, 0, 0, 1, 1, 1]),
      new Float32Array([0.5, 0.5]),
      new Float32Array([0, 0, 0]),
      new Float32Array([10, 10, 10]),
      3,
      2,
      output
    );

    expect(count).toBe(2);
    expect(output[0]).toBe(1);
    expect(output[1]).toBe(1);
  });
});
