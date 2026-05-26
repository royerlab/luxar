/**
 * Tests for `src/wasm/typescript/gsplats.ts` (ellipsoid visibility).
 *
 * Extracted from `tests/unit/wasm/typescript-reference.test.ts` per the
 * restructuring plan (wasm.md O1 / Phase D4).
 */

import { describe, it, expect } from 'vitest';
import { compute_nd_visibility_gsplats } from '../../../../wasm/typescript';

describe('gsplats: compute_nd_visibility_gsplats', () => {
  it('should filter splats based on center proximity and ellipsoid extent', () => {
    // 3D test matching Rust test: gsplats.rs::test_gsplat_visibility_basic
    const ndim = 3;
    const numSplats = 2;

    const centers = new Float32Array([
      0.0,
      0.0,
      0.0, // Splat 0 at origin
      10.0,
      10.0,
      10.0, // Splat 1 far away
    ]);
    // 3D cholesky: 6 elements per splat [L00, L10, L11, L20, L21, L22]
    // Using identity-ish (1.0 on diagonals)
    const choleskyFactors = new Float32Array([
      1.0,
      0.0,
      1.0,
      0.0,
      0.0,
      1.0, // Splat 0
      1.0,
      0.0,
      1.0,
      0.0,
      0.0,
      1.0, // Splat 1
    ]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0]);
    const tolerance = new Float32Array([2.0, 2.0, 2.0]);

    const output = new Uint8Array(numSplats);
    const count = compute_nd_visibility_gsplats(
      centers,
      choleskyFactors,
      slicePos,
      tolerance,
      ndim,
      numSplats,
      output
    );

    expect(output[0]).toBe(1); // Splat 0 visible (at origin)
    expect(output[1]).toBe(0); // Splat 1 hidden (far away)
    expect(count).toBe(1);
  });

  it('should handle 4D splats with hidden time dimension', () => {
    // 4D test matching Rust test: gsplats.rs::test_gsplat_visibility_4d
    const ndim = 4;
    const numSplats = 2;

    const centers = new Float32Array([
      0.0,
      0.0,
      0.0,
      0.0, // Splat 0 at t=0
      0.0,
      0.0,
      0.0,
      10.0, // Splat 1 at t=10
    ]);
    // 4D cholesky: 10 elements per splat
    const choleskyFactors = new Float32Array([
      1.0,
      0.0,
      1.0,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
      1.0, // Splat 0
      1.0,
      0.0,
      1.0,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
      1.0, // Splat 1
    ]);
    const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 2.0]); // Infinite for XYZ, 2.0 for T

    const output = new Uint8Array(numSplats);
    const count = compute_nd_visibility_gsplats(
      centers,
      choleskyFactors,
      slicePos,
      tolerance,
      ndim,
      numSplats,
      output
    );

    expect(output[0]).toBe(1); // Splat 0 visible (t=0)
    expect(output[1]).toBe(0); // Splat 1 hidden (t=10 > tolerance+extent)
    expect(count).toBe(1);
  });
});
