/**
 * Unit tests for the WASM TypeScript-fallback visibility / spatial helpers:
 * - spatial.query_chunks_for_view
 * - points.compute_nd_visibility_points
 * - gsplats.compute_nd_visibility_gsplats
 *
 * Pure math on typed arrays, no mocks. The same fixtures double as
 * cross-implementation reference data for the Rust→WASM build.
 */

import { describe, it, expect } from 'vitest';
import { query_chunks_for_view } from '../../../wasm/typescript/spatial';
import { compute_nd_visibility_points } from '../../../wasm/typescript/points';
import { compute_nd_visibility_gsplats } from '../../../wasm/typescript/gsplats';

describe('query_chunks_for_view', () => {
  it('returns empty result when no chunks intersect the slice', () => {
    // Two chunks both far from slicePosition=0 with tolerance=0.1
    const chunkBounds = new Float32Array([
      // chunk 0: x ∈ [10, 11]
      10, 11,
      // chunk 1: x ∈ [-5, -4]
      -5, -4,
    ]);
    const slicePosition = new Float32Array([0]);
    const tolerance = new Float32Array([0.1]);
    const output = new Uint32Array(2);
    const n = query_chunks_for_view(chunkBounds, slicePosition, tolerance, 1, 2, output);
    expect(n).toBe(0);
  });

  it('returns the chunk indices whose bounds straddle the slice (1D)', () => {
    const chunkBounds = new Float32Array([
      // 0: [-1, 1]   -> intersects [-0.1, 0.1]
      -1, 1,
      // 1: [10, 11]  -> doesn't
      10, 11,
      // 2: [0.05, 0.2] -> intersects via lower bound
      0.05, 0.2,
    ]);
    const slicePosition = new Float32Array([0]);
    const tolerance = new Float32Array([0.1]);
    const output = new Uint32Array(3);
    const n = query_chunks_for_view(chunkBounds, slicePosition, tolerance, 1, 3, output);
    expect(n).toBe(2);
    expect(Array.from(output.slice(0, 2))).toEqual([0, 2]);
  });

  it('requires intersection in *all* dimensions (AND semantics)', () => {
    const chunkBounds = new Float32Array([
      // chunk 0 in 2D: x ∈ [-1, 1], y ∈ [10, 11] — y misses
      -1, 1, 10, 11,
      // chunk 1: x ∈ [-1, 1], y ∈ [-1, 1] — both pass
      -1, 1, -1, 1,
    ]);
    const slicePosition = new Float32Array([0, 0]);
    const tolerance = new Float32Array([0.5, 0.5]);
    const output = new Uint32Array(2);
    const n = query_chunks_for_view(chunkBounds, slicePosition, tolerance, 2, 2, output);
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('handles tolerance=0 — chunk must literally span the slice position', () => {
    const chunkBounds = new Float32Array([
      // chunk 0: [-1, 1] — spans 0
      -1, 1,
      // chunk 1: [0.5, 1.5] — does not span 0
      0.5, 1.5,
    ]);
    const slicePosition = new Float32Array([0]);
    const tolerance = new Float32Array([0]);
    const output = new Uint32Array(2);
    const n = query_chunks_for_view(chunkBounds, slicePosition, tolerance, 1, 2, output);
    expect(n).toBe(1);
    expect(output[0]).toBe(0);
  });

  it('handles numChunks=0', () => {
    const output = new Uint32Array(0);
    const n = query_chunks_for_view(
      new Float32Array(0),
      new Float32Array([0]),
      new Float32Array([1]),
      1,
      0,
      output
    );
    expect(n).toBe(0);
  });
});

describe('compute_nd_visibility_points', () => {
  it('a point at the slice with non-zero tolerance is visible', () => {
    const positions = new Float32Array([0, 0, 0]);
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('a point on the tolerance boundary is visible (≤1, not <1)', () => {
    // delta=0.1 with tolerance=0.1 → normalized=1 → distSq=1 → visible
    const positions = new Float32Array([0.1]);
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      1,
      output
    );
    expect(output[0]).toBe(1);
  });

  it('a point outside the tolerance is hidden', () => {
    const positions = new Float32Array([0.5]);
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('point radius extends visibility in 3D', () => {
    // Point at delta=0.5, tolerance=0.1, radius=0.5
    // effectiveTolerance = 0.6 in each dim → normalized=0.5/0.6 = 0.833
    // distSq = 3 × 0.833² = 2.083 → still hidden (>1)
    // But with radius=2.0 in just one dim:
    const positions = new Float32Array([0.5, 0, 0]);
    const radii = new Float32Array([2.0]);
    const output = new Uint8Array(1);
    compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    // delta=0.5, tol=0.1+2=2.1, normalized=0.238, distSq=0.057 → visible
    expect(output[0]).toBe(1);
  });

  it('zero tolerance + zero radius + non-trivial delta hides the point', () => {
    const positions = new Float32Array([1, 1]);
    const radii = new Float32Array([0]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0, 0]),
      new Float32Array([0, 0]),
      2,
      1,
      output
    );
    expect(n).toBe(0);
  });

  it('mixed visibility across many points returns the correct count', () => {
    // 5 points in 1D at deltas 0, 0.05, 0.1, 0.2, 1.0
    const positions = new Float32Array([0, 0.05, 0.1, 0.2, 1.0]);
    const radii = new Float32Array([0, 0, 0, 0, 0]);
    const output = new Uint8Array(5);
    const n = compute_nd_visibility_points(
      positions,
      radii,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      5,
      output
    );
    expect(n).toBe(3);
    expect(Array.from(output)).toEqual([1, 1, 1, 0, 0]);
  });
});

describe('compute_nd_visibility_gsplats', () => {
  it('an isotropic 3D splat at the slice is visible', () => {
    // 3D splat with diagonal Cholesky [1, 0, 1, 0, 0, 1] = identity covariance
    const centers = new Float32Array([0, 0, 0]);
    const cholesky = new Float32Array([1, 0, 1, 0, 0, 1]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(1);
    expect(output[0]).toBe(1);
  });

  it('a far-away splat with small extent is hidden', () => {
    const centers = new Float32Array([100, 0, 0]);
    const cholesky = new Float32Array([0.01, 0, 0.01, 0, 0, 0.01]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      new Float32Array([0, 0, 0]),
      new Float32Array([0.1, 0.1, 0.1]),
      3,
      1,
      output
    );
    expect(n).toBe(0);
    expect(output[0]).toBe(0);
  });

  it('a stretched splat (large row norm) reaches further', () => {
    // 1D splat at distance 5 with cholesky=[6] → max extent = 6, tolerance=0.1
    // effectiveTolerance = 6.1, delta=5, normalized=5/6.1=0.82, distSq=0.67 → visible
    const centers = new Float32Array([5]);
    const cholesky = new Float32Array([6]);
    const output = new Uint8Array(1);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      1,
      output
    );
    expect(n).toBe(1);
  });

  it('mixed visibility across a batch returns the right count + mask', () => {
    // Three 1D splats: at 0 (visible), 100 (hidden), 0.05 (visible)
    const centers = new Float32Array([0, 100, 0.05]);
    const cholesky = new Float32Array([0.5, 0.5, 0.5]);
    const output = new Uint8Array(3);
    const n = compute_nd_visibility_gsplats(
      centers,
      cholesky,
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      3,
      output
    );
    expect(n).toBe(2);
    expect(Array.from(output)).toEqual([1, 0, 1]);
  });

  it('numSplats=0 returns 0 and writes nothing', () => {
    const output = new Uint8Array(2).fill(99);
    const n = compute_nd_visibility_gsplats(
      new Float32Array(0),
      new Float32Array(0),
      new Float32Array([0]),
      new Float32Array([0.1]),
      1,
      0,
      output
    );
    expect(n).toBe(0);
    expect(Array.from(output)).toEqual([99, 99]); // untouched
  });
});
