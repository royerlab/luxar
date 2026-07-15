/**
 * Tests for `src/wasm/typescript/depth-sort.ts` (back-to-front splat
 * ordering for order-dependent blending — depth-sorting Phase 2).
 *
 * Mirrors the Rust in-file tests in `wasm/rust/src/depth_sort.rs` 1:1 so
 * both backends pin the same contract (the WASM-vs-TS parity suite then
 * asserts exact-permutation equality between them).
 */

import { describe, it, expect } from 'vitest';
import { sort_splats_by_depth } from '../../../../wasm/typescript';

/** Identity model-view: view z == world z (camera at origin looking -z). */
const IDENTITY_MV = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/**
 * Build centers with the given view-space z values (x = index, y = 0) so
 * the identity model-view maps world z directly to view z.
 */
function centersWithZ(zs: number[]): Float32Array {
  const centers = new Float32Array(zs.length * 3);
  for (let i = 0; i < zs.length; i++) {
    centers[i * 3] = i;
    centers[i * 3 + 1] = 0;
    centers[i * 3 + 2] = zs[i];
  }
  return centers;
}

function sortZs(zs: number[]): { ordering: Uint32Array; sorted: number } {
  const ordering = new Uint32Array(zs.length).fill(0xffffffff);
  const sorted = sort_splats_by_depth(centersWithZ(zs), IDENTITY_MV, ordering, zs.length);
  return { ordering, sorted };
}

function assertIsPermutation(ordering: Uint32Array, count: number): void {
  const seen = new Uint8Array(count);
  for (const idx of ordering) {
    expect(idx).toBeLessThan(count);
    expect(seen[idx]).toBe(0);
    seen[idx] = 1;
  }
}

describe('depth_sort: sort_splats_by_depth', () => {
  it('orders back-to-front (farthest = most negative view z first)', () => {
    const { ordering, sorted } = sortZs([-1, -10, -5, -2]);
    expect(sorted).toBe(4);
    assertIsPermutation(ordering, 4);
    expect(Array.from(ordering)).toEqual([1, 2, 3, 0]); // z: -10, -5, -2, -1
  });

  it('keeps already-sorted input as identity', () => {
    const { ordering, sorted } = sortZs([-10, -5, -2, -1]);
    expect(sorted).toBe(4);
    expect(Array.from(ordering)).toEqual([0, 1, 2, 3]);
  });

  it('is stable: equal depths keep input order', () => {
    const { ordering, sorted } = sortZs([-5, -1, -5, -5, -1]);
    expect(sorted).toBe(5);
    expect(Array.from(ordering)).toEqual([0, 2, 3, 1, 4]);
  });

  it('keys behind-camera splats to the far bucket (drawn first)', () => {
    const { ordering, sorted } = sortZs([-1, 3, -10, 0]);
    expect(sorted).toBe(4);
    assertIsPermutation(ordering, 4);
    // Far bucket 0 holds behind-camera (1, 3) and the farthest splat (2,
    // whose z == zmin keys to 0); stable input order within the bucket.
    expect(Array.from(ordering)).toEqual([1, 2, 3, 0]);
  });

  it('falls back to identity on uniform depth (zmax == zmin)', () => {
    const { ordering, sorted } = sortZs([-4, -4, -4]);
    expect(sorted).toBe(0);
    expect(Array.from(ordering)).toEqual([0, 1, 2]);
  });

  it('falls back to identity for a single splat', () => {
    const { ordering, sorted } = sortZs([-7.5]);
    expect(sorted).toBe(0);
    expect(Array.from(ordering)).toEqual([0]);
  });

  it('falls back to identity when everything is behind the camera', () => {
    const { ordering, sorted } = sortZs([1, 2, 0.5]);
    expect(sorted).toBe(0);
    expect(Array.from(ordering)).toEqual([0, 1, 2]);
  });

  it('handles empty input', () => {
    const ordering = new Uint32Array(0);
    expect(sort_splats_by_depth(new Float32Array(0), IDENTITY_MV, ordering, 0)).toBe(0);
  });

  it('keys a NaN center to the NEAR bucket (Rust f32::min semantics)', () => {
    // Rust's `f32::min(NaN, 65535.0)` returns 65535 — the saturating
    // `as u16` is never reached with NaN. The twin must reproduce that
    // (a naive `Math.min(NaN, x) | 0` would key to 0, the far bucket,
    // and diverge from WASM). Mirrors test_nan_center_keys_to_near_bucket.
    const { ordering, sorted } = sortZs([NaN, -3, -8]);
    expect(sorted).toBe(3);
    assertIsPermutation(ordering, 3);
    expect(Array.from(ordering)).toEqual([2, 0, 1]);
  });

  it('orders nanometer-scale magnitudes (~1e-6) correctly', () => {
    // Raw f16 keys would underflow to a single bucket at this scale.
    const { ordering, sorted } = sortZs([-1e-6, -9e-6, -5e-6, -3e-6]);
    expect(sorted).toBe(4);
    assertIsPermutation(ordering, 4);
    expect(Array.from(ordering)).toEqual([1, 2, 3, 0]);
  });

  it('orders kilometer-scale magnitudes (~1e6) correctly', () => {
    // Raw f16 keys would overflow to Inf at this scale.
    const { ordering, sorted } = sortZs([-1e6, -9e6, -5e6, -3e6]);
    expect(sorted).toBe(4);
    assertIsPermutation(ordering, 4);
    expect(Array.from(ordering)).toEqual([1, 2, 3, 0]);
  });

  it('stays a valid permutation across a 12-orders-of-magnitude depth range', () => {
    const zs = [-1e-6, -1e6, -1, -1e3, -1e-3];
    const { ordering, sorted } = sortZs(zs);
    expect(sorted).toBe(5);
    assertIsPermutation(ordering, 5);
    // The km-scale splat is unambiguously farthest.
    expect(ordering[0]).toBe(1);
  });

  it('applies the model-view translation to view z', () => {
    // mv translates z by -5, so world z = +2 -> view z = -3.
    const mv = new Float32Array(IDENTITY_MV);
    mv[14] = -5;
    const centers = centersWithZ([2, -2, 4]);
    const ordering = new Uint32Array(3);
    const sorted = sort_splats_by_depth(centers, mv, ordering, 3);
    expect(sorted).toBe(3);
    // View z: -3, -7, -1 -> back-to-front: 1, 0, 2.
    expect(Array.from(ordering)).toEqual([1, 0, 2]);
  });

  it('uses the full model-view z row (rotation about y)', () => {
    // 90° rotation about y (column-major): view z comes from world x.
    const mv = new Float32Array([0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]);
    // Centers along x: view z = -x.
    const centers = new Float32Array([1, 0, 0, 9, 0, 0, 5, 0, 0]);
    const ordering = new Uint32Array(3);
    const sorted = sort_splats_by_depth(centers, mv, ordering, 3);
    expect(sorted).toBe(3);
    expect(Array.from(ordering)).toEqual([1, 2, 0]);
  });

  it('produces a monotone back-to-front permutation on 10k random depths', () => {
    // Deterministic xorshift32 depths in [-1000, -1] (matches the Rust
    // test); verify permutation + per-bucket monotonicity.
    const count = 10_000;
    const zs: number[] = [];
    let state = 0x12345678;
    for (let i = 0; i < count; i++) {
      state ^= (state << 13) >>> 0;
      state >>>= 0;
      state ^= state >>> 17;
      state ^= (state << 5) >>> 0;
      state >>>= 0;
      const unit = Math.fround(state / 0xffffffff);
      zs.push(Math.fround(-1 - unit * 999));
    }
    const { ordering, sorted } = sortZs(zs);
    expect(sorted).toBe(count);
    assertIsPermutation(ordering, count);
    const bucketWidth = 999 / 65535;
    let prev = -Infinity;
    for (const idx of ordering) {
      const z = zs[idx];
      expect(z).toBeGreaterThanOrEqual(prev - bucketWidth);
      prev = Math.max(prev, z);
    }
  });
});
