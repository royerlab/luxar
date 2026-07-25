/**
 * Fused projection-output scans (perf lever L1) — correctness against
 * independent brute-force references.
 *
 * `computeGSplatsProjectionBounds` / `computeLinesProjectionBounds` run
 * once at the projection output boundary (worker thread for the nD
 * path) so the GPU commit can skip its O(N) main-thread bbox / row-norm
 * / max-width scans. These tests pin the fused single-pass values to a
 * THREE.Box3-based brute-force scan written the way the commit-path
 * fallbacks do it — on randomized data with negative coordinates and
 * anisotropic Cholesky factors — so the fast path can never drift from
 * the fallback.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { computeGSplatsProjectionBounds } from '../../../../workers/data-worker/projection/gsplats';
import { computeLinesProjectionBounds } from '../../../../workers/data-worker/projection/lines';

/** Deterministic PRNG (mulberry32). */
function prng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('computeGSplatsProjectionBounds — fused scan == brute force', () => {
  it('matches Box3 + row-norm brute force on random anisotropic data (bit-exact)', () => {
    const rand = prng(97);
    const count = 500;
    const centers = new Float32Array(count * 3);
    const chol = new Float32Array(count * 6);
    for (let i = 0; i < count; i++) {
      // Negative and positive coordinates.
      centers.set([rand() * 2000 - 1000, rand() * 2000 - 1000, rand() * 2000 - 1000], i * 3);
      // Anisotropic Cholesky: diagonals over ~4 decades, signed off-diagonals.
      const d0 = Math.pow(10, rand() * 4 - 2);
      const d1 = Math.pow(10, rand() * 4 - 2);
      const d2 = Math.pow(10, rand() * 4 - 2);
      chol.set(
        [d0, (rand() * 2 - 1) * d1, d1, (rand() * 2 - 1) * d2, (rand() * 2 - 1) * d2, d2],
        i * 6
      );
    }

    // Brute force, written the way the adapter fallback does it.
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      v.set(centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2]);
      box.expandByPoint(v);
    }
    let maxRowNorm = 0;
    for (let i = 0; i < count; i++) {
      const c6 = i * 6;
      const row0 = Math.abs(chol[c6]);
      const row1 = Math.sqrt(chol[c6 + 1] * chol[c6 + 1] + chol[c6 + 2] * chol[c6 + 2]);
      const row2 = Math.sqrt(
        chol[c6 + 3] * chol[c6 + 3] + chol[c6 + 4] * chol[c6 + 4] + chol[c6 + 5] * chol[c6 + 5]
      );
      maxRowNorm = Math.max(maxRowNorm, row0, row1, row2);
    }

    const fused = computeGSplatsProjectionBounds(centers, chol, count);
    expect(fused.min).toEqual([box.min.x, box.min.y, box.min.z]);
    expect(fused.max).toEqual([box.max.x, box.max.y, box.max.z]);
    expect(fused.maxRowNorm).toBe(maxRowNorm);
  });

  it('handles a single splat at a negative corner', () => {
    const centers = new Float32Array([-5, -6, -7]);
    const chol = new Float32Array([2, 0, 3, 0, 0, 4]);
    const fused = computeGSplatsProjectionBounds(centers, chol, 1);
    expect(fused.min).toEqual([-5, -6, -7]);
    expect(fused.max).toEqual([-5, -6, -7]);
    expect(fused.maxRowNorm).toBe(4);
  });
});

describe('computeLinesProjectionBounds — fused scan == brute force', () => {
  it('matches Box3 + max-width brute force on random segments (bit-exact)', () => {
    const rand = prng(31);
    const count = 400;
    const starts = new Float32Array(count * 3);
    const ends = new Float32Array(count * 3);
    const startWidths = new Float32Array(count);
    const endWidths = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      starts.set([rand() * 400 - 200, rand() * 400 - 200, rand() * 400 - 200], i * 3);
      ends.set([rand() * 400 - 200, rand() * 400 - 200, rand() * 400 - 200], i * 3);
      startWidths[i] = rand() * 10;
      endWidths[i] = rand() * 10;
    }
    // Non-finite width must be ignored, exactly like computeLineBounds'
    // Number.isFinite guard.
    startWidths[7] = Infinity;
    endWidths[13] = NaN;

    // Brute force, written the way computeLineBounds' fallback does it.
    const box = new THREE.Box3(
      new THREE.Vector3(Infinity, Infinity, Infinity),
      new THREE.Vector3(-Infinity, -Infinity, -Infinity)
    );
    const v = new THREE.Vector3();
    let maxWidth = 0;
    for (let i = 0; i < count; i++) {
      const si = i * 3;
      v.set(starts[si], starts[si + 1], starts[si + 2]);
      box.expandByPoint(v);
      v.set(ends[si], ends[si + 1], ends[si + 2]);
      box.expandByPoint(v);
      const sw = startWidths[i];
      const ew = endWidths[i];
      if (Number.isFinite(sw) && sw > maxWidth) maxWidth = sw;
      if (Number.isFinite(ew) && ew > maxWidth) maxWidth = ew;
    }

    const fused = computeLinesProjectionBounds(starts, ends, startWidths, endWidths, count);
    expect(fused.min).toEqual([box.min.x, box.min.y, box.min.z]);
    expect(fused.max).toEqual([box.max.x, box.max.y, box.max.z]);
    expect(fused.maxWidth).toBe(maxWidth);
  });

  it('covers both endpoints of a single segment', () => {
    const fused = computeLinesProjectionBounds(
      new Float32Array([-1, -2, -3]),
      new Float32Array([4, 5, 6]),
      new Float32Array([0.5]),
      new Float32Array([1.5]),
      1
    );
    expect(fused.min).toEqual([-1, -2, -3]);
    expect(fused.max).toEqual([4, 5, 6]);
    expect(fused.maxWidth).toBe(1.5);
  });
});
