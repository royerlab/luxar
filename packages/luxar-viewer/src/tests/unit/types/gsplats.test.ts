/**
 * Unit tests for gsplats-specific utilities.
 *
 * [types.md/O1][P10] The isGSplatsMetadata / isGSplatsUserData guard
 * coverage has been consolidated into `geometry-guards.test.ts` (single
 * source of truth across all three geometry types). This file now hosts
 * only the gsplats-specific surface: `choleskyPackedSize` arithmetic
 * and the `CHOLESKY_SIZES` constant table.
 */

import { describe, it, expect } from 'vitest';
import {
  choleskyPackedSize,
  choleskyDiagIndices,
  choleskyOffdiagIndices,
  CHOLESKY_SIZES,
} from '../../../types/gsplats';

describe('GSplats Types', () => {
  // [types.md/O5][P4] Replaces four sibling `should return correct size for ND`
  // tests with a single it.each. Also strengthens to (a) the closed-form
  // identity n*(n+1)/2 and (b) cross-checking the CHOLESKY_SIZES constant.
  describe('choleskyPackedSize', () => {
    it.each([
      [1, 1],
      [2, 3],
      [3, 6],
      [4, 10],
      [5, 15],
      [6, 21],
      [7, 28],
      [16, 136],
    ])('packs %dD as %d elements', (n, expected) => {
      expect(choleskyPackedSize(n)).toBe(expected);
      // Closed-form identity: triangular number T_n = n*(n+1)/2.
      expect(choleskyPackedSize(n)).toBe((n * (n + 1)) / 2);
    });

    // [types.md/W6][P2] Boundary: ndim=0. Audit noted no test for ndim=0.
    // T_0 = 0; a non-trivial mutation would surface here.
    it('handles ndim=0 (empty packing)', () => {
      expect(choleskyPackedSize(0)).toBe(0);
    });

    it.each([
      [1, '1D'],
      [2, '2D'],
      [3, '3D'],
      [4, '4D'],
    ] as const)('matches CHOLESKY_SIZES[%sD]', (n, key) => {
      expect(choleskyPackedSize(n)).toBe(CHOLESKY_SIZES[key]);
    });
  });

  describe('CHOLESKY_SIZES constant', () => {
    it('should have correct values', () => {
      expect(CHOLESKY_SIZES['2D']).toBe(3);
      expect(CHOLESKY_SIZES['3D']).toBe(6);
      expect(CHOLESKY_SIZES['4D']).toBe(10);
    });

    // OOS-3 (round-2 audit): the constant previously started at 2D, but
    // LoadedGSplatsData permits ndim=1. A 1D consumer indexing
    // `CHOLESKY_SIZES['1D']` would have hit `undefined` and produced
    // silently wrong stride math. The 1D entry is now [L00] = 1.
    it('covers 1D (the new degenerate-but-legal entry)', () => {
      expect(CHOLESKY_SIZES['1D']).toBe(1);
      expect(CHOLESKY_SIZES['1D']).toBe(choleskyPackedSize(1));
    });

    it('MED-44: covers 1D through 16D (full LoadedGSplatsData range up to the WASM ceiling)', () => {
      // Every entry must match `choleskyPackedSize(n) = n*(n+1)/2` and be
      // defined (i.e. no off-by-one in the constant table). Iterates from
      // 1 to 16 inclusive so the 1D entry is covered too.
      for (let n = 1; n <= 16; n++) {
        const key = `${n}D` as keyof typeof CHOLESKY_SIZES;
        const value = (CHOLESKY_SIZES as Record<string, number>)[key];
        expect(value).toBeDefined();
        expect(value).toBe(choleskyPackedSize(n));
        expect(value).toBe((n * (n + 1)) / 2);
      }
      // Spot-check a few specific values to lock in the table.
      expect(CHOLESKY_SIZES['1D']).toBe(1);
      expect(CHOLESKY_SIZES['5D']).toBe(15);
      expect(CHOLESKY_SIZES['8D']).toBe(36);
      expect(CHOLESKY_SIZES['16D']).toBe(136);
    });
  });

  // v3.1 split-Cholesky index helpers — mirror Python
  // `luxar.gsplats.utils.trils.{diag_indices,offdiag_indices}`. The viewer
  // loader uses these to recombine `cholesky_factors_diag` +
  // `cholesky_factors_offdiag` into the packed (N, k) buffer the GPU expects.
  describe('choleskyDiagIndices / choleskyOffdiagIndices', () => {
    it.each([
      [1, [0], []],
      [2, [0, 2], [1]],
      [3, [0, 2, 5], [1, 3, 4]],
      [4, [0, 2, 5, 9], [1, 3, 4, 6, 7, 8]],
    ])('d=%d diag/offdiag positions match Python', (d, diag, off) => {
      expect(choleskyDiagIndices(d)).toEqual(diag);
      expect(choleskyOffdiagIndices(d)).toEqual(off);
    });

    it.each([1, 2, 3, 4, 5, 6, 8])('d=%d: diag ∪ offdiag partitions range(k) exactly', (d) => {
      const k = choleskyPackedSize(d);
      const combined = [...choleskyDiagIndices(d), ...choleskyOffdiagIndices(d)].sort(
        (a, b) => a - b
      );
      expect(combined).toEqual(Array.from({ length: k }, (_, i) => i));
      expect(choleskyDiagIndices(d)).toHaveLength(d);
      expect(choleskyOffdiagIndices(d)).toHaveLength(k - d);
    });

    // Guards the loader's interleave math: scattering split halves back through
    // these indices must reconstruct the original packed row (cf.
    // loadCholeskyRanges + the Python merge_tril round-trip).
    it.each([2, 3, 4, 5])('d=%d: scatter reconstructs the packed row', (d) => {
      const k = choleskyPackedSize(d);
      const packed = Array.from({ length: k }, (_, i) => i + 1); // [1..k]
      const diagIdx = choleskyDiagIndices(d);
      const offIdx = choleskyOffdiagIndices(d);
      const diag = diagIdx.map((i) => packed[i]);
      const offdiag = offIdx.map((i) => packed[i]);

      const out = new Array<number>(k).fill(0);
      diagIdx.forEach((idx, c) => (out[idx] = diag[c]));
      offIdx.forEach((idx, c) => (out[idx] = offdiag[c]));
      expect(out).toEqual(packed);
    });
  });
});
