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
import { choleskyPackedSize, CHOLESKY_SIZES } from '../../../types/gsplats';

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

    it('MED-44: covers 5D through 16D (the WASM dimension ceiling)', () => {
      // Regression: the constant previously stopped at 4D even though
      // LoadedGSplatsData and choleskyPackedSize support arbitrary ndim.
      // A 5D+ caller indexing into the constant would have hit undefined
      // and produced silently wrong stride math. The constant is now
      // populated up to 16D (the WASM 16-dim limit per CLAUDE.md), and
      // every entry must match `choleskyPackedSize(n) = n*(n+1)/2`.
      for (let n = 2; n <= 16; n++) {
        const key = `${n}D` as keyof typeof CHOLESKY_SIZES;
        const value = (CHOLESKY_SIZES as Record<string, number>)[key];
        expect(value).toBeDefined();
        expect(value).toBe(choleskyPackedSize(n));
        expect(value).toBe((n * (n + 1)) / 2);
      }
      // Spot-check a few specific values to lock in the table.
      expect(CHOLESKY_SIZES['5D']).toBe(15);
      expect(CHOLESKY_SIZES['8D']).toBe(36);
      expect(CHOLESKY_SIZES['16D']).toBe(136);
    });
  });
});
