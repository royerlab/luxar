import { describe, it, expect } from 'vitest';
import {
  isGSplatsMetadata,
  isGSplatsUserData,
  choleskyPackedSize,
  CHOLESKY_SIZES,
} from '../../../types/gsplats';

describe('GSplats Types', () => {
  describe('isGSplatsMetadata', () => {
    it('should return true for valid gsplats metadata', () => {
      const validMetadata = {
        type: 'gsplats',
        n_splats: 1000,
        ndim: 3,
        has_colors: true,
        chunk_size: 2000,
        amplitude_range: { min: 0.0, max: 10.0 },
        center_bounds: { min: [0, 0, 0], max: [100, 100, 100] },
        ordering: 'hilbert',
      };

      expect(isGSplatsMetadata(validMetadata)).toBe(true);
    });

    it('should return false for points metadata', () => {
      const pointsMetadata = {
        type: 'points',
        n_points: 1000,
        ndim: 3,
      };

      expect(isGSplatsMetadata(pointsMetadata)).toBe(false);
    });

    it('should return false for lines metadata', () => {
      const linesMetadata = {
        type: 'lines',
        n_vertices: 100,
        n_segments: 50,
      };

      expect(isGSplatsMetadata(linesMetadata)).toBe(false);
    });

    it('should return false for missing type', () => {
      const invalidMetadata = {
        n_splats: 1000,
        ndim: 3,
      };

      expect(isGSplatsMetadata(invalidMetadata)).toBe(false);
    });

    it('should return false for null/undefined', () => {
      expect(isGSplatsMetadata(null)).toBe(false);
      expect(isGSplatsMetadata(undefined)).toBe(false);
    });

    it('should return false for non-object', () => {
      expect(isGSplatsMetadata('gsplats')).toBe(false);
      expect(isGSplatsMetadata(123)).toBe(false);
      expect(isGSplatsMetadata([])).toBe(false);
    });
  });

  describe('isGSplatsUserData', () => {
    it('should return true for valid gsplats userData', () => {
      const validUserData = {
        nodeType: 'gsplats',
        loader: {}, // Actual loader would be a GSplatsDataLoader instance
        attrs: {
          type: 'gsplats',
          n_splats: 1000,
          ndim: 3,
          has_colors: true,
          chunk_size: 2000,
          amplitude_range: { min: 0.0, max: 10.0 },
          center_bounds: { min: [0, 0, 0], max: [100, 100, 100] },
          ordering: 'morton',
        },
      };

      expect(isGSplatsUserData(validUserData)).toBe(true);
    });

    it('should return false for points userData', () => {
      const pointsUserData = {
        nodeType: 'points',
        loader: {},
        attrs: {},
      };

      expect(isGSplatsUserData(pointsUserData)).toBe(false);
    });

    it('should return false for lines userData', () => {
      const linesUserData = {
        nodeType: 'lines',
        loader: {},
        attrs: {},
      };

      expect(isGSplatsUserData(linesUserData)).toBe(false);
    });

    it('should return false for missing nodeType', () => {
      const invalidUserData = {
        loader: {},
        attrs: {},
      };

      expect(isGSplatsUserData(invalidUserData)).toBe(false);
    });

    it('should return false for wrong nodeType', () => {
      const invalidUserData = {
        nodeType: 'group',
        children: [],
      };

      expect(isGSplatsUserData(invalidUserData)).toBe(false);
    });

    // [types.md/W4][P2] Audit found `isGSplatsUserData` tests never checked
    // the guard against primitives — only objects with wrong/missing fields.
    // The source predicate accepts `unknown` so non-object inputs must be
    // rejected defensively (typeof null === 'object' footgun is the
    // load-bearing branch).
    it.each([
      ['null', null],
      ['undefined', undefined],
      ['number', 42],
      ['string', 'gsplats'],
      ['boolean', true],
      ['array', [{ nodeType: 'gsplats' }]],
    ] as const)('rejects %s defensively', (_label, value) => {
      expect(isGSplatsUserData(value)).toBe(false);
    });
  });

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

// [types.md/W1][P1][P2][EXCLUDED-CATEGORY: types] Removed `GSplats Type
// Definitions` describe block (~135 lines): ten `it(...)` blocks that
// constructed typed literal objects and asserted the literal's own fields
// equaled the values just written into them. The original code openly
// admitted "if it compiles, the types are correct" — those tests
// exercised zero runtime branches and killed zero mutants. TypeScript's
// own type-checker (run as `pnpm typecheck`) is the authoritative gate
// for compile-time correctness; these vitest tests added measurement
// noise without any kill-rate signal. The runtime guards
// `isGSplatsMetadata` / `isGSplatsUserData` retain dedicated, value-
// asserting coverage above and in `geometry-guards.test.ts`.
