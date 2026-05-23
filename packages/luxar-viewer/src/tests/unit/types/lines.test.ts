import { describe, it, expect } from 'vitest';
import { isLinesMetadata, isLinesUserData, isValidLineType } from '../../../types/lines';

describe('Lines Types', () => {
  describe('isLinesMetadata', () => {
    it('should return true for valid lines metadata', () => {
      const validMetadata = {
        type: 'lines',
        n_vertices: 100,
        n_segments: 50,
        ndim: 3,
        original_line_type: 'polyline',
        max_width: 0.5,
        has_colors: true,
        has_sharpness: true,
        ordering: 'morton',
      };

      expect(isLinesMetadata(validMetadata)).toBe(true);
    });

    it('should return false for points metadata', () => {
      const pointsMetadata = {
        type: 'points',
        n_points: 1000,
        ndim: 3,
      };

      expect(isLinesMetadata(pointsMetadata)).toBe(false);
    });

    it('should return false for missing type', () => {
      const invalidMetadata = {
        n_vertices: 100,
        n_segments: 50,
      };

      expect(isLinesMetadata(invalidMetadata)).toBe(false);
    });

    it('should return false for null/undefined', () => {
      expect(isLinesMetadata(null)).toBe(false);
      expect(isLinesMetadata(undefined)).toBe(false);
    });

    it('should return false for non-object', () => {
      expect(isLinesMetadata('lines')).toBe(false);
      expect(isLinesMetadata(123)).toBe(false);
      expect(isLinesMetadata([])).toBe(false);
    });
  });

  describe('isLinesUserData', () => {
    it('should return true for valid lines userData', () => {
      // isLinesUserData only checks for nodeType === 'lines'
      const validUserData = {
        nodeType: 'lines',
        loader: {}, // Actual loader would be a LinesDataLoader instance
        attrs: {
          type: 'lines',
          n_vertices: 100,
          n_segments: 50,
          ndim: 3,
          original_line_type: 'segments',
          max_width: 0.1,
          has_colors: false,
          has_sharpness: false,
          ordering: 'none',
        },
      };

      expect(isLinesUserData(validUserData)).toBe(true);
    });

    it('should return false for points userData', () => {
      const pointsUserData = {
        nodeType: 'points',
        loader: {},
        attrs: {},
      };

      expect(isLinesUserData(pointsUserData)).toBe(false);
    });

    it('should return false for missing nodeType', () => {
      const invalidUserData = {
        loader: {},
        attrs: {},
      };

      expect(isLinesUserData(invalidUserData)).toBe(false);
    });

    it('should return false for wrong nodeType', () => {
      const invalidUserData = {
        nodeType: 'group',
        children: [],
      };

      expect(isLinesUserData(invalidUserData)).toBe(false);
    });

    // [types.md/W5][P2] Audit found `isLinesUserData` tests never covered
    // primitive / null / array inputs. Symmetric with the W4 fix in
    // gsplats.test.ts — the guard accepts `unknown` and must reject
    // non-object inputs defensively.
    it.each([
      ['null', null],
      ['undefined', undefined],
      ['number', 42],
      ['string', 'lines'],
      ['boolean', true],
      ['array', [{ nodeType: 'lines' }]],
    ] as const)('rejects %s defensively', (_label, value) => {
      expect(isLinesUserData(value)).toBe(false);
    });
  });

  describe('isValidLineType', () => {
    it('should return true for valid line types', () => {
      expect(isValidLineType('segments')).toBe(true);
      expect(isValidLineType('polyline')).toBe(true);
      expect(isValidLineType('loop')).toBe(true);
      expect(isValidLineType('indexed')).toBe(true);
    });

    it('should return false for invalid line types', () => {
      expect(isValidLineType('lines')).toBe(false);
      expect(isValidLineType('strip')).toBe(false);
      expect(isValidLineType('')).toBe(false);
      expect(isValidLineType(123 as any)).toBe(false);
      expect(isValidLineType(null as any)).toBe(false);
    });
  });
});

// [types.md/W2][P1][P2][EXCLUDED-CATEGORY: types] Removed `Lines Type
// Definitions` describe block (~107 lines): four `it(...)` blocks that
// constructed typed literal objects and asserted the literal's own fields
// equaled the values just written into them. Same pattern as gsplats W1 —
// "if it compiles, the types are correct" — these tests exercised zero
// runtime branches. TypeScript's own type-checker (`pnpm typecheck`) is
// the authoritative gate for compile-time correctness.
