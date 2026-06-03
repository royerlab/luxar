/**
 * Unit tests for the points / lines / gsplats type guards in
 * `types/{points,lines,gsplats}.ts`.
 *
 * Each pair (isXMetadata + isXUserData) checks the respective
 * `type` / `nodeType` discriminator field. These are pure runtime
 * predicates — no mocks, no DOM.
 *
 * [types.md/O1][P10] Single source of truth for the three-geometry guard
 * test suite. Per-module test files (points.test.ts, lines.test.ts,
 * gsplats.test.ts) host only module-specific tests (e.g.
 * choleskyPackedSize, CHOLESKY_SIZES); they no longer duplicate the
 * isXMetadata / isXUserData / isValidLineType coverage that lives here.
 */

import { describe, it, expect } from 'vitest';
import { isPointsMetadata, isPointsUserData } from '../../../types/points';
import { isLinesMetadata, isLinesUserData, isValidLineType } from '../../../types/lines';
import { isGSplatsMetadata, isGSplatsUserData } from '../../../types/gsplats';

// Shared defensive-rejection table — every guard accepts `unknown`, so
// every guard MUST defensively reject primitives, arrays, null and
// undefined (the `typeof null === 'object'` footgun is the load-bearing
// branch). Previously this was duplicated as scattered it.each blocks
// across the per-module files.
const NON_OBJECT_INPUTS = [
  ['null', null],
  ['undefined', undefined],
  ['number', 42],
  ['string', 'gsplats'],
  ['boolean', true],
  ['array', [{ nodeType: 'gsplats' }]],
] as const;

describe('isPointsMetadata', () => {
  it('returns true for { type: "points" }', () => {
    expect(isPointsMetadata({ type: 'points' })).toBe(true);
  });

  it('returns true for a richly-populated points metadata payload', () => {
    const valid = {
      type: 'points',
      n_points: 1000,
      ndim: 3,
      max_radius: 1.5,
      has_colors: true,
      has_radii: true,
      ordering: 'morton',
    };
    expect(isPointsMetadata(valid)).toBe(true);
  });

  it('returns false for other geometry types', () => {
    expect(isPointsMetadata({ type: 'lines' })).toBe(false);
    expect(isPointsMetadata({ type: 'gsplats' })).toBe(false);
    expect(isPointsMetadata({ type: 'group' })).toBe(false);
  });

  it('returns false for an empty object (no type field)', () => {
    expect(isPointsMetadata({})).toBe(false);
  });

  it.each(NON_OBJECT_INPUTS)('rejects %s defensively', (_label, value) => {
    expect(isPointsMetadata(value)).toBe(false);
  });
});

describe('isPointsUserData', () => {
  it('returns true for { nodeType: "points" }', () => {
    expect(isPointsUserData({ nodeType: 'points' })).toBe(true);
  });

  it('returns true with optional visiblePointCount and attrs', () => {
    const withCount = {
      nodeType: 'points' as const,
      attrs: { n_points: 100 },
      visiblePointCount: 50,
    };
    expect(isPointsUserData(withCount)).toBe(true);
  });

  it('returns false for other node types', () => {
    expect(isPointsUserData({ nodeType: 'lines' })).toBe(false);
    expect(isPointsUserData({ nodeType: 'gsplats' })).toBe(false);
  });

  it('returns false for missing nodeType', () => {
    expect(isPointsUserData({ attrs: {} })).toBe(false);
  });

  it.each(NON_OBJECT_INPUTS)('rejects %s defensively', (_label, value) => {
    expect(isPointsUserData(value)).toBe(false);
  });
});

describe('isLinesMetadata', () => {
  it('returns true for { type: "lines" }', () => {
    expect(isLinesMetadata({ type: 'lines' })).toBe(true);
  });

  it('returns true for a richly-populated lines metadata payload', () => {
    const valid = {
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
    expect(isLinesMetadata(valid)).toBe(true);
  });

  it('returns false for other geometry types', () => {
    expect(isLinesMetadata({ type: 'points' })).toBe(false);
    expect(isLinesMetadata({ type: 'gsplats' })).toBe(false);
  });

  it('returns false for missing type', () => {
    expect(isLinesMetadata({ n_vertices: 100, n_segments: 50 })).toBe(false);
  });

  it.each(NON_OBJECT_INPUTS)('rejects %s defensively', (_label, value) => {
    expect(isLinesMetadata(value)).toBe(false);
  });
});

describe('isLinesUserData', () => {
  it('returns true for { nodeType: "lines" }', () => {
    expect(isLinesUserData({ nodeType: 'lines' })).toBe(true);
  });

  it('returns true for a fully-populated lines userData payload', () => {
    const valid = {
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
    expect(isLinesUserData(valid)).toBe(true);
  });

  it('returns false for other node types', () => {
    expect(isLinesUserData({ nodeType: 'points' })).toBe(false);
    expect(isLinesUserData({ nodeType: 'gsplats' })).toBe(false);
    expect(isLinesUserData({ nodeType: 'group' })).toBe(false);
  });

  it('returns false for missing nodeType', () => {
    expect(isLinesUserData({ loader: {}, attrs: {} })).toBe(false);
  });

  it.each(NON_OBJECT_INPUTS)('rejects %s defensively', (_label, value) => {
    expect(isLinesUserData(value)).toBe(false);
  });
});

describe('isGSplatsMetadata', () => {
  it('returns true for { type: "gsplats" }', () => {
    expect(isGSplatsMetadata({ type: 'gsplats' })).toBe(true);
  });

  it('returns true for a richly-populated gsplats metadata payload', () => {
    const valid = {
      type: 'gsplats',
      n_splats: 1000,
      ndim: 3,
      has_colors: true,
      chunk_size: 2000,
      amplitude_range: { min: 0.0, max: 10.0 },
      center_bounds: { min: [0, 0, 0], max: [100, 100, 100] },
      ordering: 'hilbert',
    };
    expect(isGSplatsMetadata(valid)).toBe(true);
  });

  it('returns false for other geometry types', () => {
    expect(isGSplatsMetadata({ type: 'points' })).toBe(false);
    expect(isGSplatsMetadata({ type: 'lines' })).toBe(false);
  });

  it('returns false for missing type', () => {
    expect(isGSplatsMetadata({ n_splats: 1000, ndim: 3 })).toBe(false);
  });

  it.each(NON_OBJECT_INPUTS)('rejects %s defensively', (_label, value) => {
    expect(isGSplatsMetadata(value)).toBe(false);
  });
});

describe('isGSplatsUserData', () => {
  it('returns true for { nodeType: "gsplats" }', () => {
    expect(isGSplatsUserData({ nodeType: 'gsplats' })).toBe(true);
  });

  it('returns true for a fully-populated gsplats userData payload', () => {
    const valid = {
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
    expect(isGSplatsUserData(valid)).toBe(true);
  });

  it('returns false for other node types', () => {
    expect(isGSplatsUserData({ nodeType: 'points' })).toBe(false);
    expect(isGSplatsUserData({ nodeType: 'lines' })).toBe(false);
    expect(isGSplatsUserData({ nodeType: 'group' })).toBe(false);
  });

  it('returns false for missing nodeType', () => {
    expect(isGSplatsUserData({ loader: {}, attrs: {} })).toBe(false);
  });

  it.each(NON_OBJECT_INPUTS)('rejects %s defensively', (_label, value) => {
    expect(isGSplatsUserData(value)).toBe(false);
  });
});

describe('isValidLineType', () => {
  it.each(['segments', 'polyline', 'loop', 'indexed'] as const)('accepts %s', (t) => {
    expect(isValidLineType(t)).toBe(true);
  });

  it.each(['strip', 'wireframe', '', 'lines', 'Segments', 'SEGMENTS'])(
    'rejects unknown / case-mismatched %s',
    (t) => {
      expect(isValidLineType(t)).toBe(false);
    }
  );

  it.each([
    ['null', null],
    ['undefined', undefined],
    ['number', 42],
    ['object', {}],
    ['array', []],
  ] as const)('rejects non-string %s', (_label, value) => {
    expect(isValidLineType(value as any)).toBe(false);
  });
});

describe('cross-cutting: each metadata guard rejects every other geometry', () => {
  const cases = [
    { name: 'points', guard: isPointsMetadata, attr: { type: 'points' } },
    { name: 'lines', guard: isLinesMetadata, attr: { type: 'lines' } },
    { name: 'gsplats', guard: isGSplatsMetadata, attr: { type: 'gsplats' } },
  ];

  it('every pair of (X-guard, Y-attrs) where X !== Y returns false', () => {
    for (const a of cases) {
      for (const b of cases) {
        const expected = a.name === b.name;
        expect(a.guard(b.attr)).toBe(expected);
      }
    }
  });
});
