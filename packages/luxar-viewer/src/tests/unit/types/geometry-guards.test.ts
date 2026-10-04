/**
 * Unit tests for the points / lines / gsplats / mesh userData guards in
 * `types/{points,lines,gsplats,mesh}.ts`.
 *
 * Each isXUserData guard checks the `nodeType` discriminator a
 * scene-graph object carries. These are pure runtime predicates — no
 * mocks, no DOM.
 *
 * [types.md/O1][P10] Single source of truth for the three-geometry guard
 * test suite. Per-module test files (points.test.ts, lines.test.ts,
 * gsplats.test.ts) host only module-specific tests (e.g.
 * choleskyPackedSize, CHOLESKY_SIZES); they no longer duplicate the
 * isXUserData coverage that lives here.
 */

import { describe, it, expect } from 'vitest';
import { isPointsUserData } from '../../../types/points';
import { isLinesUserData } from '../../../types/lines';
import { isGSplatsUserData } from '../../../types/gsplats';
import { isMeshUserData } from '../../../types/mesh';

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

describe('cross-cutting: each userData guard accepts its own node type only', () => {
  const cases = [
    { name: 'points', guard: isPointsUserData },
    { name: 'lines', guard: isLinesUserData },
    { name: 'gsplats', guard: isGSplatsUserData },
    { name: 'mesh', guard: isMeshUserData },
  ];

  it('every pair of (X-guard, Y-userData) where X !== Y returns false', () => {
    for (const a of cases) {
      for (const b of cases) {
        expect(a.guard({ nodeType: b.name })).toBe(a.name === b.name);
      }
    }
  });

  it.each(NON_OBJECT_INPUTS)('isMeshUserData rejects %s defensively', (_label, value) => {
    expect(isMeshUserData(value)).toBe(false);
  });
});
