/**
 * Unit tests for the points / lines / gsplats type guards in
 * `types/{points,lines,gsplats}.ts`.
 *
 * Each pair (isXMetadata + isXUserData) checks the respective
 * `type` / `nodeType` discriminator field. These are pure runtime
 * predicates — no mocks, no DOM.
 */

import { describe, it, expect } from 'vitest';
import { isPointsMetadata, isPointsUserData } from '../../../types/points';
import { isLinesMetadata, isLinesUserData, isValidLineType } from '../../../types/lines';
import { isGSplatsMetadata, isGSplatsUserData } from '../../../types/gsplats';

describe('isPointsMetadata', () => {
  it('returns true for { type: "points" }', () => {
    expect(isPointsMetadata({ type: 'points' })).toBe(true);
  });

  it('returns false for other geometry types', () => {
    expect(isPointsMetadata({ type: 'lines' })).toBe(false);
    expect(isPointsMetadata({ type: 'gsplats' })).toBe(false);
    expect(isPointsMetadata({ type: 'group' })).toBe(false);
  });

  it('returns false defensively for null / undefined / primitives', () => {
    expect(isPointsMetadata(null)).toBe(false);
    expect(isPointsMetadata(undefined)).toBe(false);
    expect(isPointsMetadata(42)).toBe(false);
    expect(isPointsMetadata('points')).toBe(false);
  });

  it('returns false for an empty object (no type field)', () => {
    expect(isPointsMetadata({})).toBe(false);
  });
});

describe('isPointsUserData', () => {
  it('returns true for { nodeType: "points" }', () => {
    expect(isPointsUserData({ nodeType: 'points' })).toBe(true);
  });

  it('returns false for other node types', () => {
    expect(isPointsUserData({ nodeType: 'lines' })).toBe(false);
    expect(isPointsUserData({ nodeType: 'gsplats' })).toBe(false);
  });

  it('returns false defensively for null / undefined', () => {
    expect(isPointsUserData(null)).toBe(false);
    expect(isPointsUserData(undefined)).toBe(false);
  });
});

describe('isLinesMetadata', () => {
  it('returns true for { type: "lines" }', () => {
    expect(isLinesMetadata({ type: 'lines' })).toBe(true);
  });

  it('returns false for other geometry types', () => {
    expect(isLinesMetadata({ type: 'points' })).toBe(false);
    expect(isLinesMetadata({ type: 'gsplats' })).toBe(false);
  });

  it('returns false defensively for null / undefined / primitives', () => {
    expect(isLinesMetadata(null)).toBe(false);
    expect(isLinesMetadata(undefined)).toBe(false);
    expect(isLinesMetadata('lines')).toBe(false);
    expect(isLinesMetadata(42)).toBe(false);
  });
});

describe('isLinesUserData', () => {
  it('returns true for { nodeType: "lines" }', () => {
    expect(isLinesUserData({ nodeType: 'lines' })).toBe(true);
  });

  it('returns false for other node types', () => {
    expect(isLinesUserData({ nodeType: 'points' })).toBe(false);
    expect(isLinesUserData({ nodeType: 'gsplats' })).toBe(false);
  });

  it('returns false defensively for null / undefined', () => {
    expect(isLinesUserData(null)).toBe(false);
    expect(isLinesUserData(undefined)).toBe(false);
  });
});

describe('isGSplatsMetadata', () => {
  it('returns true for { type: "gsplats" }', () => {
    expect(isGSplatsMetadata({ type: 'gsplats' })).toBe(true);
  });

  it('returns false for other geometry types', () => {
    expect(isGSplatsMetadata({ type: 'points' })).toBe(false);
    expect(isGSplatsMetadata({ type: 'lines' })).toBe(false);
  });

  it('returns false defensively for null / undefined / primitives', () => {
    expect(isGSplatsMetadata(null)).toBe(false);
    expect(isGSplatsMetadata(undefined)).toBe(false);
    expect(isGSplatsMetadata('gsplats')).toBe(false);
  });
});

describe('isGSplatsUserData', () => {
  it('returns true for { nodeType: "gsplats" }', () => {
    expect(isGSplatsUserData({ nodeType: 'gsplats' })).toBe(true);
  });

  it('returns false for other node types', () => {
    expect(isGSplatsUserData({ nodeType: 'points' })).toBe(false);
    expect(isGSplatsUserData({ nodeType: 'lines' })).toBe(false);
  });

  it('returns false defensively for null / undefined', () => {
    expect(isGSplatsUserData(null)).toBe(false);
    expect(isGSplatsUserData(undefined)).toBe(false);
  });
});

describe('isValidLineType', () => {
  it('accepts the four valid line types', () => {
    expect(isValidLineType('segments')).toBe(true);
    expect(isValidLineType('polyline')).toBe(true);
    expect(isValidLineType('loop')).toBe(true);
    expect(isValidLineType('indexed')).toBe(true);
  });

  it('rejects unknown line types', () => {
    expect(isValidLineType('strip')).toBe(false);
    expect(isValidLineType('wireframe')).toBe(false);
    expect(isValidLineType('')).toBe(false);
  });

  it('rejects non-string inputs', () => {
    expect(isValidLineType(null)).toBe(false);
    expect(isValidLineType(undefined)).toBe(false);
    expect(isValidLineType(42)).toBe(false);
    expect(isValidLineType({})).toBe(false);
    expect(isValidLineType([])).toBe(false);
  });

  it('is case-sensitive (Segments vs segments)', () => {
    expect(isValidLineType('Segments')).toBe(false);
    expect(isValidLineType('SEGMENTS')).toBe(false);
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
