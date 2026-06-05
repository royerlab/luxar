/**
 * Unit tests for the pure extend-tolerance helpers extracted from
 * scene-loader.ts.
 */

import { describe, it, expect } from 'vitest';
import {
  EXTEND_TO_ALL_TOLERANCE,
  getOrComputeExtendedTolerance,
  hasOwnProperties,
  isSceneDimensions,
  validateExtendDims,
} from '../../../../../data/scene-loader/view-state/extend-tolerance';

describe('hasOwnProperties', () => {
  it('returns false for an empty object', () => {
    expect(hasOwnProperties({})).toBe(false);
  });

  it('returns true for any object with at least one own property', () => {
    expect(hasOwnProperties({ a: 1 })).toBe(true);
    expect(hasOwnProperties({ a: undefined })).toBe(true);
    expect(hasOwnProperties({ a: 1, b: 2 })).toBe(true);
  });

  it('returns false when only inherited properties exist (Object.create(parent))', () => {
    const parent = { inherited: 1 };
    const child = Object.create(parent);
    expect(hasOwnProperties(child)).toBe(false);
  });
});

describe('validateExtendDims', () => {
  const dims = [{ name: 'time' }, { name: 'channel' }, { name: 'z' }];

  it('passes silently for an empty extendDims list', () => {
    expect(() => validateExtendDims([], dims)).not.toThrow();
  });

  it('passes for all-valid names', () => {
    expect(() => validateExtendDims(['time', 'channel'], dims)).not.toThrow();
  });

  it('throws for an unknown name with the actionable message', () => {
    expect(() => validateExtendDims(['oops'], dims)).toThrow(
      /Invalid extend_to_all dimension\(s\): oops/
    );
  });

  it('throws when any one of the names is invalid (set semantics)', () => {
    expect(() => validateExtendDims(['time', 'oops', 'channel'], dims)).toThrow(/oops/);
  });

  it('lists all invalid names + the valid set in the error', () => {
    try {
      validateExtendDims(['oops', 'nope'], dims);
      throw new Error('should have thrown');
    } catch (e) {
      const msg = String(e);
      expect(msg).toContain('oops');
      expect(msg).toContain('nope');
      expect(msg).toContain('time');
      expect(msg).toContain('channel');
      expect(msg).toContain('z');
    }
  });

  it('skips dimensions whose name is undefined when building the valid set', () => {
    // A dimension without a name shouldn't be matchable.
    const dimsWithGaps = [{ name: 'time' }, {}, { name: 'z' }];
    expect(() => validateExtendDims(['time', 'z'], dimsWithGaps)).not.toThrow();
    expect(() => validateExtendDims([''], dimsWithGaps)).toThrow(/Invalid/);
  });
});

describe('getOrComputeExtendedTolerance', () => {
  const dims = [{ name: 'time' }, { name: 'channel' }, { name: 'z' }];

  it('replaces the named dimensions with EXTEND_TO_ALL_TOLERANCE', () => {
    const cache = new Map<string, number[]>();
    const result = getOrComputeExtendedTolerance([0.1, 0.1, 0.1], ['time'], dims, cache);
    expect(result).toEqual([EXTEND_TO_ALL_TOLERANCE, 0.1, 0.1]);
    expect(EXTEND_TO_ALL_TOLERANCE).toBe(1e10);
  });

  it('handles multiple extended dims in one call', () => {
    const result = getOrComputeExtendedTolerance([0.1, 0.1, 0.1], ['time', 'z'], dims, new Map());
    expect(result).toEqual([EXTEND_TO_ALL_TOLERANCE, 0.1, EXTEND_TO_ALL_TOLERANCE]);
  });

  it('returns the same cached array for repeat calls with the same set', () => {
    const cache = new Map<string, number[]>();
    const a = getOrComputeExtendedTolerance([0.1, 0.1, 0.1], ['time'], dims, cache);
    const b = getOrComputeExtendedTolerance([0.1, 0.1, 0.1], ['time'], dims, cache);
    expect(a).toBe(b); // same reference, not just equal
  });

  it('cache key is order-independent (sorted set semantics)', () => {
    const cache = new Map<string, number[]>();
    const a = getOrComputeExtendedTolerance([0.1, 0.1, 0.1], ['time', 'z'], dims, cache);
    const b = getOrComputeExtendedTolerance([0.1, 0.1, 0.1], ['z', 'time'], dims, cache);
    expect(a).toBe(b);
  });

  it('throws via the validate path when an extendDim is unknown', () => {
    expect(() => getOrComputeExtendedTolerance([0.1, 0.1, 0.1], ['oops'], dims, new Map())).toThrow(
      /Invalid extend_to_all/
    );
  });

  it('silently ignores extendDims whose index is out of range for tolerance length', () => {
    // dim 'z' is at index 2; tolerance has length 2 → out of range.
    // Helper writes nothing for it (defensive, not throw).
    const result = getOrComputeExtendedTolerance([0.1, 0.1], ['z'], dims, new Map());
    expect(result).toEqual([0.1, 0.1]);
  });

  it('returns a fresh copy of baseTolerance, not the input', () => {
    const base = [0.1, 0.1, 0.1];
    const result = getOrComputeExtendedTolerance(base, [], dims, new Map());
    expect(result).not.toBe(base);
    expect(result).toEqual(base);
  });

  it('passes through a NaN baseTolerance value at a non-extended index', () => {
    // dim 'time' (index 0) gets extended; index 1 holds NaN and must
    // survive the copy untouched (no normalization / coercion).
    const result = getOrComputeExtendedTolerance([0.1, NaN, 0.1], ['time'], dims, new Map());
    expect(result[0]).toBe(EXTEND_TO_ALL_TOLERANCE);
    expect(Number.isNaN(result[1])).toBe(true);
    expect(result[2]).toBe(0.1);
  });

  it('passes through an Infinity baseTolerance value at a non-extended index', () => {
    const result = getOrComputeExtendedTolerance([0.1, Infinity, 0.1], ['time'], dims, new Map());
    expect(result[0]).toBe(EXTEND_TO_ALL_TOLERANCE);
    expect(result[1]).toBe(Infinity);
    expect(result[2]).toBe(0.1);
  });

  it('preserves a negative baseTolerance value (no clamping)', () => {
    const result = getOrComputeExtendedTolerance([-5, 0.1, 0.1], ['time'], dims, new Map());
    // Index 0 is the extended dim, so it becomes EXTEND_TO_ALL regardless.
    expect(result[0]).toBe(EXTEND_TO_ALL_TOLERANCE);
    // A negative at a NON-extended index is preserved as-is.
    const noExtend = getOrComputeExtendedTolerance([-5, 0.1, 0.1], [], dims, new Map());
    expect(noExtend[0]).toBe(-5);
  });
});

describe('isSceneDimensions', () => {
  it('returns true for an object with a dimensions array', () => {
    expect(isSceneDimensions({ dimensions: [] })).toBe(true);
    expect(isSceneDimensions({ dimensions: [{ name: 'time' }] })).toBe(true);
  });

  it('returns false for null / undefined / primitives', () => {
    expect(isSceneDimensions(null)).toBe(false);
    expect(isSceneDimensions(undefined)).toBe(false);
    expect(isSceneDimensions(42)).toBe(false);
    expect(isSceneDimensions('dimensions')).toBe(false);
  });

  it('returns false when dimensions is missing or not an array', () => {
    expect(isSceneDimensions({})).toBe(false);
    expect(isSceneDimensions({ dimensions: 'not-an-array' })).toBe(false);
    expect(isSceneDimensions({ dimensions: { foo: 1 } })).toBe(false);
  });
});
