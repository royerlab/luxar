/**
 * Unit tests for the type guards in `types/zarr.ts`.
 *
 * Each guard is a pure runtime predicate — perfect for mock-free
 * coverage. The tests check the positive case (returns true and
 * narrows the type), the negative cases (returns false), and the
 * defensive null/undefined paths.
 */

import { describe, it, expect } from 'vitest';
import {
  hasContentsMethod,
  hasNdTransform,
  hasSceneDimensions,
  hasTransform,
  isPermutation,
  isPointsNode,
  type NdTransformEntry,
  type ZarrNodeAttrs,
  type ZarrSceneAttrs,
} from '../../../types/zarr';

describe('isPermutation', () => {
  it('returns true for an entry with a `permutation` field', () => {
    const entry: NdTransformEntry = { permutation: [0, 1, 2] };
    expect(isPermutation(entry)).toBe(true);
  });

  it('returns false for an affine entry (scale + offset)', () => {
    const entry: NdTransformEntry = { scale: 1.0, offset: 0.0 };
    expect(isPermutation(entry)).toBe(false);
  });

  it('safely returns false for null and primitives', () => {
    // HIGH-4 regression: previous `'permutation' in entry` form threw
    // TypeError for null/primitives because `in` requires a non-null
    // object. Zarr metadata is JSON-parsed at runtime so malformed
    // nd_transform payloads (e.g. `null` for an entry, or a misplaced
    // string) used to crash the loader instead of being rejected.
    // Cast to the widened type the guard now declares.
    expect(isPermutation(null)).toBe(false);
    expect(isPermutation(undefined)).toBe(false);
    expect(isPermutation(42 as unknown as NdTransformEntry)).toBe(false);
    expect(isPermutation('permutation' as unknown as NdTransformEntry)).toBe(false);
    expect(isPermutation(true as unknown as NdTransformEntry)).toBe(false);
  });
});

describe('hasContentsMethod', () => {
  it('returns true when the value has a `contents` function property', () => {
    const store = { contents: () => Promise.resolve([]) };
    expect(hasContentsMethod(store)).toBe(true);
  });

  it('returns false when `contents` is a non-function value', () => {
    expect(hasContentsMethod({ contents: 'not a function' })).toBe(false);
    expect(hasContentsMethod({ contents: 42 })).toBe(false);
    expect(hasContentsMethod({ contents: {} })).toBe(false);
  });

  it('returns false when the property is missing', () => {
    expect(hasContentsMethod({})).toBe(false);
    expect(hasContentsMethod({ other: () => null })).toBe(false);
  });

  it('returns false defensively for null / undefined / primitives', () => {
    expect(hasContentsMethod(null)).toBe(false);
    expect(hasContentsMethod(undefined)).toBe(false);
    expect(hasContentsMethod(42)).toBe(false);
    expect(hasContentsMethod('string')).toBe(false);
  });
});

describe('hasTransform', () => {
  function makeAttrs(transform?: unknown): ZarrNodeAttrs {
    return { type: 'points', transform } as unknown as ZarrNodeAttrs;
  }

  it('returns true when transform is a 16-length array', () => {
    const t = Array.from({ length: 16 }, (_, i) => i);
    expect(hasTransform(makeAttrs(t))).toBe(true);
  });

  it('returns false when transform is undefined', () => {
    expect(hasTransform(makeAttrs(undefined))).toBe(false);
    expect(hasTransform({ type: 'points' } as ZarrNodeAttrs)).toBe(false);
  });

  it('returns false when transform is the wrong length', () => {
    expect(hasTransform(makeAttrs([]))).toBe(false);
    expect(hasTransform(makeAttrs([1, 2, 3]))).toBe(false);
    expect(hasTransform(makeAttrs(new Array(15).fill(0)))).toBe(false);
    expect(hasTransform(makeAttrs(new Array(17).fill(0)))).toBe(false);
  });

  it('returns false when transform is not an array', () => {
    expect(hasTransform(makeAttrs('not-an-array'))).toBe(false);
    expect(hasTransform(makeAttrs({}))).toBe(false);
    expect(hasTransform(makeAttrs(42))).toBe(false);
  });
});

describe('hasNdTransform', () => {
  function makeAttrs(nd_transform?: unknown): ZarrNodeAttrs {
    return { type: 'points', nd_transform } as unknown as ZarrNodeAttrs;
  }

  it('returns true for a non-array object', () => {
    expect(hasNdTransform(makeAttrs({ time: { scale: 1, offset: 0 } }))).toBe(true);
    expect(hasNdTransform(makeAttrs({}))).toBe(true);
  });

  it('returns false when nd_transform is undefined', () => {
    expect(hasNdTransform(makeAttrs(undefined))).toBe(false);
  });

  it('returns false when nd_transform is an array (not a per-dimension map)', () => {
    expect(hasNdTransform(makeAttrs([]))).toBe(false);
    expect(hasNdTransform(makeAttrs([1, 2, 3]))).toBe(false);
  });

  it('returns false when nd_transform is null (typeof null === "object" footgun fixed)', () => {
    // types.md G4 / OOS fix: types/zarr.ts:hasNdTransform was previously
    // accepting `null` because `typeof null === 'object'`, `null !== undefined`,
    // and `!Array.isArray(null) === true`. The source now also checks
    // `attrs.nd_transform !== null`, so the guard correctly rejects null.
    // Downstream code that iterates the map's keys would have crashed.
    const CURRENT_BEHAVIOR_ACCEPTS_NULL = false;
    if (CURRENT_BEHAVIOR_ACCEPTS_NULL) {
      expect(hasNdTransform(makeAttrs(null))).toBe(true);
    } else {
      expect(hasNdTransform(makeAttrs(null))).toBe(false);
    }
  });
});

describe('isPointsNode', () => {
  it('returns true for type === "points"', () => {
    expect(isPointsNode({ type: 'points' } as ZarrNodeAttrs)).toBe(true);
  });

  it('returns false for other node types', () => {
    expect(isPointsNode({ type: 'lines' } as ZarrNodeAttrs)).toBe(false);
    expect(isPointsNode({ type: 'gsplats' } as ZarrNodeAttrs)).toBe(false);
    expect(isPointsNode({ type: 'group' } as ZarrNodeAttrs)).toBe(false);
  });

  it('returns false when type is undefined or attrs is empty', () => {
    // types.md W7 fix: previous version didn't exercise `attrs.type === undefined`.
    // The field is optional in the type (see zarr.ts:245); a missing type
    // must logically return false. A regression that defaulted to "points"
    // would have slipped through.
    expect(isPointsNode({} as ZarrNodeAttrs)).toBe(false);
    expect(isPointsNode({ type: undefined } as unknown as ZarrNodeAttrs)).toBe(false);
  });
});

describe('hasSceneDimensions', () => {
  it('returns true when scene_dimensions has a dimensions field', () => {
    const attrs = {
      scene_dimensions: { dimensions: [{ name: 't', unit: 's' }] },
    } as unknown as ZarrSceneAttrs;
    expect(hasSceneDimensions(attrs)).toBe(true);
  });

  it('returns false when scene_dimensions is undefined', () => {
    expect(hasSceneDimensions({} as ZarrSceneAttrs)).toBe(false);
  });

  it('returns false when scene_dimensions has no dimensions field', () => {
    const attrs = {
      scene_dimensions: {},
    } as unknown as ZarrSceneAttrs;
    expect(hasSceneDimensions(attrs)).toBe(false);
  });
});
