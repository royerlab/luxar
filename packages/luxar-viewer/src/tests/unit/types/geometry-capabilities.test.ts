import { describe, it, expect } from 'vitest';
import { GEOMETRY_TYPES, type GeometryTypeName } from '../../../types/format-contract';
import {
  GEOMETRY_CAPABILITIES,
  isGeometryType,
  supportsLod,
  supportsPartition,
  isPooledGeometry,
  isDepthSortable,
} from '../../../types/geometry-capabilities';

/**
 * Invariants of the geometry capability matrix. Written to be parametric over
 * `GEOMETRY_TYPES` so a fourth geometry type is covered automatically — the
 * table itself is a `Record<GeometryTypeName, …>`, so the compiler already
 * forces the new entry to exist; these assert it behaves.
 */
/**
 * The generated `GEOMETRY_TYPES` is declared `readonly string[]`, so iterating it
 * yields `string` — which cannot index a `Record<GeometryTypeName, …>`. Narrow
 * once here rather than per loop; the first test below asserts every member really
 * is a `GeometryTypeName`, so the narrowing is checked rather than assumed.
 * (#1204 makes the generated arrays literal-typed, after which this is a no-op.)
 */
const TYPES = GEOMETRY_TYPES as readonly GeometryTypeName[];

describe('geometry capabilities', () => {
  it('classifies every contract geometry type, and nothing else', () => {
    for (const t of TYPES) {
      expect(isGeometryType(t), t).toBe(true);
      expect(Object.hasOwn(GEOMETRY_CAPABILITIES, t), t).toBe(true);
    }
    expect(Object.keys(GEOMETRY_CAPABILITIES).sort()).toEqual([...GEOMETRY_TYPES].sort());
  });

  it('rejects non-geometry node types and non-strings', () => {
    // `group` / `scene` are node types but not geometry — the distinction the
    // guard exists to make.
    for (const value of ['group', 'scene', 'volume', '', undefined, null, 3, {}]) {
      expect(isGeometryType(value), String(value)).toBe(false);
    }
  });

  it('does not resolve prototype-chain members as geometry types', () => {
    // A bare object-literal lookup would return a truthy non-descriptor for
    // these; the Set-backed guard must not.
    for (const key of ['constructor', 'toString', 'hasOwnProperty', '__proto__']) {
      expect(isGeometryType(key), key).toBe(false);
      expect(supportsLod(key), key).toBe(false);
      expect(isPooledGeometry(key), key).toBe(false);
    }
  });

  it('every predicate agrees with the table it reads', () => {
    for (const t of TYPES) {
      const caps = GEOMETRY_CAPABILITIES[t];
      expect(supportsLod(t), t).toBe(caps.lod);
      expect(supportsPartition(t), t).toBe(caps.partition);
      expect(isPooledGeometry(t), t).toBe(caps.pooled);
      expect(isDepthSortable(t), t).toBe(caps.depthSortable);
    }
  });

  it('reports no capability for a type outside the vocabulary', () => {
    // The whole point: a type the table has not classified is excluded from
    // every feature rather than defaulting in.
    for (const predicate of [supportsLod, supportsPartition, isPooledGeometry, isDepthSortable]) {
      expect(predicate('mesh')).toBe(false);
      expect(predicate(undefined)).toBe(false);
    }
  });

  it('each predicate reads its own capability key', () => {
    // Every incumbent is capable of everything, so the four predicates are
    // black-box indistinguishable: `supportsLod` wired to `partition` passes
    // every other test here. Flip one flag at a time to pin the wiring.
    const keys = ['lod', 'partition', 'pooled', 'depthSortable'] as const;
    const predicates = {
      lod: supportsLod,
      partition: supportsPartition,
      pooled: isPooledGeometry,
      depthSortable: isDepthSortable,
    };
    const original = { ...GEOMETRY_CAPABILITIES.points };
    try {
      for (const flipped of keys) {
        Object.assign(GEOMETRY_CAPABILITIES.points, original, { [flipped]: false });
        for (const key of keys) {
          // Only the predicate reading the flipped key may change.
          expect(predicates[key]('points'), `${key} after flipping ${flipped}`).toBe(
            key !== flipped
          );
        }
      }
    } finally {
      Object.assign(GEOMETRY_CAPABILITIES.points, original);
    }
    expect(GEOMETRY_CAPABILITIES.points).toEqual(original);
  });

  it('the current three types are uniformly capable', () => {
    // Guards the documented claim in the module header. If a future type is
    // added with `false` flags this stays true (it iterates the incumbents).
    for (const t of ['points', 'lines', 'gsplats'] as const) {
      expect(GEOMETRY_CAPABILITIES[t]).toEqual({
        lod: true,
        partition: true,
        pooled: true,
        depthSortable: true,
      });
    }
  });
});
