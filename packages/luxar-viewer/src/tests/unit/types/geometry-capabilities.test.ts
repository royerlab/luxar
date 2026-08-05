import { describe, it, expect } from 'vitest';
import { GEOMETRY_TYPES } from '../../../types/format-contract';
import { POOLED_GEOMETRY_TYPES } from '../../../types/data-monitor-types';
import {
  GEOMETRY_CAPABILITIES,
  type GeometryCapabilities,
  isGeometryType,
  supportsLod,
  supportsPartition,
  isPooledGeometry,
  isDepthSortable,
  defaultBlendingMode,
} from '../../../types/geometry-capabilities';

/**
 * Invariants of the geometry capability matrix. Written to be parametric over
 * `GEOMETRY_TYPES` so a fourth geometry type is covered automatically — the
 * table itself is a `Record<GeometryTypeName, …>`, so the compiler already
 * forces the new entry to exist; these assert it behaves.
 */
describe('geometry capabilities', () => {
  it('classifies every contract geometry type, and nothing else', () => {
    for (const t of GEOMETRY_TYPES) {
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
    for (const t of GEOMETRY_TYPES) {
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

  it('the table record is frozen', () => {
    // Every predicate reads the exported object live, so replacing a row
    // would globally flip capabilities for the whole session. The record is
    // frozen (rows stay type-level readonly so the wiring test below can
    // flip flags through a deliberate cast).
    expect(Object.isFrozen(GEOMETRY_CAPABILITIES)).toBe(true);
  });

  it('agrees with POOLED_GEOMETRY_TYPES on the pooled subset', () => {
    // `POOLED_GEOMETRY_TYPES` (types/data-monitor-types.ts) keys the monitor's
    // per-type records at compile time; the table's `pooled` column answers
    // the same question at runtime. They must never drift apart.
    const pooledFromTable = GEOMETRY_TYPES.filter((t) => GEOMETRY_CAPABILITIES[t].pooled);
    expect([...POOLED_GEOMETRY_TYPES].sort()).toEqual([...pooledFromTable].sort());
  });

  it('each predicate reads its own capability key', () => {
    // Every incumbent is capable of everything, so the four predicates are
    // black-box indistinguishable: `supportsLod` wired to `partition` passes
    // every other test here. Flip one flag at a time to pin the wiring.
    // Deliberate mutation of the (type-level readonly) row, restored in
    // `finally` and asserted below — the record itself is frozen, so the row
    // cannot be swapped out from under the restore.
    const keys = ['lod', 'partition', 'pooled', 'depthSortable'] as const;
    const predicates = {
      lod: supportsLod,
      partition: supportsPartition,
      pooled: isPooledGeometry,
      depthSortable: isDepthSortable,
    };
    type MutableRow = { -readonly [K in keyof GeometryCapabilities]: boolean };
    const row = GEOMETRY_CAPABILITIES.points as MutableRow;
    const original = { ...row };
    try {
      for (const flipped of keys) {
        Object.assign(row, original, { [flipped]: false });
        for (const key of keys) {
          // Only the predicate reading the flipped key may change.
          expect(predicates[key]('points'), `${key} after flipping ${flipped}`).toBe(
            key !== flipped
          );
        }
      }
    } finally {
      Object.assign(row, original);
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

describe('defaultBlendingMode', () => {
  it('defaults mesh to opaque (the one shaded surface type — spec §6.3)', () => {
    expect(defaultBlendingMode('mesh')).toBe('opaque');
  });

  it('defaults the emissive primitives to additive', () => {
    for (const t of ['points', 'lines', 'gsplats'] as const) {
      expect(defaultBlendingMode(t), t).toBe('additive');
    }
  });

  it('falls back to additive for non-geometry node types', () => {
    for (const value of ['group', 'scene', undefined]) {
      expect(defaultBlendingMode(value), String(value)).toBe('additive');
    }
  });
});
