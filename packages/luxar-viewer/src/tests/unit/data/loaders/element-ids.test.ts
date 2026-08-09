/**
 * Unit tests for the shared slot → on-disk element-ID composer
 * (`data/loaders/element-ids.ts`).
 *
 * `PickResult.elementId` used to be the storage slot in the VISIBLE buffer,
 * which diverges from the on-disk index two independent ways: spatial range
 * loading (only visible ranges are concatenated) and visibility compaction
 * (zero effective radius for Points — issue #1421; hidden-dim attenuation for
 * GSplats — issue #1423). This map is what lets picking key the per-element
 * label CSR by the on-disk index instead.
 *
 * The Points-shaped cases were moved here verbatim from
 * `tests/unit/data/points/projection.test.ts` when the composer was extracted
 * from `buildPointElementIds`; the GSplats-shaped cases are new.
 */

import { describe, it, expect } from 'vitest';
import { buildElementIdMap, type ElementIdRange } from '../../../../data/loaders/element-ids';
import { Modules } from '../../../../utils/log';

const MOD = Modules.SPATIAL_INDEX_LOADER;

describe('buildElementIdMap', () => {
  it('returns undefined for the identity case (one range from 0, no compaction)', () => {
    expect(
      buildElementIdMap([{ start: 0, end: 5 }] as ElementIdRange[], null, 5, MOD)
    ).toBeUndefined();
  });

  it('returns undefined for an empty visible set', () => {
    expect(buildElementIdMap([] as ElementIdRange[], null, 0, MOD)).toBeUndefined();
  });

  it('maps the multi-range shape from the issue (chunks 1–2 of a 6000-element node)', () => {
    // The issue's figure, [(2048, 4096), (4096, 6000)] → 3952 visible elements.
    // Adjacent, so a real query coalesces these into one `[2048, 6000)` — kept
    // here to exercise the cursor's range-boundary step.
    const ids = buildElementIdMap(
      [
        { start: 2048, end: 4096 },
        { start: 4096, end: 6000 },
      ] as ElementIdRange[],
      null,
      3952,
      MOD
    );
    expect(ids).toBeInstanceOf(Uint32Array);
    expect(ids!.length).toBe(3952);
    expect(ids![0]).toBe(2048);
    expect(ids![2047]).toBe(4095);
    expect(ids![2048]).toBe(4096);
    expect(ids![3951]).toBe(5999);
  });

  it('offsets a single range that does not start at 0', () => {
    const ids = buildElementIdMap([{ start: 100, end: 103 }] as ElementIdRange[], null, 3, MOD);
    expect(Array.from(ids!)).toEqual([100, 101, 102]);
  });

  it('maps kept concat indices across a range boundary (compaction + range loading)', () => {
    // Kept concat index 0 → on-disk 2048; kept 2049 falls in the SECOND range
    // (first range covers concat [0, 2048)) → 4096 + (2049 - 2048) = 4097.
    const ids = buildElementIdMap(
      [
        { start: 2048, end: 4096 },
        { start: 4096, end: 6000 },
      ] as ElementIdRange[],
      [0, 2049],
      2,
      MOD
    );
    expect(Array.from(ids!)).toEqual([2048, 4097]);
  });

  it('emits a map even for a single range from 0 once compaction removed elements', () => {
    // The identity fast path is gated on "nothing was compacted out" too.
    const ids = buildElementIdMap([{ start: 0, end: 4 }] as ElementIdRange[], [1, 3], 2, MOD);
    expect(Array.from(ids!)).toEqual([1, 3]);
  });

  it('returns undefined (no throw) when the ranges are too short for the count', () => {
    expect(() =>
      buildElementIdMap([{ start: 0, end: 2 }] as ElementIdRange[], null, 5, MOD)
    ).not.toThrow();
    expect(
      buildElementIdMap([{ start: 0, end: 2 }] as ElementIdRange[], null, 5, MOD)
    ).toBeUndefined();
  });

  it('returns undefined when the KEPT LIST length disagrees with the count', () => {
    // The other arm of the same guard: with a kept list present the expected
    // count is its length, not the ranges' total.
    expect(() =>
      buildElementIdMap([{ start: 0, end: 10 }] as ElementIdRange[], [1, 3, 5], 2, MOD)
    ).not.toThrow();
    expect(
      buildElementIdMap([{ start: 0, end: 10 }] as ElementIdRange[], [1, 3, 5], 2, MOD)
    ).toBeUndefined();
  });

  it('returns undefined (no throw) for a NON-ASCENDING kept list', () => {
    // The forward cursor cannot rewind, so a descending (or repeated) index
    // would subtract a prefix already past it and wrap negative in the
    // Uint32Array. Fail closed instead.
    const ranges = [{ start: 100, end: 110 }] as ElementIdRange[];
    expect(() => buildElementIdMap(ranges, [5, 2, 7], 3, MOD)).not.toThrow();
    expect(buildElementIdMap(ranges, [5, 2, 7], 3, MOD)).toBeUndefined();
    // Repeats are equally rejected (strict ascent, not merely non-decreasing).
    expect(buildElementIdMap(ranges, [2, 2, 7], 3, MOD)).toBeUndefined();
  });

  it('returns undefined for an all-zeros kept list (STALE prebuilt WASM shape)', () => {
    // A gitignored `public/wasm/` build whose glue predates the kernel's
    // source-index out-param silently drops the argument, so the caller's
    // buffer comes back all zeros. It has the RIGHT LENGTH, so the count guard
    // passes; without the strict-ascent check every slot would resolve to
    // `ranges[0].start` and every hover would show the same label, silently.
    const ids = buildElementIdMap(
      [{ start: 2048, end: 4096 }] as ElementIdRange[],
      new Uint32Array(4), // [0, 0, 0, 0]
      4,
      Modules.GSPLATS_SPATIAL_INDEX_LOADER
    );
    expect(ids).toBeUndefined();
  });

  it('returns undefined (no throw) when a kept index runs past the end of the ranges', () => {
    expect(() =>
      buildElementIdMap([{ start: 0, end: 2 }] as ElementIdRange[], [0, 5], 2, MOD)
    ).not.toThrow();
    expect(
      buildElementIdMap([{ start: 0, end: 2 }] as ElementIdRange[], [0, 5], 2, MOD)
    ).toBeUndefined();
  });

  // -------------------------------------------------------------------------
  // GSplats-shaped inputs (issue #1423). The kept list arrives as the fused
  // kernel's recorded source indices — a Uint32Array, not a plain number[] —
  // and gsplats routinely hits BOTH divergences at once (range concat from the
  // spatial index plus hidden-dim attenuation compaction inside the kernel).
  // -------------------------------------------------------------------------

  it('accepts a Uint32Array kept list (the gsplats kernel’s recorded indices)', () => {
    const ids = buildElementIdMap(
      [{ start: 512, end: 520 }] as ElementIdRange[],
      new Uint32Array([0, 3, 7]),
      3,
      Modules.GSPLATS_SPATIAL_INDEX_LOADER
    );
    expect(Array.from(ids!)).toEqual([512, 515, 519]);
  });

  it('composes multi-range concat with kernel compaction (the gsplats case)', () => {
    // Two non-adjacent visible chunks — concat [0, 256) ↦ [1024, 1280) and
    // concat [256, 512) ↦ [4096, 4352) — with the kernel keeping four splats
    // straddling the boundary.
    const ranges = [
      { start: 1024, end: 1280 },
      { start: 4096, end: 4352 },
    ] as ElementIdRange[];
    const ids = buildElementIdMap(
      ranges,
      new Uint32Array([0, 255, 256, 511]),
      4,
      Modules.GSPLATS_SPATIAL_INDEX_LOADER
    );
    expect(Array.from(ids!)).toEqual([1024, 1279, 4096, 4351]);
  });

  it('is the identity for a whole-node gsplats load with no compaction', () => {
    // A node with no chunk index yields `[{start: 0, end: n_splats}]` and the
    // standard-3D fast path compacts nothing — slot IS the on-disk index, so
    // no array is allocated at all.
    expect(
      buildElementIdMap(
        [{ start: 0, end: 100000 }] as ElementIdRange[],
        null,
        100000,
        Modules.GSPLATS_SPATIAL_INDEX_LOADER
      )
    ).toBeUndefined();
  });
});
