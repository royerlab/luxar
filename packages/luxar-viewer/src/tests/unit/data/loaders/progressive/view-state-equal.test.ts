/**
 * Direct tests for the shared progressive-loader view-state equality
 * (`data/loaders/progressive/view-state-equal.ts`).
 *
 * The behavioral consequences (memoized-noop skip, generation reset →
 * lineage drop) are covered per geometry in the three progressive-loader
 * suites; this file pins the comparison semantics themselves, which all
 * three loaders now share. The dimensions comparison is the canonical
 * QUERY-DETERMINANT projection (see dimsQuerySig): display/navigation
 * metadata and object key order deliberately do not matter.
 */

import { describe, it, expect } from 'vitest';
import { viewStatesEqual } from '../../../../../data/loaders/progressive/view-state-equal';
import type { ViewState } from '../../../../../data/data-loader-types';

// displayDims [0,1,2] over 4 dims → index 3 is the non-displayed dim.
const base = (): ViewState => ({
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 5],
  tolerance: [0, 0, 0, 2],
});

// A 4-dim metadata array: x/y/z displayed, t non-displayed & discrete.
const dims4 = () =>
  JSON.parse(
    JSON.stringify([
      { name: 'x', unit: 'um', scale: 1, spatial: true },
      { name: 'y', unit: 'um', scale: 1, spatial: true },
      { name: 'z', unit: 'um', scale: 1, spatial: true },
      { name: 't', unit: 's', scale: 1, discrete: true, step: 1, spatial: false, cyclic: false },
    ])
  ) as ViewState['dimensions'];

describe('viewStatesEqual', () => {
  it('reports equal for element-wise identical states (distinct arrays)', () => {
    expect(viewStatesEqual(base(), base())).toBe(true);
  });

  it('detects a change in any query-affecting array', () => {
    expect(viewStatesEqual(base(), { ...base(), displayDims: [0, 1, 3] })).toBe(false);
    expect(viewStatesEqual(base(), { ...base(), slicePosition: [0, 0, 0, 6] })).toBe(false);
    expect(viewStatesEqual(base(), { ...base(), tolerance: [0, 0, 0, 3] })).toBe(false);
  });

  it('detects length changes in every array (dimensionality switch)', () => {
    // Each array's length guard is load-bearing, and only the SHORTER-a
    // direction can expose its removal: the element loop iterates a.length,
    // so an equal-prefix shorter `a` would compare equal to a longer `b`.
    // Assert both directions per array.
    expect(viewStatesEqual(base(), { ...base(), displayDims: [0, 1] })).toBe(false);
    expect(viewStatesEqual({ ...base(), displayDims: [0, 1] }, base())).toBe(false);
    expect(viewStatesEqual(base(), { ...base(), slicePosition: [0, 0, 0] })).toBe(false);
    expect(viewStatesEqual({ ...base(), slicePosition: [0, 0, 0] }, base())).toBe(false);
    expect(viewStatesEqual(base(), { ...base(), tolerance: [0, 0, 0] })).toBe(false);
    expect(viewStatesEqual({ ...base(), tolerance: [0, 0, 0] }, base())).toBe(false);
  });

  it('dimensions: reference-equal short-circuits; content-equal compares equal', () => {
    const dims = dims4();
    const a = { ...base(), dimensions: dims };
    // Same reference → equal without any projection work.
    expect(viewStatesEqual(a, { ...base(), dimensions: dims })).toBe(true);
    // Different reference, same content → equal.
    expect(viewStatesEqual(a, { ...base(), dimensions: dims4() })).toBe(true);
  });

  it('ignores the startup metadata-refresh churn (range fill, displayed-dim step, key order)', () => {
    // THE case this projection exists for: the scene REBUILDS the dimensions
    // metadata right after the first data load — dropping the `range: null`
    // key, deriving `step: null → 1` on displayed dims, and reordering
    // object keys. None of it changes what a fixed query loads, but the old
    // raw-JSON compare reset every progressive loader's generation once per
    // dataset load — discarding the ladder prefix and the append-fast-path
    // lineage. (`buildSliceViewSig` excludes the same fields from the cache
    // determinant.)
    const before = [
      { name: 'x', unit: 'units', scale: 1, range: null, display: true, step: null, spatial: true },
      { name: 'y', unit: 'units', scale: 1, range: null, display: true, step: null, spatial: true },
      { name: 'z', unit: 'units', scale: 1, range: null, display: true, step: null, spatial: true },
    ] as unknown as ViewState['dimensions'];
    const after = [
      // range key dropped, step derived, keys reordered — query-identical.
      { name: 'x', unit: 'units', scale: 1, display: true, spatial: true, step: 1 },
      { name: 'y', unit: 'units', scale: 1, display: true, spatial: true, step: 1 },
      { name: 'z', unit: 'units', scale: 1, display: true, spatial: true, step: 1 },
    ] as unknown as ViewState['dimensions'];
    const vs = (dimensions: ViewState['dimensions']): ViewState => ({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [0, 0, 0],
      dimensions,
    });
    expect(viewStatesEqual(vs(before), vs(after))).toBe(true);
  });

  it('ignores range changes on non-displayed dims too (display bound, not a query field)', () => {
    const before = dims4();
    const after = dims4();
    (after as unknown as Array<{ range?: [number, number] }>)[3].range = [0, 42];
    expect(
      viewStatesEqual({ ...base(), dimensions: before }, { ...base(), dimensions: after })
    ).toBe(true);
  });

  it('detects query-affecting changes on a NON-displayed dim (name/discrete/spatial/step/cyclic)', () => {
    const mutate = (
      patch: Partial<{
        name: string;
        discrete: boolean;
        spatial: boolean;
        step: number;
        cyclic: boolean;
      }>
    ) => {
      const changed = dims4();
      Object.assign((changed as unknown as object[])[3], patch);
      return viewStatesEqual(
        { ...base(), dimensions: dims4() },
        { ...base(), dimensions: changed }
      );
    };
    expect(mutate({ name: 'frame' })).toBe(false); // extend_to_all matches by name
    expect(mutate({ discrete: false })).toBe(false); // membership role
    expect(mutate({ spatial: true })).toBe(false); // membership role
    expect(mutate({ step: 2 })).toBe(false); // discrete half-cell gate
    expect(mutate({ cyclic: true })).toBe(false); // wrap behavior
  });

  it('displayed-dim step/discrete changes do NOT reset (keyboard-nav metadata)', () => {
    const changed = dims4();
    Object.assign((changed as unknown as object[])[0], { step: 7, discrete: true });
    expect(
      viewStatesEqual({ ...base(), dimensions: dims4() }, { ...base(), dimensions: changed })
    ).toBe(true);
    // But a displayed-dim NAME change still resets (extend_to_all matching).
    const renamed = dims4();
    Object.assign((renamed as unknown as object[])[0], { name: 'w' });
    expect(
      viewStatesEqual({ ...base(), dimensions: dims4() }, { ...base(), dimensions: renamed })
    ).toBe(false);
  });

  it('detects a dimensions-array length change', () => {
    const one = [{ name: 't' }] as unknown as ViewState['dimensions'];
    const two = [{ name: 't' }, { name: 'c' }] as unknown as ViewState['dimensions'];
    expect(viewStatesEqual({ ...base(), dimensions: one }, { ...base(), dimensions: two })).toBe(
      false
    );
  });

  it('one-sided dimensions presence is unequal', () => {
    const dims = [{ name: 't' }] as unknown as ViewState['dimensions'];
    expect(viewStatesEqual({ ...base(), dimensions: dims }, base())).toBe(false);
    expect(viewStatesEqual(base(), { ...base(), dimensions: dims })).toBe(false);
  });

  it('ignores the ride-along tolerance on non-displayed discrete non-spatial dims', () => {
    // Mirrors buildSliceViewSig: the executed query derives its reach from
    // `step` for such dims, NOT from the ride-along tolerance — and the two
    // view-state builders disagree on it (0 at init vs 0.5 from navigation),
    // which reset every progressive loader once per 4D dataset load.
    const a = { ...base(), tolerance: [0, 0, 0, 0], dimensions: dims4() };
    const b = { ...base(), tolerance: [0, 0, 0, 0.5], dimensions: dims4() };
    expect(viewStatesEqual(a, b)).toBe(true);
    expect(viewStatesEqual(b, a)).toBe(true); // symmetric

    // A step change on that dim (the REAL reach knob) still resets.
    const stepped = dims4();
    Object.assign((stepped as unknown as object[])[3], { step: 2 });
    expect(viewStatesEqual(a, { ...b, dimensions: stepped })).toBe(false);
  });

  it('never skips an extend_to_all sentinel flip as ride-along churn', () => {
    // The ride-along skip ignores the tolerance VALUE, but the worker's
    // discrete-dim membership DOES read the extend_to_all sentinel
    // (isExtendToAll): a 0.5 ↔ 1e10 flip changes which elements match, so
    // it must reset even on a discrete non-spatial non-displayed dim.
    const a = { ...base(), tolerance: [0, 0, 0, 0.5], dimensions: dims4() };
    const b = { ...base(), tolerance: [0, 0, 0, 1e10], dimensions: dims4() };
    expect(viewStatesEqual(a, b)).toBe(false);
    expect(viewStatesEqual(b, a)).toBe(false); // symmetric
    // Two different above-threshold sentinels are the same query.
    const c = { ...base(), tolerance: [0, 0, 0, 2e10], dimensions: dims4() };
    expect(viewStatesEqual(b, c)).toBe(true);
  });

  it('still compares tolerance raw on spatial, continuous, displayed, or metadata-less dims', () => {
    // Discrete SPATIAL dim: tolerance genuinely selects the decoded set.
    const spatialDims = dims4();
    Object.assign((spatialDims as unknown as object[])[3], { spatial: true });
    expect(
      viewStatesEqual(
        { ...base(), tolerance: [0, 0, 0, 0], dimensions: spatialDims },
        { ...base(), tolerance: [0, 0, 0, 0.5], dimensions: spatialDims }
      )
    ).toBe(false);
    // Continuous non-displayed dim: raw compare.
    const contDims = dims4();
    Object.assign((contDims as unknown as object[])[3], { discrete: false });
    expect(
      viewStatesEqual(
        { ...base(), tolerance: [0, 0, 0, 1], dimensions: contDims },
        { ...base(), tolerance: [0, 0, 0, 2], dimensions: contDims }
      )
    ).toBe(false);
    // No dimensions metadata at all: raw compare (current conservative behavior).
    expect(viewStatesEqual({ ...base(), tolerance: [0, 0, 0, 0] }, base())).toBe(false);
  });

  it('ignores per-pass directives (frameBudgetMs / prefetch are not query fields)', () => {
    // These must NOT defeat the memoization — a pause re-trigger arrives
    // with the same query but different directives.
    const a = { ...base(), frameBudgetMs: 8, prefetch: true } as ViewState;
    expect(viewStatesEqual(a, base())).toBe(true);
  });
});
