/**
 * Direct tests for the shared progressive-loader view-state equality
 * (`data/loaders/progressive/view-state-equal.ts`).
 *
 * The behavioral consequences (memoized-noop skip, generation reset →
 * lineage drop) are covered per geometry in the three progressive-loader
 * suites; this file pins the comparison semantics themselves, which all
 * three loaders now share.
 */

import { describe, it, expect } from 'vitest';
import { viewStatesEqual } from '../../../../../data/loaders/progressive/view-state-equal';
import type { ViewState } from '../../../../../data/data-loader-types';

const base = (): ViewState => ({
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 5],
  tolerance: [0, 0, 0, 2],
});

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

  it('dimensions: reference-equal short-circuits; content-equal via JSON; content change detected', () => {
    const dims = [{ name: 'time', unit: 's', display: false }] as ViewState['dimensions'];
    const a = { ...base(), dimensions: dims };
    // Same reference → equal without JSON work.
    expect(viewStatesEqual(a, { ...base(), dimensions: dims })).toBe(true);
    // Different reference, same content → equal (JSON compare).
    expect(viewStatesEqual(a, { ...base(), dimensions: JSON.parse(JSON.stringify(dims)) })).toBe(
      true
    );
    // Content change (e.g. a range populated after first load) → unequal.
    const changed = JSON.parse(JSON.stringify(dims));
    changed[0].name = 'frame';
    expect(viewStatesEqual(a, { ...base(), dimensions: changed })).toBe(false);
  });

  it('one-sided dimensions presence is unequal', () => {
    const dims = [{ name: 't' }] as unknown as ViewState['dimensions'];
    expect(viewStatesEqual({ ...base(), dimensions: dims }, base())).toBe(false);
    expect(viewStatesEqual(base(), { ...base(), dimensions: dims })).toBe(false);
  });

  it('ignores per-pass directives (frameBudgetMs / prefetch are not query fields)', () => {
    // These must NOT defeat the memoization — a pause re-trigger arrives
    // with the same query but different directives.
    const a = { ...base(), frameBudgetMs: 8, prefetch: true } as ViewState;
    expect(viewStatesEqual(a, base())).toBe(true);
  });
});
