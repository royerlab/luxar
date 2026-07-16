/**
 * Unit tests for the nD hypersphere effective-radius helpers.
 *
 * These pin the pure mathematical contracts the point loader relies on:
 *   - `calculateEffectiveRadii`: the Pythagorean cross-section invariant
 *     R_eff = √(R² − D²), boundary/clamp behavior, displayed-dim passthrough,
 *     discrete exact-match filtering, the `isSpatialDim` out-of-bounds default
 *     (spatial=true), and extend_to_all (tolerance ≥ 1e9) bypass.
 *   - `calculateSpatialQueryTolerance`: per-dim tolerance selection
 *     (displayed/extend_to_all → 1e10, spatial → maxRadius, discrete → the
 *     shared quarter-cell rule `0.25 × step`, fallback 0.25 — see
 *     `discreteDimTolerance` in tolerance-computer.ts; the half-step
 *     MEMBERSHIP gate in calculateEffectiveRadii is a separate concern).
 *   - `shouldApplyEffectiveRadius`: gate on config/radii/non-displayed dims.
 */

import { describe, it, expect } from 'vitest';
import {
  calculateEffectiveRadii,
  calculateSpatialQueryTolerance,
  fallbackQueryTolerance,
  shouldApplyEffectiveRadius,
} from '../../../../data/points/effective-radius-calculator';
import type { EffectiveRadiusConfig } from '../../../../types/points';
import type { ViewState } from '../../../../data/data-loader-types';

function makeViewState(overrides: Partial<ViewState> = {}): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0],
    tolerance: [0, 0, 0],
    ...overrides,
  };
}

function makeConfig(overrides: Partial<EffectiveRadiusConfig> = {}): EffectiveRadiusConfig {
  return {
    spatialExtendDims: [true, true, true],
    maxRadius: 1,
    ...overrides,
  };
}

describe('fallbackQueryTolerance — config-null query reach', () => {
  const dims = [
    { name: 'X', unit: '', scale: 1, spatial: true },
    { name: 'Y', unit: '', scale: 1, spatial: true },
    { name: 'Z', unit: '', scale: 1, spatial: true },
    { name: 'T', unit: '', scale: 1, discrete: true, spatial: false, step: 1 },
  ] as ViewState['dimensions'];

  it('discrete non-spatial reach is the quarter-cell rule (0.25×step), NOT the ride-along tolerance', () => {
    // The two viewState builders set the discrete ride-along to 0.5 (nav) vs 0
    // (init); the fetch reach MUST be independent of it (the SliceCache key
    // drops it). Pre-fix the fallback used `tolerance[d]` directly → this
    // returned 0.5 / 0, letting a cache hit serve a different-reach decode.
    const navView = makeViewState({
      slicePosition: [0, 0, 0, 3],
      tolerance: [0, 0, 0, 0.5],
      dimensions: dims,
    });
    const initView = makeViewState({
      slicePosition: [0, 0, 0, 3],
      tolerance: [0, 0, 0, 0],
      dimensions: dims,
    });
    const nav = fallbackQueryTolerance(navView, 4, 1);
    const init = fallbackQueryTolerance(initView, 4, 1);
    expect(nav[3]).toBeCloseTo(0.25); // 0.25 × step(1), not 0.5
    expect(init[3]).toBeCloseTo(0.25); // and not 0 either
    expect(nav[3]).toBe(init[3]); // builder-independent — no cache collision
  });

  it('scales the quarter-cell reach with the discrete step', () => {
    const view = makeViewState({
      slicePosition: [0, 0, 0, 3],
      tolerance: [0, 0, 0, 0.5],
      dimensions: [
        dims![0],
        dims![1],
        dims![2],
        { name: 'T', unit: '', scale: 1, discrete: true, spatial: false, step: 4 },
      ],
    });
    expect(fallbackQueryTolerance(view, 4, 1)[3]).toBeCloseTo(1.0); // 0.25 × 4
  });

  it('displayed dims and the extend_to_all sentinel load all chunks (1e10)', () => {
    const view = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 3],
      tolerance: [0, 0, 0, 1e10], // extend_to_all on the discrete dim
      dimensions: dims,
    });
    const t = fallbackQueryTolerance(view, 4, 1);
    expect(t[0]).toBe(1e10); // displayed
    expect(t[3]).toBe(1e10); // extend_to_all sentinel wins over the discrete rule
  });

  it('non-displayed spatial/continuous dims fall back to maxRadius (ride-along kept when present)', () => {
    // A 4D view where dim 3 is spatial (not discrete) and non-displayed. Unlike
    // discrete dims, the SliceCache key KEEPS a continuous dim's tolerance, so
    // the query legitimately uses `tolerance[d] ?? maxRadius` (unchanged
    // behaviour): the ride-along when present, else maxRadius.
    const spatialDims = [
      dims![0],
      dims![1],
      dims![2],
      { name: 'W', unit: '', scale: 1, spatial: true, discrete: false },
    ];
    // No ride-along at dim 3 → maxRadius.
    const noTol = makeViewState({
      slicePosition: [0, 0, 0, 5],
      tolerance: [0, 0, 0],
      dimensions: spatialDims,
    });
    expect(fallbackQueryTolerance(noTol, 4, 7)[3]).toBe(7);
    // Ride-along present → used verbatim (kept in the key, so no collision).
    const withTol = makeViewState({
      slicePosition: [0, 0, 0, 5],
      tolerance: [0, 0, 0, 2],
      dimensions: spatialDims,
    });
    expect(fallbackQueryTolerance(withTol, 4, 7)[3]).toBe(2);
  });
});

describe('calculateEffectiveRadii — Pythagorean invariant', () => {
  it('returns R_eff = √(R² − D²) ≤ R for a point offset in a non-displayed spatial dim', () => {
    // 4D: display dims 0,1,2; dim 3 is non-displayed spatial.
    // Point at offset D=3 in dim 3, radius R=5 → R_eff = √(25 − 9) = 4.
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 3]);
    const radii = new Float32Array([5]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);

    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(1);
    expect(result[0]).toBeCloseTo(4, 5);
    expect(result[0]).toBeLessThanOrEqual(radii[0]);
  });

  it('combines distance across multiple non-displayed spatial dims (sum of squares)', () => {
    // 5D: display 0,1,2; dims 3,4 non-displayed spatial.
    // Offsets D3=3, D4=4 → D² = 9 + 16 = 25, R=13 → R_eff = √(169 − 25) = 12.
    const ndim = 5;
    const positions = new Float32Array([0, 0, 0, 3, 4]);
    const radii = new Float32Array([13]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0, 0],
      tolerance: [0, 0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBeCloseTo(12, 5);
  });

  it('respects a non-zero slice position (distance measured from slice, not origin)', () => {
    // Point at value 10 in dim 3, slice at 7 → D=3, R=5 → R_eff = 4.
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 10]);
    const radii = new Float32Array([5]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 7],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBeCloseTo(4, 5);
  });

  it('returns the full radius when the point lies exactly on the slice (D=0)', () => {
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 0]);
    const radii = new Float32Array([2.5]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBeCloseTo(2.5, 6);
  });
});

describe('calculateEffectiveRadii — boundary and clamping', () => {
  it('returns exactly 0 at the boundary D == R (R² − D² == 0)', () => {
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 5]); // D=5
    const radii = new Float32Array([5]); // R=5
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBe(0);
  });

  it('clamps to 0 when D > R (point outside the slice, filtered)', () => {
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 10]); // D=10
    const radii = new Float32Array([5]); // R=5 < D
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBe(0);
  });
});

describe('calculateEffectiveRadii — displayed-dimension passthrough', () => {
  it('leaves radii unchanged when all dims are displayed (no hidden dims)', () => {
    // 3D, all displayed: no distance contribution → R_eff = R for every point.
    const ndim = 3;
    const positions = new Float32Array([1, 2, 3, 50, 60, 70]);
    const radii = new Float32Array([4, 9]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBeCloseTo(4, 6);
    expect(result[1]).toBeCloseTo(9, 6);
  });

  it('ignores offsets in displayed dims entirely (they are in the view plane)', () => {
    // Large offsets in displayed dims must NOT reduce the radius.
    const ndim = 3;
    const positions = new Float32Array([1000, -1000, 500]);
    const radii = new Float32Array([7]);
    const result = calculateEffectiveRadii(
      positions,
      radii,
      makeViewState({ displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] }),
      makeConfig({ spatialExtendDims: [true, true, true] }),
      ndim
    );
    expect(result[0]).toBeCloseTo(7, 6);
  });
});

describe('calculateEffectiveRadii — out-of-bounds spatialExtendDims default', () => {
  it('treats a dimension beyond spatialExtendDims.length as spatial (default true)', () => {
    // 4D: spatialExtendDims only covers dims 0..2. Dim 3 is uncovered →
    // default spatial=true → Pythagorean applies: D=3, R=5 → R_eff = 4.
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 3]);
    const radii = new Float32Array([5]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    });
    // Note: only 3 entries → dim 3 falls through to the permissive default.
    const config = makeConfig({ spatialExtendDims: [true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    // If the default were discrete, the |3 − 0| > 0.5 mismatch would zero it.
    // Spatial default → R_eff = 4, proving the permissive (spatial) default.
    expect(result[0]).toBeCloseTo(4, 5);
  });
});

describe('calculateEffectiveRadii — discrete-dimension exact match', () => {
  it('keeps the full radius when a discrete dim matches within the 0.5 tolerance', () => {
    // Dim 3 discrete (spatialExtendDims[3] === false). value=2.4, slice=2 →
    // |2.4 − 2| = 0.4 ≤ 0.5 → match → R_eff = R (no distance term added).
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 2.4]);
    const radii = new Float32Array([3]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 2],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, false] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBeCloseTo(3, 6);
  });

  it('zeroes the radius when a discrete dim mismatches beyond the 0.5 tolerance', () => {
    // value=3, slice=2 → |3 − 2| = 1.0 > 0.5 → no match → R_eff = 0.
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 3]);
    const radii = new Float32Array([3]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 2],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, false] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBe(0);
  });

  it('is exactly at the discrete tolerance boundary (|Δ| == 0.5 still matches)', () => {
    // |2.5 − 2| = 0.5; the check is `> discreteTolerance` so 0.5 is NOT a
    // mismatch → match → full radius preserved.
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 2.5]);
    const radii = new Float32Array([3]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 2],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, false] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBeCloseTo(3, 6);
  });
});

describe('calculateEffectiveRadii — extend_to_all bypass (tolerance ≥ 1e9)', () => {
  it('ignores a spatial dim entirely when its tolerance is ≥ 1e9 (no distance term)', () => {
    // Dim 3 is spatial with a huge offset (D would be 100), but tolerance is
    // 1e9 → extend_to_all → that dim contributes nothing → R_eff = R.
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 100]);
    const radii = new Float32Array([5]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 1e9],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBeCloseTo(5, 6);
  });

  it('ignores a discrete-dim mismatch when extend_to_all is set for that dim', () => {
    // Dim 3 discrete and mismatched (value 100 vs slice 0), but tolerance 1e9
    // bypasses the discrete check → point stays visible at full radius.
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 100]);
    const radii = new Float32Array([5]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 1e9],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, false] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result[0]).toBeCloseTo(5, 6);
  });
});

describe('calculateEffectiveRadii — multi-point batch', () => {
  it('computes each point independently in a single call', () => {
    // 3 points, 4D. Dim 3 non-displayed spatial; slice at 0.
    //  P0: offset 0  → R_eff = R = 10
    //  P1: offset 6, R=10 → √(100 − 36) = 8
    //  P2: offset 12 (D>R) → 0
    const ndim = 4;
    const positions = new Float32Array([0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 12]);
    const radii = new Float32Array([10, 10, 10]);
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, true] });

    const result = calculateEffectiveRadii(positions, radii, viewState, config, ndim);
    expect(result.length).toBe(3);
    expect(result[0]).toBeCloseTo(10, 5);
    expect(result[1]).toBeCloseTo(8, 5);
    expect(result[2]).toBe(0);
  });
});

describe('calculateSpatialQueryTolerance', () => {
  it('uses 1e10 for displayed dims, maxRadius for non-displayed spatial dims', () => {
    const ndim = 4;
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, true], maxRadius: 7 });

    const tol = calculateSpatialQueryTolerance(viewState, config, ndim);
    expect(tol).toEqual([1e10, 1e10, 1e10, 7]);
  });

  it('uses the shared quarter-cell (0.25 × step) for non-displayed discrete dims', () => {
    // The chunk-query reach must stay BELOW half a step: legacy datasets pad
    // chunk bounds ±0.5 on the write side, and 0.5 pad + 0.5 tolerance = a
    // full step — fetching the whole neighbouring category (the barrier
    // over-fetch bug). 0.25 keeps pad + tolerance < 1 step for both legacy
    // and ε-padded data.
    const ndim = 4;
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, false], maxRadius: 7 });

    const tol = calculateSpatialQueryTolerance(viewState, config, ndim);
    expect(tol).toEqual([1e10, 1e10, 1e10, 0.25]);
  });

  it('scales the discrete quarter-cell by the dimension step (step 2 → 0.5)', () => {
    const ndim = 4;
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
      dimensions: [
        { name: 'x', unit: '', scale: 1 },
        { name: 'y', unit: '', scale: 1 },
        { name: 'z', unit: '', scale: 1 },
        { name: 'time', unit: '', scale: 1, discrete: true, step: 2 },
      ],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, false], maxRadius: 7 });

    const tol = calculateSpatialQueryTolerance(viewState, config, ndim);
    expect(tol[3]).toBe(0.5); // 0.25 × step 2
  });

  it('uses 1e10 for an extend_to_all dim (tolerance ≥ 1e9) regardless of spatial flag', () => {
    const ndim = 4;
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 1e9],
    });
    const config = makeConfig({ spatialExtendDims: [true, true, true, false], maxRadius: 7 });

    const tol = calculateSpatialQueryTolerance(viewState, config, ndim);
    expect(tol[3]).toBe(1e10);
  });

  it('defaults an out-of-bounds dim to spatial → maxRadius tolerance', () => {
    const ndim = 4;
    const viewState = makeViewState({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    });
    // spatialExtendDims covers only 0..2 → dim 3 falls to the spatial default.
    const config = makeConfig({ spatialExtendDims: [true, true, true], maxRadius: 12 });

    const tol = calculateSpatialQueryTolerance(viewState, config, ndim);
    expect(tol[3]).toBe(12);
  });
});

describe('shouldApplyEffectiveRadius', () => {
  it('returns false when config is null', () => {
    expect(shouldApplyEffectiveRadius(null, [0, 1, 2], true)).toBe(false);
  });

  it('returns false when radii are unavailable', () => {
    const config = makeConfig({ spatialExtendDims: [true, true, true, true] });
    expect(shouldApplyEffectiveRadius(config, [0, 1, 2], false)).toBe(false);
  });

  it('returns false when every config dim is displayed', () => {
    const config = makeConfig({ spatialExtendDims: [true, true, true] });
    expect(shouldApplyEffectiveRadius(config, [0, 1, 2], true)).toBe(false);
  });

  it('returns true when a non-displayed dim exists (filtering needed)', () => {
    const config = makeConfig({ spatialExtendDims: [true, true, true, true] });
    expect(shouldApplyEffectiveRadius(config, [0, 1, 2], true)).toBe(true);
  });

  it('returns true for a non-displayed discrete dim too', () => {
    const config = makeConfig({ spatialExtendDims: [true, true, true, false] });
    expect(shouldApplyEffectiveRadius(config, [0, 1, 2], true)).toBe(true);
  });
});
