/**
 * Numeric tests for the line cap-factor formula.
 *
 * The visual shader computes cap falloff in the fragment shader from
 * interpolated `vT`. This test mirrors that formula in TypeScript so
 * body=1.0 and endpoint=0.5 behavior is covered without a WebGL
 * context.
 *
 * Audit acknowledgment (rendering.md [W3]): `capFactor` below is a
 * verbatim TS re-implementation of the GLSL formula in
 * `materials/line/shader-glsl.ts` — it tests the copy, not the
 * shader. This is by design: GLSL evaluation requires a WebGL
 * context that vitest jsdom cannot supply, and the structural
 * shader-pattern lock is owned by `shader-hot-path.test.ts`. The
 * GLSL/TSL parity harness + Playwright visual regression suite
 * catch divergence between this TS mirror and the real shader.
 * Same rationale applies to `gsplat-ray-integral.test.ts`.
 */
import { describe, it, expect } from 'vitest';

/**
 * Mirror of the fragment-shader cap formula in `materials/line/shader-glsl.ts`.
 * Returns the cap factor at parametric position `t ∈ [0,1]`.
 */
function capFactor(
  t: number,
  segmentLength: number,
  width: number,
  startJointCode: number | boolean,
  endJointCode: number | boolean
): number {
  const distFromStart = t * segmentLength;
  const distFromEnd = (1 - t) * segmentLength;
  // 1e-20 mirrors the shader's pure div-by-zero guard (the ratio is
  // scale-free, so sub-1e-4-unit widths keep the cap ramp).
  const startRamp = width > 1e-20 ? Math.min(Math.max(distFromStart / width, 0), 1) : 1;
  const endRamp = width > 1e-20 ? Math.min(Math.max(distFromEnd / width, 0), 1) : 1;
  // Each endpoint's ramp is lifted towards 1.0 by ITS OWN suppression
  // (`mix(baseCap, 1.0, suppression)` — a LINEAR blend, so the fractional
  // suppression a partial bend produces lands between the two), then the
  // two caps combine with min(). This makes capFactor continuous WITHIN a
  // segment even when the two width-sized cap regions overlap (segment
  // shorter than 2*width) with unequal suppressions; a residual step can
  // remain ACROSS a joint seam for sub-width segments — see the #796
  // tests below.
  const startCap = 0.5 + 0.5 * startRamp + (1.0 - (0.5 + 0.5 * startRamp)) * Number(startJointCode);
  const endCap = 0.5 + 0.5 * endRamp + (1.0 - (0.5 + 0.5 * endRamp)) * Number(endJointCode);
  return Math.min(startCap, endCap);
}

describe('line cap math (fragment-side)', () => {
  it('endpoint t=0: 0.5', () => {
    expect(capFactor(0, 10, 1, false, false)).toBeCloseTo(0.5, 5);
  });

  it('endpoint t=1: 0.5', () => {
    expect(capFactor(1, 10, 1, false, false)).toBeCloseTo(0.5, 5);
  });

  it('body well past endpoint (t=0.5 of long segment): 1.0', () => {
    // Long segment, narrow width: nearest endpoint is multiple widths away
    expect(capFactor(0.5, 10, 1, false, false)).toBeCloseTo(1.0, 5);
  });

  it('one half-width in: midpoint of ramp (≈0.75)', () => {
    // segmentLength=10, width=1 → at t=0.05, distFromStart=0.5 = 0.5*width
    // capRamp = 0.5; baseCap = 0.5 + 0.5*0.5 = 0.75
    expect(capFactor(0.05, 10, 1, false, false)).toBeCloseTo(0.75, 5);
  });

  it('start endpoint clipped → full intensity at t=0', () => {
    expect(capFactor(0, 10, 1, true, false)).toBeCloseTo(1.0, 5);
  });

  it('end endpoint clipped → full intensity at t=1', () => {
    expect(capFactor(1, 10, 1, false, true)).toBeCloseTo(1.0, 5);
  });

  it('zero width (degenerate) → cap is 1.0 (no ramp)', () => {
    expect(capFactor(0.5, 10, 0, false, false)).toBeCloseTo(1.0, 5);
  });

  it('short segment (length < 2*width): body never quite reaches 1.0', () => {
    // segmentLength=1, width=1 → at t=0.5 both endpoint distances are 0.5,
    // so both ramps are 0.5 and both caps are 0.75
    expect(capFactor(0.5, 1, 1, false, false)).toBeCloseTo(0.75, 5);
  });

  it('cap is symmetric around t=0.5', () => {
    const segLen = 10;
    const w = 1;
    for (const t of [0.1, 0.2, 0.3, 0.4]) {
      expect(capFactor(t, segLen, w, false, false)).toBeCloseTo(
        capFactor(1 - t, segLen, w, false, false)
      );
    }
  });

  // --- Interior joints (issue #780) ----------------------------------------
  // The `0.5` endpoint dip is only correct where a neighbouring quad overlaps
  // and adds the missing half back. Collinear neighbours TILE (the quad spans
  // exactly [start, end]), so an unsuppressed interior joint reads as a dark
  // notch of axial length 2*width bottoming out at 0.5. `compute_joint_codes`
  // hands the shader a per-endpoint suppression scalar; these pin how the
  // fragment stage consumes it.

  it('#780 unsuppressed interior joint is the bug: 0.5 at the joint, dip spans a full width', () => {
    // Two collinear 1-unit segments of width 1 meeting at t=1 / t=0. Without
    // suppression the shared point gets 0.5 from one side and nothing from the
    // other — this is exactly the notch the fix removes.
    expect(capFactor(1, 1, 1, false, false)).toBeCloseTo(0.5, 5);
    expect(capFactor(0, 1, 1, false, false)).toBeCloseTo(0.5, 5);
  });

  it('#780 fully suppressed joint reaches 1.0 at the shared endpoint', () => {
    expect(capFactor(1, 1, 1, false, 1)).toBeCloseTo(1.0, 5);
    expect(capFactor(0, 1, 1, 1, false)).toBeCloseTo(1.0, 5);
  });

  it('#780 a fully suppressed short segment is flat 1.0 along its whole length', () => {
    // segmentLength == width: with the cap live, NO point of the segment
    // reaches 1.0 (the worst case — the entire polyline beads). Suppressed at
    // both ends it is uniformly full intensity.
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      expect(capFactor(t, 1, 1, false, false)).toBeLessThan(0.95);
      expect(capFactor(t, 1, 1, 1, 1)).toBeCloseTo(1.0, 5);
    }
  });

  it('#780 partial suppression (a bend) blends linearly between the two regimes', () => {
    // A 45-degree turn yields suppression = cos(45°) ≈ 0.7071. At the joint
    // baseCap is 0.5, so the cap factor lands at 0.5 + 0.5 * 0.7071.
    const s = Math.SQRT1_2;
    expect(capFactor(1, 1, 1, false, s)).toBeCloseTo(0.5 + 0.5 * s, 5);
    // A 90-degree turn suppresses nothing — identical to the pre-fix value.
    expect(capFactor(1, 1, 1, false, 0)).toBeCloseTo(capFactor(1, 1, 1, false, false), 5);
  });

  it("#780 suppression lifts only ITS OWN endpoint's ramp — the far free end still dips to 0.5", () => {
    // Suppressed start, free end, long segment: near t=1 the free end still
    // dips to 0.5 — a suppressed joint must not brighten the other end.
    expect(capFactor(0, 10, 1, 1, false)).toBeCloseTo(1.0, 5);
    expect(capFactor(1, 10, 1, 1, false)).toBeCloseTo(0.5, 5);
  });

  // --- Short segments with unequal suppressions (issue #796) ----------------
  // Picking a single suppression by nearest-endpoint proximity was
  // discontinuous at the midpoint of segments shorter than 2*width whenever
  // the two endpoint suppressions differ — the routine case for the first and
  // last segment of every polyline (free end 0, interior joint ≈ 1). The fix
  // evaluates both endpoint caps independently and takes the min, which is
  // continuous WITHIN a segment. It does NOT make the field continuous across
  // a polyline seam: a sub-width segment's far-end ramp cannot reach 1.0, so a
  // residual step of 0.5*(1 - clamp(L/w)) remains at the joint — always <= the
  // old midpoint jump 0.5*(1 - L/(2w)), zero for L >= w (last test below).

  it('#796 continuous at the midpoint of a short segment with suppressions 0/1', () => {
    // L=1, w=1 (L < 2w), suppStart=0, suppEnd=1: the old nearest-endpoint
    // pick jumped by 0.5*(1 - L/(2w)) = 0.25 exactly at t=0.5.
    const eps = 1e-6;
    const below = capFactor(0.5 - eps, 1, 1, 0, 1);
    const above = capFactor(0.5 + eps, 1, 1, 0, 1);
    expect(Math.abs(above - below)).toBeLessThan(1e-4);
    // Dense sample across the whole segment: no jump anywhere (the formula
    // is 1-Lipschitz in distFromStart/width, so adjacent samples differ by
    // at most ~0.5 * step/width plus rounding).
    let prev = capFactor(0, 1, 1, 0, 1);
    for (let i = 1; i <= 1000; i++) {
      const cur = capFactor(i / 1000, 1, 1, 0, 1);
      expect(Math.abs(cur - prev)).toBeLessThan(0.002);
      prev = cur;
    }
  });

  it('#796 the free endpoint ramp runs its full course on a short segment', () => {
    // suppEnd=1 lifts the end cap to a constant 1.0, so the whole segment is
    // governed by the start ramp: 0.5 + 0.5*min(t*L/w, 1) all the way to t=1
    // (the old nearest pick cut the ramp off at the midpoint).
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      expect(capFactor(t, 1, 1, 0, 1)).toBeCloseTo(0.5 + 0.5 * Math.min(t, 1), 5);
    }
  });

  it('#796 reduces to the old nearest-endpoint formula when suppressions are equal or segments are long', () => {
    /** The pre-#796 formula: single ramp on the nearest endpoint. */
    const oldCapFactor = (t: number, len: number, w: number, s0: number, s1: number): number => {
      const dS = t * len;
      const dE = (1 - t) * len;
      const ramp = w > 1e-20 ? Math.min(Math.max(Math.min(dS, dE) / w, 0), 1) : 1;
      const baseCap = 0.5 + 0.5 * ramp;
      const s = dE >= dS ? s0 : s1;
      return baseCap + (1 - baseCap) * s;
    };
    for (const t of [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]) {
      // Equal suppressions, short segment.
      for (const s of [0, 0.5, 1]) {
        expect(capFactor(t, 1, 1, s, s)).toBeCloseTo(oldCapFactor(t, 1, 1, s, s), 6);
      }
      // Long segment (L >= 2w), unequal suppressions: cap regions disjoint.
      expect(capFactor(t, 10, 1, 0, 1)).toBeCloseTo(oldCapFactor(t, 10, 1, 0, 1), 6);
      expect(capFactor(t, 10, 1, 0.25, 0.75)).toBeCloseTo(oldCapFactor(t, 10, 1, 0.25, 0.75), 6);
    }
  });

  it('#796 residual CROSS-SEAM step for sub-width segments: 0.5*(1 - clamp(L/w)), <= old jump, 0 for L >= w', () => {
    // The min() form is continuous within a segment but NOT across a
    // polyline seam: a segment shorter than one width whose far end is not
    // fully suppressed cannot ramp to 1.0 by the joint, while the
    // neighbouring segment's side of the seam is 1.0. Pin the accurate
    // claim: the step equals 0.5*(1 - clamp(L/w, 0, 1)), is never larger
    // than the old midpoint jump 0.5*(1 - L/(2w)), and vanishes for L >= w.
    const w = 1;
    // Neighbour's side of the seam (long segment, suppressed at the joint).
    const neighbourSide = capFactor(0, 10, w, 1, 0);
    expect(neighbourSide).toBeCloseTo(1.0, 6);
    for (const L of [0.2, 0.5, 0.9, 1.0, 1.5, 2.0]) {
      // Short segment: free start (s=0), fully suppressed joint end (s=1).
      const shortSide = capFactor(1, L, w, 0, 1);
      const step = neighbourSide - shortSide;
      const expected = 0.5 * (1 - Math.min(L / w, 1));
      const oldJump = Math.max(0.5 * (1 - L / (2 * w)), 0);
      expect(step).toBeCloseTo(expected, 6);
      expect(step).toBeLessThanOrEqual(oldJump + 1e-9);
      if (L >= w) expect(step).toBeCloseTo(0, 6);
    }
  });
});
