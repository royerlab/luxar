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
  startCapSuppression: number | boolean,
  endCapSuppression: number | boolean
): number {
  const distFromStart = t * segmentLength;
  const distFromEnd = (1 - t) * segmentLength;
  const distToNearest = Math.min(distFromStart, distFromEnd);
  // 1e-20 mirrors the shader's pure div-by-zero guard (the ratio is
  // scale-free, so sub-1e-4-unit widths keep the cap ramp).
  const capRamp = width > 1e-20 ? Math.min(Math.max(distToNearest / width, 0), 1) : 1;
  const baseCap = 0.5 + 0.5 * capRamp;
  // Mirrors the shader's `step(distFromStart, distFromEnd)`: 1 when start is nearer.
  const nearestIsStart = distFromEnd >= distFromStart ? 1 : 0;
  const nearestSuppress = Number(nearestIsStart ? startCapSuppression : endCapSuppression);
  // Mirrors `mix(baseCap, 1.0, nearestSuppress)` — a LINEAR blend, so the
  // fractional suppression a partial bend produces lands between the two.
  return baseCap + (1.0 - baseCap) * nearestSuppress;
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
    // segmentLength=1, width=1 → max distToNearest = 0.5 (at t=0.5)
    // capRamp = 0.5, baseCap = 0.75
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
  // notch of axial length 2*width bottoming out at 0.5. `compute_cap_suppression`
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

  it('#780 suppression only lifts the NEAREST endpoint, never the far half', () => {
    // Suppressed start, free end, long segment: near t=1 the free end still
    // dips to 0.5 — a suppressed joint must not brighten the other end.
    expect(capFactor(0, 10, 1, 1, false)).toBeCloseTo(1.0, 5);
    expect(capFactor(1, 10, 1, 1, false)).toBeCloseTo(0.5, 5);
  });
});
