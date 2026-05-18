/**
 * Numeric tests for the line cap-factor formula.
 *
 * The visual shader computes cap falloff in the fragment shader from
 * interpolated `vT`. This test mirrors that formula in TypeScript so
 * body=1.0 and endpoint=0.5 behavior is covered without a WebGL
 * context.
 */
import { describe, it, expect } from 'vitest';

/**
 * Mirror of the fragment-shader cap formula in `line-shaders.ts`.
 * Returns the cap factor at parametric position `t ∈ [0,1]`.
 */
function capFactor(
  t: number,
  segmentLength: number,
  width: number,
  startClipped: boolean,
  endClipped: boolean
): number {
  const distFromStart = t * segmentLength;
  const distFromEnd = (1 - t) * segmentLength;
  const distToNearest = Math.min(distFromStart, distFromEnd);
  const capRamp = width > 1e-4 ? Math.min(Math.max(distToNearest / width, 0), 1) : 1;
  const baseCap = 0.5 + 0.5 * capRamp;
  // Mirrors the shader's `step(distFromStart, distFromEnd)`: 1 when start is nearer.
  const nearestIsStart = distFromEnd >= distFromStart ? 1 : 0;
  const nearestClipped = nearestIsStart ? startClipped : endClipped;
  return nearestClipped ? 1.0 : baseCap;
}

describe('line cap math (fragment-side)', () => {
  it('endpoint t=0: 0.5', () => {
    expect(capFactor(0, 10, 1, false, false)).toBeCloseTo(0.5);
  });

  it('endpoint t=1: 0.5', () => {
    expect(capFactor(1, 10, 1, false, false)).toBeCloseTo(0.5);
  });

  it('body well past endpoint (t=0.5 of long segment): 1.0', () => {
    // Long segment, narrow width: nearest endpoint is multiple widths away
    expect(capFactor(0.5, 10, 1, false, false)).toBeCloseTo(1.0);
  });

  it('one half-width in: midpoint of ramp (≈0.75)', () => {
    // segmentLength=10, width=1 → at t=0.05, distFromStart=0.5 = 0.5*width
    // capRamp = 0.5; baseCap = 0.5 + 0.5*0.5 = 0.75
    expect(capFactor(0.05, 10, 1, false, false)).toBeCloseTo(0.75);
  });

  it('start endpoint clipped → full intensity at t=0', () => {
    expect(capFactor(0, 10, 1, true, false)).toBeCloseTo(1.0);
  });

  it('end endpoint clipped → full intensity at t=1', () => {
    expect(capFactor(1, 10, 1, false, true)).toBeCloseTo(1.0);
  });

  it('zero width (degenerate) → cap is 1.0 (no ramp)', () => {
    expect(capFactor(0.5, 10, 0, false, false)).toBeCloseTo(1.0);
  });

  it('short segment (length < 2*width): body never quite reaches 1.0', () => {
    // segmentLength=1, width=1 → max distToNearest = 0.5 (at t=0.5)
    // capRamp = 0.5, baseCap = 0.75
    expect(capFactor(0.5, 1, 1, false, false)).toBeCloseTo(0.75);
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
});
