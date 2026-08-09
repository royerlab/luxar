/**
 * Unit tests for `materials/line/ray-integral.ts` — the #1352
 * volumetric-line ray integral.
 *
 * The suite is organised around the claims the design rests on:
 *
 *  1. **Correctness.** The closed form is checked against a brute-force
 *     quadrature of the underlying density along the ray — an
 *     independent computation, not a restatement of the formula.
 *  2. **The two limits.** Side-on gives the bare Gaussian cross-section
 *     (deliberately NOT the shifted-truncated one the current shader
 *     draws — that difference is pinned here too); end-on is the finite
 *     `L·G(D)` path-length answer, which is the artifact fix.
 *  3. **The numerical lanes.** Continuity across the `Δ = 0.5` threshold
 *     from both sides, non-negativity over a dense float64 sweep, and —
 *     separately, because float64 is structurally blind to it — a
 *     **float32** harness for the overflow class that poisons the
 *     unselected `mix` arm.
 *  4. **Additivity.** Splitting a segment telescopes exactly (residual
 *     pinned on both the `erfRef` and the `erfPoly` path).
 *  5. **Calibration.** The authored-width → sigma factor of 2, pinned
 *     against the same constants the shader uses.
 *
 * GLSL/TSL single-sourcing is pinned the same way `erf.test.ts` pins it:
 * `toFixed(9)` literal presence, a TSL builder smoke test, and a
 * mechanical transpile of the GLSL body to JS that must be bit-identical
 * to the TS mirror.
 */

import { describe, it, expect } from 'vitest';
import { float } from 'three/tsl';
import {
  GLSL_LINE_RAY_INTEGRAL_FUNCTIONS,
  LINE_INV_SQRT2,
  LINE_RENDERED_HALF_WIDTH_FACTOR,
  LINE_TAYLOR_MM_CLAMP,
  LINE_TWO_OVER_SQRT_PI,
  LINE_WINDOW_GAP_THRESHOLD,
  LINE_WINDOW_TAYLOR_C2,
  LINE_WINDOW_TAYLOR_C4,
  RAY_SEGMENT_PARALLEL_EPS,
  RAY_SEGMENT_PARALLEL_EPS_F32,
  lineCapsuleProfileRef,
  lineCapsuleProfileTSL,
  lineRayIntegralPoly,
  lineRayIntegralRef,
  lineRayIntegralTSL,
  lineSigmaFromAuthoredWidth,
  raySegmentDistance,
  raySegmentDistanceTSL,
  raySegmentGeometry,
  type LineRayIntegralParams,
  type Vec3,
} from '../../../../../rendering/materials/line/ray-integral';
import { erfRef, GLSL_ERF_FUNCTIONS } from '../../../../../rendering/materials/_shared/erf';
import {
  FALLOFF_FLOOR,
  FALLOFF_K,
  GAUSSIAN_EQUIVALENT_TRUNCATION,
  INV_ONE_MINUS_FALLOFF_FLOOR,
} from '../../../../../rendering/materials/_shared/falloff';

const SQRT_2PI = Math.sqrt(2 * Math.PI);

/** Radial Gaussian `exp(-D²/(2σ²))` — the bare (untruncated) profile. */
function radialGaussian(distance: number, sigma: number): number {
  return Math.exp(-(distance * distance) / (2 * sigma * sigma));
}

/**
 * Brute-force quadrature of the primitive's density along a ray — the
 * independent oracle for the closed form.
 *
 * Integrates `rho(x) = G_2D(r) · W(s)` (the module's density, amplitude
 * 1) over the ray line by trapezoid rule, then normalizes by `σ√(2π)` to
 * match the module's convention. Nothing here reuses the closed form.
 */
function quadratureIntegral(
  rayOrigin: Vec3,
  rayDirection: Vec3,
  segmentStart: Vec3,
  segmentEnd: Vec3,
  sigma: number,
  halfSpan = 80,
  steps = 160000
): number {
  const seg: Vec3 = [
    segmentEnd[0] - segmentStart[0],
    segmentEnd[1] - segmentStart[1],
    segmentEnd[2] - segmentStart[2],
  ];
  const length = Math.hypot(seg[0], seg[1], seg[2]);
  const d: Vec3 = [seg[0] / length, seg[1] / length, seg[2] / length];
  const vLen = Math.hypot(rayDirection[0], rayDirection[1], rayDirection[2]);
  const v: Vec3 = [rayDirection[0] / vLen, rayDirection[1] / vLen, rayDirection[2] / vLen];

  const dTau = (2 * halfSpan) / steps;
  let total = 0;
  for (let i = 0; i <= steps; i++) {
    const tau = -halfSpan + i * dTau;
    const px = rayOrigin[0] + tau * v[0] - segmentStart[0];
    const py = rayOrigin[1] + tau * v[1] - segmentStart[1];
    const pz = rayOrigin[2] + tau * v[2] - segmentStart[2];
    const s = px * d[0] + py * d[1] + pz * d[2];
    const rx = px - s * d[0];
    const ry = py - s * d[1];
    const rz = pz - s * d[2];
    const r = Math.hypot(rx, ry, rz);
    const w =
      0.5 * (erfRef(s / (sigma * Math.SQRT2)) - erfRef((s - length) / (sigma * Math.SQRT2)));
    const weight = i === 0 || i === steps ? 0.5 : 1;
    total += weight * Math.exp(-(r * r) / (2 * sigma * sigma)) * w;
  }
  return (total * dTau) / (sigma * SQRT_2PI);
}

describe('lineSigmaFromAuthoredWidth (calibration against the real shader chain)', () => {
  it('doubles the authored width, because the vertex stage does', () => {
    // The authored `width` attribute is a HALF-width (line-geometry.ts),
    // but luxarLineEndPixelWidth multiplies by resY/tan(fov/2) where
    // pixels-per-unit is resY/(2 z tan(fov/2)), and the result is then
    // applied at aQuadCorner.y = +/-1. Net: the world radius at
    // |vPerpNorm| = 1 is 2 x width. Pinned so nobody "simplifies" it.
    expect(LINE_RENDERED_HALF_WIDTH_FACTOR).toBe(2);
    for (const width of [0.25, 1, 3.25, 17]) {
      const drawnHalfWidth = LINE_RENDERED_HALF_WIDTH_FACTOR * width;
      expect(lineSigmaFromAuthoredWidth(width)).toBeCloseTo(
        drawnHalfWidth / GAUSSIAN_EQUIVALENT_TRUNCATION,
        12
      );
    }
  });

  it('reproduces the shader pixel chain end to end for a concrete camera', () => {
    // Independent re-derivation from the uniforms, not from the constant:
    // resY / tan(fov/2) is what the shader multiplies the width by, and
    // resY / (2 z tan(fov/2)) is the true pixels-per-world-unit.
    const resY = 1080;
    const fov = (60 * Math.PI) / 180;
    const z = 42;
    const width = 3.0;
    const perspectiveLineScale = resY / Math.tan(fov * 0.5); // shader uniform
    const shaderPixelHalfWidth = (width * perspectiveLineScale) / z;
    const pxPerUnit = resY / (2 * z * Math.tan(fov * 0.5));
    expect(shaderPixelHalfWidth / pxPerUnit).toBeCloseTo(2 * width, 9); // = 6 world units
    // ... which is exactly the radius the sigma calibration assumes.
    expect(lineSigmaFromAuthoredWidth(width) * GAUSSIAN_EQUIVALENT_TRUNCATION).toBeCloseTo(
      shaderPixelHalfWidth / pxPerUnit,
      9
    );
  });

  it('reuses GAUSSIAN_EQUIVALENT_TRUNCATION rather than a second viewer constant', () => {
    const width = 2.5;
    const sigma = lineSigmaFromAuthoredWidth(width);
    // The unshifted kernel value at the drawn edge is the 1% floor the
    // super-Gaussian's shift subtracts away (see the divergence test).
    expect(radialGaussian(LINE_RENDERED_HALF_WIDTH_FACTOR * width, sigma)).toBeCloseTo(
      FALLOFF_FLOOR,
      9
    );
  });
});

describe('divergence from the CURRENT line fragment shader (it is NOT pixel-neutral)', () => {
  /** What `line/shader-glsl.ts` actually draws at beta = 2. */
  const shifted = (p: number) =>
    Math.max(Math.exp(-FALLOFF_K * p * p) - FALLOFF_FLOOR, 0) * INV_ONE_MINUS_FALLOFF_FLOOR;
  /** What the volumetric primitive's side-on limit gives. */
  const bare = (p: number) => Math.exp(-FALLOFF_K * p * p);

  it('pins the measured relative divergence across the stroke', () => {
    const expected: Array<[number, number]> = [
      [0.25, 0.0034],
      [0.5, 0.0218],
      [0.75, 0.1246],
      [0.9, 0.411],
    ];
    for (const [p, rel] of expected) {
      expect((bare(p) - shifted(p)) / bare(p)).toBeCloseTo(rel, 3);
    }
    // At the sprite edge the current shader is EXACTLY zero (and discards
    // past it), while the volumetric primitive still carries the 1% floor.
    expect(shifted(1)).toBe(0);
    expect(bare(1)).toBeCloseTo(FALLOFF_FLOOR, 6);
  });

  it('pins the +1.46% / +1.71% cross-section mass excess', () => {
    const N = 200000;
    let mBare = 0;
    let mShifted = 0;
    for (let i = 0; i < N; i++) {
      const p = (i + 0.5) / N;
      mBare += bare(p);
      mShifted += shifted(p);
    }
    mBare /= N;
    mShifted /= N;
    expect((mBare - mShifted) / mShifted).toBeCloseTo(0.0146, 3);
    // The whole untruncated half-integral is analytic: sqrt(pi/K)/2.
    const full = 0.5 * Math.sqrt(Math.PI / FALLOFF_K);
    expect((full - mShifted) / mShifted).toBeCloseTo(0.0171, 3);
  });

  it('the side-on limit reproduces the BARE kernel, not the shifted one', () => {
    const width = 2;
    const sigma = lineSigmaFromAuthoredWidth(width);
    const drawn = LINE_RENDERED_HALF_WIDTH_FACTOR * width;
    for (const p of [0.25, 0.5, 0.75, 0.9]) {
      const g = raySegmentGeometry([300, p * drawn, 9], [0, 0, -1], [0, 0, 0], [600, 0, 0]);
      const value = lineRayIntegralRef({ sigma, ...g });
      expect(value).toBeCloseTo(bare(p), 7);
      expect(Math.abs(value - shifted(p))).toBeGreaterThan(1e-4); // and NOT the shifted one
    }
  });
});

describe('raySegmentGeometry', () => {
  const start: Vec3 = [0, 0, 0];
  const end: Vec3 = [10, 0, 0];

  it('derives |u|, D and s* for a broadside ray', () => {
    const g = raySegmentGeometry([5, 3, 7], [0, 0, -1], start, end);
    expect(g.length).toBeCloseTo(10, 12);
    expect(g.absU).toBeCloseTo(1, 12);
    expect(g.distanceToAxis).toBeCloseTo(3, 12);
    expect(g.sStar).toBeCloseTo(5, 12);
    expect(g.distanceToSegment).toBeCloseTo(3, 12);
  });

  it('derives the parallel case, where |u| is exactly zero', () => {
    const g = raySegmentGeometry([-20, 0, 4], [1, 0, 0], start, end);
    expect(g.absU).toBe(0);
    expect(g.distanceToAxis).toBeCloseTo(4, 12);
    expect(g.distanceToSegment).toBeCloseTo(4, 12);
  });

  it('clamps the closest approach into the segment for the capsule distance', () => {
    const g = raySegmentGeometry([12, 0, 7], [0, 0, -1], start, end);
    expect(g.sStar).toBeCloseTo(12, 12);
    expect(g.distanceToAxis).toBeCloseTo(0, 12);
    expect(g.distanceToSegment).toBeCloseTo(2, 12);
    expect(raySegmentDistance(g.length, g.distanceToAxis, g.sStar, g.absU)).toBeCloseTo(2, 12);
  });

  it('rejects degenerate inputs instead of inventing an axis', () => {
    expect(() => raySegmentGeometry([0, 0, 0], [0, 0, 1], start, start)).toThrow(/nonzero length/);
    expect(() => raySegmentGeometry([0, 0, 0], [0, 0, 0], start, end)).toThrow(/nonzero/);
  });

  it('exposes a float32-safe parallel epsilon, because 1e-7 is not one', () => {
    // 1e-7 is BELOW float32 machine epsilon, so a shader copying it gets
    // no guard at all. Pinned so the wiring slice cannot miss it.
    const FLOAT32_EPS = 1.1920928955078125e-7;
    expect(RAY_SEGMENT_PARALLEL_EPS).toBeLessThan(FLOAT32_EPS);
    expect(RAY_SEGMENT_PARALLEL_EPS_F32).toBeGreaterThan(FLOAT32_EPS * 1000);
  });
});

describe('lineRayIntegralRef vs an independent quadrature', () => {
  it('matches a brute-force integration of the density along the ray', () => {
    const sigma = 1;
    const start: Vec3 = [0, 0, 0];
    const end: Vec3 = [10, 0, 0];
    const cases: Array<{ origin: Vec3; dir: Vec3 }> = [
      { origin: [5, 0.5, 6], dir: [0, 0, -1] }, // broadside
      { origin: [5, 0.5, 6], dir: [0.5, 0.2, -1] }, // oblique
      { origin: [-3, 0.4, 0.3], dir: [1, 0.15, 0.05] }, // near end-on
      { origin: [11, 1.2, 0.9], dir: [-1, 0.3, 0.1] }, // past the far end
    ];
    for (const { origin, dir } of cases) {
      const g = raySegmentGeometry(origin, dir, start, end);
      const closed = lineRayIntegralRef({ sigma, ...g });
      const brute = quadratureIntegral(origin, dir, start, end, sigma);
      expect(closed).toBeCloseTo(brute, 5);
      expect(brute).toBeGreaterThan(1e-3); // sensitivity control
    }
  });
});

describe('SIDE-ON limit', () => {
  it('gives exactly exp(-D^2/(2 sigma^2)) for a long segment viewed broadside', () => {
    const sigma = 1;
    const length = 60; // L >> sigma, so the erf window saturates to 1
    for (const distance of [0, 0.25, 0.5, 1, 2, 3]) {
      const g = raySegmentGeometry(
        [length / 2, distance, 9],
        [0, 0, -1],
        [0, 0, 0],
        [length, 0, 0]
      );
      expect(g.absU).toBeCloseTo(1, 12);
      const value = lineRayIntegralRef({ sigma, ...g });
      // Tolerance is set by the toFixed(9) rounding of 1/sqrt(2), not by
      // the math: the closed form is exact here.
      expect(value).toBeCloseTo(radialGaussian(distance, sigma), 7);
    }
  });

  it('peaks at exactly 1.0 on the axis — the normalization convention', () => {
    const g = raySegmentGeometry([30, 0, 9], [0, 0, -1], [0, 0, 0], [60, 0, 0]);
    expect(lineRayIntegralRef({ sigma: 1, ...g })).toBeCloseTo(1.0, 7);
  });
});

describe('END-ON limit — the reason for the whole change', () => {
  it('gives L / (sigma*sqrt(2*pi)) * exp(-D^2/(2 sigma^2)) looking down the axis', () => {
    const sigma = 0.8;
    const length = 12;
    for (const distance of [0, 0.3, 0.8, 1.6]) {
      const g = raySegmentGeometry([-40, distance, 0], [1, 0, 0], [0, 0, 0], [length, 0, 0]);
      expect(g.absU).toBe(0);
      const expected = (length / (sigma * SQRT_2PI)) * radialGaussian(distance, sigma);
      expect(lineRayIntegralRef({ sigma, ...g })).toBeCloseTo(expected, 8);
    }
  });

  it('is finite and BRIGHTER than side-on by the honest chord factor', () => {
    const sigma = 1;
    const length = 12;
    const endOn = raySegmentGeometry([-40, 0, 0], [1, 0, 0], [0, 0, 0], [length, 0, 0]);
    const sideOn = raySegmentGeometry([length / 2, 0, 9], [0, 0, -1], [0, 0, 0], [length, 0, 0]);
    const endOnValue = lineRayIntegralRef({ sigma, ...endOn });
    const sideOnValue = lineRayIntegralRef({ sigma, ...sideOn });
    expect(Number.isFinite(endOnValue)).toBe(true);
    expect(sideOnValue).toBeCloseTo(1, 6);
    expect(endOnValue / sideOnValue).toBeCloseTo(length / (sigma * SQRT_2PI), 6);
  });
});

describe('ORIENTATION FREEDOM — sweeping through the end-on singularity', () => {
  const sigma = 1;
  const length = 20;
  const start: Vec3 = [0, 0, 0];
  const end: Vec3 = [length, 0, 0];
  const origin: Vec3 = [-50, 0.3, 0];

  /** Ray direction rotated by `theta` away from the segment axis. */
  const dirAt = (theta: number): Vec3 => [Math.cos(theta), Math.sin(theta), 0];

  /** Sample the integral over theta in [-0.5, 0.5] with `n` points. */
  function sweep(n: number): number[] {
    const samples: number[] = [];
    for (let i = 0; i < n; i++) {
      const theta = -0.5 + (1.0 * i) / (n - 1);
      samples.push(
        lineRayIntegralRef({ sigma, ...raySegmentGeometry(origin, dirAt(theta), start, end) })
      );
    }
    return samples;
  }

  const worstStep = (s: number[]) =>
    s.slice(1).reduce((worst, v, i) => Math.max(worst, Math.abs(v - s[i])), 0);

  it('the raw geometry IS discontinuous at |u| = 0 — the documented hazard', () => {
    const parallel = raySegmentGeometry(origin, dirAt(0), start, end);
    const nearly = raySegmentGeometry(origin, dirAt(1e-4), start, end);
    expect(parallel.distanceToAxis).toBeCloseTo(0.3, 12);
    expect(nearly.distanceToAxis).toBeLessThan(1e-9);
    expect(Math.abs(nearly.sStar)).toBeGreaterThan(1000);
  });

  it('stays finite, non-negative and below the absolute path-length bound', () => {
    // No ray integral through this primitive can exceed the full path
    // length through the peak density, L/(sigma*sqrt(2*pi)) — the D = 0
    // end-on value. The quad primitive this replaces has no such bound:
    // its screen-space area collapses to a sliver here, which is the
    // #1352 artifact.
    const cap = length / (sigma * SQRT_2PI);
    const samples = sweep(4001);
    for (const v of samples) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(cap * (1 + 1e-12));
    }
    expect(Math.max(...samples)).toBeGreaterThan(0.99 * cap);
    expect(Math.min(...samples)).toBeLessThan(1e-3);
    // (7 digits, not more: LINE_INV_SQRT2 is stored pre-rounded to its
    // toFixed(9) GLSL serialization, which costs ~2.5e-9 here.)
    const endOn = lineRayIntegralRef({
      sigma,
      ...raySegmentGeometry(origin, dirAt(0), start, end),
    });
    expect(endOn).toBeCloseTo(cap * radialGaussian(0.3, sigma), 7);
  });

  it('is Lipschitz across the sweep — refining the sampling halves the worst step', () => {
    // The diagnostic that separates "smooth" from "has a jump": for a
    // Lipschitz function the largest sample-to-sample step is
    // proportional to dtheta, so doubling the sample count halves it.
    // A branch seam or a 1/|u| blow-up would leave the step FLAT.
    const coarse = worstStep(sweep(2001));
    const fine = worstStep(sweep(4001));
    const finer = worstStep(sweep(8001));
    expect(fine / coarse).toBeCloseTo(0.5, 2);
    expect(finer / fine).toBeCloseTo(0.5, 2);
    expect(finer).toBeLessThan(0.05);
  });

  it('reaches the analytic parallel answer as theta -> 0, at a LEVER-DEPENDENT rate', () => {
    const parallel = lineRayIntegralRef({
      sigma,
      ...raySegmentGeometry(origin, dirAt(0), start, end),
    });
    expect(parallel).toBeCloseTo((length / (sigma * SQRT_2PI)) * radialGaussian(0.3, sigma), 7);

    // The approach is linear in theta, but the SLOPE is not a property of
    // the primitive — it scales with the axial lever arm between the ray
    // origin and the segment (~2 * lever * offset * k^2, k = 1/(sigma*sqrt2)).
    // This 50-unit origin gives ~150; do not read that as a universal
    // constant. The lever dependence is asserted below.
    for (const theta of [1e-3, 1e-4, 1e-5, RAY_SEGMENT_PARALLEL_EPS * 2, -1e-5, -1e-4, -1e-3]) {
      const g = raySegmentGeometry(origin, dirAt(theta), start, end);
      const error = Math.abs(lineRayIntegralRef({ sigma, ...g }) - parallel);
      expect(error).toBeLessThan(160 * Math.abs(theta));
    }
  });

  it('the Lipschitz constant grows with the lever arm — it is NOT modest in general', () => {
    const slopeAt = (lever: number) => {
      const o: Vec3 = [-lever, 0.3, 0];
      const f = (t: number) =>
        lineRayIntegralRef({ sigma, ...raySegmentGeometry(o, dirAt(t), start, end) });
      return Math.abs(f(1e-6) - f(0)) / 1e-6;
    };
    const s50 = slopeAt(50);
    const s500 = slopeAt(500);
    const s5000 = slopeAt(5000);
    // Ten times the lever, ten times the sensitivity (to within the
    // constant offset the segment's own half-length contributes).
    expect(s500 / s50).toBeGreaterThan(5);
    expect(s5000 / s500).toBeGreaterThan(8);
    expect(s5000).toBeGreaterThan(10000);
  });

  it('pins the RAY_SEGMENT_PARALLEL_EPS seam from BOTH sides', () => {
    // The eps switches the DERIVATION, not a math lane: below it the
    // helper uses the plain perpendicular of w0 instead of the mutual
    // perpendicular m-hat, whose relative accuracy decays like eps/|u|.
    // Both D and s* jump across that switch, but their contributions
    // cancel in the product the integral actually consumes
    // (D^2 + (s*|u|)^2 -> |w0_perp|^2 either way), so the seam is only
    // the O(|u|) remainder — and that remainder scales with the same
    // lever * offset * k^2 as the slope above.
    //
    // Every other test here steps theta by >= 2.5e-4 and so straddles
    // the guard band without ever landing inside it; these samples sit
    // one part in 1000 either side of it.
    const eps = RAY_SEGMENT_PARALLEL_EPS;
    const relStep = (lever: number, s: number) => {
      const o: Vec3 = [-lever, 0.3 * s, 0];
      const f = (t: number) =>
        lineRayIntegralRef({ sigma: s, ...raySegmentGeometry(o, dirAt(t), start, end) });
      const below = f(eps * 0.999); // parallel derivation
      const above = f(eps * 1.001); // oblique derivation
      return Math.abs(above - below) / below;
    };

    // At any framing a renderer will actually produce, the two
    // derivations are indistinguishable. Two-sided, because a zero step
    // would mean the guard band had stopped being reachable at all.
    const modest = relStep(50, 1);
    expect(modest).toBeGreaterThan(1e-6); // measured 1.80e-6
    expect(modest).toBeLessThan(3e-6);
    const long = relStep(1e4, 1);
    expect(long).toBeGreaterThan(2e-4); // measured 3.01e-4
    expect(long).toBeLessThan(4e-4);

    // The documented worst case is NOT small, and is pinned rather than
    // papered over: a long axial lever arm against a thin segment
    // amplifies the remainder without bound. Neither branch is the
    // better one here — the oblique formula's own cross-product accuracy
    // decays as |u| shrinks, so no choice of eps removes this; a
    // consumer with a lever arm this long has to widen its own guard.
    // (float32 cannot even represent this regime — see
    // RAY_SEGMENT_PARALLEL_EPS_F32.) The band is tight on BOTH sides so
    // that "simplifying" the parallel branch's s* to zero — the reading
    // its own docblock warns against, which would return the exactly-
    // parallel answer across the whole guard band — fails here (0.93)
    // instead of passing quietly.
    const worst = relStep(1e6, 0.05);
    expect(worst).toBeGreaterThan(0.4); // measured 4.56e-1
    expect(worst).toBeLessThan(0.5);
  });
});

/**
 * Build params that put the erf window at a chosen gap and midpoint.
 *
 * `gap = L·|u|/(σ√2)` and `M = (s* − L/2)·|u|/(σ√2)`, so fixing σ = 1,
 * |u| = 1 and D = 0 makes `L = gap·√2` and `s* = L/2 + M·√2` — a direct
 * handle on the two quantities the lane split is keyed on.
 */
function paramsAtGap(gap: number, m: number): LineRayIntegralParams {
  const sigma = 1;
  const length = gap / LINE_INV_SQRT2;
  return { sigma, length, distanceToAxis: 0, sStar: length / 2 + m / LINE_INV_SQRT2, absU: 1 };
}

describe('the Taylor/closed-form lane threshold', () => {
  /** Strip the common `radial · axialScale` factor to expose ½·DQ. */
  const halfDq = (v: number, p: LineRayIntegralParams) =>
    v / ((p.length * LINE_INV_SQRT2) / p.sigma);

  it('crosses at LINE_WINDOW_GAP_THRESHOLD, and both lanes are reachable', () => {
    expect(LINE_WINDOW_GAP_THRESHOLD).toBe(0.5);
    // A hair below the threshold takes the derivative lane, a hair above
    // takes the closed form — verified by the fact that the derivative
    // lane ignores `erfPoly` entirely, so the Ref/Poly split vanishes
    // below the threshold and reappears above it.
    const below = paramsAtGap(0.5 - 1e-6, 0);
    const above = paramsAtGap(0.5 + 1e-6, 0);
    expect(lineRayIntegralPoly(below)).toBe(lineRayIntegralRef(below));
    expect(lineRayIntegralPoly(above)).not.toBe(lineRayIntegralRef(above));
  });

  it('agrees across the threshold to 7e-6 absolute in DQ (accurate erf)', () => {
    const delta = 1e-9;
    let worst = 0;
    for (let i = 0; i <= 2000; i++) {
      const m = -5 + (10 * i) / 2000;
      const lo = paramsAtGap(LINE_WINDOW_GAP_THRESHOLD - delta, m);
      const hi = paramsAtGap(LINE_WINDOW_GAP_THRESHOLD + delta, m);
      worst = Math.max(
        worst,
        Math.abs(halfDq(lineRayIntegralRef(lo), lo) - halfDq(lineRayIntegralRef(hi), hi)) * 2
      );
    }
    expect(worst).toBeLessThan(7.5e-6);
    // Sensitivity control: the two lanes are genuinely different
    // computations, so the step is not identically zero.
    expect(worst).toBeGreaterThan(1e-7);
  });

  it('agrees across the threshold to 1e-4 RELATIVE where DQ is not tiny', () => {
    const delta = 1e-9;
    let worst = 0;
    for (let i = 0; i <= 2000; i++) {
      const m = -2 + (4 * i) / 2000; // DQ >= 1.128*exp(-4) ~ 0.021 here
      const lo = paramsAtGap(LINE_WINDOW_GAP_THRESHOLD - delta, m);
      const hi = paramsAtGap(LINE_WINDOW_GAP_THRESHOLD + delta, m);
      const a = halfDq(lineRayIntegralRef(lo), lo);
      const b = halfDq(lineRayIntegralRef(hi), hi);
      worst = Math.max(worst, Math.abs(a - b) / Math.abs(b));
    }
    expect(worst).toBeLessThan(1e-4); // documented: 4.5e-5
  });

  it('on the shader path the crossing step is erfPoly’s own error, ~1.45e-3', () => {
    const delta = 1e-9;
    let worst = 0;
    for (let i = 0; i <= 2000; i++) {
      const m = -5 + (10 * i) / 2000;
      const lo = paramsAtGap(LINE_WINDOW_GAP_THRESHOLD - delta, m);
      const hi = paramsAtGap(LINE_WINDOW_GAP_THRESHOLD + delta, m);
      worst = Math.max(
        worst,
        Math.abs(halfDq(lineRayIntegralPoly(lo), lo) - halfDq(lineRayIntegralPoly(hi), hi)) * 2
      );
    }
    expect(worst).toBeLessThan(2e-3);
    // ...and genuinely dominated by the polynomial, not by the lane
    // design: two orders of magnitude above the accurate path's 7e-6.
    expect(worst).toBeGreaterThan(5e-4);
  });

  it('the shader mirror tracks the reference to ~5.2e-4 everywhere', () => {
    let worst = 0;
    for (let gi = 0; gi <= 60; gi++) {
      for (let mi = 0; mi <= 60; mi++) {
        const p = paramsAtGap((6 * gi) / 60 + 1e-4, -5 + (10 * mi) / 60);
        worst = Math.max(worst, Math.abs(lineRayIntegralPoly(p) - lineRayIntegralRef(p)));
      }
    }
    // Measured 5.21e-4; the band is tight enough that a coefficient or
    // lane change moves it out, and two-sided so a mirror that silently
    // became the reference would fail too.
    expect(worst).toBeLessThan(8e-4);
    expect(worst).toBeGreaterThan(3e-4);
  });
});

describe('non-negativity and finiteness over a dense sweep', () => {
  it('never returns a negative or non-finite value', () => {
    // erfPoly can produce a slightly NEGATIVE erf window near saturation
    // (erf.ts documents -8.8e-4). A negative coverage weight would
    // DARKEN a pixel another segment already lit, so the clamp is
    // load-bearing, and this sweep drives straight through the region.
    let sawNearZero = false;
    let nonFinite = 0;
    let worstNegative = Infinity;
    let samples = 0;
    for (const sigma of [0.05, 0.5, 1, 4]) {
      for (const length of [0.01, 0.3, 2, 25, 400]) {
        for (const distance of [0, 0.7, 3, 40]) {
          for (let si = 0; si <= 24; si++) {
            const sStar = -3 * length + (7 * length * si) / 24;
            for (let ui = 0; ui <= 24; ui++) {
              const absU = ui / 24;
              const p = { sigma, length, distanceToAxis: distance, sStar, absU };
              for (const value of [lineRayIntegralRef(p), lineRayIntegralPoly(p)]) {
                // Accumulate rather than assert per sample: 100k expect()
                // calls dominate the runtime and report nothing extra.
                if (!Number.isFinite(value)) nonFinite++;
                worstNegative = Math.min(worstNegative, value);
                if (value < 1e-9) sawNearZero = true;
                samples++;
              }
            }
          }
        }
      }
    }
    expect(samples).toBeGreaterThan(50000);
    expect(nonFinite).toBe(0);
    expect(worstNegative).toBeGreaterThanOrEqual(0);
    // The sweep really does reach the saturated tail where the unclamped
    // window would have gone negative.
    expect(sawNearZero).toBe(true);
  });

  it('is exactly zero, not negative, far past the segment ends (CLOSED lane)', () => {
    const p = { sigma: 1, length: 4, distanceToAxis: 0, sStar: 1000, absU: 1 };
    // gap = 2.83 > 0.5, so this is the closed lane: the erf window
    // saturates and the result is exactly 0.
    expect(p.length * p.absU * LINE_INV_SQRT2).toBeGreaterThan(LINE_WINDOW_GAP_THRESHOLD);
    expect(lineRayIntegralPoly(p)).toBe(0);
    expect(lineRayIntegralRef(p)).toBe(0);
  });

  it('in the DERIVATIVE lane it is ~1e-45 rather than exactly 0 — the mm clamp', () => {
    // The float64 counterpart of the float32 no-op property: 104 is the
    // float32 underflow point of exp(-M^2), but float64 has ~1000x more
    // exponent range, so clamping M^2 there leaves the CPU reference a
    // denormal-scale residue where the unclamped math is exactly 0. The
    // honest thing is to pin the behaviour, not to claim it away — the
    // magnitude (~1e-45 against a side-on peak of 1.0) is 30 orders
    // below anything a renderer can express.
    const p = { sigma: 1, length: 1, distanceToAxis: 0, sStar: 100, absU: 0.7 };
    expect(p.length * p.absU * LINE_INV_SQRT2).toBeLessThan(LINE_WINDOW_GAP_THRESHOLD);
    for (const value of [lineRayIntegralRef(p), lineRayIntegralPoly(p)]) {
      expect(value).toBeGreaterThan(0); // NOT exactly zero
      expect(value).toBeLessThan(1e-40); // documented: 2.85e-45
    }
  });

  it('the TS mirror keeps the Taylor arm finite even when float64 would overflow', () => {
    // The float64 counterpart of the float32 defect below: without the
    // min-clamps on the Taylor arm's half-gap and M^2, the UNSELECTED
    // arm reaches Inf and `mix`'s `Inf * 0` poisons a correct closed-lane
    // result. float64 has ~150 decades more headroom than float32, so
    // this needs an absurd sigma to trigger — the point is that the
    // guards are structurally present in the mirror, not just the GLSL.
    const p = { sigma: 1e-150, length: 1, distanceToAxis: 0, sStar: 1, absU: 1 };
    expect(Number.isFinite(lineRayIntegralPoly(p))).toBe(true);
    expect(Number.isFinite(lineRayIntegralRef(p))).toBe(true);
  });
});

describe('float32 overflow of the unselected mix arm (the #1352 fp32 defect)', () => {
  // The transpiled-GLSL parity test below evaluates the shader body in
  // float64 and is STRUCTURALLY BLIND to overflow. This block re-runs the
  // same GLSL with a float32 rounding after every statement, and proves
  // by mutation that the two `min` guards are what keep it finite.
  //
  // The two guards are mutated SEPARATELY. Removing both at once is a
  // disjunction: either guard alone rescues the headline reproducer, so
  // a combined mutant would leave a single-guard regression green. They
  // protect different regimes (astronomical `gap` vs `|s*| >> L`) and
  // each gets its own reproducer below.
  const GLSL = GLSL_ERF_FUNCTIONS + GLSL_LINE_RAY_INTEGRAL_FUNCTIONS;
  const GAP_GUARD = `min(gap, ${LINE_WINDOW_GAP_THRESHOLD.toFixed(9)})`;
  const MM_GUARD = `min(m * m, ${LINE_TAYLOR_MM_CLAMP.toFixed(9)})`;
  const guarded = transpileGLSL(GLSL, true);
  const noGapGuard = transpileGLSL(GLSL.replace(GAP_GUARD, 'gap'), true);
  const noMmGuard = transpileGLSL(GLSL.replace(MM_GUARD, 'm * m'), true);
  const noGuards = transpileGLSL(GLSL.replace(GAP_GUARD, 'gap').replace(MM_GUARD, 'm * m'), true);

  /** Headline reproducer: authored width 0.0017, a 1000-unit segment, broadside. */
  const REPRO = { sigma: 0.0033, length: 1000, distanceToAxis: 0, sStar: 817, absU: 1 };
  /** Half-gap guard: astronomical `gap`. Only that guard rescues it. */
  const GAP_REPRO = { sigma: 1, length: 1e10, distanceToAxis: 0, sStar: 0.82e10, absU: 1 };
  /** M-squared guard: |s*| >> L, inside the SELECTED derivative lane. */
  const MM_REPRO = { sigma: 1, length: 1, distanceToAxis: 0, sStar: 1e11, absU: 0.7 };
  const call = (g: TranspiledGLSL, p: LineRayIntegralParams) =>
    g.luxarLineRayIntegral(p.sigma, p.length, p.distanceToAxis, p.sStar, p.absU);

  it('the mutation controls really do produce NaN — the block is sensitive', () => {
    // If either of these starts passing, the corresponding string
    // replacement stopped matching (the guard was renamed or removed)
    // and the mutants below have quietly stopped proving anything.
    expect(Number.isNaN(call(noGuards, REPRO))).toBe(true);
    expect(Number.isNaN(call(noMmGuard, MM_REPRO))).toBe(true);
    expect(Number.isNaN(call(noGapGuard, GAP_REPRO))).toBe(true);
  });

  it('the shipped GLSL returns the exact answer (1.0) where the mutant NaNs', () => {
    expect(call(guarded, REPRO)).toBeCloseTo(1.0, 5);
  });

  it('the HALF-GAP guard alone is what survives an astronomical gap', () => {
    // Removing only the mm clamp leaves this finite, so the combined
    // mutant could never have pinned this guard.
    expect(call(guarded, GAP_REPRO)).toBeCloseTo(1.0, 5);
    expect(Number.isNaN(call(noGapGuard, GAP_REPRO))).toBe(true);
    expect(Number.isNaN(call(noMmGuard, GAP_REPRO))).toBe(false);
  });

  it('onset is a pure aspect ratio L/sigma, and the guard removes it entirely', () => {
    const at = (ratio: number) => ({
      sigma: 1,
      length: ratio,
      distanceToAxis: 0,
      sStar: 0.82 * ratio,
      absU: 1,
    });
    // The onsets pinned here are THIS harness's, and they are LATE
    // estimates: rounding once per statement lets the
    // `q*q*(4*mm*mm - ...)` product form in float64 before it is
    // rounded, whereas a GPU rounds every operation. A per-operation
    // float32 model puts the both-guards-gone onset near L/sigma ~ 1.6e5
    // rather than 3.0e5 — so the checked-in test under-detects, never
    // over-detects.
    expect(Number.isNaN(call(noGuards, at(2.8e5)))).toBe(false);
    expect(Number.isNaN(call(noGuards, at(3.0e5)))).toBe(true);
    // With the mm clamp still in place the polynomial stays bounded far
    // longer, which is why the half-gap guard needs its own reproducer.
    expect(Number.isNaN(call(noGapGuard, at(1e9)))).toBe(false);
    expect(Number.isNaN(call(noGapGuard, at(1e10)))).toBe(true);
    for (const ratio of [1e5, 2.8e5, 3.0e5, 1e6, 1e10]) {
      expect(call(guarded, at(ratio))).toBeCloseTo(1.0, 5);
    }
  });

  it('the M-SQUARED guard fires in the SELECTED lane, where nothing else helps', () => {
    // gap = 1 * 0.7 * (1/sqrt2) = 0.495 <= 0.5, so the derivative lane is
    // the one being returned — this is not an unselected-arm artifact.
    const gap = MM_REPRO.length * MM_REPRO.absU * LINE_INV_SQRT2;
    expect(gap).toBeLessThan(LINE_WINDOW_GAP_THRESHOLD);
    // Shipped: a denormal ~2.8e-45 (the ray is 1e11 units past the end,
    // so ~0 is the right answer). mm-inert: NaN.
    const shipped = call(guarded, MM_REPRO);
    expect(Number.isFinite(shipped)).toBe(true);
    expect(shipped).toBeGreaterThanOrEqual(0);
    expect(shipped).toBeLessThan(1e-40);
    expect(Number.isNaN(call(noMmGuard, MM_REPRO))).toBe(true);
    // The half-gap guard cannot substitute for it.
    expect(Number.isNaN(call(noGapGuard, MM_REPRO))).toBe(false);
  });

  it('LINE_TAYLOR_MM_CLAMP sits exactly on the float32 exp underflow point', () => {
    // The VALUE, not just the presence, of the clamp. Bracketing it from
    // both sides is what makes a wrong value (too small -> a live change
    // to the math; too large -> the guard is structurally there but
    // semantically inert) fail rather than pass silently.
    expect(Math.fround(Math.exp(-LINE_TAYLOR_MM_CLAMP))).toBe(0);
    expect(Math.fround(Math.exp(-(LINE_TAYLOR_MM_CLAMP - 1)))).toBeGreaterThan(0);
  });

  it('is clean over a float32 grid where the mutant is not, and is bit-neutral', () => {
    let mutantBad = 0;
    let guardedBad = 0;
    let worstDiff = 0;
    let total = 0;
    for (const sigma of [0.0033, 0.01, 0.05, 0.5, 1, 4]) {
      for (const length of [0.01, 1, 25, 400, 1000, 20000]) {
        for (const distanceToAxis of [0, 0.7, 3, 40, 100]) {
          for (let si = 0; si < 5; si++) {
            const sStar = -length + (2.5 * length * si) / 4;
            for (let ui = 0; ui <= 8; ui++) {
              const p = { sigma, length, distanceToAxis, sStar, absU: ui / 8 };
              const a = call(noGuards, p);
              const b = call(guarded, p);
              total++;
              if (!Number.isFinite(a) || a < 0) mutantBad++;
              if (!Number.isFinite(b) || b < 0) guardedBad++;
              if (Number.isFinite(a) && Number.isFinite(b)) {
                worstDiff = Math.max(worstDiff, Math.abs(a - b));
              }
            }
          }
        }
      }
    }
    expect(total).toBe(8100);
    // Exact, not a floor: a 22% drift in this number went unnoticed once.
    expect(mutantBad).toBe(565);
    expect(guardedBad).toBe(0);
    // Bit-neutral: min(x, c) === x throughout the regime where the arm is
    // selected, so the guard changes nothing that was already correct.
    expect(worstDiff).toBe(0);
  });

  it('tracks the float64 mirror on benign inputs (the harness is the same math)', () => {
    for (const p of [
      { sigma: 1, length: 10, distanceToAxis: 0.5, sStar: 5, absU: 0.8 },
      { sigma: 0.4, length: 3, distanceToAxis: 0.2, sStar: 1.1, absU: 0.05 },
      { sigma: 2, length: 40, distanceToAxis: 1.5, sStar: 30, absU: 1 },
    ]) {
      expect(call(guarded, p)).toBeCloseTo(lineRayIntegralPoly(p), 5);
    }
  });
});

describe('ADDITIVITY — splitting a segment telescopes exactly', () => {
  const sigma = 1;
  const length = 14;
  const start: Vec3 = [0, 0, 0];
  const end: Vec3 = [length, 0, 0];

  /** Directions from broadside down to (and through) end-on. */
  const directions: Vec3[] = [
    [0, 0, -1],
    [0.3, 0.2, -1],
    [1, 0.6, -0.4],
    [1, 0.1, 0.05],
    [1, 0.002, 0],
    [1, 0, 0],
  ];
  const origins: Vec3[] = [
    [7, 0.6, 5],
    [-6, 1.1, 0.4],
    [20, 0.2, 0.9],
  ];
  const cuts = [0.001, 1, 7, 13, 13.999];

  /** Worst |half1 + half2 - whole| over the whole grid, for one evaluator. */
  function worstResidual(evaluate: (p: LineRayIntegralParams) => number): number {
    let worst = 0;
    for (const dir of directions) {
      for (const origin of origins) {
        for (const cut of cuts) {
          const mid: Vec3 = [cut, 0, 0];
          const whole = evaluate({ sigma, ...raySegmentGeometry(origin, dir, start, end) });
          const first = evaluate({ sigma, ...raySegmentGeometry(origin, dir, start, mid) });
          const second = evaluate({ sigma, ...raySegmentGeometry(origin, dir, mid, end) });
          worst = Math.max(worst, Math.abs(first + second - whole));
        }
      }
    }
    return worst;
  }

  it('two collinear halves sum to the whole, for every ray direction (erfRef)', () => {
    // Lane crossings are the only source of error on the accurate path: a
    // short offcut can land in the derivative lane while the whole
    // segment is in the closed form, and the two lanes agree only to
    // their documented tolerance. Everything else telescopes identically.
    const worst = worstResidual(lineRayIntegralRef);
    expect(worst).toBeLessThan(2e-6);
    // Two-sided: the residual is real (measured 6.1e-7), not a
    // vacuously-satisfied bound with 30x of headroom.
    expect(worst).toBeGreaterThan(1e-7);
  });

  it('pins the SHADER-path seam magnitude too (erfPoly): ~5.2e-4', () => {
    // What a real fragment will actually leave at a joint. Two sources:
    // erfPoly's own 5.4e-4-per-evaluation error, which no longer cancels
    // once the window is split, and the max(0, .) window clamp, which is
    // not additive by construction. Documenting it is better than
    // discovering it in a screenshot.
    const worst = worstResidual(lineRayIntegralPoly);
    expect(worst).toBeLessThan(1e-3);
    expect(worst).toBeGreaterThan(1e-4); // it is real, ~5.2e-4
    // ... and a factor ~25 worse than the accurate path, so the seam is
    // the polynomial's, not the decomposition's.
    expect(worst / worstResidual(lineRayIntegralRef)).toBeGreaterThan(5);
  });

  it('telescopes to machine precision when every piece stays in one lane', () => {
    const dir: Vec3 = [0, 0, -1];
    const origin: Vec3 = [6, 0.7, 5];
    const mid: Vec3 = [6, 0, 0];
    const whole = lineRayIntegralRef({ sigma, ...raySegmentGeometry(origin, dir, start, end) });
    const first = lineRayIntegralRef({ sigma, ...raySegmentGeometry(origin, dir, start, mid) });
    const second = lineRayIntegralRef({ sigma, ...raySegmentGeometry(origin, dir, mid, end) });
    expect(first + second).toBeCloseTo(whole, 12);
    // Control: neither half is the whole thing on its own.
    expect(first).toBeGreaterThan(0.05 * whole);
    expect(second).toBeGreaterThan(0.05 * whole);
  });

  it('telescopes across a five-way split too', () => {
    const dir: Vec3 = [1, 0.35, -0.2];
    const origin: Vec3 = [3, 1.4, 2];
    const splits = [0, 2.5, 5.5, 9, 11.5, length];
    let sum = 0;
    for (let i = 0; i < splits.length - 1; i++) {
      const a: Vec3 = [splits[i], 0, 0];
      const b: Vec3 = [splits[i + 1], 0, 0];
      sum += lineRayIntegralRef({ sigma, ...raySegmentGeometry(origin, dir, a, b) });
    }
    const whole = lineRayIntegralRef({ sigma, ...raySegmentGeometry(origin, dir, start, end) });
    expect(sum).toBeCloseTo(whole, 9);
    expect(whole).toBeGreaterThan(1e-3);
  });
});

describe('lineCapsuleProfileRef (peak / max-blend modes)', () => {
  const sigma = 1.5;
  const start: Vec3 = [0, 0, 0];
  const end: Vec3 = [10, 0, 0];

  it('equals the radial Gaussian for a perpendicular ray mid-segment', () => {
    for (const distance of [0, 0.5, 1.5, 3]) {
      const g = raySegmentGeometry([5, distance, 8], [0, 0, -1], start, end);
      expect(lineCapsuleProfileRef(sigma, g.distanceToSegment)).toBeCloseTo(
        radialGaussian(distance, sigma),
        7
      );
    }
  });

  it('is exactly 1 on the segment and rounds off past the ends (spherical cap)', () => {
    const onAxis = raySegmentGeometry([5, 0, 8], [0, 0, -1], start, end);
    expect(lineCapsuleProfileRef(sigma, onAxis.distanceToSegment)).toBeCloseTo(1, 12);

    for (const overshoot of [0.5, 1.5, 3]) {
      const past = raySegmentGeometry([10 + overshoot, 0, 8], [0, 0, -1], start, end);
      expect(past.distanceToSegment).toBeCloseTo(overshoot, 10);
      expect(lineCapsuleProfileRef(sigma, past.distanceToSegment)).toBeCloseTo(
        radialGaussian(overshoot, sigma),
        7
      );
    }
    // The cap is spherical, not a flat disc: a ray grazing at radius 2
    // from the endpoint gives the same value from EVERY direction.
    for (const degrees of [0, 30, 45, 60, 90]) {
      const phi = (degrees * Math.PI) / 180;
      const grazing: Vec3 = [10 + 2 * Math.cos(phi), 0, 2 * Math.sin(phi)];
      const g = raySegmentGeometry(grazing, [0, 1, 0], start, end);
      expect(g.distanceToSegment).toBeCloseTo(2, 10);
      expect(lineCapsuleProfileRef(sigma, g.distanceToSegment)).toBeCloseTo(
        radialGaussian(2, sigma),
        9
      );
    }
  });

  it('stays continuous through end-on, where the axis distance does not', () => {
    // distanceToAxis jumps from 0.3 to 0; distanceToSegment does not
    // move. Convergence is linear in the sweep angle (the origin's
    // 50-unit lever arm), so the near-parallel sample lands at 0.3005.
    const origin: Vec3 = [-50, 0.3, 0];
    const parallel = raySegmentGeometry(origin, [1, 0, 0], start, end);
    const nearly = raySegmentGeometry(origin, [Math.cos(1e-5), Math.sin(1e-5), 0], start, end);
    expect(parallel.distanceToAxis).toBeCloseTo(0.3, 12);
    expect(nearly.distanceToAxis).toBeLessThan(1e-9); // the jump
    expect(parallel.distanceToSegment).toBeCloseTo(0.3, 12);
    expect(nearly.distanceToSegment).toBeCloseTo(0.3, 3); // no jump
    expect(lineCapsuleProfileRef(sigma, parallel.distanceToSegment)).toBeCloseTo(
      lineCapsuleProfileRef(sigma, nearly.distanceToSegment),
      3
    );
  });

  it('agrees with the sum mode’s side-on shape (same sigma, same stroke)', () => {
    const g = raySegmentGeometry([5, 1.2, 8], [0, 0, -1], start, end);
    const peak = lineCapsuleProfileRef(sigma, g.distanceToSegment);
    const sum = lineRayIntegralRef({
      sigma,
      ...raySegmentGeometry([30, 1.2, 8], [0, 0, -1], [0, 0, 0], [60, 0, 0]),
    });
    expect(peak).toBeCloseTo(sum, 6);
  });
});

describe('GLSL / TSL single-source serialization', () => {
  it('embeds every constant with the exact toFixed(9) literal', () => {
    for (const c of [
      LINE_INV_SQRT2,
      LINE_TWO_OVER_SQRT_PI,
      LINE_WINDOW_GAP_THRESHOLD,
      LINE_WINDOW_TAYLOR_C2,
      LINE_WINDOW_TAYLOR_C4,
      LINE_TAYLOR_MM_CLAMP,
    ]) {
      expect(GLSL_LINE_RAY_INTEGRAL_FUNCTIONS).toContain(c.toFixed(9));
    }
  });

  it('declares the three entry points and does NOT redeclare luxarErf', () => {
    expect(GLSL_LINE_RAY_INTEGRAL_FUNCTIONS).toContain(
      'float luxarLineRayIntegral(float sigma, float segLength, float distanceToAxis, float sStar, float absU)'
    );
    expect(GLSL_LINE_RAY_INTEGRAL_FUNCTIONS).toContain('float luxarLineCapsuleProfile(');
    expect(GLSL_LINE_RAY_INTEGRAL_FUNCTIONS).toContain('float luxarRaySegmentDistance(');
    // It CALLS luxarErf but must not define it — a shader may already
    // have injected GLSL_ERF_FUNCTIONS for another reason.
    expect(GLSL_LINE_RAY_INTEGRAL_FUNCTIONS).toContain('luxarErf(');
    expect(GLSL_LINE_RAY_INTEGRAL_FUNCTIONS).not.toContain('float luxarErf(');
  });

  it('carries all three finiteness guards on the two mix arms', () => {
    expect(GLSL_LINE_RAY_INTEGRAL_FUNCTIONS).toContain(
      `max(gap, ${LINE_WINDOW_GAP_THRESHOLD.toFixed(9)})`
    );
    expect(GLSL_LINE_RAY_INTEGRAL_FUNCTIONS).toContain(
      `min(gap, ${LINE_WINDOW_GAP_THRESHOLD.toFixed(9)})`
    );
    expect(GLSL_LINE_RAY_INTEGRAL_FUNCTIONS).toContain(
      `min(m * m, ${LINE_TAYLOR_MM_CLAMP.toFixed(9)})`
    );
  });

  it('builds the TSL graphs without throwing (API smoke)', () => {
    const node = lineRayIntegralTSL({
      sigma: float(1),
      length: float(10),
      distanceToAxis: float(0.5),
      sStar: float(5),
      absU: float(0.8),
    });
    expect(node).toBeDefined();
    expect(typeof node).toBe('object');
    expect(lineCapsuleProfileTSL(float(1), float(0.5))).toBeDefined();
    expect(raySegmentDistanceTSL(float(10), float(0.5), float(5), float(0.8))).toBeDefined();
  });

  it('the GLSL bodies, transpiled to JS, are bit-identical to the TS mirrors', () => {
    // Same proof erf.test.ts uses: mechanically rewrite the GLSL as JS
    // and evaluate it. A structural drift between the GLSL and the
    // mirror — a reordered statement, a lane written differently, a
    // literal typed by hand — shows up as a non-zero difference.
    const glsl = transpileGLSL(GLSL_ERF_FUNCTIONS + GLSL_LINE_RAY_INTEGRAL_FUNCTIONS, false);

    for (let i = 0; i <= 40; i++) {
      const sigma = 0.05 + (5 * i) / 40;
      for (let j = 0; j <= 40; j++) {
        const length = 0.02 + (30 * j) / 40;
        for (let k = 0; k <= 8; k++) {
          const absU = k / 8;
          const sStar = -length + (3 * length * (k % 5)) / 4;
          const distanceToAxis = (k % 3) * 0.9;
          expect(glsl.luxarLineRayIntegral(sigma, length, distanceToAxis, sStar, absU)).toBe(
            lineRayIntegralPoly({ sigma, length, distanceToAxis, sStar, absU })
          );
          expect(glsl.luxarRaySegmentDistance(length, distanceToAxis, sStar, absU)).toBe(
            raySegmentDistance(length, distanceToAxis, sStar, absU)
          );
          expect(glsl.luxarLineCapsuleProfile(sigma, distanceToAxis)).toBe(
            lineCapsuleProfileRef(sigma, distanceToAxis)
          );
        }
      }
    }
  });
});

/** GLSL shape used by the transpiled-source parity and fp32 checks. */
interface TranspiledGLSL {
  luxarLineRayIntegral: (
    sigma: number,
    segLength: number,
    distanceToAxis: number,
    sStar: number,
    absU: number
  ) => number;
  luxarRaySegmentDistance: (
    segLength: number,
    distanceToAxis: number,
    sStar: number,
    absU: number
  ) => number;
  luxarLineCapsuleProfile: (sigma: number, distanceToSegment: number) => number;
}

/**
 * Mechanically rewrite a GLSL function block as JavaScript: signatures to
 * `function`, `float x =` to `let x =`, builtins to `Math.*`, and `mix` /
 * `step` to their GLSL-semantics equivalents.
 *
 * With `float32`, every statement's value is rounded through
 * `Math.fround`. That is statement granularity, not per-operation, so it
 * models float32 range CONSERVATIVELY — a product that would overflow on
 * a GPU can still form in float64 inside one statement, which pushes the
 * detected onset later than the real one (see the aspect-ratio test) —
 * and float32 PRECISION only approximately. Good enough to catch an
 * `Inf`/`NaN` class defect, never a 1-ulp one.
 */
function transpileGLSL(source: string, float32: boolean): TranspiledGLSL {
  const wrap = (expr: string) => (float32 ? `Math.fround(${expr})` : expr);
  const body = source
    .replace(
      /float\s+(\w+)\s*\(([^)]*)\)\s*\{/g,
      (_match, name: string, args: string) => `function ${name}(${args.replaceAll('float ', '')}) {`
    )
    .replaceAll('float ', 'let ')
    .replace(/\b(min|max|abs|exp|sqrt)\(/g, 'Math.$1(')
    .replace(/\blet (\w+) = ([\s\S]*?);/g, (_m, name: string, expr: string) => {
      return `let ${name} = ${wrap(expr)};`;
    })
    .replace(/\breturn ([\s\S]*?);/g, (_m, expr: string) => `return ${wrap(expr)};`);

  const factory = new Function(`
    const mix = (a, b, t) => ${wrap('a * (1 - t) + b * t')};
    const step = (edge, x) => (x < edge ? 0 : 1);
    ${body}
    return { luxarLineRayIntegral, luxarRaySegmentDistance, luxarLineCapsuleProfile };
  `) as () => TranspiledGLSL;
  return factory();
}
