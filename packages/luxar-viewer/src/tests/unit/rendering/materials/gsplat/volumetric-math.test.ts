/**
 * Volumetric-mode math invariants (VOLUMETRIC_BLENDING_SPEC.md §3),
 * checked in pure TS against the exact formulas the shader branch uses
 * (shader-glsl.ts LUXAR_VOLUMETRIC / shader-tsl.ts volumetric branch):
 *
 *   τ      = κ · opacity · intensity
 *   α      = 1 − e^(−τ)
 *   S(τ)   = series (τ < 1e-3) else α / max(τ, 1e-20)
 *   output = (emission · S(τ), α), composited src + (1 − α_src)·dst.
 *
 * I1 (κ = 0 ≡ additive) is pinned by the E2E pixel-compare; I2
 * (split-splat multiplicativity) and the S(τ) series seam live here —
 * they are properties of the math itself, no GPU needed.
 */
import { describe, it, expect } from 'vitest';
import {
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2,
  VOLUMETRIC_SERIES_C2_DIVISOR,
} from '../../../../../rendering/materials/gsplat/math';

/**
 * Exact transmittance-screening: S(τ) = (1 − e^(−τ))/τ, S(0) = 1.
 * Reference uses expm1 — the naive 1 − exp(−τ) form loses ~7 digits to
 * cancellation at τ ≈ 1e-8, which is precisely why the shader takes the
 * series branch there (the first version of this test used the naive
 * form and the SERIES was the more accurate side).
 */
function screenExact(tau: number): number {
  if (tau === 0) return 1;
  return -Math.expm1(-tau) / tau;
}

/**
 * The shader's branch: series below the threshold, guarded quotient
 * above — built FROM the exported constants in ./math (the shaders
 * interpolate the same symbols), so a drift there fails this suite
 * against `screenExact` instead of silently diverging from the test's
 * own private copy.
 */
function screenShader(tau: number): number {
  const alpha = 1 - Math.exp(-tau);
  return tau < VOLUMETRIC_SERIES_TAU_THRESHOLD
    ? 1 - VOLUMETRIC_SERIES_C1 * tau + (tau * tau) / VOLUMETRIC_SERIES_C2_DIVISOR
    : alpha / Math.max(tau, VOLUMETRIC_TAU_EPS);
}

/** One volumetric fragment: premultiplied (emission·S, α). */
function fragment(color: number, mass: number, kappa: number): { rgb: number; a: number } {
  const tau = kappa * mass;
  const alpha = 1 - Math.exp(-tau);
  return { rgb: color * mass * screenShader(tau), a: alpha };
}

/** Back-to-front One/OneMinusSrcAlpha composite of src over dst. */
function over(src: { rgb: number; a: number }, dstRgb: number): number {
  return src.rgb + (1 - src.a) * dstRgb;
}

describe('S(τ) series/exact seam', () => {
  it('constants pin the spec values (single source in ./math, consumed by both shaders)', () => {
    expect(VOLUMETRIC_SERIES_TAU_THRESHOLD).toBe(1e-3);
    expect(VOLUMETRIC_TAU_EPS).toBe(1e-20);
    expect(VOLUMETRIC_SERIES_C1).toBe(0.5);
    expect(VOLUMETRIC_SERIES_C2).toBe(1 / 6);
    // The shaders emit the τ² term as a division by 1/C2 (the historical
    // `τ·τ/6.0`, bit-identical rounding) — the derived divisor must stay
    // exactly consistent with C2.
    expect(VOLUMETRIC_SERIES_C2_DIVISOR).toBe(6);
    expect(VOLUMETRIC_SERIES_C2_DIVISOR * VOLUMETRIC_SERIES_C2).toBe(1);
  });

  it('shader branch matches the exact form across the τ range (incl. the 1e-3 seam)', () => {
    for (const tau of [0, 1e-8, 1e-6, 1e-4, 0.999e-3, 1e-3, 1.001e-3, 0.1, 1, 10]) {
      expect(screenShader(tau)).toBeCloseTo(screenExact(tau), 9);
    }
  });

  it('κ = 0 short-circuits exactly: α = 0, S = 1 (invariant I1 precondition)', () => {
    const frag = fragment(0.7, 2.5, 0);
    expect(frag.a).toBe(0);
    expect(frag.rgb).toBe(0.7 * 2.5); // pure additive contribution
  });

  it('is monotone decreasing and bounded: S(0)=1, S(∞)→0, α saturates at 1', () => {
    let prev = Number.POSITIVE_INFINITY;
    for (const tau of [0, 0.01, 0.1, 1, 10, 100]) {
      const s = screenShader(tau);
      expect(s).toBeLessThanOrEqual(prev);
      expect(s).toBeGreaterThanOrEqual(0);
      prev = s;
    }
    expect(1 - Math.exp(-1e6)).toBe(1);
  });
});

describe('I2 — split-splat multiplicativity', () => {
  it('one splat ≡ its two ray-wise halves composited back-to-front (exact)', () => {
    // The exponential absorption + screening make this EXACT, not
    // approximate — the property that keeps LOD merges/splits and the
    // additive streaming ladder visually consistent. A linear (1 − τ)
    // absorption or unscreened emission both fail this test.
    for (const kappa of [0.1, 1, 3, 10]) {
      for (const mass of [0.01, 0.5, 2, 8]) {
        const color = 0.8;
        const whole = fragment(color, mass, kappa);

        const half = fragment(color, mass / 2, kappa);
        // Composite: back half onto black, then front half over it.
        const backOverBlack = over(half, 0);
        const rgbSplit = over(half, backOverBlack);
        // Combined transmittance is multiplicative: T = T₁·T₂.
        const aSplit = 1 - (1 - half.a) * (1 - half.a);

        expect(rgbSplit).toBeCloseTo(over(whole, 0), 6);
        expect(aSplit).toBeCloseTo(whole.a, 6);
      }
    }
  });

  it('quarters compose the same as halves (associativity of the over operator)', () => {
    const kappa = 2.0;
    const mass = 1.6;
    const color = 1.0;
    const whole = over(fragment(color, mass, kappa), 0);

    let acc = 0;
    for (let i = 0; i < 4; i++) {
      acc = over(fragment(color, mass / 4, kappa), acc);
    }
    expect(acc).toBeCloseTo(whole, 6);
  });

  it('absorption darkens what is behind: bright background dimmed by exactly T = e^(−τ)', () => {
    const background = 5.0; // HDR
    const front = fragment(0, 1.0, 3.0); // black splat (color 0) — still absorbs
    const result = over(front, background);
    expect(result).toBeCloseTo(background * Math.exp(-3.0), 9);
  });
});
