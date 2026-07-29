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
} from '../../../../../rendering/materials/_shared/volumetric';

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

/**
 * One volumetric fragment with the node opacity threaded through, exactly as
 * the shader composes it (shader-glsl.ts LUXAR_VOLUMETRIC branch):
 * τ = κ·opacity·mass, emission = color·mass·opacity·S(τ). This is the model
 * behind BLENDABLE_MODES including 'volumetric' (scene/lod-fade.ts): both LOD
 * anti-popping mechanisms drive exactly this opacity multiplier.
 */
function fragmentOp(
  color: number,
  mass: number,
  kappa: number,
  opacity: number
): { rgb: number; a: number } {
  const tau = kappa * opacity * mass;
  const alpha = 1 - Math.exp(-tau);
  return { rgb: color * mass * opacity * screenShader(tau), a: alpha };
}

describe('LOD fade invariants — opacity as a linear knob on τ (BLENDABLE_MODES)', () => {
  it('cross-fade is absorption-invariant ONLY in the mass-matched-per-ray idealization', () => {
    // 1 − e^(−wτ)·e^(−(1−w)τ) = 1 − e^(−τ) for every w, because τ is additive
    // across fragments and linear in opacity. NOTE the premise: both levels
    // present the SAME per-ray τ. That is an idealization — the build
    // invariant is total mass per barrier group, not per-ray mass — so this
    // case is the reference point, not the general contract (see the next
    // test for what actually happens when τ_fine ≠ τ_coarse).
    for (const kappa of [0.1, 1, 3, 10]) {
      for (const mass of [0.01, 0.5, 2, 8]) {
        for (const w of [0.1, 0.25, 0.5, 0.75, 0.9]) {
          const whole = fragmentOp(0.8, mass, kappa, 1);
          const fineW = fragmentOp(0.8, mass, kappa, w);
          const coarseW = fragmentOp(0.8, mass, kappa, 1 - w);
          const aPair = 1 - (1 - fineW.a) * (1 - coarseW.a);
          expect(aPair).toBeCloseTo(whole.a, 12);
        }
      }
    }
  });

  it('cross-fade with DISTINCT per-ray τ: exact endpoints, monotone, always bracketed (the real dissolve contract)', () => {
    // The general case a real LOD pair presents: the coarse level is a
    // different spatial distribution, so along a given ray τ_coarse ≠ τ_fine.
    // The composited absorption is 1 − exp(−(w·τ_fine + (1−w)·τ_coarse)) — NOT
    // invariant through the fade. What IS guaranteed (and is exactly what an
    // anti-popping dissolve needs): the endpoints reproduce each level
    // exactly, and in between the absorption moves monotonically and stays
    // bracketed between the two levels' own absorptions — no overshoot, no
    // ghost darker or lighter than either level.
    for (const [mFine, mCoarse] of [
      [2, 0.5],
      [0.5, 2],
      [1, 1.4],
      [8, 0.1],
    ]) {
      const kappa = 1.5;
      const aFine = fragmentOp(0.8, mFine, kappa, 1).a;
      const aCoarse = fragmentOp(0.8, mCoarse, kappa, 1).a;
      const lo = Math.min(aFine, aCoarse);
      const hi = Math.max(aFine, aCoarse);
      const pairAlpha = (w: number): number => {
        const f = fragmentOp(0.8, mFine, kappa, w);
        const c = fragmentOp(0.8, mCoarse, kappa, 1 - w);
        return 1 - (1 - f.a) * (1 - c.a);
      };
      // Endpoints are exact (w = 1 ⇒ the fine level alone, w = 0 ⇒ coarse).
      expect(pairAlpha(1)).toBeCloseTo(aFine, 12);
      expect(pairAlpha(0)).toBeCloseTo(aCoarse, 12);
      // Monotone in w and bracketed by the two endpoint absorptions.
      let prev = pairAlpha(0);
      const ascending = aFine > aCoarse;
      for (let i = 1; i <= 20; i++) {
        const a = pairAlpha(i / 20);
        expect(a).toBeGreaterThanOrEqual(lo - 1e-12);
        expect(a).toBeLessThanOrEqual(hi + 1e-12);
        if (ascending) expect(a).toBeGreaterThanOrEqual(prev - 1e-12);
        else expect(a).toBeLessThanOrEqual(prev + 1e-12);
        prev = a;
      }
    }
  });

  it('cross-fade conserves emission to first order in τ (exact in the additive κ→0 limit)', () => {
    // Optically thin regime (τ ≪ 1): the composited pair's emission matches
    // the single full-opacity level to O(τ²).
    for (const w of [0.25, 0.5, 0.75]) {
      const kappa = 1.0;
      const mass = 0.05; // τ = 0.05
      const whole = over(fragmentOp(0.8, mass, kappa, 1), 0);
      const pair = over(
        fragmentOp(0.8, mass, kappa, w),
        over(fragmentOp(0.8, mass, kappa, 1 - w), 0)
      );
      expect(Math.abs(pair - whole) / whole).toBeLessThan(0.01);
    }
  });

  it('energy compensation restores α exactly under PROPORTIONAL thinning (the idealized prefix)', () => {
    // If a committed prefix were the whole leaf uniformly thinned to e of its
    // mass on every ray, the 1/e boost would restore τ = κ·(1/e)·(e·m) = κ·m
    // exactly. Real ladders don't work that way (next test) — this pins the
    // idealization the mechanism is derived from.
    for (const e of [0.1, 0.25, 0.5, 0.9]) {
      const kappa = 2.0;
      const mass = 1.5;
      const full = fragmentOp(0.8, mass, kappa, 1);
      const partial = fragmentOp(0.8, mass * e, kappa, 1 / e);
      expect(partial.a).toBeCloseTo(full.a, 12);
    }
  });

  it('energy compensation on a GENUINE subset prefix restores τ only in aggregate, not per ray', () => {
    // What actually streams: an energy-ordered SUBSET of splats. Model two
    // rays, one splat each, equal mass m; the prefix commits ray A's splat
    // only, so e(k) = 0.5 as a GLOBAL energy fraction. The 1/e = 2× boost
    // then doubles τ on ray A and does nothing for ray B (nothing there to
    // scale). Consequences, all on the record here:
    //   - ray A is OVER-occluded vs the full ladder (α_A > α_full)
    //   - ray B stays fully transparent (α_B = 0)
    //   - the SUM of per-ray τ is what gets restored (aggregate, not per ray)
    // This is the same structural approximation the additive/luminous path
    // has shipped since the compensation landed; it is not volumetric-specific.
    const kappa = 1.5;
    const m = 1.0;
    const e = 0.5;
    const boost = 1 / e;

    const tauFull = kappa * m; // either ray, full ladder
    const alphaFull = 1 - Math.exp(-tauFull);

    const rayA = fragmentOp(0.8, m, kappa, boost); // committed, boosted
    const rayB = fragmentOp(0.8, 0, kappa, boost); // missing splat ⇒ no mass

    expect(rayA.a).toBeGreaterThan(alphaFull); // over-occluded, not exact
    expect(rayB.a).toBe(0); // boost cannot conjure absorption

    // Aggregate τ over the two rays IS restored: boosted committed τ (2·κ·m)
    // equals the full ladder's total (κ·m + κ·m).
    const tauAggregateBoosted = kappa * boost * m + 0;
    expect(tauAggregateBoosted).toBeCloseTo(2 * tauFull, 12);

    // And the per-ray error vanishes as the ladder completes (e → 1).
    const nearComplete = fragmentOp(0.8, m, kappa, 1 / 0.99);
    expect(Math.abs(nearComplete.a - alphaFull)).toBeLessThan(0.01);
  });

  it('boost emission is ~linear on thin splats but saturates on individually thick ones (the accepted ENERGY_FLOOR-cap caveat)', () => {
    const B = 10; // the 1/ENERGY_FLOOR cap
    // Thin (τ = 1e-3): S ≈ 1, boosted emission ≈ B× — compensation works.
    const thin1 = fragmentOp(0.8, 1e-3, 1, 1).rgb;
    const thinB = fragmentOp(0.8, 1e-3, 1, B).rgb;
    expect(thinB / thin1).toBeGreaterThan(0.99 * B);
    // Thick (τ = 2): S decays as 1/τ, so emission gains far less than B —
    // the boost deepens occlusion more than it brightens (spec §6 caveat).
    const thick1 = fragmentOp(0.8, 2, 1, 1).rgb;
    const thickB = fragmentOp(0.8, 2, 1, B).rgb;
    expect(thickB / thick1).toBeLessThan(B / 5);
    expect(thickB).toBeGreaterThanOrEqual(thick1); // still monotone, never darkens the splat itself
  });
});
