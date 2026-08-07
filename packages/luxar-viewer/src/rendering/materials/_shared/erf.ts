/**
 * Shared error-function implementations — the single source of truth for
 * every erf in the codebase, CPU and GPU.
 *
 * Two implementations live here, each matched to its consumer:
 *
 * 1. `erfRef` — the Abramowitz & Stegun 7.1.26 rational approximation
 *    (max abs error 1.5e-7). The CPU-side reference: precomputing
 *    normalization factors (`computeRayIntegralFactor` in
 *    `gsplat/math.ts`), LUT generation, and the oracle the unit tests
 *    measure `erfPoly` against. Costs one `exp` and one division per
 *    call — fine on the CPU, the expensive form in a fragment shader.
 *
 * 2. `erfPoly` / `GLSL_ERF_FUNCTIONS` / `erfPolyTSL` — a pure odd
 *    polynomial on [-3, 3], clamped to ±1 outside. NO exp, NO division:
 *    the fragment-shader form (the volumetric line primitive evaluates
 *    two erfs per fragment on the most fill-heavy geometry type).
 *    Degree-13 constrained least-squares fit with P(3) = 1 EXACTLY, so
 *    the clamp is continuous. Max abs error 5.6e-4 against `erfRef`;
 *    the worst *difference* error `(erf(x1) − erf(x0))` consumers see is
 *    1.4e-3 of the peak when the two arguments are ≥ 0.5 apart (callers
 *    with closer arguments must use a midpoint/Taylor lane instead of
 *    the difference — see the volumetric line fragment shader).
 *
 * The GLSL string and the TSL builder are BOTH generated from
 * `ERF_POLY_COEFFS` with the same `toFixed(9)` serialization, so the two
 * backends cannot drift (the tsl-shader-parity harness and the codegen
 * snapshots depend on the literals matching). Same single-source pattern
 * as `falloff.ts` / `volumetric.ts`.
 *
 * @module rendering/materials/_shared/erf
 */

import { abs, float, min, select } from 'three/tsl';
import type { TSLNode } from './tsl-helpers';

/** A&S 7.1.26 auxiliary-variable constant: t = 1 / (1 + p·|x|). */
export const ERF_AS_P = 0.3275911;

/** A&S 7.1.26 polynomial coefficients (a1..a5). */
export const ERF_AS_COEFFS = [
  0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
] as const;

/**
 * CPU reference erf — Abramowitz & Stegun 7.1.26, max abs error 1.5e-7.
 */
export function erfRef(x: number): number {
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + ERF_AS_P * ax);
  const [a1, a2, a3, a4, a5] = ERF_AS_COEFFS;
  const y = 1.0 - t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * Math.exp(-ax * ax);
  return x >= 0 ? y : -y;
}

/**
 * The shader polynomial's clamp radius: erf(3) = 0.99998, and the fit is
 * constrained to reach exactly 1 there, so clamping to ±1 outside is
 * continuous.
 */
export const ERF_POLY_CLAMP = 3.0;

/**
 * Degree-13 odd-polynomial coefficients: erf(x) ≈ x·Σ cₖ·x^(2k) on
 * [0, 3], constrained least-squares (equality constraint P(3) = 1),
 * fitted against the exact erf on a 6000-point grid. The tail
 * coefficients are the `toFixed(9)` roundings of the fit; c₀ is then
 * RE-SOLVED against the rounded tail so the printed set itself satisfies
 * P(3) = 1 to 1e-9 (plain rounding of all seven broke the constraint by
 * 9.5e-5). These printed values are what both shader backends and
 * `erfPoly` evaluate, so the TS mirror is bit-faithful to the GPU math
 * up to float32 rounding. Max abs error vs the exact erf: 5.4e-4.
 */
export const ERF_POLY_COEFFS = [
  1.126454454, -0.366866314, 0.099810298, -0.018376255, 0.002118141, -0.000136121, 0.00000369,
] as const;

/**
 * TS mirror of the shader polynomial (same coefficients, same clamp).
 * Use `erfRef` when accuracy matters; use this to predict exactly what
 * the GPU computes.
 */
export function erfPoly(x: number): number {
  const ax = Math.min(Math.abs(x), ERF_POLY_CLAMP);
  const t = ax * ax;
  let p = 0;
  for (let k = ERF_POLY_COEFFS.length - 1; k >= 0; k--) {
    p = p * t + ERF_POLY_COEFFS[k];
  }
  p *= ax;
  return x < 0 ? -p : p;
}

/** Serialize a coefficient exactly as the fit produced it (toFixed(9)). */
function glslNum(v: number): string {
  return v.toFixed(9);
}

/**
 * GLSL implementation of `erfPoly` — `float luxarErf(float x)`.
 * Inject once per shader (before `main()`); pure ALU, no exp, no
 * division, so it is safe on the hottest fragment paths.
 */
export const GLSL_ERF_FUNCTIONS = `
float luxarErf(float x) {
  float ax = min(abs(x), ${glslNum(ERF_POLY_CLAMP)});
  float t = ax * ax;
  float p = ax * (${glslNum(ERF_POLY_COEFFS[0])} + t * (${glslNum(ERF_POLY_COEFFS[1])}
          + t * (${glslNum(ERF_POLY_COEFFS[2])} + t * (${glslNum(ERF_POLY_COEFFS[3])}
          + t * (${glslNum(ERF_POLY_COEFFS[4])} + t * (${glslNum(ERF_POLY_COEFFS[5])}
          + t * ${glslNum(ERF_POLY_COEFFS[6])}))))));
  return (x < 0.0) ? -p : p;
}
`;

/**
 * TSL twin of `luxarErf`, built from the SAME coefficients so WGSL and
 * GLSL emit identical literals. Branchless (`select`, not `If`) so it is
 * legal both inside and outside an `Fn()` body.
 */
export function erfPolyTSL(x: TSLNode): TSLNode {
  const ax = min(abs(x), float(ERF_POLY_CLAMP));
  const t = ax.mul(ax);
  let p: TSLNode = float(ERF_POLY_COEFFS[ERF_POLY_COEFFS.length - 1]);
  for (let k = ERF_POLY_COEFFS.length - 2; k >= 0; k--) {
    p = p.mul(t).add(float(ERF_POLY_COEFFS[k]));
  }
  p = p.mul(ax);
  return select(x.lessThan(float(0.0)), p.negate(), p);
}
