/**
 * Shared error-function implementations — the single source of truth for
 * every erf in the VIEWER, CPU and GPU. (The Python package carries its
 * own copy of the same A&S 7.1.26 form in `luxar/gsplats/lift.py` — the
 * two languages cannot share code, but the constants must stay in sync.)
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
 *    the fragment-shader form, built for the UPCOMING #1352 volumetric
 *    line primitive (two erfs per fragment on the most fill-heavy
 *    geometry type; no production shader consumes it yet — the parity
 *    fixture and codegen snapshot are its consumers until then).
 *    Degree-13 constrained least-squares fit with P(3) = 1 (to 1e-9 for
 *    the printed coefficient set), so the clamp is continuous. Max abs
 *    error 5.4e-4 against the exact erf; the worst error in the
 *    difference QUOTIENT `(erf(x1) − erf(x0)) / (x1 − x0)` is 1.4e-3
 *    absolute (0.13% of its 2/√π ≈ 1.128 peak) when the two arguments
 *    are ≥ 0.5 apart (callers with closer arguments must use a
 *    midpoint/Taylor lane instead of the difference — see the volumetric
 *    line fragment shader).
 *
 *    Two consequences of this being a least-squares fit rather than a
 *    bounded approximation — both well inside that error bound, both easy
 *    to trip over: it is NOT clamped to ±1 (it peaks at 1.00032 near
 *    |x| ≈ 2.72) and it is NOT strictly monotone near saturation, so an
 *    `erf(x1) − erf(x0)` window with x1 > x0 can come out slightly
 *    NEGATIVE (worst −8.8e-4, at x0 ≈ 2.72 / x1 ≈ 2.94). A consumer that
 *    needs a non-negative window — an optical depth, a coverage weight —
 *    must clamp at zero rather than trust the sign.
 *
 * The GLSL string and the TSL builder are BOTH generated from
 * `ERF_POLY_COEFFS`, so the two backends cannot drift in VALUE. The
 * guarantee is value-level, not textual: GLSL literals are serialized
 * here with `toFixed(9)`, while the TSL path passes the same numbers to
 * `float()` and Three's code generator owns their formatting (numeric
 * parity is what the tsl-shader-parity harness verifies). Same
 * single-source pattern as `falloff.ts` / `volumetric.ts`.
 *
 * @module rendering/materials/_shared/erf
 */

import { abs, exp, float, min, sign } from 'three/tsl';
import type { TSLNode } from './tsl-helpers';

/** A&S 7.1.26 auxiliary-variable constant: t = 1 / (1 + p·|x|). */
const ERF_AS_P = 0.3275911;

/** A&S 7.1.26 polynomial coefficients (a1..a5). */
const ERF_AS_COEFFS = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429] as const;

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
 * `erfPoly` evaluate; the f64 mirror matches a float32-chained
 * evaluation to within ~7e-6 (accumulated rounding across the Horner
 * steps — 1% of the fit error). Max abs error vs the exact erf: 5.4e-4.
 */
export const ERF_POLY_COEFFS = [
  1.126454454, -0.366866314, 0.099810298, -0.018376255, 0.002118141, -0.000136121, 0.00000369,
] as const;

/**
 * TS mirror of the shader polynomial (same coefficients, same clamp),
 * evaluated in float64. Use `erfRef` when accuracy matters; use this to
 * predict the GPU result to within accumulated float32 rounding
 * (measured ≤ ~7e-6 — see the coefficient docblock, and the ~5e-6
 * worst case observed against a real GPU).
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
 *
 * NaN caveat: GLSL `min`/`abs` make no IEEE NaN guarantee, so
 * `luxarErf(NaN)` is implementation-defined on the GPU (the TS mirror
 * propagates NaN). Callers must sanitize upstream rather than rely on
 * NaN propagation through this function.
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
 * GLSL implementation of `erfRef` — `float luxarErfAS(float x)`, the A&S
 * 7.1.26 rational form (max abs error 1.5e-7, one `exp` + one division).
 *
 * For the FEW shader lanes where the polynomial's 5.4e-4 error gets
 * amplified past visibility: the volumetric line primitive's mixed-end
 * lane multiplies its erf terms by pref ∝ 1/sin(ray, axis), which is
 * unbounded, so it pays the exp for exactness. Population there is only
 * the two chain-end segments of each polyline. Hot lanes keep
 * {@link GLSL_ERF_FUNCTIONS}. Value-locked to `erfRef` via the shared
 * constants; same NaN caveat as `luxarErf`.
 */
export const GLSL_ERF_AS_FUNCTIONS = `
float luxarErfAS(float x) {
  float ax = abs(x);
  float t = 1.0 / (1.0 + ${glslNum(ERF_AS_P)} * ax);
  float p = 1.0 - t * (${glslNum(ERF_AS_COEFFS[0])} + t * (${glslNum(ERF_AS_COEFFS[1])}
          + t * (${glslNum(ERF_AS_COEFFS[2])} + t * (${glslNum(ERF_AS_COEFFS[3])}
          + t * ${glslNum(ERF_AS_COEFFS[4])})))) * exp(-min(ax * ax, 80.0));
  return (x < 0.0) ? -p : p;
}
`;

/**
 * TSL twin of `luxarErfAS`, built from the SAME constants (see the
 * `GLSL_ERF_AS_FUNCTIONS` docblock for when to pay for it).
 */
export function erfAsTSL(x: TSLNode): TSLNode {
  const ax = abs(x);
  const t = float(1.0).div(float(1.0).add(float(ERF_AS_P).mul(ax)));
  let poly: TSLNode = float(ERF_AS_COEFFS[ERF_AS_COEFFS.length - 1]);
  for (let k = ERF_AS_COEFFS.length - 2; k >= 0; k--) {
    poly = poly.mul(t).add(float(ERF_AS_COEFFS[k]));
  }
  const p = float(1.0).sub(poly.mul(t).mul(exp(min(ax.mul(ax), float(80.0)).negate())));
  return p.mul(sign(x));
}

/**
 * TSL twin of `luxarErf`, built from the SAME coefficient values (the
 * code generator owns literal formatting — see the module header).
 * The sign is applied by multiplying with `sign(x)` — genuinely
 * branchless in the GENERATED code (a `select` would lower to an
 * `if`/`else` that duplicates the whole polynomial across both arms),
 * and legal both inside and outside an `Fn()` body. `sign(0) = 0`
 * matches the polynomial (`P(0) = 0`), and NaN falls under the same
 * caveat as the GLSL form.
 */
export function erfPolyTSL(x: TSLNode): TSLNode {
  const ax = min(abs(x), float(ERF_POLY_CLAMP));
  const t = ax.mul(ax);
  let p: TSLNode = float(ERF_POLY_COEFFS[ERF_POLY_COEFFS.length - 1]);
  for (let k = ERF_POLY_COEFFS.length - 2; k >= 0; k--) {
    p = p.mul(t).add(float(ERF_POLY_COEFFS[k]));
  }
  p = p.mul(ax);
  return p.mul(sign(x));
}
