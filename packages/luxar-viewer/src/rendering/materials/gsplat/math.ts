/**
 * Shared mathematical helpers for the GSplat material pair.
 *
 * Lives outside both `gsplat-material.ts` (GLSL `ShaderMaterial`) and
 * `gsplat-material-tsl.ts` (TSL `NodeMaterial`) so the two backends
 * compute identical numeric values from identical inputs — the parity
 * test (`tsl-shader-parity.spec.ts`) depends on this.
 *
 * @module rendering/materials/gsplat/math
 */

/**
 * Ray-integral factor for the shifted Gaussian:
 *   sqrt(2π) · erf(T/√2) − 2·T·exp(−½·T²)
 *
 * For the unshifted Gaussian this reduces to sqrt(2π) ≈ 2.507.
 * For T=3 the shifted form is ≈ 2.433.
 *
 * Uses the Abramowitz & Stegun erf approximation (max error 1.5e-7).
 */
export function computeRayIntegralFactor(truncate: number): number {
  const SQRT_2PI = Math.sqrt(2 * Math.PI);
  const x = truncate / Math.SQRT2;
  const t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
  const erfVal =
    1.0 -
    t *
      (0.254829592 +
        t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) *
      Math.exp(-x * x);
  const erf = x >= 0 ? erfVal : -erfVal;
  return SQRT_2PI * erf - 2 * truncate * Math.exp(-0.5 * truncate * truncate);
}

/**
 * Minimum truncation radius (in sigmas). Below this the shifted-Gaussian
 * normalization degenerates: `shiftC = exp(-r²/2)` approaches 1 and
 * `uInvOneMinusC = 1/(1 - shiftC)` blows up to Infinity — an invisible
 * layer with an Infinity uniform and zero diagnostics. Dataset attrs
 * pass `truncation_radius` through unvalidated, so both material
 * wrappers clamp at this boundary (mirrors `updateMaxExtentFactor`'s
 * 0.01 floor).
 */
export const MIN_TRUNCATION_RADIUS = 0.1;

/** Clamp a truncation radius to the degeneracy floor. */
export function clampTruncationRadius(radius: number): number {
  return Math.max(MIN_TRUNCATION_RADIUS, radius);
}
