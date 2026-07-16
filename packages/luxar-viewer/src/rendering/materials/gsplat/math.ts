import { log, Modules } from '../../../utils/log';

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
 * Screen-space 2D-covariance low-pass dilation, in pixels² (the standard 3DGS
 * anti-aliasing term). Added to the diagonal of the projected covariance Σ_2D
 * so every splat covers at least ~1 pixel — this prevents extremely anisotropic
 * splats (near-degenerate, edge-on flat disks, common in imported classical 3DGS
 * fits) from collapsing to razor-thin sub-pixel spikes. Negligible for the
 * near-isotropic splats produced by Luxar's own volume fitting. Kept here as the
 * single source of truth so every GLSL/TSL material + picking constructor uses
 * an identical default (avoids the uMaxExtentFactor 0.33-vs-1.0 default drift).
 */
export const GSPLAT_COV2D_DILATION_DEFAULT = 0.3;

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

let truncationClampWarned = false;

/** Clamp a truncation radius to the degeneracy floor (warns once). */
export function clampTruncationRadius(radius: number): number {
  // NaN/Inf slip past a plain comparison clamp (NaN < x is false) and
  // would poison uShiftC/uInvOneMinusC — the exact degenerate-uniform
  // failure this clamp exists to prevent. Fall back to the 3.0 default.
  if (!Number.isFinite(radius)) {
    if (!truncationClampWarned) {
      truncationClampWarned = true;
      log.warning(
        Modules.RENDERER,
        `truncation_radius ${radius} is not finite — falling back to 3.0. Further clamps are silent.`
      );
    }
    return 3.0;
  }
  if (radius < MIN_TRUNCATION_RADIUS) {
    if (!truncationClampWarned) {
      truncationClampWarned = true;
      log.warning(
        Modules.RENDERER,
        `truncation_radius ${radius} clamped to ${MIN_TRUNCATION_RADIUS}σ ` +
          '(below this the shifted-Gaussian normalization degenerates to an ' +
          'Infinity uniform / invisible layer). Further clamps are silent.'
      );
    }
    return MIN_TRUNCATION_RADIUS;
  }
  return radius;
}
