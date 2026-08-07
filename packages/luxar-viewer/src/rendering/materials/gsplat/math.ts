import { log, Modules } from '../../../utils/log';
import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../config/constants';
import { erfRef } from '../_shared/erf';

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

// The volumetric constants (ALPHA_CLAMP, VOLUMETRIC_SERIES_*) moved to
// `materials/_shared/volumetric.ts` — they are geometry-agnostic since
// points implement the same emission–absorption math (phase 3).

/**
 * Ray-integral factor for the shifted Gaussian:
 *   sqrt(2π) · erf(T/√2) − 2·T·exp(−½·T²)
 *
 * For the unshifted Gaussian this reduces to sqrt(2π) ≈ 2.507.
 * For T=3 the shifted form is ≈ 2.433.
 *
 * Uses the shared `erfRef` (A&S 7.1.26, max error 1.5e-7) from
 * `_shared/erf.ts` — the viewer's single erf source.
 */
export function computeRayIntegralFactor(truncate: number): number {
  const SQRT_2PI = Math.sqrt(2 * Math.PI);
  const erf = erfRef(truncate / Math.SQRT2);
  return SQRT_2PI * erf - 2 * truncate * Math.exp(-0.5 * truncate * truncate);
}

/**
 * Smallest truncation radius (in sigmas) whose shifted-Gaussian normalization
 * survives float32. Below it `shiftC = exp(-r²/2)` rounds to exactly 1.0 in
 * single precision (the GPU uniform's precision), so
 * `uInvOneMinusC = 1/(1 - shiftC)` blows up — an invisible layer with an
 * Infinity uniform and zero diagnostics.
 *
 * Bisected at module load against real float32 arithmetic (~2.44e-4),
 * MIRRORING the write-side validator's `MIN_TRUNCATION_RADIUS_FLOAT32` in
 * `packages/luxar/src/luxar/validation/types.py`. The two must agree: the
 * writer accepts anything that normalizes in float32, and the on-disk chunk
 * bounds are computed from the stored radius — a higher read-time floor
 * (the former 0.1) silently rendered small-but-valid radii wider than their
 * chunk bounds claim.
 */
function computeMinTruncationRadiusFloat32(): number {
  const degenerate = (t: number) => {
    const t32 = Math.fround(t);
    const exponent = Math.fround(Math.fround(-0.5 * t32) * t32);
    return Math.fround(Math.exp(exponent)) >= 1.0;
  };
  let lo = 1e-12; // degenerate
  let hi = 1.0; // fine
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (degenerate(mid)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return hi;
}

export const MIN_TRUNCATION_RADIUS = computeMinTruncationRadiusFloat32();

let truncationClampWarned = false;

/** Clamp a truncation radius to the float32 degeneracy bounds (warns once). */
export function clampTruncationRadius(radius: number): number {
  // NaN/Inf slip past a plain comparison clamp (NaN < x is false), and a
  // JS-finite value can still poison the GPU uniforms two ways: beyond
  // float32 range (a hostile attr like 1e308) `uTruncate` itself narrows to
  // Infinity, and beyond sqrt(float32.max) ≈ 1.84e19 the squared uniform
  // `uTruncateSq = T²` does. Testing the square catches both (a radius whose
  // square is float32-finite is itself float32-finite), mirroring the
  // write-side `MAX_TRUNCATION_RADIUS_FLOAT32` bound in
  // `packages/luxar/src/luxar/validation/types.py`. Fall back to the module
  // default.
  if (!Number.isFinite(Math.fround(radius * radius))) {
    if (!truncationClampWarned) {
      truncationClampWarned = true;
      log.warning(
        Modules.RENDERER,
        `truncation_radius ${radius} (or its square, uploaded as uTruncateSq) is not finite in float32 — falling back to ${GSPLAT_DEFAULT_TRUNCATION_RADIUS}. Further clamps are silent.`
      );
    }
    return GSPLAT_DEFAULT_TRUNCATION_RADIUS;
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
