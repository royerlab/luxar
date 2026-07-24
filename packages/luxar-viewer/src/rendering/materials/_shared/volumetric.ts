/**
 * Shared constants for the volumetric (emission–absorption) blending
 * mode — VOLUMETRIC_BLENDING_SPEC.md.
 *
 * Lives in `_shared/` (not `gsplat/math.ts`, their original home)
 * because the math is geometry-agnostic: gsplats (phase 1–2) and points
 * (phase 3) both evaluate the S(τ) self-screening and the alpha →
 * optical-depth map, and the GLSL/TSL twins of BOTH geometries must
 * emit identical literals (the `tsl-shader-parity` harness and the
 * codegen snapshots depend on it).
 *
 * @module rendering/materials/_shared/volumetric
 */

/**
 * Per-element opacity clamp for the volumetric optical-depth map. Opacity
 * a = 1 means "fully opaque" but maps to w = −ln(1 − a) = ∞; clamp one
 * uint9-ish step below 1 (w ≈ 6.24, transmittance ≈ 0.2%). Kept here as
 * the single TS source of truth so the GLSL and TSL shaders emit an
 * identical literal.
 *
 * Mirrors Python's `luxar.gsplats.utils.alpha.ALPHA_CLAMP` (the two
 * languages can't share a symbol — keep the value in sync across both).
 */
export const ALPHA_CLAMP = 1.0 - 1.0 / 512.0;

/**
 * Volumetric self-screening constants (VOLUMETRIC_BLENDING_SPEC.md §3).
 * S(τ) = (1 − e^(−τ))/τ is evaluated as the guarded quotient above
 * {@link VOLUMETRIC_SERIES_TAU_THRESHOLD} and as the Maclaurin
 * polynomial  S(τ) ≈ 1 − C1·τ + C2·τ²  below it (the naive quotient
 * loses ~7 digits to cancellation as τ → 0; the series' relative error
 * is < 1e-10 at the cutoff). Single TS source of truth — the ALPHA_CLAMP
 * pattern — so the GLSL templates, the TSL graphs, and the
 * volumetric-math unit test all derive from identical numbers.
 */
export const VOLUMETRIC_SERIES_TAU_THRESHOLD = 1e-3;

/**
 * Division guard for the S(τ) quotient lane. GPU ternaries/selects
 * evaluate both lanes, so max(τ, eps) keeps the unselected lane
 * NaN-free at τ = 0 (identical in the selected regime).
 */
export const VOLUMETRIC_TAU_EPS = 1e-20;

/** First series coefficient of S(τ) ≈ 1 − C1·τ + C2·τ². */
export const VOLUMETRIC_SERIES_C1 = 0.5;

/** Second series coefficient of S(τ) ≈ 1 − C1·τ + C2·τ². */
export const VOLUMETRIC_SERIES_C2 = 1 / 6;

/**
 * The τ² term is emitted as a DIVISION by 1/C2 (exactly 6 in doubles)
 * on both shader backends — the historical `τ·τ/6.0` form — so float32
 * rounding and the TSL codegen snapshot stay bit-identical (multiplying
 * by the rounded reciprocal instead can differ by 1 ulp).
 */
export const VOLUMETRIC_SERIES_C2_DIVISOR = 1 / VOLUMETRIC_SERIES_C2;
