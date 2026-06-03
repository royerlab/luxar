/**
 * Tiny shared helpers for uniform-value normalization used by the
 * Point / Line / GSplat material pairs (both GLSL and TSL variants).
 *
 * The audit that surfaced these helpers explicitly limited the scope
 * to drift points that already appear in every material constructor;
 * a wider "uniform-spec registry" was rejected as documentation
 * disguised as code. Keep this module small.
 *
 * @module rendering/material-uniform-helpers
 */

/**
 * Clamp a gamma value to a safe range and supply the canonical
 * default of `1.0` when undefined.
 *
 * The shader computes `pow(color, 1.0 / gamma)`, which divides by
 * zero when `gamma == 0`. A lower bound of `0.001` matches what each
 * material constructor was clamping inline before this helper
 * existed; using one helper across all six files (Point/Line/GSplat
 * × GLSL/TSL) keeps the clamp identical if the bound ever changes.
 */
export function clampGamma(gamma: number | undefined): number {
  return Math.max(0.001, gamma ?? 1.0);
}

/**
 * Whether `gamma == 1.0` (with ±1e-4 epsilon for float-equality
 * safety). When true, the fragment shader can skip its per-fragment
 * `pow(color, 1.0 / gamma)` calls — `pow(x, 1) == x` — by compiling in
 * the `LUXAR_GAMMA_ONE` define (GLSL) / `gammaOne` config flag (TSL).
 *
 * Shared across all three geometry types (Point / Line / GSplat) ×
 * both backends so the fast-path threshold is identical everywhere.
 */
export function isGammaOne(gamma: number): boolean {
  return Math.abs(gamma - 1.0) < 1e-4;
}
