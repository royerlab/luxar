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
