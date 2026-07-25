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
 *
 * DELIBERATELY no hysteresis (2026-07 debt-remediation decision):
 * crossing the boundary flips the LUXAR_GAMMA_ONE define and costs one
 * shader recompile per crossing. A hysteresis band wide enough to
 * matter would keep the pow()-skipping fast path engaged at gamma
 * values measurably off 1.0 — a silent visual error on scientific
 * data — while the band below is already ±1e-4 (effectively "exactly
 * 1.0"), so per-frame toggle thrash would require a slider oscillating
 * inside a 0.0002-wide window. Two bounded recompiles per deliberate
 * crossing is the correct trade.
 */
export function isGammaOne(gamma: number): boolean {
  return Math.abs(gamma - 1.0) < 1e-4;
}

/**
 * Whether `intensity == 1.0 && offset == 0.0` (with ±1e-4 epsilon) —
 * the default GOG (Gain-Offset-Gamma) configuration. When true, the
 * fragment shader can skip the `vColor * uIntensity + uOffset` chain
 * and its `max(..., vec3(0))` clamp — identity for the non-negative
 * `vColor` range — by compiling in the `LUXAR_NO_GOG` define (GLSL) /
 * `noGOG` config flag (TSL).
 *
 * Shared across all three geometry types (Point / Line / GSplat) ×
 * both backends, like {@link isGammaOne}, so the fast-path threshold
 * is identical everywhere.
 */
export function isNoGOG(intensity: number, offset: number): boolean {
  return Math.abs(intensity - 1.0) < 1e-4 && Math.abs(offset) < 1e-4;
}
