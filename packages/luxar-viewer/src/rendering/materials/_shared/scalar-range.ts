/**
 * Single source for the colormap scalar-range → uniform mapping shared by all
 * six materials (point/line/gsplat × GLSL/TSL).
 *
 * The shaders sample the LUT at `t = clamp((scalar - uScalarMin) * uScalarScale,
 * 0, 1)`. Historically each material computed `uScalarScale = 1/max(1e-10,
 * max-min)` inline (12 copies), which sent a DEGENERATE range (min === max —
 * e.g. a constant scalar attribute, or a producer-stamped `[1, 1]`
 * `scalar_data_range`) to `t = 0`: every element rendered at the very bottom
 * of the LUT (black, in `gray`). The identity for a constant scalar is the
 * MIDPOINT: with `scalarMin = min - 0.5` and `scalarScale = 1`, a constant
 * `scalar === min` maps to `t = 0.5` — visible and neutral, mirroring the
 * degenerate display-range identity fix (#631).
 *
 * @module rendering/materials/_shared/scalar-range
 */

/** Below this width a scalar range reads as degenerate (constant attribute). */
export const DEGENERATE_SCALAR_RANGE_EPS = 1e-10;

export interface ScalarRangeUniforms {
  scalarMin: number;
  scalarScale: number;
}

/**
 * Map a scalar data range to the `uScalarMin`/`uScalarScale` uniform pair.
 * Degenerate ranges (max − min < eps) map every value to the LUT midpoint.
 */
export function computeScalarRangeUniforms(min: number, max: number): ScalarRangeUniforms {
  if (max - min < DEGENERATE_SCALAR_RANGE_EPS) {
    return { scalarMin: min - 0.5, scalarScale: 1.0 };
  }
  return { scalarMin: min, scalarScale: 1.0 / (max - min) };
}

/**
 * Same mapping shaped as THREE uniform entries, for the GLSL constructors'
 * `uniforms` object literals. A missing range defaults to [0, 1] — the
 * identity mapping the constructors historically used.
 */
export function scalarRangeUniformEntries(range: readonly [number, number] | undefined): {
  uScalarMin: { value: number };
  uScalarScale: { value: number };
} {
  const { scalarMin, scalarScale } = computeScalarRangeUniforms(
    range?.[0] ?? 0.0,
    range?.[1] ?? 1.0
  );
  return { uScalarMin: { value: scalarMin }, uScalarScale: { value: scalarScale } };
}
