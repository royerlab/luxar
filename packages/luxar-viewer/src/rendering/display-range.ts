/**
 * Pure display-window ↔ shader-uniform math.
 *
 * Converts between a display ``[min, max]`` window and the shader's
 * intensity (gain) / offset uniforms. These helpers are dependency-free and
 * live in the rendering layer so both ``rendering/`` and ``ui/layers/`` can
 * import them without violating the module layer order (``rendering/`` must
 * never import from the higher ``ui/`` layer).
 *
 * @module rendering/display-range
 */

/** Computed shader uniforms from display range */
export interface DisplayUniforms {
  intensity: number;
  offset: number;
}

/**
 * Convert display [min, max] range to shader intensity (gain) and offset.
 *
 * The shader computes: `final_color = clamp(color * intensity + offset, 0, 1)`
 * We want `displayMin → 0` and `displayMax → 1`:
 *   intensity = 1 / (max - min)
 *   offset    = -min / (max - min)
 */
export function computeUniforms(displayMin: number, displayMax: number): DisplayUniforms {
  const range = displayMax - displayMin;
  if (Math.abs(range) < 1e-10) {
    // Degenerate range (min == max): there is nothing to window, so pass the
    // color through unchanged (identity gain/offset). A constant data range is
    // legitimate — e.g. classical-splat imports where per-splat opacity rides
    // in the color alpha and `amplitude_data_range` is [1, 1] (constant). The
    // old "very high contrast" mapping (gain 1000, offset -1000·min) turned
    // `color·1000 − 1000` into 0 for every non-white color, rendering the
    // whole scene black.
    return { intensity: 1.0, offset: 0.0 };
  }
  return {
    intensity: 1.0 / range,
    offset: -displayMin / range,
  };
}

/**
 * Inverse: recover display [min, max] from shader intensity and offset.
 *   displayMin = -offset / intensity
 *   displayMax = (1 - offset) / intensity
 */
export function computeDisplayRange(
  intensity: number,
  offset: number
): { min: number; max: number } {
  if (Math.abs(intensity) < 1e-10) {
    return { min: 0, max: 1 };
  }
  return {
    min: -offset / intensity,
    max: (1 - offset) / intensity,
  };
}

/**
 * Resolve the scalar LUT window for a COLORMAPPED node at load time.
 *
 * On a colormapped node an authored `intensity`/`offset` is the scalar
 * WINDOW (value→LUT mapping), NOT a post-LUT color gain — applying it as
 * both double-applies (#936). Callers therefore push identity to the color
 * GOG and this window to `updateScalarRange`.
 *
 * The identity-vs-window decision follows the RAW LEAF gain, mirroring the
 * layers panel (`layer-state.ts` `initialDisplayRange` starts from the data
 * range whenever the LEAF gain is identity) — NOT the composed value, which
 * would treat an ancestor-only gain as an authored window and discard the
 * node's own data range. When the leaf DID author a window, the COMPOSED
 * gain IS the panel's effective gain (intensity multiplies, offset adds —
 * `attrs-composer.ts`), so it is used directly. When the leaf is identity,
 * any ancestor gain is folded onto the data-range window exactly the way the
 * panel composes it: data range → window uniforms → × ancestor gain → back
 * to a window.
 *
 * Shared by the points, lines, and gsplats node factories so all three
 * geometry types agree (the three-geometry symmetry rule).
 *
 * @param dataRange   The node's own scalar/amplitude data range.
 * @param leaf        Raw leaf-authored gain/offset (uncomposed).
 * @param composed    Effective gain/offset after ancestor composition.
 */
export function resolveColormapWindow(
  dataRange: readonly [number, number],
  leaf: DisplayUniforms,
  composed: DisplayUniforms
): [number, number] {
  const leafIsIdentity = leaf.intensity === 1.0 && leaf.offset === 0.0;
  if (!leafIsIdentity) {
    const { min, max } = computeDisplayRange(composed.intensity, composed.offset);
    return [min, max];
  }
  if (composed.intensity === 1.0 && composed.offset === 0.0) {
    return [dataRange[0], dataRange[1]];
  }
  // Ancestor-only gain (leaf identity ⇒ composed == ancestor product).
  const w = computeUniforms(dataRange[0], dataRange[1]);
  const { min, max } = computeDisplayRange(
    composed.intensity * w.intensity,
    composed.offset + w.offset
  );
  return [min, max];
}
