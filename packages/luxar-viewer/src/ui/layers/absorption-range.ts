/**
 * Range for the volumetric absorption (κ) slider.
 *
 * κ converts ray mass into optical depth: every geometry family now builds
 * `τ = κ · rayMass`, where `rayMass` is the SAME quantity that family's
 * additive branch emits (points/lines `alpha`, gsplats `intensity`). Because
 * that quantity is a peak-screen-alpha-normalised number rather than a
 * world-space length, **κ is comparable across points, lines and gsplats and
 * across scene scales** — κ = 1 means "peak rendered alpha ≈ the authored
 * per-element alpha" for all three. One fixed track therefore serves every
 * scene.
 *
 * HISTORY (do not reintroduce): this module used to DERIVE a per-layer upper
 * bound as `ABSORPTION_TAU_TARGET / (thickness · chord)` from `max_width` /
 * `max_radius`, because the point and line shaders multiplied τ by a world
 * thickness (`radius · POINT_CHORD_SCALE`) that the gsplat shader had no
 * counterpart for. That made κ a per-unit-length coefficient for two families
 * and dimensionless for the third, so the stretch was a units conversion, not
 * a UX affordance — and it could never serve a MIXED subtree (a
 * `lift_points_to_gsplats` LOD ladder composes ONE κ over both families, and
 * gsplats returned no stat at all). The shaders now share one convention, so
 * the conversion is gone along with the chord constants.
 *
 * What remains here is the log-track mechanics: a fixed nominal span, widened
 * only when an AUTHORED κ falls outside it so the thumb can always represent
 * the live value.
 *
 * @module ui/layers/absorption-range
 */

/**
 * Nominal upper bound of the κ track. With the unified convention τ = κ·rayMass
 * and rayMass ≈ 1 at an element's peak, κ = 10 is already far past opaque
 * (1 − e^(−10) > 0.9999), so this tops out the useful range for every family.
 */
export const ABSORPTION_DEFAULT_MAX = 10.0;

/**
 * Hard ceiling on the track when an authored κ exceeds the nominal maximum.
 * Guards a pathological authored value from compressing the useful region off
 * the track.
 */
export const ABSORPTION_MAX_LIMIT = 1e6;

/** Decades of κ the nominal track spans below {@link ABSORPTION_DEFAULT_MAX}. */
export const ABSORPTION_LOG_DECADES = 4;

/**
 * Hard cap on the track's span in decades. Bounds the floor from below so the
 * useful region is never more than this many decades under the track's top —
 * whether the floor was lowered to include a κ far below the nominal floor, or
 * the top was raised toward the ceiling for a very large authored κ. In both
 * cases the region compressed off the track is already visually
 * indistinguishable from its neighbour.
 */
export const ABSORPTION_LOG_DECADES_MAX = 8;

/**
 * Log-track bounds for the absorption slider at a given live κ.
 *
 * The nominal track is {@link ABSORPTION_LOG_DECADES} decades below
 * {@link ABSORPTION_DEFAULT_MAX}. Both ends still move to keep the CURRENT κ
 * ON the track — otherwise the readout would show a value the thumb cannot
 * represent, and a touch that moves nothing would write the clamped end back:
 *
 *   - `max` is raised to the current κ when an author set it above the
 *     nominal maximum.
 *   - `min` is lowered to the current κ when an author set it below the
 *     nominal floor.
 *
 * TWO DELIBERATE CLAMPS bound that accommodation, because a pathological
 * authored κ would otherwise compress the useful region off the track: `max`
 * stops at {@link ABSORPTION_MAX_LIMIT}, and the floor stops at
 * {@link ABSORPTION_LOG_DECADES_MAX} decades of span. Outside those bounds the
 * thumb seats at the clamped end while the readout still shows the true κ, so
 * touching the slider writes the clamp back. That is accepted: past the ceiling
 * both κ are far beyond opaque, and below the floor both are ≥ 7 decades below
 * visible absorption — the states being swapped are visually identical (see the
 * `at the clamped extremes` tests).
 *
 * κ = 0 needs no room: it is the track's dedicated zero stop
 * (`LabeledSlider`, `scale: 'log'`).
 */
export function absorptionSliderRange(currentValue: number): { min: number; max: number } {
  const current = Number.isFinite(currentValue) ? currentValue : 0;
  const max = Math.min(ABSORPTION_MAX_LIMIT, Math.max(ABSORPTION_DEFAULT_MAX, current));
  const nominalMin = ABSORPTION_DEFAULT_MAX / Math.pow(10, ABSORPTION_LOG_DECADES);
  const min = Math.max(
    max / Math.pow(10, ABSORPTION_LOG_DECADES_MAX),
    current > 0 ? Math.min(nominalMin, current) : nominalMin
  );
  return { min, max };
}

/**
 * Readout text for a κ value. κ now ranges over decades, so a fixed
 * `toFixed(2)` would print "4035.77" for a thin-line scene and "0.00"
 * for everything below 0.005. Significant-digit formatting keeps the
 * readout short and informative across the whole track.
 */
export function formatAbsorption(v: number): string {
  if (!(v > 0)) return '0';
  if (v < 0.01) return v.toExponential(1);
  if (v < 10) return v.toFixed(2);
  if (v < 100) return v.toFixed(1);
  return String(Math.round(v));
}
