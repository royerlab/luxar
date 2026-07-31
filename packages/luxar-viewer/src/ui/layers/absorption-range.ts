/**
 * Per-layer range for the volumetric absorption (κ) slider.
 *
 * κ is a PHYSICAL coefficient with units of 1/length: the volumetric
 * shaders build optical depth as `τ = κ · density · through-thickness`,
 * where the thickness is the geometry's own world-space size —
 * `width · LINE_CHORD_SCALE` for lines, `radius · POINT_CHORD_SCALE` for
 * points (materials/{line,point}/math.ts), and the ray integral through
 * Σ for gsplats. So the κ that produces a VISIBLE amount of absorption
 * scales as 1/thickness, and a single fixed slider range cannot serve
 * both a microscopy scene (σ ≈ voxels, κ ≈ 1) and a normalised-unit-cube
 * scene (width ≈ 1.5e-3, κ ≈ 10³). A fixed 0–10 range on the latter tops
 * out at τ ≈ 0.01 — a sub-1/255 change, i.e. a knob that visibly does
 * nothing.
 *
 * This module derives the slider's bounds per layer from the geometry
 * thickness the zarr writer already records (`max_width` for lines,
 * `max_radius` for points): the top of the track lands near
 * {@link ABSORPTION_TAU_TARGET} optical depth ("clearly opaque") for the
 * layer's THINNEST descendant, and the floor is anchored to its THICKEST
 * descendant so a mixed-thickness subtree can also reach near-transparency
 * for its fattest geometry — whatever the scene's units. Gsplats carry no
 * comparable thickness stat, and their τ = κ·opacity·rayMass is already
 * O(1)-calibrated for fitted volumes, so they keep
 * {@link ABSORPTION_DEFAULT_MAX}.
 *
 * @module ui/layers/absorption-range
 */

import type { SceneNode } from '../../data/data-loader-types';
import { LINE_CHORD_SCALE } from '../../rendering/materials/line/math';
import { POINT_CHORD_SCALE } from '../../rendering/materials/point/math';

/**
 * Optical depth the TOP of the slider should reach for a layer whose
 * thickness is known. τ = 5 ⇒ 1 − e^(−τ) ≈ 0.993: fully opaque, with the
 * whole transparent→opaque sweep below it.
 */
export const ABSORPTION_TAU_TARGET = 5.0;

/**
 * Fallback upper bound when no thickness stat is available (gsplats, or
 * a layer whose nodes predate the stats). Also the FLOOR of every derived
 * bound, so an auto-derived range can never make a previously reachable
 * authored κ unreachable.
 */
export const ABSORPTION_DEFAULT_MAX = 10.0;

/**
 * Hard ceiling on the derived bound. Guards against a degenerate
 * thickness (a zero / near-zero `max_width` from malformed metadata)
 * blowing the track up to absurd magnitudes.
 */
export const ABSORPTION_MAX_LIMIT = 1e6;

/** Decades of κ spanned by the log track below its upper bound. */
export const ABSORPTION_LOG_DECADES = 4;

/**
 * Hard cap on the track's span in decades, used when the CURRENT κ sits
 * below the nominal floor and the floor has to be lowered to include it.
 * Bounds how compressed the useful region can get for a κ that is already
 * visually indistinguishable from 0.
 */
export const ABSORPTION_LOG_DECADES_MAX = 8;

/**
 * Upper κ bound for one leaf node, from its recorded geometry thickness.
 * Returns `undefined` when the node carries no usable thickness stat
 * (gsplats, groups, or missing / non-positive metadata) so callers can
 * distinguish "no information" from "a derived bound".
 */
function leafAbsorptionMax(node: SceneNode): number | undefined {
  const thickness =
    node.type === 'lines'
      ? (node.attrs.max_width as number | undefined)
      : node.type === 'points'
        ? (node.attrs.max_radius as number | undefined)
        : undefined;
  if (typeof thickness !== 'number' || !Number.isFinite(thickness) || thickness <= 0) {
    return undefined;
  }
  const chord = node.type === 'lines' ? LINE_CHORD_SCALE : POINT_CHORD_SCALE;
  return ABSORPTION_TAU_TARGET / (thickness * chord);
}

/** Derived per-layer κ bounds — see {@link absorptionBoundsForNode}. */
export interface AbsorptionBounds {
  /** Largest per-leaf bound in the subtree: the THINNEST descendant's opaque point. */
  max: number;
  /**
   * Smallest per-leaf bound in the subtree: the THICKEST descendant's opaque
   * point. Equals `max` for a single-thickness subtree or when no thickness
   * stat is available.
   */
  minBound: number;
}

/**
 * κ bounds for a layer rooted at `node`, walking the whole subtree
 * (a group / kind=lod / kind=partition layer applies one κ to every
 * descendant material).
 *
 * `max` combines descendant bounds with max: the slider must be able to
 * reach a visible τ for the THINNEST geometry under the layer. A fatter
 * sibling simply saturates earlier along the track, which is still a
 * working knob; the other way round (taking the min) would reproduce the
 * "nothing happens" bug for the thin child.
 *
 * `minBound` is the opposite extreme — the THICKEST descendant's opaque
 * point — and anchors the track FLOOR (see {@link absorptionSliderRange}):
 * with the floor fixed relative to `max` alone, a fat sibling in a
 * mixed-thickness group was stuck at visible absorption (τ ≈ 0.1 for a
 * 250× thickness spread) at the lowest positive stop, with only the abrupt
 * zero stop below it.
 *
 * Both values are clamped to `[ABSORPTION_DEFAULT_MAX, ABSORPTION_MAX_LIMIT]`.
 */
export function absorptionBoundsForNode(node: SceneNode): AbsorptionBounds {
  let hi: number | undefined;
  let lo: number | undefined;
  const visit = (n: SceneNode): void => {
    const leafMax = leafAbsorptionMax(n);
    if (leafMax !== undefined) {
      hi = hi === undefined ? leafMax : Math.max(hi, leafMax);
      lo = lo === undefined ? leafMax : Math.min(lo, leafMax);
    }
    for (const c of n.children ?? []) visit(c);
  };
  visit(node);
  if (hi === undefined || lo === undefined) {
    return { max: ABSORPTION_DEFAULT_MAX, minBound: ABSORPTION_DEFAULT_MAX };
  }
  const clampBound = (v: number): number =>
    Math.min(ABSORPTION_MAX_LIMIT, Math.max(ABSORPTION_DEFAULT_MAX, v));
  return { max: clampBound(hi), minBound: clampBound(lo) };
}

/**
 * Log-track bounds for a layer's absorption slider.
 *
 * Both ends move so that the CURRENT κ lands ON the track over the whole
 * range a scene can meaningfully use — otherwise the readout would show a
 * value the thumb cannot represent, and a touch that moves nothing would
 * write the clamped end back:
 *
 *   - `max` is the layer's derived bound, RAISED to the current κ when an
 *     author set it above the derived "opaque" point.
 *   - `min` sits {@link ABSORPTION_LOG_DECADES} decades below the layer's
 *     `minBound` (the THICKEST descendant's opaque point — equal to `max`
 *     for a single-thickness layer), so a mixed-thickness group can reach
 *     near-transparency for its fattest geometry, not only its thinnest.
 *     It is further LOWERED to the current κ when that falls beneath it:
 *     very thin geometry derives a large `max` (a 6e-4-wide line gives
 *     ≈ 1.0e4), which would otherwise put the floor above the authored
 *     default κ = 1.
 *
 * TWO DELIBERATE CLAMPS bound that accommodation, because a pathological
 * authored κ (or thickness spread) would otherwise compress the useful
 * region off the track and recreate the original "the knob does nothing"
 * bug: `max` stops at {@link ABSORPTION_MAX_LIMIT}, and the floor —
 * however it was derived — stops at {@link ABSORPTION_LOG_DECADES_MAX}
 * decades of span. Outside those bounds
 * the thumb seats at the clamped end while the readout still shows the true
 * κ, so touching the slider writes the clamp back. That is accepted: past
 * the ceiling both κ are far beyond opaque, and below the floor both are
 * ≥ 7 decades below visible absorption — the states being swapped are
 * visually identical (see the `at the clamped extremes` tests).
 *
 * κ = 0 needs no room: it is the track's dedicated zero stop
 * (`LabeledSlider`, `scale: 'log'`).
 */
export function absorptionSliderRange(
  layerMax: number,
  currentValue: number,
  layerMinBound: number = layerMax
): { min: number; max: number } {
  const current = Number.isFinite(currentValue) ? currentValue : 0;
  const max = Math.min(
    ABSORPTION_MAX_LIMIT,
    Math.max(
      ABSORPTION_DEFAULT_MAX,
      Number.isFinite(layerMax) ? layerMax : ABSORPTION_DEFAULT_MAX,
      current
    )
  );
  const floorBase =
    Number.isFinite(layerMinBound) && layerMinBound > 0 ? Math.min(layerMinBound, max) : max;
  const nominalMin = floorBase / Math.pow(10, ABSORPTION_LOG_DECADES);
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
