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
 * This module derives the slider's upper bound per layer from the
 * geometry thickness the zarr writer already records (`max_width` for
 * lines, `max_radius` for points), so the top of the track always lands
 * near {@link ABSORPTION_TAU_TARGET} optical depth ("clearly opaque")
 * whatever the scene's units. Gsplats carry no comparable thickness stat,
 * and their τ = κ·opacity·rayMass is already O(1)-calibrated for fitted
 * volumes, so they keep {@link ABSORPTION_DEFAULT_MAX}.
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

/**
 * Upper κ bound for a layer rooted at `node`, walking the whole subtree
 * (a group / kind=lod / kind=partition layer applies one κ to every
 * descendant material).
 *
 * Descendant bounds combine with `max`: the slider must be able to reach
 * a visible τ for the THINNEST geometry under the layer. A fatter sibling
 * simply saturates earlier along the track, which is still a working
 * knob; the other way round (taking the min) would reproduce the
 * "nothing happens" bug for the thin child.
 *
 * The result is clamped to `[ABSORPTION_DEFAULT_MAX, ABSORPTION_MAX_LIMIT]`.
 */
export function absorptionMaxForNode(node: SceneNode): number {
  let derived: number | undefined;
  const visit = (n: SceneNode): void => {
    const leafMax = leafAbsorptionMax(n);
    if (leafMax !== undefined) derived = Math.max(derived ?? 0, leafMax);
    for (const c of n.children ?? []) visit(c);
  };
  visit(node);
  if (derived === undefined) return ABSORPTION_DEFAULT_MAX;
  return Math.min(ABSORPTION_MAX_LIMIT, Math.max(ABSORPTION_DEFAULT_MAX, derived));
}

/**
 * Log-track bounds for a layer's absorption slider.
 *
 * `max` is the layer's derived bound, widened when needed so the CURRENT
 * κ stays on the track (an author may set κ far above the derived
 * "opaque" point). `min` sits {@link ABSORPTION_LOG_DECADES} decades
 * below — κ = 0 remains reachable as the track's zero stop
 * (`LabeledSlider`, `scale: 'log'`).
 */
export function absorptionSliderRange(
  layerMax: number,
  currentValue: number
): { min: number; max: number } {
  const max = Math.max(
    ABSORPTION_DEFAULT_MAX,
    Number.isFinite(layerMax) ? layerMax : ABSORPTION_DEFAULT_MAX,
    Number.isFinite(currentValue) ? currentValue : 0
  );
  return { min: max / Math.pow(10, ABSORPTION_LOG_DECADES), max };
}

/**
 * Readout text for a κ value. κ now ranges over decades, so a fixed
 * `toFixed(2)` would print "2858.55" for a thin-line scene and "0.00"
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
