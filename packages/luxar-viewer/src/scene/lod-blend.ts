/**
 * Pure, dependency-free math for the **substitutive-LOD cross-fade** — the
 * smooth opacity blend between two adjacent `kind=lod` levels as the camera
 * zooms across their boundary, replacing the registry's hard `object.visible`
 * swap (which pops).
 *
 * The blend is driven ENTIRELY by the viewport **coverage metric** (the
 * projected size / distance) — NOT by how a level's data streams in. Which
 * substitutive level to show is a function of distance (`coverage_fraction`),
 * so the transition between two levels is smoothed across a distance band
 * around their boundary. The additive ladder (how a single level loads
 * progressively) is an orthogonal, data-loading concern and plays no part
 * here. Brightness is preserved by the levels' build-time mass conservation
 * (both integrate to the same DC), not by anything in this module.
 *
 * Kept out of `lod-group-registry.ts` (THREE/camera-bound) so the math is
 * unit-testable over plain numbers, like `lod-freshness.ts` /
 * `lod-display-gate.ts`.
 *
 * @module scene/lod-blend
 */

/** Clamp `x` into `[0, 1]`. */
function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/**
 * GLSL-style smoothstep: the Hermite curve `3t²−2t³` remapped so it is 0 at (or
 * below) `edge0`, 1 at (or above) `edge1`, and 0.5 at their midpoint. A
 * degenerate `edge0 >= edge1` is treated as a hard step at `edge0` (no NaN from
 * a zero-width divide).
 */
export function smoothstep(edge0: number, edge1: number, x: number): number {
  if (edge1 <= edge0) return x < edge0 ? 0 : 1;
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

/**
 * A coverage-band cross-fade plan: the two adjacent levels bracketing the LOD
 * boundary the metric is currently crossing, and the FINER level's weight.
 * `lo` = coarser child index, `hi` = finer child index (== `lo + 1`),
 * `hiWeight` ∈ [0, 1] = the finer level's opacity (the coarser gets
 * `1 − hiWeight`). `hiWeight` = 0.5 exactly at the boundary.
 */
export interface CoverageBlend {
  lo: number;
  hi: number;
  hiWeight: number;
}

/**
 * Decide the coverage-band cross-fade for a dimensionless `metric` (projected
 * bbox diagonal ÷ `FILL_FACTOR·viewportDiagonal`) against a level's ascending
 * per-child `coverage_fraction` thresholds (coarsest 0 → finest 1).
 *
 * Each inter-level boundary is the activation threshold of the finer level
 * (`thresholds[i+1]`). The band half-width is PROPORTIONAL to the local
 * inter-level spacing — `Δ_i = fraction · min(gapBelow, gapAbove)` — because the
 * `coverage_fraction` thresholds are geometrically spaced (they roughly halve
 * per coarser level for a K=4 ladder), so a constant band would be a clean
 * dissolve at the finest step yet many times wider than the whole step at the
 * coarse end (perpetually-blended coarse levels, overlapping bands). Scaling to
 * the smaller adjacent gap makes the dissolve feel the same fraction of a step
 * at every level and guarantees no two bands overlap (for `fraction ≤ 0.5`).
 *
 * When the metric is within `±Δ_i` of the NEAREST such boundary, the two levels
 * straddling it cross-fade: the finer level's weight is
 * `smoothstep(boundary − Δ_i, boundary + Δ_i, metric)`, so it rises 0→1 across
 * the band and is 0.5 at the boundary. The blend is continuous through the hard
 * selection flip (which happens at the boundary): the same `(lo, hi)` pair and a
 * continuous `hiWeight` govern both sides.
 *
 * Returns `null` when no boundary's band contains the metric (a single level
 * suffices), when `fraction <= 0` (cross-fade disabled → hard step), or when
 * there are fewer than two levels. Direction-agnostic — zoom-in and zoom-out
 * cross the same band symmetrically.
 */
export function coverageBlendPlan(
  thresholds: readonly number[],
  metric: number,
  fraction: number
): CoverageBlend | null {
  if (fraction <= 0 || thresholds.length < 2) return null;
  const n = thresholds.length;
  // Nearest inter-level boundary whose (proportional) band contains the metric.
  let best = -1;
  let bestDist = Infinity;
  let bestHalf = 0;
  for (let i = 0; i + 1 < n; i++) {
    const boundary = thresholds[i + 1];
    const gapBelow = boundary - thresholds[i];
    // Finest boundary has no coarser-side neighbour above it → reuse gapBelow.
    const gapAbove = i + 2 < n ? thresholds[i + 2] - boundary : gapBelow;
    const half = fraction * Math.min(gapBelow, gapAbove);
    if (half <= 0) continue; // degenerate: coincident thresholds
    const dist = Math.abs(metric - boundary);
    if (dist < half && dist < bestDist) {
      bestDist = dist;
      best = i;
      bestHalf = half;
    }
  }
  if (best < 0) return null;
  const boundary = thresholds[best + 1];
  const hiWeight = smoothstep(boundary - bestHalf, boundary + bestHalf, metric);
  return { lo: best, hi: best + 1, hiWeight };
}
