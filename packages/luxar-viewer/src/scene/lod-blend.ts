/**
 * Pure, dependency-free opacity math for the two **orthogonal LOD anti-popping**
 * mechanisms, kept together (and THREE/camera-free) so both are unit-testable
 * over plain numbers, like `lod-freshness.ts` / `lod-display-gate.ts`.
 *
 * The two mechanisms act on independent axes and compose by multiplying their
 * opacity factors:
 *
 * 1. **Coverage cross-fade (distance axis)** — {@link coverageBlendPlan}. As the
 *    camera zooms across a `kind=lod` boundary, blend the two adjacent levels'
 *    opacity instead of a hard `object.visible` swap. Driven ENTIRELY by the
 *    group's selector metric (projected size); which level to show is a
 *    function of distance (`coverage_fraction`). Brightness across the switch is
 *    preserved by the levels' build-time mass conservation (both integrate to
 *    the same DC).
 *
 * 2. **Streaming brightness compensation (time axis)** — {@link
 *    energyCompensation}. As a single level's additive ladder streams in over
 *    time, its rendered energy climbs from `e(k)·E` toward `E` (additive/luminous
 *    compositing sums energy; the ladder commits highest-energy splats first),
 *    which reads as a brightening pop. Scaling opacity by `1/e(k)` — opacity is a
 *    linear multiplier on summed energy in additive/luminous, and on optical
 *    depth `τ` in volumetric — holds the total near `E` throughout. This is a
 *    self-energy heuristic rather than the exact conservation law used by
 *    mechanism 1: `e(k)` is quadratic in amplitude while additive brightness is
 *    linear, and the approximation becomes exact only when the leaf is complete.
 *
 * Both mechanisms apply to the modes in `BLENDABLE_MODES`
 * (`scene/lod-fade.ts`) — additive / luminous / volumetric; see that set's doc
 * for the per-mode exactness argument and the volumetric caveat.
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
 * Decide the coverage-band cross-fade for a scalar `metric` — whatever the
 * group's `selector` names, i.e. the viewport AREA fraction under
 * `'screen-area'` (what derived ladders stamp) or the normalised diagonal
 * (projected bbox diagonal ÷ `FILL_FACTOR·fittedAxisPx`, where `fittedAxisPx`
 * is `min(viewport.width, viewport.height)`) under the legacy `'coverage'` —
 * against a level's ascending per-child `coverage_fraction` thresholds
 * (coarsest 0 → finest 0.5 whole-object / 1.0 partition-bound under
 * `'screen-area'`; finest 1 for a legacy whole-object ladder, up to
 * `SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR` for an explicitly authored or
 * partition-bound one). Everything here is proportional to the inter-threshold
 * gaps, so the band scales with the ladder either way.
 *
 * Each inter-level boundary is the activation threshold of the finer level
 * (`thresholds[i+1]`). The band half-width is PROPORTIONAL to the local
 * inter-level spacing — `Δ_i = fraction · min(gapBelow, gapAbove)` — because the
 * `coverage_fraction` thresholds are geometrically spaced either way (a derived
 * `'screen-area'` ladder halves EXACTLY per coarser level, whatever the
 * compression factor K; a derived legacy `'coverage'` one steps by `√K`, from
 * the retired `sqrt(N_i/N_finest)` derivation), so a constant band would be a
 * clean dissolve at the finest step yet many times wider than the whole step at
 * the coarse end (perpetually-blended coarse levels, overlapping bands).
 * Scaling to the smaller adjacent gap makes the dissolve feel the same fraction
 * of a step at every level and guarantees no two bands overlap (for
 * `fraction ≤ 0.5`).
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

/**
 * Brightness-compensation factor for a streaming blendable LOD leaf
 * (mechanism 2 in the module doc; additive / luminous / volumetric).
 *
 * As a leaf's additive ladder streams in, its committed prefix carries only `e`
 * (∈ (0, 1]) of the leaf's full self-energy `E`, so it renders at `e·E` and
 * brightens toward `E` as chunks arrive — a pop. Because opacity linearly scales
 * summed energy in additive/luminous compositing (and per-ray optical depth τ
 * in volumetric), multiplying opacity by `1/e` holds the partial prefix near
 * the full `E` at every step. This is a heuristic, not an exact compensation:
 * e(k) is a squared-amplitude (self-energy) fraction while additive brightness
 * is linear in amplitude, so a 2:1 disjoint pair at k = 1 renders at mass 2.5
 * of 3. The factor converges to 1 as `e → 1`, so the final frame is exact.
 *
 * `floor` caps the boost at `1/floor` so a tiny early prefix can't over-brighten
 * its (energy-descending, hence core-heavy) splats into tone-map clipping.
 *
 * Returns `1` (no compensation, so the leaf is left byte-identical) whenever
 * there is nothing to compensate: `e` absent (unstamped dataset), already
 * complete (`e >= 1`; a non-progressive leaf stamps `1`), or degenerate (`e <= 0`
 * before any chunk commits, or `NaN`).
 */
export function energyCompensation(e: number | undefined, floor: number): number {
  if (typeof e !== 'number' || !(e > 0) || e >= 1) return 1;
  return 1 / Math.max(e, floor);
}
