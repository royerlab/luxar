/**
 * Pure, dependency-free opacity math for the two **orthogonal LOD anti-popping**
 * mechanisms, kept together (and THREE/camera-free) so both are unit-testable
 * over plain numbers, like `lod-freshness.ts` / `lod-display-gate.ts`.
 *
 * The two mechanisms act on independent axes (a change BETWEEN two levels; the
 * stream WITHIN one level) and compose by multiplying their opacity factors:
 *
 * 1. **Level dissolve (between levels)** — {@link smoothstep}. When a
 *    `kind=lod` group changes its displayed level, the registry dissolves the
 *    outgoing level into the incoming one over `config.lod.fadeMs` instead of
 *    a hard `object.visible` swap: the incoming at `smoothstep(elapsed/fadeMs)`,
 *    the outgoing at the complement (the state machine is `lod-dissolve.ts`).
 *    Brightness across the switch is preserved by the levels' build-time mass
 *    conservation (both integrate to the same DC).
 *
 * 2. **Streaming brightness compensation (within a level)** — {@link
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
