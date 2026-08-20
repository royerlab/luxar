/**
 * Pure display-window ↔ shader-uniform math.
 *
 * Converts between a display ``[min, max]`` window and the shader's
 * intensity (gain) / offset uniforms. These helpers pull in no THREE and no
 * ``ui/`` imports, and live in the rendering layer so both ``rendering/`` and
 * ``ui/layers/`` can import them without violating the module layer order
 * (``rendering/`` must never import from the higher ``ui/`` layer). The one
 * import — ``_shared/scalar-range`` — is itself in that same dependency-free
 * class, and is here so ``isRemappableRange`` collapses at exactly the floor the
 * MATERIAL side (``computeScalarRangeUniforms``) collapses at. It is not a
 * file-wide unification: ``computeUniforms`` and ``computeDisplayRange`` below
 * still spell their own ``1e-10`` literals, and deliberately so — the latter's
 * is a threshold on an INTENSITY (a gain), a different quantity that merely
 * happens to share the value.
 *
 * @module rendering/display-range
 */

import { DEGENERATE_SCALAR_RANGE_EPS } from './materials/_shared/scalar-range';

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
 * Whether a data range can be mapped through: wide enough to divide by, and
 * finite.
 *
 * The width floor is the SHARED `DEGENERATE_SCALAR_RANGE_EPS`, imported rather
 * than re-spelled, because the two ends of this pipeline have to agree on where
 * "degenerate" starts: `computeUniforms` collapses to the identity gain below
 * `1e-10` on the way in, and `computeScalarRangeUniforms` collapses to the LUT
 * midpoint below `1e-10` on the way out. A plain `hi > lo` accepted the sliver
 * in between and remapped a window that was no longer stated in the reference
 * basis at all.
 *
 * Finiteness is a separate arm, not a consequence: `[0, Infinity]` passes any
 * width test, and then `t₀ = 0` times an infinite span is `NaN`, which would
 * reach the shader uniforms as a NaN scalar window. `Number.isFinite` on the
 * SPAN also rejects a NaN-bearing range for free (every arithmetic on NaN is
 * NaN, and every comparison with NaN is false).
 *
 * An INVERTED range is caught by neither of those: its span is finite and merely
 * negative. It falls out of `span >= DEGENERATE_SCALAR_RANGE_EPS`, which a
 * negative span fails just as a zero one does.
 */
function isRemappableRange(range: readonly [number, number]): boolean {
  const span = range[1] - range[0];
  return Number.isFinite(span) && span >= DEGENERATE_SCALAR_RANGE_EPS;
}

/**
 * Re-express a scalar LUT window stated on a layer's REFERENCE data range in the
 * units of one leaf's OWN data range, keeping the window's relative position
 * inside the range.
 *
 * The layers panel composes ONE display window per layer, but a `kind=lod` layer
 * fans it out over LEVELS whose scalars do not share a range: gsplat LOD merging
 * SUMS amplitudes, so a coarsened level carries a different
 * `amplitude_data_range` from the finest one for the same physical signal. The
 * wrapper's reference range is just whichever descendant
 * `deriveScalarRangeFromDescendants` picked — the one with the largest
 * `n_splats`, or, when no descendant declares one (points/lines/mesh write
 * `n_points`, never `n_splats`) or several tie, simply the first visited.
 * Pushing that one window into every level renders the whole layer on the
 * reference level's window — the per-level differentiation the producer stamped
 * is discarded, and a level created lazily after the last panel commit renders
 * on its own window until the next one, so two levels of the same object can be
 * windowed differently depending only on load order (#1753).
 *
 * This is a helper about RANGES, not about structure, so it does not know which
 * sibling relations may be remapped across. The caller decides:
 * `LayerApplyEngine.composedWindowIsInReferenceBasis` allows it only across
 * `kind=lod` groups, never across a `kind=partition` boundary — partition parts
 * are disjoint spatial subsets of one field at the SAME scale, so re-expressing
 * the window per part is per-tile auto-contrast and puts a colour discontinuity
 * at every seam.
 *
 * Mapping the window through `t = (w − ref₀) / (ref₁ − ref₀)` and back out on the
 * leaf's range means "the middle 40% of the layer's signal" stays the middle 40%
 * of each leaf's signal, which is what the slider means to the user.
 *
 * The window is returned UNCHANGED — no remap at all — in four cases, each of
 * which would otherwise silently produce a *different* window rather than a
 * refined one:
 *
 * - **Either range missing.** Nothing to map between; the incoming window is
 *   already the best available answer.
 * - **A non-remappable reference range** (`isRemappableRange`, file-private). A
 *   degenerate `[x, x]` range is legitimate — every classical splat import has
 *   `amplitudes = 1`, and constant-amplitude producers emit it on purpose — and
 *   dividing by its ~zero span would yield ±∞/NaN. Below `1e-10` there is a
 *   sharper reason than the arithmetic: the panel seeds the layer's window FROM
 *   this range, so a degenerate one means the incoming window is a single point
 *   as well, and `t₀`/`t₁` come out as 0/0 or an arbitrary multiple of 1e10.
 * - **A non-remappable leaf range.** A degenerate leaf range collapses the
 *   window to a single point, and `updateScalarRange` hands that straight to
 *   `computeScalarRangeUniforms`, which answers a sub-eps span with the LUT
 *   MIDPOINT (`scalarMin = min − 0.5`, `scalarScale = 1`, deliberately, #631) —
 *   so every element of that leaf would render as one flat neutral colour.
 * - **The two ranges are equal.** The overwhelmingly common case (the finest
 *   level IS the reference, and every leaf of an ordinary single-range layer).
 *   Short-circuiting keeps the window bit-exact instead of round-tripping it
 *   through two floating-point divisions.
 *
 * @param window     The layer's OWN window (`LayerInfo.displayMin/Max`), which
 *                   is the one stated in `ref` units. Deliberately not the
 *                   COMPOSED window: that one already carries whatever gain the
 *                   ancestry above the layer contributes, so reading it as a
 *                   position inside `ref` is a basis error (it cancels only when
 *                   `ref₀/refSpan === leafRange₀/leafSpan`). The caller
 *                   re-applies the ancestor gain to the result instead.
 * @param ref        The layer's reference scalar range (`LayerInfo.scalarDataRange`).
 * @param leafRange  This leaf's own `scalar_data_range` / `amplitude_data_range`.
 */
export function remapWindowToLeafRange(
  window: { min: number; max: number },
  ref: readonly [number, number] | undefined,
  leafRange: readonly [number, number] | undefined
): { min: number; max: number } {
  if (!ref || !leafRange) return window;
  if (!isRemappableRange(ref)) return window;
  if (!isRemappableRange(leafRange)) return window;
  if (leafRange[0] === ref[0] && leafRange[1] === ref[1]) return window;
  const refSpan = ref[1] - ref[0];
  const leafSpan = leafRange[1] - leafRange[0];
  const t0 = (window.min - ref[0]) / refSpan;
  const t1 = (window.max - ref[0]) / refSpan;
  return {
    min: leafRange[0] + t0 * leafSpan,
    max: leafRange[0] + t1 * leafSpan,
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
