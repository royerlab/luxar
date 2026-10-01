/**
 * Visual → pick uniform sync: the inputs that decide WHETHER an element is
 * visible, copied from each node's visual material onto its pick material
 * before every pick render.
 *
 * A pick shader re-derives each element's footprint and visibility itself, so
 * every uniform the visual pass consults for "is this element on screen" must
 * reach the pick pass with the same value — otherwise an element culled or
 * faded out on screen stays pickable (or a visible one becomes unpickable).
 * Most of those uniforms are written on the VISUAL material only (by the
 * Layers panel, the LOD fade, the embedder API, the commit layer), so the
 * pick render pulls them across here rather than every writer having to know
 * the pick material exists.
 *
 * Duck-typed on `material.uniforms`, so one table serves the GLSL
 * `ShaderMaterial` records and the TSL wrappers' `proxyIUniform` records
 * alike. A pair whose uniform is missing on either side is skipped.
 *
 * @module rendering/picking/picking-system/visibility-sync
 */

/** `[pickUniform, visualUniform]` — copy the visual value onto the pick uniform. */
type UniformPair = readonly [pick: string, visual: string];

/**
 * Every pair the pick pass mirrors. Each pick material declares only the
 * pick-side names it consumes; the rest are skipped for it.
 *
 * - `uCoverageTruncate` ← `uTruncate`: the gsplat screen-coverage fade is a
 *   cull rule evaluated with the DRAW pass's T (per node), not with the
 *   tighter 1.5σ pick footprint radius the pick's own `uTruncate` holds.
 * - `uMaxExtentFactor`, `uCov2DDilation`: the other inputs of the gsplat
 *   coverage limit and of the dilated footprint.
 */
const VISIBILITY_UNIFORM_PAIRS: readonly UniformPair[] = [
  ['uCoverageTruncate', 'uTruncate'],
  ['uMaxExtentFactor', 'uMaxExtentFactor'],
  ['uCov2DDilation', 'uCov2DDilation'],
];

interface UniformHolder {
  uniforms?: Record<string, { value: unknown } | undefined>;
}

/** The first material of a (possibly multi-material) slot. */
function single(material: unknown): UniformHolder | undefined {
  return (Array.isArray(material) ? material[0] : material) as UniformHolder | undefined;
}

/**
 * Copy every visibility input `visual` carries onto `pick`. Cheap: a handful
 * of numeric assignments, run once per node per pick-buffer render.
 */
export function syncPickVisibilityInputs(pick: unknown, visual: unknown): void {
  const pickUniforms = single(pick)?.uniforms;
  const visualUniforms = single(visual)?.uniforms;
  if (!pickUniforms || !visualUniforms) return;
  for (const [pickName, visualName] of VISIBILITY_UNIFORM_PAIRS) {
    const target = pickUniforms[pickName];
    const source = visualUniforms[visualName];
    if (!target || !source || typeof source.value !== 'number') continue;
    if (target.value !== source.value) target.value = source.value;
  }
}
