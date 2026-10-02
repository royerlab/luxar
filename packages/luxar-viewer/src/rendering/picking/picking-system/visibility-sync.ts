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
 * alike. An input whose pick uniform is missing, or whose visual value is not
 * a finite number, is skipped.
 *
 * @module rendering/picking/picking-system/visibility-sync
 */

import { isPhysicalMeshMaterial } from '../../materials/mesh-physical/config';
import { isVolumetricMode } from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';

interface UniformHolder {
  uniforms?: Record<string, { value: unknown } | undefined>;
  getOpacity?: () => number;
}

/** One mirrored input: the pick uniform, and how to read its value off the visual material. */
interface VisibilityInput {
  readonly pick: string;
  readonly read: (visual: UniformHolder, mode: BlendingMode) => unknown;
}

/** Read a uniform of the visual material by name. */
const uniformOf =
  (name: string) =>
  (visual: UniformHolder): unknown =>
    visual.uniforms?.[name]?.value;

/**
 * Every input the pick pass mirrors. Each pick material declares only the
 * pick-side names it consumes; the rest are skipped for it.
 *
 * - `uCoverageTruncate` ← `uTruncate`: the gsplat screen-coverage fade is a
 *   cull rule evaluated with the DRAW pass's T (per node), not with the
 *   tighter 1.5σ pick footprint radius the pick's own `uTruncate` holds.
 * - `uMaxExtentFactor`, `uCov2DDilation`: the other inputs of the gsplat
 *   coverage limit and of the dilated footprint.
 * - `uOpacity`: the LIVE node opacity. Its writers are the Layers panel, the
 *   LOD cross-fade (`scene/lod-fade.ts`) and the embedder exposure path, and
 *   only the first ever touched a pick material. A physical mesh keeps its
 *   opacity off the uniform record (three's PBR material), so it is read
 *   through `getOpacity()`.
 * - `uNearFade`: whether the mesh pick mirrors the house shader's near fade.
 *   A physical visual (three's PBR material) has none, so its pick must not
 *   discard the fade band either.
 * - `uIntensity`, `uHasElementAlpha`, `uVolumetric` (with `uOpacity`): the
 *   visual weight every point/line/gsplat pick multiplies its falloff by
 *   (`../_shared/visibility-glsl.ts`). `uVolumetric` has no visual uniform —
 *   the visual shaders branch on a compile-time define — so it is derived
 *   from the node's blending mode.
 * - `uProjectionMode`, `uRayIntegralFactor`: the gsplat amplitude the draw
 *   emits (sum projection's ray-integral boost vs the plain peak).
 */
const VISIBILITY_INPUTS: readonly VisibilityInput[] = [
  { pick: 'uCoverageTruncate', read: uniformOf('uTruncate') },
  { pick: 'uMaxExtentFactor', read: uniformOf('uMaxExtentFactor') },
  { pick: 'uCov2DDilation', read: uniformOf('uCov2DDilation') },
  {
    pick: 'uOpacity',
    read: (visual) => visual.uniforms?.uOpacity?.value ?? visual.getOpacity?.(),
  },
  { pick: 'uNearFade', read: (visual) => (isPhysicalMeshMaterial(visual) ? 0 : 1) },
  { pick: 'uIntensity', read: uniformOf('uIntensity') },
  { pick: 'uHasElementAlpha', read: uniformOf('uHasElementAlpha') },
  { pick: 'uVolumetric', read: (_visual, mode) => (isVolumetricMode(mode) ? 1 : 0) },
  { pick: 'uProjectionMode', read: uniformOf('uProjectionMode') },
  { pick: 'uRayIntegralFactor', read: uniformOf('uRayIntegralFactor') },
];

/** The first material of a (possibly multi-material) slot. */
function single(material: unknown): UniformHolder | undefined {
  return (Array.isArray(material) ? material[0] : material) as UniformHolder | undefined;
}

/**
 * Copy every visibility input `visual` carries onto `pick`. Cheap: a handful
 * of numeric assignments, run once per node per pick-buffer render.
 *
 * @param mode - the node's effective blending mode (as the pick depth sync
 *   resolves it), for the inputs derived from it rather than read.
 */
export function syncPickVisibilityInputs(pick: unknown, visual: unknown, mode: BlendingMode): void {
  const pickUniforms = single(pick)?.uniforms;
  const source = single(visual);
  if (!pickUniforms || !source) return;
  for (const input of VISIBILITY_INPUTS) {
    const target = pickUniforms[input.pick];
    if (!target) continue;
    const value = input.read(source, mode);
    if (typeof value !== 'number' || !Number.isFinite(value)) continue;
    if (target.value !== value) target.value = value;
  }
}
