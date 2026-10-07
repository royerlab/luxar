/**
 * Visual → pick sync: everything that decides WHETHER an element is visible,
 * and which of two overlapping elements the user sees, copied from each
 * node's visual material onto its pick material before every pick render —
 * the one place the pick pass learns about the node it stands in for.
 *
 * A pick shader re-derives each element's footprint and visibility itself, so
 * every input the visual pass consults for "is this element on screen" must
 * reach the pick pass with the same value — otherwise an element culled or
 * faded out on screen stays pickable (or a visible one becomes unpickable).
 * Most of those inputs are written on the VISUAL material only (by the Layers
 * panel, the LOD fade, the density guard, the embedder API, the commit layer),
 * so the pick render pulls them across here rather than every writer having
 * to know the pick material exists.
 *
 * Two parts, both run by {@link syncPickFromVisual}:
 *
 *   1. The UNIFORM inputs ({@link PICK_SYNC_INPUTS}): a table keyed by the
 *      pick-side uniform. Duck-typed on `material.uniforms`, so it serves the
 *      GLSL `ShaderMaterial` records and the TSL wrappers' `proxyIUniform`
 *      records alike; each pick material declares exactly the uniforms it
 *      consumes, and an input whose pick uniform is missing (or whose visual
 *      value is not a finite number) is skipped. Which inputs reach which pick
 *      type is therefore the set of uniforms that type declares —
 *      {@link pickSyncedInputs} lists it.
 *   2. The DEPTH CONVENTION: front-most wins under the depth-ordered surface
 *      modes, brightest wins under the commutative ones, through the pick
 *      material's capability (`MeshPickAwareMaterial` for mesh, which also
 *      carries the `opaque` cutout and face culling; `SurfacePickAwareMaterial`
 *      for points, lines and gsplats).
 *
 * The camera half (pick-target resolution, near cull, pixel ratio, the TSL
 * line graphs' ortho variant) is NOT here: it is the `CameraAwareMaterial`
 * broadcast `renderPickBuffer` re-issues for the half-resolution pick target.
 * Every shader derives the ortho branch itself from the projection matrix.
 *
 * @module rendering/picking/picking-system/visibility-sync
 */

import type * as THREE from 'three';
import { isPhysicalMeshMaterial } from '../../materials/mesh-physical/config';
import { getDensityDrop } from '../../materials/_shared/density-drop';
import { isNormalMode, isOpaqueMode, isVolumetricMode } from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import { isMeshPickAwareMaterial } from '../mesh/pick-mode';
import { isSurfacePickAwareMaterial } from '../_shared/surface-pick';

export interface UniformHolder {
  uniforms?: Record<string, { value: unknown } | undefined>;
  getOpacity?: () => number;
}

/** One mirrored input: the pick uniform, and how to read its value off the visual material. */
export interface PickSyncInput {
  readonly pick: string;
  readonly read: (visual: UniformHolder, mode: BlendingMode) => unknown;
}

/** Read a uniform of the visual material by name. */
const uniformOf =
  (name: string) =>
  (visual: UniformHolder): unknown =>
    visual.uniforms?.[name]?.value;

/**
 * Every uniform input the pick pass mirrors, keyed by the pick-side name.
 *
 * - `uDensityDrop`: the density guard's thinning. The pick pass must drop
 *   exactly the elements the visual pass drops (same hash of the same storage
 *   index), or hovering a thinned-away element would resolve a pick the user
 *   cannot see. Read through `getDensityDrop`, so a visual without the
 *   uniform reads 0 (nothing dropped).
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
export const PICK_SYNC_INPUTS: readonly PickSyncInput[] = [
  { pick: 'uDensityDrop', read: (visual) => getDensityDrop(visual) },
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
function single(material: unknown): THREE.Material | undefined {
  return (Array.isArray(material) ? material[0] : material) as THREE.Material | undefined;
}

/**
 * The blending mode a node's visual material composites with: a physical mesh
 * follows its own translucency (three's PBR material carries no Luxar mode),
 * every other material its `userData.blendingMode` (additive when unset).
 */
export function effectiveBlendingMode(material: THREE.Material | undefined): BlendingMode {
  if (material && isPhysicalMeshMaterial(material)) {
    return material.transparent ? 'normal' : 'opaque';
  }
  return (material?.userData.blendingMode ?? 'additive') as BlendingMode;
}

/** The pick-side names of the inputs `pick` consumes (introspection, tests, docs). */
export function pickSyncedInputs(pick: unknown): string[] {
  const uniforms = (single(pick) as UniformHolder | undefined)?.uniforms;
  return uniforms ? PICK_SYNC_INPUTS.map((i) => i.pick).filter((name) => name in uniforms) : [];
}

function syncUniformInputs(pick: unknown, visual: UniformHolder, mode: BlendingMode): void {
  const pickUniforms = (single(pick) as UniformHolder | undefined)?.uniforms;
  if (!pickUniforms) return;
  for (const input of PICK_SYNC_INPUTS) {
    const target = pickUniforms[input.pick];
    if (!target) continue;
    const value = input.read(visual, mode);
    if (typeof value !== 'number' || !Number.isFinite(value)) continue;
    if (target.value !== value) target.value = value;
  }
}

/**
 * Pick depth convention: under the depth-ordered surface modes — 'normal'
 * (sorted alpha-over) and 'opaque' (depth-written) — the user sees an
 * occluding surface, so the pick depth must be the real projected depth
 * (front-most wins) instead of brightness-as-depth (brightest wins — right for
 * the commutative additive/luminous/max modes, but it could pick a brighter
 * element BEHIND the visible surface). 'volumetric' is DELIBERATELY excluded
 * (isNormalMode || isOpaqueMode, NOT needsDepthSort): it is emissive, so it
 * keeps brightness picking — a heavily-absorbed back splat can still win if
 * brightest; front-most-beyond-a-τ-threshold is a spec'd follow-up
 * (VOLUMETRIC_BLENDING_SPEC.md §5.2).
 *
 * MESH implements the richer `MeshPickAwareMaterial` instead, because its
 * blending mode has a SECOND pick-pass consequence the boolean cannot carry —
 * the `opaque` alpha cutout (`../mesh/pick-mode.ts`). It also needs the
 * epoch's face culling copied over, which no other pick material does: the
 * siblings' quads are view-facing, but a mesh whose back faces are culled on
 * screen must not rasterize them into the pick buffer at true surface depth.
 */
function syncDepthConvention(
  pick: unknown,
  visual: THREE.Material | undefined,
  mode: BlendingMode
): void {
  if (isMeshPickAwareMaterial(pick)) {
    pick.setPickMode(mode);
    if (visual) pick.setPickSide(visual.side);
  } else if (isSurfacePickAwareMaterial(pick)) {
    pick.setSurfacePickDepth(isNormalMode(mode) || isOpaqueMode(mode));
  }
}

/**
 * Bring `pick` in line with the node's visual material(s): every uniform input
 * it consumes, and its depth convention. Cheap — a handful of numeric
 * assignments — and run once per node per pick-buffer render.
 */
export function syncPickFromVisual(
  pick: unknown,
  visual: THREE.Material | THREE.Material[] | undefined
): void {
  const source = single(visual);
  const mode = effectiveBlendingMode(source);
  syncUniformInputs(pick, (source ?? {}) as UniformHolder, mode);
  syncDepthConvention(pick, source, mode);
}
