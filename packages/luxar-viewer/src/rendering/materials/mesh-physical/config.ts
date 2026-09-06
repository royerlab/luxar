/**
 * The physical mesh material's configuration, defaults and the ONE mapping from
 * Luxar attrs onto three's material properties.
 *
 * Both wrappers — `PhysicalMeshMaterial extends THREE.MeshPhysicalMaterial` (WebGL)
 * and `PhysicalMeshTSLMaterial extends MeshPhysicalNodeMaterial` (WebGPU) — call the
 * functions in this file rather than carrying their own copies, because the two three
 * classes share every property name this module writes. One mapping means the two
 * backends cannot disagree about what `roughness=0.4` or `opacity=0.3` does; two
 * copies would be a divergence waiting for the next knob.
 *
 * Nothing here imports `three/webgpu`: the TSL twin lives in the lazy cone and reaches
 * these helpers, never the other way round.
 *
 * @module rendering/materials/mesh-physical/config
 */

import type * as THREE from 'three';
import type { BlendingMode } from '../../../types/blending';
import { clampAppearanceFraction } from '../mesh/appearance';

/**
 * The Phase 1 knobs that are `[0, 1]` fractions, in Layers-panel order. Snake_case
 * because these are the ATTR names — `createMeshNode` reads them off the node attrs
 * and the panel lists them by the same names an author typed.
 */
export const PHYSICAL_MESH_KNOB_KEYS = [
  'roughness',
  'metalness',
  'clearcoat',
  'clearcoat_roughness',
  'iridescence',
  'sheen',
] as const;

export type PhysicalMeshKnobKey = (typeof PHYSICAL_MESH_KNOB_KEYS)[number];

/**
 * What an ABSENT knob renders as. Three's own defaults, with one deliberate
 * exception: `sheenColor` is white where three's is black, because a black sheen is
 * a no-op and an authored `sheen=1` with no colour would then render nothing — the
 * silent failure the whole material contract exists to refuse. The Python writer
 * never stamps these, so an absent attr on disk IS the default.
 */
export const PHYSICAL_MESH_DEFAULTS = {
  roughness: 1.0,
  metalness: 0.0,
  clearcoat: 0.0,
  clearcoatRoughness: 0.0,
  iridescence: 0.0,
  sheen: 0.0,
  sheenColor: '#ffffff',
} as const;

/** Construction-time config for either physical wrapper. */
export interface PhysicalMeshMaterialConfig {
  /** Node opacity, default 1. */
  opacity?: number;
  /** Base-colour gain (Luxar `intensity`), default 1. See {@link physicalUpdateIntensity}. */
  intensity?: number;
  /** Additive radiance (Luxar `offset`), default 0. See {@link physicalUpdateOffset}. */
  offset?: number;
  /** Recorded, never applied — see {@link physicalUpdateGamma}. */
  gamma?: number;
  roughness?: number;
  metalness?: number;
  clearcoat?: number;
  clearcoatRoughness?: number;
  iridescence?: number;
  sheen?: number;
  /** `#rrggbb`; default {@link PHYSICAL_MESH_DEFAULTS}.sheenColor. */
  sheenColor?: string;
  /**
   * Luxar `alpha_cutoff`, mapped onto three's `alphaTest`. On an RGBA mesh its
   * PRESENCE also selects the cutout path over translucency — see
   * {@link derivePhysicalCompositing}.
   */
  alphaCutoff?: number;
  /** `shading="flat"` (or no stored normals): derive normals per triangle. */
  flatShading?: boolean;
  /** Whether the geometry's `color` attribute carries a fourth (alpha) component. */
  vertexAlpha?: boolean;
}

/**
 * The properties this module writes — the intersection of `THREE.MeshPhysicalMaterial`
 * and `MeshPhysicalNodeMaterial` it relies on. Typed structurally so the helpers accept
 * either class (and a plain stub in a unit test) without importing `three/webgpu`.
 */
export interface PhysicalMeshHost {
  opacity: number;
  transparent: boolean;
  depthWrite: boolean;
  alphaTest: number;
  color: THREE.Color;
  emissive: THREE.Color;
  sheenColor: THREE.Color;
  roughness: number;
  metalness: number;
  clearcoat: number;
  clearcoatRoughness: number;
  iridescence: number;
  sheen: number;
  vertexColors: boolean;
  flatShading: boolean;
  toneMapped: boolean;
  needsUpdate: boolean;
  userData: Record<string, unknown>;
}

/** The three inputs the compositing decision is a pure function of. */
export interface PhysicalCompositingInputs {
  opacity: number;
  vertexAlpha: boolean;
  alphaCutoff: number | undefined;
}

/** The compositing decision, as three material state plus the coordinator's stamp. */
export interface PhysicalCompositing {
  transparent: boolean;
  depthWrite: boolean;
  alphaTest: number;
  /**
   * What `userData.blendingMode` is stamped as. `'opaque'` when the surface is
   * opaque, so the depth-sort coordinator releases it and the pick pass applies the
   * same cutout; UNSET when translucent, so the coordinator still releases it (a
   * physical mesh is never triangle-sorted — spec §3.2) and the pick pass takes
   * every fragment, matching what is drawn.
   */
  blendingMode: Extract<BlendingMode, 'opaque'> | undefined;
}

/**
 * Decide how a physical mesh composites, from its three inputs.
 *
 * A physical mesh has no Luxar blending mode — that knob is refused at authoring
 * — so translucency is read off the data instead:
 *
 * - `opacity < 1` is translucent, always.
 * - Per-vertex alpha is translucent UNLESS an `alpha_cutoff` was authored, in
 *   which case the alpha is a CUTOUT (three `alphaTest`) and the surface stays
 *   opaque — the same meaning the house shader's `opaque` mode gives the pair.
 *
 * Translucent surfaces do not write depth. With no per-triangle sort (spec §3.2)
 * that is the one choice that never drops a back face behind a front one; the
 * remaining order dependence is the ordinary alpha-over kind and is invisible on
 * a shell of uniform alpha, which is the marker case this phase exists for.
 */
export function derivePhysicalCompositing(inputs: PhysicalCompositingInputs): PhysicalCompositing {
  const cutout = inputs.vertexAlpha && inputs.alphaCutoff !== undefined;
  const translucent = inputs.opacity < 1 || (inputs.vertexAlpha && !cutout);
  return {
    transparent: translucent,
    depthWrite: !translucent,
    alphaTest: translucent ? 0 : clampAppearanceFraction(inputs.alphaCutoff, 0),
    blendingMode: translucent ? undefined : 'opaque',
  };
}

/**
 * Where the compositing inputs live on the material, so `updateOpacity` can re-derive
 * the decision without being told the other two again.
 */
const INPUTS_KEY = 'physicalCompositing';

function readInputs(host: PhysicalMeshHost): PhysicalCompositingInputs {
  const stored = host.userData[INPUTS_KEY] as PhysicalCompositingInputs | undefined;
  return stored ?? { opacity: host.opacity, vertexAlpha: false, alphaCutoff: undefined };
}

/**
 * Write a compositing decision onto the material, flagging a rebuild only when a
 * PROGRAM-affecting property changed (`transparent` and `alphaTest` both select a
 * shader variant on both backends; `depthWrite` and `opacity` are plain state).
 */
export function applyPhysicalCompositing(
  host: PhysicalMeshHost,
  inputs: PhysicalCompositingInputs
): void {
  const decision = derivePhysicalCompositing(inputs);
  const programChanged =
    host.transparent !== decision.transparent || host.alphaTest !== decision.alphaTest;
  host.userData[INPUTS_KEY] = { ...inputs };
  host.opacity = inputs.opacity;
  host.transparent = decision.transparent;
  host.depthWrite = decision.depthWrite;
  host.alphaTest = decision.alphaTest;
  if (decision.blendingMode) host.userData.blendingMode = decision.blendingMode;
  else delete host.userData.blendingMode;
  if (programChanged) host.needsUpdate = true;
}

/**
 * Configure a freshly constructed physical material from Luxar attrs.
 *
 * Knobs are clamped to `[0, 1]` with the same `clampAppearanceFraction` policy the
 * house knobs use (NaN/Inf → the documented default), because they arrive from
 * authored metadata rather than code: three clamps them too, but silently, and a
 * corrupt value should resolve to a default rather than a boundary that looks chosen.
 */
export function applyPhysicalMeshConfig(
  host: PhysicalMeshHost,
  config: PhysicalMeshMaterialConfig
): void {
  // The base colour is the vertex colour; `color` is left as a scalar GAIN so
  // `intensity` has the same meaning it has on the house shader.
  host.vertexColors = true;
  // The scene pass runs with `renderer.toneMapping = NoToneMapping` and tone-maps in
  // post, exactly like every emissive node; stated on the material too so a future
  // renderer-level tone map can never double-apply.
  host.toneMapped = false;
  host.flatShading = config.flatShading === true;

  host.roughness = clampAppearanceFraction(config.roughness, PHYSICAL_MESH_DEFAULTS.roughness);
  host.metalness = clampAppearanceFraction(config.metalness, PHYSICAL_MESH_DEFAULTS.metalness);
  host.clearcoat = clampAppearanceFraction(config.clearcoat, PHYSICAL_MESH_DEFAULTS.clearcoat);
  host.clearcoatRoughness = clampAppearanceFraction(
    config.clearcoatRoughness,
    PHYSICAL_MESH_DEFAULTS.clearcoatRoughness
  );
  host.iridescence = clampAppearanceFraction(
    config.iridescence,
    PHYSICAL_MESH_DEFAULTS.iridescence
  );
  host.sheen = clampAppearanceFraction(config.sheen, PHYSICAL_MESH_DEFAULTS.sheen);
  // `Color.set('#rrggbb')` reads the hex as sRGB and converts to the working
  // (linear) space under three's colour management — the human reading of a hex
  // tint, and what the Python `sheen_color` validator promises.
  host.sheenColor.set(config.sheenColor ?? PHYSICAL_MESH_DEFAULTS.sheenColor);

  physicalUpdateIntensity(host, config.intensity ?? 1.0);
  physicalUpdateOffset(host, config.offset ?? 0.0);
  physicalUpdateGamma(host, config.gamma ?? 1.0);

  host.userData.material = 'physical';
  applyPhysicalCompositing(host, {
    opacity: clampAppearanceFraction(config.opacity, 1.0),
    vertexAlpha: config.vertexAlpha === true,
    alphaCutoff: config.alphaCutoff,
  });
}

// ---------------------------------------------------------------------------
// The `LuxarMaterial` surface, as functions both wrappers delegate to.
// ---------------------------------------------------------------------------

/** Luxar `opacity`: re-derives the whole compositing decision (translucency is data). */
export function physicalUpdateOpacity(host: PhysicalMeshHost, opacity: number): void {
  const inputs = readInputs(host);
  applyPhysicalCompositing(host, { ...inputs, opacity: clampAppearanceFraction(opacity, 1.0) });
}

/** The node opacity as last written — the LOD cross-fade's fade base. */
export function physicalGetOpacity(host: PhysicalMeshHost): number {
  return readInputs(host).opacity;
}

/**
 * Tell the material whether its geometry's colours carry alpha.
 *
 * Known only at the first COMMIT: `MeshMetadata` says `has_colors`, not how many
 * components, and the placeholder node is built before any data is fetched. The
 * commit path calls this with the decoded `colorComponents`; a no-op when nothing
 * changed, so every later commit is free.
 */
export function physicalSetVertexAlpha(host: PhysicalMeshHost, vertexAlpha: boolean): void {
  const inputs = readInputs(host);
  if (inputs.vertexAlpha === vertexAlpha) return;
  applyPhysicalCompositing(host, { ...inputs, vertexAlpha });
}

/**
 * Luxar `intensity` → a scalar `color`. With `vertexColors` on, three's base colour is
 * `color × vertexColor`, so a scalar `color` is exactly a gain on the authored colour —
 * the same thing `uIntensity` is on the house shader.
 */
export function physicalUpdateIntensity(host: PhysicalMeshHost, intensity: number): void {
  host.color.setScalar(Number.isFinite(intensity) ? Math.max(0, intensity) : 1.0);
}

/**
 * Luxar `offset` → a scalar `emissive`. An additive brightness shift IS an emitted
 * radiance, so the mapping is faithful for `offset >= 0`. A NEGATIVE offset (black-level
 * subtraction) has no physical counterpart — a surface cannot emit less than nothing —
 * and clamps to 0.
 */
export function physicalUpdateOffset(host: PhysicalMeshHost, offset: number): void {
  host.emissive.setScalar(Number.isFinite(offset) ? Math.max(0, offset) : 0.0);
}

/**
 * Luxar `gamma`: RECORDED, not applied.
 *
 * Gamma on the house shader warps the colour before the colour GOG; a physically based
 * material has no such term (its radiance is what the lighting model says it is), and
 * the Layers panel hides the slider for a physical layer rather than offer a knob that
 * does nothing. Kept as a method so `isLuxarMaterial` recognises the material and the
 * generic opacity/exposure paths keep working, and the value is kept on `userData` so
 * a round-trip through the panel's state does not lose what the author wrote.
 */
export function physicalUpdateGamma(host: PhysicalMeshHost, gamma: number): void {
  host.userData.gamma = gamma;
}

/**
 * Whether a material is one of the two physical wrappers, by the stamp the config
 * writes. Structural on purpose: the TSL twin lives behind the lazy `three/webgpu`
 * boundary, so an `instanceof` here would drag that chunk onto the eager path.
 */
export function isPhysicalMeshMaterial(material: unknown): boolean {
  return (
    typeof material === 'object' &&
    material !== null &&
    (material as { userData?: { material?: unknown } }).userData?.material === 'physical'
  );
}
