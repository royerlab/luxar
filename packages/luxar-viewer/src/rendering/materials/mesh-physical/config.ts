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
 * The knobs themselves are ONE table, {@link PHYSICAL_MESH_KNOBS}: the attr name, the
 * three property it lands on, its domain and default, how a slider should present it,
 * and whether crossing zero selects a shader variant. The construction mapping, the
 * live Layers-panel sliders and the unit tests all read that table, so a knob added
 * there exists everywhere at once.
 *
 * Nothing here imports `three/webgpu`: the TSL twin lives in the lazy cone and reaches
 * these helpers, never the other way round.
 *
 * @module rendering/materials/mesh-physical/config
 */

import type * as THREE from 'three';
import type { BlendingMode } from '../../../types/blending';
import { log, Modules } from '../../../utils/log';
import { clampAppearanceFraction } from '../mesh/appearance';

/** The three property a knob writes — the same name on both three classes. */
export type PhysicalKnobProp =
  | 'roughness'
  | 'metalness'
  | 'clearcoat'
  | 'clearcoatRoughness'
  | 'iridescence'
  | 'sheen'
  | 'transmission'
  | 'ior'
  | 'thickness'
  | 'attenuationDistance'
  | 'dispersion';

/** Everything the viewer knows about one numeric physical knob. */
export interface PhysicalKnobSpec {
  /** Three property name (camelCase); the attr name is the table key (snake_case). */
  readonly prop: PhysicalKnobProp;
  /** Layers-panel label. */
  readonly label: string;
  /** Smallest value the material accepts; lower values clamp here. */
  readonly min: number;
  /** Largest value the material accepts (`Infinity` = unbounded); higher values clamp. */
  readonly max: number;
  /** What an ABSENT knob renders as — three's own default unless documented otherwise. */
  readonly default: number;
  /** Slider step (ignored on a log track). */
  readonly step: number;
  /**
   * Top of the slider track when the domain is unbounded. With
   * {@link PhysicalKnobSpec.maxIsInfinite} the top stop MEANS "no limit"
   * (`Infinity`), which is how `attenuation_distance`'s default is reachable from a
   * finite slider.
   */
  readonly sliderMax?: number;
  /** The slider's top stop maps to `Infinity` (see `sliderMax`). */
  readonly maxIsInfinite?: boolean;
  /** Present on a geometric track (a length whose useful range spans decades). */
  readonly logScale?: boolean;
  /**
   * Whether `0 → >0` (and back) selects a shader variant. Three's WebGL setters
   * bump the material version on that crossing already; `MeshPhysicalNodeMaterial`
   * stores plain properties and bakes the flags into its lighting model at setup,
   * so {@link setPhysicalKnob} flags `needsUpdate` itself — one rule for both twins.
   */
  readonly programAffecting: boolean;
}

/**
 * The knob table, in Layers-panel order: the Phase 1 surface knobs, then the Phase 2
 * glass family (spec §3.4). Keys are the ATTR names an author typed — `createMeshNode`
 * reads them off the node attrs and the panel lists them by the same names.
 *
 * Defaults are three's, with one exception documented on {@link PHYSICAL_MESH_DEFAULTS}
 * (the sheen colour). `attenuation_distance` defaults to `Infinity` = no volume
 * attenuation, exactly three's; the slider reaches it through its top stop.
 */
export const PHYSICAL_MESH_KNOBS = {
  roughness: {
    prop: 'roughness',
    label: 'Roughness',
    min: 0,
    max: 1,
    default: 1,
    step: 0.01,
    programAffecting: false,
  },
  metalness: {
    prop: 'metalness',
    label: 'Metalness',
    min: 0,
    max: 1,
    default: 0,
    step: 0.01,
    programAffecting: false,
  },
  clearcoat: {
    prop: 'clearcoat',
    label: 'Clearcoat',
    min: 0,
    max: 1,
    default: 0,
    step: 0.01,
    programAffecting: true,
  },
  clearcoat_roughness: {
    prop: 'clearcoatRoughness',
    label: 'Clearcoat roughness',
    min: 0,
    max: 1,
    default: 0,
    step: 0.01,
    programAffecting: false,
  },
  iridescence: {
    prop: 'iridescence',
    label: 'Iridescence',
    min: 0,
    max: 1,
    default: 0,
    step: 0.01,
    programAffecting: true,
  },
  sheen: {
    prop: 'sheen',
    label: 'Sheen',
    min: 0,
    max: 1,
    default: 0,
    step: 0.01,
    programAffecting: true,
  },
  transmission: {
    prop: 'transmission',
    label: 'Transmission',
    min: 0,
    max: 1,
    default: 0,
    step: 0.01,
    programAffecting: true,
  },
  ior: {
    prop: 'ior',
    label: 'IOR',
    min: 1,
    max: 2.333,
    default: 1.5,
    step: 0.001,
    programAffecting: false,
  },
  thickness: {
    prop: 'thickness',
    label: 'Thickness',
    min: 0,
    max: Number.POSITIVE_INFINITY,
    default: 0,
    step: 0.01,
    sliderMax: 10,
    programAffecting: false,
  },
  attenuation_distance: {
    prop: 'attenuationDistance',
    label: 'Attenuation distance',
    min: 1e-3,
    max: Number.POSITIVE_INFINITY,
    default: Number.POSITIVE_INFINITY,
    step: 0.01,
    sliderMax: 100,
    maxIsInfinite: true,
    logScale: true,
    programAffecting: false,
  },
  dispersion: {
    prop: 'dispersion',
    label: 'Dispersion',
    min: 0,
    max: Number.POSITIVE_INFINITY,
    default: 0,
    step: 0.01,
    sliderMax: 1,
    programAffecting: true,
  },
} as const satisfies Record<string, PhysicalKnobSpec>;

/** An attr name from the knob table. */
export type PhysicalMeshKnobKey = keyof typeof PHYSICAL_MESH_KNOBS;

/** The knob keys in panel order (object key order is insertion order). */
export const PHYSICAL_MESH_KNOB_KEYS = Object.keys(
  PHYSICAL_MESH_KNOBS
) as readonly PhysicalMeshKnobKey[];

/**
 * What an ABSENT knob renders as, keyed by three PROPERTY name (the construction
 * config's spelling). Three's own defaults, with one deliberate exception:
 * `sheenColor` is white where three's is black, because a black sheen is a no-op and
 * an authored `sheen=1` with no colour would then render nothing — the silent failure
 * the whole material contract exists to refuse. The Python writer never stamps these,
 * so an absent attr on disk IS the default.
 */
export const PHYSICAL_MESH_DEFAULTS = {
  roughness: PHYSICAL_MESH_KNOBS.roughness.default,
  metalness: PHYSICAL_MESH_KNOBS.metalness.default,
  clearcoat: PHYSICAL_MESH_KNOBS.clearcoat.default,
  clearcoatRoughness: PHYSICAL_MESH_KNOBS.clearcoat_roughness.default,
  iridescence: PHYSICAL_MESH_KNOBS.iridescence.default,
  sheen: PHYSICAL_MESH_KNOBS.sheen.default,
  sheenColor: '#ffffff',
  transmission: PHYSICAL_MESH_KNOBS.transmission.default,
  ior: PHYSICAL_MESH_KNOBS.ior.default,
  thickness: PHYSICAL_MESH_KNOBS.thickness.default,
  attenuationDistance: PHYSICAL_MESH_KNOBS.attenuation_distance.default,
  /** Three's own default; white = no tint, so an absent colour changes nothing. */
  attenuationColor: '#ffffff',
  dispersion: PHYSICAL_MESH_KNOBS.dispersion.default,
} as const;

/** The two `#rrggbb` colour knobs, by three property name. */
export type PhysicalColorProp = 'sheenColor' | 'attenuationColor';

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
  /** Share of light transmitted through the surface (glass); `> 0` is translucent. */
  transmission?: number;
  /** Index of refraction in `[1, 2.333]`. */
  ior?: number;
  /** Volume thickness for the refraction, in scene units. */
  thickness?: number;
  /** Beer–Lambert distance at which `attenuationColor` is reached; `Infinity` = none. */
  attenuationDistance?: number;
  /** `#rrggbb`; default {@link PHYSICAL_MESH_DEFAULTS}.attenuationColor. */
  attenuationColor?: string;
  /** Chromatic dispersion strength (Abbe-number-like), `>= 0`. */
  dispersion?: number;
  /**
   * Luxar `refract_data` (spec §3.4 Phase 3): draw the glass AFTER the emissive data
   * so it refracts what is behind it. Default false. See {@link derivePhysicalCompositing}.
   */
  refractData?: boolean;
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
export interface PhysicalMeshHost extends Record<PhysicalKnobProp, number> {
  opacity: number;
  transparent: boolean;
  depthWrite: boolean;
  alphaTest: number;
  color: THREE.Color;
  emissive: THREE.Color;
  sheenColor: THREE.Color;
  attenuationColor: THREE.Color;
  vertexColors: boolean;
  flatShading: boolean;
  toneMapped: boolean;
  needsUpdate: boolean;
  userData: Record<string, unknown>;
}

/** The five inputs the compositing decision is a pure function of. */
export interface PhysicalCompositingInputs {
  opacity: number;
  vertexAlpha: boolean;
  alphaCutoff: number | undefined;
  /** The material's `transmission`; `> 0` is glass and composites as translucent. */
  transmission: number;
  /**
   * Luxar `refract_data`: the glass draws AFTER the emissive data instead of before
   * it (spec §3.4 Phase 3). Meaningless without `transmission > 0`, and inert then.
   */
  refractData: boolean;
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
  /**
   * Whether the mesh must be drawn BEFORE the emissive data around it. True for
   * glass: on WebGPU a transmissive mesh shares the transparent render list with
   * every point, line and splat layer, and a glass fragment lands with alpha 1, so
   * a cluster whose centre sorts farther than the sphere would be drawn first and
   * then painted over. The depth-sort coordinator reads this off
   * `userData.drawBeforeEmissive` and orders the mesh first within its
   * `layer_order` band on both backends (on WebGL it is already in the earlier
   * transmissive list, so the rule is harmless there). Exactly one of this and
   * {@link PhysicalCompositing.drawAfterEmissive} is set for glass; neither for
   * anything else.
   */
  drawBeforeEmissive: boolean;
  /**
   * The Phase 3 inverse (spec §3.4): glass authored with `refract_data` must draw
   * AFTER the emissive data so that what it samples already holds the data. The
   * coordinator reads `userData.drawAfterEmissive` and ranks such a mesh LAST within
   * its `layer_order` band (that is the whole mechanism on WebGPU); on WebGL the
   * post-processing pipeline additionally splits the scene pass so three's
   * transmission target sees the data.
   */
  drawAfterEmissive: boolean;
}

/**
 * Decide how a physical mesh composites, from its inputs.
 *
 * A physical mesh has no Luxar blending mode — that knob is refused at authoring
 * — so translucency is read off the data instead:
 *
 * - `opacity < 1` is translucent, always.
 * - `transmission > 0` (glass) is translucent, always. Glass therefore does NOT
 *   write depth, and that is the spec §3.4 contract made concrete: emissive data
 *   behind a glass surface stays visible — crisp and unrefracted — where a
 *   depth-writing shell would HIDE the cluster inside it, the worse failure for the
 *   marker case this family exists for. With `refractData` the glass instead draws
 *   AFTER the data and refracts it (Phase 3); the compositing state is the same,
 *   only the ordering stamp flips.
 * - Per-vertex alpha is translucent UNLESS an `alpha_cutoff` was authored, in
 *   which case the alpha is a CUTOUT (three `alphaTest`) and the surface stays
 *   opaque — the same meaning the house shader's `opaque` mode gives the pair.
 *
 * Translucent surfaces do not write depth. With no per-triangle sort (spec §3.2)
 * that is the one choice that never drops a back face behind a front one; the
 * remaining order dependence is the ordinary alpha-over kind and is invisible on
 * a shell of uniform alpha.
 */
export function derivePhysicalCompositing(inputs: PhysicalCompositingInputs): PhysicalCompositing {
  const cutout = inputs.vertexAlpha && inputs.alphaCutoff !== undefined;
  const glass = inputs.transmission > 0;
  const translucent = inputs.opacity < 1 || glass || (inputs.vertexAlpha && !cutout);
  return {
    transparent: translucent,
    depthWrite: !translucent,
    alphaTest: translucent ? 0 : clampAppearanceFraction(inputs.alphaCutoff, 0),
    blendingMode: translucent ? undefined : 'opaque',
    drawBeforeEmissive: glass && !inputs.refractData,
    drawAfterEmissive: glass && inputs.refractData,
  };
}

/**
 * Where the compositing inputs live on the material, so `updateOpacity` can re-derive
 * the decision without being told the others again.
 */
const INPUTS_KEY = 'physicalCompositing';

function readInputs(host: PhysicalMeshHost): PhysicalCompositingInputs {
  const stored = host.userData[INPUTS_KEY] as PhysicalCompositingInputs | undefined;
  return (
    stored ?? {
      opacity: host.opacity,
      vertexAlpha: false,
      alphaCutoff: undefined,
      transmission: host.transmission,
      refractData: false,
    }
  );
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
  if (decision.drawBeforeEmissive) host.userData.drawBeforeEmissive = true;
  else delete host.userData.drawBeforeEmissive;
  if (decision.drawAfterEmissive) host.userData.drawAfterEmissive = true;
  else delete host.userData.drawAfterEmissive;
  if (programChanged) host.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// The transmitted-alpha pin (spec §3.4, Phase 3).
// ---------------------------------------------------------------------------

/**
 * The one line of three's `transmission_fragment` chunk that copies the alpha of the
 * SAMPLED framebuffer into the glass fragment's alpha (`opaque_fragment` then multiplies
 * `diffuseColor.a` by it). Three means it as coverage, for transparent canvases where a
 * glass over an empty background should itself be see-through.
 *
 * Luxar's HDR framebuffer alpha is not coverage. The additive, luminous and normal
 * point and line shaders are alpha-weighted (RGB unweighted, alpha = intensity·opacity,
 * blended SrcAlpha/One with One/One on the alpha channel), so behind a point cloud the
 * alpha is an overdraw count — 8 to 18 was measured, `Infinity` on dense scenes. The
 * house already treats it as undefined: the mega shader reads RGB only, and the EXR
 * capture forces alpha to 1 before export. A glass that reads it composites with an
 * UNCLAMPED blend factor on the half-float target and comes out thousands of times too
 * bright, or negative (measured on both backends). So the physical family refuses that
 * channel too: the line below is replaced by {@link TRANSMISSION_ALPHA_PIN_LINE}.
 *
 * A pixel no-op for the glass-first path, whose transmission target only ever holds the
 * alpha-1 clear and opaque meshes. It changes what an `opaque`-mode house layer
 * (alpha = intensity·opacity) or a cutout mesh looks like THROUGH glass — they follow the
 * house rule now — and it is what makes `refract_data` glass composite correctly.
 */
export const TRANSMISSION_ALPHA_MIX_LINE =
  'material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );';

/** What replaces {@link TRANSMISSION_ALPHA_MIX_LINE}: the glass keeps its own alpha. */
export const TRANSMISSION_ALPHA_PIN_LINE = 'material.transmissionAlpha = 1.0;';

/** The include three resolves AFTER `onBeforeCompile`, so the hook must expand it itself. */
const TRANSMISSION_INCLUDE = '#include <transmission_fragment>';

/** Fixed program-cache key for the WebGL wrapper (three's default is `onBeforeCompile.toString()`). */
export const PHYSICAL_PROGRAM_CACHE_KEY = 'luxar-physical:transmitted-alpha-pinned';

let warnedChunkDrift = false;

/**
 * Patch a WebGL fragment shader so its transmission pass ignores the sampled alpha.
 *
 * Pure over its inputs so the substitution is unit-testable against three's real chunk:
 * `#include <transmission_fragment>` (still unresolved when `onBeforeCompile` runs) is
 * replaced by the chunk body with {@link TRANSMISSION_ALPHA_MIX_LINE} pinned. A shader
 * without the include (no transmission) is returned untouched. If a three upgrade moves
 * the line, the shader is left to three's own include resolution and one warning names
 * the consequence; the unit test on the real chunk turns that drift into a red build.
 *
 * @param fragmentShader - `shader.fragmentShader` as handed to `onBeforeCompile`
 * @param transmissionChunk - `THREE.ShaderChunk.transmission_fragment` (passed in so this
 *   module keeps its type-only `three` import)
 */
export function pinTransmittedAlphaGlsl(fragmentShader: string, transmissionChunk: string): string {
  if (!fragmentShader.includes(TRANSMISSION_INCLUDE)) return fragmentShader;
  if (!transmissionChunk.includes(TRANSMISSION_ALPHA_MIX_LINE)) {
    if (!warnedChunkDrift) {
      warnedChunkDrift = true;
      log.warning(
        Modules.RENDERER,
        "three's transmission_fragment chunk no longer contains the transmissionAlpha mix " +
          'line; the physical glass will read the framebuffer alpha again, and refract_data ' +
          'glass will composite wrongly over emissive layers. Update TRANSMISSION_ALPHA_MIX_LINE.'
      );
    }
    return fragmentShader;
  }
  const pinned = transmissionChunk.replace(
    TRANSMISSION_ALPHA_MIX_LINE,
    TRANSMISSION_ALPHA_PIN_LINE
  );
  return fragmentShader.replace(TRANSMISSION_INCLUDE, pinned);
}

/**
 * Luxar `refract_data`, live (the Layers-panel toggle): re-derives the ordering stamp.
 * Plain state on both backends — no program rebuild — so a toggle costs one frame.
 */
export function physicalUpdateRefractData(host: PhysicalMeshHost, refractData: boolean): void {
  applyPhysicalCompositing(host, { ...readInputs(host), refractData: refractData === true });
}

/** The material's current `refract_data` decision input. */
export function physicalGetRefractData(host: PhysicalMeshHost): boolean {
  return readInputs(host).refractData;
}

/**
 * Sanitize a knob value against its spec: a non-number or NaN resolves to the
 * documented DEFAULT (the sibling sanitizer policy — a corrupt value should not
 * land on a boundary that looks chosen), `+Infinity` is kept only where the spec
 * says the top is infinite, and everything else clamps into `[min, max]`.
 */
export function clampPhysicalKnob(key: PhysicalMeshKnobKey, value: unknown): number {
  const spec: PhysicalKnobSpec = PHYSICAL_MESH_KNOBS[key];
  if (typeof value !== 'number' || Number.isNaN(value)) return spec.default;
  if (!Number.isFinite(value)) {
    return value > 0 && spec.maxIsInfinite ? Number.POSITIVE_INFINITY : spec.default;
  }
  return Math.min(spec.max, Math.max(spec.min, value));
}

/**
 * Value → slider position domain: `Infinity` (and anything past the track) lands on
 * the top stop. What the Layers panel STORES for a knob, so its state stays
 * JSON-safe (a literal `Infinity` serialises as `null`).
 */
export function physicalKnobToSlider(key: PhysicalMeshKnobKey, value: number): number {
  const spec: PhysicalKnobSpec = PHYSICAL_MESH_KNOBS[key];
  const top = spec.sliderMax ?? spec.max;
  return Math.min(top, Math.max(spec.min, Number.isFinite(value) ? value : top));
}

/** Slider position → material value: the top stop means `Infinity` where the spec says so. */
export function physicalKnobFromSlider(key: PhysicalMeshKnobKey, value: number): number {
  const spec: PhysicalKnobSpec = PHYSICAL_MESH_KNOBS[key];
  const top = spec.sliderMax ?? spec.max;
  if (spec.maxIsInfinite && value >= top) return Number.POSITIVE_INFINITY;
  return clampPhysicalKnob(key, value);
}

/**
 * Write ONE knob onto the material — the shared path for construction AND the live
 * Layers-panel sliders. Clamps by the table, assigns the three property, and
 * requests a program rebuild when a `programAffecting` knob crosses zero (see
 * {@link PhysicalKnobSpec.programAffecting} for why both twins need this here).
 * `transmission` additionally re-derives the compositing decision, since glass is
 * translucent by definition. Returns the value actually written.
 */
export function setPhysicalKnob(
  host: PhysicalMeshHost,
  key: PhysicalMeshKnobKey,
  value: unknown
): number {
  const spec: PhysicalKnobSpec = PHYSICAL_MESH_KNOBS[key];
  const next = clampPhysicalKnob(key, value);
  const prev = host[spec.prop];
  if (prev === next) return next;
  host[spec.prop] = next;
  if (spec.programAffecting && prev > 0 !== next > 0) host.needsUpdate = true;
  if (key === 'transmission') {
    applyPhysicalCompositing(host, { ...readInputs(host), transmission: next });
  }
  return next;
}

/**
 * Write one of the two `#rrggbb` colour knobs. `Color.set('#rrggbb')` reads the hex
 * as sRGB and converts to the working (linear) space under three's colour
 * management — the human reading of a hex tint, and what the Python validators
 * promise. An absent or non-string value is the documented default.
 */
export function setPhysicalColor(
  host: PhysicalMeshHost,
  prop: PhysicalColorProp,
  hex: unknown
): void {
  host[prop].set(typeof hex === 'string' ? hex : PHYSICAL_MESH_DEFAULTS[prop]);
}

/**
 * Configure a freshly constructed physical material from Luxar attrs.
 *
 * Every numeric knob goes through {@link setPhysicalKnob}, so construction and the
 * live sliders cannot disagree about clamping or rebuild rules.
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

  for (const key of PHYSICAL_MESH_KNOB_KEYS) {
    const spec: PhysicalKnobSpec = PHYSICAL_MESH_KNOBS[key];
    // A knob three constructs at a non-default value (none today, but the table is
    // the contract) still lands on OUR default when the attr is absent.
    const requested = config[spec.prop];
    host[spec.prop] = requested === undefined ? spec.default : clampPhysicalKnob(key, requested);
  }
  setPhysicalColor(host, 'sheenColor', config.sheenColor);
  setPhysicalColor(host, 'attenuationColor', config.attenuationColor);

  physicalUpdateIntensity(host, config.intensity ?? 1.0);
  physicalUpdateOffset(host, config.offset ?? 0.0);
  physicalUpdateGamma(host, config.gamma ?? 1.0);

  host.userData.material = 'physical';
  applyPhysicalCompositing(host, {
    opacity: clampAppearanceFraction(config.opacity, 1.0),
    vertexAlpha: config.vertexAlpha === true,
    alphaCutoff: config.alphaCutoff,
    transmission: host.transmission,
    refractData: config.refractData === true,
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
 * Why a knob currently changes NOTHING on screen, or `null` when it is live — the
 * dependencies three's physical shader imposes between knobs, stated once so the
 * Layers panel can grey a slider out with the reason instead of offering a dead
 * control (measured on the reflections demo: at `metalness = 1` the whole glass
 * family is inert because three scales transmission by `1 − metalness` and takes a
 * metal's reflectance from its base colour, not its IOR).
 *
 * `values` are the LIVE knob values (material domain, not slider space) plus the
 * attenuation colour, all optional so a partial record still answers.
 */
export function physicalKnobInertReason(
  key: PhysicalMeshKnobKey,
  values: PhysicalKnobLiveValues
): string | null {
  const state = physicalKnobDependencyState(values);
  const needsTransmission = state.transmitting ? null : INERT_NEEDS_TRANSMISSION;
  const rules: Partial<Record<PhysicalMeshKnobKey, string | null>> = {
    clearcoat_roughness: state.coated ? null : 'No effect until Clearcoat is above 0.',
    transmission: state.metal
      ? 'A metal transmits nothing: three scales transmission by 1 − Metalness.'
      : null,
    ior: state.metal
      ? 'No effect on a metal: its reflectance comes from the base colour, not the IOR.'
      : null,
    thickness: needsTransmission,
    dispersion: needsTransmission,
    attenuation_distance: needsTransmission ?? (state.whiteAttenuation ? INERT_WHITE : null),
  };
  return rules[key] ?? null;
}

/** The live knob values (material domain) plus the attenuation colour, all optional. */
export type PhysicalKnobLiveValues = Partial<Record<PhysicalMeshKnobKey, number>> & {
  attenuation_color?: string;
};

const INERT_NEEDS_TRANSMISSION = 'No effect until Transmission is above 0 (and Metalness below 1).';
const INERT_WHITE =
  'No effect while the attenuation colour is white: author attenuation_color to tint the glass.';

/** The four facts the inert rules are stated in terms of. */
function physicalKnobDependencyState(values: PhysicalKnobLiveValues): {
  metal: boolean;
  coated: boolean;
  transmitting: boolean;
  whiteAttenuation: boolean;
} {
  const metal = (values.metalness ?? PHYSICAL_MESH_DEFAULTS.metalness) >= 1;
  const colour = values.attenuation_color ?? PHYSICAL_MESH_DEFAULTS.attenuationColor;
  return {
    metal,
    coated: (values.clearcoat ?? PHYSICAL_MESH_DEFAULTS.clearcoat) > 0,
    transmitting: (values.transmission ?? PHYSICAL_MESH_DEFAULTS.transmission) > 0 && !metal,
    whiteAttenuation: colour.toLowerCase() === '#ffffff',
  };
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
