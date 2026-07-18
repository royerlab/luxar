/**
 * Shared blending-state helper used by all node-type materials.
 *
 * Centralises the per-mode mapping from the high-level Luxar blending
 * mode (`additive` | `normal` | `max` | `opaque` | `luminous`) to the
 * complete THREE.js material state. This is the single source of truth
 * for both creation-time wiring (in MaterialManager) and runtime
 * transitions (in {Point,Line,GSplat}Material.applyBlendingMode and
 * LayersPanel's runtime path).
 *
 * Why "complete": runtime layer updates must set the same low-level
 * blend factors as material creation. Updating only the high-level
 * mode or blend equation can leave stale `blendSrc`/`blendDst` values
 * on the material, so this helper returns the full state to apply.
 *
 * `shaderOutputMode` is a hint for materials whose fragment shader
 * needs to know how to compose RGB. For example, Points/Lines in `max`
 * mode want `rgb-contribution` so the fragment emits
 * `finalColor * intensity * opacity` (premultiplied), which the
 * MaxEquation + OneFactor / OneFactor framebuffer state turns into a
 * correct max-of-contribution. In additive/normal mode the shader
 * uses `alpha-weighted` output (RGB unweighted, alpha =
 * intensity*opacity). GSplats already premultiply intensity into RGB
 * unconditionally so they can ignore this hint.
 */

import * as THREE from 'three';
import type { BlendingMode } from './material-manager';
import { log, Modules } from '../utils/log';

/**
 * Discriminator predicates over `BlendingMode`. Centralising the
 * `mode === 'foo'` literal comparisons here keeps the modes string-typed
 * (zero runtime cost) while making intent explicit at call sites and
 * giving us one place to change if the mode enum ever gets reshaped.
 *
 * @param mode - the blending mode to test
 * @returns `true` when `mode` matches the predicate's mode
 * @public
 */
export function isAdditiveMode(mode: BlendingMode): boolean {
  return mode === 'additive';
}
/**
 * @param mode - the blending mode to test
 * @returns `true` when `mode === 'opaque'`
 * @public
 */
export function isOpaqueMode(mode: BlendingMode): boolean {
  return mode === 'opaque';
}
/**
 * @param mode - the blending mode to test
 * @returns `true` when `mode === 'max'`
 * @public
 */
export function isMaxMode(mode: BlendingMode): boolean {
  return mode === 'max';
}
/**
 * @param mode - the blending mode to test
 * @returns `true` when `mode === 'luminous'`
 * @public
 */
export function isLuminousMode(mode: BlendingMode): boolean {
  return mode === 'luminous';
}
/**
 * @param mode - the blending mode to test
 * @returns `true` when `mode === 'normal'`
 * @public
 */
export function isNormalMode(mode: BlendingMode): boolean {
  return mode === 'normal';
}

/**
 * Projection-model taxonomy over the blending modes (PR #561): SURFACE
 * modes composite the projected 2D-Gaussian PEAK (surface density at
 * the ray hit) — `max` compares peak contributions, `normal` and
 * `opaque` alpha-over a surface — while EMISSIVE modes
 * (`additive`/`luminous`) integrate the ray through the 3D Gaussian
 * (the ~2.4×·sigmaRay line-integral boost). GSplat materials key the
 * sum↔peak projection split off this single predicate so the two
 * backends and the rebuild-boundary logic can never disagree on the
 * grouping.
 *
 * @param mode - the blending mode to test
 * @returns `true` when `mode` projects the 2D-Gaussian peak
 * @public
 */
export function usesPeakProjection(mode: BlendingMode): boolean {
  return isMaxMode(mode) || isNormalMode(mode) || isOpaqueMode(mode);
}

/**
 * The five canonical Luxar blending modes. Runtime source of truth for
 * validating raw `blending_mode` strings (zarr attrs, URL params)
 * before they reach the string-typed `BlendingMode` world.
 */
export const BLENDING_MODES = ['additive', 'normal', 'max', 'opaque', 'luminous'] as const;

/** Unknown mode strings already warned about — one warning per distinct value. */
const warnedUnknownModes = new Set<string>();

/**
 * Normalize a raw `blending_mode` string to a canonical
 * {@link BlendingMode}.
 *
 * - `undefined` → `'additive'` (the composition identity —
 *   `composeAttrs`' default when no ancestor sets a mode).
 * - A member of {@link BLENDING_MODES} → passed through unchanged.
 * - Anything else → `'normal'`, warning once per distinct string.
 *   `'normal'` matches `getCompleteBlendingState`'s fallthrough AND
 *   keeps `isNormalMode()` true, so unknown-mode gsplats still get
 *   depth-sorted instead of rendering alpha-over unsorted.
 *
 * @param raw - the raw attribute value (possibly absent or malformed)
 * @returns The validated blending mode.
 * @public
 */
export function normalizeBlendingMode(raw: string | undefined): BlendingMode {
  if (raw === undefined) return 'additive';
  if ((BLENDING_MODES as readonly string[]).includes(raw)) return raw as BlendingMode;
  if (!warnedUnknownModes.has(raw)) {
    warnedUnknownModes.add(raw);
    log.warning(Modules.RENDERER, `Unknown blending_mode "${raw}" — falling back to 'normal'.`);
  }
  return 'normal';
}

/**
 * The generic `normal`-mode depthWrite predicate: a (near-)fully-opaque
 * normal layer writes depth so it occludes additive layers behind it.
 * Single source of truth — the material caches key on this SAME
 * predicate so two opacities on either side of the threshold can never
 * share a cached material (1%-opacity buckets straddle 0.99: 0.985 and
 * 0.994 both bucket to 99 but need different depthWrite).
 */
export function normalModeDepthWrite(opacity: number): boolean {
  return opacity >= 0.99;
}

export interface CompleteBlendingState {
  blending: THREE.Blending;
  blendEquation: THREE.BlendingEquation;
  blendSrc: THREE.BlendingSrcFactor;
  blendDst: THREE.BlendingDstFactor;
  depthTest: boolean;
  depthWrite: boolean;
  transparent: boolean;
  /**
   * Hint for shader RGB output composition. Only `'rgb-contribution'`
   * is machine-consumed: the Point and Line `applyBlendingMode`
   * wrappers toggle the `LUXAR_MAX_RGB_CONTRIBUTION` define off it so
   * `max` mode premultiplies RGB by intensity·opacity (required
   * because MaxEquation + OneFactor compares contributions, not flat
   * colour). The other three values document the fragment-output
   * contract for readers but are consumed by no wrapper:
   * `'alpha-weighted'` (additive/luminous/normal — RGB unweighted,
   * alpha = intensity·opacity), `'opaque'`, and `'premultiplied-alpha'`
   * (gsplat `normal` — RGB carries the full premultiplied contribution,
   * alpha a clamped coverage term for `OneMinusSrcAlpha` destination
   * attenuation; the gsplat wrappers key that branch off
   * `isNormalMode`, not off this field).
   */
  shaderOutputMode: 'alpha-weighted' | 'rgb-contribution' | 'opaque' | 'premultiplied-alpha';
}

/**
 * Compute the complete blending state for a Luxar blending mode.
 *
 * `opacity` only affects `depthWrite` for `normal` mode (a fully-opaque
 * normal layer should write depth so it occludes additive layers
 * behind it). All other modes are opacity-independent.
 *
 * @param mode - One of `'normal'`, `'additive'`, `'max'`, `'opaque'`,
 *   `'luminous'`. See the file-level header for per-mode semantics.
 * @param opacity - Layer opacity in `[0, 1]`. Currently only consulted
 *   for `mode === 'normal'`, where `opacity >= 0.99` flips on
 *   `depthWrite`. Default `1.0`.
 * @returns The complete THREE.js material state to apply (blending,
 *   blend factors, depth, transparency, and a `shaderOutputMode` hint
 *   for fragment shaders that need to know how to compose RGB).
 * @public
 */
export function getCompleteBlendingState(
  mode: BlendingMode,
  opacity: number = 1.0
): CompleteBlendingState {
  if (mode === 'max') {
    // Custom blending with MaxEquation + OneFactor expects RGB to
    // already include falloff/intensity (premultiplied) so the per-pixel
    // max captures contribution, not flat colour.
    return {
      blending: THREE.CustomBlending,
      blendEquation: THREE.MaxEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      depthTest: true,
      depthWrite: false,
      transparent: true,
      shaderOutputMode: 'rgb-contribution',
    };
  }

  if (mode === 'additive') {
    // AdditiveBlending = SrcAlpha + One. Shader emits RGB unweighted,
    // alpha = intensity*opacity, so the framebuffer multiplies by
    // intensity*opacity at composite time.
    return {
      blending: THREE.AdditiveBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.SrcAlphaFactor,
      blendDst: THREE.OneFactor,
      depthTest: false, // Additive ignores depth — renders on top
      depthWrite: false,
      transparent: true,
      shaderOutputMode: 'alpha-weighted',
    };
  }

  if (mode === 'luminous') {
    // Luminous = additive but with depth test (so far primitives
    // occlude near ones).
    return {
      blending: THREE.AdditiveBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.SrcAlphaFactor,
      blendDst: THREE.OneFactor,
      depthTest: true,
      depthWrite: false,
      transparent: true,
      shaderOutputMode: 'alpha-weighted',
    };
  }

  if (mode === 'opaque') {
    // Opaque: standard alpha-blended primitive that writes depth.
    return {
      blending: THREE.NormalBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.SrcAlphaFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      depthTest: true,
      depthWrite: true,
      transparent: false,
      shaderOutputMode: 'opaque',
    };
  }

  // 'normal' (default) — opacity-aware depthWrite so a fully opaque
  // normal layer occludes additive layers behind it.
  return {
    blending: THREE.NormalBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.SrcAlphaFactor,
    blendDst: THREE.OneMinusSrcAlphaFactor,
    depthTest: true,
    depthWrite: normalModeDepthWrite(opacity),
    transparent: true,
    shaderOutputMode: 'alpha-weighted',
  };
}

/**
 * GSplat-specific `normal`-mode blending state: premultiplied alpha-over.
 *
 * The gsplat fragment shader premultiplies intensity into RGB and (only
 * in this mode) emits a clamped coverage alpha, so the source factor is
 * `One`, not `SrcAlpha` (which would multiply the contribution by
 * coverage a second time). This replaces the generic `normal` entry for
 * gsplats, whose alpha-always-1 output made `normal` degenerate
 * ("opaque dimmed by opacity") — see GSPLAT_DEPTH_SORTING_SPEC.md §3.
 *
 * Deliberate choices (each load-bearing):
 * - `CustomBlending` with SYMMETRIC alpha channel (the material's
 *   `blendEquationAlpha`/`blendSrcAlpha`/`blendDstAlpha` stay null):
 *   separate alpha-channel blend state is exactly what trips a
 *   `gl.getError()` flag under WebGPURenderer's WebGL2 bridge (see
 *   materials/gsplat/material-tsl.ts). Alpha then composites as
 *   `a_src + (1 - a_src)·a_dst` — correct coverage accumulation.
 * - NEVER set `material.premultipliedAlpha` instead: on the TSL path
 *   `NodeMaterial.setup()` auto-injects an output RGB×alpha transform
 *   when that flag is set, double-premultiplying a shader that already
 *   premultiplies (and desyncing GLSL vs TSL).
 * - `depthWrite: false` unconditionally (unlike the generic `normal`
 *   entry's `opacity >= 0.99` flip): a coverage-alpha splat fragment
 *   with alpha as low as ~1e-4 survives the shader's discards, and
 *   letting it write depth punches occlusion halos across everything
 *   behind the splat's footprint. Sorted transparency never
 *   depth-writes. Trade-off: a gsplat `normal` layer does not occlude
 *   additive layers behind it.
 *
 * No `opacity` parameter — the state is opacity-independent by
 * construction (depthWrite never flips).
 *
 * @returns The complete THREE.js material state for gsplat `normal`.
 * @public
 */
export function getGSplatNormalBlendingState(): CompleteBlendingState {
  return {
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneMinusSrcAlphaFactor,
    depthTest: true,
    depthWrite: false,
    transparent: true,
    shaderOutputMode: 'premultiplied-alpha',
  };
}

/**
 * Apply a CompleteBlendingState to a THREE material in-place. Returns
 * `true` when any THREE.js side state actually changed (callers can
 * use this to decide whether to set `needsUpdate`).
 *
 * Does NOT update `userData.blendingMode` or shader defines —
 * material-specific `applyBlendingMode()` methods own those.
 *
 * @param material - The THREE material to mutate in place.
 * @param state - Output of {@link getCompleteBlendingState}.
 * @returns `true` if any of `blending` / `blendEquation` /
 *   `blendSrc` / `blendDst` / `depthTest` / `depthWrite` / `transparent`
 *   actually changed on the material. Use this to decide whether to set
 *   `material.needsUpdate = true`.
 * @public
 */
export function applyBlendingStateToMaterial(
  material: THREE.Material,
  state: CompleteBlendingState
): boolean {
  let changed = false;

  if (material.blending !== state.blending) {
    material.blending = state.blending;
    changed = true;
  }
  if (material.blendEquation !== state.blendEquation) {
    material.blendEquation = state.blendEquation;
    changed = true;
  }
  if (material.blendSrc !== state.blendSrc) {
    material.blendSrc = state.blendSrc;
    changed = true;
  }
  if (material.blendDst !== state.blendDst) {
    material.blendDst = state.blendDst;
    changed = true;
  }
  if (material.depthTest !== state.depthTest) {
    material.depthTest = state.depthTest;
    changed = true;
  }
  if (material.depthWrite !== state.depthWrite) {
    material.depthWrite = state.depthWrite;
    changed = true;
  }
  if (material.transparent !== state.transparent) {
    material.transparent = state.transparent;
    changed = true;
  }
  return changed;
}
