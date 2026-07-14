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

export interface CompleteBlendingState {
  blending: THREE.Blending;
  blendEquation: THREE.BlendingEquation;
  blendSrc: THREE.BlendingSrcFactor;
  blendDst: THREE.BlendingDstFactor;
  depthTest: boolean;
  depthWrite: boolean;
  transparent: boolean;
  /**
   * Hint for shader RGB output composition. `max` mode needs RGB to
   * already include the soft kernel contribution because the framebuffer
   * equation compares premultiplied contributions. `premultiplied-alpha`
   * (GSplat `normal` mode) means RGB carries the full premultiplied
   * contribution AND alpha carries a clamped coverage term for
   * `OneMinusSrcAlpha` destination attenuation.
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
    depthWrite: opacity >= 0.99,
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
