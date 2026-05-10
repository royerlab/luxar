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

export interface CompleteBlendingState {
  blending: THREE.Blending;
  blendEquation: THREE.BlendingEquation;
  blendSrc: THREE.BlendingSrcFactor;
  blendDst: THREE.BlendingDstFactor;
  /** Optional alpha-channel blending. Defaults to RGB equivalents. */
  blendEquationAlpha?: THREE.BlendingEquation;
  blendSrcAlpha?: THREE.BlendingSrcFactor;
  blendDstAlpha?: THREE.BlendingDstFactor;
  depthTest: boolean;
  depthWrite: boolean;
  transparent: boolean;
  /**
   * Hint for shader RGB output composition. `max` mode needs RGB to
   * already include the soft kernel contribution because the framebuffer
   * equation compares premultiplied contributions.
   */
  shaderOutputMode: 'alpha-weighted' | 'rgb-contribution' | 'opaque';
}

/**
 * Compute the complete blending state for a Luxar blending mode.
 *
 * `opacity` only affects `depthWrite` for `normal` mode (a fully-opaque
 * normal layer should write depth so it occludes additive layers
 * behind it). All other modes are opacity-independent.
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
 * Apply a CompleteBlendingState to a THREE material in-place. Returns
 * `true` when any THREE.js side state actually changed (callers can
 * use this to decide whether to set `needsUpdate`).
 *
 * Does NOT update `userData.blendingMode` or shader defines —
 * material-specific `applyBlendingMode()` methods own those.
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
