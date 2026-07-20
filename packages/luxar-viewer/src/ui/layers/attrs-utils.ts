/**
 * Pure helpers for the layers panel — kept here so the blending-mode
 * mapping and live-attrs derivation can be unit-tested without
 * instantiating a `LayersPanel` (which needs a real DOM container,
 * AnimationController, and zarr-backed scene graph).
 *
 * The DOM-facing `LayersPanel` calls these and applies the returned
 * shape to its THREE.Material instances.
 *
 * @module ui/layers/attrs-utils
 */

import * as THREE from 'three';
import { computeUniforms, type LayerInfo } from './layer-state';
import type { ComposableAttrs } from '../../data/attrs-composer';
import { clamp } from '../gui/format/value-formatting';
import {
  getCompleteBlendingState,
  normalizeBlendingMode,
  type CompleteBlendingState,
} from '../../rendering/blending-state';

/** Clamp gamma to a sensible UI range. Centralised to match material defaults. */
export function clampGamma(gamma: number): number {
  return clamp(gamma, 0.2, 5.0);
}

/**
 * Material-level blending state for a given high-level blending mode.
 *
 * The viewer exposes five named blending modes that map onto specific
 * THREE.js blending + depth + transparency combinations. Pulling the
 * mapping into a pure function makes the truth table explicit and
 * testable, and lets the caller decide how to apply it (e.g. set on a
 * material, or compose into a snapshot).
 *
 * `blendEquation` is part of the *total* state — every mode reports it
 * (even non-max modes report `THREE.AddEquation`, the default). That way
 * switching from 'max' back to 'additive' resets the equation instead of
 * stranding `MaxEquation` on the material from a previous selection.
 */
export interface BlendingState {
  blending: THREE.Blending;
  depthTest: boolean;
  depthWrite: boolean;
  transparent: boolean;
  blendEquation: THREE.BlendingEquation;
  /** blend factors needed for max-mode parity with creation-time state. */
  blendSrc?: THREE.BlendingSrcFactor;
  blendDst?: THREE.BlendingDstFactor;
}

/**
 * Map a blending-mode name to its concrete THREE.js material settings.
 *
 * Unknown modes coerce to `'normal'`-equivalent state via the shared
 * `normalizeBlendingMode` chokepoint so a malformed zarr attribute
 * can't crash the panel (and gets the same one-time warning every
 * other consumer of the raw string gets).
 *
 * this delegates to `getCompleteBlendingState` so the LayersPanel
 * generic fallback path agrees with material-side `applyBlendingMode`
 * implementations on `blendSrc`/`blendDst`. Without this, switching a
 * material to `max` via the UI used to leave `blendSrc`/`blendDst`
 * stale at SrcAlpha/OneMinusSrcAlpha — different from the creation-
 * time max state (`OneFactor`/`OneFactor`).
 */
export function getBlendingState(mode: string, opacity: number = 1.0): BlendingState {
  const safeMode = normalizeBlendingMode(mode);
  const complete: CompleteBlendingState = getCompleteBlendingState(safeMode, opacity);
  return {
    blending: complete.blending,
    blendEquation: complete.blendEquation,
    blendSrc: complete.blendSrc,
    blendDst: complete.blendDst,
    depthTest: complete.depthTest,
    depthWrite: complete.depthWrite,
    transparent: complete.transparent,
  };
}

/**
 * Derive the layer's current live composable attributes from its UI state.
 *
 * For layers whose user hasn't touched a control, these match the
 * authored zarr values — so composition stays a no-op for untouched
 * scenes. Pulled into a pure function so the data-flow from
 * (display range, gamma, opacity, blending) → ComposableAttrs is
 * test-isolated from the DOM panel.
 */
export function liveLayerAttrs(layer: LayerInfo): ComposableAttrs {
  const { intensity, offset } = computeUniforms(layer.displayMin, layer.displayMax);
  return {
    opacity: layer.opacity,
    // Identity-valued (multiplicative 1.0) ⇒ always-emitting is safe —
    // unlike blending_mode, which has no identity value (the campaign's
    // setter-valued-vs-identity-valued doctrine).
    absorption: layer.absorption,
    gamma: clampGamma(layer.gamma),
    intensity,
    offset,
    blending_mode: layer.blendingMode as string,
  };
}
