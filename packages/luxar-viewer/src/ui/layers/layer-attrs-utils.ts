/**
 * Pure helpers for the layers panel — kept here so the blending-mode
 * mapping and live-attrs derivation can be unit-tested without
 * instantiating a `LayersPanel` (which needs a real DOM container,
 * AnimationController, and zarr-backed scene graph).
 *
 * The DOM-facing `LayersPanel` calls these and applies the returned
 * shape to its THREE.Material instances.
 *
 * @module ui/layers/layer-attrs-utils
 */

import * as THREE from 'three';
import { computeUniforms, type LayerInfo } from './layer-state';
import type { ComposableAttrs } from '../../data/utils/attrs-composer';
import { clamp } from '../gui/utils/value-formatting';

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
 * `blendEquation` is only meaningful for the 'max' mode (which uses
 * `CustomBlending` with `MaxEquation`); leaving it `undefined` for the
 * other modes signals "leave whatever the material already had".
 */
export interface BlendingState {
  blending: THREE.Blending;
  depthTest: boolean;
  depthWrite: boolean;
  transparent: boolean;
  blendEquation?: THREE.BlendingEquation;
}

/**
 * Map a blending-mode name to its concrete THREE.js material settings.
 *
 * Falls back to `'normal'`-equivalent state for unknown modes so a
 * malformed zarr attribute can't crash the panel (the existing
 * implementation simply ignored unknown modes — same effect).
 */
export function getBlendingState(mode: string): BlendingState {
  switch (mode) {
    case 'additive':
      return {
        blending: THREE.AdditiveBlending,
        depthTest: false,
        depthWrite: false,
        transparent: true,
      };
    case 'normal':
      return {
        blending: THREE.NormalBlending,
        depthTest: true,
        depthWrite: false,
        transparent: true,
      };
    case 'max':
      return {
        blending: THREE.CustomBlending,
        blendEquation: THREE.MaxEquation,
        depthTest: true,
        depthWrite: false,
        transparent: true,
      };
    case 'opaque':
      return {
        blending: THREE.NormalBlending,
        depthTest: true,
        depthWrite: true,
        transparent: false,
      };
    case 'luminous':
      return {
        blending: THREE.AdditiveBlending,
        depthTest: true,
        depthWrite: false,
        transparent: true,
      };
    default:
      return {
        blending: THREE.NormalBlending,
        depthTest: true,
        depthWrite: false,
        transparent: true,
      };
  }
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
    gamma: clampGamma(layer.gamma),
    intensity,
    offset,
    blending_mode: layer.blendingMode as string,
  };
}
