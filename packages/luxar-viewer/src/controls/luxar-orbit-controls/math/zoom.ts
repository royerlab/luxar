/**
 * Zoom math (vendored from THREE.js OrbitControls).
 * Extracted from `luxar-orbit-controls.ts` so the orchestrator file
 * stays focused on its sequenced update step and DOM lifecycle.
 */

import * as THREE from 'three';
import type { LuxarCamera } from '../../../utils/camera-utils';
import { clamp } from '../../../utils/clamp';

/** Convert a raw wheel/pointer delta into a multiplicative zoom factor. */
export function computeZoomScale(delta: number, zoomSpeed: number): number {
  const normalizedDelta = Math.abs(delta * 0.01);
  return Math.pow(0.95, zoomSpeed * normalizedDelta);
}

/**
 * Apply a zoom scale.
 *
 * For perspective cameras: the camera position is derived from
 * `distance × orientation` elsewhere, so we just return the new
 * `distance` value (caller assigns it).
 *
 * For orthographic cameras: `zoom` is mutated in place (clamped against
 * the supplied limits) and the projection matrix is refreshed. The
 * returned distance is unchanged.
 */
export function applyZoomScale(
  camera: LuxarCamera,
  currentDistance: number,
  scale: number,
  minZoom: number,
  maxZoom: number
): number {
  if (camera instanceof THREE.PerspectiveCamera) {
    return currentDistance * scale;
  }
  const cam = camera as THREE.OrthographicCamera;
  cam.zoom = clamp(cam.zoom / scale, minZoom, maxZoom);
  cam.updateProjectionMatrix();
  return currentDistance;
}
