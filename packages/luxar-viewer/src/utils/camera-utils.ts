/**
 * Camera type utilities for supporting multiple projection modes.
 *
 * Provides a union type and type guards so the codebase can work with
 * both PerspectiveCamera and OrthographicCamera without scattering
 * instanceof checks everywhere.
 */

import * as THREE from 'three';

/** Union type for all camera types supported by the viewer */
export type LuxarCamera = THREE.PerspectiveCamera | THREE.OrthographicCamera;

/** Type guard: returns true if camera is perspective */
export function isPerspectiveCamera(camera: LuxarCamera): camera is THREE.PerspectiveCamera {
  return camera instanceof THREE.PerspectiveCamera;
}

/** Type guard: returns true if camera is orthographic */
export function isOrthographicCamera(camera: LuxarCamera): camera is THREE.OrthographicCamera {
  return camera instanceof THREE.OrthographicCamera;
}

/**
 * Get the vertical FOV in radians for material/shader calculations.
 * - Perspective: converts camera.fov (degrees) to radians
 * - Orthographic: returns 0 (no perspective foreshortening)
 */
export function getCameraFovRadians(camera: LuxarCamera): number {
  if (isPerspectiveCamera(camera)) {
    return (camera.fov * Math.PI) / 180;
  }
  return 0;
}

/**
 * Update camera projection for new viewport dimensions.
 * - Perspective: updates aspect ratio
 * - Orthographic: scales frustum bounds proportionally
 * Calls updateProjectionMatrix() internally.
 */
export function updateCameraAspect(camera: LuxarCamera, width: number, height: number): void {
  if (isPerspectiveCamera(camera)) {
    camera.aspect = width / height;
  } else {
    // Orthographic: scale frustum to maintain aspect ratio while preserving vertical extent
    const currentHeight = camera.top - camera.bottom;
    const halfHeight = currentHeight / 2;
    const halfWidth = halfHeight * (width / height);
    camera.left = -halfWidth;
    camera.right = halfWidth;
  }
  camera.updateProjectionMatrix();
}

/**
 * Get the effective frustum height in world units for an orthographic camera.
 * Accounts for camera.zoom (which acts as a divisor on the frustum).
 */
export function getOrthoFrustumHeight(camera: THREE.OrthographicCamera): number {
  return (camera.top - camera.bottom) / camera.zoom;
}
