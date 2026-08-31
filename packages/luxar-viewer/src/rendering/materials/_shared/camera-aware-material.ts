/**
 * Interface for materials that need camera parameter updates.
 *
 * Implemented by all four main rendering materials (PointMaterial, LineMaterial,
 * GSplatMaterial, MeshMaterial) and their picking counterparts. The MaterialManager
 * broadcasts camera changes to all registered CameraAwareMaterials.
 *
 * Mesh implements only HALF of it, and that is intended rather than a gap: it has
 * no screen-space sprite extent to size, so `fov` / `resolution` are accepted and
 * ignored, while `isOrtho` / `nearCull` are consumed for the shared near fade
 * (#1431).
 */
import * as THREE from 'three';

export interface CameraAwareMaterial {
  /**
   * Update camera-dependent uniforms (FOV, resolution, projection mode).
   *
   * @param fov - Field of view in radians (perspective) or frustum height in world units (ortho)
   * @param resolution - Viewport resolution in pixels
   * @param isOrtho - Whether camera is orthographic (default false)
   * @param nearCull - Near-fade start distance in world units (consumed by all four geometry materials via the shared perspectiveNearFade)
   * @param pixelRatio - Active renderer pixel ratio; screen-space appearance constants scale with it
   */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho?: boolean,
    nearCull?: number,
    pixelRatio?: number
  ): void;
}

/**
 * Type guard to check if a material implements CameraAwareMaterial.
 */
export function isCameraAwareMaterial(material: unknown): material is CameraAwareMaterial {
  return (
    typeof material === 'object' &&
    material !== null &&
    'updateCameraParams' in material &&
    typeof (material as Record<string, unknown>).updateCameraParams === 'function'
  );
}
