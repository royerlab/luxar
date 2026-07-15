/**
 * Interface for materials that need camera parameter updates.
 *
 * Implemented by all main rendering materials (PointMaterial, LineMaterial,
 * GSplatMaterial) and their picking counterparts. The MaterialManager
 * broadcasts camera changes to all registered CameraAwareMaterials.
 */
import * as THREE from 'three';

export interface CameraAwareMaterial {
  /**
   * Update camera-dependent uniforms (FOV, resolution, projection mode).
   *
   * @param fov - Field of view in radians (perspective) or frustum height in world units (ortho)
   * @param resolution - Viewport resolution in pixels
   * @param isOrtho - Whether camera is orthographic (default false)
   * @param nearCull - Near-fade start distance in world units (consumed by all three geometry materials via the shared perspectiveNearFade)
   */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho?: boolean,
    nearCull?: number
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
