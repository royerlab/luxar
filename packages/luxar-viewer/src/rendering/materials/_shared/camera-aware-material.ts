/**
 * Interface for materials that need camera parameter updates.
 *
 * Implemented by all four main rendering materials (PointMaterial, LineMaterial,
 * GSplatMaterial, MeshMaterial) and their picking counterparts. The MaterialManager
 * broadcasts camera changes to all registered CameraAwareMaterials.
 *
 * What it carries is deliberately NOT the projection itself: every projection-derived
 * term (the perspective/ortho size scale, the ortho test, the focal length) is read
 * in shader from the projection matrix three binds per draw, so a zoom, an FOV
 * change, an off-axis frustum or an ortho/perspective swap needs no push (every
 * GLSL shader and TSL graph, lines included, takes its ortho branch from P).
 * What remains is state the projection matrix does not hold: the viewport size,
 * the scene's near-cull distance and the render target's pixel ratio.
 *
 * Mesh implements only HALF of it, and that is intended rather than a gap: it has
 * no screen-space sprite extent to size, so `resolution` is accepted and ignored
 * (its ortho test reads the projection matrix), while `nearCull` is
 * consumed for the shared near fade (#1431).
 */
import * as THREE from 'three';

export interface CameraAwareMaterial {
  /**
   * Update camera-dependent uniforms (viewport, near cull, pixel ratio).
   *
   * Push again after a resize, a near-cull change or a pixel-ratio change. The
   * projection itself — FOV, ortho zoom, off-axis frustum AND the
   * ortho/perspective kind — is read in shader from the projection matrix of
   * the draw (`luxarIsOrthoProjection()` / `isOrthoProjectionTSL()`), so a
   * camera-kind flip or a draw through another camera needs no push.
   *
   * @param resolution - Viewport resolution in pixels
   * @param nearCull - Near-fade start distance in world units (consumed by all four geometry materials via the shared perspectiveNearFade)
   * @param pixelRatio - Physical render-target pixels per CSS pixel; screen-space appearance constants scale with it above 1×
   */
  updateCameraParams(resolution: THREE.Vector2, nearCull?: number, pixelRatio?: number): void;
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
