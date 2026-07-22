/**
 * Pure camera-uniform math shared by Point/GSplat materials and their
 * picking counterparts.
 *
 * Each rendering material had its own inline copy of these formulas, so
 * picking and rendering could drift out of sync (a screen-space hit test
 * would no longer match what the user saw). Centralising them here keeps
 * the math identical across the four materials and gives the formulas
 * one place to be unit-tested.
 *
 * For perspective cameras `fov` is in radians; for orthographic cameras
 * the camera-aware-material contract reuses `fov` to carry the world-
 * space frustum height. The functions branch on `isOrtho` so callers do
 * not have to special-case the two projections themselves.
 */

/**
 * Point-size factor for Point/PointPicking materials.
 *
 * The shader multiplies this by the inverse view-space DEPTH
 * (`1 / max(-mvPosition.z, 1e-20)`, matching the line + gsplat shaders;
 * the floor is a pure INF guard, not a scale floor)
 * to size a point by its world-space radius (perspective) or by 1
 * (ortho, where the full distance scaling is baked into the factor
 * below).
 *
 * - Perspective: `2 * resY / tan(fov/2)`.
 * - Ortho: `(2 * resY) / (frustumHeight / 2) = (4 * resY) / frustumHeight`.
 *   The shader uses `invDistance = 1`, so the full factor is folded in.
 */
export function computePointSizeFactor(fov: number, resY: number, isOrtho: boolean): number {
  if (isOrtho) {
    const halfFrustum = fov * 0.5;
    return (2.0 * resY) / halfFrustum;
  }
  return (2.0 * resY) / Math.tan(fov / 2);
}

/**
 * Maximum point size in pixels — half the viewport height. This tracks
 * the hardware-typical `gl_PointSize` ceiling and keeps points from
 * blowing out when the camera gets very close.
 */
export function computeMaxPointSize(resY: number): number {
  return resY * 0.5;
}

/**
 * Focal length in pixels for GSplat/GSplatPicking materials.
 *
 * The shader divides by the per-vertex view-space depth to project the
 * covariance ellipse onto screen space. We assume isotropic pixel
 * aspect (`fx === fy`), which holds for every viewport that does not
 * explicitly squish the framebuffer.
 *
 * - Perspective: `fy = resY / (2 * tan(fov/2))`.
 * - Ortho: `fy = resY / frustumHeight` (direct linear mapping; depth
 *   does not enter, but the shader still multiplies by `fy`).
 */
export function computeFocalLength(fov: number, resY: number, isOrtho: boolean): number {
  if (isOrtho) {
    return resY / fov;
  }
  return resY / (2 * Math.tan(fov / 2));
}
