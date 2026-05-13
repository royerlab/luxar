/**
 * Helpers that synchronize material uniforms with the underlying geometry
 * after a commit. Live in the rendering layer because they operate on
 * Materials and Geometries rather than on data-loading state.
 *
 * F.6 of viewer-code-review-rerun action plan: moved from
 * `data/scene-loader/commit-points-geometry.ts` so the helper can be
 * reused by future rendering paths (e.g., picking shadow rebuilds)
 * without pulling in the data-loading dependency cone.
 *
 * @module rendering/material-sync-helpers
 */

import * as THREE from 'three';
import { PointMaterial } from './point-material';
import { PointPickingMaterial } from './picking/point-picking-material';

/**
 * Synchronize a Points material's dtype-scale uniforms after a geometry
 * commit.
 *
 * The placeholder-first loading pattern creates a `PointMaterial` from
 * an empty geometry (radiusScale=1, sharpnessScale=1) before real data
 * arrives. When the first commit replaces the geometry with real
 * normalized Uint8 radii/sharpness, the material uniforms must be
 * updated or the points render at `[0,1]` scale instead of
 * `[0,max_radius]` / `[0,max_sharpness]`.
 *
 * Reads `geometry.userData.{radiusScale, sharpnessScale}` and propagates
 * the values to both the render and pick materials. Idempotent.
 */
export function syncPointMaterialWithGeometry(points: THREE.Mesh): void {
  const geometry = points.geometry;
  if (!geometry) return;
  const radiusScale = (geometry.userData?.radiusScale as number | undefined) ?? 1.0;
  const sharpnessScale = (geometry.userData?.sharpnessScale as number | undefined) ?? 1.0;

  const renderMat = points.material as THREE.Material | null;
  if (renderMat instanceof PointMaterial) {
    renderMat.updateRadiusScale(radiusScale);
    renderMat.updateSharpnessScale(sharpnessScale);
  }

  // Picking shadow node was wired into userData by
  // PickingSystem.registerNode (see picking-system.ts). When picking
  // is disabled this is undefined and the helper is a no-op.
  const pickNode = points.userData?.pickNode as THREE.Object3D | undefined;
  if (pickNode) {
    const pickMat = (pickNode as THREE.Mesh | THREE.Points).material as THREE.Material | undefined;
    if (pickMat instanceof PointPickingMaterial) {
      pickMat.updateRadiusScale(radiusScale);
      pickMat.updateSharpnessScale(sharpnessScale);
    }
  }
}
