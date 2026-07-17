/**
 * Helpers that synchronize material uniforms with the underlying geometry
 * after a commit. Live in the rendering layer because they operate on
 * Materials and Geometries rather than on data-loading state.
 *
 * Kept here so rendering paths can reuse the helper without pulling in
 * the data-loading dependency cone.
 *
 * There is deliberately no `syncLineMaterialWithGeometry`: line
 * materials carry no geometry-derived uniform (widths are raw Float32
 * per-instance attributes with no dtype-scale analog of
 * `radiusScale`, and there is no mesh-owned texture like `uSplatTex`).
 *
 * @module rendering/material-sync-helpers
 */

import * as THREE from 'three';
import { PointMaterial } from './materials/point/material-glsl';
import { PointTSLMaterial } from './materials/point/material-tsl';
import { PointPickingMaterial } from './picking/point/material';
import { PointPickingTSLMaterial } from './picking/point/material-tsl';
import { GSplatMaterial } from './materials/gsplat/material-glsl';
import { GSplatTSLMaterial } from './materials/gsplat/material-tsl';
import { GSplatPickingMaterial } from './picking/gsplat/material';
import { GSplatPickingTSLMaterial } from './picking/gsplat/material-tsl';
import { getSplatTexture } from './gsplat-geometry';

/**
 * Synchronize a Points material's dtype-scale uniforms after a geometry
 * commit.
 *
 * The placeholder-first loading pattern creates a `PointMaterial` /
 * `PointTSLMaterial` from an empty geometry (radiusScale=1) before
 * real data arrives. When the first commit replaces the geometry with
 * real normalized Uint8 radii, the material uniforms must be updated
 * or the points render at `[0,1]` scale instead of `[0,max_radius]`.
 *
 * Reads `geometry.userData.radiusScale` and propagates the value to
 * both the render and pick materials (GLSL and TSL wrappers — both
 * expose identical `updateRadiusScale` surfaces). Idempotent.
 */
export function syncPointMaterialWithGeometry(points: THREE.Mesh): void {
  const geometry = points.geometry;
  if (!geometry) return;
  const radiusScale = (geometry.userData?.radiusScale as number | undefined) ?? 1.0;

  const renderMat = points.material as THREE.Material | null;
  if (renderMat instanceof PointMaterial || renderMat instanceof PointTSLMaterial) {
    renderMat.updateRadiusScale(radiusScale);
  }

  // Picking shadow node was wired into userData by
  // PickingSystem.registerNode (see picking-system.ts). When picking
  // is disabled this is undefined and the helper is a no-op.
  const pickNode = points.userData?.pickNode as THREE.Object3D | undefined;
  if (pickNode) {
    const pickMat = (pickNode as THREE.Mesh | THREE.Points).material as THREE.Material | undefined;
    if (pickMat instanceof PointPickingMaterial || pickMat instanceof PointPickingTSLMaterial) {
      pickMat.updateRadiusScale(radiusScale);
    }
  }
}

/**
 * Synchronize a GSplats material's splat-texture binding after a
 * geometry commit.
 *
 * Splat data lives in an RGBA32F texture that shares the geometry's
 * lifetime (`gsplat-geometry.ts::attachSplatStorage`). A pool acquire
 * may hand the node a DIFFERENT geometry+texture pair (growth,
 * best-fit reuse, first commit after the placeholder mesh), so the
 * commit rebinds `uSplatTex` on the render material and — via
 * `userData.pickNode` — the pick material. Idempotent: both wrapper
 * classes no-op when the texture identity is unchanged (the common
 * same-geometry commit), so this is safe to call on every commit.
 * (The geometry itself is re-pointed by the commit's ownership
 * handoff + `invalidateRenderObjectFor`; this helper covers only the
 * material side.)
 */
export function syncGSplatMaterialWithGeometry(mesh: THREE.Mesh): void {
  const geometry = mesh.geometry;
  if (!geometry) return;
  const splatTexture = getSplatTexture(geometry);
  if (!splatTexture) return;

  const renderMat = mesh.material as THREE.Material | null;
  if (renderMat instanceof GSplatMaterial || renderMat instanceof GSplatTSLMaterial) {
    renderMat.updateSplatTexture(splatTexture);
  }

  const pickNode = mesh.userData?.pickNode as THREE.Object3D | undefined;
  if (pickNode) {
    const pickMat = (pickNode as THREE.Mesh).material as THREE.Material | undefined;
    if (pickMat instanceof GSplatPickingMaterial || pickMat instanceof GSplatPickingTSLMaterial) {
      pickMat.updateSplatTexture(splatTexture);
    }
  }
}
