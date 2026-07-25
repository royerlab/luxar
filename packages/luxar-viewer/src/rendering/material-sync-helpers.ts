/**
 * Helpers that synchronize material uniforms with the underlying geometry
 * after a commit. Live in the rendering layer because they operate on
 * Materials and Geometries rather than on data-loading state.
 *
 * Kept here so rendering paths can reuse the helper without pulling in
 * the data-loading dependency cone.
 *
 * @module rendering/material-sync-helpers
 */

import * as THREE from 'three';
import { PointMaterial } from './materials/point/material-glsl';
import { PointTSLMaterial } from './materials/point/material-tsl';
import { PointPickingMaterial } from './picking/point/material';
import { PointPickingTSLMaterial } from './picking/point/material-tsl';
import { LineMaterial } from './materials/line/material-glsl';
import { LineTSLMaterial } from './materials/line/material-tsl';
import { LinePickingMaterial } from './picking/line/material';
import { LinePickingTSLMaterial } from './picking/line/material-tsl';
import { GSplatMaterial } from './materials/gsplat/material-glsl';
import { GSplatTSLMaterial } from './materials/gsplat/material-tsl';
import { GSplatPickingMaterial } from './picking/gsplat/material';
import { GSplatPickingTSLMaterial } from './picking/gsplat/material-tsl';
import { getSplatTexture } from './gsplat-geometry';
import { getPointTexture } from './point-geometry';
import { getLineTexture } from './line-geometry';

/**
 * Synchronize a Points material's geometry-derived state after a
 * geometry commit: the dtype-scale uniform AND the point-texture
 * binding.
 *
 * The placeholder-first loading pattern creates a `PointMaterial` /
 * `PointTSLMaterial` from an empty geometry (radiusScale=1) before
 * real data arrives. When the first commit replaces the geometry with
 * real normalized Uint8 radii, the material uniforms must be updated
 * or the points render at `[0,1]` scale instead of `[0,max_radius]`.
 *
 * Point data lives in an RGBA32F texture that shares the geometry's
 * lifetime (`point-geometry.ts::attachPointStorage`). A pool acquire
 * may hand the node a DIFFERENT geometry+texture pair (growth,
 * best-fit reuse, first commit after the placeholder mesh), so the
 * commit rebinds `uPointTex` on the render material and — via
 * `userData.pickNode` — the pick material. Idempotent: both wrapper
 * classes no-op or cheaply re-write on an unchanged identity (the
 * common same-geometry commit), so this is safe to call on every
 * commit. Mirrors {@link syncGSplatMaterialWithGeometry}.
 */
export function syncPointMaterialWithGeometry(points: THREE.Mesh): void {
  const geometry = points.geometry;
  if (!geometry) return;
  const radiusScale = (geometry.userData?.radiusScale as number | undefined) ?? 1.0;
  const pointTexture = getPointTexture(geometry);

  const renderMat = points.material as THREE.Material | null;
  if (renderMat instanceof PointMaterial || renderMat instanceof PointTSLMaterial) {
    renderMat.updateRadiusScale(radiusScale);
    if (pointTexture) renderMat.updatePointTexture(pointTexture);
    // RGBA-alpha presence: gates the volumetric w(a) optical-depth map
    // (uHasElementAlpha). Stamped by both texel-write paths; refreshed
    // on every commit so pool geometry swaps can't leak a previous
    // tenant's flag. All three sync helpers push it identically
    // (stampPointPresenceFlags / stampLinePresenceFlags /
    // stampGSplatPresenceFlags are the stamping chokepoints).
    renderMat.updateHasElementAlpha(geometry.userData?.hasElementAlpha === true);
  }

  // Picking shadow node was wired into userData by
  // PickingSystem.registerNode (see picking-system.ts). When picking
  // is disabled this is undefined and the helper is a no-op.
  const pickNode = points.userData?.pickNode as THREE.Object3D | undefined;
  if (pickNode) {
    const pickMat = (pickNode as THREE.Mesh | THREE.Points).material as THREE.Material | undefined;
    if (pickMat instanceof PointPickingMaterial || pickMat instanceof PointPickingTSLMaterial) {
      pickMat.updateRadiusScale(radiusScale);
      if (pointTexture) pickMat.updatePointTexture(pointTexture);
    }
  }
}

/**
 * Synchronize a Lines material's line-texture binding after a geometry
 * commit.
 *
 * Segment data lives in an RGBA32F texture that shares the geometry's
 * lifetime (`line-geometry.ts::attachLineStorage`). A pool acquire may
 * hand the node a DIFFERENT geometry+texture pair (growth, best-fit
 * reuse, first commit after the placeholder mesh), so the commit
 * rebinds `uLineTex` on the render material and — via
 * `userData.pickNode` — the pick material. Idempotent: both wrapper
 * classes no-op or cheaply re-write on an unchanged identity (the
 * common same-geometry commit), so this is safe to call on every
 * commit. Mirrors {@link syncPointMaterialWithGeometry} (minus the
 * dtype scale — lines have no `radiusScale` analog; widths are raw
 * Float32 world units).
 *
 * Two documented divergences from the points twin (phase-4 lifecycle
 * review; deliberate, unit-tested):
 * - The no-texture early return below skips the `hasElementAlpha` push
 *   entirely (a placeholder geometry carries no presence information),
 *   whereas the points twin pushes the flag unconditionally. Both are
 *   safe — every real lines mesh attaches storage — but the policies
 *   differ for the unreachable placeholder case.
 * - On a THROWING pool write, the commit's `finally` still runs this
 *   sync (the texture rebind must happen so the material never samples
 *   a disposed texture), so the uniform can transiently reflect the
 *   PREVIOUS tenant's stamp — harmless because adoption pinned
 *   `instanceCount = 0` (nothing draws) and the next successful commit
 *   re-stamps.
 */
export function syncLineMaterialWithGeometry(mesh: THREE.Mesh): void {
  const geometry = mesh.geometry;
  if (!geometry) return;
  const lineTexture = getLineTexture(geometry);
  if (!lineTexture) return;

  const renderMat = mesh.material as THREE.Material | null;
  if (renderMat instanceof LineMaterial || renderMat instanceof LineTSLMaterial) {
    renderMat.updateLineTexture(lineTexture);
    // RGBA-alpha presence: gates the volumetric w(a) optical-depth map
    // (uHasElementAlpha). Stamped by every texel-write path
    // (stampLinePresenceFlags); refreshed on every commit so pool
    // geometry swaps can't leak a previous tenant's flag. Mirrors
    // syncPointMaterialWithGeometry.
    renderMat.updateHasElementAlpha(geometry.userData?.hasElementAlpha === true);
  }

  const pickNode = mesh.userData?.pickNode as THREE.Object3D | undefined;
  if (pickNode) {
    const pickMat = (pickNode as THREE.Mesh).material as THREE.Material | undefined;
    if (pickMat instanceof LinePickingMaterial || pickMat instanceof LinePickingTSLMaterial) {
      pickMat.updateLineTexture(lineTexture);
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
    // RGBA-alpha presence: gates the volumetric w(a) optical-depth map
    // (uHasElementAlpha). Stamped by every texel-write path
    // (stampGSplatPresenceFlags); refreshed on every commit so pool
    // geometry swaps can't leak a previous tenant's flag. Consolidated
    // here from the commit's former duck-typed direct call so all three
    // geometry types share the sync-helper chokepoint decomposition.
    renderMat.updateHasElementAlpha(geometry.userData?.hasElementAlpha === true);
  }

  const pickNode = mesh.userData?.pickNode as THREE.Object3D | undefined;
  if (pickNode) {
    const pickMat = (pickNode as THREE.Mesh).material as THREE.Material | undefined;
    if (pickMat instanceof GSplatPickingMaterial || pickMat instanceof GSplatPickingTSLMaterial) {
      pickMat.updateSplatTexture(splatTexture);
    }
  }
}
