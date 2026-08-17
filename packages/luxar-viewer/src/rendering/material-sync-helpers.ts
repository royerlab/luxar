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
import { getSplatTexture } from './gsplat-geometry';
import { getPointTexture } from './point-geometry';
import { getLineTexture } from './line-geometry';

/*
 * Family detection is STRUCTURAL, not `instanceof`.
 *
 * These helpers sit on the hot commit path (`commit-points-geometry.ts` and
 * siblings) and must stay synchronous, so they cannot await the lazy
 * `three/webgpu` boundary — and an `instanceof PointTSLMaterial` check would
 * have to import that class as a value, dragging the whole TSL cone into the
 * eager bundle for every WebGL user (issue #1679). Probing for the methods we
 * are about to call is both cheaper and a tighter contract: it asserts exactly
 * what this module needs and nothing about the class hierarchy. Same trick, and
 * the same reasoning, as `renderer-capabilities.ts::isWebGLRenderer`.
 *
 * The probe method is unique per geometry family — `updatePointTexture`,
 * `updateLineTexture`, `updateSplatTexture` — so a family can never be mistaken
 * for another, and `updateHasElementAlpha` separates a visual material from its
 * picking twin (only the visual one carries the volumetric alpha flag).
 */

/** Visual point material surface touched after a geometry commit. */
interface PointSyncTarget {
  updateRadiusScale(scale: number): void;
  updatePointTexture(texture: THREE.DataTexture): void;
  updateHasElementAlpha(hasElementAlpha: boolean): void;
}

/** Point picking material surface: the same minus the alpha flag. */
type PointPickSyncTarget = Omit<PointSyncTarget, 'updateHasElementAlpha'>;

/** Visual line material surface touched after a geometry commit. */
interface LineSyncTarget {
  updateLineTexture(texture: THREE.DataTexture): void;
  updateHasElementAlpha(hasElementAlpha: boolean): void;
}

/** Line picking material surface. */
type LinePickSyncTarget = Omit<LineSyncTarget, 'updateHasElementAlpha'>;

/** Visual gsplat material surface touched after a geometry commit. */
interface GSplatSyncTarget {
  updateSplatTexture(texture: THREE.DataTexture): void;
  updateHasElementAlpha(hasElementAlpha: boolean): void;
}

/** GSplat picking material surface. */
type GSplatPickSyncTarget = Omit<GSplatSyncTarget, 'updateHasElementAlpha'>;

/** True when `m` exposes `name` as a callable. */
function hasMethod<K extends string>(
  m: THREE.Material | null | undefined,
  name: K
): m is THREE.Material & Record<K, (...args: never[]) => unknown> {
  return typeof (m as unknown as Record<string, unknown> | null | undefined)?.[name] === 'function';
}

function isPointSyncTarget(
  m: THREE.Material | null | undefined
): m is THREE.Material & PointSyncTarget {
  return (
    hasMethod(m, 'updatePointTexture') &&
    hasMethod(m, 'updateRadiusScale') &&
    hasMethod(m, 'updateHasElementAlpha')
  );
}

function isPointPickSyncTarget(
  m: THREE.Material | null | undefined
): m is THREE.Material & PointPickSyncTarget {
  return hasMethod(m, 'updatePointTexture') && hasMethod(m, 'updateRadiusScale');
}

function isLineSyncTarget(
  m: THREE.Material | null | undefined
): m is THREE.Material & LineSyncTarget {
  return hasMethod(m, 'updateLineTexture') && hasMethod(m, 'updateHasElementAlpha');
}

function isLinePickSyncTarget(
  m: THREE.Material | null | undefined
): m is THREE.Material & LinePickSyncTarget {
  return hasMethod(m, 'updateLineTexture');
}

function isGSplatSyncTarget(
  m: THREE.Material | null | undefined
): m is THREE.Material & GSplatSyncTarget {
  return hasMethod(m, 'updateSplatTexture') && hasMethod(m, 'updateHasElementAlpha');
}

function isGSplatPickSyncTarget(
  m: THREE.Material | null | undefined
): m is THREE.Material & GSplatPickSyncTarget {
  return hasMethod(m, 'updateSplatTexture');
}

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
  if (isPointSyncTarget(renderMat)) {
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
    if (isPointPickSyncTarget(pickMat)) {
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
  if (isLineSyncTarget(renderMat)) {
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
    if (isLinePickSyncTarget(pickMat)) {
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
  if (isGSplatSyncTarget(renderMat)) {
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
    if (isGSplatPickSyncTarget(pickMat)) {
      pickMat.updateSplatTexture(splatTexture);
    }
  }
}
