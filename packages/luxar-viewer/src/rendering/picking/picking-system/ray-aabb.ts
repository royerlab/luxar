/**
 * Ray-AABB culling for PickingSystem.
 *
 * Quick cursor-over-empty-space check that lets the orchestrator skip
 * the GPU readback when the ray doesn't intersect any registered pick
 * node's world-space AABB.
 *
 * @module rendering/picking/picking-system/ray-aabb
 */

import * as THREE from 'three';
import type { PickNodeEntry } from './registration';

/**
 * Resolve a node's cached world-space AABB, computing it from its
 * geometry's local-space bounds on cache miss. The cache survives
 * camera motion — only register/unregister/geometry-commit
 * invalidates it via {@link invalidateBoxCache}.
 */
export function getOrComputeWorldBox(
  pickId: number,
  entry: PickNodeEntry,
  cache: Map<number, THREE.Box3>
): THREE.Box3 | null {
  const geom = (entry.main as THREE.Mesh).geometry;
  if (!geom || !geom.boundingBox) return null;
  let worldBox = cache.get(pickId);
  if (!worldBox) {
    // The cull box must cover each element's rendered footprint, not just
    // its center — an element whose center sits on the box surface (e.g. a
    // point at the scene's +X extreme) would otherwise be pickable only
    // from its inward side; aiming at its outward half misses the box, the
    // readback is skipped, and no tooltip appears.
    //
    // Lines and gsplats already fold their footprint into `boundingBox` at
    // geometry creation (`line-geometry.ts` expands by max half-width,
    // `gsplat-geometry.ts` by maxRowNorm × truncation) because their meshes
    // are frustum-culled and need a footprint-aware box for that too — so
    // the pick cull inherits the correct margin for free.
    //
    // Points are the exception: their mesh has frustum culling DISABLED, so
    // `boundingBox` is left as the centers-only bounds (set identically in
    // `create-points-node.ts` and both spots in `commit-points-geometry.ts`).
    // We add the radius margin here instead — one cull-side site that covers
    // all three points paths via the `userData.maxActualRadius` they each set.
    // Use maxActualRadius (the world-space max radius) rather than
    // radiusScale: radiusScale is only a shader normalization factor and is
    // 1.0 for Float32 radii, so a large Float32-radius point at the scene
    // edge would otherwise be culled before readback. Fall back to
    // radiusScale for any geometry that predates maxActualRadius.
    // The cull is a perf shortcut only, so the margin is harmless even when
    // generous: at worst it permits a readback that votes all-background and
    // returns null.
    const local = geom.boundingBox.clone();
    const maxRadius =
      (geom.userData?.maxActualRadius as number | undefined) ??
      (geom.userData?.radiusScale as number | undefined) ??
      0;
    if (maxRadius > 0) local.expandByScalar(maxRadius);
    worldBox = local.applyMatrix4(entry.main.matrixWorld);
    cache.set(pickId, worldBox);
  }
  return worldBox;
}

/**
 * Return true when `ray` intersects at least one registered node's
 * world-space AABB. Short-circuits at the first hit so the cost is
 * O(N) only in the cache-miss case.
 */
export function rayHitsAnyNode(
  ray: THREE.Ray,
  nodeMap: ReadonlyMap<number, PickNodeEntry>,
  cache: Map<number, THREE.Box3>
): boolean {
  for (const [pickId, entry] of nodeMap) {
    const worldBox = getOrComputeWorldBox(pickId, entry, cache);
    if (worldBox && ray.intersectsBox(worldBox)) return true;
  }
  return false;
}

/** Invalidate one or all cached world-space AABBs. */
export function invalidateBoxCache(cache: Map<number, THREE.Box3>, pickId?: number): void {
  if (pickId === undefined) cache.clear();
  else cache.delete(pickId);
}
