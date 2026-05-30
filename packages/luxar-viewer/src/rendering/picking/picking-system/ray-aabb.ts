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
    // `boundingBox` already covers each element's rendered footprint, not
    // just its center: all three geometries bake their footprint in at
    // creation/commit — points expand by the per-instance disc radius
    // (`create-points-node.ts` / `commit-points-geometry.ts`), lines by max
    // half-width (`line-geometry.ts`), gsplats by maxRowNorm × truncation
    // (`gsplat-geometry.ts`). So the cull treats all three identically here:
    // clone the local box and transform it to world space. Without the
    // footprint, an element whose center sits on the box surface (e.g. a
    // point at the scene's +X extreme) would be pickable only from its
    // inward side — the readback is skipped and no tooltip appears.
    worldBox = geom.boundingBox.clone().applyMatrix4(entry.main.matrixWorld);
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
