/**
 * Scene-bounds cache extracted from SceneManager.
 *
 * Holds the lazily-computed 3D bounding box (projected from the
 * loaded scene's `position_bounds` metadata to the current display
 * dimensions) plus its derived bounding sphere and near-cull margin.
 *
 * Invalidated on scene load / clear. Lazy `ensure(scene)` is called
 * by callers that need a value; first call walks the scene graph
 * once for `userData.positionBounds`, projects nD min/max through
 * the current `sceneDimsManager.getDims().displayed`, computes the
 * sphere + near-cull, and caches everything. Both outcomes are
 * cached — a miss (no metadata anywhere) is remembered too — so
 * subsequent calls are O(1) either way until `invalidate()` is
 * called.
 *
 * Pure with respect to SceneManager — the cache reads its inputs
 * from the supplied scene + the global `sceneDimsManager` singleton.
 * SceneManager owns one instance and threads it through clipping-
 * policy helpers.
 *
 * @module scene/scene-manager/clipping/scene-bounds-cache
 */

import * as THREE from 'three';
import { sceneDimsManager } from '../../scene-dims-manager';
import {
  type BoundingBox,
  type BoundingSphere,
  boundingBoxToSphere,
  getBoundingBoxDiagonal,
  projectBoundsToDisplayDims,
} from './bounds-math';

/**
 * Lazily-computed, invalidatable cache of the 3D bounds /
 * bounding sphere / near-cull margin derived from the scene's
 * `position_bounds` metadata, projected to the current display
 * dimensions.
 */
export class SceneBoundsCache {
  private _bounds: BoundingBox | null = null;
  private _sphere: BoundingSphere | null = null;
  private _nearCull: number = 0.1;
  /**
   * True once `ensure()` has performed the metadata search, whether
   * it found bounds or not. Negative-caches the miss: without it, a
   * scene lacking `positionBounds` metadata would pay a full graph
   * walk on EVERY `ensure()` call (twice per frame via the clipping
   * policy + camera materials).
   */
  private _computed: boolean = false;

  /**
   * Clear the cache (including the cached "no metadata found"
   * result). Call on scene load/clear — content added after a miss
   * (late-arriving `positionBounds`) is picked up because the scene
   * loader invalidates when it attaches loaded content.
   */
  invalidate(): void {
    this._bounds = null;
    this._sphere = null;
    this._nearCull = 0.1;
    this._computed = false;
  }

  /**
   * Compute the cache from the scene metadata if not already
   * computed. Idempotent: subsequent calls are O(1) until
   * `invalidate()` is called — the search runs at most once per
   * invalidation cycle, even when it finds nothing.
   *
   * If no `position_bounds` metadata is found in the scene graph,
   * the miss itself is cached and `getBounds()` / `getSphere()`
   * return null until `invalidate()`.
   */
  ensure(scene: THREE.Scene): void {
    if (this._computed) return;
    this._computed = true;
    const bounds = computeBoundsFromMetadata(scene);
    if (!bounds) return;
    this._bounds = bounds;
    this._sphere = boundingBoxToSphere(bounds);
    this._nearCull = getBoundingBoxDiagonal(bounds) * 0.001;
  }

  /**
   * @returns the cached 3D bounds, or null if `ensure()` has not
   *   been called or no metadata bounds were found in the scene.
   */
  getBounds(): BoundingBox | null {
    return this._bounds;
  }

  /**
   * @returns the cached bounding sphere, or null if `ensure()` has
   *   not been called or no metadata bounds were found.
   */
  getSphere(): BoundingSphere | null {
    return this._sphere;
  }

  /**
   * @returns the cached near-cull safety margin (~0.1% of the
   *   bounding box diagonal). Defaults to 0.1 when no bounds are
   *   cached, matching the value `invalidate()` resets to.
   */
  getNearCull(): number {
    return this._nearCull;
  }
}

/**
 * Get the 3D bounding box from scene metadata, projecting nD
 * bounds (from `userData.positionBounds`) to display dimensions
 * (from `sceneDimsManager.getDims().displayed`, defaulting to
 * `[0,1,2]`).
 *
 * @returns 3D bounding box or null if metadata bounds not found.
 */
export function computeBoundsFromMetadata(scene: THREE.Scene): BoundingBox | null {
  const foundBounds = findPositionBoundsInScene(scene);
  if (!foundBounds) return null;

  const dims = sceneDimsManager.getDims();
  const displayDims: number[] = dims?.displayed ?? [0, 1, 2];

  return projectBoundsToDisplayDims(foundBounds.min, foundBounds.max, displayDims);
}

/**
 * Search for `userData.positionBounds` in the scene graph. Returns
 * the first match in depth-first pre-order (root-level metadata is
 * set on the root group by the scene loader, so a hit is normally
 * found within the first few nodes).
 *
 * Uses a manual stack instead of `THREE.Object3D.traverse` so the
 * walk genuinely stops at the first match — `traverse` cannot
 * early-exit (returning from the callback skips bodies, not
 * recursion). Misses still visit every node; `SceneBoundsCache`
 * negative-caches that outcome so the walk runs at most once per
 * invalidation cycle.
 */
export function findPositionBoundsInScene(
  scene: THREE.Scene
): { min: number[]; max: number[] } | null {
  const stack: THREE.Object3D[] = [scene];

  while (stack.length > 0) {
    const object = stack.pop()!;
    const bounds = object.userData?.positionBounds;
    if (bounds && Array.isArray(bounds.min) && Array.isArray(bounds.max)) {
      return { min: bounds.min, max: bounds.max };
    }
    // Push children in reverse so pop order matches traverse()'s
    // depth-first pre-order (first match is unchanged).
    for (let i = object.children.length - 1; i >= 0; i--) {
      stack.push(object.children[i]);
    }
  }

  return null;
}
