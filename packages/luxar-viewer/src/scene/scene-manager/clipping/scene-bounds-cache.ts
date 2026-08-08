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
 * sphere + near-cull, and caches everything. On a HIT the result is
 * cached and subsequent calls are O(1) until `invalidate()` is
 * called. On a MISS (no `position_bounds` metadata anywhere in the
 * graph) nothing is cached — there is no negative caching, so every
 * subsequent call repeats the full scene-graph traversal until
 * metadata appears in the graph.
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
 * Fraction of the bounding-box diagonal used as the near-cull margin fed to
 * the point / line / gsplat materials (`uNearCull`).
 *
 * Named and exported because it is not local: `MAX_NEAR_FAR_RATIO`'s
 * losslessness derivation (`bounds-math.ts`) is stated relative to this
 * factor, and the tests that pin that derivation import it from here rather
 * than re-typing 0.001 — otherwise retuning the near cull would leave those
 * tests green while quietly invalidating the bound.
 */
export const NEAR_CULL_DIAGONAL_FACTOR = 0.001;

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

  /** Clear the cache. Call on scene load/clear. */
  invalidate(): void {
    this._bounds = null;
    this._sphere = null;
    this._nearCull = 0.1;
  }

  /**
   * Compute the cache from the scene metadata if not already
   * computed. Idempotent: after a successful hit subsequent calls
   * are O(1) until `invalidate()` is called.
   *
   * If no `position_bounds` metadata is found in the scene graph,
   * the cache remains empty and `getBounds()` / `getSphere()`
   * return null. There is no negative caching, so a miss re-walks
   * the whole scene graph on every call until metadata appears.
   */
  ensure(scene: THREE.Scene): void {
    if (this._bounds !== null) return;
    const bounds = computeBoundsFromMetadata(scene);
    if (!bounds) return;
    this._bounds = bounds;
    this._sphere = boundingBoxToSphere(bounds);
    this._nearCull = getBoundingBoxDiagonal(bounds) * NEAR_CULL_DIAGONAL_FACTOR;
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
 * the first match. The scene loader attaches the metadata to the
 * root group, which is added as a child of the `THREE.Scene`, so
 * the match is normally found on that root group near the TOP of
 * the graph — not on the first visited object (which is the bare
 * `THREE.Scene`, carrying no `positionBounds`). But
 * `THREE.Object3D.traverse` cannot stop early (the `if (result)
 * return` below skips visit bodies, not the recursion), so this
 * walks the WHOLE graph on every call, hit or miss. Callers must
 * cache the result (`SceneBoundsCache.ensure` caches the hit; a
 * metadata-less scene re-pays the full walk each
 * call — measured harmless at real scene sizes, ~0.1 ms at ~2k
 * objects, since Luxar geometry is instanced and graphs stay small).
 */
export function findPositionBoundsInScene(
  scene: THREE.Scene
): { min: number[]; max: number[] } | null {
  let result: { min: number[]; max: number[] } | null = null;

  scene.traverse((object) => {
    if (result) return; // Already found
    const bounds = object.userData?.positionBounds;
    if (bounds && Array.isArray(bounds.min) && Array.isArray(bounds.max)) {
      result = { min: bounds.min, max: bounds.max };
    }
  });

  return result;
}
