/**
 * Scene-graph disposal helpers extracted from `scene/scene-manager.ts`.
 *
 * Three independent behaviors live here, all pure with respect to the
 * scene manager:
 *
 *   - `disposeObjectTree` — recursively dispose a single Object3D and
 *     its descendants, removing children from their parents as it goes.
 *     Used by `clearSceneContent()` for objects below the scene root
 *     that are being unloaded.
 *   - `clearLoadedSceneContent` — walk the direct children of the scene
 *     root, skip lights and `userData.isBackground` markers, dispose
 *     and remove the rest. Returns the count of removed objects so the
 *     caller can log it.
 *   - `disposeSceneGraphResources` — traverse the scene one last time
 *     on shutdown, disposing every geometry + material reference. The
 *     scene root is NOT cleared from `scene.children`; the caller is
 *     about to drop the renderer/scene/camera anyway.
 *
 * @module scene/scene-manager/render-pipeline/scene-disposal
 */

import * as THREE from 'three';
import {
  releaseDepthSortNode,
  releaseAllDepthSortNodes,
} from '../../../rendering/depth-sort-coordinator';
import { isDepthShard } from '../../../rendering/depth-sort-coordinator/depth-shards';

/**
 * Recursively dispose `obj` and every descendant, removing each child
 * from its parent as it walks. Disposes geometry and material(s) on
 * `Mesh` and `InstancedMesh` instances; non-renderable Object3Ds
 * (Group / Object3D / Light) are walked through but contribute
 * nothing to dispose themselves.
 *
 * The walk is depth-first by always disposing `children[0]` until the
 * children array is empty — the same shape `clearSceneContent` had
 * inline. This avoids re-indexing after each removal.
 */
export function disposeObjectTree(obj: THREE.Object3D): void {
  if (obj instanceof THREE.Mesh || obj instanceof THREE.InstancedMesh) {
    // A DEPTH SHARD owns nothing. Its geometry shares every attribute object —
    // base quad, index, and subarray views of the parent's ordering — with the
    // node's own geometry, and it shares the material OBJECT too. Disposing it
    // would free the parent's GPU buffers out from under it and dispose the
    // shared material a second time, destroying a compiled program still in use.
    // Dropping the reference (the recursion below removes it) is the whole
    // teardown. See `rendering/depth-sort-coordinator/depth-shards.ts`.
    if (isDepthShard(obj)) {
      while (obj.children.length > 0) {
        disposeObjectTree(obj.children[0]);
        obj.remove(obj.children[0]);
      }
      return;
    }
    // Depth sorting: drop the node's SortWorker registration (transferred
    // center buffers) with the mesh, plus any written-but-undrawn index
    // permutation's profiler lifecycle. Unconditional — only sortable nodes
    // (all four geometry types) ever register, and the release
    // is a cheap map-delete no-op for everything else; an in-flight sort
    // resolves onto the deleted coordinator state and is discarded.
    // Also detaches any shard children, so the recursion below never meets one.
    releaseDepthSortNode(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) {
        obj.material.forEach((m) => m.dispose());
      } else {
        obj.material.dispose();
      }
    }
  }

  // Recursively dispose children
  while (obj.children.length > 0) {
    disposeObjectTree(obj.children[0]);
    obj.remove(obj.children[0]);
  }
}

/**
 * Clear loaded content from a scene root, preserving lights and any
 * objects flagged with `userData.isBackground = true`.
 *
 * For each remaining direct child, disposes its full subtree via
 * `disposeObjectTree` and removes it from the scene.
 *
 * This is the dataset-switch teardown, so it ALSO drops every
 * depth-sort registration wholesale: the per-mesh release inside
 * `disposeObjectTree` covers meshes reachable from the scene walk, and
 * `releaseAllDepthSortNodes` sweeps any coordinator/worker state whose
 * mesh was never attached (or was detached before the switch) — the old
 * dataset's registrations must not outlive it either way.
 *
 * @returns number of removed objects (for logging by the caller).
 */
export function clearLoadedSceneContent(scene: THREE.Scene): number {
  releaseAllDepthSortNodes();

  const objectsToRemove: THREE.Object3D[] = [];

  // Snapshot the removable children — iterate from the back so the
  // index walk is correct even though we don't mutate during this loop.
  for (let i = scene.children.length - 1; i >= 0; i--) {
    const child = scene.children[i];
    if (child instanceof THREE.Light) continue;
    if (child.userData?.isBackground) continue;
    objectsToRemove.push(child);
  }

  for (const obj of objectsToRemove) {
    disposeObjectTree(obj);
    scene.remove(obj);
  }

  return objectsToRemove.length;
}

/**
 * Final-shutdown traversal: dispose every renderable's geometry and
 * material(s) under the scene. Unlike `clearLoadedSceneContent`, this
 * does NOT remove anything from the scene graph — the caller is about
 * to drop the scene, renderer, and camera, so detaching is unnecessary
 * and traversing once is faster than walking + removing.
 *
 * Skips non-renderable Object3Ds; those have nothing to dispose.
 */
export function disposeSceneGraphResources(scene: THREE.Scene): void {
  // Defense-in-depth: the canonical teardown (dispose-pipeline) calls
  // disposeDepthSort() separately, but an embedder driving THIS shutdown
  // path alone would otherwise leave the module-scoped coordinator map
  // pinning every sorted mesh (+ geometry + element texture) it ever
  // registered. Releasing here makes the final-shutdown traversal
  // self-sufficient; it is a no-op under the wired pipeline.
  releaseAllDepthSortNodes();
  scene.traverse((object) => {
    if ('geometry' in object && 'material' in object) {
      const mesh = object as THREE.Mesh;

      // Dispose geometry - frees vertex and index buffers on GPU
      mesh.geometry.dispose();

      // Handle both single materials and material arrays
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      } else if (Array.isArray(mesh.material)) {
        mesh.material.forEach((material) => material.dispose());
      }
    }
  });
}
