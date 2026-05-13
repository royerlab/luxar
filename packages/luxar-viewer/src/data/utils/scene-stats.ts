/**
 * Pure scene-graph statistics — counts the loaded geometry in a scene
 * tree.
 *
 * Extracted from `zarr-loader.ts::logSceneStats` so the traversal /
 * counting logic can be tested directly against real THREE objects
 * (no WebGL needed). The logging layer in `zarr-loader.ts` calls this
 * helper and emits one info line per stat.
 *
 * Counts mirror the legacy logSceneStats output:
 * - `pointsObjects` / `totalPoints`: `THREE.Mesh` nodes tagged with
 *   `userData.nodeType === 'points'`. After the container migration
 *   points are instanced quad meshes; the per-instance `aCenter`
 *   attribute's `count` is the source of truth for visible points.
 * - `gsplatsObjects` / `totalGSplats`: `THREE.Mesh` nodes tagged with
 *   `userData.nodeType === 'gsplats'`. `userData.visibleSplatCount`
 *   contributes to the running total (default 0 if absent).
 * - `spatialIndexed`: how many of the above also set
 *   `userData.attrs.has_spatial_index`.
 *
 * Returns null if `scene.traverse` is not a function (e.g. mocked
 * scene); callers should treat that as "no stats available".
 */

import * as THREE from 'three';

export interface SceneStats {
  pointsObjects: number;
  totalPoints: number;
  gsplatsObjects: number;
  totalGSplats: number;
  spatialIndexed: number;
}

export function computeSceneStats(scene: THREE.Object3D | null | undefined): SceneStats | null {
  if (!scene || typeof scene.traverse !== 'function') return null;

  let pointsObjects = 0;
  let totalPoints = 0;
  let gsplatsObjects = 0;
  let totalGSplats = 0;
  let spatialIndexed = 0;

  scene.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh)) return;
    const nodeType = (obj.userData as { nodeType?: string })?.nodeType;

    if (nodeType === 'points') {
      pointsObjects++;
      const centers = obj.geometry?.getAttribute?.('aCenter');
      if (centers) {
        totalPoints += centers.count;
      }
      if (obj.userData.attrs?.has_spatial_index) {
        spatialIndexed++;
      }
    } else if (nodeType === 'gsplats') {
      gsplatsObjects++;
      totalGSplats += obj.userData.visibleSplatCount ?? 0;
      if (obj.userData.attrs?.has_spatial_index) {
        spatialIndexed++;
      }
    }
  });

  return { pointsObjects, totalPoints, gsplatsObjects, totalGSplats, spatialIndexed };
}
