/**
 * Pure scene-graph statistics — counts the loaded geometry in a scene
 * tree.
 *
 * The logging layer in `zarr-loader.ts` calls this helper and emits one
 * info line per stat. Keeping traversal and counting here lets tests run
 * directly against real THREE objects without WebGL.
 *
 * Counted fields:
 * - `pointsObjects` / `totalPoints`: `THREE.Mesh` nodes tagged with
 *   `userData.nodeType === 'points'`. Points are instanced quad meshes;
 *   `geometry.instanceCount` is the source of truth for visible points
 *   because pooled attributes may be over-allocated beyond the visible
 *   count.
 * - `linesObjects` / `totalSegments`: `THREE.Mesh` nodes tagged with
 *   `userData.nodeType === 'lines'`. Like Points, Lines are instanced
 *   quad meshes; `geometry.instanceCount` (one instance per segment)
 *   is the source of truth, with `userData.visibleSegmentCount` as a
 *   fallback for tests that synthesize Lines meshes without setting
 *   instanceCount.
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
  linesObjects: number;
  totalSegments: number;
  gsplatsObjects: number;
  totalGSplats: number;
  spatialIndexed: number;
}

export function computeSceneStats(scene: THREE.Object3D | null | undefined): SceneStats | null {
  if (!scene || typeof scene.traverse !== 'function') return null;

  let pointsObjects = 0;
  let totalPoints = 0;
  let linesObjects = 0;
  let totalSegments = 0;
  let gsplatsObjects = 0;
  let totalGSplats = 0;
  let spatialIndexed = 0;

  scene.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh)) return;
    const nodeType = (obj.userData as { nodeType?: string })?.nodeType;

    if (nodeType === 'points') {
      pointsObjects++;
      const geometry = obj.geometry as THREE.InstancedBufferGeometry | undefined;
      // Per-point data lives in the point texture (no per-instance
      // `aCenter` attribute to fall back on); instanceCount is the
      // source of truth, visiblePointCount the commit-stamped fallback.
      if (geometry?.isInstancedBufferGeometry && Number.isFinite(geometry.instanceCount)) {
        totalPoints += geometry.instanceCount;
      } else if (obj.userData.visiblePointCount != null) {
        totalPoints += obj.userData.visiblePointCount;
      }
      if (obj.userData.attrs?.has_spatial_index) {
        spatialIndexed++;
      }
    } else if (nodeType === 'lines') {
      linesObjects++;
      const geometry = obj.geometry as THREE.InstancedBufferGeometry | undefined;
      if (geometry?.isInstancedBufferGeometry && Number.isFinite(geometry.instanceCount)) {
        totalSegments += geometry.instanceCount;
      } else if (obj.userData.visibleSegmentCount != null) {
        totalSegments += obj.userData.visibleSegmentCount;
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

  return {
    pointsObjects,
    totalPoints,
    linesObjects,
    totalSegments,
    gsplatsObjects,
    totalGSplats,
    spatialIndexed,
  };
}
