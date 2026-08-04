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
 * - `meshObjects` / `totalTriangles`: `THREE.Mesh` objects tagged
 *   `userData.nodeType === 'mesh'`. `userData.visibleTriangleCount`
 *   contributes (default 0 if absent). Mesh never bumps `spatialIndexed` —
 *   it has no spatial index, so counting it would inflate a metric that
 *   means "nodes that can skip chunks on a slice change".
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
  meshObjects: number;
  totalTriangles: number;
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
  let meshObjects = 0;
  let totalTriangles = 0;
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
    } else if (nodeType === 'mesh') {
      meshObjects++;
      // The commit-stamped count, with no geometry fallback: mesh draws a plain
      // indexed BufferGeometry, so there is no `instanceCount` to read, and
      // `index.count / 3` would count the placeholder's empty index as 0 anyway.
      totalTriangles += obj.userData.visibleTriangleCount ?? 0;
      // No `spatialIndexed` bump: mesh has no spatial index in v1 (§7), and its
      // `ordering` attr is always 'none'. Counting it here would inflate a metric
      // that means "how many nodes can skip chunks on a slice change".
    }
  });

  return {
    pointsObjects,
    totalPoints,
    linesObjects,
    totalSegments,
    gsplatsObjects,
    totalGSplats,
    meshObjects,
    totalTriangles,
    spatialIndexed,
  };
}
