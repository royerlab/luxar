/**
 * Camera-framing helpers extracted from `scene/scene-manager.ts`.
 *
 * Shared FOV-aware fit math for two camera-framing callers:
 *
 *   - `SceneManager.centerCameraOnScene()` — F-key recenter, computes
 *     bounds from loaded geometry.
 *   - `SceneManager.autoFrameCamera()` — auto-frame on scene load,
 *     uses bounds from zarr metadata.
 *
 * Both paths (1) compute / get bounds, (2) feed scene scale to the
 * controls, (3) compute distance (perspective) or zoom
 * (orthographic), (4) position the camera, (5) lookAt the target,
 * (6) reinitialize + saveState. Steps 3-6 live here as the shared
 * `fitCameraToBounds` helper; step 1 has its own helper
 * `computeSceneBoundingBox` since the metadata path doesn't need it.
 *
 * @module scene/scene-setup/camera-framing
 */

import * as THREE from 'three';
import { config } from '../../config';
import { log, Modules } from '../../utils/log';
import {
  calculateCameraDistance,
  type BoundingBox,
} from '../scene-manager-utils';
import {
  isPerspectiveCamera,
  isOrthographicCamera,
  type LuxarCamera,
} from '../../utils/camera-utils';
import type { ControlsManager } from '../../controls/controls-manager';

/**
 * How far the user can zoom in or out relative to the "scene fits in
 * view" distance/zoom. A value of 100 means 100x zoom-in and
 * 100x zoom-out from the auto-framed view. Mirrors the original
 * inline constant in scene-manager.ts.
 */
export const ZOOM_RANGE_FACTOR = 100;

/** Result of a scene-graph traversal that aggregates Points / Lines / GSplats / InstancedMesh bounds. */
export interface SceneBoundingBoxResult {
  /** Combined world-space bounding box. Empty if no primitives were found. */
  box: THREE.Box3;
  /** Total primitive count seen during traversal (Points position counts + InstancedMesh / instanced-geometry instance counts). */
  primitiveCount: number;
}

/**
 * Walk `scene` and aggregate the world-space bounding box of every
 * renderable primitive. Handles three rendering shapes:
 *
 *   - `THREE.Points` — bounding box from the position attribute.
 *   - `THREE.InstancedMesh` — bounding box from the geometry, plus
 *     the count from `mesh.count` for the primitive count.
 *   - `THREE.Mesh` with `InstancedBufferGeometry` (used by Lines and
 *     GSplats so they don't exceed WebGL's 16 attribute-location
 *     limit) — bounding box from the geometry, plus the
 *     `instanceCount` from the geometry for the primitive count.
 *
 * Other Object3D types contribute nothing to bounds. Empty
 * geometries / zero-count primitives are skipped so a returned
 * box of `box.isEmpty() === true` actually means "no visible
 * geometry."
 *
 * Pure with respect to the scene — does NOT mutate object world
 * matrices; the caller should call `scene.updateMatrixWorld(true)`
 * before calling if needed.
 */
export function computeSceneBoundingBox(scene: THREE.Scene): SceneBoundingBoxResult {
  const box = new THREE.Box3();
  let primitiveCount = 0;

  scene.traverse((object) => {
    // Handle Points objects (point clouds)
    if (object instanceof THREE.Points) {
      const geometry = object.geometry;
      const positions = geometry.attributes.position;
      if (positions && positions.count > 0) {
        primitiveCount += positions.count;

        if (!geometry.boundingBox) {
          geometry.computeBoundingBox();
        }
        if (geometry.boundingBox) {
          const tempBox = geometry.boundingBox.clone();
          tempBox.applyMatrix4(object.matrixWorld);
          if (!tempBox.isEmpty()) {
            box.union(tempBox);
          }
        }
      }
    }

    // Handle THREE.InstancedMesh AND Mesh + InstancedBufferGeometry
    // (Lines and GSplats use the second form to dodge the 16-attribute
    // location limit).
    const isLineMesh =
      object instanceof THREE.Mesh &&
      object.userData?.nodeType === 'lines' &&
      object.geometry instanceof THREE.InstancedBufferGeometry;
    const isGSplatMesh =
      object instanceof THREE.Mesh &&
      object.userData?.nodeType === 'gsplats' &&
      object.geometry instanceof THREE.InstancedBufferGeometry;

    if (object instanceof THREE.InstancedMesh || isLineMesh || isGSplatMesh) {
      const geometry = object.geometry;
      if (!geometry.boundingBox) {
        geometry.computeBoundingBox();
      }
      if (geometry.boundingBox) {
        const instanceCount =
          object instanceof THREE.InstancedMesh
            ? object.count
            : ((geometry as THREE.InstancedBufferGeometry).instanceCount ?? 0);
        primitiveCount += instanceCount;

        const tempBox = geometry.boundingBox.clone();
        tempBox.applyMatrix4(object.matrixWorld);
        if (!tempBox.isEmpty()) {
          box.union(tempBox);
        }
      }
    }
  });

  return { box, primitiveCount };
}

/** Options for `fitCameraToBounds`. */
export interface FitCameraOptions {
  /**
   * The point the camera should look at. Typically the bounding box
   * center, but `autoFrameCamera` allows preserving an author-set
   * target by passing `controls.getFocusTarget()` instead.
   */
  lookAtTarget: THREE.Vector3;
  /**
   * If true, the controls' target is NOT updated to `lookAtTarget`
   * (the caller has already set it elsewhere). When false, the
   * helper calls `controls.setTarget(lookAtTarget)`.
   */
  preserveControlsTarget?: boolean;
  /**
   * Identifier for the log line emitted on success — distinguishes
   * "Camera centered on scene (F key)" from "Auto-framed camera"
   * messages without forcing the caller to log separately.
   */
  logLabel?: string;
}

/**
 * Apply the shared "fit camera to a bounding box" math:
 *
 *   - Set the controls' scene scale from the box diagonal.
 *   - For perspective cameras: compute the FOV/aspect-aware distance,
 *     position the camera at `lookAtTarget + (0, 0, distance)`, and
 *     set distance limits to ±`ZOOM_RANGE_FACTOR`.
 *   - For orthographic cameras: compute the zoom that fits the largest
 *     dimension into the frustum, position the camera at
 *     `lookAtTarget + (0, 0, diagonal)`, and set zoom limits to
 *     ±`ZOOM_RANGE_FACTOR`.
 *   - Run `lookAt(lookAtTarget) → updateMatrixWorld(true) →
 *     controls.setTarget(...) → controls.reinitialize() →
 *     controls.update() → controls.saveState()` so the orbit state is
 *     consistent with the new pose.
 *
 * Returns the diagonal of the bounding box (used by the caller for
 * downstream logging or empty-bounds detection); returns 0 when the
 * box is empty or has zero extent.
 */
export function fitCameraToBounds(
  camera: LuxarCamera,
  controls: ControlsManager,
  bounds: BoundingBox,
  options: FitCameraOptions
): number {
  const { lookAtTarget, preserveControlsTarget = false, logLabel = 'Fit camera' } = options;

  const sizeX = bounds.max.x - bounds.min.x;
  const sizeY = bounds.max.y - bounds.min.y;
  const sizeZ = bounds.max.z - bounds.min.z;
  const diagonal = Math.sqrt(sizeX * sizeX + sizeY * sizeY + sizeZ * sizeZ);
  if (diagonal <= 0) return 0;

  controls.setSceneScale(diagonal);

  if (isPerspectiveCamera(camera)) {
    const cameraConfig = {
      fov: camera.fov,
      aspect: camera.aspect,
      near: camera.near,
      far: camera.far,
    };
    const distance = calculateCameraDistance(bounds, cameraConfig);
    camera.position.set(lookAtTarget.x, lookAtTarget.y, lookAtTarget.z + distance);
    controls.setDistanceLimits(distance / ZOOM_RANGE_FACTOR, distance * ZOOM_RANGE_FACTOR);
  } else if (isOrthographicCamera(camera)) {
    const frustumHeight = camera.top - camera.bottom;
    const frustumWidth = camera.right - camera.left;
    const maxDim = Math.max(sizeX, sizeY, sizeZ);
    if (maxDim > 0 && frustumHeight > 0 && frustumWidth > 0) {
      const fitRatio = config.scene.defaultFitRatio;
      const zoomH = frustumHeight / (maxDim / fitRatio);
      const zoomW = frustumWidth / (maxDim / fitRatio);
      camera.zoom = Math.min(zoomH, zoomW);
      camera.updateProjectionMatrix();
      controls.setZoomLimits(camera.zoom / ZOOM_RANGE_FACTOR, camera.zoom * ZOOM_RANGE_FACTOR);
    }
    camera.position.set(lookAtTarget.x, lookAtTarget.y, lookAtTarget.z + diagonal);
  }

  camera.lookAt(lookAtTarget);
  camera.updateMatrixWorld(true);

  // Sync orbit controls with the new camera state.
  // CRITICAL: set target FIRST, then reinitialize() so the controls
  // re-derive their internal distance from the camera position we
  // just set. Without reinitialize(), the next update() would snap
  // the camera back to the old distance.
  if (!preserveControlsTarget) {
    controls.setTarget(lookAtTarget);
  }
  controls.reinitialize();
  controls.update();
  controls.saveState();

  log.success(
    Modules.SCENE_MANAGER,
    `${logLabel} (target: [${lookAtTarget.x.toFixed(2)}, ${lookAtTarget.y.toFixed(2)}, ${lookAtTarget.z.toFixed(2)}], diagonal: ${diagonal.toFixed(2)})`
  );

  return diagonal;
}
