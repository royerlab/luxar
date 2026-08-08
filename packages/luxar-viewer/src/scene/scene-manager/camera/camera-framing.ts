/**
 * Camera-framing helpers for `scene/scene-manager.ts`.
 *
 * Shared FOV-aware fit math for two camera-framing callers:
 *
 *   - `SceneManager.centerCameraOnScene()` — F-key recenter, computes
 *     bounds from loaded geometry.
 *   - `SceneManager.autoFrameCamera()` — auto-frame on scene load,
 *     uses bounds from zarr metadata.
 *
 * Both paths compute or get bounds, feed scene scale to the controls,
 * compute distance (perspective) or zoom (orthographic), position the
 * camera, look at the target, then reinitialize + saveState.
 * `fitCameraToBounds` owns the shared fit math; `computeSceneBoundingBox`
 * owns scene traversal for the F-key path.
 *
 * @module scene/scene-manager/camera/camera-framing
 */

import * as THREE from 'three';
import { config } from '../../../config';
import { log, Modules } from '../../../utils/log';
import {
  calculateCameraDistance,
  getBoundingBoxCenter,
  type BoundingBox,
} from '../clipping/bounds-math';
import {
  isPerspectiveCamera,
  isOrthographicCamera,
  type LuxarCamera,
} from '../../../utils/camera-utils';
import type { ControlsManager } from '../../../controls/controls-manager';

/**
 * How far the user can zoom IN relative to the "scene fits in view"
 * distance/zoom (1000 = down to 1/1000 of the framed distance).
 * Kept finite because orbit math degenerates as the target distance
 * approaches zero.
 *
 * Coupled to `MAX_NEAR_FAR_RATIO` in `clipping/bounds-math.ts`: at
 * maximum zoom-in the near-plane floor (`far / MAX_NEAR_FAR_RATIO`) must sit in
 * front of the orbit target, which is what pins that ratio's lower
 * bound. Loosening this factor without re-deriving that one would clip
 * whatever the user zoomed in on.
 */
export const ZOOM_IN_FACTOR = 1000;

/**
 * How far the user can zoom OUT relative to the "scene fits in view"
 * distance/zoom (10000 = up to 10000x the framed distance). The
 * dynamic clipping planes follow the camera outward, so this limit is
 * a "don't lose the scene" guardrail rather than a technical one —
 * hence much looser than {@link ZOOM_IN_FACTOR}.
 */
export const ZOOM_OUT_FACTOR = 10000;

/** Result of a scene-graph traversal that aggregates Points / Lines / GSplats / InstancedMesh bounds. */
export interface SceneBoundingBoxResult {
  /** Combined world-space bounding box. Empty if no primitives were found. */
  box: THREE.Box3;
  /** Total primitive count seen during traversal (instance counts for instanced-mesh geometry). */
  primitiveCount: number;
}

/**
 * Triangles a mesh geometry actually draws.
 *
 * `drawRange`, not `index.count`. Mesh allocates its index buffer once at the node's
 * full face-count capacity and draws the visible prefix via `drawRange`
 * (`rendering/mesh-geometry.ts` explains why: replacing `geometry.index` per epoch
 * leaks its GPU buffer). So `index.count` is the capacity — reading it here would
 * report every face as on-screen and, worse, give a fully-culled mesh a non-zero
 * count, which is exactly the "contributes bounds while drawing nothing" bug the
 * count guard exists to prevent.
 *
 * `drawRange.count` defaults to `Infinity` on a geometry nobody has set it on, so the
 * fallback keeps this finite rather than poisoning `primitiveCount`.
 */
function drawnTriangleCount(geometry: THREE.BufferGeometry): number {
  const drawn = geometry.drawRange.count;
  if (Number.isFinite(drawn)) return drawn / 3;
  return (geometry.index?.count ?? 0) / 3;
}

/**
 * Walk `scene` and aggregate the world-space bounding box of every
 * renderable primitive. Points, Lines, and GSplats all render as
 * `THREE.Mesh + InstancedBufferGeometry`, so one shape covers those three;
 * Mesh renders a plain indexed `BufferGeometry` and needs its own:
 *
 *   - `THREE.Mesh` with `InstancedBufferGeometry` and
 *     `userData.nodeType` in {'points', 'lines', 'gsplats'} —
 *     bounding box from the geometry, `instanceCount` for the
 *     primitive count.
 *   - `THREE.Mesh` with `userData.nodeType === 'mesh'` — bounding box from the
 *     geometry (which a mesh commit sets from the projection's INDEXED-vertex
 *     bounds, so it already excludes culled geometry — see `computeMeshBounds`),
 *     and the DRAWN TRIANGLE count (`drawRange.count / 3`, see
 *     {@link drawnTriangleCount}) for the primitive count.
 *   - `THREE.InstancedMesh` — bounding box from the geometry, plus
 *     the count from `mesh.count` for the primitive count.
 *
 * Other Object3D types contribute nothing to bounds. Empty
 * geometries / zero-count primitives are skipped so a returned
 * box of `box.isEmpty() === true` actually means "no visible
 * geometry."
 *
 * The instanced arm's `nodeType` list is NOT the geometry vocabulary, and widening
 * it alone would be a false fix: that arm also requires an
 * `InstancedBufferGeometry`, so a type rendered from a plain `BufferGeometry`
 * would still contribute nothing and frame the camera wrongly, silently. This
 * warning was left by the groundwork phase and it described a real bug — a
 * mesh-only scene returned empty bounds and zero primitives until the mesh arm
 * below was added. Any future non-instanced type needs its own arm too.
 *
 * Pure with respect to the scene — does NOT mutate object world
 * matrices; the caller should call `scene.updateMatrixWorld(true)`
 * before calling if needed.
 */
export function computeSceneBoundingBox(scene: THREE.Scene): SceneBoundingBoxResult {
  const box = new THREE.Box3();
  let primitiveCount = 0;

  scene.traverse((object) => {
    const isInstancedMesh = object instanceof THREE.InstancedMesh;
    const nodeType =
      object instanceof THREE.Mesh
        ? (object.userData as { nodeType?: string })?.nodeType
        : undefined;
    const isLuxarInstancedMesh =
      object instanceof THREE.Mesh &&
      object.geometry instanceof THREE.InstancedBufferGeometry &&
      (nodeType === 'points' || nodeType === 'lines' || nodeType === 'gsplats');
    // Mesh's own arm: a plain indexed `BufferGeometry`, so it matches neither of the
    // two branches above. Deliberately not folded into the instanced test — see the
    // false-fix note in the docstring.
    const isLuxarMesh =
      object instanceof THREE.Mesh &&
      !(object.geometry instanceof THREE.InstancedBufferGeometry) &&
      nodeType === 'mesh';

    if (!isInstancedMesh && !isLuxarInstancedMesh && !isLuxarMesh) return;

    const geometry = object.geometry;
    if (!geometry.boundingBox) {
      geometry.computeBoundingBox();
    }
    if (!geometry.boundingBox) return;

    // Primitives, counted per type in the unit each type actually draws: instances for
    // the instanced-quad types, and DRAWN TRIANGLES for a mesh. Reading what is drawn
    // rather than `n_faces` is what makes a culled mesh report what is on screen — the
    // draw range is the only thing a slice change rewrites.
    const instanceCount = isLuxarMesh
      ? drawnTriangleCount(geometry)
      : isInstancedMesh
        ? object.count
        : ((geometry as THREE.InstancedBufferGeometry).instanceCount ?? 0);
    if (instanceCount <= 0) return;
    primitiveCount += instanceCount;

    const tempBox = geometry.boundingBox.clone();
    tempBox.applyMatrix4(object.matrixWorld);
    if (!tempBox.isEmpty()) {
      box.union(tempBox);
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
  /**
   * The up vector the fitted view is squared to (see the up-reset in the
   * implementation). Defaults to world +Y; SceneManager passes the scene's
   * authored up (zarr `viewer_config.up`) so an author-oriented scene
   * frames upright in ITS OWN frame rather than snapping to world Y.
   */
  up?: THREE.Vector3;
}

/**
 * Apply the shared "fit camera to a bounding box" math:
 *
 *   - Set the controls' scene scale from the box diagonal.
 *   - For perspective cameras: compute the FOV/aspect-aware distance,
 *     position the camera at `lookAtTarget + (0, 0, distance)`, and
 *     set distance limits to `ZOOM_IN_FACTOR` in / `ZOOM_OUT_FACTOR` out.
 *   - For orthographic cameras: compute the zoom that fits the largest
 *     dimension into the frustum, position the camera at
 *     `lookAtTarget + (0, 0, diagonal)`, and set zoom limits to
 *     `ZOOM_IN_FACTOR` in / `ZOOM_OUT_FACTOR` out.
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
  const {
    lookAtTarget,
    preserveControlsTarget = false,
    logLabel = 'Fit camera',
    up = THREE.Object3D.DEFAULT_UP,
  } = options;

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
    controls.setDistanceLimits(distance / ZOOM_IN_FACTOR, distance * ZOOM_OUT_FACTOR);
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
      // Ortho zoom-IN means a LARGER camera.zoom, so the factors swap
      // sides relative to the perspective distance clamp above.
      controls.setZoomLimits(camera.zoom / ZOOM_OUT_FACTOR, camera.zoom * ZOOM_IN_FACTOR);
    }
    camera.position.set(lookAtTarget.x, lookAtTarget.y, lookAtTarget.z + diagonal);
  }

  // Reset the up vector before lookAt: orbiting overwrites `camera.up` every
  // frame (see luxar-orbit-controls/camera-application.ts), so without this a
  // fit after any orbit inherits the accumulated tilt — lookAt derives its
  // roll from `camera.up` — and "Home" lands on an oblique, rolled framing
  // instead of the same face-on view a fresh camera gets on load. The reset
  // targets `options.up` (the scene's authored up, world +Y by default) so a
  // zarr `viewer_config.up` scene squares to ITS OWN horizon, not world Y.
  camera.up.copy(up);
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

// ============================================================================
// Centering helpers.
// ============================================================================

/**
 * Result of `centerCameraOnScene`. Returns the bounding-box center
 * so the caller (SceneManager) can update its `lastBoundingBoxCenter`
 * tracking field. Returns `null` when the scene has no visible
 * geometry to center on.
 */
export type CenterResult = THREE.Vector3 | null;

/**
 * Compute scene bounds from loaded geometry and frame the camera
 * around them. Returns the bounding-box center so the caller can
 * update centering-state tracking; returns null when the scene
 * has no visible geometry.
 *
 * Called by F-key recenter via SceneManager.
 */
export function centerCameraOnScene(
  scene: THREE.Scene,
  camera: LuxarCamera,
  controls: ControlsManager,
  up?: THREE.Vector3
): CenterResult {
  // Ensure world matrices are up to date before computing bounds.
  scene.updateMatrixWorld(true);

  const { box, primitiveCount } = computeSceneBoundingBox(scene);
  if (box.isEmpty() || primitiveCount === 0) {
    log.warning(Modules.SCENE_MANAGER, 'No visible geometry found to center camera on');
    return null;
  }

  const center = box.getCenter(new THREE.Vector3());

  fitCameraToBounds(
    camera,
    controls,
    {
      min: { x: box.min.x, y: box.min.y, z: box.min.z },
      max: { x: box.max.x, y: box.max.y, z: box.max.z },
    },
    { lookAtTarget: center, logLabel: 'Camera centered on scene', up }
  );
  log.success(Modules.CONTROLS, 'Controls target updated and state saved');

  return center;
}

/**
 * Reset camera + controls target to the origin (0, 0, 0) at the
 * camera's current distance from its current target. Saves the
 * new origin-centered state as the controls' default so a reset
 * later returns here.
 */
export function centerOnOrigin(
  camera: LuxarCamera,
  controls: ControlsManager,
  up: THREE.Vector3 = THREE.Object3D.DEFAULT_UP
): void {
  // Get current camera distance from target. getFocusTarget() returns a
  // clone, so it is safe to use as a one-shot read.
  const currentDistance = camera.position.distanceTo(controls.getFocusTarget());

  const origin = new THREE.Vector3(0, 0, 0);

  camera.position.set(0, 0, currentDistance);
  // Same up-reset as fitCameraToBounds (scene-authored up, world +Y by
  // default): without it, lookAt keeps the roll accumulated by orbiting and
  // the "centered" view comes out tilted.
  camera.up.copy(up);
  camera.lookAt(origin);
  camera.updateMatrixWorld(true);

  controls.setTarget(origin);
  controls.update();
  // NOTE: Do NOT call reset() before saveState() — that would undo the
  // centering and return the camera to the previous default.
  controls.saveState();

  log.success(Modules.SCENE_MANAGER, 'Centered on origin');
}

/**
 * Result of `autoFrameCamera`. `bounds` is null when no metadata
 * is available; otherwise the bounding-box center is returned so
 * the caller can update centering-state tracking.
 */
export interface AutoFrameResult {
  /** True when bounds were found and the camera was framed. */
  framed: boolean;
  /** Bounding-box center, or null when bounds are missing or zero-extent. */
  center: THREE.Vector3 | null;
}

/**
 * Auto-frame the camera to fit a pre-computed (typically metadata-
 * derived) 3D bounding box. For orthographic cameras, adjusts zoom
 * instead of distance.
 *
 * Used by SceneManager.loadSceneData() to auto-frame on scene load.
 *
 * @param preserveTarget If true, keep the current controls target
 *   (set by zarr viewer_config) instead of overwriting it with the
 *   bounding box center.
 */
export function autoFrameCamera(
  camera: LuxarCamera,
  controls: ControlsManager,
  bounds: BoundingBox | null,
  preserveTarget: boolean = false,
  up?: THREE.Vector3
): AutoFrameResult {
  if (!bounds) {
    log.warning(Modules.SCENE_MANAGER, 'No metadata bounds available for auto-framing');
    return { framed: false, center: null };
  }

  const center = getBoundingBoxCenter(bounds);
  const lookAtTarget = preserveTarget
    ? controls.getFocusTarget()
    : new THREE.Vector3(center.x, center.y, center.z);

  const diagonal = fitCameraToBounds(camera, controls, bounds, {
    lookAtTarget,
    preserveControlsTarget: preserveTarget,
    logLabel: 'Auto-framed camera on scene',
    up,
  });
  if (diagonal === 0) {
    log.warning(Modules.SCENE_MANAGER, 'Scene bounds have zero extent, skipping auto-frame');
    return { framed: false, center: null };
  }

  return { framed: true, center: new THREE.Vector3(center.x, center.y, center.z) };
}
