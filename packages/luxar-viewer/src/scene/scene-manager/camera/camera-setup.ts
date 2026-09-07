/**
 * Camera-setup helpers extracted from `scene/scene-manager.ts`.
 *
 * Three independent behaviors live here, each pure with respect to the
 * scene manager (they take camera / scene / controls references in by
 * parameter and never reach out for instance state):
 *
 *   - `createDefaultPerspectiveCamera` — read FOV / clipping defaults
 *     from `config` and instantiate a `THREE.PerspectiveCamera`. Does
 *     not append anything to the scene.
 *   - `resetCameraToInitialPosition` — reset position + look-at to the
 *     `config.camera.initialPosition` home pose; the orbit-state reset
 *     and saveState() stay in the caller because they own controls.
 *   - `applyZarrViewerConfig` — apply per-scene camera overrides
 *     (position, target, target_node, up) and the optional background
 *     color from the loaded zarr's `viewer_config` userData. Returns a
 *     boolean so the caller can decide whether to skip auto-framing.
 *
 * @module scene/scene-manager/camera/camera-setup
 */

import * as THREE from 'three';
import { config } from '../../../config';
import {
  extractCameraOverrides,
  extractBackgroundColor,
} from '../../../config/zarr-bridge/viewer-config-utils';
import type { ZarrViewerConfig } from '../../../types/zarr';
import type { ControlsManager } from '../../../controls/controls-manager';
import { log, Modules } from '../../../utils/log';
import { isOrthographicCamera, type LuxarCamera } from '../../../utils/camera-utils';

/**
 * Build a fresh `PerspectiveCamera` configured with the FOV /
 * near / far defaults from `config.renderingControls.defaults`, sized
 * for `canvas.clientWidth × clientHeight` (falls back to
 * `window.innerWidth × innerHeight` when the canvas hasn't been laid
 * out yet), positioned at `config.camera.initialPosition`.
 *
 * Mirrors the original `setupCamera()` body byte-for-byte; the canvas
 * is the only thing the helper needs from the SceneManager.
 */
export function createDefaultPerspectiveCamera(canvas: HTMLCanvasElement): THREE.PerspectiveCamera {
  const width = canvas.clientWidth || window.innerWidth;
  const height = canvas.clientHeight || window.innerHeight;

  const camera = new THREE.PerspectiveCamera(
    config.renderingControls.defaults.fov,
    width / height,
    config.renderingControls.defaults.near,
    config.renderingControls.defaults.far
  );

  camera.position.set(
    config.camera.initialPosition.x,
    config.camera.initialPosition.y,
    config.camera.initialPosition.z
  );

  return camera;
}

/**
 * Reset the camera to the configured home pose: move it to
 * `config.camera.initialPosition` and aim it at the world origin.
 *
 * Note: this only resets the camera. The orbit-state reset (controls
 * reset + update + saveState) is still done by the caller because the
 * controls manager is the source of truth for orbit-target / orbit-radius.
 */
export function resetCameraToInitialPosition(camera: LuxarCamera): void {
  camera.position.set(
    config.camera.initialPosition.x,
    config.camera.initialPosition.y,
    config.camera.initialPosition.z
  );
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld(true);
}

/**
 * Walk `root` and return the world-space center of the first object
 * named `nodeName`. Returns `null` when:
 *   - no descendant has that exact `name`, OR
 *   - the descendant exists but its world-space bounding box is empty.
 *
 * The traversal stops at the first match (the early-exit `!targetObject`
 * guard inside `traverse`).
 */
export function resolveTargetNodeCenter(root: THREE.Group, nodeName: string): THREE.Vector3 | null {
  let targetObject: THREE.Object3D | null = null;

  root.traverse((obj) => {
    if (obj.name === nodeName && !targetObject) {
      targetObject = obj;
    }
  });

  if (!targetObject) return null;

  const box = new THREE.Box3().setFromObject(targetObject);
  if (box.isEmpty()) return null;

  const center = new THREE.Vector3();
  box.getCenter(center);
  return center;
}

/**
 * Apply the zarr-side `viewer_config` (camera position/target/up,
 * background color) onto the scene + camera + controls.
 *
 * Behavior is identical to the inline implementation in
 * `SceneManager.applyZarrViewerConfig`:
 *
 *   - position: assigned directly to `camera.position` if present.
 *   - target_node takes precedence over an explicit target — resolves
 *     the named node's center via `resolveTargetNodeCenter` and feeds
 *     it to `controls.setTarget(...)`. Logs a warning if the name
 *     doesn't resolve. Falls through to the explicit target only when
 *     target_node is absent.
 *   - up: applied via `camera.up.set(...)` followed by
 *     `camera.lookAt(controls.getFocusTarget())` so `reinitialize()`
 *     (which reads quaternion, not `camera.up`) picks up the author's roll.
 *   - If any of {position, target, target_node, up} were applied:
 *     `camera.updateMatrixWorld(true)` then
 *     `controls.reinitialize()` then `controls.update()`.
 *   - background color: assigned directly to `scene.background`.
 *
 * @returns `positionApplied` — true when an explicit camera position was
 *   applied; the caller uses this to suppress auto-framing. (Author target
 *   alone does NOT suppress auto-framing — a target without a position
 *   means the author wants the orbit pivot set but still wants a sensible
 *   distance.) `appliedUp` — the author's up vector when one was applied
 *   (null otherwise); the caller stores it as the scene up so camera
 *   fits/resets square to the AUTHOR's horizon instead of world +Y.
 */
export function applyZarrViewerConfig(
  root: THREE.Group,
  camera: LuxarCamera,
  controls: ControlsManager,
  scene: THREE.Scene
): { positionApplied: boolean; appliedUp: THREE.Vector3 | null } {
  const viewerConfig = root.userData?.viewerConfig as ZarrViewerConfig | undefined;
  if (!viewerConfig) return { positionApplied: false, appliedUp: null };

  const camOverrides = extractCameraOverrides(viewerConfig);
  if (camOverrides.position) {
    camera.position.set(camOverrides.position.x, camOverrides.position.y, camOverrides.position.z);
  }

  // target_node takes precedence over explicit target coordinates.
  // Use setTarget() (not lookAt()) to avoid an intermediate update() that
  // would snap the camera back before reinitialize() derives the new orbit state.
  if (camOverrides.targetNode) {
    const resolved = resolveTargetNodeCenter(root, camOverrides.targetNode);
    if (resolved) {
      controls.setTarget(resolved);
      log.info(
        Modules.SCENE_MANAGER,
        `Resolved target_node '${camOverrides.targetNode}' to (${resolved.x.toFixed(2)}, ${resolved.y.toFixed(2)}, ${resolved.z.toFixed(2)})`
      );
    } else {
      log.warning(
        Modules.SCENE_MANAGER,
        `target_node '${camOverrides.targetNode}' not found in scene graph`
      );
    }
  } else if (camOverrides.target) {
    const targetVec = new THREE.Vector3(
      camOverrides.target.x,
      camOverrides.target.y,
      camOverrides.target.z
    );
    controls.setTarget(targetVec);
  }

  if (camOverrides.up) {
    camera.up.set(camOverrides.up.x, camOverrides.up.y, camOverrides.up.z);
    // Sync camera.quaternion with the new up vector so that
    // reinitialize() (which reads quaternion, not camera.up) picks up the
    // author's roll. Use the current orbit target as the look-at point.
    camera.lookAt(controls.getFocusTarget());
  }

  // Orthographic framing: distance changes nothing under an ortho projection,
  // only zoom does. A perspective camera ignores the field.
  if (camOverrides.zoom !== undefined && isOrthographicCamera(camera)) {
    camera.zoom = camOverrides.zoom;
    camera.updateProjectionMatrix();
  }

  if (camOverrides.position || camOverrides.target || camOverrides.targetNode || camOverrides.up) {
    camera.updateMatrixWorld(true);
    controls.reinitialize();
    controls.update();
    log.info(Modules.SCENE_MANAGER, 'Applied camera config from zarr viewer_config');
  }

  // Apply background color
  const bgColor = extractBackgroundColor(viewerConfig);
  if (bgColor) {
    scene.background = new THREE.Color(bgColor);
    log.info(Modules.SCENE_MANAGER, `Applied background color from zarr: ${bgColor}`);
  }

  return {
    positionApplied: !!camOverrides.position,
    appliedUp: camOverrides.up
      ? new THREE.Vector3(camOverrides.up.x, camOverrides.up.y, camOverrides.up.z)
      : null,
  };
}
