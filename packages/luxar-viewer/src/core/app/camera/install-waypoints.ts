/**
 * Bind a loaded scene's authored story waypoints (`viewer_config.waypoints`)
 * to the dims manager: match once now (snap — the opening framing, ahead of the
 * plain `camera` block), then fly whenever a dimension change makes a different
 * waypoint match. Each port is a piece the app already owns; the driver only
 * sequences them.
 *
 * @module core/app/camera/install-waypoints
 */

import type * as THREE from 'three';

import { config } from '../../../config';
import { resolveTargetNodeCenter } from '../../../scene/scene-manager/camera/camera-setup';
import { computeBoundsFromMetadata } from '../../../scene/scene-manager/clipping/scene-bounds-cache';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import type { SceneManager } from '../../../scene/scene-manager';
import type { ZarrWaypoint } from '../../../types/zarr';
import {
  captureSnapshot as captureViewerSnapshot,
  restoreCamera,
} from '../snapshot/viewer-snapshot';
import type { CameraFlight } from './camera-flight';
import { WaypointDriver, resolveWaypointPose } from './waypoint-driver';

/** A live waypoint binding: its driver, and the teardown of its dims listener. */
export interface InstalledWaypoints {
  driver: WaypointDriver;
  dispose(): void;
}

/** The two story events, as the embedder bus carries them. */
export type StoryWaypointEvent =
  | { event: 'waypoint-departed'; index: number }
  | { event: 'waypoint-arrived'; index: number; completed: boolean };

export interface StoryWaypointPorts {
  sceneManager: SceneManager;
  /** The flyTo() driver; absent during the first load's config pass. */
  getCameraFlight(): CameraFlight | undefined;
  /** Apply a waypoint's `rendering` patch (snake_case keys, validated path). */
  applyRendering(rendering: Record<string, unknown>): void;
  /** Publish a story event on the embedder bus. */
  emit(event: StoryWaypointEvent): void;
  /** Tell the sound layer (its `on_depart` / `on_arrive` nodes). */
  notifySound(kind: 'depart' | 'arrive', when: NonNullable<ZarrWaypoint['when']>): void;
  /** Re-run the overlay manager's reveal pass (`reveal: "on_arrival"`). */
  updateOverlayVisibility(): void;
}

function sceneRoot(sceneManager: SceneManager): THREE.Group | undefined {
  return sceneManager.scene?.children?.find((c) => c.name === 'LuxarScene') as
    THREE.Group | undefined;
}

/**
 * Install `waypoints`, or return null when the scene authors none. Either way
 * the overlay reveal pass runs once, so a gate the previous scene closed opens.
 */
export function installStoryWaypoints(
  waypoints: ZarrWaypoint[] | undefined,
  ports: StoryWaypointPorts
): InstalledWaypoints | null {
  if (!Array.isArray(waypoints) || waypoints.length === 0) {
    ports.updateOverlayVisibility();
    return null;
  }
  const { sceneManager } = ports;
  const driver = new WaypointDriver(waypoints, {
    getDims: () => sceneDimsManager.getDims(),
    getLivePose: () => captureViewerSnapshot(sceneManager).camera,
    resolvePose: (camera, live) =>
      resolveWaypointPose(camera, live, {
        resolveNodeCenter: (name) => {
          const root = sceneRoot(sceneManager);
          return root ? resolveTargetNodeCenter(root, name) : null;
        },
        fovPresets: config.camera.fovPresets,
      }),
    snapTo: (pose) => restoreCamera(sceneManager, pose),
    autoRotateActive: () => sceneManager.controls.isAutoRotateActive(),
    // The default pivot of a `swing` flight: the centre of the scene's data.
    sceneCentre: () => {
      const scene = sceneManager.scene;
      const b = typeof scene?.traverse === 'function' ? computeBoundsFromMetadata(scene) : null;
      if (!b) return null;
      return [(b.min.x + b.max.x) / 2, (b.min.y + b.max.y) / 2, (b.min.z + b.max.z) / 2];
    },
    flyTo: (pose, opts) => {
      // The first load's config pass runs before the embedder hooks build the
      // flight driver; a snap is the faithful fallback.
      const flight = ports.getCameraFlight();
      if (flight) return flight.flyTo(pose, opts);
      restoreCamera(sceneManager, pose);
      return Promise.resolve({ completed: true });
    },
    applyRendering: (rendering) => ports.applyRendering(rendering),
    // The two story events: onto the embedder bus for controllers, and to the
    // sound layer for its `on_depart` / `on_arrive` nodes.
    emit: (event, payload) => {
      if (event === 'waypoint-departed') {
        ports.emit({ event, index: payload.index });
      } else {
        ports.emit({
          event,
          index: payload.index,
          completed: 'completed' in payload ? payload.completed : true,
        });
        // The flight resolved and the driver's gate is open: reveal the
        // overlays a `reveal: "on_arrival"` waypoint held back.
        ports.updateOverlayVisibility();
      }
      const when = waypoints[payload.index]?.when;
      if (when) ports.notifySound(event === 'waypoint-departed' ? 'depart' : 'arrive', when);
    },
  });
  const listener = (): void => {
    driver.evaluate('fly');
    // The overlay manager listens to the same dims manager and may have run
    // first with the previous gate state. Re-run its pass now whether the new
    // match closes OR opens the gate — same task, so nothing paints between.
    ports.updateOverlayVisibility();
  };
  sceneDimsManager.addListener(listener);
  driver.evaluate('snap');
  ports.updateOverlayVisibility();
  return {
    driver,
    dispose: () => sceneDimsManager.removeListener(listener),
  };
}
