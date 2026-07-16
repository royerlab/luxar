/**
 * Factory functions that build a configured control instance for each
 * supported control mode. Extracted from `controls-manager.ts` so the
 * orchestrator stays focused on lifecycle and dispatch.
 *
 * Each factory consumes a `ControlsCreationCtx` (a narrow view of the
 * manager's state) and returns the new instance. The caller is
 * responsible for attaching event forwarders and resetting the timer.
 */

import * as THREE from 'three';
import { LuxarOrbitControls } from '../luxar-orbit-controls';
import { LuxarFlyControls } from '../luxar-fly-controls';
import { config } from '../../config';
import type { LuxarCamera } from '../../utils/camera-utils';
import type { ControlsManagerConfig } from '../controls-manager';

export interface ControlsCreationCtx {
  camera: LuxarCamera;
  domElement: HTMLElement;
  config: ControlsManagerConfig;
  sceneScale: number;
  storedDistanceLimits: { min: number; max: number } | null;
  storedZoomLimits: { min: number; max: number } | null;
}

export type MouseButtonMap = {
  LEFT: THREE.MOUSE | null;
  MIDDLE: THREE.MOUSE | null;
  RIGHT: THREE.MOUSE | null;
};

/**
 * Build the LEFT/MIDDLE/RIGHT mouse-button mapping for orbit (3D) mode.
 * `enabled=true` returns the touchpad-friendly mapping (LEFT=rotate,
 * RIGHT=pan); `enabled=false` returns the CAD/Blender mapping
 * (LEFT=pan, RIGHT=rotate). Used both at orbit-control creation time
 * and live by `setNaturalDrag` to swap without recreating controls.
 */
export function naturalDragButtonMap(enabled: boolean): MouseButtonMap {
  return enabled
    ? { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN }
    : { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
}

export function createOrbitControls(ctx: ControlsCreationCtx): LuxarOrbitControls {
  const m = config.controls.scaleMultipliers;

  // Use stored limits (from auto-frame) if available, then scale-derived,
  // then hardcoded config defaults.
  const minDist =
    ctx.storedDistanceLimits?.min ??
    (ctx.sceneScale > 0
      ? ctx.sceneScale * m.minDistanceFactor
      : config.controls.orbit.zoom.minDistance);
  const maxDist =
    ctx.storedDistanceLimits?.max ??
    (ctx.sceneScale > 0
      ? ctx.sceneScale * m.maxDistanceFactor
      : config.controls.orbit.zoom.maxDistance);

  const controls = new LuxarOrbitControls(ctx.camera, ctx.domElement, {
    enableDamping: config.controls.orbit.damping.enabled,
    screenSpacePanning: true,
    autoRotate: ctx.config.autoRotate || false,
    autoRotateSpeed: ctx.config.autoRotateSpeed || 0.25,
    zoomSpeed: ctx.config.orbitZoomSpeed,
    dampingFactor: ctx.config.orbitDampingFactor,
    minDistance: minDist,
    maxDistance: maxDist,
  });

  // Shift+scroll = view-axis rotation (roll)
  controls.enableViewAxisRotation();

  // Apply "natural drag" mapping (touchpad-friendly: LEFT=rotate, RIGHT=pan)
  // when enabled. The default in LuxarOrbitControls is the mouse-friendly
  // mapping (LEFT=pan, RIGHT=rotate); we only need to act when swapping in.
  if (ctx.config.naturalDrag) {
    controls.mouseButtons = naturalDragButtonMap(true);
  }

  return controls;
}

export function createFlyControls(ctx: ControlsCreationCtx): LuxarFlyControls {
  return new LuxarFlyControls(ctx.camera, ctx.domElement, {
    movementSpeed: ctx.config.flyMovementSpeed,
    rotationSpeed: ctx.config.flyRotationSpeed,
    lookSpeed: ctx.config.flyLookSpeed,
    inertialMode: ctx.config.flyInertialMode,
    damping: ctx.config.flyDamping,
    rotationDamping: ctx.config.flyRotationDamping,
    // Keyboard is routed through InputContextManager — see input/README.md.
    externalInputManagement: true,
  });
}

export function createOrthoControls(ctx: ControlsCreationCtx): LuxarOrbitControls {
  const m = config.controls.scaleMultipliers;

  // Use stored zoom limits (from auto-frame) if available, else wide defaults.
  // Ortho zoom ~ 1/distance, so each zoom bound maps to the OPPOSITE distance
  // factor: minZoom (zoom-out floor) ← maxDistanceFactor, maxZoom (zoom-in
  // ceiling) ← minDistanceFactor. (With the old symmetric 0.01/100 factors
  // the two legs were coincidentally equal; the asymmetric split makes the
  // mapping load-bearing.)
  const minZoom = ctx.storedZoomLimits?.min ?? 1.0 / m.maxDistanceFactor;
  const maxZoom = ctx.storedZoomLimits?.max ?? 1.0 / m.minDistanceFactor;

  const controls = new LuxarOrbitControls(ctx.camera, ctx.domElement, {
    enableDamping: config.controls.orbit.damping.enabled,
    screenSpacePanning: true,
    enableRotate: false,
    // Same feel knobs as orbit — ortho is the same class, and the live
    // setOrbitZoomSpeed/-DampingFactor setters mutate whichever is current,
    // so construction must match to avoid feel-flips on mode switch.
    zoomSpeed: ctx.config.orbitZoomSpeed,
    dampingFactor: ctx.config.orbitDampingFactor,
    minZoom,
    maxZoom,
    minDistance: 0,
    maxDistance: Infinity,
  });

  // Remap: left-click = pan (Napari/Google Maps convention)
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.PAN,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: null,
  };

  // Shift+scroll = view-axis rotation (roll)
  controls.enableViewAxisRotation();

  return controls;
}
