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
import { dollyAmplitudeFromPercent } from '../types';

/**
 * The narrow view of `ControlsManager` state each control factory reads:
 * the camera and DOM element to bind, the manager config (feel/toggles), the
 * scene scale, and any auto-frame-derived distance/zoom limits that take
 * precedence over scale-derived defaults.
 */
export interface ControlsCreationCtx {
  camera: LuxarCamera;
  domElement: HTMLElement;
  config: ControlsManagerConfig;
  sceneScale: number;
  storedDistanceLimits: { min: number; max: number } | null;
  storedZoomLimits: { min: number; max: number } | null;
}

/** LEFT/MIDDLE/RIGHT mouse-button → action mapping (null = unbound). */
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

/**
 * Build the orbit (3D) control instance. Distance limits are chosen as stored
 * auto-frame limits, else scale-derived from `sceneScale`, else config
 * defaults. Enables Shift+scroll view-axis roll, and applies the natural-drag
 * button map when configured (the class default is the mouse-friendly map).
 */
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
    autoRotateAxis: ctx.config.autoRotateAxis,
    zoomSpeed: ctx.config.orbitZoomSpeed,
    dampingFactor: ctx.config.orbitDampingFactor,
    autoDolly: ctx.config.autoDolly || false,
    autoDollyAmplitude: dollyAmplitudeFromPercent(
      ctx.config.autoDollyAmplitudePercent ??
        config.controls.orbit.autoDolly.amplitudePercent.default
    ),
    autoDollyPeriod: ctx.config.autoDollyPeriod ?? config.controls.orbit.autoDolly.period.default,
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

/**
 * Build the fly control instance from the config's fly speeds/damping, with
 * `externalInputManagement` enabled so keyboard input is routed through the
 * InputContextManager rather than registered on `window`.
 */
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

/**
 * Build the ortho (2D) control instance — the same `LuxarOrbitControls` class
 * with rotation disabled. Zoom limits come from stored auto-frame limits or
 * wide defaults, each mapped to the OPPOSITE distance factor (ortho zoom ~
 * 1/distance); feel knobs match orbit so switching modes doesn't flip the
 * feel. Remaps left-click to pan (Napari/Maps convention) and enables
 * Shift+scroll view-axis roll.
 */
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
    // All three turntable fields are carried even though `enableRotate: false`
    // makes auto-rotation inert here. They live on the INSTANCE, and
    // `syncCurrentState` reads them off whichever instance is live (ortho is
    // this same class, so its `isOrbitControls` guard passes) — then
    // `applyControlType` persists the result. A default-valued ortho instance
    // therefore wipes a stored turntable choice the moment the user visits
    // ortho. `naturalDrag` and the fly config dodge this only because they
    // live on the manager rather than on the control. Animation liveness uses
    // `isAutoRotateActive()`, which also requires `enableRotate`.
    autoRotate: ctx.config.autoRotate || false,
    autoRotateSpeed: ctx.config.autoRotateSpeed || 0.25,
    autoRotateAxis: ctx.config.autoRotateAxis,
    // The auto-dolly, unlike the turntable, is NOT inert here: it is gated on
    // `enableZoom`, and zoom is exactly what "closer" means in 2D. It
    // modulates `camera.zoom` through the same `applyZoomScale` seam.
    autoDolly: ctx.config.autoDolly || false,
    autoDollyAmplitude: dollyAmplitudeFromPercent(
      ctx.config.autoDollyAmplitudePercent ??
        config.controls.orbit.autoDolly.amplitudePercent.default
    ),
    autoDollyPeriod: ctx.config.autoDollyPeriod ?? config.controls.orbit.autoDolly.period.default,
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
