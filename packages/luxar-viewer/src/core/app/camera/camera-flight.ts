/**
 * Camera flight — a smooth, interruptible transition from the live camera
 * pose to a target {@link CameraSnapshot}.
 *
 * The embedder API's `setCameraPose()` snaps. A kiosk or a remote controller
 * that "flies" to a story waypoint needs the same end state reached over
 * time, without fighting the viewer's own controls. This module owns that
 * tween and nothing else:
 *
 * - The interpolation is done in the ORBIT parameterisation, not on raw
 *   positions: the focus target moves linearly, the camera's offset from it
 *   is slerped in direction and log-interpolated in distance, and `up` is
 *   slerped. A straight-line position lerp between two orbit poses passes
 *   through the object (and through the target) — this one keeps the arc.
 * - Every frame ends with the same hand-off `restoreCamera()` performs:
 *   `controls.setTarget()` + `controls.reinitialize()` + a controls `change`
 *   event, so the orbit state never snaps back, the render loop wakes, and
 *   LOD / depth-sort / picking see the new pose. The final frame IS
 *   `restoreCamera()`, so a completed flight lands on the pose bit-exactly.
 * - It is driven by the {@link AnimationController} per-frame callback (as a
 *   `continuous` callback so the idle timer cannot stop the loop mid-flight;
 *   `startAnimation()` is paired with the registration — see
 *   `reference_per_frame_callbacks_need_startanimation`).
 * - Any user input on the canvas (pointer, wheel, touch) or the keyboard
 *   cancels the flight immediately and leaves the camera where it is. The
 *   user's own volition always wins; the caller learns about it through the
 *   resolved `{ completed: false }`.
 *
 * Nothing here knows about waypoints, dimensions, or transports — it is the
 * one primitive those layers compose.
 */

import * as THREE from 'three';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import { isPerspectiveCamera, isOrthographicCamera } from '../../../utils/camera-utils';
import {
  captureSnapshot,
  dynamicClippingActive,
  restoreCamera,
  type CameraSnapshot,
} from '../snapshot/viewer-snapshot';

/** Easing curve applied to normalised flight time. */
export type FlightEasing = 'linear' | 'ease-in-out';

export interface FlyToOptions {
  /** Flight duration in milliseconds. `0` (or negative) applies the pose immediately. Default 1500. */
  durationMs?: number;
  /** Easing curve. Default `'ease-in-out'` (smoothstep). */
  easing?: FlightEasing;
  /**
   * Keep the CURRENT viewing direction and up vector; only the focus target,
   * the distance to it, and the projection parameters (fov / zoom / planes)
   * travel to the pose. The pose's own orientation is ignored.
   *
   * This is how a flight composes with the orbit turntable: `controls.update()`
   * runs before the flight's frame callback, so the auto-rotation keeps
   * advancing the direction each frame and the flight simply carries that
   * direction to the new target — the spin never pauses, and landing does not
   * swing the camera to the author's azimuth. The waypoint driver sets it
   * whenever auto-rotate is active; a controller may ask for it explicitly.
   * Meaningless in fly mode (no orbit target) and ignored there in effect.
   */
  keepOrientation?: boolean;
}

export interface FlightResult {
  /**
   * `true` when the flight reached its pose; `false` when it was cancelled by
   * user input, by a newer `flyTo()`, by `cancel()`, or by disposal.
   */
  completed: boolean;
}

/** Per-frame driver subset of {@link AnimationController} the flight needs. */
export type FlightFrameDriver = Pick<
  AnimationController,
  'addPerFrameCallback' | 'removePerFrameCallback' | 'startAnimation'
>;

export interface CameraFlightDeps {
  sceneManager: SceneManager;
  animationController: FlightFrameDriver;
  /**
   * Element whose pointer / wheel / touch input cancels a flight (the
   * canvas). `null` disables input cancellation (headless use).
   */
  inputElement: HTMLElement | null;
  /** Clock, injectable for tests. Defaults to `performance.now`. */
  now?: () => number;
}

export const DEFAULT_FLIGHT_DURATION_MS = 1500;
/** Per-frame callback id — one flight at a time, so a fixed id is correct. */
export const FLIGHT_CALLBACK_ID = 'camera-flight';

/** DOM events on the input element that count as "the user took over". */
const CANCEL_ON_ELEMENT_EVENTS = ['pointerdown', 'wheel', 'touchstart'] as const;
/** Keyboard input is dispatched at the document level by the input handler. */
const CANCEL_ON_DOCUMENT_EVENTS = ['keydown'] as const;

function isEditableKeyTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return (
    target.matches('input, textarea, select') || target.closest('[contenteditable="true"]') !== null
  );
}

export function easeFlight(t: number, easing: FlightEasing): number {
  const x = t < 0 ? 0 : t > 1 ? 1 : t;
  if (easing === 'linear') return x;
  return x * x * (3 - 2 * x);
}

type Vec3Tuple = readonly [number, number, number];

function vec3(a: Vec3Tuple): THREE.Vector3 {
  return new THREE.Vector3(a[0], a[1], a[2]);
}

/** Exponential interpolation; falls back to linear when either end is not positive. */
function lerpLog(a: number, b: number, s: number): number {
  if (a > 0 && b > 0) return Math.exp(THREE.MathUtils.lerp(Math.log(a), Math.log(b), s));
  return THREE.MathUtils.lerp(a, b, s);
}

/**
 * Rotate unit vector `from` toward unit vector `to` by fraction `s` of the
 * angle between them (spherical interpolation of directions).
 */
function slerpDirection(from: THREE.Vector3, to: THREE.Vector3, s: number): THREE.Vector3 {
  const q = new THREE.Quaternion().setFromUnitVectors(from, to);
  const partial = new THREE.Quaternion().slerp(q, s);
  return from.clone().applyQuaternion(partial).normalize();
}

/** A pre-computed interpolation between two poses; `at(s)` yields the pose at eased time `s`. */
export interface FlightPath {
  at(s: number): CameraSnapshot;
}

/**
 * Build the interpolant between two camera snapshots. Exported so the path
 * geometry can be tested without a render loop.
 */
export function buildFlightPath(from: CameraSnapshot, to: CameraSnapshot): FlightPath {
  const target0 = vec3(from.target);
  const target1 = vec3(to.target);
  const offset0 = vec3(from.position).sub(target0);
  const offset1 = vec3(to.position).sub(target1);
  const dist0 = offset0.length();
  const dist1 = offset1.length();
  // A degenerate (zero-length) offset has no direction: borrow the other end's.
  const dir0 = dist0 > 0 ? offset0.clone().divideScalar(dist0) : null;
  const dir1 = dist1 > 0 ? offset1.clone().divideScalar(dist1) : null;
  const dirA = dir0 ?? dir1 ?? new THREE.Vector3(0, 0, 1);
  const dirB = dir1 ?? dir0 ?? new THREE.Vector3(0, 0, 1);
  const up0 = vec3(from.up).normalize();
  const up1 = vec3(to.up).normalize();

  return {
    at(s: number): CameraSnapshot {
      const target = target0.clone().lerp(target1, s);
      const dir = slerpDirection(dirA, dirB, s);
      const dist = lerpLog(dist0, dist1, s);
      const position = target.clone().addScaledVector(dir, dist);
      const up = slerpDirection(up0, up1, s);
      const pose: CameraSnapshot = {
        position: [position.x, position.y, position.z],
        target: [target.x, target.y, target.z],
        up: [up.x, up.y, up.z],
        isOrtho: to.isOrtho,
        near: THREE.MathUtils.lerp(from.near, to.near, s),
        far: THREE.MathUtils.lerp(from.far, to.far, s),
      };
      if (from.fov !== undefined && to.fov !== undefined) {
        pose.fov = THREE.MathUtils.lerp(from.fov, to.fov, s);
      } else if (to.fov !== undefined) {
        pose.fov = to.fov;
      }
      if (from.zoom !== undefined && to.zoom !== undefined) {
        pose.zoom = lerpLog(from.zoom, to.zoom, s);
      } else if (to.zoom !== undefined) {
        pose.zoom = to.zoom;
      }
      return pose;
    },
  };
}

/**
 * Re-seat `pathPose` on the camera's CURRENT viewing direction and up: same
 * target, same distance and projection as the path says, but the offset from
 * the target points the way the live camera already points. Exported for
 * tests. Falls back to the path's own direction when the live offset is
 * degenerate (camera sitting on its target).
 */
export function keepOrientationPose(
  pathPose: CameraSnapshot,
  livePosition: THREE.Vector3,
  liveTarget: THREE.Vector3,
  liveUp: THREE.Vector3
): CameraSnapshot {
  const target = vec3(pathPose.target);
  const dist = vec3(pathPose.position).sub(target).length();
  const liveOffset = livePosition.clone().sub(liveTarget);
  const dir =
    liveOffset.lengthSq() > 0
      ? liveOffset.normalize()
      : vec3(pathPose.position).sub(target).normalize();
  const position = target.clone().addScaledVector(dir, dist);
  return {
    ...pathPose,
    position: [position.x, position.y, position.z],
    up: [liveUp.x, liveUp.y, liveUp.z],
  };
}

export interface ActiveFlight {
  path: FlightPath;
  pose: CameraSnapshot;
  startedAt: number;
  durationMs: number;
  easing: FlightEasing;
  keepOrientation: boolean;
  resolve: (result: FlightResult) => void;
}

/**
 * Owns at most one in-flight camera transition. Create once per app; call
 * {@link dispose} on teardown.
 */
export class CameraFlight {
  private readonly now: () => number;
  private active: ActiveFlight | null = null;
  private readonly onUserInput = (event: Event): void => {
    if (event.type === 'keydown' && isEditableKeyTarget(event.target)) return;
    this.cancel();
  };

  constructor(private readonly deps: CameraFlightDeps) {
    this.now = deps.now ?? (() => performance.now());
  }

  /** Whether a flight is currently in progress. */
  get isActive(): boolean {
    return this.active !== null;
  }

  /**
   * Fly the camera to `pose`. A flight already in progress is cancelled
   * first (its promise resolves `{ completed: false }`). Resolves when the
   * pose is reached or the flight is interrupted.
   */
  flyTo(pose: CameraSnapshot, opts?: FlyToOptions): Promise<FlightResult> {
    this.cancel();

    const durationMs = opts?.durationMs ?? DEFAULT_FLIGHT_DURATION_MS;
    const keepOrientation = opts?.keepOrientation === true;
    if (!(durationMs > 0)) {
      restoreCamera(this.deps.sceneManager, keepOrientation ? this.reseat(pose) : pose);
      return Promise.resolve({ completed: true });
    }

    const from = captureSnapshot(this.deps.sceneManager).camera;
    return new Promise<FlightResult>((resolve) => {
      this.active = {
        path: buildFlightPath(from, pose),
        pose,
        startedAt: this.now(),
        durationMs,
        easing: opts?.easing ?? 'ease-in-out',
        keepOrientation,
        resolve,
      };
      this.attachCancelListeners();
      // Register + start: registration alone never starts a stopped loop.
      // The `camera` phase: the step moves the camera before clipping, depth
      // sort and LOD read it, so each flight frame renders with its own
      // near/far rather than the previous frame's.
      this.deps.animationController.addPerFrameCallback(FLIGHT_CALLBACK_ID, () => this.step(), {
        continuous: true,
        phase: 'camera',
      });
      this.deps.animationController.startAnimation();
    });
  }

  /** Stop the current flight (if any) where it is. Idempotent. */
  cancel(): void {
    this.finish(false);
  }

  /** Cancel and release listeners. Safe to call repeatedly. */
  dispose(): void {
    this.cancel();
  }

  /** One frame of the flight. Exposed for the per-frame driver only. */
  private step(): void {
    const flight = this.active;
    if (!flight) return;
    const t = (this.now() - flight.startedAt) / flight.durationMs;
    if (t >= 1) {
      // Land exactly: restoreCamera is the same hand-off setCameraPose uses.
      // Under keepOrientation the landing keeps whatever direction the
      // turntable has reached by now, at the pose's target and distance.
      restoreCamera(
        this.deps.sceneManager,
        flight.keepOrientation ? this.reseat(flight.pose) : flight.pose
      );
      this.finish(true);
      return;
    }
    const pathPose = flight.path.at(easeFlight(t, flight.easing));
    this.applyIntermediate(flight.keepOrientation ? this.reseat(pathPose) : pathPose);
  }

  /** `keepOrientationPose` against the live camera + focus target. */
  private reseat(pathPose: CameraSnapshot): CameraSnapshot {
    const { camera, controls } = this.deps.sceneManager;
    return keepOrientationPose(pathPose, camera.position, controls.getFocusTarget(), camera.up);
  }

  /**
   * Write an intermediate pose onto the live camera + controls. Mirrors
   * `restoreCamera` (same hand-off), minus the log noise of the final call.
   */
  private applyIntermediate(pose: CameraSnapshot): void {
    const { sceneManager } = this.deps;
    const camera = sceneManager.camera;
    const target = vec3(pose.target);

    camera.position.copy(vec3(pose.position));
    camera.up.copy(vec3(pose.up));
    // Same rule as restoreCamera: under dynamic clipping the per-frame
    // updater owns near/far. It runs BEFORE this callback each frame (it was
    // registered at init), so writing the interpolated planes here overrode
    // it every frame — geometry clipped away during the flight and came back
    // on landing (reported on the stories demo).
    if (!dynamicClippingActive(sceneManager)) {
      camera.near = pose.near;
      camera.far = pose.far;
    }
    if (isPerspectiveCamera(camera) && pose.fov !== undefined) camera.fov = pose.fov;
    if (isOrthographicCamera(camera) && pose.zoom !== undefined) camera.zoom = pose.zoom;
    camera.lookAt(target);
    camera.updateProjectionMatrix();

    sceneManager.controls.setTarget(target);
    sceneManager.controls.reinitialize();
    // Also brings the world matrices up to date: this runs AFTER the frame's
    // controls.update(), so without it the view-phase callbacks (LOD, depth
    // sort) read the previous frame's view matrix.
    sceneManager.commitCameraChange();
  }

  private finish(completed: boolean): void {
    const flight = this.active;
    if (!flight) return;
    this.active = null;
    this.deps.animationController.removePerFrameCallback(FLIGHT_CALLBACK_ID);
    this.detachCancelListeners();
    flight.resolve({ completed });
  }

  private attachCancelListeners(): void {
    const el = this.deps.inputElement;
    if (el) {
      for (const type of CANCEL_ON_ELEMENT_EVENTS) {
        el.addEventListener(type, this.onUserInput, { capture: true, passive: true });
      }
    }
    if (typeof document !== 'undefined') {
      for (const type of CANCEL_ON_DOCUMENT_EVENTS) {
        document.addEventListener(type, this.onUserInput, { capture: true, passive: true });
      }
    }
  }

  private detachCancelListeners(): void {
    const el = this.deps.inputElement;
    if (el) {
      for (const type of CANCEL_ON_ELEMENT_EVENTS) {
        el.removeEventListener(type, this.onUserInput, { capture: true });
      }
    }
    if (typeof document !== 'undefined') {
      for (const type of CANCEL_ON_DOCUMENT_EVENTS) {
        document.removeEventListener(type, this.onUserInput, { capture: true });
      }
    }
  }
}
