/**
 * Camera flight — a smooth, interruptible transition from the live camera
 * pose to a target {@link CameraSnapshot}.
 *
 * The embedder API's `setCameraPose()` snaps. A kiosk or a remote controller
 * that "flies" to a story waypoint needs the same end state reached over
 * time, without fighting the viewer's own controls. This module owns that
 * tween and nothing else:
 *
 * - The PATH is a trajectory from `./flight-trajectories.ts` (`orbit` by
 *   default; `zoom-pan`, `arc`, `straight`, `swing`, `fly-through`, `via`), and
 *   the TIMING is either a fixed `durationMs` or a pace: `speed` divides the
 *   path's perceived length, so travel time follows what the viewer sees.
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
import {
  buildFlightPath,
  normalizeTrajectory,
  trajectoryOwnsOrientation,
  type FlightPath,
  type FlightTrajectorySpec,
  type Vec3Tuple,
} from './flight-trajectories';

export {
  buildFlightPath,
  normalizeTrajectory,
  zoomPanProfile,
  ZOOM_PAN_RHO,
  type FlightPath,
  type FlightTrajectory,
  type FlightTrajectoryKind,
  type FlightTrajectorySpec,
} from './flight-trajectories';

/**
 * Easing curve applied to normalised flight time. `ease-in-out` is smoothstep
 * (zero velocity at both ends); `smooth` is smootherstep, which also starts and
 * ends with zero ACCELERATION — no kick at departure, no jolt on landing.
 * `cruise` ramps up over the first {@link CRUISE_RAMP} of the flight, holds a
 * constant speed, and ramps down over the last: paced travel rather than a
 * jump, since the camera visibly moves for nearly all of the flight. Its
 * ramps are smoothstep in VELOCITY, so acceleration is continuous and zero at
 * both ends.
 */
export type FlightEasing = 'linear' | 'ease-in-out' | 'smooth' | 'cruise';

/** Fraction of a `cruise` flight spent ramping up (and, again, ramping down). */
export const CRUISE_RAMP = 0.2;

/** Duration bounds for a paced (`speed`) flight when the caller gives none. */
export const DEFAULT_PACED_DURATION_RANGE_MS: readonly [number, number] = [1500, 8000];

export interface FlyToOptions {
  /** Flight duration in milliseconds. `0` (or negative) applies the pose immediately. Default 1500. */
  durationMs?: number;
  /** Easing curve. Default `'ease-in-out'` (smoothstep). */
  easing?: FlightEasing;
  /**
   * Path between the poses: a name (`'orbit'`, the default, `'zoom-pan'`,
   * `'arc'`, `'straight'`, `'swing'`, `'fly-through'`) or a parameterised
   * trajectory (`{ kind: 'arc', lift: 0.8 }`, `{ kind: 'via', via: pose }`…).
   * See `./flight-trajectories.ts`.
   */
  trajectory?: FlightTrajectorySpec;
  /** Centre a `swing` turns about when it names no pivot (the scene centre). */
  sceneCentre?: Vec3Tuple;
  /**
   * Pace the flight instead of timing it: it lasts its path's perceived length
   * divided by this speed (units per second), clamped to {@link durationRangeMs}.
   * The length counts panning in view heights, zooming in log scale and turning
   * in radians, so travel time follows the distance the viewer perceives — zoom
   * included — for every trajectory.
   */
  speed?: number;
  /** `[min, max]` duration of a paced flight. Default {@link DEFAULT_PACED_DURATION_RANGE_MS}. */
  durationRangeMs?: readonly [number, number];
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
  if (easing === 'smooth') return x * x * x * (x * (6 * x - 15) + 10);
  if (easing === 'cruise') return cruiseEase(x, CRUISE_RAMP);
  return x * x * (3 - 2 * x);
}

/**
 * Ramp–cruise–ramp position profile. Velocity rises as smoothstep over `ramp`,
 * holds its peak `1 / (1 − ramp)`, and falls symmetrically, so the position
 * covers exactly 0 → 1. ∫₀ˣ smoothstep = x³ − x⁴/2 gives the ramp's position.
 */
function cruiseEase(x: number, ramp: number): number {
  if (!(ramp > 0)) return x;
  const peak = 1 / (1 - ramp);
  const up = (t: number): number => {
    const u = t / ramp;
    return peak * ramp * (u * u * u - (u * u * u * u) / 2);
  };
  if (x <= ramp) return up(x);
  if (x >= 1 - ramp) return 1 - up(1 - x);
  return peak * (ramp / 2 + (x - ramp));
}

function vec3(a: Vec3Tuple): THREE.Vector3 {
  return new THREE.Vector3(a[0], a[1], a[2]);
}

/**
 * How long a flight along `path` lasts: paced by `opts.speed` (its perceived
 * length over the speed, clamped to the duration range), otherwise
 * `opts.durationMs` or the default. Exported for tests.
 */
export function flightDurationMs(path: FlightPath, opts?: FlyToOptions): number {
  const speed = opts?.speed;
  if (speed !== undefined && speed > 0) {
    const [lo, hi] = opts?.durationRangeMs ?? DEFAULT_PACED_DURATION_RANGE_MS;
    return THREE.MathUtils.clamp((path.length / speed) * 1000, lo, hi);
  }
  return opts?.durationMs ?? DEFAULT_FLIGHT_DURATION_MS;
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

    const trajectory = normalizeTrajectory(opts?.trajectory);
    // A trajectory that defines the view direction along the way takes it over
    // from the turntable for the flight; the spin resumes from where it lands.
    const keepOrientation =
      opts?.keepOrientation === true && !trajectoryOwnsOrientation(trajectory);
    const from = captureSnapshot(this.deps.sceneManager).camera;
    const centre = opts?.sceneCentre ? vec3(opts.sceneCentre) : null;
    const path = buildFlightPath(from, pose, trajectory, centre);
    const durationMs = flightDurationMs(path, opts);
    if (!(durationMs > 0)) {
      restoreCamera(this.deps.sceneManager, keepOrientation ? this.reseat(pose) : pose);
      return Promise.resolve({ completed: true });
    }

    return new Promise<FlightResult>((resolve) => {
      this.active = {
        path,
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
      // Returns false: a flight frame moves only the camera (pose, near/far,
      // fov/zoom), which the loop's view signature already detects.
      this.deps.animationController.addPerFrameCallback(
        FLIGHT_CALLBACK_ID,
        () => {
          this.step();
          return false;
        },
        { continuous: true, phase: 'camera' }
      );
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
    const ease = (x: number): number => easeFlight(x, flight.easing);
    const pathPose = flight.path.atTime ? flight.path.atTime(t, ease) : flight.path.at(ease(t));
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
    // Same rule as restoreCamera: under dynamic clipping the view-phase
    // updater owns near/far and runs after this camera-phase callback, so it
    // sees this frame's pose.
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
