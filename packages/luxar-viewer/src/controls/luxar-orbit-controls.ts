/**
 * Unified orbit controls with quaternion-based rotation (no gimbal lock)
 * and exponential damping (smooth interaction feel).
 *
 * Combines the best of THREE.js OrbitControls (damping, pan, zoom math)
 * and ArcballControls (quaternion rotation via virtual trackball).
 *
 * Mouse mapping:
 * - 3D mode: left-drag = pan, right-drag = rotate, Shift+left = rotate, scroll = zoom
 * - Ortho mode: left-drag = pan, scroll = zoom, Shift+scroll = view-axis rotate
 *
 * Supports both PerspectiveCamera and OrthographicCamera.
 */

import * as THREE from 'three';
import { isOrthographicCamera, type LuxarCamera } from '../utils/camera-utils';
import { config } from '../config';
import { applyPan, type PanCtx } from './luxar-orbit-controls/math/pan';
import { applyToCamera, initializeFromCamera } from './luxar-orbit-controls/camera-application';
import { runUpdateStep, type OrbitUpdateCtx } from './luxar-orbit-controls/update';
import { autoRotateAxisVector } from './luxar-orbit-controls/math/auto-rotate';
import { dollyAmplitudeChangeScale, dollyScale } from './luxar-orbit-controls/math/auto-dolly';
import { applyZoomScale } from './luxar-orbit-controls/math/zoom';
import { clamp } from '../utils/clamp';
import { normalizeWheelDelta } from '../utils/wheel-delta';
import {
  type AutoRotateAxis,
  DEFAULT_AUTO_DOLLY_AMPLITUDE,
  DEFAULT_AUTO_DOLLY_PERIOD,
  DEFAULT_AUTO_ROTATE_AXIS,
} from './types';
import {
  type ControlAction,
  type OrbitInputCtx,
  handlePointerDown,
  handlePointerMove,
  handlePointerUp,
  handleWheel,
} from './luxar-orbit-controls/input/pointer';
import { handleTouchStart, handleTouchMove } from './luxar-orbit-controls/input/touch';
import { attachKeyboardPan } from './luxar-orbit-controls/input/keyboard';

/**
 * Re-export of the pointer-interaction state discriminant used internally by
 * the orbit input handlers (`'rotate' | 'pan' | 'zoom' | 'none'`).
 *
 * Surfaced here so consumers can reference the active gesture without
 * reaching into the `./luxar-orbit-controls/input/pointer` module.
 */
export type { ControlAction };

/**
 * Optional construction parameters for {@link LuxarOrbitControls}.
 *
 * Every field is optional and falls back to a built-in default (matching
 * three.js `OrbitControls` where applicable — e.g. `maxDistance`/`maxZoom`
 * default to `Infinity`). Groups: interaction feel (`enableDamping`,
 * `dampingFactor`, `*Speed`), gesture toggles (`enableRotate`/`Pan`/`Zoom`),
 * auto-rotation, zoom/distance constraints, and `trackballRadius` for the
 * virtual-trackball rotation.
 */
export interface LuxarOrbitControlsConfig {
  enableDamping?: boolean;
  dampingFactor?: number;
  rotateSpeed?: number;
  panSpeed?: number;
  zoomSpeed?: number;
  enableRotate?: boolean;
  enablePan?: boolean;
  enableZoom?: boolean;
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  autoRotateAxis?: AutoRotateAxis;
  autoDolly?: boolean;
  autoDollyAmplitude?: number;
  autoDollyPeriod?: number;
  minDistance?: number;
  maxDistance?: number;
  minZoom?: number;
  maxZoom?: number;
  screenSpacePanning?: boolean;
  trackballRadius?: number;
}

/**
 * Unified orbit/ortho camera controls with quaternion rotation and
 * exponential damping.
 *
 * Combines the damping, pan, and zoom math of three.js `OrbitControls` with
 * the gimbal-lock-free quaternion rotation of a virtual-trackball
 * `ArcballControls`, and works with both `PerspectiveCamera` and
 * `OrthographicCamera`. Pointer gestures accumulate into per-frame delta
 * buffers (rotation/pan/zoom/roll) that are applied fractionally and decayed
 * each {@link update}, giving the smooth "weighted" feel; Shift+scroll
 * view-axis roll feeds the same damped buffers, while auto-rotation is
 * applied directly in the same update step.
 *
 * This same class backs both the manager's orbit (3D) and ortho (2D) modes,
 * each with its own instance — `ControlsManager` disposes and reconstructs a
 * fresh instance on every orbit↔ortho switch, which is why the manager
 * persists its stored distance/zoom limits across that recreation.
 * {@link reinitialize} re-derives the internal distance and orientation from
 * the live camera after external camera edits, and
 * {@link saveState}/{@link reset} snapshot and restore the full pose.
 *
 * @see {@link ControlsManager} which owns and switches between control modes
 */
export class LuxarOrbitControls extends THREE.EventDispatcher<{
  change: {};
  start: {};
  end: {};
}> {
  // --- Public API ---
  public enabled: boolean = true;
  public target: THREE.Vector3;

  // Configuration
  public enableDamping: boolean;
  public dampingFactor: number;
  public rotateSpeed: number;
  public panSpeed: number;
  public zoomSpeed: number;
  public enableRotate: boolean;
  public enablePan: boolean;
  public enableZoom: boolean;
  public autoRotate: boolean;
  public autoRotateSpeed: number;
  /**
   * Camera-frame axis or fixed scene axis the turntable revolves around; see
   * {@link AutoRotateAxis}.
   * Also the default axis of {@link applyOrbitRotation}, so programmatic
   * turntables (recording) follow the same choice as the interactive one.
   */
  public autoRotateAxis: AutoRotateAxis;
  /**
   * Auto-dolly: oscillate the orbit distance on a sine while enabled — the
   * turntable's radial sibling. Gated on {@link enableZoom} rather than
   * `enableRotate`, so it also runs in ortho mode (where it modulates
   * `camera.zoom`; see `math/auto-dolly.ts`).
   *
   * Switching it OFF leaves the camera exactly where it is, mid-swing or not,
   * and freezes the phase there. That is deliberate, and it is what the
   * turntable does: disabling auto-rotation does not rewind the scene to the
   * angle it started from, so disabling the dolly must not rewind the
   * distance either (measured: rotation jumps 0.000° on disable, and a
   * snap-back would have teleported the camera by up to 95% of the viewing
   * distance in a single frame at max amplitude). Re-enabling resumes from
   * the frozen phase, so nothing jumps in either direction and a completed
   * cycle still returns to wherever the user left it.
   */
  public autoDolly: boolean;
  private _autoDollyAmplitude: number;
  /**
   * Peak dolly swing as a fraction of distance (0.15 = ±15%). While the dolly
   * is running, assigning re-derives the position at the current phase while
   * preserving the baseline. While it is switched off, assigning does not move
   * the camera because the frozen position is the user's framing (see
   * {@link autoDolly}).
   */
  public get autoDollyAmplitude(): number {
    return this._autoDollyAmplitude;
  }

  public set autoDollyAmplitude(amplitude: number) {
    const scale = dollyAmplitudeChangeScale(this.dollyPhase, this._autoDollyAmplitude, amplitude);
    this._autoDollyAmplitude = amplitude;
    if (!this.autoDolly || scale === 1) return;
    this.applyDollyScale(scale);
  }
  /** Seconds per full dolly oscillation. */
  public autoDollyPeriod: number;
  public screenSpacePanning: boolean;

  // Constraints
  public minDistance: number;
  public maxDistance: number;
  public minZoom: number;
  public maxZoom: number;

  // Mouse button mapping
  public mouseButtons: {
    LEFT: THREE.MOUSE | null;
    MIDDLE: THREE.MOUSE | null;
    RIGHT: THREE.MOUSE | null;
  };

  // --- Internal state ---
  private camera: LuxarCamera;
  private domElement: HTMLElement;

  // Quaternion orbit state
  private orientation = new THREE.Quaternion();
  private distance: number = 1;

  // Damping accumulators (applied fractionally each frame, then decayed)
  private rotationDelta = new THREE.Quaternion(); // identity = no pending rotation
  private panDelta = new THREE.Vector3();
  private zoomDelta: number = 0;
  private rollDelta: number = 0; // view-axis rotation (radians, damped)

  /**
   * Auto-dolly oscillation phase (radians, wrapped to `[0, 2π)`). NOT a
   * damping accumulator: the dolly is applied to the distance directly, and
   * this is only the clock it reads. Phase 0 is the baseline distance moving
   * inward, so enabling the feature starts with an approach.
   */
  private dollyPhase: number = 0;

  // Pointer state
  private pointers: PointerEvent[] = [];
  private pointerPositions = new Map<number, THREE.Vector2>();
  private state: ControlAction = 'none';
  private rotateStart = new THREE.Vector2();
  private panStart = new THREE.Vector2();
  private dollyStart = new THREE.Vector2();
  private trackballRadius: number;

  // Change detection
  private lastPosition = new THREE.Vector3();
  private lastQuaternion = new THREE.Quaternion();

  // Saved state for reset()
  private target0 = new THREE.Vector3();
  private position0 = new THREE.Vector3();
  private orientation0 = new THREE.Quaternion();
  private zoom0: number = 1;

  // Bound event handlers (for cleanup)
  private boundOnPointerDown: (e: PointerEvent) => void;
  private boundOnPointerMove: (e: PointerEvent) => void;
  private boundOnPointerUp: (e: PointerEvent) => void;
  private boundOnWheel: (e: WheelEvent) => void;
  private boundOnContextMenu: (e: Event) => void;

  // Ortho view-axis rotation
  private viewAxisRotationHandler: ((e: WheelEvent) => void) | null = null;

  // Keyboard pan
  public keyPanSpeed: number = 7; // pixels per arrow key press
  private keyboardDisposer: (() => void) | null = null;

  constructor(camera: LuxarCamera, domElement: HTMLElement, config?: LuxarOrbitControlsConfig) {
    super();

    this.camera = camera;
    this.domElement = domElement;
    this.target = new THREE.Vector3();

    // Apply configuration with defaults
    this.enableDamping = config?.enableDamping ?? true;
    this.dampingFactor = config?.dampingFactor ?? 0.25;
    this.rotateSpeed = config?.rotateSpeed ?? 3.0;
    this.panSpeed = config?.panSpeed ?? 1.0;
    this.zoomSpeed = config?.zoomSpeed ?? 1.0;
    this.enableRotate = config?.enableRotate ?? true;
    this.enablePan = config?.enablePan ?? true;
    this.enableZoom = config?.enableZoom ?? true;
    this.autoRotate = config?.autoRotate ?? false;
    this.autoRotateSpeed = config?.autoRotateSpeed ?? 0.25;
    this.autoRotateAxis = config?.autoRotateAxis ?? DEFAULT_AUTO_ROTATE_AXIS;
    this.autoDolly = config?.autoDolly ?? false;
    this._autoDollyAmplitude = config?.autoDollyAmplitude ?? DEFAULT_AUTO_DOLLY_AMPLITUDE;
    this.autoDollyPeriod = config?.autoDollyPeriod ?? DEFAULT_AUTO_DOLLY_PERIOD;
    this.screenSpacePanning = config?.screenSpacePanning ?? true;
    this.trackballRadius = config?.trackballRadius ?? 1.0;

    // MED-34 (audit-ack): `Infinity` is intentional API parity with
    // three.js `OrbitControls` (which also defaults max-distance and
    // max-zoom to `Infinity`). Callers that need a finite ceiling pass
    // it via `config.maxDistance` / `config.maxZoom`. The audit's
    // suggested `1e6` upper bound would silently cap users who rely on
    // the THREE-compatible default. The wheel-zoom path is already
    // hardened against runaway distance: `computeZoomScale()` clamps
    // `scale` to `(0, 1]` for outward wheel deltas, so the distance
    // never grows by more than ×1 per frame before the next clamp at
    // `update.ts`'s distance-clamp step.
    this.minDistance = config?.minDistance ?? 0.01;
    this.maxDistance = config?.maxDistance ?? Infinity;
    this.minZoom = config?.minZoom ?? 0.01;
    this.maxZoom = config?.maxZoom ?? Infinity;

    // Default: left=pan, right=rotate, scroll=zoom
    this.mouseButtons = {
      LEFT: THREE.MOUSE.PAN,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.ROTATE,
    };

    // Initialize orbit state from camera
    this.initializeFromCamera();

    // Ensure camera matrix is up-to-date (needed by pan math which reads matrix columns)
    this.camera.updateMatrixWorld();

    // Save initial state for reset()
    this.saveState();

    // Bind event handlers
    this.boundOnPointerDown = this.onPointerDown.bind(this);
    this.boundOnPointerMove = this.onPointerMove.bind(this);
    this.boundOnPointerUp = this.onPointerUp.bind(this);
    this.boundOnWheel = this.onWheel.bind(this);
    this.boundOnContextMenu = (e: Event) => e.preventDefault();

    // Attach listeners
    this.domElement.addEventListener('pointerdown', this.boundOnPointerDown);
    this.domElement.addEventListener('wheel', this.boundOnWheel, { passive: false });
    this.domElement.addEventListener('contextmenu', this.boundOnContextMenu);
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Update controls. Call every frame.
   * @param deltaTime - Optional time since last frame (seconds). Used for frame-rate independent auto-rotation.
   * @returns true if the view changed — camera position, orientation, or
   *   orthographic zoom (useful for render-on-demand).
   */
  public update(deltaTime?: number): boolean {
    return runUpdateStep(this.makeUpdateCtx(), deltaTime);
  }

  private makeUpdateCtx(): OrbitUpdateCtx {
    return {
      enableRotate: this.enableRotate,
      enableDamping: this.enableDamping,
      dampingFactor: this.dampingFactor,
      autoRotate: this.autoRotate,
      autoRotateSpeed: this.autoRotateSpeed,
      autoRotateAxis: this.autoRotateAxis,
      enableZoom: this.enableZoom,
      autoDolly: this.autoDolly,
      autoDollyAmplitude: this.autoDollyAmplitude,
      autoDollyPeriod: this.autoDollyPeriod,
      orientation: this.orientation,
      rotationDelta: this.rotationDelta,
      panDelta: this.panDelta,
      target: this.target,
      getRollDelta: () => this.rollDelta,
      setRollDelta: (v) => {
        this.rollDelta = v;
      },
      getZoomDelta: () => this.zoomDelta,
      setZoomDelta: (v) => {
        this.zoomDelta = v;
      },
      getDistance: () => this.distance,
      setDistance: (v) => {
        this.distance = v;
      },
      getDollyPhase: () => this.dollyPhase,
      setDollyPhase: (v) => {
        this.dollyPhase = v;
      },
      camera: this.camera,
      minDistance: this.minDistance,
      maxDistance: this.maxDistance,
      minZoom: this.minZoom,
      maxZoom: this.maxZoom,
      lastPosition: this.lastPosition,
      lastQuaternion: this.lastQuaternion,
      dispatch: (type) => this.dispatchEvent({ type }),
    };
  }

  /** Save current state for reset(). */
  public saveState(): void {
    this.target0.copy(this.target);
    this.position0.copy(this.camera.position);
    this.orientation0.copy(this.orientation);
    // Both PerspectiveCamera and OrthographicCamera have a `.zoom` field that
    // affects the projection matrix — save it unconditionally so reset()
    // restores user-modified zoom on perspective cameras too.
    this.zoom0 = this.camera.zoom;
  }

  /** Restore to last saved state. */
  public reset(): void {
    this.target.copy(this.target0);
    this.orientation.copy(this.orientation0);
    // Scale-relative degenerate-distance floor (matches
    // initializeFromCamera): the class's own minDistance zoom bound is
    // scene-relative once scale limits are known; an absolute 0.001
    // floor flung the camera out of tiny-unit scenes on reset().
    this.distance = Math.max(
      this.position0.distanceTo(this.target0),
      this.minDistance > 0 ? this.minDistance : 0.001
    );
    this.camera.zoom = this.zoom0;
    this.camera.updateProjectionMatrix();
    this.rotationDelta.identity();
    this.panDelta.set(0, 0, 0);
    this.zoomDelta = 0;
    this.rollDelta = 0;
    // Restart the oscillation at its baseline: reset() has just restored the
    // saved distance, so leaving the phase mid-cycle would have the next frame
    // apply the REMAINDER of a swing that no longer has a matching outbound
    // half, permanently offsetting the centre the camera breathes around.
    this.dollyPhase = 0;
    this.applyToCamera();
    this.update();
  }

  /**
   * Apply an orbit rotation by the given angle (radians).
   *
   * @param angle - Rotation angle in radians (positive = counter-clockwise when looking along the axis).
   * @param axis  - World-space axis to rotate around. Defaults to the axis
   *               {@link autoRotateAxis} names (screen-up unless changed) — the
   *               SAME axis interactive auto-rotation uses, so a recorded
   *               turntable cannot rotate unlike the preview it was set up from.
   *
   * This is the same quaternion math that auto-rotation uses — call it from turntable
   * recording or any other code that needs to orbit the camera programmatically.
   */
  public applyOrbitRotation(angle: number, axis?: THREE.Vector3): void {
    const rotAxis =
      axis ?? autoRotateAxisVector(this.autoRotateAxis, this.orientation, new THREE.Vector3());
    const q = new THREE.Quaternion().setFromAxisAngle(rotAxis, angle);
    this.orientation.premultiply(q);
    this.orientation.normalize();
    this.applyToCamera();
  }

  /**
   * Move the auto-dolly to an absolute oscillation `phase` (radians) and apply
   * the resulting distance change.
   *
   * The programmatic counterpart of the per-frame dolly, for the same reason
   * {@link applyOrbitRotation} exists: offline capture drives the camera from
   * a FRAME INDEX, not from wall-clock time (its frames wait on LOD settling,
   * so `deltaTime` there is meaningless), and both paths must agree on what
   * the oscillation is. Passing an absolute phase rather than an increment
   * lets a recording close its loop exactly — phase `2π` lands on the same
   * distance as phase `0`, with no accumulated error.
   *
   * Uses the configured {@link autoDollyAmplitude}, so a recording breathes as
   * deeply as the preview. Does nothing while {@link enableZoom} is false.
   *
   * @param phase - Absolute oscillation phase in radians.
   */
  public applyOrbitDolly(phase: number): void {
    if (!this.enableZoom || !Number.isFinite(phase)) return;
    const scale = dollyScale(this.dollyPhase, phase, this.autoDollyAmplitude);
    this.dollyPhase = phase;
    this.applyDollyScale(scale);
  }

  /** Remove the currently applied dolly offset and restart from phase zero. */
  public returnAutoDollyToBaseline(): void {
    const scale = dollyAmplitudeChangeScale(this.dollyPhase, this.autoDollyAmplitude, 0);
    this.dollyPhase = 0;
    this.applyDollyScale(scale);
  }

  /**
   * Re-derive orientation and distance from the current camera position and target.
   * Call after changing the target externally to keep the orbit state consistent.
   */
  public reinitialize(): void {
    this.initializeFromCamera();
    this.rotationDelta.identity();
    this.panDelta.set(0, 0, 0);
    this.zoomDelta = 0;
    this.rollDelta = 0;
    this.dollyPhase = 0;
  }

  /** Keep the current pose but discard residual user-input damping. */
  public settleDamping(): void {
    this.rotationDelta.identity();
    this.panDelta.set(0, 0, 0);
    this.zoomDelta = 0;
    this.rollDelta = 0;
  }

  private applyDollyScale(scale: number): void {
    if (scale === 1) return;
    this.distance = applyZoomScale(this.camera, this.distance, scale, this.minZoom, this.maxZoom);
    this.distance = clamp(this.distance, this.minDistance, this.maxDistance);
    this.applyToCamera();
  }

  /**
   * Enable Shift+scroll view-axis rotation (roll around the viewing axis).
   * Uses capture phase to intercept before other wheel handlers (zoom, FOV).
   * Works in both orbit and ortho modes.
   */
  public enableViewAxisRotation(speed: number = 0.0005): void {
    if (this.viewAxisRotationHandler) return; // Already enabled

    this.viewAxisRotationHandler = (event: WheelEvent) => {
      if (!event.shiftKey || !this.enabled) return;
      event.preventDefault();
      event.stopImmediatePropagation();

      // Accumulate into rollDelta — damping is applied in update().
      // The delta is normalized to pixel-mode equivalent first, which puts a
      // line-mode browser (Firefox reports 3 lines where Chromium reports
      // 100 px) in the same ballpark per notch: 0.024 rad against 0.050,
      // instead of 0.0015 against 0.050.
      this.rollDelta += normalizeWheelDelta(event, this.domElement) * speed;

      // Wake up animation loop (rollDelta is applied in update())
      this.dispatchEvent({ type: 'change' });
    };

    this.domElement.addEventListener('wheel', this.viewAxisRotationHandler, {
      capture: true,
      passive: false,
    });
  }

  /** Clean up all event listeners. */
  public dispose(): void {
    this.domElement.removeEventListener('pointerdown', this.boundOnPointerDown);
    this.domElement.removeEventListener('pointermove', this.boundOnPointerMove);
    this.domElement.removeEventListener('pointerup', this.boundOnPointerUp);
    this.domElement.removeEventListener('pointercancel', this.boundOnPointerUp);
    this.domElement.removeEventListener('wheel', this.boundOnWheel);
    this.domElement.removeEventListener('contextmenu', this.boundOnContextMenu);

    if (this.viewAxisRotationHandler) {
      // The DOM removeEventListener overload that accepts an options
      // object is typed `EventListenerOptions`; the `passive` flag in
      // capture mode isn't part of that subset, so we cast to the
      // structural shape we actually pass.
      this.domElement.removeEventListener('wheel', this.viewAxisRotationHandler, {
        capture: true,
      } as EventListenerOptions);
      this.viewAxisRotationHandler = null;
    }

    this.stopListenToKeyEvents();

    // Release any active pointer captures
    for (const pointer of this.pointers) {
      try {
        this.domElement.releasePointerCapture(pointer.pointerId);
      } catch {
        /* ignore */
      }
    }
    this.pointers.length = 0;
    this.pointerPositions.clear();
  }

  // ---------------------------------------------------------------------------
  // Math delegates (luxar-orbit-controls/math/*)
  // ---------------------------------------------------------------------------

  private makePanCtx(): PanCtx {
    return {
      camera: this.camera,
      distance: this.distance,
      panSpeed: this.panSpeed,
      screenSpacePanning: this.screenSpacePanning,
      domElement: this.domElement,
    };
  }

  private pan(deltaX: number, deltaY: number): void {
    applyPan(this.panDelta, deltaX, deltaY, this.makePanCtx());
  }

  // ---------------------------------------------------------------------------
  // Camera application (delegated to luxar-orbit-controls/camera-application.ts)
  // ---------------------------------------------------------------------------

  /** Apply orientation + distance + target to camera transform. */
  private applyToCamera(): void {
    applyToCamera(this.camera, this.target, this.orientation, this.distance);
  }

  /**
   * Extract orientation and distance from current camera state.
   * `minDistance` (scene diagonal × minDistanceFactor once scale limits
   * are known) supplies the scale-relative degenerate-distance floor —
   * see camera-application.ts.
   */
  private initializeFromCamera(): void {
    this.distance = initializeFromCamera(
      this.camera,
      this.target,
      this.orientation,
      this.minDistance
    );
  }

  // ---------------------------------------------------------------------------
  // Input handling (delegated to luxar-orbit-controls/input/*)
  // ---------------------------------------------------------------------------

  private makeInputCtx(): OrbitInputCtx {
    return {
      enabled: this.enabled,
      enableRotate: this.enableRotate,
      enablePan: this.enablePan,
      enableZoom: this.enableZoom,
      isOrthographic: isOrthographicCamera(this.camera),
      mouseButtons: this.mouseButtons,
      domElement: this.domElement,
      trackballRadius: this.trackballRadius,
      rotateSpeed: this.rotateSpeed,
      zoomSpeed: this.zoomSpeed,
      // Read from config per event (the ctx is rebuilt per call) so the
      // Settings slider applies to the very next wheel notch.
      wheelZoomSensitivity: config.controls.wheelZoomSensitivity,
      boundOnPointerMove: this.boundOnPointerMove,
      boundOnPointerUp: this.boundOnPointerUp,
      pointers: this.pointers,
      pointerPositions: this.pointerPositions,
      rotateStart: this.rotateStart,
      panStart: this.panStart,
      dollyStart: this.dollyStart,
      rotationDelta: this.rotationDelta,
      getState: () => this.state,
      setState: (s) => {
        this.state = s;
      },
      addZoomDelta: (delta) => {
        this.zoomDelta += delta;
      },
      addRollDelta: (delta) => {
        this.rollDelta += delta;
      },
      pan: (dx, dy) => this.pan(dx, dy),
      dispatch: (type) => this.dispatchEvent({ type }),
      onTouchStart: () => handleTouchStart(this.makeInputCtx()),
      onTouchMove: (event) => handleTouchMove(this.makeInputCtx(), event),
      setPointers: (pointers) => {
        this.pointers = pointers;
      },
    };
  }

  private onPointerDown(event: PointerEvent): void {
    handlePointerDown(this.makeInputCtx(), event);
  }

  private onPointerMove(event: PointerEvent): void {
    handlePointerMove(this.makeInputCtx(), event);
  }

  private onPointerUp(event: PointerEvent): void {
    handlePointerUp(this.makeInputCtx(), event);
  }

  private onWheel(event: WheelEvent): void {
    handleWheel(this.makeInputCtx(), event);
  }

  /**
   * Enable keyboard controls (arrow keys for panning).
   * Call with the element that should receive key events (typically window or canvas).
   */
  public listenToKeyEvents(element: HTMLElement | Window): void {
    if (this.keyboardDisposer) return; // Already listening
    this.keyboardDisposer = attachKeyboardPan(element, {
      enabled: () => this.enabled,
      enablePan: () => this.enablePan,
      keyPanSpeed: () => this.keyPanSpeed,
      pan: (dx, dy) => this.pan(dx, dy),
    });
  }

  /** Stop listening for keyboard events. */
  public stopListenToKeyEvents(): void {
    if (this.keyboardDisposer) {
      this.keyboardDisposer();
      this.keyboardDisposer = null;
    }
  }
}
