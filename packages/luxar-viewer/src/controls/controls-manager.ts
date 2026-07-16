/**
 * Manages switching between different camera control types.
 *
 * Provides unified interface for camera control modes:
 * - Orbit: Quaternion-based orbit (no gimbal lock) with smooth damping
 * - Fly: First-person WASD movement for exploring inside datasets
 * - Ortho: Orthographic pan + zoom (2D viewing mode)
 *
 * Handles:
 * - Seamless switching between control types (V key)
 * - Camera state preservation during switches
 * - Event listener lifecycle management
 * - Configuration persistence and updates
 * - Auto-rotation support (orbit only)
 * - Physics-based movement (fly mode)
 */

import * as THREE from 'three';
import { LuxarOrbitControls } from './luxar-orbit-controls';
import { LuxarFlyControls } from './luxar-fly-controls';
import { config } from '../config';
import { log, Modules, LogEmoji } from '../utils/log';
import { EventGroup } from '../utils/cross-layer/event-group';
import type { LuxarCamera } from '../utils/camera-utils';
import type { ControlType } from './types';
import { isMacPlatform } from '../utils/platform';
import {
  createOrbitControls,
  createFlyControls,
  createOrthoControls,
  naturalDragButtonMap,
  type ControlsCreationCtx,
} from './controls-manager/factories';
import {
  saveCameraState,
  restoreCameraState,
  type CameraStateCtx,
} from './controls-manager/camera-state';
import {
  attachControlEventForwarders,
  type ControlEventDispatcher,
} from './controls-manager/event-forwarders';
import { deriveScaleLimits } from './controls-manager/scene-scale';
export type { ControlType };

export interface ControlsManagerConfig {
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  /**
   * Swap LEFT ↔ RIGHT mouse-button mapping in orbit (3D) mode. When true,
   * one-finger drag rotates and right-drag pans (touchpad ergonomics);
   * when false, the classic CAD/Blender mapping (left-drag pans, right-
   * drag rotates) stays in place. Ignored in ortho and fly modes.
   */
  naturalDrag?: boolean;
  /**
   * Orbit feel parameters. Also applied to ortho mode — it is the same
   * LuxarOrbitControls class, and the live setters mutate whichever
   * instance is current, so both modes share one zoom/damping feel.
   */
  orbitZoomSpeed?: number;
  orbitDampingFactor?: number;
  flyMovementSpeed?: number;
  flyRotationSpeed?: number;
  flyLookSpeed?: number;
  flyInertialMode?: boolean;
  flyDamping?: number;
  flyRotationDamping?: number;
}

interface ControlsManagerEventMap {
  change: { controlType?: ControlType };
  start: {};
  end: {};
}

type ActiveControls = LuxarOrbitControls | LuxarFlyControls;

export class ControlsManager extends THREE.EventDispatcher<ControlsManagerEventMap> {
  private camera: LuxarCamera;
  private domElement: HTMLElement;
  // Current control instance
  private currentControls: ActiveControls | null = null;
  /**
   * Cleanup group for the change/start/end event-forwarders attached to
   * the active controls instance. Rebuilt on every switch so disposing
   * the previous group detaches the old listeners atomically.
   */
  private controlEvents: EventGroup = new EventGroup();
  private currentType: ControlType = 'orbit';

  // Configuration - uses defaults from config
  private config: ControlsManagerConfig = {
    autoRotate: false,
    autoRotateSpeed: config.controls.orbit.autoRotate.speed.default,
    // Default to true on macOS; rendering-controls persistence overrides
    // this with any stored user choice as soon as settings load.
    naturalDrag: isMacPlatform(),
    orbitZoomSpeed: config.controls.orbit.zoom.speed.default,
    orbitDampingFactor: config.controls.orbit.damping.factor.default,
    flyMovementSpeed: config.controls.fly.movement.speed.default,
    flyRotationSpeed: config.controls.fly.rotation.speed.default,
    flyLookSpeed: config.controls.fly.look.mouseSpeed.default,
    flyInertialMode: config.controls.fly.inertialMode.default,
    flyDamping: config.controls.fly.movement.damping.default,
    flyRotationDamping: config.controls.fly.rotation.damping.default,
  };

  // Scene scale (bounding box diagonal) for scale-aware control parameters.
  // 0 = not yet set, use hardcoded config defaults.
  private sceneScale: number = 0;

  // Stored zoom/distance limits (set by auto-frame, persisted across mode switches).
  // null = not yet set, use defaults from config.
  private storedDistanceLimits: { min: number; max: number } | null = null;
  private storedZoomLimits: { min: number; max: number } | null = null;

  // Saved camera state for switching
  private savedCameraPosition = new THREE.Vector3();
  private savedCameraRotation = new THREE.Euler();
  private savedTarget = new THREE.Vector3();
  private savedCameraUp = new THREE.Vector3(0, 1, 0);

  // Delta time tracking for fly controls
  private clock = new THREE.Timer();

  /**
   * Create controls manager for camera interaction.
   *
   * @param camera - Camera to control (perspective or orthographic)
   * @param domElement - DOM element for mouse/touch input (typically canvas)
   * @param _scene - Optional scene reference
   */
  constructor(camera: LuxarCamera, domElement: HTMLElement, _scene?: THREE.Scene) {
    super();

    this.camera = camera;
    this.domElement = domElement;

    // Page-visibility integration: pauses the timer when the tab is hidden.
    // Clock did this implicitly; Timer requires opt-in (three r174+).
    if (typeof document !== 'undefined') {
      this.clock.connect(document);
    }

    // Initialize with orbit controls by default
    this.setControlType('orbit');
  }

  /**
   * Update internal camera reference (e.g., when swapping between perspective and orthographic).
   * Must be called BEFORE setControlType when the camera object itself changes.
   */
  public setCamera(camera: LuxarCamera): void {
    this.camera = camera;
  }

  /**
   * Switch to different camera control type.
   */
  public setControlType(type: ControlType): void {
    if (type === this.currentType && this.currentControls) {
      return;
    }

    // Save current camera state
    this.saveCameraState();

    // Dispose of current controls
    this.disposeCurrentControls();

    // Create new controls
    switch (type) {
      case 'orbit':
        this.createOrbitControls();
        break;
      case 'fly':
        this.createFlyControls();
        break;
      case 'ortho':
        this.createOrthoControls();
        break;
    }

    this.currentType = type;

    // Restore camera state
    this.restoreCameraState();

    // Emit change event
    this.dispatchEvent({ type: 'change', controlType: type });

    log.custom(LogEmoji.CONTROLS, Modules.CONTROLS, `Switched to ${type} controls`);
  }

  /** Get currently active control type. */
  public getControlType(): ControlType {
    return this.currentType;
  }

  /** Get current controls instance. */
  public getControls(): ActiveControls | null {
    return this.currentControls;
  }

  // ---------------------------------------------------------------------------
  // Control creation (delegated to controls-manager/factories.ts)
  // ---------------------------------------------------------------------------

  private makeCreationCtx(): ControlsCreationCtx {
    return {
      camera: this.camera,
      domElement: this.domElement,
      config: this.config,
      sceneScale: this.sceneScale,
      storedDistanceLimits: this.storedDistanceLimits,
      storedZoomLimits: this.storedZoomLimits,
    };
  }

  private createOrbitControls(): void {
    // Target is set in restoreCameraState() after creation.
    this.currentControls = createOrbitControls(this.makeCreationCtx());
    this.attachControlEventForwarders(this.currentControls);
    this.clock.update(); // Reset baseline; next getDelta() reads from now.
  }

  private createFlyControls(): void {
    this.currentControls = createFlyControls(this.makeCreationCtx());
    this.attachControlEventForwarders(this.currentControls);
    this.clock.update(); // Reset baseline; next getDelta() reads from now.
  }

  private createOrthoControls(): void {
    // Target is set in restoreCameraState() after creation.
    this.currentControls = createOrthoControls(this.makeCreationCtx());
    this.attachControlEventForwarders(this.currentControls);
    this.clock.update(); // Reset baseline; next getDelta() reads from now.
  }

  // ---------------------------------------------------------------------------
  // Camera state save/restore (delegated to controls-manager/camera-state.ts)
  // ---------------------------------------------------------------------------

  private makeCameraStateCtx(): CameraStateCtx {
    return {
      camera: this.camera,
      currentControls: this.currentControls,
      sceneScale: this.sceneScale,
      savedCameraPosition: this.savedCameraPosition,
      savedCameraRotation: this.savedCameraRotation,
      savedCameraUp: this.savedCameraUp,
      savedTarget: this.savedTarget,
    };
  }

  private saveCameraState(): void {
    saveCameraState(this.makeCameraStateCtx());
  }

  private restoreCameraState(): void {
    restoreCameraState(this.makeCameraStateCtx());
  }

  private attachControlEventForwarders(controls: ControlEventDispatcher): void {
    attachControlEventForwarders(
      controls,
      (type) => this.dispatchEvent({ type }),
      this.controlEvents
    );
  }

  private disposeCurrentControls(): void {
    if (this.currentControls) {
      this.controlEvents.dispose();
      this.controlEvents = new EventGroup();
      this.currentControls.dispose();
      this.currentControls = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Update loop
  // ---------------------------------------------------------------------------

  public update(): void {
    if (!this.currentControls) return;

    this.clock.update();
    const delta = this.clock.getDelta();
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.update(delta);
    } else if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.update(delta);
    }
  }

  // ---------------------------------------------------------------------------
  // Configuration setters
  // ---------------------------------------------------------------------------

  public setEnabled(enabled: boolean): void {
    if (this.currentControls) {
      this.currentControls.enabled = enabled;
    }
  }

  public setAutoRotate(enabled: boolean): void {
    this.config.autoRotate = enabled;
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.autoRotate = enabled;
    }
  }

  public setAutoRotateSpeed(speed: number): void {
    this.config.autoRotateSpeed = speed;
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.autoRotateSpeed = speed;
    }
  }

  /**
   * Toggle the orbit-mode LEFT ↔ RIGHT mouse-button mapping. Updates stored
   * config and, only when the active control is orbit (3D), mutates the
   * live mouseButtons in place so the change applies immediately without a
   * mode switch. Ortho and fly are intentionally ignored: ortho uses its
   * own RIGHT=null mapping, fly doesn't use mouseButtons at all.
   */
  public setNaturalDrag(enabled: boolean): void {
    this.config.naturalDrag = enabled;
    if (this.currentType !== 'orbit') return;
    if (!(this.currentControls instanceof LuxarOrbitControls)) return;
    this.currentControls.mouseButtons = naturalDragButtonMap(enabled);
  }

  public getNaturalDrag(): boolean {
    return this.config.naturalDrag ?? false;
  }

  public getAutoRotate(): boolean {
    if (this.currentControls instanceof LuxarOrbitControls) {
      return this.currentControls.autoRotate;
    }
    return false;
  }

  /**
   * Orbit wheel-zoom speed. `zoomSpeed` is read live per wheel event, so
   * mutating the field applies immediately. Also affects ortho mode (same
   * LuxarOrbitControls class) — one shared zoom feel across both.
   */
  public setOrbitZoomSpeed(speed: number): void {
    this.config.orbitZoomSpeed = speed;
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.zoomSpeed = speed;
    }
  }

  /**
   * Orbit damping factor (camera "weight"). `dampingFactor` is read live
   * per frame; applies immediately. Shared with ortho like the zoom speed.
   */
  public setOrbitDampingFactor(factor: number): void {
    this.config.orbitDampingFactor = factor;
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.dampingFactor = factor;
    }
  }

  public setFlyLookSpeed(speed: number): void {
    this.config.flyLookSpeed = speed;
    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.lookSpeed = speed;
    }
  }

  public setFlyMovementSpeed(speed: number): void {
    this.config.flyMovementSpeed = speed;
    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.movementSpeed = speed;
    }
  }

  public setFlyRotationSpeed(speed: number): void {
    this.config.flyRotationSpeed = speed;
    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.rotationSpeed = speed;
    }
  }

  public setFlyInertialMode(inertial: boolean): void {
    this.config.flyInertialMode = inertial;
    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.setInertialMode(inertial);
    }
  }

  public setFlyDamping(damping: number): void {
    this.config.flyDamping = damping;
    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.damping = damping;
    }
  }

  public setFlyRotationDamping(damping: number): void {
    this.config.flyRotationDamping = damping;
    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.rotationDamping = damping;
    }
  }

  public reset(): void {
    if (this.currentControls) {
      this.currentControls.reset();
    }
  }

  public saveState(): void {
    if (this.currentControls) {
      this.currentControls.saveState();
    }
  }

  public setEnableZoom(enabled: boolean): void {
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.enableZoom = enabled;
    }
  }

  /**
   * Set the orbit target without triggering an update.
   * Use this when you also need to reinitialize() afterward (e.g., after auto-frame).
   * For fly controls, instantly orients the camera toward the target.
   */
  public setTarget(target: THREE.Vector3): void {
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.target.copy(target);
    } else if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.lookAtSmooth(target, 0);
    }
  }

  /**
   * Re-derive internal orbit state (distance, orientation) from the current camera
   * position and target. Must be called after externally setting camera.position
   * to avoid the next update() snapping the camera back to the old distance.
   */
  public reinitialize(): void {
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.reinitialize();
    }
  }

  public lookAt(target: THREE.Vector3, smooth: boolean = true): void {
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.target.copy(target);
      this.currentControls.update();
    } else if (this.currentControls instanceof LuxarFlyControls) {
      if (smooth) {
        this.currentControls.lookAtSmooth(target, 0.9);
      } else {
        this.currentControls.lookAtSmooth(target, 0);
      }
    }
  }

  public getFocusTarget(): THREE.Vector3 {
    if (this.currentControls instanceof LuxarOrbitControls) {
      return this.currentControls.target.clone();
    } else {
      // For fly controls, return a point in front of the camera
      const forward = new THREE.Vector3();
      this.camera.getWorldDirection(forward);
      return this.camera.position.clone().add(forward.multiplyScalar(this.sceneScale || 10));
    }
  }

  public getFlyControls(): LuxarFlyControls | null {
    if (this.currentControls instanceof LuxarFlyControls) {
      return this.currentControls;
    }
    return null;
  }

  public getFlyConfig(): {
    inertialMode: boolean;
    damping: number;
    rotationDamping: number;
    movementSpeed: number;
    rotationSpeed: number;
  } {
    return {
      inertialMode: this.config.flyInertialMode ?? config.controls.fly.inertialMode.default,
      damping: this.config.flyDamping ?? config.controls.fly.movement.damping.default,
      rotationDamping:
        this.config.flyRotationDamping ?? config.controls.fly.rotation.damping.default,
      movementSpeed: this.config.flyMovementSpeed ?? config.controls.fly.movement.speed.default,
      rotationSpeed: this.config.flyRotationSpeed ?? config.controls.fly.rotation.speed.default,
    };
  }

  public setSceneScale(diagonal: number): void {
    if (diagonal <= 0) return;
    if (this.sceneScale > 0 && Math.abs(diagonal - this.sceneScale) < 0.001 * this.sceneScale)
      return;
    this.sceneScale = diagonal;

    const { minDist, maxDist, flySpeed } = deriveScaleLimits(diagonal);

    // Update active orbit/ortho controls with scale-derived distance limits,
    // but only if auto-frame hasn't set precise limits yet.
    // Once setDistanceLimits/setZoomLimits have been called (by autoFrameCamera),
    // those take precedence over the rough scale-derived approximation.
    if (this.currentControls instanceof LuxarOrbitControls && !this.storedDistanceLimits) {
      this.currentControls.minDistance = minDist;
      this.currentControls.maxDistance = maxDist;
    }

    // Update fly movement speed
    this.config.flyMovementSpeed = flySpeed;
    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.movementSpeed = flySpeed;
    }

    log.custom(
      LogEmoji.CONTROLS,
      Modules.CONTROLS,
      `Scale-aware controls: diagonal=${diagonal.toFixed(1)}, ` +
        `dist=[${minDist.toFixed(3)}, ${maxDist.toFixed(1)}], ` +
        `flySpeed=${flySpeed.toFixed(2)}`
    );
  }

  public getSceneScale(): number {
    return this.sceneScale;
  }

  /**
   * Set orbit distance limits (perspective camera zoom range).
   * Called after auto-framing with the scene-fitting distance to give an
   * asymmetric zoom range (ZOOM_IN_FACTOR in, ZOOM_OUT_FACTOR out).
   * Persisted across mode switches so limits survive control recreation.
   */
  public setDistanceLimits(min: number, max: number): void {
    this.storedDistanceLimits = { min, max };
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.minDistance = min;
      this.currentControls.maxDistance = max;
    }
  }

  /**
   * Set ortho zoom limits (orthographic camera zoom range).
   * Called after auto-framing with the scene-fitting zoom to give an
   * asymmetric zoom range (ZOOM_IN_FACTOR in, ZOOM_OUT_FACTOR out).
   * Persisted across mode switches so limits survive control recreation.
   */
  public setZoomLimits(min: number, max: number): void {
    this.storedZoomLimits = { min, max };
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.currentControls.minZoom = min;
      this.currentControls.maxZoom = max;
    }
  }

  public dispose(): void {
    this.disposeCurrentControls();
    this.clock.dispose(); // Disconnects from document visibility events.
  }
}
