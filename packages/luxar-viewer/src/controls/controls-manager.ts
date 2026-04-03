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
import type { LuxarCamera } from '../scene/camera-utils';
import type { ControlType } from './types';
export type { ControlType };

export interface ControlsManagerConfig {
  autoRotate?: boolean;
  autoRotateSpeed?: number;
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

export class ControlsManager extends THREE.EventDispatcher<ControlsManagerEventMap> {
  private camera: LuxarCamera;
  private domElement: HTMLElement;
  // Current control instance
  private currentControls: LuxarOrbitControls | LuxarFlyControls | null = null;
  private currentType: ControlType = 'orbit';

  // Configuration - uses defaults from config
  private config: ControlsManagerConfig = {
    autoRotate: false,
    autoRotateSpeed: config.controls.orbit.autoRotate.speed.default,
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
  private clock = new THREE.Clock();

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
  public getControls(): LuxarOrbitControls | LuxarFlyControls | null {
    return this.currentControls;
  }

  // ---------------------------------------------------------------------------
  // Control creation
  // ---------------------------------------------------------------------------

  private createOrbitControls(): void {
    const m = config.controls.scaleMultipliers;

    // Use stored limits (from auto-frame) if available, then scale-derived,
    // then hardcoded config defaults.
    const minDist =
      this.storedDistanceLimits?.min ??
      (this.sceneScale > 0
        ? this.sceneScale * m.minDistanceFactor
        : config.controls.orbit.zoom.minDistance);
    const maxDist =
      this.storedDistanceLimits?.max ??
      (this.sceneScale > 0
        ? this.sceneScale * m.maxDistanceFactor
        : config.controls.orbit.zoom.maxDistance);

    const controls = new LuxarOrbitControls(this.camera, this.domElement, {
      enableDamping: true,
      screenSpacePanning: true,
      autoRotate: this.config.autoRotate || false,
      autoRotateSpeed: this.config.autoRotateSpeed || 0.25,
      minDistance: minDist,
      maxDistance: maxDist,
    });

    // Shift+scroll = view-axis rotation (roll)
    controls.enableViewAxisRotation();

    // Event forwarding
    controls.addEventListener('change', () => this.dispatchEvent({ type: 'change' }));
    controls.addEventListener('start', () => this.dispatchEvent({ type: 'start' }));
    controls.addEventListener('end', () => this.dispatchEvent({ type: 'end' }));

    // Target is set in restoreCameraState() after creation
    this.currentControls = controls;
    this.clock.getDelta();
  }

  private createFlyControls(): void {
    const controls = new LuxarFlyControls(this.camera, this.domElement, {
      movementSpeed: this.config.flyMovementSpeed,
      rotationSpeed: this.config.flyRotationSpeed,
      lookSpeed: this.config.flyLookSpeed,
      inertialMode: this.config.flyInertialMode,
      damping: this.config.flyDamping,
      rotationDamping: this.config.flyRotationDamping,
    });

    controls.setExternalInputManagement(true);

    controls.addEventListener('change', () => this.dispatchEvent({ type: 'change' }));
    controls.addEventListener('start', () => this.dispatchEvent({ type: 'start' }));
    controls.addEventListener('end', () => this.dispatchEvent({ type: 'end' }));

    this.currentControls = controls;
    this.clock.start();
  }

  private createOrthoControls(): void {
    const m = config.controls.scaleMultipliers;

    // Use stored zoom limits (from auto-frame) if available, else wide defaults.
    const minZoom = this.storedZoomLimits?.min ?? m.minDistanceFactor;
    const maxZoom = this.storedZoomLimits?.max ?? 1.0 / m.minDistanceFactor;

    const controls = new LuxarOrbitControls(this.camera, this.domElement, {
      enableDamping: true,
      screenSpacePanning: true,
      enableRotate: false,
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

    // Target is set in restoreCameraState() after creation
    controls.addEventListener('change', () => this.dispatchEvent({ type: 'change' }));
    controls.addEventListener('start', () => this.dispatchEvent({ type: 'start' }));
    controls.addEventListener('end', () => this.dispatchEvent({ type: 'end' }));

    this.currentControls = controls;
    this.clock.getDelta();
  }

  // ---------------------------------------------------------------------------
  // Camera state save/restore
  // ---------------------------------------------------------------------------

  private saveCameraState(): void {
    this.savedCameraPosition.copy(this.camera.position);
    this.savedCameraRotation.copy(this.camera.rotation);
    this.savedCameraUp.copy(this.camera.up);

    // Save target for all control types.
    // For orbit/ortho: use the explicit orbit target.
    // For fly: derive from camera look direction so switching to orbit/ortho
    // gets a sensible pivot point (not a stale target from a previous mode).
    if (this.currentControls instanceof LuxarOrbitControls) {
      this.savedTarget.copy(this.currentControls.target);
    } else {
      const forward = new THREE.Vector3();
      this.camera.getWorldDirection(forward);
      this.savedTarget
        .copy(this.camera.position)
        .add(forward.multiplyScalar(this.sceneScale || 10));
    }
  }

  private restoreCameraState(): void {
    if (this.currentControls instanceof LuxarOrbitControls) {
      // Set the target, then re-derive orientation from the current camera state
      // (important: the constructor initialized with target=(0,0,0), which is wrong)
      this.currentControls.target.copy(this.savedTarget);
      this.currentControls.reinitialize();
      this.currentControls.update();
    }
    // Fly controls automatically initialize from current camera state
  }

  private disposeCurrentControls(): void {
    if (this.currentControls) {
      this.currentControls.dispose();
      this.currentControls = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Update loop
  // ---------------------------------------------------------------------------

  public update(): void {
    if (!this.currentControls) return;

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

  public getAutoRotate(): boolean {
    if (this.currentControls instanceof LuxarOrbitControls) {
      return this.currentControls.autoRotate;
    }
    return false;
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

    const m = config.controls.scaleMultipliers;
    const minDist = diagonal * m.minDistanceFactor;
    const maxDist = diagonal * m.maxDistanceFactor;

    // Update active orbit/ortho controls with scale-derived distance limits,
    // but only if auto-frame hasn't set precise limits yet.
    // Once setDistanceLimits/setZoomLimits have been called (by autoFrameCamera),
    // those take precedence over the rough scale-derived approximation.
    if (this.currentControls instanceof LuxarOrbitControls && !this.storedDistanceLimits) {
      this.currentControls.minDistance = minDist;
      this.currentControls.maxDistance = maxDist;
    }

    // Update fly movement speed
    this.config.flyMovementSpeed = diagonal * m.flySpeedFactor;
    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.movementSpeed = this.config.flyMovementSpeed;
    }

    log.custom(
      LogEmoji.CONTROLS,
      Modules.CONTROLS,
      `Scale-aware controls: diagonal=${diagonal.toFixed(1)}, ` +
        `dist=[${minDist.toFixed(3)}, ${maxDist.toFixed(1)}], ` +
        `flySpeed=${this.config.flyMovementSpeed!.toFixed(2)}`
    );
  }

  public getSceneScale(): number {
    return this.sceneScale;
  }

  /**
   * Set orbit distance limits (perspective camera zoom range).
   * Called after auto-framing with the scene-fitting distance to give
   * symmetric zoom range (e.g. 100x in, 100x out).
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
   * Called after auto-framing with the scene-fitting zoom to give
   * symmetric zoom range (e.g. 100x in, 100x out).
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
    this.clock.stop();
  }
}
