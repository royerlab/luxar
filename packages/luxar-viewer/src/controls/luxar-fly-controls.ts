/**
 * Custom fly controls for Luxar with quaternion-based rotation and inertial physics
 *
 * Mouse mapping (consistent with orbit/ortho modes):
 * - Left drag: strafe (screen-space translation)
 * - Right drag: rotate (look around)
 * - Scroll: move forward/backward (velocity impulse)
 * - Shift+scroll: roll (rotate around viewing axis)
 * - Ctrl+scroll: FOV change (handled by InputHandler, not here)
 *
 * Keyboard:
 * - WASD: movement (forward/back/strafe)
 * - Alt+W/S: vertical movement
 * - Arrow keys: camera rotation
 * - Q/E: roll
 * - Shift: speed boost
 *
 * Quaternion-based rotation (no gimbal lock, unlimited freedom).
 * Unified physics model with configurable damping.
 */

import * as THREE from 'three';
import { config } from '../config';
import type { LuxarCamera } from '../utils/camera-utils';
import {
  initializeFromCamera as initializeFromCameraHelper,
  updateOrientation as updateOrientationHelper,
  lookAtSmooth as lookAtSmoothHelper,
} from './luxar-fly-controls/camera-application';
import {
  handleKeyDown as handleKeyDownHelper,
  handleKeyUp as handleKeyUpHelper,
  type FlyKeyboardCtx,
  type FlyMoveState,
  type FlyLookState,
  type FlyMouseAction,
} from './luxar-fly-controls/input/keyboard';
import {
  handleMouseDown as handleMouseDownHelper,
  handleMouseUp as handleMouseUpHelper,
  handleMouseMove as handleMouseMoveHelper,
  type FlyMouseCtx,
} from './luxar-fly-controls/input/mouse';
import {
  handleWheel as handleWheelHelper,
  type FlyWheelCtx,
} from './luxar-fly-controls/input/wheel';
import {
  integrateTranslation,
  integrateRotation,
  type FlyPhysicsCtx,
} from './luxar-fly-controls/physics';

export interface LuxarFlyControlsConfig {
  movementSpeed?: number; // Units per second
  rotationSpeed?: number; // Radians per second for arrow keys
  lookSpeed?: number; // Radians per pixel for mouse
  inertialMode?: boolean; // True for low damping, false for high damping
  damping?: number; // Translation damping: 0.9-0.99 for inertial mode
  rotationDamping?: number; // Rotation damping: 0.9-0.99 for inertial mode
  // When true, keyboard listeners are not attached — the caller is responsible
  // for forwarding key events via handleKeyDown/handleKeyUp. Mouse events are
  // always handled internally for free-look functionality.
  externalInputManagement?: boolean;
}

export class LuxarFlyControls extends THREE.EventDispatcher<{
  change: {};
  start: {};
  end: {};
}> {
  public enabled: boolean = true;

  // Configuration
  public movementSpeed: number = config.controls.fly.movement.speed.default;
  public rotationSpeed: number = config.controls.fly.rotation.speed.default;
  public lookSpeed: number = config.controls.fly.look.mouseSpeed.default;
  public inertialMode: boolean = config.controls.fly.inertialMode.default;
  public damping: number = config.controls.fly.movement.damping.default;
  public rotationDamping: number = config.controls.fly.rotation.damping.default;

  // Movement state
  private moveState: FlyMoveState = {
    forward: 0,
    back: 0,
    left: 0,
    right: 0,
    up: 0,
    down: 0,
  };

  // Look state for arrow key camera rotation
  private lookState: FlyLookState = {
    horizontal: 0, // -1 for left, 1 for right
    vertical: 0, // -1 for up, 1 for down
    roll: 0, // -1 for Q (roll left), 1 for E (roll right)
  };

  // Speed boost state
  private speedBoost: boolean = false;

  // Track whether keyboard listeners are currently attached
  private keyListenersAttached = false;

  // Velocity vectors for physics
  private velocity = new THREE.Vector3(0, 0, 0); // Translational velocity in world space
  private angularVelocity = new THREE.Vector3(0, 0, 0); // Angular velocity in world space (rad/s)

  // Quaternion-based orientation
  private orientation = new THREE.Quaternion();

  // Saved state for saveState/reset
  private savedPosition = new THREE.Vector3();
  private savedOrientation = new THREE.Quaternion();

  // Mouse state — tracks which button is down for different drag actions
  // Left drag = strafe (translate), Right drag = rotate (look)
  private activeMouseAction: FlyMouseAction = 'none';
  private mouseX = 0;
  private mouseY = 0;

  // References
  private camera: LuxarCamera;
  private domElement: HTMLElement;

  // Bound DOM event handlers, captured for clean removal in dispose().
  private boundHandlers: {
    keydown: (e: KeyboardEvent) => void;
    keyup: (e: KeyboardEvent) => void;
    mousedown: (e: MouseEvent) => void;
    mouseup: (e: MouseEvent) => void;
    mousemove: (e: MouseEvent) => void;
    wheel: (e: WheelEvent) => void;
    contextmenu: (e: Event) => void;
  };

  // When true, keyboard input is forwarded by the caller (e.g.
  // InputContextManager) rather than registered on `window`. Set at
  // construction; not mutable afterward.
  private readonly externalInputManagement: boolean;

  constructor(camera: LuxarCamera, domElement: HTMLElement, config?: LuxarFlyControlsConfig) {
    super();

    this.camera = camera;
    this.domElement = domElement;

    // Apply configuration
    if (config) {
      this.movementSpeed = config.movementSpeed ?? this.movementSpeed;
      this.rotationSpeed = config.rotationSpeed ?? this.rotationSpeed;
      this.lookSpeed = config.lookSpeed ?? this.lookSpeed;
      this.inertialMode = config.inertialMode ?? this.inertialMode;
      this.damping = config.damping ?? this.damping;
      this.rotationDamping = config.rotationDamping ?? this.rotationDamping;
    }
    this.externalInputManagement = config?.externalInputManagement ?? false;

    // Initialize orientation from current camera
    this.initializeFromCamera();

    // Save initial state so reset() has a valid baseline
    this.saveState();

    this.boundHandlers = this.bindEventHandlers();
    this.addEventListeners();
  }

  private bindEventHandlers(): typeof this.boundHandlers {
    return {
      keydown: this.onKeyDown.bind(this),
      keyup: this.onKeyUp.bind(this),
      mousedown: this.onMouseDown.bind(this),
      mouseup: this.onMouseUp.bind(this),
      mousemove: this.onMouseMove.bind(this),
      wheel: this.onWheel.bind(this),
      contextmenu: (e: Event) => e.preventDefault(),
    };
  }

  private addEventListeners(): void {
    if (!this.externalInputManagement) {
      window.addEventListener('keydown', this.boundHandlers.keydown);
      window.addEventListener('keyup', this.boundHandlers.keyup);
      this.keyListenersAttached = true;
    }

    // Mouse events are always handled internally
    this.domElement.addEventListener('mousedown', this.boundHandlers.mousedown);
    window.addEventListener('mouseup', this.boundHandlers.mouseup);
    window.addEventListener('mousemove', this.boundHandlers.mousemove);
    this.domElement.addEventListener('wheel', this.boundHandlers.wheel, { passive: false });
    this.domElement.addEventListener('contextmenu', this.boundHandlers.contextmenu);
  }

  private removeEventListeners(): void {
    if (this.keyListenersAttached) {
      window.removeEventListener('keydown', this.boundHandlers.keydown);
      window.removeEventListener('keyup', this.boundHandlers.keyup);
      this.keyListenersAttached = false;
    }

    this.domElement.removeEventListener('mousedown', this.boundHandlers.mousedown);
    window.removeEventListener('mouseup', this.boundHandlers.mouseup);
    window.removeEventListener('mousemove', this.boundHandlers.mousemove);
    this.domElement.removeEventListener('wheel', this.boundHandlers.wheel);
    this.domElement.removeEventListener('contextmenu', this.boundHandlers.contextmenu);
  }

  /**
   * Public method to handle key down events (for external input management)
   */
  public handleKeyDown(event: KeyboardEvent): void {
    if (!this.enabled) return;
    this.onKeyDown(event);
  }

  /**
   * Public method to handle key up events (for external input management)
   */
  public handleKeyUp(event: KeyboardEvent): void {
    if (!this.enabled) return;
    this.onKeyUp(event);
  }

  // ---------------------------------------------------------------------------
  // Input handling (delegated to luxar-fly-controls/input/*)
  // ---------------------------------------------------------------------------

  private makeKeyboardCtx(): FlyKeyboardCtx {
    return {
      enabled: this.enabled,
      moveState: this.moveState,
      lookState: this.lookState,
      setSpeedBoost: (v) => {
        this.speedBoost = v;
      },
      dispatch: (type) => this.dispatchEvent({ type }),
    };
  }

  private makeMouseCtx(): FlyMouseCtx {
    return {
      enabled: this.enabled,
      inertialMode: this.inertialMode,
      lookSpeed: this.lookSpeed,
      movementSpeed: this.movementSpeed,
      camera: this.camera,
      orientation: this.orientation,
      velocity: this.velocity,
      angularVelocity: this.angularVelocity,
      getActiveMouseAction: () => this.activeMouseAction,
      setActiveMouseAction: (v) => {
        this.activeMouseAction = v;
      },
      getMouseX: () => this.mouseX,
      setMouseX: (v) => {
        this.mouseX = v;
      },
      getMouseY: () => this.mouseY,
      setMouseY: (v) => {
        this.mouseY = v;
      },
      dispatch: (type) => this.dispatchEvent({ type }),
    };
  }

  private makeWheelCtx(): FlyWheelCtx {
    return {
      enabled: this.enabled,
      inertialMode: this.inertialMode,
      movementSpeed: this.movementSpeed,
      rotationSpeed: this.rotationSpeed,
      camera: this.camera,
      orientation: this.orientation,
      velocity: this.velocity,
      angularVelocity: this.angularVelocity,
      dispatch: (type) => this.dispatchEvent({ type }),
    };
  }

  private onKeyDown(event: KeyboardEvent): void {
    handleKeyDownHelper(this.makeKeyboardCtx(), event);
  }

  private onKeyUp(event: KeyboardEvent): void {
    handleKeyUpHelper(this.makeKeyboardCtx(), event);
  }

  private onMouseDown(event: MouseEvent): void {
    handleMouseDownHelper(this.makeMouseCtx(), event);
  }

  private onMouseUp(event: MouseEvent): void {
    handleMouseUpHelper(this.makeMouseCtx(), event);
  }

  private onMouseMove(event: MouseEvent): void {
    handleMouseMoveHelper(this.makeMouseCtx(), event);
  }

  private onWheel(event: WheelEvent): void {
    handleWheelHelper(this.makeWheelCtx(), event);
  }

  private initializeFromCamera(): void {
    initializeFromCameraHelper(this.camera, this.orientation);
  }

  private updateOrientation(): void {
    updateOrientationHelper(this.camera, this.orientation);
  }

  /**
   * Update controls - must be called in animation loop
   * @param delta - Time since last frame in seconds
   */
  public update(delta: number): void {
    if (!this.enabled) return;

    const ctx = this.makePhysicsCtx();
    const translated = integrateTranslation(ctx, delta);
    const rotated = integrateRotation(ctx, delta);

    // Update camera orientation
    this.updateOrientation();

    // Dispatch change event if we're moving to keep animation running
    if (translated || rotated) {
      this.dispatchEvent({ type: 'change' });
    }
  }

  private makePhysicsCtx(): FlyPhysicsCtx {
    return {
      camera: this.camera,
      orientation: this.orientation,
      velocity: this.velocity,
      angularVelocity: this.angularVelocity,
      moveState: this.moveState,
      lookState: this.lookState,
      inertialMode: this.inertialMode,
      damping: this.damping,
      rotationDamping: this.rotationDamping,
      movementSpeed: this.movementSpeed,
      rotationSpeed: this.rotationSpeed,
      speedBoost: this.speedBoost,
    };
  }

  /**
   * Set movement mode
   * @param inertial - True for low damping (momentum), false for high damping (immediate)
   */
  public setInertialMode(inertial: boolean): void {
    this.inertialMode = inertial;
    // Both modes now use physics, just with different damping
    // No need to clear velocity as it will quickly dampen out
  }

  /**
   * Smoothly look at a target position
   * @param target - Target position to look at
   * @param smoothness - Smoothing factor (0-1, higher = smoother)
   */
  public lookAtSmooth(target: THREE.Vector3, smoothness: number = 0.9): void {
    lookAtSmoothHelper(this.camera, this.orientation, target, smoothness);
  }

  /**
   * Save current state (for reset functionality)
   */
  public saveState(): void {
    this.savedPosition.copy(this.camera.position);
    this.savedOrientation.copy(this.orientation);
  }

  /**
   * Reset to saved state
   */
  public reset(): void {
    // Restore saved position and orientation
    this.camera.position.copy(this.savedPosition);
    this.orientation.copy(this.savedOrientation);

    // Reset velocities
    this.velocity.set(0, 0, 0);
    this.angularVelocity.set(0, 0, 0);

    // Reset movement state
    this.moveState.forward = 0;
    this.moveState.back = 0;
    this.moveState.left = 0;
    this.moveState.right = 0;
    this.moveState.up = 0;
    this.moveState.down = 0;

    // Reset look state
    this.lookState.horizontal = 0;
    this.lookState.vertical = 0;
    this.lookState.roll = 0;

    // Reset speed boost and mouse state
    this.speedBoost = false;
    this.activeMouseAction = 'none';

    this.updateOrientation();
  }

  /**
   * Dispose of controls and clean up event listeners
   */
  public dispose(): void {
    this.removeEventListeners();
  }
}
