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
import type { LuxarCamera } from '../scene/camera-utils';

export interface LuxarFlyControlsConfig {
  movementSpeed?: number; // Units per second
  rotationSpeed?: number; // Radians per second for arrow keys
  lookSpeed?: number; // Radians per pixel for mouse
  inertialMode?: boolean; // True for low damping, false for high damping
  damping?: number; // Translation damping: 0.9-0.99 for inertial mode
  rotationDamping?: number; // Rotation damping: 0.9-0.99 for inertial mode
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
  private moveState = {
    forward: 0,
    back: 0,
    left: 0,
    right: 0,
    up: 0,
    down: 0,
  };

  // Look state for arrow key camera rotation
  private lookState = {
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
  private activeMouseAction: 'none' | 'strafe' | 'rotate' = 'none';
  private mouseX = 0;
  private mouseY = 0;

  // References
  private camera: LuxarCamera;
  private domElement: HTMLElement;

  // Event listeners to clean up
  private boundHandlers: { [key: string]: any } = {};

  // Flag to track if we're using external input management
  private externalInputManagement: boolean = false;

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

    // Initialize orientation from current camera
    this.initializeFromCamera();

    // Save initial state so reset() has a valid baseline
    this.saveState();

    // Only add event listeners if not using external input management
    // This will be controlled by setExternalInputManagement()
    this.bindEventHandlers();
    this.addEventListeners();
  }

  private bindEventHandlers(): void {
    this.boundHandlers.keydown = this.onKeyDown.bind(this);
    this.boundHandlers.keyup = this.onKeyUp.bind(this);
    this.boundHandlers.mousedown = this.onMouseDown.bind(this);
    this.boundHandlers.mouseup = this.onMouseUp.bind(this);
    this.boundHandlers.mousemove = this.onMouseMove.bind(this);
    this.boundHandlers.wheel = this.onWheel.bind(this);
    this.boundHandlers.contextmenu = (e: Event) => e.preventDefault();
  }

  private addEventListeners(): void {
    // Only add keyboard listeners if not using external input management
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

  /**
   * Remove event listeners
   */
  private removeEventListeners(): void {
    // Remove keyboard listeners (only if currently attached)
    if (this.keyListenersAttached) {
      window.removeEventListener('keydown', this.boundHandlers.keydown);
      window.removeEventListener('keyup', this.boundHandlers.keyup);
      this.keyListenersAttached = false;
    }

    // Remove mouse listeners
    this.domElement.removeEventListener('mousedown', this.boundHandlers.mousedown);
    window.removeEventListener('mouseup', this.boundHandlers.mouseup);
    window.removeEventListener('mousemove', this.boundHandlers.mousemove);
    this.domElement.removeEventListener('wheel', this.boundHandlers.wheel);
    this.domElement.removeEventListener('contextmenu', this.boundHandlers.contextmenu);
  }

  /**
   * Set whether keyboard input is managed externally (by InputContextManager)
   * When true, the control won't register its own keyboard event listeners
   * Note: Mouse events are always handled internally for free-look functionality
   */
  public setExternalInputManagement(external: boolean): void {
    if (external !== this.externalInputManagement) {
      this.externalInputManagement = external;

      if (external) {
        // Remove only keyboard event listeners
        if (this.keyListenersAttached) {
          window.removeEventListener('keydown', this.boundHandlers.keydown);
          window.removeEventListener('keyup', this.boundHandlers.keyup);
          this.keyListenersAttached = false;
        }
      } else {
        // Add keyboard event listeners back
        if (!this.keyListenersAttached) {
          window.addEventListener('keydown', this.boundHandlers.keydown);
          window.addEventListener('keyup', this.boundHandlers.keyup);
          this.keyListenersAttached = true;
        }
      }
    }
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

  private onKeyDown(event: KeyboardEvent): void {
    if (!this.enabled) return;

    // Only prevent default for arrow keys (always used for camera look)
    if (event.key.startsWith('Arrow')) {
      event.preventDefault();
    }

    // Only prevent default for WASD if we're not typing in an input field
    const activeElement = document.activeElement;
    const isTyping =
      activeElement &&
      (activeElement.tagName === 'INPUT' ||
        activeElement.tagName === 'TEXTAREA' ||
        activeElement.getAttribute('contenteditable') === 'true');

    if (
      !isTyping &&
      ['w', 'a', 's', 'd', 'q', 'e', 'W', 'A', 'S', 'D', 'Q', 'E'].includes(event.key)
    ) {
      event.preventDefault();
    }

    // WASD for movement
    switch (event.key.toLowerCase()) {
      case 'w':
        if (event.altKey || event.metaKey) {
          this.moveState.up = 1; // Alt/Option+W for up
        } else {
          this.moveState.forward = 1; // W for forward
        }
        break;
      case 's':
        if (event.altKey || event.metaKey) {
          this.moveState.down = 1; // Alt/Option+S for down
        } else {
          this.moveState.back = 1; // S for backward
        }
        break;
      case 'a':
        this.moveState.left = 1; // A for strafe left
        break;
      case 'd':
        this.moveState.right = 1; // D for strafe right
        break;
      case 'q':
        this.lookState.roll = -1; // Q for roll left
        // Debug logging - commented out for production
        // log.info(Modules.CONTROLS, 'Q pressed - roll left', this.lookState.roll);
        break;
      case 'e':
        this.lookState.roll = 1; // E for roll right
        // Debug logging - commented out for production
        // log.info(Modules.CONTROLS, 'E pressed - roll right', this.lookState.roll);
        break;
    }

    // Speed boost with Shift key
    if (event.key === 'Shift') {
      this.speedBoost = true;
      // Debug logging - commented out for production
      // log.info(Modules.CONTROLS, 'Shift pressed - speed boost ON');
    }

    // Arrow keys for camera look direction
    switch (event.key) {
      case 'ArrowUp':
        this.startLookChange(0, -1); // Look up
        break;
      case 'ArrowDown':
        this.startLookChange(0, 1); // Look down
        break;
      case 'ArrowLeft':
        this.startLookChange(-1, 0); // Look left
        break;
      case 'ArrowRight':
        this.startLookChange(1, 0); // Look right
        break;
    }

    this.dispatchEvent({ type: 'change' });
  }

  private onKeyUp(event: KeyboardEvent): void {
    if (!this.enabled) return;

    // WASD movement release
    switch (event.key.toLowerCase()) {
      case 'w':
        this.moveState.forward = 0;
        this.moveState.up = 0; // Also clear up in case Alt was held
        break;
      case 's':
        this.moveState.back = 0;
        this.moveState.down = 0; // Also clear down in case Alt was held
        break;
      case 'a':
        this.moveState.left = 0;
        break;
      case 'd':
        this.moveState.right = 0;
        break;
      case 'q':
        this.lookState.roll = 0;
        break;
      case 'e':
        this.lookState.roll = 0;
        break;
    }

    // Release speed boost
    if (event.key === 'Shift') {
      this.speedBoost = false;
    }

    // Arrow keys for camera look release
    switch (event.key) {
      case 'ArrowUp':
      case 'ArrowDown':
        this.lookState.vertical = 0;
        break;
      case 'ArrowLeft':
      case 'ArrowRight':
        this.lookState.horizontal = 0;
        break;
    }

    this.dispatchEvent({ type: 'change' });
  }

  private onMouseDown(event: MouseEvent): void {
    if (!this.enabled) return;

    // Left button (0) = strafe, Right button (2) = rotate
    if (event.button === 0) {
      this.activeMouseAction = 'strafe';
      this.mouseX = event.clientX;
      this.mouseY = event.clientY;
      event.preventDefault();
      this.dispatchEvent({ type: 'start' });
    } else if (event.button === 2) {
      this.activeMouseAction = 'rotate';
      this.mouseX = event.clientX;
      this.mouseY = event.clientY;
      event.preventDefault();
      this.dispatchEvent({ type: 'start' });
    }
  }

  private onMouseUp(event: MouseEvent): void {
    if (!this.enabled) return;

    if (
      (event.button === 0 && this.activeMouseAction === 'strafe') ||
      (event.button === 2 && this.activeMouseAction === 'rotate')
    ) {
      this.activeMouseAction = 'none';
      this.dispatchEvent({ type: 'end' });
    }
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.enabled || this.activeMouseAction === 'none') return;

    const deltaX = event.clientX - this.mouseX;
    const deltaY = event.clientY - this.mouseY;

    this.mouseX = event.clientX;
    this.mouseY = event.clientY;

    if (this.activeMouseAction === 'rotate') {
      // Right-drag: apply angular impulse for rotation (look around)
      const torquePitch = -deltaY * this.lookSpeed * 2.5;
      const torqueYaw = -deltaX * this.lookSpeed * 2.5;

      const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(this.orientation);
      const cameraUp = new THREE.Vector3(0, 1, 0).applyQuaternion(this.orientation);

      this.angularVelocity.addScaledVector(cameraRight, torquePitch);
      this.angularVelocity.addScaledVector(cameraUp, torqueYaw);
    } else if (this.activeMouseAction === 'strafe') {
      // Left-drag: screen-space translation (strafe up/down/left/right)
      // Drag direction matches on-screen movement, consistent with pan in orbit/ortho.
      const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(this.orientation);
      const cameraUp = new THREE.Vector3(0, 1, 0).applyQuaternion(this.orientation);

      // Scale by movementSpeed for scene-appropriate sensitivity.
      // The 0.005 factor converts pixel deltas to reasonable world-space distances.
      const strafeScale = this.movementSpeed * 0.005;

      if (this.inertialMode) {
        // Inertial: add velocity impulse
        this.velocity.addScaledVector(cameraRight, -deltaX * strafeScale);
        this.velocity.addScaledVector(cameraUp, deltaY * strafeScale);
      } else {
        // Non-inertial: move directly
        this.camera.position.addScaledVector(cameraRight, -deltaX * strafeScale);
        this.camera.position.addScaledVector(cameraUp, deltaY * strafeScale);
      }
    }

    this.dispatchEvent({ type: 'change' });
  }

  /**
   * Handle mouse wheel for forward/back movement and roll.
   * - Plain scroll: forward/backward velocity impulse
   * - Shift+scroll: roll (rotate around viewing axis)
   * - Ctrl/Meta+scroll: FOV (handled by InputHandler, not intercepted here)
   */
  private onWheel(event: WheelEvent): void {
    if (!this.enabled) return;

    // Let Ctrl/Meta+scroll pass through to InputHandler for FOV control
    if (event.ctrlKey || event.metaKey) return;

    event.preventDefault();

    // Normalize deltaY across browsers (line vs pixel vs page scrolling)
    const delta = -Math.sign(event.deltaY);

    if (event.shiftKey) {
      // Shift+scroll: roll around viewing axis
      const cameraForward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.orientation);
      const rollImpulse = delta * this.rotationSpeed * 0.06;

      if (this.inertialMode) {
        this.angularVelocity.addScaledVector(cameraForward, rollImpulse);
      } else {
        // Non-inertial: apply rotation directly
        const rollQuat = new THREE.Quaternion().setFromAxisAngle(cameraForward, rollImpulse);
        this.orientation.premultiply(rollQuat);
        this.orientation.normalize();
      }
    } else {
      // Plain scroll: move forward/backward
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.orientation);
      const impulse = delta * this.movementSpeed * 0.3;

      if (this.inertialMode) {
        this.velocity.addScaledVector(forward, impulse);
      } else {
        // Non-inertial: move directly
        this.camera.position.addScaledVector(forward, impulse * 0.2);
      }
    }

    this.dispatchEvent({ type: 'change' });
  }

  /**
   * Initialize orientation from current camera quaternion
   */
  private initializeFromCamera(): void {
    // Copy the camera's current orientation
    this.orientation.copy(this.camera.quaternion);
  }

  private updateOrientation(): void {
    // Apply the orientation quaternion to the camera
    this.camera.quaternion.copy(this.orientation);
  }

  /**
   * Update controls - must be called in animation loop
   * @param delta - Time since last frame in seconds
   */
  public update(delta: number): void {
    if (!this.enabled) return;

    // Get movement vectors from orientation quaternion for consistency
    // This ensures movement is perfectly tied to the control model
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.orientation).normalize();
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.orientation).normalize();
    const up = new THREE.Vector3(0, 1, 0); // Keep world up for vertical rise/fall

    let isMoving = false;

    // Determine effective damping based on mode
    // Non-inertial mode uses high damping for immediate response
    const effectiveDamping = this.inertialMode ? this.damping : 0.5;
    const effectiveRotationDamping = this.inertialMode ? this.rotationDamping : 0.5;

    // Apply speed boost multiplier (2x speed when Shift is held)
    const speedMultiplier = this.speedBoost ? 2.0 : 1.0;

    // Always use physics-based movement (unified approach)
    // Calculate acceleration from input.
    // movementSpeed is the user-facing "speed" parameter (controlled by UI slider
    // and scale-aware system).
    const accel = new THREE.Vector3();
    accel.addScaledVector(
      forward,
      (this.moveState.forward - this.moveState.back) * this.movementSpeed * speedMultiplier
    );
    accel.addScaledVector(
      right,
      (this.moveState.right - this.moveState.left) * this.movementSpeed * speedMultiplier
    );
    accel.addScaledVector(
      up,
      (this.moveState.up - this.moveState.down) * this.movementSpeed * speedMultiplier
    );

    // Update velocity
    this.velocity.addScaledVector(accel, delta);

    // Apply damping
    this.velocity.multiplyScalar(
      Math.pow(effectiveDamping, delta * config.controls.fly.physics.dampingPower)
    );

    // Apply velocity to position
    this.camera.position.addScaledVector(this.velocity, delta);

    // Check if we're still moving (using configured threshold)
    if (this.velocity.length() < config.controls.fly.physics.velocityThreshold) {
      this.velocity.set(0, 0, 0);
    } else {
      isMoving = true;
    }

    // Handle angular velocity for rotation with arrow keys and Q/E roll
    // True airplane-like fly controls: all rotations relative to camera's local axes
    if (
      this.lookState.horizontal !== 0 ||
      this.lookState.vertical !== 0 ||
      this.lookState.roll !== 0
    ) {
      // Get camera's local axes in world space
      // These define the rotation axes for consistent airplane-like controls
      const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(this.orientation);
      const cameraUp = new THREE.Vector3(0, 1, 0).applyQuaternion(this.orientation);
      const cameraForward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.orientation);

      if (this.inertialMode) {
        // Apply angular acceleration (torque)
        const torque = new THREE.Vector3();
        // Pitch: rotate around camera's local right axis (negative for correct up/down)
        torque.addScaledVector(cameraRight, -this.lookState.vertical * this.rotationSpeed);
        // Yaw: rotate around camera's local up axis
        torque.addScaledVector(cameraUp, -this.lookState.horizontal * this.rotationSpeed);
        // Roll: rotate around camera's local forward axis
        torque.addScaledVector(cameraForward, this.lookState.roll * this.rotationSpeed);
        if (this.lookState.roll !== 0) {
          // Debug logging - commented out for production
          // log.info(Modules.CONTROLS, 'Applying roll torque:', this.lookState.roll * this.rotationSpeed);
        }

        // Add torque to world-space angular velocity
        this.angularVelocity.addScaledVector(torque, delta);
      } else {
        // Non-inertial: directly set angular velocity
        this.angularVelocity.set(0, 0, 0);
        this.angularVelocity.addScaledVector(
          cameraRight,
          -this.lookState.vertical * this.rotationSpeed
        );
        this.angularVelocity.addScaledVector(
          cameraUp,
          -this.lookState.horizontal * this.rotationSpeed
        );
        this.angularVelocity.addScaledVector(
          cameraForward,
          this.lookState.roll * this.rotationSpeed
        );
      }
    } else if (!this.inertialMode) {
      // In non-inertial mode, stop rotation when keys are released
      // High damping will handle this quickly
    }

    // Apply angular velocity to orientation
    const angularSpeed = this.angularVelocity.length();
    if (angularSpeed > config.controls.fly.physics.angularVelocityThreshold) {
      // Create rotation from angular velocity
      const angle = angularSpeed * delta;
      const axis = this.angularVelocity.clone().normalize();
      const deltaRotation = new THREE.Quaternion().setFromAxisAngle(axis, angle);

      // Apply WORLD-space delta rotation (pre-multiply)
      this.orientation.premultiply(deltaRotation);
      this.orientation.normalize();

      isMoving = true;
    }

    // Apply angular damping to world-space angular velocity
    this.angularVelocity.multiplyScalar(
      Math.pow(effectiveRotationDamping, delta * config.controls.fly.physics.dampingPower)
    );

    // Stop tiny rotations
    if (this.angularVelocity.length() < config.controls.fly.physics.angularVelocityThreshold) {
      this.angularVelocity.set(0, 0, 0);
    }

    // Update camera orientation
    this.updateOrientation();

    // Dispatch change event if we're moving to keep animation running
    if (isMoving) {
      this.dispatchEvent({ type: 'change' });
    }
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
   * Start continuous look change with arrow keys
   * @param horizontal - Horizontal look direction (-1 left, 1 right)
   * @param vertical - Vertical look direction (-1 up, 1 down)
   */
  private startLookChange(horizontal: number, vertical: number): void {
    this.lookState.horizontal = horizontal;
    this.lookState.vertical = vertical;
  }

  /**
   * Smoothly look at a target position
   * @param target - Target position to look at
   * @param smoothness - Smoothing factor (0-1, higher = smoother)
   */
  public lookAtSmooth(target: THREE.Vector3, smoothness: number = 0.9): void {
    // Calculate desired look direction
    const direction = new THREE.Vector3();
    direction.subVectors(target, this.camera.position);
    direction.normalize();

    // Create a quaternion that looks in the target direction
    const targetQuaternion = new THREE.Quaternion();
    const tempMatrix = new THREE.Matrix4();
    tempMatrix.lookAt(this.camera.position, target, new THREE.Vector3(0, 1, 0));
    targetQuaternion.setFromRotationMatrix(tempMatrix);

    // Smoothly interpolate to target orientation
    this.orientation.slerp(targetQuaternion, 1 - smoothness);

    this.updateOrientation();
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
