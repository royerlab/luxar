/**
 * Custom fly controls for Luxar with quaternion-based rotation and inertial physics
 *
 * Features:
 * - WASD keys for movement (forward/back/strafe)
 * - Alt+W/S for vertical movement
 * - Arrow keys and mouse drag for camera rotation
 * - Quaternion-based rotation (no gimbal lock, unlimited freedom)
 * - Unified physics model with configurable damping
 * - Smooth inertial physics for both translation and rotation
 */

import * as THREE from 'three';
import { CONTROL_CONFIG } from './control-config';

export interface LuxarFlyControlsConfig {
  movementSpeed?: number; // Units per second
  rotationSpeed?: number; // Radians per second for arrow keys
  lookSpeed?: number; // Radians per pixel for mouse
  inertialMode?: boolean; // True for low damping, false for high damping
  damping?: number; // Translation damping: 0.9-0.99 for inertial mode
  rotationDamping?: number; // Rotation damping: 0.9-0.99 for inertial mode
  acceleration?: number; // Acceleration rate for inertial mode
}

export class LuxarFlyControls extends THREE.EventDispatcher<{
  change: {};
  start: {};
  end: {};
}> {
  public enabled: boolean = true;

  // Configuration
  public movementSpeed: number = CONTROL_CONFIG.fly.movement.speed.default;
  public rotationSpeed: number = CONTROL_CONFIG.fly.rotation.speed.default;
  public lookSpeed: number = CONTROL_CONFIG.fly.look.mouseSpeed.default;
  public inertialMode: boolean = CONTROL_CONFIG.fly.inertialMode.default;
  public damping: number = CONTROL_CONFIG.fly.movement.damping.default;
  public rotationDamping: number = CONTROL_CONFIG.fly.rotation.damping.default;
  public acceleration: number = CONTROL_CONFIG.fly.movement.acceleration.default;

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

  // Velocity vectors for physics
  private velocity = new THREE.Vector3(0, 0, 0); // Translational velocity in world space
  private angularVelocity = new THREE.Vector3(0, 0, 0); // Angular velocity in world space (rad/s)

  // Quaternion-based orientation
  private orientation = new THREE.Quaternion();

  // Mouse state for looking
  private isMouseDown = false;
  private mouseX = 0;
  private mouseY = 0;

  // References
  private camera: THREE.PerspectiveCamera;
  private domElement: HTMLElement;

  // Event listeners to clean up
  private boundHandlers: { [key: string]: any } = {};

  // Flag to track if we're using external input management
  private externalInputManagement: boolean = false;

  constructor(
    camera: THREE.PerspectiveCamera,
    domElement: HTMLElement,
    config?: LuxarFlyControlsConfig
  ) {
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
      this.acceleration = config.acceleration ?? this.acceleration;
    }

    // Initialize orientation from current camera
    this.initializeFromCamera();

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
    this.boundHandlers.contextmenu = (e: Event) => e.preventDefault();
  }

  private addEventListeners(): void {
    // Only add keyboard listeners if not using external input management
    if (!this.externalInputManagement) {
      window.addEventListener('keydown', this.boundHandlers.keydown);
      window.addEventListener('keyup', this.boundHandlers.keyup);
    }

    // Mouse events are always handled internally
    // These are critical for fly controls' free-look feature
    this.domElement.addEventListener('mousedown', this.boundHandlers.mousedown);
    window.addEventListener('mouseup', this.boundHandlers.mouseup);
    window.addEventListener('mousemove', this.boundHandlers.mousemove);
    this.domElement.addEventListener('contextmenu', this.boundHandlers.contextmenu);
  }

  /**
   * Remove event listeners
   */
  private removeEventListeners(): void {
    // Remove keyboard listeners
    window.removeEventListener('keydown', this.boundHandlers.keydown);
    window.removeEventListener('keyup', this.boundHandlers.keyup);

    // Remove mouse listeners
    this.domElement.removeEventListener('mousedown', this.boundHandlers.mousedown);
    window.removeEventListener('mouseup', this.boundHandlers.mouseup);
    window.removeEventListener('mousemove', this.boundHandlers.mousemove);
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
        window.removeEventListener('keydown', this.boundHandlers.keydown);
        window.removeEventListener('keyup', this.boundHandlers.keyup);
      } else {
        // Add keyboard event listeners back
        window.addEventListener('keydown', this.boundHandlers.keydown);
        window.addEventListener('keyup', this.boundHandlers.keyup);
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
        console.log('🔧 [Luxar] Q pressed - roll left', this.lookState.roll);
        break;
      case 'e':
        this.lookState.roll = 1; // E for roll right
        console.log('🔧 [Luxar] E pressed - roll right', this.lookState.roll);
        break;
    }

    // Speed boost with Shift key
    if (event.key === 'Shift') {
      this.speedBoost = true;
      console.log('🔧 [Luxar] Shift pressed - speed boost ON');
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
        this.stopLookChange();
        break;
      case 'ArrowLeft':
      case 'ArrowRight':
        this.stopLookChange();
        break;
    }

    this.dispatchEvent({ type: 'change' });
  }

  private onMouseDown(event: MouseEvent): void {
    if (!this.enabled) return;

    // Only respond to left mouse button
    if (event.button === 0) {
      this.isMouseDown = true;
      this.mouseX = event.clientX;
      this.mouseY = event.clientY;

      // Prevent text selection
      event.preventDefault();

      this.dispatchEvent({ type: 'start' });
    }
  }

  private onMouseUp(event: MouseEvent): void {
    if (!this.enabled) return;

    if (event.button === 0) {
      this.isMouseDown = false;

      this.dispatchEvent({ type: 'end' });
    }
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.enabled || !this.isMouseDown) return;

    const deltaX = event.clientX - this.mouseX;
    const deltaY = event.clientY - this.mouseY;

    this.mouseX = event.clientX;
    this.mouseY = event.clientY;

    // Apply angular impulse based on current camera orientation
    // Note: deltaY is positive when moving down, negative when moving up
    // We want to pitch down (positive rotation) when dragging down
    const torquePitch = -deltaY * this.lookSpeed * 10; // Pitch (up/down) - inverted for natural feel
    const torqueYaw = -deltaX * this.lookSpeed * 10; // Yaw (left/right)

    // Get camera's local axes for consistent airplane-like controls
    const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(this.orientation);
    const cameraUp = new THREE.Vector3(0, 1, 0).applyQuaternion(this.orientation);

    // Add impulse to world-space angular velocity
    this.angularVelocity.addScaledVector(cameraRight, torquePitch);
    this.angularVelocity.addScaledVector(cameraUp, torqueYaw);

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
    // Calculate acceleration from input
    const accel = new THREE.Vector3();
    accel.addScaledVector(
      forward,
      (this.moveState.forward - this.moveState.back) * this.acceleration * speedMultiplier
    );
    accel.addScaledVector(
      right,
      (this.moveState.right - this.moveState.left) * this.acceleration * speedMultiplier
    );
    accel.addScaledVector(
      up,
      (this.moveState.up - this.moveState.down) * this.acceleration * speedMultiplier
    );

    // Update velocity
    this.velocity.addScaledVector(accel, delta);

    // Apply damping
    this.velocity.multiplyScalar(
      Math.pow(effectiveDamping, delta * CONTROL_CONFIG.fly.physics.dampingPower)
    );

    // Apply velocity to position
    this.camera.position.addScaledVector(this.velocity, delta);

    // Check if we're still moving (using configured threshold)
    if (this.velocity.length() < CONTROL_CONFIG.fly.physics.velocityThreshold) {
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
          console.log('🔧 [Luxar] Applying roll torque:', this.lookState.roll * this.rotationSpeed);
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
    if (angularSpeed > CONTROL_CONFIG.fly.physics.angularVelocityThreshold) {
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
      Math.pow(effectiveRotationDamping, delta * CONTROL_CONFIG.fly.physics.dampingPower)
    );

    // Stop tiny rotations
    if (this.angularVelocity.length() < CONTROL_CONFIG.fly.physics.angularVelocityThreshold) {
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
   * Stop continuous look change
   */
  private stopLookChange(): void {
    this.lookState.horizontal = 0;
    this.lookState.vertical = 0;
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
    // Store current camera position and orientation
    // This would be used by a reset() method if needed
  }

  /**
   * Reset to saved state
   */
  public reset(): void {
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

    // Reset speed boost
    this.speedBoost = false;

    // Reset to identity quaternion (looking forward)
    this.orientation.set(0, 0, 0, 1);

    this.updateOrientation();
  }

  /**
   * Dispose of controls and clean up event listeners
   */
  public dispose(): void {
    this.removeEventListeners();
  }
}
