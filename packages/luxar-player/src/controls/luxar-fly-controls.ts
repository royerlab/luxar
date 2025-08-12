/**
 * Custom fly controls for Luxar with inertial and non-inertial modes
 * 
 * Features:
 * - Arrow keys for movement (forward/back/strafe)
 * - Shift+arrows for vertical movement
 * - Mouse drag for camera rotation
 * - Two modes: Direct (velocity) and Inertial (acceleration)
 * - Configurable damping for smooth deceleration
 */

import * as THREE from 'three';
import { CONTROL_CONFIG } from './control-config';

export interface LuxarFlyControlsConfig {
  movementSpeed?: number;  // Units per second
  lookSpeed?: number;      // Radians per pixel
  inertialMode?: boolean;  // True for acceleration, false for velocity
  damping?: number;        // 0.9-0.99 for inertial mode
  acceleration?: number;   // Acceleration rate for inertial mode
}

export class LuxarFlyControls extends THREE.EventDispatcher<{
  change: {};
  start: {};
  end: {};
}> {
  public enabled: boolean = true;
  
  // Configuration
  public movementSpeed: number = CONTROL_CONFIG.fly.movement.speed.default;
  public lookSpeed: number = CONTROL_CONFIG.fly.look.mouseSpeed.default;
  public inertialMode: boolean = false;
  public damping: number = CONTROL_CONFIG.fly.movement.damping.default;
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
    horizontal: 0,  // -1 for left, 1 for right
    vertical: 0,    // -1 for up, 1 for down
  };
  
  // Look speed for arrow keys (radians per second)
  public arrowLookSpeed: number = CONTROL_CONFIG.fly.look.keyboardSpeed.default;
  
  // Velocity vector for inertial mode
  private velocity = new THREE.Vector3(0, 0, 0);
  
  // Mouse state for looking
  private isMouseDown = false;
  private mouseX = 0;
  private mouseY = 0;
  private lat = 0;
  private lon = 0;
  private phi = 0;
  private theta = 0;
  
  // References
  private camera: THREE.PerspectiveCamera;
  private domElement: HTMLElement;
  
  // Event listeners to clean up
  private boundHandlers: { [key: string]: any } = {};
  
  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement, config?: LuxarFlyControlsConfig) {
    super();
    
    this.camera = camera;
    this.domElement = domElement;
    
    // Apply configuration
    if (config) {
      this.movementSpeed = config.movementSpeed ?? this.movementSpeed;
      this.lookSpeed = config.lookSpeed ?? this.lookSpeed;
      this.inertialMode = config.inertialMode ?? this.inertialMode;
      this.damping = config.damping ?? this.damping;
      this.acceleration = config.acceleration ?? this.acceleration;
    }
    
    // Initialize orientation from current camera
    this.initializeFromCamera();
    
    // Bind event handlers
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
    window.addEventListener('keydown', this.boundHandlers.keydown);
    window.addEventListener('keyup', this.boundHandlers.keyup);
    this.domElement.addEventListener('mousedown', this.boundHandlers.mousedown);
    window.addEventListener('mouseup', this.boundHandlers.mouseup);
    window.addEventListener('mousemove', this.boundHandlers.mousemove);
    this.domElement.addEventListener('contextmenu', this.boundHandlers.contextmenu);
  }
  
  private onKeyDown(event: KeyboardEvent): void {
    if (!this.enabled) return;
    
    // Only prevent default for arrow keys (always used for camera look)
    if (event.key.startsWith('Arrow')) {
      event.preventDefault();
    }
    
    // Only prevent default for WASD if we're not typing in an input field
    const activeElement = document.activeElement;
    const isTyping = activeElement && (
      activeElement.tagName === 'INPUT' ||
      activeElement.tagName === 'TEXTAREA' ||
      activeElement.getAttribute('contenteditable') === 'true'
    );
    
    if (!isTyping && ['w', 'a', 's', 'd', 'W', 'A', 'S', 'D'].includes(event.key)) {
      event.preventDefault();
    }
    
    // WASD for movement
    switch (event.key.toLowerCase()) {
      case 'w':
        if (event.altKey || event.metaKey) {
          this.moveState.up = 1;  // Alt/Option+W for up
        } else {
          this.moveState.forward = 1;  // W for forward
        }
        break;
      case 's':
        if (event.altKey || event.metaKey) {
          this.moveState.down = 1;  // Alt/Option+S for down
        } else {
          this.moveState.back = 1;  // S for backward
        }
        break;
      case 'a':
        this.moveState.left = 1;  // A for strafe left
        break;
      case 'd':
        this.moveState.right = 1;  // D for strafe right
        break;
    }
    
    // Arrow keys for camera look direction
    switch (event.key) {
      case 'ArrowUp':
        this.startLookChange(0, -1);  // Look up
        break;
      case 'ArrowDown':
        this.startLookChange(0, 1);   // Look down
        break;
      case 'ArrowLeft':
        this.startLookChange(-1, 0);  // Look left
        break;
      case 'ArrowRight':
        this.startLookChange(1, 0);   // Look right
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
        this.moveState.up = 0;  // Also clear up in case Alt was held
        break;
      case 's':
        this.moveState.back = 0;
        this.moveState.down = 0;  // Also clear down in case Alt was held
        break;
      case 'a':
        this.moveState.left = 0;
        break;
      case 'd':
        this.moveState.right = 0;
        break;
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
    
    // Update look angles
    this.lon -= deltaX * this.lookSpeed * 100;
    this.lat += deltaY * this.lookSpeed * 100;
    
    // Clamp vertical rotation
    this.lat = Math.max(-85, Math.min(85, this.lat));
    
    this.updateOrientation();
    
    this.dispatchEvent({ type: 'change' });
  }
  
  /**
   * Initialize lat/lon from current camera orientation
   */
  private initializeFromCamera(): void {
    // Get the camera's forward direction
    const forward = new THREE.Vector3();
    this.camera.getWorldDirection(forward);
    forward.normalize();
    
    // Convert to spherical coordinates
    // Longitude (horizontal rotation)
    this.lon = THREE.MathUtils.radToDeg(Math.atan2(forward.z, forward.x));
    
    // Latitude (vertical rotation)
    const horizontalLength = Math.sqrt(forward.x * forward.x + forward.z * forward.z);
    this.lat = THREE.MathUtils.radToDeg(Math.atan2(forward.y, horizontalLength));
    
    // Update phi and theta
    this.phi = THREE.MathUtils.degToRad(90 - this.lat);
    this.theta = THREE.MathUtils.degToRad(this.lon);
  }
  
  private updateOrientation(): void {
    // Convert lat/lon to spherical coordinates
    this.phi = THREE.MathUtils.degToRad(90 - this.lat);
    this.theta = THREE.MathUtils.degToRad(this.lon);
    
    // Calculate look direction
    const lookAt = new THREE.Vector3();
    lookAt.x = this.camera.position.x + Math.sin(this.phi) * Math.cos(this.theta);
    lookAt.y = this.camera.position.y + Math.cos(this.phi);
    lookAt.z = this.camera.position.z + Math.sin(this.phi) * Math.sin(this.theta);
    
    this.camera.lookAt(lookAt);
  }
  
  /**
   * Update controls - must be called in animation loop
   * @param delta - Time since last frame in seconds
   */
  public update(delta: number): void {
    if (!this.enabled) return;
    
    // Get movement vectors relative to camera orientation
    const forward = new THREE.Vector3();
    const right = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0); // World up
    
    // Get camera's forward direction (negative Z in camera space)
    this.camera.getWorldDirection(forward);
    forward.normalize();
    
    // Get camera's right direction (X axis in camera space)
    right.setFromMatrixColumn(this.camera.matrix, 0);
    right.normalize();
    
    let isMoving = false;
    
    if (this.inertialMode) {
      // Inertial mode: apply acceleration
      const accel = new THREE.Vector3();
      
      // Calculate acceleration from input
      accel.addScaledVector(forward, (this.moveState.forward - this.moveState.back) * this.acceleration);
      accel.addScaledVector(right, (this.moveState.right - this.moveState.left) * this.acceleration);
      accel.addScaledVector(up, (this.moveState.up - this.moveState.down) * this.acceleration);
      
      // Update velocity
      this.velocity.addScaledVector(accel, delta);
      
      // Apply damping
      this.velocity.multiplyScalar(Math.pow(this.damping, delta * CONTROL_CONFIG.fly.physics.dampingPower)); // Normalize to 60fps
      
      // Apply velocity to position
      this.camera.position.addScaledVector(this.velocity, delta);
      
      // Check if we're still moving (using configured threshold)
      const velocityMagnitude = this.velocity.length();
      if (velocityMagnitude < CONTROL_CONFIG.fly.physics.velocityThreshold) {
        // Stop tiny movements
        this.velocity.set(0, 0, 0);
      } else {
        // We're still moving - need to keep rendering
        isMoving = true;
      }
    } else {
      // Direct mode: immediate velocity control
      const movement = new THREE.Vector3();
      
      // Calculate movement from input - in camera's frame of reference
      movement.addScaledVector(forward, (this.moveState.forward - this.moveState.back));
      movement.addScaledVector(right, (this.moveState.right - this.moveState.left));
      movement.addScaledVector(up, (this.moveState.up - this.moveState.down));
      
      // Check if there's any active movement
      if (movement.length() > 0) {
        isMoving = true;
      }
      
      // Apply movement scaled by speed and delta
      movement.multiplyScalar(this.movementSpeed * delta);
      this.camera.position.add(movement);
    }
    
    // Apply arrow key look changes
    if (this.lookState.horizontal !== 0 || this.lookState.vertical !== 0) {
      // Update look angles based on arrow keys
      this.lon += this.lookState.horizontal * this.arrowLookSpeed * delta * 100;
      this.lat -= this.lookState.vertical * this.arrowLookSpeed * delta * 100;
      
      // Clamp vertical rotation
      this.lat = Math.max(-85, Math.min(85, this.lat));
      
      isMoving = true;  // Keep rendering active while looking
    }
    
    // Update camera orientation for look changes
    this.updateOrientation();
    
    // Dispatch change event if we're moving to keep animation running
    if (isMoving) {
      this.dispatchEvent({ type: 'change' });
    }
  }
  
  /**
   * Set movement mode
   * @param inertial - True for inertial (acceleration), false for direct (velocity)
   */
  public setInertialMode(inertial: boolean): void {
    this.inertialMode = inertial;
    if (!inertial) {
      // Clear velocity when switching to direct mode
      this.velocity.set(0, 0, 0);
    }
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
    
    // Convert to spherical coordinates
    const targetLon = THREE.MathUtils.radToDeg(Math.atan2(direction.z, direction.x));
    const horizontalLength = Math.sqrt(direction.x * direction.x + direction.z * direction.z);
    const targetLat = THREE.MathUtils.radToDeg(Math.atan2(direction.y, horizontalLength));
    
    // Smooth interpolation
    this.lon = this.lon * smoothness + targetLon * (1 - smoothness);
    this.lat = this.lat * smoothness + targetLat * (1 - smoothness);
    
    // Clamp vertical rotation
    this.lat = Math.max(-85, Math.min(85, this.lat));
    
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
    // Reset velocity
    this.velocity.set(0, 0, 0);
    
    // Reset movement state
    this.moveState.forward = 0;
    this.moveState.back = 0;
    this.moveState.left = 0;
    this.moveState.right = 0;
    this.moveState.up = 0;
    this.moveState.down = 0;
    
    // Reset look angles
    this.lat = 0;
    this.lon = 0;
    
    this.updateOrientation();
  }
  
  /**
   * Dispose of controls and clean up event listeners
   */
  public dispose(): void {
    window.removeEventListener('keydown', this.boundHandlers.keydown);
    window.removeEventListener('keyup', this.boundHandlers.keyup);
    this.domElement.removeEventListener('mousedown', this.boundHandlers.mousedown);
    window.removeEventListener('mouseup', this.boundHandlers.mouseup);
    window.removeEventListener('mousemove', this.boundHandlers.mousemove);
    this.domElement.removeEventListener('contextmenu', this.boundHandlers.contextmenu);
  }
}