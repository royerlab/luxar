/**
 * Manages switching between different camera control types
 *
 * Handles:
 * - Switching between OrbitControls and LuxarFlyControls
 * - Preserving camera state during switches
 * - Proper cleanup of event listeners
 * - Configuration persistence
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { LuxarFlyControls } from './luxar-fly-controls';
import { CONTROL_CONFIG } from './control-config';

export type ControlType = 'orbit' | 'fly';

export interface ControlsManagerConfig {
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  flyMovementSpeed?: number;
  flyRotationSpeed?: number;
  flyLookSpeed?: number;
  flyInertialMode?: boolean;
  flyDamping?: number;
  flyRotationDamping?: number;
  flyAcceleration?: number;
}

interface ControlsManagerEventMap {
  change: { controlType?: ControlType };
  start: {};
  end: {};
}

export class ControlsManager extends THREE.EventDispatcher<ControlsManagerEventMap> {
  private camera: THREE.PerspectiveCamera;
  private domElement: HTMLElement;
  // Current control instance
  private currentControls: OrbitControls | LuxarFlyControls | null = null;
  private currentType: ControlType = 'orbit';

  // Configuration - uses defaults from CONTROL_CONFIG
  private config: ControlsManagerConfig = {
    autoRotate: false,
    autoRotateSpeed: CONTROL_CONFIG.orbit.autoRotate.speed.default,
    flyMovementSpeed: CONTROL_CONFIG.fly.movement.speed.default,
    flyRotationSpeed: CONTROL_CONFIG.fly.rotation.speed.default,
    flyLookSpeed: CONTROL_CONFIG.fly.look.mouseSpeed.default,
    flyInertialMode: CONTROL_CONFIG.fly.inertialMode.default,
    flyDamping: CONTROL_CONFIG.fly.movement.damping.default,
    flyRotationDamping: CONTROL_CONFIG.fly.rotation.damping.default,
    flyAcceleration: CONTROL_CONFIG.fly.movement.acceleration.default,
  };

  // Saved camera state for switching
  private savedCameraPosition = new THREE.Vector3();
  private savedCameraRotation = new THREE.Euler();
  private savedTarget = new THREE.Vector3();

  // Delta time tracking for fly controls
  private clock = new THREE.Clock();

  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement, _scene?: THREE.Scene) {
    super();

    this.camera = camera;
    this.domElement = domElement;

    // Initialize with orbit controls by default
    this.setControlType('orbit');
  }

  /**
   * Switch to a different control type
   * @param type - The control type to switch to
   */
  public setControlType(type: ControlType): void {
    if (type === this.currentType && this.currentControls) {
      return; // Already using this type
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
    }

    this.currentType = type;

    // Restore camera state
    this.restoreCameraState();

    // Emit change event
    this.dispatchEvent({ type: 'change', controlType: type });

    console.log(`🎮 [Luxar] Switched to ${type} controls`);
  }

  /**
   * Get the current control type
   */
  public getControlType(): ControlType {
    return this.currentType;
  }

  /**
   * Get the current controls instance
   */
  public getControls(): OrbitControls | LuxarFlyControls | null {
    return this.currentControls;
  }

  /**
   * Create orbit controls
   */
  private createOrbitControls(): void {
    const controls = new OrbitControls(this.camera, this.domElement);

    // Configure orbit controls
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = true;
    controls.minDistance = 0.1;
    controls.maxDistance = 1000;
    controls.autoRotate = this.config.autoRotate || false;
    controls.autoRotateSpeed = this.config.autoRotateSpeed || 0.25;

    // Set target
    controls.target.copy(this.savedTarget);

    // Listen for changes
    controls.addEventListener('change', () => {
      this.dispatchEvent({ type: 'change' });
    });

    controls.addEventListener('start', () => {
      this.dispatchEvent({ type: 'start' });
    });

    controls.addEventListener('end', () => {
      this.dispatchEvent({ type: 'end' });
    });

    this.currentControls = controls;

    // Reset clock for proper delta time when switching back
    this.clock.getDelta();
  }

  /**
   * Create fly controls
   */
  private createFlyControls(): void {
    const controls = new LuxarFlyControls(this.camera, this.domElement, {
      movementSpeed: this.config.flyMovementSpeed,
      rotationSpeed: this.config.flyRotationSpeed,
      lookSpeed: this.config.flyLookSpeed,
      inertialMode: this.config.flyInertialMode,
      damping: this.config.flyDamping,
      rotationDamping: this.config.flyRotationDamping,
      acceleration: this.config.flyAcceleration,
    });

    // Enable external input management for better control
    controls.setExternalInputManagement(true);

    // The fly controls will initialize from the current camera state
    // so no need to manually set position/rotation

    // Listen for changes
    controls.addEventListener('change', () => {
      this.dispatchEvent({ type: 'change' });
    });

    controls.addEventListener('start', () => {
      this.dispatchEvent({ type: 'start' });
    });

    controls.addEventListener('end', () => {
      this.dispatchEvent({ type: 'end' });
    });

    this.currentControls = controls;

    // Start clock for delta time
    this.clock.start();
  }

  /**
   * Save current camera state before switching
   */
  private saveCameraState(): void {
    this.savedCameraPosition.copy(this.camera.position);
    this.savedCameraRotation.copy(this.camera.rotation);

    // Save orbit target if using orbit controls
    if (this.currentType === 'orbit' && this.currentControls instanceof OrbitControls) {
      this.savedTarget.copy(this.currentControls.target);
    }
  }

  /**
   * Restore camera state after switching
   */
  private restoreCameraState(): void {
    // Don't restore position/rotation - keep them continuous
    // Only restore orbit target if switching TO orbit controls
    if (this.currentControls instanceof OrbitControls) {
      // For orbit controls, update the target
      this.currentControls.target.copy(this.savedTarget);
      this.currentControls.update();
    }
    // Fly controls automatically initialize from current camera state
  }

  /**
   * Dispose of current controls
   */
  private disposeCurrentControls(): void {
    if (this.currentControls) {
      if ('dispose' in this.currentControls) {
        this.currentControls.dispose();
      }
      this.currentControls = null;
    }
  }

  /**
   * Update controls - must be called in animation loop
   */
  public update(): void {
    if (!this.currentControls) return;

    if (this.currentControls instanceof OrbitControls) {
      this.currentControls.update();
    } else if (this.currentControls instanceof LuxarFlyControls) {
      const delta = this.clock.getDelta();
      this.currentControls.update(delta);
    }
  }

  /**
   * Enable/disable controls
   */
  public setEnabled(enabled: boolean): void {
    if (this.currentControls) {
      this.currentControls.enabled = enabled;
    }
  }

  /**
   * Set auto-rotation (for orbit controls)
   */
  public setAutoRotate(enabled: boolean): void {
    this.config.autoRotate = enabled;

    if (this.currentControls instanceof OrbitControls) {
      this.currentControls.autoRotate = enabled;
    }
  }

  /**
   * Set auto-rotation speed (for orbit controls)
   */
  public setAutoRotateSpeed(speed: number): void {
    this.config.autoRotateSpeed = speed;

    if (this.currentControls instanceof OrbitControls) {
      this.currentControls.autoRotateSpeed = speed;
    }
  }

  /**
   * Get auto-rotation state
   */
  public getAutoRotate(): boolean {
    if (this.currentControls instanceof OrbitControls) {
      return this.currentControls.autoRotate;
    }
    return false;
  }

  /**
   * Set fly controls movement speed
   */
  public setFlyMovementSpeed(speed: number): void {
    this.config.flyMovementSpeed = speed;

    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.movementSpeed = speed;
    }
  }

  /**
   * Set fly controls rotation speed
   */
  public setFlyRotationSpeed(speed: number): void {
    this.config.flyRotationSpeed = speed;

    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.rotationSpeed = speed;
    }
  }

  /**
   * Set fly controls inertial mode
   */
  public setFlyInertialMode(inertial: boolean): void {
    this.config.flyInertialMode = inertial;

    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.setInertialMode(inertial);
    }
  }

  /**
   * Set fly controls damping
   */
  public setFlyDamping(damping: number): void {
    this.config.flyDamping = damping;

    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.damping = damping;
    }
  }

  /**
   * Set fly controls rotation damping
   */
  public setFlyRotationDamping(damping: number): void {
    this.config.flyRotationDamping = damping;

    if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.rotationDamping = damping;
    }
  }

  /**
   * Reset controls to default state
   */
  public reset(): void {
    if (this.currentControls) {
      if (this.currentControls instanceof OrbitControls) {
        this.currentControls.reset();
      } else if (this.currentControls instanceof LuxarFlyControls) {
        this.currentControls.reset();
      }
    }
  }

  /**
   * Save current state (for reset functionality)
   */
  public saveState(): void {
    if (this.currentControls instanceof OrbitControls) {
      this.currentControls.saveState();
    } else if (this.currentControls instanceof LuxarFlyControls) {
      this.currentControls.saveState();
    }
  }

  /**
   * Enable/disable zoom (for orbit controls)
   */
  public setEnableZoom(enabled: boolean): void {
    if (this.currentControls instanceof OrbitControls) {
      this.currentControls.enableZoom = enabled;
    }
  }

  /**
   * Look at a target position (smoothly for fly controls, directly for orbit)
   * @param target - Target position to look at
   * @param smooth - Whether to animate smoothly (only for fly controls)
   */
  public lookAt(target: THREE.Vector3, smooth: boolean = true): void {
    if (this.currentControls instanceof OrbitControls) {
      // For orbit controls, update the target
      this.currentControls.target.copy(target);
      this.currentControls.update();
    } else if (this.currentControls instanceof LuxarFlyControls) {
      // For fly controls, smoothly rotate to look at target
      if (smooth) {
        // Will be called repeatedly in animation loop for smooth transition
        this.currentControls.lookAtSmooth(target, 0.9);
      } else {
        // Immediate look at
        this.currentControls.lookAtSmooth(target, 0);
      }
    }
  }

  /**
   * Get the current focus target (orbit target or look-at point)
   */
  public getFocusTarget(): THREE.Vector3 {
    if (this.currentControls instanceof OrbitControls) {
      return this.currentControls.target.clone();
    } else {
      // For fly controls, return a point in front of the camera
      const forward = new THREE.Vector3();
      this.camera.getWorldDirection(forward);
      return this.camera.position.clone().add(forward.multiplyScalar(10));
    }
  }

  /**
   * Get the current fly controls if active
   */
  public getFlyControls(): LuxarFlyControls | null {
    if (this.currentControls instanceof LuxarFlyControls) {
      return this.currentControls;
    }
    return null;
  }

  /**
   * Get the fly controls configuration
   * This returns the stored config regardless of which control type is active
   */
  public getFlyConfig(): { inertialMode: boolean; damping: number; rotationDamping: number; movementSpeed: number; rotationSpeed: number } {
    return {
      inertialMode: this.config.flyInertialMode ?? CONTROL_CONFIG.fly.inertialMode.default,
      damping: this.config.flyDamping ?? CONTROL_CONFIG.fly.movement.damping.default,
      rotationDamping: this.config.flyRotationDamping ?? CONTROL_CONFIG.fly.rotation.damping.default,
      movementSpeed: this.config.flyMovementSpeed ?? CONTROL_CONFIG.fly.movement.speed.default,
      rotationSpeed: this.config.flyRotationSpeed ?? CONTROL_CONFIG.fly.rotation.speed.default,
    };
  }

  /**
   * Clean up and dispose all controls
   */
  public dispose(): void {
    this.disposeCurrentControls();
    this.clock.stop();
  }
}
