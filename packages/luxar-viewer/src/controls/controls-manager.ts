/**
 * Manages switching between different camera control types.
 *
 * Provides unified interface for three control modes:
 * - Orbit: Traditional orbit camera with target-based rotation
 * - Arcball: Quaternion-based trackball for free rotation without gimbal lock
 * - Fly: First-person WASD movement for exploring inside datasets
 *
 * Handles:
 * - Seamless switching between control types (V key)
 * - Camera state preservation during switches
 * - Event listener lifecycle management
 * - Configuration persistence and updates
 * - Auto-rotation support (orbit/arcball only)
 * - Physics-based movement (fly mode)
 *
 * @example
 * ```typescript
 * const controlsManager = new ControlsManager(camera, canvas);
 *
 * // Switch to fly mode
 * controlsManager.setControlType('fly');
 *
 * // Enable auto-rotation (orbit/arcball)
 * controlsManager.setControlType('orbit');
 * controlsManager.setAutoRotate(true);
 * ```
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls';
import { LuxarFlyControls } from './luxar-fly-controls';
import { config } from '../config';
import { log, Modules, LogEmoji } from '../utils/log';

export type ControlType = 'orbit' | 'arcball' | 'fly';

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
  private scene?: THREE.Scene;
  // Current control instance
  private currentControls: OrbitControls | ArcballControls | LuxarFlyControls | null = null;
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
    flyAcceleration: config.controls.fly.movement.acceleration.default,
  };

  // Saved camera state for switching
  private savedCameraPosition = new THREE.Vector3();
  private savedCameraRotation = new THREE.Euler();
  private savedTarget = new THREE.Vector3();
  private savedCameraUp = new THREE.Vector3(0, 1, 0); // Save the original up vector

  // Delta time tracking for fly controls
  private clock = new THREE.Clock();

  /**
   * Create controls manager for camera interaction.
   *
   * Initializes with orbit controls by default. Starts Three.js Clock for
   * frame-rate independent physics (fly mode).
   *
   * @param camera - Perspective camera to control
   * @param domElement - DOM element for mouse/touch input (typically canvas)
   * @param scene - Optional scene reference for advanced controls
   */
  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement, scene?: THREE.Scene) {
    super();

    this.camera = camera;
    this.domElement = domElement;
    this.scene = scene;

    // Initialize with orbit controls by default
    this.setControlType('orbit');
  }

  /**
   * Switch to different camera control type.
   *
   * Saves current camera state, disposes old controls, creates new controls,
   * and restores camera state. Emits 'change' event with new control type.
   *
   * @param type - Control type to switch to: 'orbit', 'arcball', or 'fly'
   *
   * @example
   * ```typescript
   * // Switch to fly mode for first-person exploration
   * controlsManager.setControlType('fly');
   *
   * // Switch back to orbit for traditional viewing
   * controlsManager.setControlType('orbit');
   * ```
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
      case 'arcball':
        this.createArcballControls();
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

    log.custom(LogEmoji.CONTROLS, Modules.CONTROLS, `Switched to ${type} controls`);
  }

  /**
   * Get currently active control type.
   *
   * @returns 'orbit', 'arcball', or 'fly'
   */
  public getControlType(): ControlType {
    return this.currentType;
  }

  /**
   * Get current controls instance for advanced manipulation.
   *
   * @returns Active controls instance, or null if not initialized
   */
  public getControls(): OrbitControls | ArcballControls | LuxarFlyControls | null {
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
   * Create arcball controls
   */
  private createArcballControls(): void {
    const controls = new ArcballControls(this.camera, this.domElement, this.scene) as any;

    // Configure arcball controls
    // Note: ArcballControls has these properties but TypeScript definitions are incomplete
    controls.enableDamping = true;
    controls.dampingFactor = 25; // ArcballControls uses different damping scale
    controls.enablePan = true;
    controls.enableRotate = true;
    controls.enableZoom = true;
    controls.minDistance = 0.1;
    controls.maxDistance = 1000;

    // IMPORTANT: Disable gizmos completely - both the flag and visibility
    controls.enableGizmos = false;
    controls.setGizmosVisible(false); // This actually hides the gizmos

    // Note: ArcballControls doesn't support auto-rotation

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

    // Don't set target or call setCamera here - let restoreCameraState handle it
    // to avoid double initialization
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
    this.savedCameraUp.copy(this.camera.up); // Save the current up vector

    // Save target if using orbit or arcball controls
    if (this.currentType === 'orbit' && this.currentControls instanceof OrbitControls) {
      this.savedTarget.copy(this.currentControls.target);
    } else if (this.currentType === 'arcball' && this.currentControls instanceof ArcballControls) {
      this.savedTarget.copy((this.currentControls as any).target);
    }
  }

  /**
   * Restore camera state after switching
   */
  private restoreCameraState(): void {
    // Don't restore position/rotation - keep them continuous
    // Only restore target if switching TO orbit or arcball controls
    if (this.currentControls instanceof OrbitControls) {
      // For orbit controls, update the target
      this.currentControls.target.copy(this.savedTarget);
      this.currentControls.update();
    } else if (this.currentControls instanceof ArcballControls) {
      // For arcball controls, set target and initialize camera
      const controls = this.currentControls as any;
      controls.target.copy(this.savedTarget);

      // CRITICAL FIX: Reset the camera's up vector to prevent jumps
      // ArcballControls modifies the up vector during rotation, which causes issues
      // when recreating controls if not properly reset
      this.camera.up.set(0, 1, 0); // Reset to default up vector
      this.camera.updateMatrixWorld();

      // Initialize the control with the current camera state
      // This must be done AFTER setting the target and resetting up vector
      controls.setCamera(this.camera);

      // IMPORTANT: Sync the internal up vector states with the camera's up vector
      // This prevents the "jump" at the start/end of dragging
      if (controls._up0 && controls._upState) {
        controls._up0.copy(this.camera.up);
        controls._upState.copy(this.camera.up);
      }

      // Update once to sync everything
      controls.update();
    }
    // Fly controls automatically initialize from current camera state
  }

  /**
   * Dispose of current controls
   */
  private disposeCurrentControls(): void {
    if (this.currentControls) {
      // For ArcballControls, reset before disposing to prevent state issues
      if (this.currentControls instanceof ArcballControls) {
        // Reset the control state before disposal to prevent up vector issues
        this.currentControls.reset();
      }

      if ('dispose' in this.currentControls) {
        this.currentControls.dispose();
      }
      this.currentControls = null;
    }
  }

  /**
   * Update controls - must be called every frame in animation loop.
   *
   * Processes user input, applies damping, and updates camera. For fly
   * controls, requires delta time from Clock for frame-rate independence.
   */
  public update(): void {
    if (!this.currentControls) return;

    if (this.currentControls instanceof OrbitControls) {
      this.currentControls.update();
    } else if (this.currentControls instanceof ArcballControls) {
      // ArcballControls need manual update call
      this.currentControls.update();
      // Note: ArcballControls doesn't support auto-rotation
    } else if (this.currentControls instanceof LuxarFlyControls) {
      const delta = this.clock.getDelta();
      this.currentControls.update(delta);
    }
  }

  /**
   * Enable or disable camera controls globally.
   *
   * @param enabled - true to allow user interaction, false to disable
   */
  public setEnabled(enabled: boolean): void {
    if (this.currentControls) {
      this.currentControls.enabled = enabled;
    }
  }

  /**
   * Enable/disable auto-rotation (orbit controls only).
   *
   * Has no effect in arcball or fly modes.
   *
   * @param enabled - true to enable auto-rotation, false to disable
   */
  public setAutoRotate(enabled: boolean): void {
    this.config.autoRotate = enabled;

    if (this.currentControls instanceof OrbitControls) {
      this.currentControls.autoRotate = enabled;
    }
    // ArcballControls and FlyControls don't support auto-rotation
  }

  /**
   * Set auto-rotation speed (only for orbit controls)
   */
  public setAutoRotateSpeed(speed: number): void {
    this.config.autoRotateSpeed = speed;

    if (this.currentControls instanceof OrbitControls) {
      this.currentControls.autoRotateSpeed = speed;
    }
    // ArcballControls and FlyControls don't support auto-rotation
  }

  /**
   * Get auto-rotation state.
   *
   * @returns true if auto-rotation is enabled (orbit mode only), false otherwise
   */
  public getAutoRotate(): boolean {
    if (this.currentControls instanceof OrbitControls) {
      return this.currentControls.autoRotate;
    }
    // ArcballControls and FlyControls don't support auto-rotation
    return false;
  }

  /**
   * Set fly mode movement speed (WASD keys).
   *
   * @param speed - Movement speed in world units per second (default 10)
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
   * Enable/disable fly mode inertial physics (momentum).
   *
   * @param inertial - true for momentum (continues after key release), false for instant stop
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
   * Reset controls to saved state (position, rotation, target).
   *
   * Restores camera to last saved state (from saveState() or control initialization).
   */
  public reset(): void {
    if (this.currentControls) {
      if (this.currentControls instanceof OrbitControls) {
        this.currentControls.reset();
      } else if (this.currentControls instanceof ArcballControls) {
        this.currentControls.reset();
      } else if (this.currentControls instanceof LuxarFlyControls) {
        this.currentControls.reset();
      }
    }
  }

  /**
   * Save current camera state for reset() functionality.
   *
   * Captures current position, rotation, and target as new default state.
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
    } else if (this.currentControls instanceof ArcballControls) {
      // For arcball controls, update the target
      (this.currentControls as any).target.copy(target);
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
    } else if (this.currentControls instanceof ArcballControls) {
      return (this.currentControls as any).target.clone();
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

  /**
   * Clean up and dispose all controls
   */
  public dispose(): void {
    this.disposeCurrentControls();
    this.clock.stop();
  }
}
