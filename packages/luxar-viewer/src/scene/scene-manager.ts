// Scene, renderer, and camera management for the Luxar scene player
//
// This module handles all Three.js setup and 3D graphics configuration:
// - WebGL renderer initialization with optimal settings
// - Camera setup with proper projection and positioning
// - ArcballControls for intuitive 3D navigation
// - Scene graph management and Zarr data loading
// - Resource disposal for memory management

import * as THREE from 'three';
import { ControlsManager } from '../controls/controls-manager';
import { loadScene } from '../data';
import { showLoadingIndicator, hideLoadingIndicator, showError } from '../ui/helpers';
import { config } from '../config';
import { PostProcessingManager } from '../rendering/post-processing-manager';
import { materialManager } from '../rendering/material-manager';
import {
  detectHDRCapabilities,
  configureHDRRenderer,
  logHDRCapabilities,
} from '../utils/hdr-detection';
import { validateFOV, calculateClippingPlanes } from './scene-manager-utils';
import { log, Modules, LogEmoji } from '../utils/log';

/**
 * SceneManager orchestrates all Three.js components for 3D rendering
 *
 * Responsibilities:
 * - WebGL renderer setup with HDR capabilities
 * - Camera configuration for optimal 3D viewing
 * - Control system for user interaction (rotation, zoom, pan)
 * - Scene graph management for 3D objects
 * - HDR post-processing pipeline with bloom effects
 * - Advanced shader-based points rendering
 * - Dynamic loading of points data from Zarr sources
 * - Resource cleanup to prevent memory leaks
 *
 * Technical Details:
 * - Uses perspective camera for realistic 3D projection
 * - ArcballControls provide constraint-based camera movement
 * - HDR post-processing with ACES tone mapping and bloom
 * - Custom Gaussian point shaders for enhanced visual quality
 * - Automatic canvas resizing for responsive design
 */
export class SceneManager extends THREE.EventDispatcher<{
  change: {};
}> {
  /** Three.js WebGL renderer - handles all GPU-accelerated rendering */
  public renderer!: THREE.WebGLRenderer;

  /** Three.js scene graph - container for all 3D objects and lights */
  public scene!: THREE.Scene;

  /** Perspective camera - provides realistic 3D viewing with depth */
  public camera!: THREE.PerspectiveCamera;

  /** ControlsManager - manages different camera control types (orbit, arcball, fly) */
  public controls!: ControlsManager;

  /** HDR post-processing manager for bloom and tone mapping effects */
  public postProcessing!: PostProcessingManager;

  /** The HTML canvas element where 3D rendering occurs */
  private canvasElement!: HTMLCanvasElement;

  /** Track whether we're centered on bounding box or origin */
  private isCenteredOnBoundingBox: boolean = false;

  /** Store the last calculated bounding box center */
  private lastBoundingBoxCenter: THREE.Vector3 = new THREE.Vector3();

  /**
   * Initialize the complete 3D scene setup
   */
  constructor() {
    super();
  }

  async init(): Promise<void> {
    this.setupCanvas();
    this.setupRenderer();
    this.setupScene();
    this.setupCamera();
    this.setupControls();
    this.setupPostProcessing();

    // Call updateSize() during initialization to ensure consistent behavior
    // This makes initialization go through the same path as resize events
    this.updateSize();
  }

  /**
   * Get and validate the canvas element
   */
  private setupCanvas(): void {
    const element = document.getElementById(config.canvasId) as HTMLCanvasElement;
    if (!element) {
      showError(
        `Canvas element with id '${config.canvasId}' not found. Please check the HTML structure.`
      );
      throw new Error('Required canvas element not found');
    }
    this.canvasElement = element;
  }

  /**
   * Initialize Three.js WebGL renderer with optimal settings
   *
   * This creates the WebGL rendering context that will handle all GPU operations.
   * Key configurations:
   * - Antialiasing for smooth edges (MSAA)
   * - Custom canvas element for precise DOM control
   * - High DPI display support via pixel ratio
   * - Accessibility attributes for screen readers
   * - Fullscreen immersive experience
   */
  private setupRenderer(): void {
    // Try to get HDR canvas context first using config values
    let gl: WebGLRenderingContext | null = null;
    try {
      gl = this.canvasElement.getContext(
        'webgl2',
        config.webgl.context
      ) as WebGLRenderingContext | null;

      if (!gl) {
        log.warning(
          Modules.SCENE_MANAGER,
          'WebGL2 context creation failed, falling back to default'
        );
      }
    } catch (error) {
      log.error(Modules.SCENE_MANAGER, 'Error creating WebGL2 context:', error);
      showError('Failed to create WebGL2 context. Your browser may not support WebGL2.');
    }

    // Create WebGL renderer using configuration values
    // This ensures consistent settings across all rendering components
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvasElement, // Use our pre-existing canvas element
      context: gl || undefined, // Use our HDR context if available
      ...config.webgl.renderer, // Apply all renderer settings from config
    });

    // Configure page for immersive fullscreen 3D experience
    // Remove default margins to eliminate whitespace around canvas
    document.body.style.margin = '0';

    // Hide scrollbars since 3D scene uses entire viewport
    document.body.style.overflow = 'hidden';

    // NOTE: We don't append renderer.domElement because we're using the existing HTML canvas

    // Report hardware point size limits in debug mode
    const debugParams = new URLSearchParams(window.location.search);
    if (debugParams.has('debug')) {
      const glContext = this.renderer.getContext();
      const pointSizeRange = glContext.getParameter(glContext.ALIASED_POINT_SIZE_RANGE);
      log.info(
        Modules.RENDERER,
        `Hardware point size limits: ${pointSizeRange[0]}-${pointSizeRange[1]} pixels`
      );
    }
    // This allows for better integration into complex HTML pages

    // Configure renderer dimensions and high-DPI support
    this.updateRendererSize();

    // Detect and configure HDR capabilities
    const hdrCapabilities = detectHDRCapabilities(this.renderer);
    logHDRCapabilities(hdrCapabilities);
    configureHDRRenderer(this.renderer, hdrCapabilities);
  }

  /**
   * Initialize the Three.js scene
   */
  private setupScene(): void {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(config.scene.backgroundColor);
  }

  /**
   * Initialize perspective camera with optimal 3D viewing parameters
   *
   * Perspective camera provides realistic 3D projection with depth perception.
   * Key parameters:
   * - FOV: Field of view angle (wider = more visible, narrower = more focused)
   * - Aspect ratio: Width/height ratio matching canvas dimensions
   * - Near/far planes: Define visible depth range (Z-clipping)
   * - Initial position: Starting camera location in 3D space
   */
  private setupCamera(): void {
    // Get canvas dimensions for proper aspect ratio
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;

    // Create perspective camera with realistic 3D projection
    // FOV of 60° provides natural human-like viewing angle
    this.camera = new THREE.PerspectiveCamera(
      config.camera.fov, // Field of view (60 degrees)
      width / height, // Aspect ratio (canvas width/height)
      config.camera.near, // Near clipping plane (0.1 units)
      config.camera.far // Far clipping plane (1000 units)
    );

    // Position camera at initial viewing location
    // Z=8 provides good overview of typical points scenes
    // X=0, Y=0 centers the view on the origin
    this.camera.position.set(
      config.camera.initialPosition.x, // X position (0 = centered)
      config.camera.initialPosition.y, // Y position (0 = centered)
      config.camera.initialPosition.z // Z position (8 = pulled back for overview)
    );
  }

  /**
   * Initialize ControlsManager for flexible camera control
   *
   * ControlsManager supports multiple control types:
   * - Orbit: Traditional orbital camera with auto-rotation
   * - Fly: First-person flying controls with inertia
   *
   * Features:
   * - Hot-swapping between control types
   * - Auto-rotation for presentations
   * - Inertial and non-inertial movement modes
   * - Smooth transitions and state preservation
   */
  private setupControls(): void {
    // Create controls manager with camera and DOM element
    this.controls = new ControlsManager(this.camera, this.renderer.domElement, this.scene);

    // Set default control type (orbit)
    this.controls.setControlType('orbit');

    // Listen for control changes to trigger renders
    this.controls.addEventListener('change', () => {
      this.dispatchEvent({ type: 'change' });
    });

    // Save initial state so reset() works properly
    this.controls.saveState();
  }

  /**
   * Reset controls to default state
   * This is needed when loading a new dataset to prevent accumulated transformations
   */
  private resetControls(): void {
    // Reset camera to default position
    this.camera.position.set(
      config.camera.initialPosition.x,
      config.camera.initialPosition.y,
      config.camera.initialPosition.z
    );

    // Reset camera rotation to look at origin
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld(true);

    // Reset controls to default state
    this.controls.reset();

    // Update controls to sync with camera
    this.controls.update();

    // Save this configuration as the new default state
    this.controls.saveState();

    log.success(Modules.CONTROLS, 'Controls reset to default state');
  }

  /**
   * Initialize HDR post-processing pipeline for advanced visual effects
   *
   * This creates a sophisticated rendering chain:
   * 1. Scene renders to HDR buffer (16-bit float precision)
   * 2. UnrealBloomPass creates glow effects on bright areas
   * 3. OutputPass applies ACES tone mapping and sRGB conversion
   *
   * The result is professional-quality rendering with realistic bloom
   * effects and proper color management for accurate display.
   */
  private setupPostProcessing(): void {
    // Get canvas dimensions for proper HDR render target sizing
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;

    // Create post-processing manager with actual canvas dimensions
    this.postProcessing = new PostProcessingManager(this.renderer, this.scene, this.camera, {
      width,
      height,
    });

    // Log shader configuration for debugging
    // Shader configuration logging removed - now handled by PointMaterial

    log.success(Modules.POST_PROCESSING, 'HDR pipeline initialized');
  }

  /**
   * Load scene data from Zarr source
   */
  async loadSceneData(src: string): Promise<void> {
    showLoadingIndicator();

    try {
      // Clear existing scene content (keep lights and background)
      this.clearSceneContent();

      // Reset controls to default state before loading new content
      this.resetControls();

      // Update material manager BEFORE loading scene so materials are created with correct params
      if (this.camera && this.renderer) {
        const fovRadians = (this.camera.fov * Math.PI) / 180;
        const drawingBufferSize = this.renderer.getDrawingBufferSize(new THREE.Vector2());
        materialManager.updateCameraParams(fovRadians, drawingBufferSize);
      }

      const root = await loadScene(src);
      hideLoadingIndicator();
      this.scene.add(root);

      // Update all materials with current camera parameters after loading
      // This is critical - materials may have been created with wrong params
      if (this.camera && this.renderer) {
        const fovRadians = (this.camera.fov * Math.PI) / 180;
        const drawingBufferSize = this.renderer.getDrawingBufferSize(new THREE.Vector2());
        materialManager.updateCameraParams(fovRadians, drawingBufferSize);

        // Also update any materials that might have been created directly
        this.scene.traverse((object) => {
          if (object instanceof THREE.Points) {
            const material = object.material as THREE.ShaderMaterial;
            if (material.uniforms && material.uniforms.fov && material.uniforms.resolution) {
              material.uniforms.fov.value = fovRadians;
              // Ensure the resolution value is properly set
              if (material.uniforms.resolution.value && material.uniforms.resolution.value.copy) {
                material.uniforms.resolution.value.copy(drawingBufferSize);
              } else {
                material.uniforms.resolution.value = drawingBufferSize.clone();
              }
            }
          }
        });
      }

      // Don't automatically center - let the scene designer's positioning take precedence
      // User can press 'F' to center on bounding box if desired
      log.info(Modules.SCENE_MANAGER, 'Scene loaded. Press F to toggle centering on bounding box.');
    } catch (error) {
      hideLoadingIndicator();
      log.error(Modules.SCENE_MANAGER, 'Failed to load scene:', error);
      showError(`Failed to load scene from "${src}". Please check the path and try again.`);
      throw error;
    }
  }

  /**
   * Clear all loaded content from the scene, keeping lights and background
   */
  private clearSceneContent(): void {
    // Helper function to recursively dispose of objects
    const disposeObject = (obj: THREE.Object3D) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Points) {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose());
          } else {
            obj.material.dispose();
          }
        }
      }

      // Recursively dispose children
      while (obj.children.length > 0) {
        disposeObject(obj.children[0]);
        obj.remove(obj.children[0]);
      }
    };

    // Find all objects to remove (direct children of scene)
    const objectsToRemove: THREE.Object3D[] = [];

    for (let i = this.scene.children.length - 1; i >= 0; i--) {
      const child = this.scene.children[i];

      // Keep lights and any background/environment objects
      if (child instanceof THREE.Light) continue;
      if (child.userData?.isBackground) continue;

      // Mark everything else for removal
      objectsToRemove.push(child);
    }

    // Remove and dispose marked objects
    for (const obj of objectsToRemove) {
      disposeObject(obj);
      this.scene.remove(obj);
    }

    log.info(Modules.SCENE_MANAGER, `Cleared ${objectsToRemove.length} objects from scene`);
  }

  /**
   * Center camera on the bounding box of all visible objects
   */
  public centerCameraOnScene(): void {
    // Ensure world matrices are up to date before computing bounds
    this.scene.updateMatrixWorld(true);

    // Create a bounding box that encompasses all visible objects
    const box = new THREE.Box3();
    let totalPointCount = 0;

    // Traverse the scene and expand the box to include all geometries
    this.scene.traverse((object) => {
      if (object instanceof THREE.Points) {
        const geometry = object.geometry;

        // For points, compute bounding box from position attribute
        const positions = geometry.attributes.position;
        if (positions && positions.count > 0) {
          totalPointCount += positions.count;

          // First, compute the bounding box
          if (!geometry.boundingBox) {
            geometry.computeBoundingBox();
          }

          if (geometry.boundingBox) {
            const tempBox = geometry.boundingBox.clone();

            // Apply object's world transform
            tempBox.applyMatrix4(object.matrixWorld);

            // Only include if box has valid size (not empty)
            if (!tempBox.isEmpty()) {
              box.union(tempBox);
            }
          }
        }
      }
    });

    // Only center camera if we have a reasonable scene
    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);

      // Don't center if bounding box is too small or too few points
      // This prevents awkward camera positioning on edge cases
      if (maxDim < 1.0 || totalPointCount < 100) {
        // Keep default camera position for better user experience
        log.warning(
          Modules.SCENE_MANAGER,
          `Scene too small for auto-centering (size: ${maxDim.toFixed(2)}, points: ${totalPointCount})`
        );
        return;
      }

      const center = box.getCenter(new THREE.Vector3());

      // Store the center for later use
      this.lastBoundingBoxCenter.copy(center);

      // Position camera to see the entire scene
      const distance = maxDim * 1.2; // Closer for better visibility
      this.camera.position.set(center.x, center.y, center.z + distance);

      // Point camera at the center
      this.camera.lookAt(center);
      this.camera.updateMatrixWorld(true);

      // Update controls to orbit around the center
      // Note: ArcballControls doesn't have full TypeScript definitions, so we use type assertion
      const controlsAny = this.controls as any;

      // Set the new target position
      controlsAny.target.copy(center);

      // CRITICAL: Force the controls to recalculate internal state after target change
      // ArcballControls maintains internal gizmos that need to be synchronized
      this.controls.update();

      // Reset the saved state to the current configuration
      // This prevents the "jump" on first zoom interaction by ensuring
      // the saved state matches the actual current state
      this.controls.reset();
      this.controls.saveState();

      log.success(Modules.CONTROLS, 'Controls target updated and state saved');

      log.success(
        Modules.SCENE_MANAGER,
        `Camera centered on scene (center: [${center.x.toFixed(2)}, ${center.y.toFixed(2)}, ${center.z.toFixed(2)}], distance: ${distance.toFixed(2)})`
      );
    } else {
      log.warning(Modules.SCENE_MANAGER, 'No visible geometry found to center camera on');
    }
  }

  /**
   * Get the current center point (either origin or bounding box center)
   * @returns The current center as a Vector3
   */
  public getCurrentCenter(): THREE.Vector3 {
    if (this.isCenteredOnBoundingBox) {
      return this.lastBoundingBoxCenter.clone();
    }
    return new THREE.Vector3(0, 0, 0); // Origin
  }

  /**
   * Get the controls manager instance
   * @returns The ControlsManager instance
   */
  public getControlsManager(): ControlsManager {
    return this.controls;
  }

  /**
   * Toggle between centering on origin (native) and bounding box center
   */
  public toggleCentering(): void {
    if (this.isCenteredOnBoundingBox) {
      // Switch to origin (native center)
      this.centerOnOrigin();
      this.isCenteredOnBoundingBox = false;
      log.success(Modules.SCENE_MANAGER, 'Centered on origin (native center)');
    } else {
      // Switch to bounding box center
      this.centerCameraOnScene();
      this.isCenteredOnBoundingBox = true;
      log.success(Modules.SCENE_MANAGER, 'Centered on bounding box');
    }
  }

  /**
   * Center camera and controls on the origin
   */
  private centerOnOrigin(): void {
    // Get current camera distance from target
    const currentDistance = this.camera.position.distanceTo((this.controls as any).target);

    // Reset target to origin
    const origin = new THREE.Vector3(0, 0, 0);

    // Position camera at same distance from origin
    this.camera.position.set(0, 0, currentDistance);
    this.camera.lookAt(origin);
    this.camera.updateMatrixWorld(true);

    // Update controls target
    (this.controls as any).target.copy(origin);
    this.controls.update();

    // Reset and save state to prevent jumps
    this.controls.reset();
    this.controls.saveState();

    log.success(
      Modules.SCENE_MANAGER,
      `Camera reset to origin with distance: ${currentDistance.toFixed(2)}`
    );
  }

  /**
   * Update renderer and camera for window resize
   */
  updateSize(): void {
    // Always use window dimensions for consistency
    // Canvas dimensions can become stale after fullscreen transitions
    const width = window.innerWidth;
    const height = window.innerHeight;

    if (document.fullscreenElement) {
      log.success(Modules.SCENE_MANAGER, `Using fullscreen dimensions: ${width}x${height}`);
    } else {
      log.success(Modules.SCENE_MANAGER, `Using windowed dimensions: ${width}x${height}`);
    }

    // Only update camera if it exists (might be called during init)
    if (this.camera) {
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }

    this.updateRendererSize(width, height);

    // Update post-processing pipeline for new dimensions
    if (this.postProcessing) {
      this.postProcessing.resize(width, height);
    }
  }

  /**
   * Update renderer size and pixel ratio
   */
  private updateRendererSize(width?: number, height?: number): void {
    // Use provided dimensions or fall back to window dimensions
    const w = width || window.innerWidth;
    const h = height || window.innerHeight;

    // Set pixel ratio BEFORE size for correct buffer calculations
    this.renderer.setPixelRatio(window.devicePixelRatio);
    // Let Three.js handle CSS sizing normally
    this.renderer.setSize(w, h); // Allow Three.js to set CSS size

    // Update material uniforms for world-space point sizing (only if camera exists)
    if (this.camera) {
      const fovRadians = (this.camera.fov * Math.PI) / 180;
      const drawingBufferSize = this.renderer.getDrawingBufferSize(new THREE.Vector2());
      materialManager.updateCameraParams(fovRadians, drawingBufferSize);

      // Also update any materials in the scene directly
      this.scene.traverse((object) => {
        if (object instanceof THREE.Points) {
          const material = object.material as THREE.ShaderMaterial;
          if (material.uniforms && material.uniforms.fov && material.uniforms.resolution) {
            material.uniforms.fov.value = fovRadians;
            // Ensure the resolution value is properly set
            if (material.uniforms.resolution.value && material.uniforms.resolution.value.copy) {
              material.uniforms.resolution.value.copy(drawingBufferSize);
            } else {
              material.uniforms.resolution.value = drawingBufferSize.clone();
            }
          }
        }
      });
    }
  }

  /**
   * Update camera FOV with bounds checking
   */
  updateFOV(deltaY: number): void {
    const fovChange = deltaY * config.camera.fovSensitivity;
    this.camera.fov = validateFOV(
      this.camera.fov + fovChange,
      config.camera.fovMin,
      config.camera.fovMax
    );
    this.camera.updateProjectionMatrix();

    // Update material uniforms for world-space point sizing
    const fovRadians = (this.camera.fov * Math.PI) / 180;
    const drawingBufferSize = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    materialManager.updateCameraParams(fovRadians, drawingBufferSize);

    // Also update any materials in the scene directly
    this.scene.traverse((object) => {
      if (object instanceof THREE.Points) {
        const material = object.material as THREE.ShaderMaterial;
        if (material.uniforms && material.uniforms.fov && material.uniforms.resolution) {
          material.uniforms.fov.value = fovRadians;
          material.uniforms.resolution.value.copy(drawingBufferSize);
        }
      }
    });
  }

  /**
   * Update camera clipping planes with validation
   */
  updateClippingPlanes(near: number, far: number): void {
    // Validate near/far relationship
    if (near >= far) {
      log.warning(Modules.SCENE_MANAGER, 'Near plane must be less than far plane');
      return;
    }

    // Warn about Z-buffer precision if ratio is too high
    const ratio = far / near;
    if (ratio > 10000) {
      log.warning(
        Modules.SCENE_MANAGER,
        `High near/far ratio (${ratio.toFixed(0)}:1) may cause Z-buffer precision issues. Consider adjusting clipping planes.`
      );
    }

    // Update camera clipping planes
    this.camera.near = near;
    this.camera.far = far;
    this.camera.updateProjectionMatrix();

    log.info(
      Modules.SCENE_MANAGER,
      `Clipping planes updated - Near: ${near.toFixed(3)}, Far: ${far.toFixed(1)} (ratio: ${ratio.toFixed(0)}:1)`
    );
  }

  /**
   * Auto-adjust clipping planes based on current scene bounds
   */
  autoAdjustClippingPlanes(): { near: number; far: number } {
    // Calculate scene bounding box
    const box = new THREE.Box3().setFromObject(this.scene);

    if (box.isEmpty()) {
      log.warning(Modules.SCENE_MANAGER, 'No scene content for clipping plane calculation');
      return { near: config.camera.near, far: config.camera.far };
    }

    // Get camera distance to scene center
    const center = box.getCenter(new THREE.Vector3());
    const cameraDistance = this.camera.position.distanceTo(center);

    // Use existing utility function
    const { near, far } = calculateClippingPlanes(
      {
        min: { x: box.min.x, y: box.min.y, z: box.min.z },
        max: { x: box.max.x, y: box.max.y, z: box.max.z },
      },
      cameraDistance
    );

    // Apply the calculated planes
    this.updateClippingPlanes(near, far);

    return { near, far };
  }

  /**
   * Update HDR multiplier for all point materials in the scene
   *
   * @param multiplier - New HDR multiplier value (1.0 to 20.0)
   */
  updateHDRMultiplier(multiplier: number): void {
    // Update all materials in the material manager
    materialManager.updateHDRMultiplier(multiplier);

    // Also update any materials in the scene directly
    this.scene.traverse((object) => {
      if (object instanceof THREE.Points) {
        const material = object.material as THREE.ShaderMaterial;
        if (material.uniforms && material.uniforms.hdrMultiplier) {
          material.uniforms.hdrMultiplier.value = multiplier;
        }
      }
    });

    log.success(Modules.RENDERER, `HDR multiplier updated for all point materials: ${multiplier}`);
  }

  /**
   * Clean up all Three.js resources to prevent memory leaks
   *
   * WebGL resources (textures, buffers, shaders) are not automatically
   * garbage collected and must be explicitly disposed. This method ensures
   * proper cleanup of all GPU resources:
   *
   * 1. Post-processing: Dispose HDR render targets and effect composer
   * 2. Controls: Remove event listeners and internal references
   * 3. Renderer: Clean up WebGL context and associated resources
   * 4. Scene objects: Dispose geometry buffers and material shaders
   * 5. Materials: Free texture memory and shader programs
   *
   * Critical for preventing memory leaks in long-running applications.
   */
  dispose(): void {
    // Dispose post-processing resources first
    // This includes HDR render targets, effect composer, and all passes
    this.postProcessing.dispose();

    // Dispose controls - removes all event listeners and internal references
    // This prevents memory leaks from mouse/touch event handlers
    this.controls.dispose();

    // Dispose material manager - cleans up all cached materials
    materialManager.dispose();

    // Dispose renderer - cleans up WebGL context and associated GPU resources
    // This frees vertex buffers, textures, and shader programs
    this.renderer.dispose();

    // Traverse scene graph and dispose all geometry and material resources
    // This is critical because WebGL resources are not garbage collected
    this.scene.traverse((object) => {
      if ('geometry' in object && 'material' in object) {
        const mesh = object as THREE.Mesh;

        // Dispose geometry - frees vertex and index buffers on GPU
        mesh.geometry.dispose();

        // Handle both single materials and material arrays
        if (mesh.material instanceof THREE.Material) {
          // Single material - dispose textures and shader programs
          mesh.material.dispose();
        } else if (Array.isArray(mesh.material)) {
          // Multiple materials - dispose each one individually
          mesh.material.forEach((material) => material.dispose());
        }
      }
    });
  }

  /**
   * Enable or disable automatic camera rotation
   * @param enabled - Whether to enable auto-rotation
   */
  setAutoRotate(enabled: boolean): void {
    this.controls.setAutoRotate(enabled);
    log.custom(
      LogEmoji.SCENE,
      Modules.SCENE_MANAGER,
      `Auto-rotation ${enabled ? 'enabled' : 'disabled'}`
    );
  }

  /**
   * Set the speed of automatic rotation
   * @param speed - Rotation speed (default 2.0 = 30 seconds per orbit at 60fps)
   */
  setAutoRotateSpeed(speed: number): void {
    this.controls.setAutoRotateSpeed(speed);
  }

  /**
   * Get current auto-rotation state
   */
  getAutoRotate(): boolean {
    return this.controls.getAutoRotate();
  }

  /**
   * Switch camera control type
   * @param type - Control type ('orbit', 'arcball', or 'fly')
   */
  setControlType(type: 'orbit' | 'arcball' | 'fly'): void {
    this.controls.setControlType(type);
  }

  /**
   * Get current control type
   */
  getControlType(): 'orbit' | 'arcball' | 'fly' {
    return this.controls.getControlType();
  }

  /**
   * Set fly controls movement speed
   */
  setFlyMovementSpeed(speed: number): void {
    this.controls.setFlyMovementSpeed(speed);
  }

  /**
   * Set fly controls rotation speed
   */
  setFlyRotationSpeed(speed: number): void {
    this.controls.setFlyRotationSpeed(speed);
  }

  /**
   * Set fly controls inertial mode
   */
  setFlyInertialMode(inertial: boolean): void {
    this.controls.setFlyInertialMode(inertial);
  }

  /**
   * Set fly controls damping
   */
  setFlyDamping(damping: number): void {
    this.controls.setFlyDamping(damping);
  }

  /**
   * Set fly controls rotation damping
   */
  setFlyRotationDamping(damping: number): void {
    this.controls.setFlyRotationDamping(damping);
  }
}
