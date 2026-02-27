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
import {
  validateFOV,
  calculateClippingPlanes,
  calculateDistancesToBoundingBox,
  BoundingBox,
  CLIPPING_SAFETY_MARGIN,
  MIN_NEAR_PLANE,
} from './scene-manager-utils';
import { log, Modules, LogEmoji } from '../utils/log';
import { sceneDimsManager } from './scene-dims-manager';

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

  /** Resize debouncing with requestAnimationFrame for smooth resizing */
  private resizeRAF: number | null = null;
  private pendingResize: { width: number; height: number } | null = null;

  /** WebGL context loss handling */
  private isContextLost: boolean = false;
  private contextLostHandler: ((event: Event) => void) | null = null;
  private contextRestoredHandler: ((event: Event) => void) | null = null;

  /** Dynamic clipping planes state */
  private dynamicClippingEnabled: boolean =
    config.renderingControls.defaults.dynamicClippingEnabled;
  private clippingAdaptSpeed: number = config.renderingControls.defaults.clippingAdaptSpeed;
  private smoothedNear: number = config.renderingControls.defaults.near;
  private smoothedFar: number = config.renderingControls.defaults.far;

  /**
   * Create a new scene manager instance.
   *
   * Sets up the EventDispatcher base class. Does not initialize Three.js
   * components - call init() to set up renderer, scene, camera, and controls.
   *
   * @example
   * ```typescript
   * const sceneManager = new SceneManager();
   * await sceneManager.init();  // Initialize Three.js components
   * await sceneManager.loadSceneFromUrl(url);  // Load data
   * ```
   */
  constructor() {
    super();
  }

  /**
   * Initialize complete Three.js rendering pipeline.
   *
   * Sets up all required components in order:
   * 1. Canvas element validation
   * 2. WebGL renderer with HDR support
   * 3. WebGL context loss handling
   * 4. Scene graph
   * 5. Perspective camera
   * 6. Camera controls (orbit/arcball/fly)
   * 7. Post-processing (bloom, HDR tone mapping)
   * 8. Initial canvas sizing
   *
   * Must be called once before using scene manager. Async because HDR
   * detection and post-processing setup involve async operations.
   *
   * @returns Promise that resolves when initialization is complete
   *
   * @example
   * ```typescript
   * const sceneManager = new SceneManager();
   * await sceneManager.init();
   * // Now ready to load scenes and render
   * ```
   */
  async init(): Promise<void> {
    this.setupCanvas();
    this.setupRenderer();
    this.setupContextLossHandling(); // Setup context loss recovery
    this.setupScene();
    this.setupCamera();
    this.setupControls();
    this.setupPostProcessing();

    // Call doUpdateSize() directly during initialization (no debounce needed)
    // This ensures immediate sizing without waiting for requestAnimationFrame
    this.doUpdateSize(window.innerWidth, window.innerHeight);
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
   * Setup WebGL context loss and restoration handling
   *
   * WebGL context can be lost due to:
   * - GPU driver crashes or resets
   * - System sleep/hibernate
   * - Too many contexts (browser limit)
   * - Out of GPU memory
   *
   * This setup ensures the app can recover gracefully instead of crashing.
   */
  private setupContextLossHandling(): void {
    const canvas = this.canvasElement;

    // Handle context loss - prevent default and prepare for restoration
    this.contextLostHandler = (event: Event) => {
      event.preventDefault(); // Required to allow context restoration
      this.isContextLost = true;

      log.error(
        Modules.SCENE_MANAGER,
        'WebGL context lost! This can happen due to GPU driver issues, system sleep, or memory pressure.'
      );

      showError(
        'Graphics context lost - attempting to restore. This can happen if your GPU driver crashes or the system runs out of video memory. The app will try to recover automatically.'
      );
    };

    // Handle context restoration - recreate all WebGL resources
    this.contextRestoredHandler = async (_event: Event) => {
      log.info(Modules.SCENE_MANAGER, 'WebGL context restored - recreating resources...');

      try {
        // Mark context as restored
        this.isContextLost = false;

        // Force renderer to recreate its internal state
        this.renderer.resetState();

        // Recreate post-processing resources (render targets, shaders)
        // Note: This is handled by PostProcessingManager's dispose/recreate cycle
        // For now, we log that resources need recreation
        log.info(
          Modules.SCENE_MANAGER,
          'Post-processing resources will be recreated on next render'
        );

        // Trigger a render to force resource recreation
        this.dispatchEvent({ type: 'change' });

        hideLoadingIndicator();
        log.success(Modules.SCENE_MANAGER, 'WebGL context successfully restored');
      } catch (error) {
        log.error(Modules.SCENE_MANAGER, 'Failed to restore WebGL context:', error);
        showError('Failed to restore graphics context. Please refresh the page to continue.');
      }
    };

    // Add event listeners
    canvas.addEventListener('webglcontextlost', this.contextLostHandler, false);
    canvas.addEventListener('webglcontextrestored', this.contextRestoredHandler, false);

    log.info(Modules.SCENE_MANAGER, 'WebGL context loss handling initialized');
  }

  /**
   * Check if WebGL context is currently lost
   */
  public isWebGLContextLost(): boolean {
    return this.isContextLost;
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
      config.renderingControls.defaults.fov, // Field of view (60 degrees)
      width / height, // Aspect ratio (canvas width/height)
      config.renderingControls.defaults.near, // Near clipping plane (0.1 units)
      config.renderingControls.defaults.far // Far clipping plane (1000 units)
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

      // NOTE: Material parameters were already updated BEFORE loadScene() above
      // Materials created during loading already have correct FOV/resolution
      // No need to update again - this would be redundant work

      // Auto-adjust clipping planes using scene bounds from metadata
      // This uses position_bounds stored in zarr, which represents the full dataset extent
      // and doesn't require waiting for point data to load
      this.autoAdjustClippingPlanes();

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
      // Handle Mesh, Points, and InstancedMesh (used for lines)
      if (
        obj instanceof THREE.Mesh ||
        obj instanceof THREE.Points ||
        obj instanceof THREE.InstancedMesh
      ) {
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
   * Center camera on the bounding box of all visible objects in scene.
   *
   * Computes the bounding box of all Points and InstancedMesh objects,
   * positions camera to view entire scene, and updates controls target.
   * Skips centering if scene is too small (< 1.0 units) or has too few
   * points (< 100) to avoid awkward positioning.
   *
   * Called automatically after scene loading if dataset has reasonable size.
   * Can be called manually via F key to recenter after navigation.
   *
   * @example
   * ```typescript
   * // After loading scene
   * await sceneManager.loadSceneFromUrl(url);
   * sceneManager.centerCameraOnScene();
   * // Camera now frames entire dataset
   * ```
   */
  public centerCameraOnScene(): void {
    // Ensure world matrices are up to date before computing bounds
    this.scene.updateMatrixWorld(true);

    // Create a bounding box that encompasses all visible objects
    const box = new THREE.Box3();
    let totalPrimitiveCount = 0;

    // Traverse the scene and expand the box to include all geometries
    this.scene.traverse((object) => {
      // Handle Points objects (point clouds)
      if (object instanceof THREE.Points) {
        const geometry = object.geometry;

        // For points, compute bounding box from position attribute
        const positions = geometry.attributes.position;
        if (positions && positions.count > 0) {
          totalPrimitiveCount += positions.count;

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

      // Handle InstancedMesh objects (legacy) and instanced Mesh objects
      // Lines and GSplats use THREE.Mesh with InstancedBufferGeometry (not InstancedMesh)
      // to avoid exceeding WebGL's 16 attribute location limit
      const isLineMesh =
        object instanceof THREE.Mesh &&
        object.userData?.nodeType === 'lines' &&
        object.geometry instanceof THREE.InstancedBufferGeometry;

      const isGSplatMesh =
        object instanceof THREE.Mesh &&
        object.userData?.nodeType === 'gsplats' &&
        object.geometry instanceof THREE.InstancedBufferGeometry;

      if (object instanceof THREE.InstancedMesh || isLineMesh || isGSplatMesh) {
        const geometry = object.geometry;

        // For instanced meshes/geometries, use the precomputed bounding box
        if (!geometry.boundingBox) {
          geometry.computeBoundingBox();
        }

        if (geometry.boundingBox) {
          // Get instance count (InstancedMesh has count, InstancedBufferGeometry has instanceCount)
          const instanceCount =
            object instanceof THREE.InstancedMesh
              ? object.count
              : ((geometry as THREE.InstancedBufferGeometry).instanceCount ?? 0);
          totalPrimitiveCount += instanceCount;

          const tempBox = geometry.boundingBox.clone();

          // Apply object's world transform
          tempBox.applyMatrix4(object.matrixWorld);

          // Only include if box has valid size (not empty)
          if (!tempBox.isEmpty()) {
            box.union(tempBox);
          }
        }
      }
    });

    // Only center camera if we have a reasonable scene
    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);

      // Don't center if bounding box is too small or too few primitives
      // This prevents awkward camera positioning on edge cases
      if (maxDim < 1.0 || totalPrimitiveCount < 100) {
        // Keep default camera position for better user experience
        log.warning(
          Modules.SCENE_MANAGER,
          `Scene too small for auto-centering (size: ${maxDim.toFixed(2)}, primitives: ${totalPrimitiveCount})`
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

      // Save the new centered state as the default
      // NOTE: Do NOT call reset() before saveState() - that would undo the centering!
      // reset() reverts to the previously saved state, defeating the purpose
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
   * Get current camera target center point.
   *
   * Returns either the bounding box center (if auto-centered) or origin
   * (0,0,0) if using default positioning. The returned vector is a clone,
   * safe to modify.
   *
   * @returns Current center as Vector3 (bounding box center or origin)
   */
  public getCurrentCenter(): THREE.Vector3 {
    if (this.isCenteredOnBoundingBox) {
      return this.lastBoundingBoxCenter.clone();
    }
    return new THREE.Vector3(0, 0, 0); // Origin
  }

  /**
   * Get controls manager for camera interaction.
   *
   * Provides access to orbit, arcball, and fly controls for advanced
   * camera manipulation.
   *
   * @returns ControlsManager instance managing camera controls
   */
  public getControlsManager(): ControlsManager {
    return this.controls;
  }

  /**
   * Toggle camera centering between origin and bounding box center.
   *
   * Switches between two centering modes:
   * - Origin (0,0,0): Default Three.js behavior
   * - Bounding box center: Computed from all visible objects
   *
   * Useful when dataset is not centered at origin or when you want to
   * return to default camera position.
   *
   * @example
   * ```typescript
   * // Switch to origin centering
   * sceneManager.toggleCentering();
   * ```
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

    // Save the new origin-centered state as the default
    // NOTE: Do NOT call reset() before saveState() - that would undo the centering!
    this.controls.saveState();

    log.success(
      Modules.SCENE_MANAGER,
      `Camera reset to origin with distance: ${currentDistance.toFixed(2)}`
    );
  }

  /**
   * Update renderer and camera for window resize (debounced)
   *
   * This method debounces resize events using requestAnimationFrame to prevent
   * excessive WebGL buffer reallocations during window dragging. Multiple rapid
   * resize events are coalesced into a single update on the next frame.
   */
  /**
   * Update canvas size and camera aspect ratio for window resize.
   *
   * Handles window resize events with debouncing via requestAnimationFrame.
   * Updates:
   * - Canvas dimensions to match window size
   * - Camera aspect ratio to prevent distortion
   * - Renderer viewport
   * - Post-processing effect sizes
   *
   * Called automatically on window resize. Debouncing ensures smooth
   * resize without excessive recomputations.
   *
   * @example
   * ```typescript
   * // Manual resize (usually not needed, window resize auto-triggers)
   * sceneManager.updateSize();
   * ```
   */
  updateSize(): void {
    // Store the latest dimensions
    this.pendingResize = {
      width: window.innerWidth,
      height: window.innerHeight,
    };

    // Cancel any pending resize
    if (this.resizeRAF !== null) {
      cancelAnimationFrame(this.resizeRAF);
    }

    // Schedule resize for next frame (coalesces multiple events)
    this.resizeRAF = requestAnimationFrame(() => {
      if (!this.pendingResize) return;

      this.doUpdateSize(this.pendingResize.width, this.pendingResize.height);
      this.pendingResize = null;
      this.resizeRAF = null;
    });
  }

  /**
   * Actual resize logic (called once per frame at most)
   */
  private doUpdateSize(width: number, height: number): void {
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
      // Material manager updates all registered materials (no scene traversal needed)
      materialManager.updateCameraParams(fovRadians, drawingBufferSize);
    }
  }

  /**
   * Update pixel ratio for adaptive performance optimization.
   *
   * Uses setSize with updateStyle=false to keep canvas CSS size constant
   * while reducing the internal buffer resolution for better performance.
   *
   * This method is called by the AdaptiveDPRManager when FPS drops below
   * acceptable thresholds.
   *
   * @param dpr - The new device pixel ratio to use
   */
  public setAdaptivePixelRatio(dpr: number): void {
    const w = window.innerWidth;
    const h = window.innerHeight;

    // Set new pixel ratio
    this.renderer.setPixelRatio(dpr);

    // Update size with updateStyle=false to keep CSS dimensions constant
    // This allows the internal render buffer to be smaller while the canvas
    // still fills the viewport
    this.renderer.setSize(w, h, false);

    // Update post-processing pipeline for new buffer dimensions
    if (this.postProcessing) {
      this.postProcessing.resize(w, h);

      // Scale noise parameters based on DPR to maintain perceptual consistency
      // At lower DPR, each pixel covers more area, so noise should be scaled down
      const nativeDPR = window.devicePixelRatio;
      const normalizedDPR = dpr / nativeDPR; // 1.0 at native, <1.0 when reduced
      this.postProcessing.setDPRScale(normalizedDPR);
    }

    // Update material uniforms for world-space point sizing
    if (this.camera) {
      const fovRadians = (this.camera.fov * Math.PI) / 180;
      const drawingBufferSize = this.renderer.getDrawingBufferSize(new THREE.Vector2());
      materialManager.updateCameraParams(fovRadians, drawingBufferSize);
    }

    log.update(
      Modules.SCENE_MANAGER,
      `Adaptive DPR: ${dpr.toFixed(2)} (buffer: ${Math.round(w * dpr)}x${Math.round(h * dpr)})`
    );
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
    // Material manager updates all registered materials (no scene traversal needed)
    materialManager.updateCameraParams(fovRadians, drawingBufferSize);
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
   * Auto-adjust clipping planes based on scene bounds from metadata.
   *
   * This uses the position_bounds stored in zarr metadata, which represents
   * the full dataset extent computed at compile time. This is more reliable
   * than computing bounds from loaded geometry because:
   * 1. It includes the full dataset, not just currently loaded points
   * 2. It works correctly for nD data (we project to display dimensions)
   * 3. It's available immediately without waiting for data to load
   *
   * Falls back to geometry-based calculation if metadata bounds are not available.
   */
  autoAdjustClippingPlanes(): { near: number; far: number } {
    // Get camera position for distance calculations
    const cameraPos = {
      x: this.camera.position.x,
      y: this.camera.position.y,
      z: this.camera.position.z,
    };

    // Try to get scene bounds from metadata first
    const sceneBounds = this.getSceneBoundsFromMetadata();

    if (sceneBounds) {
      // Use unified utility function with camera position
      const { near, far } = calculateClippingPlanes(sceneBounds, cameraPos);

      // Apply the calculated planes
      this.updateClippingPlanes(near, far);

      log.success(
        Modules.SCENE_MANAGER,
        `Clipping planes set from metadata bounds (near: ${near.toFixed(4)}, far: ${far.toFixed(1)})`
      );

      return { near, far };
    }

    // Fallback: Calculate scene bounding box from loaded geometry
    const box = new THREE.Box3().setFromObject(this.scene);

    if (box.isEmpty()) {
      log.warning(Modules.SCENE_MANAGER, 'No scene content for clipping plane calculation');
      return {
        near: config.renderingControls.defaults.near,
        far: config.renderingControls.defaults.far,
      };
    }

    // Use unified utility function with camera position
    const { near, far } = calculateClippingPlanes(
      {
        min: { x: box.min.x, y: box.min.y, z: box.min.z },
        max: { x: box.max.x, y: box.max.y, z: box.max.z },
      },
      cameraPos
    );

    // Apply the calculated planes
    this.updateClippingPlanes(near, far);

    return { near, far };
  }

  /**
   * Update dynamic clipping planes using exponential smoothing.
   *
   * Called each frame by AnimationController to smoothly adjust clipping planes
   * based on camera position relative to scene bounds. This prevents clipping
   * artifacts when navigating and maintains optimal Z-buffer precision.
   *
   * Uses unified calculation with CLIPPING_SAFETY_MARGIN (50%) for consistency
   * with autoAdjustClippingPlanes(). When inside the bounding box, uses distance
   * to nearest surface to prevent clipping nearby geometry.
   */
  updateDynamicClippingPlanes(): void {
    if (!this.dynamicClippingEnabled) return;

    // Get scene bounds
    const bounds = this.getSceneBoundsFromMetadata();
    if (!bounds) return;

    // Get camera position
    const cameraPos = {
      x: this.camera.position.x,
      y: this.camera.position.y,
      z: this.camera.position.z,
    };

    // Calculate distances from camera to bounding box (includes face centers)
    const { nearDist, farDist, isInside } = calculateDistancesToBoundingBox(cameraPos, bounds);

    // Calculate optimal clipping planes using unified margin (50%)
    let optimalNear: number;
    if (isInside) {
      // When inside the bounding box, use minimum near plane
      // This ensures we can see all geometry around us without clipping
      optimalNear = MIN_NEAR_PLANE;
    } else {
      // When outside, use nearest point distance with margin
      // margin of 0.5 means near = nearDist * 0.5
      optimalNear = Math.max(MIN_NEAR_PLANE, nearDist * (1 - CLIPPING_SAFETY_MARGIN));
    }

    // Far plane: farthest point plus margin (~150% of distance)
    const optimalFar = farDist * (1 + CLIPPING_SAFETY_MARGIN);

    // Exponential smoothing: new = (1-α)*current + α*optimal
    const α = this.clippingAdaptSpeed;
    this.smoothedNear = (1 - α) * this.smoothedNear + α * optimalNear;
    this.smoothedFar = (1 - α) * this.smoothedFar + α * optimalFar;

    // Apply safety clamps
    this.smoothedNear = Math.max(MIN_NEAR_PLANE, this.smoothedNear);

    // Prevent excessive far/near ratio (Z-buffer precision)
    const maxRatio = 100000;
    if (this.smoothedFar / this.smoothedNear > maxRatio) {
      this.smoothedNear = this.smoothedFar / maxRatio;
    }

    // Only update camera if values changed significantly (>0.1%)
    const nearChanged = Math.abs(this.camera.near - this.smoothedNear) / this.camera.near > 0.001;
    const farChanged = Math.abs(this.camera.far - this.smoothedFar) / this.camera.far > 0.001;

    if (nearChanged || farChanged) {
      this.camera.near = this.smoothedNear;
      this.camera.far = this.smoothedFar;
      this.camera.updateProjectionMatrix();
    }
  }

  // NOTE: calculateDistancesToBounds removed - now using calculateDistancesToBoundingBox
  // from scene-manager-utils.ts which includes face centers for better accuracy

  /**
   * Set dynamic clipping configuration.
   *
   * @param enabled - Whether dynamic clipping is enabled
   * @param adaptSpeed - Exponential smoothing factor (0.01-0.5)
   */
  setDynamicClipping(enabled: boolean, adaptSpeed?: number): void {
    this.dynamicClippingEnabled = enabled;
    if (adaptSpeed !== undefined) {
      this.clippingAdaptSpeed = Math.max(0.01, Math.min(0.5, adaptSpeed));
    }

    log.info(
      Modules.SCENE_MANAGER,
      `Dynamic clipping ${enabled ? 'enabled' : 'disabled'}${adaptSpeed !== undefined ? ` (adapt speed: ${this.clippingAdaptSpeed})` : ''}`
    );
  }

  /**
   * Get current dynamic clipping state.
   */
  getDynamicClippingState(): { enabled: boolean; adaptSpeed: number; near: number; far: number } {
    return {
      enabled: this.dynamicClippingEnabled,
      adaptSpeed: this.clippingAdaptSpeed,
      near: this.smoothedNear,
      far: this.smoothedFar,
    };
  }

  /**
   * Get 3D bounding box from scene metadata, projecting nD bounds to display dimensions.
   *
   * @returns 3D bounding box or null if metadata bounds not available
   */
  private getSceneBoundsFromMetadata(): BoundingBox | null {
    // Find the root group with position bounds
    const foundBounds = this.findPositionBoundsInScene();

    if (!foundBounds) {
      return null;
    }

    // Get display dimensions from scene dimensions manager
    const dims = sceneDimsManager.getDims();
    const displayDims: number[] = dims?.displayed ?? [0, 1, 2];

    // Project nD bounds to 3D using display dimensions
    const minBounds = foundBounds.min;
    const maxBounds = foundBounds.max;
    const min3D = { x: 0, y: 0, z: 0 };
    const max3D = { x: 0, y: 0, z: 0 };

    // Map display dimensions to X, Y, Z
    if (displayDims.length > 0 && displayDims[0] < minBounds.length) {
      min3D.x = minBounds[displayDims[0]];
      max3D.x = maxBounds[displayDims[0]];
    }
    if (displayDims.length > 1 && displayDims[1] < minBounds.length) {
      min3D.y = minBounds[displayDims[1]];
      max3D.y = maxBounds[displayDims[1]];
    }
    if (displayDims.length > 2 && displayDims[2] < minBounds.length) {
      min3D.z = minBounds[displayDims[2]];
      max3D.z = maxBounds[displayDims[2]];
    }

    return { min: min3D, max: max3D };
  }

  /**
   * Search for position bounds in the scene graph
   */
  private findPositionBoundsInScene(): { min: number[]; max: number[] } | null {
    let result: { min: number[]; max: number[] } | null = null;

    this.scene.traverse((object) => {
      if (result) return; // Already found

      const bounds = object.userData?.positionBounds;
      if (bounds && Array.isArray(bounds.min) && Array.isArray(bounds.max)) {
        result = { min: bounds.min, max: bounds.max };
      }
    });

    return result;
  }

  /**
   * Update HDR multiplier for all point materials in the scene
   *
   * @param multiplier - New HDR multiplier value (1.0 to 20.0)
   */
  updateHDRMultiplier(multiplier: number): void {
    // Material manager updates all registered materials (no scene traversal needed)
    materialManager.updateHDRMultiplier(multiplier);
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
  /**
   * Clean up all Three.js resources and prevent memory leaks.
   *
   * Disposes:
   * - All geometries and materials in scene
   * - Renderer and render targets
   * - Post-processing effects
   * - Camera controls
   * - Event listeners (context loss, resize, etc.)
   *
   * Should be called when scene manager is no longer needed. After calling
   * dispose(), the scene manager cannot be reused - create a new instance.
   *
   * @example
   * ```typescript
   * // During application teardown
   * sceneManager.dispose();
   * // All WebGL resources released
   * ```
   */
  dispose(): void {
    // Cancel any pending resize operations to prevent memory leaks
    if (this.resizeRAF !== null) {
      cancelAnimationFrame(this.resizeRAF);
      this.resizeRAF = null;
    }
    this.pendingResize = null;

    // Remove WebGL context loss event listeners
    if (this.contextLostHandler) {
      this.canvasElement.removeEventListener('webglcontextlost', this.contextLostHandler);
      this.contextLostHandler = null;
    }
    if (this.contextRestoredHandler) {
      this.canvasElement.removeEventListener('webglcontextrestored', this.contextRestoredHandler);
      this.contextRestoredHandler = null;
    }

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
