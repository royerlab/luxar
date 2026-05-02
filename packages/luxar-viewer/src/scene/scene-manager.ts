// Scene, renderer, and camera management for the Luxar scene player
//
// This module handles all Three.js setup and 3D graphics configuration:
// - WebGL renderer initialization with optimal settings
// - Camera setup with proper projection and positioning
// - Camera controls for intuitive 3D navigation
// - Scene graph management and Zarr data loading
// - Resource disposal for memory management

import * as THREE from 'three';
import { ControlsManager } from '../controls/controls-manager';
import { loadScene } from '../data';
import type { LoaderConfig } from '../data/data-loader-types';
import { showLoadingIndicator, hideLoadingIndicator, showError } from '../ui/helpers';
import { config } from '../config';
import { extractCameraOverrides, extractBackgroundColor } from '../config/viewer-config-utils';
import type { ZarrViewerConfig } from '../types/zarr';
import { PostProcessingManager } from '../rendering/post-processing-manager';
import { materialManager } from '../rendering/material-manager';
import {
  detectHDRCapabilities,
  configureHDRRenderer,
  logHDRCapabilities,
} from '../utils/hdr-detection';
import {
  validateFOV,
  calculateCameraDistance,
  getBoundingBoxDiagonal,
  getBoundingBoxCenter,
  BoundingBox,
  BoundingSphere,
  boundingBoxToSphere,
  calculateClippingPlanesFromSphere,
  SPHERE_SAFETY_EXPANSION,
  MIN_NEAR_PLANE,
} from './scene-manager-utils';
import { log, Modules, LogEmoji } from '../utils/log';
import { sceneDimsManager } from './scene-dims-manager';
import {
  type LuxarCamera,
  isPerspectiveCamera,
  isOrthographicCamera,
  getCameraFovRadians,
  updateCameraAspect,
  getOrthoFrustumHeight,
} from './camera-utils';
import type { ControlType } from '../controls/controls-manager';

/**
 * How far the user can zoom in or out relative to the "scene fits in view" distance/zoom.
 * A value of 100 means 100x zoom-in and 100x zoom-out from the auto-framed view.
 */
const ZOOM_RANGE_FACTOR = 100;

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
 * - LuxarOrbitControls provide quaternion-based camera movement (no gimbal lock)
 * - HDR post-processing with ACES tone mapping and bloom
 * - Custom Gaussian point shaders for enhanced visual quality
 * - Automatic canvas resizing for responsive design
 */
export class SceneManager extends THREE.EventDispatcher<{
  change: {};
  'camera-changed': {};
}> {
  /** Three.js WebGL renderer - handles all GPU-accelerated rendering */
  public renderer!: THREE.WebGLRenderer;

  /** Three.js scene graph - container for all 3D objects and lights */
  public scene!: THREE.Scene;

  /** Camera for 3D viewing (perspective or orthographic) */
  public camera!: LuxarCamera;

  /** ControlsManager - manages different camera control types (orbit, fly, ortho) */
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

  /** When true, resize events are suppressed (used during recording to prevent resolution changes) */
  public resizeLocked: boolean = false;

  /** Cached ortho zoom level to avoid redundant material updates during panning */
  private lastOrthoZoom: number = 1;

  /** Dynamic clipping planes state */
  private dynamicClippingEnabled: boolean =
    config.renderingControls.defaults.dynamicClippingEnabled;

  /** Cached scene bounds (invalidated on scene load/clear, lazily recomputed) */
  private _cachedBounds: BoundingBox | null = null;
  private _cachedSphere: BoundingSphere | null = null;
  private _cachedNearCull: number = 0.1;

  /** Reusable Vector2 for getDrawingBufferSize (avoids per-call allocation) */
  private readonly _bufferSize = new THREE.Vector2();

  /** Current FOV in degrees (perspective) or the default FOV (orthographic). */
  get currentFov(): number {
    return isPerspectiveCamera(this.camera)
      ? this.camera.fov
      : config.renderingControls.defaults.fov;
  }

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
   * await sceneManager.loadSceneData(url);  // Load data
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
   * 6. Camera controls (orbit/fly/ortho)
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
  /**
   * When true, additional hardware/runtime info is logged at startup.
   * Set via {@link init}'s `debug` flag (forwarded from `?debug` URL parameter).
   */
  private debug: boolean = false;

  /**
   * Initialize the renderer pipeline.
   *
   * @param options.canvas - The HTMLCanvasElement to render into. Callers
   *   resolve this themselves (`document.getElementById(...)` in the
   *   standalone app's main.ts; arbitrary container child for embedders).
   *   SceneManager performs no DOM lookups of its own.
   * @param options.debug - Verbose hardware/runtime logging.
   */
  async init(options: { canvas: HTMLCanvasElement; debug?: boolean }): Promise<void> {
    this.canvasElement = options.canvas;
    this.debug = options.debug ?? false;
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

    // Create WebGL renderer using configuration values.
    // Shared attributes (antialias, powerPreference, etc.) come from webgl.context;
    // renderer-specific settings (precision, shadowMap, etc.) come from webgl.renderer.
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvasElement, // Use our pre-existing canvas element
      context: gl || undefined, // Use our HDR context if available
      // Shared attributes from context config
      alpha: config.webgl.context.alpha,
      antialias: config.webgl.context.antialias,
      depth: config.webgl.context.depth,
      stencil: config.webgl.context.stencil,
      powerPreference: config.webgl.context.powerPreference,
      preserveDrawingBuffer: config.webgl.context.preserveDrawingBuffer,
      premultipliedAlpha: config.webgl.context.premultipliedAlpha,
      // Renderer-specific settings
      ...config.webgl.renderer,
    });

    // NOTE: We don't append renderer.domElement because we're using the
    // existing HTML canvas. Page-chrome styling (body margin/overflow,
    // background) is the responsibility of the host page (index.html for
    // the standalone app), not of SceneManager.

    // Report hardware point size limits when debug logging is requested.
    if (this.debug) {
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

    // Immediately clear to the scene background color to avoid a white flash
    // before the first frame renders (alpha:false makes the canvas opaque white by default)
    this.renderer.setClearColor(config.scene.backgroundColor);
    this.renderer.clear();
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

        // Recreate post-processing resources that own WebGL render targets.
        if (this.postProcessing) {
          this.postProcessing.dispose();
          this.setupPostProcessing();
        }
        this.updateRendererSize();

        // Trigger a render to force Three.js material/program resource recreation.
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
      // Ortho zoom changes camera.zoom, which affects material frustum height.
      // Only update materials if zoom actually changed (skip during panning).
      if (isOrthographicCamera(this.camera) && this.camera.zoom !== this.lastOrthoZoom) {
        this.lastOrthoZoom = this.camera.zoom;
        this.updateMaterialsForCurrentCamera();
      }
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
   * Initialize the HDR post-processing pipeline (pmndrs/postprocessing).
   *
   * Creates an EffectComposer with 16-bit float buffers, bloom, tone mapping,
   * and optional AA effects. See PostProcessingManager for the full pipeline.
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

    log.success(Modules.POST_PROCESSING, 'HDR pipeline initialized');
  }

  /**
   * Load scene data from Zarr source.
   *
   * Cache and prefetch flags propagate through `loaderConfig` from
   * LuxarApp (originally derived from `?no-cache`/`?cache-debug`/etc URL
   * parameters in main.ts).
   */
  async loadSceneData(src: string, loaderConfig?: LoaderConfig): Promise<void> {
    showLoadingIndicator();

    try {
      // Clear existing scene content (keep lights and background)
      this.clearSceneContent();

      // Reset controls to default state before loading new content
      this.resetControls();

      // Update material manager BEFORE loading scene so materials are created with correct params
      if (this.camera && this.renderer) {
        this.updateMaterialsForCurrentCamera();
      }

      const root = await loadScene(src, loaderConfig);
      hideLoadingIndicator();
      this.scene.add(root);
      this.invalidateBoundsCache();

      // NOTE: Material parameters were already updated BEFORE loadScene() above
      // Materials created during loading already have correct FOV/resolution
      // No need to update again - this would be redundant work

      // Apply viewer config from zarr (camera position, background color)
      this.applyZarrViewerConfig(root);

      // Auto-frame camera to fit scene contents, unless the zarr author specified a camera position.
      // Only an explicit position suppresses auto-framing — a target/targetNode alone means the
      // author wants the orbit pivot set but still expects the camera to be at a sensible distance.
      const viewerConfig = root.userData?.viewerConfig as ZarrViewerConfig | undefined;
      const camOverrides = viewerConfig ? extractCameraOverrides(viewerConfig) : {};
      const hasAuthorTarget = !!(camOverrides.target || camOverrides.targetNode);

      if (!camOverrides.position) {
        // No author camera position — auto-frame using metadata bounds.
        // If the author set a target, preserve it as the look-at point
        // instead of overwriting with bounding box center.
        this.autoFrameCamera(hasAuthorTarget);
      }

      // Auto-adjust clipping planes using scene bounds from metadata
      // (must run AFTER autoFrameCamera since camera position affects clipping)
      this.autoAdjustClippingPlanes();

      log.info(Modules.SCENE_MANAGER, 'Scene loaded. Press F to re-center camera on bounding box.');
    } catch (error) {
      hideLoadingIndicator();
      log.error(Modules.SCENE_MANAGER, 'Failed to load scene:', error);
      showError(`Failed to load scene from "${src}". Please check the path and try again.`);
      throw error;
    }
  }

  /**
   * Get the viewer config from the loaded scene's root group userData.
   */
  getSceneViewerConfig(): ZarrViewerConfig | undefined {
    const root = this.scene.children.find((c) => c.name === 'LuxarScene');
    return root?.userData?.viewerConfig as ZarrViewerConfig | undefined;
  }

  /**
   * Apply viewer config from zarr (camera position/target/up, background color).
   * Camera config is applied on every load — it's the data author's intended "home" view.
   */
  private applyZarrViewerConfig(root: THREE.Group): void {
    const viewerConfig = root.userData?.viewerConfig as ZarrViewerConfig | undefined;
    if (!viewerConfig) return;

    // Apply camera position/target/up
    const camOverrides = extractCameraOverrides(viewerConfig);
    if (camOverrides.position) {
      this.camera.position.set(
        camOverrides.position.x,
        camOverrides.position.y,
        camOverrides.position.z
      );
    }

    // target_node takes precedence over explicit target coordinates.
    // Use setTarget() (not lookAt()) to avoid an intermediate update() that
    // would snap the camera back before reinitialize() derives the new orbit state.
    if (camOverrides.targetNode) {
      const resolved = this.resolveTargetNode(root, camOverrides.targetNode);
      if (resolved) {
        this.controls.setTarget(resolved);
        log.info(
          Modules.SCENE_MANAGER,
          `Resolved target_node '${camOverrides.targetNode}' to (${resolved.x.toFixed(2)}, ${resolved.y.toFixed(2)}, ${resolved.z.toFixed(2)})`
        );
      } else {
        log.warning(
          Modules.SCENE_MANAGER,
          `target_node '${camOverrides.targetNode}' not found in scene graph`
        );
      }
    } else if (camOverrides.target) {
      const targetVec = new THREE.Vector3(
        camOverrides.target.x,
        camOverrides.target.y,
        camOverrides.target.z
      );
      this.controls.setTarget(targetVec);
    }

    if (camOverrides.up) {
      this.camera.up.set(camOverrides.up.x, camOverrides.up.y, camOverrides.up.z);
      // Sync camera.quaternion with the new up vector so that
      // reinitialize() (which reads quaternion, not camera.up) picks up the
      // author's roll.  Use the current orbit target as the look-at point.
      this.camera.lookAt(this.controls.getFocusTarget());
    }
    if (
      camOverrides.position ||
      camOverrides.target ||
      camOverrides.targetNode ||
      camOverrides.up
    ) {
      this.camera.updateMatrixWorld(true);
      this.controls.reinitialize();
      this.controls.update();
      log.info(Modules.SCENE_MANAGER, 'Applied camera config from zarr viewer_config');
    }

    // Apply background color
    const bgColor = extractBackgroundColor(viewerConfig);
    if (bgColor) {
      this.scene.background = new THREE.Color(bgColor);
      log.info(Modules.SCENE_MANAGER, `Applied background color from zarr: ${bgColor}`);
    }
  }

  /**
   * Find a named node in the scene graph and return its bounding box center.
   *
   * @param root - Scene graph root to search
   * @param nodeName - Name of the node to find
   * @returns Bounding box center, or null if node not found
   */
  private resolveTargetNode(root: THREE.Group, nodeName: string): THREE.Vector3 | null {
    let targetObject: THREE.Object3D | null = null;

    root.traverse((obj) => {
      if (obj.name === nodeName && !targetObject) {
        targetObject = obj;
      }
    });

    if (!targetObject) return null;

    const box = new THREE.Box3().setFromObject(targetObject);
    if (box.isEmpty()) return null;

    const center = new THREE.Vector3();
    box.getCenter(center);
    return center;
  }

  /**
   * Clear all loaded content from the scene, keeping lights and background
   */
  private clearSceneContent(): void {
    this.invalidateBoundsCache();

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
   * Skips centering if no geometry is found in the scene.
   *
   * Called automatically after scene loading if dataset has reasonable size.
   * Can be called manually via F key to recenter after navigation.
   *
   * @example
   * ```typescript
   * // After loading scene
   * await sceneManager.loadSceneData(url);
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

      // Handle both THREE.InstancedMesh and Mesh + InstancedBufferGeometry objects
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

    // Only center camera if we have geometry
    if (!box.isEmpty() && totalPrimitiveCount > 0) {
      const size = box.getSize(new THREE.Vector3());

      // Update scale-aware controls from geometry bounding box
      const diagonal = size.length();
      if (diagonal > 0) {
        this.controls.setSceneScale(diagonal);
      }

      const center = box.getCenter(new THREE.Vector3());

      // Store the center for later use
      this.lastBoundingBoxCenter.copy(center);

      // Compute optimal distance using FOV-aware calculation
      const geoBounds: BoundingBox = {
        min: { x: box.min.x, y: box.min.y, z: box.min.z },
        max: { x: box.max.x, y: box.max.y, z: box.max.z },
      };

      if (isPerspectiveCamera(this.camera)) {
        const cameraConfig = {
          fov: this.camera.fov,
          aspect: this.camera.aspect,
          near: this.camera.near,
          far: this.camera.far,
        };
        const distance = calculateCameraDistance(geoBounds, cameraConfig);
        this.camera.position.set(center.x, center.y, center.z + distance);

        // Set distance limits relative to the scene-fitting distance
        this.controls.setDistanceLimits(distance / ZOOM_RANGE_FACTOR, distance * ZOOM_RANGE_FACTOR);
      } else if (isOrthographicCamera(this.camera)) {
        const frustumHeight = this.camera.top - this.camera.bottom;
        const frustumWidth = this.camera.right - this.camera.left;
        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 0 && frustumHeight > 0 && frustumWidth > 0) {
          const fitRatio = config.scene.defaultFitRatio;
          const zoomH = frustumHeight / (maxDim / fitRatio);
          const zoomW = frustumWidth / (maxDim / fitRatio);
          this.camera.zoom = Math.min(zoomH, zoomW);
          this.camera.updateProjectionMatrix();

          // Set zoom limits relative to the scene-fitting zoom (100x in each direction)
          this.controls.setZoomLimits(
            this.camera.zoom / ZOOM_RANGE_FACTOR,
            this.camera.zoom * ZOOM_RANGE_FACTOR
          );
        }
        this.camera.position.set(center.x, center.y, center.z + diagonal);
      }

      // Point camera at the center
      this.camera.lookAt(center);
      this.camera.updateMatrixWorld(true);

      // Sync orbit controls with the new camera state.
      // CRITICAL: Set target first, then reinitialize() so the controls re-derive
      // their internal distance from the camera position we just set.
      this.controls.setTarget(center);
      this.controls.reinitialize();
      this.controls.update();

      // Save the new centered state as the default
      // NOTE: Do NOT call reset() before saveState() - that would undo the centering!
      // reset() reverts to the previously saved state, defeating the purpose
      this.controls.saveState();

      log.success(Modules.CONTROLS, 'Controls target updated and state saved');

      log.success(
        Modules.SCENE_MANAGER,
        `Camera centered on scene (center: [${center.x.toFixed(2)}, ${center.y.toFixed(2)}, ${center.z.toFixed(2)}])`
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
   * Provides access to orbit, fly, and ortho controls for advanced
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
    // Get current camera distance from target. getFocusTarget() returns a
    // clone, so it is safe to use as a one-shot read.
    const currentDistance = this.camera.position.distanceTo(this.controls.getFocusTarget());

    // Reset target to origin
    const origin = new THREE.Vector3(0, 0, 0);

    // Position camera at same distance from origin
    this.camera.position.set(0, 0, currentDistance);
    this.camera.lookAt(origin);
    this.camera.updateMatrixWorld(true);

    // Update controls target through the typed setter.
    this.controls.setTarget(origin);
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
    // Suppress resize during recording to prevent resolution changes mid-capture
    if (this.resizeLocked) return;

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
      updateCameraAspect(this.camera, width, height);
    }

    // Ensure pixel ratio stays current (matters when dragging between monitors
    // with different DPI — devicePixelRatio changes and a resize fires).
    this.renderer.setPixelRatio(window.devicePixelRatio);

    // PostProcessingManager owns renderer + composer sizing — it calls
    // renderer.setSize() and composer.setSize() internally via resize().
    // Only fall back to direct updateRendererSize() during early init
    // before PostProcessingManager has been created.
    if (this.postProcessing) {
      this.postProcessing.resize(width, height);
    } else {
      this.updateRendererSize(width, height);
    }

    // Update material uniforms for world-space point sizing
    if (this.camera) {
      this.updateMaterialsForCurrentCamera();
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
      this.updateMaterialsForCurrentCamera();
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

    // Set new pixel ratio — PostProcessingManager's updateRendererSize()
    // will pick this up when it calls renderer.setSize().
    this.renderer.setPixelRatio(dpr);

    // PostProcessingManager owns renderer + composer sizing.
    // Its resize() → updateRendererSize() calls renderer.setSize(w, h, false)
    // which keeps CSS dimensions constant while reducing the render buffer.
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
      this.updateMaterialsForCurrentCamera();
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
    if (!isPerspectiveCamera(this.camera)) return; // No FOV in orthographic mode
    const fovChange = deltaY * config.camera.fovSensitivity;
    this.camera.fov = validateFOV(
      this.camera.fov + fovChange,
      config.camera.fovMin,
      config.camera.fovMax
    );
    this.camera.updateProjectionMatrix();

    // Update material uniforms for world-space point sizing
    this.updateMaterialsForCurrentCamera();
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
      `Clipping planes updated - Near: ${near < 0.001 ? near.toExponential(1) : near.toFixed(3)}, Far: ${far.toFixed(1)} (ratio: ${ratio.toFixed(0)}:1)`
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
      // Update scale-aware controls from metadata bounds (available before geometry loads)
      const diagonal = getBoundingBoxDiagonal(sceneBounds);
      if (diagonal > 0) {
        this.controls.setSceneScale(diagonal);
      }

      // Use bounding sphere for smooth clipping (no box-edge discontinuities)
      const sphere = boundingBoxToSphere(sceneBounds);
      const { near, far } = calculateClippingPlanesFromSphere(sphere, cameraPos);

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
    const fallbackBounds = {
      min: { x: box.min.x, y: box.min.y, z: box.min.z },
      max: { x: box.max.x, y: box.max.y, z: box.max.z },
    };

    // Update scale-aware controls from geometry bounds as fallback
    const diagonal = getBoundingBoxDiagonal(fallbackBounds);
    if (diagonal > 0) {
      this.controls.setSceneScale(diagonal);
    }

    const sphere = boundingBoxToSphere(fallbackBounds);
    const { near, far } = calculateClippingPlanesFromSphere(sphere, cameraPos);

    // Apply the calculated planes
    this.updateClippingPlanes(near, far);

    return { near, far };
  }

  /**
   * Auto-frame the camera to fit the scene contents using metadata bounds.
   *
   * Uses position_bounds from zarr metadata (available immediately, no geometry load needed)
   * to compute the optimal camera distance via FOV-aware calculation. This ensures
   * the initial view fits the scene regardless of its physical scale.
   *
   * For orthographic cameras, adjusts zoom instead of distance.
   *
   * @param preserveTarget - If true, keep the current controls target (set by zarr viewer_config)
   *   instead of overwriting it with the bounding box center.
   */
  private autoFrameCamera(preserveTarget: boolean = false): void {
    const bounds = this.getSceneBoundsFromMetadata();
    if (!bounds) {
      log.warning(Modules.SCENE_MANAGER, 'No metadata bounds available for auto-framing');
      return;
    }

    const center = getBoundingBoxCenter(bounds);
    const diagonal = getBoundingBoxDiagonal(bounds);

    if (diagonal <= 0) {
      log.warning(Modules.SCENE_MANAGER, 'Scene bounds have zero extent, skipping auto-frame');
      return;
    }

    // Adapt control speeds to scene scale
    this.controls.setSceneScale(diagonal);

    // Determine the look-at target: author's target if set, otherwise bounding box center
    const lookAtTarget = preserveTarget
      ? this.controls.getFocusTarget()
      : new THREE.Vector3(center.x, center.y, center.z);

    if (isPerspectiveCamera(this.camera)) {
      // Compute optimal distance using FOV, aspect ratio, and fitRatio
      const cameraConfig = {
        fov: this.camera.fov,
        aspect: this.camera.aspect,
        near: this.camera.near,
        far: this.camera.far,
      };
      const distance = calculateCameraDistance(bounds, cameraConfig);

      this.camera.position.set(lookAtTarget.x, lookAtTarget.y, lookAtTarget.z + distance);

      // Set distance limits relative to the scene-fitting distance
      this.controls.setDistanceLimits(distance / ZOOM_RANGE_FACTOR, distance * ZOOM_RANGE_FACTOR);
    } else if (isOrthographicCamera(this.camera)) {
      // For ortho, compute zoom to fit the scene in the frustum
      const frustumHeight = this.camera.top - this.camera.bottom;
      const frustumWidth = this.camera.right - this.camera.left;
      const maxDim = Math.max(
        bounds.max.x - bounds.min.x,
        bounds.max.y - bounds.min.y,
        bounds.max.z - bounds.min.z
      );
      if (maxDim > 0 && frustumHeight > 0 && frustumWidth > 0) {
        const fitRatio = config.scene.defaultFitRatio;
        const zoomH = frustumHeight / (maxDim / fitRatio);
        const zoomW = frustumWidth / (maxDim / fitRatio);
        this.camera.zoom = Math.min(zoomH, zoomW);
        this.camera.updateProjectionMatrix();

        // Set zoom limits relative to the scene-fitting zoom (100x in each direction)
        this.controls.setZoomLimits(
          this.camera.zoom / ZOOM_RANGE_FACTOR,
          this.camera.zoom * ZOOM_RANGE_FACTOR
        );
      }
      // Position along Z for correct depth ordering
      this.camera.position.set(lookAtTarget.x, lookAtTarget.y, lookAtTarget.z + diagonal);
    }

    // Point camera at the look-at target
    this.camera.lookAt(lookAtTarget);
    this.camera.updateMatrixWorld(true);

    // Sync orbit controls with the new camera state.
    // CRITICAL: We must set the target first, then reinitialize() so the controls
    // re-derive their internal distance from the camera position we just set.
    // Without reinitialize(), the next update() would snap the camera back to the
    // old distance (e.g., the default 8 units from resetControls).
    if (!preserveTarget) {
      this.controls.setTarget(lookAtTarget);
    }
    this.controls.reinitialize();
    this.controls.update();
    this.controls.saveState();

    // Track centering state
    this.isCenteredOnBoundingBox = !preserveTarget;
    this.lastBoundingBoxCenter.set(center.x, center.y, center.z);

    log.success(
      Modules.SCENE_MANAGER,
      `Auto-framed camera on scene (target: [${lookAtTarget.x.toFixed(2)}, ${lookAtTarget.y.toFixed(2)}, ${lookAtTarget.z.toFixed(2)}], diagonal: ${diagonal.toFixed(2)})`
    );
  }

  /**
   * Invalidate cached scene bounds. Called on scene load and scene clear.
   * Display dims (sceneDimsManager.getDims().displayed) are immutable per scene,
   * so no invalidation is needed for dimension navigation.
   */
  private invalidateBoundsCache(): void {
    this._cachedBounds = null;
    this._cachedSphere = null;
    this._cachedNearCull = 0.1;
  }

  /** Lazily recompute cached bounds/sphere/nearCull from scene metadata. */
  private ensureBoundsCache(): void {
    if (this._cachedBounds !== null) return;
    const bounds = this.getSceneBoundsFromMetadata();
    if (!bounds) return;
    this._cachedBounds = bounds;
    this._cachedSphere = boundingBoxToSphere(bounds);
    this._cachedNearCull = getBoundingBoxDiagonal(bounds) * 0.001;
  }

  /**
   * Update dynamic clipping planes using cached bounding sphere projection.
   *
   * Called each frame by AnimationController. Uses a cached bounding sphere
   * (invalidated on scene load/clear) for smooth near/far values with zero
   * per-frame scene graph traversal or object allocations.
   */
  updateDynamicClippingPlanes(): void {
    if (!this.dynamicClippingEnabled) return;

    this.ensureBoundsCache();
    const s = this._cachedSphere;
    if (!s) return;

    // Inline sphere-based clipping math (no intermediate object allocations)
    const cam = this.camera.position;
    const dx = cam.x - s.center.x;
    const dy = cam.y - s.center.y;
    const dz = cam.z - s.center.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const R = s.radius * SPHERE_SAFETY_EXPANSION;
    const far = dist + R;
    const near = dist < R ? MIN_NEAR_PLANE : Math.max(MIN_NEAR_PLANE, dist - R);

    // Only update camera if values changed significantly (>0.1%)
    const nearChanged = Math.abs(this.camera.near - near) / this.camera.near > 0.001;
    const farChanged = Math.abs(this.camera.far - far) / this.camera.far > 0.001;

    if (nearChanged || farChanged) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }
  }

  /**
   * Set dynamic clipping enabled/disabled.
   */
  setDynamicClipping(enabled: boolean): void {
    this.dynamicClippingEnabled = enabled;
    log.info(Modules.SCENE_MANAGER, `Dynamic clipping ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Get current dynamic clipping state.
   */
  getDynamicClippingState(): { enabled: boolean; near: number; far: number } {
    return {
      enabled: this.dynamicClippingEnabled,
      near: this.camera.near,
      far: this.camera.far,
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

  // ======================================================================
  // Global EOG (Exposure-Offset-Gamma) — routed to post-processing
  // ======================================================================

  /**
   * Update global exposure (log2 stops).
   * Applied in the vendored tone mapping shader before tone mapping.
   */
  updateExposure(value: number): void {
    this.postProcessing.updateExposure(value);
  }

  /**
   * Update global offset (additive brightness shift).
   * Applied in the vendored tone mapping shader before tone mapping.
   */
  updateGlobalOffset(value: number): void {
    this.postProcessing.updateGlobalOffset(value);
  }

  /**
   * Update global gamma correction.
   * Applied in the vendored tone mapping shader before tone mapping.
   */
  updateGlobalGamma(value: number): void {
    this.postProcessing.updateGlobalGamma(value);
  }

  /**
   * Clean up all Three.js resources to prevent memory leaks.
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
   * After calling dispose(), the scene manager cannot be reused.
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
   * Switch camera control type. Handles camera swap for ortho mode.
   * @param type - Control type ('orbit', 'fly', or 'ortho')
   */
  setControlType(type: ControlType): void {
    const needsOrtho = type === 'ortho';
    const hasOrtho = isOrthographicCamera(this.camera);
    const cameraChanged = needsOrtho !== hasOrtho;

    // Swap camera if projection mode changes
    if (needsOrtho && !hasOrtho) {
      this.swapToOrthographic();
    } else if (!needsOrtho && hasOrtho) {
      this.swapToPerspective();
    }

    // Update controls with new camera reference (may have changed)
    this.controls.setCamera(this.camera);
    this.controls.setControlType(type);

    // Update materials for new projection mode
    this.updateMaterialsForCurrentCamera();

    // Notify listeners that the camera object was replaced (picking system, etc.)
    if (cameraChanged) {
      this.dispatchEvent({ type: 'camera-changed' });
    }
  }

  /**
   * Get current control type
   */
  getControlType(): ControlType {
    return this.controls.getControlType();
  }

  /**
   * Swap from perspective to orthographic camera, matching the current view.
   * Frustum is computed to show the same visible area at the target distance.
   */
  private swapToOrthographic(): void {
    if (!isPerspectiveCamera(this.camera)) return;

    const focusTarget = this.controls.getFocusTarget();
    const distance = Math.max(this.camera.position.distanceTo(focusTarget), 0.001);
    const fovRad = (this.camera.fov * Math.PI) / 180;
    const frustumHeight = 2 * distance * Math.tan(fovRad / 2);
    const aspect = this.camera.aspect || 1;

    const ortho = new THREE.OrthographicCamera(
      (-frustumHeight * aspect) / 2,
      (frustumHeight * aspect) / 2,
      frustumHeight / 2,
      -frustumHeight / 2,
      this.camera.near,
      this.camera.far
    );

    // Reset to a clean front view (looking along -Z, up = Y)
    // Ortho is for 2D viewing — carrying over a tilted 3D orientation is confusing
    ortho.position.set(focusTarget.x, focusTarget.y, focusTarget.z + distance);
    ortho.up.set(0, 1, 0);
    ortho.lookAt(focusTarget);
    ortho.updateMatrixWorld();

    // Old camera not disposed — THREE.js cameras hold no GPU resources
    this.camera = ortho;
    this.lastOrthoZoom = ortho.zoom;
    this.postProcessing.setCamera(ortho);
  }

  /**
   * Swap from orthographic back to perspective camera.
   * Restores the default FOV.
   */
  private swapToPerspective(): void {
    if (isPerspectiveCamera(this.camera)) return;

    const canvas = this.renderer.domElement;
    const aspect =
      (canvas.clientWidth || window.innerWidth) / (canvas.clientHeight || window.innerHeight);

    const persp = new THREE.PerspectiveCamera(
      config.renderingControls.defaults.fov,
      aspect,
      this.camera.near,
      this.camera.far
    );

    persp.position.copy(this.camera.position);
    persp.quaternion.copy(this.camera.quaternion);
    persp.up.copy(this.camera.up);
    persp.updateMatrixWorld();

    // Old camera not disposed — THREE.js cameras hold no GPU resources
    this.camera = persp;
    this.postProcessing.setCamera(persp);
  }

  /**
   * Update all materials with current camera projection parameters.
   * Handles both perspective (FOV-based) and orthographic (frustum-based) modes.
   *
   * Called internally on resize and camera changes. Also used by
   * RecordingPanel when the renderer is resized for offline capture.
   */
  updateMaterialsForCurrentCamera(): void {
    this.renderer.getDrawingBufferSize(this._bufferSize);
    this.ensureBoundsCache();
    const nearCull = this._cachedNearCull;

    if (isOrthographicCamera(this.camera)) {
      const frustumHeight = getOrthoFrustumHeight(this.camera);
      materialManager.updateCameraParams(frustumHeight, this._bufferSize, true, nearCull);
    } else {
      materialManager.updateCameraParams(
        getCameraFovRadians(this.camera),
        this._bufferSize,
        false,
        nearCull
      );
    }
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

  /**
   * Get current scene scale (bounding box diagonal). Returns 0 if not yet set.
   */
  getSceneScale(): number {
    return this.controls.getSceneScale();
  }
}
