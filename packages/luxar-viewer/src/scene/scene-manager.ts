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
import { notifier } from '../utils/notifier';
import { config } from '../config';
import { extractCameraOverrides } from '../config/viewer-config-utils';
import type { ZarrViewerConfig } from '../types/zarr';
import { PostProcessingManager } from '../rendering/post-processing/post-processing-manager';
import { materialManager } from '../rendering';
import { disposeColormapTextures } from '../rendering/colormap-textures';
import {
  createRendererCapabilities,
  type RendererCapabilities,
  type Renderer,
} from '../rendering/renderer-capabilities';
import { configureHDRRenderer, logHDRCapabilities } from '../utils/hdr-detection';
import {
  validateFOV,
  getBoundingBoxDiagonal,
  getBoundingBoxCenter,
  BoundingBox,
  BoundingSphere,
  boundingBoxToSphere,
  calculateClippingPlanesFromSphere,
  projectBoundsToDisplayDims,
  SPHERE_SAFETY_EXPANSION,
  MIN_NEAR_PLANE,
} from './scene-manager-utils';
import { log, Modules, LogEmoji } from '../utils/log';
import { sceneDimsManager } from './scene-dims-manager';
import { clearLoadedSceneContent, disposeSceneGraphResources } from './scene-setup/scene-disposal';
import {
  applyZarrViewerConfig as applyZarrViewerConfigHelper,
  createDefaultPerspectiveCamera,
  resetCameraToInitialPosition,
} from './scene-setup/camera-setup';
import { computeSceneBoundingBox, fitCameraToBounds } from './scene-setup/camera-framing';
import { WebGLContextRecovery } from './scene-setup/webgl-context-recovery';
import {
  type LuxarCamera,
  isPerspectiveCamera,
  isOrthographicCamera,
  getCameraFovRadians,
  updateCameraAspect,
  getOrthoFrustumHeight,
} from '../utils/camera-utils';
import type { ControlType } from '../controls/controls-manager';

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
  /**
   * Fired after a successful WebGL context restore. Subscribers (e.g.
   * SceneLoader, which owns the NodeFactory and picking registrations)
   * use this to re-register / rebuild any GPU-bound resources their
   * objects depend on. SceneManager itself rebuilds the renderer +
   * post-processing + material cache before dispatching.
   */
  'webgl-context-restored': {};
}> {
  /**
   * The graphics-API renderer. Typed as the `Renderer` union from
   * `renderer-capabilities.ts` so the WebGPU port widens this in
   * one place. Today the union has a single arm (WebGLRenderer).
   */
  public renderer!: Renderer;

  /**
   * Capabilities snapshot for the active renderer. Hides raw-GL queries
   * behind a typed interface so the eventual WebGPU port has a single
   * implementation seam.
   */
  public capabilities!: RendererCapabilities;

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

  /**
   * WebGL context-loss / restoration concern. Constructed lazily in
   * setupContextLossHandling() once the canvas + renderer are wired
   * up. The class owns the canvas listeners and the isContextLost
   * flag — SceneManager just forwards events through it.
   */
  private contextRecovery: WebGLContextRecovery | null = null;

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

  /**
   * Explicit DPR selected by adaptive/manual resolution control.
   *
   * `null` means "track the browser's native `window.devicePixelRatio`".
   * Non-null values must survive ordinary window resizes; otherwise a
   * resize event immediately after a manual DPR change silently restores
   * native resolution while the AdaptiveDPRManager/UI still reports the
   * reduced DPR.
   */
  private pixelRatioOverride: number | null = null;

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
    // Try to get HDR canvas context first using config values.
    //
    // Allow-list rule: a `getContext` call is permitted ONLY if it
    // runs before the renderer exists (no `this.capabilities` to
    // route through yet). Today there are exactly two such sites:
    //
    //   1. This line — creates the WebGL2 context the WebGLRenderer
    //      wraps.
    //   2. `src/utils/webgpu-availability.ts` — page-load probe that
    //      classifies the browser as `'webgpu' | 'webgl2' | 'unsupported'`.
    //
    // Under WebGPU the parallel call at site (1) becomes
    // `navigator.gpu.requestAdapter()` / async `renderer.init()`;
    // everything else in the codebase must go through
    // `this.capabilities`.
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
      notifier.error('Failed to create WebGL2 context. Your browser may not support WebGL2.');
    }

    // Create WebGL renderer using configuration values.
    // Shared attributes (antialias, powerPreference, etc.) come from webgl.context;
    // renderer-specific settings (precision, shadowMap, etc.) come from webgl.renderer.
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvasElement, // Use the scene manager's canvas element
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

    // Build the renderer-capabilities snapshot. This is the single
    // module that owns raw-GL probes (MAX_SAMPLES, point-size range,
    // HDR extensions). All downstream consumers read from here, never
    // from `renderer.getContext()`.
    this.capabilities = createRendererCapabilities(this.renderer);

    // Log the active graphics API. Doubles as a live consumer of
    // `capabilities.api` so the discriminator field can't silently
    // rot before the WebGPU port adds the second arm.
    log.info(Modules.RENDERER, `Rendering API: ${this.capabilities.api}`);

    // Report hardware point size limits when debug logging is requested.
    if (this.debug) {
      const [minPt, maxPt] = this.capabilities.pointSizeRange;
      log.info(Modules.RENDERER, `Hardware point size limits: ${minPt}-${maxPt} pixels`);
    }
    // This allows for better integration into complex HTML pages

    // Configure renderer dimensions and high-DPI support
    this.updateRendererSize();

    // Detect and configure HDR capabilities
    const hdrCapabilities = this.capabilities.hdr;
    logHDRCapabilities(hdrCapabilities);
    configureHDRRenderer(this.renderer, hdrCapabilities);

    // Immediately clear to the scene background color to avoid a white flash
    // before the first frame renders (alpha:false makes the canvas opaque white by default)
    this.renderer.setClearColor(config.scene.backgroundColor);
    this.renderer.clear();
  }

  /**
   * Construct the WebGLContextRecovery concern and attach its
   * canvas listeners. Thin delegate over scene-setup/webgl-context-recovery.
   * The recovery instance owns the loss/restored handlers, the
   * `isContextLost` flag, and the deterministic rebuild order.
   */
  private setupContextLossHandling(): void {
    this.contextRecovery = new WebGLContextRecovery({
      canvas: this.canvasElement,
      // Lazy lookup — `setupContextLossHandling` runs before
      // `setupScene` in init() order; capturing `this.scene` at
      // construction would freeze in `undefined`.
      getScene: () => this.scene,
      renderer: this.renderer,
      getPostProcessing: () => this.postProcessing ?? null,
      updateRendererSize: () => this.updateRendererSize(),
      onContextRestored: () => this.dispatchEvent({ type: 'webgl-context-restored' }),
      triggerChange: () => this.dispatchEvent({ type: 'change' }),
    });
    this.contextRecovery.attach();
  }

  /**
   * Check if WebGL context is currently lost. Forwards to the
   * recovery instance; returns `false` when recovery hasn't been
   * wired up yet (pre-init / post-dispose).
   */
  public isWebGLContextLost(): boolean {
    return this.contextRecovery?.getIsContextLost() ?? false;
  }

  /**
   * Initialize the Three.js scene
   */
  private setupScene(): void {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(config.scene.backgroundColor);
  }

  /**
   * Initialize the perspective camera. Thin delegate over
   * `createDefaultPerspectiveCamera` in scene-setup/camera-setup so
   * the FOV/clip/initial-position pose is unit-testable in isolation.
   */
  private setupCamera(): void {
    this.camera = createDefaultPerspectiveCamera(this.renderer.domElement);
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
   * Reset camera + controls to default state when loading a new
   * dataset. Camera-side reset is delegated to
   * `resetCameraToInitialPosition`; the controls-side state machine
   * (reset → update → saveState) stays here because it's the
   * controls manager's contract.
   */
  private resetControls(): void {
    resetCameraToInitialPosition(this.camera);
    this.controls.reset();
    this.controls.update();
    this.controls.saveState();
    log.success(Modules.CONTROLS, 'Controls reset to default state');
  }

  /**
   * Initialize the HDR post-processing pipeline.
   *
   * Wires up the mega-shader pipeline with 16-bit float (HalfFloat)
   * buffers, the bloom pre-pass, and the optional FXAA post-pass.
   * See PostProcessingManager for the full pipeline.
   */
  private setupPostProcessing(): void {
    // Get canvas dimensions for proper HDR render target sizing
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;

    // Create post-processing manager with actual canvas dimensions.
    // The onResize callback fires whenever the manager reallocates
    // its render-target pyramid (window resize, SSAA toggle, MSAA
    // toggle, DPR change). Scene material uniforms cache
    // pointSizeFactor / uResolution from `renderer.getDrawingBufferSize()`
    // and would otherwise stay stale until the next manual window
    // resize.
    this.postProcessing = new PostProcessingManager(
      this.renderer,
      this.capabilities,
      this.scene,
      this.camera,
      { width, height },
      () => {
        if (this.camera) this.updateMaterialsForCurrentCamera();
      }
    );

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
    notifier.showLoading();

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
      notifier.hideLoading();
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
      notifier.hideLoading();
      log.error(Modules.SCENE_MANAGER, 'Failed to load scene:', error);
      notifier.error(`Failed to load scene from "${src}". Please check the path and try again.`);
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
   * Apply viewer config from zarr (camera position/target/up, background
   * color). Thin delegate over `applyZarrViewerConfig` in
   * scene-setup/camera-setup; the helper returns whether an explicit
   * camera position was applied so `loadSceneData` can suppress
   * auto-framing. (Author target alone does NOT suppress auto-framing.)
   */
  private applyZarrViewerConfig(root: THREE.Group): void {
    applyZarrViewerConfigHelper(root, this.camera, this.controls, this.scene);
  }

  /**
   * Clear all loaded content from the scene, keeping lights and background.
   * Thin delegate over `clearLoadedSceneContent` in scene-setup/scene-disposal.
   */
  private clearSceneContent(): void {
    this.invalidateBoundsCache();
    const removed = clearLoadedSceneContent(this.scene);
    log.info(Modules.SCENE_MANAGER, `Cleared ${removed} objects from scene`);
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
    // Ensure world matrices are up to date before computing bounds.
    this.scene.updateMatrixWorld(true);

    const { box, primitiveCount } = computeSceneBoundingBox(this.scene);
    if (box.isEmpty() || primitiveCount === 0) {
      log.warning(Modules.SCENE_MANAGER, 'No visible geometry found to center camera on');
      return;
    }

    const center = box.getCenter(new THREE.Vector3());
    this.lastBoundingBoxCenter.copy(center);

    fitCameraToBounds(
      this.camera,
      this.controls,
      {
        min: { x: box.min.x, y: box.min.y, z: box.min.z },
        max: { x: box.max.x, y: box.max.y, z: box.max.z },
      },
      { lookAtTarget: center, logLabel: 'Camera centered on scene' }
    );
    log.success(Modules.CONTROLS, 'Controls target updated and state saved');
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

    // Ensure pixel ratio stays current. When adaptive/manual DPR is active,
    // preserve that explicit override across ordinary window resizes; when
    // no override is active, track native devicePixelRatio changes (e.g.
    // dragging between monitors with different DPI).
    this.renderer.setPixelRatio(this.getActivePixelRatio());

    // PostProcessingManager owns renderer + composer sizing — it calls
    // renderer.setSize() and composer.setSize() internally via resize().
    // Only fall back to direct updateRendererSize() during early init
    // before PostProcessingManager has been created.
    if (this.postProcessing) {
      this.postProcessing.resize(width, height);
      this.syncPostProcessingDPRScale();
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

    // Set pixel ratio BEFORE size for correct buffer calculations.
    this.renderer.setPixelRatio(this.getActivePixelRatio());
    // Let Three.js handle CSS sizing normally
    this.renderer.setSize(w, h); // Allow Three.js to set CSS size

    // Update material uniforms for world-space point sizing (only if camera exists)
    if (this.camera) {
      this.updateMaterialsForCurrentCamera();
    }
  }

  /** Return the DPR currently applied to renderer sizing. */
  private getActivePixelRatio(): number {
    return (this.pixelRatioOverride ?? window.devicePixelRatio) || 1;
  }

  /**
   * Store/clear the explicit DPR override and return the effective DPR.
   * Native DPR clears the override so future monitor-DPI changes continue
   * to track `window.devicePixelRatio` automatically.
   */
  private setPixelRatioOverride(dpr: number): number {
    const nativeDPR = window.devicePixelRatio || 1;
    const safeDPR = Number.isFinite(dpr) && dpr > 0 ? dpr : nativeDPR;
    this.pixelRatioOverride = Math.abs(safeDPR - nativeDPR) < 0.01 ? null : safeDPR;
    return this.getActivePixelRatio();
  }

  /** Normalize active DPR relative to current native DPR for perceptual effect scaling. */
  private getNormalizedDPRScale(dpr: number = this.getActivePixelRatio()): number {
    const nativeDPR = window.devicePixelRatio || 1;
    return dpr / nativeDPR;
  }

  /** Keep DPR-dependent post-processing effects consistent after DPR/resize changes. */
  private syncPostProcessingDPRScale(): void {
    if (!this.postProcessing) return;
    this.postProcessing.setDPRScale(this.getNormalizedDPRScale());
  }

  /**
   * Update pixel ratio for adaptive performance optimization.
   *
   * Uses setSize with updateStyle=false to keep canvas CSS size constant
   * while reducing the internal buffer resolution for better performance.
   *
   * This method is called by the AdaptiveDPRManager when FPS drops below
   * acceptable thresholds, and by the manual DPR control when adaptive
   * mode is disabled.
   *
   * @param dpr - The new device pixel ratio to use
   */
  public setAdaptivePixelRatio(dpr: number): void {
    const canvas = this.renderer.domElement;
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    const activeDPR = this.setPixelRatioOverride(dpr);

    // Set new pixel ratio — PostProcessingManager's updateRendererSize()
    // will pick this up when it calls renderer.setSize().
    this.renderer.setPixelRatio(activeDPR);

    // PostProcessingManager owns renderer + composer sizing.
    // Its resize() → updateRendererSize() calls renderer.setSize(w, h, false)
    // which keeps CSS dimensions constant while reducing the render buffer.
    if (this.postProcessing) {
      this.postProcessing.resize(w, h);

      // Scale noise parameters based on DPR to maintain perceptual consistency
      // At lower DPR, each pixel covers more area, so noise should be scaled down.
      this.postProcessing.setDPRScale(this.getNormalizedDPRScale(activeDPR));
    }

    // Update material uniforms for world-space point sizing
    if (this.camera) {
      this.updateMaterialsForCurrentCamera();
    }

    log.update(
      Modules.SCENE_MANAGER,
      `Adaptive DPR: ${activeDPR.toFixed(2)} (buffer: ${Math.round(w * activeDPR)}x${Math.round(
        h * activeDPR
      )})`
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

    // Determine the look-at target: author's target if set, otherwise bounding box center.
    const center = getBoundingBoxCenter(bounds);
    const lookAtTarget = preserveTarget
      ? this.controls.getFocusTarget()
      : new THREE.Vector3(center.x, center.y, center.z);

    const diagonal = fitCameraToBounds(this.camera, this.controls, bounds, {
      lookAtTarget,
      preserveControlsTarget: preserveTarget,
      logLabel: 'Auto-framed camera on scene',
    });
    if (diagonal === 0) {
      log.warning(Modules.SCENE_MANAGER, 'Scene bounds have zero extent, skipping auto-frame');
      return;
    }

    // Track centering state
    this.isCenteredOnBoundingBox = !preserveTarget;
    this.lastBoundingBoxCenter.set(center.x, center.y, center.z);
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
    const foundBounds = this.findPositionBoundsInScene();
    if (!foundBounds) return null;

    const dims = sceneDimsManager.getDims();
    const displayDims: number[] = dims?.displayed ?? [0, 1, 2];

    return projectBoundsToDisplayDims(foundBounds.min, foundBounds.max, displayDims);
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

    // Tear down the WebGL context-recovery listeners.
    if (this.contextRecovery) {
      this.contextRecovery.dispose();
      this.contextRecovery = null;
    }

    // Dispose post-processing resources first
    // This includes HDR render targets, effect composer, and all passes
    this.postProcessing.dispose();

    // Dispose controls - removes all event listeners and internal references
    // This prevents memory leaks from mouse/touch event handlers
    this.controls.dispose();

    // Dispose material manager - cleans up all cached materials
    materialManager.dispose();

    // dispose shared colormap textures (built-in cache + custom-LUT
    // cache) AFTER materials are released — material disposal doesn't
    // touch shared LUT textures because they live in module-scope
    // caches. Without this call, repeated app-construction cycles leak
    // a 256×1 RGBA DataTexture per built-in colormap plus one per unique
    // custom LUT.
    disposeColormapTextures();

    // Dispose renderer - cleans up WebGL context and associated GPU resources
    // This frees vertex buffers, textures, and shader programs
    this.renderer.dispose();

    // Traverse scene graph and dispose all geometry and material resources
    // (WebGL resources are not garbage collected). Delegated to
    // scene-setup/scene-disposal so the same one-shot final-dispose pass
    // is unit-testable in isolation.
    disposeSceneGraphResources(this.scene);
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
   * Toggle "natural drag" — swap LEFT ↔ RIGHT mouse buttons in orbit mode so
   * a one-finger touchpad drag rotates and right-drag pans. Applies to
   * orbit (3D) only; ortho and fly modes ignore.
   */
  setNaturalDrag(enabled: boolean): void {
    this.controls.setNaturalDrag(enabled);
    log.custom(
      LogEmoji.SCENE,
      Modules.SCENE_MANAGER,
      `Natural drag ${enabled ? 'enabled' : 'disabled'}`
    );
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

    // Prior camera is not disposed — THREE.js cameras hold no GPU resources.
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

    // Prior camera is not disposed — THREE.js cameras hold no GPU resources.
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
