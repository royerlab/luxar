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
import { notifier } from '../utils/cross-layer/notifier';
import { config } from '../config';
import {
  extractCameraOverrides,
  extractRenderingOverrides,
} from '../config/zarr-bridge/viewer-config-utils';
import type { ZarrViewerConfig } from '../types/zarr';
import type { PostProcessingManager } from '../rendering/post-processing/post-processing-manager';
import { materialManager } from '../rendering';
import { resolveMaterialBackend } from '../rendering/material-manager/factories';
import {
  createSceneEnvironment,
  type SceneEnvironment,
} from '../rendering/environment/scene-environment';
import type { BakedEnvironment } from '../types/environment';
import { loadTslMaterials } from '../rendering/tsl/load';
import { disposeColormapTextures } from '../rendering/colormap-textures';
import {
  clearBlendModeProgramWarmup,
  configureBlendModeProgramWarmup,
  warmSceneBlendModePrograms,
} from '../rendering/webgl-blend-warmup';
import {
  type Renderer,
  type RendererCapabilities,
  isWebGLRenderer,
} from '../rendering/renderer-capabilities';
import {
  BoundingBox,
  getBoundingBoxDiagonal,
  validateFOV,
} from './scene-manager/clipping/bounds-math';
import {
  SceneBoundsCache,
  computeBoundsFromMetadata,
} from './scene-manager/clipping/scene-bounds-cache';
import {
  type ClippingCtx,
  applyClippingPlanes,
  autoAdjustFromBounds,
  updateDynamicFromCache,
} from './scene-manager/clipping/clipping-policy';
import {
  type CameraMaterialsCtx,
  updateMaterialsForCurrentCamera as updateCameraMaterials,
  adjustFOV,
} from './scene-manager/camera/camera-materials';
import { log, Modules, LogEmoji } from '../utils/log';
import {
  clearLoadedSceneContent,
  disposeSceneGraphResources,
} from './scene-manager/render-pipeline/scene-disposal';
import {
  applyZarrViewerConfig as applyZarrViewerConfigHelper,
  createDefaultPerspectiveCamera,
  resetCameraToInitialPosition,
} from './scene-manager/camera/camera-setup';
import {
  autoFrameCamera,
  centerCameraOnScene,
  frameCameraOnObject,
  centerOnOrigin,
  ZOOM_IN_FACTOR,
  ZOOM_OUT_FACTOR,
} from './scene-manager/camera/camera-framing';
import {
  type CameraModeCtx,
  setControlType as cameraModeSetControlType,
} from './scene-manager/camera/camera-mode';
import { sceneDimsManager } from './scene-dims-manager';
import { WebGLContextRecovery } from './scene-manager/render-pipeline/webgl-context-recovery';
import { reduceGpuByteBudgetForContextLoss } from '../rendering/gpu-byte-budget';
import {
  createWebGLRenderer,
  createWebGPURenderer,
  selectBackend,
} from './scene-manager/render-pipeline/renderer-setup';
import { createPostProcessing } from './scene-manager/render-pipeline/post-processing-setup';
import { ResizeOrchestrator } from './scene-manager/viewport/resize-orchestrator';
import {
  computePixelRatioOverride,
  getActivePixelRatio,
} from './scene-manager/viewport/dpr-policy';
import { type LuxarCamera, isPerspectiveCamera, isOrthographicCamera } from '../utils/camera-utils';
import { isDocumentFullscreen } from '../utils/fullscreen';
import type { ControlType } from '../controls/controls-manager';
import type { AutoRotateAxis } from '../controls/types';

/** Default scene up (world +Y) — overridden per scene by `viewer_config.up`. */
const DEFAULT_SCENE_UP = new THREE.Vector3(0, 1, 0);
/** Skip projection/material refreshes within the 0.5° deadband preserved from prior call sites. */
const FOV_APPLY_DEADBAND_DEG = 0.5;

/** Caller-resolved scene-load decisions that affect initial camera setup. */
export interface SceneLoadOptions {
  /** Apply scene FOV before auto-framing; authored positions carry it regardless. */
  applyViewerConfigFov?: boolean;
}

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
  /** The view moved or its projection changed; redraw and invalidate picking. */
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
  /**
   * Fired when the WebGPU device.lost Promise resolves. Treated as
   * **unrecoverable** in this release — Luxar does not rebuild GPU
   * resources after WebGPU device loss; the host application is
   * expected to prompt for a reload. See
   * `setupContextLossHandling` for the rationale.
   *
   * Payload carries the diagnostic reason / message returned by the
   * browser's `GPUDevice.lost` info struct (both may be undefined
   * when the browser doesn't supply them).
   */
  'webgpu-device-lost': { reason?: string; message?: string };
}> {
  /**
   * The graphics-API renderer. Holds the `Renderer` union honestly:
   * a `THREE.WebGLRenderer` on the default path, or a `WebGPURenderer`
   * when opted in via `?renderer=webgpu` / `VITE_LUXAR_USE_WEBGPU=1`
   * (which itself may dispatch to a real WebGPU adapter or transparently
   * fall back to its internal WebGL2 backend depending on browser support).
   *
   * Every method called on this field across the codebase
   * (`PostProcessingManager`, `picking-system`, `BloomChain`,
   * `FxaaPass`, UI panels) is part of the common `Renderer` surface
   * in Three r185 — no `WebGLRenderer`-only API is used
   * unconditionally. The discriminator for callers that genuinely
   * must branch is `this.capabilities.apiSurface` (see
   * `RendererCapabilities`).
   */
  public renderer!: Renderer;

  /**
   * The lazily built scene environment that lights `material="physical"` meshes;
   * `null` until {@link init}. Public so a debug surface or test can ask
   * `isReady()`; nothing else should need to touch it.
   */
  public environment: SceneEnvironment | null = null;
  private unsubscribeEnvironment: (() => void) | null = null;

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

  /**
   * The scene's up vector — world +Y unless the zarr `viewer_config`
   * authored one (set in loadSceneData). Every camera fit/reset
   * (Home/F, center-on-origin, load-time auto-frame) squares to this,
   * so author-oriented scenes reset to THEIR horizon, not world Y.
   */
  private sceneUp: THREE.Vector3 = DEFAULT_SCENE_UP.clone();

  /** Resize debouncing with requestAnimationFrame for smooth resizing */
  private resizer = new ResizeOrchestrator();

  /**
   * WebGL context-loss / restoration concern. Constructed lazily in
   * setupContextLossHandling() once the canvas + renderer are wired
   * up. The class owns the canvas listeners and the isContextLost
   * flag — SceneManager just forwards events through it.
   */
  private contextRecovery: WebGLContextRecovery | null = null;

  /**
   * When true, resize events are suppressed (used during recording to
   * prevent resolution changes). Delegated to ResizeOrchestrator —
   * exposed as a getter/setter so recording-panel.ts continues to read
   * + write `sceneManager.resizeLocked` directly.
   */
  get resizeLocked(): boolean {
    return this.resizer.resizeLocked;
  }
  set resizeLocked(v: boolean) {
    this.resizer.resizeLocked = v;
  }

  /**
   * Perspective FOV in effect at the last perspective→ortho swap, restored on
   * the inverse ortho→perspective swap so a control-mode round trip preserves
   * the user's FOV instead of resetting it to the config default.
   */
  private lastPerspectiveFov: number = config.renderingControls.defaults.fov;

  /** Dynamic clipping planes state */
  private dynamicClippingEnabled: boolean =
    config.renderingControls.defaults.dynamicClippingEnabled;

  /**
   * Cached scene bounds + bounding sphere + near-cull margin.
   * Invalidated on scene load/clear; lazily recomputed by
   * `boundsCache.ensure(scene)`.
   */
  private readonly boundsCache = new SceneBoundsCache();

  /**
   * Reusable Vector2 for getDrawingBufferSize (avoids per-call allocation).
   *
   * **Ownership contract**: This Vector2 instance is owned by SceneManager
   * and is reused across every camera-materials update — every resize
   * mutates it in place. Consumers reached via {@link makeCameraMaterialsCtx}
   * MUST treat the supplied `bufferSize` as **read-only borrow scoped to
   * the current call**. If a consumer needs to hold the value across
   * frames, it must `.copy()` the Vector2 into its own storage, not
   * retain the shared reference — otherwise the next resize will silently
   * corrupt the held value.
   */
  private readonly _bufferSize = new THREE.Vector2();

  /**
   * Explicit DPR selected by adaptive/manual resolution control.
   *
   * `null` means "track the live pixel-ratio ceiling".
   * Non-null values must survive ordinary window resizes; otherwise a
   * resize event immediately after a manual DPR change silently restores
   * ceiling resolution while the AdaptiveDPRManager/UI still reports the
   * reduced DPR.
   */
  private pixelRatioOverride: number | null = null;

  /**
   * Current FOV in degrees. Perspective: the live camera FOV. Orthographic:
   * the stashed perspective FOV (`lastPerspectiveFov`) — the single source of
   * truth for the FOV that a future ortho→perspective swap will restore, so a
   * FOV set while in ortho (Reset-to-Defaults / zarr-authored) is reported and
   * honored rather than masked by the config default.
   */
  get currentFov(): number {
    return isPerspectiveCamera(this.camera) ? this.camera.fov : this.lastPerspectiveFov;
  }

  /**
   * Set an absolute perspective FOV using rendering-setting validation semantics.
   * Invalid values fall back to the configured default rather than clamping.
   */
  setFov(degrees: number): boolean {
    const fov =
      Number.isFinite(degrees) && degrees >= config.camera.fovMin && degrees <= config.camera.fovMax
        ? degrees
        : config.renderingControls.defaults.fov;
    if (Math.abs(this.currentFov - fov) <= FOV_APPLY_DEADBAND_DEG) return false;

    if (isOrthographicCamera(this.camera)) {
      this.lastPerspectiveFov = fov;
      return true;
    }

    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
    this.updateMaterialsForCurrentCamera();
    this.dispatchEvent({ type: 'change' });
    return true;
  }

  /** Apply an authored zoom after the camera has switched to ortho projection. */
  setCameraZoom(zoom: number): void {
    if (!isOrthographicCamera(this.camera) || !Number.isFinite(zoom) || zoom <= 0) return;
    this.camera.zoom = zoom;
    this.camera.updateProjectionMatrix();
    this.updateMaterialsForCurrentCamera();
    this.controls.setZoomLimits(zoom / ZOOM_OUT_FACTOR, zoom * ZOOM_IN_FACTOR);
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
   * Optional per-instance backend override. When set, `setupRenderer`
   * uses it instead of consulting the env var. Threaded from the
   * `?renderer=webgl|webgpu` URL parameter through `LuxarAppOptions`.
   */
  private rendererOverride: 'webgl' | 'webgpu' | undefined;
  /**
   * Diagnostic WebGPURenderer mode. When true and the selected renderer is
   * WebGPURenderer, pass `{ forceWebGL: true }` so Three.js uses its
   * internal WebGL2 backend while Luxar still dispatches TSL materials.
   */
  private webgpuForceWebGL = false;

  /**
   * Opt-in to `WebGPURenderer({ trackTimestamp: true })` for the perf
   * bench. Off by default; flipped via `?perfTimestamp` URL param.
   */
  private perfTimestamp = false;
  /** WebGL-only blend-variant warm-up (`?noBlendWarmup` disables). */
  private blendWarmup = true;

  /**
   * Initialize the renderer pipeline.
   *
   * @param options.canvas - The HTMLCanvasElement to render into. Callers
   *   resolve this themselves (`document.getElementById(...)` in the
   *   standalone app's main.ts; arbitrary container child for embedders).
   *   SceneManager performs no DOM lookups of its own.
   * @param options.debug - Verbose hardware/runtime logging.
   */
  async init(options: {
    canvas: HTMLCanvasElement;
    debug?: boolean;
    /**
     * Optional backend override. When provided, wins over the
     * `VITE_LUXAR_USE_WEBGPU` / `VITE_LUXAR_USE_LEGACY_WEBGL` env vars
     * and the WebGL default. Threaded from `LuxarAppOptions.renderer`,
     * ultimately from the `?renderer=webgl|webgpu` URL parameter.
     */
    renderer?: 'webgl' | 'webgpu';
    /**
     * Diagnostic flag for `WebGPURenderer({ forceWebGL: true })`.
     * Threaded from `LuxarAppOptions.webgpuForceWebGL`, ultimately from
     * the `?webgpuForceWebgl` URL parameter.
     */
    webgpuForceWebGL?: boolean;
    /**
     * Opt-in to GPU timestamp queries. Threaded from
     * `LuxarAppOptions.perfTimestamp`, ultimately from the
     * `?perfTimestamp` URL flag set by the perf bench.
     */
    perfTimestamp?: boolean;
    /**
     * WebGL-only blend warm-up. When true, classic `THREE.WebGLRenderer`
     * sessions pre-compile each DISTINCT blend-mode program variant a
     * material can reach, one compile per post-frame idle opportunity, so the first
     * Layers-panel blend switch does not pay SwiftShader's synchronous
     * link cost on the click path.
     */
    blendWarmup?: boolean;
  }): Promise<void> {
    this.canvasElement = options.canvas;
    this.debug = options.debug ?? false;
    this.rendererOverride = options.renderer;
    this.webgpuForceWebGL = options.webgpuForceWebGL ?? false;
    this.perfTimestamp = options.perfTimestamp ?? false;
    this.blendWarmup = options.blendWarmup ?? true;
    await this.setupRenderer();
    this.setupContextLossHandling(); // Setup context loss recovery
    this.setupScene();
    this.setupEnvironment();
    this.setupCamera();
    this.configureBlendWarmup();
    this.setupControls();
    this.setupPostProcessing();

    // Apply resize directly during initialization (no rAF coalescing) so
    // dimensions are available before the first render.
    this.resizeToCanvas();
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
  private async setupRenderer(): Promise<void> {
    const { backend, source } = selectBackend(this.rendererOverride);
    log.info(
      Modules.RENDERER,
      `Backend selection: ${backend === 'webgl' ? 'WebGLRenderer (GLSL)' : 'WebGPURenderer (TSL)'} ` +
        `[source: ${source}${backend === 'webgpu' && this.webgpuForceWebGL ? ', forceWebGL backend' : ''}]`
    );
    if (backend === 'webgl') {
      await this.setupWebGLRenderer();
      return;
    }
    await this.setupWebGPURenderer();
  }

  private async setupWebGLRenderer(): Promise<void> {
    const { renderer, capabilities } = await createWebGLRenderer(this.canvasElement);
    this.renderer = renderer;
    this.capabilities = capabilities;

    // Hand the capabilities to the material manager so its
    // getPoint/Line/GSplatMaterial dispatch can pick the GLSL or TSL
    // backend. Must precede any node-factory material requests.
    materialManager.setCaps(this.capabilities);

    if (this.debug) {
      const [minPt, maxPt] = this.capabilities.pointSizeRange;
      log.info(Modules.RENDERER, `Hardware point size limits: ${minPt}-${maxPt} pixels`);
    }

    // Configure renderer dimensions and high-DPI support.
    this.resizeToCanvas();

    // Clear immediately to the scene background color to avoid the
    // brief white flash before the first frame renders.
    this.renderer.setClearColor(config.scene.backgroundColor);
    this.renderer.clear();
  }

  /**
   * Opt-in renderer setup — selected via `?renderer=webgpu` URL flag
   * or `VITE_LUXAR_USE_WEBGPU=1`. Constructs a `WebGPURenderer` and
   * runs its async `init()`. On browsers with WebGPU support, the
   * renderer acquires a WebGPU adapter and dispatches TSL graphs
   * to WGSL. On browsers without WebGPU (Firefox today, older
   * Safari), Three's WebGPURenderer transparently falls back to a
   * WebGL2 backend — the TSL graphs target both from one source.
   *
   * NOTE: WebGL is the **production default**; this method is
   * reached only when the user explicitly opts into WebGPU.
   * See `BROWSER_SUPPORT_POLICY.md` for the policy.
   */
  private async setupWebGPURenderer(): Promise<void> {
    const result = await createWebGPURenderer(this.canvasElement, {
      debug: this.debug,
      webgpuForceWebGL: this.webgpuForceWebGL,
      perfTimestamp: this.perfTimestamp,
      rendererOverride: this.rendererOverride,
    });
    if (result.fallback) {
      // Adapter below the WebGPU spec minimum (< 8 vertex buffers)
      // and the user didn't force the path; drop to WebGL.
      await this.setupWebGLRenderer();
      return;
    }
    this.renderer = result.renderer;
    this.capabilities = result.capabilities;

    // Fetch the TSL/WebGPU material cone before anything can ask for a
    // material. This is the ONLY place it is loaded on the production path, and
    // the reason the default WebGL session never downloads the ~182 kB gzipped
    // `three-webgpu` chunk (issue #1679).
    //
    // Ordering is load-bearing and already guaranteed: `init()` awaits
    // `setupRenderer()` (this) before `setupPostProcessing()`, which builds the
    // first material of the whole app (the mega-shader, then bloom and FXAA).
    // Because the registry is installed by then, `buildMaterial` and the
    // `MaterialManager` dispatch tables can stay synchronous.
    await loadTslMaterials();

    // Hand the capabilities to the material manager so its
    // getPoint/Line/GSplatMaterial dispatch picks the TSL backend.
    // Must precede any node-factory or post-processing material
    // requests — without this, the WebGPU path silently dispatches
    // GLSL ShaderMaterial and the NodeBuilder rejects it.
    materialManager.setCaps(this.capabilities);

    this.resizeToCanvas();
    this.renderer.setClearColor(config.scene.backgroundColor);
    this.renderer.clear();
  }

  /**
   * Construct the WebGLContextRecovery concern and attach its
   * canvas listeners (WebGL2 path), OR attach a `device.lost`
   * observer that surfaces the failure as an event (WebGPU path).
   *
   * **WebGL2 path.** The recovery instance owns the loss/restored
   * handlers, the `isContextLost` flag, and the deterministic
   * rebuild order. Bound here because `webglcontextlost` /
   * `webglcontextrestored` canvas events fire only on WebGL
   * contexts.
   *
   * **WebGPU path.** Three's `WebGPURenderer` recreates its own
   * GPU device internally when `device.lost` resolves, but it does
   * **not** rebuild Luxar-owned resources (post-processing render
   * targets, picking buffers, material caches, interleaved geometry
   * buffers). A full rebuild path mirroring WebGL2 is non-trivial
   * and untested in CI today, so for now we treat WebGPU device
   * loss as **unrecoverable**: log it loudly and dispatch a
   * `webgpu-device-lost` event so the host application can prompt
   * for a reload. The event payload carries the device-loss reason
   * for diagnostics.
   */
  private setupContextLossHandling(): void {
    if (this.capabilities.apiSurface === 'webgl2') {
      this.contextRecovery = new WebGLContextRecovery({
        canvas: this.canvasElement,
        // Lazy lookup — `setupContextLossHandling` runs before
        // `setupScene` in init() order; capturing `this.scene` at
        // construction would freeze in `undefined`.
        getScene: () => this.scene,
        renderer: this.renderer as THREE.WebGLRenderer,
        getPostProcessing: () => this.postProcessing ?? null,
        updateRendererSize: () => this.resizeToCanvas(),
        onContextRestored: () => {
          try {
            if (this.environment?.isReady()) this.environment.rebuild();
          } catch (error) {
            log.warning(Modules.SCENE_MANAGER, 'Failed to rebuild scene environment', error);
          }
          this.dispatchEvent({ type: 'webgl-context-restored' });
          void this.warmBlendModePrograms();
        },
        triggerChange: () => this.dispatchEvent({ type: 'change' }),
        onContextLost: () => {
          clearBlendModeProgramWarmup();
          reduceGpuByteBudgetForContextLoss();
        },
      });
      this.contextRecovery.attach();
      return;
    }

    // WebGPU: attach a best-effort device.lost observer. The
    // backend's `device` is created during `gpuRenderer.init()`,
    // so by the time `setupContextLossHandling` runs the device is
    // either present or we accept "device.lost reporting unavailable
    // on this Three.js build" silently. Structural typing avoids the
    // @webgpu/types dependency on consumers that don't enable WebGPU
    // in TypeScript's lib.
    type DeviceLostInfo = { reason?: string; message?: string };
    type GpuDeviceLike = { lost: Promise<DeviceLostInfo> };
    const backend = (this.renderer as { backend?: { device?: GpuDeviceLike } }).backend;
    const device = backend?.device;
    if (!device || typeof device.lost !== 'object') {
      return;
    }
    void device.lost.then((info: DeviceLostInfo) => {
      log.error(
        Modules.RENDERER,
        `WebGPU device lost (reason=${info.reason ?? 'unknown'}). ` +
          'Luxar does not auto-recover WebGPU device loss in this release — ' +
          `please reload the page. Message: ${info.message ?? '(none)'}`
      );
      this.dispatchEvent({
        type: 'webgpu-device-lost',
        reason: info.reason,
        message: info.message,
      });
    });
  }

  /**
   * Check if WebGL context is currently lost. Forwards to the
   * recovery instance; returns `false` when recovery hasn't been
   * wired up yet (pre-init / post-dispose).
   */
  public isWebGLContextLost(): boolean {
    return this.contextRecovery?.getIsContextLost() ?? false;
  }

  private configureBlendWarmup(): void {
    const renderer = isWebGLRenderer(this.renderer) ? this.renderer : null;
    configureBlendModeProgramWarmup({
      enabled: this.blendWarmup && this.capabilities.apiSurface === 'webgl2' && renderer !== null,
      renderer,
      camera: this.camera,
      targetScene: this.scene,
    });
  }

  /**
   * Initialize the Three.js scene
   */
  private setupScene(): void {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(config.scene.backgroundColor);
  }

  /**
   * Wire the LAZY scene environment (`rendering/environment/scene-environment.ts`).
   *
   * Nothing is built here. The environment — a prefiltered `RoomEnvironment` on
   * `scene.environment`, the one lighting input a physically based mesh material
   * needs — is constructed the first time the material manager creates a physical
   * mesh material and never otherwise, so a scene without one keeps
   * `scene.environment === null` and renders exactly as it did before the
   * environment existed. House materials never read it either way (spec
   * `MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3).
   */
  private setupEnvironment(): void {
    this.environment = createSceneEnvironment(
      this.renderer,
      resolveMaterialBackend(this.capabilities),
      this.scene
    );
    this.unsubscribeEnvironment = materialManager.onPhysicalMaterialCreated(() => {
      try {
        this.environment?.ensure();
      } catch (error) {
        log.warning(Modules.SCENE_MANAGER, 'Failed to build scene environment', error);
      }
    });
  }

  /**
   * Initialize the perspective camera. Thin delegate over
   * `createDefaultPerspectiveCamera` in scene-manager/camera/camera-setup
   * so the FOV/clip/initial-position pose is unit-testable in isolation.
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
    // (An ortho zoom needs no material push: it lives in the projection
    // matrix, which every shader reads per draw.)
    this.controls.addEventListener('change', () => {
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
   * Initialize the HDR post-processing pipeline. Wires the manager's
   * resize callback to refresh material uniforms that cache the
   * drawing-buffer size.
   */
  private setupPostProcessing(): void {
    this.postProcessing = createPostProcessing({
      renderer: this.renderer,
      capabilities: this.capabilities,
      scene: this.scene,
      camera: this.camera,
      onResize: () => {
        if (this.camera) this.updateMaterialsForCurrentCamera();
      },
    });
  }

  /**
   * Load scene data from Zarr source.
   *
   * Cache and prefetch flags propagate through `loaderConfig` from
   * LuxarApp (originally derived from `?noCache`/`?cacheDebug`/etc URL
   * parameters in main.ts).
   * `options.applyViewerConfigFov` is the caller's localStorage-precedence
   * decision, not a feature switch. Returning visitors keep their stored FOV
   * for auto-framed scenes; an authored position instead carries the resolved
   * scene FOV with it as one framing contract. Under an orthographic camera the
   * FOV only stashes the next perspective value; framing uses camera zoom, and
   * the later projection swap preserves that frustum.
   */
  async loadSceneData(
    src: string,
    loaderConfig?: LoaderConfig,
    options: SceneLoadOptions = {}
  ): Promise<void> {
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

      // Resolve the scene's DISPLAYED dimensions before anything below reads
      // bounds. Every metadata-bounds consumer in this method (scene scale,
      // auto-frame, clipping planes, near-cull) projects the nD
      // `position_bounds` through `sceneDimsManager.getDims().displayed`, and
      // falls back to [0, 1, 2] when the manager is uninitialised. The
      // dimension-navigation UI initialises it too, but only AFTER this method
      // resolves — so a scene whose displayed dims are not the first three
      // (e.g. a leading non-displayed time / order / channel axis) used to be
      // framed around the WRONG axes: the non-displayed axis' extent landed on
      // world X, putting the look-at target off to one side of the geometry and
      // inflating the fit distance by that axis' range. `initFromScene` is a
      // pure metadata read (no listeners fire, no camera touched), and the
      // later UI call re-runs it identically.
      sceneDimsManager.initFromScene(this.scene);

      // Material parameters were initialized before loadScene(). A scene FOV
      // applied below refreshes them again before the first rendered frame.

      // Establish scale-aware orbit distance limits from scene bounds BEFORE
      // applying the author's camera. The orbit controls start with a small
      // default maxDistance (config.controls.orbit.zoom.maxDistance); a
      // viewer_config that places the camera far from its target (a wide
      // establishing shot) would otherwise have that distance clamped to the
      // default max by reinitialize()+update(), snapping the camera near the
      // target. setSceneScale only sets the distance limits (it never moves
      // the camera) and is idempotent, so the later autoAdjustClippingPlanes()
      // call — which sets the same scale — is a no-op for it.
      const metaBoundsForScale = this.getSceneBoundsFromMetadata();
      if (metaBoundsForScale) {
        this.controls.setSceneScale(getBoundingBoxDiagonal(metaBoundsForScale));
      }

      // Apply viewer config from zarr (camera position, background color).
      // The helper returns whether an explicit camera position was applied;
      // also extract once more to detect author-set target/targetNode.
      const viewerConfig = root.userData?.viewerConfig as ZarrViewerConfig | undefined;
      const { positionApplied, appliedUp } = this.applyZarrViewerConfig(root);
      if ((options.applyViewerConfigFov || positionApplied) && viewerConfig) {
        const fovOverride = extractRenderingOverrides(viewerConfig).fov;
        const validFovOverride =
          fovOverride !== undefined &&
          Number.isFinite(fovOverride) &&
          fovOverride >= config.camera.fovMin &&
          fovOverride <= config.camera.fovMax;
        if (fovOverride !== undefined && (options.applyViewerConfigFov || validFovOverride)) {
          this.setFov(fovOverride);
        }
      }
      // The scene up governs every camera fit/reset (Home/F, center-on-
      // origin, this auto-frame): world +Y unless the author set one.
      this.sceneUp.copy(appliedUp ?? DEFAULT_SCENE_UP);
      const camOverrides = viewerConfig ? extractCameraOverrides(viewerConfig) : {};
      const hasAuthorTarget = !!(camOverrides.target || camOverrides.targetNode);

      // Auto-frame camera to fit scene contents, unless the zarr author specified
      // a camera position. Only an explicit position suppresses auto-framing — a
      // target/targetNode alone means the author wants the orbit pivot set but
      // still expects the camera to be at a sensible distance.
      if (!positionApplied) {
        // No author camera position — auto-frame using metadata bounds.
        // If the author set a target, preserve it as the look-at point
        // instead of overwriting with bounding box center.
        this.autoFrameCamera(hasAuthorTarget);
      }

      // Auto-adjust clipping planes using scene bounds from metadata
      // (must run AFTER autoFrameCamera since camera position affects clipping)
      this.autoAdjustClippingPlanes();

      log.info(
        Modules.SCENE_MANAGER,
        'Scene loaded. Press F to reset the camera (authored view if set, else fit to bounds).'
      );
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

  /** The baked environment map the loader found in the store, if any (see `loadBakedEnvironment`). */
  getSceneBakedEnvironment(): BakedEnvironment | null {
    const root = this.scene.children.find((c) => c.name === 'LuxarScene');
    return (root?.userData?.bakedEnvironment as BakedEnvironment | undefined) ?? null;
  }

  /**
   * Give the scene environment what a live `scene` capture needs (the init pipeline
   * calls this once the load-activity predicate exists). The capture pushes the cube
   * camera's params (a square drawing buffer, perspective, pixel ratio 1) to the
   * material manager so point and line footprints render at the right size in the six
   * faces — the 90° projection itself is read in shader from the cube camera's matrix
   * — and restores the main camera's push afterwards through the ordinary path.
   */
  attachEnvironmentRuntime(isSettled: () => boolean): void {
    const root = (): THREE.Object3D | null =>
      this.scene.children.find((c) => c.name === 'LuxarScene') ?? null;
    this.environment?.attachRuntime({
      sceneRoot: root,
      pushCaptureCameraParams: (resolution) => {
        materialManager.updateCameraParams(
          new THREE.Vector2(resolution, resolution),
          false,
          undefined,
          1
        );
      },
      restoreCameraParams: () => this.updateMaterialsForCurrentCamera(),
      isSettled,
      baseUrl: () => root()?.userData?.zarrBaseUrl as string | undefined,
    });
  }

  /** Arm WebGL blend warm-up after all scene-dependent dataset setup completes. */
  public warmBlendModePrograms(): Promise<void> {
    return warmSceneBlendModePrograms(this.scene);
  }

  /**
   * Apply viewer config from zarr (camera position/target/up, background
   * color). Thin delegate over `applyZarrViewerConfig` in
   * scene-manager/camera/camera-setup; forwards the helper's
   * `positionApplied` flag so `loadSceneData` can suppress auto-framing.
   * (Author target alone does NOT suppress auto-framing.)
   */
  private applyZarrViewerConfig(root: THREE.Group): {
    positionApplied: boolean;
    appliedUp: THREE.Vector3 | null;
  } {
    return applyZarrViewerConfigHelper(root, this.camera, this.controls, this.scene);
  }

  /**
   * Clear all loaded content from the scene, keeping lights and background.
   * Thin delegate over `clearLoadedSceneContent` in
   * scene-manager/render-pipeline/scene-disposal.
   */
  private clearSceneContent(): void {
    clearBlendModeProgramWarmup();
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
    // Prefer the scene's AUTHORED camera (zarr viewer_config) when it pins an
    // explicit position: F should return to the author's intended framing, not
    // re-fit to the raw min/max bounding box. A bounds fit zooms out to include
    // sparse outliers (e.g. Gaia halo stars out to ~10 kpc), shrinking the
    // subject to a dot — the author already framed around a robust extent, so
    // honour it. Fall back to the bounds fit only when there is no authored
    // camera position.
    const root = this.scene.children.find((c) => c.name === 'LuxarScene') as
      THREE.Group | undefined;
    const viewerConfig = root?.userData?.viewerConfig as ZarrViewerConfig | undefined;
    if (root && viewerConfig?.camera?.position) {
      // Reset the up vector to the scene up first: orbiting overwrites
      // camera.up every frame, so re-applying the authored camera without this
      // would inherit the accumulated roll (the bounds-fit path does the same
      // reset). If the author pinned an up, applyZarrViewerConfig overrides it.
      this.camera.up.copy(this.sceneUp);
      this.applyZarrViewerConfig(root);
      this.controls.saveState();
      this.lastBoundingBoxCenter.copy(this.controls.getFocusTarget());
      log.info(Modules.SCENE_MANAGER, 'Recentered to authored camera (viewer_config)');
      return;
    }
    const center = centerCameraOnScene(this.scene, this.camera, this.controls, this.sceneUp);
    if (center) this.lastBoundingBoxCenter.copy(center);
  }

  /**
   * Frame the camera on one object subtree (the per-layer sibling of
   * {@link centerCameraOnScene}). Returns false when the subtree holds no
   * framable geometry (e.g. a partition whose parts haven't streamed yet).
   */
  public fitCameraToObject(obj: THREE.Object3D): boolean {
    return frameCameraOnObject(obj, this.camera, this.controls, this.sceneUp) !== null;
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
      log.success(Modules.SCENE_MANAGER, 'Centered on origin (native center)');
    } else {
      // Switch to bounding box center
      this.centerCameraOnScene();
      this.isCenteredOnBoundingBox = true;
      log.success(Modules.SCENE_MANAGER, 'Centered on bounding box');
    }
  }

  /**
   * Center camera and controls on the origin (0,0,0) at the current
   * distance. Also used by the rail Home popover's "Center on origin"
   * action. Clears the bounding-box-centered flag so a subsequent
   * {@link toggleCentering} switches back to the bounding box.
   */
  public centerOnOrigin(): void {
    centerOnOrigin(this.camera, this.controls, this.sceneUp);
    this.isCenteredOnBoundingBox = false;
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
    this.resizer.scheduleResize(
      () => this.makeResizeCtx(),
      () => this.measureViewport()
    );
  }

  /**
   * Measure the viewport the canvas should fill.
   *
   * **Fullscreen-first**: while any fullscreen is active the canvas is styled
   * to fill the screen (`100vw/100vh`, see
   * `window-event-handler.onFullscreenChange`, which keys off the same
   * `isDocumentFullscreen()` check — standard + webkit), so its DOM parent — an embed
   * container — no longer reflects its displayed size. Measure the window.
   *
   * **Parent-first** otherwise: Three.js stamps inline `width`/`height` px
   * styles onto the canvas on every `setSize`, so the canvas's own client box
   * reflects our *last stamp*, not the host's layout. The parent element (the
   * embedder's frame, or `document.body` in the standalone app — sized 100% by
   * `base/layout.css`) is the box that actually tracks layout changes. Falls
   * back to the canvas's own box, then the window, when the parent reports
   * zero (detached canvas, jsdom).
   */
  private measureViewport(): { width: number; height: number } {
    if (typeof document !== 'undefined' && isDocumentFullscreen()) {
      return { width: window.innerWidth, height: window.innerHeight };
    }
    const canvas = this.renderer.domElement;
    const parent = canvas.parentElement;
    return {
      width: parent?.clientWidth || canvas.clientWidth || window.innerWidth,
      height: parent?.clientHeight || canvas.clientHeight || window.innerHeight,
    };
  }

  /**
   * Resize to the canvas's container box rather than the window.
   *
   * This is the embedding-safe resize path: an embedded canvas lives inside
   * a host container whose size can change without the window changing
   * (sidebars, splitters, flex/grid reflow), so a `window.innerWidth`-based
   * resize would stamp window-sized inline styles onto the canvas and
   * overflow the host frame. Sizing comes from {@link measureViewport}
   * (parent-first). Synchronous via `resizeNow`; callers that fire it from
   * a `ResizeObserver` already get browser-batched delivery (~once/frame).
   */
  public resizeToCanvas(): void {
    const { width, height } = this.measureViewport();
    this.resizer.resizeNow(width, height, this.makeResizeCtx());
  }

  /** Build the per-call ResizeCtx snapshot used by the resize orchestrator. */
  private makeResizeCtx() {
    return {
      renderer: this.renderer,
      camera: this.camera,
      postProcessing: this.postProcessing,
      pixelRatioOverride: this.pixelRatioOverride,
      updateMaterialsForCurrentCamera: () => this.updateMaterialsForCurrentCamera(),
    };
  }

  /**
   * The pixel ratio the renderer is currently sized for: the explicit
   * override if one is engaged, otherwise the live ceiling — and clamped
   * to that ceiling either way.
   *
   * Exposed so callers that need to re-apply the pixel ratio by hand
   * (the recording session, after a scaled capture) can ask for the
   * value the resize path would use instead of reaching for
   * `window.devicePixelRatio`, which respects neither the override nor
   * the pixel-ratio cap.
   */
  get activePixelRatio(): number {
    return getActivePixelRatio(this.pixelRatioOverride);
  }

  /**
   * Store/clear the explicit DPR override and return the effective DPR.
   * Delegates the math to dpr-policy.computePixelRatioOverride.
   */
  private setPixelRatioOverride(dpr: number): number {
    const { override, active } = computePixelRatioOverride(dpr);
    this.pixelRatioOverride = override;
    return active;
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

    // Resize through the orchestrator so the renderer / post-processing /
    // material-uniform pipeline stays in lockstep with the normal resize
    // path. resizeNow() is synchronous (no rAF coalescing), matching the
    // frame-level granularity AdaptiveDPRManager already runs at.
    this.resizer.resizeNow(w, h, this.makeResizeCtx());

    log.update(
      Modules.SCENE_MANAGER,
      `Adaptive DPR: ${activeDPR.toFixed(2)} (buffer: ${Math.round(w * activeDPR)}x${Math.round(
        h * activeDPR
      )})`
    );
  }

  /**
   * Update perspective camera FOV with bounds checking. Returns true when
   * applied. In orthographic mode there is no live FOV to change, but the
   * request is applied to the stashed perspective FOV so a FOV set while in
   * ortho is honored on the next ortho→perspective swap.
   */
  updateFOV(deltaY: number): boolean {
    if (!isPerspectiveCamera(this.camera)) {
      // Ortho renders no FOV, but keep the perspective stash coherent so a
      // Reset-to-Defaults / zarr-authored FOV applied while in ortho is honored
      // on the next ortho→perspective swap.
      const fovChange = deltaY * config.camera.fovSensitivity;
      this.lastPerspectiveFov = validateFOV(
        this.lastPerspectiveFov + fovChange,
        config.camera.fovMin,
        config.camera.fovMax
      );
      return true;
    }
    const applied = adjustFOV(this.makeCameraMaterialsCtx(), deltaY);
    if (applied) this.dispatchEvent({ type: 'change' });
    return applied;
  }

  /** Update camera clipping planes with validation. */
  updateClippingPlanes(near: number, far: number): void {
    if (applyClippingPlanes(this.camera, near, far)) {
      this.dispatchEvent({ type: 'change' });
    }
  }

  /**
   * Auto-adjust clipping planes from scene bounds (metadata first,
   * geometry fallback). Also feeds the bounding-box diagonal into
   * the scale-aware controls.
   */
  autoAdjustClippingPlanes(): { near: number; far: number } {
    const { near, far, applied } = autoAdjustFromBounds(this.makeClippingCtx());
    if (applied) this.dispatchEvent({ type: 'change' });
    return { near, far };
  }

  /**
   * Auto-frame the camera to fit the scene contents using metadata
   * bounds. Delegates to the camera-framing helper. Updates the
   * centering-state tracking fields when framing succeeds.
   *
   * @param preserveTarget If true, keep the current controls target
   *   (set by zarr viewer_config) instead of overwriting it with
   *   the bounding box center.
   */
  private autoFrameCamera(preserveTarget: boolean = false): void {
    const result = autoFrameCamera(
      this.camera,
      this.controls,
      this.getSceneBoundsFromMetadata(),
      preserveTarget,
      this.sceneUp
    );
    if (result.framed && result.center) {
      this.isCenteredOnBoundingBox = !preserveTarget;
      this.lastBoundingBoxCenter.copy(result.center);
    }
  }

  /**
   * Invalidate cached scene bounds. Called on scene load / clear.
   * Display dims are immutable per scene, so no invalidation is
   * needed for dimension navigation.
   */
  private invalidateBoundsCache(): void {
    this.boundsCache.invalidate();
  }

  /**
   * Update dynamic clipping planes using cached bounding sphere projection.
   *
   * Called each frame by AnimationController. Uses a cached bounding sphere
   * (invalidated on scene load/clear) for smooth near/far values with zero
   * object allocations and — once the metadata bounds are cached — zero
   * per-frame scene graph traversal. A metadata-less scene is the exception:
   * the cache has no negative caching, so the per-frame `ensure()` re-walks
   * the graph each frame (see `clipping/scene-bounds-cache.ts`).
   */
  updateDynamicClippingPlanes(): void {
    if (!this.dynamicClippingEnabled) return;
    updateDynamicFromCache(this.makeClippingCtx());
  }

  /**
   * Build the narrow ctx that the clipping-policy helpers consume.
   * Created on demand to keep helper signatures stable as
   * subsystem fields evolve.
   */
  private makeClippingCtx(): ClippingCtx {
    return {
      camera: this.camera,
      controls: this.controls,
      scene: this.scene,
      boundsCache: this.boundsCache,
      getSceneBoundsFromMetadata: () => this.getSceneBoundsFromMetadata(),
    };
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
   * Get 3D bounding box from scene metadata, projecting nD bounds
   * to display dimensions. Delegates to the bounds-cache helper.
   *
   * @returns 3D bounding box or null if metadata bounds not available
   */
  private getSceneBoundsFromMetadata(): BoundingBox | null {
    return computeBoundsFromMetadata(this.scene);
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
    // Cancel any pending resize operations to prevent memory leaks.
    this.resizer.dispose();
    clearBlendModeProgramWarmup();

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

    // The prefiltered environment, if a physical mesh ever caused it to be built.
    // Before the renderer goes: the PMREM target is a GPU resource of that renderer.
    this.unsubscribeEnvironment?.();
    this.unsubscribeEnvironment = null;
    this.environment?.dispose();
    this.environment = null;

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
    // scene-manager/render-pipeline/scene-disposal so the same one-shot
    // final-dispose pass is unit-testable in isolation.
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
   * Set the turntable speed, in REVOLUTIONS PER MINUTE.
   *
   * A full turn takes `60 / speed` seconds — 1.0 is one turn a minute, the
   * 0.25 default is one turn every four minutes. The unit is inherited from
   * three.js `OrbitControls` and is what `auto_rotate_speed` means in every
   * authored scene, so it is the stored unit; the Navigation popover shows the
   * equivalent PERIOD in seconds (see `secondsPerTurnFromRpm`).
   *
   * Frame-rate independent: the update step scales by `deltaTime`, so the
   * turn takes the same wall-clock time at 30 fps and at 144.
   *
   * @param speed - Revolutions per minute (> 0).
   */
  setAutoRotateSpeed(speed: number): void {
    this.controls.setAutoRotateSpeed(speed);
  }

  /**
   * Set the camera-frame or fixed scene axis the turntable revolves around;
   * see {@link AutoRotateAxis}.
   */
  setAutoRotateAxis(axis: AutoRotateAxis): void {
    this.controls.setAutoRotateAxis(axis);
    log.custom(LogEmoji.SCENE, Modules.SCENE_MANAGER, `Auto-rotation axis: ${axis}`);
  }

  /**
   * Enable/disable the auto-dolly: a sinusoidal in-and-out motion along the
   * view direction, the turntable's radial sibling. Live in orbit AND ortho
   * (in 2D it breathes `camera.zoom`).
   */
  setAutoDolly(enabled: boolean): void {
    this.controls.setAutoDolly(enabled);
    log.custom(
      LogEmoji.SCENE,
      Modules.SCENE_MANAGER,
      `Auto-dolly ${enabled ? 'enabled' : 'disabled'}`
    );
  }

  /** Dolly amplitude as a percent of the viewing distance (15 → ±15%). */
  setAutoDollyAmplitudePercent(percent: number): void {
    this.controls.setAutoDollyAmplitudePercent(percent);
  }

  /** Dolly period in seconds (one full in-and-out oscillation). */
  setAutoDollyPeriod(seconds: number): void {
    this.controls.setAutoDollyPeriod(seconds);
  }

  /** Orbit wheel-zoom speed (live; shared with ortho — same control class). */
  setOrbitZoomSpeed(speed: number): void {
    this.controls.setOrbitZoomSpeed(speed);
  }

  /** Orbit damping factor (live; shared with ortho — same control class). */
  setOrbitDampingFactor(factor: number): void {
    this.controls.setOrbitDampingFactor(factor);
  }

  /** Fly-mode mouse-look sensitivity (live on the current fly controls). */
  setFlyLookSpeed(speed: number): void {
    this.controls.setFlyLookSpeed(speed);
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

  /** Current auto-dolly state (see {@link setAutoDolly}). */
  getAutoDolly(): boolean {
    return this.controls.getAutoDolly();
  }

  /** Current turntable axis (see {@link setAutoRotateAxis}). */
  getAutoRotateAxis(): AutoRotateAxis {
    return this.controls.getAutoRotateAxis();
  }

  /**
   * Switch camera control type. Handles camera swap for ortho mode and
   * dispatches `camera-changed` at this public call site.
   *
   * @param type Control type ('orbit', 'fly', or 'ortho')
   */
  setControlType(type: ControlType): void {
    const { cameraChanged } = cameraModeSetControlType(type, this.makeCameraModeCtx());
    if (cameraChanged) {
      this.dispatchEvent({ type: 'camera-changed' });
    }
  }

  /** Get current control type. */
  getControlType(): ControlType {
    return this.controls.getControlType();
  }

  /** Build the narrow ctx that the camera-mode helpers consume. */
  private makeCameraModeCtx(): CameraModeCtx {
    return {
      getCamera: () => this.camera,
      setCamera: (camera) => {
        this.camera = camera;
      },
      controls: this.controls,
      renderer: this.renderer,
      postProcessing: this.postProcessing,
      updateMaterialsForCurrentCamera: () => this.updateMaterialsForCurrentCamera(),
      getLastPerspectiveFov: () => this.lastPerspectiveFov,
      setLastPerspectiveFov: (fov) => {
        this.lastPerspectiveFov = fov;
      },
    };
  }

  /**
   * Update all materials with current camera projection parameters.
   * Perspective: FOV-based. Orthographic: frustum-based.
   *
   * Called internally on resize and camera changes. Also called
   * by RecordingPanel when the renderer is resized for offline
   * capture.
   */
  updateMaterialsForCurrentCamera(): void {
    updateCameraMaterials(this.makeCameraMaterialsCtx());
  }

  /** Build the narrow ctx that the camera-materials helpers consume. */
  private makeCameraMaterialsCtx(): CameraMaterialsCtx {
    return {
      renderer: this.renderer,
      camera: this.camera,
      scene: this.scene,
      boundsCache: this.boundsCache,
      bufferSize: this._bufferSize,
    };
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
