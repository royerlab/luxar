// Main application class for the Luxar scene player

import * as THREE from 'three';
import { SceneManager } from '../scene/scene-manager';
import {
  captureSnapshot as captureViewerSnapshot,
  restoreSnapshot as restoreViewerSnapshot,
  type ViewerSnapshot,
} from './viewer-snapshot';
import { AnimationController } from '../scene/animation-controller';
import { InputHandler } from '../input/input-handler';
import { DimensionSliders } from '../ui/panels/dimension-sliders';
import { RenderingControls } from '../ui/rendering-controls';
import { cleanupUI, clearError, showError, showHelpOverlay } from '../ui/helpers';
import { config } from '../config';
import { DatasetBrowser } from '../ui/panels/dataset-browser';
import { log, Modules } from '../utils/log';
import { getManagerRegistry } from './manager-registry';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';
import { ResolutionIndicator } from '../ui/components/resolution-indicator';
import { PerformanceMonitor } from '../ui/monitors/performance-monitor';
import { DataMonitorManager } from '../ui/monitors/data-monitor-manager';
import { DebugConsole } from '../ui/panels/debug-console';
import { SceneLoaderManager, getSceneLoader } from '../data/scene-loader-manager';
import { ScaleBar } from '../ui/components/scale-bar';
import { ColormapLegend } from '../ui/components/colormap-legend';
import { RecordingPanel } from '../ui/recording-panel';
import { LayersPanel } from '../ui/layers';
import { ThemeManager } from '../themes/theme-manager';
import type { ZarrViewerConfig } from '../types/zarr';
import { OverlayManager } from '../ui/helpers/overlay-manager';
import * as zarr from 'zarrita';
import { PickingSystem, type PickResult } from '../rendering/picking/picking-system';
import { LabelLoader } from '../data/loaders/label-loader';
import { ImageLabelLoader } from '../data/loaders/image-label-loader';
import type { LoaderConfig } from '../data/data-loader-types';
import { consoleInterceptor } from '../utils/console-interceptor';
import { EventGroup } from '../utils/event-group';
import { setWasmJsUrl } from '../wasm';
import { setDataWorkerUrl } from '../workers/worker-pool';
import { replaceBrowserDataSourceUrl } from '../config/url-params';
import { classifyBrowserUrl } from './browser-decision';
import { applyViewerConfigState as applyViewerConfigStateHelper } from './viewer-config-applier';
import { computeDebugState } from './debug-state';
import { buildDebugCacheHelpers } from './debug-cache-helpers';

/**
 * Init-time options for {@link LuxarApp.init}.
 *
 * Typically constructed by `main.ts` from `readUrlParams()`, but any caller
 * can provide values directly — useful for tests, embedding, and notebook
 * integrations where `window.location` is not the right source.
 */
export interface LuxarAppOptions {
  /**
   * Canvas element to render into. The standalone app's main.ts resolves
   * this via `document.getElementById('app')`; embedders pass any
   * HTMLCanvasElement they own.
   */
  canvas: HTMLCanvasElement;
  /** Dataset URL. Defaults to {@link config.defaultZarrPath}. */
  src?: string;
  /** Expose `window.__luxarDebug` and verbose hardware logging. */
  debug?: boolean;
  /** Cache and prefetch flags forwarded to the data loader. */
  loaderConfig?: LoaderConfig;
  /**
   * Reflect the loaded dataset URL in the browser address bar via
   * `history.replaceState` so the page can be reloaded or shared.
   *
   * Defaults to `false` for programmatic/embedded safety. The standalone
   * bootstrap sets this to `true` explicitly.
   */
  updateBrowserUrl?: boolean;

  /**
   * Absolute URL to the WASM JS shim (`luxar_wasm.js`).
   *
   * Defaults to `new URL('../wasm/luxar_wasm.js', import.meta.url)` —
   * resolved relative to the bundled JS, which works for Vite, Rollup,
   * webpack 5, and most modern bundlers. Embedders whose bundlers don't
   * support `import.meta.url` for asset URLs (or who ship the WASM files
   * from a non-default location) override this.
   */
  wasmPath?: string;

  /**
   * Absolute URL to the data-worker module bundle.
   *
   * Defaults to `new URL('./data-worker.ts', import.meta.url)`. Override
   * if your bundler can't resolve worker URLs that way.
   */
  workerPath?: string;

  /**
   * Open the data-loading monitor in expanded mode on the Cache tab as
   * soon as the scene is wired up. Set by the standalone bootstrap when
   * `?cache-stats` is in the URL; embedders can pass it explicitly when
   * profiling cache behaviour.
   */
  openCacheStats?: boolean;
}

export class LuxarApp {
  private sceneManager!: SceneManager;
  private animationController!: AnimationController;
  private performanceMonitor!: PerformanceMonitor;
  private debugConsole!: DebugConsole;
  private inputHandler!: InputHandler;
  private renderingControls!: RenderingControls;
  private adaptiveDPRManager!: AdaptiveDPRManager;
  private resolutionIndicator!: ResolutionIndicator;
  private datasetBrowser?: DatasetBrowser;
  private scaleBar?: ScaleBar;
  private colormapLegend?: ColormapLegend;
  private recordingPanel?: RecordingPanel;
  private layersPanel?: LayersPanel;
  private overlayManager?: OverlayManager;
  private pickingSystem?: PickingSystem;
  private labelLoader?: LabelLoader;
  private imageLabelLoader?: ImageLabelLoader;
  /**
   * Per-init EventGroup for picking-system listeners (canvas mousemove,
   * window resize, controls/scene-manager subscriptions). Re-disposed and
   * rebuilt on each scene load.
   */
  private pickingEvents = new EventGroup();
  private isInitialized = false;
  /**
   * Re-entrance guard for {@link dispose}. Set while a dispose is in flight
   * so a `beforeunload` callback that fires mid-dispose (or any nested call)
   * is a no-op rather than running the teardown a second time.
   */
  private isDisposing = false;
  /**
   * Idempotency guard for {@link dispose}. Once teardown completes, further
   * `dispose()` calls are no-ops — fields still reference disposed instances,
   * so without this flag we would invoke `dispose()` on already-disposed
   * components (potentially double-freeing GPU resources). Reset by `init()`.
   */
  private isDisposed = false;
  /**
   * App-level event listeners (beforeunload, focus, visibilitychange,
   * open-dataset-browser, plus the picking system's mousemove + control
   * change subscriptions). Disposed in one call from {@link dispose}.
   */
  private events = new EventGroup();

  /**
   * Snapshot of init-time options. Populated by `init()` and read by
   * setupDebugInterface, dataset-browser callbacks, and other components
   * that need URL-derived flags without re-reading `window.location`.
   *
   * Definitely-assigned: every method that reads `this.options` runs after
   * `init()`, which assigns the field as its first action.
   */
  private options!: LuxarAppOptions;

  /**
   * Initialize the complete Luxar application.
   *
   * Sets up the complete visualization pipeline including:
   * - WebGL renderer and scene manager
   * - Animation loop with post-processing
   * - Input handling (keyboard/mouse)
   * - UI controls (dimension sliders, rendering settings)
   * - Data loading with spatial indexing
   *
   * The initialization sequence is carefully ordered to ensure the
   * animation loop starts BEFORE data loading, providing visual feedback
   * even during long load operations.
   *
   * @param options - Init-time options. URL parameters are not consulted
   *                  here — main.ts is responsible for reading them and
   *                  passing the result.
   *
   * @returns Promise that resolves when initialization is complete and
   *          dataset loading has started (may still be loading in background).
   *          Does NOT wait for all chunks to load.
   *
   * @throws {Error} If WebGL is not supported by browser
   * @throws {Error} If scene manager initialization fails
   * @throws {Error} Dataset loading errors are caught and displayed to user
   *
   * @example
   * ```typescript
   * const app = new LuxarApp();
   * await app.init({
   *   canvas: document.getElementById('app') as HTMLCanvasElement,
   *   src: 'https://example.com/cells.zarr',
   * });
   * ```
   *
   * @see {@link SceneManager} for rendering pipeline setup
   * @see README.md - initialization sequence section for detailed init flow
   */
  async init(options: LuxarAppOptions): Promise<void> {
    if (this.isInitialized) {
      throw new Error('LuxarApp is already initialized. Call dispose() before initializing again.');
    }

    // Browser-environment guard. SceneManager and InputHandler reach for
    // window/document/localStorage unconditionally, so a friendly upfront
    // error beats a cryptic ReferenceError half-way through init for SSR
    // or non-browser callers.
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      throw new Error(
        'LuxarApp requires a browser environment (window and document must be defined).'
      );
    }

    // Reset the idempotency guard so a fresh init followed by dispose works
    // even if the same instance was previously initialized and disposed.
    this.isDisposed = false;
    this.options = options;

    // Forward asset-URL overrides to the WASM and worker modules. Skipped
    // when the option is undefined so the modules use their default
    // `import.meta.url`-based resolution. NOTE: the override is module-level
    // and sticks across init() calls — once set, a subsequent init() without
    // the option does not reset to the default. In practice we only support
    // one LuxarApp per page in v1, so this is fine.
    if (options.wasmPath) setWasmJsUrl(options.wasmPath);
    if (options.workerPath) setDataWorkerUrl(options.workerPath);

    try {
      // Inform users about expected console messages
      log.info(
        Modules.LUXAR,
        'Note: You may see 404 errors for optional features like spatial indexes and array attributes.'
      );
      log.info(
        Modules.LUXAR,
        'These are expected and do not indicate a problem - the app checks for optional features that may not exist.'
      );

      const sceneSrc = this.options.src ?? config.defaultZarrPath;

      // Initialize scene manager first
      this.sceneManager = new SceneManager();
      await this.sceneManager.init({
        canvas: this.options.canvas,
        debug: this.options.debug,
      });

      // Initialize animation controller with HDR post-processing.
      // The PerformanceMonitor UI panel is constructed up here (not in
      // the controller) and subscribes to the bus events the
      // controller emits each frame. Owning it at the app level keeps
      // the lower scene/ layer free of UI imports.
      this.animationController = new AnimationController(
        this.sceneManager.controls,
        this.sceneManager.postProcessing
      );
      this.performanceMonitor = new PerformanceMonitor();
      this.debugConsole = new DebugConsole();

      // Set up per-frame callback for dynamic clipping plane updates
      // Uses unique ID so it won't conflict with other per-frame callbacks (e.g., dimension animation)
      this.animationController.addPerFrameCallback('dynamic-clipping', () => {
        this.sceneManager.updateDynamicClippingPlanes();
      });

      // Initialize adaptive DPR manager for dynamic resolution scaling
      this.adaptiveDPRManager = new AdaptiveDPRManager();
      this.adaptiveDPRManager.setRenderer(this.sceneManager);
      this.animationController.setAdaptiveDPRManager(this.adaptiveDPRManager);

      // Initialize resolution indicator and connect to DPR manager
      this.resolutionIndicator = new ResolutionIndicator();
      // Display target FPS rounded up from maxFPS (58 → 60) since targetFPS (55) is a hysteresis threshold
      const displayTargetFPS = Math.ceil(config.adaptiveDPR.maxFPS / 5) * 5;
      this.resolutionIndicator.setTargetFPS(displayTargetFPS);
      this.adaptiveDPRManager.setOnDPRChangeCallback((dpr, isReducedResolution) => {
        if (isReducedResolution) {
          this.resolutionIndicator.show(dpr);
        } else {
          // Reset the indicator so it can show again on next reduced resolution mode activation
          this.resolutionIndicator.reset();
        }
      });

      // Re-register picking-system / GPU-pool resources after a WebGL
      // context-restore event. SceneManager rebuilds the renderer +
      // post-processing + material cache before dispatching, then we
      // call NodeFactory.rebuildAfterContextRestore on the loaded
      // scene so the picking system gets fresh registrations against
      // the new context.
      if (typeof this.sceneManager.addEventListener === 'function') {
        this.sceneManager.addEventListener('webgl-context-restored', () => {
          const sceneLoader = getSceneLoader('default');
          if (sceneLoader && this.sceneManager.scene) {
            sceneLoader.nodeFactory.rebuildAfterContextRestore(this.sceneManager.scene);
          }
        });
      }

      // Inject the monitor factory into SceneLoaderManager so each
      // SceneLoader can resolve its UI monitor without the data/
      // layer importing ui/ directly. Closes the last layer-cruiser
      // exception (Phase 8.6.e).
      SceneLoaderManager.getInstance().setMonitorFactory((monitorId) => {
        if (typeof document === 'undefined') return null;
        const mgr = DataMonitorManager.getInstance();
        if (!mgr.hasMonitor(monitorId)) {
          mgr.createMonitor(monitorId, document.body);
        }
        return mgr.getMonitor(monitorId) ?? null;
      });

      // Initialize input handler. The DimensionSliders factory is
      // injected here so the input layer never imports the concrete
      // ui/ panel — input → ui is a layer-cruiser violation.
      this.inputHandler = new InputHandler(
        this.sceneManager,
        this.animationController,
        this.performanceMonitor,
        this.debugConsole,
        (config) => new DimensionSliders(config)
      );
      this.inputHandler.init();

      // Initialize rendering controls
      this.renderingControls = new RenderingControls(
        this.sceneManager.postProcessing,
        this.sceneManager
      );

      // Connect rendering controls to animation controller
      this.renderingControls.setAnimationController(this.animationController);

      // Connect rendering controls to adaptive DPR manager for performance UI
      this.renderingControls.setAdaptiveDPRManager(this.adaptiveDPRManager);

      // Connect rendering controls to input handler
      this.inputHandler.setRenderingControls(this.renderingControls);

      // Initialize recording panel (screenshot/video capture)
      this.recordingPanel = new RecordingPanel(this.sceneManager, this.animationController);
      this.recordingPanel.setPanelStateCallbacks(
        () => this.getPanelVisibilityStates(),
        (states) => this.restorePanelVisibilityStates(states)
      );
      this.recordingPanel.setAdaptiveDPRManager(this.adaptiveDPRManager);
      this.inputHandler.setRecordingPanel(this.recordingPanel);

      // Initialize layers panel (per-node controls)
      this.layersPanel = new LayersPanel(document.body, this.animationController);
      this.inputHandler.setLayersPanel(this.layersPanel);

      // Start animation loop first to ensure background is rendered
      this.animationController.startAnimation();

      // Check if source might be a directory (for navigation)
      if (await this.shouldShowBrowser(sceneSrc)) {
        // Show dataset browser for directory navigation
        try {
          this.showDatasetBrowser();
        } catch (error) {
          log.warning(
            Modules.APP,
            'Dataset browser initialization had issues, but browser is shown:',
            error
          );
          // Browser is shown even if navigation fails - user can use manual entry
        }
      } else {
        // Load scene data directly
        await this.loadDataset(sceneSrc);
      }

      // Dispose on page unload (cleans up listeners, workers, GPU resources).
      this.setupDisposeOnUnload();

      // Setup dataset browser keyboard shortcut
      this.setupDatasetBrowserShortcut();

      // Setup window focus handling to trigger render on focus
      this.setupFocusHandling();

      // Expose debug interface for testing and AI-assisted development
      this.setupDebugInterface();

      this.isInitialized = true;
    } catch (error) {
      log.error(Modules.APP, 'Failed to initialize Luxar app:', error);
      // Tear down whatever partial state was constructed before the throw.
      // dispose() is now defensive (per-field `if (this.x)` guards) so it
      // safely handles a half-built app. The caller's error handler is
      // expected to surface a fresh, top-level error UI; any in-progress
      // error UI from sub-loaders is wiped along with everything else.
      this.dispose();
      throw error;
    }
  }

  /**
   * Check if we should show the dataset browser
   */
  private async shouldShowBrowser(src: string): Promise<boolean> {
    // Synchronous classification: empty / trailing-slash URLs always
    // need the browser, no point firing a zarr-metadata probe.
    if (classifyBrowserUrl(src) === 'must-browse') return true;

    // Check if it's a Zarr dataset by looking for zarr metadata files
    // Try both v2 (.zgroup) and v3 (zarr.json) formats
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const zarrChecks = [
      fetch(src + '/.zgroup', { method: 'HEAD', signal: controller.signal }),
      fetch(src + '/.zattrs', { method: 'HEAD', signal: controller.signal }),
      fetch(src + '/zarr.json', { method: 'HEAD', signal: controller.signal }),
    ];

    try {
      // Short-circuit: return as soon as any probe confirms zarr metadata exists.
      // This avoids waiting for the zarr.json 404 on v2 stores (~200ms on slow networks).
      await Promise.any(
        zarrChecks.map((p) =>
          p.then((r) => {
            if (!r.ok) throw new Error('not ok');
            return r;
          })
        )
      );
      return false; // At least one zarr metadata file exists — load directly
    } catch {
      // All probes failed or errored — likely a directory, show browser
    } finally {
      clearTimeout(timeoutId);
    }

    return true;
  }

  /**
   * Show the dataset browser UI
   */
  private showDatasetBrowser(): void {
    // Close existing browser if any
    if (this.datasetBrowser) {
      return; // Browser already open
    }

    // Clear any existing error messages when opening the browser
    clearError();

    this.datasetBrowser = new DatasetBrowser({
      container: document.body,
      currentSrc: this.options.src,
      onDatasetSelect: async (fullUrl: string) => {
        // The browser now passes full URLs directly, preserving directory context
        // Strip any trailing slashes to ensure consistent URL format
        const cleanUrl = fullUrl.replace(/\/+$/, '');

        // Reflect the chosen dataset in the URL bar only for callers that opt in.
        // The standalone bootstrap opts in; programmatic/embedded usage defaults
        // to no host-page URL mutation.
        if (this.options.updateBrowserUrl === true) {
          replaceBrowserDataSourceUrl(cleanUrl);
        }

        // Track the new src in our options snapshot so a subsequent browser
        // open lands in the right directory.
        this.options = { ...this.options, src: cleanUrl };

        // Load the dataset
        await this.loadDataset(cleanUrl);
      },
      onClose: () => {
        this.datasetBrowser = undefined;
      },
    });
  }

  /**
   * Load a dataset and initialize UI
   */
  private async loadDataset(src: string): Promise<void> {
    // Clear any existing dimension UI
    this.inputHandler.clearDimensionUI();

    // Dispose previous-scene overlays upfront so they are cleared in lockstep
    // with clearSceneContent() — otherwise a failing scene load leaves the old
    // overlay DOM elements visible on top of an empty canvas.
    this.disposeOverlays();

    // Note: Monitor cleanup is handled by SceneLoader.loadScene() which calls
    // monitor.disconnectAllLoaders() when loading a new scene

    // Set scene ID for rendering controls persistence BEFORE loading scene
    // This ensures saved settings (like HDR intensity) are applied before materials are created
    this.renderingControls.setSceneId(src);

    // Load scene data (animation loop will continue even if this fails)
    await this.sceneManager.loadSceneData(src, this.options.loaderConfig);

    // Pass zarr viewer_config to rendering controls (available after scene loads).
    // If no localStorage settings exist for this scene, apply zarr defaults.
    const viewerConfig = this.sceneManager.getSceneViewerConfig();
    this.renderingControls.setZarrViewerConfig(viewerConfig);
    if (!this.renderingControls.hasStoredSettings() && viewerConfig) {
      this.renderingControls.applyZarrDefaults();
    }

    // Update fly speed slider range and value based on scene scale
    this.renderingControls.updateSceneScale();

    // Initialize UI components that depend on loaded scene data
    this.inputHandler.initDimensionSliders();
    this.initScaleBar();

    const sceneLoader = getSceneLoader('default');
    if (sceneLoader?.sceneGraph && this.layersPanel) {
      const root = this.sceneManager.scene.children.find((c) => c.name === 'LuxarScene');
      if (root) {
        this.layersPanel.initFromScene(root as THREE.Group, sceneLoader.sceneGraph);
      }
    }

    // Initialize colormap legend after layers panel (needs layer state)
    if (this.layersPanel) {
      this.initColormapLegend();
    }

    // Initialize overlays (screen-space annotations from zarr)
    await this.initOverlays();

    // Initialize GPU picking system (if any node has labels)
    await this.initPicking();

    // Apply zarr viewer_config: UI visibility, theme, dimension state, animation
    this.applyViewerConfigState(viewerConfig);

    // ?cache-stats: open the data-loading monitor on the Cache tab. The
    // monitor was created during sceneManager.loadSceneData() above, so
    // it's safe to look it up via DataMonitorManager now.
    if (this.options.openCacheStats) {
      this.openCacheStatsView();
    }

    // Trigger animation to ensure scene is rendered immediately
    this.animationController.startAnimation();
  }

  /**
   * Open the data-loading monitor in expanded mode on the Cache tab.
   * Best-effort: silently skips when no monitor was created (e.g.
   * embedded contexts that disable the monitor).
   */
  private openCacheStatsView(): void {
    const monitor = DataMonitorManager.getInstance().getDefaultMonitor();
    if (!monitor) return;
    monitor.show();
    monitor.expand();
    monitor.setActiveTab('cache');
  }

  /**
   * Apply zarr viewer_config state that isn't handled by RenderingControls.
   *
   * RenderingControls handles the 47 rendering settings (bloom, AO, AA, etc.).
   * This method handles everything else: UI panel visibility, theme,
   * dimension navigation state, and animation state.
   *
   * Only explicitly set fields (not undefined) are applied — unset fields
   * preserve the viewer's built-in defaults.
   */
  private applyViewerConfigState(viewerConfig: ZarrViewerConfig | undefined): void {
    applyViewerConfigStateHelper(viewerConfig, {
      showHelp: showHelpOverlay,
      renderingControls: this.renderingControls,
      performanceMonitor: this.performanceMonitor,
      inputHandler: this.inputHandler,
      scaleBar: this.scaleBar,
      layersPanel: this.layersPanel,
      overlayManager: this.overlayManager,
      setTheme: (id) => ThemeManager.getInstance().setTheme(id),
      setDimensionValue: (i, v) => sceneDimsManager.setDimensionValue(i, v),
    });
  }

  /**
   * Initialize or recreate the scale bar overlay.
   * Creates a ScaleBar component and registers a per-frame callback
   * to update it as the camera moves.
   */
  private initScaleBar(): void {
    // Dispose previous instance if reloading
    if (this.scaleBar) {
      this.animationController.removePerFrameCallback('scale-bar');
      this.scaleBar.dispose();
    }

    this.scaleBar = new ScaleBar({
      camera: this.sceneManager.camera,
      controls: this.sceneManager.controls,
      canvas: this.sceneManager.renderer.domElement,
      targetWidthPx: config.ui.scaleBar.targetWidthPx,
      position: config.ui.scaleBar.position,
    });

    // Register per-frame update for live camera tracking
    this.animationController.addPerFrameCallback('scale-bar', () => {
      this.scaleBar?.update();
    });

    // Wire to input handler for keyboard toggle
    this.inputHandler.setScaleBar(this.scaleBar);
  }

  /**
   * Initialize the colormap legend overlay.
   * Shows per-layer colormap gradients with names and data ranges.
   */
  private initColormapLegend(): void {
    if (this.colormapLegend) {
      this.colormapLegend.dispose();
    }

    if (!this.layersPanel) return;

    try {
      this.colormapLegend = new ColormapLegend({
        layerState: this.layersPanel.layerState,
      });

      // Wire to input handler for keyboard toggle
      this.inputHandler.setColormapLegend(this.colormapLegend);
    } catch {
      // ColormapLegend requires DOM; may fail in test environments
    }
  }

  /**
   * Tear down the current OverlayManager, removing its DOM elements.
   */
  private disposeOverlays(): void {
    if (this.overlayManager) {
      this.overlayManager.dispose();
      this.overlayManager = undefined;
    }
  }

  /**
   * Initialize screen-space overlays from zarr metadata.
   * Creates an OverlayManager if the loaded scene contains overlays.
   */
  private async initOverlays(): Promise<void> {
    // Defensive: loadDataset() already disposes overlays upfront, but keep
    // this idempotent in case initOverlays() is called from another path.
    this.disposeOverlays();

    const root = this.sceneManager.scene?.children?.find((c) => c.name === 'LuxarScene') as
      | THREE.Group
      | undefined;

    const overlayConfigs = root?.userData?.overlayConfigs;
    const zarrBaseUrl = root?.userData?.zarrBaseUrl;

    if (overlayConfigs?.length > 0 && zarrBaseUrl) {
      this.overlayManager = new OverlayManager();
      await this.overlayManager.loadOverlays(overlayConfigs, zarrBaseUrl);
      this.inputHandler.setOverlayManager(this.overlayManager);
      this.recordingPanel?.setOverlayManager(this.overlayManager);
    }
  }

  /**
   * Initialize GPU picking system for hover tooltips.
   * Only activates if any scene node has labels or image labels
   * (has_labels / has_image_labels in .zattrs).
   * Wires up: PickingSystem → LabelLoader/ImageLabelLoader → OverlayManager.updateHoverContent.
   */
  private async initPicking(): Promise<void> {
    // Re-init: tear down listeners from any previous picking session.
    // (initPicking() also resets pickingEvents at the listener registration
    // site below, but doing it here too lets us early-return on no-labels
    // without leaking the previous session's listeners.)
    this.pickingEvents.dispose();
    this.pickingSystem?.dispose();
    this.labelLoader?.dispose();
    this.imageLabelLoader?.dispose();
    this.pickingSystem = undefined;
    this.labelLoader = undefined;
    this.imageLabelLoader = undefined;

    // Check if any node has labels or image labels
    const root = this.sceneManager.scene?.children?.find((c) => c.name === 'LuxarScene') as
      | THREE.Group
      | undefined;
    if (!root) return;

    let hasAnyLabels = false;
    let hasAnyImageLabels = false;
    root.traverse((obj) => {
      if (obj.userData?.attrs?.has_labels) {
        hasAnyLabels = true;
      }
      if (obj.userData?.attrs?.has_image_labels) {
        hasAnyImageLabels = true;
      }
    });
    if (!hasAnyLabels && !hasAnyImageLabels) return;

    // Get the scene loader for store/rootLoc access
    const sceneLoader = getSceneLoader('default');
    if (!sceneLoader) return;

    // Create label loader using the scene loader's zarr store
    const store = sceneLoader.zarrStore;
    if (!store) {
      log.warning(Modules.APP, 'Cannot init picking: zarr store not available');
      return;
    }
    const rootLoc = zarr.root(store);
    if (hasAnyLabels) {
      this.labelLoader = new LabelLoader(store, rootLoc);
    }
    if (hasAnyImageLabels) {
      this.imageLabelLoader = new ImageLabelLoader(store, rootLoc);
    }

    // Create picking system with result callback
    this.pickingSystem = new PickingSystem(
      this.sceneManager.renderer,
      this.sceneManager.camera,
      async (result: PickResult | null) => {
        try {
          if (!result) {
            this.overlayManager?.updateHoverContent(null);
            return;
          }
          const nodePath = result.mainNode.name;
          // Fetch text label and image URL in parallel
          const [label, imageUrl] = await Promise.all([
            this.labelLoader?.getLabel(nodePath, result.elementId) ?? Promise.resolve(null),
            this.imageLabelLoader?.getImageUrl(nodePath, result.elementId) ?? Promise.resolve(null),
          ]);
          const hasContent = label || imageUrl;
          this.overlayManager?.updateHoverContent(
            hasContent
              ? { label, imageUrl, nodeName: nodePath, elementIndex: result.elementId }
              : null
          );
        } catch (err) {
          // Don't let label loading errors kill the hover loop
          log.warning(Modules.APP, `Picking callback error: ${err}`);
          this.overlayManager?.updateHoverContent(null);
        }
      }
    );

    // Wire NodeFactory to create pick nodes for future scene loads
    sceneLoader.nodeFactory.setPickingSystem(this.pickingSystem);

    // Wire post-processing for lens distortion coordinate correction
    this.pickingSystem.setPostProcessing(this.sceneManager.postProcessing);

    // Retroactively register already-loaded nodes (scene loads before picking init)
    if (root) {
      sceneLoader.nodeFactory.registerExistingSceneNodes(root);
    }

    // DOM events go through EventGroup.on(); Three.js EventDispatcher events
    // (controls, sceneManager) use add() with a manual remove closure since
    // their addEventListener/removeEventListener signatures aren't EventTarget.
    const canvas = this.sceneManager.renderer.domElement;
    const handler = (e: MouseEvent) => this.pickingSystem?.onMouseMove(e);
    this.pickingEvents.on(canvas, 'mousemove', handler);

    const dirtyHandler = () => this.pickingSystem?.markDirty();
    this.sceneManager.controls.addEventListener('change', dirtyHandler);
    this.pickingEvents.add(() =>
      this.sceneManager.controls.removeEventListener('change', dirtyHandler)
    );
    this.pickingEvents.on(window, 'resize', dirtyHandler);

    // Suppress picking during orbit/pan/zoom — no expensive offscreen renders
    // while the user is navigating, and fade out stale hover labels.
    const controls = this.sceneManager.controls;
    const interactionStart = () => {
      this.pickingSystem?.suppress(true);
      this.overlayManager?.updateHoverContent(null);
    };
    const interactionEnd = () => {
      this.pickingSystem?.suppress(false);
    };
    controls.addEventListener('start', interactionStart);
    controls.addEventListener('end', interactionEnd);
    this.pickingEvents.add(() => controls.removeEventListener('start', interactionStart));
    this.pickingEvents.add(() => controls.removeEventListener('end', interactionEnd));

    // Update picking camera when perspective ↔ orthographic swap occurs
    const cameraChangedHandler = () => {
      this.pickingSystem?.setCamera(this.sceneManager.camera);
    };
    this.sceneManager.addEventListener('camera-changed', cameraChangedHandler);
    this.pickingEvents.add(() =>
      this.sceneManager.removeEventListener('camera-changed', cameraChangedHandler)
    );

    log.info(Modules.APP, 'GPU picking system initialized (labels detected)');
  }

  /**
   * Register a beforeunload handler that disposes the app on page unload.
   */
  private setupDisposeOnUnload(): void {
    this.events.on(window, 'beforeunload', () => this.dispose());
  }

  /**
   * Setup keyboard shortcut for opening dataset browser
   */
  private setupDatasetBrowserShortcut(): void {
    this.events.on(window, 'open-dataset-browser', () => {
      if (!this.datasetBrowser) {
        this.showDatasetBrowser();
      }
    });
  }

  /**
   * Setup window focus handling to trigger render on focus
   * This prevents stale renders when switching between windows/tabs
   */
  private setupFocusHandling(): void {
    this.events.on(window, 'focus', () => {
      // Suppress focus-triggered renders during recording — they can interfere
      // with the deterministic capture loop or cause resize side effects
      if (this.recordingPanel?.isCurrentlyRecording()) return;
      this.animationController.startAnimation();
      log.info(Modules.LUXAR, 'Window focused - triggering render refresh');
    });
    this.events.on(document, 'visibilitychange', () => {
      // Don't stop animation during recording (offline capture needs the loop alive)
      if (this.recordingPanel?.isCurrentlyRecording()) return;
      if (document.hidden) {
        this.animationController.stopAnimation();
        log.info(Modules.LUXAR, 'Document hidden - stopping animation to save resources');
      } else {
        this.animationController.startAnimation();
        log.info(Modules.LUXAR, 'Document became visible - resuming animation');
      }
    });
  }

  /**
   * Setup debug interface for testing and AI-assisted development
   *
   * This extends the existing debug interface (seeded by bootstrapStandalone()
   * before init() runs) with runtime components that are only available after
   * initialization:
   * - Three.js scene, camera, renderer
   * - Controls and animation state
   * - Helper functions for testing
   *
   * Preserves existing properties (app, consoleInterceptor, version) from
   * the bootstrap-side seeding.
   *
   * Only enabled when `LuxarAppOptions.debug` is set (the standalone bootstrap
   * derives that from the `?debug` URL param or persisted `luxar.debug` flag).
   */
  private setupDebugInterface(): void {
    if (!this.options.debug) {
      return;
    }

    log.info(Modules.LUXAR, 'Extending debug interface with runtime components');

    // Extend whatever bootstrap seeded (app/consoleInterceptor/version). When
    // LuxarApp is instantiated outside the standalone-app entry point
    // (tests, embeds), bootstrap hasn't run; fall back to a fresh base.
    const existing = window.__luxarDebug ?? {
      app: this,
      consoleInterceptor: consoleInterceptor,
      version: '1.0.0',
    };

    window.__luxarDebug = {
      // Preserve existing properties from main.ts
      ...existing,

      // Add runtime components (only available after initialization)
      scene: this.sceneManager.scene,
      camera: this.sceneManager.camera,
      renderer: this.sceneManager.renderer,
      controls: this.sceneManager.controls,
      postProcessing: this.sceneManager.postProcessing,
      animationController: this.animationController,
      inputHandler: this.inputHandler,
      renderingControls: this.renderingControls,
      recordingPanel: this.recordingPanel,
      sceneDimsManager: sceneDimsManager,
      app: this,

      // Helper function to get current state snapshot.
      // Implementation lives in `core/debug-state.ts` so the
      // scene-walking logic can be unit-tested directly.
      getState: () =>
        computeDebugState({
          scene: this.sceneManager.scene,
          camera: this.sceneManager.camera,
          currentFov: this.sceneManager.currentFov,
          isAnimating: this.animationController.isActive,
          initialized: this.isInitialized,
          dims: sceneDimsManager.getDims(),
        }),

      // Helper to trigger a single frame render (for stable screenshots)
      renderOnce: () => {
        this.animationController.startAnimation();
      },

      // Helper to get scene loader manager (for cache inspection)
      getSceneLoader: () => {
        return SceneLoaderManager.getInstance();
      },

      // Cache-specific helpers — thin wrappers over the SceneLoader cache
      // API. Implementation lives in `core/debug-cache-helpers.ts` so the
      // not-found / no-cache / success branches can be unit-tested
      // directly with a stub loader.
      cache: buildDebugCacheHelpers(() =>
        SceneLoaderManager.getInstance().getDefaultLoader()
      ),

      // Test-friendly hook for the error-dialog component. Lets
      // visual-regression specs render the dialog directly without going
      // through URL-routing failure paths (whose semantics evolve
      // independently of the dialog's appearance).
      showError,

      // Mark that runtime components are now available
      runtimeReady: true,
    };

    // Log available debug commands
    log.info(Modules.LUXAR, 'Debug interface ready:');
    log.info(Modules.LUXAR, '  __luxarDebug.getState() - Get current state snapshot');
    log.info(Modules.LUXAR, '  __luxarDebug.renderOnce() - Trigger single frame render');
    log.info(Modules.LUXAR, '  __luxarDebug.scene - Access THREE.js scene');
    log.info(Modules.LUXAR, '  __luxarDebug.camera - Access camera');
    log.info(Modules.LUXAR, '  __luxarDebug.app - Access LuxarApp instance');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.getStats() - Get cache statistics (L0, L1, L2)');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.clearL0() - Clear L0 decompressed chunk cache');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.clearL1() - Clear L1 memory cache');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.clearL2() - Clear L2 OPFS cache');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.clearAll() - Clear all caches (L0, L1, L2)');
  }

  /**
   * Get application components (for testing or advanced usage)
   */
  get components() {
    return {
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      inputHandler: this.inputHandler,
      renderingControls: this.renderingControls,
      adaptiveDPRManager: this.adaptiveDPRManager,
    };
  }

  /**
   * Get initialization state
   */
  get initialized(): boolean {
    return this.isInitialized;
  }

  /**
   * Dispose all application resources.
   *
   * Tears down the animation loop, scene, input handlers, UI panels, and
   * registered listeners. Idempotent: safe to call repeatedly. After
   * dispose(), the LuxarApp instance is in an uninitialized state — call
   * init() again to re-create resources, or discard the instance.
   */
  /**
   * Capture a JSON-serialisable snapshot of the current viewer state.
   *
   * Includes camera placement (position, target, up, projection params)
   * and per-dimension slice positions. Layer-panel state and rendering-
   * controls settings are not included in v1 — see
   * `src/core/viewer-snapshot.ts` for the rationale and the schema.
   *
   * Use the returned object to share a view, write a regression fixture,
   * or hand to {@link restoreSnapshot} on another LuxarApp instance.
   *
   * @throws if the app has not been initialised yet.
   */
  captureSnapshot(): ViewerSnapshot {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.captureSnapshot called before init()');
    }
    return captureViewerSnapshot(this.sceneManager);
  }

  /**
   * Restore viewer state from a snapshot produced by {@link captureSnapshot}.
   *
   * Returns which parts of the snapshot were applied. Camera always applies
   * if the version matches; dims apply only when the snapshot's `ndim`
   * matches the loaded dataset (otherwise skipped with a warning rather
   * than throwing — common for cross-dataset link sharing).
   *
   * @throws if the app has not been initialised yet.
   */
  restoreSnapshot(snapshot: ViewerSnapshot): { cameraApplied: boolean; dimsApplied: boolean } {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.restoreSnapshot called before init()');
    }
    return restoreViewerSnapshot(this.sceneManager, snapshot);
  }

  dispose(): void {
    // Idempotency: a second dispose() after a successful one is a no-op.
    // Component fields still reference their (already disposed) instances,
    // so without this guard we would call dispose() on disposed components.
    if (this.isDisposed) return;
    // Re-entrance guard: if a beforeunload (or any nested) call fires while
    // we are already tearing down, do nothing.
    if (this.isDisposing) return;
    this.isDisposing = true;

    // Flip initialized at entry so any concurrent observer of `app.initialized`
    // sees the correct state from the first instant of teardown, even if
    // teardown throws partway through.
    this.isInitialized = false;

    try {
      // Stop animation first
      if (this.animationController) {
        this.animationController.dispose();
      }

      // Clean up adaptive DPR manager
      if (this.adaptiveDPRManager) {
        this.adaptiveDPRManager.dispose();
      }

      // Clean up resolution indicator
      if (this.resolutionIndicator) {
        this.resolutionIndicator.dispose();
      }

      // Clean up scale bar
      if (this.scaleBar) {
        this.scaleBar.dispose();
        this.scaleBar = undefined;
      }

      // Clean up colormap legend
      if (this.colormapLegend) {
        this.colormapLegend.dispose();
        this.colormapLegend = undefined;
      }

      // Clean up overlay manager (before recording panel so we can clear the reference)
      if (this.overlayManager) {
        this.recordingPanel?.setOverlayManager(null);
        this.overlayManager.dispose();
        this.overlayManager = undefined;
      }

      // Clean up recording panel
      if (this.recordingPanel) {
        this.recordingPanel.dispose();
        this.recordingPanel = undefined;
      }

      // Clean up layers panel
      if (this.layersPanel) {
        this.layersPanel.dispose();
        this.layersPanel = undefined;
      }

      // Clean up picking system listeners (DOM mousemove, controls/scene
      // event subscriptions). pickingEvents is reusable: dispose() leaves it
      // in an empty state ready for the next initPicking() call.
      this.pickingEvents.dispose();
      if (this.pickingSystem) {
        this.pickingSystem.dispose();
        this.pickingSystem = undefined;
      }
      if (this.labelLoader) {
        this.labelLoader.dispose();
        this.labelLoader = undefined;
      }
      if (this.imageLabelLoader) {
        this.imageLabelLoader.dispose();
        this.imageLabelLoader = undefined;
      }

      // Clean up input handlers
      if (this.inputHandler) {
        this.inputHandler.dispose();
      }

      // Clean up rendering controls
      if (this.renderingControls) {
        this.renderingControls.dispose();
      }

      // Clean up scene resources
      if (this.sceneManager) {
        this.sceneManager.dispose();
      }

      // Tear down the theme manager (disconnects glass-refraction MutationObserver,
      // removes injected SVG filters, clears CSS custom properties).
      ThemeManager.disposeInstance();

      // Clean up UI resources
      cleanupUI();

      // Tear down all app-level event listeners (focus, visibility, beforeunload,
      // open-dataset-browser, and any picking-system subscriptions added later
      // via this.events.add()) in one call.
      this.events.dispose();

      // Dispose any singletons that registered themselves with the
      // ManagerRegistry. Walks them in reverse registration order so
      // the most-recently-created tears down first. Idempotent on
      // repeat calls; non-disposed managers fall through silently.
      getManagerRegistry().disposeAll();
    } catch (error) {
      log.error(Modules.LUXAR, 'Error during dispose:', error);
    } finally {
      this.isDisposing = false;
      this.isDisposed = true;
    }
  }

  /**
   * Get visibility states of all UI panels for save/restore during recording.
   */
  private getPanelVisibilityStates(): Map<string, boolean> {
    const states = new Map<string, boolean>();
    states.set('renderingControls', this.renderingControls?.isVisible() ?? false);
    states.set('recordingPanel', this.recordingPanel?.isVisible() ?? false);
    return states;
  }

  /**
   * Restore UI panel visibility from a saved state map.
   */
  private restorePanelVisibilityStates(states: Map<string, boolean>): void {
    if (states.get('renderingControls')) {
      this.renderingControls?.show();
    } else {
      if (this.renderingControls?.isVisible()) this.renderingControls.hide();
    }
    if (states.get('recordingPanel')) {
      this.recordingPanel?.show();
    } else {
      if (this.recordingPanel?.isVisible()) this.recordingPanel.hide();
    }
  }
}
