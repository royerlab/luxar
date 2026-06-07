// Main application class for the Luxar scene player

import type { SceneManager } from '../scene/scene-manager';
import {
  captureSnapshot as captureViewerSnapshot,
  restoreSnapshot as restoreViewerSnapshot,
  type ViewerSnapshot,
} from './app/snapshot/viewer-snapshot';
import type { AnimationController } from '../scene/animation/animation-controller';
import type { InputHandler } from '../input/input-handler';
import type { RenderingControls } from '../ui/rendering-controls';
import { showHelpOverlay } from '../ui/help-overlay';
import type { DatasetBrowser } from '../ui/dataset-browser';
import { log, Modules } from '../utils/log';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import type { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';
import type { ResolutionIndicator } from '../ui/resolution-indicator';
import type { PerformanceMonitor } from '../ui/performance-monitor';
import type { ScaleBar } from '../ui/scale-bar';
import type { ColormapLegend } from '../ui/colormap-legend';
import type { RecordingPanel } from '../ui/recording-panel';
import type { LayersPanel } from '../ui/layers';
import { ThemeManager } from '../themes/theme-manager';
import type { ZarrViewerConfig } from '../types/zarr';
import type { OverlayManager } from '../ui/overlay-manager';
import type { PickingSystem } from '../rendering/picking/picking-system';
import type { LabelLoader, ImageLabelLoader } from '../data/loaders';
import { EventGroup } from '../utils/cross-layer/event-group';
import { setViewerContainer } from '../utils/viewer-container';
import { assertBrowserEnvironment, assertThreeRevision } from './app/init/environment-guards';
import { applyModuleOverrides } from './app/init/module-overrides';
import { runInitPipeline, type InitPipelineResult } from './app/init/pipeline';
import { runDisposePipeline } from './app/lifecycle/dispose-pipeline';
import { shouldShowBrowser as shouldShowBrowserImpl } from './app/dataset/should-show-browser';
import { showDatasetBrowser as showDatasetBrowserImpl } from './app/dataset/show-browser';
import { loadDataset as loadDatasetImpl } from './app/dataset/load-dataset';
import { installDebugInterface } from './app/debug/debug-interface';
import { initPicking as initPickingImpl } from './app/picking/init-picking';
import { applyViewerConfigState as applyViewerConfigStateHelper } from './app/viewer-config/apply-state';
import {
  getPanelVisibilityStates as getPanelVisibilityStatesHelper,
  restorePanelVisibilityStates as restorePanelVisibilityStatesHelper,
} from './app/viewer-config/panel-visibility';
import { installUnloadHandler } from './app/lifecycle/unload-handling';
import { installBrowserShortcut } from './app/dataset/browser-shortcut';
import { openCacheStatsView as openCacheStatsViewImpl } from './app/debug/cache-stats-view';
import { disposeOverlays as disposeOverlaysImpl } from './app/overlays/dispose-overlays';
import { initColormapLegend as initColormapLegendImpl } from './app/overlays/init-colormap-legend';
import { initOverlays as initOverlaysImpl } from './app/overlays/init-overlays';
import { installFocusHandling } from './app/lifecycle/focus-handling';
import { initScaleBar as initScaleBarImpl } from './app/overlays/init-scale-bar';

import type { LuxarAppOptions } from './app/options';
export type { LuxarAppOptions } from './app/options';

export class LuxarApp {
  private sceneManager!: SceneManager;
  private animationController!: AnimationController;
  private performanceMonitor!: PerformanceMonitor;
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

    assertBrowserEnvironment();
    assertThreeRevision();

    // Reset the idempotency guard so a fresh init followed by dispose
    // works even if the same instance was already initialized and
    // disposed.
    this.isDisposed = false;
    this.options = options;

    applyModuleOverrides(options);

    // Point all viewer overlays/panels at the host-provided container (or
    // document.body by default) before any subsystem mounts DOM. A custom
    // container is also promoted to a containing block so fixed overlays
    // scope to it; resetViewerContainer() in the dispose pipeline restores it.
    setViewerContainer(options.container ?? document.body);

    // Mutable accumulator: pipeline writes each subsystem here as it
    // constructs it, so even if init() throws partway through, the
    // already-constructed pieces are visible to dispose().
    const partial: Partial<InitPipelineResult> = {};
    const assignFromPartial = (): void => {
      if (partial.sceneManager) this.sceneManager = partial.sceneManager;
      if (partial.animationController) this.animationController = partial.animationController;
      if (partial.performanceMonitor) this.performanceMonitor = partial.performanceMonitor;
      if (partial.adaptiveDPRManager) this.adaptiveDPRManager = partial.adaptiveDPRManager;
      if (partial.resolutionIndicator) this.resolutionIndicator = partial.resolutionIndicator;
      if (partial.inputHandler) this.inputHandler = partial.inputHandler;
      if (partial.renderingControls) this.renderingControls = partial.renderingControls;
      if (partial.recordingPanel) this.recordingPanel = partial.recordingPanel;
      if (partial.layersPanel) this.layersPanel = partial.layersPanel;
    };

    try {
      const result = await runInitPipeline(
        {
          options: this.options,
          events: this.events,
          getPanelVisibilityStates: () => this.getPanelVisibilityStates(),
          restorePanelVisibilityStates: (states) => this.restorePanelVisibilityStates(states),
        },
        partial
      );

      assignFromPartial();

      // Dataset routing: subsystems are wired up, fields are assigned —
      // the orchestrator delegates can now safely read `this.*`.
      if (await this.shouldShowBrowser(result.sceneSrc)) {
        try {
          this.showDatasetBrowser();
        } catch (error) {
          log.warning(
            Modules.APP,
            'Dataset browser initialization had issues, but browser is shown:',
            error
          );
        }
      } else {
        await this.loadDataset(result.sceneSrc);
      }

      this.setupDisposeOnUnload();
      this.setupDatasetBrowserShortcut();
      this.setupFocusHandling();
      this.setupDebugInterface();

      this.isInitialized = true;
    } catch (error) {
      log.error(Modules.APP, 'Failed to initialize Luxar app:', error);
      // Tear down whatever partial state was constructed before the throw.
      // The pipeline writes each subsystem into `partial` as it builds
      // it, so a mid-init failure still surfaces every disposable on
      // this.* before we call dispose() (which uses per-field guards).
      // The caller's error handler is expected to surface a fresh,
      // top-level error UI; any in-progress error UI from sub-loaders
      // is wiped along with everything else.
      assignFromPartial();
      this.dispose();
      throw error;
    }
  }

  /**
   * Check if we should show the dataset browser
   */
  private async shouldShowBrowser(src: string): Promise<boolean> {
    return shouldShowBrowserImpl(src);
  }

  /**
   * Show the dataset browser UI
   */
  private showDatasetBrowser(): void {
    if (this.datasetBrowser) return; // Browser already open
    this.datasetBrowser = showDatasetBrowserImpl({
      currentSrc: this.options.src,
      updateBrowserUrl: this.options.updateBrowserUrl === true,
      inputHandler: this.inputHandler,
      onSrcChange: (src) => {
        this.options = { ...this.options, src };
      },
      loadDataset: (src) => this.loadDataset(src),
      onClose: () => {
        this.datasetBrowser = undefined;
      },
    });
  }

  /**
   * Load a dataset and initialize UI
   */
  private async loadDataset(src: string): Promise<void> {
    await loadDatasetImpl(src, {
      inputHandler: this.inputHandler,
      renderingControls: this.renderingControls,
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      layersPanel: this.layersPanel,
      loaderConfig: this.options.loaderConfig,
      openCacheStats: !!this.options.openCacheStats,
      disposeOverlays: () => this.disposeOverlays(),
      initScaleBar: () => this.initScaleBar(),
      initColormapLegend: () => this.initColormapLegend(),
      initOverlays: () => this.initOverlays(),
      initPicking: () => this.initPicking(),
      applyViewerConfigState: (config) => this.applyViewerConfigState(config),
      openCacheStatsView: () => this.openCacheStatsView(),
    });
  }

  /**
   * Open the data-loading monitor in expanded mode on the Cache tab.
   * Best-effort: silently skips when no monitor was created (e.g.
   * embedded contexts that disable the monitor).
   */
  private openCacheStatsView(): void {
    openCacheStatsViewImpl();
  }

  /**
   * Apply zarr viewer_config state that isn't handled by RenderingControls.
   *
   * RenderingControls handles rendering settings (bloom, AA, tone mapping, etc.).
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
    this.scaleBar = initScaleBarImpl({
      previous: this.scaleBar,
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      inputHandler: this.inputHandler,
    });
  }

  /**
   * Initialize the colormap legend overlay.
   * Shows per-layer colormap gradients with names and data ranges.
   */
  private initColormapLegend(): void {
    this.colormapLegend = initColormapLegendImpl({
      previous: this.colormapLegend,
      layersPanel: this.layersPanel,
      inputHandler: this.inputHandler,
    });
  }

  /**
   * Tear down the current OverlayManager, removing its DOM elements.
   */
  private disposeOverlays(): void {
    disposeOverlaysImpl({
      manager: this.overlayManager,
      onDisposed: () => {
        this.overlayManager = undefined;
      },
    });
  }

  /**
   * Initialize screen-space overlays from zarr metadata.
   *
   * Always constructs an `OverlayManager` (even when the scene declares
   * no overlays) so the rest of the app — input handler, recording panel,
   * `__luxarDebug.getOverlayManager()` probe — sees a stable, non-null
   * collaborator. The manager just stays empty until `loadOverlays` (or a
   * runtime caller, e.g. a test) populates it.
   */
  private async initOverlays(): Promise<void> {
    this.overlayManager = await initOverlaysImpl({
      disposePrevious: () => this.disposeOverlays(),
      sceneManager: this.sceneManager,
      inputHandler: this.inputHandler,
      recordingPanel: this.recordingPanel,
    });
  }

  /**
   * Initialize GPU picking system for hover tooltips.
   * Only activates if any scene node has labels or image labels
   * (has_labels / has_image_labels in .zattrs).
   * Wires up: PickingSystem → LabelLoader/ImageLabelLoader → OverlayManager.updateHoverContent.
   */
  private async initPicking(): Promise<void> {
    const result = await initPickingImpl({
      sceneManager: this.sceneManager,
      pickingEvents: this.pickingEvents,
      previous: {
        pickingSystem: this.pickingSystem,
        labelLoader: this.labelLoader,
        imageLabelLoader: this.imageLabelLoader,
      },
      getOverlayManager: () => this.overlayManager,
    });
    this.pickingSystem = result.pickingSystem;
    this.labelLoader = result.labelLoader;
    this.imageLabelLoader = result.imageLabelLoader;
  }

  /**
   * Register a beforeunload handler that disposes the app on page unload.
   */
  private setupDisposeOnUnload(): void {
    installUnloadHandler({ events: this.events, dispose: () => this.dispose() });
  }

  /**
   * Setup keyboard shortcut for opening dataset browser
   */
  private setupDatasetBrowserShortcut(): void {
    installBrowserShortcut({
      events: this.events,
      hasOpenBrowser: () => !!this.datasetBrowser,
      showBrowser: () => this.showDatasetBrowser(),
    });
  }

  /**
   * Setup window focus handling to trigger render on focus
   * This prevents stale renders when switching between windows/tabs
   */
  private setupFocusHandling(): void {
    installFocusHandling({
      events: this.events,
      animationController: this.animationController,
      getRecordingPanel: () => this.recordingPanel,
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
    installDebugInterface({
      debug: !!this.options.debug,
      app: this,
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      inputHandler: this.inputHandler,
      renderingControls: this.renderingControls,
      recordingPanel: this.recordingPanel,
      getPickingSystem: () => this.pickingSystem,
      getOverlayManager: () => this.overlayManager,
      isInitialized: () => this.isInitialized,
    });
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
   * Capture a JSON-serialisable snapshot of the current viewer state.
   *
   * Includes camera placement (position, target, up, projection params)
   * and per-dimension slice positions. Layer-panel state and rendering-
   * controls settings are not included in v1 — see
   * `src/core/app/snapshot/viewer-snapshot.ts` for the rationale and the schema.
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

  /**
   * Dispose all application resources.
   *
   * Tears down the animation loop, scene, input handlers, UI panels, and
   * registered listeners. Idempotent: safe to call repeatedly. After
   * dispose(), the LuxarApp instance is in an uninitialized state — call
   * init() again to re-create resources, or discard the instance.
   */
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

    runDisposePipeline({
      events: this.events,
      pickingEvents: this.pickingEvents,
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      adaptiveDPRManager: this.adaptiveDPRManager,
      resolutionIndicator: this.resolutionIndicator,
      inputHandler: this.inputHandler,
      renderingControls: this.renderingControls,
      recordingPanel: this.recordingPanel,
      layersPanel: this.layersPanel,
      scaleBar: this.scaleBar,
      colormapLegend: this.colormapLegend,
      overlayManager: this.overlayManager,
      pickingSystem: this.pickingSystem,
      labelLoader: this.labelLoader,
      imageLabelLoader: this.imageLabelLoader,
      datasetBrowser: this.datasetBrowser,
      clearScaleBar: () => {
        this.scaleBar = undefined;
      },
      clearColormapLegend: () => {
        this.colormapLegend = undefined;
      },
      clearOverlayManager: () => {
        this.overlayManager = undefined;
      },
      clearRecordingPanel: () => {
        this.recordingPanel = undefined;
      },
      clearLayersPanel: () => {
        this.layersPanel = undefined;
      },
      clearPickingSystem: () => {
        this.pickingSystem = undefined;
      },
      clearLabelLoader: () => {
        this.labelLoader = undefined;
      },
      clearImageLabelLoader: () => {
        this.imageLabelLoader = undefined;
      },
      clearDatasetBrowser: () => {
        this.datasetBrowser = undefined;
      },
    });

    this.isDisposing = false;
    this.isDisposed = true;
  }

  /**
   * Get visibility states of all UI panels for save/restore during recording.
   * Implementation lives in `core/app/viewer-config/panel-visibility.ts`.
   */
  private getPanelVisibilityStates(): Map<string, boolean> {
    return getPanelVisibilityStatesHelper({
      renderingControls: this.renderingControls,
      recordingPanel: this.recordingPanel,
    });
  }

  /**
   * Restore UI panel visibility from a saved state map.
   */
  private restorePanelVisibilityStates(states: Map<string, boolean>): void {
    restorePanelVisibilityStatesHelper(states, {
      renderingControls: this.renderingControls,
      recordingPanel: this.recordingPanel,
    });
  }
}
