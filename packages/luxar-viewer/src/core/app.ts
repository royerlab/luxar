// Main application class for the Luxar scene player

import type { SceneManager } from '../scene/scene-manager';
import {
  captureSnapshot as captureViewerSnapshot,
  restoreSnapshot as restoreViewerSnapshot,
  restoreCamera,
  type ViewerSnapshot,
  type CameraSnapshot,
} from './app/snapshot/viewer-snapshot';
import { createEventBus, type Unsubscribe } from '../utils/cross-layer/event-bus';
import type {
  LuxarEmbedderEventMap,
  EmbedderDimensions,
  ScreenshotOptions,
  DatasetFaultPayload,
  LayerPatch,
  LayerSummary,
  ViewerState,
} from './app/embedder/events';
import { CameraFlight, type FlyToOptions, type FlightResult } from './app/camera/camera-flight';
import { WaypointDriver, resolveWaypointPose } from './app/camera/waypoint-driver';
import { ControlClient } from './app/control/control-client';
import { extractRenderingOverrides } from '../config/zarr-bridge/viewer-config-utils';
import { extractAudioConfig } from '../config/zarr-bridge/audio-config';
import type { AudioEngine } from '../audio/audio-engine';
import type { AudioPatch, AudioState } from '../types/audio';
import { resolveTargetNodeCenter } from '../scene/scene-manager/camera/camera-setup';
import { config, type RenderingSettings } from '../config';
import type * as THREE from 'three';
import type { ZarrWaypoint } from '../types/zarr';
import { captureScreenshot } from './app/embedder/screenshot';
import type { AnimationController } from '../scene/animation/animation-controller';
import type { ContextConfig, InputContextId, InputHandler, KeyBinding } from '../input';
import type { RenderingControls } from '../ui/rendering-controls';
import { showHelpOverlay } from '../ui/help-overlay';
import type { DatasetBrowser } from '../ui/dataset-browser';
import { log, Modules } from '../utils/log';
import { getErrorMessage } from '../utils/format-error';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import type { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';
import type { ResolutionIndicator } from '../ui/resolution-indicator';
import type { PerformanceMonitor } from '../ui/performance-monitor';
import type { ScaleBar } from '../ui/scale-bar';
import type { ColormapLegend } from '../ui/colormap-legend';
import type { RecordingPanel } from '../ui/recording-panel';
import type { LayersPanel } from '../ui/layers';
import type { ControlRail } from '../ui/control-rail';
import { ThemeManager } from '../themes/theme-manager';
import type { ZarrViewerConfig } from '../types/zarr';
import type { OverlayManager } from '../ui/overlay-manager';
import type { PickingSystem } from '../rendering/picking/picking-system';
import type { LabelLoader, ImageLabelLoader } from '../data/loaders';
import { EventGroup } from '../utils/cross-layer/event-group';
import { installCanvasGestureOwnership } from './app/interaction/canvas-gesture-ownership';
import { installContextMenuOwnership } from './app/interaction/context-menu-ownership';
import { setViewerContainer } from '../utils/viewer-container';
import { assertBrowserEnvironment, assertThreeRevision } from './app/init/environment-guards';
import { applyModuleOverrides } from './app/init/module-overrides';
import { runInitPipeline, type InitPipelineResult } from './app/init/pipeline';
import { runDisposePipeline } from './app/lifecycle/dispose-pipeline';
import { shouldShowBrowser as shouldShowBrowserImpl } from './app/dataset/should-show-browser';
import { showDatasetBrowser as showDatasetBrowserImpl } from './app/dataset/show-browser';
import { loadDataset as loadDatasetImpl } from './app/dataset/load-dataset';
import { dataSourceDocumentTitle, setDocumentTitle } from './document-title';
import { installDebugInterface } from './app/debug/debug-interface';
import {
  initPicking as initPickingImpl,
  disposePickingSession as disposePickingSessionImpl,
} from './app/picking/init-picking';
import { installDoubleTapToFit } from './app/interaction/double-tap-to-fit';
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
import { installOnlineRetry } from './app/lifecycle/online-retry';
import { getSceneLoader, SceneLoaderManager } from '../data/scene-loader-manager';
import type { SceneLoader } from '../data/scene-loader';
import { notifier } from '../utils/cross-layer/notifier';
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
  private controlRail?: ControlRail;
  /** The sound layer; created by the init pipeline, nodes attach per scene. */
  private audioEngine?: AudioEngine;
  private overlayManager?: OverlayManager;
  private pickingSystem?: PickingSystem;
  private labelLoader?: LabelLoader;
  private imageLabelLoader?: ImageLabelLoader;
  /** Reads the per-element `keys` CSR — a LabelLoader on the 'keys' channel (#1917). */
  private keyLoader?: LabelLoader;
  /**
   * Per-init EventGroup for picking-system listeners (canvas mousemove,
   * window resize, controls/scene-manager subscriptions). Re-disposed and
   * rebuilt on each scene load.
   */
  private pickingEvents = new EventGroup();
  private isInitialized = false;
  /** True from the start of init() until routing and app-level wiring complete. */
  private isInitializing = false;
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
   * Per-app emitter for the public embedder events ({@link on}). Deliberately
   * a per-instance bus (not the global `eventBus` singleton) so the embedder
   * surface stays decoupled from the internal frame/UI plumbing and is
   * multi-instance-ready.
   */
  private embedderEvents = createEventBus<LuxarEmbedderEventMap>();
  private datasetFaultUnsubscribe?: Unsubscribe;
  private datasetFaultLoader?: SceneLoader;
  private currentDatasetSrc?: string;
  /** Smooth camera transitions for {@link flyTo}; created in {@link setupEmbedderHooks}. */
  private cameraFlight?: CameraFlight;
  /**
   * Dims-manager listener driving the loaded scene's authored story waypoints
   * (`viewer_config.waypoints`); the driver itself lives in its closure.
   */
  private waypointListener?: () => void;
  private waypointDriver?: WaypointDriver;
  /** Remote-control channel, present only when `options.control` is set. */
  private controlClient?: ControlClient;

  /**
   * Observes the canvas box so the viewer re-fits when the host container
   * resizes (not just the window). Disconnected on dispose via {@link events}.
   */
  private resizeObserver?: ResizeObserver;

  /**
   * In-flight guard for every post-init dataset switch, whether requested by
   * the public {@link switchDataset} API or the built-in dataset browser.
   * `loadDataset` does a full teardown+reload, so overlapping switches would
   * corrupt scene state.
   */
  private switchInFlight?: Promise<void>;

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

    // Claim the input surface before any async initialization or dataset
    // loading leaves an embedder-supplied canvas browser-owned.
    if (options.canvas instanceof HTMLElement) {
      installCanvasGestureOwnership(options.canvas, this.events);
    }

    // The same claim for the secondary click, across every surface the viewer
    // mounts — WebKit resolves the context-menu target to the overlay above
    // the canvas, not to the canvas the controls listen on.
    installContextMenuOwnership(
      options.canvas instanceof HTMLCanvasElement ? options.canvas : null,
      this.events
    );

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
      if (partial.controlRail) this.controlRail = partial.controlRail;
      if (partial.audioEngine) this.audioEngine = partial.audioEngine;
    };

    this.isInitializing = true;
    try {
      const result = await runInitPipeline(
        {
          options: this.options,
          events: this.events,
          getPanelVisibilityStates: () => this.getPanelVisibilityStates(),
          restorePanelVisibilityStates: (states) => this.restorePanelVisibilityStates(states),
          emitEmbedderEvent: (event, payload) => this.embedderEvents.emit(event, payload),
        },
        partial
      );

      assignFromPartial();

      // The layers panel edits a mesh's pick coverage (opacity/cutoff/blending)
      // with a stationary camera, which nothing else invalidates — give it a
      // late-bound hook to the current picking system so those edits refresh the
      // cached pick buffer. `this.pickingSystem` is (re)assigned per dataset load,
      // so read it lazily rather than capturing.
      this.layersPanel?.setPickBufferInvalidator(() => this.pickingSystem?.markDirty());

      // Dataset routing: subsystems are wired up, fields are assigned —
      // the orchestrator delegates can now safely read `this.*`.
      // Install this before routing so O / the rail control can open the
      // browser while a slow initial dataset load is still in progress.
      this.setupDatasetBrowserShortcut();
      // Install before the initial load so its first recorded transient
      // failure can arm the bounded retry backoff immediately.
      this.setupOnlineRetry();
      // The control client eagerly attaches the selection / element event
      // consumers that picking checks once, during dataset provisioning.
      this.installControlClient();
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
      this.setupFocusHandling();
      this.setupDebugInterface();
      this.setupEmbedderHooks(options.canvas);
      // Touch double-tap re-frames on EVERY scene — not only the ones the
      // picking session (and with it canvas-actions) gets provisioned for.
      // (Unit tests hand `init` a stub canvas with no event surface.)
      if (options.canvas instanceof HTMLElement) {
        installDoubleTapToFit(options.canvas, this.events, () => this.recenterCamera());
      }

      this.isInitialized = true;
    } catch (error) {
      log.error(Modules.APP, `Failed to initialize Luxar app: ${getErrorMessage(error)}`, error);
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
    } finally {
      this.isInitializing = false;
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
    // Do not reopen the browser while a selected dataset is still switching.
    // The modal closes immediately after an accepted selection, so without this guard
    // the O shortcut / dataset rail item could open a second modal and offer
    // another overlapping full teardown+reload.
    if (this.datasetBrowser || this.switchInFlight) return;
    this.datasetBrowser = showDatasetBrowserImpl({
      currentSrc: this.options.src,
      updateBrowserUrl: this.options.updateBrowserUrl === true,
      inputHandler: this.inputHandler,
      onSrcChange: (src) => {
        this.options = { ...this.options, src };
      },
      // Lets the selection handler refuse startup-time choices before any
      // side effects or guarded switch dispatch.
      isInitializing: () => this.isInitializing,
      isSwitchInFlight: () => this.switchInFlight !== undefined,
      // Browser selections must share the same in-flight guard as the public
      // embedder API. Calling loadDataset() directly here used to allow two
      // full teardown+reload passes to interleave.
      loadDataset: (src) => this.switchDataset(src),
      shortcutForAction: (actionId) => this.shortcutForAction(actionId),
      onClose: () => {
        this.datasetBrowser = undefined;
      },
    });
  }

  /**
   * Load a dataset and initialize UI
   */
  private async loadDataset(src: string): Promise<void> {
    this.datasetFaultUnsubscribe?.();
    this.datasetFaultUnsubscribe = undefined;
    this.datasetFaultLoader = undefined;
    this.currentDatasetSrc = undefined;
    // The outgoing scene's waypoints must not fire on the incoming scene's dims.
    this.disposeWaypoints();
    // ...and its sounds must stop before the new store loads.
    this.audioEngine?.detachScene();
    try {
      await loadDatasetImpl(src, {
        inputHandler: this.inputHandler,
        renderingControls: this.renderingControls,
        sceneManager: this.sceneManager,
        animationController: this.animationController,
        layersPanel: this.layersPanel,
        loaderConfig: this.options.loaderConfig,
        openCacheStats: !!this.options.openCacheStats,
        disposeOverlays: () => this.disposeOverlays(),
        disposePicking: () => this.disposePicking(),
        initScaleBar: () => this.initScaleBar(),
        initColormapLegend: () => this.initColormapLegend(),
        initOverlays: () => this.initOverlays(),
        initPicking: () => this.initPicking(),
        applyViewerConfigState: (config) => this.applyViewerConfigState(config),
        openCacheStatsView: () => this.openCacheStatsView(),
      });
      this.currentDatasetSrc = src;
      const sceneLoader = getSceneLoader();
      this.datasetFaultLoader = sceneLoader ?? undefined;
      // A replayed latched fault is post-load state, so preserve the public
      // ordering: the dataset becomes available before its fault is reported.
      this.embedderEvents.emit('dataset-loaded', { src });
      if (sceneLoader) {
        this.datasetFaultUnsubscribe = sceneLoader.onArchiveFault(
          (error) => this.embedderEvents.emit('dataset-fault', { src, error }),
          { replayCurrent: true }
        );
      }
    } catch (error) {
      this.embedderEvents.emit('dataset-error', {
        src,
        error: error instanceof Error ? error : new Error(String(error)),
      });
      throw error;
    }
  }

  /**
   * Wire the programmatic-embedder hooks: re-emit dimension changes as the
   * public `dimensions-changed` event, and auto-resize to the canvas box via
   * a ResizeObserver. Both are guarded by `isInitialized` so they no-op
   * outside the live window, and both are torn down through {@link events}.
   */
  private setupEmbedderHooks(canvas: HTMLCanvasElement): void {
    const dimsListener = (): void => {
      if (this.isInitialized) {
        this.embedderEvents.emit('dimensions-changed', this.getDimensions());
      }
    };
    sceneDimsManager.addListener(dimsListener);
    this.events.add(() => sceneDimsManager.removeListener(dimsListener));

    // Camera pose feedback: the ControlsManager is created once per app and
    // forwards every active control's `change`, so one listener covers
    // orbit/fly/ortho input, setCameraPose(), flight frames and auto-rotate.
    const controls = this.sceneManager.controls;
    const cameraListener = (): void => {
      if (this.isInitialized) {
        this.embedderEvents.emit('camera-changed', captureViewerSnapshot(this.sceneManager).camera);
      }
    };
    controls.addEventListener('change', cameraListener);
    this.events.add(() => controls.removeEventListener('change', cameraListener));

    // flyTo() driver. The canvas is the input surface whose pointer / wheel /
    // touch events hand control back to the user mid-flight.
    this.cameraFlight = new CameraFlight({
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      inputElement: canvas instanceof HTMLElement ? canvas : null,
    });
    this.events.add(() => {
      this.cameraFlight?.dispose();
      this.cameraFlight = undefined;
    });
    this.events.add(() => this.disposeWaypoints());

    // Guard on a real Element: ResizeObserver may be absent (some test
    // environments) and observing a non-Element throws.
    if (typeof ResizeObserver !== 'undefined' && canvas instanceof Element) {
      // ResizeObserver callbacks are browser-batched (~once per frame), so a
      // direct resizeToCanvas() (synchronous resizeNow) needs no extra debounce.
      this.resizeObserver = new ResizeObserver(() => {
        if (this.isInitialized) this.sceneManager.resizeToCanvas();
      });
      // Observe the canvas's PARENT, not the canvas: Three stamps inline px
      // sizes on the canvas on every setSize, so the canvas's own box stops
      // tracking host layout changes — the parent (the embedder's frame) is
      // the box that actually resizes.
      this.resizeObserver.observe(canvas.parentElement ?? canvas);
      this.events.add(() => {
        this.resizeObserver?.disconnect();
        this.resizeObserver = undefined;
      });
    }
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
    this.applyViewerConfigStateCore(viewerConfig);
    // AFTER the core pass: `dimensions.current_step` has been applied, so the
    // load-time match sees the authored opening position and snaps to it.
    this.installWaypoints(viewerConfig?.waypoints);
    // AFTER the waypoints: the opening slice is final, so the first slab
    // evaluation starts exactly the sounds the opening story owns.
    this.installAudio(viewerConfig?.audio);
  }

  /**
   * Bind the sound layer to the loaded scene: apply the authored
   * `viewer_config.audio` defaults (the listener's persisted mute / master gain
   * win), then hand the engine the scene root so it finds the sound-node
   * placeholders `loadSoundNode` attached and starts decoding their clips.
   */
  private installAudio(audio: unknown): void {
    if (!this.audioEngine) return;
    this.audioEngine.detachScene();
    this.audioEngine.applySceneConfig(extractAudioConfig(audio));
    const root = this.sceneManager.scene?.children?.find((c) => c.name === 'LuxarScene');
    if (root) {
      this.audioEngine.attachScene(root);
      this.layersPanel?.pushAudioMutes();
    }
    // The waypoints install first and snap to the opening waypoint before any
    // sound node exists, so the load-time arrival is replayed here — otherwise
    // the opening story's `on_arrive` narration would never fire.
    const driver = this.waypointDriver;
    const opening = driver ? driver.getWaypoint(driver.currentIndex) : undefined;
    if (opening?.when) this.audioEngine.notifyWaypoint('arrive', opening.when);
  }

  /**
   * Bind the scene's authored story waypoints to the dims manager: match once
   * now (snap — the opening framing, ahead of the plain `camera` block), then
   * fly whenever a dimension change makes a different waypoint match. Each
   * port is a piece the app already owns; the driver only sequences them.
   */
  private installWaypoints(waypoints: ZarrWaypoint[] | undefined): void {
    this.disposeWaypoints();
    if (!Array.isArray(waypoints) || waypoints.length === 0) {
      this.overlayManager?.updateVisibility();
      return;
    }

    const driver = new WaypointDriver(waypoints, {
      getDims: () => sceneDimsManager.getDims(),
      getLivePose: () => captureViewerSnapshot(this.sceneManager).camera,
      resolvePose: (camera, live) =>
        resolveWaypointPose(camera, live, {
          resolveNodeCenter: (name) => {
            const root = this.sceneManager.scene?.children?.find((c) => c.name === 'LuxarScene') as
              THREE.Group | undefined;
            return root ? resolveTargetNodeCenter(root, name) : null;
          },
          fovPresets: config.camera.fovPresets,
        }),
      snapTo: (pose) => restoreCamera(this.sceneManager, pose),
      autoRotateActive: () => this.sceneManager.controls.isAutoRotateActive(),
      flyTo: (pose, opts) => {
        // The first load's config pass runs before setupEmbedderHooks(), so the
        // flight driver may not exist yet; a snap is the faithful fallback.
        if (this.cameraFlight) return this.cameraFlight.flyTo(pose, opts);
        restoreCamera(this.sceneManager, pose);
        return Promise.resolve({ completed: true });
      },
      // Snake_case keys → the same validated override path authored defaults take.
      applyRendering: (rendering) =>
        this.renderingControls.applyOverrides(
          extractRenderingOverrides(rendering as ZarrViewerConfig),
          'Applied waypoint rendering overrides'
        ),
      // The two story events: onto the embedder bus for controllers, and to the
      // sound layer for its `on_depart` / `on_arrive` nodes.
      emit: (event, payload) => {
        if (event === 'waypoint-departed') {
          this.embedderEvents.emit(event, { index: payload.index });
        } else {
          this.embedderEvents.emit(event, {
            index: payload.index,
            completed: 'completed' in payload ? payload.completed : true,
          });
          // The flight resolved and the driver's gate is open: reveal the
          // overlays a `reveal: "on_arrival"` waypoint held back.
          this.overlayManager?.updateVisibility();
        }
        const when = waypoints[payload.index]?.when;
        if (when && this.audioEngine) {
          this.audioEngine.notifyWaypoint(
            event === 'waypoint-departed' ? 'depart' : 'arrive',
            when
          );
        }
      },
    });
    const listener = (): void => {
      driver.evaluate('fly');
      // The overlay manager listens to the same dims manager and may have run
      // first with the previous gate state. Re-run its pass now whether the
      // new match closes OR opens the gate — same task, so nothing paints in
      // between.
      this.overlayManager?.updateVisibility();
    };
    sceneDimsManager.addListener(listener);
    this.waypointListener = listener;
    this.waypointDriver = driver;
    driver.evaluate('snap');
    this.overlayManager?.updateVisibility();
  }

  private disposeWaypoints(): void {
    if (this.waypointListener) {
      sceneDimsManager.removeListener(this.waypointListener);
      this.waypointListener = undefined;
    }
    this.waypointDriver = undefined;
  }

  /**
   * Attach the remote-control channel, when one was asked for.
   *
   * Built before initial dataset routing because its eager `selection` and
   * element listeners must exist when picking is provisioned. `this.events`
   * owns the teardown (see `CONTROL_FORWARDED_EVENTS`).
   *
   * `invoke` indexes the app by method name, which is safe precisely because
   * `isControlMethodAllowed` has already vetted the name against a list the
   * lock test keeps exhaustive.
   */
  private installControlClient(): void {
    const socketUrl = this.options.control;
    if (socketUrl === undefined || socketUrl === null || socketUrl.length === 0) return;
    const callable = this as unknown as Record<string, (...args: unknown[]) => unknown>;
    // `on` is generic over the event map, so its payload type is narrowed per
    // event name. The client subscribes by dynamic name and sanitizes whatever
    // arrives, so it wants the un-narrowed shape.
    const subscribeByName = this.on.bind(this) as unknown as (
      event: string,
      listener: (payload: unknown) => void
    ) => () => void;
    this.controlClient = new ControlClient({
      socketUrl,
      token: this.options.controlToken ?? null,
      invoke: (method, params) => callable[method].apply(this, params),
      subscribe: subscribeByName,
      events: this.events,
    });
    this.controlClient.connect();
    this.events.add(() => {
      this.controlClient?.dispose();
      this.controlClient = undefined;
    });
  }

  private applyViewerConfigStateCore(viewerConfig: ZarrViewerConfig | undefined): void {
    applyViewerConfigStateHelper(viewerConfig, {
      showHelp: () => showHelpOverlay(this.inputHandler.getRegisteredShortcutBindings()),
      renderingControls: this.renderingControls,
      performanceMonitor: this.performanceMonitor,
      inputHandler: this.inputHandler,
      scaleBar: this.scaleBar,
      layersPanel: this.layersPanel,
      overlayManager: this.overlayManager,
      setTheme: (id) => ThemeManager.getInstance().setTheme(id),
      setDimensionValue: (i, v) => sceneDimsManager.setDimensionValue(i, v),
      setDocumentTitle,
      startDimensionAnimation: (dim, options) => {
        // Resolved lazily: the animation manager is built when a scene with an
        // animatable dimension loads, which for the first dataset happens in
        // the same init pass as this call.
        const manager = this.inputHandler.getAnimationManager();
        manager?.play(dim, {
          targetFPS: options.targetFPS,
          loopMode: options.loopMode,
          direction: options.direction,
        });
        // Keep the playback options together, then apply the independent step
        // override. `setStepSize` validates the value itself and rejects a
        // non-positive one rather than animating nowhere.
        if (options.stepSize !== undefined) {
          manager?.setStepSize(dim, options.stepSize);
        }
      },
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
    // Story captions wait for the camera when a waypoint asks for it: the gate
    // reads the LIVE driver, so it holds whichever scene's waypoints are
    // installed, before or after the overlays themselves were created.
    this.overlayManager.setTransitGate(() => this.waypointDriver?.inTransit === true);
  }

  /**
   * Initialize GPU picking for hover tooltips, element actions, and embedder
   * listeners. Activates for labels, image labels, keys, link/copy templates,
   * or pick consumers. Wires PickingSystem to the label/key loaders,
   * ImageLabelLoader, and OverlayManager.
   */
  private async initPicking(): Promise<void> {
    const result = await initPickingImpl({
      sceneManager: this.sceneManager,
      pickingEvents: this.pickingEvents,
      previous: {
        pickingSystem: this.pickingSystem,
        labelLoader: this.labelLoader,
        imageLabelLoader: this.imageLabelLoader,
        keyLoader: this.keyLoader,
      },
      getOverlayManager: () => this.overlayManager,
      onSelection: (sel) => this.embedderEvents.emit('selection', sel),
      hasSelectionConsumer: () => this.embedderEvents.hasListeners('selection'),
      hasElementActionConsumer: () =>
        this.embedderEvents.hasListeners('element-click') ||
        this.embedderEvents.hasListeners('element-contextmenu'),
      allowLinks: this.options.allowLinks ?? true,
      onElementClick: (p) => this.embedderEvents.emit('element-click', p),
      onElementContextMenu: (p) => this.embedderEvents.emit('element-contextmenu', p),
    });
    this.pickingSystem = result.pickingSystem;
    this.labelLoader = result.labelLoader;
    this.imageLabelLoader = result.imageLabelLoader;
    this.keyLoader = result.keyLoader;
  }

  /**
   * Tear down the current picking session (listeners, system, loaders).
   * Called by `loadDataset` UP-FRONT — alongside `disposeOverlays`, in
   * lockstep with `clearSceneContent()` — so a failing mid-load never
   * leaves a stale session firing picks against disposed geometries.
   * `initPicking` at the end of the load is the (re)creation point;
   * between the two no pick can fire (all listeners are removed here).
   */
  private disposePicking(): void {
    disposePickingSessionImpl({
      pickingEvents: this.pickingEvents,
      previous: {
        pickingSystem: this.pickingSystem,
        labelLoader: this.labelLoader,
        imageLabelLoader: this.imageLabelLoader,
        keyLoader: this.keyLoader,
      },
    });
    this.pickingSystem = undefined;
    this.labelLoader = undefined;
    this.imageLabelLoader = undefined;
    this.keyLoader = undefined;
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
      closeBrowser: () => this.datasetBrowser?.close(),
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
   * Auto-retry failed loaders when connectivity is restored — the trigger
   * for `SceneLoader.retryAllFailedLoaders` (whose doc names "after
   * connectivity is restored" as the intended use). Live accessor through
   * the manager so dataset switches keep pointing at the current loader.
   */
  private setupOnlineRetry(): void {
    const manager = SceneLoaderManager.getInstance();
    installOnlineRetry({
      events: this.events,
      getLoader: () => getSceneLoader(),
      toast: (message, durationMs) => notifier.toast(message, durationMs),
      subscribeAutoRetryableFailure: (listener) => {
        manager.setAutoRetryableFailureCallback(listener);
        return () => manager.setAutoRetryableFailureCallback(null);
      },
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
      adaptiveDPRManager: this.adaptiveDPRManager,
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

  // ==================== Programmatic embedder API ====================
  // Flat, additive methods so a host page can drive the viewer without the
  // built-in UI. Mutating methods guard on `isInitialized`; shortcut lookup
  // also supports initialization and returns undefined before input exists.

  /** Register a custom keyboard-routing context. */
  registerContext(context: InputContextId, config: ContextConfig): void {
    this.requireInputHandler('registerContext').registerContext(context, config);
  }

  /** Remove a custom keyboard-routing context and all of its bindings. */
  unregisterContext(context: InputContextId): void {
    this.requireInputHandler('unregisterContext').unregisterContext(context);
  }

  /** Register a keyboard binding in a built-in or custom context. */
  registerBinding(context: InputContextId, binding: KeyBinding): void {
    this.requireInputHandler('registerBinding').registerBinding(context, binding);
  }

  /** Remove a keyboard binding. Missing bindings are ignored. */
  unregisterBinding(
    context: InputContextId,
    key: string,
    modifiers?: KeyBinding['modifiers']
  ): void {
    this.requireInputHandler('unregisterBinding').unregisterBinding(context, key, modifiers);
  }

  /** Activate a nested keyboard-routing context. */
  pushContext(context: InputContextId): void {
    this.requireInputHandler('pushContext').pushContext(context);
  }

  /** Restore the context active before the latest {@link pushContext}. */
  popContext(): void {
    this.requireInputHandler('popContext').popContext();
  }

  /** Enable or disable all viewer keyboard shortcuts. */
  setInputEnabled(enabled: boolean): void {
    this.requireInputHandler('setInputEnabled').setEnabled(enabled);
  }

  /** Resolve the active chord label for a registered action, if available. */
  shortcutForAction(actionId: string): string | undefined {
    if (!this.isInitialized && !this.isInitializing) return undefined;
    return this.inputHandler?.getShortcutLabel(actionId);
  }

  private requireInputHandler(method: string): InputHandler {
    if (!this.isInitialized) {
      throw new Error(`LuxarApp.${method} called before init()`);
    }
    return this.inputHandler;
  }

  /**
   * Subscribe to a public embedder event. Returns an unsubscribe function.
   *
   * Events: `dataset-loaded`, `dataset-error`, `dataset-fault`, `dimensions-changed`,
   * `selection` (see {@link LuxarEmbedderEventMap}). Safe to call before
   * `init()`; the per-app emitter outlives individual init/dispose cycles.
   *
   * @example
   * ```ts
   * const off = app.on('dataset-loaded', ({ src }) => console.log('loaded', src));
   * // later: off();
   * ```
   */
  on<K extends keyof LuxarEmbedderEventMap>(
    event: K,
    listener: (payload: LuxarEmbedderEventMap[K]) => void
  ): Unsubscribe {
    // Isolate embedder callbacks at the public boundary: the event bus does
    // not catch listener errors, and some events (dataset-loaded,
    // dimensions-changed, selection) are emitted from inside the viewer's
    // own control flow. A throwing consumer listener must NOT corrupt that —
    // e.g. without this guard a throwing `dataset-loaded` handler would
    // propagate into loadDataset's catch, emit a spurious `dataset-error`,
    // and reject switchDataset() on an otherwise-successful load.
    const safe = (payload: LuxarEmbedderEventMap[K]): void => {
      try {
        listener(payload);
      } catch (err) {
        log.warning(Modules.APP, `embedder '${String(event)}' listener threw:`, err);
      }
    };
    return this.embedderEvents.on(event, safe);
  }

  /**
   * Load a different dataset into the running viewer, reusing the full
   * teardown+reload path (the same one the built-in dataset browser uses).
   * Resolves when the new scene is loaded; emits `dataset-loaded` /
   * `dataset-error`.
   *
   * Rejects if a switch is already in progress (the reload does a full scene
   * teardown — overlapping calls would corrupt state).
   *
   * @throws if the app has not been initialised yet.
   */
  switchDataset(src: string): Promise<void> {
    if (!this.isInitialized) {
      if (this.isInitializing) {
        throw new Error('Luxar is still starting up; try again in a moment.');
      }
      throw new Error('LuxarApp.switchDataset called before init()');
    }
    if (this.switchInFlight) {
      return Promise.reject(
        new Error('LuxarApp.switchDataset: a dataset switch is already in progress')
      );
    }
    // A flight aimed at the outgoing scene has nothing to land on.
    this.cameraFlight?.cancel();
    this.options.src = src;
    // Re-title the tab for the incoming scene BEFORE the load. Neither of the
    // two things that could be naming it survives a switch: `?title=` names
    // the dataset the server started with (and is dropped from the address bar
    // by `buildDataSourceBrowserUrl`), and an authored `viewer_config.title`
    // names the scene we are about to tear down. Leaving either in place is
    // how a tab ends up advertising a scene it no longer shows. The new
    // scene's own authored title, if it has one, wins again a moment later in
    // `applyViewerConfigState`.
    setDocumentTitle(dataSourceDocumentTitle(src));
    this.switchInFlight = this.loadDataset(src).finally(() => {
      this.switchInFlight = undefined;
    });
    return this.switchInFlight;
  }

  /**
   * Current nD dimension state (metadata + current slice positions + ranges).
   * All array fields are cloned — safe to read without mutating internals;
   * use {@link setDimensionValue} to change a slice.
   *
   * @throws if the app has not been initialised yet.
   */
  getDimensions(): EmbedderDimensions {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.getDimensions called before init()');
    }
    const dims = sceneDimsManager.getDims();
    if (!dims) {
      return { ndim: 0, displayed: [], currentStep: [], metadata: [], ranges: [] };
    }
    const ranges = sceneDimsManager.getDimensionRanges() ?? [];
    return {
      ndim: dims.ndim,
      displayed: [...dims.displayed],
      currentStep: [...dims.currentStep],
      // Deep-clone metadata: a spread alone would leave the nested `range`
      // and `categories` arrays aliasing the scene-dims manager's internals,
      // so an embedder mutating them would corrupt viewer state.
      metadata: sceneDimsManager.getDimensionMetadata().map((m) => ({
        ...m,
        ...(m.range ? { range: [m.range[0], m.range[1]] as [number, number] } : {}),
        ...(m.categories ? { categories: [...m.categories] } : {}),
      })),
      ranges: ranges.map((r) => [r[0], r[1]] as [number, number]),
    };
  }

  /** Current latched fault episode, or null when no dataset is loaded or the latch is clear. */
  getDatasetFault(): DatasetFaultPayload | null {
    const error = this.datasetFaultLoader?.archiveFault;
    if (!error || !this.currentDatasetSrc) return null;
    return { src: this.currentDatasetSrc, error };
  }

  /**
   * Set the slice position of a single (non-displayed) dimension. Clamped and
   * quantized by the scene-dims manager; triggers a data update and emits
   * `dimensions-changed`. Await {@link awaitDimensionUpdate} for the load.
   *
   * @throws if the app has not been initialised yet.
   */
  setDimensionValue(index: number, value: number): void {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.setDimensionValue called before init()');
    }
    sceneDimsManager.setDimensionValue(index, value);
  }

  /** Resolve once any in-flight dimension data update has settled. */
  awaitDimensionUpdate(): Promise<void> {
    return sceneDimsManager.waitForUpdate();
  }

  /**
   * Recenter/fit the camera on the loaded scene (the built-in `F`-key
   * behaviour).
   *
   * @throws if the app has not been initialised yet.
   */
  recenterCamera(): void {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.recenterCamera called before init()');
    }
    this.sceneManager.centerCameraOnScene();
  }

  /**
   * Current camera pose (position, target, up, projection params) — the same
   * `camera` block {@link captureSnapshot} produces.
   *
   * @throws if the app has not been initialised yet.
   */
  getCameraPose(): CameraSnapshot {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.getCameraPose called before init()');
    }
    return captureViewerSnapshot(this.sceneManager).camera;
  }

  /**
   * Apply a camera pose previously obtained from {@link getCameraPose} (or a
   * snapshot's `camera`). Controls are re-initialised so orbit/fly updates
   * don't snap back.
   *
   * @throws if the app has not been initialised yet.
   */
  setCameraPose(pose: CameraSnapshot): void {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.setCameraPose called before init()');
    }
    restoreCamera(this.sceneManager, pose);
  }

  /**
   * Fly the camera smoothly to `pose` (the same shape {@link getCameraPose}
   * returns). Interpolates in the orbit parameterisation — focus target,
   * viewing direction, distance, up — so the transition arcs around the
   * scene instead of cutting through it, and lands on `pose` exactly.
   *
   * Any user input on the canvas or keyboard cancels the flight where it is,
   * as does a newer `flyTo()` or a dataset switch; the promise then resolves
   * `{ completed: false }`. `durationMs: 0` is equivalent to
   * {@link setCameraPose}.
   *
   * @throws if the app has not been initialised yet.
   */
  flyTo(pose: CameraSnapshot, opts?: FlyToOptions): Promise<FlightResult> {
    if (!this.isInitialized || !this.cameraFlight) {
      throw new Error('LuxarApp.flyTo called before init()');
    }
    return this.cameraFlight.flyTo(pose, opts);
  }

  /**
   * Copy of the live rendering settings (tone mapping, exposure, bloom,
   * anti-aliasing, navigation feel, …) — the `RenderingSettings` object the
   * Rendering panel edits.
   *
   * @throws if the app has not been initialised yet.
   */
  getRenderingSettings(): RenderingSettings {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.getRenderingSettings called before init()');
    }
    return this.renderingControls.getSettingsSnapshot();
  }

  /**
   * Apply a partial rendering-settings override. Takes the same path an
   * authored `viewer_config` takes at load (validation, camera FOV / planes,
   * navigation, post-processing), so anything an author can bake a
   * controller can set live. Values are validated and clamped; nothing is
   * persisted to the user's stored preferences.
   *
   * @throws if the app has not been initialised yet.
   */
  setRenderingSettings(patch: Partial<RenderingSettings>): void {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.setRenderingSettings called before init()');
    }
    this.renderingControls.applyOverrides(patch);
  }

  /**
   * Per-layer appearance summaries in Layers-panel order (copies). Empty
   * before a scene has loaded.
   *
   * @throws if the app has not been initialised yet.
   */
  getLayers(): LayerSummary[] {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.getLayers called before init()');
    }
    return this.layersPanel?.getLayerSummaries() ?? [];
  }

  /**
   * Patch one layer's appearance (visibility, opacity, display window, gamma,
   * colormap, blending mode, absorption, order). Each field takes the same
   * route the Layers panel's own control does. `path` is the
   * {@link LayerSummary.path} of a layer from {@link getLayers}.
   *
   * @throws if the app has not been initialised yet, or on an unknown path.
   */
  setLayer(path: string, patch: LayerPatch): void {
    if (!this.isInitialized || !this.layersPanel) {
      throw new Error('LuxarApp.setLayer called before init()');
    }
    this.layersPanel.setLayer(path, patch);
  }

  /**
   * One-call mirror of everything a remote controller needs: dataset,
   * camera, slice position, rendering settings, layers. All copies.
   *
   * @throws if the app has not been initialised yet.
   */
  getViewerState(): ViewerState {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.getViewerState called before init()');
    }
    return {
      src: this.currentDatasetSrc ?? this.options.src ?? null,
      // Read from the document rather than kept as a field: the authored
      // title, the `?title=` parameter and the built-in default all land
      // there already, so this reports what is actually on the tab.
      title: document.title,
      camera: this.getCameraPose(),
      dimensions: this.getDimensions(),
      rendering: this.getRenderingSettings(),
      layers: this.getLayers(),
      audio: this.getAudioState(),
    };
  }

  /**
   * The sound layer's state: `AudioContext` state (`suspended` = the display
   * still needs its tap), mute, master gain, panning model, bus gains, the
   * names of the nodes playing, and whether the scene has sound nodes at all.
   *
   * @throws if the app has not been initialised yet.
   */
  getAudioState(): AudioState {
    if (!this.isInitialized || !this.audioEngine) {
      throw new Error('LuxarApp.getAudioState called before init()');
    }
    return this.audioEngine.getState();
  }

  /**
   * Live mixer patch: master gain and mute (both persisted like the rail's own
   * controls), per-bus gains, panning model. Fields left out are untouched.
   *
   * @throws if the app has not been initialised yet.
   */
  setAudio(patch: AudioPatch): void {
    if (!this.isInitialized || !this.audioEngine) {
      throw new Error('LuxarApp.setAudio called before init()');
    }
    this.audioEngine.setAudio(patch);
  }

  /**
   * Start one sound node by name (its last path segment, or the full path)
   * regardless of its slab audibility. Returns `false` for an unknown name or
   * a clip that has not decoded yet.
   *
   * @throws if the app has not been initialised yet.
   */
  playSound(name: string): boolean {
    if (!this.isInitialized || !this.audioEngine) {
      throw new Error('LuxarApp.playSound called before init()');
    }
    return this.audioEngine.play(name);
  }

  /**
   * Stop one sound node by name with its own fade-out. Returns `false` when
   * nothing by that name was playing.
   *
   * @throws if the app has not been initialised yet.
   */
  stopSound(name: string): boolean {
    if (!this.isInitialized || !this.audioEngine) {
      throw new Error('LuxarApp.stopSound called before init()');
    }
    return this.audioEngine.stop(name);
  }

  /**
   * Resize the viewer to its canvas's current client box. Called
   * automatically when the canvas resizes (via a ResizeObserver); expose it
   * for explicit/programmatic relayout (e.g. right after toggling a host
   * panel synchronously).
   *
   * @throws if the app has not been initialised yet.
   */
  resize(): void {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.resize called before init()');
    }
    this.sceneManager.resizeToCanvas();
  }

  /**
   * Render the current frame to an encoded image Blob (PNG by default).
   * Async because the WebGPU readback path is async.
   *
   * @throws if the app has not been initialised yet, or if encoding fails.
   */
  screenshot(opts?: ScreenshotOptions): Promise<Blob> {
    if (!this.isInitialized) {
      throw new Error('LuxarApp.screenshot called before init()');
    }
    return captureScreenshot(this.sceneManager, this.overlayManager ?? null, opts);
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
    this.isInitializing = false;

    // Hand the page its own <title> back. The viewer overwrites document.title
    // with the scene's name, which is a mutation of a host-page global: an
    // embedder that removes the viewer would otherwise be left titled after a
    // scene that no longer exists. Restored BEFORE the teardown pipeline so a
    // step that throws partway can't strand it.
    setDocumentTitle(null);

    this.datasetFaultUnsubscribe?.();
    this.datasetFaultUnsubscribe = undefined;
    this.datasetFaultLoader = undefined;
    this.currentDatasetSrc = undefined;

    runDisposePipeline({
      events: this.events,
      pickingEvents: this.pickingEvents,
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      audioEngine: this.audioEngine,
      performanceMonitor: this.performanceMonitor,
      adaptiveDPRManager: this.adaptiveDPRManager,
      resolutionIndicator: this.resolutionIndicator,
      inputHandler: this.inputHandler,
      renderingControls: this.renderingControls,
      recordingPanel: this.recordingPanel,
      layersPanel: this.layersPanel,
      controlRail: this.controlRail,
      scaleBar: this.scaleBar,
      colormapLegend: this.colormapLegend,
      overlayManager: this.overlayManager,
      pickingSystem: this.pickingSystem,
      labelLoader: this.labelLoader,
      imageLabelLoader: this.imageLabelLoader,
      keyLoader: this.keyLoader,
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
      clearControlRail: () => {
        this.controlRail = undefined;
      },
      clearPickingSystem: () => {
        this.pickingSystem = undefined;
      },
      clearLabelLoader: () => {
        this.labelLoader = undefined;
      },
      clearKeyLoader: () => {
        this.keyLoader = undefined;
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
