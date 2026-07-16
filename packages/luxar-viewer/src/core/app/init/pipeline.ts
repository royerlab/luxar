import { SceneManager } from '../../../scene/scene-manager';
import { AnimationController } from '../../../scene/animation/animation-controller';
import { PerformanceMonitor } from '../../../ui/performance-monitor';
import { DebugConsole } from '../../../ui/debug-console';
import { AdaptiveDPRManager } from '../../../rendering/adaptive-dpr-manager';
import { ResolutionIndicator } from '../../../ui/resolution-indicator';
import { InputHandler } from '../../../input/input-handler';
import { DimensionSliders } from '../../../ui/dimension-sliders';
import { RenderingControls } from '../../../ui/rendering-controls';
import { RecordingPanel } from '../../../ui/recording-panel';
import { LayersPanel } from '../../../ui/layers';
import { ControlRail } from '../../../ui/control-rail';
import { buildRailItems } from './build-rail-items';
import { DataMonitorManager } from '../../../ui/data-monitor-manager';
import { SceneLoaderManager, getSceneLoader } from '../../../data/scene-loader-manager';
import { LODGroupRegistry } from '../../../scene/lod-group-registry';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { notifier } from '../../../utils/cross-layer/notifier';
import { log, Modules } from '../../../utils/log';
import { config } from '../../../config';
import { getGpuByteBudget } from '../../../rendering/gpu-byte-budget';
import {
  configureDepthSort,
  setDepthSortEnabled,
  evaluateDepthSortPerFrame,
} from '../../../rendering/depth-sort-coordinator';
import { materialManager } from '../../../rendering';
import { readUrlParams } from '../../../config/url-params';
import { resolveFactories, type AppFactories } from '../factories';
import type { LuxarAppOptions } from '../options';
import type { EventGroup } from '../../../utils/cross-layer/event-group';

/**
 * Everything `LuxarApp.init()` constructs is returned in this result.
 * The orchestrator copies the fields onto itself after the pipeline
 * resolves, so the helper never reaches into LuxarApp directly.
 */
export interface InitPipelineResult {
  sceneManager: SceneManager;
  animationController: AnimationController;
  performanceMonitor: PerformanceMonitor;
  debugConsole: DebugConsole;
  adaptiveDPRManager: AdaptiveDPRManager;
  resolutionIndicator: ResolutionIndicator;
  inputHandler: InputHandler;
  renderingControls: RenderingControls;
  recordingPanel: RecordingPanel;
  layersPanel: LayersPanel;
  controlRail: ControlRail;
  /** Resolved dataset URL — orchestrator routes to browser or load. */
  sceneSrc: string;
}

/**
 * Pipeline only builds the subsystem graph. Post-construction work
 * (dataset routing, beforeunload/focus listeners, debug interface)
 * runs in the orchestrator after it copies the result onto its own
 * fields, because those steps call back into orchestrator methods
 * that read `this.inputHandler` / `this.sceneManager` etc.
 *
 * `getPanelVisibilityStates` / `restorePanelVisibilityStates` are
 * the only callbacks the pipeline captures during construction —
 * they wire into `RecordingPanel.setPanelStateCallbacks` at the
 * moment of recording-panel build, and the orchestrator's closures
 * over `this` resolve correctly by the time the recording panel
 * actually calls them.
 */
export interface InitPipelinePorts {
  options: LuxarAppOptions;
  events: EventGroup;
  getPanelVisibilityStates: () => Map<string, boolean>;
  restorePanelVisibilityStates: (states: Map<string, boolean>) => void;
}

/**
 * Build the complete viewer subsystem graph (scene manager, animation
 * controller, panels, input handler, etc.), wire context-loss /
 * device-loss listeners, and kick the animation loop. Returns the
 * constructed components so the orchestrator can store them on its
 * fields. The order of construction is part of observable behaviour
 * (e.g. PerformanceMonitor depends on the bus the animation controller
 * emits each frame), so the helper keeps the steps strictly in their
 * original order.
 *
 * @param partial Mutable accumulator the pipeline fills as each
 *   subsystem is constructed. The caller pre-allocates and passes it
 *   in so partial state survives an exception: if `sceneManager.init`
 *   throws, `partial.sceneManager` is still set and `LuxarApp.dispose`
 *   can clean it up. The pipeline ALSO returns the same object cast
 *   to the full result type — on the happy path the caller can use
 *   either reference.
 */
export async function runInitPipeline(
  ports: InitPipelinePorts,
  partial: Partial<InitPipelineResult>
): Promise<InitPipelineResult> {
  // Inform users about expected console messages. The browser logs a
  // `GET … 404` line (with a JS stack trace) for every failed network
  // request; these cannot be suppressed from JS — only avoided by not
  // making the request. The loader intentionally probes for OPTIONAL
  // arrays/groups that many datasets omit, so 404s here are normal:
  //   • sharpnesses/  — optional per-point sharpness (falls back to default)
  //   • overlays/     — optional scene overlays group
  //   • */.zattrs     — optional zarr attributes on an existing array
  // Any of these returning 404 is expected and does NOT indicate a problem.
  log.info(
    Modules.LUXAR,
    'Note: the browser may log "GET … 404" lines for optional dataset features ' +
      '(sharpnesses, overlays, optional .zattrs/.zarray probes).'
  );
  log.info(
    Modules.LUXAR,
    'These 404s are expected and harmless — the loader probes for optional features ' +
      'that many datasets omit, and falls back cleanly when they are absent.'
  );

  const sceneSrc = ports.options.src ?? config.defaultZarrPath;

  // Resolve construction-factory overrides once. Without
  // overrides each entry simply calls the matching `new X(...)`.
  const factories: Required<AppFactories> = resolveFactories(ports.options.factories);

  // Initialize scene manager first. Stored on `partial` *before*
  // awaiting init() so a thrown init still leaves a disposable
  // reference behind for the orchestrator's error handler.
  const sceneManager = factories.sceneManager();
  partial.sceneManager = sceneManager;
  await sceneManager.init({
    canvas: ports.options.canvas,
    debug: ports.options.debug,
    renderer: ports.options.renderer,
    webgpuForceWebGL: ports.options.webgpuForceWebGL,
    perfTimestamp: ports.options.perfTimestamp,
  });

  // Initialize animation controller with HDR post-processing.
  // The PerformanceMonitor UI panel is constructed up here (not in
  // the controller) and subscribes to the bus events the
  // controller emits each frame. Owning it at the app level keeps
  // the lower scene/ layer free of UI imports.
  const animationController = factories.animationController(
    sceneManager.controls,
    sceneManager.postProcessing
  );
  partial.animationController = animationController;
  // Skip GPU rendering while the rendering context is dead.
  // isWebGLContextLost covers WebGL2 (contextRecovery flips it in the
  // webglcontextlost/restored handlers) but is hard-false under
  // ?renderer=webgpu, where no contextRecovery is constructed — so a
  // local latch, set by the webgpu-device-lost listener below, folds
  // WebGPU device loss (unrecoverable in this release) into the same
  // predicate. Everything keyed on it — render skips, adaptive-DPR
  // frame recording, the idle-restore render — becomes WebGPU-aware
  // through this one closure.
  let gpuDeviceLost = false;
  animationController.setContextLostPredicate(
    () => gpuDeviceLost || sceneManager.isWebGLContextLost()
  );
  // When the perf readout is shown, kick the loop once so it gets a live
  // reading if the scene had idled — but do NOT force continuous rendering
  // (that would defeat the idle-pause / battery saving). The FPS is live while
  // the scene renders and freezes at the last value when it idles.
  const performanceMonitor = new PerformanceMonitor({
    request: () => animationController.startAnimation(),
    release: () => {
      /* nothing to release — we never forced continuous rendering */
    },
  });
  partial.performanceMonitor = performanceMonitor;
  const debugConsole = new DebugConsole();
  partial.debugConsole = debugConsole;

  // Set up per-frame callback for dynamic clipping plane updates
  // Uses unique ID so it won't conflict with other per-frame callbacks (e.g., dimension animation)
  animationController.addPerFrameCallback('dynamic-clipping', () => {
    sceneManager.updateDynamicClippingPlanes();
  });

  // Wire LOD-group selection. The factory closes over the live
  // SceneManager so the registry's getters always read the current
  // camera / viewport / displayDims — no stale snapshots even after
  // resize or ortho-mode swaps. SceneLoaderManager forwards the
  // factory to each new SceneLoader instance.
  //
  // The per-frame callback reads `getSceneLoader('default')` rather
  // than closing over a specific registry: when the user loads a new
  // dataset, the SceneLoaderManager swaps loaders under the hood and
  // the callback keeps pointing at whichever is current. (Known
  // multi-instance limitation, deferred with the embeddability roadmap:
  // only the DEFAULT loader's registry is evaluated per frame. The
  // registry DEPS below are per-owner already, so when per-loader
  // callbacks arrive no further wiring changes are needed.)
  // LOD cross-fade is ON by default; ?no-lod-fade disables it. Streaming energy
  // compensation is ON by default; ?no-lod-energy disables it. Both captured once
  // at wiring time (a reload re-reads them).
  const lodUrlParams = readUrlParams();
  const lodCrossFadeEnabled = lodUrlParams.lodFade;
  const lodEnergyCompEnabled = lodUrlParams.lodEnergyComp;
  SceneLoaderManager.getInstance().setLODGroupRegistryFactory((owner) => {
    return new LODGroupRegistry({
      getCamera: () => sceneManager.camera,
      getViewportSize: () => {
        const canvas = sceneManager.renderer.domElement;
        return {
          width: canvas.clientWidth || window.innerWidth,
          height: canvas.clientHeight || window.innerHeight,
        };
      },
      // Return an empty list when scene dimensions aren't initialized
      // yet rather than the misleading ``[0, 1, 2]`` default — for
      // 2D scenes the latter projected onto a phantom Z axis. The
      // registry's existing ``displayDims.length < 2`` early-return
      // skips evaluation in this state.
      getDisplayDims: () => sceneDimsManager.getDims()?.displayed ?? [],
      // Resident-byte budget for loaded LOD geometry = the single,
      // adaptive GPU-geometry budget shared with the buffer pool (one VRAM
      // authority). Read dynamically so context-loss backoff applies live.
      getResidentByteBudget: () => getGpuByteBudget(),
      // Measured resident VRAM (active + pooled, real capacities) from the
      // buffer pool — the single accounting truth the registry uses to
      // decide when to demote cold levels. Read from the OWNING loader
      // (the factory receives it) so a non-default loader's registry never
      // consults the default loader's pool; a null pool (pre-construction /
      // pooling disabled) reads as 0 bytes, so the registry never evicts in
      // that state.
      getResidentBytes: () => owner.gpuBufferPool?.getResidentBytes() ?? 0,
      // Current view-update version. Lets the registry detect when a level's
      // committed geometry is stale for the current slice/displayDims (a
      // re-slice reloads geometry in place without flipping readiness) and
      // display a coarser FRESH level until the re-slice commits — the
      // slice-aware coarse-while-reloading fallback. Read from the OWNING
      // loader for the same per-instance reason as getResidentBytes.
      getViewVersion: () => owner.currentViewVersion,
      // Keep the on-demand render loop alive while a lazy fine level reloads
      // (it commits outside the per-slice sweep and can outlast the idle
      // timeout), so the swap-up to the fresh level fires when it lands.
      requestRender: () => animationController.startAnimation(),
      // LOD cross-fade (ON by default; ?no-lod-fade disables): the registry
      // blends adjacent LOD levels' opacity across a zoom transition instead of
      // a hard swap (additive/luminous only). Read once at wiring time.
      getCrossFadeEnabled: () => lodCrossFadeEnabled,
      // Streaming brightness compensation (ON by default; ?no-lod-energy disables):
      // scale a streaming additive/luminous leaf's opacity by 1/e(k) so its
      // partial ladder prefix renders at full-level brightness (no brightening
      // pop as chunks arrive). Read once at wiring time.
      getEnergyCompEnabled: () => lodEnergyCompEnabled,
      // Register a fade's clone-on-first-use material so it keeps receiving
      // per-frame camera-uniform updates (an unregistered gsplat clone would
      // project with stale camera params).
      registerMaterial: (material) =>
        materialManager.register(material as Parameters<typeof materialManager.register>[0]),
    });
  });
  // Wake the render loop after EVERY geometry commit (forwarded to each
  // SceneLoader). Late commits — progressive-refinement passes, failed-load
  // retries, the online auto-retry — land after the sweep that started
  // them; without this, a loop that idle-paused meanwhile keeps showing
  // the stale frame until the next user input. startAnimation is
  // idempotent (early-out while animating + idle-timer re-arm), so
  // per-node calls inside an atomic sweep are harmless.
  SceneLoaderManager.getInstance().setRequestRender(() => animationController.startAnimation());
  // Depth-sort coordinator (Phases 2-3): the gsplats commit path has no
  // camera (SceneLoader deliberately owns no camera state), so the
  // coordinator gets the live camera + render wake-up here — the same
  // dependency-inversion as setRequestRender above.
  setDepthSortEnabled(config.depthSort.enabled && lodUrlParams.depthSort);
  configureDepthSort({
    camera: sceneManager.camera,
    requestRender: () => animationController.startAnimation(),
    // Blending-mode-switch hook (spec §5.4): switching a gsplat layer TO
    // `normal` clears its noop stamp and forces a reprocess so the next
    // commit registers with the SortWorker.
    requestReprocess: () => {
      void getSceneLoader('default')?.updateView({});
    },
    // Phase 3: the per-frame scheduler skips dispatching while a view
    // update is in flight — the pending commit sorts from the
    // then-current pose anyway (same signal the refinement loop reads).
    isLoadInProgress: () => getSceneLoader('default')?.isUpdateInProgress() ?? false,
    // Sort round-trips show up as the monitor's 'Depth Sort' line.
    getProfiler: () => SceneLoaderManager.getInstance().getProfiler(),
  });
  // Camera-motion re-sort scheduler (Phase 3, spec §6). Same per-frame
  // slot pattern as 'lod-group-selector' below; the evaluation is
  // allocation-free and early-outs when no order-dependent node exists.
  animationController.addPerFrameCallback('depth-sort-scheduler', () => {
    evaluateDepthSortPerFrame();
  });
  animationController.addPerFrameCallback('lod-group-selector', () => {
    const loader = getSceneLoader('default');
    // When a substitutive-LOD group swaps its active level (a camera-move
    // event with no data reload), refresh the monitor's visible-element
    // tally so it reflects the level now rendering rather than staying
    // pinned to the default/coarsest level from the last updateView.
    if (loader?.lodGroupRegistry?.evaluatePerFrame()) {
      loader.refreshVisibleCounts();
      // A level swap changes what is being rendered — learned DPR
      // bounds (floor/backoff) describe the old level's render cost.
      // notifyContentChanged is internally coalesced, so per-frame
      // swap bursts during a zoom don't spam the ledger.
      partial.adaptiveDPRManager?.notifyContentChanged();
    }
  });

  // Initialize adaptive DPR manager for dynamic resolution scaling
  const adaptiveDPRManager = new AdaptiveDPRManager();
  partial.adaptiveDPRManager = adaptiveDPRManager;
  adaptiveDPRManager.setRenderer(sceneManager);
  animationController.setAdaptiveDPRManager(adaptiveDPRManager);

  // `?dpr=` pins a fixed pixel ratio for the whole session (deterministic
  // E2E/visual runs, repros). Must be applied here — before rendering
  // controls load persisted settings — and locks setEnabled() so those
  // settings can't re-enable adaptation later in init.
  if (ports.options.pinnedDPR !== undefined) {
    adaptiveDPRManager.pinManualDPR(ports.options.pinnedDPR);
  }

  // While an updateView sweep is in flight, frame jank reflects
  // decode/upload work, not steady-state render cost — the manager
  // suppresses probe/estimator learning for those samples.
  adaptiveDPRManager.setLoadActivityPredicate(
    () => getSceneLoader('default')?.isUpdateInProgress() ?? false
  );

  // Dataset/layer changes invalidate the learned DPR bounds (the floor
  // was evidence about the OLD content). Tracked via ports.events so
  // dispose removes it like every other app-level listener.
  const onLayersChanged = (): void => adaptiveDPRManager.notifyContentChanged();
  window.addEventListener('luxar-layers-changed', onLayersChanged);
  ports.events.add(() => window.removeEventListener('luxar-layers-changed', onLayersChanged));

  // Initialize resolution indicator and connect to DPR manager
  const resolutionIndicator = new ResolutionIndicator();
  partial.resolutionIndicator = resolutionIndicator;
  // Display target FPS: the warmup refresh cap rounded to a friendly
  // multiple of 5 (60 → 60). The live thresholds are refresh-relative
  // ratios, not user-facing targets, so the indicator shows the nominal
  // cap instead.
  const displayTargetFPS = Math.ceil(config.adaptiveDPR.refreshRateFallback / 5) * 5;
  resolutionIndicator.setTargetFPS(displayTargetFPS);
  adaptiveDPRManager.setOnDPRChangeCallback((dpr, isReducedResolution) => {
    if (isReducedResolution) {
      // The indicator displays percent-of-native resolution, so normalize
      // the absolute DPR here — on a 2x retina display a reduced DPR of
      // 1.8 must read as "90%", not "180%".
      resolutionIndicator.show(dpr / adaptiveDPRManager.getNativeDPR());
    } else {
      // Reset the indicator so it can show again on next reduced resolution mode activation
      resolutionIndicator.reset();
    }
  });

  // Re-register picking-system / GPU-pool resources after a WebGL
  // context-restore event. SceneManager rebuilds the renderer +
  // post-processing + material cache before dispatching, then we
  // call NodeFactory.rebuildAfterContextRestore on the loaded
  // scene so the picking system gets fresh registrations against
  // the new context.
  //
  // Track the listener via ports.events so dispose() removes it.
  // An untracked anonymous arrow here would leak if sceneManager
  // outlives app teardown — inconsistent with every other
  // app-level listener.
  if (typeof sceneManager.addEventListener === 'function') {
    // Give the SceneManager's 'change' event a live subscriber. The
    // context-restore path ends with `triggerChange()` ("trigger a
    // render") — without this, a context restored while the rAF loop is
    // idle-paused rebuilds + resizes (clearing the canvas) and then no
    // frame ever renders: blank viewer until the next input event.
    // startAnimation is idempotent, so the redundant dispatches from the
    // controls handler are harmless.
    const onSceneChange = (): void => animationController.startAnimation();
    sceneManager.addEventListener('change', onSceneChange);
    ports.events.add(() => sceneManager.removeEventListener('change', onSceneChange));

    const onContextRestored = (): void => {
      const sceneLoader = getSceneLoader('default');
      if (sceneLoader && sceneManager.scene) {
        sceneLoader.nodeFactory.rebuildAfterContextRestore(sceneManager.scene);
      }
    };
    sceneManager.addEventListener('webgl-context-restored', onContextRestored);
    ports.events.add(() =>
      sceneManager.removeEventListener('webgl-context-restored', onContextRestored)
    );

    // WebGPU device-loss is unrecoverable in this release (see
    // `scene-manager.setupContextLossHandling`). Surface it as a
    // user-facing error dialog with reload guidance — the only
    // remediation. Console diagnostics are already emitted by the
    // scene-manager handler; this listener exists to make sure the
    // user is told too.
    const onWebGPUDeviceLost = (event: { reason?: string; message?: string }): void => {
      // Latch the shared context-lost predicate (see its definition
      // above): stops draw calls against the dead device AND stops the
      // adaptive DPR manager from evaluating the artificially cheap
      // no-op frames (which would drive bogus scale-ups / false probe
      // verdicts). Unrecoverable in this release, so it never unlatches.
      gpuDeviceLost = true;
      adaptiveDPRManager.notifyPaused();
      const reason = event.reason ? ` (${event.reason})` : '';
      const detail = event.message ? `: ${event.message}` : '';
      notifier.error(
        `WebGPU device lost${reason}${detail}. ` + 'Please reload the page to continue.'
      );
    };
    sceneManager.addEventListener('webgpu-device-lost', onWebGPUDeviceLost);
    ports.events.add(() =>
      sceneManager.removeEventListener('webgpu-device-lost', onWebGPUDeviceLost)
    );
  }

  // Inject the monitor factory into SceneLoaderManager so each
  // SceneLoader can resolve its UI monitor without the data/ layer
  // importing ui/ directly.
  //
  // NOTE: the `typeof document === 'undefined'` guard below is
  // effectively dead in this pipeline — `sceneManager.init()` (called
  // earlier in this function) constructs a real WebGL/WebGPU renderer
  // and would throw long before reaching this factory in any genuine
  // SSR run. We keep the guard purely as belt-and-braces for the case
  // where the factory is invoked from a non-browser caller in the
  // future (or from a fake-DOM unit test that mocks the renderer but
  // not `document`). If you ever hoist SSR rejection to a single
  // top-of-pipeline check, this branch can be removed.
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
  const inputHandler = new InputHandler(
    sceneManager,
    animationController,
    performanceMonitor,
    debugConsole,
    (config) => new DimensionSliders(config)
  );
  partial.inputHandler = inputHandler;
  inputHandler.init();

  // Initialize rendering controls
  const renderingControls = factories.renderingControls(sceneManager.postProcessing, sceneManager);
  partial.renderingControls = renderingControls;

  // Connect rendering controls to animation controller
  renderingControls.setAnimationController(animationController);

  // Connect rendering controls to adaptive DPR manager for performance UI
  renderingControls.setAdaptiveDPRManager(adaptiveDPRManager);

  // Connect rendering controls to input handler
  inputHandler.setRenderingControls(renderingControls);

  // Initialize recording panel (screenshot/video capture)
  const recordingPanel = factories.recordingPanel(sceneManager, animationController);
  partial.recordingPanel = recordingPanel;
  recordingPanel.setPanelStateCallbacks(
    () => ports.getPanelVisibilityStates(),
    (states) => ports.restorePanelVisibilityStates(states)
  );
  recordingPanel.setAdaptiveDPRManager(adaptiveDPRManager);
  inputHandler.setRecordingPanel(recordingPanel);
  // The idle-pause native-DPR restore must never fire mid-capture —
  // recording resolution stays locked for the whole session.
  animationController.setIdleRestorePredicate(() => !recordingPanel.isCurrentlyRecording());

  // Initialize layers panel (per-node controls)
  const layersPanel = factories.layersPanel(document.body, animationController);
  partial.layersPanel = layersPanel;
  inputHandler.setLayersPanel(layersPanel);

  // Left activity rail — the always-visible, discoverable entry point to the
  // otherwise keyboard-only panels. Each button fires the SAME command as its
  // shortcut (via inputHandler.getUiActions()), so behaviour never drifts.
  const ui = inputHandler.getUiActions();
  const railItems = buildRailItems({
    ui,
    sceneManager,
    sceneDims: sceneDimsManager,
    renderingControls,
    animationController,
    adaptiveDPRManager,
    performanceMonitor,
    layersPanel,
    debugConsole,
    recordingPanel,
  });
  // Dock the perf readout as the rail's footer; the gauge above toggles it.
  const controlRail = new ControlRail(railItems, performanceMonitor.element);
  partial.controlRail = controlRail;

  // Start animation loop first to ensure background is rendered
  animationController.startAnimation();

  partial.sceneSrc = sceneSrc;
  return partial as InitPipelineResult;
}
