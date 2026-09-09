import type * as THREE from 'three';
import { SceneManager } from '../../../scene/scene-manager';
import { AnimationController } from '../../../scene/animation/animation-controller';
import { PerformanceMonitor } from '../../../ui/performance-monitor';
import { DebugConsole } from '../../../ui/debug-console';
import {
  AdaptiveDPRManager,
  mobileAdaptiveDprOverrides,
} from '../../../rendering/adaptive-dpr-manager';
import { ResolutionIndicator } from '../../../ui/resolution-indicator';
import { InputHandler } from '../../../input';
import { DimensionSliders } from '../../../ui/dimension-sliders';
import { RenderingControls } from '../../../ui/rendering-controls';
import { RecordingPanel } from '../../../ui/recording-panel';
import { LayersPanel } from '../../../ui/layers';
import { ControlRail } from '../../../ui/control-rail';
import { buildRailItems } from './build-rail-items';
import { AudioEngine } from '../../../audio/audio-engine';
import { resolveTargetNodeCenter } from '../../../scene/scene-manager/camera/camera-setup';
import { getViewerContainer } from '../../../utils/viewer-container';
import { DataMonitorManager } from '../../../ui/data-monitor-manager';
import { SceneLoaderManager, getSceneLoader } from '../../../data/scene-loader-manager';
import { LODGroupRegistry } from '../../../scene/lod-group-registry';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { notifier } from '../../../utils/cross-layer/notifier';
import { log, Modules } from '../../../utils/log';
import { config } from '../../../config';
import { getGpuByteBudget } from '../../../rendering/gpu-byte-budget';
import { getMaxPixelRatio, setHighDPRAllowed } from '../../../rendering/pixel-ratio-cap';
import {
  configureDepthSort,
  setDepthSortEnabled,
  warmUpDepthSortWorker,
  evaluateDepthSortPerFrame,
} from '../../../rendering/depth-sort-coordinator';
import { materialManager } from '../../../rendering';
import { createKTX2TextureDecoder } from '../../../rendering/ktx2-texture-decoder';
import { resolveFactories, type AppFactories } from '../factories';
import type { LuxarAppOptions } from '../options';
import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { wireDensityGuard } from './density-guard-wiring';
import { buildLoadActivityPredicate } from './load-activity';
import { wireSceneEnvironment } from './environment-wiring';
import { getInputProfile } from '../../../utils/input-capabilities';

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
  /** The sound layer (nodes attach per scene in `LuxarApp.installAudio`). */
  audioEngine: AudioEngine;
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
  /** Emit a public embedder event (the audio engine's `sound-started` / `sound-ended`). */
  emitEmbedderEvent: (event: 'sound-started' | 'sound-ended', payload: { name: string }) => void;
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
    // Mobile GPUs can spend 50-300 ms linking each blend variant, and Safari
    // before 18.2 has no requestIdleCallback to hide that work. Resolve the
    // device default here so direct LuxarApp embedders get the same protection
    // as the standalone bootstrap; an explicit false remains the opt-out.
    blendWarmup: ports.options.blendWarmup !== false && getInputProfile().deviceClass !== 'mobile',
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
  // compensation is ON by default; ?no-lod-energy disables it. Both come in as
  // app OPTIONS (the standalone bootstrap threads them from the URL params;
  // embedders set them directly — the pipeline never reads window.location)
  // and are captured once at wiring time (a reload re-reads them).
  const lodCrossFadeEnabled = ports.options.lodFade ?? true;
  const lodEnergyCompEnabled = ports.options.lodEnergyComp ?? true;
  // Opt-in: force the finest LOD for capture-quality output (?lod-finest).
  const lodFinestEnabled = ports.options.lodFinest ?? false;
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
      hasArchiveFault: () => owner.archiveFault !== null,
      requestReprocess: (paths) => owner.requestReprocess(paths),
      // A view PASS in flight or queued — not a refinement hold, which the
      // loader parks a resync through (see `LODGroupRegistryOwner`).
      isUpdateInProgress: () => owner.isLoadPassInProgress(),
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
      // a hard swap (blendable modes only: additive/luminous/volumetric).
      // Read once at wiring time.
      getCrossFadeEnabled: () => lodCrossFadeEnabled,
      // Streaming brightness compensation (ON by default; ?no-lod-energy disables):
      // scale a streaming additive/luminous/volumetric leaf's opacity by 1/e(k) so its
      // partial ladder prefix renders at full-level brightness (no brightening
      // pop as chunks arrive). Read once at wiring time.
      getEnergyCompEnabled: () => lodEnergyCompEnabled,
      // Force-finest capture override (?lod-finest via LuxarAppOptions.lodFinest):
      // always select the finest level and never coarsen off-screen.
      getForceFinestLOD: () => lodFinestEnabled,
      // Register a fade's clone-on-first-use material so it keeps receiving
      // per-frame camera-uniform updates (an unregistered gsplat clone would
      // project with stale camera params).
      // No cast: `register` takes a plain `THREE.Material` and dispatches on
      // camera-awareness internally. The `as Parameters<typeof register>[0]` that used
      // to sit here existed only to satisfy an `& CameraAwareMaterial` requirement the
      // manager no longer imposes — and being self-referential, it would have silently
      // accepted anything the parameter type later became.
      registerMaterial: (material) => materialManager.register(material),
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
  SceneLoaderManager.getInstance().setKTX2TextureDecoder(
    createKTX2TextureDecoder(sceneManager.renderer)
  );
  // Depth-sort coordinator (Phases 2-3): the gsplats commit path has no
  // camera (SceneLoader deliberately owns no camera state), so the
  // coordinator gets the live camera + render wake-up here — the same
  // dependency-inversion as setRequestRender above.
  setDepthSortEnabled(config.depthSort.enabled && (ports.options.depthSort ?? true));
  configureDepthSort({
    // A live GETTER, not sceneManager.camera captured by value: the
    // ortho-mode toggle replaces the camera object, and sorts must track
    // whichever camera is current (same pattern as the LOD registry's
    // getCamera above).
    getCamera: () => sceneManager.camera,
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
    // A partition's BSP `axis` is a center-column index; the painter's-order
    // traversal needs to know which column is displayed as x/y/z. Same
    // accessor the LOD registry gets above, and read live so nD navigation
    // is tracked (the stored tree stays valid, the mapping does not).
    getDisplayDims: () => sceneDimsManager.getDims()?.displayed ?? null,
  });
  // Start the SortWorker NOW, while the app is still idle, rather than
  // letting the first order-dependent commit spawn it. That commit lands
  // when this thread and the data-worker pool are saturated decoding the
  // scene, and the worker's `initialize()` reply has to be dispatched on
  // this very thread — on a multi-million-element scene it misses its
  // deadline there, which used to disable sorting for the whole session.
  // Honours `setDepthSortEnabled` above (so `?depthSort=0` spawns nothing).
  warmUpDepthSortWorker();
  // Camera-motion re-sort scheduler (Phase 3, spec §6) + global cross-node
  // renderOrder assignment. Same per-frame slot pattern as
  // 'lod-group-selector' below; the evaluation early-outs when no
  // order-dependent node exists (with nodes it allocates only the small
  // per-frame order slots — documented in assignGlobalRenderOrder).
  animationController.addPerFrameCallback('depth-sort-scheduler', () => {
    evaluateDepthSortPerFrame();
  });
  // Projected-density guard walker (config.densityGuard; `?no-density-guard`):
  // measures elements per drawing-buffer pixel for every committed data mesh
  // once per frame. Consumers read it through getProjectedDensityTracker()
  // (shader keep fraction, refinement rung cap, getPerf().density).
  // The keep-fraction ladder, the refinement rung-gate provider and the
  // per-frame walk are wired in density-guard-wiring.ts (unit-tested there).
  const loaderManager = SceneLoaderManager.getInstance();
  const densityWiring = wireDensityGuard({
    configEnabled: config.densityGuard.enabled,
    option: ports.options.densityGuard,
    config: config.densityGuard,
    capOverride: ports.options.densityCap,
    energyComp: lodEnergyCompEnabled,
    sceneManager,
    registerMaterial: (material) => materialManager.register(material),
    setRefinementDensityProvider: (provider, caps) =>
      loaderManager.setRefinementDensityProvider(provider, caps),
    getDefaultLoader: () => getSceneLoader('default'),
    getAdaptiveDpr: () => partial.adaptiveDPRManager,
    requestRender: () => animationController.startAnimation(),
  });
  animationController.addPerFrameCallback('projected-density', densityWiring.perFrame);
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

  // Seed the pixel-ratio cap from config BEFORE anything sizes a frame.
  // The renderer boundary (dpr-policy.getActivePixelRatio) reads the cap
  // on every resize, so with high DPR disallowed — the default — even the
  // opening frame renders at CSS resolution instead of paying 4x the
  // fragment cost until the adaptive loop reacts. A scene's authored
  // `allow_high_dpr`, and any persisted per-scene choice, land later via
  // RenderingControls.
  setHighDPRAllowed(config.renderingControls.defaults.allowHighDPR);

  // Initialize adaptive DPR manager for dynamic resolution scaling
  // On a phone/tablet: a legible floor and a 60 Hz threshold ceiling (a
  // ProMotion iPad's learned 120 Hz mark would otherwise read a healthy
  // 60 fps as distress). `undefined` on a laptop/desktop — unchanged.
  const adaptiveDPRManager = new AdaptiveDPRManager(mobileAdaptiveDprOverrides());
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

  // While the viewer is not SETTLED — an updateView sweep, any loader's load
  // pass, a lazy LOD level load, or the post-load refinement drain (each
  // rung is fetch + decode + commit with the lock released between passes)
  // — frame jank reflects that work, not steady-state render cost, and the
  // manager suppresses probe/estimator learning for those samples. Same
  // predicate the perf probes read as `getPerf().isSettled`, inverted.
  // TRUE while load activity is in flight (the adaptive-DPR manager's sense).
  const isLoadActive = buildLoadActivityPredicate({
    getDefaultLoader: () => getSceneLoader('default'),
    isAnyLoadPassInProgress: () => SceneLoaderManager.getInstance().isAnyLoadPassInProgress(),
  });
  adaptiveDPRManager.setLoadActivityPredicate(isLoadActive);

  // The scene environment's live behaviour (re-capture on commit / slice /
  // appearance change once SETTLED — the same predicate, inverted) and the
  // `?bake-env` one-shot.
  wireSceneEnvironment({
    sceneManager,
    animationController,
    events: ports.events,
    options: ports.options,
    isSettled: () => !isLoadActive(),
  });

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
      // The indicator displays percent-of-FULL-QUALITY resolution, so
      // normalize the absolute DPR here — on a 2x retina display a
      // reduced DPR of 1.8 must read as "90%", not "180%".
      //
      // Normalized against the CEILING, not the display's DPR: with high
      // DPR disallowed, 1.0 IS full quality, and dividing by native would
      // both report a permanent "50%" and fire the toast on every load of
      // every scene.
      resolutionIndicator.show(dpr / getMaxPixelRatio());
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
      // Elected default (argless), not the literal 'default' id — matches
      // the sibling consumers (core/app.ts, build-rail-items.ts) and stays
      // correct if an embedder ever names its loader.
      const sceneLoader = getSceneLoader();
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
      // The density guard is app-scoped (it outlives scenes), so its provider
      // is wired here, once per monitor, rather than through the per-scene
      // monitor wiring in data/ — which cannot import scene/ anyway.
      mgr
        .getMonitor(monitorId)
        ?.setDensityProvider({ getDensityStates: () => densityWiring.densityStates() });
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
  // ...and to the density guard, so the persisted per-scene Density Guard
  // choice is applied by loadSettings alongside the two DPR flags.
  renderingControls.setDensityGuardControl(densityWiring);

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
  // An offline capture renders its own pipeline pass per frame, so the
  // loop's render is discarded work — and during an EXR sequence it
  // paints a blown-out frame under the translucent overlay, because the
  // capture holds raw-HDR shader flags across its async readback. The
  // loop itself keeps running (per-frame callbacks must follow the
  // camera); only its render is skipped. Offline-only: the real-time
  // path records the canvas the loop paints. Keyed on the narrow
  // render-suppression flag rather than the capture's mutual-exclusion
  // flag, so a wedged capture teardown can't freeze the viewport.
  animationController.setRenderSkipPredicate(() => recordingPanel.isLoopRenderSuppressed());
  // Frame pacing must stay off for the whole of a capture, because both
  // capture families depend on the loop's untouched cadence: the real-time
  // MediaRecorder path records the canvas the loop paints (a paced gap is a
  // dropped frame in the video), and the offline capture drives its own
  // `await requestAnimationFrame` cadence while registering one-shot
  // per-frame orbit callbacks on this controller (a paced frame could miss
  // the capture's window and drop the orbit step). Hence the BROAD
  // `isCurrentlyRecording()` — `session.isAnyCaptureActive()`, i.e.
  // `session.isRecording`, the flag both of those paths set for the whole of
  // their run — rather than the narrow render-suppression flag used just
  // above. A plain screenshot does not set it and does not need it: it reads
  // the canvas after its own awaited frame rather than depending on the
  // loop's cadence.
  animationController.setPacingSuspendPredicate(() => recordingPanel.isCurrentlyRecording());
  // An offline turntable keeps the rAF loop running, so the auto-LOD selector
  // is live and frustum-aware for the whole sweep: a tile that left the
  // frustum mid-orbit is demoted (and may have had its fine level released by
  // the resident-byte budget), then reloads ASYNCHRONOUSLY on re-entry.
  // Without a wait those frames are exported at the coarse level and pop back
  // a few frames later (#1695). Force-finest was rejected as the fix — a
  // capture visits the whole scene, so pinning finest across a tiled partition
  // would make peak residency the entire dataset — so the capture waits
  // (bounded) instead.
  //
  // Read live, because the scene loader is created after the panel. The `null`
  // answer is load-bearing rather than a convenience: this provider is wired
  // unconditionally, so on a plain points/lines scene the capture would
  // otherwise spend its mandatory selector-catch-up rAF on every exported
  // frame waiting for a selector that does not exist. `null` = "no lod_group
  // to wait for" and skips the drain outright; only a scene with at least one
  // registered lod_group gets the boolean. Narrow on purpose, and narrower
  // than "nothing here can be mid-load" — a `--recipe stream` leaf has no
  // lod_group but does have a progressive ladder still streaming.
  recordingPanel.setLODSettledProvider(() => {
    const registry = getSceneLoader('default')?.lodGroupRegistry;
    if (!registry || registry.size() === 0) return null;
    return registry.isCaptureQuiescent();
  });

  // Initialize layers panel (per-node controls)
  const layersPanel = factories.layersPanel(document.body, animationController);
  partial.layersPanel = layersPanel;
  inputHandler.setLayersPanel(layersPanel);

  // Left activity rail — the always-visible, discoverable entry point to the
  // otherwise keyboard-only panels. Buttons dispatch through the same command
  // surface as keyboard shortcuts (via inputHandler.getUiActions()).
  // The sound layer. Constructs no AudioContext until a scene with sound nodes
  // attaches; every viewer piece it needs arrives as a port so `audio/` stays
  // below `scene/` in the layer order (see src/audio/README.md).
  const audioEngine = new AudioEngine({
    getCamera: () => sceneManager.camera,
    onCameraReplaced: (cb) => {
      const handler = (): void => cb();
      sceneManager.addEventListener('camera-changed', handler);
      return () => sceneManager.removeEventListener('camera-changed', handler);
    },
    getDims: () => sceneDimsManager.getDims(),
    onDimsChanged: (cb) => {
      const listener = (): void => cb();
      sceneDimsManager.addListener(listener);
      return () => sceneDimsManager.removeListener(listener);
    },
    getSceneGraph: () => getSceneLoader('default')?.sceneGraph ?? null,
    getSceneScale: () => sceneManager.getSceneScale(),
    container: getViewerContainer,
    emit: (event, payload) => ports.emitEmbedderEvent(event, payload),
    resolveNodeCenter: (name) => {
      const root = sceneManager.scene?.children?.find((c) => c.name === 'LuxarScene');
      return root ? resolveTargetNodeCenter(root as THREE.Group, name) : null;
    },
    notifyUiChanged: () => {
      // The rail refreshes on this. Guarded like a unit test's bare window stub
      // expects: no dispatcher, no event (the layers panel's own event is the model).
      if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
        window.dispatchEvent(new CustomEvent('luxar-audio-changed'));
      }
    },
  });
  partial.audioEngine = audioEngine;
  // Real-time recordings carry the mix the listener hears (SOUND_SPEC §6, Phase 3).
  recordingPanel.setAudioCapture({
    acquire: () => audioEngine.acquireCaptureStream(),
    release: (stream) => audioEngine.releaseCaptureStream(stream),
  });
  // Sound rows in the Layers panel: eye = mute, slider = gain (SOUND_SPEC §4.2).
  layersPanel.setAudioPort({
    setNodeMuted: (path, muted) => audioEngine.setNodeMuted(path, muted),
    setNodeGain: (path, gain) => audioEngine.setNodeGain(path, gain),
  });

  const ui = inputHandler.getUiActions();
  const railItems = buildRailItems({
    ui,
    shortcutForAction: (actionId) => inputHandler.getShortcutLabel(actionId),
    sceneManager,
    sceneDims: sceneDimsManager,
    renderingControls,
    animationController,
    adaptiveDPRManager,
    densityGuard: densityWiring,
    performanceMonitor,
    layersPanel,
    debugConsole,
    recordingPanel,
    audioEngine,
  });
  // Dock the perf readout as the rail's footer; the gauge above toggles it.
  const controlRail = new ControlRail(railItems, performanceMonitor.element);
  partial.controlRail = controlRail;
  inputHandler.setControlRail(controlRail);

  // Start animation loop first to ensure background is rendered
  animationController.startAnimation();

  partial.sceneSrc = sceneSrc;
  return partial as InitPipelineResult;
}
