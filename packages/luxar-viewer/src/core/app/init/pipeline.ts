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
import { ControlRail, RAIL_ICONS, type ControlRailItem } from '../../../ui/control-rail';
import { buildSettingsPopover } from '../../../ui/rail-panels/settings-popover';
import { buildNavigationPopover } from '../../../ui/rail-panels/navigation-popover';
import { buildPerformancePopover } from '../../../ui/rail-panels/performance-popover';
import { nextControlType } from '../../../input/input-handler/commands/control-mode';
import { DataMonitorManager } from '../../../ui/data-monitor-manager';
import { SceneLoaderManager, getSceneLoader } from '../../../data/scene-loader-manager';
import { LODGroupRegistry } from '../../../scene/lod-group-registry';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { notifier } from '../../../utils/cross-layer/notifier';
import { log, Modules } from '../../../utils/log';
import { config } from '../../../config';
import { getGpuByteBudget } from '../../../rendering/gpu-byte-budget';
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
  // Skip GPU rendering while the WebGL context is lost.
  // SceneManager flips this flag in its webglcontextlost/restored
  // handlers; the loop polls each frame.
  animationController.setContextLostPredicate(() => sceneManager.isWebGLContextLost());
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
    });
  });
  animationController.addPerFrameCallback('lod-group-selector', () => {
    const loader = getSceneLoader('default');
    // When a substitutive-LOD group swaps its active level (a camera-move
    // event with no data reload), refresh the monitor's visible-element
    // tally so it reflects the level now rendering rather than staying
    // pinned to the default/coarsest level from the last updateView.
    if (loader?.lodGroupRegistry?.evaluatePerFrame()) {
      loader.refreshVisibleCounts();
    }
  });

  // Initialize adaptive DPR manager for dynamic resolution scaling
  const adaptiveDPRManager = new AdaptiveDPRManager();
  partial.adaptiveDPRManager = adaptiveDPRManager;
  adaptiveDPRManager.setRenderer(sceneManager);
  animationController.setAdaptiveDPRManager(adaptiveDPRManager);

  // Initialize resolution indicator and connect to DPR manager
  const resolutionIndicator = new ResolutionIndicator();
  partial.resolutionIndicator = resolutionIndicator;
  // Display target FPS rounded up from maxFPS (58 → 60) since targetFPS (55) is a hysteresis threshold
  const displayTargetFPS = Math.ceil(config.adaptiveDPR.maxFPS / 5) * 5;
  resolutionIndicator.setTargetFPS(displayTargetFPS);
  adaptiveDPRManager.setOnDPRChangeCallback((dpr, isReducedResolution) => {
    if (isReducedResolution) {
      resolutionIndicator.show(dpr);
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

  // Initialize layers panel (per-node controls)
  const layersPanel = factories.layersPanel(document.body, animationController);
  partial.layersPanel = layersPanel;
  inputHandler.setLayersPanel(layersPanel);

  // Left activity rail — the always-visible, discoverable entry point to the
  // otherwise keyboard-only panels. Each button fires the SAME command as its
  // shortcut (via inputHandler.getUiActions()), so behaviour never drifts.
  const ui = inputHandler.getUiActions();
  const railItems: ControlRailItem[] = [
    {
      id: 'help',
      title: 'Help & shortcuts',
      shortcut: 'H',
      icon: RAIL_ICONS.help,
      activate: () => ui.commands.toggleHelp(),
      openSelector: '#luxar-help-overlay',
    },
    {
      // Navigation: left-click cycles orbit → fly → ortho (reuses the exact V
      // shortcut command, incl. the input-context switch); right-click opens
      // the current mode's parameters. The icon mirrors the live control mode.
      id: 'nav',
      title: 'Navigation',
      shortcut: 'V',
      icon: RAIL_ICONS.navOrbit,
      activate: () => ui.commands.toggleControlMode(),
      render: (btn) => {
        const type = sceneManager.getControlType();
        if (btn.dataset.navMode === type) return;
        btn.dataset.navMode = type;
        const icon =
          type === 'fly'
            ? RAIL_ICONS.navFly
            : type === 'ortho'
              ? RAIL_ICONS.navOrtho
              : RAIL_ICONS.navOrbit;
        const svg = btn.querySelector('svg');
        if (svg) svg.outerHTML = icon;
        // `type` is a fixed enum (orbit|fly|ortho) — safe to interpolate.
        const modeLabel = type.charAt(0).toUpperCase() + type.slice(1);
        const next = nextControlType(type); // next mode a click would select
        btn.setAttribute(
          'aria-label',
          `Navigation: ${modeLabel} — click for ${next} (V), right-click for options`
        );
        // The button icon is cryptic on its own, so name the current mode in
        // the hover tooltip too: e.g. "Navigation · Orbit  V".
        const tip = btn.querySelector('.luxar-control-rail__tip');
        if (tip) tip.innerHTML = `Navigation · ${modeLabel}<kbd>V</kbd>`;
      },
      popover: {
        trigger: 'context',
        title: 'Navigation',
        build: (host) =>
          buildNavigationPopover(host, {
            settings: renderingControls.settings,
            sceneManager,
            animationController,
            saveSettings: () => renderingControls.saveSettings(),
            triggerAnimation: () => animationController.startAnimation(),
            setMode: (type) => ui.commands.setControlMode(type),
          }),
      },
    },
    {
      id: 'dims',
      title: 'Dimensions',
      shortcut: 'N',
      icon: RAIL_ICONS.dims,
      activate: () => ui.commands.toggleDimensionSliders(),
      openSelector: '.luxar-dimension-sliders',
    },
    {
      id: 'render',
      title: 'Rendering',
      shortcut: 'R',
      icon: RAIL_ICONS.render,
      activate: () => ui.commands.toggleRenderingControls(),
      isActive: () => renderingControls.isVisible(),
    },
    {
      id: 'layers',
      title: 'Layers',
      shortcut: 'L',
      icon: RAIL_ICONS.layers,
      activate: () => ui.panels.getLayersPanel()?.toggle(),
      isActive: () => layersPanel.isVisible(),
      // Grayed + non-clickable until the scene actually has layers (nodes with
      // layer=true). Refreshed on the 'luxar-layers-changed' event after load.
      disabled: () => layersPanel.layerState.count === 0,
    },
    {
      id: 'monitor',
      title: 'Data monitor',
      shortcut: 'M',
      icon: RAIL_ICONS.monitor,
      activate: () => ui.commands.cycleDataMonitor(),
      openSelector: '.luxar-data-monitor',
    },
    {
      id: 'data',
      title: 'Datasets',
      shortcut: 'O',
      icon: RAIL_ICONS.data,
      activate: () => window.dispatchEvent(new CustomEvent('open-dataset-browser')),
      openSelector: '.luxar-dataset-browser',
    },
    {
      id: 'recording',
      title: 'Recording',
      shortcut: 'T',
      icon: RAIL_ICONS.recording,
      activate: () => ui.panels.getRecordingPanel()?.toggle(),
      isActive: () => recordingPanel.isVisible(),
      separatorBefore: true,
    },
    {
      id: 'logs',
      title: 'Logs (console)',
      shortcut: 'Ctrl+L',
      icon: RAIL_ICONS.logs,
      activate: () => debugConsole.toggle(),
      isActive: () => debugConsole.getIsVisible(),
      separatorBefore: true,
    },
    {
      id: 'view',
      title: 'View options',
      icon: RAIL_ICONS.view,
      activate: () => {}, // unused — opens the flyout below
      separatorBefore: true,
      flyout: [
        {
          id: 'scalebar',
          title: 'Scale bar',
          shortcut: 'B',
          icon: RAIL_ICONS.scalebar,
          activate: () => ui.panels.getScaleBar()?.toggle(),
          openSelector: '.luxar-scale-bar',
        },
        {
          id: 'legend',
          title: 'Colormap legend',
          shortcut: 'J',
          icon: RAIL_ICONS.legend,
          activate: () => ui.panels.getColormapLegend()?.toggle(),
          openSelector: '.luxar-colormap-legend',
        },
        {
          id: 'overlays',
          title: 'Overlays',
          shortcut: 'U',
          icon: RAIL_ICONS.overlays,
          activate: () => ui.panels.getOverlayManager()?.toggle(),
          openSelector: '.luxar-overlay:not(.luxar-overlay--hidden)',
        },
        {
          id: 'cinematic',
          title: 'Cinematic mode',
          shortcut: 'C',
          icon: RAIL_ICONS.cinematic,
          activate: () => ui.commands.toggleCinematicMode(),
          isActive: () => renderingControls.settings.cinematicMode,
        },
      ],
    },
    {
      // Viewer-wide preferences (theme, and future settings) — kept off the top
      // level so the rail stays focused. Click opens the Settings popover.
      id: 'settings',
      title: 'Settings',
      icon: RAIL_ICONS.settings,
      activate: () => {}, // unused — opens the popover below
      popover: {
        trigger: 'click',
        title: 'Settings',
        build: (host) =>
          buildSettingsPopover(host, {
            triggerAnimation: () => animationController.startAnimation(),
          }),
      },
    },
    {
      // The gauge toggles the perf readout docked below (rail footer). Placed
      // just under the eye so the readout appears at the very bottom of the rail.
      // Right-click opens the adaptive-resolution (DPR) controls popover.
      id: 'perf',
      title: 'Performance',
      shortcut: 'P',
      icon: RAIL_ICONS.perf,
      activate: () => ui.commands.togglePerformanceStats(),
      isActive: () => performanceMonitor.visible,
      popover: {
        trigger: 'context',
        title: 'Performance',
        build: (host) =>
          buildPerformancePopover(host, {
            settings: renderingControls.settings,
            manager: adaptiveDPRManager,
            saveSettings: () => renderingControls.saveSettings(),
            triggerAnimation: () => animationController.startAnimation(),
          }),
      },
    },
  ];
  // Dock the perf readout as the rail's footer; the gauge above toggles it.
  const controlRail = new ControlRail(railItems, performanceMonitor.element);
  partial.controlRail = controlRail;

  // Start animation loop first to ensure background is rendered
  animationController.startAnimation();

  partial.sceneSrc = sceneSrc;
  return partial as InitPipelineResult;
}
