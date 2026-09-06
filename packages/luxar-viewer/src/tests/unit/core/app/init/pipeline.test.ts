// @vitest-environment jsdom
/**
 * Unit tests for core/app/init/pipeline.ts.
 *
 * The load-bearing invariant is the **partial-accumulator contract**:
 * the pipeline writes each subsystem into `partial` *before* any await
 * that could throw, so a mid-init failure still leaves a disposable
 * reference behind for LuxarApp.dispose() to clean up.
 *
 * The construction graph is too large to assert step-by-step, so this
 * suite focuses on:
 *   - partial-accumulator timing at each await boundary
 *   - happy-path returns the same object reference as `partial`
 *   - sceneSrc resolves from options.src (or default fallback)
 *   - factories.* are invoked, with overrides honored
 */

import { describe, it, expect, expectTypeOf, vi, beforeEach } from 'vitest';
import {
  runInitPipeline,
  type InitPipelineResult,
  type InitPipelinePorts,
} from '../../../../../core/app/init/pipeline';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import type { DimensionSlidersConfig } from '../../../../../input/input-handler/panel-capabilities';
import type { SliderConfig } from '../../../../../ui/dimension-sliders';

// Stub every heavy constructor at module level. Each one returns a
// minimal object that satisfies the pipeline's subsequent member access.
function makeSceneStub(opts: { initThrows?: boolean } = {}) {
  return {
    init: vi.fn().mockImplementation(async () => {
      if (opts.initThrows) throw new Error('sceneManager.init failed');
    }),
    controls: { kind: 'controls' },
    postProcessing: { kind: 'pp' },
    isWebGLContextLost: vi.fn().mockReturnValue(false),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    scene: { kind: 'scene' },
    // The scene-environment wiring (`environment-wiring.ts`) attaches its capture
    // runtime here; `environment` stays null so the per-frame tick is a no-op.
    attachEnvironmentRuntime: vi.fn(),
    environment: null,
  };
}
function makeAnimationStub() {
  return {
    setContextLostPredicate: vi.fn(),
    setIdleRestorePredicate: vi.fn(),
    setRenderSkipPredicate: vi.fn(),
    setPacingSuspendPredicate: vi.fn(),
    addPerFrameCallback: vi.fn(),
    setAdaptiveDPRManager: vi.fn(),
    setDensityGuardControl: vi.fn(),
    startAnimation: vi.fn(),
  };
}
function makeRenderingControlsStub() {
  return {
    setAnimationController: vi.fn(),
    setAdaptiveDPRManager: vi.fn(),
    setDensityGuardControl: vi.fn(),
  };
}
function makeRecordingPanelStub() {
  return {
    setPanelStateCallbacks: vi.fn(),
    setAdaptiveDPRManager: vi.fn(),
    setDensityGuardControl: vi.fn(),
    // The offline loop's LOD-quiescence predicate (#1695) is injected here
    // too, since reaching `getSceneLoader` from the panel itself would pull
    // the whole data/cache stack into its module graph.
    setLODSettledProvider: vi.fn(),
    // The sound layer's capture tap ("Include Audio").
    setAudioCapture: vi.fn(),
    // The two capture flags the pipeline's injected predicates read.
    // `isCurrentlyRecording()` covers BOTH capture kinds; only the
    // narrower `isLoopRenderSuppressed()` may gate the render skip.
    isCurrentlyRecording: vi.fn(() => false),
    isLoopRenderSuppressed: vi.fn(() => false),
  };
}
function makeLayersPanelStub() {
  // `setAudioPort` is the sound layer's late-bound port for the sound rows.
  return { kind: 'layers', setAudioPort: vi.fn() };
}
function makeInputHandlerStub() {
  return {
    init: vi.fn(),
    setRenderingControls: vi.fn(),
    setRecordingPanel: vi.fn(),
    setLayersPanel: vi.fn(),
    setControlRail: vi.fn(),
    // The control rail reads this to wire its buttons to the same commands the
    // keyboard uses; the closures are only invoked on click (never in tests).
    getUiActions: vi.fn(() => ({ commands: {}, panels: {} })),
    // Stands in for the live key-binding registry: exactly one action resolves,
    // and to a label the real config never produces, so a test can tell "the
    // registry's answer reached the rail" apart from both "some truthy stub
    // did" and "the production letter was baked in". Everything else is
    // unbound, which is what an un-init'd registry answers for every action.
    getShortcutLabel: vi.fn((actionId: string) =>
      actionId === KeyAction.toggleHelp ? '?' : undefined
    ),
  };
}

vi.mock('../../../../../scene/scene-manager', () => ({
  SceneManager: vi.fn(),
}));
vi.mock('../../../../../scene/animation/animation-controller', () => ({
  AnimationController: vi.fn(),
}));
vi.mock('../../../../../ui/performance-monitor', () => ({
  PerformanceMonitor: vi.fn().mockImplementation(() => ({ kind: 'perf-monitor' })),
}));
// Mock the rail so the pipeline test doesn't build a real one (which would
// attach document listeners / a rAF loop that outlive the test).
vi.mock('../../../../../ui/control-rail', () => ({
  ControlRail: vi.fn().mockImplementation(() => ({ setCollapsed: vi.fn(), dispose: vi.fn() })),
  RAIL_ICONS: {},
}));
vi.mock('../../../../../ui/debug-console', () => ({
  DebugConsole: vi.fn().mockImplementation(() => ({ kind: 'debug-console' })),
}));
vi.mock('../../../../../rendering/adaptive-dpr-manager', () => ({
  AdaptiveDPRManager: vi.fn().mockImplementation(() => ({
    setRenderer: vi.fn(),
    setOnDPRChangeCallback: vi.fn(),
    getNativeDPR: vi.fn(() => 2),
    setLoadActivityPredicate: vi.fn(),
    notifyContentChanged: vi.fn(),
    notifyPaused: vi.fn(),
  })),
}));
vi.mock('../../../../../ui/resolution-indicator', () => ({
  ResolutionIndicator: vi.fn().mockImplementation(() => ({
    setTargetFPS: vi.fn(),
    show: vi.fn(),
    reset: vi.fn(),
  })),
}));
vi.mock('../../../../../input', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../../../input')>();
  // Spread rather than enumerate: only the constructor needs replacing, and an
  // explicit export list turns the day some module in the pipeline's import
  // graph value-imports another facade symbol into an opaque
  // "does not provide an export" with no type error pointing here.
  return {
    ...actual,
    InputHandler: vi.fn(),
  };
});
vi.mock('../../../../../ui/dimension-sliders', () => ({
  DimensionSliders: vi.fn(),
}));
vi.mock('../../../../../ui/rendering-controls', () => ({
  RenderingControls: vi.fn(),
}));
vi.mock('../../../../../ui/recording-panel', () => ({
  RecordingPanel: vi.fn(),
}));
vi.mock('../../../../../ui/layers', () => ({
  LayersPanel: vi.fn(),
}));
vi.mock('../../../../../ui/data-monitor-manager', () => ({
  DataMonitorManager: {
    getInstance: vi.fn().mockReturnValue({
      hasMonitor: vi.fn().mockReturnValue(true),
      createMonitor: vi.fn(),
      getMonitor: vi.fn().mockReturnValue(null),
    }),
  },
}));
vi.mock('../../../../../data/scene-loader-manager', () => ({
  SceneLoaderManager: {
    getInstance: vi.fn().mockReturnValue({
      setMonitorFactory: vi.fn(),
      setLODGroupRegistryFactory: vi.fn(),
      setRequestRender: vi.fn(),
      setKTX2TextureDecoder: vi.fn(),
      setRefinementDensityProvider: vi.fn(),
      isAnyLoadPassInProgress: vi.fn().mockReturnValue(false),
    }),
  },
  getSceneLoader: vi.fn().mockReturnValue(null),
}));
vi.mock('../../../../../utils/cross-layer/notifier', () => ({
  notifier: { error: vi.fn() },
}));
// The coordinator is module-scoped live authority; mocked so the warm-up
// call is observable (and so no real SortWorker/WASM is spawned here).
vi.mock('../../../../../rendering/depth-sort-coordinator', () => ({
  configureDepthSort: vi.fn(),
  setDepthSortEnabled: vi.fn(),
  warmUpDepthSortWorker: vi.fn(),
  evaluateDepthSortPerFrame: vi.fn(),
}));

import { InputHandler, KeyAction } from '../../../../../input';
import { ControlRail } from '../../../../../ui/control-rail';
import { getSceneLoader, SceneLoaderManager } from '../../../../../data/scene-loader-manager';
import {
  configureDepthSort,
  setDepthSortEnabled,
  warmUpDepthSortWorker,
} from '../../../../../rendering/depth-sort-coordinator';
import {
  DEFAULT_MAX_PIXEL_RATIO,
  setMaxPixelRatioCap,
} from '../../../../../rendering/pixel-ratio-cap';
import { setNativeDPR } from '../../../../helpers/device-pixel-ratio';

function makePorts(): InitPipelinePorts {
  const canvas = document.createElement('canvas');
  return {
    options: {
      canvas,
      src: 'http://example.com/scene.zarr',
      debug: false,
    },
    events: new EventGroup(),
    getPanelVisibilityStates: vi.fn().mockReturnValue(new Map()),
    restorePanelVisibilityStates: vi.fn(),
    emitEmbedderEvent: vi.fn(),
  };
}

function makeFactoryOverrides(opts: { sceneInitThrows?: boolean } = {}) {
  const scene = makeSceneStub({ initThrows: opts.sceneInitThrows });
  return {
    factories: {
      sceneManager: vi.fn().mockReturnValue(scene),
      animationController: vi.fn().mockReturnValue(makeAnimationStub()),
      renderingControls: vi.fn().mockReturnValue(makeRenderingControlsStub()),
      recordingPanel: vi.fn().mockReturnValue(makeRecordingPanelStub()),
      layersPanel: vi.fn().mockReturnValue(makeLayersPanelStub()),
    },
    sceneStub: scene,
  };
}

describe('runInitPipeline', () => {
  it('keeps the injected dimension-slider config identical to the UI config', () => {
    expectTypeOf<DimensionSlidersConfig>().toEqualTypeOf<SliderConfig>();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    (InputHandler as unknown as ReturnType<typeof vi.fn>).mockImplementation(() =>
      makeInputHandlerStub()
    );
  });

  describe('partial-accumulator contract (load-bearing)', () => {
    it('writes partial.sceneManager BEFORE awaiting sceneManager.init (preserves dispose-on-init-throw)', async () => {
      const { factories, sceneStub } = makeFactoryOverrides({ sceneInitThrows: true });
      const ports = makePorts();
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await expect(runInitPipeline(ports, partial)).rejects.toThrow(/sceneManager\.init failed/);

      // The throw came from sceneStub.init — but sceneStub itself
      // was written to partial BEFORE the await, so dispose can find it.
      expect(partial.sceneManager).toBe(sceneStub as never);
    });

    it('writes partial.animationController + downstream fields when sceneManager.init succeeds', async () => {
      // core.md W7 strengthening + G18 (partial.sceneManager identity):
      // previously 10 sequential `toBeDefined`s. A mutation that swapped
      // two fields (e.g. `partial.performanceMonitor = animationController`)
      // would survive `toBeDefined`. We now assert IDENTITY against the
      // factory return values (so wrong-field-assigned regressions die),
      // and confirm sceneManager is the EXACT object the factory returned.
      const { factories, sceneStub } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await runInitPipeline(ports, partial);

      // G18: partial.sceneManager IS the factory's return value (not a
      // copy / replacement) — dispose() reads this exact reference.
      expect(partial.sceneManager).toBe(sceneStub as never);

      // Identity checks on every factory-built field. Mutating any of
      // these wires would now fail (e.g. swapping animationController
      // and renderingControls).
      expect(partial.animationController).toBe(factories.animationController.mock.results[0].value);
      expect(partial.renderingControls).toBe(factories.renderingControls.mock.results[0].value);
      expect(partial.recordingPanel).toBe(factories.recordingPanel.mock.results[0].value);
      expect(partial.layersPanel).toBe(factories.layersPanel.mock.results[0].value);

      // For the inline-constructed (non-factory) fields, type + shape
      // are the strongest local invariants we can assert without
      // pulling in the real constructors.
      expect(partial.performanceMonitor).toMatchObject({ kind: 'perf-monitor' });
      expect(partial.debugConsole).toMatchObject({ kind: 'debug-console' });
      expect(partial.adaptiveDPRManager).toBeDefined();
      expect(partial.resolutionIndicator).toBeDefined();
      expect(partial.inputHandler).toBeDefined();

      // sceneSrc is a string (P5 boundary), and EQUALS the supplied src.
      expect(typeof partial.sceneSrc).toBe('string');
      expect(partial.sceneSrc).toBe('http://example.com/scene.zarr');
    });

    it('normalizes the DPR-change callback to percent-of-CEILING before showing the indicator', async () => {
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await runInitPipeline(ports, partial);

      const manager = partial.adaptiveDPRManager as unknown as {
        setOnDPRChangeCallback: ReturnType<typeof vi.fn>;
      };
      const indicator = partial.resolutionIndicator as unknown as {
        show: ReturnType<typeof vi.fn>;
        reset: ReturnType<typeof vi.fn>;
      };
      const callback = manager.setOnDPRChangeCallback.mock.calls[0][0] as (
        dpr: number,
        isReducedResolution: boolean
      ) => void;

      // Reduced at DPR 1.8 with a 2.0 ceiling → indicator shows 0.9
      // (percent-of-full-quality), NOT the absolute DPR (the retina
      // "180%" bug).
      //
      // Against the CEILING, not the display: with high DPR disallowed
      // the ceiling is 1.0, so dividing by a native 2.0 would report a
      // permanent "50%" and pop the toast on every scene load.
      setMaxPixelRatioCap(Infinity);
      const restoreNative = setNativeDPR(2);
      try {
        callback(1.8, true);
        expect(indicator.show).toHaveBeenCalledWith(0.9);

        // Back at the ceiling → reset branch, no further show.
        callback(2, false);
        expect(indicator.reset).toHaveBeenCalledTimes(1);
        expect(indicator.show).toHaveBeenCalledTimes(1);

        // Capped at 1.0: a DPR of 0.9 is 90% of full quality, not 45%.
        setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO);
        callback(0.9, true);
        expect(indicator.show).toHaveBeenLastCalledWith(0.9);
      } finally {
        restoreNative();
        setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO);
      }
    });

    it('returns the SAME object reference as `partial` (happy path)', async () => {
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      const result = await runInitPipeline(ports, partial);
      // The pipeline returns `partial as InitPipelineResult` — same ref.
      expect(result).toBe(partial as InitPipelineResult);
    });
  });

  describe('factory dispatch', () => {
    it('calls every factory exactly once', async () => {
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;

      await runInitPipeline(ports, {});

      expect(factories.sceneManager).toHaveBeenCalledOnce();
      expect(factories.animationController).toHaveBeenCalledOnce();
      expect(factories.renderingControls).toHaveBeenCalledOnce();
      expect(factories.recordingPanel).toHaveBeenCalledOnce();
      expect(factories.layersPanel).toHaveBeenCalledOnce();
    });

    it('installs the renderer-owned KTX2 decoder on the loader manager', async () => {
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;

      await runInitPipeline(ports, {});

      expect(SceneLoaderManager.getInstance().setKTX2TextureDecoder).toHaveBeenCalledWith(
        expect.any(Function)
      );
    });

    it('forwards canvas + renderer flags from options to sceneManager.init', async () => {
      const { factories, sceneStub } = makeFactoryOverrides();
      const canvas = document.createElement('canvas');
      const ports: InitPipelinePorts = {
        options: {
          canvas,
          src: 'http://example.com/scene.zarr',
          debug: true,
          renderer: 'webgpu',
          webgpuForceWebGL: false,
          perfTimestamp: true,
          blendWarmup: false,
          factories: factories as never,
        },
        events: new EventGroup(),
        getPanelVisibilityStates: vi.fn().mockReturnValue(new Map()),
        restorePanelVisibilityStates: vi.fn(),
        emitEmbedderEvent: vi.fn(),
      };

      await runInitPipeline(ports, {});

      expect(sceneStub.init).toHaveBeenCalledExactlyOnceWith({
        canvas,
        debug: true,
        renderer: 'webgpu',
        webgpuForceWebGL: false,
        perfTimestamp: true,
        blendWarmup: false,
      });
    });
  });

  describe('sceneSrc resolution', () => {
    it('uses options.src when present', async () => {
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.src = 'http://example.com/custom.zarr';
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await runInitPipeline(ports, partial);

      expect(partial.sceneSrc).toBe('http://example.com/custom.zarr');
    });

    it('falls back to config.defaultZarrPath when options.src is undefined (no undefined leak)', async () => {
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.src = undefined;
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await runInitPipeline(ports, partial);

      // The exact default isn't important (it's '' in the default
      // config). What matters is that sceneSrc resolved to a string,
      // not undefined — the orchestrator's routing reads it directly.
      expect(partial.sceneSrc).toBeTypeOf('string');
    });
  });

  describe('animation loop kick', () => {
    it('startAnimation fires before the pipeline returns', async () => {
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await runInitPipeline(ports, partial);

      expect(partial.animationController?.startAnimation).toHaveBeenCalled();
    });
  });

  describe('depth-sort warm-up', () => {
    it('warms up the SortWorker during init, AFTER the enable + config wiring', async () => {
      // The warm-up is the whole point of starting the worker here: spawned
      // lazily by the first order-dependent commit it races the scene decode
      // for the main thread and misses its init deadline at a few million
      // elements, which used to disable sorting for the session.
      //
      // Order is load-bearing, not cosmetic: `warmUpDepthSortWorker` early-outs
      // on the `depthSortEnabled` flag, so warming up before
      // `setDepthSortEnabled` would spawn a worker (and load WASM) for a
      // `?depthSort=0` session, and before `configureDepthSort` a starved
      // retry's self-wake would have no `requestRender` to call.
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;

      await runInitPipeline(ports, {});

      expect(warmUpDepthSortWorker).toHaveBeenCalledTimes(1);
      const warmUpOrder = vi.mocked(warmUpDepthSortWorker).mock.invocationCallOrder[0];
      expect(vi.mocked(setDepthSortEnabled).mock.invocationCallOrder[0]).toBeLessThan(warmUpOrder);
      expect(vi.mocked(configureDepthSort).mock.invocationCallOrder[0]).toBeLessThan(warmUpOrder);
    });
  });

  describe('recording-panel predicate wiring', () => {
    it('the render-skip predicate follows the loop-render-suppression flag, never real-time recording', async () => {
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;

      await runInitPipeline(ports, {});

      const animation = factories.animationController.mock.results[0].value as {
        setRenderSkipPredicate: ReturnType<typeof vi.fn>;
      };
      const panel = factories.recordingPanel.mock.results[0].value as {
        isCurrentlyRecording: ReturnType<typeof vi.fn>;
        isLoopRenderSuppressed: ReturnType<typeof vi.fn>;
      };
      expect(animation.setRenderSkipPredicate).toHaveBeenCalledTimes(1);
      const predicate = animation.setRenderSkipPredicate.mock.calls[0][0] as () => boolean;

      expect(predicate()).toBe(false);

      // Real-time MediaRecorder capture: `isCurrentlyRecording()` is true
      // for it too, and its video IS the canvas the loop paints — keying
      // the skip off that flag would record an empty video.
      panel.isCurrentlyRecording.mockReturnValue(true);
      expect(predicate()).toBe(false);

      // Offline (frame-by-frame) capture renders its own pipeline pass per
      // frame, so the loop's render is discarded work — and during an EXR
      // sequence it paints a blown-out frame under the translucent overlay.
      // That is exactly what `isLoopRenderSuppressed()` reports, and it is
      // dropped before the capture's teardown awaits its driver abort.
      panel.isLoopRenderSuppressed.mockReturnValue(true);
      expect(predicate()).toBe(true);
    });

    it('the pacing-suspend predicate follows the BROAD recording flag, not loop-render suppression', async () => {
      // Opposite polarity to the render-skip predicate above, and that is
      // the point: frame pacing must be off for BOTH capture families,
      // because both depend on the loop's untouched cadence (the real-time
      // MediaRecorder path records the canvas this loop paints, so a paced
      // gap is a dropped frame in the video; the offline capture drives its
      // own rAF cadence with one-shot per-frame orbit callbacks). Keying it
      // on the narrow `isLoopRenderSuppressed()` instead would silently drop
      // a frame per gap from an exported video, and nothing else in the
      // suite would notice.
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;

      await runInitPipeline(ports, {});

      const animation = factories.animationController.mock.results[0].value as {
        setPacingSuspendPredicate: ReturnType<typeof vi.fn>;
      };
      const panel = factories.recordingPanel.mock.results[0].value as {
        isCurrentlyRecording: ReturnType<typeof vi.fn>;
        isLoopRenderSuppressed: ReturnType<typeof vi.fn>;
      };
      expect(animation.setPacingSuspendPredicate).toHaveBeenCalledTimes(1);
      const predicate = animation.setPacingSuspendPredicate.mock.calls[0][0] as () => boolean;

      expect(predicate()).toBe(false);

      panel.isCurrentlyRecording.mockReturnValue(true);
      expect(predicate()).toBe(true);

      // The narrow flag alone must NOT suspend pacing: it is set only for
      // an offline capture, which already reports `isCurrentlyRecording()`.
      panel.isCurrentlyRecording.mockReturnValue(false);
      panel.isLoopRenderSuppressed.mockReturnValue(true);
      expect(predicate()).toBe(false);
    });

    it('the LOD-settle provider delegates to the registry, and answers null when there is nothing to wait for', async () => {
      // The only integration seam of the #1695 fix: the offline capture drains
      // on this provider, and the panel cannot build it itself (reaching the
      // registry from `ui/recording-panel` would pull the whole data/cache
      // stack into its module graph).
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;

      await runInitPipeline(ports, {});

      const panel = factories.recordingPanel.mock.results[0].value as {
        setLODSettledProvider: ReturnType<typeof vi.fn>;
      };
      expect(panel.setLODSettledProvider).toHaveBeenCalledTimes(1);
      const provider = panel.setLODSettledProvider.mock.calls[0][0] as () => boolean | null;

      try {
        // Read LIVE — the scene loader is created after the panel, so at wiring
        // time there is none. That is `null` ("nothing here could ever need
        // waiting for"), NOT `true` ("settled"): a `true` still costs the
        // capture a mandatory selector-catch-up rAF on every exported frame.
        expect(provider()).toBeNull();

        const isCaptureQuiescent = vi.fn(() => false);
        const size = vi.fn(() => 0);
        vi.mocked(getSceneLoader).mockReturnValue({
          lodGroupRegistry: { size, isCaptureQuiescent },
        } as never);

        // A registry with no lod_group registered — a plain points/lines scene
        // — is equally nothing to wait for, and the predicate is not consulted.
        expect(provider()).toBeNull();
        expect(isCaptureQuiescent).not.toHaveBeenCalled();

        // With entries registered the answer is the registry's own, both ways.
        size.mockReturnValue(2);
        expect(provider()).toBe(false);
        isCaptureQuiescent.mockReturnValue(true);
        expect(provider()).toBe(true);
        expect(isCaptureQuiescent).toHaveBeenCalledTimes(2);
      } finally {
        // The module mock is shared; restore the suite-wide default.
        vi.mocked(getSceneLoader).mockReturnValue(null as never);
      }
    });
  });

  describe('context-loss listener wiring', () => {
    it("wires the SceneManager 'change' event to startAnimation (repaint after idle-time context restore)", async () => {
      // The context-restore path ends with SceneManager dispatching
      // 'change' ("trigger a render"). Without a subscriber, a context
      // restored while the rAF loop is idle-paused rebuilds + clears the
      // canvas and never paints — blank viewer until the next input.
      const { factories, sceneStub } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await runInitPipeline(ports, partial);

      const call = sceneStub.addEventListener.mock.calls.find((c: unknown[]) => c[0] === 'change');
      expect(call).toBeDefined();

      const animation = partial.animationController as unknown as {
        startAnimation: ReturnType<typeof vi.fn>;
      };
      animation.startAnimation.mockClear();
      (call![1] as () => void)();
      expect(animation.startAnimation).toHaveBeenCalled();
    });

    it('registers webgl-context-restored + webgpu-device-lost via ports.events', async () => {
      const { factories, sceneStub } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;

      await runInitPipeline(ports, {});

      const events = sceneStub.addEventListener.mock.calls.map((c) => c[0]);
      expect(events).toContain('webgl-context-restored');
      expect(events).toContain('webgpu-device-lost');
    });

    it('webgpu-device-lost latches the shared context-lost predicate and pauses the DPR manager', async () => {
      const { factories, sceneStub } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await runInitPipeline(ports, partial);

      const animation = factories.animationController.mock.results[0].value as {
        setContextLostPredicate: ReturnType<typeof vi.fn>;
      };
      const predicate = animation.setContextLostPredicate.mock.calls[0][0] as () => boolean;
      sceneStub.isWebGLContextLost = vi.fn(() => false);
      expect(predicate()).toBe(false);

      // Fire the registered webgpu-device-lost handler.
      const handler = sceneStub.addEventListener.mock.calls.find(
        (c: unknown[]) => c[0] === 'webgpu-device-lost'
      )![1] as (event: object) => void;
      handler({ reason: 'destroyed' });

      // The predicate is latched even though isWebGLContextLost stays
      // false (it is hard-false under WebGPU), and the DPR manager is
      // paused so cheap no-op frames can't drive scale-ups.
      expect(predicate()).toBe(true);
      const manager = partial.adaptiveDPRManager as unknown as {
        notifyPaused: ReturnType<typeof vi.fn>;
      };
      expect(manager.notifyPaused).toHaveBeenCalled();
    });
  });

  describe('control-rail shortcut wiring', () => {
    it('labels rail buttons from the live key-binding registry, and only after inputHandler.init()', async () => {
      // Every rail button's tooltip/aria-label and <kbd> chip comes from the
      // LIVE registry (`inputHandler.getShortcutLabel`), not a letter baked
      // into build-rail-items, so a rebound key shows up on screen. Two ways
      // that seam breaks with the buttons still working and every chip
      // silently blank: the pipeline stops supplying `shortcutForAction` (the
      // build-rail-items suite injects its own, so it stays green and proves
      // only that the rail CONSUMES the port), or the rail is built before
      // `init()` has registered the bindings, so the registry is empty and
      // every action resolves `undefined`.
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await runInitPipeline(ports, partial);

      const inputHandler = partial.inputHandler as unknown as {
        init: ReturnType<typeof vi.fn>;
        getShortcutLabel: ReturnType<typeof vi.fn>;
      };

      // Asked with the REAL action id (a typo'd/renamed id resolves nothing).
      expect(inputHandler.getShortcutLabel).toHaveBeenCalledWith(KeyAction.toggleHelp);

      // And the registry's ANSWER is what the item carries — read off the real
      // buildRailItems output the pipeline handed to the (mocked) rail. Assert
      // the item exists first, so renaming it reads as a missing item rather
      // than as a broken label.
      const railItems = vi.mocked(ControlRail).mock.calls[0][0];
      const helpItem = railItems.find((item) => item.id === 'help');
      expect(helpItem).toBeDefined();
      expect(helpItem?.shortcut).toBe('?');

      // Bindings exist only after init(), so the rail must be built later.
      expect(inputHandler.init.mock.invocationCallOrder[0]).toBeLessThan(
        inputHandler.getShortcutLabel.mock.invocationCallOrder[0]
      );
    });

    it('hands the rail it built to inputHandler.setControlRail', async () => {
      // The rail is also an input-handler collaborator: a handled keydown
      // notifies it, and Escape closes its overlay through PanelCoordinator.
      // Passing anything but this instance (or nothing) silently kills both
      // routes while the buttons keep working — so assert IDENTITY, not that
      // the setter was merely called.
      const { factories } = makeFactoryOverrides();
      const ports = makePorts();
      ports.options.factories = factories as never;
      const partial: Partial<InitPipelineResult> = {};

      await runInitPipeline(ports, partial);

      const rail = vi.mocked(ControlRail).mock.results[0].value;
      const inputHandler = partial.inputHandler as unknown as {
        setControlRail: ReturnType<typeof vi.fn>;
      };
      expect(inputHandler.setControlRail.mock.calls.length).toBe(1);
      expect(inputHandler.setControlRail.mock.calls[0][0]).toBe(rail);
      // …and the app keeps the very same instance for its dispose path.
      expect(partial.controlRail).toBe(rail);
    });
  });
});
