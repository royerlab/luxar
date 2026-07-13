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

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  runInitPipeline,
  type InitPipelineResult,
  type InitPipelinePorts,
} from '../../../../../core/app/init/pipeline';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';

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
  };
}
function makeAnimationStub() {
  return {
    setContextLostPredicate: vi.fn(),
    setIdleRestorePredicate: vi.fn(),
    addPerFrameCallback: vi.fn(),
    setAdaptiveDPRManager: vi.fn(),
    startAnimation: vi.fn(),
  };
}
function makeRenderingControlsStub() {
  return {
    setAnimationController: vi.fn(),
    setAdaptiveDPRManager: vi.fn(),
  };
}
function makeRecordingPanelStub() {
  return {
    setPanelStateCallbacks: vi.fn(),
    setAdaptiveDPRManager: vi.fn(),
  };
}
function makeLayersPanelStub() {
  return { kind: 'layers' };
}
function makeInputHandlerStub() {
  return {
    init: vi.fn(),
    setRenderingControls: vi.fn(),
    setRecordingPanel: vi.fn(),
    setLayersPanel: vi.fn(),
    // The control rail reads this to wire its buttons to the same commands the
    // keyboard uses; the closures are only invoked on click (never in tests).
    getUiActions: vi.fn(() => ({ commands: {}, panels: {} })),
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
vi.mock('../../../../../input/input-handler', () => ({
  InputHandler: vi.fn(),
}));
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
    }),
  },
  getSceneLoader: vi.fn().mockReturnValue(null),
}));
vi.mock('../../../../../utils/cross-layer/notifier', () => ({
  notifier: { error: vi.fn() },
}));

import { InputHandler } from '../../../../../input/input-handler';

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

    it('normalizes the DPR-change callback to percent-of-native before showing the indicator', async () => {
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

      // Reduced at DPR 1.8 on a native-2 display → indicator shows 0.9
      // (percent-of-native), NOT the absolute DPR (the retina "180%" bug).
      callback(1.8, true);
      expect(indicator.show).toHaveBeenCalledWith(0.9);

      // Back at native → reset branch, no further show.
      callback(2, false);
      expect(indicator.reset).toHaveBeenCalledTimes(1);
      expect(indicator.show).toHaveBeenCalledTimes(1);
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
          factories: factories as never,
        },
        events: new EventGroup(),
        getPanelVisibilityStates: vi.fn().mockReturnValue(new Map()),
        restorePanelVisibilityStates: vi.fn(),
      };

      await runInitPipeline(ports, {});

      expect(sceneStub.init).toHaveBeenCalledExactlyOnceWith({
        canvas,
        debug: true,
        renderer: 'webgpu',
        webgpuForceWebGL: false,
        perfTimestamp: true,
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
});
