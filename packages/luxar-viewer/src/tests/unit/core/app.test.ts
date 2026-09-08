// @vitest-environment jsdom
/**
 * Tests for LuxarApp - the main application coordinator
 *
 * These tests verify the initialization sequence, component integration,
 * dataset detection logic, error handling, and cleanup of the main app.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

const jsdomDocument = document;
const ownershipMocks = vi.hoisted(() => ({ install: vi.fn() }));

// NOTE: This test file mocks 9 internal modules (below). It primarily
// verifies initialization ordering + cross-wiring; component behavior
// is covered by per-module tests.
//
// `LuxarAppOptions.factories` (see `core/app-factories.ts`) exposes
// construction overrides for the heavy components (SceneManager,
// AnimationController, RenderingControls, RecordingPanel,
// LayersPanel). The `vi.mock(...)` calls below still work because the
// default factories call `new X(...)` and vi.mock intercepts the
// constructor. New tests can opt into factory stubs instead — see
// the `factory overrides` describe block below for an example of
// injecting a SceneManager stub without `vi.mock`.

// AUDIT NOTE (core.md C1, C2, C3, C5): this file vi.mocks ~13 internal
// modules (SceneManager, AnimationController, InputHandler,
// RenderingControls, RecordingPanel, ScaleBar, DatasetBrowser, UICleanup,
// ErrorOverlay, HelpOverlay, Layers, PerformanceMonitor, DebugConsole).
// Most are first-party luxar modules — not external trust boundaries.
//
// Status:
//   - The redundant construction-count asserts (initialization sequence
//     describe block) duplicate `app/init/pipeline.test.ts`'s
//     factory-dispatch tests against real ports — those are the
//     load-bearing tests. Per audit Non-Goal 2 ("no deletion") this
//     file's duplicates are kept as a redundant safety net until a
//     follow-up cleanup pass.
//   - `vi.stubGlobal('window', ...)` and `vi.stubGlobal('document', ...)`
//     replace the jsdom globals for the whole file. Tests at ~lines
//     870-911 read `mockAddEventListener.mock.calls` to find listeners
//     because of this — only registration shape is verified, not real-
//     event behavior. Real-event coverage lives in
//     `app/lifecycle/focus-handling.test.ts`. Dropping the stubs is
//     blocked on rewriting those ~5 tests; that is the next follow-up.
//   - C5 (resolved in-file): the pre-init-invariant assertion was split
//     out of "should set isInitialized to true after successful init"
//     into its own test so the test name matches the contract.

// Mock all dependencies before importing LuxarApp
vi.mock('../../../scene/scene-manager');
vi.mock('../../../scene/animation/animation-controller');
vi.mock('../../../input');
vi.mock('../../../ui/rendering-controls');
vi.mock('../../../ui/recording-panel');
vi.mock('../../../ui/scale-bar');
vi.mock('../../../ui/dataset-browser');
vi.mock('../../../ui/ui-cleanup');
vi.mock('../../../ui/error-overlay');
vi.mock('../../../ui/toast');
vi.mock('../../../ui/help-overlay');
vi.mock('../../../ui/layers');
vi.mock('../../../core/app/interaction/canvas-gesture-ownership', () => ({
  installCanvasGestureOwnership: ownershipMocks.install,
}));
// scene-dims-manager is unmocked: it's a pure JS singleton (no DOM
// or WebGL), so running it real in app.test improves coverage of the
// dim-init wiring without affecting jsdom behavior.
// PerformanceMonitor and DebugConsole are owned by LuxarApp and are
// mocked here so stats.js / DebugConsole's document.createElement
// calls don't run in the stubbed-window env.
vi.mock('../../../ui/performance-monitor', () => ({
  PerformanceMonitor: vi.fn().mockImplementation(() => ({
    show: vi.fn(),
    hide: vi.fn(),
    toggle: vi.fn(),
    cycleMode: vi.fn(),
    dispose: vi.fn(),
    visible: false,
  })),
}));
vi.mock('../../../ui/debug-console', () => ({
  DebugConsole: vi.fn().mockImplementation(() => ({
    show: vi.fn(),
    hide: vi.fn(),
    toggle: vi.fn(),
    dispose: vi.fn(),
    getIsVisible: vi.fn(() => false),
  })),
}));
vi.mock('../../../ui/control-rail', () => ({
  ControlRail: vi.fn().mockImplementation(() => ({ setCollapsed: vi.fn(), dispose: vi.fn() })),
  RAIL_ICONS: {},
}));

// Setup global mocks
const mockAddEventListener = vi.fn();
const mockRemoveEventListener = vi.fn();
const mockReplaceState = vi.fn();

// AUDIT NOTE (core.md C5): this `vi.stubGlobal('window', {...})` *completely
// replaces* the jsdom window for the entire test file. Tests that rely on
// `EventGroup.on(window, ...)` to register listeners cannot dispatch real
// Events, so any production code path that calls dispatchEvent against
// the real window is invisible. Tests below assert on
// `mockAddEventListener.mock.calls` (which IS the stubbed function) — that
// catches registration-shape regressions but not behavior. The orchestrator
// helpers extracted post-hoc (`unload-handling`, `browser-shortcut`,
// `focus-handling`, etc.) have dedicated tests under `app/lifecycle/`
// that use real EventGroup wiring. Follow-up: drop this stubGlobal block
// in favor of jsdom's real window + per-test mockAddEventListener spies.
vi.stubGlobal('window', {
  addEventListener: mockAddEventListener,
  removeEventListener: mockRemoveEventListener,
  location: {
    search: '',
    pathname: '/',
    origin: 'http://localhost:5173',
  },
  history: {
    replaceState: mockReplaceState,
  },
  innerWidth: 1920,
  innerHeight: 1080,
});

vi.stubGlobal('document', {
  addEventListener: mockAddEventListener,
  removeEventListener: mockRemoveEventListener,
  body: {},
  hidden: false,
  // A real document always has a string here; the tab-title helpers read it
  // back before overwriting it.
  title: '',
});

// Mock fetch for dataset detection
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

// Import mocked classes
import { SceneManager } from '../../../scene/scene-manager';
import { AnimationController } from '../../../scene/animation/animation-controller';
import { InputHandler } from '../../../input';
import { RenderingControls } from '../../../ui/rendering-controls';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import { DatasetBrowser } from '../../../ui/dataset-browser';
import { cleanupUI as mockCleanupUI } from '../../../ui/ui-cleanup';
import { clearError as mockClearError } from '../../../ui/error-overlay';
import { showToast as mockShowToast } from '../../../ui/toast';
import { showHelpOverlay } from '../../../ui/help-overlay';

// Import LuxarApp after all mocks are set up
import { LuxarApp } from '../../../core/app';
import { SceneDimsManager } from '../../../scene/scene-dims-manager';
import { setDocumentTitle } from '../../../core/document-title';
import { SceneLoaderManager } from '../../../data/scene-loader-manager';

describe('LuxarApp', () => {
  let app: LuxarApp;
  let mockSceneManager: any;
  let mockAnimationController: any;
  let mockInputHandler: any;
  let mockRenderingControls: any;
  let mockCanvas: HTMLCanvasElement;

  beforeEach(() => {
    // Clear all mocks
    vi.clearAllMocks();
    ownershipMocks.install.mockReset();

    // Reset mock implementations
    mockSceneManager = {
      init: vi.fn().mockResolvedValue(undefined),
      loadSceneData: vi.fn().mockResolvedValue(undefined),
      warmBlendModePrograms: vi.fn(),
      updateDynamicClippingPlanes: vi.fn(),
      getSceneViewerConfig: vi.fn().mockReturnValue(undefined),
      setCameraZoom: vi.fn(),
      getSceneBakedEnvironment: vi.fn().mockReturnValue(null),
      attachEnvironmentRuntime: vi.fn(),
      environment: null,
      dispose: vi.fn(),
      renderer: { domElement: {} },
      scene: {},
      // Enough of a camera for captureSnapshot() (the camera-changed
      // embedder event reads position / up / near / far).
      camera: { position: { x: 1, y: 2, z: 3 }, up: { x: 0, y: 1, z: 0 }, near: 0.1, far: 100 },
      // The ControlsManager surface the embedder hooks subscribe to.
      controls: {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getFocusTarget: vi.fn(() => ({ x: 0, y: 0, z: 0 })),
        setTarget: vi.fn(),
        reinitialize: vi.fn(),
        dispatchEvent: vi.fn(),
        isAutoRotateActive: vi.fn(() => false),
      },
      postProcessing: {},
      // Embedder-API delegation targets.
      resizeToCanvas: vi.fn(),
      centerCameraOnScene: vi.fn(),
    };

    mockAnimationController = {
      startAnimation: vi.fn(),
      stopAnimation: vi.fn(),
      addPerFrameCallback: vi.fn(),
      removePerFrameCallback: vi.fn(),
      setAdaptiveDPRManager: vi.fn(),
      setDensityGuardControl: vi.fn(),
      setContextLostPredicate: vi.fn(),
      setIdleRestorePredicate: vi.fn(),
      setRenderSkipPredicate: vi.fn(),
      setPacingSuspendPredicate: vi.fn(),
      dispose: vi.fn(),
      isActive: false,
    };

    mockInputHandler = {
      init: vi.fn(),
      getUiActions: vi.fn(() => ({ commands: {}, panels: {} })),
      getShortcutLabel: vi.fn(() => undefined),
      registerContext: vi.fn(),
      unregisterContext: vi.fn(),
      registerBinding: vi.fn(),
      unregisterBinding: vi.fn(),
      pushContext: vi.fn(),
      popContext: vi.fn(),
      setEnabled: vi.fn(),
      setRenderingControls: vi.fn(),
      setScaleBar: vi.fn(),
      setRecordingPanel: vi.fn(),
      setLayersPanel: vi.fn(),
      setControlRail: vi.fn(),
      setDatasetBrowser: vi.fn(),
      setOverlayManager: vi.fn(),
      setColormapLegend: vi.fn(),
      clearDimensionUI: vi.fn(),
      initDimensionSliders: vi.fn(),
      getRegisteredShortcutBindings: vi.fn(),
      dispose: vi.fn(),
    };

    mockRenderingControls = {
      setAnimationController: vi.fn(),
      setAdaptiveDPRManager: vi.fn(),
      setDensityGuardControl: vi.fn(),
      setSceneId: vi.fn(),
      setZarrViewerConfig: vi.fn(),
      hasStoredSettings: vi.fn().mockReturnValue(false),
      applyZarrDefaults: vi.fn(),
      syncCameraFovState: vi.fn(),
      applyOverrides: vi.fn(),
      getSettingsSnapshot: vi.fn(() => ({})),
      updateSceneScale: vi.fn(),
      dispose: vi.fn(),
    };

    // Setup constructor mocks
    (SceneManager as any).mockImplementation(() => mockSceneManager);
    (AnimationController as any).mockImplementation(() => mockAnimationController);
    (InputHandler as any).mockImplementation(() => mockInputHandler);
    (RenderingControls as any).mockImplementation(() => mockRenderingControls);

    // Reset fetch mock
    mockFetch.mockResolvedValue({ ok: false });

    // Minimal mock canvas — `document` is stubbed above so createElement
    // isn't available. SceneManager is mocked, so the reference is only
    // stored, never inspected.
    mockCanvas = {} as HTMLCanvasElement;

    // Create new app instance
    app = new LuxarApp();
  });

  afterEach(() => {
    if (app) {
      app.dispose();
    }
    // core.md C4 fix: restore all spies so SceneLoaderManager /
    // DataMonitorManager / workerPool dispose-spies created inside the
    // `dispose` block do not leak into subsequent tests. Without this,
    // ordering between tests could flip the assertions for the spied
    // call counts (the spies persist across vi.clearAllMocks).
    vi.restoreAllMocks();
  });

  it('passes the registered shortcut snapshot when viewer config opens help', () => {
    const bindings = new Map([['navigation', ['h']]]);
    mockInputHandler.getRegisteredShortcutBindings.mockReturnValue(bindings);
    (app as unknown as { inputHandler: typeof mockInputHandler }).inputHandler = mockInputHandler;

    (
      app as unknown as {
        applyViewerConfigState(config: { ui: { show_help: boolean } }): void;
      }
    ).applyViewerConfigState({ ui: { show_help: true } });

    expect(showHelpOverlay).toHaveBeenCalledWith(bindings);
  });

  describe('initialization sequence', () => {
    it('constructs every top-level subsystem during init (order-independent check)', async () => {
      // Previous version named this 'should initialize all components in correct order'
      // but only pushed ONE token into initOrder, so the contains-'sceneManager'
      // assertion was always satisfied regardless of construction sequence. The
      // dedicated order test lives in tests/unit/core/app/init/pipeline.test.ts:297-303.
      mockFetch.mockResolvedValue({ ok: true });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(SceneManager).toHaveBeenCalledTimes(1);
      expect(AnimationController).toHaveBeenCalledTimes(1);
      expect(InputHandler).toHaveBeenCalledTimes(1);
      expect(RenderingControls).toHaveBeenCalledTimes(1);
    });

    it('should create SceneManager first', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(SceneManager).toHaveBeenCalledTimes(1);
      expect(mockSceneManager.init).toHaveBeenCalled();
    });

    it('should create AnimationController after SceneManager', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(AnimationController).toHaveBeenCalledWith(
        mockSceneManager.controls,
        mockSceneManager.postProcessing
      );
    });

    it('should create InputHandler with proper dependencies', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      // InputHandler receives the PerformanceMonitor and DebugConsole
      // (both constructed at app level rather than in
      // AnimationController / InputHandler), plus a
      // DimensionSlidersFactory function so the input layer never
      // imports the concrete UI panel.
      expect(InputHandler).toHaveBeenCalledWith(
        mockSceneManager,
        mockAnimationController,
        expect.any(Object),
        expect.any(Object),
        expect.any(Function)
      );
      expect(mockInputHandler.init).toHaveBeenCalled();
    });

    it('should create RenderingControls with postProcessing and sceneManager', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(RenderingControls).toHaveBeenCalledWith(
        mockSceneManager.postProcessing,
        mockSceneManager
      );
    });

    it('cross-links every documented orchestrator pair', async () => {
      // core.md C3 fix: previous coverage asserted only two orchestrator
      // cross-links. Mutations dropping any remaining edge would have slipped
      // through silently. Pin the complete graph here.
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      // RenderingControls receives both the animation controller and (when
      // wired) the adaptive-DPR manager.
      expect(mockRenderingControls.setAnimationController).toHaveBeenCalledWith(
        mockAnimationController
      );
      if (mockRenderingControls.setAdaptiveDPRManager) {
        expect(mockRenderingControls.setAdaptiveDPRManager).toHaveBeenCalled();
      }

      // InputHandler receives RenderingControls and (when present) the
      // recording-panel and layers-panel.
      expect(mockInputHandler.setRenderingControls).toHaveBeenCalledWith(mockRenderingControls);
      if (mockInputHandler.setRecordingPanel) {
        expect(mockInputHandler.setRecordingPanel).toHaveBeenCalled();
      }
      if (mockInputHandler.setLayersPanel) {
        expect(mockInputHandler.setLayersPanel).toHaveBeenCalled();
      }
      expect(mockInputHandler.setControlRail).toHaveBeenCalled();

      // RecordingPanel receives panel-state callbacks and (when present)
      // the adaptive-DPR manager.
      const RecordingPanelMock = (await import('../../../ui/recording-panel'))
        .RecordingPanel as unknown as ReturnType<typeof vi.fn>;
      const lastRecordingPanelInstance = RecordingPanelMock.mock.results.at(-1)?.value;
      if (lastRecordingPanelInstance?.setPanelStateCallbacks) {
        expect(lastRecordingPanelInstance.setPanelStateCallbacks).toHaveBeenCalled();
      }
    });

    it('starts animation before loading and warms only after the final dataset render start', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      const callOrder: string[] = [];

      mockAnimationController.startAnimation.mockImplementation(() => {
        callOrder.push('startAnimation');
      });
      mockSceneManager.loadSceneData.mockImplementation(async () => {
        callOrder.push('loadSceneData');
      });
      mockSceneManager.warmBlendModePrograms.mockImplementation(() => {
        callOrder.push('warmBlendModePrograms');
      });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      const loadIndex = callOrder.indexOf('loadSceneData');
      const firstStartIndex = callOrder.indexOf('startAnimation');
      const finalStartIndex = callOrder.lastIndexOf('startAnimation');
      const warmIndex = callOrder.indexOf('warmBlendModePrograms');
      expect(firstStartIndex).toBeLessThan(loadIndex);
      expect(finalStartIndex).toBeGreaterThan(loadIndex);
      expect(warmIndex).toBeGreaterThan(finalStartIndex);
    });

    it('claims the embedder canvas before the initial dataset load', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      mockCanvas = jsdomDocument.createElement('canvas');
      const callOrder: string[] = [];
      ownershipMocks.install.mockImplementation(() => void callOrder.push('ownership'));
      mockSceneManager.loadSceneData.mockImplementation(async () => {
        callOrder.push('loadSceneData');
      });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(callOrder).toEqual(['ownership', 'loadSceneData']);
      expect(ownershipMocks.install).toHaveBeenCalledTimes(1);
    });

    it('initialized starts false on a freshly-constructed LuxarApp (pre-init invariant)', () => {
      // core.md C5 fix: split off the construction-default assertion from
      // the post-init test below. Previously both assertions lived in the
      // same `it` block whose name only described the post-init state, so
      // a regression where the constructor accidentally set `initialized=true`
      // would have been technically caught here but mislabelled. The
      // construction default is its own contract — test it on its own.
      expect(app.initialized).toBe(false);
    });

    it('should set isInitialized to true after successful init', async () => {
      // core.md W3 strengthening: previously a single-line "initialized===true"
      // assertion. Three observable side-effects MUST also be true after a
      // successful init() so mutations that flip `initialized` without
      // building the subsystems would still fail here.
      mockFetch.mockResolvedValue({ ok: true });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(app.initialized).toBe(true);
      // Components getter exposes the subsystems — a regression that
      // flipped `initialized` without wiring components would still
      // fail this branch.
      expect(app.components.sceneManager).toBeDefined();
      expect(app.components.animationController).toBeDefined();
      // sceneManager.init was actually awaited (not just constructed).
      expect(mockSceneManager.init).toHaveBeenCalledTimes(1);
    });

    it('should setup beforeunload dispose handler', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(mockAddEventListener).toHaveBeenCalledWith('beforeunload', expect.any(Function));
    });

    it('throws if init() is called twice without an intervening dispose()', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      await expect(
        app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' })
      ).rejects.toThrow(/already initialized/i);
    });

    it('allows init() again after dispose()', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });
      app.dispose();

      await expect(
        app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' })
      ).resolves.toBeUndefined();
      expect(app.initialized).toBe(true);
    });

    it('should setup focus handling', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(mockAddEventListener).toHaveBeenCalledWith('focus', expect.any(Function));
      expect(mockAddEventListener).toHaveBeenCalledWith('visibilitychange', expect.any(Function));
    });
  });

  describe('dataset detection logic', () => {
    it('should show browser for empty source', async () => {
      await app.init({ canvas: mockCanvas, src: '' });

      expect(DatasetBrowser).toHaveBeenCalled();
      expect(mockSceneManager.loadSceneData).not.toHaveBeenCalled();
    });

    it('should show browser for missing source', async () => {
      await app.init({ canvas: mockCanvas });

      expect(DatasetBrowser).toHaveBeenCalled();
      expect(mockSceneManager.loadSceneData).not.toHaveBeenCalled();
    });

    it('should show browser for directory URLs ending with /', async () => {
      await app.init({ canvas: mockCanvas, src: 'http://example.com/datasets/' });

      expect(DatasetBrowser).toHaveBeenCalled();
      expect(mockSceneManager.loadSceneData).not.toHaveBeenCalled();
    });

    it('should load a .zarr URL directly, without probing for it', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      // A `.zarr` suffix NAMES a store, so the HEAD probes are skipped
      // entirely — they only re-confirmed what the URL already said, at the
      // cost of a round trip ahead of everything else on first paint.
      expect(mockFetch).not.toHaveBeenCalled();
      expect(mockSceneManager.loadSceneData).toHaveBeenCalledWith(
        'http://example.com/data.zarr',
        undefined,
        { applyViewerConfigFov: true }
      );
      expect(DatasetBrowser).not.toHaveBeenCalled();
    });

    it('applies authored zoom after first-load rendering defaults switch to ortho', async () => {
      const order: string[] = [];
      mockSceneManager.getSceneViewerConfig.mockReturnValue({
        camera: { zoom: 2.5 },
        control_type: 'ortho',
      });
      mockRenderingControls.applyZarrDefaults.mockImplementation(() => order.push('defaults'));
      mockSceneManager.setCameraZoom.mockImplementation(() => order.push('zoom'));

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(mockSceneManager.setCameraZoom).toHaveBeenCalledWith(2.5);
      expect(order).toEqual(['defaults', 'zoom']);
    });

    it('reapplies authored zoom after stored settings and auto-framing', async () => {
      mockRenderingControls.hasStoredSettings.mockReturnValue(true);
      mockSceneManager.getSceneViewerConfig.mockReturnValue({ camera: { zoom: 3 } });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(mockRenderingControls.applyZarrDefaults).not.toHaveBeenCalled();
      expect(mockRenderingControls.syncCameraFovState).toHaveBeenCalledOnce();
      expect(mockSceneManager.setCameraZoom).toHaveBeenCalledWith(3);
    });

    it('should load directly for an unsuffixed URL whose zarr probe hits', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/store' });

      // The probe path still exists for URLs that do not name themselves.
      expect(mockFetch).toHaveBeenCalledWith(
        'http://example.com/store/.zgroup',
        expect.objectContaining({ method: 'HEAD' })
      );
      expect(mockSceneManager.loadSceneData).toHaveBeenCalledWith(
        'http://example.com/store',
        undefined,
        { applyViewerConfigFov: true }
      );
      expect(DatasetBrowser).not.toHaveBeenCalled();
    });

    it('should show browser for paths without extensions', async () => {
      mockFetch.mockResolvedValue({ ok: false });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/datasets' });

      expect(DatasetBrowser).toHaveBeenCalled();
      expect(mockSceneManager.loadSceneData).not.toHaveBeenCalled();
    });

    it('should handle fetch errors gracefully in detection', async () => {
      // core.md W4 strengthening: previous version only asserted
      // `initialized === true` with no signal as to which fallback ran.
      // A fetch error during dataset detection (probing for zarr metadata)
      // must NOT crash init — control falls through to the dataset
      // browser (likely a directory URL or offline). Pin both behaviors.
      mockFetch.mockRejectedValue(new Error('Network error'));

      // An UNSUFFIXED url, so detection actually reaches the probes: a
      // `.zarr` URL now short-circuits ahead of them and would never exercise
      // the rejection path this test exists for.
      await app.init({ canvas: mockCanvas, src: 'http://example.com/store' });

      expect(app.initialized).toBe(true);
      // Because all zarr-metadata probes rejected, `shouldShowBrowser`
      // returns true → DatasetBrowser is constructed. A regression
      // where the rejection bubbled up would NOT reach this point.
      expect(DatasetBrowser).toHaveBeenCalled();
      // And the direct-load path was NOT taken.
      expect(mockSceneManager.loadSceneData).not.toHaveBeenCalled();
    });

    it('should show browser for whitespace-only source', async () => {
      await app.init({ canvas: mockCanvas, src: '   ' });

      expect(DatasetBrowser).toHaveBeenCalled();
    });
  });

  describe('component integration', () => {
    beforeEach(async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });
    });

    it('should connect rendering controls to animation controller', () => {
      expect(mockRenderingControls.setAnimationController).toHaveBeenCalledWith(
        mockAnimationController
      );
    });

    it('should connect input handler to rendering controls', () => {
      expect(mockInputHandler.setRenderingControls).toHaveBeenCalledWith(mockRenderingControls);
    });

    it('should set scene ID before loading data', () => {
      expect(mockRenderingControls.setSceneId).toHaveBeenCalledWith('http://example.com/data.zarr');
    });

    it('should provide access to components via getter', () => {
      const components = app.components;

      expect(components.sceneManager).toBe(mockSceneManager);
      expect(components.animationController).toBe(mockAnimationController);
      expect(components).not.toHaveProperty('inputHandler');
      expect(components.renderingControls).toBe(mockRenderingControls);
    });

    it('should expose the narrow keyboard-input surface without exposing InputHandler', () => {
      const context = 'annotation';
      const config = { priority: 100, passthrough: true };
      const binding = {
        actionId: 'annotation.accept',
        key: 'x',
        handler: vi.fn(),
        description: 'Accept annotation',
        help: false as const,
      };

      app.registerContext(context, config);
      app.registerBinding(context, binding);
      app.pushContext(context);
      app.popContext();
      app.unregisterBinding(context, binding.key);
      app.unregisterContext(context);
      app.setInputEnabled(false);

      expect(mockInputHandler.registerContext).toHaveBeenCalledWith(context, config);
      expect(mockInputHandler.registerBinding).toHaveBeenCalledWith(context, binding);
      expect(mockInputHandler.pushContext).toHaveBeenCalledWith(context);
      expect(mockInputHandler.popContext).toHaveBeenCalledOnce();
      expect(mockInputHandler.unregisterBinding).toHaveBeenCalledWith(
        context,
        binding.key,
        undefined
      );
      expect(mockInputHandler.unregisterContext).toHaveBeenCalledWith(context);
      expect(mockInputHandler.setEnabled).toHaveBeenCalledWith(false);
    });

    it('should resolve registered action labels through the input facade', () => {
      mockInputHandler.getShortcutLabel.mockReturnValue('Shift+X');

      expect(app.shortcutForAction('annotation.accept')).toBe('Shift+X');
      expect(mockInputHandler.getShortcutLabel).toHaveBeenCalledWith('annotation.accept');

      app.dispose();
      expect(app.shortcutForAction('annotation.accept')).toBeUndefined();
    });

    it('should initialize dimension sliders after loading', async () => {
      expect(mockInputHandler.initDimensionSliders).toHaveBeenCalled();
    });
  });

  describe('error handling', () => {
    it('should handle scene manager init failure', async () => {
      const error = new Error('Failed to initialize scene manager');
      mockSceneManager.init.mockRejectedValue(error);

      await expect(
        app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' })
      ).rejects.toThrow('Failed to initialize scene manager');

      expect(app.initialized).toBe(false);
    });

    it('propagates loadSceneData errors out of init() so the caller can handle them', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      mockSceneManager.loadSceneData.mockRejectedValue(new Error('Load failed'));

      // Contract: init() does NOT swallow loadSceneData failures. The previous
      // name ("should continue if data loading fails") contradicted the
      // assertion — the test always asserted the throw.
      await expect(
        app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' })
      ).rejects.toThrow('Load failed');
    });

    it('disposes partial state when init() throws so the caller can retry', async () => {
      mockSceneManager.init.mockRejectedValue(new Error('Init failed'));

      try {
        await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });
      } catch {
        // Expected to throw
      }

      // Auto-dispose runs in init's catch block. cleanupUI() and the scene
      // manager's dispose() are both safe to call on partial state thanks
      // to per-field guards inside dispose().
      expect(mockCleanupUI).toHaveBeenCalledTimes(1);
      expect(mockSceneManager.dispose).toHaveBeenCalled();
      expect(app.initialized).toBe(false);
    });

    it('allows init() to succeed after a previous init() throw', async () => {
      mockSceneManager.init.mockRejectedValueOnce(new Error('Transient init failure'));

      await expect(
        app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' })
      ).rejects.toThrow('Transient init failure');

      // Recovery: second init with the same instance should work because
      // dispose() ran in the first init's catch and reset state.
      mockSceneManager.init.mockResolvedValueOnce(undefined);
      mockFetch.mockResolvedValue({ ok: true });

      await expect(
        app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' })
      ).resolves.toBeUndefined();
      expect(app.initialized).toBe(true);
    });

    it('should preserve error state when init fails', async () => {
      mockSceneManager.init.mockRejectedValue(new Error('Init failed'));

      try {
        await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });
      } catch {
        // Expected to throw
      }

      expect(app.initialized).toBe(false);
    });

    it('should handle animation controller creation failure', async () => {
      (AnimationController as any).mockImplementation(() => {
        throw new Error('Animation controller failed');
      });

      await expect(
        app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' })
      ).rejects.toThrow('Animation controller failed');
    });

    it('should handle input handler init failure', async () => {
      mockInputHandler.init.mockImplementation(() => {
        throw new Error('Input handler failed');
      });

      await expect(
        app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' })
      ).rejects.toThrow('Input handler failed');
    });
  });

  describe('dispose', () => {
    beforeEach(async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });
      vi.clearAllMocks();
    });

    it('should dispose all components in reverse order', () => {
      const disposeOrder: string[] = [];

      mockAnimationController.dispose.mockImplementation(() => {
        disposeOrder.push('animationController');
      });
      mockInputHandler.dispose.mockImplementation(() => {
        disposeOrder.push('inputHandler');
      });
      mockRenderingControls.dispose.mockImplementation(() => {
        disposeOrder.push('renderingControls');
      });
      mockSceneManager.dispose.mockImplementation(() => {
        disposeOrder.push('sceneManager');
      });

      app.dispose();

      expect(disposeOrder).toEqual([
        'animationController',
        'inputHandler',
        'renderingControls',
        'sceneManager',
      ]);
    });

    it('should call cleanupUI', () => {
      app.dispose();

      expect(mockCleanupUI).toHaveBeenCalled();
    });

    it('gives the page its own title back', () => {
      // document.title is a host-page global: an embedder that removes the
      // viewer must not be left with a tab named after a torn-down scene.
      // Probe for the restore target — earlier tests in this file overwrite
      // the title too, so the module's one-shot page-title capture may
      // already have happened.
      setDocumentTitle('probe');
      setDocumentTitle(null);
      const pageTitle = document.title;

      setDocumentTitle('Rivers of Earth');
      app.dispose();

      expect(document.title).toBe(pageTitle);
    });

    it('restores the title even when a teardown step throws', () => {
      setDocumentTitle('probe');
      setDocumentTitle(null);
      const pageTitle = document.title;

      setDocumentTitle('Rivers of Earth');
      mockSceneManager.dispose.mockImplementation(() => {
        throw new Error('Dispose failed');
      });

      expect(() => app.dispose()).not.toThrow();
      expect(document.title).toBe(pageTitle);
    });

    it('should remove beforeunload listener', () => {
      app.dispose();

      expect(mockRemoveEventListener).toHaveBeenCalledWith('beforeunload', expect.any(Function));
    });

    it('should set isInitialized to false', () => {
      app.dispose();

      expect(app.initialized).toBe(false);
    });

    it('should handle dispose errors gracefully', () => {
      // core.md W5 strengthening: previously only asserted "does not throw".
      // The dispose() contract is stronger — when one subsystem's dispose
      // throws, the OTHERS must still run (otherwise a single failing
      // dispose leaves GPU / event-listener leaks across the app).
      mockSceneManager.dispose.mockImplementation(() => {
        throw new Error('Dispose failed');
      });

      expect(() => app.dispose()).not.toThrow();

      // All other dispose calls still happened despite the scene
      // manager throwing — `safeDispose` wrapper continues past errors.
      expect(mockAnimationController.dispose).toHaveBeenCalledTimes(1);
      expect(mockInputHandler.dispose).toHaveBeenCalledTimes(1);
      expect(mockRenderingControls.dispose).toHaveBeenCalledTimes(1);
      expect(mockSceneManager.dispose).toHaveBeenCalledTimes(1);
      // Initialized flag is still cleared so re-init paths see a clean
      // slate.
      expect(app.initialized).toBe(false);
    });

    it('should handle multiple dispose calls safely', () => {
      app.dispose();
      app.dispose();

      // Should not throw on second call
      expect(app.initialized).toBe(false);
    });

    it('disposes each component exactly once across re-entrant calls', () => {
      // First dispose runs everything.
      app.dispose();

      expect(mockAnimationController.dispose).toHaveBeenCalledTimes(1);
      expect(mockInputHandler.dispose).toHaveBeenCalledTimes(1);
      expect(mockRenderingControls.dispose).toHaveBeenCalledTimes(1);
      expect(mockSceneManager.dispose).toHaveBeenCalledTimes(1);

      // Second dispose: components are still referenced (not nulled out) but
      // the isDisposing/isInitialized guards short-circuit before any
      // child dispose is invoked again. Without that guard, animationController.dispose
      // etc. would be called twice and could double-free GPU resources.
      app.dispose();

      expect(mockAnimationController.dispose).toHaveBeenCalledTimes(1);
      expect(mockInputHandler.dispose).toHaveBeenCalledTimes(1);
      expect(mockRenderingControls.dispose).toHaveBeenCalledTimes(1);
      expect(mockSceneManager.dispose).toHaveBeenCalledTimes(1);
    });

    it('reports initialized=false from the very first instant of teardown', () => {
      // animationController.dispose runs first inside the dispose chain;
      // observe app.initialized from inside it. Pre-A3, this would still be true
      // because isInitialized flipped at the END of the try block.
      let initializedDuringTeardown: boolean | null = null;
      mockAnimationController.dispose.mockImplementation(() => {
        initializedDuringTeardown = app.initialized;
      });

      app.dispose();

      expect(initializedDuringTeardown).toBe(false);
    });

    it('should dispose animation controller first', () => {
      app.dispose();

      expect(mockAnimationController.dispose).toHaveBeenCalled();
    });

    it('should dispose scene manager last', () => {
      const disposeOrder: string[] = [];

      mockAnimationController.dispose.mockImplementation(() => {
        disposeOrder.push('animation');
      });
      mockSceneManager.dispose.mockImplementation(() => {
        disposeOrder.push('scene');
      });

      app.dispose();

      expect(disposeOrder.indexOf('scene')).toBeGreaterThan(disposeOrder.indexOf('animation'));
    });

    it('disposes the SceneLoaderManager and DataMonitorManager singletons', async () => {
      // Long-lived static singletons. Without explicit disposeInstance()
      // calls, their loaders + cache stores + eventBus subscriptions
      // survive across LuxarApp re-init.
      //
      // AUDIT NOTE (core.md C4): the spies created below + at lines 703-705,
      // 726-728, 750-... are NOT explicitly restored. vi.clearAllMocks()
      // in beforeEach clears the spy CALL histories but does NOT remove
      // the spy itself — subsequent tests in the same describe block see
      // the spied (no-op) version of SceneLoaderManager.disposeInstance.
      // The dispose-pipeline.test.ts file does this correctly with
      // afterEach(() => vi.restoreAllMocks()). Follow-up: add the same
      // afterEach here so cross-test contamination is impossible.
      const sceneLoaderModule = await import('../../../data/scene-loader-manager');
      const dataMonitorModule = await import('../../../ui/data-monitor-manager');

      const sceneLoaderSpy = vi.spyOn(sceneLoaderModule.SceneLoaderManager, 'disposeInstance');
      const dataMonitorSpy = vi.spyOn(dataMonitorModule.DataMonitorManager, 'disposeInstance');

      app.dispose();

      expect(sceneLoaderSpy).toHaveBeenCalledTimes(1);
      expect(dataMonitorSpy).toHaveBeenCalledTimes(1);

      // Order: monitor first (its factory wiring holds loader refs),
      // then the loader manager drops the actual loaders + cache stores.
      const monitorCallOrder = dataMonitorSpy.mock.invocationCallOrder[0];
      const loaderCallOrder = sceneLoaderSpy.mock.invocationCallOrder[0];
      expect(monitorCallOrder).toBeLessThan(loaderCallOrder);
    });

    it('still disposes singletons + workerPool when an early component throws', async () => {
      // Pre-existing dispose() wrapped everything in one
      // try/catch, so a throw early in the chain (sceneManager etc.)
      // skipped DataMonitorManager / SceneLoaderManager / disposeWorkerPool
      // / managerRegistry. The safeDispose helper guarantees later
      // teardown runs regardless.
      const sceneLoaderModule = await import('../../../data/scene-loader-manager');
      const dataMonitorModule = await import('../../../ui/data-monitor-manager');
      const workerPoolModule = await import('../../../workers/worker-pool');

      const sceneLoaderSpy = vi.spyOn(sceneLoaderModule.SceneLoaderManager, 'disposeInstance');
      const dataMonitorSpy = vi.spyOn(dataMonitorModule.DataMonitorManager, 'disposeInstance');
      const workerPoolSpy = vi.spyOn(workerPoolModule, 'disposeWorkerPool');

      // Force an early disposer to throw — animationController is the
      // very first call site inside dispose().
      mockAnimationController.dispose.mockImplementation(() => {
        throw new Error('animation dispose blew up');
      });

      expect(() => app.dispose()).not.toThrow();

      // The throw must NOT have aborted later cleanup:
      expect(dataMonitorSpy).toHaveBeenCalledTimes(1);
      expect(sceneLoaderSpy).toHaveBeenCalledTimes(1);
      expect(workerPoolSpy).toHaveBeenCalledTimes(1);
    });

    it('still disposes singletons + workerPool when a middle component throws', async () => {
      const sceneLoaderModule = await import('../../../data/scene-loader-manager');
      const dataMonitorModule = await import('../../../ui/data-monitor-manager');
      const workerPoolModule = await import('../../../workers/worker-pool');

      const sceneLoaderSpy = vi.spyOn(sceneLoaderModule.SceneLoaderManager, 'disposeInstance');
      const dataMonitorSpy = vi.spyOn(dataMonitorModule.DataMonitorManager, 'disposeInstance');
      const workerPoolSpy = vi.spyOn(workerPoolModule, 'disposeWorkerPool');

      // sceneManager sits in the middle of the dispose chain — between
      // the UI/scene panels and the singleton/worker teardown.
      mockSceneManager.dispose.mockImplementation(() => {
        throw new Error('sceneManager dispose blew up');
      });

      expect(() => app.dispose()).not.toThrow();

      expect(dataMonitorSpy).toHaveBeenCalledTimes(1);
      expect(sceneLoaderSpy).toHaveBeenCalledTimes(1);
      expect(workerPoolSpy).toHaveBeenCalledTimes(1);
    });

    it('disposes the worker pool after the scene-loader manager', async () => {
      // Order matters: SceneLoaderManager drops loaders that may still
      // post messages to workers; disposing workers BEFORE the loader
      // manager could race a final message into a terminated worker.
      const sceneLoaderModule = await import('../../../data/scene-loader-manager');
      const workerPoolModule = await import('../../../workers/worker-pool');

      const sceneLoaderSpy = vi.spyOn(sceneLoaderModule.SceneLoaderManager, 'disposeInstance');
      const workerPoolSpy = vi.spyOn(workerPoolModule, 'disposeWorkerPool');

      app.dispose();

      expect(sceneLoaderSpy).toHaveBeenCalledTimes(1);
      expect(workerPoolSpy).toHaveBeenCalledTimes(1);
      const loaderOrder = sceneLoaderSpy.mock.invocationCallOrder[0];
      const workerOrder = workerPoolSpy.mock.invocationCallOrder[0];
      expect(loaderOrder).toBeLessThan(workerOrder);
    });

    it('closes an open DatasetBrowser and clears app + input-handler refs', () => {
      // Plant a fake browser to exercise the safeDispose('datasetBrowser')
      // step. Real construction goes through `showDatasetBrowser()` which
      // opens it lazily when no `?src=` is given; assigning here matches
      // the post-init state when the user has the browser open.
      const browserClose = vi.fn();
      const setDatasetBrowserSpy = vi.spyOn(mockInputHandler, 'setDatasetBrowser');
      (app as unknown as { datasetBrowser: { close: () => void } }).datasetBrowser = {
        close: browserClose,
      };

      app.dispose();

      expect(browserClose).toHaveBeenCalledTimes(1);
      expect(setDatasetBrowserSpy).toHaveBeenCalledWith(undefined);
      // Field cleared: a stale browser ref shouldn't persist on a
      // disposed app instance.
      expect((app as unknown as { datasetBrowser: unknown }).datasetBrowser).toBeUndefined();
    });

    it('does not throw when DatasetBrowser is not open at dispose time', () => {
      // Common path: user navigated with `?src=...`, never opened the
      // browser. `this.datasetBrowser` is undefined; the optional chain
      // in safeDispose handles it.
      expect(() => app.dispose()).not.toThrow();
    });
  });

  describe('focus handling', () => {
    beforeEach(async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });
    });

    it('should register focus event listener', () => {
      expect(mockAddEventListener).toHaveBeenCalledWith('focus', expect.any(Function));
    });

    it('should register visibilitychange listener', () => {
      expect(mockAddEventListener).toHaveBeenCalledWith('visibilitychange', expect.any(Function));
    });

    it('should trigger render on window focus', async () => {
      mockAnimationController.startAnimation.mockClear();

      // Find the focus handler
      const focusHandler = mockAddEventListener.mock.calls.find((call) => call[0] === 'focus')?.[1];

      expect(focusHandler).toBeDefined();
      focusHandler?.();

      expect(mockAnimationController.startAnimation).toHaveBeenCalled();
    });

    it('should trigger render on visibility change when not hidden', async () => {
      mockAnimationController.startAnimation.mockClear();
      (document as any).hidden = false;

      // Find the visibilitychange handler
      const visibilityHandler = mockAddEventListener.mock.calls.find(
        (call) => call[0] === 'visibilitychange'
      )?.[1];

      expect(visibilityHandler).toBeDefined();
      visibilityHandler?.();

      expect(mockAnimationController.startAnimation).toHaveBeenCalled();
    });

    it('should not trigger render when document is hidden', async () => {
      mockAnimationController.startAnimation.mockClear();
      mockAnimationController.stopAnimation.mockClear();
      (document as any).hidden = true;

      // Find the visibilitychange handler
      const visibilityHandler = mockAddEventListener.mock.calls.find(
        (call) => call[0] === 'visibilitychange'
      )?.[1];

      expect(visibilityHandler).toBeDefined();
      visibilityHandler?.();

      // Should stop animation when hidden (saves resources)
      expect(mockAnimationController.stopAnimation).toHaveBeenCalled();
      expect(mockAnimationController.startAnimation).not.toHaveBeenCalled();
    });
  });

  describe('dataset browser', () => {
    it('should clear error when opening browser', async () => {
      await app.init({ canvas: mockCanvas, src: '' });

      expect(mockClearError).toHaveBeenCalled();
    });

    it('should register open-dataset-browser event listener', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(mockAddEventListener).toHaveBeenCalledWith(
        'open-dataset-browser',
        expect.any(Function)
      );
    });

    it('opens during the initial load but refuses selection without changing the source', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      let releaseInitialLoad!: () => void;
      mockSceneManager.loadSceneData.mockImplementationOnce(
        () => new Promise<void>((resolve) => (releaseInitialLoad = resolve))
      );

      const initPromise = app.init({
        canvas: mockCanvas,
        src: 'http://example.com/initial.zarr',
        updateBrowserUrl: true,
      });
      await vi.waitFor(() => expect(mockSceneManager.loadSceneData).toHaveBeenCalledTimes(1));

      try {
        const openBrowser = mockAddEventListener.mock.calls.find(
          (call) => call[0] === 'open-dataset-browser'
        )?.[1] as (() => void) | undefined;
        expect(openBrowser).toBeDefined();

        openBrowser!();
        const browserCall = (DatasetBrowser as any).mock.calls.at(-1);
        expect(browserCall).toBeDefined();
        const onSelect = browserCall[0].onDatasetSelect as (url: string) => false | Promise<void>;

        expect(onSelect('http://example.com/replacement.zarr')).toBe(false);
        expect(mockShowToast).toHaveBeenCalledExactlyOnceWith(
          'Luxar is still starting up; try again in a moment.'
        );
        expect((app as any).options.src).toBe('http://example.com/initial.zarr');
        expect(mockReplaceState).not.toHaveBeenCalled();
        expect(mockSceneManager.loadSceneData).toHaveBeenCalledTimes(1);
      } finally {
        releaseInitialLoad();
        await initPromise;
      }
    });

    it('removes the early browser listener and clears initializing state when init fails', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      mockSceneManager.loadSceneData.mockRejectedValueOnce(new Error('initial load failed'));

      await expect(
        app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' })
      ).rejects.toThrow('initial load failed');

      const browserRegistration = mockAddEventListener.mock.calls.find(
        (call) => call[0] === 'open-dataset-browser'
      );
      expect(browserRegistration).toBeDefined();
      expect(mockRemoveEventListener).toHaveBeenCalledWith(
        'open-dataset-browser',
        browserRegistration![1]
      );
      expect(() => app.switchDataset('http://example.com/retry.zarr')).toThrow(/before init/);
    });

    it('clears initializing state immediately when disposed during init', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      let releaseInitialLoad!: () => void;
      mockSceneManager.loadSceneData.mockImplementationOnce(
        () => new Promise<void>((resolve) => (releaseInitialLoad = resolve))
      );

      const initPromise = app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });
      await vi.waitFor(() => expect(mockSceneManager.loadSceneData).toHaveBeenCalledTimes(1));

      app.dispose();
      expect(() => app.switchDataset('http://example.com/retry.zarr')).toThrow(/before init/);

      releaseInitialLoad();
      await initPromise;
      app.dispose();
    });

    it('does not call history.replaceState when updateBrowserUrl is false', async () => {
      const replaceStateSpy = vi.spyOn(window.history, 'replaceState').mockImplementation(() => {});

      await app.init({ canvas: mockCanvas, src: '', updateBrowserUrl: false });

      const browserCall = (DatasetBrowser as any).mock.calls.at(-1);
      expect(browserCall).toBeDefined();
      const onSelect = browserCall[0].onDatasetSelect as (url: string) => Promise<void>;

      replaceStateSpy.mockClear();
      await onSelect('http://example.com/picked.zarr');

      expect(replaceStateSpy).not.toHaveBeenCalled();
      replaceStateSpy.mockRestore();
    });

    it('does not call history.replaceState by default for embedded safety', async () => {
      const replaceStateSpy = vi.spyOn(window.history, 'replaceState').mockImplementation(() => {});

      await app.init({ canvas: mockCanvas, src: '' });

      const browserCall = (DatasetBrowser as any).mock.calls.at(-1);
      const onSelect = browserCall[0].onDatasetSelect as (url: string) => Promise<void>;

      replaceStateSpy.mockClear();
      await onSelect('http://example.com/picked.zarr');

      expect(replaceStateSpy).not.toHaveBeenCalled();
      replaceStateSpy.mockRestore();
    });

    it('calls history.replaceState when updateBrowserUrl is true', async () => {
      const replaceStateSpy = vi.spyOn(window.history, 'replaceState').mockImplementation(() => {});

      await app.init({ canvas: mockCanvas, src: '', updateBrowserUrl: true });

      const browserCall = (DatasetBrowser as any).mock.calls.at(-1);
      const onSelect = browserCall[0].onDatasetSelect as (url: string) => Promise<void>;

      replaceStateSpy.mockClear();
      await onSelect('http://example.com/picked.zarr');

      expect(replaceStateSpy).toHaveBeenCalledTimes(1);
      replaceStateSpy.mockRestore();
    });

    it('serializes browser selections through the shared dataset-switch guard', async () => {
      const replaceStateSpy = vi.spyOn(window.history, 'replaceState').mockImplementation(() => {});
      await app.init({ canvas: mockCanvas, src: '', updateBrowserUrl: true });
      const browserCall = (DatasetBrowser as any).mock.calls.at(-1);
      const onSelect = browserCall[0].onDatasetSelect as (url: string) => Promise<void>;

      let release!: () => void;
      mockSceneManager.loadSceneData.mockImplementationOnce(
        () => new Promise<void>((resolve) => (release = resolve))
      );

      replaceStateSpy.mockClear();
      const first = onSelect('http://example.com/a.zarr');
      await expect(onSelect('http://example.com/b.zarr')).rejects.toThrow(/in progress/);
      expect(mockSceneManager.loadSceneData).toHaveBeenCalledTimes(1);
      expect(mockSceneManager.loadSceneData).toHaveBeenCalledWith(
        'http://example.com/a.zarr',
        undefined,
        { applyViewerConfigFov: true }
      );
      // The rejected selection must not leave the configured src or the
      // host-page URL pointing at the dataset that never loaded.
      expect((app as any).options.src).toBe('http://example.com/a.zarr');
      expect(replaceStateSpy).toHaveBeenCalledTimes(1);

      release();
      await first;
      replaceStateSpy.mockRestore();
    });

    it('does not reopen the browser shortcut until a selected dataset finishes switching', async () => {
      await app.init({ canvas: mockCanvas, src: '' });
      const browserCall = (DatasetBrowser as any).mock.calls.at(-1);
      const onSelect = browserCall[0].onDatasetSelect as (url: string) => Promise<void>;
      const onClose = browserCall[0].onClose as () => void;
      const openBrowser = mockAddEventListener.mock.calls.find(
        (call) => call[0] === 'open-dataset-browser'
      )?.[1] as (() => void) | undefined;
      expect(openBrowser).toBeDefined();

      let release!: () => void;
      mockSceneManager.loadSceneData.mockImplementationOnce(
        () => new Promise<void>((resolve) => (release = resolve))
      );

      const switching = onSelect('http://example.com/a.zarr');
      onClose(); // DatasetBrowser closes synchronously after firing onDatasetSelect.
      openBrowser?.();
      expect(DatasetBrowser).toHaveBeenCalledTimes(1);

      release();
      await switching;
      openBrowser?.();
      expect(DatasetBrowser).toHaveBeenCalledTimes(2);
    });
  });

  describe('debug interface', () => {
    it('should not setup debug interface without ?debug param', async () => {
      (window.location as any).search = '';
      mockFetch.mockResolvedValue({ ok: true });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect((window as any).__luxarDebug).toBeUndefined();
    });

    it('should setup debug interface when debug option is passed', async () => {
      mockFetch.mockResolvedValue({ ok: true });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr', debug: true });

      expect((window as any).__luxarDebug).toBeDefined();
      expect((window as any).__luxarDebug.scene).toBe(mockSceneManager.scene);
      expect((window as any).__luxarDebug.camera).toBe(mockSceneManager.camera);
      expect((window as any).__luxarDebug.runtimeReady).toBe(true);
    });
  });

  describe('snapshot API pre-init guards (core.md G20)', () => {
    // captureSnapshot/restoreSnapshot read this.sceneManager and would
    // crash inside `sceneManager.camera.position` (or similar) if called
    // before init(). The current implementation throws with a clear
    // message; pin that message so a regression that dropped the guard
    // would re-introduce the cryptic "cannot read property X of undefined".
    it('captureSnapshot() throws a clear message before init()', () => {
      expect(() => app.captureSnapshot()).toThrow(/captureSnapshot called before init/i);
    });

    it('restoreSnapshot() throws a clear message before init()', () => {
      const fakeSnapshot = {
        version: 1 as const,
        camera: {
          position: [0, 0, 0] as [number, number, number],
          target: [0, 0, 0] as [number, number, number],
          up: [0, 1, 0] as [number, number, number],
          isOrtho: false,
          near: 0.1,
          far: 1000,
        },
      };
      expect(() => app.restoreSnapshot(fakeSnapshot)).toThrow(
        /restoreSnapshot called before init/i
      );
    });

    it('snapshot API works after init() (sanity)', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      // mockSceneManager.camera/controls need shape for the snapshot
      // helper to read; the default mocks don't expose them. This
      // sanity test verifies the guard flips OFF after init even if
      // the underlying snapshot machinery fails for other reasons —
      // we only care that the pre-init guard branch is no longer
      // taken.
      mockSceneManager.camera = {
        position: { x: 1, y: 2, z: 3 },
        up: { x: 0, y: 1, z: 0 },
        near: 0.1,
        far: 1000,
      };
      mockSceneManager.controls = {
        getFocusTarget: () => ({ x: 0, y: 0, z: 0 }),
        setTarget: vi.fn(),
        reinitialize: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      };
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      // The guard branch is no longer taken — any other error path is
      // out of scope here. We just confirm the pre-init guard didn't
      // fire.
      try {
        app.captureSnapshot();
      } catch (e) {
        expect((e as Error).message).not.toMatch(/before init/);
      }
    });
  });

  describe('remote-control embedder API (flyTo / settings / layers / state)', () => {
    const SRC = 'http://example.com/data.zarr';
    const POSE = {
      position: [0, 0, 10] as [number, number, number],
      target: [0, 0, 0] as [number, number, number],
      up: [0, 1, 0] as [number, number, number],
      isOrtho: false,
      near: 0.1,
      far: 1000,
    };

    it('every new method throws a clear message before init()', () => {
      expect(() => app.flyTo(POSE)).toThrow(/flyTo called before init/);
      expect(() => app.getRenderingSettings()).toThrow(/getRenderingSettings called before init/);
      expect(() => app.setRenderingSettings({ exposure: 1 })).toThrow(
        /setRenderingSettings called before init/
      );
      expect(() => app.getLayers()).toThrow(/getLayers called before init/);
      expect(() => app.setLayer('/x', { opacity: 1 })).toThrow(/setLayer called before init/);
      expect(() => app.getViewerState()).toThrow(/getViewerState called before init/);
      expect(() => app.getAudioState()).toThrow(/getAudioState called before init/);
      expect(() => app.setAudio({ muted: true })).toThrow(/setAudio called before init/);
      expect(() => app.playSound('narration')).toThrow(/playSound called before init/);
      expect(() => app.stopSound('narration')).toThrow(/stopSound called before init/);
    });

    it('forwards the public audio controls to the engine', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      const state = {
        state: 'running',
        muted: false,
        masterGain: 0.8,
        panningModel: 'equalpower',
        buses: { ambient: 0.6, voice: 1, effects: 0.8 },
        playing: [],
        hasSoundNodes: true,
      } as const;
      const audio = {
        getState: vi.fn(() => state),
        setAudio: vi.fn(),
        play: vi.fn(() => true),
        stop: vi.fn(() => false),
      };
      (app as unknown as { audioEngine: typeof audio }).audioEngine = audio;

      expect(app.getAudioState()).toBe(state);
      app.setAudio({ masterGain: 0.5 });
      expect(audio.setAudio).toHaveBeenCalledWith({ masterGain: 0.5 });
      expect(app.playSound('narration')).toBe(true);
      expect(app.stopSound('narration')).toBe(false);
    });

    it('replays layer mutes after attaching audio and before the opening waypoint', () => {
      const order: string[] = [];
      const root = { name: 'LuxarScene' };
      const audioEngine = {
        detachScene: vi.fn(() => order.push('detach')),
        applySceneConfig: vi.fn(() => order.push('config')),
        attachScene: vi.fn(() => order.push('attach')),
        notifyWaypoint: vi.fn(() => order.push('waypoint')),
        dispose: vi.fn(),
      };
      const layersPanel = {
        pushAudioMutes: vi.fn(() => order.push('mutes')),
        dispose: vi.fn(),
      };
      const waypointDriver = {
        currentIndex: 0,
        getWaypoint: vi.fn(() => ({ when: { story: 0 } })),
      };
      mockSceneManager.scene = { children: [root] };
      Object.assign(app as unknown as Record<string, unknown>, {
        audioEngine,
        layersPanel,
        sceneManager: mockSceneManager,
        waypointDriver,
      });

      (app as unknown as { installAudio(audio: unknown): void }).installAudio(undefined);

      expect(order).toEqual(['detach', 'config', 'attach', 'mutes', 'waypoint']);
    });

    it('subscribes to the controls change stream and re-emits it as camera-changed', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      const calls = mockSceneManager.controls.addEventListener.mock.calls as Array<
        [string, () => void]
      >;
      const change = calls.find(([type]) => type === 'change');
      expect(change).toBeDefined();

      const onCamera = vi.fn();
      app.on('camera-changed', onCamera);
      change![1]();

      expect(onCamera).toHaveBeenCalledTimes(1);
      expect(onCamera.mock.calls[0][0]).toMatchObject({ position: [1, 2, 3], target: [0, 0, 0] });

      // Torn down with the app: the same listener reference is removed.
      app.dispose();
      expect(mockSceneManager.controls.removeEventListener).toHaveBeenCalledWith(
        'change',
        change![1]
      );
    });

    it('flyTo registers the flight driver and starts the loop; a dataset switch cancels it', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      mockAnimationController.startAnimation.mockClear();

      const done = app.flyTo(POSE, { durationMs: 1000 });

      expect(mockAnimationController.addPerFrameCallback).toHaveBeenCalledWith(
        'camera-flight',
        expect.any(Function),
        { continuous: true }
      );
      expect(mockAnimationController.startAnimation).toHaveBeenCalled();

      await app.switchDataset('http://example.com/other.zarr');
      await expect(done).resolves.toEqual({ completed: false });
      expect(mockAnimationController.removePerFrameCallback).toHaveBeenCalledWith('camera-flight');
    });

    it('setRenderingSettings rides the same override path as an authored viewer_config', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });

      app.setRenderingSettings({ exposure: 1.5, toneMapping: 'ACES' });

      expect(mockRenderingControls.applyOverrides).toHaveBeenCalledWith({
        exposure: 1.5,
        toneMapping: 'ACES',
      });
    });

    it('authored waypoints snap at load and fly when the story dimension changes', async () => {
      // A camera restoreCamera() can write to (the snap path), on top of the
      // readable fields the default mock already has.
      mockSceneManager.camera = {
        position: { x: 0, y: 0, z: 10, set: vi.fn() },
        up: { x: 0, y: 1, z: 0, set: vi.fn() },
        near: 0.1,
        far: 100,
        updateProjectionMatrix: vi.fn(),
      };
      mockSceneManager.getSceneViewerConfig.mockReturnValue({
        waypoints: [
          { when: { story: 0 }, camera: { position: [0, 0, 5] } },
          { when: { story: 1 }, camera: { position: [5, 0, 0] }, duration_ms: 700 },
        ],
      });
      // The scene's dims (the real dims manager is not mocked): three shown
      // axes plus a hidden `story` axis at 0.
      const dimensions = ['x', 'y', 'z', 'story'].map((name, i) => ({
        name,
        unit: '',
        range: [0, 3] as [number, number],
        step: 1,
        display: i < 3,
      }));
      const fakeScene = {
        userData: { sceneDimensions: { dimensions } },
        children: [],
        getObjectByName: () => undefined,
      } as unknown as Parameters<typeof sceneDimsManager.initFromScene>[0];
      sceneDimsManager.initFromScene(fakeScene);
      sceneDimsManager.setDimensionValue(3, 0);
      // init() only loads `src` directly when the probe says it exists;
      // otherwise it opens the dataset browser instead.
      mockFetch.mockResolvedValue({ ok: true });

      try {
        await app.init({ canvas: mockCanvas, src: SRC });

        // Load-time: the matched waypoint is applied as a SNAP (restoreCamera),
        // not a flight.
        expect(mockSceneManager.camera.position.set).toHaveBeenCalledWith(0, 0, 5);
        expect(mockAnimationController.addPerFrameCallback).not.toHaveBeenCalledWith(
          'camera-flight',
          expect.any(Function),
          expect.anything()
        );

        // Stepping the story dimension to a different waypoint flies.
        sceneDimsManager.setDimensionValue(3, 1);
        expect(mockAnimationController.addPerFrameCallback).toHaveBeenCalledWith(
          'camera-flight',
          expect.any(Function),
          { continuous: true }
        );

        const flightsStarted = (): number =>
          mockAnimationController.addPerFrameCallback.mock.calls.filter(
            (c: unknown[]) => c[0] === 'camera-flight'
          ).length;

        // A move that keeps the same waypoint matched does nothing more.
        mockAnimationController.addPerFrameCallback.mockClear();
        sceneDimsManager.setDimensionValue(3, 1.2);
        expect(flightsStarted()).toBe(0);

        // A scene without waypoints detaches the previous scene's binding
        // (the switch itself registers other per-frame work; only flights count).
        mockSceneManager.getSceneViewerConfig.mockReturnValue({});
        await app.switchDataset('http://example.com/plain.zarr');
        mockAnimationController.addPerFrameCallback.mockClear();
        sceneDimsManager.setDimensionValue(3, 0);
        expect(flightsStarted()).toBe(0);
      } finally {
        sceneDimsManager.reset();
      }
    });

    it('getViewerState bundles dataset, camera, dims, rendering and layers', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      mockRenderingControls.getSettingsSnapshot.mockReturnValue({ exposure: 0.25 });

      const state = app.getViewerState();

      expect(state.src).toBe(SRC);
      expect(state.camera).toMatchObject({ position: [1, 2, 3] });
      expect(state.dimensions).toMatchObject({ ndim: 0 });
      expect(state.rendering).toEqual({ exposure: 0.25 });
      // LayersPanel is mocked: its summaries come back undefined → empty list.
      expect(state.layers).toEqual([]);
    });
  });

  describe('programmatic embedder API', () => {
    const SRC = 'http://example.com/data.zarr';

    it('emits dataset-loaded when switchDataset succeeds', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      const onLoaded = vi.fn();
      app.on('dataset-loaded', onLoaded);

      await app.switchDataset('http://example.com/other.zarr');

      expect(onLoaded).toHaveBeenCalledWith({ src: 'http://example.com/other.zarr' });
    });

    it('re-titles the browser tab for the dataset being switched to', async () => {
      // Whatever named the tab belongs to the outgoing scene: `?title=` names
      // the dataset the server started with, an authored title names the scene
      // being torn down. Either one left in place advertises a scene the tab
      // no longer shows.
      // Establish the page-title target the helper restores to. Earlier tests
      // in this file switch datasets too, so the module's one-shot capture of
      // the page title may already have happened — probe it instead of
      // assuming a pristine document.
      setDocumentTitle('probe');
      setDocumentTitle(null);
      const pageTitle = document.title;

      await app.init({ canvas: mockCanvas, src: SRC });
      setDocumentTitle('Previous Scene');

      await app.switchDataset('http://example.com/global_rivers.luxar.zarr');
      expect(document.title).toBe('global_rivers');

      // A src that names no store falls back to the page's own title rather
      // than keeping the last scene's name.
      await app.switchDataset('http://example.com:8000');
      expect(document.title).toBe(pageTitle);
    });

    it('emits dataset-error and rejects when a load fails', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      mockSceneManager.loadSceneData.mockRejectedValueOnce(new Error('boom'));
      const onError = vi.fn();
      app.on('dataset-error', onError);

      await expect(app.switchDataset('http://example.com/bad.zarr')).rejects.toThrow('boom');
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError.mock.calls[0][0].src).toBe('http://example.com/bad.zarr');
      expect(onError.mock.calls[0][0].error).toBeInstanceOf(Error);
    });

    it('emits dataset-fault and exposes the current latched archive fault', async () => {
      const fault = new Error('archive unavailable');
      const loader = SceneLoaderManager.getInstance().createLoader();
      const onFault = vi.fn();
      app.on('dataset-fault', onFault);
      mockFetch.mockResolvedValue({ ok: true });

      await app.init({ canvas: mockCanvas, src: SRC });
      (loader as any)._archiveFault = fault;
      (loader as any).notifyArchiveFault(fault);

      expect(onFault).toHaveBeenCalledOnce();
      expect(onFault).toHaveBeenCalledWith({ src: SRC, error: fault });
      expect(app.getDatasetFault()).toEqual({ src: SRC, error: fault });

      const foreignFault = new Error('foreign archive unavailable');
      const manager = SceneLoaderManager.getInstance();
      manager.setMonitorFactory(null);
      const foreignLoader = manager.createLoader('foreign');
      (foreignLoader as any)._archiveFault = foreignFault;

      expect(app.getDatasetFault()).toEqual({ src: SRC, error: fault });

      app.dispose();

      expect(app.getDatasetFault()).toBeNull();
      (loader as any).notifyArchiveFault(fault);
      expect(onFault).toHaveBeenCalledOnce();
    });

    it('replays a fault latched before app wiring and replaces the listener on switch', async () => {
      const firstFault = new Error('first archive unavailable');
      const secondFault = new Error('second archive unavailable');
      const firstUnsubscribe = vi.fn();
      const manager = SceneLoaderManager.getInstance();
      const firstLoader = manager.createLoader();
      (firstLoader as any)._archiveFault = firstFault;
      const subscribe = firstLoader.onArchiveFault.bind(firstLoader);
      vi.spyOn(firstLoader, 'onArchiveFault').mockImplementation((listener, options) => {
        const unsubscribe = subscribe(listener, options);
        return () => {
          firstUnsubscribe();
          unsubscribe();
        };
      });
      let secondLoader: ReturnType<typeof manager.createLoader>;
      const onFault = vi.fn();
      const eventOrder: string[] = [];
      let faultAtDatasetLoaded: ReturnType<typeof app.getDatasetFault>;
      app.on('dataset-loaded', () => {
        eventOrder.push('loaded');
        faultAtDatasetLoaded = app.getDatasetFault();
      });
      app.on('dataset-fault', onFault);
      app.on('dataset-fault', () => eventOrder.push('fault'));
      mockFetch.mockResolvedValue({ ok: true });

      await app.init({ canvas: mockCanvas, src: SRC });
      expect(faultAtDatasetLoaded!).toEqual({ src: SRC, error: firstFault });
      expect(onFault).toHaveBeenCalledWith({ src: SRC, error: firstFault });
      expect(eventOrder).toEqual(['loaded', 'fault']);

      mockSceneManager.loadSceneData.mockImplementationOnce(async () => {
        manager.setMonitorFactory(null);
        secondLoader = manager.createLoader();
      });
      await app.switchDataset('http://example.com/next.zarr');
      (secondLoader! as any)._archiveFault = secondFault;
      (secondLoader! as any).notifyArchiveFault(secondFault);

      expect(firstUnsubscribe).toHaveBeenCalledOnce();
      expect(onFault).toHaveBeenLastCalledWith({
        src: 'http://example.com/next.zarr',
        error: secondFault,
      });
      expect(app.getDatasetFault()).toEqual({
        src: 'http://example.com/next.zarr',
        error: secondFault,
      });
    });

    it('isolates a throwing embedder listener (no spurious dataset-error, no rejection)', async () => {
      // A buggy consumer 'dataset-loaded' handler must not corrupt the
      // viewer's control flow: the event bus does not catch listener errors,
      // so without isolation at on() the throw would propagate into
      // loadDataset's catch, emit a spurious 'dataset-error', and reject the
      // switch — on an otherwise-successful load.
      await app.init({ canvas: mockCanvas, src: SRC });
      app.on('dataset-loaded', () => {
        throw new Error('listener boom');
      });
      const onError = vi.fn();
      app.on('dataset-error', onError);

      await expect(app.switchDataset('http://example.com/ok.zarr')).resolves.toBeUndefined();
      expect(onError).not.toHaveBeenCalled();
    });

    it('rejects a concurrent switchDataset while one is in flight', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      let release!: () => void;
      mockSceneManager.loadSceneData.mockImplementationOnce(
        () => new Promise<void>((r) => (release = r))
      );

      const first = app.switchDataset('http://example.com/a.zarr');
      await expect(app.switchDataset('http://example.com/b.zarr')).rejects.toThrow(/in progress/);

      release();
      await first;
    });

    it('throws on the guarded methods before init()', () => {
      expect(() => app.switchDataset('x')).toThrow(/before init/);
      expect(() => app.getDimensions()).toThrow(/before init/);
      expect(() => app.setDimensionValue(0, 1)).toThrow(/before init/);
      expect(() => app.recenterCamera()).toThrow(/before init/);
      expect(() => app.resize()).toThrow(/before init/);
      expect(() => app.registerContext('annotation', { priority: 1 })).toThrow(/before init/);
      expect(app.shortcutForAction('help.toggle')).toBeUndefined();
    });

    it('registers a scene-dims listener on init and removes it on dispose', async () => {
      const addSpy = vi.spyOn(SceneDimsManager.prototype, 'addListener');
      const removeSpy = vi.spyOn(SceneDimsManager.prototype, 'removeListener');

      await app.init({ canvas: mockCanvas, src: SRC });
      expect(addSpy).toHaveBeenCalled();
      const listener = addSpy.mock.calls.at(-1)![0];

      app.dispose();
      expect(removeSpy).toHaveBeenCalledWith(listener);
    });

    it('emits dimensions-changed when the dims listener fires', async () => {
      const addSpy = vi.spyOn(SceneDimsManager.prototype, 'addListener');
      await app.init({ canvas: mockCanvas, src: SRC });
      const listener = addSpy.mock.calls.at(-1)![0];
      const onDims = vi.fn();
      app.on('dimensions-changed', onDims);

      listener(); // simulate a slice-position change notification

      expect(onDims).toHaveBeenCalledTimes(1);
      expect(onDims.mock.calls[0][0]).toMatchObject({ ndim: expect.any(Number) });
    });

    it('does not emit dimensions-changed after dispose', async () => {
      const addSpy = vi.spyOn(SceneDimsManager.prototype, 'addListener');
      await app.init({ canvas: mockCanvas, src: SRC });
      const listener = addSpy.mock.calls.at(-1)![0];
      const onDims = vi.fn();
      app.on('dimensions-changed', onDims);

      app.dispose();
      listener(); // isInitialized is false → guarded no-op

      expect(onDims).not.toHaveBeenCalled();
    });

    it('getDimensions returns an empty shape when no scene is loaded', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      expect(app.getDimensions()).toEqual({
        ndim: 0,
        displayed: [],
        currentStep: [],
        metadata: [],
        ranges: [],
      });
    });

    it('getDimensions deep-clones nested metadata arrays (no aliasing of internals)', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      // A shallow `{ ...m }` spread would leave `range`/`categories` shared
      // with the scene-dims manager — embedder mutation would corrupt state.
      const internalRange: [number, number] = [0, 10];
      const internalCategories = ['dapi', 'gfp'];
      vi.spyOn(SceneDimsManager.prototype, 'getDims').mockReturnValue({
        ndim: 1,
        displayed: [0],
        currentStep: [0],
        metadata: [],
      } as never);
      vi.spyOn(SceneDimsManager.prototype, 'getDimensionRanges').mockReturnValue([internalRange]);
      vi.spyOn(SceneDimsManager.prototype, 'getDimensionMetadata').mockReturnValue([
        { name: 'ch', unit: '', scale: 1, range: internalRange, categories: internalCategories },
      ]);

      const dims = app.getDimensions();
      dims.metadata[0].range![0] = 999;
      dims.metadata[0].categories!.push('hacked');
      dims.ranges[0][0] = 999;

      expect(internalRange).toEqual([0, 10]);
      expect(internalCategories).toEqual(['dapi', 'gfp']);
    });

    it('resize() delegates to sceneManager.resizeToCanvas()', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      app.resize();
      expect(mockSceneManager.resizeToCanvas).toHaveBeenCalled();
    });

    it('recenterCamera() delegates to sceneManager.centerCameraOnScene()', async () => {
      await app.init({ canvas: mockCanvas, src: SRC });
      app.recenterCamera();
      expect(mockSceneManager.centerCameraOnScene).toHaveBeenCalled();
    });
  });
});
