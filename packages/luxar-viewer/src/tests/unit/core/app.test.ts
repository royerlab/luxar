/**
 * Tests for LuxarApp - the main application coordinator
 *
 * These tests verify the initialization sequence, component integration,
 * dataset detection logic, error handling, and cleanup of the main app.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// NOTE: This test file mocks 9 internal modules (below). It primarily
// verifies initialization ordering + cross-wiring; component behavior
// is covered by per-module tests.
//
// Phase 21C audit: the original plan was to add an `AppFactories`
// interface to `LuxarAppOptions` so tests could inject stubs without
// `vi.mock` indirection. On close inspection the trade-off doesn't
// pay: every `new X(...)` site in `app.ts:init()` would gain a
// `(this.options.factories?.X ?? defaultFactories.X)(...)` indirection,
// the test stubs would relocate from `vi.mock` to factory functions
// (same complexity, different shape), and the tests' constructor-call
// assertions (e.g. `expect(AnimationController).toHaveBeenCalledWith
// (mockSceneManager.controls, mockSceneManager.postProcessing)`) would
// need to be rewritten to assert on factory spies. Net: production
// code reads worse, test code reads ~the same, no new capability.
//
// Per the standing "no complexification" rule, the DI refactor is
// deferred. If a future need (e.g. a non-mocked embedding test) makes
// `vi.mock` actually painful, factory injection can be added per-
// component without churning the rest.

// Mock all dependencies before importing LuxarApp
vi.mock('../../../scene/scene-manager');
vi.mock('../../../scene/animation-controller');
vi.mock('../../../input/input-handler');
vi.mock('../../../ui/rendering-controls');
vi.mock('../../../ui/recording-panel');
vi.mock('../../../ui/components/scale-bar');
vi.mock('../../../ui/panels/dataset-browser');
vi.mock('../../../ui/helpers');
vi.mock('../../../ui/layers');
// Phase 17A.4: scene-dims-manager unmocked. It's a pure JS singleton
// (no DOM or WebGL); running it real in app.test improves coverage
// of the dim-init wiring without affecting jsdom behavior.
// Phase 8.6 migrated PerformanceMonitor and DebugConsole ownership
// from AnimationController / InputHandler to LuxarApp. Mock both here
// so stats.js / DebugConsole's document.createElement calls don't run
// in the stubbed-window env.
vi.mock('../../../ui/monitors/performance-monitor', () => ({
  PerformanceMonitor: vi.fn().mockImplementation(() => ({
    show: vi.fn(),
    hide: vi.fn(),
    toggle: vi.fn(),
    cyclePanels: vi.fn(),
    dispose: vi.fn(),
    visible: false,
  })),
}));
vi.mock('../../../ui/panels/debug-console', () => ({
  DebugConsole: vi.fn().mockImplementation(() => ({
    show: vi.fn(),
    hide: vi.fn(),
    toggle: vi.fn(),
    dispose: vi.fn(),
    getIsVisible: vi.fn(() => false),
  })),
}));

// Setup global mocks
const mockAddEventListener = vi.fn();
const mockRemoveEventListener = vi.fn();
const mockReplaceState = vi.fn();

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
});

// Mock fetch for dataset detection
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

// Import mocked classes
import { SceneManager } from '../../../scene/scene-manager';
import { AnimationController } from '../../../scene/animation-controller';
import { InputHandler } from '../../../input/input-handler';
import { RenderingControls } from '../../../ui/rendering-controls';
import { DatasetBrowser } from '../../../ui/panels/dataset-browser';
import { cleanupUI as mockCleanupUI, clearError as mockClearError } from '../../../ui/helpers';

// Import LuxarApp after all mocks are set up
import { LuxarApp } from '../../../core/app';

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

    // Reset mock implementations
    mockSceneManager = {
      init: vi.fn().mockResolvedValue(undefined),
      loadSceneData: vi.fn().mockResolvedValue(undefined),
      updateDynamicClippingPlanes: vi.fn(),
      getSceneViewerConfig: vi.fn().mockReturnValue(undefined),
      dispose: vi.fn(),
      renderer: { domElement: {} },
      scene: {},
      camera: {},
      controls: {},
      postProcessing: {},
    };

    mockAnimationController = {
      startAnimation: vi.fn(),
      stopAnimation: vi.fn(),
      addPerFrameCallback: vi.fn(),
      removePerFrameCallback: vi.fn(),
      setAdaptiveDPRManager: vi.fn(),
      setContextLostPredicate: vi.fn(),
      dispose: vi.fn(),
      isActive: false,
    };

    mockInputHandler = {
      init: vi.fn(),
      setRenderingControls: vi.fn(),
      setScaleBar: vi.fn(),
      setRecordingPanel: vi.fn(),
      setLayersPanel: vi.fn(),
      setDatasetBrowser: vi.fn(),
      clearDimensionUI: vi.fn(),
      initDimensionSliders: vi.fn(),
      dispose: vi.fn(),
    };

    mockRenderingControls = {
      setAnimationController: vi.fn(),
      setAdaptiveDPRManager: vi.fn(),
      setSceneId: vi.fn(),
      setZarrViewerConfig: vi.fn(),
      hasStoredSettings: vi.fn().mockReturnValue(false),
      applyZarrDefaults: vi.fn(),
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
  });

  describe('initialization sequence', () => {
    it('should initialize all components in correct order', async () => {
      mockFetch.mockResolvedValue({ ok: true }); // Valid zarr dataset
      const initOrder: string[] = [];

      mockSceneManager.init.mockImplementation(async () => {
        initOrder.push('sceneManager');
      });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(initOrder).toContain('sceneManager');
      expect(SceneManager).toHaveBeenCalled();
      expect(AnimationController).toHaveBeenCalled();
      expect(InputHandler).toHaveBeenCalled();
      expect(RenderingControls).toHaveBeenCalled();
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

      // Phase 8.6: InputHandler receives the PerformanceMonitor and
      // DebugConsole (both constructed at app level rather than in
      // AnimationController / InputHandler).
      // Phase 8.6.d: also receives a DimensionSlidersFactory function,
      // injected so the input layer never imports the concrete UI panel
      // (closes the input → ui layer-cruiser exception).
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

    it('should cross-link components properly', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(mockRenderingControls.setAnimationController).toHaveBeenCalledWith(
        mockAnimationController
      );
      expect(mockInputHandler.setRenderingControls).toHaveBeenCalledWith(mockRenderingControls);
    });

    it('should start animation loop before loading data', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      const callOrder: string[] = [];

      mockAnimationController.startAnimation.mockImplementation(() => {
        callOrder.push('startAnimation');
      });
      mockSceneManager.loadSceneData.mockImplementation(async () => {
        callOrder.push('loadSceneData');
      });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(callOrder.indexOf('startAnimation')).toBeLessThan(callOrder.indexOf('loadSceneData'));
    });

    it('should set isInitialized to true after successful init', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(app.initialized).toBe(true);
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

    it('should load directly for valid zarr datasets', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(mockFetch).toHaveBeenCalledWith(
        'http://example.com/data.zarr/.zgroup',
        expect.objectContaining({ method: 'HEAD' })
      );
      expect(mockSceneManager.loadSceneData).toHaveBeenCalledWith(
        'http://example.com/data.zarr',
        undefined
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
      mockFetch.mockRejectedValue(new Error('Network error'));
      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      // Should continue with browser or direct load based on extension check
      expect(app.initialized).toBe(true);
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
      expect(components.inputHandler).toBe(mockInputHandler);
      expect(components.renderingControls).toBe(mockRenderingControls);
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

    it('should continue if data loading fails', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      mockSceneManager.loadSceneData.mockRejectedValue(new Error('Load failed'));

      // Should throw because loadSceneData error propagates
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

    it('should remove beforeunload listener', () => {
      app.dispose();

      expect(mockRemoveEventListener).toHaveBeenCalledWith('beforeunload', expect.any(Function));
    });

    it('should set isInitialized to false', () => {
      app.dispose();

      expect(app.initialized).toBe(false);
    });

    it('should handle dispose errors gracefully', () => {
      mockSceneManager.dispose.mockImplementation(() => {
        throw new Error('Dispose failed');
      });

      expect(() => app.dispose()).not.toThrow();
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
      // Pre-existing static singletons that pre-date ManagerRegistry.
      // Without explicit disposeInstance() calls, their loaders + cache stores
      // + eventBus subscriptions survive across LuxarApp re-init.
      const sceneLoaderModule = await import('../../../data/scene-loader-manager');
      const dataMonitorModule = await import('../../../ui/monitors/data-monitor-manager');

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
      // Phase 13.1: Pre-existing dispose() wrapped everything in one
      // try/catch, so a throw early in the chain (sceneManager etc.)
      // skipped DataMonitorManager / SceneLoaderManager / disposeWorkerPool
      // / managerRegistry. The safeDispose helper guarantees later
      // teardown runs regardless.
      const sceneLoaderModule = await import('../../../data/scene-loader-manager');
      const dataMonitorModule = await import('../../../ui/monitors/data-monitor-manager');
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
      const dataMonitorModule = await import('../../../ui/monitors/data-monitor-manager');
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

    it('closes an open DatasetBrowser and clears app + input-handler refs (Phase 14.5)', () => {
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
      expect(
        (app as unknown as { datasetBrowser: unknown }).datasetBrowser
      ).toBeUndefined();
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

  describe('initialization order details', () => {
    it('should initialize scene manager before creating animation controller', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      const initOrder: string[] = [];

      mockSceneManager.init.mockImplementation(async () => {
        initOrder.push('sceneManager.init');
      });

      (AnimationController as any).mockImplementation(() => {
        initOrder.push('AnimationController');
        return mockAnimationController;
      });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      expect(initOrder.indexOf('sceneManager.init')).toBeLessThan(
        initOrder.indexOf('AnimationController')
      );
    });

    it('should create all managers before starting animation', async () => {
      mockFetch.mockResolvedValue({ ok: true });
      const initOrder: string[] = [];

      (SceneManager as any).mockImplementation(() => {
        initOrder.push('SceneManager');
        return mockSceneManager;
      });
      (AnimationController as any).mockImplementation(() => {
        initOrder.push('AnimationController');
        return mockAnimationController;
      });
      (InputHandler as any).mockImplementation(() => {
        initOrder.push('InputHandler');
        return mockInputHandler;
      });
      (RenderingControls as any).mockImplementation(() => {
        initOrder.push('RenderingControls');
        return mockRenderingControls;
      });
      mockAnimationController.startAnimation.mockImplementation(() => {
        initOrder.push('startAnimation');
      });

      await app.init({ canvas: mockCanvas, src: 'http://example.com/data.zarr' });

      const startAnimationIndex = initOrder.indexOf('startAnimation');
      expect(initOrder.indexOf('SceneManager')).toBeLessThan(startAnimationIndex);
      expect(initOrder.indexOf('AnimationController')).toBeLessThan(startAnimationIndex);
      expect(initOrder.indexOf('InputHandler')).toBeLessThan(startAnimationIndex);
      expect(initOrder.indexOf('RenderingControls')).toBeLessThan(startAnimationIndex);
    });
  });
});
