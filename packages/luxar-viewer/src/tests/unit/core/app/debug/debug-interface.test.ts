// @vitest-environment jsdom
/**
 * Unit tests for core/app/debug/debug-interface.ts (G11).
 *
 * `installDebugInterface` populates `window.__luxarDebug` with runtime
 * fields. Contract under test:
 *   - When `ports.debug === false`, the function is a no-op (no
 *     mutation of window.__luxarDebug).
 *   - When `ports.debug === true` and bootstrap has already seeded
 *     `__luxarDebug`, the runtime fields are MERGED into the existing
 *     object (preserves app/consoleInterceptor/version/showError).
 *   - When `ports.debug === true` and bootstrap has NOT seeded, the
 *     helper builds a fresh base with app/consoleInterceptor/version.
 *   - Live-accessor fields (`getPickingSystem`, `getOverlayManager`)
 *     reflect the CURRENT result of the supplied port closures, not
 *     a snapshot at install time.
 *   - `getState()` returns the current scene/dim snapshot.
 *   - `renderOnce()` triggers a single animationController.renderOnce().
 *
 * Heavy collaborators (SceneLoaderManager, worker-pool, console-interceptor)
 * are mocked at the module boundary — those are external trust boundaries
 * for this helper.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import { VIEWER_VERSION } from '../../../../../version';

// Mock the heavy imports so the helper runs without GPU / zarr.
vi.mock('../../../../../utils/console-interceptor', () => ({
  consoleInterceptor: { patch: vi.fn() },
}));
// Mutable stand-in for the manager's live answers, so a test can flip one
// BETWEEN two getState() calls and prove the field is read per snapshot.
const managerState = vi.hoisted(() => ({
  loadPassInProgress: false,
  poolStats: undefined as Record<string, number> | undefined,
  residencyStop: undefined as Record<string, unknown> | undefined,
}));
vi.mock('../../../../../data/scene-loader-manager', () => ({
  SceneLoaderManager: {
    getInstance: vi.fn(() => ({
      getDefaultLoader: vi.fn(() => null),
      isAnyLoadPassInProgress: vi.fn(() => managerState.loadPassInProgress),
      gpuPoolStats: vi.fn(() => managerState.poolStats),
      refinementResidencyStop: vi.fn(() => managerState.residencyStop),
    })),
  },
}));
vi.mock('../../../../../workers/worker-pool', () => ({
  getWorkerPool: vi.fn(() => ({
    getQueueDepth: vi.fn(() => 7),
    getStats: vi.fn(() => ({ inFlight: 1, completed: 99 })),
  })),
}));
vi.mock('../../../../../ui/error-overlay', () => ({
  showError: vi.fn(),
}));
vi.mock('../../../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    getDims: vi.fn(() => null),
  },
}));
// Spy on log; the helper logs noisy info messages we want to silence.
vi.mock('../../../../../utils/log', () => ({
  log: {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    custom: vi.fn(),
  },
  Modules: { LUXAR: 'LUXAR' },
  LogEmoji: {},
}));

import { installDebugInterface } from '../../../../../core/app/debug/debug-interface';

function makePorts(overrides: Partial<Parameters<typeof installDebugInterface>[0]> = {}) {
  // computeDebugState (called from getState) does `ctx.scene.traverse`
  // and reads `ctx.camera.position`, so both must be real THREE
  // objects. Other fields are stored as opaque references.
  const sceneManager = {
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(60, 1, 0.1, 1000),
    renderer: { id: 'renderer' },
    controls: { id: 'controls' },
    postProcessing: { id: 'pp' },
    currentFov: 60,
  };
  const animationController = {
    isActive: false,
    startAnimation: vi.fn(),
    renderOnce: vi.fn(),
  };
  const inputHandler = { id: 'input' };
  const renderingControls = { id: 'rendering' };
  const adaptiveDPRManager = {
    getCurrentFPS: vi.fn(() => 60),
    getState: vi.fn(() => ({ enabled: true })),
  };
  return {
    debug: true,
    app: { id: 'app', shortcutForAction: vi.fn().mockReturnValue('F1') } as never,
    sceneManager: sceneManager as never,
    animationController: animationController as never,
    inputHandler: inputHandler as never,
    renderingControls: renderingControls as never,
    adaptiveDPRManager: adaptiveDPRManager as never,
    recordingPanel: { id: 'recording' } as never,
    getPickingSystem: () => undefined,
    getOverlayManager: () => undefined,
    isInitialized: () => true,
    ...overrides,
  };
}

describe('installDebugInterface', () => {
  const resetManagerState = (): void => {
    managerState.loadPassInProgress = false;
    managerState.poolStats = undefined;
    managerState.residencyStop = undefined;
  };

  beforeEach(() => {
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
    resetManagerState();
  });

  afterEach(() => {
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
    resetManagerState();
  });

  describe('debug=false short-circuit', () => {
    it('does NOT mutate window.__luxarDebug when ports.debug is false', () => {
      installDebugInterface(makePorts({ debug: false }));
      expect(window.__luxarDebug).toBeUndefined();
    });

    it('preserves a pre-seeded __luxarDebug when debug=false (no overwrite)', () => {
      (window as { __luxarDebug?: unknown }).__luxarDebug = { preExisting: 'value' };
      installDebugInterface(makePorts({ debug: false }));
      expect((window.__luxarDebug as unknown as { preExisting: string }).preExisting).toBe('value');
    });
  });

  describe('runtime field installation', () => {
    it('attaches scene/camera/renderer/controls/postProcessing from sceneManager', () => {
      const ports = makePorts();
      installDebugInterface(ports);

      expect(window.__luxarDebug).toBeDefined();
      const dbg = window.__luxarDebug!;
      // scene + camera are real THREE objects — identity checks.
      expect(dbg.scene).toBe((ports.sceneManager as unknown as { scene: THREE.Scene }).scene);
      expect(dbg.camera).toBe((ports.sceneManager as unknown as { camera: THREE.Camera }).camera);
      expect(dbg.renderer).toEqual({ id: 'renderer' });
      expect(dbg.controls).toEqual({ id: 'controls' });
      expect(dbg.postProcessing).toEqual({ id: 'pp' });
    });

    it('attaches the animation controller, input handler, rendering controls, recording panel', () => {
      installDebugInterface(makePorts());
      const dbg = window.__luxarDebug!;
      expect(dbg.animationController).toBeDefined();
      expect(dbg.inputHandler).toEqual({ id: 'input' });
      expect(dbg.renderingControls).toEqual({ id: 'rendering' });
      expect(dbg.recordingPanel).toEqual({ id: 'recording' });
    });

    it('exposes runtimeReady=true to signal init is complete', () => {
      // Playwright + visual-regression specs poll on this flag to know
      // the runtime is wired (vs the bootstrap-seeded pre-init shape).
      installDebugInterface(makePorts());
      expect(window.__luxarDebug?.runtimeReady).toBe(true);
    });

    it('exposes workers.getQueueDepth + workers.getStats as live accessors', () => {
      installDebugInterface(makePorts());
      const dbg = window.__luxarDebug!;
      // Both are functions, not snapshot values.
      expect(typeof dbg.workers?.getQueueDepth).toBe('function');
      expect(typeof dbg.workers?.getStats).toBe('function');
      // Live read through the mocked worker pool.
      expect(dbg.workers!.getQueueDepth()).toBe(7);
      expect(dbg.workers!.getStats()).toEqual({ inFlight: 1, completed: 99 });
    });

    it('exposes showError so visual-regression specs can drive the error dialog', async () => {
      const { showError } = await import('../../../../../ui/error-overlay');
      const showErrorMock = vi.mocked(showError);
      const ports = makePorts();
      installDebugInterface(ports);
      window.__luxarDebug!.showError!('Synthetic failure');

      expect(showErrorMock).toHaveBeenCalledWith('Synthetic failure', expect.any(Function), {
        datasetBrowser: 'dataset-browser.toggle',
        help: 'help.toggle',
      });
      const shortcutForAction = showErrorMock.mock.calls[0][1]!;
      expect(shortcutForAction('help.toggle')).toBe('F1');
    });
  });

  describe('pre-existing __luxarDebug preservation (bootstrap-seeded fields)', () => {
    it('merges runtime fields into bootstrap-seeded version/app/consoleInterceptor', () => {
      // Simulate bootstrap.ts:198-203 having seeded the debug surface.
      (window as { __luxarDebug?: unknown }).__luxarDebug = {
        app: { id: 'preSeededApp' },
        consoleInterceptor: { patched: true },
        // An arbitrary sentinel — the point is that whatever bootstrap put
        // here survives the merge, not what the value is.
        version: 'seeded-by-bootstrap',
        showError: vi.fn(),
      };

      installDebugInterface(makePorts());

      const dbg = window.__luxarDebug!;
      // Bootstrap-seeded fields survive the merge…
      expect(dbg.version).toBe('seeded-by-bootstrap');
      expect((dbg.consoleInterceptor as unknown as { patched: boolean }).patched).toBe(true);
      // …but the new app reference takes precedence (helper sets
      // ports.app last, so it overrides the pre-seeded one).
      expect((dbg.app as unknown as { id: string }).id).toBe('app');
      // Runtime fields are now attached (scene is a real THREE.Scene).
      expect(dbg.scene).toBeInstanceOf(THREE.Scene);
      expect(dbg.runtimeReady).toBe(true);
    });

    it('builds a fresh base when nothing was pre-seeded (tests/embedders)', () => {
      // No bootstrap seeding — __luxarDebug doesn't exist before install.
      expect(window.__luxarDebug).toBeUndefined();

      installDebugInterface(makePorts());

      const dbg = window.__luxarDebug!;
      // Fresh base: app + consoleInterceptor + version are still
      // populated even though bootstrap never ran.
      expect(dbg.app).toBeDefined();
      expect(dbg.consoleInterceptor).toBeDefined();
      // The fresh base reports the package version (`VIEWER_VERSION`), not a
      // hardcoded constant.
      expect(dbg.version).toBe(VIEWER_VERSION);
    });
  });

  describe('live accessor functions', () => {
    it('getPickingSystem() / getOverlayManager() reflect the LIVE port closure', () => {
      // The picking system + overlay manager are reset between dataset
      // loads, so the accessors must NOT capture an initial snapshot.
      let pickingSystem: unknown = undefined;
      let overlayManager: unknown = undefined;
      installDebugInterface(
        makePorts({
          getPickingSystem: () => pickingSystem as never,
          getOverlayManager: () => overlayManager as never,
        })
      );

      const dbg = window.__luxarDebug!;
      // Initial: both undefined.
      expect(dbg.getPickingSystem!()).toBeUndefined();
      expect(dbg.getOverlayManager!()).toBeUndefined();

      // Simulate a dataset load that built new instances.
      pickingSystem = { id: 'picking' };
      overlayManager = { id: 'overlay' };
      expect(dbg.getPickingSystem!()).toEqual({ id: 'picking' });
      expect(dbg.getOverlayManager!()).toEqual({ id: 'overlay' });
    });

    it('renderOnce() invokes animationController.renderOnce (not startAnimation)', () => {
      const ports = makePorts();
      installDebugInterface(ports);

      const dbg = window.__luxarDebug!;
      dbg.renderOnce!();
      const anim = ports.animationController as unknown as {
        renderOnce: ReturnType<typeof vi.fn>;
        startAnimation: ReturnType<typeof vi.fn>;
      };
      expect(anim.renderOnce).toHaveBeenCalledOnce();
      expect(anim.startAnimation).not.toHaveBeenCalled();
    });
  });

  describe('getState()', () => {
    it('returns the current scene snapshot via computeDebugState', () => {
      installDebugInterface(makePorts());
      const dbg = window.__luxarDebug!;
      const state = dbg.getState!();
      // The state is whatever computeDebugState returns; the helper
      // forwards its fields. We only care that getState exists and is
      // callable here — the underlying computeDebugState is unit-tested
      // in debug-state.test.ts.
      expect(state).toBeDefined();
      expect(typeof state).toBe('object');
    });

    // #1639 — the snapshot never carried the `isLoading` flag the E2E
    // data-wait helpers have always polled, so `!state.isLoading` was
    // `!undefined` and every one of them resolved on its first poll. Pin the
    // WIRING here (the pure helper's own forwarding is covered in
    // debug-state.test.ts): the field must come from the loader manager and be
    // re-read on EVERY snapshot, so a captured-once refactor fails.
    it('sources isLoading from SceneLoaderManager on each snapshot', () => {
      installDebugInterface(makePorts());
      const dbg = window.__luxarDebug!;

      expect((dbg.getState!() as { isLoading?: boolean }).isLoading).toBe(false);

      // A load starts AFTER install: a value captured at install time would
      // still report false here.
      managerState.loadPassInProgress = true;
      expect((dbg.getState!() as { isLoading?: boolean }).isLoading).toBe(true);

      managerState.loadPassInProgress = false;
      expect((dbg.getState!() as { isLoading?: boolean }).isLoading).toBe(false);
    });

    // #2508 — the same wiring lesson as `isLoading`, for the two signals a
    // capture tool refuses on. `gpuPoolStats` had been declared on the context
    // and documented for as long as the field existed while production never
    // passed it, so `state.gpuPool` was permanently undefined; the residency
    // stop had no snapshot representation at all and lived only in a console
    // warning. Both must be read from the manager on EVERY snapshot.
    it('sources gpuPool and refinementResidency from SceneLoaderManager per snapshot', () => {
      installDebugInterface(makePorts());
      const dbg = window.__luxarDebug!;
      type CapState = { gpuPool?: Record<string, number>; refinementResidency?: unknown };

      // Before anything has happened: absent, NOT a synthesised zero record —
      // `capture-readiness.ts` treats a present residency record as the stop.
      const idle = dbg.getState!() as CapState;
      expect(idle.gpuPool).toBeUndefined();
      expect(idle.refinementResidency).toBeUndefined();

      // A FULL `PoolStats`, not just the eight projected fields: the manager
      // really does return the whole record, and a mock that pre-projects it
      // makes the projection assertion below unfalsifiable — replacing the
      // whole projection with `return stats` would pass.
      managerState.poolStats = {
        activeBuffers: 2,
        pooledBuffers: 1,
        activeBytes: 2048,
        pooledBytes: 512,
        totalBytes: 2560,
        largestPooledBytes: 512,
        evictions: 9,
        byteBudgetEvictions: 4,
        allocations: 11,
        reuses: 5,
        capacityGrowths: 1,
        deferredEvictions: 2,
        byType: {
          points: { allocations: 1, reuses: 0, evictions: 0 },
          lines: { allocations: 0, reuses: 0, evictions: 0 },
          gsplats: { allocations: 0, reuses: 0, evictions: 0 },
        },
      } as unknown as Record<string, number>;
      managerState.residencyStop = { declinedPathCount: 6, firstPath: '/galaxies/bright' };

      const stopped = dbg.getState!() as CapState;
      expect(stopped.gpuPool?.byteBudgetEvictions).toBe(4);
      expect(stopped.gpuPool?.evictions).toBe(9);
      expect(stopped.refinementResidency).toEqual({
        declinedPathCount: 6,
        firstPath: '/galaxies/bright',
      });
      // The pool read is a PROJECTION onto the debug subset: the EXACT key set,
      // so neither a dropped field nor a forwarded extra can slip through. The
      // snapshot crosses a `page.evaluate` boundary, and `byType` alone would
      // multiply what every `getState()` poll serialises.
      expect(Object.keys(stopped.gpuPool ?? {}).sort()).toEqual(
        [
          'activeBuffers',
          'pooledBuffers',
          'activeBytes',
          'pooledBytes',
          'totalBytes',
          'largestPooledBytes',
          'evictions',
          'byteBudgetEvictions',
        ].sort()
      );
    });

    it('forwards isAnimating from animationController.isActive', () => {
      const ports = makePorts();
      (ports.animationController as unknown as { isActive: boolean }).isActive = true;
      installDebugInterface(ports);
      const dbg = window.__luxarDebug!;
      const state = dbg.getState!() as { isAnimating?: boolean };
      expect(state.isAnimating).toBe(true);
    });
  });

  describe('fps', () => {
    it('exposes frame cadence, sampling state, and an empty window distinctly', () => {
      const getCurrentFPS = vi.fn(() => 144);
      const getState = vi.fn(() => ({ enabled: true }));
      installDebugInterface(
        makePorts({ adaptiveDPRManager: { getCurrentFPS, getState } as never })
      );

      expect(window.__luxarDebug?.fps).toBe(144);
      expect(window.__luxarDebug?.fpsSamplingEnabled).toBe(true);
      getCurrentFPS.mockReturnValue(0);
      expect(window.__luxarDebug?.fps).toBeUndefined();
      getState.mockReturnValue({ enabled: false });
      expect(window.__luxarDebug?.fpsSamplingEnabled).toBe(false);
    });
  });

  describe('cache helpers', () => {
    it('exposes a cache namespace built from SceneLoaderManager.getDefaultLoader', () => {
      installDebugInterface(makePorts());
      const dbg = window.__luxarDebug!;
      expect(dbg.cache).toBeDefined();
      // Specific helper functions are unit-tested in debug-cache-helpers.test.ts;
      // here we only confirm the wiring exists.
      expect(typeof dbg.cache).toBe('object');
    });
  });

  // [core OOS] injectSyntheticScene wraps its dynamic-import chain in
  // try/catch. Pre-fix, a rejected dynamic import (e.g. missing code-
  // split chunk, transient network failure) became an unhandled promise
  // rejection — debug consumers typically don't await with their own
  // try/catch, and a URL like `?debug=1&inject=lines` could leave the
  // page broken with no visible signal. Now the rejection is surfaced
  // via showError + log.error AND re-thrown so awaiting callers see it.
  describe('injectSyntheticScene error handling', () => {
    it('rejects with the underlying error AND surfaces it via showError + log.error', async () => {
      const { showError } = await import('../../../../../ui/error-overlay');
      const { log } = await import('../../../../../utils/log');
      // Force the synthetic-scene module to throw at use-time (when
      // `generateSyntheticLines(spec)` is called). The helper's try/catch
      // around the dynamic-import chain + invocation should convert the
      // throw into a structured user-facing error AND re-throw.
      vi.doMock('../../../../../scene/synthetic-scene', () => ({
        generateSyntheticLines: () => {
          throw new Error('simulated bundle-load failure');
        },
        // The injector destructures this alongside the generator (it
        // sizes the line primitive from the generation volume), and
        // vitest throws on an export a mock factory omits.
        syntheticLinesBoundsDiagonal: () => 1,
      }));

      installDebugInterface(makePorts());
      const dbg = window.__luxarDebug!;
      expect(typeof dbg.injectSyntheticScene).toBe('function');

      // Awaiting callers must see the rejection re-thrown.
      await expect(dbg.injectSyntheticScene!({ type: 'lines', count: 5 })).rejects.toThrow(
        /simulated bundle-load failure/
      );

      // AND the rejection must have been surfaced via the user-facing
      // error overlay so silent-await callers (typical debug-URL flow)
      // still see something.
      expect(showError).toHaveBeenCalledWith(
        expect.stringMatching(/Synthetic-scene injection failed.*simulated bundle-load failure/),
        expect.any(Function),
        { datasetBrowser: 'dataset-browser.toggle', help: 'help.toggle' }
      );
      // AND log.error must record the failure with module + payload.
      expect(log.error).toHaveBeenCalledWith(
        'LUXAR',
        expect.stringMatching(/injectSyntheticScene failed.*simulated bundle-load failure/),
        expect.any(Error)
      );

      vi.doUnmock('../../../../../scene/synthetic-scene');
    });
  });

  // The injected node is the perf bench's `default`-arm subject, so it
  // must be sized by the SAME auto rule (#1352) a compiled node is:
  // authored count × a rendered-width factor normalized by the node's
  // extent. Count-only sizing would let a wide swept arm benchmark the
  // capsule where production builds the quad.
  describe('injectSyntheticScene line-primitive sizing', () => {
    /** A wide-lines spec whose WIDTH alone carries it over the threshold. */
    const wide = { type: 'lines' as const, count: 40_000, bounds: 10, width: 3 };

    it('crosses the auto threshold on rendered width, far below the count threshold', async () => {
      installDebugInterface(makePorts());
      const result = await window.__luxarDebug!.injectSyntheticScene!(wide);
      const material = (result.mesh as THREE.Mesh).material as THREE.Material;
      expect(material.userData.linePrimitive).toBe('screen-space');
    });

    it('keeps the capsule for the same count at a sub-clamp width', async () => {
      // Same 40 k count, width far below the shader's 1.5 px floor — the
      // negative control, without which "reads the width" would also be
      // satisfied by flipping every injected node to the quad.
      installDebugInterface(makePorts());
      const result = await window.__luxarDebug!.injectSyntheticScene!({ ...wide, width: 0.02 });
      const material = (result.mesh as THREE.Mesh).material as THREE.Material;
      expect(material.userData.linePrimitive).toBe('capsule');
    });
  });
});
