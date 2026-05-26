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
 *   - `renderOnce()` triggers a single startAnimation.
 *
 * Heavy collaborators (SceneLoaderManager, worker-pool, console-interceptor)
 * are mocked at the module boundary — those are external trust boundaries
 * for this helper.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';

// Mock the heavy imports so the helper runs without GPU / zarr.
vi.mock('../../../../../utils/console-interceptor', () => ({
  consoleInterceptor: { patch: vi.fn() },
}));
vi.mock('../../../../../data/scene-loader-manager', () => ({
  SceneLoaderManager: {
    getInstance: vi.fn(() => ({
      getDefaultLoader: vi.fn(() => null),
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
  };
  const inputHandler = { id: 'input' };
  const renderingControls = { id: 'rendering' };
  return {
    debug: true,
    app: { id: 'app' } as never,
    sceneManager: sceneManager as never,
    animationController: animationController as never,
    inputHandler: inputHandler as never,
    renderingControls: renderingControls as never,
    recordingPanel: { id: 'recording' } as never,
    getPickingSystem: () => undefined,
    getOverlayManager: () => undefined,
    isInitialized: () => true,
    ...overrides,
  };
}

describe('installDebugInterface', () => {
  beforeEach(() => {
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
  });

  afterEach(() => {
    delete (window as { __luxarDebug?: unknown }).__luxarDebug;
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
      expect(dbg.scene).toBe(
        (ports.sceneManager as unknown as { scene: THREE.Scene }).scene
      );
      expect(dbg.camera).toBe(
        (ports.sceneManager as unknown as { camera: THREE.Camera }).camera
      );
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

    it('exposes showError so visual-regression specs can drive the error dialog', () => {
      installDebugInterface(makePorts());
      expect(typeof window.__luxarDebug?.showError).toBe('function');
    });
  });

  describe('pre-existing __luxarDebug preservation (bootstrap-seeded fields)', () => {
    it('merges runtime fields into bootstrap-seeded version/app/consoleInterceptor', () => {
      // Simulate bootstrap.ts:198-203 having seeded the debug surface.
      (window as { __luxarDebug?: unknown }).__luxarDebug = {
        app: { id: 'preSeededApp' },
        consoleInterceptor: { patched: true },
        version: '1.0.0',
        showError: vi.fn(),
      };

      installDebugInterface(makePorts());

      const dbg = window.__luxarDebug!;
      // Bootstrap-seeded fields survive the merge…
      expect(dbg.version).toBe('1.0.0');
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
      expect(dbg.version).toBe('1.0.0');
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

    it('renderOnce() invokes animationController.startAnimation', () => {
      const ports = makePorts();
      installDebugInterface(ports);

      const dbg = window.__luxarDebug!;
      dbg.renderOnce!();
      expect(
        (ports.animationController as unknown as { startAnimation: ReturnType<typeof vi.fn> })
          .startAnimation
      ).toHaveBeenCalledOnce();
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

    it('forwards isAnimating from animationController.isActive', () => {
      const ports = makePorts();
      (ports.animationController as unknown as { isActive: boolean }).isActive = true;
      installDebugInterface(ports);
      const dbg = window.__luxarDebug!;
      const state = dbg.getState!() as { isAnimating?: boolean };
      expect(state.isAnimating).toBe(true);
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
      }));

      installDebugInterface(makePorts());
      const dbg = window.__luxarDebug!;
      expect(typeof dbg.injectSyntheticScene).toBe('function');

      // Awaiting callers must see the rejection re-thrown.
      await expect(
        dbg.injectSyntheticScene!({ type: 'lines', count: 5 })
      ).rejects.toThrow(/simulated bundle-load failure/);

      // AND the rejection must have been surfaced via the user-facing
      // error overlay so silent-await callers (typical debug-URL flow)
      // still see something.
      expect(showError).toHaveBeenCalledWith(
        expect.stringMatching(/Synthetic-scene injection failed.*simulated bundle-load failure/)
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
});
