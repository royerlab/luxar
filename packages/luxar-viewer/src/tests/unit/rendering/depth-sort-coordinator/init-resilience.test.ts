/**
 * SortWorker STARTUP: when it happens, on whose deadline, and how the two
 * failure arms the shared guard adds are classified.
 *
 * The retry/give-up/epoch behaviour of a STARVED init lives in
 * `src/tests/unit/rendering/depth-sort-coordinator.test.ts` and is
 * deliberately not repeated here. What this file pins is the rest of the
 * startup contract:
 * - the worker is spawned + initialized at APP INIT (`warmUpDepthSortWorker`),
 *   while nothing is loading — the first order-dependent commit is the worst
 *   possible moment, because it lands exactly when this thread is saturated
 *   decoding and the worker's Comlink reply has to be dispatched on it;
 * - the deadline is `config.depthSort.workerInitTimeoutMs`, not a constant;
 * - `onmessageerror` (which the coordinator's hand-rolled race never had)
 *   settles init, permanently;
 * - `isDepthSortAvailable()` — what the data-loading monitor's footer reads —
 *   says "unavailable" only once the subsystem has actually given up.
 *
 * Split into its own file (rather than added to depth-sort-coordinator.test.ts)
 * because it is the only one here that needs FAKE TIMERS, which do not mix
 * with that file's frame-pump helpers — `src/tests/setup.ts` backs the rAF
 * mock with real `setTimeout`. Mirrors the shape of
 * `src/tests/unit/workers/worker-pool/lifecycle/`.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import type { AppConfig } from '../../../../config/types';

/** How `initialize()` behaves for a given test. */
type InitBehavior = 'immediate' | 'never-settles' | 'rejects';

const terminatedWorkers: unknown[] = [];
let constructedWorkers: unknown[] = [];
let mockApi: {
  initialize: ReturnType<typeof vi.fn>;
  registerNode: ReturnType<typeof vi.fn>;
  sort: ReturnType<typeof vi.fn>;
  releaseNode: ReturnType<typeof vi.fn>;
  releaseAllNodes: ReturnType<typeof vi.fn>;
};
/**
 * The config instance the freshly imported coordinator actually reads. It
 * must come from the SAME module registry generation, so it is captured in
 * `loadCoordinator` rather than imported at the top of this file — a
 * `vi.resetModules()` would otherwise leave the two looking at different
 * objects and any knob written here would be invisible to the coordinator.
 */
let liveConfig: AppConfig;

function makeMockApi(behavior: InitBehavior) {
  return {
    initialize: vi.fn(() => {
      if (behavior === 'immediate') return Promise.resolve({ wasmFallback: true });
      if (behavior === 'rejects') return Promise.reject(new Error('WASM unavailable'));
      return new Promise<{ wasmFallback: boolean }>(() => {});
    }),
    registerNode: vi.fn(async () => undefined),
    // Never settles: these tests only assert that a sort was DISPATCHED.
    sort: vi.fn(() => new Promise(() => {})),
    releaseNode: vi.fn(async () => undefined),
    releaseAllNodes: vi.fn(async () => undefined),
  };
}

async function loadCoordinator(behavior: InitBehavior = 'never-settles') {
  vi.resetModules();
  terminatedWorkers.length = 0;
  constructedWorkers = [];
  // Drop the previous test's worker handle: `__lastMockWorker` is global while
  // the coordinator is re-imported per test, so a stale one would let a test
  // fire an event into the PREVIOUS module instance and assert nothing.
  delete (globalThis as unknown as { __lastMockWorker?: unknown }).__lastMockWorker;
  mockApi = makeMockApi(behavior);

  vi.doMock('../../../../utils/log', () => ({
    log: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
    Modules: new Proxy({}, { get: (_t, p) => String(p) }),
  }));
  vi.doMock('comlink', () => ({
    wrap: vi.fn(() => mockApi),
    transfer: vi.fn((value: unknown) => value),
  }));
  vi.doMock('../../../../workers/sort-worker?worker', () => ({
    default: class MockSortWorker {
      onerror: ((e: unknown) => void) | null = null;
      onmessageerror: ((e: unknown) => void) | null = null;
      constructor() {
        constructedWorkers.push(this);
        (globalThis as unknown as { __lastMockWorker?: unknown }).__lastMockWorker = this;
      }
      terminate = vi.fn(() => terminatedWorkers.push(this));
    },
  }));

  const coord = await import('../../../../rendering/depth-sort-coordinator');
  liveConfig = (await import('../../../../config')).config;
  return coord;
}

/** Drain microtasks so ensureWorker → registerNode → scheduleSort settles. */
async function flush(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

/** The live mocked worker (for firing worker-level events). */
function liveWorker(): {
  onerror?: ((e: unknown) => void) | null;
  onmessageerror?: ((e: unknown) => void) | null;
} {
  return (
    globalThis as unknown as {
      __lastMockWorker: {
        onerror?: ((e: unknown) => void) | null;
        onmessageerror?: ((e: unknown) => void) | null;
      };
    }
  ).__lastMockWorker;
}

function makeCamera(): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera();
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  return camera;
}

/** A minimal order-dependent points mesh with real ordering attributes. */
function makeSortableMesh(count: number, blendingMode = 'volumetric'): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute(
    'aSortedIndex',
    new THREE.InstancedBufferAttribute(new Uint32Array(count), 1)
  );
  geometry.setAttribute(
    'aSortedIndexB',
    new THREE.InstancedBufferAttribute(new Uint32Array(count), 1)
  );
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 10);
  const material = new THREE.Material();
  material.userData.blendingMode = blendingMode;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.nodeType = 'points';
  mesh.userData.committedData = { some: 'source' };
  return mesh;
}

const CENTERS = () => new Float32Array([0, 0, -1, 1, 0, -2]);

/** The default init deadline (`config.depthSort.workerInitTimeoutMs`). */
const DEFAULT_INIT_DEADLINE_MS = 30_000;

describe('SortWorker startup (warm-up, configured deadline, guard arms)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('warms up at app init with no commit — and the first commit reuses that worker', async () => {
    // The prevention half of the fix: initialize() runs while the app is idle
    // instead of racing the scene decode. Reuse matters as much as the early
    // start — a warm-up that spawned its own throwaway worker would leave the
    // commit path paying the very startup cost this avoids.
    const coord = await loadCoordinator('immediate');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      isLoadInProgress: () => false,
    });

    coord.warmUpDepthSortWorker();
    await flush();

    expect(constructedWorkers).toHaveLength(1);
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'ready', initTimeouts: 0 });
    // Nothing has committed, so nothing is registered yet.
    expect(mockApi.registerNode).not.toHaveBeenCalled();

    // The first order-dependent commit now finds a READY worker.
    coord.noteDepthSortCommit(makeSortableMesh(2), CENTERS(), 2);
    await flush();

    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    expect(constructedWorkers).toHaveLength(1);
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
  });

  it('warms up nothing when depth sorting is disabled (?depthSort=0)', async () => {
    // The master switch must keep the subsystem fully inert: a spawned worker
    // would load WASM for a session that never sorts.
    const coord = await loadCoordinator('immediate');
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    coord.setDepthSortEnabled(false);

    coord.warmUpDepthSortWorker();
    await flush();

    expect(constructedWorkers).toHaveLength(0);
    expect(mockApi.initialize).not.toHaveBeenCalled();
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'idle', initTimeouts: 0 });
  });

  it('a failing warm-up stays fire-and-forget: no throw, no unhandled rejection', async () => {
    // `core/app/init/pipeline.ts` calls this synchronously in the middle of a
    // sequence that must not be derailed by a worker that cannot start.
    // Real timers: an unhandled rejection is only observable after a real
    // macrotask turn.
    vi.useRealTimers();
    const coord = await loadCoordinator('rejects');
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on('unhandledRejection', onUnhandled);
    try {
      expect(() => coord.warmUpDepthSortWorker()).not.toThrow();
      await flush();
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }

    // The failure is not swallowed either — it is a reported verdict.
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'failed', initTimeouts: 0 });
    expect(coord.isDepthSortAvailable()).toBe(false);
  });

  it('takes the init deadline from config.depthSort.workerInitTimeoutMs', async () => {
    // The deadline used to be a module constant, so an embedder on a slow
    // machine had no way to widen it (and no test could shorten it).
    const coord = await loadCoordinator('never-settles');
    liveConfig.depthSort.workerInitTimeoutMs = 250;
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      isLoadInProgress: () => false,
    });

    coord.warmUpDepthSortWorker();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    // Just short of the configured budget: no verdict yet.
    await vi.advanceTimersByTimeAsync(249);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'idle', initTimeouts: 0 });

    // Just past it — on THIS budget, two orders of magnitude before the 30 s
    // default a hardcoded deadline would have waited for.
    await vi.advanceTimersByTimeAsync(2);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });
  });

  it('an unserializable message during init is PERMANENT (the onmessageerror arm)', async () => {
    // New with the shared guard: the coordinator's own race listened to
    // `onerror` only, so a worker that answered with an unstructured-cloneable
    // payload left `initialize()` pending until the deadline — reported as a
    // starved main thread and RETRIED, three times, for a worker that had
    // already failed. It is a dead-script-class failure and must latch.
    const coord = await loadCoordinator('never-settles');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      isLoadInProgress: () => false,
    });

    coord.warmUpDepthSortWorker();
    await flush();
    const live = liveWorker();
    expect(live.onmessageerror).toBeTypeOf('function');

    live.onmessageerror!({});
    await flush();

    // `initTimeouts: 0` is the pin: this was classified as broken, not slow.
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'failed', initTimeouts: 0 });
    expect(coord.isDepthSortAvailable()).toBe(false);
    expect(terminatedWorkers).toHaveLength(1);

    // No retry is armed: neither elapsed time nor further frames respawn it.
    coord.noteDepthSortCommit(makeSortableMesh(2), CENTERS(), 2);
    await flush();
    for (let frame = 0; frame < 3; frame++) {
      await vi.advanceTimersByTimeAsync(DEFAULT_INIT_DEADLINE_MS);
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    expect(constructedWorkers).toHaveLength(1);
    expect(mockApi.registerNode).not.toHaveBeenCalled();
  });

  it('reports depth sorting AVAILABLE while a starved init is still recoverable', async () => {
    // What the data-loading monitor footer reads. 'idle' (never spawned) and
    // 'ready' are obviously available; the interesting one is 'starved' — a
    // retry is armed and expected to land, so announcing UNAVAILABLE there
    // would cry wolf on every slow load.
    const coord = await loadCoordinator('never-settles');
    expect(coord.isDepthSortAvailable()).toBe(true); // never spawned

    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      isLoadInProgress: () => false,
    });
    coord.warmUpDepthSortWorker();
    await flush();
    await vi.advanceTimersByTimeAsync(DEFAULT_INIT_DEADLINE_MS + 1);

    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });
    expect(coord.isDepthSortAvailable()).toBe(true);

    // Ready is available too (the same load, once init succeeds elsewhere, is
    // covered by the warm-up test above — here the point is the contrast).
    const ready = await loadCoordinator('immediate');
    ready.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    ready.warmUpDepthSortWorker();
    await flush();
    expect(ready.getDepthSortWorkerStatus().state).toBe('ready');
    expect(ready.isDepthSortAvailable()).toBe(true);
  });

  it('reports depth sorting UNAVAILABLE once the session has given up', async () => {
    // The other half of issue #705's rule: a scene drawn in storage order
    // must never look like a correctly sorted one. A dead worker script is
    // the terminal case — nothing later can recover it.
    const coord = await loadCoordinator('never-settles');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      isLoadInProgress: () => false,
    });
    coord.warmUpDepthSortWorker();
    await flush();

    // A plain object, not `new ErrorEvent(...)`: this suite runs in the
    // default `node` environment, where that constructor does not exist.
    // The guard reads `.message` structurally, so this is the same input.
    liveWorker().onerror!({ message: 'module evaluation failed' });
    await flush();

    expect(coord.getDepthSortWorkerStatus().state).toBe('failed');
    expect(coord.isDepthSortAvailable()).toBe(false);

    // A dispose restores a truthful, unspent verdict for the next app.
    coord.disposeDepthSort();
    expect(coord.isDepthSortAvailable()).toBe(true);
  });
});
