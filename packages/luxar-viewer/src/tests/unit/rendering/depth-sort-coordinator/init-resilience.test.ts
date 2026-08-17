/**
 * SortWorker init lifecycle: what happens when startup is SLOW rather than
 * broken.
 *
 * The coordinator used to spawn the worker on the first order-dependent
 * commit and race `initialize()` against a fixed deadline, caching the
 * rejection forever on any failure. On a multi-million-element scene that
 * first commit lands while this thread is saturated decoding, so the
 * worker's reply — which must be dispatched on this thread — misses the
 * deadline. A healthy worker was then terminated and depth sorting stayed
 * off for the whole session, silently.
 *
 * These tests pin the distinction the fix rests on: a DEADLINE MISS is
 * transient and retried, while an `onerror` death or an explicit
 * `initialize()` rejection is permanent and still latches.
 *
 * Split into its own file (rather than added to depth-sort-coordinator.test.ts)
 * because it is the only one here that needs FAKE TIMERS, which do not mix
 * with that file's frame-pump helpers — `src/tests/setup.ts` backs the rAF
 * mock with real `setTimeout`. Mirrors the shape of
 * `src/tests/unit/workers/worker-pool/lifecycle/`.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';

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
/** Resolver for the pending `initialize()` under 'never-settles'. */
let resolveInit: ((r: { wasmFallback: boolean }) => void) | null = null;
/** When true the mocked worker CONSTRUCTOR throws (CSP-blocked script). */
let workerConstructThrows = false;
let logMock: {
  info: ReturnType<typeof vi.fn>;
  warning: ReturnType<typeof vi.fn>;
  error: ReturnType<typeof vi.fn>;
};

function makeMockApi(behavior: InitBehavior) {
  return {
    initialize: vi.fn(() => {
      if (behavior === 'immediate') return Promise.resolve({ wasmFallback: true });
      if (behavior === 'rejects') return Promise.reject(new Error('WASM unavailable'));
      return new Promise<{ wasmFallback: boolean }>((resolve) => {
        resolveInit = resolve;
      });
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
  resolveInit = null;
  workerConstructThrows = false;
  mockApi = makeMockApi(behavior);
  logMock = { info: vi.fn(), warning: vi.fn(), error: vi.fn() };

  vi.doMock('../../../../utils/log', () => ({
    log: logMock,
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
        if (workerConstructThrows) throw new Error('worker construction blocked');
        constructedWorkers.push(this);
        (globalThis as unknown as { __lastMockWorker?: unknown }).__lastMockWorker = this;
      }
      terminate = vi.fn(() => terminatedWorkers.push(this));
    },
  }));

  return await import('../../../../rendering/depth-sort-coordinator');
}

/** Drain microtasks so ensureWorker → registerNode → scheduleSort settles. */
async function flush(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
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

/** The configured init deadline (config.depthSort.workerInitTimeoutMs). */
const INIT_DEADLINE_MS = 30_000;

describe('SortWorker init resilience', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('retries after a deadline miss and re-registers, without a second initialize()', async () => {
    // The regression gate. Against the pre-fix coordinator the worker is
    // terminated on timeout and `initPromise` stays rejected, so no retry
    // ever happens and `requestReprocess` is never called.
    const coord = await loadCoordinator('never-settles');
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
      isLoadInProgress: () => false,
    });

    const mesh = makeSortableMesh(2);
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    // The deadline passes with the worker still silent (this thread busy).
    await vi.advanceTimersByTimeAsync(INIT_DEADLINE_MS + 1);
    await flush();

    // Transient: the worker is NOT killed for being late.
    expect(terminatedWorkers).toHaveLength(0);
    expect(coord.isDepthSortAvailable()).toBe(true);
    expect(mockApi.registerNode).not.toHaveBeenCalled();

    // The worker answers late — exactly what the 3M repro showed happening
    // just after the main thread gave up on it.
    resolveInit!({ wasmFallback: false });
    await flush();

    // A frame past the loader-idle gate drives the retry.
    coord.evaluateDepthSortPerFrame();
    await flush();

    // The ORIGINAL in-flight RPC was adopted rather than a second one
    // issued — worker-side initWasm() is not memoized, so a duplicate call
    // would fetch and instantiate the module twice.
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    expect(constructedWorkers).toHaveLength(1);
    expect(terminatedWorkers).toHaveLength(0);

    // Nodes are pushed back through a commit, because the worker never
    // received their centers and nothing else would re-commit them.
    expect(requestReprocess).toHaveBeenCalled();

    // …and that re-commit now reaches the worker.
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalled();
    expect(mockApi.sort).toHaveBeenCalled();
  });

  it('gives up after MAX_INIT_TIMEOUT_ATTEMPTS deadline misses, and says so', async () => {
    const coord = await loadCoordinator('never-settles');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
      isLoadInProgress: () => false,
    });

    const mesh = makeSortableMesh(2);
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();

    // Three deadline misses in a row, each retried on a later idle frame.
    // The first two must leave the worker ALIVE — that is the bound being
    // a bound, rather than the old give-up-on-first-miss behaviour.
    for (let attempt = 0; attempt < 3; attempt++) {
      await vi.advanceTimersByTimeAsync(INIT_DEADLINE_MS + 1);
      await flush();
      if (attempt < 2) {
        expect(terminatedWorkers, `miss ${attempt + 1} must not kill the worker`).toHaveLength(0);
        expect(coord.isDepthSortAvailable()).toBe(true);
      }
      coord.evaluateDepthSortPerFrame();
      await flush();
    }

    // Latched: the worker is released and the state is reported, not hidden.
    expect(coord.isDepthSortAvailable()).toBe(false);
    expect(terminatedWorkers).toHaveLength(1);

    // No spawn storm — further frames must not keep constructing workers.
    const spawned = constructedWorkers.length;
    for (let i = 0; i < 5; i++) {
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(constructedWorkers).toHaveLength(spawned);
  });

  it('an initialize() rejection is PERMANENT — terminated, never retried', async () => {
    // The worker answered: it cannot work (e.g. WASM genuinely missing).
    // Unlike a deadline miss this is evidence, so the old latch is right.
    const coord = await loadCoordinator('rejects');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
      isLoadInProgress: () => false,
    });

    const mesh = makeSortableMesh(2);
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();

    expect(terminatedWorkers).toHaveLength(1);
    expect(coord.isDepthSortAvailable()).toBe(false);

    for (let i = 0; i < 5; i++) {
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    expect(constructedWorkers).toHaveLength(1);
    expect(mockApi.registerNode).not.toHaveBeenCalled();
  });

  it('a worker error during startup is PERMANENT — terminated, never retried', async () => {
    const coord = await loadCoordinator('never-settles');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
      isLoadInProgress: () => false,
    });

    const mesh = makeSortableMesh(2);
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();

    const live = (
      globalThis as unknown as { __lastMockWorker?: { onerror?: (e: unknown) => void } }
    ).__lastMockWorker;
    expect(live?.onerror).toBeTypeOf('function');
    live!.onerror!(new ErrorEvent('error', { message: 'module evaluation failed' }));
    await flush();

    expect(terminatedWorkers).toHaveLength(1);
    expect(coord.isDepthSortAvailable()).toBe(false);

    for (let i = 0; i < 5; i++) {
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    expect(constructedWorkers).toHaveLength(1);
  });

  it('survives two misses then succeeds on the LAST allowed attempt', async () => {
    // The fencepost from the other side: attempt 3 is permitted, it is only
    // a THIRD miss that latches. Off-by-one here would give up while the
    // worker was about to answer — the exact failure this change exists to
    // stop, reintroduced one attempt later.
    const coord = await loadCoordinator('never-settles');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
      isLoadInProgress: () => false,
    });

    const mesh = makeSortableMesh(2);
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();

    for (let miss = 0; miss < 2; miss++) {
      await vi.advanceTimersByTimeAsync(INIT_DEADLINE_MS + 1);
      await flush();
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(coord.isDepthSortAvailable()).toBe(true);
    expect(terminatedWorkers).toHaveLength(0);

    // The worker finally answers, on the third and last permitted attempt.
    resolveInit!({ wasmFallback: false });
    await flush();
    expect(coord.isDepthSortAvailable()).toBe(true);

    // And the subsystem is genuinely usable, not merely "not latched".
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalled();
    expect(mockApi.sort).toHaveBeenCalled();
  });

  it('a commit landing DURING a retry joins it rather than starting a rival attempt', async () => {
    // `ensureWorker()` dedupes on `initPromise`, so the commit rides the
    // retry already in flight; when it resolves, the commit's own
    // continuation registers the node. The hazard would be a second worker,
    // or a node whose registration is dropped because the retry "owned" it.
    const coord = await loadCoordinator('never-settles');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
      isLoadInProgress: () => false,
    });

    const first = makeSortableMesh(2);
    coord.noteDepthSortCommit(first, CENTERS(), 2);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_DEADLINE_MS + 1);
    await flush();

    // Retry starts on the next idle frame…
    coord.evaluateDepthSortPerFrame();
    await flush();

    // …and a SECOND node commits while it is still in flight.
    const second = makeSortableMesh(3);
    coord.noteDepthSortCommit(second, new Float32Array([0, 0, -1, 1, 0, -2, 2, 0, -3]), 3);
    await flush();

    // One worker, one initialize() — no rival attempt.
    expect(constructedWorkers).toHaveLength(1);
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    resolveInit!({ wasmFallback: false });
    await flush();

    // The node that committed mid-retry is registered and sorted.
    expect(mockApi.registerNode).toHaveBeenCalled();
    expect(mockApi.sort).toHaveBeenCalled();
  });

  it('an offline capture during the retry window does not sort an uninitialized worker', async () => {
    // `resortForCapture`'s force loop dispatches for every visible
    // order-dependent node WITHOUT consulting `state.registered`. Before the
    // worker survived a deadline miss, `api !== null` implied initialized, so
    // that was safe; now it can mean "spawned, awaiting retry", and sorting
    // against it would hit the worker-side NOT_INITIALIZED guard.
    const coord = await loadCoordinator('never-settles');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      isLoadInProgress: () => false,
    });

    const mesh = makeSortableMesh(2);
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_DEADLINE_MS + 1);
    await flush();

    // Worker alive but NOT initialized — the window this guards.
    expect(terminatedWorkers).toHaveLength(0);
    await coord.resortForCapture(0);
    await flush();
    expect(mockApi.sort).not.toHaveBeenCalled();
  });

  it('a dispose mid-init makes the in-flight attempt stale, not authoritative', async () => {
    // The generation token's whole job. `disposeDepthSort()` bumps it while
    // an attempt is awaiting its guard; when that attempt finally settles it
    // must not publish `workerReady` over the disposed state, nor clear a
    // newer attempt's cached promise.
    const coord = await loadCoordinator('never-settles');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      isLoadInProgress: () => false,
    });

    const mesh = makeSortableMesh(2);
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();
    expect(constructedWorkers).toHaveLength(1);

    coord.disposeDepthSort();
    await flush();
    // Dispose terminates the worker it owned.
    expect(terminatedWorkers).toHaveLength(1);

    // The superseded attempt answers LATE.
    resolveInit!({ wasmFallback: false });
    await flush();

    // It must not resurrect the subsystem behind dispose's back.
    expect(coord.isDepthSortAvailable()).toBe(true); // dispose resets the flag
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      isLoadInProgress: () => false,
    });
    coord.evaluateDepthSortPerFrame();
    await flush();
    // No node state survives dispose, so nothing was registered against the
    // dead worker by the stale resolution.
    expect(mockApi.registerNode).not.toHaveBeenCalled();
  });

  it('a CSP-blocked constructor is PERMANENT — and is REPORTED, not silent', async () => {
    // `new Worker()` throws synchronously, so this arm reaches the failure
    // bookkeeping only because the construction lives inside the try. With
    // it outside, the session degraded to unsorted while
    // `isDepthSortAvailable()` still answered true — silent, which is the
    // one outcome this whole change exists to prevent.
    const coord = await loadCoordinator('immediate');
    workerConstructThrows = true;
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
      isLoadInProgress: () => false,
    });

    const mesh = makeSortableMesh(2);
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();

    expect(coord.isDepthSortAvailable()).toBe(false);
    expect(mockApi.registerNode).not.toHaveBeenCalled();

    // Latched: no retry storm against a script that will never load.
    for (let i = 0; i < 5; i++) {
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(constructedWorkers).toHaveLength(0);
    expect(terminatedWorkers).toHaveLength(0);
  });

  it('warms up at configure time, with no commit — and honours ?depthSort=0', async () => {
    // The prevention half of the fix: starting here means initialize() runs
    // while the app is idle, instead of racing the scene decode.
    const coord = await loadCoordinator('immediate');
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    coord.warmUpDepthSortWorker();
    await flush();

    expect(constructedWorkers).toHaveLength(1);
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    // A disabled session must spawn nothing at all.
    const off = await loadCoordinator('immediate');
    off.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    off.setDepthSortEnabled(false);
    off.warmUpDepthSortWorker();
    await flush();
    expect(constructedWorkers).toHaveLength(0);
  });

  it('warm-up means the first commit registers immediately', async () => {
    const coord = await loadCoordinator('immediate');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      isLoadInProgress: () => false,
    });
    coord.warmUpDepthSortWorker();
    await flush();

    const mesh = makeSortableMesh(2);
    coord.noteDepthSortCommit(mesh, CENTERS(), 2);
    await flush();

    expect(mockApi.registerNode).toHaveBeenCalled();
    expect(mockApi.sort).toHaveBeenCalled();
    // Still exactly one worker: the commit path reuses the warmed one.
    expect(constructedWorkers).toHaveLength(1);
  });
});
