/**
 * Depth-sort coordinator tests (depth-sorting Phase 2, spec §5).
 *
 * Mocks the Comlink boundary (`wrap`/`transfer`) and the Vite `?worker`
 * constructor — the module-scoped coordinator is loaded fresh per test
 * via `vi.resetModules()` + dynamic import (the worker-pool lifecycle
 * test idiom). Geometry/attribute writes use REAL THREE objects so the
 * `aSortedIndex` application path is exercised end-to-end.
 *
 * Spec exit criteria covered here: generation-guard drops stale results;
 * single-in-flight per node; release-on-dispose; transfer-detach intent
 * (centers buffer in the transfer list).
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';

type SortResult = { generation: number; ordering: Uint32Array } | null;

interface MockApi {
  initialize: ReturnType<typeof vi.fn>;
  registerNode: ReturnType<typeof vi.fn>;
  sort: ReturnType<typeof vi.fn>;
  releaseNode: ReturnType<typeof vi.fn>;
  releaseAllNodes: ReturnType<typeof vi.fn>;
}

const terminatedWorkers: unknown[] = [];
let mockApi: MockApi;
let transferCalls: Array<{ value: unknown; transferables: Transferable[] }>;
/** Pending resolvers for controllable in-flight sorts (FIFO). */
let sortResolvers: Array<(r: SortResult) => void>;

function makeMockApi(): MockApi {
  return {
    initialize: vi.fn(async () => ({ wasmFallback: true })),
    registerNode: vi.fn(async () => undefined),
    sort: vi.fn(
      () =>
        new Promise<SortResult>((resolve) => {
          sortResolvers.push(resolve);
        })
    ),
    releaseNode: vi.fn(async () => undefined),
    releaseAllNodes: vi.fn(async () => undefined),
  };
}

async function loadCoordinator() {
  vi.resetModules();
  terminatedWorkers.length = 0;
  transferCalls = [];
  sortResolvers = [];
  mockApi = makeMockApi();

  vi.doMock('../../../utils/log', () => ({
    log: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
    Modules: new Proxy({}, { get: (_t, p) => String(p) }),
  }));
  vi.doMock('comlink', () => ({
    wrap: vi.fn(() => mockApi),
    transfer: vi.fn((value: unknown, transferables: Transferable[]) => {
      transferCalls.push({ value, transferables });
      return value;
    }),
  }));
  vi.doMock('../../../workers/sort-worker?worker', () => ({
    default: class MockSortWorker {
      terminate = vi.fn(() => terminatedWorkers.push(this));
    },
  }));

  return await import('../../../rendering/depth-sort-coordinator');
}

/** A minimal gsplats mesh with a REAL aSortedIndex attribute. */
function makeGSplatsMesh(count: number, blendingMode: string): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  const attr = new THREE.InstancedBufferAttribute(new Uint32Array(count), 1);
  geometry.setAttribute('aSortedIndex', attr);
  const material = new THREE.Material();
  material.userData.blendingMode = blendingMode;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.nodeType = 'gsplats';
  mesh.userData.committedData = { some: 'source' };
  return mesh;
}

function makeCamera(): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera();
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  return camera;
}

/** Drain microtasks so ensureWorker → registerNode → scheduleSort settles. */
async function flush(): Promise<void> {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

describe('depth-sort coordinator', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('registers + sorts an order-dependent (normal) commit and applies the ordering', async () => {
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ camera: makeCamera(), requestRender });

    const mesh = makeGSplatsMesh(3, 'normal');
    const centers = new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]);
    coord.noteGSplatsCommit(mesh, centers, 3);
    await flush();

    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    const registered = transferCalls[0];
    expect(registered.transferables).toContain(centers.buffer);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    expect(mockApi.sort.mock.calls[0][0]).toMatchObject({ nodeId: mesh.uuid, generation: 1 });

    sortResolvers[0]({ generation: 1, ordering: new Uint32Array([0, 2, 1]) });
    await flush();

    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 2, 1]);
    expect(requestRender).toHaveBeenCalled();
  });

  it('does not register order-independent (additive) commits', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ camera: makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(3, 'additive');
    coord.noteGSplatsCommit(mesh, new Float32Array(9), 3);
    await flush();

    expect(mockApi.registerNode).not.toHaveBeenCalled();
    expect(mockApi.sort).not.toHaveBeenCalled();
  });

  it('generation guard: a stale ordering resolving after a newer commit is dropped', async () => {
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ camera: makeCamera(), requestRender });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // A newer commit lands while the sort is in flight (generation -> 2).
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -10, 2, 0, -5]), 3);
    await flush();

    // The stale generation-1 ordering resolves — must NOT be applied.
    sortResolvers[0]({ generation: 1, ordering: new Uint32Array([2, 1, 0]) });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0, 0]); // untouched

    // The queued re-sort was issued for the current generation.
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    expect(mockApi.sort.mock.calls[1][0]).toMatchObject({ generation: 2 });

    // The fresh ordering applies.
    sortResolvers[1]({ generation: 2, ordering: new Uint32Array([1, 2, 0]) });
    await flush();
    expect(Array.from(attr.array as Uint32Array)).toEqual([1, 2, 0]);
  });

  it('enforces at most one in-flight sort per node', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ camera: makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    // Three commits, but only ONE sort RPC outstanding.
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // Resolving it triggers exactly one queued re-sort (for the latest generation).
    sortResolvers[0]({ generation: 1, ordering: new Uint32Array([0, 1]) });
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    expect(mockApi.sort.mock.calls[1][0]).toMatchObject({ generation: 3 });
  });

  it('releaseDepthSortNode drops state and discards an in-flight result', async () => {
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ camera: makeCamera(), requestRender });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.releaseDepthSortNode(mesh);
    expect(mockApi.releaseNode).toHaveBeenCalledWith(mesh.uuid);

    sortResolvers[0]({ generation: 1, ordering: new Uint32Array([1, 0]) });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0]); // untouched
    expect(requestRender).not.toHaveBeenCalled();
  });

  it('a cleared committedData stamp (LOD demotion) blocks a resolving ordering', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ camera: makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    delete mesh.userData.committedData; // demotion released the geometry
    sortResolvers[0]({ generation: 1, ordering: new Uint32Array([1, 0]) });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0]); // untouched
  });

  it('mode switch TO normal clears the noop stamp and requests a reprocess', async () => {
    const coord = await loadCoordinator();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      camera: makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteGSplatsBlendingModeSwitch(mesh, 'normal', 'additive');
    expect(mesh.userData.committedData).toBeUndefined();
    expect(requestReprocess).toHaveBeenCalledTimes(1);
  });

  it('mode switch AWAY from normal releases the node and kills in-flight applies', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ camera: makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.noteGSplatsBlendingModeSwitch(mesh, 'additive', 'normal');
    expect(mockApi.releaseNode).toHaveBeenCalledWith(mesh.uuid);
    // The stamp is NOT cleared on the way out (no reprocess needed).
    expect(mesh.userData.committedData).toBeDefined();

    // The in-flight generation-1 sort resolves — dropped (generation bumped).
    sortResolvers[0]({ generation: 1, ordering: new Uint32Array([1, 0]) });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0]);
  });

  it('disposeDepthSort terminates the worker and resets state', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ camera: makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.disposeDepthSort();
    expect(terminatedWorkers.length).toBe(1);

    // Safe to call again when never spawned.
    coord.disposeDepthSort();
    expect(terminatedWorkers.length).toBe(1);
  });

  it('a zero-splat commit releases instead of registering', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ camera: makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    // First a real commit so the worker exists.
    coord.noteGSplatsCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);

    coord.noteGSplatsCommit(mesh, new Float32Array(0), 0);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1); // unchanged
    expect(mockApi.releaseNode).toHaveBeenCalledWith(mesh.uuid);
  });
});
