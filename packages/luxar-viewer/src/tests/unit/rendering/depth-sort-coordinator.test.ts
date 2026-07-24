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
/** Matching rejectors (same FIFO index as sortResolvers). */
let sortRejectors: Array<(e: Error) => void>;

function makeMockApi(): MockApi {
  return {
    initialize: vi.fn(async () => ({ wasmFallback: true })),
    registerNode: vi.fn(async () => undefined),
    sort: vi.fn(
      () =>
        new Promise<SortResult>((resolve, reject) => {
          sortResolvers.push(resolve);
          sortRejectors.push(reject);
        })
    ),
    releaseNode: vi.fn(async () => undefined),
    releaseAllNodes: vi.fn(async () => undefined),
  };
}

/** When true the mocked worker CONSTRUCTOR throws (CSP-blocked script). */
let workerConstructThrows = false;

async function loadCoordinator() {
  vi.resetModules();
  terminatedWorkers.length = 0;
  transferCalls = [];
  sortResolvers = [];
  sortRejectors = [];
  workerConstructThrows = false;
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
      onerror: ((e: unknown) => void) | null = null;
      constructor() {
        if (workerConstructThrows) throw new Error('worker construction blocked');
        // Expose the live instance so tests can fire worker events
        // (the init settle-guard test drives `onerror`).
        (globalThis as unknown as { __lastMockWorker?: unknown }).__lastMockWorker = this;
      }
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
  // Committed gsplat geometry always carries bounds (the commit path
  // computes them); the Phase-3 translation threshold is bounds-relative.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 10);
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

/** A camera parked at a world position (identity rotation) for BSP tests. */
function cameraAt(x: number, y: number, z: number): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(x, y, z);
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  return camera;
}

/**
 * Wrap `parts` in a kind=partition `THREE.Group` carrying `bspTree`, tagging
 * each part with its `partIndex` — exactly what `load-partition-group-node`
 * stamps at load. Parts stay at the wrapper's (identity) local space, so the
 * BSP `split` coordinates are directly comparable to the camera position.
 */
function makePartitionWrapper(bspTree: unknown, parts: THREE.Mesh[]): THREE.Group {
  const wrapper = new THREE.Group();
  wrapper.userData.kind = 'partition';
  wrapper.userData.bspTree = bspTree;
  parts.forEach((mesh, i) => {
    mesh.userData.partIndex = i;
    wrapper.add(mesh);
  });
  wrapper.updateMatrixWorld(true);
  return wrapper;
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
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender });

    const mesh = makeGSplatsMesh(3, 'normal');
    const centers = new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]);
    coord.noteDepthSortCommit(mesh, centers, 3);
    await flush();

    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    const registered = transferCalls[0];
    expect(registered.transferables).toContain(centers.buffer);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    expect(mockApi.sort.mock.calls[0][0]).toMatchObject({ nodeId: mesh.uuid });
    // Generations are a module-scoped monotonic counter (lifetime-unique),
    // so absolute values are execution-order-dependent — assert RELATIVE
    // facts: the sort carries the same generation the registration did.
    const sentGeneration = mockApi.sort.mock.calls[0][0].generation as number;
    expect(sentGeneration).toBe(mockApi.registerNode.mock.calls[0][0].generation);
    expect(sentGeneration).toBeGreaterThan(0);

    sortResolvers[0]({ generation: sentGeneration, ordering: new Uint32Array([0, 2, 1]) });
    await flush();

    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 2, 1]);
    expect(requestRender).toHaveBeenCalled();
  });

  it('does not register order-independent (additive) commits', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(3, 'additive');
    coord.noteDepthSortCommit(mesh, new Float32Array(9), 3);
    await flush();

    expect(mockApi.registerNode).not.toHaveBeenCalled();
    expect(mockApi.sort).not.toHaveBeenCalled();
  });

  it('registers + sorts a volumetric commit (the second order-dependent mode)', async () => {
    // needsDepthSort = normal ∪ volumetric: volumetric compositing is
    // non-commutative (emission–absorption attenuates what is behind),
    // so it takes the exact same sort path as 'normal'. Fails against
    // an isNormalMode-gated coordinator.
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender });

    const mesh = makeGSplatsMesh(3, 'volumetric');
    const centers = new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]);
    coord.noteDepthSortCommit(mesh, centers, 3);
    await flush();

    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 2, 1]),
    });
    await flush();

    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 2, 1]);
    expect(requestRender).toHaveBeenCalled();
  });

  it('normal↔volumetric mode switch is a sorted→sorted no-op (no reprocess, no release)', async () => {
    const coord = await loadCoordinator();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);

    // Both modes are sorted — the ordering stays valid; the projection/
    // output change is the material's problem, not the coordinator's.
    coord.noteDepthSortBlendingModeSwitch(mesh, 'volumetric', 'normal');
    expect(requestReprocess).not.toHaveBeenCalled();
    expect(mockApi.releaseNode).not.toHaveBeenCalled();

    // Switching to a commutative mode DOES release worker-side state.
    coord.noteDepthSortBlendingModeSwitch(mesh, 'additive', 'volumetric');
    expect(mockApi.releaseNode).toHaveBeenCalledTimes(1);

    // And switching back to a sorted mode forces the reprocess.
    coord.noteDepthSortBlendingModeSwitch(mesh, 'volumetric', 'additive');
    expect(requestReprocess).toHaveBeenCalledTimes(1);
  });

  it('generation guard: a stale ordering resolving after a newer commit is dropped', async () => {
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    const staleGeneration = mockApi.sort.mock.calls[0][0].generation as number;

    // A newer commit lands while the sort is in flight (generation bumps).
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -10, 2, 0, -5]), 3);
    await flush();

    // The FIRST dispatch's (now-stale) ordering resolves — must NOT be applied.
    sortResolvers[0]({ generation: staleGeneration, ordering: new Uint32Array([2, 1, 0]) });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0, 0]); // untouched

    // The queued re-sort was issued for the current (newer) generation.
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    const freshGeneration = mockApi.sort.mock.calls[1][0].generation as number;
    expect(freshGeneration).not.toBe(staleGeneration);

    // The fresh ordering applies.
    sortResolvers[1]({ generation: freshGeneration, ordering: new Uint32Array([1, 2, 0]) });
    await flush();
    expect(Array.from(attr.array as Uint32Array)).toEqual([1, 2, 0]);
  });

  it("stale sort from a released node's previous LIFETIME is dropped (demote → re-promote)", async () => {
    // Fuzz-found bug: releaseDepthSortNode deletes the node state, and a
    // per-node counter restarting at 1 on re-promotion let a stale
    // in-flight sort from the PREVIOUS lifetime pass the generation guard
    // — applying a wrong-length permutation over the new commit. The
    // module-scoped monotonic counter makes generations lifetime-unique.
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender });

    // First lifetime: 3 splats committed, sort dispatched with generation A.
    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    const generationA = mockApi.sort.mock.calls[0][0].generation as number;

    // Demote (LOD): the node's state + worker registration are released
    // while lifetime A's sort is still in flight.
    coord.releaseDepthSortNode(mesh);

    // Re-promote: the SAME mesh recommits with a DIFFERENT count (2).
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    const generationB = mockApi.sort.mock.calls[1][0].generation as number;
    expect(generationB).not.toBe(generationA); // lifetime-unique

    // Lifetime A's sort resolves with its length-3 ordering — must be
    // dropped: applied over the 2-splat commit it would be corrupt.
    sortResolvers[0]({ generation: generationA, ordering: new Uint32Array([2, 0, 1]) });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0, 0]); // untouched

    // Lifetime B's own ordering applies normally.
    sortResolvers[1]({ generation: generationB, ordering: new Uint32Array([1, 0]) });
    await flush();
    expect(Array.from(attr.array as Uint32Array)).toEqual([1, 0, 0]);
  });

  it('enforces at most one in-flight sort per node', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    // Three commits, but only ONE sort RPC outstanding — and only ONE
    // worker ever spawned/initialized (the initPromise cache).
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    // Resolving it triggers exactly one queued re-sort (for the latest generation).
    const firstGeneration = mockApi.sort.mock.calls[0][0].generation as number;
    sortResolvers[0]({ generation: firstGeneration, ordering: new Uint32Array([0, 1]) });
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    // The re-sort carries the LATEST commit's generation — the one the
    // most recent registration sent — not the first dispatch's.
    const latestRegistered = mockApi.registerNode.mock.calls[
      mockApi.registerNode.mock.calls.length - 1
    ][0].generation as number;
    expect(mockApi.sort.mock.calls[1][0].generation).toBe(latestRegistered);
    expect(mockApi.sort.mock.calls[1][0].generation).not.toBe(firstGeneration);
  });

  it('derives the model-view from fresh matrices, not renderer-maintained caches', async () => {
    // A commit can fire before the next render (first commit of a load,
    // idle-paused loop): camera.matrixWorldInverse and mesh.matrixWorld
    // are then STALE. The coordinator must refresh both at sort time —
    // reading the cached inverse here would send an identity view matrix.
    const coord = await loadCoordinator();
    const camera = new THREE.PerspectiveCamera();
    camera.position.set(0, 0, 50);
    // Deliberately DO NOT update matrixWorld / matrixWorldInverse — that
    // is the renderer's job, which has not run yet.
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    mesh.position.set(0, 0, 10);
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    const mv = mockApi.sort.mock.calls[0][0].modelView as Float32Array;
    // modelView = inverse(camera at z=50) × mesh at z=10 → z-translation
    // 10 − 50 = −40. Stale matrices would give 0 (identity × identity).
    expect(mv[14]).toBeCloseTo(-40, 5);
  });

  it('respawns a fresh worker after dispose + reconfigure (app re-init)', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const meshA = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(meshA, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    coord.disposeDepthSort();
    expect(terminatedWorkers.length).toBe(1);

    // Re-init (a second LuxarApp.init in the same page/session).
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const meshB = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(meshB, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    // A FRESH worker was spawned and initialized; sorts flow again.
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
    const lastSortCall = mockApi.sort.mock.calls[mockApi.sort.mock.calls.length - 1][0];
    sortResolvers[sortResolvers.length - 1]({
      generation: lastSortCall.generation as number,
      ordering: new Uint32Array([1, 0]),
    });
    await flush();
    const attrB = (meshB.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attrB.array as Uint32Array)).toEqual([1, 0]);
  });

  it('degrades gracefully (warn-once, no throw) when the worker cannot be created', async () => {
    const coord = await loadCoordinator();
    // Simulate a broken embedder override / CSP-blocked worker script.
    mockApi.initialize.mockRejectedValue(new Error('worker init blocked'));
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    // Two commits: neither may throw; register/sort never happen.
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.registerNode).not.toHaveBeenCalled();
    expect(mockApi.sort).not.toHaveBeenCalled();
    // Rendering itself is unaffected — identity ordering stays in place.
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0]);
  });

  it('cross-node renderOrder still assigns when the worker was never constructed', async () => {
    // The renderOrder pass is pure main-thread — the degraded
    // no-SortWorker mode (constructor throw, e.g. a CSP-blocked worker
    // script, leaves `api` null forever) loses within-mesh splat order
    // (unavoidable) but must NOT lose cross-node back-to-front mesh
    // order (avoidable: it needs no worker).
    const coord = await loadCoordinator();
    workerConstructThrows = true;
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const far = makeGSplatsMesh(2, 'normal');
    far.geometry.boundingSphere!.center.set(0, 0, -30);
    const near = makeGSplatsMesh(2, 'normal');
    near.geometry.boundingSphere!.center.set(0, 0, -10);
    coord.noteDepthSortCommit(near, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(far, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(far.renderOrder).toBe(0);
    expect(near.renderOrder).toBe(1);
    expect(mockApi.sort).not.toHaveBeenCalled(); // worker path stays inert
  });

  it('keeps per-node state independent across two nodes sharing the worker', async () => {
    // Two order-dependent nodes → two independent in-flight sorts on
    // the SAME worker; resolving one must not touch the other, and
    // releasing one mid-flight must not disturb the other's resolve.
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender });

    const meshA = makeGSplatsMesh(2, 'normal');
    const meshB = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(meshA, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    coord.noteDepthSortCommit(meshB, new Float32Array([0, 0, -3, 1, 0, -4]), 2);
    await flush();

    // One sort per node, concurrently in flight (per-node rule, not global).
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    expect(mockApi.initialize).toHaveBeenCalledTimes(1); // still one worker

    // Release A mid-flight, then resolve BOTH (A's first) — each with the
    // generation its OWN dispatch carried (they differ: one shared counter).
    coord.releaseDepthSortNode(meshA);
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([1, 0]),
    });
    sortResolvers[1]({
      generation: mockApi.sort.mock.calls[1][0].generation as number,
      ordering: new Uint32Array([1, 0]),
    });
    await flush();

    const attrA = (meshA.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    const attrB = (meshB.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attrA.array as Uint32Array)).toEqual([0, 0]); // discarded
    expect(Array.from(attrB.array as Uint32Array)).toEqual([1, 0]); // applied
  });

  it('orders BSP-partition parts back-to-front from the stored split planes (camera outside)', async () => {
    // Partition parts share the world origin (splat centers baked in), so
    // THREE's per-object transparent sort gives every part the SAME key and
    // draws them in creation order. With a stored bspTree the coordinator
    // instead assigns each part its EXACT painter's-order rank (0 = farthest).
    // Tree: three splits on x (axis 0) → 4 leaf cells left→right along x.
    const bspTree = {
      axis: 0,
      split: 0,
      left: { axis: 0, split: -50, left: { part: 0 }, right: { part: 1 } },
      right: { axis: 0, split: 50, left: { part: 2 }, right: { part: 3 } },
    };
    const coord = await loadCoordinator();
    // Camera far out on +x: the x>=50 cell (part 3) is nearest, x<-50 (part 0)
    // farthest → ranks must be [0, 1, 2, 3] left→right.
    coord.configureDepthSort({ getCamera: () => cameraAt(1000, 0, 0), requestRender: vi.fn() });

    const parts = [0, 1, 2, 3].map(() => makeGSplatsMesh(2, 'normal'));
    makePartitionWrapper(bspTree, parts);
    for (const m of parts) coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // renderOrder == painter's rank; THREE draws transparent objects by
    // renderOrder ASCENDING, so rank 0 (x<-50, farthest) draws first.
    expect(parts.map((m) => m.renderOrder)).toEqual([0, 1, 2, 3]);
  });

  it('orders BSP-partition parts correctly with the camera INSIDE the volume (centroid fails here)', async () => {
    // The camera-inside case the per-part centroid heuristic gets wrong: parts
    // spread perpendicular to the view axis all share ~the same centroid
    // view-z (→ a THREE tie → creation order), and a part can sit behind the
    // camera (positive view-z). The BSP traversal uses the split geometry, not
    // a projected centroid, so it stays exact from any interior viewpoint.
    const bspTree = {
      axis: 0,
      split: 0,
      left: { axis: 0, split: -50, left: { part: 0 }, right: { part: 1 } },
      right: { axis: 0, split: 50, left: { part: 2 }, right: { part: 3 } },
    };
    const coord = await loadCoordinator();
    // Camera at x=+25 — INSIDE the x∈[-100,100] span, between the inner splits.
    // Far side of x=0 is the left branch (drawn first); within the near (right)
    // branch, x>=50 (part 3) is farther than 0<=x<50 (part 2). Correct
    // back-to-front leaf order = part0, part1, part3, part2.
    coord.configureDepthSort({ getCamera: () => cameraAt(25, 0, 0), requestRender: vi.fn() });

    const parts = [0, 1, 2, 3].map(() => makeGSplatsMesh(2, 'normal'));
    makePartitionWrapper(bspTree, parts);
    for (const m of parts) coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // ranks: part0→0, part1→1, part3→2, part2→3.
    expect(parts.map((m) => m.renderOrder)).toEqual([0, 1, 3, 2]);
  });

  it('falls back to content-centroid view depth when a mesh has no BSP tree (single leaf / legacy)', async () => {
    // Leaf meshes without a partition wrapper: each is its own order group,
    // and groups sort by their bounding-sphere-center view-space z (camera
    // at the origin looking −z, so view ≈ identity and view-z ≈ center.z).
    // renderOrder is the GLOBAL sequential rank (0 = farthest), not the raw
    // z — all normal-mode gsplat meshes share one comparable integer scale.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const near = makeGSplatsMesh(2, 'normal'); // centroid closest to camera
    const mid = makeGSplatsMesh(2, 'normal');
    const far = makeGSplatsMesh(2, 'normal'); // centroid farthest
    near.geometry.boundingSphere!.center.set(0, 0, -10);
    mid.geometry.boundingSphere!.center.set(0, 0, -20);
    far.geometry.boundingSphere!.center.set(0, 0, -30);
    // Insertion order deliberately NOT depth order — the fallback must reorder.
    coord.noteDepthSortCommit(mid, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(far, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(near, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(far.renderOrder).toBe(0);
    expect(mid.renderOrder).toBe(1);
    expect(near.renderOrder).toBe(2);
  });

  it('meshes without usable depth (no bounds / non-finite center) rank at view-z 0, no NaN poisoning', async () => {
    // A boundless mesh has no depth reference; a NaN bounding-sphere
    // center (NaN input data propagates into the bbox) must not poison
    // the group-sort comparators — both degrade to view-z 0 and stay
    // comparable, ranking between farther (< 0) and behind-camera (> 0)
    // content.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const farMesh = makeGSplatsMesh(2, 'normal');
    farMesh.geometry.boundingSphere!.center.set(0, 0, -30);
    const boundless = makeGSplatsMesh(2, 'normal');
    boundless.geometry.boundingSphere = null;
    const nanCenter = makeGSplatsMesh(2, 'normal');
    nanCenter.geometry.boundingSphere!.center.set(NaN, NaN, NaN);

    for (const m of [boundless, nanCenter, farMesh]) {
      coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    // The finite far mesh (view-z -30) ranks first; the two degenerate
    // meshes tie at view-z 0 and take the remaining ranks in insertion
    // order — every mesh got exactly one integer rank (no NaN fallout).
    expect(farMesh.renderOrder).toBe(0);
    expect([boundless.renderOrder, nanCenter.renderOrder].sort()).toEqual([1, 2]);
  });

  it('two BSP wrappers land on ONE global scale: the far wrapper draws entirely first', async () => {
    // The core cross-domain fix: per-wrapper painter ranks are only
    // comparable WITHIN a wrapper. Two partitions must interleave on a
    // shared global scale — previously both wrappers' parts got 0..N-1 and
    // THREE drew them arbitrarily interleaved.
    const xSplitTree = {
      axis: 0,
      split: 0,
      left: { part: 0 },
      right: { part: 1 },
    };
    const coord = await loadCoordinator();
    // Camera at the origin looking −z; group depth = mean member view-z.
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const farParts = [0, 1].map(() => makeGSplatsMesh(2, 'normal'));
    farParts[0].geometry.boundingSphere!.center.set(-50, 0, -100);
    farParts[1].geometry.boundingSphere!.center.set(50, 0, -100);
    makePartitionWrapper(xSplitTree, farParts);

    const nearParts = [0, 1].map(() => makeGSplatsMesh(2, 'normal'));
    nearParts[0].geometry.boundingSphere!.center.set(-50, 0, -20);
    nearParts[1].geometry.boundingSphere!.center.set(50, 0, -20);
    makePartitionWrapper(xSplitTree, nearParts);

    // Commit near wrapper FIRST so insertion order can't fake the result.
    for (const m of [...nearParts, ...farParts]) {
      coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    // Far wrapper (mean z −100) ranks 0..1, near wrapper (mean z −20)
    // ranks 2..3; each wrapper internally keeps its BSP traversal order
    // (eye x=0 is not < split 0 → left leaf drawn first).
    expect(farParts.map((m) => m.renderOrder)).toEqual([0, 1]);
    expect(nearParts.map((m) => m.renderOrder)).toEqual([2, 3]);
  });

  it('a partition and single leaves share the global scale by depth', async () => {
    // Partition + leaf was the worst mixed case: leaves carried raw
    // negative view-z while parts carried ranks >= 0, so every leaf drew
    // before every part regardless of actual depth.
    const xSplitTree = {
      axis: 0,
      split: 0,
      left: { part: 0 },
      right: { part: 1 },
    };
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const parts = [0, 1].map(() => makeGSplatsMesh(2, 'normal'));
    parts[0].geometry.boundingSphere!.center.set(-50, 0, -100);
    parts[1].geometry.boundingSphere!.center.set(50, 0, -100);
    makePartitionWrapper(xSplitTree, parts);

    const behindLeaf = makeGSplatsMesh(2, 'normal'); // farther than the wrapper
    behindLeaf.geometry.boundingSphere!.center.set(0, 0, -200);
    const frontLeaf = makeGSplatsMesh(2, 'normal'); // between wrapper and camera
    frontLeaf.geometry.boundingSphere!.center.set(0, 0, -20);

    for (const m of [frontLeaf, ...parts, behindLeaf]) {
      coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    // behind leaf → wrapper parts (BSP order) → front leaf.
    expect(behindLeaf.renderOrder).toBe(0);
    expect(parts.map((m) => m.renderOrder)).toEqual([1, 2]);
    expect(frontLeaf.renderOrder).toBe(3);
  });

  it('camera inside wrapper A: wrapper B behind still draws first, A keeps exact interior BSP order', async () => {
    const treeA = {
      axis: 0,
      split: 0,
      left: { axis: 0, split: -50, left: { part: 0 }, right: { part: 1 } },
      right: { axis: 0, split: 50, left: { part: 2 }, right: { part: 3 } },
    };
    const coord = await loadCoordinator();
    // Camera at x=25 (inside A's x∈[-100,100] span), looking −z.
    coord.configureDepthSort({ getCamera: () => cameraAt(25, 0, 0), requestRender: vi.fn() });

    const partsA = [0, 1, 2, 3].map(() => makeGSplatsMesh(2, 'normal'));
    // A's content straddles the camera plane (mean view-z 0 — the
    // camera-inside signature).
    partsA.forEach((m, i) => m.geometry.boundingSphere!.center.set(-75 + i * 50, 0, 0));
    makePartitionWrapper(treeA, partsA);

    const partsB = [0, 1].map(() => makeGSplatsMesh(2, 'normal'));
    partsB[0].geometry.boundingSphere!.center.set(-50, 0, -50);
    partsB[1].geometry.boundingSphere!.center.set(50, 0, -50);
    makePartitionWrapper({ axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } }, partsB);

    for (const m of [...partsA, ...partsB]) {
      coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    // B (mean view-z −50) is globally farther than A (mean 0) → ranks 0..1.
    expect(partsB.map((m) => m.renderOrder)).toEqual([0, 1]);
    // A keeps the exact interior FKN order (eye x=25): part0, part1, part3, part2.
    expect(partsA.map((m) => m.renderOrder)).toEqual([2, 3, 5, 4]);
  });

  it('clears renderOrder to 0 when a mesh is no longer order-dependent (additive)', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    // Two leaves so the NEARER one carries a nonzero global rank — a
    // single mesh would rank 0 and be indistinguishable from "cleared".
    const far = makeGSplatsMesh(2, 'normal');
    far.geometry.boundingSphere!.center.set(0, 0, -30);
    const near = makeGSplatsMesh(2, 'normal');
    near.geometry.boundingSphere!.center.set(0, 0, -15);
    coord.noteDepthSortCommit(far, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(near, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    coord.evaluateDepthSortPerFrame();
    expect(near.renderOrder).toBe(1); // biased while normal

    // Switch to a commutative mode — renderOrder bias must be cleared so it
    // doesn't strand a stale ordering (additive is order-independent).
    (near.material as THREE.Material).userData.blendingMode = 'additive';
    coord.evaluateDepthSortPerFrame();
    expect(near.renderOrder).toBe(0);
  });

  it('applies the mesh ROTATION to the model-view (not just translation)', async () => {
    // A 180° rotation about y negates the mesh-local z axis: local
    // z = +1 lands at world z = position.z − 1. A translation-only
    // model-view would put it at position.z + 1 — the opposite depth
    // order. Pins the full matrixWorld path.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    mesh.position.set(0, 0, -10);
    mesh.rotation.y = Math.PI;
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, 1, 0, 0, 3]), 2);
    await flush();

    const mv = mockApi.sort.mock.calls[0][0].modelView as Float32Array;
    // Rotation flips the z column: m10 ≈ −1; translation stays −10.
    expect(mv[10]).toBeCloseTo(-1, 5);
    expect(mv[14]).toBeCloseTo(-10, 5);
    // View z of the two splats: −10−1 = −11 and −10−3 = −13 → the
    // second (farther) splat must draw first under this model-view.
    const z0 = mv[10] * 1 + mv[14];
    const z1 = mv[10] * 3 + mv[14];
    expect(z1).toBeLessThan(z0);
  });

  it('drains a queued re-sort even when the in-flight sort RPC fails', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    // A second commit queues a re-sort while the first sort is in flight.
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // The in-flight sort FAILS — the queued request must still run
    // (dropping it would leave the node stale until the next commit),
    // carrying the SECOND commit's (newer) generation.
    sortRejectors[0](new Error('worker transport error'));
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    expect(mockApi.sort.mock.calls[1][0].generation).not.toBe(
      mockApi.sort.mock.calls[0][0].generation
    );
    expect(mockApi.sort.mock.calls[1][0].generation).toBe(
      mockApi.registerNode.mock.calls[1][0].generation
    );
  });

  it('releaseDepthSortNode drops state and discards an in-flight result', async () => {
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.releaseDepthSortNode(mesh);
    expect(mockApi.releaseNode).toHaveBeenCalledWith(mesh.uuid);

    // Echo the dispatched generation — the drop comes from the DELETED
    // state, not a generation mismatch.
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([1, 0]),
    });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0]); // untouched
    expect(requestRender).not.toHaveBeenCalled();
  });

  it('a cleared committedData stamp (LOD demotion) blocks a resolving ordering', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    delete mesh.userData.committedData; // demotion released the geometry
    // Echo the dispatched generation — the block must come from the
    // cleared stamp alone, not a generation mismatch.
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([1, 0]),
    });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0]); // untouched
  });

  it('mode switch TO normal clears the noop stamp and requests a reprocess', async () => {
    const coord = await loadCoordinator();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortBlendingModeSwitch(mesh, 'normal', 'additive');
    expect(mesh.userData.committedData).toBeUndefined();
    expect(requestReprocess).toHaveBeenCalledTimes(1);
  });

  it('mode switch AWAY from normal releases the node and kills in-flight applies', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.noteDepthSortBlendingModeSwitch(mesh, 'additive', 'normal');
    expect(mockApi.releaseNode).toHaveBeenCalledWith(mesh.uuid);
    // The stamp is NOT cleared on the way out (no reprocess needed).
    expect(mesh.userData.committedData).toBeDefined();

    // The in-flight sort resolves with the generation it was dispatched
    // with — dropped (the mode switch bumped the node's generation).
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([1, 0]),
    });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 0]);
  });

  it('disposeDepthSort terminates the worker and resets state', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.disposeDepthSort();
    expect(terminatedWorkers.length).toBe(1);

    // Safe to call again when never spawned.
    coord.disposeDepthSort();
    expect(terminatedWorkers.length).toBe(1);
  });

  it('a zero-splat commit releases instead of registering', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    // First a real commit so the worker exists.
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);

    coord.noteDepthSortCommit(mesh, new Float32Array(0), 0);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1); // unchanged
    expect(mockApi.releaseNode).toHaveBeenCalledWith(mesh.uuid);
  });
});

/**
 * Phase 3 (spec §6): the per-frame camera-motion re-sort scheduler.
 *
 * Defaults from config.depthSort: angleThresholdDeg = 3,
 * translationFraction = 0.05; test meshes carry a bounding sphere of
 * radius 10, so the view-axis translation threshold is 0.5 world units.
 */
describe('depth-sort scheduler (Phase 3)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  /**
   * Commit one normal-mode mesh and resolve its commit-time sort so the
   * node is registered, quiet, and has a recorded dispatch pose.
   */
  async function sortedSetup(
    coord: Awaited<ReturnType<typeof loadCoordinator>>,
    camera: THREE.Camera,
    extra: Partial<Parameters<(typeof coord)['configureDepthSort']>[0]> = {}
  ): Promise<THREE.Mesh> {
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn(), ...extra });
    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 1]),
    });
    await flush();
    return mesh;
  }

  it('rotation past the angle threshold dispatches ONE re-sort; sub-threshold and post-dispatch frames stay quiet', async () => {
    const coord = await loadCoordinator();
    const camera = makeCamera();
    await sortedSetup(coord, camera);

    // Stationary camera: no dispatch, frame after frame.
    coord.evaluateDepthSortPerFrame();
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // 1° — below the 3° threshold.
    camera.rotateY((1 * Math.PI) / 180);
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // 5° total — past the threshold: exactly one dispatch...
    camera.rotateY((4 * Math.PI) / 180);
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);

    // ...and hysteresis: while in flight AND after it resolves, the same
    // pose never re-dispatches (the dispatch re-recorded the reference).
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    sortResolvers[1]({
      generation: mockApi.sort.mock.calls[1][0].generation as number,
      ordering: new Uint32Array([1, 0]),
    });
    await flush();
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('view-axis translation past fraction×radius dispatches; orthogonal translation never does', async () => {
    const coord = await loadCoordinator();
    const camera = makeCamera();
    await sortedSetup(coord, camera);

    // Orthogonal to the view axis (camera looks -z; slide along x): the
    // permutation cannot change (view-z of every splat is unchanged), so
    // even a huge slide must not dispatch.
    camera.position.x += 100;
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // Along the view axis, below 0.05 × radius 10 = 0.5 units: quiet.
    camera.position.z += 0.3;
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // Past the threshold: dispatch (the behind-camera set may change).
    camera.position.z += 0.7;
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('tracks a REPLACED camera object (ortho-mode toggle swaps the camera)', async () => {
    // The scene manager's ortho toggle constructs a NEW camera and
    // replaces `sceneManager.camera` — the coordinator must read the
    // camera through the live getter, not a reference captured at init.
    const coord = await loadCoordinator();
    let current: THREE.Camera = makeCamera();
    await sortedSetup(coord, current, { getCamera: () => current });

    // Swap in a differently-posed camera (as the ortho toggle does).
    const swapped = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
    swapped.position.set(100, 0, 0);
    swapped.lookAt(0, 0, 0);
    current = swapped;

    // The 90° pose change must dispatch on the next frame; a captured
    // init-time camera would never see it.
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('translation threshold scales with the mesh transform (offset normalized by |axis|)', async () => {
    // Scaled mesh (×10): the offset is measured in MODEL-LOCAL units
    // (e[14]/|axis|), matching the model-local bounding radius, so the
    // world-space trigger distance is fraction × radius × scale =
    // 0.05 × 10 × 10 = 5 world units. An UN-normalized offset (raw e[14],
    // world units) would trip at 0.5 world units — 10× too eager.
    const coord = await loadCoordinator();
    const camera = makeCamera();
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(2, 'normal');
    mesh.scale.set(10, 10, 10);
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 1]),
    });
    await flush();

    camera.position.z += 3; // 0.3 local units — under the 0.5 threshold
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    camera.position.z += 4; // 0.7 local units total — past it
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('without geometry bounds the translation trigger is inert (angle-only)', async () => {
    const coord = await loadCoordinator();
    const camera = makeCamera();
    const mesh = await sortedSetup(coord, camera);
    (mesh.geometry as THREE.InstancedBufferGeometry).boundingSphere = null;

    camera.position.z += 100;
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    camera.rotateY((10 * Math.PI) / 180);
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('skips dispatch while a view update is in flight (pending-load signal)', async () => {
    const coord = await loadCoordinator();
    const camera = makeCamera();
    let loading = false;
    await sortedSetup(coord, camera, { isLoadInProgress: () => loading });

    loading = true;
    camera.rotateY(Math.PI / 2);
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // Load settles → the very next frame dispatches.
    loading = false;
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('skips invisible, LOD-demoted, and mode-switched-away nodes', async () => {
    const coord = await loadCoordinator();
    const camera = makeCamera();
    const mesh = await sortedSetup(coord, camera);
    camera.rotateY(Math.PI / 2); // way past the threshold from here on

    mesh.visible = false;
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    mesh.visible = true;

    // A hidden ANCESTOR must gate too: an LOD level can be a group
    // (partition tiles) whose visibility toggle never touches the
    // member meshes' own flags.
    const parent = new THREE.Group();
    parent.add(mesh);
    parent.visible = false;
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    parent.visible = true;

    const committed = mesh.userData.committedData;
    delete mesh.userData.committedData;
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    mesh.userData.committedData = committed;

    (mesh.material as THREE.Material).userData.blendingMode = 'additive';
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    (mesh.material as THREE.Material).userData.blendingMode = 'normal';

    // All gates lifted: the pending camera motion dispatches.
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('setDepthSortEnabled(false) makes the subsystem inert (?depthSort=0)', async () => {
    const coord = await loadCoordinator();
    coord.setDepthSortEnabled(false);
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    // A normal-mode commit neither spawns the worker nor sorts: the
    // identity (storage) ordering is pinned.
    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.initialize).not.toHaveBeenCalled();
    expect(mockApi.sort).not.toHaveBeenCalled();

    // The mode-switch hook must not force a reprocess either.
    coord.noteDepthSortBlendingModeSwitch(mesh, 'normal', 'additive');
    expect(requestReprocess).not.toHaveBeenCalled();

    // And the per-frame scheduler no-ops.
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).not.toHaveBeenCalled();
  });

  it('records each sort round-trip as a Depth Sort profiler pass with splat count + upload bytes', async () => {
    const coord = await loadCoordinator();
    // Fresh module instance to match the coordinator's post-reset module graph.
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    await sortedSetup(coord, makeCamera(), { getProfiler: () => profiler });

    const root = profiler.getDepthSortTimings();
    expect(root.count).toBe(1);
    expect(root.metadata?.splats).toBe(2);
    expect(root.metadata?.info).toMatch(/up$/);
  });

  it('recovers a node whose first dispatch raced a null camera (registered, never sorted)', async () => {
    // Init-ordering window: the commit lands while getCamera still
    // returns null. Registration reaches the worker (it does not need a
    // camera) but no sort dispatches, so `lastSortAxis` stays null and
    // no camera motion could ever re-trigger it — only the per-frame
    // recovery branch can.
    const coord = await loadCoordinator();
    let camera: THREE.Camera | null = null;
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => camera, requestRender });

    const mesh = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(mockApi.sort).not.toHaveBeenCalled();

    // Camera appears: the next frame dispatches exactly one recovery
    // sort; the frame after stays quiet (in flight).
    camera = makeCamera();
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // The recovered sort applies like any other.
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([1, 0]),
    });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([1, 0]);
    expect(requestRender).toHaveBeenCalled();
  });

  it('an empty (count=0) commit silences camera-motion re-sorts until real content returns', async () => {
    // The zero-splat commit releases the worker registration; keeping
    // the recorded sort pose would fire a guaranteed-null sort RPC on
    // every threshold crossing (the slice has no visible splats).
    const coord = await loadCoordinator();
    const camera = makeCamera();
    const mesh = await sortedSetup(coord, camera);

    coord.noteDepthSortCommit(mesh, new Float32Array(0), 0);
    await flush();
    expect(mockApi.releaseNode).toHaveBeenCalledWith(mesh.uuid);

    // Way past the angle threshold: neither the motion trigger (pose
    // cleared) nor the recovery branch (registration cleared) may fire.
    camera.rotateY(Math.PI / 2);
    coord.evaluateDepthSortPerFrame();
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1); // the setup sort only

    // A later non-empty commit restores the full register + sort cycle.
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(2);
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });
});

/**
 * Points integration (the volumetric-phases arc PR-B): points share the
 * whole coordinator mechanism — the deltas under test are the LAZY
 * centers provider, the EFFECTIVE-mode gate (points render `volumetric`
 * as additive until phase 3, so they must not sort in it), and the
 * shared global renderOrder scale.
 */
describe('depth-sort coordinator — points integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  /** A minimal points mesh — same texture-backed shape, points nodeType. */
  function makePointsMesh(count: number, blendingMode: string): THREE.Mesh {
    const mesh = makeGSplatsMesh(count, blendingMode);
    mesh.userData.nodeType = 'points';
    return mesh;
  }

  it('registers + sorts a normal-mode points commit from a LAZY centers provider', async () => {
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender });

    const mesh = makePointsMesh(3, 'normal');
    const produced: Float32Array[] = [];
    const provider = vi.fn(() => {
      const out = new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]);
      produced.push(out);
      return out;
    });
    coord.noteDepthSortCommit(mesh, provider, 3);
    await flush();

    // The provider ran exactly once, and ITS buffer (not some other
    // array) was transferred to the worker.
    expect(provider).toHaveBeenCalledTimes(1);
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(transferCalls[0].transferables).toContain(produced[0].buffer);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 2, 1]),
    });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 2, 1]);
    expect(requestRender).toHaveBeenCalled();
  });

  it('never invokes the provider for an order-independent (additive) commit', async () => {
    // The whole point of the lazy form: the common additive path must
    // not pay the O(N) positions copy.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makePointsMesh(3, 'additive');
    const provider = vi.fn(() => new Float32Array(9));
    coord.noteDepthSortCommit(mesh, provider, 3);
    await flush();

    expect(provider).not.toHaveBeenCalled();
    expect(mockApi.registerNode).not.toHaveBeenCalled();
  });

  it('never invokes the provider when a newer commit superseded the registration', async () => {
    // The provider resolves AFTER the generation re-check inside the
    // worker-init continuation — a commit superseded while the worker
    // was spawning must not pay the copy either.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makePointsMesh(2, 'normal');
    const staleProvider = vi.fn(() => new Float32Array([0, 0, -1, 1, 0, -2]));
    // Two commits back-to-back BEFORE the async worker init settles: the
    // first registration's continuation sees a newer generation and bails.
    coord.noteDepthSortCommit(mesh, staleProvider, 2);
    const freshProvider = vi.fn(() => new Float32Array([0, 0, -3, 1, 0, -4]));
    coord.noteDepthSortCommit(mesh, freshProvider, 2);
    await flush();

    expect(staleProvider).not.toHaveBeenCalled();
    expect(freshProvider).toHaveBeenCalledTimes(1);
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
  });

  it('DOES sort a volumetric points commit (phase 3: real emission–absorption)', async () => {
    // needsDepthSort('volumetric') is true — the material composites
    // order-dependently, so the commit registers with the SortWorker
    // exactly like a normal-mode points commit. Since phase 4 all
    // three geometry types sort volumetric (the per-type
    // effectiveGeometryMode downgrade helper is gone).
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makePointsMesh(3, 'volumetric');
    const provider = vi.fn(() => new Float32Array(9));
    coord.noteDepthSortCommit(mesh, provider, 3);
    await flush();

    expect(provider).toHaveBeenCalledTimes(1);
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(mockApi.sort).toHaveBeenCalled();
  });

  it('mode switches judge the EFFECTIVE mode: points normal→volumetric is sorted→sorted', async () => {
    const coord = await loadCoordinator();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    // Points: normal→volumetric is sorted→sorted since phase 3 — no
    // release, no reprocess (the same contract gsplats have carried
    // since phase 1).
    const points = makePointsMesh(2, 'normal');
    coord.noteDepthSortCommit(points, () => new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    coord.noteDepthSortBlendingModeSwitch(points, 'volumetric', 'normal');
    expect(mockApi.releaseNode).not.toHaveBeenCalled();
    expect(requestReprocess).not.toHaveBeenCalled();

    // ...and additive→volumetric is unsorted→SORTED: the coordinator
    // must request the reprocess that re-commits (and registers) the
    // node — mirroring additive→normal.
    const additivePoints = makePointsMesh(2, 'additive');
    coord.noteDepthSortBlendingModeSwitch(additivePoints, 'volumetric', 'additive');
    expect(requestReprocess).toHaveBeenCalled();

    // Gsplats: the same normal→volumetric switch is sorted→sorted — no
    // release, no reprocess (the existing contract, now shared with
    // points via the effective-mode path).
    requestReprocess.mockClear();
    const gsplats = makeGSplatsMesh(2, 'normal');
    coord.noteDepthSortCommit(gsplats, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    coord.noteDepthSortBlendingModeSwitch(gsplats, 'volumetric', 'normal');
    expect(mockApi.releaseNode).not.toHaveBeenCalled();
    expect(requestReprocess).not.toHaveBeenCalled();
  });

  it('mode switch TO normal clears the points noop stamp and requests a reprocess', async () => {
    const coord = await loadCoordinator();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    const mesh = makePointsMesh(2, 'normal');
    coord.noteDepthSortBlendingModeSwitch(mesh, 'normal', 'additive');
    expect(mesh.userData.committedData).toBeUndefined();
    expect(requestReprocess).toHaveBeenCalledTimes(1);
  });

  it('points and gsplats land on ONE global renderOrder scale by depth', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const farPoints = makePointsMesh(2, 'normal');
    farPoints.geometry.boundingSphere!.center.set(0, 0, -30);
    const nearGSplats = makeGSplatsMesh(2, 'normal');
    nearGSplats.geometry.boundingSphere!.center.set(0, 0, -10);
    const midPoints = makePointsMesh(2, 'normal');
    midPoints.geometry.boundingSphere!.center.set(0, 0, -20);
    // Insertion order deliberately NOT depth order.
    coord.noteDepthSortCommit(nearGSplats, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(farPoints, () => new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(midPoints, () => new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(farPoints.renderOrder).toBe(0);
    expect(midPoints.renderOrder).toBe(1);
    expect(nearGSplats.renderOrder).toBe(2);
  });

  it('camera motion past the angle threshold re-sorts a points node (Phase 3 scheduler)', async () => {
    const coord = await loadCoordinator();
    const camera = makeCamera();
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn() });

    const mesh = makePointsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, () => new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 1]),
    });
    await flush();

    camera.rotateY((5 * Math.PI) / 180); // past the 3° default threshold
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('a zero-count points commit releases instead of registering (empty slice)', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makePointsMesh(2, 'normal');
    coord.noteDepthSortCommit(mesh, () => new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);

    const provider = vi.fn(() => new Float32Array(0));
    coord.noteDepthSortCommit(mesh, provider, 0);
    await flush();
    expect(provider).not.toHaveBeenCalled();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1); // unchanged
    expect(mockApi.releaseNode).toHaveBeenCalledWith(mesh.uuid);
  });
});

/**
 * Lines integration (the volumetric-phases arc PR-C): lines share the
 * whole coordinator mechanism through the same geometry-neutral entry
 * points — the deltas under test are the `'line'` kind mapping (the
 * commit passes segment MIDPOINTS as the lazy centers) and, since
 * phase 4, the shared volumetric contract: lines implement the real
 * emission–absorption state like points/gsplats, so volumetric SORTS.
 */
describe('depth-sort coordinator — lines integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  /** A minimal lines mesh — same texture-backed shape, lines nodeType. */
  function makeLinesMesh(count: number, blendingMode: string): THREE.Mesh {
    const mesh = makeGSplatsMesh(count, blendingMode);
    mesh.userData.nodeType = 'lines';
    return mesh;
  }

  it('registers + sorts a normal-mode lines commit from a LAZY midpoint provider', async () => {
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender });

    const mesh = makeLinesMesh(3, 'normal');
    const produced: Float32Array[] = [];
    const provider = vi.fn(() => {
      const out = new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]);
      produced.push(out);
      return out;
    });
    coord.noteDepthSortCommit(mesh, provider, 3);
    await flush();

    expect(provider).toHaveBeenCalledTimes(1);
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(transferCalls[0].transferables).toContain(produced[0].buffer);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 2, 1]),
    });
    await flush();
    const attr = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute('aSortedIndex');
    expect(Array.from(attr.array as Uint32Array)).toEqual([0, 2, 1]);
    expect(requestRender).toHaveBeenCalled();
  });

  it('DOES sort a volumetric lines commit (phase 4: real emission–absorption)', async () => {
    // needsDepthSort('volumetric') is true — the material composites
    // order-dependently, so the commit registers with the SortWorker
    // exactly like a normal-mode lines commit. Since phase 4 all
    // three geometry types sort volumetric (the per-type
    // effectiveGeometryMode downgrade helper is gone).
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeLinesMesh(3, 'volumetric');
    const provider = vi.fn(() => new Float32Array(9));
    coord.noteDepthSortCommit(mesh, provider, 3);
    await flush();

    expect(provider).toHaveBeenCalledTimes(1);
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(mockApi.sort).toHaveBeenCalled();
  });

  it('mode switches: lines normal→volumetric is sorted→sorted', async () => {
    const coord = await loadCoordinator();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    // Lines: normal→volumetric is sorted→sorted since phase 4 — no
    // release, no reprocess (the same contract points have carried
    // since phase 3 and gsplats since phase 1).
    const lines = makeLinesMesh(2, 'normal');
    coord.noteDepthSortCommit(lines, () => new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);

    coord.noteDepthSortBlendingModeSwitch(lines, 'volumetric', 'normal');
    expect(mockApi.releaseNode).not.toHaveBeenCalled();
    expect(requestReprocess).not.toHaveBeenCalled();

    // ...and additive→volumetric is unsorted→SORTED: the coordinator
    // must request the reprocess that re-commits (and registers) the
    // node — mirroring additive→normal.
    const additiveLines = makeLinesMesh(2, 'additive');
    coord.noteDepthSortBlendingModeSwitch(additiveLines, 'volumetric', 'additive');
    expect(requestReprocess).toHaveBeenCalled();
  });
});

describe('depth-sort coordinator — provider failure semantics', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('a throwing centers provider leaves the node unregistered (no phantom recovery sorts)', async () => {
    // The provider resolves BEFORE the `registered` flag flips: a throw
    // must behave like a registerNode rejection — otherwise the node is
    // stranded with registered=true and no worker-side centers, and the
    // per-frame recovery branch (registered && !lastSortAxis) dispatches
    // a guaranteed-null sort.
    const coord = await loadCoordinator();
    const camera = makeCamera();
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    mesh.userData.nodeType = 'points';
    coord.noteDepthSortCommit(
      mesh,
      () => {
        throw new Error('provider OOM');
      },
      2
    );
    await flush();

    // Nothing reached the worker, no sort dispatched.
    expect(mockApi.registerNode).not.toHaveBeenCalled();
    expect(mockApi.sort).not.toHaveBeenCalled();

    // The per-frame scheduler must NOT enter the recovery branch for
    // this node (registered stayed false).
    coord.evaluateDepthSortPerFrame();
    coord.evaluateDepthSortPerFrame();
    expect(mockApi.sort).not.toHaveBeenCalled();

    // A later healthy commit recovers the full register + sort cycle.
    coord.noteDepthSortCommit(mesh, () => new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
  });
});

describe('depth-sort coordinator — lazy-LOD mode-switch recovery + init settle guard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('switch TO a sorted mode clears the freshness stamp too (lazy-LOD recovery path)', async () => {
    // A hidden RESIDENT lazy LOD level is structurally outside the
    // reprocess sweep — clearing only committedData left it "ready +
    // fresh" in the LOD registry with nothing ever re-committing it, so
    // on re-show it rendered normal-mode UNSORTED until an unrelated
    // slice change. Marking it stale routes it through the registry's
    // settle-gated ready-but-stale reload (maybeKickReload), whose
    // re-commit registers it with the SortWorker.
    const coord = await loadCoordinator();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    const mesh = makeGSplatsMesh(2, 'additive');
    mesh.userData.nodeType = 'points';
    mesh.userData.loadedViewVersion = 7; // committed fresh for view 7
    coord.noteDepthSortBlendingModeSwitch(mesh, 'normal', 'additive');
    expect(mesh.userData.committedData).toBeUndefined();
    expect(mesh.userData.loadedViewVersion).toBeUndefined();
    expect(requestReprocess).toHaveBeenCalledTimes(1);

    // Switching AWAY does NOT touch either stamp (no reprocess needed).
    const away = makeGSplatsMesh(2, 'normal');
    away.userData.loadedViewVersion = 9;
    coord.noteDepthSortBlendingModeSwitch(away, 'additive', 'normal');
    expect(away.userData.committedData).toBeDefined();
    expect(away.userData.loadedViewVersion).toBe(9);
  });

  it('a worker error during startup settles the init promise (no unbounded closure pile-up)', async () => {
    // A worker that dies during ASYNC module evaluation emits 'error'
    // but never settles the Comlink initialize RPC. Every commit's
    // continuation (closing over its centers provider — the full
    // LoadedPointsData for points) would otherwise accumulate on the
    // forever-pending initPromise. The onerror guard must settle it and
    // drain the continuations into the warn-once degrade path.
    const coord = await loadCoordinator();
    // initialize never settles — simulates the wedged RPC.
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(2, 'normal');
    mesh.userData.nodeType = 'points';
    const provider = vi.fn(() => new Float32Array([0, 0, -1, 1, 0, -2]));
    coord.noteDepthSortCommit(mesh, provider, 2);
    await flush();
    // Pending init: nothing registered yet, provider unpaid.
    expect(mockApi.registerNode).not.toHaveBeenCalled();
    expect(provider).not.toHaveBeenCalled();

    // The worker emits an error event (async startup death).
    const w = terminatedWorkers.length === 0 ? null : null; // (worker not terminated yet)
    expect(w).toBeNull();
    const liveWorker = (
      globalThis as unknown as { __lastMockWorker?: { onerror?: (e: unknown) => void } }
    ).__lastMockWorker;
    expect(liveWorker?.onerror).toBeTypeOf('function');
    liveWorker!.onerror!({ message: 'module evaluation failed' });
    await flush();

    // The init settled: the wedged worker was terminated, and later
    // commits degrade gracefully (no throw, no registration).
    expect(terminatedWorkers.length).toBe(1);
    coord.noteDepthSortCommit(mesh, provider, 2);
    await flush();
    expect(mockApi.registerNode).not.toHaveBeenCalled();
    expect(provider).not.toHaveBeenCalled();
  });
});
