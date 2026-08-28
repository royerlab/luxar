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

import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
// Type-only (erased at runtime, so no clash with the dynamically imported
// live profiler module the coordinator holds) — used to type the fake
// profiler that pins the per-pass session lifecycle (issue #713).
import type { UpdateProfiler, UpdateSession } from '../../../profiling/update-profiler';

type SortResult = {
  generation: number;
  ordering: Uint32Array;
  /** Worker timing split (optional in mocks; the real worker always sends both). */
  kernelMs?: number;
  workerMs?: number;
} | null;

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

/**
 * @param syncSortMaxElements - `config.depthSort.syncSortMaxElements` for this
 *   coordinator instance. Defaults to **0** (synchronous first sort OFF)
 *   because everything in this file was written to exercise the ASYNC worker
 *   pipeline: a synchronous sort answers before the worker is even asked, so
 *   it would pre-fill the ordering buffer that these tests read back as the
 *   worker's output. The dedicated `synchronous first sort` block below turns
 *   it on.
 */
async function loadCoordinator(syncSortMaxElements = 0) {
  vi.resetModules();
  terminatedWorkers.length = 0;
  transferCalls = [];
  sortResolvers = [];
  sortRejectors = [];
  workerConstructThrows = false;
  // Drop the previous test's worker instance: `__lastMockWorker` is global
  // while the coordinator is re-imported per test, so a stale handle would
  // let a test fire `onerror` into the PREVIOUS module instance and silently
  // assert nothing about the one under test.
  delete (globalThis as unknown as { __lastMockWorker?: unknown }).__lastMockWorker;
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

  const coordinator = await import('../../../rendering/depth-sort-coordinator');
  // `vi.resetModules()` above means this is the SAME fresh config instance the
  // coordinator just imported, so mutating it here reaches it.
  const { config } = await import('../../../config');
  config.depthSort.syncSortMaxElements = syncSortMaxElements;
  return coordinator;
}

/**
 * Drive the per-frame pump until a staged ordering has swapped in, and
 * return the buffer the shaders now read.
 *
 * A resolve no longer changes what renders on its own: the ordering
 * streams into the INACTIVE buffer and the slot flips once that buffer
 * holds the whole permutation, so the pump owns every slice AND the
 * swap. Imported dynamically so it resolves to the same module instance
 * the freshly-loaded coordinator holds (`loadCoordinator` resets the
 * registry).
 */
async function applyStagedOrdering(mesh: THREE.Mesh): Promise<Uint32Array> {
  const { pumpSortedIndexOrderingApply, getActiveSortedIndexAttribute } =
    await import('../../../rendering/element-storage');
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
  for (let guard = 0; ; guard++) {
    const result = pumpSortedIndexOrderingApply(geometry);
    // Production pumps before every rendered frame. The draw uploads the
    // staged slice (releasing #715 back-pressure), then onAfterRender
    // acknowledges a newly selected ordering (#713).
    simulateMeshRender(mesh);
    if (!result.more) break;
    if (guard > 64) throw new Error('ordering stream did not converge');
  }
  return getActiveSortedIndexAttribute(geometry)!.array as Uint32Array;
}

/**
 * Simulate THREE completing a visible mesh draw. Attribute uploads happen
 * before the draw callback: fire `onUploadCallback` only for attributes with
 * pending ranges, then invoke the coordinator's chained `onAfterRender` hook.
 * This models both #715 upload back-pressure and #713 applied acknowledgement
 * without constructing a WebGL renderer.
 */
function simulateMeshRender(mesh: THREE.Mesh): void {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
  for (const name of ['aSortedIndex', 'aSortedIndexB'] as const) {
    const attr = geometry.getAttribute(name) as THREE.InstancedBufferAttribute | null;
    if (attr && attr.updateRanges.length > 0) {
      attr.onUploadCallback?.();
      attr.clearUpdateRanges();
    }
  }
  const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  mesh.onAfterRender(
    {} as THREE.WebGLRenderer,
    new THREE.Scene(),
    makeCamera(),
    geometry,
    material,
    null as unknown as THREE.Group
  );
}

/** Run the coordinator's pre-render work, then acknowledge visible draws. */
function evaluateAndRender(
  coord: { evaluateDepthSortPerFrame(): void },
  ...meshes: THREE.Mesh[]
): void {
  coord.evaluateDepthSortPerFrame();
  for (const mesh of meshes) simulateMeshRender(mesh);
}

/** The buffer the shaders are reading right now (no pumping). */
function drawnOrdering(mesh: THREE.Mesh): Uint32Array {
  const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
  const slot = (geometry.userData as { sortedIndexSlot?: 0 | 1 }).sortedIndexSlot ?? 0;
  const attr = geometry.getAttribute(slot === 1 ? 'aSortedIndexB' : 'aSortedIndex');
  return attr.array as Uint32Array;
}

/** A minimal gsplats mesh with a REAL aSortedIndex attribute. */
function makeGSplatsMesh(count: number, blendingMode: string): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  const attr = new THREE.InstancedBufferAttribute(new Uint32Array(count), 1);
  geometry.setAttribute('aSortedIndex', attr);
  // Mirror attachElementStorage: TWO distinct ordering buffers from the
  // start, so the attribute set never changes under a cached WebGPU
  // vertex layout.
  geometry.setAttribute(
    'aSortedIndexB',
    new THREE.InstancedBufferAttribute(new Uint32Array(count), 1)
  );
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

    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 2, 1]);
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

    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 2, 1]);
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
    // Nothing was even staged, so the drawn buffer is untouched.
    expect(Array.from(drawnOrdering(mesh))).toEqual([0, 0, 0]);

    // The queued re-sort was issued for the current (newer) generation.
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    const freshGeneration = mockApi.sort.mock.calls[1][0].generation as number;
    expect(freshGeneration).not.toBe(staleGeneration);

    // The fresh ordering applies.
    sortResolvers[1]({ generation: freshGeneration, ordering: new Uint32Array([1, 2, 0]) });
    await flush();
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([1, 2, 0]);
  });

  it('records a profiler completion once per APPLIED ordering (issue #711)', async () => {
    const coord = await loadCoordinator();
    // Fresh module instance to match the coordinator's post-reset module graph.
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    const recordSpy = vi.spyOn(profiler, 'recordDepthSortCompletion');
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    const generation = mockApi.sort.mock.calls[0][0].generation as number;

    sortResolvers[0]({
      generation,
      ordering: new Uint32Array([0, 2, 1]),
      kernelMs: 4,
      workerMs: 6,
    });
    await flush();

    // A worker resolve only stages the ordering. The completion stream must
    // stay empty until the inactive buffer has been filled and flipped in.
    expect(recordSpy).not.toHaveBeenCalled();
    expect(profiler.getDepthSortCompletions().total).toBe(0);
    await applyStagedOrdering(mesh);

    // Exactly one APPLIED completion landed on the monotonic stream.
    expect(recordSpy).toHaveBeenCalledTimes(1);
    const { total, events } = profiler.getDepthSortCompletions();
    expect(total).toBe(1);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ seq: 1, kernelMs: 4, elements: 3 });
    // boundaryMs = workerMs − kernelMs; queueMs/lastMs are wall-clock, so
    // just assert boundaryMs and non-negative/finite latencies.
    expect(events[0].boundaryMs).toBe(2);
    expect(events[0].queueMs).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(events[0].lastMs)).toBe(true);
  });

  it('does NOT record a profiler completion when a stale ordering is dropped (issue #711)', async () => {
    const coord = await loadCoordinator();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    const staleGeneration = mockApi.sort.mock.calls[0][0].generation as number;

    // A newer commit lands while the first sort is in flight (generation bumps).
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -10, 2, 0, -5]), 3);
    await flush();

    // The stale ordering resolves — dropped by the generation guard, so no
    // fresh ordering and no completion recorded.
    sortResolvers[0]({
      generation: staleGeneration,
      ordering: new Uint32Array([2, 1, 0]),
      kernelMs: 4,
      workerMs: 6,
    });
    await flush();
    expect(profiler.getDepthSortCompletions().total).toBe(0);

    // The re-issued fresh sort DOES record exactly one completion.
    const freshGeneration = mockApi.sort.mock.calls[1][0].generation as number;
    sortResolvers[1]({
      generation: freshGeneration,
      ordering: new Uint32Array([1, 2, 0]),
      kernelMs: 4,
      workerMs: 6,
    });
    await flush();
    expect(profiler.getDepthSortCompletions().total).toBe(0);
    await applyStagedOrdering(mesh);
    expect(profiler.getDepthSortCompletions().total).toBe(1);
  });

  it('records both completions when two nodes resolve OUT OF ORDER (issue #711 acceptance criterion 3)', async () => {
    const coord = await loadCoordinator();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const meshA = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(meshA, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    const meshB = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(meshB, new Float32Array([0, 0, -1, 1, 0, -10, 2, 0, -5]), 3);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);

    // Resolve and apply meshB (dispatch 1) FIRST, then meshA (dispatch 0).
    // The dedicated stream must retain one event per applied ordering in
    // application order, independent of the aggregate timing root.
    sortResolvers[1]({
      generation: mockApi.sort.mock.calls[1][0].generation as number,
      ordering: new Uint32Array([2, 0, 1]),
      kernelMs: 4,
      workerMs: 6,
    });
    await flush();
    expect(profiler.getDepthSortCompletions().total).toBe(0);
    await applyStagedOrdering(meshB);

    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 2, 1]),
      kernelMs: 4,
      workerMs: 6,
    });
    await flush();
    expect(profiler.getDepthSortCompletions().total).toBe(1);
    await applyStagedOrdering(meshA);

    const { total, events } = profiler.getDepthSortCompletions();
    expect(total).toBe(2);
    expect(events.map((e) => e.seq)).toEqual([1, 2]);
  });

  it('does NOT record a completion when the mesh was demoted mid-sort (stillCommitted guard)', async () => {
    const coord = await loadCoordinator();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    const generation = mockApi.sort.mock.calls[0][0].generation as number;

    // LOD demotion mid-sort: the geometry returned to the pool, so the
    // committedData stamp is cleared while the sort is still out.
    delete mesh.userData.committedData;

    // The ordering resolves with the CURRENT (matching) generation, so the
    // generation guard passes — but hasCommittedData() is now false, so the
    // ordering is NOT applied and NO completion is recorded. This pins the
    // recording inside the `stillCommitted` conjunct, not just the generation
    // guard.
    sortResolvers[0]({
      generation,
      ordering: new Uint32Array([0, 2, 1]),
      kernelMs: 4,
      workerMs: 6,
    });
    await flush();

    expect(profiler.getDepthSortCompletions().total).toBe(0);
  });

  it('does NOT record a completion when the profiler was reset while the sort was in flight', async () => {
    const coord = await loadCoordinator();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    const generation = mockApi.sort.mock.calls[0][0].generation as number;

    // reset() bumps the profiler generation, abandoning the in-flight
    // sort's session — the completion record must honor the same
    // reset-isolation contract and NOT land in the fresh stream.
    profiler.reset();

    sortResolvers[0]({
      generation,
      ordering: new Uint32Array([0, 2, 1]),
      kernelMs: 4,
      workerMs: 6,
    });
    await flush();
    await applyStagedOrdering(mesh);

    expect(profiler.getDepthSortCompletions().total).toBe(0);
  });

  it('records against the DISPATCH-TIME profiler, not one swapped in mid-sort', async () => {
    const coord = await loadCoordinator();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const oldProfiler = new UpdateProfiler();
    let profiler = oldProfiler;
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    const generation = mockApi.sort.mock.calls[0][0].generation as number;

    // A replacement profiler installed while the sort is in flight must
    // start with a clean completion stream; the completion belongs to the
    // profiler whose session tracked the dispatch.
    const newProfiler = new UpdateProfiler();
    profiler = newProfiler;

    sortResolvers[0]({
      generation,
      ordering: new Uint32Array([0, 2, 1]),
      kernelMs: 4,
      workerMs: 6,
    });
    await flush();
    expect(oldProfiler.getDepthSortCompletions().total).toBe(0);
    await applyStagedOrdering(mesh);

    expect(newProfiler.getDepthSortCompletions().total).toBe(0);
    expect(oldProfiler.getDepthSortCompletions().total).toBe(1);
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
    // Nothing was even staged, so the drawn buffer is untouched.
    expect(Array.from(drawnOrdering(mesh))).toEqual([0, 0, 0]);

    // Lifetime B's own ordering applies normally.
    sortResolvers[1]({ generation: generationB, ordering: new Uint32Array([1, 0]) });
    await flush();
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([1, 0, 0]);
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

  it('resortForCapture dispatches a pose-fresh sort and drains it without re-arming the rAF loop', async () => {
    // Simulates offline capture (gallery orbit): the rAF loop is stopped, so
    // the per-frame depth-sort scheduler never runs. resortForCapture must
    // both dispatch a fresh sort for the current pose AND drain it all the
    // way to the drawn buffer — WITHOUT calling requestRender (which would
    // restart the very loop the capture stopped).
    const coord = await loadCoordinator();
    const camera = makeCamera();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => camera, requestRender });

    // Establish an initial ordering the way a normal commit would.
    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    const gen0 = mockApi.sort.mock.calls[0][0].generation as number;
    sortResolvers[0]({ generation: gen0, ordering: new Uint32Array([0, 2, 1]) });
    await flush();
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 2, 1]);

    const before = mockApi.sort.mock.calls.length;
    const rrBefore = requestRender.mock.calls.length;

    // Kick the capture re-sort but DON'T await yet: a fresh sort must have
    // been dispatched synchronously for the current pose.
    const p = coord.resortForCapture();
    expect(mockApi.sort.mock.calls.length).toBe(before + 1);

    // Resolve the newest sort with a DIFFERENT ordering, then let the drain
    // stream + swap it in (real timers: resortForCapture uses setTimeout(0)).
    const genCap = mockApi.sort.mock.calls[before][0].generation as number;
    sortResolvers[sortResolvers.length - 1]({
      generation: genCap,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await p;

    // The drain applied the fresh ordering to the drawn buffer …
    expect(Array.from(drawnOrdering(mesh))).toEqual([2, 0, 1]);
    // … and resortForCapture never re-armed the (stopped) rAF loop.
    expect(requestRender.mock.calls.length).toBe(rrBefore);

    // The QUIESCENT exit (the common successful-capture path) must ALSO have
    // restored the suppressed requestRender hook — guards against the restore
    // drifting out of `finally` into the drain body (a mutation that would
    // leave this path unrestored yet still pass the assertions above). Prove
    // it via a NORMAL commit (not a capture): its resolve/staging path calls
    // requestRender?.(), which only fires if the real fn is back in place.
    const rrAfterCapture = requestRender.mock.calls.length;
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -3, 1, 0, -7, 2, 0, -2]), 3);
    await flush();
    const genPost = mockApi.sort.mock.calls[mockApi.sort.mock.calls.length - 1][0]
      .generation as number;
    sortResolvers[sortResolvers.length - 1]({
      generation: genPost,
      ordering: new Uint32Array([1, 0, 2]),
    });
    await flush();
    expect(requestRender.mock.calls.length).toBeGreaterThan(rrAfterCapture);
  });

  it('resortForCapture dispatches exactly ONE sort per call when the camera moved past the threshold', async () => {
    // The capture path composes TWO dispatchers: the force loop and the
    // per-frame pass's camera-motion trigger. When the camera moved past
    // angleThresholdDeg since the last sort (the first orbit frame after
    // repositioning), both are eligible for the SAME pose — run in the
    // wrong order, the second only sets `resortQueued` and a redundant
    // identical full sort runs serially after the first. Guard the
    // compose-to-one-sort property.
    const coord = await loadCoordinator();
    const camera = makeCamera();
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 2, 1]),
    });
    await flush();
    await applyStagedOrdering(mesh);

    // Rotate the view axis ~45° about the mesh — far past the 3° default
    // angle threshold, so the per-frame motion trigger is armed.
    camera.position.set(10, 0, 10);
    camera.lookAt(0, 0, -5);
    camera.updateMatrixWorld();

    const before = mockApi.sort.mock.calls.length;
    const p = coord.resortForCapture(200);
    expect(mockApi.sort.mock.calls.length).toBe(before + 1);
    sortResolvers[sortResolvers.length - 1]({
      generation: mockApi.sort.mock.calls[before][0].generation as number,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await p;

    // Exactly one sort for the pose — the motion trigger did not queue a
    // duplicate behind the force loop's dispatch (nor vice versa).
    expect(mockApi.sort.mock.calls.length).toBe(before + 1);
    expect(Array.from(drawnOrdering(mesh))).toEqual([2, 0, 1]);
  });

  it('resortForCapture is bounded by maxWaitMs and restores requestRender in finally', async () => {
    // Guards two properties: (a) a wedged/never-resolving worker sort can
    // NEVER hang the offline capture — resortForCapture(maxWaitMs) exits on
    // the time bound regardless; and (b) the `finally` restores the
    // `requestRender` hook it suppressed (the helper is exposed on
    // `__luxarDebug`), so a later sort resolve wakes the loop again.
    const coord = await loadCoordinator();
    const camera = makeCamera();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => camera, requestRender });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    // First sort dispatched; its resolver is parked and NEVER resolved here.
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    const gen = mockApi.sort.mock.calls[0][0].generation as number;

    const rrBefore = requestRender.mock.calls.length;

    // The parked in-flight sort keeps the node non-quiescent, so the drain
    // loop runs until the time bound. Simply resolving proves no hang.
    await coord.resortForCapture(50);

    // No frame request escaped during the suppressed drain.
    expect(requestRender.mock.calls.length).toBe(rrBefore);

    // The parked sort finally resolves — the resolve/staging path calls
    // requestRender?.(), which is only possible if `finally` restored it.
    sortResolvers[sortResolvers.length - 1]({
      generation: gen,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await flush();
    expect(requestRender.mock.calls.length).toBeGreaterThan(rrBefore);
  });

  it('resortForCapture drains a MULTI-SLICE chunked apply to the drawn buffer without any draw', async () => {
    // The offline drain never draws, but on the chunked (WebGL) path a
    // multi-slice apply stalls after its first slice until a DRAW's upload
    // ack releases the #715 back-pressure. Without the capture-time bypass,
    // any order-dependent node past one slice (>1M elements in production —
    // e.g. the 3M-star gaia gallery demo) could never reach quiescence:
    // every orbit frame would burn the full maxWaitMs AND still film a
    // stale ordering. Shrink the slice to 1 element so a 3-element ordering
    // needs 3 slices, then prove the drain completes and flips WITHOUT a
    // single simulated draw.
    const coord = await loadCoordinator();
    const camera = makeCamera();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => camera, requestRender });

    const { setSortedIndexChunkElementsForTests, configureSortedIndexChunkedApply } =
      await import('../../../rendering/element-storage');
    configureSortedIndexChunkedApply(true);
    setSortedIndexChunkElementsForTests(1);

    // Establish an initial drawn ordering the normal way (draws allowed here).
    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 2, 1]),
    });
    await flush();
    await applyStagedOrdering(mesh);
    expect(Array.from(drawnOrdering(mesh))).toEqual([0, 2, 1]);

    // Capture: the forced sort resolves to a 3-slice apply. NO draw happens
    // between here and the await — the drain must still converge well inside
    // the bound (a stall would spin until maxWaitMs and leave the old
    // ordering drawn, failing both assertions below).
    const before = mockApi.sort.mock.calls.length;
    const rrBefore = requestRender.mock.calls.length;
    const p = coord.resortForCapture(2000);
    expect(mockApi.sort.mock.calls.length).toBe(before + 1);
    const start = performance.now();
    sortResolvers[sortResolvers.length - 1]({
      generation: mockApi.sort.mock.calls[before][0].generation as number,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await p;

    expect(Array.from(drawnOrdering(mesh))).toEqual([2, 0, 1]);
    expect(performance.now() - start).toBeLessThan(1500);
    expect(requestRender.mock.calls.length).toBe(rrBefore);

    // The bypass is scoped to the capture: a post-capture stream stalls on
    // its unflushed slice again (back-pressure restored).
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -3, 1, 0, -7, 2, 0, -2]), 3);
    await flush();
    sortResolvers[sortResolvers.length - 1]({
      generation: mockApi.sort.mock.calls[mockApi.sort.mock.calls.length - 1][0]
        .generation as number,
      ordering: new Uint32Array([1, 0, 2]),
    });
    await flush();
    coord.evaluateDepthSortPerFrame(); // slice 1 written (no draw yet)
    const drawnBefore = Array.from(drawnOrdering(mesh));
    coord.evaluateDepthSortPerFrame(); // must STALL, not advance to a flip
    coord.evaluateDepthSortPerFrame();
    expect(Array.from(drawnOrdering(mesh))).toEqual(drawnBefore);

    setSortedIndexChunkElementsForTests(null);
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
    expect(Array.from(await applyStagedOrdering(meshB))).toEqual([1, 0]);
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
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 0]);
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

    expect(far.renderOrder).toBe(1);
    expect(near.renderOrder).toBe(2);
    expect(mockApi.sort).not.toHaveBeenCalled(); // worker path stays inert
  });

  it('keeps partition renderOrder a permutation while a load sweep is in progress', async () => {
    const bspTree = {
      axis: 0,
      split: 0,
      left: { axis: 0, split: -50, left: { part: 0 }, right: { part: 1 } },
      right: { axis: 0, split: 50, left: { part: 2 }, right: { part: 3 } },
    };
    const coord = await loadCoordinator();
    coord.configureDepthSort({
      getCamera: () => cameraAt(1000, 0, 0),
      requestRender: vi.fn(),
      isLoadInProgress: () => true,
    });

    const parts = [0, 1, 2, 3].map(() => makeGSplatsMesh(2, 'normal'));
    makePartitionWrapper(bspTree, parts);
    // The last part is still the empty, never-committed placeholder and is
    // therefore not tracked by the coordinator yet.
    for (const part of parts.slice(0, 3)) {
      coord.noteDepthSortCommit(part, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    const ranks = parts.map((part) => part.renderOrder);
    expect(new Set(ranks).size).toBe(parts.length);
    expect(ranks).toEqual([1, 2, 3, 0]);
  });

  it('keeps the BSP rank of a stamp-less tracked part during a load sweep', async () => {
    const bspTree = {
      axis: 0,
      split: 0,
      left: { axis: 0, split: -50, left: { part: 0 }, right: { part: 1 } },
      right: { axis: 0, split: 50, left: { part: 2 }, right: { part: 3 } },
    };
    const coord = await loadCoordinator();
    coord.configureDepthSort({
      getCamera: () => cameraAt(1000, 0, 0),
      requestRender: vi.fn(),
      isLoadInProgress: () => true,
    });

    const parts = [0, 1, 2, 3].map(() => makeGSplatsMesh(2, 'normal'));
    makePartitionWrapper(bspTree, parts);
    for (const part of parts) {
      coord.noteDepthSortCommit(part, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();
    // Remove the FAR part's stamp. Merely assigning unranked parts after all
    // ranked parts would preserve uniqueness but produce the wrong BSP order;
    // the stored BSP tree still has enough information for the exact order.
    delete parts[0].userData.committedData;

    coord.evaluateDepthSortPerFrame();

    const ranks = parts.map((part) => part.renderOrder);
    expect(new Set(ranks).size).toBe(parts.length);
    expect(ranks).toEqual([1, 2, 3, 4]);
  });

  it('a switch into a sorted mode keeps its rank while the commit stamp is invalidated', async () => {
    const coord = await loadCoordinator();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
      isLoadInProgress: () => false,
    });

    const switched = makeGSplatsMesh(2, 'additive');
    switched.geometry.boundingSphere!.center.set(0, 0, -30);
    const stable = makeGSplatsMesh(2, 'normal');
    stable.geometry.boundingSphere!.center.set(0, 0, -10);
    coord.noteDepthSortCommit(switched, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(stable, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    // LayersPanel restamps the material before notifying the coordinator.
    // The invalidated commit stamp forces a reprocess, but the existing
    // geometry remains visible and must keep its cross-mesh rank meanwhile.
    (switched.material as THREE.Material).userData.blendingMode = 'normal';
    coord.noteDepthSortBlendingModeSwitch(switched, 'normal', 'additive');
    expect(switched.userData.committedData).toBeUndefined();
    expect(requestReprocess).toHaveBeenCalledTimes(1);

    coord.evaluateDepthSortPerFrame();

    expect(switched.renderOrder).toBe(1);
    expect(stable.renderOrder).toBe(2);
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

    // A was released mid-flight, so its ordering never even staged.
    expect(Array.from(drawnOrdering(meshA))).toEqual([0, 0]); // discarded
    expect(Array.from(await applyStagedOrdering(meshB))).toEqual([1, 0]); // applied
  });

  it('orders BSP-partition parts back-to-front from the stored split planes (camera outside)', async () => {
    // Partition parts share the world origin (splat centers baked in), so
    // THREE's per-object transparent sort gives every part the SAME key and
    // draws them in creation order. With a stored bspTree the coordinator
    // instead assigns each part its exact painter's order (the BSP's internal
    // rank 0 becomes the first live renderOrder rank, 1).
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

    // THREE draws transparent objects by renderOrder ASCENDING, so the
    // farthest part (x<-50) receives the first reserved live rank.
    expect(parts.map((m) => m.renderOrder)).toEqual([1, 2, 3, 4]);
  });

  it('a tree that names only SOME parts falls the whole group back to depth order', async () => {
    // `partRank` is `ranks.get(partIndex) ?? -1`, so a stored tree whose leaf set
    // does not cover every part leaves ranked and unranked members side by side.
    // Comparing those two keys in one comparator is non-transitive (ranked pairs
    // by rank, mixed pairs by depth), and Array.sort may then return an order
    // satisfying NEITHER key. The group must fall back to depth wholesale, which
    // is approximate but well-defined.
    //
    // Tree ranks parts 0 and 1 only; parts 2 and 3 are unnamed. Depths are set so
    // the correct depth order (far → near) is 3, 2, 1, 0 — the exact REVERSE of
    // the partial ranking, so a comparator that let the ranks leak through could
    // not produce this result by luck.
    const bspTree = { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } };
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const parts = [0, 1, 2, 3].map(() => makeGSplatsMesh(2, 'normal'));
    parts.forEach((mesh, i) => {
      // More negative view-z = farther, so part 3 is farthest.
      mesh.geometry.boundingSphere!.center.set(0, 0, -10 * (i + 1));
    });
    makePartitionWrapper(bspTree, parts);
    for (const m of parts) coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // Farthest (part 3) draws first.
    expect(parts.map((m) => m.renderOrder)).toEqual([4, 3, 2, 1]);
  });

  it('maps BSP split axes through displayDims, not straight to x/y/z', async () => {
    // A serialized `axis` is a CENTER-COLUMN index; `eyeLocal` is in display
    // space (x/y/z = displayDims[0..2]). Reading axis 1 as "y" is only right
    // when displayDims == [0,1,2]. Here the scene displays [3, 1, 0], so center
    // column 0 is the THIRD display axis (z) — the split must consult
    // eyeLocal.z, not eyeLocal.x.
    const bspTree = { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } };

    const orderFor = async (displayedDims: number[]): Promise<number[]> => {
      const coord = await loadCoordinator();
      // Camera on -z only. Under the correct mapping (column 0 → z) the eye is
      // on the small-coord side, so the BSP flips the parts → [1, 0]. Under the
      // old naive reading (column 0 → x) it would read eyeLocal.x == 0 and land
      // on the other branch → [0, 1].
      coord.configureDepthSort({
        getCamera: () => cameraAt(0, 0, -1000),
        requestRender: vi.fn(),
        getDisplayDims: () => displayedDims,
      });
      const parts = [0, 1].map(() => makeGSplatsMesh(2, 'normal'));
      makePartitionWrapper(bspTree, parts);
      for (const m of parts) {
        coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
      }
      await flush();
      coord.evaluateDepthSortPerFrame();
      return parts.map((m) => m.renderOrder);
    };

    // Column 0 displayed as z → split read on z (eye at -1000) → parts flipped.
    expect(await orderFor([3, 1, 0])).toEqual([2, 1]);
    // Same tree, same camera, but column 0 displayed as x → eye.x == 0 → the
    // other branch. Different answer from the SAME tree is the whole point:
    // the mapping is consulted, not assumed.
    expect(await orderFor([0, 1, 2])).toEqual([1, 2]);
  });

  it('maps BSP split axes above column 2 when that column is displayed', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({
      getCamera: () => cameraAt(0, 0, -1000),
      requestRender: vi.fn(),
      getDisplayDims: () => [1, 2, 3],
    });
    const parts = [0, 1].map(() => makeGSplatsMesh(2, 'normal'));
    makePartitionWrapper({ axis: 3, split: 0, left: { part: 0 }, right: { part: 1 } }, parts);
    for (const mesh of parts) {
      coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(parts.map((mesh) => mesh.renderOrder)).toEqual([2, 1]);
  });

  it.each([
    ['identity display dims', [0, 1, 2]],
    ['empty display dims', []],
    ['missing display dims accessor', undefined],
  ] as const)('falls back when axis 3 is hidden with %s', async (_label, displayedDims) => {
    const orderFor = async (withTree: boolean): Promise<number[]> => {
      const coord = await loadCoordinator();
      coord.configureDepthSort({
        getCamera: () => cameraAt(0, 0, -1000),
        requestRender: vi.fn(),
        ...(displayedDims === undefined ? {} : { getDisplayDims: () => displayedDims }),
      });
      const parts = [0, 1].map(() => makeGSplatsMesh(2, 'normal'));
      makePartitionWrapper(
        withTree ? { axis: 3, split: 0, left: { part: 0 }, right: { part: 1 } } : undefined,
        parts
      );
      for (const mesh of parts) {
        coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
      }
      await flush();
      coord.evaluateDepthSortPerFrame();
      return parts.map((mesh) => mesh.renderOrder);
    };

    expect(await orderFor(true)).toEqual(await orderFor(false));
  });

  it('falls back to the centroid heuristic when a split axis is not displayed', async () => {
    // displayDims == [1, 2] (a 2D view of 3D+ data) leaves center column 0 off
    // screen, so its split plane carries no on-screen depth information: the
    // coordinator must decline the BSP ranks rather than order along an axis
    // the viewer isn't showing.
    //
    // Each assertion picks the camera that makes ITS property observable — a
    // single pose can't do both, and a pose where the outcomes coincide would
    // make the assertion vacuous.
    const bspTree = { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } };

    const run = async (
      displayedDims: number[],
      tree: unknown,
      cam: [number, number, number]
    ): Promise<number[]> => {
      const coord = await loadCoordinator();
      coord.configureDepthSort({
        getCamera: () => cameraAt(...cam),
        requestRender: vi.fn(),
        getDisplayDims: () => displayedDims,
      });
      const parts = [0, 1].map(() => makeGSplatsMesh(2, 'normal'));
      makePartitionWrapper(tree, parts);
      for (const m of parts) {
        coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
      }
      await flush();
      coord.evaluateDepthSortPerFrame();
      return parts.map((m) => m.renderOrder);
    };

    // Declining an unusable tree == having no tree at all. Camera on -z: were an
    // unmapped axis (component -1) to fall through to eyeLocal.z instead of
    // being declined, the traversal would flip the parts here — so this pose is
    // what makes the decline observable rather than coincidental.
    const onZ: [number, number, number] = [0, 0, -1000];
    expect(await run([1, 2], bspTree, onZ)).toEqual(await run([1, 2], undefined, onZ));

    // ...and with that same axis ON screen the tree really does change the
    // answer, so the equivalence above is a genuine fallback. Camera on -x,
    // where the BSP's column-0 split and the centroid heuristic disagree.
    const onX: [number, number, number] = [-1000, 0, 0];
    expect(await run([0, 1, 2], bspTree, onX)).not.toEqual(await run([0, 1, 2], undefined, onX));
  });

  it('warns once when a valid BSP tree is rejected only by display-axis mapping', async () => {
    const coord = await loadCoordinator();
    let displayedDims = [0, 1, 2];
    coord.configureDepthSort({
      getCamera: () => cameraAt(0, 0, -1000),
      requestRender: vi.fn(),
      getDisplayDims: () => displayedDims,
    });
    const parts = [0, 1].map(() => makeGSplatsMesh(2, 'normal'));
    const wrapper = makePartitionWrapper(
      { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } },
      parts
    );
    wrapper.name = '/hidden-first';
    for (const mesh of parts) {
      coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    const { log } = await import('../../../utils/log');
    expect(log.warning).not.toHaveBeenCalled();

    displayedDims = [1, 2, 3];
    coord.evaluateDepthSortPerFrame();
    coord.evaluateDepthSortPerFrame();

    expect(log.warning).toHaveBeenCalledTimes(1);
    expect(log.warning).toHaveBeenCalledWith(
      'RENDERER',
      expect.stringContaining(
        'partition-kind group /hidden-first has a valid bsp_tree whose split axes are not all displayed'
      )
    );
    expect(log.warning).toHaveBeenCalledWith(
      'RENDERER',
      expect.stringContaining('falling back to centroid ordering')
    );
  });

  it('orders a partition wrapper without a bsp tree like the centroid-only path', async () => {
    // `undefined` is the state a rejected tree leaves; the loader half of that
    // contract is covered in load-partition-group-node.test.ts.
    const run = async (wrapped: boolean): Promise<number[]> => {
      const coord = await loadCoordinator();
      coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
      const parts = [0, 1, 2].map(() => makeGSplatsMesh(2, 'normal'));
      parts[0].geometry.boundingSphere!.center.z = -10;
      parts[1].geometry.boundingSphere!.center.z = -30;
      parts[2].geometry.boundingSphere!.center.z = -20;
      if (wrapped) makePartitionWrapper(undefined, parts);
      for (const mesh of parts) {
        coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
      }
      await flush();
      coord.evaluateDepthSortPerFrame();
      return parts.map((mesh) => mesh.renderOrder);
    };

    const wrappedOrder = await run(true);
    expect(wrappedOrder).toEqual(await run(false));
    expect(wrappedOrder).toEqual([3, 1, 2]);
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

    // ranks: part0→1, part1→2, part3→3, part2→4.
    expect(parts.map((m) => m.renderOrder)).toEqual([1, 2, 4, 3]);
  });

  it('falls back to content-centroid view depth when a mesh has no BSP tree (single leaf / legacy)', async () => {
    // Leaf meshes without a partition wrapper: each is its own order group,
    // and groups sort by their bounding-sphere-center view-space z (camera
    // at the origin looking −z, so view ≈ identity and view-z ≈ center.z).
    // renderOrder is the GLOBAL sequential rank (1 = farthest), not the raw
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

    expect(far.renderOrder).toBe(1);
    expect(mid.renderOrder).toBe(2);
    expect(near.renderOrder).toBe(3);
  });

  it('a node whose bounds contain another node draws FIRST even when its centroid is nearer', async () => {
    // The embedded-marker case (galaxy demo): a tiny reference-marker node
    // sits INSIDE a huge cloud node. The cloud's bounds-center view-z is
    // NEARER than the marker's here, so pure centroid depth would draw the
    // cloud last — and in an order-dependent mode (volumetric) the cloud's
    // fragments would multiply the already-drawn marker by transmittance
    // ≈ 0, erasing it for ~half of all camera orientations. Containment
    // must override depth: container first, embedded content on top.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const cloud = makeGSplatsMesh(2, 'volumetric');
    cloud.geometry.boundingSphere!.center.set(0, 0, -10);
    cloud.geometry.boundingSphere!.radius = 100;
    const marker = makeGSplatsMesh(2, 'volumetric');
    marker.geometry.boundingSphere!.center.set(0, 0, -30); // inside the cloud
    marker.geometry.boundingSphere!.radius = 1;

    // Marker committed FIRST so neither insertion order nor centroid depth
    // (marker -30 is farther than cloud -10) could accidentally pass this.
    coord.noteDepthSortCommit(marker, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(cloud, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(cloud.renderOrder).toBe(1);
    expect(marker.renderOrder).toBe(2);
  });

  it('nested containment orders container → inner container → innermost', async () => {
    // A ⊃ B ⊃ C: edges point strictly larger → smaller sphere, so the
    // topological pass must emit A, B, C even though pure depth would put
    // C (view-z -32, farthest) before B (-30) before A (-10).
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const a = makeGSplatsMesh(2, 'volumetric');
    a.geometry.boundingSphere!.center.set(0, 0, -10);
    a.geometry.boundingSphere!.radius = 100;
    const b = makeGSplatsMesh(2, 'volumetric');
    b.geometry.boundingSphere!.center.set(0, 0, -30);
    b.geometry.boundingSphere!.radius = 10;
    const c = makeGSplatsMesh(2, 'volumetric');
    c.geometry.boundingSphere!.center.set(0, 0, -32);
    c.geometry.boundingSphere!.radius = 1;

    for (const m of [c, b, a]) {
      coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(a.renderOrder).toBe(1);
    expect(b.renderOrder).toBe(2);
    expect(c.renderOrder).toBe(3);
  });

  it('containment leaves DISJOINT nodes on plain farthest-first depth order', async () => {
    // A far node outside the cloud must keep its depth rank (drawn first);
    // only the embedded marker is hoisted after its container. Expected
    // order: farAway (-200), cloud (container), marker (embedded).
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const farAway = makeGSplatsMesh(2, 'volumetric');
    farAway.geometry.boundingSphere!.center.set(0, 0, -200);
    farAway.geometry.boundingSphere!.radius = 5;
    const cloud = makeGSplatsMesh(2, 'volumetric');
    cloud.geometry.boundingSphere!.center.set(0, 0, -10);
    cloud.geometry.boundingSphere!.radius = 100;
    const marker = makeGSplatsMesh(2, 'volumetric');
    marker.geometry.boundingSphere!.center.set(0, 0, -30);
    marker.geometry.boundingSphere!.radius = 1;

    for (const m of [marker, cloud, farAway]) {
      coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(farAway.renderOrder).toBe(1);
    expect(cloud.renderOrder).toBe(2);
    expect(marker.renderOrder).toBe(3);
  });

  it('partially overlapping spheres do not form a containment edge', async () => {
    // The spheres overlap (distance 8 < radii sum 16) but neither contains
    // the other (8 + 6 > 10). Plain depth order must therefore survive.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const outer = makeGSplatsMesh(2, 'volumetric');
    outer.geometry.boundingSphere!.center.set(0, 0, -10);
    outer.geometry.boundingSphere!.radius = 10;
    const overlap = makeGSplatsMesh(2, 'volumetric');
    overlap.geometry.boundingSphere!.center.set(0, 0, -18);
    overlap.geometry.boundingSphere!.radius = 6;

    coord.noteDepthSortCommit(outer, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(overlap, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(overlap.renderOrder).toBe(1);
    expect(outer.renderOrder).toBe(2);
  });

  it('applies containment only inside the documented grazing epsilon band', async () => {
    // Outer radius 10 gives 0.01 slack. `justOutside` misses the boundary
    // by 0.001; `justInside` lands 0.001 inside it.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const outer = makeGSplatsMesh(2, 'volumetric');
    outer.geometry.boundingSphere!.center.set(0, 0, -10);
    outer.geometry.boundingSphere!.radius = 10;
    const justInside = makeGSplatsMesh(2, 'volumetric');
    justInside.geometry.boundingSphere!.center.set(0, 0, -19.009);
    justInside.geometry.boundingSphere!.radius = 1;
    const justOutside = makeGSplatsMesh(2, 'volumetric');
    justOutside.geometry.boundingSphere!.center.set(0, 0, -19.011);
    justOutside.geometry.boundingSphere!.radius = 1;

    for (const mesh of [outer, justInside, justOutside]) {
      coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    // The outside sphere remains the farthest ready group. The container
    // then draws before only the epsilon-contained sphere.
    expect(justOutside.renderOrder).toBe(1);
    expect(outer.renderOrder).toBe(2);
    expect(justInside.renderOrder).toBe(3);
  });

  it('scales containment radii under a non-identity mesh transform', async () => {
    // Local radius 10 becomes world/view radius 20. That transformed radius
    // contains the marker at z=-25; the unscaled radius would not.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const transformedOuter = makeGSplatsMesh(2, 'volumetric');
    transformedOuter.position.set(0, 0, -10);
    transformedOuter.scale.setScalar(2);
    transformedOuter.geometry.boundingSphere!.center.set(0, 0, 0);
    transformedOuter.geometry.boundingSphere!.radius = 10;
    const marker = makeGSplatsMesh(2, 'volumetric');
    marker.geometry.boundingSphere!.center.set(0, 0, -25);
    marker.geometry.boundingSphere!.radius = 1;

    coord.noteDepthSortCommit(marker, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(transformedOuter, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(transformedOuter.renderOrder).toBe(1);
    expect(marker.renderOrder).toBe(2);
  });

  it('excludes a zero-radius inner group from containment edges', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const outer = makeGSplatsMesh(2, 'volumetric');
    outer.geometry.boundingSphere!.center.set(0, 0, -10);
    outer.geometry.boundingSphere!.radius = 100;
    const zeroRadius = makeGSplatsMesh(2, 'volumetric');
    zeroRadius.geometry.boundingSphere!.center.set(0, 0, -30);
    zeroRadius.geometry.boundingSphere!.radius = 0;

    coord.noteDepthSortCommit(outer, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(zeroRadius, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(zeroRadius.renderOrder).toBe(1);
    expect(outer.renderOrder).toBe(2);
  });

  it('treats a non-finite transformed radius as unusable', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const outer = makeGSplatsMesh(2, 'volumetric');
    outer.geometry.boundingSphere!.center.set(0, 0, -10);
    outer.geometry.boundingSphere!.radius = 100;
    const marker = makeGSplatsMesh(2, 'volumetric');
    marker.geometry.boundingSphere!.center.set(0, 0, -30);
    marker.geometry.boundingSphere!.radius = 1;

    coord.noteDepthSortCommit(outer, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(marker, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    const scaleSpy = vi
      .spyOn(THREE.Matrix4.prototype, 'getMaxScaleOnAxis')
      .mockReturnValueOnce(Infinity)
      .mockReturnValueOnce(1);
    coord.evaluateDepthSortPerFrame();
    scaleSpy.mockRestore();

    // Infinity must become the radius=-1 sentinel instead of a container
    // that captures every finite group.
    expect(marker.renderOrder).toBe(1);
    expect(outer.renderOrder).toBe(2);
  });

  it('keeps the farthest-ready tie-break across independent containers', async () => {
    // Both containers start with indegree zero. Each contains only its own
    // child, so Kahn's ready-set choice must preserve farthest-first order.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const farContainer = makeGSplatsMesh(2, 'volumetric');
    farContainer.geometry.boundingSphere!.center.set(0, 0, -50);
    farContainer.geometry.boundingSphere!.radius = 10;
    const farChild = makeGSplatsMesh(2, 'volumetric');
    farChild.geometry.boundingSphere!.center.set(0, 0, -58);
    farChild.geometry.boundingSphere!.radius = 1;
    const nearContainer = makeGSplatsMesh(2, 'volumetric');
    nearContainer.geometry.boundingSphere!.center.set(0, 0, -10);
    nearContainer.geometry.boundingSphere!.radius = 10;
    const nearChild = makeGSplatsMesh(2, 'volumetric');
    nearChild.geometry.boundingSphere!.center.set(0, 0, -18);
    nearChild.geometry.boundingSphere!.radius = 1;

    for (const mesh of [nearChild, nearContainer, farChild, farContainer]) {
      coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    }
    await flush();

    coord.evaluateDepthSortPerFrame();

    expect(farContainer.renderOrder).toBe(1);
    expect(farChild.renderOrder).toBe(2);
    expect(nearContainer.renderOrder).toBe(3);
    expect(nearChild.renderOrder).toBe(4);
  });

  it('identical bounding spheres produce no containment edge (no self-lock, stable depth order)', async () => {
    // Exactly co-located equal-bounds layers (two channels of one dataset)
    // have no meaningful cross order; the strict-radius guard must not
    // fabricate edges between them — they keep plain depth/insertion order.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const chanA = makeGSplatsMesh(2, 'volumetric');
    chanA.geometry.boundingSphere!.center.set(0, 0, -20);
    chanA.geometry.boundingSphere!.radius = 10;
    const chanB = makeGSplatsMesh(2, 'volumetric');
    chanB.geometry.boundingSphere!.center.set(0, 0, -20);
    chanB.geometry.boundingSphere!.radius = 10;

    coord.noteDepthSortCommit(chanA, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(chanB, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // Tie → stable sort keeps insertion order; both got exactly one rank.
    expect(chanA.renderOrder).toBe(1);
    expect(chanB.renderOrder).toBe(2);
  });

  it('two co-located volumetric LOD siblings (near-identical bounds, mid cross-fade) each get a rank', async () => {
    // The LOD coverage cross-fade shows two sibling levels of one lod_group
    // simultaneously (scene/lod-group-registry.ts). Same content ⇒ nearly
    // identical spheres, differing slightly in radius (different splat
    // subsets). Both must be sorted and ranked; with the coarser slightly
    // larger, the containment rule fires and draws it first. Which order wins
    // is acceptable because combined transmittance is order-INDEPENDENT
    // (transmittances multiply), so occlusion behind the pair is exact at every
    // fade weight; emission's ordering residual is the product of both alphas
    // and the levels' local radiance difference—always first order in that
    // difference, and second order in the alphas only in the thin regime (spec
    // §6). What matters here is deterministic order—no frame-to-frame flicker
    // mid-fade.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const coarse = makeGSplatsMesh(2, 'volumetric');
    coarse.geometry.boundingSphere!.center.set(0, 0, -20);
    coarse.geometry.boundingSphere!.radius = 10.5;
    const fine = makeGSplatsMesh(2, 'volumetric');
    fine.geometry.boundingSphere!.center.set(0, 0, -20);
    fine.geometry.boundingSphere!.radius = 10;

    coord.noteDepthSortCommit(fine, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(coarse, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // Containment edge (coarse ⊃ fine): container draws first.
    expect(coarse.renderOrder).toBe(1);
    expect(fine.renderOrder).toBe(2);
  });

  it('a partition wrapper whose AGGREGATE bounds contain a leaf draws all its parts first', async () => {
    // Multi-mesh groups aggregate an enclosing sphere from their members
    // (the tighter of a Ritter union and a centroid+max-reach bound). Two wide
    // parts flank the origin; a marker leaf sits inside it. The wrapper's MEAN view-z (-10)
    // is nearer than the marker (-30), so pure depth would draw the
    // marker first and the wrapper's parts would erase it — the aggregate
    // containment edge must hoist the whole wrapper, while the BSP ranks
    // still order the parts within it.
    const bspTree = { axis: 0, split: 0, left: { part: 0 }, right: { part: 1 } };
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const parts = [0, 1].map(() => makeGSplatsMesh(2, 'volumetric'));
    parts[0].geometry.boundingSphere!.center.set(-50, 0, -10);
    parts[0].geometry.boundingSphere!.radius = 100;
    parts[1].geometry.boundingSphere!.center.set(50, 0, -10);
    parts[1].geometry.boundingSphere!.radius = 100;
    makePartitionWrapper(bspTree, parts);
    const marker = makeGSplatsMesh(2, 'volumetric');
    marker.geometry.boundingSphere!.center.set(0, 0, -30);
    marker.geometry.boundingSphere!.radius = 1;

    coord.noteDepthSortCommit(marker, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    for (const m of parts) coord.noteDepthSortCommit(m, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // Camera at the origin sits on the split's high side → left subtree
    // (part 0) is the far side and draws first; marker draws last.
    expect(parts[0].renderOrder).toBe(1);
    expect(parts[1].renderOrder).toBe(2);
    expect(marker.renderOrder).toBe(3);
  });

  it('a loose-bound false positive: spread-out tiles do NOT hoist a disjoint neighbour', async () => {
    // The biodiversity-demo inversion (#1227): a small sorted-mode (volumetric)
    // backdrop — standing in for the demo's globe, since an actually-opaque mesh
    // never enters the sorted set — sits OUTSIDE a partition group whose tiles
    // are spread far along one axis.
    // The old loose bound (centroid of member centers + max reach) centered
    // the group sphere on the tile CENTROID and inflated its radius to reach
    // the lone far tile, wrongly swallowing the disjoint globe. The TIGHT
    // union hugs the real tile footprint, so no containment edge forms and
    // plain farthest-first depth order survives.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    // Four clustered tiles near x=0 plus one far outlier at x=100 (all at
    // z=-50). Centroid lands at x≈22.4, and the loose radius (reach to the
    // x=100 tile, ≈78.6) sweeps a huge sphere back across the -x half-space
    // that holds no actual content. The tight union is centered near x=50
    // with radius ≈51 — it never reaches the globe.
    const tileX = [0, 2, 4, 6, 100];
    const tiles = tileX.map((x) => {
      const t = makeGSplatsMesh(2, 'volumetric');
      t.geometry.boundingSphere!.center.set(x, 0, -50);
      t.geometry.boundingSphere!.radius = 1;
      return t;
    });
    makePartitionWrapper(undefined, tiles);

    // Disjoint backdrop on the -x side, FARTHER than the tiles (z=-60 vs -50)
    // so farthest-first draws it first — unless a false containment edge
    // hoists the tiles ahead of it.
    const globe = makeGSplatsMesh(2, 'volumetric');
    globe.geometry.boundingSphere!.center.set(-30, 0, -60);
    globe.geometry.boundingSphere!.radius = 2;

    coord.noteDepthSortCommit(globe, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    for (const t of tiles) coord.noteDepthSortCommit(t, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // Tight bound: no edge → depth order. The farther globe draws first; all
    // tiles follow. Under the old loose bound the globe would be hoisted to
    // LAST (rank 5), inverting the backdrop behind its neighbour.
    expect(globe.renderOrder).toBe(1);
    expect(Math.min(...tiles.map((t) => t.renderOrder))).toBeGreaterThan(1);
  });

  it('a genuine containment inside one union member still hoists the partition first', async () => {
    // Guard the other side of the tight bound: a leaf truly embedded inside
    // one of a partition's member spheres must STILL be detected. Here a big
    // member sphere fully encloses the globe, so the union contains it and the
    // whole wrapper is hoisted ahead — even though the globe is FARTHER
    // (z=-80) than the tiles' mean depth, so only containment (not depth) can
    // produce this order.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const big = makeGSplatsMesh(2, 'volumetric');
    big.geometry.boundingSphere!.center.set(0, 0, -10);
    big.geometry.boundingSphere!.radius = 100;
    const small = makeGSplatsMesh(2, 'volumetric');
    small.geometry.boundingSphere!.center.set(0, 0, -12);
    small.geometry.boundingSphere!.radius = 5;
    const tiles = [big, small];
    makePartitionWrapper(undefined, tiles);

    const globe = makeGSplatsMesh(2, 'volumetric');
    globe.geometry.boundingSphere!.center.set(0, 0, -80); // deep inside `big`
    globe.geometry.boundingSphere!.radius = 1;

    coord.noteDepthSortCommit(globe, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    for (const t of tiles) coord.noteDepthSortCommit(t, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // Containment edge holds: both tiles draw before the embedded globe.
    expect(Math.max(...tiles.map((t) => t.renderOrder))).toBeLessThan(globe.renderOrder);
  });

  it('a symmetric (equilateral) tile layout does not hoist a neighbour Ritter alone would swallow', async () => {
    // The min-of-two bound guard: on a near-symmetric footprint Ritter's union
    // OVERSHOOTS the exact circumsphere (here ~11.9 vs 10.1) and off-centers it,
    // which can re-introduce the very #1227 false-positive edge. The exact
    // centroid+max-reach bound wins on this layout, so a neighbour that sits
    // OUTSIDE the circumsphere but INSIDE the raw Ritter overshoot must NOT be
    // hoisted. Fails if the bound is the raw Ritter sphere.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    // Three tiles on an equilateral triangle, circumradius 10, coplanar at
    // z=-50. Centroid == circumcenter (origin), so the exact bound is r≈10.1;
    // Ritter seeds on a side (r≈8.76) then expands to ~11.93 off-center.
    const tileXY: Array<[number, number]> = [
      [10, 0],
      [-5, 8.66],
      [-5, -8.66],
    ];
    const tiles = tileXY.map(([x, y]) => {
      const t = makeGSplatsMesh(2, 'volumetric');
      t.geometry.boundingSphere!.center.set(x, y, -50);
      t.geometry.boundingSphere!.radius = 0.1;
      return t;
    });
    makePartitionWrapper(undefined, tiles);

    // Neighbour ~11 out along Ritter's bulge direction: outside the r≈10.1
    // circumsphere (dist ≈ 11.0) but well inside the r≈11.9 Ritter overshoot.
    // Slightly farther in z so farthest-first draws it first absent a false edge.
    const neighbour = makeGSplatsMesh(2, 'volumetric');
    neighbour.geometry.boundingSphere!.center.set(5.5, 9.53, -50.5);
    neighbour.geometry.boundingSphere!.radius = 0.2;

    coord.noteDepthSortCommit(neighbour, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    for (const t of tiles) coord.noteDepthSortCommit(t, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // No false edge → depth order: the farther neighbour draws first. Under a
    // raw Ritter bound it would be swallowed and hoisted to last.
    expect(neighbour.renderOrder).toBe(1);
    expect(Math.min(...tiles.map((t) => t.renderOrder))).toBeGreaterThan(1);
  });

  it('an asymmetric layout does not hoist an off-center neighbour only the Ritter sphere covers', async () => {
    // A smaller Ritter radius does NOT make its containment relation a subset
    // of the legacy one: the Ritter center is OFFSET, so its bulge can cover a
    // disjoint neighbour the centroid sphere never reached. Three tiles at
    // (0,0), (10,0), (0,8) give Ritter center (5,4) r≈6.5 vs centroid
    // (3.33,2.67) r≈7.28 — Ritter is smaller and gets picked as the tight
    // bound, yet a neighbour at (11,4) sits INSIDE it (dist 6.0) while the
    // centroid sphere excludes it (dist ≈7.78 > 7.28). The edge must require
    // BOTH bounds to agree, so this neighbour keeps plain depth order. Fails
    // if containment is tested against the min-radius sphere alone.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const tileXY: Array<[number, number]> = [
      [0, 0],
      [10, 0],
      [0, 8],
    ];
    const tiles = tileXY.map(([x, y]) => {
      const t = makeGSplatsMesh(2, 'volumetric');
      t.geometry.boundingSphere!.center.set(x, y, -50);
      t.geometry.boundingSphere!.radius = 0.1;
      return t;
    });
    makePartitionWrapper(undefined, tiles);

    // Slightly farther in z so farthest-first draws it first absent a false edge.
    const neighbour = makeGSplatsMesh(2, 'volumetric');
    neighbour.geometry.boundingSphere!.center.set(11, 4, -50.5);
    neighbour.geometry.boundingSphere!.radius = 0.2;

    coord.noteDepthSortCommit(neighbour, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    for (const t of tiles) coord.noteDepthSortCommit(t, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();

    coord.evaluateDepthSortPerFrame();

    // No new false edge relative to the legacy bound → depth order survives.
    expect(neighbour.renderOrder).toBe(1);
    expect(Math.min(...tiles.map((t) => t.renderOrder))).toBeGreaterThan(1);
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
    expect(farMesh.renderOrder).toBe(1);
    expect(boundless.renderOrder).toBe(2);
    expect(nanCenter.renderOrder).toBe(3);
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

    // Far wrapper (mean z −100) ranks 1..2, near wrapper (mean z −20)
    // ranks 3..4; each wrapper internally keeps its BSP traversal order
    // (eye x=0 is not < split 0 → left leaf drawn first).
    expect(farParts.map((m) => m.renderOrder)).toEqual([1, 2]);
    expect(nearParts.map((m) => m.renderOrder)).toEqual([3, 4]);
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
    expect(behindLeaf.renderOrder).toBe(1);
    expect(parts.map((m) => m.renderOrder)).toEqual([2, 3]);
    expect(frontLeaf.renderOrder).toBe(4);
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

    // B (mean view-z −50) is globally farther than A (mean 0) → ranks 1..2.
    expect(partsB.map((m) => m.renderOrder)).toEqual([1, 2]);
    // A keeps the exact interior FKN order (eye x=25): part0, part1, part3, part2.
    expect(partsA.map((m) => m.renderOrder)).toEqual([3, 4, 6, 5]);
  });

  it('clears renderOrder to 0 when a mesh is no longer order-dependent (additive)', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    // Two leaves so the NEARER one carries a later live rank; switching it to
    // additive must still clear that rank back to the reserved default 0.
    const far = makeGSplatsMesh(2, 'normal');
    far.geometry.boundingSphere!.center.set(0, 0, -30);
    const near = makeGSplatsMesh(2, 'normal');
    near.geometry.boundingSphere!.center.set(0, 0, -15);
    coord.noteDepthSortCommit(far, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    coord.noteDepthSortCommit(near, new Float32Array([0, 0, -1, 1, 0, -2]), 2);
    await flush();
    coord.evaluateDepthSortPerFrame();
    expect(near.renderOrder).toBe(2); // biased while normal

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
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 0]); // untouched
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
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 0]); // untouched
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
    const elementIdMap = new Uint32Array([2048, 2049]);
    mesh.userData.elementIdMap = elementIdMap;
    coord.noteDepthSortBlendingModeSwitch(mesh, 'normal', 'additive');
    expect(mesh.userData.committedData).toBeUndefined();
    expect(requestReprocess).toHaveBeenCalledTimes(1);
    // The picking slot → on-disk map SURVIVES: the geometry stays on the GPU
    // and pickable until the async reprocess commits, so the map still
    // describes it. Dropping it would make hover in that window fall back to
    // the raw storage slot — a silently wrong label (#1421/#1423).
    expect(mesh.userData.elementIdMap).toBe(elementIdMap);
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
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 0]);
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
      // Worker timing split as the real sortNode reports it — the
      // profiler round-trip test below pins the derived metadata.
      kernelMs: 1.5,
      workerMs: 2,
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

  it("pauses a HIDDEN node's chunked apply and resumes it when shown (#715)", async () => {
    // Issue #715: a hidden/off-screen node must not advance (and thus
    // accumulate) its chunked ordering stream — writing a slice while not
    // drawn leaves an unflushed range that the next slice would union
    // onto, collapsing the ladder into one giant upload. The coordinator
    // therefore skips the pump for a node that is not effectively visible.
    const coord = await loadCoordinator();
    const camera = makeCamera();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => camera, requestRender });

    // Same element-storage instance the coordinator drives (loadCoordinator
    // reset the module). A 1-index slice makes the 3-element apply span
    // three frames instead of finishing in one.
    const { setSortedIndexChunkElementsForTests, configureSortedIndexChunkedApply } =
      await import('../../../rendering/element-storage');
    configureSortedIndexChunkedApply(true);
    setSortedIndexChunkElementsForTests(1);

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]), 3);
    await flush();
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await flush();

    // Staged but un-pumped: the inactive buffer (aSortedIndexB, slot 0 is
    // active) is still all zeros.
    const back = (mesh.geometry as THREE.InstancedBufferGeometry).getAttribute(
      'aSortedIndexB'
    ) as THREE.InstancedBufferAttribute;
    expect(Array.from(back.array as Uint32Array)).toEqual([0, 0, 0]);

    // Hidden: several frames advance NOTHING — no slice written, no
    // pending range, no render requested for the paused stream.
    mesh.visible = false;
    requestRender.mockClear();
    for (let i = 0; i < 3; i++) coord.evaluateDepthSortPerFrame();
    expect(Array.from(back.array as Uint32Array)).toEqual([0, 0, 0]);
    expect(back.updateRanges.length).toBe(0);
    expect(requestRender).not.toHaveBeenCalled();

    // Shown: the very next frame writes exactly the first slice
    // (ordering[0] = 2 → inactive slot 0) and requests a render to resume.
    mesh.visible = true;
    coord.evaluateDepthSortPerFrame();
    expect((back.array as Uint32Array)[0]).toBe(2);
    expect(back.updateRanges[0]).toMatchObject({ start: 0, count: 1 });
    expect(requestRender).toHaveBeenCalled();

    setSortedIndexChunkElementsForTests(null);
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

  it('records each sort round-trip as a Depth Sort profiler pass, closed when the ordering is applied', async () => {
    const coord = await loadCoordinator();
    // Fresh module instance to match the coordinator's post-reset module graph.
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    const mesh = await sortedSetup(coord, makeCamera(), { getProfiler: () => profiler });

    const root = profiler.getDepthSortTimings();
    // The pass spans dispatch→applied: the worker resolve in sortedSetup only
    // STAGES the ordering, so the pass is still open — no sample recorded yet.
    // The pump selects the buffer; only the following onAfterRender closes it.
    expect(root.count).toBe(0);
    coord.evaluateDepthSortPerFrame();
    expect(root.count).toBe(0);
    simulateMeshRender(mesh);

    expect(root.count).toBe(1);
    expect(root.metadata?.elements).toBe(2);
    // Uploaded (`up`) now, not merely scheduled — the buffer reached the GPU.
    expect(root.metadata?.info).toMatch(/up$/);
    // The apply duration is recorded separately from the resolve round-trip.
    expect(Number.isFinite(root.metadata?.applyMs)).toBe(true);
    expect(root.metadata?.applyMs).toBeGreaterThanOrEqual(0);
    // Timing split: kernelMs passes through; boundaryMs = workerMs −
    // kernelMs; queueMs = locally-timed round-trip − workerMs. The mock
    // resolves through microtask flushes so the measured round-trip vs
    // the mocked 2 ms workerMs is timing-dependent — only queueMs's
    // non-negativity (it is clamped at 0) and finiteness are pinned.
    expect(root.metadata?.kernelMs).toBe(1.5);
    expect(root.metadata?.boundaryMs).toBeCloseTo(0.5);
    expect(Number.isFinite(root.metadata?.queueMs)).toBe(true);
    expect(root.metadata?.queueMs).toBeGreaterThanOrEqual(0);
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
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([1, 0]);
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
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 2, 1]);
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

    expect(farPoints.renderOrder).toBe(1);
    expect(midPoints.renderOrder).toBe(2);
    expect(nearGSplats.renderOrder).toBe(3);
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
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 2, 1]);
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

describe('depth-sort coordinator — chunked ordering apply (perf lever L8)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  /**
   * The coordinator and the test must share ONE element-storage module
   * instance — import it from the same registry AFTER loadCoordinator's
   * vi.resetModules(). Shrinks the chunk to 4 indices; resetModules in
   * the next test discards the override with the instance.
   */
  async function loadWithTinyChunks() {
    const coord = await loadCoordinator();
    const storage = await import('../../../rendering/element-storage');
    storage.setSortedIndexChunkElementsForTests(4);
    return { coord, storage };
  }

  /** Reversed permutation over [0, n). */
  function reversed(n: number): Uint32Array {
    const out = new Uint32Array(n);
    for (let i = 0; i < n; i++) out[i] = n - 1 - i;
    return out;
  }

  /** Commit + resolve one sort with `ordering` for a fresh normal-mode mesh. */
  async function resolveLargeOrdering(
    coord: Awaited<ReturnType<typeof loadCoordinator>>,
    count: number
  ) {
    const mesh = makeGSplatsMesh(count, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array(count * 3), count);
    await flush();
    // Use the LAST dispatched sort — tests may resolve several in sequence.
    const calls = mockApi.sort.mock.calls;
    const generation = calls[calls.length - 1][0].generation as number;
    const ordering = reversed(count);
    sortResolvers[sortResolvers.length - 1]({ generation, ordering });
    await flush();
    return { mesh, ordering };
  }

  /**
   * Advance ONE drawn frame: run the pre-render pump, then simulate both the
   * classic WebGLRenderer's attribute upload acknowledgement (#715) and the
   * mesh's completed-draw acknowledgement (#713). Without the upload between
   * pumps, the stream correctly stalls; without `onAfterRender`, a selected
   * ordering remains scheduled rather than applied.
   */
  function drawFrame(coord: Awaited<ReturnType<typeof loadCoordinator>>, mesh: THREE.Mesh): void {
    coord.evaluateDepthSortPerFrame();
    simulateMeshRender(mesh);
  }

  it('resolve records the pending apply; the per-frame pump streams the slices and keeps frames flowing', async () => {
    const { coord, storage } = await loadWithTinyChunks();
    const camera = makeCamera();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => camera, requestRender });

    const { mesh, ordering } = await resolveLargeOrdering(coord, 12);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;

    // Resolve only staged the apply and requested the bootstrap frame.
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(true);
    expect(Array.from(drawnOrdering(mesh).subarray(0, 12))).toEqual(new Array(12).fill(0));
    expect(requestRender).toHaveBeenCalled();

    // Frames 1-2: slices upload into the INACTIVE buffer but the slot has
    // not flipped, so rendering still uses the previous whole ordering. Each
    // requests a follow-up frame — from the pump (progress, not stalled) AND from the
    // drawn slice's onUpload acknowledgement; the two coalesce into one
    // real frame in production (requestRender is idempotent per frame), so
    // here we only assert a follow-up WAS requested.
    for (let frame = 1; frame <= 2; frame++) {
      requestRender.mockClear();
      drawFrame(coord, mesh);
      expect(Array.from(drawnOrdering(mesh).subarray(0, 12))).toEqual(new Array(12).fill(0));
      expect(requestRender).toHaveBeenCalled();
    }

    // Frame 3: the final slice completes the back buffer and flips it in.
    // The write and the flip ride THIS frame's flush.
    requestRender.mockClear();
    drawFrame(coord, mesh);
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    expect(Array.from(drawnOrdering(mesh).subarray(0, 12))).toEqual(Array.from(ordering));
    expect(requestRender).toHaveBeenCalledTimes(1); // the flip repaints

    // Steady state: further frames neither write nor request.
    requestRender.mockClear();
    drawFrame(coord, mesh);
    expect(requestRender).not.toHaveBeenCalled();
  });

  it('a STALLED pump (visible but un-drawn) advances nothing and requests NO render (#715)', async () => {
    // The coordinator's `if (!stalled) requestRender?.()` contract: a
    // VISIBLE mesh whose previous slice was not yet drawn must not spin
    // the on-demand loop. Reverting the guard to an unconditional
    // requestRender makes this test fail (the second frame would request).
    const { coord, storage } = await loadWithTinyChunks();
    const camera = makeCamera();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => camera, requestRender });

    const { mesh, ordering } = await resolveLargeOrdering(coord, 12); // 3 slices
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    const back = geometry.getAttribute('aSortedIndexB') as THREE.InstancedBufferAttribute;

    // Frame 1 draws slice 1: progress → exactly one requestRender.
    requestRender.mockClear();
    coord.evaluateDepthSortPerFrame();
    expect((back.array as Uint32Array)[0]).toBe(ordering[0]); // slice 1 written
    expect(requestRender).toHaveBeenCalledTimes(1);

    // Frame 2 WITHOUT acknowledging the upload (mesh stays visible but is
    // not drawn): the pump STALLS — nothing new written, and NO render is
    // requested for the idle stream.
    requestRender.mockClear();
    const writtenBefore = Array.from(back.array as Uint32Array);
    coord.evaluateDepthSortPerFrame();
    expect(Array.from(back.array as Uint32Array)).toEqual(writtenBefore); // no slice 2
    expect(requestRender).not.toHaveBeenCalled();
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(true);
  });

  it('a cleared committedData stamp (LOD demotion) aborts the remaining slices', async () => {
    const { coord, storage } = await loadWithTinyChunks();
    const camera = makeCamera();
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn() });

    const { mesh } = await resolveLargeOrdering(coord, 12);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(true);

    // Demotion: geometry returned to the pool (may already serve another
    // node) — the remaining slices describe the demoted population.
    delete mesh.userData.committedData;
    const before = Array.from(drawnOrdering(mesh));
    coord.evaluateDepthSortPerFrame();
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    expect(Array.from(drawnOrdering(mesh))).toEqual(before); // no further writes
  });

  it('demotion CLEARS a queued re-sort instead of dispatching it (pump branch)', async () => {
    // A queued re-sort surviving demotion would start a worker sort over
    // the released registration's centers, only for the resolve path to
    // discard the result — wasted multi-million-element worker time that
    // delays live nodes. Re-promotion re-commits, which schedules the
    // sort the node actually needs.
    const { coord, storage } = await loadWithTinyChunks();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const { mesh } = await resolveLargeOrdering(coord, 12); // sort #1 resolved, apply streaming
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(true);

    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12); // sort #2 in flight
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12); // queues a re-sort
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);

    // Demotion, then a pump frame: the apply is aborted AND the queued
    // request is dropped — not re-dispatched through scheduleSort.
    delete mesh.userData.committedData;
    coord.evaluateDepthSortPerFrame();
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);

    // Sort #2 resolves onto the demoted node: stale generation, cleared
    // stamp, no queued request left — nothing new may be dispatched.
    const gen2 = mockApi.sort.mock.calls[1][0].generation as number;
    sortResolvers[1]({ generation: gen2, ordering: reversed(12) });
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('a queued re-sort resolving onto a demoted node is dropped, and re-promotion re-sorts', async () => {
    // Same waste, resolve-path flavor: the drain after a resolve must not
    // dispatch for a mesh whose committedData stamp is gone.
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(4, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array(12), 4); // sort #1 in flight
    await flush();
    coord.noteDepthSortCommit(mesh, new Float32Array(12), 4); // queues a re-sort
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // Demoted before the sort resolves.
    delete mesh.userData.committedData;
    const gen1 = mockApi.sort.mock.calls[0][0].generation as number;
    sortResolvers[0]({ generation: gen1, ordering: new Uint32Array([3, 2, 1, 0]) });
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1); // queued request dropped

    // Re-promotion re-commits (the commit path re-stamps committedData
    // before calling the coordinator) — THAT schedules the needed sort.
    mesh.userData.committedData = { some: 'source' };
    coord.noteDepthSortCommit(mesh, new Float32Array(12), 4);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('a stale-generation resolve never starts a chunked apply; the queued re-sort does', async () => {
    const { coord, storage } = await loadWithTinyChunks();
    const camera = makeCamera();
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(12, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    const staleGeneration = mockApi.sort.mock.calls[0][0].generation as number;
    // A newer commit lands mid-sort (bumps the generation, queues a re-sort).
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();

    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    sortResolvers[0]({ generation: staleGeneration, ordering: reversed(12) });
    await flush();
    // Stale result dropped BEFORE the writer — no partial apply started.
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);

    // The queued re-sort dispatched on resolve; its CURRENT-generation
    // result streams normally.
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
    const freshGeneration = mockApi.sort.mock.calls[1][0].generation as number;
    const ordering = reversed(12);
    sortResolvers[1]({ generation: freshGeneration, ordering });
    await flush();
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(true);
    drawFrame(coord, mesh);
    drawFrame(coord, mesh);
    drawFrame(coord, mesh);
    expect(Array.from(drawnOrdering(mesh).subarray(0, 12))).toEqual(Array.from(ordering));
  });

  it('the flip reaches BOTH the visual and the pick material', async () => {
    // The pick pass shares the geometry and emits vElementId from the
    // same index, so a pick material left pointing at the other slot
    // would resolve hovers against a stale permutation.
    const { coord } = await loadWithTinyChunks();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(12, 'normal');
    const visualUniform = { value: 0 };
    const pickUniform = { value: 0 };
    (mesh.material as THREE.Material & { uniforms: unknown }).uniforms = {
      uSortedIndexSlot: visualUniform,
    };
    const pickMaterial = new THREE.Material() as THREE.Material & { uniforms: unknown };
    pickMaterial.uniforms = { uSortedIndexSlot: pickUniform };
    mesh.userData.pickNode = new THREE.Mesh(mesh.geometry, pickMaterial);

    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    const generation = mockApi.sort.mock.calls[0][0].generation as number;
    sortResolvers[0]({ generation, ordering: reversed(12) });
    await flush();

    expect(visualUniform.value).toBe(0);
    drawFrame(coord, mesh);
    drawFrame(coord, mesh);
    expect(visualUniform.value).toBe(0); // still streaming
    drawFrame(coord, mesh); // completes + flips
    expect(visualUniform.value).toBe(1);
    expect(pickUniform.value).toBe(1);
  });

  it('reaches every material of a multi-material mesh, visual and pick alike', async () => {
    // `applySortedIndexSlotToMaterial` accepts `Material | Material[]`
    // because THREE meshes may carry an array. Nothing in the suite
    // exercised the array branch, so silently dropping it (`Array.isArray
    // ? [] : [material]`) passed every other test — while in a
    // multi-material mesh the extra materials would keep reading the
    // stale buffer.
    const { coord } = await loadWithTinyChunks();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(12, 'normal');
    const visA = { value: 0 };
    const visB = { value: 0 };
    const matA = new THREE.Material() as THREE.Material & { uniforms: unknown };
    const matB = new THREE.Material() as THREE.Material & { uniforms: unknown };
    matA.uniforms = { uSortedIndexSlot: visA };
    matB.uniforms = { uSortedIndexSlot: visB };
    matA.userData.blendingMode = 'normal';
    mesh.material = [matA, matB];

    const pickA = { value: 0 };
    const pickB = { value: 0 };
    const pmA = new THREE.Material() as THREE.Material & { uniforms: unknown };
    const pmB = new THREE.Material() as THREE.Material & { uniforms: unknown };
    pmA.uniforms = { uSortedIndexSlot: pickA };
    pmB.uniforms = { uSortedIndexSlot: pickB };
    mesh.userData.pickNode = new THREE.Mesh(mesh.geometry, [pmA, pmB]);

    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: reversed(12),
    });
    await flush();
    for (let i = 0; i < 4; i++) drawFrame(coord, mesh);

    expect(visA.value).toBe(1);
    expect(visB.value).toBe(1);
    expect(pickA.value).toBe(1);
    expect(pickB.value).toBe(1);
  });

  it('re-asserts the slot every frame, not only on the frame it flips', async () => {
    // The slot lives on the geometry, the selector uniform on the material,
    // and they get re-paired behind the coordinator's back: the pool hands a
    // geometry to another node, a TSL graph rebuild replaces the uniform
    // leaves, a pick material is created after the flip. So the sync must be
    // unconditional per frame — syncing only on `flipped` would leave a
    // desynced pair rendering the WRONG buffer indefinitely, because a
    // settled node never flips again.
    const { coord, storage } = await loadWithTinyChunks();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(12, 'normal');
    const uniform = { value: 0 };
    (mesh.material as THREE.Material & { uniforms: unknown }).uniforms = {
      uSortedIndexSlot: uniform,
    };
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: reversed(12),
    });
    await flush();
    for (let i = 0; i < 4; i++) drawFrame(coord, mesh);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    expect(uniform.value).toBe(1);

    // Something re-pairs the pair behind our back (fresh/rebuilt material).
    uniform.value = 0;
    coord.evaluateDepthSortPerFrame();
    expect(uniform.value).toBe(1);
  });

  it('the commit itself pushes the slot to the materials — no frame needed', async () => {
    // A commit's writers can re-home the geometry on slot 0 (full identity
    // write) or hand the mesh a different geometry (pool acquire), and a
    // render that bypasses the frame loop — the settle-scheduled pick
    // pass — can fire before the pump's per-frame re-assert runs. So the
    // sync must ride the commit itself, not only the pump.
    const { coord, storage } = await loadWithTinyChunks();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeGSplatsMesh(12, 'normal');
    const uniform = { value: 0 };
    (mesh.material as THREE.Material & { uniforms: unknown }).uniforms = {
      uSortedIndexSlot: uniform,
    };
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: reversed(12),
    });
    await flush();
    for (let i = 0; i < 4; i++) drawFrame(coord, mesh);
    expect(uniform.value).toBe(1);

    // A count-changing recommit: the writers run BEFORE noteDepthSortCommit
    // and re-home the geometry on slot 0. The commit call must push that
    // slot to the material immediately, without a pump frame in between.
    storage.writeSortedIndexIdentity(mesh.geometry as THREE.InstancedBufferGeometry, 12);
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    expect(uniform.value).toBe(0);
  });

  it('camera motion past the threshold dispatches WHILE a stream runs (no apply-gate)', async () => {
    // The inverse of the old apply-gate test. A stream writes into the
    // inactive buffer, so what renders stays a whole permutation the
    // entire time — there is no reason to make a fresher sort wait for
    // it. Sorting and applying run concurrently.
    const { coord, storage } = await loadWithTinyChunks();
    const camera = makeCamera();
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn() });

    const { mesh } = await resolveLargeOrdering(coord, 12); // 3 slices pending
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    camera.rotateY((5 * Math.PI) / 180); // past the 3° threshold
    coord.evaluateDepthSortPerFrame(); // slice [0, 4) AND a dispatch
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(true);
    expect(mockApi.sort).toHaveBeenCalledTimes(2);
  });

  it('a sort landing mid-stream is held, then swaps in right after the current one', async () => {
    const { coord, storage } = await loadWithTinyChunks();
    const camera = makeCamera();
    coord.configureDepthSort({ getCamera: () => camera, requestRender: vi.fn() });

    const { mesh, ordering: first } = await resolveLargeOrdering(coord, 12);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;

    // A commit lands mid-stream (test harness commits don't write
    // identity, so the stream keeps running — the production identity
    // write would cancel it first). Its sort dispatches IMMEDIATELY now.
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(2);
    expect(mockApi.sort).toHaveBeenCalledTimes(2);

    // Its result is held rather than restarting the stream: the first
    // ordering finishes and swaps in, then the newer one streams.
    const secondGeneration = mockApi.sort.mock.calls[1][0].generation as number;
    const second = new Uint32Array([5, 4, 7, 6, 1, 0, 3, 2, 9, 8, 11, 10]);
    sortResolvers[1]({ generation: secondGeneration, ordering: second });
    await flush();

    drawFrame(coord, mesh);
    drawFrame(coord, mesh);
    drawFrame(coord, mesh); // first completes + flips
    expect(Array.from(drawnOrdering(mesh).subarray(0, 12))).toEqual(Array.from(first));

    drawFrame(coord, mesh);
    drawFrame(coord, mesh);
    drawFrame(coord, mesh); // second completes + flips
    expect(Array.from(drawnOrdering(mesh).subarray(0, 12))).toEqual(Array.from(second));
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
  });

  it("releaseDepthSortNode cancels the node's in-flight apply (no orphaned pump entry)", async () => {
    const { coord, storage } = await loadWithTinyChunks();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const { mesh } = await resolveLargeOrdering(coord, 12);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(true);
    coord.releaseDepthSortNode(mesh);
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
  });

  it('releaseAllDepthSortNodes and disposeDepthSort sweep every in-flight apply', async () => {
    const { coord, storage } = await loadWithTinyChunks();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const { mesh } = await resolveLargeOrdering(coord, 12);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(true);
    coord.releaseAllDepthSortNodes();
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);

    // disposeDepthSort covers the app-teardown path the same way.
    const second = await resolveLargeOrdering(coord, 12);
    const secondGeometry = second.mesh.geometry as THREE.InstancedBufferGeometry;
    expect(storage.hasPendingSortedIndexOrderingApply(secondGeometry)).toBe(true);
    coord.disposeDepthSort();
    expect(storage.hasPendingSortedIndexOrderingApply(secondGeometry)).toBe(false);
  });

  // === Profiler lifecycle across the chunked apply (issue #713) ===

  it('keeps the Depth Sort pass open through the flip, closing it only after the rendered upload', async () => {
    const { coord } = await loadWithTinyChunks();
    // Fresh module instance to match the coordinator's post-reset module graph.
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    const camera = makeCamera();
    coord.configureDepthSort({
      getCamera: () => camera,
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    // 12 indices over a 4-index chunk = three slices → three pump frames.
    const { mesh } = await resolveLargeOrdering(coord, 12);
    const root = profiler.getDepthSortTimings();

    // Resolve alone must NOT close the pass — the bytes are only staged, not
    // yet uploaded. This is the whole point of the fix.
    expect(root.count).toBe(0);

    // Slices 1-2 stream into the buffer NOT being drawn; rendering the old
    // slot cannot acknowledge the new ordering, so the pass stays open.
    evaluateAndRender(coord, mesh);
    evaluateAndRender(coord, mesh);
    expect(root.count).toBe(0);

    // Slice 3 completes the back buffer and flips it in, but flip-time alone
    // still is not proof of upload/draw.
    coord.evaluateDepthSortPerFrame();
    expect(root.count).toBe(0);
    simulateMeshRender(mesh);
    expect(root.count).toBe(1);
    expect(root.metadata?.elements).toBe(12);
    expect(root.metadata?.info).toMatch(/up$/); // uploaded, not scheduled
    expect(Number.isFinite(root.metadata?.applyMs)).toBe(true);
    expect(root.metadata?.applyMs).toBeGreaterThanOrEqual(0);

    // Steady state: another render does not close the pass a second time.
    evaluateAndRender(coord, mesh);
    expect(root.count).toBe(1);
  });

  it('a selected ordering with no completed draw stays scheduled and is abandoned on teardown', async () => {
    const { coord } = await loadWithTinyChunks();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const { mesh } = await resolveLargeOrdering(coord, 12);
    const root = profiler.getDepthSortTimings();
    // Draw the first two frames so their slices upload and release #715
    // back-pressure. Pump the final slice WITHOUT rendering that frame: the
    // complete buffer is selected, but THREE has neither uploaded its final
    // range nor completed a draw with it.
    evaluateAndRender(coord, mesh);
    evaluateAndRender(coord, mesh);
    coord.evaluateDepthSortPerFrame();
    expect(root.count).toBe(0);
    expect(profiler.getDepthSortCompletions().total).toBe(0);

    coord.releaseDepthSortNode(mesh);
    expect(root.count).toBe(1);
    expect(root.metadata?.info).toMatch(/sched$/);
    expect(root.metadata?.applyMs).toBeUndefined();
    expect(profiler.getDepthSortCompletions().total).toBe(0);

    // A late render callback cannot resurrect the abandoned completion.
    simulateMeshRender(mesh);
    expect(root.count).toBe(1);
  });

  it('a chunked apply aborted by LOD demotion closes the pass exactly once, as scheduled (never uploaded)', async () => {
    const { coord, storage } = await loadWithTinyChunks();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    const camera = makeCamera();
    coord.configureDepthSort({
      getCamera: () => camera,
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const { mesh } = await resolveLargeOrdering(coord, 12);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    const root = profiler.getDepthSortTimings();
    expect(root.count).toBe(0); // pass open, streaming
    expect(profiler.getDepthSortCompletions().total).toBe(0);

    // Stream one slice, then demote: the remaining slices abort.
    coord.evaluateDepthSortPerFrame();
    expect(root.count).toBe(0);
    delete mesh.userData.committedData;
    coord.evaluateDepthSortPerFrame(); // demotion cancels the apply
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);

    // The pass closed once, labeled scheduled — the bytes never reached the
    // GPU, so they must not be reported as uploaded.
    expect(root.count).toBe(1);
    expect(root.metadata?.info).toMatch(/sched$/);
    // It was staged but never became drawable, so the per-applied-ordering
    // completion stream must stay empty.
    expect(profiler.getDepthSortCompletions().total).toBe(0);

    // Exactly once: further frames must not re-close it.
    coord.evaluateDepthSortPerFrame();
    expect(root.count).toBe(1);
  });

  it('a superseded held ordering closes its pass exactly once as scheduled; the drawn ones close as uploaded', async () => {
    const { coord } = await loadWithTinyChunks();
    const camera = makeCamera();

    // A fake profiler records every pass so we can assert exactly-once
    // per-session (the real profiler aggregates passes into one root and does
    // not expose each session's close count). The generation/completion methods
    // satisfy the coordinator's #711 integration; this test only inspects pass
    // lifecycles.
    const passes: Array<{ meta: Record<string, unknown>; ended: number }> = [];
    const fakeProfiler = {
      _currentGeneration: () => 0,
      recordDepthSortCompletion: () => {},
      beginDepthSortPass(): UpdateSession {
        const pass = { meta: {} as Record<string, unknown>, ended: 0 };
        passes.push(pass);
        const session: UpdateSession = {
          begin: () => session,
          end: () => {
            pass.ended++;
          },
          setMetadata: (m) => {
            Object.assign(pass.meta, m);
          },
          markSkipped: () => {},
        };
        return session;
      },
    } as unknown as UpdateProfiler;
    coord.configureDepthSort({
      getCamera: () => camera,
      requestRender: vi.fn(),
      getProfiler: () => fakeProfiler,
    });

    // Sort 1 resolves → apply 1 starts streaming (pass 0 = current).
    const { mesh } = await resolveLargeOrdering(coord, 12);

    // Sort 2 resolves mid-stream → held behind apply 1 (pass 1 = pending).
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    sortResolvers[1]({
      generation: mockApi.sort.mock.calls[1][0].generation as number,
      ordering: reversed(12),
    });
    await flush();

    // Sort 3 resolves mid-stream → supersedes the held pending (pass 2).
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    sortResolvers[2]({
      generation: mockApi.sort.mock.calls[2][0].generation as number,
      ordering: reversed(12),
    });
    await flush();

    expect(passes).toHaveLength(3);
    // Pass 1 — the superseded held ordering — closed immediately, exactly
    // once, labeled scheduled: it never streamed a single slice.
    expect(passes[1].ended).toBe(1);
    expect(String(passes[1].meta.info)).toMatch(/sched$/);
    // Passes 0 (streaming) and 2 (now held) are still open.
    expect(passes[0].ended).toBe(0);
    expect(passes[2].ended).toBe(0);

    // Drain with a real render acknowledgement after every pre-render pump:
    // pass 0 uploads/draws first, then the promoted pass 2 does likewise.
    // Eight frames comfortably covers two 3-slice streams.
    for (let i = 0; i < 8; i++) evaluateAndRender(coord, mesh);
    expect(String(passes[0].meta.info)).toMatch(/up$/);
    expect(String(passes[2].meta.info)).toMatch(/up$/);
    // Every pass closed exactly once — no leak, no double-close.
    expect(passes.map((p) => p.ended)).toEqual([1, 1, 1]);
  });

  it('records the applied sample of a pass that completes after a superseded sibling (no seq-drop)', async () => {
    // Regression guard for the real profiler under sort churn: sessions end
    // at APPLY, out of dispatch order — a held ordering superseded mid-stream
    // ends BEFORE the currently-streaming pass flips. With a dispatch-time
    // seq the streaming pass's applied sample would be dropped as "stale" by
    // the profiler's merge guard, so the dispatch→applied timing this fix
    // exists to record would vanish. The completion-ordered seq keeps it.
    const { coord } = await loadWithTinyChunks();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    const camera = makeCamera();
    coord.configureDepthSort({
      getCamera: () => camera,
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    // Sort 1 streaming; sort 2 held; sort 3 supersedes the held sort 2.
    const { mesh } = await resolveLargeOrdering(coord, 12);
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    sortResolvers[1]({
      generation: mockApi.sort.mock.calls[1][0].generation as number,
      ordering: reversed(12),
    });
    await flush();
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    sortResolvers[2]({
      generation: mockApi.sort.mock.calls[2][0].generation as number,
      ordering: reversed(12),
    });
    await flush();

    // Drain: sort 1 applies (up), then the promoted sort 3 applies (up).
    for (let i = 0; i < 8; i++) evaluateAndRender(coord, mesh);

    const root = profiler.getDepthSortTimings();
    // Three passes recorded: sort 2 (abandoned → scheduled) plus the two
    // applied ones — the sort-1 applied sample is NOT dropped despite its
    // lower dispatch order.
    expect(root.count).toBe(3);
    // The last-completing pass (sort 3, applied) shows uploaded bytes + an
    // apply duration — the freshest state, correctly labeled.
    expect(root.metadata?.info).toMatch(/up$/);
    expect(Number.isFinite(root.metadata?.applyMs)).toBe(true);
  });

  it('a post-handoff requestRender throw does not close the pass early (applied sample stays uploaded)', async () => {
    // Once the ordering is handed to the chunked-apply callbacks, a throw
    // from the post-handoff requestRender() must NOT route into .catch and
    // close the pass — the callbacks own its close. Otherwise the pass would
    // record a premature scheduled sample and the real applied one would be
    // swallowed by end()'s idempotency (issue #713).
    const { coord } = await loadWithTinyChunks();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    const camera = makeCamera();
    let throwOnce = true;
    const requestRender = vi.fn(() => {
      if (throwOnce) {
        throwOnce = false;
        throw new Error('boom');
      }
    });
    coord.configureDepthSort({
      getCamera: () => camera,
      requestRender,
      getProfiler: () => profiler,
    });

    // Resolve stages the ordering, hands off the session, then the bootstrap
    // requestRender() throws → the .then rejects into .catch.
    const { mesh } = await resolveLargeOrdering(coord, 12);
    const root = profiler.getDepthSortTimings();
    // The pass was already handed off, so .catch left it open — no premature
    // scheduled sample.
    expect(root.count).toBe(0);

    // The chunked apply still completes and closes the pass as uploaded,
    // but only once the selected buffer is actually rendered.
    for (let i = 0; i < 4; i++) evaluateAndRender(coord, mesh);
    expect(root.count).toBe(1);
    expect(root.metadata?.info).toMatch(/up$/);
  });

  it('an RPC failure closes the pass in .catch exactly once (no leak, no double-close)', async () => {
    // The .catch's `if (!applyOwnsSession)` guard must still close the pass
    // when the RPC genuinely fails (nothing was ever staged) — the failure
    // side of the latch, complementing the post-handoff-throw test above.
    const { coord } = await loadWithTinyChunks();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeGSplatsMesh(12, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);

    // The sort RPC rejects: no ordering staged, so the pass closes here.
    sortRejectors[0](new Error('worker crashed'));
    await flush();

    const root = profiler.getDepthSortTimings();
    expect(root.count).toBe(1);
    // No later frame re-closes it.
    coord.evaluateDepthSortPerFrame();
    expect(root.count).toBe(1);
  });

  it('releaseAllDepthSortNodes (dataset switch) closes an open streaming pass, as scheduled', async () => {
    const { coord, storage } = await loadWithTinyChunks();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const { mesh } = await resolveLargeOrdering(coord, 12);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    const root = profiler.getDepthSortTimings();
    expect(root.count).toBe(0); // pass open, still streaming

    coord.releaseAllDepthSortNodes(); // dataset-switch teardown sweep
    expect(storage.hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    // The sweep's onAbandoned closed the open pass — recorded once, scheduled
    // (the bytes never reached the GPU).
    expect(root.count).toBe(1);
    expect(root.metadata?.info).toMatch(/sched$/);
  });

  it('a teardown sweep closes BOTH the streaming and the held-pending pass', async () => {
    const { coord } = await loadWithTinyChunks();
    const { UpdateProfiler } = await import('../../../profiling/update-profiler');
    const profiler = new UpdateProfiler();
    const camera = makeCamera();
    coord.configureDepthSort({
      getCamera: () => camera,
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    // Pass A streaming; pass B resolves mid-stream → held as pending.
    const { mesh } = await resolveLargeOrdering(coord, 12);
    coord.noteDepthSortCommit(mesh, new Float32Array(36), 12);
    await flush();
    sortResolvers[1]({
      generation: mockApi.sort.mock.calls[1][0].generation as number,
      ordering: reversed(12),
    });
    await flush();
    const root = profiler.getDepthSortTimings();
    expect(root.count).toBe(0); // both open

    // The sweep must abandon BOTH the current and the held ordering.
    coord.releaseAllDepthSortNodes();
    expect(root.count).toBe(2);
    expect(root.metadata?.info).toMatch(/sched$/);
  });
});

/**
 * The INDEXED (mesh) apply, driven through the real coordinator.
 *
 * `triangle-ordering.test.ts` pins the write in isolation. What can only be
 * seen here is the coordinator's ownership of it: that a resolved ordering
 * reaches `geometry.index` at all, and that a COMMIT supersedes a pending one.
 * The second is not the same case as sort-supersedes-sort — the commit rewrites
 * the index buffer itself (`updateMeshGeometry` runs before
 * `noteDepthSortCommit`), so an ordering left in the acknowledgement map would
 * be reported as uploaded on the next draw even though nothing of it survives.
 */
function makeIndexedMesh(faceCount: number, blendingMode: string): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();
  const triples = new Uint32Array(faceCount * 3);
  for (let i = 0; i < triples.length; i++) triples[i] = i;
  geometry.setIndex(new THREE.BufferAttribute(triples.slice(), 1, false));
  geometry.setDrawRange(0, faceCount * 3);
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 10);
  const material = new THREE.Material();
  material.userData.blendingMode = blendingMode;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.nodeType = 'mesh';
  mesh.userData.committedData = { some: 'source' };
  return mesh;
}

/** The canonical triples a mesh commit hands over as the permutation source. */
function sourceTriples(faceCount: number): Uint32Array {
  const src = new Uint32Array(faceCount * 3);
  for (let i = 0; i < src.length; i++) src[i] = i;
  return src;
}

/**
 * A REAL profiler from the coordinator's post-reset module graph — the same
 * idiom the instanced lifecycle tests use. A hand-written double would not pin
 * the applied/scheduled label, which is the whole observable here.
 */
async function freshProfiler(): Promise<UpdateProfiler> {
  const { UpdateProfiler } = await import('../../../profiling/update-profiler');
  return new UpdateProfiler();
}

function drawnTriples(mesh: THREE.Mesh, faceCount: number): string[] {
  const a = mesh.geometry.index!.array;
  const out: string[] = [];
  for (let f = 0; f < faceCount; f++) out.push(`${a[f * 3]},${a[f * 3 + 1]},${a[f * 3 + 2]}`);
  return out;
}

describe('depth-sort coordinator — the indexed (mesh) apply', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('permutes geometry.index when the ordering resolves', async () => {
    const coord = await loadCoordinator();
    const requestRender = vi.fn();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender });

    const mesh = makeIndexedMesh(3, 'normal');
    const centers = new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]);
    coord.noteDepthSortCommit(mesh, centers, 3, sourceTriples(3));
    await flush();

    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    const generation = mockApi.sort.mock.calls[0][0].generation as number;
    sortResolvers[0]({ generation, ordering: new Uint32Array([2, 0, 1]) });
    await flush();

    // Faces are (0,1,2) (3,4,5) (6,7,8); the ordering moves whole triples.
    expect(drawnTriples(mesh, 3)).toEqual(['6,7,8', '0,1,2', '3,4,5']);
    expect(requestRender).toHaveBeenCalled();
  });

  it('drops an ordering whose generation was superseded', async () => {
    const coord = await loadCoordinator();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });

    const mesh = makeIndexedMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array(9), 3, sourceTriples(3));
    await flush();
    const stale = mockApi.sort.mock.calls[0][0].generation as number;
    // A newer commit lands before the first sort resolves.
    coord.noteDepthSortCommit(mesh, new Float32Array(9), 3, sourceTriples(3));
    await flush();

    sortResolvers[0]({ generation: stale, ordering: new Uint32Array([2, 0, 1]) });
    await flush();
    // Untouched: the stale ordering describes the previous commit's faces.
    expect(drawnTriples(mesh, 3)).toEqual(['0,1,2', '3,4,5', '6,7,8']);
  });

  it('a COMMIT supersedes an ordering written but not yet drawn', async () => {
    // The reviewer-found gap. `updateMeshGeometry` has already overwritten the
    // index buffer by the time the commit reaches the coordinator, so the
    // pending acknowledgement must be abandoned — otherwise the next render
    // closes its profiler pass as UPLOADED for a permutation nothing kept.
    const profiler = await freshProfiler();
    const coord = await loadCoordinator();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeIndexedMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array(9), 3, sourceTriples(3));
    await flush();
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await flush();
    const root = profiler.getDepthSortTimings();
    expect(root.count).toBe(0); // written, awaiting its first draw

    // The next commit rewrites the index and registers afresh.
    coord.noteDepthSortCommit(mesh, new Float32Array(9), 3, sourceTriples(3));
    await flush();
    expect(root.count).toBe(1);
    expect(root.metadata?.info).toMatch(/sched$/); // abandoned, NOT uploaded

    // And a later draw must not resurrect it as a second, applied sample.
    simulateMeshRender(mesh);
    expect(root.count).toBe(1);
  });

  it('a commit that turns commutative also abandons the pending ordering', async () => {
    // The release branch's peer of the case above — same requirement, reached
    // through the other arm of the commit.
    const profiler = await freshProfiler();
    const coord = await loadCoordinator();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeIndexedMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array(9), 3, sourceTriples(3));
    await flush();
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await flush();

    (mesh.material as THREE.Material).userData.blendingMode = 'opaque';
    coord.noteDepthSortCommit(mesh, new Float32Array(9), 3, sourceTriples(3));
    await flush();
    expect(profiler.getDepthSortTimings().count).toBe(1);
    expect(profiler.getDepthSortTimings().metadata?.info).toMatch(/sched$/);
  });

  it('reports the ordering as applied once the mesh is drawn', async () => {
    // The positive control for the two abandonment cases: without it they
    // would pass against a coordinator that never reports applied at all.
    const profiler = await freshProfiler();
    const coord = await loadCoordinator();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeIndexedMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array(9), 3, sourceTriples(3));
    await flush();
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await flush();

    simulateMeshRender(mesh);
    const root = profiler.getDepthSortTimings();
    expect(root.count).toBe(1);
    expect(root.metadata?.info).toMatch(/up$/);
  });

  it('releasing the node abandons a pending indexed ordering', async () => {
    const profiler = await freshProfiler();
    const coord = await loadCoordinator();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      getProfiler: () => profiler,
    });

    const mesh = makeIndexedMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, new Float32Array(9), 3, sourceTriples(3));
    await flush();
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await flush();

    coord.releaseDepthSortNode(mesh);
    expect(profiler.getDepthSortTimings().count).toBe(1);
    expect(profiler.getDepthSortTimings().metadata?.info).toMatch(/sched$/);
  });
});

/**
 * Starved-vs-dead SortWorker init (issue #1694).
 *
 * The init deadline exists to guarantee the cached `initPromise` SETTLES, so
 * per-commit continuations closing over their centers providers cannot pile up
 * unbounded. But the settle is not a verdict on the worker's health: at ~3M
 * points the main thread stays saturated through a load long enough for worker
 * startup to lose that 30 s race, and the old unconditional stays-failed
 * degrade then drew the identity (storage) order for the rest of the session
 * over a condition that would have cleared in seconds.
 *
 * These tests pin the split: a deadline miss is retried (bounded, backed off,
 * scheduler-driven, and only when something actually wants sorting), while a
 * worker `error` event — a script that really is dead — is not.
 */
describe('depth-sort coordinator — starved init retry (issue #1694)', () => {
  /** Mirrors the DEFAULT of `config.depthSort.workerInitTimeoutMs`. */
  const INIT_TIMEOUT_MS = 30_000;
  /** Mirrors the module-private SORT_WORKER_INIT_RETRY_BASE_MS (× attempt). */
  const RETRY_BASE_MS = 2_000;
  /** Mirrors the module-private SORT_WORKER_INIT_MAX_ATTEMPTS (initial + 2 retries). */
  const SORT_WORKER_INIT_MAX_ATTEMPTS_MIRROR = 3;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  // Every test here installs fake timers (see useDeadlineTimers) — restoring
  // them once, centrally, instead of in five per-test `finally` blocks. Safe
  // for the tests that never install any: `useRealTimers` is a no-op then.
  afterEach(() => {
    vi.useRealTimers();
  });

  /**
   * Fake timers covering BOTH clocks the retry path reads: the 30 s deadline
   * is a `setTimeout`, while the backoff is measured on the monotonic
   * `performance.now()`. Faking only one would either never fire the deadline
   * or make every retry wait out a real 2 s. Must be installed before the
   * coordinator arms either, i.e. before the first commit.
   */
  function useDeadlineTimers(): void {
    vi.useFakeTimers({
      toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'performance', 'Date'],
    });
  }

  /** The live mocked worker instance (for firing worker-level events). */
  function liveWorker(): { onerror?: ((e: unknown) => void) | null } {
    return (
      globalThis as unknown as {
        __lastMockWorker: { onerror?: ((e: unknown) => void) | null };
      }
    ).__lastMockWorker;
  }

  /** Centers for a 3-splat node (distinct view-z values). */
  function centers3(): Float32Array {
    return new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]);
  }

  it('recovers depth sorting after init loses the deadline race to a starved main thread', async () => {
    const coord = await loadCoordinator();
    // The load-starvation shape: `initialize` is simply too slow to settle
    // inside the deadline (nothing is broken).
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    mesh.userData.loadedViewVersion = 7; // committed fresh for view 7
    coord.noteDepthSortCommit(mesh, centers3(), 3);
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    // Init in flight — not yet a verdict either way.
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'idle', initTimeouts: 0 });

    // The deadline fires: the wedged worker is terminated, the commit's
    // continuation drains into the degrade path, nothing sorts.
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(terminatedWorkers.length).toBe(1);
    expect(mockApi.registerNode).not.toHaveBeenCalled();
    expect(mockApi.sort).not.toHaveBeenCalled();
    // …but the state says STARVED, not failed: a retry is armed.
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    // Inside the backoff no attempt may be spent (a retry issued into the
    // same stall would just miss the deadline again).
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    // The stall clears (init would now succeed) and the backoff elapses.
    mockApi.initialize.mockImplementation(async () => ({ wasmFallback: true }));
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 1);
    coord.evaluateDepthSortPerFrame();
    await flush();

    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'ready', initTimeouts: 1 });
    // The fresh worker holds no centers (they were never transferred), so
    // the node is marked for re-commit and ONE reprocess is requested.
    expect(requestReprocess).toHaveBeenCalledTimes(1);
    expect(mesh.userData.committedData).toBeUndefined();
    expect(mesh.userData.loadedViewVersion).toBeUndefined();

    // Simulate that reprocess landing: the re-commit registers + sorts, and
    // a real ordering finally reaches the drawn buffer.
    mesh.userData.committedData = { some: 'source' };
    coord.noteDepthSortCommit(mesh, centers3(), 3);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    sortResolvers[0]({
      generation: mockApi.sort.mock.calls[0][0].generation as number,
      ordering: new Uint32Array([0, 2, 1]),
    });
    await flush();
    expect(Array.from(await applyStagedOrdering(mesh))).toEqual([0, 2, 1]);
  });

  it('does NOT retry a dead worker script (onerror) — that failure is permanent', async () => {
    // Negative control for the test above: `onerror` means the script died
    // during async module evaluation, so waiting/retrying cannot help. This
    // is the case the stays-failed degrade was designed for and it must stay
    // exactly as it was.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, centers3(), 3);
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    // The worker emits its error event BEFORE the deadline (which the
    // handler clears), so the failure is the dead-script kind.
    liveWorker().onerror!({ message: 'module evaluation failed' });
    await flush();
    expect(terminatedWorkers.length).toBe(1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'failed', initTimeouts: 0 });

    // Neither elapsed time nor any number of frames may respawn it.
    for (let frame = 0; frame < 5; frame++) {
      await vi.advanceTimersByTimeAsync(60_000);
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    expect(requestReprocess).not.toHaveBeenCalled();
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'failed', initTimeouts: 0 });
  });

  it('bounds the starved retry at 3 init attempts, then gives up', async () => {
    // The bound is what keeps the anti-leak contract intact: each attempt
    // opens a fresh 30 s window in which commits attach continuations to a
    // pending promise, so the number of such windows must be finite.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {})); // never recovers
    useDeadlineTimers();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, centers3(), 3);
    await flush();

    for (let attempt = 1; attempt <= SORT_WORKER_INIT_MAX_ATTEMPTS_MIRROR; attempt++) {
      await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1); // this attempt's deadline
      expect(coord.getDepthSortWorkerStatus().initTimeouts).toBe(attempt);
      // Backoff grows with the miss count (2 s, then 4 s).
      await vi.advanceTimersByTimeAsync(RETRY_BASE_MS * attempt + 1);
      coord.evaluateDepthSortPerFrame();
      await flush();
    }

    // Three attempts spent (initial + 2 retries) and the third deadline has
    // fired, so the coordinator has given up.
    expect(mockApi.initialize).toHaveBeenCalledTimes(SORT_WORKER_INIT_MAX_ATTEMPTS_MIRROR);
    expect(coord.getDepthSortWorkerStatus()).toEqual({
      state: 'failed',
      initTimeouts: SORT_WORKER_INIT_MAX_ATTEMPTS_MIRROR,
    });

    // Further frames and further elapsed time add nothing.
    for (let frame = 0; frame < 5; frame++) {
      await vi.advanceTimersByTimeAsync(60_000);
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(mockApi.initialize).toHaveBeenCalledTimes(SORT_WORKER_INIT_MAX_ATTEMPTS_MIRROR);
  });

  it('does not spend a retry on a scene where nothing visible wants sorting', async () => {
    // A retry budget of 3 is easy to waste: an idle/hidden scene would burn
    // it before the data that needs sorting is even on screen.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, centers3(), 3);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    // The only tracked node is hidden (a collapsed layer / demoted LOD
    // level): nothing on screen needs an ordering.
    mesh.visible = false;
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 1);
    for (let frame = 0; frame < 5; frame++) {
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    // Positive control: the moment it IS visible, the armed retry fires —
    // proving the visibility gate (not some other block) held it back.
    mesh.visible = true;
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
  });

  it('does not retry while a load sweep is in flight', async () => {
    // The in-flight sweep IS the main-thread saturation that starved init;
    // retrying into it would spend an attempt on a guaranteed miss.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    let loading = true;
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
      isLoadInProgress: () => loading,
    });

    const mesh = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(mesh, centers3(), 3);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 1);
    for (let frame = 0; frame < 5; frame++) {
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    // Positive control: the sweep finishes and the next frame retries.
    loading = false;
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
  });
  it('orphans a stale attempt so its deadline cannot maul the next app (epoch guard)', async () => {
    // The deadline `setTimeout` lives in the init closure and only `onerror` /
    // `initialize` settling clears it — a dispose CANNOT. So app A's timer is
    // still armed after teardown, and without the attempt-epoch guard it
    // classifies a starved miss against app B's healthy worker: retry armed,
    // B's RESOLVED initPromise dropped, a third worker spawned over the live
    // one (leaked thread + its transferred centers), and — since every node
    // still reads `registered` — a new worker holding no registrations at all.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    const configure = (requestRender: () => void): void =>
      coord.configureDepthSort({
        getCamera: () => makeCamera(),
        requestRender,
        requestReprocess: vi.fn(),
      });

    configure(vi.fn());
    coord.noteDepthSortCommit(makeGSplatsMesh(3, 'normal'), centers3(), 3);
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    const staleWorker = liveWorker();

    // App A torn down mid-init; app B re-inits in the same page and comes up.
    coord.disposeDepthSort();
    mockApi.initialize.mockImplementation(async () => ({ wasmFallback: true }));
    configure(vi.fn());
    const meshB = makeGSplatsMesh(3, 'normal');
    coord.noteDepthSortCommit(meshB, centers3(), 3);
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'ready', initTimeouts: 0 });
    const workerB = liveWorker();
    expect(workerB).not.toBe(staleWorker);

    // App A's orphan deadline fires against the LIVE session.
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'ready', initTimeouts: 0 });
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
    expect(terminatedWorkers).not.toContain(workerB);
  });

  it('does not let a stale init SUCCESS stamp a disposed session ready (epoch guard)', async () => {
    // The symmetric direction: app A's slow `initialize` finally answers after
    // the teardown. Unguarded it would stamp 'ready' and clear the retry
    // bookkeeping on behalf of a worker that no longer belongs to anything.
    const coord = await loadCoordinator();
    let resolveInit: (r: { wasmFallback: boolean }) => void = () => {};
    mockApi.initialize.mockImplementation(
      () =>
        new Promise<{ wasmFallback: boolean }>((resolve) => {
          resolveInit = resolve;
        })
    );
    useDeadlineTimers();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
    });
    coord.noteDepthSortCommit(makeGSplatsMesh(3, 'normal'), centers3(), 3);
    await flush();
    const staleWorker = liveWorker();

    coord.disposeDepthSort();
    resolveInit({ wasmFallback: true });
    await flush();

    // THE pin: unguarded, the late success would have stamped 'ready' (and
    // cleared the retry bookkeeping) for a worker that belongs to nothing.
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'idle', initTimeouts: 0 });
    // The orphaned worker is not left running — but this line credits the
    // DISPOSE, not the epoch guard: `disposeDepthSort()` terminated this exact
    // worker before the resolve, so it cannot fail either way. The stale-success
    // path's own `w.terminate()` is defensive: every epoch bump a pending
    // SUCCESS can observe today comes from a dispose, which has already
    // terminated the worker (the other bumper, a retry, only runs after the
    // previous attempt REJECTED and terminated its own worker in that catch).
    expect(terminatedWorkers).toContain(staleWorker);
  });

  it('wakes the idle render loop so an armed retry actually gets a frame', async () => {
    // The per-frame scheduler is a NON-continuous callback, so the on-demand
    // loop stops `config.animation.idleTimeoutMs` (2000 ms — exactly the first
    // backoff) after the last render. Without a self-wake a static
    // "load it and look at it" scene would never run the retry at all.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    const requestRender = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender,
      requestReprocess: vi.fn(),
    });
    coord.noteDepthSortCommit(makeGSplatsMesh(3, 'normal'), centers3(), 3);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    requestRender.mockClear();
    // Not before the backoff expires (a wake there would be refused by the
    // eligibility check and waste the only scheduled frame).
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS - 1);
    expect(requestRender).not.toHaveBeenCalled();
    // …and shortly after it, exactly once.
    await vi.advanceTimersByTimeAsync(200);
    expect(requestRender).toHaveBeenCalledTimes(1);

    // The woken frame is genuinely past the backoff, so it spends the attempt.
    mockApi.initialize.mockImplementation(async () => ({ wasmFallback: true }));
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
  });

  it('cancels the pending wake on dispose so it cannot poke the next app', async () => {
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    const requestRenderA = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: requestRenderA,
      requestReprocess: vi.fn(),
    });
    coord.noteDepthSortCommit(makeGSplatsMesh(3, 'normal'), centers3(), 3);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });
    requestRenderA.mockClear();

    // Teardown + re-init well inside the backoff, then let the wake's instant
    // pass: the retry it was scheduled for no longer exists.
    coord.disposeDepthSort();
    const requestRenderB = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: requestRenderB,
      requestReprocess: vi.fn(),
    });
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 1_000);
    expect(requestRenderA).not.toHaveBeenCalled();
    expect(requestRenderB).not.toHaveBeenCalled();
  });

  it('grows the backoff with the miss count (× attempt, not a constant 2 s)', async () => {
    // An upper bound alone would be satisfied by a constant 2 s backoff; this
    // pins the multiplication by refusing a frame that is past 2 s but inside
    // the second miss's 4 s.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
    });
    coord.noteDepthSortCommit(makeGSplatsMesh(3, 'normal'), centers3(), 3);
    await flush();

    // Miss 1 → 2 s: refused just before, spent just after.
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS - 10);
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(20);
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);

    // Miss 2 → 4 s: a frame at 2 s + ε must still be refused.
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 2 });
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 10);
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS);
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(3);
  });

  it('disposeDepthSort() resets the init verdict and restores the retry budget', async () => {
    // An embedder that disposes and re-inits in one page must not inherit a
    // stale 'starved'/'failed' verdict about a worker that is already gone.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    const startSession = async (): Promise<void> => {
      coord.configureDepthSort({
        getCamera: () => makeCamera(),
        requestRender: vi.fn(),
        requestReprocess: vi.fn(),
      });
      coord.noteDepthSortCommit(makeGSplatsMesh(3, 'normal'), centers3(), 3);
      await flush();
    };

    // From 'starved'.
    await startSession();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });
    coord.disposeDepthSort();
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'idle', initTimeouts: 0 });

    // …and from 'failed' (the budget spent to exhaustion).
    await startSession();
    for (let attempt = 1; attempt <= SORT_WORKER_INIT_MAX_ATTEMPTS_MIRROR; attempt++) {
      await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
      await vi.advanceTimersByTimeAsync(RETRY_BASE_MS * attempt + 1);
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(coord.getDepthSortWorkerStatus()).toEqual({
      state: 'failed',
      initTimeouts: SORT_WORKER_INIT_MAX_ATTEMPTS_MIRROR,
    });
    expect(mockApi.initialize).toHaveBeenCalledTimes(1 + SORT_WORKER_INIT_MAX_ATTEMPTS_MIRROR);
    coord.disposeDepthSort();
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'idle', initTimeouts: 0 });

    // A FULL budget, not a session that starts already given-up: the fresh
    // session's first deadline miss is 'starved', with a retry armed.
    await startSession();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2 + SORT_WORKER_INIT_MAX_ATTEMPTS_MIRROR);
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });
  });

  it('does not spend a retry inside an offline capture, but the next frame does', async () => {
    // The capture drains synchronously against a time bound and cannot await a
    // 30 s init, so its own `evaluateDepthSortPerFrame` calls must not retry.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
    });
    coord.noteDepthSortCommit(makeGSplatsMesh(3, 'normal'), centers3(), 3);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 1);

    await coord.resortForCapture(0);
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);

    // Positive control: an ordinary frame right afterwards spends it.
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
  });

  it('retries a starved init even with no requestReprocess wired', async () => {
    // The reprocess gate belongs on the re-registration SWEEP, not on the
    // attempt: a recovered worker still lets every FUTURE commit register.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'normal');
    mesh.userData.loadedViewVersion = 9;
    coord.noteDepthSortCommit(mesh, centers3(), 3);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    mockApi.initialize.mockImplementation(async () => ({ wasmFallback: true }));
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 1);
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'ready', initTimeouts: 1 });
    // Nothing could act on an invalidation, so the stamps were left alone.
    expect(mesh.userData.committedData).toBeDefined();
    expect(mesh.userData.loadedViewVersion).toBe(9);

    // The next natural commit registers + sorts against the recovered worker.
    coord.noteDepthSortCommit(mesh, centers3(), 3);
    await flush();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
  });

  it('leaves a demoted (stamp-less) node alone when re-registering after a late init', async () => {
    // A node with no `committedData` stamp is LOD-demoted, and re-promotion
    // always re-commits — so invalidating its freshness stamp only forces a
    // reload of something that is not on screen.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });
    const shown = makeGSplatsMesh(3, 'normal');
    shown.userData.loadedViewVersion = 4;
    const demoted = makeGSplatsMesh(3, 'normal');
    demoted.userData.loadedViewVersion = 4;
    coord.noteDepthSortCommit(shown, centers3(), 3);
    coord.noteDepthSortCommit(demoted, centers3(), 3);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    // The LOD registry demotes one of the two while the retry is armed.
    delete demoted.userData.committedData;

    mockApi.initialize.mockImplementation(async () => ({ wasmFallback: true }));
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 1);
    coord.evaluateDepthSortPerFrame();
    await flush();

    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'ready', initTimeouts: 1 });
    expect(shown.userData.loadedViewVersion).toBeUndefined();
    expect(demoted.userData.loadedViewVersion).toBe(4);
    expect(requestReprocess).toHaveBeenCalledTimes(1);
  });

  it('does not spend a retry when every tracked node is COMMUTATIVE', async () => {
    // The order-dependence gate in the "does anything want sorting?" scan, on
    // its own: an all-additive scene never needed an ordering, so spending one
    // of three attempts on it would burn the budget before the sorted data the
    // retry exists for arrives. (The visibility/committed gates are pinned by
    // the hidden-node test above; this one keeps both nodes visible and
    // committed and varies only the blending mode.)
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess: vi.fn(),
    });

    // A `normal` node is what starts init at all (a commutative commit returns
    // before `ensureWorker`), so it seeds the starved state and then switches
    // to additive — the LIVE mode is what the gate reads.
    const sorted = makeGSplatsMesh(3, 'normal');
    const additive = makeGSplatsMesh(3, 'additive');
    coord.noteDepthSortCommit(sorted, centers3(), 3);
    coord.noteDepthSortCommit(additive, centers3(), 3);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    (sorted.material as THREE.Material).userData.blendingMode = 'additive';
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 1);
    for (let frame = 0; frame < 5; frame++) {
      coord.evaluateDepthSortPerFrame();
      await flush();
    }
    expect(mockApi.initialize).toHaveBeenCalledTimes(1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    // Positive control: the moment an order-dependent node is live again the
    // armed retry fires — proving the mode gate (not the backoff, not the
    // budget) held it back.
    (sorted.material as THREE.Material).userData.blendingMode = 'normal';
    coord.evaluateDepthSortPerFrame();
    await flush();
    expect(mockApi.initialize).toHaveBeenCalledTimes(2);
  });

  it('leaves a COMMUTATIVE node untouched when re-registering after a late init', async () => {
    // The order-dependence gate in the re-registration sweep. An additive node
    // never wanted an ordering, so invalidating its stamps would buy nothing
    // and cost a needless O(N) re-projection plus (via the cleared
    // `loadedViewVersion`) an LOD reload of a layer that was never sorted.
    const coord = await loadCoordinator();
    mockApi.initialize.mockImplementation(() => new Promise(() => {}));
    useDeadlineTimers();
    const requestReprocess = vi.fn();
    coord.configureDepthSort({
      getCamera: () => makeCamera(),
      requestRender: vi.fn(),
      requestReprocess,
    });
    const sorted = makeGSplatsMesh(3, 'normal');
    sorted.userData.loadedViewVersion = 11;
    const additive = makeGSplatsMesh(3, 'additive');
    additive.userData.loadedViewVersion = 11;
    coord.noteDepthSortCommit(sorted, centers3(), 3);
    coord.noteDepthSortCommit(additive, centers3(), 3);
    await flush();
    await vi.advanceTimersByTimeAsync(INIT_TIMEOUT_MS + 1);
    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'starved', initTimeouts: 1 });

    mockApi.initialize.mockImplementation(async () => ({ wasmFallback: true }));
    await vi.advanceTimersByTimeAsync(RETRY_BASE_MS + 1);
    coord.evaluateDepthSortPerFrame();
    await flush();

    expect(coord.getDepthSortWorkerStatus()).toEqual({ state: 'ready', initTimeouts: 1 });
    // The sorted node is marked for re-commit…
    expect(sorted.userData.committedData).toBeUndefined();
    expect(sorted.userData.loadedViewVersion).toBeUndefined();
    // …and the additive one is untouched by it.
    expect(additive.userData.committedData).toBeDefined();
    expect(additive.userData.loadedViewVersion).toBe(11);
    expect(requestReprocess).toHaveBeenCalledTimes(1);
  });
});

/**
 * The synchronous first sort (#2290's flash).
 *
 * Every commit of an order-dependent node used to write a fallback ordering and
 * then wait for the SortWorker — a Comlink round trip, so at least one frame
 * rendered unsorted. Invisible on a one-off load; continuous during nD
 * playback, where a commit lands at EVERY timepoint. On the `cloud` demo (4D
 * Points, `volumetric` blending) that reached ~14 unsorted frames a second, and
 * the fallback composited only 61.7% of sampled element pairs in correct
 * back-to-front order against 100% for a real sort.
 */
describe('depth-sort coordinator — synchronous first sort', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  /** Farthest-first is the contract: ordering[0] is the most-negative view z. */
  const CENTERS = new Float32Array([
    0,
    0,
    -1, // element 0: nearest
    0,
    0,
    -9, // element 1: farthest
    0,
    0,
    -5, // element 2: middle
  ]);
  const EXPECTED_BACK_TO_FRONT = [1, 2, 0];

  function activeOrdering(mesh: THREE.Mesh, count: number): number[] {
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    const slot = (geometry.userData.sortedIndexSlot as number | undefined) ?? 0;
    const attr = geometry.getAttribute(slot === 0 ? 'aSortedIndex' : 'aSortedIndexB');
    return Array.from((attr.array as Uint32Array).subarray(0, count));
  }

  /**
   * Seed the ordering buffer with a value no permutation can produce, so
   * "declined" is distinguishable from "wrote something that happens to look
   * like the initial state". These coordinator unit tests call
   * `noteDepthSortCommit` directly, bypassing the commit path — so a decline
   * leaves the buffer UNTOUCHED rather than storage-ordered (writing storage
   * order is the adapter's job, covered in `sorted-index-repair.test.ts`).
   */
  const SENTINEL = [77, 77, 77];
  function seedSentinel(mesh: THREE.Mesh): void {
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    for (const name of ['aSortedIndex', 'aSortedIndexB']) {
      (geometry.getAttribute(name).array as Uint32Array).set(SENTINEL, 0);
    }
  }

  it('orders the buffer BEFORE the worker is asked — no unsorted frame', async () => {
    const coord = await loadCoordinator(250_000);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'normal');
    seedSentinel(mesh);

    coord.noteDepthSortCommit(mesh, CENTERS.slice(), 3);

    // No `await flush()`: this is the state of the buffer in the very frame the
    // commit returns, which is the frame that used to flash.
    expect(mockApi.sort).not.toHaveBeenCalled();
    expect(activeOrdering(mesh, 3)).toEqual(EXPECTED_BACK_TO_FRONT);
  });

  it('writes NOTHING with the synchronous path off (mutation guard)', async () => {
    // The pre-fix behaviour: the commit leaves the ordering to the adapter's
    // storage-order fallback and the worker, so this frame is unsorted. If this
    // ever starts matching EXPECTED_BACK_TO_FRONT, the ceiling stopped gating.
    const coord = await loadCoordinator(0);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'normal');
    seedSentinel(mesh);

    coord.noteDepthSortCommit(mesh, CENTERS.slice(), 3);

    expect(activeOrdering(mesh, 3)).toEqual(SENTINEL);
  });

  it('declines above the configured element ceiling', async () => {
    const coord = await loadCoordinator(2); // ceiling below the node's count
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'normal');
    seedSentinel(mesh);

    coord.noteDepthSortCommit(mesh, CENTERS.slice(), 3);

    expect(activeOrdering(mesh, 3)).toEqual(SENTINEL);
  });

  it('still registers and still dispatches the async sort', async () => {
    // The synchronous sort makes the FIRST frame correct; keeping the ordering
    // current as the camera moves is still the pipeline's job, so its dispatch
    // contract is deliberately unchanged.
    const coord = await loadCoordinator(250_000);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'normal');

    coord.noteDepthSortCommit(mesh, CENTERS.slice(), 3);
    await flush();

    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
  });

  it('resolves a lazy centers provider exactly ONCE across both paths', async () => {
    // The sort needs the centers, and so does the worker registration. Paying
    // the O(N) copy twice per commit would be a real cost on a scrub.
    const coord = await loadCoordinator(250_000);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'normal');
    const provider = vi.fn(() => CENTERS.slice());

    coord.noteDepthSortCommit(mesh, provider, 3);
    await flush();

    expect(provider).toHaveBeenCalledTimes(1);
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
  });

  it('does not resolve an indexed mesh provider for a synchronous sort it cannot apply', async () => {
    const coord = await loadCoordinator(250_000);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeIndexedMesh(3, 'normal');
    const provider = vi.fn(() => CENTERS.slice());

    coord.noteDepthSortCommit(mesh, provider, 3, sourceTriples(3));
    await flush();

    expect(provider).toHaveBeenCalledTimes(1);
    expect(mockApi.registerNode).toHaveBeenCalledTimes(1);
  });

  it('spends at most one configured element budget per frame', async () => {
    const coord = await loadCoordinator(3);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const first = makeGSplatsMesh(3, 'normal');
    const second = makeGSplatsMesh(3, 'normal');
    seedSentinel(first);
    seedSentinel(second);

    coord.evaluateDepthSortPerFrame();
    coord.noteDepthSortCommit(first, CENTERS.slice(), 3);
    coord.noteDepthSortCommit(second, CENTERS.slice(), 3);

    expect(activeOrdering(first, 3)).toEqual(EXPECTED_BACK_TO_FRONT);
    expect(activeOrdering(second, 3)).toEqual(SENTINEL);

    coord.evaluateDepthSortPerFrame();
    coord.noteDepthSortCommit(second, CENTERS.slice(), 3);
    expect(activeOrdering(second, 3)).toEqual(EXPECTED_BACK_TO_FRONT);
  });

  it('restores the synchronous budget when the coordinator is disposed and reconfigured', async () => {
    const coord = await loadCoordinator(3);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const first = makeGSplatsMesh(3, 'normal');
    coord.evaluateDepthSortPerFrame();
    coord.noteDepthSortCommit(first, CENTERS.slice(), 3);
    expect(activeOrdering(first, 3)).toEqual(EXPECTED_BACK_TO_FRONT);

    coord.disposeDepthSort();
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const afterReinit = makeGSplatsMesh(3, 'normal');
    seedSentinel(afterReinit);
    coord.noteDepthSortCommit(afterReinit, CENTERS.slice(), 3);

    expect(activeOrdering(afterReinit, 3)).toEqual(EXPECTED_BACK_TO_FRONT);
  });

  it('declines without throwing when the centers provider throws', async () => {
    // Must land where it always did: the async path's outer catch, which owns
    // the once-per-episode report.
    const coord = await loadCoordinator(250_000);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'normal');
    const provider = vi.fn(() => {
      throw new Error('projection gone');
    });

    seedSentinel(mesh);
    expect(() => coord.noteDepthSortCommit(mesh, provider, 3)).not.toThrow();
    await flush();
    expect(activeOrdering(mesh, 3)).toEqual(SENTINEL);
  });

  it('declines when no camera is configured yet', async () => {
    const coord = await loadCoordinator(250_000);
    coord.configureDepthSort({ getCamera: () => null, requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'normal');

    seedSentinel(mesh);
    expect(() => coord.noteDepthSortCommit(mesh, CENTERS.slice(), 3)).not.toThrow();
    expect(activeOrdering(mesh, 3)).toEqual(SENTINEL);
  });

  it('does not sort an order-INDEPENDENT node', async () => {
    const coord = await loadCoordinator(250_000);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'additive');

    seedSentinel(mesh);
    coord.noteDepthSortCommit(mesh, CENTERS.slice(), 3);

    expect(activeOrdering(mesh, 3)).toEqual(SENTINEL);
    expect(mockApi.registerNode).not.toHaveBeenCalled();
  });

  it('declines a provider that under-delivers rather than sorting garbage', async () => {
    const coord = await loadCoordinator(250_000);
    coord.configureDepthSort({ getCamera: () => makeCamera(), requestRender: vi.fn() });
    const mesh = makeGSplatsMesh(3, 'normal');

    seedSentinel(mesh);
    // Two elements' worth of centers for a three-element count.
    coord.noteDepthSortCommit(mesh, new Float32Array([0, 0, -1, 0, 0, -9]), 3);

    expect(activeOrdering(mesh, 3)).toEqual(SENTINEL);
  });
});
