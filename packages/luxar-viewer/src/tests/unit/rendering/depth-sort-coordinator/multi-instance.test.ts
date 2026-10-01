/**
 * Two hosts on one page — a LuxarApp and a LuxarLayer, or two layers — each
 * with its own `DepthSortCoordinator` over the ONE shared SortWorker.
 *
 * The coordinator used to be module-scoped and configured by both hosts, so the
 * last configuration won: every node was sorted against the last-configured
 * camera and the first host's `requestRender` never fired. These tests pin the
 * per-instance contract: each coordinator sorts its own nodes against its own
 * camera, wakes only its own render loop, and its teardown leaves the other
 * host's registrations — and the shared worker — alone.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';

interface SortCall {
  nodeId: string;
  generation: number;
  modelView: Float32Array;
}

let mockApi: {
  initialize: ReturnType<typeof vi.fn>;
  registerNode: ReturnType<typeof vi.fn>;
  sort: ReturnType<typeof vi.fn>;
  releaseNode: ReturnType<typeof vi.fn>;
  releaseAllNodes: ReturnType<typeof vi.fn>;
};
let sortResolvers: Array<(r: { generation: number; ordering: Uint32Array } | null) => void>;
let terminated: number;

async function loadModule() {
  vi.resetModules();
  sortResolvers = [];
  terminated = 0;
  mockApi = {
    initialize: vi.fn(async () => ({ wasmFallback: true })),
    registerNode: vi.fn(async () => undefined),
    sort: vi.fn(() => new Promise((resolve) => sortResolvers.push(resolve))),
    releaseNode: vi.fn(async () => undefined),
    releaseAllNodes: vi.fn(async () => undefined),
  };
  vi.doMock('../../../../utils/log', () => ({
    log: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
    Modules: new Proxy({}, { get: (_t, p) => String(p) }),
  }));
  vi.doMock('comlink', () => ({ wrap: vi.fn(() => mockApi), transfer: vi.fn((v: unknown) => v) }));
  vi.doMock('../../../../workers/sort-worker?worker', () => ({
    default: class MockSortWorker {
      onerror: ((e: unknown) => void) | null = null;
      onmessageerror: ((e: unknown) => void) | null = null;
      terminate = vi.fn(() => {
        terminated++;
      });
    },
  }));
  const mod = await import('../../../../rendering/depth-sort-coordinator');
  const { config } = await import('../../../../config');
  // The async worker path is what crosses hosts; the synchronous first sort
  // would answer before the worker is asked.
  config.depthSort.syncSortMaxElements = 0;
  return mod;
}

function makeMesh(count: number): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute(
    'aSortedIndex',
    new THREE.InstancedBufferAttribute(new Uint32Array(count), 1)
  );
  geometry.setAttribute(
    'aSortedIndexB',
    new THREE.InstancedBufferAttribute(new Uint32Array(count), 1)
  );
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 10);
  const material = new THREE.Material();
  material.userData.blendingMode = 'normal';
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.nodeType = 'gsplats';
  mesh.userData.committedData = { some: 'source' };
  return mesh;
}

function cameraLookingFrom(x: number, z: number): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(x, 0, z);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  return camera;
}

/** The model-view the sort kernel must receive for `mesh` seen from `camera`. */
function expectedModelView(camera: THREE.Camera, mesh: THREE.Mesh): number[] {
  mesh.updateWorldMatrix(true, false);
  const mv = new THREE.Matrix4().copy(camera.matrixWorld).invert().multiply(mesh.matrixWorld);
  return Array.from(new Float32Array(mv.elements));
}

function centers(): Float32Array {
  return new Float32Array([0, 0, -1, 0, 0, 1, 0, 0, 0]);
}

function sortCallIndex(mesh: THREE.Mesh): number {
  return mockApi.sort.mock.calls.findIndex((c) => (c[0] as SortCall).nodeId === mesh.uuid);
}

function sortCall(mesh: THREE.Mesh): SortCall {
  return mockApi.sort.mock.calls[sortCallIndex(mesh)][0] as SortCall;
}

async function flush(): Promise<void> {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

/** Two configured coordinators with distinct cameras, one committed node each. */
async function twoHosts() {
  const mod = await loadModule();
  const camA = cameraLookingFrom(0, 20);
  const camB = cameraLookingFrom(20, 0);
  const renderA = vi.fn();
  const renderB = vi.fn();
  const a = new mod.DepthSortCoordinator();
  const b = new mod.DepthSortCoordinator();
  const meshA = makeMesh(3);
  const meshB = makeMesh(3);
  // Configure-then-commit for A BEFORE B is configured: the order that made
  // the module-scoped coordinator sort A's node against B's camera.
  a.configure({ getCamera: () => camA, requestRender: renderA });
  a.noteCommit(meshA, centers(), 3);
  b.configure({ getCamera: () => camB, requestRender: renderB });
  b.noteCommit(meshB, centers(), 3);
  await flush();
  return { mod, a, b, camA, camB, renderA, renderB, meshA, meshB };
}

describe('two depth-sort coordinators on one page', () => {
  it('sorts each node against its own host camera and wakes only its own host', async () => {
    const { camA, camB, renderA, renderB, meshA, meshB } = await twoHosts();

    expect(Array.from(sortCall(meshA).modelView)).toEqual(expectedModelView(camA, meshA));
    expect(Array.from(sortCall(meshB).modelView)).toEqual(expectedModelView(camB, meshB));

    renderA.mockClear();
    renderB.mockClear();
    sortResolvers[sortCallIndex(meshA)]({
      generation: sortCall(meshA).generation,
      ordering: new Uint32Array([1, 2, 0]),
    });
    await flush();
    expect(renderA).toHaveBeenCalled();
    expect(renderB).not.toHaveBeenCalled();

    renderA.mockClear();
    sortResolvers[sortCallIndex(meshB)]({
      generation: sortCall(meshB).generation,
      ordering: new Uint32Array([2, 0, 1]),
    });
    await flush();
    expect(renderB).toHaveBeenCalled();
    expect(renderA).not.toHaveBeenCalled();
  });

  it("each host's per-frame pass re-sorts only its own nodes, from its own camera", async () => {
    const { a, b, camA, meshA, meshB } = await twoHosts();
    for (const mesh of [meshA, meshB]) {
      sortResolvers[sortCallIndex(mesh)]({
        generation: sortCall(mesh).generation,
        ordering: new Uint32Array([0, 1, 2]),
      });
    }
    await flush();
    mockApi.sort.mockClear();

    // Swing A's camera a quarter turn: past every re-sort threshold.
    camA.position.set(20, 0, 0);
    camA.lookAt(0, 0, 0);
    camA.updateMatrixWorld();
    a.evaluatePerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
    expect(sortCall(meshA).nodeId).toBe(meshA.uuid);
    expect(Array.from(sortCall(meshA).modelView)).toEqual(expectedModelView(camA, meshA));

    // B's camera did not move, so B's pass dispatches nothing.
    b.evaluatePerFrame();
    expect(mockApi.sort).toHaveBeenCalledTimes(1);
  });

  it("one host's dataset switch and dispose leave the other host's registrations and the worker alone", async () => {
    const { a, b, meshA, meshB } = await twoHosts();
    expect(mockApi.registerNode).toHaveBeenCalledTimes(2);

    a.releaseAllNodes();
    expect(mockApi.releaseAllNodes).not.toHaveBeenCalled();
    expect(mockApi.releaseNode.mock.calls.map((c) => c[0] as string)).toEqual([meshA.uuid]);

    a.dispose();
    expect(terminated).toBe(0);
    // B still sorts on the shared worker.
    b.noteCommit(meshB, centers(), 3);
    await flush();
    expect(mockApi.sort.mock.calls.at(-1)?.[0]).toMatchObject({ nodeId: meshB.uuid });

    // The LAST host's dispose terminates it.
    b.dispose();
    expect(terminated).toBe(1);
  });

  it('routes a mesh release to the coordinator the mesh was committed through', async () => {
    const { mod, meshA, meshB } = await twoHosts();
    mockApi.releaseNode.mockClear();

    mod.releaseDepthSortNode(meshB);
    expect(mockApi.releaseNode.mock.calls.map((c) => c[0] as string)).toEqual([meshB.uuid]);
    // B no longer tracks it: a second release is a no-op …
    mod.releaseDepthSortNode(meshB);
    expect(mockApi.releaseNode).toHaveBeenCalledTimes(1);
    // … and A's node was never touched.
    mod.releaseDepthSortNode(meshA);
    expect(mockApi.releaseNode.mock.calls.map((c) => c[0] as string)).toEqual([
      meshB.uuid,
      meshA.uuid,
    ]);
  });
});
