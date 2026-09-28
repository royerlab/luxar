// @vitest-environment jsdom
/**
 * Scene-graph name lookups on the update path (B9a, tracking #2944).
 *
 * Every per-node step of an update pass — the sweep's eligibility probe, the
 * per-type process and commit, the monitor tallies — resolves its node by PATH.
 * `Object3D.getObjectByName` answers that with a full subtree walk, so a pass
 * over N nodes costs O(N²) node visits: on a 2000-part partition tree that is
 * millions of visits per slice move. The loader keeps a path→Object3D index
 * instead, and this pins the metric: a settled update pass over a populated
 * scene makes (essentially) no `getObjectByName` calls at all.
 *
 * Harness: the same mocked-zarrita `SceneLoader` scaffolding as
 * `mesh-update-sweep.test.ts` — an empty root, mesh placeholders attached by
 * hand under nested groups, driven through the real `updateView`.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import * as zarr from 'zarrita';
import { SceneLoader } from '../../../../data';
import { createEmptyMeshNode } from '../../../../rendering/node-factory/create-mesh-node';
import type { LoadedMeshData, MeshDataLoader, MeshMetadata } from '../../../../types/mesh';

vi.mock('zarrita', () => ({
  FetchStore: vi.fn(),
  withMaybeConsolidatedMetadata: vi.fn(),
  registry: {},
  root: vi.fn(),
  open: (() => {
    const openMock = vi.fn();
    return Object.assign(openMock, { v2: openMock, v3: openMock });
  })(),
  NotFoundError: class NotFoundError extends Error {},
  InvalidMetadataError: class InvalidMetadataError extends Error {},
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

vi.mock('../../../../rendering/material-manager', () => ({
  materialManager: {
    getMeshMaterial: vi.fn(() => ({
      uniforms: {},
      userData: {},
      defines: {},
      side: 0,
      needsUpdate: false,
      updateCameraParams: vi.fn(),
      updateShading: vi.fn(),
      updateColormapTexture: vi.fn(),
      updateScalarRange: vi.fn(),
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
    })),
  },
  SOFT_DISPOSE_FLAG: Symbol.for('luxar.material.softDispose.test-mock'),
}));

vi.mock('../../../../utils/cross-layer/notifier', () => ({
  notifier: {
    toast: vi.fn(),
    error: vi.fn(),
    showHelp: vi.fn(),
    hideHelp: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
    clearError: vi.fn(),
    showSceneIdentityBanner: vi.fn(),
    hideSceneIdentityBanner: vi.fn(),
  },
}));

const ATTRS: MeshMetadata = {
  type: 'mesh',
  n_vertices: 3,
  n_faces: 1,
  ndim: 4,
  has_normals: false,
  has_colors: false,
  has_scalars: false,
  has_uvs: false,
  has_texture: false,
  shading: 'flat',
  double_sided: true,
  ordering: 'none',
};

function loaded(): LoadedMeshData {
  return {
    vertices: new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
    faces: new Uint32Array([0, 1, 2]),
    normals: null,
    colors: null,
    scalars: undefined,
    vertexCount: 3,
    faceCount: 1,
    ndim: 4,
  };
}

/** Attach `groups × perGroup` mesh leaves under nested groups and register them. */
function populate(sceneLoader: SceneLoader, groups: number, perGroup: number): void {
  const root = (sceneLoader as unknown as { rootGroup: THREE.Group }).rootGroup;
  const registry = (
    sceneLoader as unknown as {
      registry: { registerMeshLoader(path: string, loader: MeshDataLoader): void };
    }
  ).registry;
  for (let g = 0; g < groups; g++) {
    const group = new THREE.Group();
    group.name = `/g${g}`;
    root.add(group);
    for (let i = 0; i < perGroup; i++) {
      const path = `/g${g}/m${i}`;
      const data = loaded();
      const loader = {
        updateView: vi.fn().mockResolvedValue(data),
        loadMesh: vi.fn().mockResolvedValue(data),
        dispose: vi.fn(),
      } as unknown as MeshDataLoader;
      group.add(createEmptyMeshNode(path, ATTRS, loader, null));
      registry.registerMeshLoader(path, loader);
    }
  }
}

describe('SceneLoader update pass — name lookups (B9a)', () => {
  let sceneLoader: SceneLoader;

  beforeEach(() => {
    vi.clearAllMocks();
    const mockStore = { contents: vi.fn().mockResolvedValue([{ path: '/', kind: 'group' }]) };
    const mockRootLoc = {
      resolve: vi.fn().mockImplementation(() => ({
        resolve: vi.fn().mockImplementation(() => ({ resolve: vi.fn() })),
      })),
    };
    const mockZarrGroup = {
      attrs: {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'z', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'w', unit: 'um', range: [0, 100], display: false, step: 1 },
          ],
        },
      },
    };
    vi.mocked(zarr.FetchStore).mockImplementation(() => mockStore as unknown as zarr.FetchStore);
    vi.mocked(zarr.withMaybeConsolidatedMetadata).mockResolvedValue(mockStore as never);
    vi.mocked(zarr.root).mockReturnValue(mockRootLoc as never);
    vi.mocked(zarr.open).mockResolvedValue(mockZarrGroup as never);
    sceneLoader = new SceneLoader();
  });

  afterEach(async () => {
    await sceneLoader.dispose();
    vi.restoreAllMocks();
  });

  it('a settled update pass over 40 nodes makes no getObjectByName walks', async () => {
    await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    populate(sceneLoader, 4, 10);
    const view = (w: number) => ({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, w],
      tolerance: [1e10, 1e10, 1e10, 0.5],
    });
    // Warm-up pass: the first commit of every node.
    await sceneLoader.updateView(view(0));

    const spy = vi.spyOn(THREE.Object3D.prototype, 'getObjectByName');
    await sceneLoader.updateView(view(1));
    await sceneLoader.updateView(view(0));
    const calls = spy.mock.calls.length;
    spy.mockRestore();

    // Sanity: the passes really did visit every node (each re-committed).
    const root = (sceneLoader as unknown as { rootGroup: THREE.Group }).rootGroup;
    const leaf = root.getObjectByName('/g3/m9') as THREE.Mesh;
    expect(leaf.geometry.drawRange.count).toBe(3);
    expect(calls).toBe(0);
  });
});
