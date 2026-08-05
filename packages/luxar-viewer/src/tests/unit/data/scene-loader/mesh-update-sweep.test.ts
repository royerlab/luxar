/**
 * Mesh participates in the scene-loader's per-update sweep.
 *
 * A mesh is whole-node resident: `MeshDataLoader.updateView` returns the SAME
 * `LoadedMeshData` on every call, and what actually depends on the view is the
 * nD-slab cull performed downstream in `processMeshData` → `projectMeshTo3D`. So
 * the mesh MUST be re-projected on every `SceneLoader.updateView`, not only at
 * initial load — otherwise the indexed triangle set freezes at the first
 * slice's answer and never tracks a slice move / tolerance change.
 *
 * This test drives one mesh through `updateView` at two slice positions and
 * asserts the committed index buffer re-projects: full triangle at the slice it
 * lives on, empty once the slice moves off it. The empty-index assertion can
 * only pass if the mesh joined the Stage-1 sweep and re-projected.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SceneLoader } from '../../../../data';
import { createEmptyMeshNode } from '../../../../rendering/node-factory/create-mesh-node';
import type { LoadedMeshData, MeshDataLoader, MeshMetadata } from '../../../../types/mesh';
import * as zarr from 'zarrita';

// Mock zarrita (external dependency — network I/O for zarr stores). Same
// scaffolding as scene-loader.test.ts.
vi.mock('zarrita', () => ({
  FetchStore: vi.fn(),
  withMaybeConsolidatedMetadata: vi.fn(),
  registry: {},
  root: vi.fn(),
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

// Material manager depends on WebGL shader compilation — must be mocked in jsdom.
vi.mock('../../../../rendering/material-manager', () => ({
  materialManager: {
    getPointMaterial: vi.fn().mockReturnValue({
      uniforms: {},
      vertexShader: '',
      fragmentShader: '',
      userData: {},
      updateCameraParams: vi.fn(),
    }),
  },
  SOFT_DISPOSE_FLAG: Symbol.for('luxar.material.softDispose.test-mock'),
}));

// SceneLoader uses `notifier.toast` for the >16D scene-dimensions warning.
const notifierMocks = vi.hoisted(() => ({ toast: vi.fn() }));
vi.mock('../../../../utils/cross-layer/notifier', () => ({
  notifier: {
    toast: notifierMocks.toast,
    error: vi.fn(),
    showHelp: vi.fn(),
    hideHelp: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
    clearError: vi.fn(),
  },
}));

// Node metadata: 4D mesh, authored double-sided (no stored normals, so no
// `normal_dims` — honouring the "present iff has_normals" invariant in
// types/mesh.ts). Winding is decidable-as-double-sided, which raises no
// undecidable-winding notice; the slab cull is side-independent, so the
// index-count assertions below are unaffected by the render side.
const ATTRS: MeshMetadata = {
  type: 'mesh',
  n_vertices: 3,
  n_faces: 1,
  ndim: 4,
  has_normals: false,
  has_colors: false,
  has_scalars: false,
  shading: 'flat',
  double_sided: true,
  ordering: 'none',
};

/** One triangle in 4D, all three vertices at hidden dim `w`. */
function loaded(w = 0): LoadedMeshData {
  return {
    vertices: new Float32Array([0, 0, 0, w, 1, 0, 0, w, 0, 1, 0, w]),
    faces: new Uint32Array([0, 1, 2]),
    normals: null,
    colors: null,
    scalars: undefined,
    vertexCount: 3,
    faceCount: 1,
    ndim: 4,
  };
}

describe('SceneLoader.updateView — mesh re-projects on every sweep', () => {
  let sceneLoader: SceneLoader;

  beforeEach(() => {
    vi.clearAllMocks();

    const mockStore = {
      // Bare scene root: no auto-built child nodes, so the mesh node we attach
      // manually is the only geometry in the sweep.
      contents: vi.fn().mockResolvedValue([{ path: '/', kind: 'group' }]),
    };
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

    (zarr.FetchStore as any).mockImplementation(() => mockStore);
    (zarr as any).withMaybeConsolidatedMetadata.mockResolvedValue(mockStore);
    (zarr.root as any).mockReturnValue(mockRootLoc);
    (zarr.open as any).mockResolvedValue(mockZarrGroup);

    sceneLoader = new SceneLoader();
  });

  afterEach(() => {
    sceneLoader.dispose();
  });

  it('culls the triangle when the slice moves off the plane it lives on', async () => {
    await sceneLoader.loadScene('http://localhost:8000/test.zarr');

    // The mesh loader is whole-node resident: it hands back the SAME data on
    // every call. Any view-dependence must come from the downstream projection.
    const meshData = loaded();
    const meshLoader = {
      updateView: vi.fn().mockResolvedValue(meshData),
      loadMesh: vi.fn().mockResolvedValue(meshData),
      dispose: vi.fn(),
    } as unknown as MeshDataLoader;

    const meshNode = createEmptyMeshNode('/mesh', ATTRS, meshLoader);
    (sceneLoader as any).rootGroup.add(meshNode);
    (sceneLoader as any).registry.registerMeshLoader('/mesh', meshLoader);

    // First sweep: slice sits on w = 0, where all three vertices live.
    await sceneLoader.updateView({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1e10, 1e10, 1e10, 0.5],
    });
    expect((meshLoader.updateView as any).mock.calls.length).toBeGreaterThan(0);
    const callsAfterFirst = (meshLoader.updateView as any).mock.calls.length;
    // `drawRange`, not `index.count`: the index buffer is allocated once at the
    // node's face-count capacity and the visible prefix is drawn via the draw range
    // (see `rendering/mesh-geometry.ts` — replacing the index per epoch leaks its GPU
    // buffer). `index.count` therefore stays at the capacity across sweeps and could
    // not witness a cull at all.
    expect(meshNode.geometry.drawRange.count).toBe(3);

    // Second sweep: slice moves to w = 99. Every vertex is at w = 0, far outside
    // the ±0.5 slab, so the whole triangle culls. This can ONLY hold if the mesh
    // re-joined the sweep and re-projected — a frozen initial projection would
    // leave the draw range at 3.
    await sceneLoader.updateView({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 99],
      tolerance: [1e10, 1e10, 1e10, 0.5],
    });
    expect((meshLoader.updateView as any).mock.calls.length).toBeGreaterThan(callsAfterFirst);
    expect(meshNode.geometry.drawRange.count).toBe(0);
  });

  it('extend_to_all survives the sweep: an extended mesh away from w = 0 stays visible', async () => {
    await sceneLoader.loadScene('http://localhost:8000/test.zarr');

    // Triangle at w = 50, node extended across w. The derivation pins an
    // extended dim's slicePosition to a constant 0 (purely to stabilize the
    // same-view no-op — see derive-node-view-state.ts Step 4), so visibility
    // rests ENTIRELY on the infinite-tolerance sentinel. `processMeshData`
    // RECOMPUTES the membership tolerance (discarding the derived sentinel) and
    // re-applies the extension from the attrs the sweep handler passes it —
    // dropping `extend_to_all` there culls every extended mesh that does not
    // happen to sit at w = 0 (|50 − 0| ≫ the recomputed step × 1 slab).
    const meshData = loaded(50);
    const meshLoader = {
      updateView: vi.fn().mockResolvedValue(meshData),
      loadMesh: vi.fn().mockResolvedValue(meshData),
      dispose: vi.fn(),
    } as unknown as MeshDataLoader;

    const meshNode = createEmptyMeshNode('/mesh', { ...ATTRS, extend_to_all: ['w'] }, meshLoader);
    (sceneLoader as any).rootGroup.add(meshNode);
    (sceneLoader as any).registry.registerMeshLoader('/mesh', meshLoader);

    await sceneLoader.updateView({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [1e10, 1e10, 1e10, 0.5],
    });
    // `drawRange`, not `index.count` — the latter is the node's fixed capacity, so it
    // would read 3 even with every triangle culled and could not witness visibility.
    expect(meshNode.geometry.drawRange.count).toBe(3);

    // Scrub w: an extended node must stay slice-invariant.
    await sceneLoader.updateView({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 99],
      tolerance: [1e10, 1e10, 1e10, 0.5],
    });
    expect(meshNode.geometry.drawRange.count).toBe(3);
  });
});
