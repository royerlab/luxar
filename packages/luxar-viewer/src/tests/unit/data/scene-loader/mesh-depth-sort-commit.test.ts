/**
 * The wiring between the mesh commit and the depth-sort coordinator.
 *
 * `triangle-ordering.test.ts` pins the apply in isolation; this pins the half no
 * unit test of that module can reach — that the commit actually hands the
 * coordinator the four things a mesh sort needs, and that the ONE gate the
 * coordinator's resolve path checks before applying anything (`committedData`)
 * is satisfied. That gate is easy to get wrong invisibly: with it unset, every
 * sort still dispatches, still resolves, and is then silently dropped, so the
 * surface renders unsorted while nothing anywhere reports a problem.
 *
 * The coordinator is mocked at the module boundary — not because it is hard to
 * run, but because the real one spawns a SortWorker on the first order-dependent
 * commit, and what is under test here is the CALL, not the sort.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

const noteDepthSortCommit = vi.fn();
vi.mock('../../../../rendering/depth-sort-coordinator', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../../../rendering/depth-sort-coordinator')>()),
  noteDepthSortCommit: (...args: unknown[]) => noteDepthSortCommit(...args),
}));

const { commitMeshGeometry } =
  await import('../../../../data/scene-loader/commit/commit-mesh-geometry');
const { processMeshData } =
  await import('../../../../data/scene-loader/process/data-processor-mesh');
const { createEmptyMeshNode } = await import('../../../../rendering/node-factory/create-mesh-node');
const { hasCommittedData, getCommittedData } = await import('../../../../types/committed-data');

import type {
  LoadedMeshData,
  MeshDataLoader,
  MeshMetadata,
  MeshViewState,
} from '../../../../types/mesh';

const ATTRS: MeshMetadata = {
  type: 'mesh',
  n_vertices: 4,
  n_faces: 2,
  ndim: 4,
  has_normals: false,
  has_colors: false,
  has_scalars: false,
  shading: 'flat',
  double_sided: true,
  ordering: 'none',
};

/**
 * Two triangles in 4D on DIFFERENT hidden-dim slices, so the nD cull keeps
 * exactly one of them. That is what makes the "visible faces, not all faces"
 * claims below falsifiable — with `faceCount` passed where `visibleFaceCount`
 * belongs, the worker would sort a face the geometry is not drawing.
 */
function loaded(): LoadedMeshData {
  return {
    vertices: new Float32Array([
      // face 0 — at w = 0, so visible
      0, 0, 0, 0, 3, 0, 0, 0, 0, 6, 0, 0,
      // face 1 — at w = 10, culled
      0, 0, 0, 10, 1, 0, 0, 10, 0, 1, 0, 10,
    ]),
    faces: new Uint32Array([0, 1, 2, 3, 4, 5]),
    normals: null,
    colors: null,
    scalars: undefined,
    vertexCount: 6,
    faceCount: 2,
    ndim: 4,
  };
}

const VIEW: MeshViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [1e10, 1e10, 1e10, 0.5],
} as MeshViewState;

async function commitOnce(attrs: MeshMetadata = ATTRS): Promise<THREE.Mesh> {
  const root = new THREE.Group();
  const mesh = createEmptyMeshNode('/surface', attrs, {} as MeshDataLoader, null);
  root.add(mesh);
  const staged = await processMeshData('/surface', loaded(), VIEW, {
    normal_dims: undefined,
    double_sided: attrs.double_sided,
  });
  commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);
  return mesh;
}

beforeEach(() => {
  noteDepthSortCommit.mockClear();
});

describe('commitMeshGeometry — depth-sort registration', () => {
  it('stamps committedData, without which every resolved ordering is dropped', async () => {
    const mesh = await commitOnce();
    // The coordinator's resolve path, per-frame scheduler and capture drain all
    // gate on this stamp; mesh is the one geometry type with no memoized-noop
    // reason of its own to set it.
    expect(hasCommittedData(mesh)).toBe(true);
    expect(getCommittedData(mesh)).toBeDefined();
  });

  it('registers the VISIBLE face count and the epoch index, not the whole node', async () => {
    await commitOnce();
    expect(noteDepthSortCommit).toHaveBeenCalledTimes(1);
    const [object, centers, count, source] = noteDepthSortCommit.mock.calls[0];
    expect(object).toBeInstanceOf(THREE.Mesh);
    // One of the two authored faces survives the nD cull.
    expect(count).toBe(1);
    expect(source).toBeInstanceOf(Uint32Array);
    expect(Array.from(source as Uint32Array)).toEqual([0, 1, 2]);
    expect(typeof centers).toBe('function');
  });

  it('passes the centroids LAZILY — the O(F) pass is not paid by the commit', async () => {
    // The thunk contract: the coordinator resolves it only past the
    // order-dependence and generation checks, so an opaque mesh never pays it.
    // If the commit evaluated it eagerly that saving would be gone, and nothing
    // else would look different.
    await commitOnce();
    const centers = noteDepthSortCommit.mock.calls[0][1] as () => Float32Array;
    const produced = centers();
    // Face 0 is (0,0,0) (3,0,0) (0,6,0) in the displayed axes → centroid (1, 2, 0).
    expect(Array.from(produced)).toEqual([1, 2, 0]);
    // Fresh each call: the buffer is transferred to the worker and detached.
    expect(centers()).not.toBe(produced);
  });

  it('registers on EVERY commit, so a slice move invalidates the old ordering', async () => {
    // The generation bump is what drops an in-flight sort whose centers describe
    // the previous slice. Skipping registration when the mode is opaque would
    // break that — the coordinator, not the commit, decides what to do with it.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', ATTRS, {} as MeshDataLoader, null);
    root.add(mesh);
    for (const w of [0, 10]) {
      const staged = await processMeshData(
        '/surface',
        loaded(),
        { ...VIEW, slicePosition: [0, 0, 0, w] } as MeshViewState,
        { normal_dims: undefined, double_sided: true }
      );
      commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);
    }
    expect(noteDepthSortCommit).toHaveBeenCalledTimes(2);
    // The second slice shows the OTHER triangle, so the registered source moved
    // with it rather than being pinned at the first epoch.
    expect(Array.from(noteDepthSortCommit.mock.calls[1][3] as Uint32Array)).toEqual([3, 4, 5]);
  });

  it('registers a zero count for a fully-culled slice', async () => {
    // The coordinator releases the worker registration on an empty commit; it
    // can only do that if the commit still reports in.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', ATTRS, {} as MeshDataLoader, null);
    root.add(mesh);
    const staged = await processMeshData(
      '/surface',
      loaded(),
      { ...VIEW, slicePosition: [0, 0, 0, 100] } as MeshViewState,
      { normal_dims: undefined, double_sided: true }
    );
    commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);
    expect(noteDepthSortCommit.mock.calls[0][2]).toBe(0);
  });
});
