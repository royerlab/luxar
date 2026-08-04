/**
 * The mesh process → commit pair.
 *
 * Three behaviours here are easy to get subtly wrong and invisible when they are:
 *
 * 1. The undecidable-winding notice must fire **once per node**, not once per index
 *    build. The projection runs on every slice move, so a per-call warning turns a
 *    scrub into console spam.
 * 2. The commit must resolve its target by TYPE as well as name.
 *    `getObjectByName` searches the whole subtree, so a path collision would
 *    otherwise let mesh geometry be written into a points node — silently.
 * 3. The whole-triangle cull must run on the mesh's own recomputed membership slab
 *    (`computeTolerance('mesh', …)`), not the navigation ride-along
 *    `viewState.tolerance` — whose flat 0.5 / point-radius values have nothing to do
 *    with a mesh's cell size.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  processMeshData,
  resetWindingNoticesForTesting,
} from '../../../../data/scene-loader/process/data-processor-mesh';
import { commitMeshGeometry } from '../../../../data/scene-loader/commit/commit-mesh-geometry';
import { createEmptyMeshNode } from '../../../../rendering/node-factory/create-mesh-node';
import { log } from '../../../../utils/log';
import type {
  LoadedMeshData,
  MeshDataLoader,
  MeshMetadata,
  MeshViewState,
} from '../../../../types/mesh';

const ATTRS: MeshMetadata = {
  type: 'mesh',
  n_vertices: 3,
  n_faces: 1,
  ndim: 4,
  has_normals: false,
  has_colors: false,
  has_scalars: false,
  shading: 'flat',
  double_sided: false,
  ordering: 'none',
};

function loaded(): LoadedMeshData {
  return {
    // One triangle in 4D, all three vertices at w = 0.
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

const VIEW: MeshViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [1e10, 1e10, 1e10, 0.5],
} as MeshViewState;

/** One triangle in 4D, all three vertices at hidden dim `w = w`. */
function loadedAtW(w: number): LoadedMeshData {
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

/** Build a 4D view (displayDims [0,1,2], hidden dim 3) with per-dim metadata. */
function viewWithDim(
  toleranceW: number,
  dim3: { name: string; discrete?: boolean; step?: number }
): MeshViewState {
  // Full DimensionMetadata objects (unit/scale are required fields) so a single
  // `as MeshViewState` cast suffices, matching the VIEW const above.
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, 0],
    tolerance: [1e10, 1e10, 1e10, toleranceW],
    dimensions: [
      { name: 'x', unit: '', scale: 1 },
      { name: 'y', unit: '', scale: 1 },
      { name: 'z', unit: '', scale: 1 },
      { unit: '', scale: 1, ...dim3 },
    ],
  } as MeshViewState;
}

describe('processMeshData — the undecidable-winding notice', () => {
  beforeEach(() => {
    resetWindingNoticesForTesting();
    vi.restoreAllMocks();
  });

  it('warns ONCE per node however many times the projection runs', async () => {
    // A single-sided mesh with no stored normals declares no winding frame, so the
    // epoch is undecidable and falls back to double-sided with a notice.
    const warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
    for (let i = 0; i < 5; i++) {
      await processMeshData('/surface', loaded(), VIEW, {
        normal_dims: undefined,
        double_sided: false,
      });
    }
    const meshWarnings = warn.mock.calls.filter((c) => String(c[1]).includes('/surface'));
    expect(meshWarnings).toHaveLength(1);
    expect(String(meshWarnings[0][1])).toMatch(/single-sided/);
  });

  it('stays quiet for an authored double-sided mesh', async () => {
    // The overwhelmingly common case: both orientations draw, so parity is
    // unobservable and there is nothing to report.
    const warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
    await processMeshData('/surface', loaded(), VIEW, {
      normal_dims: undefined,
      double_sided: true,
    });
    expect(warn.mock.calls.filter((c) => String(c[1]).includes('/surface'))).toHaveLength(0);
  });

  it('reports the epoch side and the cull result on the staged commit', async () => {
    const staged = await processMeshData('/surface', loaded(), VIEW, {
      normal_dims: [0, 1, 2],
      double_sided: false,
    });
    expect(staged.path).toBe('/surface');
    expect(staged.projected.visibleFaceCount).toBe(1);
    expect(staged.projected.side).toBe('front');
  });
});

describe('commitMeshGeometry', () => {
  const loader = {} as MeshDataLoader;

  function sceneWithMesh(path: string): { root: THREE.Group; mesh: THREE.Mesh } {
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode(path, ATTRS, loader);
    root.add(mesh);
    return { root, mesh };
  }

  it('populates the placeholder and stamps the visible counts', async () => {
    const { root, mesh } = sceneWithMesh('/surface');
    const staged = await processMeshData('/surface', loaded(), VIEW, {
      normal_dims: [0, 1, 2],
      double_sided: false,
    });
    commitMeshGeometry({ rootGroup: root, currentVersion: 7 }, staged);

    expect(mesh.geometry.index?.count).toBe(3);
    expect(mesh.geometry.getAttribute('position').count).toBe(3);
    expect(mesh.userData.visibleTriangleCount).toBe(1);
    expect(mesh.userData.loadedViewVersion).toBe(7);
  });

  it('applies the epoch side, which can differ from the authored double_sided', async () => {
    // Authored single-sided, but no winding frame → the epoch must render both
    // faces or an open surface would vanish.
    const { root, mesh } = sceneWithMesh('/surface');
    expect((mesh.material as THREE.Material).side).toBe(THREE.FrontSide);
    const staged = await processMeshData('/surface', loaded(), VIEW, {
      normal_dims: undefined,
      double_sided: false,
    });
    commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);
    expect((mesh.material as THREE.Material).side).toBe(THREE.DoubleSide);
  });

  it('declines and warns when the named object is NOT a mesh node', async () => {
    // `getObjectByName` searches the whole subtree by name, so a collision could
    // hand back another type. Writing mesh geometry into it would corrupt it
    // silently.
    const warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const root = new THREE.Group();
    const impostor = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
    impostor.name = '/surface';
    impostor.userData = { nodeType: 'points' };
    root.add(impostor);

    const staged = await processMeshData('/surface', loaded(), VIEW, {
      normal_dims: [0, 1, 2],
      double_sided: false,
    });
    commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);

    expect(impostor.geometry.index).toBeNull();
    expect(warn).toHaveBeenCalled();
    expect(String(warn.mock.calls.at(-1)?.[1])).toMatch(/no mesh node named/);
  });

  it('is a no-op with no root group', async () => {
    const staged = await processMeshData('/surface', loaded(), VIEW, {
      normal_dims: [0, 1, 2],
      double_sided: false,
    });
    expect(() => commitMeshGeometry({ rootGroup: null, currentVersion: 1 }, staged)).not.toThrow();
  });

  it('commits an empty index when the slice culls every triangle', async () => {
    const { root, mesh } = sceneWithMesh('/surface');
    const staged = await processMeshData(
      '/surface',
      loaded(),
      { ...VIEW, slicePosition: [0, 0, 0, 99] } as MeshViewState,
      { normal_dims: [0, 1, 2], double_sided: false }
    );
    commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);
    // An empty index, not a stale one: the previous epoch's triangles must stop
    // drawing rather than lingering.
    expect(mesh.geometry.index?.count).toBe(0);
    expect(mesh.userData.visibleTriangleCount).toBe(0);
  });
});

describe('processMeshData — membership tolerance', () => {
  beforeEach(() => {
    resetWindingNoticesForTesting();
    vi.restoreAllMocks();
  });

  it('culls a discrete hidden dim at the half-cell from step, not the ride-along 0.5', async () => {
    // Triangle at w = 3, slice at w = 0. The ride-along tolerance for dim 3 is
    // 0.5 (what `simpleDimsToViewState` would emit) — under that value |3 − 0| > 0.5
    // and the triangle culls. The mesh's own membership slab is the half-cell of the
    // step: 0.5 × 10 = 5, so |3 − 0| ≤ 5 and the triangle stays visible.
    const staged = await processMeshData(
      '/surface',
      loadedAtW(3),
      viewWithDim(0.5, { name: 'w', discrete: true, step: 10 }),
      { normal_dims: undefined, double_sided: true }
    );
    expect(staged.projected.visibleFaceCount).toBe(1);

    // Upper edge: w = 7 exceeds the discrete half-cell (5) and must cull. This
    // pins the discrete arm specifically — a fall-through to the CONTINUOUS arm
    // (step × meshSlabTolerance = 10 × 1 = 10) would wrongly keep it.
    const overEdge = await processMeshData(
      '/surface',
      loadedAtW(7),
      viewWithDim(0.5, { name: 'w', discrete: true, step: 10 }),
      { normal_dims: undefined, double_sided: true }
    );
    expect(overEdge.projected.visibleFaceCount).toBe(0);
  });

  it('a large ride-along maxRadius no longer widens the continuous slab', async () => {
    // Triangle at w = 50, slice at w = 0. The ride-along tolerance is a huge
    // point-radius (1e6) that would keep the triangle. The mesh's continuous slab is
    // step × meshSlabTolerance = 1 × 1 = 1, so |50| > 1 and the triangle culls.
    const staged = await processMeshData(
      '/surface',
      loadedAtW(50),
      viewWithDim(1e6, { name: 'w', discrete: false, step: 1 }),
      { normal_dims: undefined, double_sided: true }
    );
    expect(staged.projected.visibleFaceCount).toBe(0);
  });

  it('extend_to_all keeps an extended dim slice-invariant', async () => {
    // Triangle at w = 1000, far outside any finite slab. Without extend_to_all the
    // half-cell membership (0.5 × 10 = 5) culls it; naming 'w' in extend_to_all lifts
    // dim 3 to the infinite sentinel, so the far vertex stays visible.
    const view = viewWithDim(0.5, { name: 'w', discrete: true, step: 10 });

    const without = await processMeshData('/surface', loadedAtW(1000), view, {
      normal_dims: undefined,
      double_sided: true,
    });
    expect(without.projected.visibleFaceCount).toBe(0);

    const withExtend = await processMeshData('/surface', loadedAtW(1000), view, {
      normal_dims: undefined,
      double_sided: true,
      extend_to_all: ['w'],
    });
    expect(withExtend.projected.visibleFaceCount).toBe(1);
  });
});
