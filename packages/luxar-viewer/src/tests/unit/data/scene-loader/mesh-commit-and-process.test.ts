/**
 * The mesh process → commit pair.
 *
 * Two behaviours here are easy to get subtly wrong and invisible when they are:
 *
 * 1. The undecidable-winding notice must fire **once per node**, not once per index
 *    build. The projection runs on every slice move, so a per-call warning turns a
 *    scrub into console spam.
 * 2. The commit must resolve its target by TYPE as well as name.
 *    `getObjectByName` searches the whole subtree, so a path collision would
 *    otherwise let mesh geometry be written into a points node — silently.
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

const ATTRS_COLORS: MeshMetadata = { ...ATTRS, has_colors: true };

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

// Same triangle, but with authored per-vertex RGB colors.
function loadedWithColors(): LoadedMeshData {
  return {
    ...loaded(),
    colors: new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]),
    colorComponents: 3,
  };
}

const VIEW: MeshViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [1e10, 1e10, 1e10, 0.5],
} as MeshViewState;

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

  it('installs authored per-vertex colors so the mesh actually displays them', async () => {
    // Regression for #1243: the node is born with the 1-vertex placeholder color,
    // and the commit must grow `color` to the authored buffer. Before the fix the
    // color attribute stayed count 1 (the placeholder) and authored colors never
    // rendered.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', ATTRS_COLORS, loader);
    root.add(mesh);

    const staged = await processMeshData('/surface', loadedWithColors(), VIEW, {
      normal_dims: [0, 1, 2],
      double_sided: false,
    });
    commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);

    const color = mesh.geometry.getAttribute('color');
    expect(color.count).toBe(3);
    expect(color.itemSize).toBe(4); // uint8 RGB padded to RGBA
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
