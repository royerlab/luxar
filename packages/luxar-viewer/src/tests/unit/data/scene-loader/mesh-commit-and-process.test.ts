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
  resetMeshNoticesForTesting,
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
  dim3: { name: string; discrete?: boolean; step?: number; unit?: string }
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
    resetMeshNoticesForTesting();
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
    const mesh = createEmptyMeshNode(path, ATTRS, loader, null);
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

    // `drawRange` is the drawn quantity; `index.count` is the node's capacity.
    expect(mesh.geometry.drawRange.count).toBe(3);
    expect(mesh.geometry.getAttribute('position').count).toBe(3);
    expect(mesh.userData.visibleTriangleCount).toBe(1);
    expect(mesh.userData.loadedViewVersion).toBe(7);
  });

  it('applies the node transform to the placeholder, like the sibling factories', () => {
    // `MeshMetadata.transform` is column-major (THREE.js layout, translation at
    // [12..14]). Points/lines/gsplats all apply it at creation; a mesh that skips
    // it renders in untransformed coordinates — silently, since nothing else
    // consumes the attr.
    const withTransform: MeshMetadata = {
      ...ATTRS,
      transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1],
    };
    const mesh = createEmptyMeshNode('/surface', withTransform, loader, null);
    expect(mesh.position.toArray()).toEqual([5, 6, 7]);
  });

  it('installs authored per-vertex colors so the mesh actually displays them', async () => {
    // Regression for #1243: the node is born with the 1-vertex placeholder color,
    // and the commit must grow `color` to the authored buffer. Before the fix the
    // color attribute stayed count 1 (the placeholder) and authored colors never
    // rendered.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', ATTRS_COLORS, loader, null);
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
    // Nothing drawn, and not a stale draw. The index buffer keeps its capacity (it is
    // allocated once per node), so what has to go to zero is the DRAW RANGE.
    expect(mesh.geometry.drawRange.count).toBe(0);
    expect(mesh.userData.visibleTriangleCount).toBe(0);
  });
});

describe('processMeshData — the continuous-hidden-dim notice (§9 evidence gate)', () => {
  beforeEach(() => {
    resetMeshNoticesForTesting();
    // Restore here, not with a trailing `mockRestore()` per test: a failing assertion
    // throws before the trailing call runs, leaving the spy installed so the NEXT test
    // sees this one's calls and fails for a reason that has nothing to do with it.
    vi.restoreAllMocks();
  });

  it('reports a continuous hidden dim, naming it and its unit', async () => {
    // The measurement behind §9's deferral of exact nD clipping: a continuous hidden
    // dim is exactly when the whole-triangle slab stops being a true cut. The name and
    // unit are in the message for the one judgement no metadata flag can make — a dim
    // can be declared spatial and still be a time axis, and only the name says so.
    const info = vi.spyOn(log, 'info').mockImplementation(() => {});
    // A real unit, because the unit is half of what the message exists to convey — it
    // is what lets a reader tell a spatial axis from a temporal one. The test was named
    // for it and did not assert it.
    const view = viewWithDim(1.0, { name: 'z2', step: 1, unit: 'um' });
    await processMeshData('/continuous', loadedAtW(0), view, {
      normal_dims: [0, 1, 2],
      double_sided: false,
    });
    const messages = info.mock.calls.map((c) => String(c[1]));
    const hit = messages.find((m) => m.includes('/continuous') && m.includes('§5.2.1'));
    expect(hit).toBeDefined();
    expect(hit).toContain('z2');
    expect(hit).toContain('[um]');
  });

  it('names every continuous hidden dim, batched into one line', async () => {
    // Two continuous hidden axes are one node's worth of evidence, so they are named
    // together in a single line rather than one line each — the shape §5.2.1 states.
    // Both must appear: WHICH axes turn up hidden-and-continuous is the entire payload,
    // so a message that reported only the first would lose half the measurement.
    const info = vi.spyOn(log, 'info').mockImplementation(() => {});
    const data: LoadedMeshData = {
      // One triangle in 5D, all three vertices at hidden (z2, z3) = (0, 0).
      vertices: new Float32Array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
      faces: new Uint32Array([0, 1, 2]),
      normals: null,
      colors: null,
      scalars: undefined,
      vertexCount: 3,
      faceCount: 1,
      ndim: 5,
    };
    // z3 carries no unit, so it also covers the bracket-less arm of the description.
    const view = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0, 0],
      tolerance: [1e10, 1e10, 1e10, 1, 1],
      dimensions: [
        { name: 'x', unit: '', scale: 1 },
        { name: 'y', unit: '', scale: 1 },
        { name: 'z', unit: '', scale: 1 },
        { name: 'z2', unit: 'um', scale: 1, step: 1 },
        { name: 'z3', unit: '', scale: 1, step: 1 },
      ],
    } as MeshViewState;
    await processMeshData('/two-axes', data, view, {
      normal_dims: [0, 1, 2],
      double_sided: false,
    });
    const hits = info.mock.calls.map((c) => String(c[1])).filter((m) => m.includes('§5.2.1'));
    expect(hits).toHaveLength(1);
    expect(hits[0]).toContain('z2 [um]');
    expect(hits[0]).toContain('z3');
  });

  it('stays silent when the hidden dim is discrete — the dominant real case', async () => {
    // Time/channel hidden dims get a TRUE cut from the half-cell membership rule, so
    // there is nothing to report. If this fired here the signal would be worthless:
    // almost every mesh in the wild has a discrete hidden dim or none at all.
    const info = vi.spyOn(log, 'info').mockImplementation(() => {});
    await processMeshData(
      '/discrete',
      loadedAtW(0),
      viewWithDim(0.5, { name: 't', discrete: true, step: 1 }),
      { normal_dims: [0, 1, 2], double_sided: false }
    );
    expect(info.mock.calls.map((c) => String(c[1])).some((m) => m.includes('§5.2.1'))).toBe(false);
  });

  it('stays silent when the continuous hidden dim is extend_to_all', async () => {
    // An extended dim is slice-invariant, so its membership slab is infinite
    // (EXTEND_TO_ALL_TOLERANCE) and there is no finite thickness to report. Reporting it
    // would put the one case the approximation provably cannot bite into the evidence.
    const info = vi.spyOn(log, 'info').mockImplementation(() => {});
    await processMeshData(
      '/extended',
      loadedAtW(0),
      viewWithDim(1.0, { name: 'z2', step: 1, unit: 'um' }),
      { normal_dims: [0, 1, 2], double_sided: false, extend_to_all: ['z2'] }
    );
    expect(info.mock.calls.map((c) => String(c[1])).some((m) => m.includes('§5.2.1'))).toBe(false);
  });

  it('reports the same axis again when it comes back with a different unit', async () => {
    // The dedup key is the message text — name AND unit — not the name alone. A dataset
    // switch that reuses a node path and an axis name but changes the unit is new
    // evidence: the unit is half of what the reader classifies on, so suppressing the
    // second line would hide exactly the distinction the message exists to draw.
    const info = vi.spyOn(log, 'info').mockImplementation(() => {});
    for (const unit of ['s', 'um']) {
      const view = viewWithDim(1.0, { name: 'w', step: 1, unit });
      await processMeshData('/reload', loadedAtW(0), view, {
        normal_dims: [0, 1, 2],
        double_sided: false,
      });
    }
    const hits = info.mock.calls.map((c) => String(c[1])).filter((m) => m.includes('§5.2.1'));
    expect(hits).toHaveLength(2);
    expect(hits[0]).toContain('[s]');
    expect(hits[1]).toContain('[um]');
  });

  it('reports a NEWLY hidden dimension, even after the node was already noticed', async () => {
    // The failure keying by path alone would cause, and the one that matters most:
    // this notice exists to COLLECT evidence about which axes turn up
    // hidden-and-continuous. A 4D mesh that first reports a continuous time axis would
    // then have the early return suppress a continuous Z forever once displayDims
    // changed — so the one configuration the measurement is looking for is the one it
    // would never see.
    const info = vi.spyOn(log, 'info').mockImplementation(() => {});
    const first = viewWithDim(1.0, { name: 'time', step: 1 });
    await processMeshData('/swaps', loadedAtW(0), first, {
      normal_dims: [0, 1, 2],
      double_sided: false,
    });

    // Same node, different displayDims: dim 2 ("z") is now hidden and continuous.
    // Built as a new literal rather than mutated, because `displayDims` is readonly.
    const second: MeshViewState = {
      ...viewWithDim(1.0, { name: 'time', step: 1 }),
      displayDims: [0, 1, 3],
    };
    await processMeshData('/swaps', loadedAtW(0), second, {
      normal_dims: [0, 1, 2],
      double_sided: false,
    });

    const hits = info.mock.calls.map((c) => String(c[1])).filter((m) => m.includes('§5.2.1'));
    expect(hits).toHaveLength(2);
    expect(hits[0]).toContain('time');
    expect(hits[1]).toContain('z');
  });

  it('reports once per node, not once per slice move', async () => {
    const info = vi.spyOn(log, 'info').mockImplementation(() => {});
    for (let i = 0; i < 3; i++) {
      await processMeshData('/scrub', loadedAtW(0), viewWithDim(1.0, { name: 'z2', step: 1 }), {
        normal_dims: [0, 1, 2],
        double_sided: false,
      });
    }
    const hits = info.mock.calls.map((c) => String(c[1])).filter((m) => m.includes('§5.2.1'));
    expect(hits).toHaveLength(1);
  });
});

describe('processMeshData — membership tolerance', () => {
  beforeEach(() => {
    resetMeshNoticesForTesting();
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

  it('DRAWS a triangle inside the continuous slab — the arm’s reason for existing', async () => {
    // The positive direction of the trap the continuous arm exists for, and the one
    // direction nothing asserted: the three unit tests in `tolerance-computer.test.ts`
    // pin the NUMBER the arm returns, so replacing the arm with Lines' `0` fails them —
    // but it left every mesh BEHAVIOUR suite green, including this describe block,
    // whose continuous cases all assert culls. The documented consequence of `0` is
    // "the node renders nothing", and that was pinned nowhere.
    //
    // Triangle at w = 0.4, slice at w = 0, step 1 ⇒ slab = step × 1 cell = 1, so
    // |0.4| ≤ 1 and it draws. The ride-along tolerance is passed as 0 so it cannot
    // rescue the assertion: the recomputed mesh slab is the only thing that can make
    // this triangle visible.
    const inside = await processMeshData(
      '/surface',
      loadedAtW(0.4),
      viewWithDim(0, { name: 'w', discrete: false, step: 1 }),
      { normal_dims: undefined, double_sided: true }
    );
    expect(inside.projected.visibleFaceCount).toBe(1);

    // Exactly on the slab edge. Both kernels compare with `>=` / `<=` against
    // `fround(slice ± tolerance)`, so the edge is INCLUSIVE — a vertex one cell away
    // still draws rather than flickering out at the boundary.
    const onEdge = await processMeshData(
      '/surface',
      loadedAtW(1),
      viewWithDim(0, { name: 'w', discrete: false, step: 1 }),
      { normal_dims: undefined, double_sided: true }
    );
    expect(onEdge.projected.visibleFaceCount).toBe(1);

    // Absent dimension metadata takes the same arm through its `slabCells` fallback
    // (VIEW carries no `dimensions`), so a store with no per-dim step still draws
    // instead of silently culling everything.
    const noMetadata = await processMeshData('/surface', loadedAtW(0.4), VIEW, {
      normal_dims: undefined,
      double_sided: true,
    });
    expect(noMetadata.projected.visibleFaceCount).toBe(1);
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

describe('process -> commit: the position buffer is uploaded once per epoch (#1245)', () => {
  const loader = {} as MeshDataLoader;

  /** Loaded data carrying the loader-owned projection scratch, as production does. */
  function withScratch(): LoadedMeshData {
    const data = loaded();
    return {
      ...data,
      projection: {
        position: new Float32Array(data.vertexCount * 3),
        displayDimsKey: null,
        mask: new Uint8Array(data.vertexCount),
        faceScratch: new Uint32Array(data.faceCount * 3),
      },
    };
  }

  async function commitAt(root: THREE.Group, data: LoadedMeshData, view: MeshViewState) {
    const staged = await processMeshData('/surface', data, view, {
      normal_dims: [0, 1, 2],
      double_sided: true,
    });
    commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);
  }

  it('binds the reused buffer, then re-uploads only when displayDims changes', async () => {
    // This is the production path the unit tests could not reach: after the first
    // commit the geometry's `position` attribute IS the loader's reused buffer, so
    // array identity matches from then on. Gating the upload on identity would
    // suppress EVERY later upload — including after an axis permutation, leaving the
    // mesh in the stale frame while the rest of the scene moves.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', ATTRS, loader, null);
    root.add(mesh);
    const data = withScratch();

    await commitAt(root, data, VIEW);
    const positionAttr = mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    // The attribute now wraps the loader's buffer — hence identity is useless below.
    expect(positionAttr.array).toBe(data.projection!.position);
    const afterFirst = positionAttr.version;

    // A pure slice move: same displayDims, so positions cannot have changed.
    await commitAt(root, data, { ...VIEW, slicePosition: [0, 0, 0, 10] } as MeshViewState);
    expect(mesh.geometry.getAttribute('position')).toBe(positionAttr);
    expect(positionAttr.version).toBe(afterFirst);

    // An axis permutation: positions are re-extracted into the same buffer, so the
    // upload MUST be re-flagged.
    await commitAt(root, data, { ...VIEW, displayDims: [3, 0, 1] } as MeshViewState);
    expect(mesh.geometry.getAttribute('position')).toBe(positionAttr);
    expect(positionAttr.version).toBeGreaterThan(afterFirst);
  });
});

describe('process -> commit: geometry bounds cover only the drawn triangles (#1252)', () => {
  const loader = {} as MeshDataLoader;

  /** Two independent triangles in 4D: one at w = 0 near the origin, one at w = 10 far away. */
  function twoTriangles(): LoadedMeshData {
    return {
      // prettier-ignore
      vertices: new Float32Array([
        0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        900, 900, 900, 10, 901, 900, 900, 10, 900, 901, 900, 10,
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

  async function commitAt(root: THREE.Group, data: LoadedMeshData, w: number) {
    const staged = await processMeshData(
      '/surface',
      data,
      { ...VIEW, slicePosition: [0, 0, 0, w] } as MeshViewState,
      { normal_dims: [0, 1, 2], double_sided: true }
    );
    commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);
  }

  it('bounds the DRAWN triangle, not the whole position buffer', async () => {
    // The regression. `position` holds all six vertices, so `computeBoundingBox()`
    // would span out to 900 — framing a 4D surface's whole trajectory rather than the
    // slice on screen. The mechanism is the projection's bounds reaching
    // `computeMeshBounds`, so the assertion belongs on the geometry, where camera
    // framing (and frustum culling, and the raycast broad phase) all read it.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', ATTRS, loader, null);
    root.add(mesh);

    await commitAt(root, twoTriangles(), 0);
    expect(mesh.geometry.boundingBox!.max.toArray()).toEqual([1, 1, 0]);
  });

  it('follows the slice to the far triangle', async () => {
    // Anti-vacuity: a box that always described the near triangle would pass above.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', ATTRS, loader, null);
    root.add(mesh);

    await commitAt(root, twoTriangles(), 10);
    expect(mesh.geometry.boundingBox!.min.toArray()).toEqual([900, 900, 900]);
  });

  it('empties the box and the sphere when nothing is drawn', async () => {
    // Three's own empty-box guard supplies the sphere: `Box3.getBoundingSphere()` calls
    // `Sphere.makeEmpty()`, i.e. center (0,0,0) and radius -1, which the frustum test
    // rejects. Asserted so a future "simplification" that derives the sphere some other
    // way cannot silently produce a NaN center.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', ATTRS, loader, null);
    root.add(mesh);

    await commitAt(root, twoTriangles(), 999);
    expect(mesh.geometry.boundingBox!.isEmpty()).toBe(true);
    expect(mesh.geometry.boundingSphere!.radius).toBe(-1);
    expect(mesh.geometry.boundingSphere!.center.toArray()).toEqual([0, 0, 0]);
  });
});

describe('process -> commit: the vertex attribute set is frozen after the first commit', () => {
  const loader = {} as MeshDataLoader;

  function withScratch(): LoadedMeshData {
    const d = loaded();
    return {
      ...d,
      colors: new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]),
      colorComponents: 3,
      projection: {
        position: new Float32Array(d.vertexCount * 3),
        displayDimsKey: null,
        mask: new Uint8Array(d.vertexCount),
        faceScratch: new Uint32Array(d.faceCount * 3),
      },
    };
  }

  it('rebinds nothing after the first commit, across slice moves AND axis permutations', async () => {
    // Three r184's WebGPU backend keys a pipeline's vertex-buffer layout by attribute
    // IDENTITY, and `getGeometryCacheKey` hashes only names/itemSize/normalized — so a
    // rebind that the cache key cannot see leaves the draw reading the OLD buffer. The
    // commit's `invalidateRenderObjectFor` covers the first-commit rebind; what must hold
    // afterwards is that NO further rebind happens at all, or a scrub would evict the
    // render object on every frame.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', ATTRS, loader, null);
    root.add(mesh);
    const data = withScratch();

    const epochs: MeshViewState[] = [
      VIEW,
      { ...VIEW, slicePosition: [0, 0, 0, 10] } as MeshViewState,
      { ...VIEW, displayDims: [3, 0, 1] } as MeshViewState,
      { ...VIEW, displayDims: [0, 1, 2], slicePosition: [0, 0, 0, 99] } as MeshViewState,
    ];

    const seen: { names: string[]; position: unknown; color: unknown }[] = [];
    for (const view of epochs) {
      const staged = await processMeshData('/surface', data, view, {
        normal_dims: [0, 1, 2],
        double_sided: true,
      });
      commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);
      seen.push({
        names: Object.keys(mesh.geometry.attributes).sort(),
        position: mesh.geometry.getAttribute('position'),
        color: mesh.geometry.getAttribute('color'),
      });
    }

    // The attribute SET is identical at every epoch — nothing joins or leaves.
    for (const s of seen) expect(s.names).toEqual(['color', 'position']);
    // And every attribute OBJECT is the one installed by the first commit.
    for (const s of seen.slice(1)) {
      expect(s.position).toBe(seen[0].position);
      expect(s.color).toBe(seen[0].color);
    }
    // The index attribute is likewise stable (its capacity buffer is allocated once).
    expect(mesh.geometry.index).not.toBeNull();
  });
});

describe('process -> commit: the shading variant follows the epoch (§3.4)', () => {
  const loader = {} as MeshDataLoader;

  /** A smooth-shaded, normal-bearing mesh — the only configuration that reads normals. */
  const SMOOTH: MeshMetadata = {
    ...ATTRS,
    shading: 'smooth',
    has_normals: true,
    normal_dims: [0, 1, 2],
  };

  function withNormals(): LoadedMeshData {
    return { ...loaded(), normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]) };
  }

  const isFlatVariant = (mesh: THREE.Mesh): boolean => {
    const defines = (mesh.material as THREE.ShaderMaterial).defines ?? {};
    return 'LUXAR_MESH_FLAT_NORMAL' in defines;
  };

  async function commitAt(root: THREE.Group, data: LoadedMeshData, view: MeshViewState) {
    const staged = await processMeshData('/surface', data, view, {
      normal_dims: SMOOTH.normal_dims,
      double_sided: true,
    });
    commitMeshGeometry({ rootGroup: root, currentVersion: 1 }, staged);
  }

  it('flips smooth -> flat when displayDims leaves the authored frame, and back', async () => {
    // The WIRING test. `applyMeshShading` and `storedNormalsUsable` each have their own
    // unit coverage, but the connection between them lives only in the commit — and
    // deleting that one call broke NOTHING in the suite before this test existed. A
    // mesh would then keep reading normals authored for other axes, silently shading
    // against a tilted frame with no diagnostic anywhere.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', SMOOTH, loader, null);
    root.add(mesh);
    const data = withNormals();

    // displayDims == normal_dims: the stored normals are meaningful.
    await commitAt(root, data, { ...VIEW, displayDims: [0, 1, 2] } as MeshViewState);
    expect(isFlatVariant(mesh), 'smooth variant expected at the authored frame').toBe(false);

    // A PERMUTATION of the same triple. Winding still calls this decidable, but a
    // normal's components are positionally bound to normal_dims, so it must drop to
    // the derivative fallback rather than shade against a tilted frame.
    await commitAt(root, data, { ...VIEW, displayDims: [1, 0, 2] } as MeshViewState);
    expect(isFlatVariant(mesh), 'permuted frame must fall back to derivatives').toBe(true);

    // A DIFFERENT triple: likewise flat.
    await commitAt(root, data, { ...VIEW, displayDims: [1, 2, 3] } as MeshViewState);
    expect(isFlatVariant(mesh)).toBe(true);

    // Back to the authored frame: the variant must recover, not latch.
    await commitAt(root, data, { ...VIEW, displayDims: [0, 1, 2] } as MeshViewState);
    expect(isFlatVariant(mesh), 'the variant must not latch flat').toBe(false);
  });

  it('costs no recompile across a pure slice move', async () => {
    // The commit calls this on every epoch, and a variant flip recompiles the
    // program, so an unguarded write would pay that on every frame of a scrub.
    const root = new THREE.Group();
    const mesh = createEmptyMeshNode('/surface', SMOOTH, loader, null);
    root.add(mesh);
    const data = withNormals();

    await commitAt(root, data, { ...VIEW, displayDims: [0, 1, 2] } as MeshViewState);
    const version = (mesh.material as THREE.Material).version;
    for (const w of [1, 2, 3]) {
      await commitAt(root, data, {
        ...VIEW,
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, w],
      } as MeshViewState);
    }
    expect((mesh.material as THREE.Material).version).toBe(version);
  });
});
