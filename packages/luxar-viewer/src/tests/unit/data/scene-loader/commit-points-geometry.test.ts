/**
 * Unit tests for the points geometry-commit handler.
 *
 * Strategy: real `THREE.Group` + `THREE.Points` so the
 * getObjectByName lookup is exercised; mocked NodeFactory and
 * GPUBufferPool stubs to detect which code path ran.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import { commitPointsGeometry } from '../../../../data/scene-loader/commit/commit-points-geometry';
import type { LoadedPointsData } from '../../../../data/data-loader-types';
import type { NodeFactory } from '../../../../rendering/node-factory';

function makeData(pointCount: number, withRadii = false): LoadedPointsData {
  return {
    positions: new Float32Array(pointCount * 3),
    colors: new Uint8Array(pointCount * 3),
    radii: withRadii ? new Float32Array(pointCount) : undefined,
    sharpness: undefined,
    pointCount,
    metadata: {
      bounds: new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1)),
    },
  } as unknown as LoadedPointsData;
}

function makePoints(name: string): THREE.Mesh {
  // Points are THREE.Mesh with instanced quad geometry. Per-instance
  // attribute is `aCenter` (InstancedBufferAttribute).
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(new Float32Array(0), 3));
  const material = new THREE.MeshBasicMaterial();
  const points = new THREE.Mesh(geometry, material);
  points.name = name;
  points.userData = { nodeType: 'points', visiblePointCount: 0 };
  return points;
}

const mockCreatePointsGeometry = vi.fn(() => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('aCenter', new THREE.InstancedBufferAttribute(new Float32Array(9), 3));
  return g;
});
const mockNodeFactory = {
  createPointsGeometry: mockCreatePointsGeometry,
} as unknown as NodeFactory;

beforeEach(() => {
  mockCreatePointsGeometry.mockClear();
});

describe('commitPointsGeometry', () => {
  it('no-ops when rootGroup is null', () => {
    expect(() =>
      commitPointsGeometry('/p', makeData(5), null, null, mockNodeFactory, undefined, 0)
    ).not.toThrow();
  });

  it('no-ops when no points node with the given path exists', () => {
    expect(() =>
      commitPointsGeometry(
        '/missing',
        makeData(5),
        new THREE.Group(),
        null,
        mockNodeFactory,
        undefined,
        0
      )
    ).not.toThrow();
    expect(mockCreatePointsGeometry).not.toHaveBeenCalled();
  });

  it('updates visiblePointCount on the userData (pool disabled, pointCount=0)', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const oldGeom = points.geometry;
    const disposeSpy = vi.spyOn(oldGeom, 'dispose');
    // C8[P2][P11]: a mutant dropping the GPU geometry rebuild would still pass
    // the userData write below. The pool-disabled path with pointCount=0 vs an
    // existing 0-count geometry takes the dispose+recreate branch (the in-place
    // branch requires pointCount > 0). Pin the actual dispatch AND assert no crash.
    expect(() =>
      commitPointsGeometry('/p', makeData(0), root, null, mockNodeFactory, undefined, 0)
    ).not.toThrow();
    expect((points.userData as { visiblePointCount: number }).visiblePointCount).toBe(0);
    expect(disposeSpy).toHaveBeenCalledTimes(1);
    expect(mockCreatePointsGeometry).toHaveBeenCalledTimes(1);
    // The rebuilt geometry replaces the old slot.
    expect(points.geometry).not.toBe(oldGeom);
  });

  it('uses GPU buffer pool when supplied', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);

    const newGeometry = new THREE.BufferGeometry();
    const gpuBufferPool = {
      acquirePointsGeometry: vi.fn(() => newGeometry),
      updatePointsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => false),
    };

    commitPointsGeometry(
      '/p',
      makeData(3),
      root,
      gpuBufferPool as never,
      mockNodeFactory,
      undefined,
      0
    );
    expect(gpuBufferPool.acquirePointsGeometry).toHaveBeenCalledTimes(1);
    expect(gpuBufferPool.updatePointsGeometry).toHaveBeenCalledTimes(1);
    expect(points.geometry).toBe(newGeometry);
    expect(mockCreatePointsGeometry).not.toHaveBeenCalled();
  });

  it('stamps loadedViewVersion onto the mesh user-data (three-geometry symmetry)', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory, undefined, 9);
    expect((points.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(9);
  });

  it('falls back to dispose+create when pool disabled and counts differ', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const oldGeom = points.geometry;
    const disposeSpy = vi.spyOn(oldGeom, 'dispose');

    // Existing geometry has 0 positions; new data has 3 → counts differ.
    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory, undefined, 0);
    expect(disposeSpy).toHaveBeenCalledTimes(1);
    expect(mockCreatePointsGeometry).toHaveBeenCalledTimes(1);
  });

  it('updates attributes in place when pool disabled and counts match', () => {
    const root = new THREE.Group();
    const points = new THREE.Mesh();
    points.name = '/p';
    points.userData = { nodeType: 'points', visiblePointCount: 0 };
    // 3-point geometry: production points geometries pack per-instance
    // attributes into one `InstancedInterleavedBuffer` with views per
    // attribute. The in-place commit path writes through these views,
    // so the test fixture must mirror that shape.
    const geom = new THREE.InstancedBufferGeometry();
    const stride = 3; // aCenter only — minimal layout for this test.
    const interleaved = new THREE.InstancedInterleavedBuffer(
      new Float32Array(3 * stride),
      stride,
      1
    );
    geom.setAttribute('aCenter', new THREE.InterleavedBufferAttribute(interleaved, 3, 0));
    points.geometry = geom;
    root.add(points);

    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory, undefined, 0);
    // No dispose / no recreate → in-place path.
    expect(mockCreatePointsGeometry).not.toHaveBeenCalled();
    // The same geometry instance is preserved.
    expect(points.geometry).toBe(geom);
    expect((points.userData as { visiblePointCount: number }).visiblePointCount).toBe(3);
  });

  it('bakes the radius footprint into boundingBox (three-geometry invariant)', () => {
    // Symmetry with lines/gsplats: the committed boundingBox must include
    // the rendered disc footprint, not just the centers — so the pick cull
    // (ray-aabb) and camera framing cover large radii. Here centers bounds
    // are [-1,1] and max_radius is 10 → boundingBox grows to [-11,11].
    const root = new THREE.Group();
    const points = new THREE.Mesh();
    points.name = '/p';
    points.userData = {
      nodeType: 'points',
      visiblePointCount: 0,
      attrs: { max_radius: 10 },
    };
    const stride = 3;
    const interleaved = new THREE.InstancedInterleavedBuffer(
      new Float32Array(3 * stride),
      stride,
      1
    );
    const geom = new THREE.InstancedBufferGeometry();
    geom.setAttribute('aCenter', new THREE.InterleavedBufferAttribute(interleaved, 3, 0));
    geom.setAttribute('aRadius', new THREE.InterleavedBufferAttribute(interleaved, 1, 0));
    points.geometry = geom;
    root.add(points);

    commitPointsGeometry(
      '/p',
      makeData(3, /*withRadii=*/ true),
      root,
      null,
      mockNodeFactory,
      undefined,
      0
    );

    expect(points.geometry).toBe(geom);
    expect(geom.boundingBox).not.toBeNull();
    expect(geom.boundingBox!.min.x).toBeCloseTo(-11, 5);
    expect(geom.boundingBox!.max.x).toBeCloseTo(11, 5);
  });
});

describe('commitPointsGeometry — no-op commit skip (committedData)', () => {
  it('stamps committedData with the raw data on a real commit', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const data = makeData(3);
    commitPointsGeometry('/p', data, root, null, mockNodeFactory, undefined, 4);
    expect((points.userData as { committedData?: unknown }).committedData).toBe(data);
  });

  it('skips geometry work when the SAME data reference is committed again, but refreshes the freshness stamp', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const data = makeData(3);
    commitPointsGeometry('/p', data, root, null, mockNodeFactory, undefined, 4);
    const geometryAfterFirst = points.geometry;
    mockCreatePointsGeometry.mockClear();

    commitPointsGeometry('/p', data, root, null, mockNodeFactory, undefined, 9);

    // Geometry untouched, no rebuild dispatched, stamp refreshed.
    expect(points.geometry).toBe(geometryAfterFirst);
    expect(mockCreatePointsGeometry).not.toHaveBeenCalled();
    expect((points.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(9);
  });

  it('recommits when a DIFFERENT data reference arrives', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const first = makeData(3);
    commitPointsGeometry('/p', first, root, null, mockNodeFactory, undefined, 4);
    mockCreatePointsGeometry.mockClear();

    const second = makeData(5); // different reference AND count → rebuild path
    commitPointsGeometry('/p', second, root, null, mockNodeFactory, undefined, 5);

    expect(mockCreatePointsGeometry).toHaveBeenCalledTimes(1);
    expect((points.userData as { committedData?: unknown }).committedData).toBe(second);
  });
});

describe('commitPointsGeometry — committedLadderComplete stamp', () => {
  const ladderComplete = (points: THREE.Mesh) =>
    (points.userData as { committedLadderComplete?: boolean }).committedLadderComplete;

  it('stamps false while the committing progressive loader has more LODs', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    points.userData.loader = { hasMoreLODs: true };
    root.add(points);
    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory, undefined, 0);
    expect(ladderComplete(points)).toBe(false);
  });

  it('stamps true on the final ladder pass and for non-progressive / loaderless meshes', () => {
    for (const loader of [{ hasMoreLODs: false }, {}, undefined]) {
      const root = new THREE.Group();
      const points = makePoints('/p');
      if (loader) points.userData.loader = loader;
      root.add(points);
      commitPointsGeometry('/p', makeData(2), root, null, mockNodeFactory, undefined, 0);
      expect(ladderComplete(points)).toBe(true);
    }
  });

  it('no-op (already-committed) commit refreshes the ladder stamp too', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    points.userData.loader = { hasMoreLODs: true };
    root.add(points);
    const data = makeData(3);
    commitPointsGeometry('/p', data, root, null, mockNodeFactory, undefined, 1);
    // Force a stale stamp, then recommit the SAME reference → stamp-only path.
    points.userData.committedLadderComplete = true;
    commitPointsGeometry('/p', data, root, null, mockNodeFactory, undefined, 2);
    expect(ladderComplete(points)).toBe(false); // refreshed from the live loader
  });
});
