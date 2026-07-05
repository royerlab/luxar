/**
 * Direct tests for `commitGSplatsGeometry`, mirroring
 * `commit-points-geometry.test.ts` and `commit-lines-geometry.test.ts`.
 *
 * Mocks `updateInstancedGSplatsMesh` so the test doesn't pull in the
 * gsplat shader / r184 instanced mesh paths; we only verify the
 * orchestration (rootGroup / mesh guards + visibleSplatCount write).
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';

const mockUpdateInstancedMesh = vi.fn();
vi.mock('../../../../rendering/gsplat-geometry', () => ({
  updateInstancedGSplatsMesh: (...args: unknown[]) => mockUpdateInstancedMesh(...args),
  packCholeskyForShader: vi.fn(),
}));

import { commitGSplatsGeometry } from '../../../../data/scene-loader/commit/commit-gsplats-geometry';
import type { StagedGSplatsCommit } from '../../../../data/scene-loader/process/data-processor-gsplats';

function makeProcessed(splatCount = 2) {
  return {
    centers3D: new Float32Array(splatCount * 3),
    choleskyFactors3D: new Float32Array(splatCount * 6),
    amplitudes: new Float32Array(splatCount),
    colors: new Float32Array(splatCount * 3),
    splatCount,
  };
}

function makeStaged(splatCount = 5): StagedGSplatsCommit {
  return {
    path: '/g',
    sourceData: {
      positions: new Float32Array(splatCount * 3),
      amplitudes: new Float32Array(splatCount),
      choleskyFactors: new Float32Array(splatCount * 6),
      colors: null,
      splatCount,
      ndim: 3,
    },
    processed: makeProcessed(splatCount),
    cholesky01: new Float32Array(),
    cholesky23: new Float32Array(),
    cholesky45: new Float32Array(),
  };
}

// The view-version stamped onto the mesh is now an explicit commit arg (shared
// across gsplats/points/lines via stampLoadedViewVersion), not a staged field.
const V = 0;

function makeMesh(name: string): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = name;
  mesh.userData = {
    nodeType: 'gsplats',
    attrs: {},
    visibleSplatCount: 0,
  };
  return mesh;
}

describe('commitGSplatsGeometry', () => {
  it('no-ops when rootGroup is null', () => {
    expect(() => commitGSplatsGeometry(makeStaged(), null, null, undefined, V)).not.toThrow();
  });

  it('no-ops silently when mesh has gone missing', () => {
    expect(() =>
      commitGSplatsGeometry(makeStaged(), new THREE.Group(), null, undefined, V)
    ).not.toThrow();
  });

  it('writes visibleSplatCount on the mesh user-data', () => {
    mockUpdateInstancedMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(11), root, null, undefined, V);
    expect((mesh.userData as { visibleSplatCount: number }).visibleSplatCount).toBe(11);
    expect(mockUpdateInstancedMesh).toHaveBeenCalledTimes(1);
    // C7[P2][P11]: a mutant that drops the GPU update call would still pass
    // the userData write above — pin the actual dispatch. The no-pool path
    // calls updateInstancedGSplatsMesh(mesh, {centers, cholesky*, amplitudes,
    // colors, splatCount}). Assert the target mesh AND the splatCount payload.
    const [calledMesh, payload] = mockUpdateInstancedMesh.mock.calls[0] as [
      THREE.Mesh,
      { splatCount: number },
    ];
    expect(calledMesh).toBe(mesh);
    expect(payload.splatCount).toBe(11);
  });

  it('stamps loadedViewVersion (the explicit commit arg) onto the mesh user-data', () => {
    // Drives the slice-aware LOD fallback: the registry reads this stamp to tell
    // whether the level's geometry is fresh for the current view version.
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(3), root, null, undefined, 7);
    expect((mesh.userData as { loadedViewVersion: number }).loadedViewVersion).toBe(7);
  });

  // data.md G3 fix: parallel coverage to commit-points-geometry.test.ts.
  // Pin the pool-supplied dispatch path (previously untested for GSplats).
  it('accepts a buffer-pool argument without throwing and still writes visibleSplatCount', () => {
    mockUpdateInstancedMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const mockPool: any = {
      acquireGSplatsGeometry: () => ({ geometry: new THREE.BufferGeometry(), pointCount: 0 }),
      updateGSplatsGeometry: () => undefined,
      releaseGSplatsGeometry: () => undefined,
      didLastAcquireRebuildAttributes: () => false,
    };
    expect(() => commitGSplatsGeometry(makeStaged(7), root, mockPool, undefined, V)).not.toThrow();
    expect((mesh.userData as { visibleSplatCount: number }).visibleSplatCount).toBe(7);
    // Pool path must NOT fall through to the no-pool instanced-mesh update.
    expect(mockUpdateInstancedMesh).not.toHaveBeenCalled();
  });

  // data.md C7[P2][P8] three-geometry symmetry: mirror the Points and Lines
  // pool-path tests — assert acquire/update call counts AND that the
  // mesh's geometry slot was replaced with the pool-acquired one. The
  // pool branch does NOT call updateInstancedGSplatsMesh (that's the
  // ELSE/no-pool path) — pin BOTH directions: pool path SKIPS the
  // instanced-mesh update.
  it('[C7] pool path: calls acquire/update exactly once, replaces mesh.geometry, SKIPS updateInstancedGSplatsMesh', () => {
    mockUpdateInstancedMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const beforeGeom = mesh.geometry;

    const newGeometry = new THREE.BufferGeometry();
    const pool = {
      acquireGSplatsGeometry: vi.fn(() => newGeometry),
      updateGSplatsGeometry: vi.fn(),
      releaseGSplatsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => false),
    };
    commitGSplatsGeometry(makeStaged(7), root, pool as never, undefined, V);

    expect(pool.acquireGSplatsGeometry).toHaveBeenCalledTimes(1);
    expect(pool.updateGSplatsGeometry).toHaveBeenCalledTimes(1);
    expect(mesh.geometry).toBe(newGeometry);
    expect(mesh.geometry).not.toBe(beforeGeom);
    // Pool path skips updateInstancedGSplatsMesh — that's only the
    // no-pool fallback path. Pin BOTH directions of the contract.
    expect(mockUpdateInstancedMesh).not.toHaveBeenCalled();
  });
});

describe('commitGSplatsGeometry — no-op commit skip (committedData)', () => {
  it('stamps committedData with the raw source data on a real commit', () => {
    mockUpdateInstancedMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const staged = makeStaged(5);
    commitGSplatsGeometry(staged, root, null, undefined, 7);
    if (staged.noop) throw new Error('expected geometry staged commit');
    expect((mesh.userData as { committedData?: unknown }).committedData).toBe(staged.sourceData);
  });

  it('noop staged commit stamps loadedViewVersion but touches no geometry', () => {
    mockUpdateInstancedMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const geometryBefore = mesh.geometry;

    const sourceData = makeStaged(5);
    const noop: StagedGSplatsCommit = {
      path: '/g',
      noop: true,
      sourceData: sourceData.noop ? (undefined as never) : sourceData.sourceData,
    };
    commitGSplatsGeometry(noop, root, null, undefined, 9);

    // Freshness stamp written, geometry + GPU dispatch untouched.
    expect((mesh.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(9);
    expect(mesh.geometry).toBe(geometryBefore);
    expect(mockUpdateInstancedMesh).not.toHaveBeenCalled();
  });
});
