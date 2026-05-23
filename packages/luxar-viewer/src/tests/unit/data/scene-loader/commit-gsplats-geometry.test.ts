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
    processed: makeProcessed(splatCount),
    cholesky01: new Float32Array(),
    cholesky23: new Float32Array(),
    cholesky45: new Float32Array(),
  };
}

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
    expect(() => commitGSplatsGeometry(makeStaged(), null, null)).not.toThrow();
  });

  it('no-ops silently when mesh has gone missing', () => {
    expect(() => commitGSplatsGeometry(makeStaged(), new THREE.Group(), null)).not.toThrow();
  });

  it('writes visibleSplatCount on the mesh user-data', () => {
    mockUpdateInstancedMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(11), root, null);
    expect((mesh.userData as { visibleSplatCount: number }).visibleSplatCount).toBe(11);
    expect(mockUpdateInstancedMesh).toHaveBeenCalledTimes(1);
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
    expect(() => commitGSplatsGeometry(makeStaged(7), root, mockPool)).not.toThrow();
    expect((mesh.userData as { visibleSplatCount: number }).visibleSplatCount).toBe(7);
  });
});
