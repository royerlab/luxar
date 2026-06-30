/**
 * Direct tests for `commitLinesGeometry`, mirroring
 * `commit-points-geometry.test.ts`.
 *
 * End-to-end exercise of the surrounding processLinesData /
 * projectLinesTo3DUsingWorker still lives in
 * `data-processor-lines.test.ts`. These tests pin the commit step in
 * isolation: missing rootGroup, missing mesh, and the visibleSegmentCount
 * userData write on a real mesh.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';

// Mock the GPU update fn so the no-pool path's actual dispatch can be
// asserted (a mutant that drops this call would otherwise still pass the
// userData write). Mirrors the gsplats commit test's mock of
// updateInstancedGSplatsMesh.
const mockUpdateInstancedLinesMesh = vi.fn();
vi.mock('../../../../rendering/line-geometry', () => ({
  updateInstancedLinesMesh: (...args: unknown[]) => mockUpdateInstancedLinesMesh(...args),
}));

import { commitLinesGeometry } from '../../../../data/scene-loader/commit/commit-lines-geometry';
import type { StagedLinesCommit } from '../../../../data/scene-loader/process/data-processor-lines';
import type { ProcessedLinesData } from '../../../../types/lines';

function makeProcessed(segmentCount = 2): ProcessedLinesData {
  return {
    startPositions: new Float32Array(segmentCount * 3),
    endPositions: new Float32Array(segmentCount * 3),
    startColors: new Float32Array(segmentCount * 3),
    endColors: new Float32Array(segmentCount * 3),
    startWidths: new Float32Array(segmentCount),
    endWidths: new Float32Array(segmentCount),
    startSharpness: new Float32Array(segmentCount),
    endSharpness: new Float32Array(segmentCount),
    segmentLengths: new Float32Array(segmentCount),
    startClipped: new Uint8Array(segmentCount),
    endClipped: new Uint8Array(segmentCount),
    segmentCount,
  };
}

function makeMesh(name: string): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = name;
  mesh.userData = {
    nodeType: 'lines',
    attrs: {},
    visibleSegmentCount: 0,
  };
  return mesh;
}

describe('commitLinesGeometry', () => {
  it('no-ops when rootGroup is null', () => {
    const staged: StagedLinesCommit = { path: '/lines', processed: makeProcessed() };
    expect(() => commitLinesGeometry(staged, null, null, undefined, 0)).not.toThrow();
  });

  it('no-ops silently when the mesh has gone missing', () => {
    const root = new THREE.Group();
    const staged: StagedLinesCommit = { path: '/missing', processed: makeProcessed() };
    expect(() => commitLinesGeometry(staged, root, null, undefined, 0)).not.toThrow();
  });

  it('writes visibleSegmentCount on the mesh userData', () => {
    mockUpdateInstancedLinesMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const processed = makeProcessed(7);
    const staged: StagedLinesCommit = { path: '/lines', processed };
    commitLinesGeometry(staged, root, null, undefined, 0);
    expect(mesh.userData.visibleSegmentCount).toBe(7);
    // C6[P2][P11]: pin the actual GPU dispatch — a mutant dropping the
    // updateInstancedLinesMesh call would still pass the userData write.
    // No-pool path calls updateInstancedLinesMesh(mesh, processed).
    expect(mockUpdateInstancedLinesMesh).toHaveBeenCalledTimes(1);
    const [calledMesh, calledProcessed] = mockUpdateInstancedLinesMesh.mock.calls[0] as [
      THREE.Mesh,
      { segmentCount: number },
    ];
    expect(calledMesh).toBe(mesh);
    expect(calledProcessed).toBe(processed);
  });

  it('stamps loadedViewVersion onto the mesh user-data (three-geometry symmetry)', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const staged: StagedLinesCommit = { path: '/lines', processed: makeProcessed(4) };
    commitLinesGeometry(staged, root, null, undefined, 9);
    expect((mesh.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(9);
  });

  // data.md G3 fix: parallel coverage to commit-points-geometry.test.ts. The
  // Points test for the pool-supplied path is mirrored here so a regression
  // in the Lines pool-dispatch path is caught (previously the buffer-pool
  // branch in commitLinesGeometry had no direct coverage at all).
  it('accepts a buffer-pool argument without throwing and still writes visibleSegmentCount', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    // Minimal pool stub — commitLinesGeometry's pool-supplied path should
    // route through this object. We pin the contract that visibleSegmentCount
    // is still written regardless of pool presence.
    const mockPool: any = {
      acquireLinesGeometry: () => ({ geometry: new THREE.BufferGeometry(), pointCount: 0 }),
      updateLinesGeometry: () => undefined,
      releaseLinesGeometry: () => undefined,
      didLastAcquireRebuildAttributes: () => false,
    };
    const staged: StagedLinesCommit = { path: '/lines', processed: makeProcessed(11) };
    mockUpdateInstancedLinesMesh.mockReset();
    expect(() => commitLinesGeometry(staged, root, mockPool, undefined, 0)).not.toThrow();
    expect(mesh.userData.visibleSegmentCount).toBe(11);
    // Pool path must NOT fall through to the no-pool instanced-mesh update.
    expect(mockUpdateInstancedLinesMesh).not.toHaveBeenCalled();
  });

  // data.md C6[P2][P8] three-geometry symmetry: the Points equivalent
  // (commit-points-geometry.test.ts:75-92) asserts that
  // (a) the mesh's geometry slot was REPLACED with the pool-acquired one,
  // (b) acquireLinesGeometry was called EXACTLY once, and
  // (c) updateLinesGeometry was called EXACTLY once.
  // Pin the same contract for Lines so a mutation that swapped to the
  // dispose+create fallback (or double-invoked update) is caught.
  it('[C6] pool path: replaces mesh.geometry and calls acquire/update exactly once', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const beforeGeom = mesh.geometry;

    const newGeometry = new THREE.BufferGeometry();
    const pool = {
      acquireLinesGeometry: vi.fn(() => newGeometry),
      updateLinesGeometry: vi.fn(),
      releaseLinesGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => false),
    };
    const staged: StagedLinesCommit = { path: '/lines', processed: makeProcessed(11) };
    commitLinesGeometry(staged, root, pool as never, undefined, 0);

    expect(mesh.geometry).toBe(newGeometry);
    expect(mesh.geometry).not.toBe(beforeGeom);
    expect(pool.acquireLinesGeometry).toHaveBeenCalledTimes(1);
    expect(pool.updateLinesGeometry).toHaveBeenCalledTimes(1);
  });
});
