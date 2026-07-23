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

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';

const mockNoteDepthSortCommit = vi.fn();
vi.mock('../../../../rendering/depth-sort-coordinator', () => ({
  noteDepthSortCommit: (...args: unknown[]) => mockNoteDepthSortCommit(...args),
}));

// Mock the GPU update fn so the no-pool path's actual dispatch can be
// asserted (a mutant that drops this call would otherwise still pass the
// userData write). Mirrors the gsplats commit test's mock of
// updateInstancedGSplatsMesh. `getLineTexture` must ride along: the
// commit's syncLineMaterialWithGeometry (material-sync-helpers.ts) reads
// it from this same module — null keeps the sync a no-op on the stub
// geometries used here.
const mockUpdateInstancedLinesMesh = vi.fn();
vi.mock('../../../../rendering/line-geometry', () => ({
  updateInstancedLinesMesh: (...args: unknown[]) => mockUpdateInstancedLinesMesh(...args),
  getLineTexture: () => null,
}));

import { commitLinesGeometry } from '../../../../data/scene-loader/commit/commit-lines-geometry';
import {
  configureElementTextureLayout,
  resetElementTextureLayoutForTests,
} from '../../../../rendering/element-texture-layout';
import { getPrefixParent, setPrefixParent } from '../../../../types/prefix-lineage';
import { SOFT_DISPOSE_FLAG } from '../../../../rendering/material-manager';
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

function makeSourceData(segmentCount = 2) {
  return {
    positions: new Float32Array(segmentCount * 2 * 3),
    segments: new Uint32Array(segmentCount * 2),
    widths: new Float32Array(segmentCount * 2),
    colors: null,
    sharpness: null,
    segmentCount,
    vertexCount: segmentCount * 2,
    ndim: 3,
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
    const staged: StagedLinesCommit = {
      path: '/lines',
      sourceData: makeSourceData(),
      processed: makeProcessed(),
    };
    expect(() => commitLinesGeometry(staged, null, null, undefined, 0)).not.toThrow();
  });

  it('no-ops silently when the mesh has gone missing', () => {
    const root = new THREE.Group();
    const staged: StagedLinesCommit = {
      path: '/missing',
      sourceData: makeSourceData(),
      processed: makeProcessed(),
    };
    expect(() => commitLinesGeometry(staged, root, null, undefined, 0)).not.toThrow();
  });

  it('writes visibleSegmentCount on the mesh userData', () => {
    mockUpdateInstancedLinesMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const processed = makeProcessed(7);
    const staged: StagedLinesCommit = { path: '/lines', sourceData: makeSourceData(), processed };
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
    const staged: StagedLinesCommit = {
      path: '/lines',
      sourceData: makeSourceData(4),
      processed: makeProcessed(4),
    };
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
      acquireLinesGeometry: () => new THREE.BufferGeometry(),
      updateLinesGeometry: () => undefined,
      releaseLinesGeometry: () => undefined,
      didLastAcquireRebuildAttributes: () => false,
    };
    const staged: StagedLinesCommit = {
      path: '/lines',
      sourceData: makeSourceData(11),
      processed: makeProcessed(11),
    };
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
  it('hands the acquired geometry to the mesh even when the pool update throws', () => {
    // Exception-window ownership handoff — see commit-points-geometry.test.ts.
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const oldGeometry = mesh.geometry;

    const newGeometry = new THREE.BufferGeometry();
    const pool = {
      acquireLinesGeometry: vi.fn(() => newGeometry),
      updateLinesGeometry: vi.fn(() => {
        throw new Error('upload failed');
      }),
      releaseLinesGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
    };
    const staged: StagedLinesCommit = {
      path: '/lines',
      sourceData: makeSourceData(3),
      processed: makeProcessed(3),
    };
    expect(() => commitLinesGeometry(staged, root, pool as never, undefined, 0)).toThrow(
      'upload failed'
    );

    expect(mesh.geometry).toBe(newGeometry);
    expect(mesh.geometry).not.toBe(oldGeometry);
    expect((mesh.userData as { committedData?: unknown }).committedData).toBeUndefined();
  });

  it('clears gpuPrefixIntact when the pool update throws (append-safety defense-in-depth)', () => {
    // Prime a SUCCESSFUL commit (stamps the flag true), then make a
    // bigger commit throw: the write left the buffer content unproven,
    // so the catch must clear the append-safety flag — the next commit
    // full-rewrites even if a future loader/cache change re-stamps
    // prefix lineage. Pins the commit's `catch { gpuPrefixIntact =
    // false; throw }` block (the lineage consume-and-clear masks it
    // behaviorally today, so nothing else fails when it's removed).
    // Twin of the points/gsplats tests of the same name.
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);

    const geometry = new THREE.BufferGeometry();
    const pool = {
      acquireLinesGeometry: vi.fn(() => geometry),
      updateLinesGeometry: vi.fn(),
      releaseLinesGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => false),
    };
    const first: StagedLinesCommit = {
      path: '/lines',
      sourceData: makeSourceData(2),
      processed: makeProcessed(2),
    };
    commitLinesGeometry(first, root, pool as never, undefined, 1);
    expect((mesh.userData as { gpuPrefixIntact?: boolean }).gpuPrefixIntact).toBe(true);

    pool.updateLinesGeometry.mockImplementation(() => {
      throw new Error('upload failed');
    });
    const second: StagedLinesCommit = {
      path: '/lines',
      sourceData: makeSourceData(5),
      processed: makeProcessed(5),
    };
    expect(() => commitLinesGeometry(second, root, pool as never, undefined, 2)).toThrow(
      'upload failed'
    );

    expect((mesh.userData as { gpuPrefixIntact?: boolean }).gpuPrefixIntact).toBe(false);
  });

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
    const staged: StagedLinesCommit = {
      path: '/lines',
      sourceData: makeSourceData(11),
      processed: makeProcessed(11),
    };
    commitLinesGeometry(staged, root, pool as never, undefined, 0);

    expect(mesh.geometry).toBe(newGeometry);
    expect(mesh.geometry).not.toBe(beforeGeom);
    expect(pool.acquireLinesGeometry).toHaveBeenCalledTimes(1);
    expect(pool.updateLinesGeometry).toHaveBeenCalledTimes(1);
  });
});

describe('commitLinesGeometry — append fast path (Phase 4 Stage 2, fromInstance)', () => {
  // The gate lives in the pool branch; these tests pin `fromInstance` in the
  // options threaded to updateLinesGeometry. The suffix-write behavior
  // itself is covered in line-texture-storage.test.ts.
  const makePool = (geometry: THREE.BufferGeometry) => ({
    acquireLinesGeometry: vi.fn(() => geometry),
    updateLinesGeometry: vi.fn(),
    releaseLinesGeometry: vi.fn(),
    didLastAcquireRebuildAttributes: vi.fn(() => false),
  });
  const lastOpts = (pool: ReturnType<typeof makePool>) =>
    (pool.updateLinesGeometry.mock.calls.at(-1) as unknown[])[3] as {
      fromInstance: number;
    };
  const makeStaged = (segmentCount: number): StagedLinesCommit => ({
    path: '/lines',
    sourceData: makeSourceData(segmentCount),
    processed: makeProcessed(segmentCount),
  });

  // Arrange a committed prefix, then stage a genuine extension of it.
  const primeAndExtend = (
    root: THREE.Group,
    pool: ReturnType<typeof makePool>,
    prevCount: number,
    newCount: number
  ): StagedLinesCommit => {
    commitLinesGeometry(makeStaged(prevCount), root, pool as never, undefined, 0);
    const committed = (root.children[0].userData as { committedData: object }).committedData;
    const next = makeStaged(newCount);
    setPrefixParent(next.sourceData, committed); // forward-chain lineage
    return next;
  };

  it('fires the append (fromInstance = prevCount) when the commit extends the committed prefix', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    commitLinesGeometry(next, root, pool as never, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(4);
    // Positive-path bookkeeping stamp re-enables the NEXT append.
    expect((root.children[0].userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact).toBe(true);
    // Consume-and-clear: the gate consumed the lineage entry, unpinning the
    // parent concat (prefix-lineage.ts retention contract).
    expect(getPrefixParent(next.sourceData)).toBeUndefined();
  });

  it('does NOT append (fromInstance 0) when there is no prefix lineage (unrelated reload)', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    commitLinesGeometry(makeStaged(4), root, pool as never, undefined, 0);
    // A larger commit with NO lineage stamp: full rewrite.
    commitLinesGeometry(makeStaged(6), root, pool as never, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append after a context restore cleared gpuPrefixIntact', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    (root.children[0].userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact = false;
    commitLinesGeometry(next, root, pool as never, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append when the acquire rebuilt attributes (pool grow / best-fit swap)', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    // Grow handed back a different geometry with rebuilt attributes.
    pool.acquireLinesGeometry.mockReturnValue(new THREE.BufferGeometry());
    pool.didLastAcquireRebuildAttributes.mockReturnValue(true);
    commitLinesGeometry(next, root, pool as never, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append on an equal-count or shrinking recommit', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    const same = primeAndExtend(root, pool, 5, 5); // same count, lineage set
    commitLinesGeometry(same, root, pool as never, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
    const shrunk = makeStaged(3);
    setPrefixParent(
      shrunk.sourceData,
      (root.children[0].userData as { committedData: object }).committedData
    );
    commitLinesGeometry(shrunk, root, pool as never, undefined, 2);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append when an optional field flips presence vs the committed parent', () => {
    // A presence flip (here: the new level introduces colors) re-fills the
    // prefix through the interpolation kernel, which need not be bit-exact
    // with the constant white the prefix was committed with — full rewrite.
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6); // committed sourceData.colors = null
    (next.sourceData as { colors: Float32Array | null }).colors = new Float32Array(6 * 2 * 3);
    commitLinesGeometry(next, root, pool as never, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });
});

describe('commitLinesGeometry — no-op commit skip (committedData)', () => {
  it('stamps committedData with the raw source data on a real commit', () => {
    mockUpdateInstancedLinesMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const staged: StagedLinesCommit = {
      path: '/lines',
      sourceData: makeSourceData(3),
      processed: makeProcessed(3),
    };
    commitLinesGeometry(staged, root, null, undefined, 4);
    if (staged.noop) throw new Error('expected geometry staged commit');
    expect((mesh.userData as { committedData?: unknown }).committedData).toBe(staged.sourceData);
  });

  it('noop staged commit stamps loadedViewVersion but touches no geometry', () => {
    mockUpdateInstancedLinesMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const geometryBefore = mesh.geometry;

    const noop: StagedLinesCommit = {
      path: '/lines',
      noop: true,
      sourceData: makeSourceData(3),
    };
    commitLinesGeometry(noop, root, null, undefined, 9);

    expect((mesh.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(9);
    expect(mesh.geometry).toBe(geometryBefore);
    expect(mockUpdateInstancedLinesMesh).not.toHaveBeenCalled();
  });
});

describe('commitLinesGeometry — committedLadderComplete stamp', () => {
  const ladderComplete = (mesh: THREE.Mesh) =>
    (mesh.userData as { committedLadderComplete?: boolean }).committedLadderComplete;

  it('stamps false while the committing progressive loader has more LODs', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    mesh.userData.loader = { hasMoreLODs: true };
    root.add(mesh);
    commitLinesGeometry(
      { path: '/lines', sourceData: makeSourceData(3), processed: makeProcessed(3) },
      root,
      null,
      undefined,
      0
    );
    expect(ladderComplete(mesh)).toBe(false);
  });

  it('stamps true on the final ladder pass and for non-progressive / loaderless meshes', () => {
    for (const loader of [{ hasMoreLODs: false }, {}, undefined]) {
      const root = new THREE.Group();
      const mesh = makeMesh('/lines');
      if (loader) mesh.userData.loader = loader;
      root.add(mesh);
      commitLinesGeometry(
        { path: '/lines', sourceData: makeSourceData(2), processed: makeProcessed(2) },
        root,
        null,
        undefined,
        0
      );
      expect(ladderComplete(mesh)).toBe(true);
    }
  });

  it('noop (stamp-only) commit refreshes the ladder stamp too', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    mesh.userData.loader = { hasMoreLODs: true };
    mesh.userData.committedLadderComplete = true; // stale value from a previous view
    root.add(mesh);
    const noop: StagedLinesCommit = { path: '/lines', noop: true, sourceData: makeSourceData(2) };
    commitLinesGeometry(noop, root, null, undefined, 0);
    expect(ladderComplete(mesh)).toBe(false); // refreshed from the live loader
  });
});

describe('commitLinesGeometry — committedEnergyFraction stamp', () => {
  const energy = (mesh: THREE.Mesh) =>
    (mesh.userData as { committedEnergyFraction?: number }).committedEnergyFraction;

  it('stamps the progressive loader committed-energy fraction e(k) mid-ladder', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    mesh.userData.loader = { hasMoreLODs: true, committedEnergyFraction: 0.42 };
    root.add(mesh);
    commitLinesGeometry(
      { path: '/lines', sourceData: makeSourceData(3), processed: makeProcessed(3) },
      root,
      null,
      undefined,
      0
    );
    expect(energy(mesh)).toBe(0.42);
  });

  it('REMOVES the stamp on an unstamped (legacy) dataset; stamps 1 for non-progressive', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    mesh.userData.loader = { hasMoreLODs: true, committedEnergyFraction: null };
    mesh.userData.committedEnergyFraction = 0.9; // stale
    root.add(mesh);
    commitLinesGeometry(
      { path: '/lines', sourceData: makeSourceData(3), processed: makeProcessed(3) },
      root,
      null,
      undefined,
      0
    );
    expect(energy(mesh)).toBeUndefined();

    const root2 = new THREE.Group();
    const plain = makeMesh('/lines');
    plain.userData.loader = {};
    root2.add(plain);
    commitLinesGeometry(
      { path: '/lines', sourceData: makeSourceData(2), processed: makeProcessed(2) },
      root2,
      null,
      undefined,
      0
    );
    expect(energy(plain)).toBe(1);
  });
});

describe('commitLinesGeometry — RenderObject invalidation on non-pool rebuild', () => {
  // When updateInstancedLinesMesh reports a REBUILD (a COUNT change builds
  // a fresh exact-size geometry+texture pair — scalar toggles no longer
  // rebuild under the fixed texel layout), the commit must dispatch the
  // SOFT_DISPOSE-flagged material event so Three's cached RenderObject
  // (stale `vertexBuffers` on the WebGPU backend) is evicted — the same
  // contract the pool branch honors via didLastAcquireRebuildAttributes.
  const softDisposeSeen = (mesh: THREE.Mesh): (() => boolean) => {
    let seen = false;
    (mesh.material as THREE.Material).addEventListener('dispose', () => {
      seen = (mesh.material as unknown as Record<symbol, boolean>)[SOFT_DISPOSE_FLAG] === true;
    });
    return () => seen;
  };

  it('soft-disposes the material when the update reports a rebuild', () => {
    mockUpdateInstancedLinesMesh.mockReset();
    mockUpdateInstancedLinesMesh.mockReturnValue(true);
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const saw = softDisposeSeen(mesh);
    commitLinesGeometry(
      { path: '/lines', sourceData: makeSourceData(3), processed: makeProcessed(3) },
      root,
      null,
      undefined,
      0
    );
    expect(saw()).toBe(true);
  });

  it('does NOT dispatch when the update was in-place (no rebuild)', () => {
    mockUpdateInstancedLinesMesh.mockReset();
    mockUpdateInstancedLinesMesh.mockReturnValue(false);
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const saw = softDisposeSeen(mesh);
    commitLinesGeometry(
      { path: '/lines', sourceData: makeSourceData(3), processed: makeProcessed(3) },
      root,
      null,
      undefined,
      0
    );
    expect(saw()).toBe(false);
  });
});

describe('commitLinesGeometry — depth-sort integration (lines sort registration)', () => {
  beforeEach(() => {
    mockNoteDepthSortCommit.mockReset();
    mockUpdateInstancedLinesMesh.mockReset();
  });

  const makePool = (geometry: THREE.BufferGeometry) => ({
    acquireLinesGeometry: vi.fn(() => geometry),
    updateLinesGeometry: vi.fn(),
    releaseLinesGeometry: vi.fn(),
    didLastAcquireRebuildAttributes: vi.fn(() => false),
  });
  const makeStaged = (segmentCount: number): StagedLinesCommit => ({
    path: '/lines',
    sourceData: makeSourceData(segmentCount),
    processed: makeProcessed(segmentCount),
  });

  it('pool path: notifies the coordinator with a LAZY midpoints provider and the count', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const staged = makeStaged(2);
    if (staged.noop) throw new Error('expected geometry staged commit');
    staged.processed.startPositions.set([0, 2, 4, 10, 20, 30]);
    staged.processed.endPositions.set([2, 4, 6, 30, 40, 50]);
    const pool = makePool(new THREE.BufferGeometry());
    commitLinesGeometry(staged, root, pool as never, undefined, 0);

    expect(mockNoteDepthSortCommit).toHaveBeenCalledTimes(1);
    const [calledMesh, provider, count] = mockNoteDepthSortCommit.mock.calls[0] as [
      THREE.Mesh,
      () => Float32Array,
      number,
    ];
    expect(calledMesh).toBe(mesh);
    expect(count).toBe(2);
    // Lines pass a THUNK (deferring the O(N) midpoint copy to the sorted
    // path). Segment sort keys are the endpoint midpoints (start+end)/2.
    expect(typeof provider).toBe('function');
    const centers = provider();
    expect(centers).toBeInstanceOf(Float32Array);
    expect(Array.from(centers)).toEqual([1, 3, 5, 20, 30, 40]);
    // FRESH allocation, not a view: the coordinator transfers the returned
    // buffer to the SortWorker; sharing the processed arrays' ArrayBuffer
    // would detach the texel source with it.
    expect(centers.buffer).not.toBe(staged.processed.startPositions.buffer);
    expect(centers.buffer).not.toBe(staged.processed.endPositions.buffer);
  });

  it('non-pool path: notifies too (both commit branches register)', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    commitLinesGeometry(makeStaged(4), root, null, undefined, 0);
    expect(mockNoteDepthSortCommit).toHaveBeenCalledTimes(1);
    expect(mockNoteDepthSortCommit.mock.calls[0][0]).toBe(mesh);
    expect(mockNoteDepthSortCommit.mock.calls[0][2]).toBe(4);
  });

  it('is success-only: a throwing GPU write must NOT bump the sort generation', () => {
    // Mirrors the points/gsplats ordering (noteDepthSortCommit after the
    // write block): the texture holds partially-written data, committedData
    // was not stamped, and the next commit full-rewrites + registers.
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    pool.updateLinesGeometry.mockImplementation(() => {
      throw new Error('device lost');
    });
    expect(() => commitLinesGeometry(makeStaged(3), root, pool as never, undefined, 0)).toThrow(
      'device lost'
    );
    expect(mockNoteDepthSortCommit).not.toHaveBeenCalled();
  });

  it('is skipped on the staged.noop path (stamp-only — in-flight sorts stay valid)', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const noop: StagedLinesCommit = { path: '/lines', noop: true, sourceData: makeSourceData(3) };
    commitLinesGeometry(noop, root, null, undefined, 1);
    expect(mockNoteDepthSortCommit).not.toHaveBeenCalled();
  });

  it('reports count 0 on an empty commit (coordinator release path)', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    commitLinesGeometry(makeStaged(0), root, null, undefined, 0);
    expect(mockNoteDepthSortCommit).toHaveBeenCalledTimes(1);
    expect(mockNoteDepthSortCommit.mock.calls[0][2]).toBe(0);
  });

  describe('capacity-clamp consistency', () => {
    afterEach(() => {
      resetElementTextureLayoutForTests();
    });

    it('notifies the coordinator with the CLAMPED count and a same-length provider', () => {
      // maxTextureSize 6 → width 6 (multiple of texelsPerElement 6),
      // per-node bound = 6×6/6 = 6 segments. An unclamped count would make
      // the SortWorker return permutation values ≥ the texture capacity
      // (OOB texel fetches → segments vanish).
      configureElementTextureLayout(6);
      const root = new THREE.Group();
      const mesh = makeMesh('/lines');
      root.add(mesh);
      const pool = makePool(new THREE.BufferGeometry());
      commitLinesGeometry(makeStaged(100), root, pool as never, undefined, 0);

      expect((mesh.userData as { visibleSegmentCount: number }).visibleSegmentCount).toBe(6);
      const [, provider, count] = mockNoteDepthSortCommit.mock.calls[0] as [
        THREE.Mesh,
        () => Float32Array,
        number,
      ];
      expect(count).toBe(6);
      expect((provider() as Float32Array).length).toBe(6 * 3);
    });
  });
});

describe('commitLinesGeometry — preserve-ordering on same-node same-count recommits', () => {
  // The commit path decides; the writer obeys the flag (its skip behavior
  // is covered in line-texture-storage tests). These pin the predicate
  // (hadCommittedData && !attributesRebuilt && same geometry && same count)
  // by asserting the options arg threaded to updateLinesGeometry —
  // mirroring the points/gsplats twins.
  const makePool = (geometry: THREE.BufferGeometry) => ({
    acquireLinesGeometry: vi.fn(() => geometry),
    updateLinesGeometry: vi.fn(),
    releaseLinesGeometry: vi.fn(),
    didLastAcquireRebuildAttributes: vi.fn(() => false),
  });
  const lastOpts = (pool: ReturnType<typeof makePool>) =>
    (pool.updateLinesGeometry.mock.calls.at(-1) as unknown[])[3];
  const makeStaged = (segmentCount: number): StagedLinesCommit => ({
    path: '/lines',
    sourceData: makeSourceData(segmentCount),
    processed: makeProcessed(segmentCount),
  });

  it('same-count recommit → preserveOrdering true (first commit → false)', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    // First commit: no committedData stamp yet → the geometry's ordering
    // is unvouched-for, identity must be written.
    commitLinesGeometry(makeStaged(7), root, pool as never, undefined, 0);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
    // Same-node same-count recommit on the SAME pooled geometry: the
    // previous permutation of [0,7) is still valid — keep it as a
    // no-worse prior until the commit-triggered re-sort lands. Equal
    // count is NOT an append, so fromInstance stays 0.
    commitLinesGeometry(makeStaged(7), root, pool as never, undefined, 1);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: true, fromInstance: 0 });
  });

  it('count-change recommit → preserveOrdering false', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    commitLinesGeometry(makeStaged(7), root, pool as never, undefined, 0);
    // A permutation of [0,7) is not a permutation of [0,9).
    commitLinesGeometry(makeStaged(9), root, pool as never, undefined, 1);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
  });

  it('recommit after committedData was cleared (LOD demotion) → false', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/lines');
    root.add(mesh);
    const pool = makePool(new THREE.BufferGeometry());
    commitLinesGeometry(makeStaged(7), root, pool as never, undefined, 0);
    delete (mesh.userData as { committedData?: unknown }).committedData;
    commitLinesGeometry(makeStaged(7), root, pool as never, undefined, 1);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
  });

  it('geometry swap / attribute rebuild defeats the flag', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const pool = makePool(new THREE.BufferGeometry());
    commitLinesGeometry(makeStaged(7), root, pool as never, undefined, 0);
    // Best-fit reuse handed the node a DIFFERENT geometry (holding some
    // other node's permutation over a different prior count) and reported
    // an attribute rebuild — identity must be written.
    pool.acquireLinesGeometry.mockReturnValue(new THREE.BufferGeometry());
    pool.didLastAcquireRebuildAttributes.mockReturnValue(true);
    commitLinesGeometry(makeStaged(7), root, pool as never, undefined, 1);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
  });
});
