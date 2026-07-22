/**
 * Direct tests for `commitGSplatsGeometry`, mirroring
 * `commit-points-geometry.test.ts` and `commit-lines-geometry.test.ts`.
 *
 * Mocks `updateInstancedGSplatsMesh` so the test doesn't pull in the
 * gsplat shader / r184 instanced mesh paths; we only verify the
 * orchestration (rootGroup / mesh guards + visibleSplatCount write).
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  configureElementTextureLayout,
  resetElementTextureLayoutForTests,
} from '../../../../rendering/element-texture-layout';

const mockUpdateInstancedMesh = vi.fn();
const mockGetSplatTexture = vi.fn((..._args: unknown[]) => null);
const mockSyncGSplatMaterial = vi.fn();
const mockNoteGSplatsCommit = vi.fn();
vi.mock('../../../../rendering/material-sync-helpers', () => ({
  syncGSplatMaterialWithGeometry: (...args: unknown[]) => mockSyncGSplatMaterial(...args),
}));
vi.mock('../../../../rendering/depth-sort-coordinator', () => ({
  noteGSplatsCommit: (...args: unknown[]) => mockNoteGSplatsCommit(...args),
}));
vi.mock('../../../../rendering/gsplat-geometry', () => ({
  updateInstancedGSplatsMesh: (...args: unknown[]) => mockUpdateInstancedMesh(...args),
  packCholeskyForShader: vi.fn(),
  // material-sync-helpers reaches getSplatTexture through this module;
  // returning null makes syncGSplatMaterialWithGeometry a no-op here
  // (the sync itself is covered by its own describe below).
  getSplatTexture: (...args: unknown[]) => mockGetSplatTexture(...args),
}));

import { commitGSplatsGeometry } from '../../../../data/scene-loader/commit/commit-gsplats-geometry';
import { SOFT_DISPOSE_FLAG } from '../../../../rendering/material-manager';
import type { StagedGSplatsCommit } from '../../../../data/scene-loader/process/data-processor-gsplats';
import { getPrefixParent, setPrefixParent } from '../../../../types/prefix-lineage';

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
    // Non-pool commits rebind uSplatTex too (rebuilds swap in a fresh
    // mesh-owned texture).
    expect(mockSyncGSplatMaterial).toHaveBeenCalledWith(mesh);
    // Depth-sorting Phase 2: every non-noop commit notifies the sort
    // coordinator (generation bump + order-dependent register/sort).
    expect(mockNoteGSplatsCommit).toHaveBeenCalledWith(mesh, expect.any(Float32Array), 11);
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
  it('hands the acquired geometry to the mesh even when the pool update throws', () => {
    // Exception-window ownership handoff — see commit-points-geometry.test.ts.
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const oldGeometry = mesh.geometry;

    const newGeometry = new THREE.BufferGeometry();
    const pool = {
      acquireGSplatsGeometry: vi.fn(() => newGeometry),
      updateGSplatsGeometry: vi.fn(() => {
        throw new Error('upload failed');
      }),
      releaseGSplatsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
    };
    expect(() => commitGSplatsGeometry(makeStaged(3), root, pool as never, undefined, 0)).toThrow(
      'upload failed'
    );

    expect(mesh.geometry).toBe(newGeometry);
    expect(mesh.geometry).not.toBe(oldGeometry);
    expect((mesh.userData as { committedData?: unknown }).committedData).toBeUndefined();
    // The splat-texture rebind lives in the same finally as the
    // ownership handoff: even on a throw, the material must be
    // re-pointed at the acquired geometry's texture.
    expect(mockSyncGSplatMaterial).toHaveBeenCalledWith(mesh);
  });

  it('leaves the freshness stamps untouched when the pool update throws (success-only stamps)', () => {
    // Prime a SUCCESSFUL commit first so the stamps hold real values, then
    // make a bigger commit throw: visibleSplatCount / loadedViewVersion /
    // committedData must all still describe the last SUCCESSFUL commit —
    // a throwing write must not stamp the mesh fresh-for-the-new-view with
    // a count that never landed (the LOD freshness registry would trust it).
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);

    const geometry = new THREE.BufferGeometry();
    const pool = {
      acquireGSplatsGeometry: vi.fn(() => geometry),
      updateGSplatsGeometry: vi.fn(),
      releaseGSplatsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => false),
    };
    const first = makeStaged(2);
    commitGSplatsGeometry(first, root, pool as never, undefined, 1);
    if (first.noop) throw new Error('expected geometry staged commit');
    expect((mesh.userData as { visibleSplatCount: number }).visibleSplatCount).toBe(2);

    pool.updateGSplatsGeometry.mockImplementation(() => {
      throw new Error('upload failed');
    });
    expect(() => commitGSplatsGeometry(makeStaged(5), root, pool as never, undefined, 2)).toThrow(
      'upload failed'
    );

    expect((mesh.userData as { visibleSplatCount: number }).visibleSplatCount).toBe(2);
    expect((mesh.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(1);
    expect((mesh.userData as { committedData?: unknown }).committedData).toBe(first.sourceData);
  });

  it('disposes a replaced non-pool creation geometry at the pool handoff (placeholder leak)', () => {
    // The creation-time placeholder geometry carries a minimum-row splat
    // texture; nobody else owns it once the pool hands the node its first
    // real geometry, so the handoff must dispose it — see the
    // commit-points-geometry.test.ts twin.
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const prevGeometry = mesh.geometry;
    const disposeSpy = vi.spyOn(prevGeometry, 'dispose');

    const newGeometry = new THREE.BufferGeometry();
    newGeometry.userData = { luxarPooled: true };
    const pool = {
      acquireGSplatsGeometry: vi.fn(() => newGeometry),
      updateGSplatsGeometry: vi.fn(),
      releaseGSplatsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
    };
    commitGSplatsGeometry(makeStaged(3), root, pool as never, undefined, V);
    expect(mesh.geometry).toBe(newGeometry);
    expect(disposeSpy).toHaveBeenCalledTimes(1);
  });

  it('never disposes a replaced POOL-owned geometry (luxarPooled marker — acquire released it)', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const prevGeometry = mesh.geometry;
    prevGeometry.userData = { ...prevGeometry.userData, luxarPooled: true };
    const disposeSpy = vi.spyOn(prevGeometry, 'dispose');

    const newGeometry = new THREE.BufferGeometry();
    newGeometry.userData = { luxarPooled: true };
    const pool = {
      acquireGSplatsGeometry: vi.fn(() => newGeometry),
      updateGSplatsGeometry: vi.fn(),
      releaseGSplatsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
    };
    commitGSplatsGeometry(makeStaged(3), root, pool as never, undefined, V);
    expect(mesh.geometry).toBe(newGeometry);
    expect(disposeSpy).not.toHaveBeenCalled();
  });

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
    // Every pool commit rebinds uSplatTex (acquire may hand the node a
    // different geometry+texture pair).
    expect(mockSyncGSplatMaterial).toHaveBeenCalledWith(mesh);
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
    mockNoteGSplatsCommit.mockReset();
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
    // The sort generation must NOT bump on a stamp-only noop — an
    // in-flight sort stays valid across it (spec §5 generation contract).
    expect(mockNoteGSplatsCommit).not.toHaveBeenCalled();
  });
});

describe('commitGSplatsGeometry — committedLadderComplete stamp', () => {
  const ladderComplete = (mesh: THREE.Mesh) =>
    (mesh.userData as { committedLadderComplete?: boolean }).committedLadderComplete;

  it('stamps false while the committing progressive loader has more LODs', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    mesh.userData.loader = { hasMoreLODs: true };
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(3), root, null, undefined, V);
    expect(ladderComplete(mesh)).toBe(false);
  });

  it('stamps true on the final ladder pass (hasMoreLODs false at commit time)', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    mesh.userData.loader = { hasMoreLODs: false };
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(3), root, null, undefined, V);
    expect(ladderComplete(mesh)).toBe(true);
  });

  it('stamps true for a non-progressive loader (no hasMoreLODs getter) and for no loader at all', () => {
    const root = new THREE.Group();
    const plain = makeMesh('/g');
    plain.userData.loader = {}; // single-LOD spatial-index loader shape
    root.add(plain);
    commitGSplatsGeometry(makeStaged(3), root, null, undefined, V);
    expect(ladderComplete(plain)).toBe(true);

    const root2 = new THREE.Group();
    const loaderless = makeMesh('/g');
    root2.add(loaderless);
    commitGSplatsGeometry(makeStaged(3), root2, null, undefined, V);
    expect(ladderComplete(loaderless)).toBe(true);
  });

  it('noop (stamp-only) commit refreshes the ladder stamp too', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    mesh.userData.loader = { hasMoreLODs: true };
    mesh.userData.committedLadderComplete = true; // stale value from a previous view
    root.add(mesh);
    const staged = makeStaged(5);
    const noop: StagedGSplatsCommit = {
      path: '/g',
      noop: true,
      sourceData: staged.noop ? (undefined as never) : staged.sourceData,
    };
    commitGSplatsGeometry(noop, root, null, undefined, V);
    expect(ladderComplete(mesh)).toBe(false); // refreshed from the live loader
  });
});

describe('commitGSplatsGeometry — committedEnergyFraction stamp', () => {
  const energy = (mesh: THREE.Mesh) =>
    (mesh.userData as { committedEnergyFraction?: number }).committedEnergyFraction;

  it('stamps the progressive loader committed-energy fraction e(k) mid-ladder', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    mesh.userData.loader = { hasMoreLODs: true, committedEnergyFraction: 0.42 };
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(3), root, null, undefined, V);
    expect(energy(mesh)).toBe(0.42);
  });

  it('REMOVES the stamp on an unstamped (legacy) dataset — absence means count fallback', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    mesh.userData.loader = { hasMoreLODs: true, committedEnergyFraction: null };
    mesh.userData.committedEnergyFraction = 0.9; // stale value from a previous loader
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(3), root, null, undefined, V);
    expect(energy(mesh)).toBeUndefined();
    expect('committedEnergyFraction' in mesh.userData).toBe(false);
  });

  it('stamps 1 for a non-progressive loader (a single-set leaf IS its full content)', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    mesh.userData.loader = {}; // single-LOD spatial-index loader shape
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(3), root, null, undefined, V);
    expect(energy(mesh)).toBe(1);
  });

  it('noop (stamp-only) commit refreshes the energy stamp too', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    mesh.userData.loader = { hasMoreLODs: true, committedEnergyFraction: 0.8 };
    mesh.userData.committedEnergyFraction = 0.2; // stale value from a previous view
    root.add(mesh);
    const staged = makeStaged(5);
    const noop: StagedGSplatsCommit = {
      path: '/g',
      noop: true,
      sourceData: staged.noop ? (undefined as never) : staged.sourceData,
    };
    commitGSplatsGeometry(noop, root, null, undefined, V);
    expect(energy(mesh)).toBe(0.8); // refreshed from the live loader
  });
});

describe('commitGSplatsGeometry — capacity-clamp consistency', () => {
  afterEach(() => {
    resetElementTextureLayoutForTests();
  });

  // The GPU writers clamp the WRITTEN splats to the per-node texture bound
  // (element-texture-layout), so every count the commit records or hands out
  // must be the clamped one — an unclamped count fed to the sort
  // coordinator makes the SortWorker return permutation values ≥ the
  // texture capacity (OOB texel fetches → splats vanish).
  it('notifies the sort coordinator and stamps visibleSplatCount with the CLAMPED count', () => {
    // maxTextureSize 8 → width 8, per-node bound = 8×8/4 = 16 splats.
    configureElementTextureLayout(8);
    mockUpdateInstancedMesh.mockReset();
    mockNoteGSplatsCommit.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(100), root, null, undefined, V);

    expect((mesh.userData as { visibleSplatCount: number }).visibleSplatCount).toBe(16);
    expect(mockNoteGSplatsCommit).toHaveBeenCalledWith(mesh, expect.any(Float32Array), 16);
    // The non-pool GPU dispatch carries the clamped count too — every
    // consumer downstream of the commit sees ONE consistent count.
    const [, payload] = mockUpdateInstancedMesh.mock.calls[0] as [
      THREE.Mesh,
      { splatCount: number },
    ];
    expect(payload.splatCount).toBe(16);
  });
});

describe('commitGSplatsGeometry — preserve-ordering on same-node same-count recommits', () => {
  // The commit path decides; the writers obey the flag. These tests pin the
  // predicate (hadCommittedData && !attributesRebuilt && same geometry &&
  // same count) by asserting the options arg threaded to the writers —
  // the writers' own skip behavior is covered in splat-texture-storage.test.ts
  // and gsplat-geometry.test.ts.
  const makePool = (geometry: THREE.BufferGeometry) => ({
    acquireGSplatsGeometry: vi.fn(() => geometry),
    updateGSplatsGeometry: vi.fn(),
    releaseGSplatsGeometry: vi.fn(),
    didLastAcquireRebuildAttributes: vi.fn(() => false),
  });
  const lastPoolPreserve = (pool: ReturnType<typeof makePool>) =>
    (pool.updateGSplatsGeometry.mock.calls.at(-1) as unknown[])[4];
  const lastNonPoolPreserve = () => (mockUpdateInstancedMesh.mock.calls.at(-1) as unknown[])[2];

  it('pool path: same-count recommit → preserveOrdering true (first commit → false)', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const pool = makePool(new THREE.BufferGeometry());
    // First commit: no committedData stamp yet → the geometry's ordering
    // is unvouched-for, identity must be written.
    commitGSplatsGeometry(makeStaged(7), root, pool as never, undefined, V);
    expect(lastPoolPreserve(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
    // Same-node same-count recommit on the SAME pooled geometry: the
    // previous permutation of [0,7) is still valid — keep it. Equal count is
    // NOT an append (that needs a strict extension), so fromInstance stays 0.
    commitGSplatsGeometry(makeStaged(7), root, pool as never, undefined, V);
    expect(lastPoolPreserve(pool)).toEqual({ preserveOrdering: true, fromInstance: 0 });
  });

  it('pool path: count-change recommit → preserveOrdering false', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const pool = makePool(new THREE.BufferGeometry());
    commitGSplatsGeometry(makeStaged(7), root, pool as never, undefined, V);
    // A permutation of [0,7) is not a permutation of [0,9). No lineage was
    // stamped here, so this is a full rewrite (not an append) → fromInstance 0.
    commitGSplatsGeometry(makeStaged(9), root, pool as never, undefined, V);
    expect(lastPoolPreserve(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
  });

  it('pool path: recommit after committedData was cleared (LOD demotion) → false', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const pool = makePool(new THREE.BufferGeometry());
    commitGSplatsGeometry(makeStaged(7), root, pool as never, undefined, V);
    delete (mesh.userData as { committedData?: unknown }).committedData;
    commitGSplatsGeometry(makeStaged(7), root, pool as never, undefined, V);
    expect(lastPoolPreserve(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
  });

  it('pool path: geometry swap / attribute rebuild defeats the flag', () => {
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const pool = makePool(new THREE.BufferGeometry());
    commitGSplatsGeometry(makeStaged(7), root, pool as never, undefined, V);
    // Best-fit reuse handed the node a DIFFERENT geometry (holding some
    // other node's permutation over a different prior count) and reported
    // an attribute rebuild — identity must be written.
    pool.acquireGSplatsGeometry.mockReturnValue(new THREE.BufferGeometry());
    pool.didLastAcquireRebuildAttributes.mockReturnValue(true);
    commitGSplatsGeometry(makeStaged(7), root, pool as never, undefined, V);
    expect(lastPoolPreserve(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
  });

  it('non-pool path: same-count recommit → preserveOrdering true (first commit → false)', () => {
    mockUpdateInstancedMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(11), root, null, undefined, V);
    expect(lastNonPoolPreserve()).toEqual({ preserveOrdering: false });
    commitGSplatsGeometry(makeStaged(11), root, null, undefined, V);
    expect(lastNonPoolPreserve()).toEqual({ preserveOrdering: true });
    expect((mesh.userData as { visibleSplatCount: number }).visibleSplatCount).toBe(11);
  });

  it('non-pool path: count-change recommit → preserveOrdering false', () => {
    mockUpdateInstancedMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(11), root, null, undefined, V);
    commitGSplatsGeometry(makeStaged(5), root, null, undefined, V);
    expect(lastNonPoolPreserve()).toEqual({ preserveOrdering: false });
  });

  it('non-pool path: recommit after committedData was cleared → false', () => {
    mockUpdateInstancedMesh.mockReset();
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    commitGSplatsGeometry(makeStaged(11), root, null, undefined, V);
    delete (mesh.userData as { committedData?: unknown }).committedData;
    commitGSplatsGeometry(makeStaged(11), root, null, undefined, V);
    expect(lastNonPoolPreserve()).toEqual({ preserveOrdering: false });
  });
});

describe('commitGSplatsGeometry — append fast path (Phase 4 Stage 2, fromInstance)', () => {
  // The gate lives in the pool branch; these tests pin `fromInstance` in the
  // options threaded to updateGSplatsGeometry. The suffix-write behavior
  // itself is covered in splat-texture-storage.test.ts.
  const makePool = (geometry: THREE.BufferGeometry) => ({
    acquireGSplatsGeometry: vi.fn(() => geometry),
    updateGSplatsGeometry: vi.fn(),
    releaseGSplatsGeometry: vi.fn(),
    didLastAcquireRebuildAttributes: vi.fn(() => false),
  });
  const lastOpts = (pool: ReturnType<typeof makePool>) =>
    (pool.updateGSplatsGeometry.mock.calls.at(-1) as unknown[])[4] as {
      preserveOrdering: boolean;
      fromInstance: number;
    };

  // Arrange a committed prefix, then stage a genuine extension of it.
  const primeAndExtend = (
    root: THREE.Group,
    pool: ReturnType<typeof makePool>,
    prevCount: number,
    newCount: number
  ): StagedGSplatsCommit => {
    commitGSplatsGeometry(makeStaged(prevCount), root, pool as never, undefined, V);
    const committed = (root.children[0].userData as { committedData: object }).committedData;
    const next = makeStaged(newCount);
    setPrefixParent(next.sourceData, committed); // forward-chain lineage
    return next;
  };

  it('fires the append (fromInstance = prevCount) when the commit extends the committed prefix', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    commitGSplatsGeometry(next, root, pool as never, undefined, V);
    expect(lastOpts(pool).fromInstance).toBe(4);
    // Positive-path bookkeeping stamps re-enable the NEXT append.
    const ud = root.children[0].userData as { gpuPrefixIntact: boolean; committedTruncate: number };
    expect(ud.gpuPrefixIntact).toBe(true);
    expect(ud.committedTruncate).toBe(3.0);
    // Consume-and-clear: the gate consumed the lineage entry, unpinning the
    // parent concat (prefix-lineage.ts retention contract).
    expect(getPrefixParent(next.sourceData)).toBeUndefined();
  });

  it('does NOT append (fromInstance 0) when there is no prefix lineage (unrelated reload)', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const pool = makePool(new THREE.BufferGeometry());
    commitGSplatsGeometry(makeStaged(4), root, pool as never, undefined, V);
    // A larger commit with NO lineage stamp: full rewrite.
    commitGSplatsGeometry(makeStaged(6), root, pool as never, undefined, V);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append after a context restore cleared gpuPrefixIntact', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    (root.children[0].userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact = false;
    commitGSplatsGeometry(next, root, pool as never, undefined, V);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append when the acquire rebuilt attributes (pool grow / best-fit swap)', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    // Grow handed back a different geometry with rebuilt attributes.
    pool.acquireGSplatsGeometry.mockReturnValue(new THREE.BufferGeometry());
    pool.didLastAcquireRebuildAttributes.mockReturnValue(true);
    commitGSplatsGeometry(next, root, pool as never, undefined, V);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append on an equal-count recommit (that is the preserveOrdering path)', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 5, 5); // same count, lineage set
    commitGSplatsGeometry(next, root, pool as never, undefined, V);
    const opts = lastOpts(pool);
    expect(opts.fromInstance).toBe(0);
    expect(opts.preserveOrdering).toBe(true);
  });

  it('does NOT append when the truncate uniform changed since the committed prefix', () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6); // committedTruncate stamped 3.0
    // A truncate change is invisible to the view state / lineage — give the
    // mesh a material with a different uTruncate so readTruncate diverges.
    (root.children[0] as THREE.Mesh).material = {
      uniforms: { uTruncate: { value: 5.0 } },
    } as never;
    commitGSplatsGeometry(next, root, pool as never, undefined, V);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });
});

describe('commitGSplatsGeometry — RenderObject invalidation on non-pool rebuild', () => {
  // When updateInstancedGSplatsMesh reports a REBUILD (size change rebinds
  // a fresh InstancedInterleavedBuffer), the commit must dispatch the
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
    mockUpdateInstancedMesh.mockReset();
    mockUpdateInstancedMesh.mockReturnValue(true);
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const saw = softDisposeSeen(mesh);
    commitGSplatsGeometry(makeStaged(3), root, null, undefined, V);
    expect(saw()).toBe(true);
  });

  it('does NOT dispatch when the update was in-place (no rebuild)', () => {
    mockUpdateInstancedMesh.mockReset();
    mockUpdateInstancedMesh.mockReturnValue(false);
    const root = new THREE.Group();
    const mesh = makeMesh('/g');
    root.add(mesh);
    const saw = softDisposeSeen(mesh);
    commitGSplatsGeometry(makeStaged(3), root, null, undefined, V);
    expect(saw()).toBe(false);
  });
});
