/**
 * Unit tests for the points geometry-commit handler.
 *
 * Strategy: real `THREE.Group` + `THREE.Points` so the
 * getObjectByName lookup is exercised; mocked NodeFactory and
 * GPUBufferPool stubs to detect which code path ran.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';

const mockNoteDepthSortCommit = vi.fn();
vi.mock('../../../../rendering/depth-sort-coordinator', () => ({
  noteDepthSortCommit: (...args: unknown[]) => mockNoteDepthSortCommit(...args),
}));

import { commitPointsGeometry } from '../../../../data/scene-loader/commit/commit-points-geometry';
import {
  configureElementTextureLayout,
  resetElementTextureLayoutForTests,
} from '../../../../rendering/element-texture-layout';
import { createPointsGeometry } from '../../../../rendering/node-factory/create-points-node';
import { getPointTexture } from '../../../../rendering/point-geometry';
import { getPrefixParent, setPrefixParent } from '../../../../types/prefix-lineage';
import { SOFT_DISPOSE_FLAG } from '../../../../rendering/material-manager';
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

// The REAL geometry factory — regression tests below must exercise the
// commit path against production-shaped geometry (plain
// InstancedBufferAttributes), which the mocked factory masked.
const realNodeFactory = {
  createPointsGeometry: (data: LoadedPointsData, maxRadius?: number) =>
    createPointsGeometry(data, maxRadius),
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

  it('disposes a replaced non-pool creation geometry at the pool handoff (placeholder leak)', () => {
    // The creation-time placeholder geometry (createPointsNode) carries a
    // minimum-row element texture; nobody else owns it once the pool hands
    // the node its first real geometry, so the handoff must dispose it.
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const prevGeometry = points.geometry;
    const disposeSpy = vi.spyOn(prevGeometry, 'dispose');

    const newGeometry = new THREE.BufferGeometry();
    newGeometry.userData = { luxarPooled: true };
    const gpuBufferPool = {
      acquirePointsGeometry: vi.fn(() => newGeometry),
      updatePointsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
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
    expect(points.geometry).toBe(newGeometry);
    expect(disposeSpy).toHaveBeenCalledTimes(1);
  });

  it('never disposes a replaced POOL-owned geometry (luxarPooled marker — acquire released it)', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const prevGeometry = points.geometry;
    prevGeometry.userData = { ...prevGeometry.userData, luxarPooled: true };
    const disposeSpy = vi.spyOn(prevGeometry, 'dispose');

    const newGeometry = new THREE.BufferGeometry();
    newGeometry.userData = { luxarPooled: true };
    const gpuBufferPool = {
      acquirePointsGeometry: vi.fn(() => newGeometry),
      updatePointsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
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
    expect(points.geometry).toBe(newGeometry);
    expect(disposeSpy).not.toHaveBeenCalled();
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

  it('recreates the geometry when pool disabled and counts match (always-recreate contract)', () => {
    // The non-pool path recreates unconditionally — the historical
    // same-count in-place branch assumed interleaved attributes, but
    // createPointsGeometry binds plain InstancedBufferAttributes, so
    // the branch threw on the 2nd same-count commit. Recreation via
    // NodeFactory owns all the dtype logic.
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory, undefined, 0);
    const firstGeometry = points.geometry;
    const disposeSpy = vi.spyOn(firstGeometry, 'dispose');
    mockCreatePointsGeometry.mockClear();

    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory, undefined, 1);
    expect(mockCreatePointsGeometry).toHaveBeenCalledTimes(1);
    expect(disposeSpy).toHaveBeenCalledTimes(1);
    expect(points.geometry).not.toBe(firstGeometry);
    expect((points.userData as { visiblePointCount: number }).visiblePointCount).toBe(3);
  });

  it('bakes the radius footprint into boundingBox (three-geometry invariant)', () => {
    // Symmetry with lines/gsplats: the committed boundingBox must include
    // the rendered disc footprint, not just the centers — so the pick cull
    // (ray-aabb) and camera framing cover large radii. Here centers bounds
    // are [-1,1] and max_radius is 10 → boundingBox grows to [-11,11].
    // Uses the REAL createPointsGeometry: the recreate path delegates the
    // footprint expansion to the factory (which receives maxRadius).
    const root = new THREE.Group();
    const points = makePoints('/p');
    points.userData.attrs = { max_radius: 10 };
    root.add(points);

    commitPointsGeometry(
      '/p',
      makeData(3, /*withRadii=*/ true),
      root,
      null,
      realNodeFactory,
      undefined,
      0
    );

    expect(points.geometry.boundingBox).not.toBeNull();
    expect(points.geometry.boundingBox!.min.x).toBeCloseTo(-11, 5);
    expect(points.geometry.boundingBox!.max.x).toBeCloseTo(11, 5);
  });
});

describe('commitPointsGeometry — non-pool path against REAL factory geometry', () => {
  // Regression suite for the broken useGPUBufferPool:false fallback. The
  // mocked-factory tests above masked this: production non-pool geometry
  // is built by createPointsGeometry with plain InstancedBufferAttributes,
  // but the historical same-count in-place branch cast `.data` to an
  // interleaved buffer (undefined) and threw on the SECOND same-count
  // commit — the routine case while scrubbing a dimension whose visible
  // count is constant.

  it('a second same-count commit does not throw and uploads the new positions', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);

    commitPointsGeometry('/p', makeData(3), root, null, realNodeFactory, undefined, 0);

    const second = makeData(3); // fresh reference (no no-op skip), same count
    second.positions[0] = 7;
    expect(() =>
      commitPointsGeometry('/p', second, root, null, realNodeFactory, undefined, 1)
    ).not.toThrow();

    // Per-point data lives in the point texture: center.x is float 0 of
    // texel 0 (see point-geometry.ts).
    const texels = getPointTexture(points.geometry)!.image.data as Float32Array;
    expect(texels[0]).toBe(7);
    expect((points.userData as { visiblePointCount: number }).visiblePointCount).toBe(3);
  });

  it('Uint16 colors keep integer-normalized (÷65535) semantics across a same-count commit', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);

    const first = makeData(3);
    first.colors = new Uint16Array(9).fill(65535);
    commitPointsGeometry('/p', first, root, null, realNodeFactory, undefined, 0);

    const second = makeData(3);
    second.colors = new Uint16Array(9).fill(65535);
    second.positions[0] = 1;
    commitPointsGeometry('/p', second, root, null, realNodeFactory, undefined, 1);

    // Uint16 sources widen into the texture with the ÷65535 divisor, so
    // 65535 lands as exactly 1.0 in texel 1's color slots. A ÷255
    // regression would land 257.0 instead (colors 257× too bright).
    const texels = getPointTexture(points.geometry)!.image.data as Float32Array;
    expect(texels[4]).toBeCloseTo(1.0, 6); // color.r of point 0
    expect(texels[5]).toBeCloseTo(1.0, 6);
    expect(texels[6]).toBeCloseTo(1.0, 6);
  });

  it('evicts Three’s cached RenderObject via a soft material dispose on every non-pool commit', () => {
    // The recreate path rebinds fresh GPU buffers; on the WebGPU backend
    // the mesh's cached RenderObject keeps a stale `vertexBuffers` set
    // unless the commit dispatches the SOFT_DISPOSE-flagged event (same
    // contract as the pool path's attributesRebuilt branch).
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);

    let sawSoftDispose = false;
    (points.material as THREE.Material).addEventListener('dispose', () => {
      sawSoftDispose =
        (points.material as unknown as Record<symbol, boolean>)[SOFT_DISPOSE_FLAG] === true;
    });

    commitPointsGeometry('/p', makeData(3), root, null, realNodeFactory, undefined, 0);
    expect(sawSoftDispose).toBe(true);
  });
});

describe('commitPointsGeometry — exception-window ownership handoff', () => {
  it('hands the acquired geometry to the mesh even when the pool update throws', () => {
    // Grow / spec-mismatch acquires RELEASE the mesh's current geometry
    // into the free pool before updatePointsGeometry runs. If the update
    // throws, the mesh must still be switched to the acquired geometry —
    // otherwise it keeps rendering a free-pooled geometry the evictor can
    // dispose (or another node adopt) mid-render. committedData must stay
    // unstamped so the next update re-uploads in full.
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const oldGeometry = points.geometry;

    const newGeometry = new THREE.BufferGeometry();
    const gpuBufferPool = {
      acquirePointsGeometry: vi.fn(() => newGeometry),
      updatePointsGeometry: vi.fn(() => {
        throw new Error('upload failed');
      }),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
    };

    const data = makeData(3);
    expect(() =>
      commitPointsGeometry('/p', data, root, gpuBufferPool as never, mockNodeFactory, undefined, 0)
    ).toThrow('upload failed');

    expect(points.geometry).toBe(newGeometry);
    expect(points.geometry).not.toBe(oldGeometry);
    expect((points.userData as { committedData?: unknown }).committedData).toBeUndefined();
  });

  it('leaves the freshness stamps untouched when the pool update throws (success-only stamps)', () => {
    // Prime a SUCCESSFUL commit first so the stamps hold real values, then
    // make a bigger commit throw: visiblePointCount / loadedViewVersion /
    // committedData must all still describe the last SUCCESSFUL commit —
    // a throwing write must not stamp the mesh fresh-for-the-new-view with
    // a count that never landed (the LOD freshness registry would trust it).
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);

    const geometry = new THREE.BufferGeometry();
    const gpuBufferPool = {
      acquirePointsGeometry: vi.fn(() => geometry),
      updatePointsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => false),
    };
    const first = makeData(2);
    commitPointsGeometry('/p', first, root, gpuBufferPool as never, mockNodeFactory, undefined, 1);
    expect((points.userData as { visiblePointCount: number }).visiblePointCount).toBe(2);

    gpuBufferPool.updatePointsGeometry.mockImplementation(() => {
      throw new Error('upload failed');
    });
    expect(() =>
      commitPointsGeometry(
        '/p',
        makeData(5),
        root,
        gpuBufferPool as never,
        mockNodeFactory,
        undefined,
        2
      )
    ).toThrow('upload failed');

    expect((points.userData as { visiblePointCount: number }).visiblePointCount).toBe(2);
    expect((points.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(1);
    expect((points.userData as { committedData?: unknown }).committedData).toBe(first);
  });

  it('clears gpuPrefixIntact when the pool update throws (append-safety defense-in-depth)', () => {
    // Prime a SUCCESSFUL commit (stamps the flag true), then make a
    // bigger commit throw: the write left the buffer content unproven,
    // so the catch must clear the append-safety flag — the next commit
    // full-rewrites even if a future loader/cache change re-stamps
    // prefix lineage. Pins the commit's `catch { gpuPrefixIntact =
    // false; throw }` block (the lineage consume-and-clear masks it
    // behaviorally today, so nothing else fails when it's removed).
    // Twin of the gsplats/lines tests of the same name.
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);

    const geometry = new THREE.BufferGeometry();
    const gpuBufferPool = {
      acquirePointsGeometry: vi.fn(() => geometry),
      updatePointsGeometry: vi.fn(),
      didLastAcquireRebuildAttributes: vi.fn(() => false),
    };
    commitPointsGeometry(
      '/p',
      makeData(2),
      root,
      gpuBufferPool as never,
      mockNodeFactory,
      undefined,
      1
    );
    expect((points.userData as { gpuPrefixIntact?: boolean }).gpuPrefixIntact).toBe(true);

    gpuBufferPool.updatePointsGeometry.mockImplementation(() => {
      throw new Error('upload failed');
    });
    expect(() =>
      commitPointsGeometry(
        '/p',
        makeData(5),
        root,
        gpuBufferPool as never,
        mockNodeFactory,
        undefined,
        2
      )
    ).toThrow('upload failed');

    expect((points.userData as { gpuPrefixIntact?: boolean }).gpuPrefixIntact).toBe(false);
  });

  it('non-pool path: keeps the previous geometry undisposed when the factory throws (create-then-swap-then-dispose)', () => {
    // The non-pool fallback builds the replacement geometry BEFORE touching
    // the mesh: a throwing factory (malformed data) must leave the mesh on
    // its old, still-valid geometry — dispose-first would strand the mesh
    // on a freed element texture.
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const prevGeometry = points.geometry;
    const disposeSpy = vi.spyOn(prevGeometry, 'dispose');

    const throwingFactory = {
      createPointsGeometry: vi.fn(() => {
        throw new Error('malformed data');
      }),
    } as unknown as NodeFactory;

    expect(() =>
      commitPointsGeometry('/p', makeData(3), root, null, throwingFactory, undefined, 0)
    ).toThrow('malformed data');

    expect(points.geometry).toBe(prevGeometry);
    expect(disposeSpy).not.toHaveBeenCalled();
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

describe('commitPointsGeometry — append fast path (Phase 4 Stage 2, fromInstance)', () => {
  // The gate lives in the pool branch; these tests pin `fromInstance` in the
  // options threaded to updatePointsGeometry. The suffix-write behavior
  // itself is covered in interleaved-attributes.test.ts.
  const makePool = (geometry: THREE.BufferGeometry) => ({
    acquirePointsGeometry: vi.fn(() => geometry),
    updatePointsGeometry: vi.fn(),
    didLastAcquireRebuildAttributes: vi.fn(() => false),
  });
  const lastOpts = (pool: ReturnType<typeof makePool>) =>
    (pool.updatePointsGeometry.mock.calls.at(-1) as unknown[])[3] as {
      fromInstance: number;
    };

  // Arrange a committed prefix, then stage a genuine extension of it.
  const primeAndExtend = (
    root: THREE.Group,
    pool: ReturnType<typeof makePool>,
    prevCount: number,
    newCount: number
  ): LoadedPointsData => {
    commitPointsGeometry(
      '/p',
      makeData(prevCount),
      root,
      pool as never,
      mockNodeFactory,
      undefined,
      0
    );
    const committed = (root.children[0].userData as { committedData: object }).committedData;
    const next = makeData(newCount);
    setPrefixParent(next, committed); // forward-chain lineage
    return next;
  };

  it('fires the append (fromInstance = prevCount) when the commit extends the committed prefix', () => {
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    commitPointsGeometry('/p', next, root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(4);
    // Positive-path bookkeeping stamp re-enables the NEXT append.
    expect((root.children[0].userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact).toBe(true);
    // Consume-and-clear: the gate consumed the lineage entry, unpinning the
    // parent concat (prefix-lineage.ts retention contract).
    expect(getPrefixParent(next)).toBeUndefined();
  });

  it('does NOT append (fromInstance 0) when there is no prefix lineage (unrelated reload)', () => {
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    commitPointsGeometry('/p', makeData(4), root, pool as never, mockNodeFactory, undefined, 0);
    // A larger commit with NO lineage stamp: full rewrite.
    commitPointsGeometry('/p', makeData(6), root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append after a context restore cleared gpuPrefixIntact', () => {
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    (root.children[0].userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact = false;
    commitPointsGeometry('/p', next, root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append when the acquire rebuilt attributes (pool grow / best-fit swap)', () => {
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    // Grow handed back a different geometry with rebuilt attributes.
    pool.acquirePointsGeometry.mockReturnValue(new THREE.BufferGeometry());
    pool.didLastAcquireRebuildAttributes.mockReturnValue(true);
    commitPointsGeometry('/p', next, root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append on an equal-count or shrinking recommit', () => {
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    const same = primeAndExtend(root, pool, 5, 5); // same count, lineage set
    commitPointsGeometry('/p', same, root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
    const shrunk = makeData(3);
    setPrefixParent(shrunk, (root.children[0].userData as { committedData: object }).committedData);
    commitPointsGeometry('/p', shrunk, root, pool as never, mockNodeFactory, undefined, 2);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append when an optional field flips presence vs the committed parent', () => {
    // concatOptionalField is all-or-nothing: a new level WITHOUT Float32
    // radii drops the merged field entirely, and the adapter's 0.5 fill
    // would differ from the prefix's committed radii. Presence-flip must
    // force a full rewrite. (Float32↔absent is invisible to the pool's
    // dtype matching, so the gate owns this case.)
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    commitPointsGeometry(
      '/p',
      makeData(4, /*withRadii=*/ true),
      root,
      pool as never,
      mockNodeFactory,
      undefined,
      0
    );
    const committed = (root.children[0].userData as { committedData: object }).committedData;
    const next = makeData(6, /*withRadii=*/ false); // radii dropped
    setPrefixParent(next, committed);
    commitPointsGeometry('/p', next, root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
  });

  it('does NOT append when colors/sharpness/scalars presence flips vs the committed parent', () => {
    // Same all-or-nothing rationale as the radii flip above — the gate
    // carries one presence conjunct PER optional field, so each field's
    // flip must force the full rewrite on its own (a mutant dropping any
    // single conjunct would append a fill-valued suffix onto a prefix
    // whose committed values differ).
    const makeDataWithField = (
      pointCount: number,
      field: 'colors' | 'sharpness' | 'scalars',
      present: boolean
    ): LoadedPointsData => {
      const data = makeData(pointCount);
      if (field === 'colors') {
        data.colors = present ? new Uint8Array(pointCount * 3) : undefined;
      } else if (field === 'sharpness') {
        data.sharpness = present ? new Float32Array(pointCount) : undefined;
      } else {
        data.scalars = present ? new Float32Array(pointCount) : undefined;
      }
      return data;
    };

    for (const field of ['colors', 'sharpness', 'scalars'] as const) {
      const root = new THREE.Group();
      root.add(makePoints('/p'));
      const pool = makePool(new THREE.BufferGeometry());
      commitPointsGeometry(
        '/p',
        makeDataWithField(4, field, /*present=*/ true),
        root,
        pool as never,
        mockNodeFactory,
        undefined,
        0
      );
      const committed = (root.children[0].userData as { committedData: object }).committedData;
      const next = makeDataWithField(6, field, /*present=*/ false); // field dropped
      setPrefixParent(next, committed);
      commitPointsGeometry('/p', next, root, pool as never, mockNodeFactory, undefined, 1);
      expect(lastOpts(pool).fromInstance, `presence flip: ${field}`).toBe(0);
    }
  });

  it('does NOT append when the pool hands back a DIFFERENT geometry, even without a reported rebuild', () => {
    // geometry === prevGeometry is a load-bearing conjunct of its own: a
    // swap that (hypothetically) reported no attribute rebuild still means
    // the committed prefix lives in ANOTHER buffer — appending would
    // extend a stranger's texels.
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    const next = primeAndExtend(root, pool, 4, 6);
    pool.acquirePointsGeometry.mockReturnValue(new THREE.BufferGeometry());
    // didLastAcquireRebuildAttributes stays FALSE (makePool default) — the
    // geometry-identity conjunct must gate alone.
    commitPointsGeometry('/p', next, root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool).fromInstance).toBe(0);
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

describe('commitPointsGeometry — committedEnergyFraction stamp', () => {
  const energy = (points: THREE.Mesh) =>
    (points.userData as { committedEnergyFraction?: number }).committedEnergyFraction;

  it('stamps the progressive loader committed-energy fraction e(k) mid-ladder', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    points.userData.loader = { hasMoreLODs: true, committedEnergyFraction: 0.42 };
    root.add(points);
    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory, undefined, 0);
    expect(energy(points)).toBe(0.42);
  });

  it('REMOVES the stamp on an unstamped (legacy) dataset; stamps 1 for non-progressive', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    points.userData.loader = { hasMoreLODs: true, committedEnergyFraction: null };
    points.userData.committedEnergyFraction = 0.9; // stale
    root.add(points);
    commitPointsGeometry('/p', makeData(3), root, null, mockNodeFactory, undefined, 0);
    expect(energy(points)).toBeUndefined();

    const root2 = new THREE.Group();
    const plain = makePoints('/p');
    plain.userData.loader = {};
    root2.add(plain);
    commitPointsGeometry('/p', makeData(2), root2, null, mockNodeFactory, undefined, 0);
    expect(energy(plain)).toBe(1);
  });
});

describe('commitPointsGeometry — depth-sort integration (points sort registration)', () => {
  beforeEach(() => {
    mockNoteDepthSortCommit.mockReset();
  });

  const makePool = (geometry: THREE.BufferGeometry) => ({
    acquirePointsGeometry: vi.fn(() => geometry),
    updatePointsGeometry: vi.fn(),
    didLastAcquireRebuildAttributes: vi.fn(() => false),
  });

  it('pool path: notifies the coordinator with a LAZY centers provider and the count', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const data = makeData(3);
    (data.positions as Float32Array).set([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const pool = makePool(new THREE.BufferGeometry());
    commitPointsGeometry('/p', data, root, pool as never, mockNodeFactory, undefined, 0);

    expect(mockNoteDepthSortCommit).toHaveBeenCalledTimes(1);
    const [mesh, provider, count] = mockNoteDepthSortCommit.mock.calls[0] as [
      THREE.Mesh,
      () => Float32Array,
      number,
    ];
    expect(mesh).toBe(points);
    expect(count).toBe(3);
    // Points pass a THUNK (deferring the O(N) copy to the sorted path),
    // unlike gsplats' fresh centers3D array.
    expect(typeof provider).toBe('function');
    const centers = provider();
    expect(centers).toBeInstanceOf(Float32Array);
    expect(Array.from(centers)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    // FRESH allocation, not a view: the coordinator transfers the returned
    // buffer to the SortWorker; sharing data.positions' ArrayBuffer would
    // detach the committed/lineage reference with it.
    expect(centers.buffer).not.toBe((data.positions as Float32Array).buffer);
  });

  it('non-pool path: notifies too (both commit branches register)', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    commitPointsGeometry('/p', makeData(4), root, null, mockNodeFactory, undefined, 0);
    expect(mockNoteDepthSortCommit).toHaveBeenCalledTimes(1);
    expect(mockNoteDepthSortCommit.mock.calls[0][0]).toBe(points);
    expect(mockNoteDepthSortCommit.mock.calls[0][2]).toBe(4);
  });

  it('the provider widens Float16 positions to the Float32 the sort kernel expects', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const data = makeData(2);
    // 0.5 / 1.5 / -2 are exactly representable in fp16 — the widened
    // copy must be value-identical. The Float16Array GLOBAL is absent on
    // Node < 23 (CI), so fall back to a Float64Array stand-in there: both
    // take the thunk's same non-Float32 element-wise branch, and the
    // fp16-exact values make the two engines byte-identical.
    const F16 = (globalThis as unknown as { Float16Array?: Float16ArrayConstructor }).Float16Array;
    const values = [0.5, 1.5, -2, 3, -0.25, 8];
    data.positions = (F16 ? new F16(values) : new Float64Array(values)) as never;
    const pool = makePool(new THREE.BufferGeometry());
    commitPointsGeometry('/p', data, root, pool as never, mockNodeFactory, undefined, 0);

    const provider = mockNoteDepthSortCommit.mock.calls[0][1] as () => Float32Array;
    const centers = provider();
    expect(centers).toBeInstanceOf(Float32Array);
    expect(Array.from(centers)).toEqual([0.5, 1.5, -2, 3, -0.25, 8]);
  });

  it('is success-only: a throwing GPU write must NOT bump the sort generation', () => {
    // Mirrors the gsplats ordering (noteDepthSortCommit after the write
    // block): the buffer holds partially-written data, committedData was
    // not stamped, and the next commit full-rewrites + registers.
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const pool = makePool(new THREE.BufferGeometry());
    pool.updatePointsGeometry.mockImplementation(() => {
      throw new Error('device lost');
    });
    expect(() =>
      commitPointsGeometry('/p', makeData(3), root, pool as never, mockNodeFactory, undefined, 0)
    ).toThrow('device lost');
    expect(mockNoteDepthSortCommit).not.toHaveBeenCalled();
  });

  it('is skipped on the memoized no-op recommit (stamp-only — in-flight sorts stay valid)', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const data = makeData(3);
    commitPointsGeometry('/p', data, root, null, mockNodeFactory, undefined, 0);
    expect(mockNoteDepthSortCommit).toHaveBeenCalledTimes(1);
    // SAME reference again: the no-op path must not bump the generation
    // (spec §5 generation contract, shared with the gsplats staged.noop).
    commitPointsGeometry('/p', data, root, null, mockNodeFactory, undefined, 1);
    expect(mockNoteDepthSortCommit).toHaveBeenCalledTimes(1);
  });

  it('reports count 0 on an empty commit (coordinator release path)', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    commitPointsGeometry('/p', makeData(0), root, null, mockNodeFactory, undefined, 0);
    expect(mockNoteDepthSortCommit).toHaveBeenCalledTimes(1);
    expect(mockNoteDepthSortCommit.mock.calls[0][2]).toBe(0);
  });

  describe('capacity-clamp consistency', () => {
    afterEach(() => {
      resetElementTextureLayoutForTests();
    });

    it('notifies the coordinator with the CLAMPED count and a same-length provider', () => {
      // maxTextureSize 6 → width 6 (multiple of texelsPerElement 3),
      // per-node bound = 6×6/3 = 12 points. An unclamped count would make
      // the SortWorker return permutation values ≥ the texture capacity
      // (OOB texel fetches → points vanish).
      configureElementTextureLayout(6);
      const root = new THREE.Group();
      const points = makePoints('/p');
      root.add(points);
      const pool = makePool(new THREE.BufferGeometry());
      commitPointsGeometry('/p', makeData(100), root, pool as never, mockNodeFactory, undefined, 0);

      expect((points.userData as { visiblePointCount: number }).visiblePointCount).toBe(12);
      const [, provider, count] = mockNoteDepthSortCommit.mock.calls[0] as [
        THREE.Mesh,
        () => Float32Array,
        number,
      ];
      expect(count).toBe(12);
      expect((provider() as Float32Array).length).toBe(12 * 3);
    });
  });
});

describe('commitPointsGeometry — preserve-ordering on same-node same-count recommits', () => {
  // The commit path decides; the writer obeys the flag (its skip behavior
  // is covered in points-texture-storage tests). These pin the predicate
  // (hadCommittedData && !attributesRebuilt && same geometry && same count)
  // by asserting the options arg threaded to updatePointsGeometry —
  // mirroring the gsplats twin in commit-gsplats-geometry.test.ts.
  const makePool = (geometry: THREE.BufferGeometry) => ({
    acquirePointsGeometry: vi.fn(() => geometry),
    updatePointsGeometry: vi.fn(),
    didLastAcquireRebuildAttributes: vi.fn(() => false),
  });
  const lastOpts = (pool: ReturnType<typeof makePool>) =>
    (pool.updatePointsGeometry.mock.calls.at(-1) as unknown[])[3];

  it('same-count recommit → preserveOrdering true (first commit → false)', () => {
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    // First commit: no committedData stamp yet → the geometry's ordering
    // is unvouched-for, identity must be written.
    commitPointsGeometry('/p', makeData(7), root, pool as never, mockNodeFactory, undefined, 0);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
    // Same-node same-count recommit on the SAME pooled geometry: the
    // previous permutation of [0,7) is still valid — keep it as a
    // no-worse prior until the commit-triggered re-sort lands. Equal
    // count is NOT an append, so fromInstance stays 0.
    commitPointsGeometry('/p', makeData(7), root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: true, fromInstance: 0 });
  });

  it('count-change recommit → preserveOrdering false', () => {
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    commitPointsGeometry('/p', makeData(7), root, pool as never, mockNodeFactory, undefined, 0);
    // A permutation of [0,7) is not a permutation of [0,9).
    commitPointsGeometry('/p', makeData(9), root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
  });

  it('recommit after committedData was cleared (LOD demotion) → false', () => {
    const root = new THREE.Group();
    const points = makePoints('/p');
    root.add(points);
    const pool = makePool(new THREE.BufferGeometry());
    commitPointsGeometry('/p', makeData(7), root, pool as never, mockNodeFactory, undefined, 0);
    delete (points.userData as { committedData?: unknown }).committedData;
    commitPointsGeometry('/p', makeData(7), root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
  });

  it('geometry swap / attribute rebuild defeats the flag', () => {
    const root = new THREE.Group();
    root.add(makePoints('/p'));
    const pool = makePool(new THREE.BufferGeometry());
    commitPointsGeometry('/p', makeData(7), root, pool as never, mockNodeFactory, undefined, 0);
    // Best-fit reuse handed the node a DIFFERENT geometry (holding some
    // other node's permutation over a different prior count) and reported
    // an attribute rebuild — identity must be written.
    pool.acquirePointsGeometry.mockReturnValue(new THREE.BufferGeometry());
    pool.didLastAcquireRebuildAttributes.mockReturnValue(true);
    commitPointsGeometry('/p', makeData(7), root, pool as never, mockNodeFactory, undefined, 1);
    expect(lastOpts(pool)).toEqual({ preserveOrdering: false, fromInstance: 0 });
  });
});
