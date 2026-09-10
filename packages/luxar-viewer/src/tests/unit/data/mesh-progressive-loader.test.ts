/**
 * Unit tests for `MeshProgressiveLoader` — the reveal-ladder composite.
 *
 * Two things here are worth more than the rest, and both are silent when wrong:
 *
 *  - **Face-index offsetting.** Each level's indices are local to its own vertex
 *    array, so a verbatim copy does not crash — it draws a plausible-looking
 *    garbage surface stitched out of the wrong vertices. Every assertion about
 *    it is therefore on VALUES, not lengths.
 *  - **`committedEnergyFraction` existing and returning `null`.** Absent, the
 *    commit stamps a half-revealed mesh as carrying all its energy; numeric, the
 *    LOD fade brightens it by `1/e`. Only the middle case is right, and the two
 *    wrong ones are invisible to any test that just reads the getter — so the
 *    consequence is asserted through `stampLadderComplete` itself, with a
 *    sensitivity control that pins the OTHER branch.
 */

import { afterEach, beforeEach, describe, it, expect, vi } from 'vitest';
import {
  MeshProgressiveLoader,
  concatenateMeshData,
} from '../../../data/mesh/mesh-progressive-loader';
import { stampLadderComplete } from '../../../data/scene-loader/commit/stamp-view-version';
import { LoaderError } from '../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import { MESH_DECODE_BUDGET_BYTES } from '../../../config/constants';
import { CACHE_HIT_THRESHOLD_MS } from '../../../data/loaders/progressive/constants';
import type { LoadedMeshData, MeshViewState } from '../../../types/mesh';
import type { MeshWholeNodeLoader } from '../../../data/mesh/mesh-whole-node-loader';
import { testLadderFoldContract } from './_shared/ladder-fold-contract';
import {
  resetLodLoadStats,
  setLodLoadStatsEnabled,
  snapshotLodLoadStats,
} from '../../../data/scene-loader/lod-load-stats';

// ============================================================================
// Fixtures
// ============================================================================

/**
 * One ladder level: `nVertices` vertices at ascending coordinates and
 * `faces` given as LOCAL indices, exactly as the writer's `split_mesh_by_faces`
 * emits them.
 */
function level(
  nVertices: number,
  faces: number[],
  opts: {
    ndim?: number;
    normals?: boolean;
    colors?: 3 | 4 | null;
    scalars?: boolean;
    base?: number;
  } = {}
): LoadedMeshData {
  const ndim = opts.ndim ?? 3;
  const base = opts.base ?? 0;
  const vertices = new Float32Array(nVertices * ndim);
  for (let v = 0; v < nVertices; v++) {
    for (let d = 0; d < ndim; d++) vertices[v * ndim + d] = base + v + d / 10;
  }
  const colorK = opts.colors ?? null;
  return {
    vertices,
    faces: new Uint32Array(faces),
    normals: opts.normals ? new Float32Array(nVertices * 3).fill(base + 1) : null,
    colors: colorK ? new Uint8Array(nVertices * colorK).fill(base + 2) : null,
    ...(colorK ? { colorComponents: colorK } : {}),
    ...(opts.scalars ? { scalars: new Float32Array(nVertices).fill(base + 3) } : {}),
    vertexCount: nVertices,
    faceCount: faces.length / 3,
    ndim,
  };
}

/**
 * A sub-loader stub with the methods the ladder loop and the byte-budget gate
 * call.
 *
 * `gate` lets a test hold a level mid-flight, which is the only way to exercise
 * the post-await dispose check: the ladder loop's hazard is a `dispose()` that
 * lands BETWEEN the await resolving and the push, and a synchronous stub never
 * opens that window.
 *
 * `accountedBytes` defaults to a tiny value so every pre-existing test above —
 * none of which cares about the byte budget — keeps passing untouched; a
 * ladder of default-tiny levels never trips the aggregate the budget describe
 * below exercises. `preflightError`, when set, makes `runPreflight()` REJECT
 * instead of resolving — simulating a level whose own metadata open fails (a
 * transient network blip or a malformed store; the gate propagates either one
 * unlatched, without reading its kind).
 * `preflightErrorAfterDispose` rejects only once `dispose()` has been called,
 * which is what the REAL `MeshWholeNodeLoader` does to an in-flight
 * `runPreflight()`: its `dispose()` nulls `preflight`, so the call throws
 * `LoaderError('Unexpected', …, 'mesh loader not initialized')`.
 */
function subLoader(
  data: LoadedMeshData,
  opts: {
    resident?: boolean;
    gate?: Promise<void>;
    accountedBytes?: number;
    preflightError?: Error | null;
    preflightErrorAfterDispose?: Error | null;
    preflightGate?: Promise<void>;
  } = {}
): MeshWholeNodeLoader & { calls: number; disposed: boolean; preflightCalls: number } {
  const stub = {
    calls: 0,
    disposed: false,
    preflightCalls: 0,
    runPreflight: vi.fn(async () => {
      stub.preflightCalls++;
      if (opts.preflightGate) await opts.preflightGate;
      if (opts.preflightErrorAfterDispose && stub.disposed) throw opts.preflightErrorAfterDispose;
      if (opts.preflightError) throw opts.preflightError;
      return { accountedBytes: opts.accountedBytes ?? 1, nVertices: 0, nFaces: 0, ndim: 0 };
    }),
    updateViewWithResidency: vi.fn(async () => {
      stub.calls++;
      if (opts.gate) await opts.gate;
      return { data, allResident: opts.resident ?? true };
    }),
    releaseData: vi.fn(),
    dispose: vi.fn(() => {
      stub.disposed = true;
    }),
  };
  return stub as unknown as MeshWholeNodeLoader & {
    calls: number;
    disposed: boolean;
    preflightCalls: number;
  };
}

const VIEW: MeshViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0],
  tolerance: [0, 0, 0],
};

// ============================================================================
// concatenateMeshData
// ============================================================================

describe('concatenateMeshData', () => {
  it('offsets each level face index by the preceding levels VERTEX count', () => {
    // Level 0: 4 vertices, one triangle. Level 1: 3 vertices, one triangle whose
    // indices are 0,1,2 IN ITS OWN array — i.e. global 4,5,6.
    const merged = concatenateMeshData([level(4, [0, 1, 2]), level(3, [0, 1, 2], { base: 100 })]);

    expect(merged.vertexCount).toBe(7);
    expect(merged.faceCount).toBe(2);
    // The whole point: a verbatim copy would give [0,1,2, 0,1,2] — the same
    // length, the same face count, and a surface stitched from level 0's
    // vertices twice. Assert the VALUES.
    expect(Array.from(merged.faces)).toEqual([0, 1, 2, 4, 5, 6]);
    // And every index must address a real vertex of the merged buffer.
    for (const index of merged.faces) expect(index).toBeLessThan(merged.vertexCount);
  });

  it('keeps the second level vertices reachable through the offset indices', () => {
    // The end-to-end version of the check above: follow an index back to the
    // coordinates it lands on, so an offset that is merely *present* but wrong
    // (off by a face count rather than a vertex count, say) still fails.
    const merged = concatenateMeshData([level(4, [0, 1, 2]), level(3, [0, 1, 2], { base: 100 })]);
    const firstOfSecondTriangle = merged.faces[3];
    expect(merged.vertices[firstOfSecondTriangle * 3]).toBeCloseTo(100);
  });

  it('strides vertices by ndim, not by 3', () => {
    const merged = concatenateMeshData([
      level(2, [0, 1, 1], { ndim: 4 }),
      level(2, [0, 1, 1], { ndim: 4, base: 50 }),
    ]);
    expect(merged.ndim).toBe(4);
    expect(merged.vertices.length).toBe(4 * 4);
    // Level 1's first vertex starts at element 2*4 = 8, not 2*3 = 6.
    expect(merged.vertices[8]).toBeCloseTo(50);
  });

  it('throws on levels that disagree about ndim', () => {
    expect(() =>
      concatenateMeshData([level(2, [0, 1, 1]), level(2, [0, 1, 1], { ndim: 4 })])
    ).toThrow(/mixed dimensionality/);
  });

  it('throws when one level carries normals and another omits them', () => {
    // Not a fill-with-default case, unlike the sibling ladders' colours: the
    // `normal` attribute is bound once from the node's metadata and can never be
    // added or removed later, so a level without normals would leave the previous
    // level's bound at the wrong length.
    expect(() =>
      concatenateMeshData([level(2, [0, 1, 1], { normals: true }), level(2, [0, 1, 1])])
    ).toThrow(/'normals'/);
  });

  it('throws when levels disagree about the colour layout', () => {
    expect(() =>
      concatenateMeshData([level(2, [0, 1, 1], { colors: 3 }), level(2, [0, 1, 1], { colors: 4 })])
    ).toThrow(/mixed color layouts|'colors'/);
  });

  it('concatenates the optional per-vertex arrays alongside the vertices', () => {
    const merged = concatenateMeshData([
      level(2, [0, 1, 1], { normals: true, colors: 3, scalars: true }),
      level(3, [0, 1, 2], { normals: true, colors: 3, scalars: true, base: 10 }),
    ]);
    expect(merged.normals?.length).toBe(5 * 3);
    expect(merged.colors?.length).toBe(5 * 3);
    expect(merged.scalars?.length).toBe(5);
    // Level 1's values start where level 0's end — the same offset the faces use.
    expect(merged.scalars?.[2]).toBeCloseTo(13);
  });

  it('sizes the projection scratch to the REVEALED prefix', () => {
    const merged = concatenateMeshData([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
    expect(merged.projection?.position.length).toBe(7 * 3);
    expect(merged.projection?.mask.length).toBe(7);
    expect(merged.projection?.faceScratch.length).toBe(2 * 3);
    // A fresh scratch has not been extracted into yet, so the epoch key must be
    // null — otherwise the projection would skip the re-extract and draw a stale
    // frame into a buffer that has never held one.
    expect(merged.projection?.displayDimsKey).toBeNull();
  });

  it('hands back the single level unchanged, scratch included', () => {
    const only = level(4, [0, 1, 2]);
    expect(concatenateMeshData([only])).toBe(only);
  });
});

// ============================================================================
// MeshProgressiveLoader
// ============================================================================

describe('MeshProgressiveLoader', () => {
  let clockSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // Depth assertions must not depend on scheduler speed; timing-policy tests
    // override this stable clock with their own moving implementation.
    clockSpy = vi.spyOn(performance, 'now').mockReturnValue(0);
  });

  afterEach(() => {
    clockSpy.mockRestore();
  });

  function makeLadder(
    levels: LoadedMeshData[],
    opts: { resident?: boolean } = {}
  ): {
    loader: MeshProgressiveLoader;
    subs: ReturnType<typeof subLoader>[];
  } {
    const subs = levels.map((d) => subLoader(d, opts));
    return { loader: new MeshProgressiveLoader(subs, levels.length, '/surf'), subs };
  }

  it('records mesh additive load timing keys', async () => {
    resetLodLoadStats();
    setLodLoadStatsEnabled(true);
    try {
      const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
      await loader.updateView(VIEW);
      expect(Object.keys(snapshotLodLoadStats())).toEqual([
        'additive:mesh:level:0:resident',
        'additive:mesh:level:1:resident',
      ]);
    } finally {
      setLodLoadStatsEnabled(false);
      resetLodLoadStats();
    }
  });

  it('streams every level on a refine pass and reports completion', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])]);

    expect(loader.hasMoreLODs).toBe(true);
    const data = await loader.updateView(VIEW);

    expect(loader.loadedLODCount).toBe(3);
    expect(loader.totalLODCount).toBe(3);
    expect(loader.hasMoreLODs).toBe(false);
    expect(data.faceCount).toBe(3);
    expect(data.vertexCount).toBe(10);
  });

  it(`stops after a RESIDENT level that took > ${CACHE_HIT_THRESHOLD_MS}ms`, async () => {
    // The other disjunct of `shouldStopAfterLevel`, and the one the freeze above
    // deliberately puts out of that test's reach: `!allResident || elapsedMs >
    // CACHE_HIT_THRESHOLD_MS`. A level can be fully cache-resident and still be
    // too expensive to continue past — residency spares it the fetch, not the
    // dequant+project — so the pass must yield the frame and let a later pass
    // take the rest. The
    // `!allResident` half is pinned by the test below; without this one the mesh
    // ladder would cover only that half, while `points-`, `lines-` and
    // `gsplats-progressive-loader.test.ts` each pin the timing half too — and the
    // four geometry types are expected to keep mirrored test coverage.
    let now = 0;
    const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => now);
    try {
      const levels = [level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])];
      const subs = levels.map((d) => subLoader(d));
      // Level 1 alone "costs" more than the threshold; every other level is
      // free. Residency stays TRUE throughout, so only the timing disjunct can
      // end the pass here.
      subs[1].updateViewWithResidency = vi.fn(async () => {
        now += CACHE_HIT_THRESHOLD_MS + 1;
        return { data: levels[1], allResident: true };
      });
      const loader = new MeshProgressiveLoader(subs, levels.length, '/surf');

      const data = await loader.updateView(VIEW);

      expect(loader.loadedLODCount).toBe(2);
      expect(loader.hasMoreLODs).toBe(true);
      // The slow level itself IS committed (the break is after the push) —
      // what never loads is the level after it.
      expect(data.faceCount).toBe(2);
      expect(subs[2].calls).toBe(0);
    } finally {
      nowSpy.mockRestore();
    }
  });

  it('stops after the first COLD level on a refine pass', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])], {
      resident: false,
    });

    await loader.updateView(VIEW);

    // The first-paint floor always loads (level 0), and the next level's cache
    // miss ends the pass so the frame can render. A later pass continues.
    expect(loader.loadedLODCount).toBe(2);
    expect(loader.hasMoreLODs).toBe(true);
  });

  it('retains one cumulative payload after concatenation', async () => {
    const { loader, subs } = makeLadder([
      level(4, [0, 1, 2]),
      level(3, [0, 1, 2]),
      level(3, [0, 1, 2]),
    ]);

    const result = await loader.updateView(VIEW);
    const retained = (loader as unknown as { loadedLODs: LoadedMeshData[] }).loadedLODs;

    expect(loader.loadedLODCount).toBe(3);
    expect(retained).toEqual([result]);
    for (const sub of subs) expect(sub.releaseData).toHaveBeenCalledOnce();
  });

  it('unwinds intact rung payloads when concatenation fails before folding', async () => {
    const { loader } = makeLadder([
      level(4, [0, 1, 2]),
      level(3, [0, 1, 2], { ndim: 4 }),
      level(3, [0, 1, 2]),
    ]);

    await expect(loader.updateView(VIEW)).rejects.toThrow('mixed dimensionality');
    expect(loader.loadedLODCount).toBe(3);

    expect(loader.rollbackToPassStart()).toBe(3);
    expect(loader.loadedLODCount).toBe(0);
    expect(loader.hasMoreLODs).toBe(true);
  });

  it('replays a failed pass without duplicating levels behind a folded prefix', async () => {
    const levels = [
      level(4, [0, 1, 2]),
      level(3, [0, 1, 2]),
      level(3, [0, 1, 2]),
      level(3, [0, 1, 2]),
    ];
    const subs = [
      subLoader(levels[0]),
      subLoader(levels[1], { resident: false }),
      subLoader(levels[2]),
      subLoader(levels[3]),
    ];
    const loader = new MeshProgressiveLoader(subs, levels.length, '/surf');

    const prefix = await loader.updateView(VIEW);
    expect(prefix.vertexCount).toBe(7);
    expect(prefix.faceCount).toBe(2);

    vi.mocked(subs[3].updateViewWithResidency).mockRejectedValueOnce(new Error('fetch failed'));
    await expect(loader.updateView(VIEW)).rejects.toThrow('fetch failed');
    expect(loader.rollbackToPassStart()).toBe(1);

    const replayed = await loader.updateView(VIEW);
    expect(replayed.vertexCount).toBe(13);
    expect(replayed.faceCount).toBe(4);
  });

  it('keeps an incrementally folded prefix and cursor paired', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])], {
      resident: false,
    });

    await loader.updateView(VIEW);
    expect(loader.loadedLODCount).toBe(2);

    await loader.updateView(VIEW);
    expect(loader.loadedLODCount).toBe(3);
    expect(loader.rollbackToPassStart()).toBe(0);
    expect(loader.loadedLODCount).toBe(3);
    expect(loader.hasMoreLODs).toBe(true);
  });

  it('keeps a completed pass schedulable when its commit fails', async () => {
    const { loader, subs } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);

    const result = await loader.updateView(VIEW);
    expect(loader.loadedLODCount).toBe(2);
    expect(loader.hasMoreLODs).toBe(false);

    await expect(loader.updateView(VIEW)).resolves.toBe(result);
    expect(loader.rollbackToPassStart()).toBe(0);
    expect(loader.loadedLODCount).toBe(2);
    expect(loader.hasMoreLODs).toBe(true);

    for (const sub of subs) vi.mocked(sub.updateViewWithResidency).mockClear();
    await expect(loader.updateView(VIEW)).resolves.toBe(result);
    for (const sub of subs) expect(vi.mocked(sub.updateViewWithResidency)).not.toHaveBeenCalled();
    expect(loader.hasMoreLODs).toBe(false);
  });

  it('streams resident levels under a playback frame budget', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);

    await loader.updateView({ ...VIEW, frameBudgetMs: 8 });

    // Playback spends its budget on levels that come back cache-resident
    // rather than committing the LOD-0 floor and stopping (#2374, #2376) —
    // on a sliced node the floor alone can be a near-empty frame.
    expect(loader.loadedLODCount).toBe(2);
    // The budgeted prefix IS the target while playback is running, so nothing
    // should schedule background refinement between animation ticks.
    expect(loader.hasMoreLODs).toBe(false);
  });

  it('stops a warmed-cache refinement pass after spending its residency allowance', async () => {
    const { loader, subs } = makeLadder([
      level(4, [0, 1, 2]),
      level(3, [0, 1, 2]),
      level(3, [0, 1, 2]),
    ]);

    await loader.updateView(VIEW, undefined, undefined, 1);

    expect(loader.loadedLODCount).toBe(1);
    expect(subs[1].calls).toBe(0);
  });

  it('deepens an existing playback prefix only while budget remains', async () => {
    let now = 0;
    const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => (now += 5));
    try {
      const { loader, subs } = makeLadder([
        level(4, [0, 1, 2]),
        level(3, [0, 1, 2]),
        level(3, [0, 1, 2]),
      ]);

      await loader.updateView({ ...VIEW, frameBudgetMs: 0 });
      expect(loader.loadedLODCount).toBe(1); // empty-ladder first-paint floor

      await loader.updateView({ ...VIEW, frameBudgetMs: 0 });
      expect(loader.loadedLODCount).toBe(1); // non-empty ladder has no floor
      expect(subs[1].calls).toBe(0);

      nowSpy.mockImplementation(() => now);
      await loader.updateView({ ...VIEW, frameBudgetMs: 8 });
      expect(loader.loadedLODCount).toBe(3);
      expect(subs[0].calls).toBe(1); // no reset
    } finally {
      nowSpy.mockRestore();
    }
  });

  it('returns the SAME object across view changes once the ladder is complete', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);

    const first = await loader.updateView(VIEW);
    const afterSliceMove = await loader.updateView({
      ...VIEW,
      slicePosition: [5, 5, 5],
    });

    // A mesh is whole-node resident, so a slice move changes nothing about what
    // is LOADED. Identity is what keeps the loader-owned projection scratch (and
    // with it the uploaded vertex buffer) alive across a scrub — the sibling
    // loaders cannot do this because their levels answer a range query.
    expect(afterSliceMove).toBe(first);
    expect(first.projection).toBe(afterSliceMove.projection);
  });

  it('returns a NEW object once a further level lands', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])], {
      resident: false,
    });

    const partial = await loader.updateView(VIEW);
    const deeper = await loader.updateView(VIEW);

    expect(deeper).not.toBe(partial);
    expect(deeper.faceCount).toBeGreaterThan(partial.faceCount);
  });

  it('re-serves each level from its own decode rather than re-fetching', async () => {
    const { loader, subs } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);

    await loader.updateView(VIEW);
    await loader.updateView({ ...VIEW, slicePosition: [1, 1, 1] });

    // Once a level is in the ladder it is never asked again — the ladder is
    // view-independent. A per-view reset (what the three siblings do) would show
    // up here as a second call per level.
    for (const sub of subs) expect(sub.calls).toBe(1);
  });

  it('disposes every sub-loader and reports no further work', async () => {
    const { loader, subs } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
    await loader.updateView(VIEW);

    loader.dispose();

    for (const sub of subs) expect(sub.disposed).toBe(true);
    expect(loader.hasMoreLODs).toBe(false);
  });

  it('does not push a level that resolved AFTER dispose', async () => {
    // The teardown race the loop guards twice (before the await and after it).
    // Without the post-await check the level would be pushed onto a ladder
    // `dispose()` just emptied — resurrecting it on a dead loader and pinning
    // the whole prefix's arrays that nothing will ever read.
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const subs = [subLoader(level(4, [0, 1, 2]), { gate }), subLoader(level(3, [0, 1, 2]))];
    const loader = new MeshProgressiveLoader(subs, 2, '/surf');

    const inFlight = loader.updateView(VIEW);
    loader.dispose();
    release();
    const data = await inFlight;

    expect(loader.loadedLODCount).toBe(0);
    expect(data.faceCount).toBe(0);
    // And it must report no further work, so a refinement loop holding a stale
    // reference stops instead of indexing into the emptied array.
    expect(loader.hasMoreLODs).toBe(false);
  });

  it('survives updateView called after dispose', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
    loader.dispose();

    const data = await loader.updateView(VIEW);

    expect(data.faceCount).toBe(0);
    expect(data.vertexCount).toBe(0);
  });

  it('crosses the uint16 index boundary as the prefix grows', async () => {
    // The index buffer dtype is chosen from the VERTEX count: level 0 alone fits
    // uint16, the ladder does not. Nothing here binds a real geometry, but the
    // concat is what decides which side of 65536 the commit sees, so pin that it
    // reports the summed count rather than any single level's.
    const { loader } = makeLadder([level(40_000, [0, 1, 2]), level(40_000, [0, 1, 2])]);

    const data = await loader.updateView(VIEW);

    // Each level on its own is a uint16 index buffer; the ladder is not.
    expect(40_000).toBeLessThan(65_536);
    expect(data.vertexCount).toBeGreaterThan(65_535);
    expect(data.vertexCount).toBe(80_000);
    // And the offset that crosses the boundary is a real index, not a wrap.
    expect(data.faces[3]).toBe(40_000);
  });

  describe('energy stamps', () => {
    it('reports a null committed energy fraction', () => {
      const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
      expect(loader.committedEnergyFraction).toBeNull();
    });

    it('makes the commit leave `committedEnergyFraction` ABSENT', () => {
      // The consequence, not the getter. `stampLadderComplete` probes
      // `'committedEnergyFraction' in loader`, so this is what distinguishes a
      // null-returning getter from no getter at all.
      const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
      const userData: { loader: unknown; committedEnergyFraction?: number } = { loader };

      stampLadderComplete(userData);

      expect('committedEnergyFraction' in userData).toBe(false);
    });

    it('SENSITIVITY: a loader without the getter is stamped as fully committed', () => {
      // The control for the assertion above. Without it, "the key is absent"
      // would pass against a loader that simply has no ladder at all — and the
      // real hazard is exactly that misreading: a half-revealed mesh stamped
      // `committedEnergyFraction: 1`, i.e. "all of it is on screen".
      const userData: { loader: unknown; committedEnergyFraction?: number } = {
        loader: { hasMoreLODs: true },
      };

      stampLadderComplete(userData);

      expect(userData.committedEnergyFraction).toBe(1);
    });

    it('stamps the ladder INCOMPLETE while levels remain', async () => {
      const { loader } = makeLadder(
        [level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])],
        {
          resident: false,
        }
      );
      await loader.updateView(VIEW);
      const userData: { loader: unknown; committedLadderComplete?: boolean } = { loader };

      stampLadderComplete(userData);

      expect(userData.committedLadderComplete).toBe(false);
    });
  });
});

// ============================================================================
// Aggregate byte budget (#1517)
// ============================================================================

describe('MeshProgressiveLoader — aggregate byte budget (#1517)', () => {
  // The bug: each level's `runPreflight()` only ever sees its OWN
  // `MESH_DECODE_BUDGET_BYTES` ceiling. Four levels individually under budget
  // but jointly N times over it used to sail through — the same geometry as a
  // plain leaf, which the leaf's single preflight would have refused up
  // front. `assertWithinByteBudget` sums the levels' `accountedBytes` and
  // charges that once against the SAME ceiling, at the ladder's first
  // `updateView` — before any level's chunks are fetched.

  it('refuses a ladder whose levels are individually fine but jointly over the ceiling', async () => {
    const perLevel = Math.floor(MESH_DECODE_BUDGET_BYTES * 0.3);
    const subs = [
      subLoader(level(4, [0, 1, 2]), { accountedBytes: perLevel }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: perLevel }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: perLevel }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: perLevel }),
    ];
    const loader = new MeshProgressiveLoader(subs, subs.length, '/surf');

    let thrown: unknown;
    try {
      await loader.updateView(VIEW);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(LoaderError);
    expect((thrown as LoaderError).kind).toBe('Validation');
    expect((thrown as Error).message).toMatch(/4 levels account for.*combined, over the.*budget/);
  });

  it('latches the aggregate refusal — a later updateView rethrows the SAME error and re-runs NO preflight', async () => {
    // The over-budget verdict is deterministic once every level's own preflight
    // has already succeeded and cached its result (#1517 round 4): re-deriving it
    // on a later `updateView` can only repeat the same answer. Without latching,
    // `hasMoreLODs` would stay `true` forever and a refinement loop holding this
    // node would re-run this gate — and burn its failure budget — on every scrub.
    const subs = [
      subLoader(level(4, [0, 1, 2]), { accountedBytes: MESH_DECODE_BUDGET_BYTES }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: MESH_DECODE_BUDGET_BYTES }),
    ];
    const loader = new MeshProgressiveLoader(subs, subs.length, '/surf');

    let firstError: unknown;
    try {
      await loader.updateView(VIEW);
    } catch (e) {
      firstError = e;
    }
    expect(firstError).toBeInstanceOf(LoaderError);
    for (const sub of subs) expect(sub.preflightCalls).toBe(1);

    let secondError: unknown;
    try {
      await loader.updateView({ ...VIEW, slicePosition: [1, 1, 1] });
    } catch (e) {
      secondError = e;
    }
    // Same object, not merely an equal message — proof the gate rethrew the
    // cached refusal rather than recomputing it.
    expect(secondError).toBe(firstError);
    for (const sub of subs) expect(sub.preflightCalls).toBe(1);
  });

  it('reports hasMoreLODs false once the aggregate refusal has latched, so the refinement loop leaves the node', async () => {
    const subs = [
      subLoader(level(4, [0, 1, 2]), { accountedBytes: MESH_DECODE_BUDGET_BYTES }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: MESH_DECODE_BUDGET_BYTES }),
    ];
    const loader = new MeshProgressiveLoader(subs, subs.length, '/surf');

    expect(loader.hasMoreLODs).toBe(true);
    await expect(loader.updateView(VIEW)).rejects.toThrow(LoaderError);
    // This is the property that actually stops `queue-next.ts` from scheduling
    // `runMeshRefinement` on the dead node — the latched error alone is not
    // enough if a caller never re-observes it.
    expect(loader.hasMoreLODs).toBe(false);
  });

  it('SENSITIVITY: levels summing to exactly the ceiling still load (not `>=`)', async () => {
    // Without this, the refusal above would also pass against a guard that
    // refuses every ladder outright, or one whose comparison is inverted.
    // Four levels summing to precisely MESH_DECODE_BUDGET_BYTES — the leaf's
    // own `> budget` boundary — must be ADMITTED.
    const per = Math.floor(MESH_DECODE_BUDGET_BYTES / 4);
    const last = MESH_DECODE_BUDGET_BYTES - 3 * per; // exact sum === budget
    const subs = [
      subLoader(level(4, [0, 1, 2]), { accountedBytes: per }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: per }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: per }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: last }),
    ];
    const loader = new MeshProgressiveLoader(subs, subs.length, '/surf');

    const data = await loader.updateView(VIEW);
    expect(data.faceCount).toBeGreaterThan(0);
  });

  it('refuses BEFORE any level is fetched', async () => {
    const subs = [
      subLoader(level(4, [0, 1, 2]), { accountedBytes: MESH_DECODE_BUDGET_BYTES }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: MESH_DECODE_BUDGET_BYTES }),
    ];
    const loader = new MeshProgressiveLoader(subs, subs.length, '/surf');

    await expect(loader.updateView(VIEW)).rejects.toThrow(LoaderError);
    // The refusal must cost no fetch — the streaming loop never ran.
    for (const sub of subs) expect(sub.calls).toBe(0);
  });

  it('runs the gate ONCE for a ladder that passes — a later slice scrub costs nothing', async () => {
    const subs = [subLoader(level(4, [0, 1, 2])), subLoader(level(3, [0, 1, 2]))];
    const loader = new MeshProgressiveLoader(subs, subs.length, '/surf');

    await loader.updateView(VIEW);
    await loader.updateView({ ...VIEW, slicePosition: [1, 1, 1] });

    for (const sub of subs) expect(sub.preflightCalls).toBe(1);
  });

  it("propagates a level's own runPreflight() rejection unchanged, and does not latch the ladder", async () => {
    // The transient-blip path the placement (inside the contained
    // `loadMeshNodeExpensive` try, in production) exists to get right: the
    // level's own error surfaces AS ITSELF, not rewritten into a ladder-wide
    // message and not swallowed — and the ladder is not admitted, so a later
    // retry (here: the flaky level recovering) genuinely re-checks rather than
    // replaying a cached pass.
    const failure = new Error('simulated transient network failure');
    const flaky = subLoader(level(4, [0, 1, 2]), { preflightError: failure });
    const subs = [flaky, subLoader(level(3, [0, 1, 2]))];
    const loader = new MeshProgressiveLoader(subs, subs.length, '/surf');

    await expect(loader.updateView(VIEW)).rejects.toBe(failure);
    // Nothing was fetched — the rejection happened at the gate, before the
    // streaming loop.
    for (const sub of subs) expect(sub.calls).toBe(0);

    // The flaky level "recovers" (a real retry re-enters `initialize()` fresh,
    // since a failed attempt caches nothing); a later updateView must re-run
    // the gate rather than replay the earlier rejection or skip it as already
    // checked.
    flaky.runPreflight = vi.fn(async () => ({
      accountedBytes: 1,
      nVertices: 0,
      nFaces: 0,
      ndim: 0,
    }));
    const data = await loader.updateView(VIEW);
    expect(data.faceCount).toBeGreaterThan(0);
  });

  it('a dispose() racing the gate whose in-flight runPreflight REJECTS still resolves with the empty payload', async () => {
    // The shape the REAL sub-loader produces, and the one a post-await
    // `_disposed` check alone cannot cover: `MeshProgressiveLoader.dispose()`
    // disposes every level and `MeshWholeNodeLoader.dispose()` bumps its
    // generation and nulls `preflight`, so the in-flight `runPreflight()` throws
    // `LoaderError('Unexpected', …, 'mesh loader not initialized')` — the
    // `Promise.all` REJECTS instead of resolving late.
    //
    // The mutation this kills: delete the catch's `if (this._disposed) return;`
    // and `updateView` REJECTS with that teardown `LoaderError` instead of
    // resolving with the ladder's empty payload — a dataset switch mis-counted
    // as a load failure, the very outcome the streaming loop's two `_disposed`
    // re-checks exist to prevent. The `faceCount === 0` assertion below IS the
    // test.
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const tornDown = new LoaderError(
      'Unexpected',
      '/surf/additive_0',
      new Error('mesh loader not initialized')
    );
    const subs = [
      subLoader(level(4, [0, 1, 2]), {
        preflightGate: gate,
        preflightErrorAfterDispose: tornDown,
      }),
      subLoader(level(3, [0, 1, 2])),
    ];
    const loader = new MeshProgressiveLoader(subs, subs.length, '/surf');

    const inFlight = loader.updateView(VIEW);
    loader.dispose();
    release();

    const data = await inFlight;
    expect(data.faceCount).toBe(0);
  });

  it('a dispose() racing the gate on an OVER-BUDGET ladder resolves (empty payload) rather than rejecting', async () => {
    // The RESOLVE shape of the dispose race (its rejection sibling is the test
    // above): this stub keeps resolving after `dispose()`, which the real
    // sub-loader does not do, so what this pins is the post-await `_disposed`
    // re-check on the aggregate comparison. Without the test, "make
    // the post-await `_disposed` re-check a no-op" survives, because a ladder
    // whose levels are all comfortably under budget can't tell "returned early"
    // from "computed the sum and it happened to pass". Push every level's own
    // `accountedBytes` over the ceiling so the two paths diverge — dropping the
    // re-check would have this reject with the budget `LoaderError` instead of
    // resolving, throwing a refusal at a dataset that is already gone.
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const subs = [
      subLoader(level(4, [0, 1, 2]), {
        accountedBytes: MESH_DECODE_BUDGET_BYTES,
        preflightGate: gate,
      }),
      subLoader(level(3, [0, 1, 2]), { accountedBytes: MESH_DECODE_BUDGET_BYTES }),
    ];
    const loader = new MeshProgressiveLoader(subs, subs.length, '/surf');

    const inFlight = loader.updateView(VIEW);
    loader.dispose();
    release();

    // Must RESOLVE, not reject: a torn-down node must not throw a budget refusal
    // at a dataset nothing will read. The empty payload is the ladder's own
    // "nothing committed" answer, from the streaming loop's own `_disposed`
    // break — not a side effect of the budget check being skipped.
    const data = await inFlight;
    expect(data.faceCount).toBe(0);
  });
});

testLadderFoldContract('Mesh', async () => {
  const levels = [
    level(6, [0, 1, 2]),
    level(6, [1, 2, 3], { base: 6 }),
    level(6, [2, 3, 4], { base: 12 }),
  ];
  const subs = levels.map((d) => subLoader(d, { resident: true }));
  const l = new MeshProgressiveLoader(subs, levels.length, '/fold');
  const multiPassSubs = levels.map((data) => subLoader(data, { resident: false }));
  const multiPassLoader = new MeshProgressiveLoader(
    multiPassSubs,
    levels.length,
    '/fold-multi-pass'
  );
  return {
    loader: l,
    totalLevels: levels.length,
    loadAll: async () => {
      await l.updateView(VIEW);
    },
    multiPass: {
      loader: multiPassLoader,
      totalLevels: levels.length,
      loadAll: async () => {
        await multiPassLoader.updateView(VIEW);
      },
    },
  };
});
