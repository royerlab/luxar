/**
 * Integration test for issue #1535: a lazy substitutive mesh child with an
 * additive reveal ladder must actually DRAIN that ladder, not freeze at its
 * first commit.
 *
 * `load-lod-group-node.ts` cheap-attaches every non-default `kind=lod` child
 * and defers its expensive fetch to the registry (`attachLazyChild`); a mesh
 * child that ALSO carries `n_additive_sublods > 1` is laddered underneath that
 * deferral — two independent lazy mechanisms stacked on the same node. The
 * mesh branch of `attachLazyChild` passes a `hasMoreLODs` probe (reading the
 * real `MeshProgressiveLoader.hasMoreLODs` getter) as its final argument, and
 * `LODGroupRegistry.evaluateEntry` re-fires `ensureLoaded` (settle-gated, see
 * `FINE_RELOAD_SETTLE_TICKS`) for as long as that probe reports `true`. Drop
 * that probe (pass `undefined` instead, the pre-#1499 shape) and the level
 * still gets its FIRST commit — the initial not-ready kick that promotes the
 * level from cheap-attached to displayed is unconditional — but nothing ever
 * asks for a second one: `hasMoreLODs?.() ?? false` is permanently `false`, so
 * the ladder is stuck at whatever its first streaming pass happened to load. A
 * mesh reveal ladder is a growing SUBSET OF FACES — one connected patch grown
 * best-first through face adjacency, keyed on radius from the reveal centre
 * (spec §9.1, explicitly not a radius sort), not a decimation — so that stuck
 * state is a permanently holed surface, not a blurrier-but-complete one.
 *
 * ## What is real here, and what is not
 *
 * Real: `loadLodGroupNode` (cheap-attach + defer dispatch + registry wiring),
 * `loadMeshNodeCheap` / `loadMeshNodeExpensive` (NOT mocked — unlike the
 * sibling unit test `load-lod-group-node.test.ts`, which stubs that module out
 * via `loadMeshNodeCheapMock`), the real `MeshProgressiveLoader` (streaming
 * loop, concat, `hasMoreLODs` getter), and the real `LODGroupRegistry`
 * selector + settle + re-fire logic. Stubbed: the zarr/network edge
 * (`loader-factory.ts`, mocked below), AND every orchestrator callback on
 * `NodeBuildCtx` (`processMeshData`, `commitMeshGeometry`, the node factory,
 * etc. — supplied by `makeTestNodeBuildCtx`'s `vi.fn()` defaults, with only
 * the handful this test actually reads overridden). The whole ctx is a fake;
 * it is not "everything except the zarr edge."
 *
 * That sibling module-mocks `load-mesh-node` for an unrelated reason (it is a
 * `load-lod-group-node.ts` DISPATCH test, not a loader test) — it is NOT
 * blind to the #1535 probe. Its two `hasMoreLODs` tests
 * (`load-lod-group-node.test.ts:998` and `:1103`) cover the probe's two
 * cases from opposite sides: `:998` overrides the mocked `loadMeshNodeCheap`
 * to return a loader stub `{ hasMoreLODs: true }` (a laddered level, still
 * streaming); `:1103` takes no override at all and gets the shared default
 * `cheapImpl` (`:84-89`), whose loader is a bare `{}` with no `hasMoreLODs`
 * property (an unladdered level — the probe's negative case). Either way, the
 * probe at `load-lod-group-node.ts:426` reads `hasMoreLODs` straight off
 * whatever loader the cheap half handed back, so both tests assert
 * `typeof … .hasMoreLODs === 'function'` (`:1027` / `:1124`) and both fail
 * exactly like this one does if that probe is passed `undefined`. What the
 * sibling genuinely cannot see is the DRAIN: neither loader stub is a real
 * `MeshProgressiveLoader`, `ensureLoaded` there only reaches the mocked,
 * no-op `loadMeshNodeExpensiveMock`, and neither test ever calls
 * `registry.evaluatePerFrame()` — `:998` flips its stub's `hasMoreLODs` field
 * by hand and reads the thunk directly, rather than driving a real re-fire.
 * This file is the only one that proves repeated settle-gated re-fires
 * actually walk a real ladder to completion.
 *
 * That real/fake distinction is load-bearing, not just accurate bookkeeping:
 * see the comment on the `createEmptyMeshNode` override below, which explains
 * why one specific property of that fake — NOT stamping `userData.nodeType` —
 * is what isolates the `hasMoreLODs` probe as the thing under test, rather
 * than the unrelated staleness path also live in the registry's re-fire gate.
 *
 * ## Why the stub sub-loaders report `allResident: false`
 *
 * `MeshWholeNodeLoader.updateViewWithResidency` samples its residency flag
 * BEFORE the fetch it is about to make, specifically so that "a level this
 * very call fetches reports false" (see that method's own doc). Since each
 * ladder level's sub-loader is asked for its data exactly once per node
 * lifetime (the streaming loop's `startLevel` only ever advances past a level
 * once it is loaded), a REAL sub-loader always reports `allResident: false`
 * on its one-and-only call — there is no such thing as a "warm" first fetch.
 * Reporting `allResident: true` instead (which looks like the simpler,
 * more "obviously fast and deterministic" choice) would make this test
 * VACUOUS: `shouldStopAfterLevel`'s `refine`-pass rule only stops a streaming
 * pass at a level that is NOT resident or was slow, so with every level
 * reported resident the very first (unconditional, `hasMoreLODs`-INDEPENDENT)
 * "not-ready" kick drains the entire three-level ladder in one pass, and the
 * `hasMoreLODs`-gated second pass this test exists to cover would never run —
 * with or without the #1535 fix. `allResident: false` is both the faithful
 * and the discriminating choice: the first pass loads levels 0 and 1 (the
 * `level > startLevel` stop check never fires for `level === startLevel`, so
 * the first level of any pass always lands), then stops; only a SECOND,
 * settle-gated pass — driven purely by `hasMoreLODs` — loads level 2 and
 * completes the surface.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

const { createMeshLoaderMock, createProgressiveMeshLoaderMock } = vi.hoisted(() => ({
  createMeshLoaderMock: vi.fn(),
  createProgressiveMeshLoaderMock: vi.fn(),
}));
vi.mock('../../../../../data/scene-loader/loaders/loader-factory', () => ({
  createMeshLoader: createMeshLoaderMock,
  createProgressiveMeshLoader: createProgressiveMeshLoaderMock,
}));

import { loadLodGroupNode } from '../../../../../data/scene-loader/nodes/load-lod-group-node';
import type { LoadSceneChildren } from '../../../../../data/scene-loader/nodes/load-children-concurrently';
import { loadMeshNode } from '../../../../../data/scene-loader/nodes/load-mesh-node';
import { LODGroupRegistry } from '../../../../../scene/lod-group-registry';
import { MeshProgressiveLoader } from '../../../../../data/mesh/mesh-progressive-loader';
import { makeTestNodeBuildCtx } from '../../../../helpers/make-test-node-build-ctx';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';
import type { LoadedMeshData, MeshDataLoader } from '../../../../../types/mesh';
import type { MeshWholeNodeLoader } from '../../../../../data/mesh/mesh-whole-node-loader';

// ────────────────────────────────────────────────────────────────────────
// Fixtures
// ────────────────────────────────────────────────────────────────────────

/**
 * Production wires both the leaf loaders' `ctx.getViewVersion()` (the
 * per-commit `loadedViewVersion` stamp) and the `LODGroupRegistry`'s own
 * `getViewVersion` dep from the SAME `_updateVersion` counter. This constant
 * keeps the registry's copy (`makeMeshLadderRegistry`, below) textually
 * pinned to that one value; the ctx side relies on `makeTestNodeBuildCtx`'s
 * own default (`() => 1`) instead of repeating it as an override — the test
 * body asserts the two agree right after the ctx is built, so a future change
 * to that default cannot silently decouple them without failing loudly here.
 */
const VIEW_VERSION = 1;

/**
 * One ladder level's decoded payload: `faceCount` independent triangles (no
 * shared vertices — irrelevant to the offset-adjusting concat this test
 * exercises, which only cares that each level's face indices are LOCAL to its
 * own vertex array and get renumbered on concat).
 */
function levelData(faceCount: number): LoadedMeshData {
  const vertexCount = faceCount * 3;
  const vertices = new Float32Array(vertexCount * 3);
  for (let v = 0; v < vertexCount; v++) {
    vertices[v * 3] = v;
    vertices[v * 3 + 1] = v + 0.1;
    vertices[v * 3 + 2] = v + 0.2;
  }
  const faces = new Uint32Array(faceCount * 3);
  for (let i = 0; i < faces.length; i++) faces[i] = i; // local indices [0, vertexCount)
  return {
    vertices,
    faces,
    normals: null,
    colors: null,
    vertexCount,
    faceCount,
    ndim: 3,
  };
}

/**
 * A `MeshWholeNodeLoader`-shaped stub for one ladder level: the four methods
 * `MeshProgressiveLoader` calls — the `updateViewWithResidency` its streaming
 * loop fetches through, the `runPreflight` its aggregate byte-budget gate calls
 * once per level before any level is fetched, the `releaseData` it calls after
 * folding a level into the cumulative payload, and the `dispose` it forwards
 * to every level on teardown. Always reports
 * `allResident: false` — see the module doc for why that is the faithful
 * (not merely convenient) choice.
 */
function stubLevelLoader(faceCount: number): MeshWholeNodeLoader {
  const data = levelData(faceCount);
  return {
    // The preflight figure is derived from the payload this stub actually hands
    // back rather than invented, but its exact value is immaterial here: nine
    // tiny triangles sit many orders of magnitude under the per-node budget, so
    // this test exercises the ladder DRAIN, not the budget gate (which has its
    // own coverage in mesh-progressive-loader.test.ts).
    runPreflight: vi.fn(async () => ({
      nVertices: data.vertexCount,
      nFaces: data.faceCount,
      ndim: data.ndim,
      accountedBytes: data.vertices.byteLength + data.faces.byteLength,
    })),
    updateViewWithResidency: vi.fn(async () => ({ data, allResident: false })),
    releaseData: vi.fn(),
    dispose: vi.fn(),
  } as unknown as MeshWholeNodeLoader;
}

/** The fine child's three-level reveal ladder: 2 + 3 + 4 = 9 faces total. */
const FINE_LEVEL_FACE_COUNTS = [2, 3, 4];
const FINE_TOTAL_FACES = FINE_LEVEL_FACE_COUNTS.reduce((a, b) => a + b, 0);

/** The coarse (eager, non-laddered) child's single-shot payload. */
const COARSE_FACE_COUNT = 1;

const LOD_PATH = '/lod';
const COARSE_PATH = '/lod/coarse';
const FINE_PATH = '/lod/fine';

/** A leaf mesh child node — coarse (no ladder) or fine (3-level ladder). */
function meshChildNode(path: string, coverageFraction: number, additiveSubLods: number): SceneNode {
  return {
    path,
    type: 'mesh',
    attrs: {
      type: 'mesh',
      coverage_fraction: coverageFraction,
      position_bounds: { min: [0, 0, 0], max: [10, 10, 10] },
      ...(additiveSubLods > 1 ? { n_additive_sublods: additiveSubLods } : {}),
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

function lodGroupNode(children: SceneNode[]): SceneNode {
  return {
    path: LOD_PATH,
    type: 'group',
    attrs: {
      type: 'group',
      kind: 'lod',
      display_type: 'mesh',
      selector: 'coverage',
      default_level: 0,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children,
  };
}

/** Full microtask + macrotask drain — lets a fire-and-forget async chain land. */
async function flush(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/** Minimal stub for the zarr `Location` surface: only `resolve()` is ever called. */
function makeStubLoc(): never {
  return { resolve: () => makeStubLoc() } as never;
}

/**
 * Registry wired the same way `lod-group-registry.test.ts`'s
 * `makeRegistry`/never-downgrade tests are: an identity camera maps world
 * coordinates straight to NDC, so a [0,0,0]-[10,10,10] box with thresholds
 * [0, 0.5] resolves the fine child as `desired` from the very first frame — a
 * proven-working pattern, not something re-derived here. `getViewVersion` is
 * the fixed `VIEW_VERSION` constant so the settle tracker genuinely counts
 * quiet frames (`FINE_RELOAD_SETTLE_TICKS` ~ 8) rather than being permanently
 * inert.
 */
function makeMeshLadderRegistry(): LODGroupRegistry {
  const camera = new THREE.Camera();
  camera.matrixWorldInverse.identity();
  camera.projectionMatrix.identity();
  return new LODGroupRegistry({
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => [0, 1, 2],
    getViewVersion: () => VIEW_VERSION,
  });
}

beforeEach(() => {
  createMeshLoaderMock.mockReset();
  createProgressiveMeshLoaderMock.mockReset();

  // Eager/coarse child: a plain (non-laddered) whole-node loader stub.
  createMeshLoaderMock.mockImplementation(
    (): MeshDataLoader =>
      ({
        loadMesh: vi.fn(async () => levelData(COARSE_FACE_COUNT)),
        updateView: vi.fn(async () => levelData(COARSE_FACE_COUNT)),
        dispose: vi.fn(),
      }) as unknown as MeshDataLoader
  );

  // Lazy/fine child: the REAL MeshProgressiveLoader over three stub levels —
  // this is the object under test, not a fake of it.
  createProgressiveMeshLoaderMock.mockImplementation(
    async (node: SceneNode, nAdditive: number): Promise<MeshDataLoader> => {
      const subLoaders = FINE_LEVEL_FACE_COUNTS.map((count) => stubLevelLoader(count));
      return new MeshProgressiveLoader(subLoaders, nAdditive, node.path);
    }
  );
});

describe('lazy substitutive mesh child with an additive reveal ladder (#1535)', () => {
  it('drains its reveal ladder to completion via registry-driven re-fires, never joining the sweep', async () => {
    // Records every commit as { path, faceCount } in commit order, across
    // BOTH children — the coarse eager commit and however many fine commits
    // the ladder needs.
    const commits: { path: string; faceCount: number }[] = [];

    // Hoisted above the ctx so it can be passed as a normal `lodGroupRegistry`
    // override (a first-class optional member of `NodeBuildCtx`) rather than
    // bolted on afterwards through a cast.
    const registry = makeMeshLadderRegistry();

    // Only the members this test OVERRIDES are named here; everything else
    // (`applyEffectiveAttrs`, `isDatasetLive`, `connectLoaderToMonitor`,
    // `kickRefinementIfIdle`, `releaseLazyMesh`, the node factory's
    // `markPickingDirty`, ...) comes from the shared factory's defaults — see
    // make-test-node-build-ctx.ts. `getViewVersion` and `registry` ARE read
    // below (the `VIEW_VERSION` assertion; the two spies) but neither needs
    // an override: `getViewVersion` already defaults to `() => 1`, exactly
    // `VIEW_VERSION`, and `registry` defaults to a real `LoaderRegistry` the
    // spies wrap in place.
    const ctx: NodeBuildCtx = makeTestNodeBuildCtx({
      viewState: { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      lodGroupRegistry: registry,
      processMeshData: vi.fn(async (path: string, data: LoadedMeshData) => ({
        path,
        data,
        projected: { visibleFaceCount: data.faceCount },
      })) as unknown as NodeBuildCtx['processMeshData'],
      commitMeshGeometry: vi.fn((staged: unknown) => {
        const s = staged as { path: string; projected: { visibleFaceCount: number } };
        commits.push({ path: s.path, faceCount: s.projected.visibleFaceCount });
      }),
    });

    // Pins the coupling `VIEW_VERSION`'s own doc claims: production feeds the
    // ctx's view version and the registry's from the same counter, so this
    // test's two copies must agree too — not just today, by coincidence of
    // two literals, but enforced here.
    expect(ctx.getViewVersion()).toBe(VIEW_VERSION);

    // LOAD-BEARING, do not "faithfully" stamp this. Production's real
    // `createEmptyMeshNode` (create-mesh-node.ts) stamps
    // `userData.nodeType = 'mesh'` on the placeholder; this fake deliberately
    // does NOT. That absence is what keeps `isTrackedLeaf` false for this
    // child (lod-freshness.ts:91-94), which routes the registry's
    // `childFreshAndCount` to the untracked-subtree branch instead of reading
    // any `loadedViewVersion` stamp: `countFromUserData` returns `null` for an
    // untracked `nodeType` (lod-freshness.ts:145-165), so the subtree fold
    // (`foldProgress`) finds nothing and `subtreeDisplayProgress` returns
    // `null` (lod-display-gate.ts:201) — which `childFreshAndCount` then
    // reports as `{ fresh: true, count: null }` (lod-group-registry.ts:1138)
    // for any READY child with no tracked leaves under it (a not-ready child
    // short-circuits to `fresh: false` first, lod-group-registry.ts:1131).
    // That keeps `aspirationFresh` permanently `true` once this child is
    // ready, in the registry's re-fire gate (`aspirationReady &&
    // (!aspirationFresh || hasMoreLODs())`, lod-group-registry.ts ~991), so
    // `hasMoreLODs` is the SOLE thing that can make that disjunction true —
    // exactly what this test needs to isolate the #1535 probe. Stamping a
    // `loadedViewVersion` in the `commitMeshGeometry` fake above would do
    // NOTHING on its own (that fake has no handle on this placeholder to
    // stamp it onto, and it would go unread anyway without the `nodeType`
    // stamp). Stamping `nodeType` ALONE, though, opens the `!aspirationFresh`
    // door: nothing here ever writes `loadedViewVersion`, so the child would
    // read stale forever and the registry would re-fire `ensureLoaded` every
    // settled frame regardless of `hasMoreLODs`.
    //
    // Two different scenarios, kept separate. Against the ACTUAL #1535
    // regression (the probe below passed `undefined`), assertion 1
    // (`hasMoreLODs` is a function) still correctly fails — it is a
    // structural fact about the 7th `attachLazyChild` argument, untouched by
    // freshness. Against the HYPOTHETICAL `nodeType`-stamped fake instead
    // (probe still deleted), assertion 1 would PASS (freshness doesn't touch
    // it either), assertion 2 (the ladder draining to the full face count)
    // would stop discriminating — the real drain would still happen, just
    // driven by staleness instead of the probe — and assertion 3 (no more
    // commits once complete) would then fail on its own, since a
    // permanently-stale child never stops re-firing.
    vi.mocked(ctx.nodeFactory.createEmptyMeshNode).mockImplementation((path: string) => {
      const mesh = new THREE.Mesh();
      mesh.name = path;
      return mesh;
    });

    const registerMeshLoaderSpy = vi.spyOn(ctx.registry, 'registerMeshLoader');
    const recordFailureSpy = vi.spyOn(ctx.registry, 'recordFailure');

    // The generic recursion `loadLodGroupNode` calls for its EAGER default
    // child (the lazy children are dispatched directly, without this
    // callback). Routes straight through the real `loadMeshNode` so the
    // coarse level goes through the identical cheap+expensive+register path
    // as production, just like the lazy fine child does.
    const loadChildren: LoadSceneChildren = async (node, parentThree, loc, nodeCtx) => {
      await loadMeshNode(node, parentThree, loc, nodeCtx);
    };

    const sceneRoot = new THREE.Group();
    const node = lodGroupNode([meshChildNode(COARSE_PATH, 0, 0), meshChildNode(FINE_PATH, 0.5, 3)]);

    await loadLodGroupNode(node, sceneRoot, makeStubLoc(), ctx, loadChildren);

    // ── Sanity: the coarse eager child committed once and registered. ──
    expect(commits).toEqual([{ path: COARSE_PATH, faceCount: COARSE_FACE_COUNT }]);
    expect(registerMeshLoaderSpy).toHaveBeenCalledWith(COARSE_PATH, expect.anything());

    const entry = registry.get(LOD_PATH);
    expect(entry).toBeDefined();
    const fineChild = entry!.children[1];

    // ── Assertion 1: the lazy child exposes a live hasMoreLODs thunk. ──
    // This is exactly what a `hasMoreLODs: undefined` regression breaks — see
    // the module doc above.
    expect(typeof fineChild.hasMoreLODs).toBe('function');
    expect(fineChild.hasMoreLODs!()).toBe(true); // nothing loaded yet — 3 sub-LODs outstanding

    // ── Drive the registry: identity camera selects the fine child from
    // frame 1 (unconditional "not ready" kick — loads levels 0+1, 5/9
    // faces); the settle gate (~8 quiet frames) then lets a second,
    // hasMoreLODs-driven pass load level 2 and complete the ladder. Loop well
    // past that so a stuck ladder is unambiguous, not a timing near-miss. ──
    for (let i = 0; i < 20; i++) {
      registry.evaluatePerFrame();
      await flush();
    }

    const fineCommits = commits.filter((c) => c.path === FINE_PATH).map((c) => c.faceCount);

    // ── Assertion 2: a genuine PARTIAL first commit, followed by at least one
    // further re-fire that reaches the FULL surface. This is the assertion
    // that actually pins the settle-gated re-fire under test — the thing this
    // file adds over the sibling unit test. Without the `< FINE_TOTAL_FACES`
    // and `> 1` checks, simplifying `stubLevelLoader` to `allResident: true`
    // (exactly what the module doc above warns against) would make the
    // unconditional first kick drain all three levels in one pass,
    // `fineCommits` would collapse to `[9]`, and every remaining assertion
    // would still pass — proving nothing about the re-fire. This does NOT pin
    // the exact per-fire step beyond that first partial commit: a refine pass
    // may legitimately drain more than one level per fire, so a real second
    // pass could in principle also be the last one; it only has to be more
    // than the first. ──
    expect(fineCommits.length).toBeGreaterThan(1);
    expect(fineCommits[0]).toBeLessThan(FINE_TOTAL_FACES);
    for (let i = 1; i < fineCommits.length; i++) {
      expect(fineCommits[i]).toBeGreaterThanOrEqual(fineCommits[i - 1]);
    }
    expect(fineCommits[fineCommits.length - 1]).toBe(FINE_TOTAL_FACES);

    // ── Assertion 3: complete means complete — no further commits, no
    // runaway per-frame refinement once the ladder is drained. ──
    expect(fineChild.hasMoreLODs!()).toBe(false);
    const commitCountAtCompletion = commits.length;
    for (let i = 0; i < 5; i++) {
      registry.evaluatePerFrame();
      await flush();
    }
    expect(commits.length).toBe(commitCountAtCompletion);

    // ── Assertion 4: the lazy level never joined the per-slice sweep — the
    // registry is the ONLY thing driving its ladder. Without this, assertion
    // 2 would be meaningless: a sweep-registered loader could have been the
    // one advancing the ladder instead of the registry's settle-gated
    // re-fire, and this test would pass for the wrong reason. ──
    expect(registerMeshLoaderSpy).not.toHaveBeenCalledWith(FINE_PATH, expect.anything());

    // Guards against the lazy expensive half silently throwing mid-ladder,
    // which would also stop the drain — but for a reason having nothing to do
    // with the #1535 probe. Without this, a broken `loadMeshNodeExpensive`
    // could masquerade as "hasMoreLODs was never wired".
    expect(recordFailureSpy).not.toHaveBeenCalled();
  });
});
