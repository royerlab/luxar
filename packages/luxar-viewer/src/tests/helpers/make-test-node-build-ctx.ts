/**
 * Shared test factory for {@link NodeBuildCtx}, the context the per-geometry
 * leaf loaders (`load-points-node`, `load-lines-node`, `load-gsplats-node`,
 * `load-mesh-node`) read from, along with the group loaders that forward it
 * (`load-lod-group-node`, `load-partition-group-node`, `load-scene-nodes`).
 *
 * Named `makeTestNodeBuildCtx`, not `makeNodeBuildCtx`, because production
 * already owns the latter name: `SceneLoader.makeNodeBuildCtx()` builds the
 * real ctx per call into `loadSceneNodes`. Same shape, different provenance —
 * keeping the names distinct means a reference to either one resolves.
 *
 * WHY this exists: every member of `NodeBuildCtx` except the optional
 * `lodGroupRegistry` is required, so a hand-rolled ctx literal in a test file
 * must name all 22 of them. Five spec files each carried their own copy, which
 * meant adding ONE member to the interface broke all five with the same
 * one-line edit — and because neither `eslint` nor `vitest` typechecks, those
 * files kept passing lint and passing their own runs while `tsc` was red.
 * (That is exactly how `releaseLazyMesh` landed: one production member, four
 * identical test-file fixes — the fifth copy, in `load-lod-group-node.test.ts`,
 * already named it, being the spec the new behaviour was written for.) With
 * this factory the new member is added in exactly ONE place.
 *
 * The four `releaseLazy*` peers in particular must stay interchangeable — they
 * are called from the same lod_group demotion path and a spec that stubs three
 * of them but not the fourth is a latent crash, not a compile error, once the
 * ctx is built behind a cast.
 *
 * Call sites keep their EXPLICITNESS where it matters: a spec still names the
 * handful of members its assertions read (its own spies, its own `nodeFactory`,
 * its own `processXData` resolution) as `overrides`. The factory only absorbs
 * the members nobody in that file cares about.
 *
 * Vitest-only module — imports `vi`, so it must not be imported from a
 * Playwright spec or a plain script.
 */

import { vi } from 'vitest';
import { LoaderRegistry } from '../../data/scene-loader/loaders/loader-registry';
import type { NodeBuildCtx } from '../../data/scene-loader/nodes/build-ctx';
import type { SceneNode, ViewState } from '../../data/data-loader-types';
import { createLineWorkingSetGate } from '../../data/scene-loader/nodes/load-children-concurrently';

/** The ctx `viewState` a spec gets when it does not supply one. */
function defaultViewState(): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, 0],
    tolerance: [0, 0, 0, 1],
    dimensions: undefined,
  };
}

/**
 * Build a fully-populated {@link NodeBuildCtx} of `vi.fn()` stubs, with
 * `overrides` applied last.
 *
 * Notable defaults:
 *  - `registry` is a REAL {@link LoaderRegistry} — the leaf specs assert on
 *    `registry.loaders` / `failedLoaders` membership, so a stub would not do.
 *  - `deriveNodeViewState` resolves to the EFFECTIVE view state, i.e.
 *    `overrides.viewState` when one is supplied and the default otherwise, so
 *    the derived state and `ctx.viewState` never disagree. That matches what
 *    the production `deriveNodeViewState` yields for a node with no
 *    `extend_to_all` and no `nd_transform` (production returns an equal copy
 *    rather than the same object; no spec asserts on the identity either way).
 *  - the lines/gsplats processors resolve `null` — their real "nothing staged"
 *    branch. `processMeshData` has no null branch by contract (and
 *    `load-mesh-node` dereferences the staged commit with no truthiness guard),
 *    so there is no honest zero-value to default it to; no spec currently
 *    reaches this default — the mesh spec supplies its own resolved commit — so
 *    it is left a bare stub.
 *  - `lodGroupRegistry` is optional on the interface and therefore has NO
 *    default — it comes only from `overrides`, so nothing forces a spec to
 *    fabricate a registry it never reads. Only `load-lod-group-node` reads it,
 *    and its spec always supplies one.
 */
export function makeTestNodeBuildCtx(overrides: Partial<NodeBuildCtx> = {}): NodeBuildCtx {
  const viewState = overrides.viewState ?? defaultViewState();

  const nodeFactory = {
    createEmptyPointsNode: vi.fn(),
    createEmptyLinesNode: vi.fn(),
    createEmptyGSplatsNode: vi.fn(),
    createEmptyMeshNode: vi.fn(),
    applyTransform: vi.fn(),
    markPickingDirty: vi.fn(),
  } as unknown as NodeBuildCtx['nodeFactory'];

  return {
    registry: new LoaderRegistry(),
    lineWorkingSetGate: createLineWorkingSetGate(),
    nodeFactory,
    viewState,
    getViewVersion: () => 1,
    factoryDeps: {} as never,
    isDatasetLive: () => true,
    applyEffectiveAttrs: vi.fn((node: SceneNode) => node.attrs),
    deriveNodeViewState: vi.fn(() => ({ skip: false as const, viewState })),
    connectLoaderToMonitor: vi.fn(),
    kickRefinementIfIdle: vi.fn(),
    reportArchiveFault: vi.fn(),
    releaseLazyGSplats: vi.fn(),
    releaseLazyPoints: vi.fn(),
    releaseLazyLines: vi.fn(),
    releaseLazyMesh: vi.fn(),
    processPointsData: vi.fn(),
    commitPointsGeometry: vi.fn(),
    processLinesData: vi.fn().mockResolvedValue(null),
    commitLinesGeometry: vi.fn(),
    processGSplatsData: vi.fn().mockResolvedValue(null),
    commitGSplatsGeometry: vi.fn(),
    processMeshData: vi.fn(),
    commitMeshGeometry: vi.fn(),
    ...overrides,
  };
}
