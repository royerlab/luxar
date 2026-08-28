/**
 * Unit tests for the retry helpers (`retryFailedLoaderUnlocked` and
 * `retryAllFailedLoadersUnlocked`) in `scene-loader/retry.ts`.
 *
 * The retry path's three load-bearing invariants:
 *   1. Verify-and-clear guard — if the named scene object has been
 *      removed between failure and retry, the data fetch is allowed
 *      to complete but the failure entry is NOT cleared from the
 *      registry. Without this guard, retry would falsely report
 *      success while the data has nowhere to land.
 *   2. A fully-extended node retries with the DERIVED extended-tolerance +
 *      pinned-slice view state (#1157) — the same slice-invariant query the
 *      initial-load path uses, so a failed fully-extended node re-fetches
 *      its whole extent.
 *   3. Per-attempt retryCount accounting — when a retry itself throws,
 *      the failedLoaders entry is updated with retryCount + 1 so the
 *      UI can surface "tried N times" diagnostics.
 *
 * All tests exercise the helper directly via a stub `RetryCtx` — the
 * orchestrator's `_updateInProgress` lock dance lives one layer above
 * and is out of scope here.
 */

import { describe, it, expect, vi } from 'vitest';
import { processPointsData } from '../../../../../data/scene-loader/process/data-processor-points';
import * as THREE from 'three';
import {
  retryFailedLoaderUnlocked,
  retryAllFailedLoadersUnlocked,
  type RetryCtx,
} from '../../../../../data/scene-loader/lifecycle/retry';
import { LoaderRegistry } from '../../../../../data/scene-loader/loaders/loader-registry';
import type {
  DataLoader,
  LoadedPointsData,
  ViewState,
} from '../../../../../data/data-loader-types';
import type { LinesDataLoader, LoadedLinesData } from '../../../../../types/lines';
import type { GSplatsDataLoader, LoadedGSplatsData } from '../../../../../types/gsplats';
import {
  createLineWorkingSetGate,
  EAGER_CHILD_LOAD_CONCURRENCY,
} from '../../../../../data/scene-loader/nodes/load-children-concurrently';

// ============================================================================
// Local fixtures — flat ctx-stubbing per the data/scene-loader test pattern.
// ============================================================================

function makeViewState(): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, 0],
    tolerance: [0, 0, 0, 1],
    dimensions: undefined,
  };
}

/**
 * Build a real `THREE.Group` containing a single named Mesh, so the
 * helper's `rootGroup.getObjectByName(path)` lookups exercise real
 * scene-graph traversal rather than a mocked getter.
 */
function makeRootGroupWith(path: string, attrs?: Record<string, unknown>): THREE.Group {
  const root = new THREE.Group();
  root.name = 'LuxarScene';
  const mesh = new THREE.Mesh();
  mesh.name = path;
  if (attrs) mesh.userData.attrs = attrs;
  root.add(mesh);
  return root;
}

/**
 * Build a `RetryCtx` whose registry and rootGroup are real, and whose
 * callbacks are individual vi.fn() spies that the test can assert against.
 */
function makeRetryCtx(overrides: Partial<RetryCtx> = {}): RetryCtx & {
  // Surface the spies for easy assertion.
  spies: {
    deriveNodeViewState: ReturnType<typeof vi.fn>;
    commitPointsGeometry: ReturnType<typeof vi.fn>;
    processLinesData: ReturnType<typeof vi.fn>;
    commitLinesGeometry: ReturnType<typeof vi.fn>;
    processGSplatsData: ReturnType<typeof vi.fn>;
    commitGSplatsGeometry: ReturnType<typeof vi.fn>;
  };
} {
  const viewState = makeViewState();
  const deriveNodeViewState = vi.fn(
    (_path: string, _attrs: unknown, _opts: { applyPartialExtendTolerance: boolean }) => ({
      skip: false as const,
      viewState,
    })
  );
  const commitPointsGeometry = vi.fn();
  const processLinesData = vi.fn().mockResolvedValue(null);
  const commitLinesGeometry = vi.fn();
  const processGSplatsData = vi.fn().mockResolvedValue(null);
  const commitGSplatsGeometry = vi.fn();
  // Mesh's processor never resolves null (no worker projection to decline), so the
  // stub returns a staged shape rather than the null the other two use.
  const processMeshData = vi.fn().mockResolvedValue({ path: '/m', data: {}, projected: {} });
  const commitMeshGeometry = vi.fn();

  const ctx: RetryCtx = {
    registry: new LoaderRegistry(),
    lineWorkingSetGate: createLineWorkingSetGate(),
    rootGroup: null,
    deriveNodeViewState,
    processPointsData,
    commitPointsGeometry,
    processLinesData,
    commitLinesGeometry,
    processGSplatsData,
    commitGSplatsGeometry,
    processMeshData,
    commitMeshGeometry,
    ...overrides,
  };
  return Object.assign(ctx, {
    spies: {
      deriveNodeViewState,
      commitPointsGeometry,
      processLinesData,
      commitLinesGeometry,
      processGSplatsData,
      commitGSplatsGeometry,
    },
  });
}

/** Minimal loader stubs — only the surface the helper actually calls. */
function makePointsLoader(
  updateView: (vs: ViewState) => Promise<LoadedPointsData | null>
): DataLoader {
  return { updateView } as unknown as DataLoader;
}
function makeLinesLoader(
  updateView: (vs: ViewState) => Promise<LoadedLinesData | null>
): LinesDataLoader {
  return { updateView } as unknown as LinesDataLoader;
}
function makeGSplatsLoader(
  updateView: (vs: ViewState) => Promise<LoadedGSplatsData | null>
): GSplatsDataLoader {
  return { updateView } as unknown as GSplatsDataLoader;
}

const PATH = '/scene/node';

// ============================================================================
// Tests
// ============================================================================

describe('retryFailedLoaderUnlocked — path-not-in-failed-loaders', () => {
  it('returns false immediately when the path is not in registry.failedLoaders', async () => {
    const ctx = makeRetryCtx();
    // Registry empty; no failure recorded.
    const ok = await retryFailedLoaderUnlocked(PATH, ctx);
    expect(ok).toBe(false);
    expect(ctx.spies.deriveNodeViewState).not.toHaveBeenCalled();
  });
});

describe('retryFailedLoaderUnlocked — Points loader success path', () => {
  it('updates geometry and clears the failure on success', async () => {
    const data = { pointCount: 7 } as unknown as LoadedPointsData;
    const updateView = vi.fn().mockResolvedValue(data);
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH),
    });
    ctx.registry.registerPointsLoader(PATH, makePointsLoader(updateView));
    ctx.registry.recordFailure(PATH, new Error('initial failure'));

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    expect(ok).toBe(true);
    expect(updateView).toHaveBeenCalledTimes(1);
    expect(ctx.spies.commitPointsGeometry).toHaveBeenCalledWith({ path: PATH, data });
    expect(ctx.registry.failedLoaders.has(PATH)).toBe(false);
    // Points uses applyPartialExtendTolerance: true (matches initial-load path).
    expect(ctx.spies.deriveNodeViewState).toHaveBeenCalledWith(PATH, undefined, {
      applyPartialExtendTolerance: true,
    });
  });

  it('does not commit points geometry when loader.updateView resolves null', async () => {
    const updateView = vi.fn().mockResolvedValue(null);
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH),
    });
    ctx.registry.registerPointsLoader(PATH, makePointsLoader(updateView));
    ctx.registry.recordFailure(PATH, new Error('initial failure'));

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    // verifyAndClear still runs — fetched (null) data still counts as a successful retry.
    expect(ok).toBe(true);
    expect(ctx.spies.commitPointsGeometry).not.toHaveBeenCalled();
    expect(ctx.registry.failedLoaders.has(PATH)).toBe(false);
  });
});

describe('retryFailedLoaderUnlocked — Lines loader paths', () => {
  const oversizedLineAttrs = {
    type: 'lines',
    n_vertices: 1_600_000,
    n_segments: 1_600_000,
    ndim: 3,
  };

  it('runs process → commit on a successful fetch', async () => {
    const linesData = { segmentCount: 4 } as unknown as LoadedLinesData;
    const stagedFromProcess = { path: PATH } as never;
    const updateView = vi.fn().mockResolvedValue(linesData);
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH),
    });
    ctx.spies.processLinesData.mockResolvedValue(stagedFromProcess);
    ctx.registry.registerLinesLoader(PATH, makeLinesLoader(updateView));
    ctx.registry.recordFailure(PATH, new Error('initial failure'));

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    expect(ok).toBe(true);
    const derivedViewState = ctx.spies.deriveNodeViewState.mock.results[0].value.viewState;
    expect(ctx.spies.processLinesData).toHaveBeenCalledWith(PATH, linesData, derivedViewState);
    expect(ctx.spies.commitLinesGeometry).toHaveBeenCalledWith(stagedFromProcess);
    // Lines variant: applyPartialExtendTolerance: false.
    expect(ctx.spies.deriveNodeViewState).toHaveBeenCalledWith(PATH, undefined, {
      applyPartialExtendTolerance: false,
    });
  });

  it('skips commit when processLinesData returns null but still clears the failure', async () => {
    const linesData = { segmentCount: 4 } as unknown as LoadedLinesData;
    const updateView = vi.fn().mockResolvedValue(linesData);
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH),
    });
    // processLinesData defaults to resolving null (set in makeRetryCtx).
    ctx.registry.registerLinesLoader(PATH, makeLinesLoader(updateView));
    ctx.registry.recordFailure(PATH, new Error('x'));

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    expect(ok).toBe(true);
    expect(ctx.spies.processLinesData).toHaveBeenCalledTimes(1);
    expect(ctx.spies.commitLinesGeometry).not.toHaveBeenCalled();
    expect(ctx.registry.failedLoaders.has(PATH)).toBe(false);
  });

  it('waits for shared line working-set admission before a single retry', async () => {
    let settleRetry!: (data: LoadedLinesData) => void;
    const updateView = vi.fn(
      () =>
        new Promise<LoadedLinesData>((resolve) => {
          settleRetry = resolve;
        })
    );
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH, oversizedLineAttrs),
    });
    ctx.registry.registerLinesLoader(PATH, makeLinesLoader(updateView));
    ctx.registry.recordFailure(PATH, new Error('allocation failed'));

    const releaseEagerAdmission = await ctx.lineWorkingSetGate.acquire({
      path: '/eager-line',
      type: 'lines',
      attrs: oversizedLineAttrs,
    });
    const retry = retryFailedLoaderUnlocked(PATH, ctx);

    await Promise.resolve();
    expect(updateView).not.toHaveBeenCalled();

    releaseEagerAdmission();
    await vi.waitFor(() => expect(updateView).toHaveBeenCalledTimes(1));
    settleRetry({ segmentCount: 0 } as unknown as LoadedLinesData);
    await expect(retry).resolves.toBe(true);
  });

  it('releases line working-set admission when a retry fails', async () => {
    const failingPath = '/line-failing';
    const nextPath = '/line-next';
    const root = new THREE.Group();
    for (const path of [failingPath, nextPath]) {
      const mesh = new THREE.Mesh();
      mesh.name = path;
      mesh.userData.attrs = oversizedLineAttrs;
      root.add(mesh);
    }

    let settleNext!: (data: LoadedLinesData) => void;
    const nextUpdateView = vi.fn(
      () =>
        new Promise<LoadedLinesData>((resolve) => {
          settleNext = resolve;
        })
    );
    const ctx = makeRetryCtx({ rootGroup: root });
    ctx.registry.registerLinesLoader(
      failingPath,
      makeLinesLoader(vi.fn().mockRejectedValue(new Error('still cannot allocate')))
    );
    ctx.registry.registerLinesLoader(nextPath, makeLinesLoader(nextUpdateView));
    ctx.registry.recordFailure(failingPath, new Error('allocation failed'));
    ctx.registry.recordFailure(nextPath, new Error('allocation failed'));

    await expect(retryFailedLoaderUnlocked(failingPath, ctx)).resolves.toBe(false);
    const nextRetry = retryFailedLoaderUnlocked(nextPath, ctx);

    await vi.waitFor(() => expect(nextUpdateView).toHaveBeenCalledTimes(1));
    settleNext({ segmentCount: 0 } as unknown as LoadedLinesData);
    await expect(nextRetry).resolves.toBe(true);
  });
});

describe('retryFailedLoaderUnlocked — fully-extended GSplats node', () => {
  it('retries with the derived extended-tolerance + pinned-slice view state (#1157)', async () => {
    const splatsData = { splatCount: 9 } as unknown as LoadedGSplatsData;
    const updateView = vi.fn().mockResolvedValue(splatsData);
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH, { extend_to_all: ['t', 'c'] }),
    });
    // A fully-extended node is derived as a normal node with a slice-invariant
    // query (extend-to-all tolerance + pinned slice); the retry must load with
    // it so the failed node re-fetches its whole extent.
    const extendedViewState = { ...makeViewState(), tolerance: [1e10, 1e10, 1e10, 1e10] };
    ctx.spies.deriveNodeViewState.mockReturnValue({
      skip: false,
      viewState: extendedViewState,
    });
    ctx.registry.registerGSplatsLoader(PATH, makeGSplatsLoader(updateView));
    ctx.registry.recordFailure(PATH, new Error('initial failure'));

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    expect(ok).toBe(true);
    expect(updateView).toHaveBeenCalledTimes(1);
    // Same reference passed straight through to the loader.
    expect(updateView.mock.calls[0][0]).toBe(extendedViewState);
    // Confirm derive was passed the node's extend_to_all attrs from the rootGroup mesh.
    expect(ctx.spies.deriveNodeViewState).toHaveBeenCalledWith(
      PATH,
      { extend_to_all: ['t', 'c'] },
      { applyPartialExtendTolerance: true }
    );
  });
});

describe('retryFailedLoaderUnlocked — derived view state reaches the loader intact', () => {
  // `deriveNodeViewState` sets `noPreimage` when a node's discrete
  // `nd_transform` maps the current world slice between grid points; each
  // spatial-index loader honours it by returning no ranges, so the node
  // renders nothing. A retry arm that rebuilds the view state field-by-field
  // drops the flag and commits geometry at a position that must stay empty —
  // and `verifyAndClear` then blesses that result.

  // `Promise<null>` is assignable to every loader's `Promise<LoadedX | null>`,
  // so one signature serves all three makers.
  type StubUpdateView = (vs: ViewState) => Promise<null>;
  const registerFor: Record<
    string,
    (registry: LoaderRegistry, updateView: StubUpdateView) => void
  > = {
    points: (registry, updateView) =>
      registry.registerPointsLoader(PATH, makePointsLoader(updateView)),
    lines: (registry, updateView) =>
      registry.registerLinesLoader(PATH, makeLinesLoader(updateView)),
    gsplats: (registry, updateView) =>
      registry.registerGSplatsLoader(PATH, makeGSplatsLoader(updateView)),
  };

  it.each(['points', 'lines', 'gsplats'])('forwards noPreimage for %s', async (kind) => {
    const updateView = vi.fn().mockResolvedValue(null);
    const ctx = makeRetryCtx({ rootGroup: makeRootGroupWith(PATH) });
    const derivedViewState: ViewState = { ...makeViewState(), noPreimage: true };
    ctx.spies.deriveNodeViewState.mockReturnValue({ skip: false, viewState: derivedViewState });
    registerFor[kind](ctx.registry, updateView);
    ctx.registry.recordFailure(PATH, new Error('initial failure'));

    await retryFailedLoaderUnlocked(PATH, ctx);

    expect(updateView).toHaveBeenCalledTimes(1);
    expect(updateView.mock.calls[0][0].noPreimage).toBe(true);
  });
});

describe('retryFailedLoaderUnlocked — verifyAndClear stale-scene guard', () => {
  it('returns false WITHOUT clearing the failure when the scene object is gone', async () => {
    const data = { pointCount: 7 } as unknown as LoadedPointsData;
    const updateView = vi.fn().mockResolvedValue(data);
    // rootGroup is a real group but does NOT contain a mesh named PATH —
    // the scene was reloaded or the node was programmatically removed
    // between failure and retry.
    const ctx = makeRetryCtx({
      rootGroup: new THREE.Group(),
    });
    ctx.registry.registerPointsLoader(PATH, makePointsLoader(updateView));
    ctx.registry.recordFailure(PATH, new Error('initial failure'));

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    expect(ok).toBe(false);
    // Data was fetched (the fetch is async + completes), commit was attempted...
    expect(updateView).toHaveBeenCalledTimes(1);
    expect(ctx.spies.commitPointsGeometry).toHaveBeenCalledTimes(1);
    // ...but the failure stays — verifyAndClear refused to clear it.
    expect(ctx.registry.failedLoaders.has(PATH)).toBe(true);
  });
});

describe('retryFailedLoaderUnlocked — no loader registered', () => {
  it('logs warning, deletes the stale failedLoaders entry, returns false', async () => {
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH),
    });
    // failedLoaders has an entry but no registry map has a loader for this path —
    // can happen if the loader was disposed mid-flight.
    ctx.registry.recordFailure(PATH, new Error('initial failure'));

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    expect(ok).toBe(false);
    expect(ctx.registry.failedLoaders.has(PATH)).toBe(false);
    // None of the per-type paths ran.
    expect(ctx.spies.commitPointsGeometry).not.toHaveBeenCalled();
    expect(ctx.spies.processLinesData).not.toHaveBeenCalled();
    expect(ctx.spies.processGSplatsData).not.toHaveBeenCalled();
  });
});

describe('retryFailedLoaderUnlocked — lazy LOD level fallback', () => {
  // Lazy substitutive levels never join the sweep maps but DO record
  // failures; previously the no-loader branch silently discarded them.
  it('kicks the lazy child via the LOD registry, keeps the record, returns true', async () => {
    const retryLazyChildByLeafPath = vi.fn().mockReturnValue(true);
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH),
      lodGroupRegistry: { retryLazyChildByLeafPath } as never,
    });
    ctx.registry.recordFailure(PATH, new Error('lazy load failed'));

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    expect(ok).toBe(true);
    expect(retryLazyChildByLeafPath).toHaveBeenCalledWith(PATH);
    // The record is KEPT across the kick — the fire-and-forget thunk owns the
    // outcome (success clears it; a repeat failure re-records). Deleting it
    // here reset autoRetryCount, so MAX_AUTO_RETRY_ATTEMPTS never bound a
    // permanently-failing lazy level.
    expect(ctx.registry.failedLoaders.has(PATH)).toBe(true);
  });

  it('preserves autoRetryCount across a lazy kick so the auto-retry budget binds', async () => {
    const retryLazyChildByLeafPath = vi.fn().mockReturnValue(true);
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH),
      lodGroupRegistry: { retryLazyChildByLeafPath } as never,
    });
    // A network failure that has already burned two automatic attempts.
    ctx.registry.recordFailure(PATH, new Error('lazy 404'), 'Network');
    ctx.registry.markAutoRetryAttempt(PATH);
    ctx.registry.markAutoRetryAttempt(PATH);
    expect(ctx.registry.failedLoaders.get(PATH)!.autoRetryCount).toBe(2);

    await retryFailedLoaderUnlocked(PATH, ctx);
    // The kick leaves the record intact; a repeat failure re-records without
    // resetting the accumulated budget.
    ctx.registry.recordFailure(PATH, new Error('lazy 404 again'), 'Network');

    expect(ctx.registry.failedLoaders.get(PATH)!.autoRetryCount).toBe(2);
    expect(ctx.registry.autoRetryablePaths()).toContain(PATH);
    ctx.registry.markAutoRetryAttempt(PATH); // reaches MAX_AUTO_RETRY_ATTEMPTS (3)
    expect(ctx.registry.autoRetryablePaths()).not.toContain(PATH);
  });

  it('falls through to the stale-entry cleanup when the registry has no lazy child', async () => {
    const retryLazyChildByLeafPath = vi.fn().mockReturnValue(false);
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH),
      lodGroupRegistry: { retryLazyChildByLeafPath } as never,
    });
    ctx.registry.recordFailure(PATH, new Error('orphaned failure'));

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    expect(ok).toBe(false);
    expect(ctx.registry.failedLoaders.has(PATH)).toBe(false); // stale entry cleaned
  });

  it('no lodGroupRegistry on ctx (headless) keeps the legacy no-loader behavior', async () => {
    const ctx = makeRetryCtx({ rootGroup: makeRootGroupWith(PATH) });
    ctx.registry.recordFailure(PATH, new Error('initial failure'));
    expect(await retryFailedLoaderUnlocked(PATH, ctx)).toBe(false);
  });
});

describe('retryFailedLoaderUnlocked — per-attempt retryCount accounting', () => {
  it('increments retryCount when the retry itself throws', async () => {
    const updateView = vi.fn().mockRejectedValue(new Error('network down'));
    const ctx = makeRetryCtx({
      rootGroup: makeRootGroupWith(PATH),
    });
    ctx.registry.registerPointsLoader(PATH, makePointsLoader(updateView));
    // First failure recorded with retryCount 0; the retry's catch block
    // bumps that to retryCount 1.
    ctx.registry.recordFailure(PATH, new Error('initial failure'));
    const initial = ctx.registry.failedLoaders.get(PATH);
    expect(initial?.retryCount).toBe(0);

    const ok = await retryFailedLoaderUnlocked(PATH, ctx);

    expect(ok).toBe(false);
    const updated = ctx.registry.failedLoaders.get(PATH);
    expect(updated?.retryCount).toBe(1);
    expect(updated?.error.message).toBe('network down');
  });
});

describe('retryAllFailedLoadersUnlocked — partition', () => {
  it('returns empty buckets when given no paths', async () => {
    const ctx = makeRetryCtx();
    const result = await retryAllFailedLoadersUnlocked([], ctx);
    expect(result).toEqual({ succeeded: [], failed: [] });
  });

  it('partitions a mixed batch into succeeded / failed', async () => {
    const okData = { pointCount: 1 } as unknown as LoadedPointsData;
    const okLoader = makePointsLoader(vi.fn().mockResolvedValue(okData));
    const failLoader = makePointsLoader(vi.fn().mockRejectedValue(new Error('boom')));

    const ctx = makeRetryCtx({
      rootGroup: (() => {
        // Group containing both paths so verifyAndClear succeeds for the
        // ok path.
        const root = new THREE.Group();
        for (const p of ['/a', '/b', '/c']) {
          const m = new THREE.Mesh();
          m.name = p;
          root.add(m);
        }
        return root;
      })(),
    });
    ctx.registry.registerPointsLoader('/a', okLoader);
    ctx.registry.registerPointsLoader('/b', failLoader);
    ctx.registry.registerPointsLoader('/c', okLoader);
    ctx.registry.recordFailure('/a', new Error('e'));
    ctx.registry.recordFailure('/b', new Error('e'));
    ctx.registry.recordFailure('/c', new Error('e'));

    const result = await retryAllFailedLoadersUnlocked(['/a', '/b', '/c'], ctx);

    expect(result.succeeded.sort()).toEqual(['/a', '/c']);
    expect(result.failed).toEqual(['/b']);
    // The failing path keeps its failure entry with retryCount bumped.
    expect(ctx.registry.failedLoaders.get('/b')?.retryCount).toBe(1);
    // The succeeded paths have their failure entries cleared.
    expect(ctx.registry.failedLoaders.has('/a')).toBe(false);
    expect(ctx.registry.failedLoaders.has('/c')).toBe(false);
  });

  it('starts every retry in a sub-cap batch before any settles', async () => {
    // This three-path batch is deliberately below EAGER_CHILD_LOAD_CONCURRENCY.
    // Each updateView resolves after the next microtask, so record call order
    // to assert every admitted retry enters before any resolves.
    const callOrder: string[] = [];
    const makeSlowLoader = (p: string) =>
      makePointsLoader(async () => {
        callOrder.push(`enter:${p}`);
        await Promise.resolve(); // microtask yield
        callOrder.push(`resolve:${p}`);
        return { pointCount: 0 } as unknown as LoadedPointsData;
      });

    const ctx = makeRetryCtx({
      rootGroup: (() => {
        const root = new THREE.Group();
        for (const p of ['/x', '/y', '/z']) {
          const m = new THREE.Mesh();
          m.name = p;
          root.add(m);
        }
        return root;
      })(),
    });
    for (const p of ['/x', '/y', '/z']) {
      ctx.registry.registerPointsLoader(p, makeSlowLoader(p));
      ctx.registry.recordFailure(p, new Error('e'));
    }

    await retryAllFailedLoadersUnlocked(['/x', '/y', '/z'], ctx);

    // All three `enter:*` events come before any `resolve:*`.
    const firstResolveIdx = callOrder.findIndex((s) => s.startsWith('resolve:'));
    expect(firstResolveIdx).toBe(3);
    expect(callOrder.slice(0, 3).sort()).toEqual(['enter:/x', 'enter:/y', 'enter:/z']);
  });

  it('caps retry fan-out at the eager child concurrency', async () => {
    const paths = Array.from({ length: EAGER_CHILD_LOAD_CONCURRENCY + 1 }, (_, i) => `/p-${i}`);
    const root = new THREE.Group();
    const started: string[] = [];
    const settle: Array<() => void> = [];
    const ctx = makeRetryCtx({ rootGroup: root });

    for (const path of paths) {
      const mesh = new THREE.Mesh();
      mesh.name = path;
      root.add(mesh);
      ctx.registry.registerPointsLoader(
        path,
        makePointsLoader(
          () =>
            new Promise((resolve) => {
              started.push(path);
              settle.push(() => resolve({ pointCount: 0 } as unknown as LoadedPointsData));
            })
        )
      );
      ctx.registry.recordFailure(path, new Error('offline'));
    }

    const retry = retryAllFailedLoadersUnlocked(paths, ctx);
    await vi.waitFor(() => expect(started).toHaveLength(EAGER_CHILD_LOAD_CONCURRENCY));
    expect(started).toEqual(paths.slice(0, EAGER_CHILD_LOAD_CONCURRENCY));

    settle.shift()!();
    await vi.waitFor(() => expect(started).toEqual(paths));
    for (const resolve of settle.splice(0)) resolve();

    await expect(retry).resolves.toEqual({ succeeded: paths, failed: [] });
  });

  it('serializes oversized line retries and releases admission after the batch', async () => {
    const paths = ['/line-a', '/line-b', '/line-c'];
    const root = new THREE.Group();
    const started: string[] = [];
    const settle: Array<() => void> = [];
    const ctx = makeRetryCtx({ rootGroup: root });

    for (const path of paths) {
      const mesh = new THREE.Mesh();
      mesh.name = path;
      mesh.userData.attrs = {
        type: 'lines',
        n_vertices: 1_600_000,
        n_segments: 1_600_000,
        ndim: 3,
      };
      root.add(mesh);
      ctx.registry.registerLinesLoader(
        path,
        makeLinesLoader(
          () =>
            new Promise((resolve) => {
              started.push(path);
              settle.push(() => resolve({} as LoadedLinesData));
            })
        )
      );
    }

    ctx.registry.recordFailure(paths[0], new Error('allocation failed'));
    ctx.registry.recordFailure(paths[1], new Error('allocation failed'));
    const firstBatch = retryAllFailedLoadersUnlocked(paths.slice(0, 2), ctx);

    await vi.waitFor(() => expect(started).toEqual([paths[0]]));
    settle.shift()!();
    await vi.waitFor(() => expect(started).toEqual(paths.slice(0, 2)));
    settle.shift()!();
    await expect(firstBatch).resolves.toEqual({ succeeded: paths.slice(0, 2), failed: [] });

    ctx.registry.recordFailure(paths[2], new Error('allocation failed'));
    const secondBatch = retryAllFailedLoadersUnlocked([paths[2]], ctx);
    await vi.waitFor(() => expect(started).toEqual(paths));
    settle.shift()!();
    await expect(secondBatch).resolves.toEqual({ succeeded: [paths[2]], failed: [] });
  });
});
