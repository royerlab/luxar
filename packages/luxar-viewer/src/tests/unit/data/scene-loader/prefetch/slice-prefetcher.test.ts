/**
 * Unit tests for SlicePrefetcher (data/scene-loader/prefetch/
 * slice-prefetcher.ts) — the background t+1 shadow-load pass that warms the
 * SliceCache during dimension playback.
 *
 * Contracts pinned here:
 *   - shadow loaders are built from the factory (plain vs progressive by
 *     n_additive_sublods), cached per path, and NEVER the foreground
 *     registry loaders;
 *   - every shadow pass carries frameBudgetMs (the store-prefix invariant)
 *     and the pass's abort signal;
 *   - extend_to_all-skipped nodes and no-hidden-dims views are skipped;
 *   - persist-across-ticks: a prefetch while a batch is in flight is a no-op
 *     (NOT abort+restart — a cold level outlives one frame), so background
 *     deepening can complete; abortInFlight / releaseShadows (dispose + lazy
 *     rebuild) / dispose still tear it down;
 *   - a failed shadow build is dropped so the next pass retries.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { SceneNode, ViewState, DataLoader } from '../../../../../data/data-loader-types';

const factoryCalls: Array<{ helper: string; path: string }> = [];
const shadowLoaders = new Map<string, ReturnType<typeof makeShadowLoader>>();
let failNextBuild = false;

function makeShadowLoader(path: string) {
  return {
    path,
    updateView: vi.fn().mockResolvedValue({}),
    dispose: vi.fn(),
  };
}

vi.mock('../../../../../data/scene-loader/loaders/loader-factory', () => {
  const record = (helper: string, node: SceneNode) => {
    factoryCalls.push({ helper, path: node.path });
    if (failNextBuild) {
      failNextBuild = false;
      if (helper.startsWith('createProgressive')) return Promise.reject(new Error('build failed'));
      throw new Error('build failed');
    }
    const loader = makeShadowLoader(node.path);
    shadowLoaders.set(node.path, loader);
    return helper.startsWith('createProgressive') ? Promise.resolve(loader) : loader;
  };
  return {
    createPointsLoader: vi.fn((node: SceneNode) => record('createPointsLoader', node)),
    createLinesLoader: vi.fn((node: SceneNode) => record('createLinesLoader', node)),
    createGSplatsLoader: vi.fn((node: SceneNode) => record('createGSplatsLoader', node)),
    createProgressivePointsLoader: vi.fn((node: SceneNode) =>
      record('createProgressivePointsLoader', node)
    ),
    createProgressiveLinesLoader: vi.fn((node: SceneNode) =>
      record('createProgressiveLinesLoader', node)
    ),
    createProgressiveGSplatsLoader: vi.fn((node: SceneNode) =>
      record('createProgressiveGSplatsLoader', node)
    ),
  };
});

vi.mock('../../../../../data/zarr', () => ({
  root: vi.fn(() => ({ resolve: vi.fn() })),
}));

import { SlicePrefetcher } from '../../../../../data/scene-loader/prefetch/slice-prefetcher';

function makeNode(path: string, attrs: SceneNode['attrs'] = {}): SceneNode {
  return { path, type: 'gsplats', attrs, hasSpatialIndex: true, children: [] };
}

const view: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 7],
  tolerance: [0, 0, 0, 0.25],
};

function flushAsync(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe('SlicePrefetcher', () => {
  let graph: SceneNode;
  let registry: {
    loaders: Map<string, DataLoader>;
    linesLoaders: Map<string, DataLoader>;
    gsplatLoaders: Map<string, DataLoader>;
  };
  let foregroundLoader: { updateView: ReturnType<typeof vi.fn> };
  let prefetcher: SlicePrefetcher;

  beforeEach(() => {
    factoryCalls.length = 0;
    shadowLoaders.clear();
    failNextBuild = false;

    graph = {
      ...makeNode('/'),
      children: [makeNode('/splats'), makeNode('/pts')],
    };
    foregroundLoader = { updateView: vi.fn() };
    registry = {
      loaders: new Map([['/pts', foregroundLoader as unknown as DataLoader]]),
      linesLoaders: new Map(),
      gsplatLoaders: new Map([['/splats', foregroundLoader as unknown as DataLoader]]),
    };
    prefetcher = new SlicePrefetcher({
      getSceneGraph: () => graph,
      factoryDeps: () => ({ zarrStore: {} }) as never,
      registry: registry as never,
      applyEffectiveAttrs: (node) => node.attrs,
    });
  });

  it('builds shadow loaders from the factory and runs them with budget + signal — never the foreground loaders', async () => {
    prefetcher.prefetch(view, 42);
    await flushAsync();

    expect(factoryCalls.map((c) => c.helper).sort()).toEqual([
      'createGSplatsLoader',
      'createPointsLoader',
    ]);
    const shadow = shadowLoaders.get('/splats')!;
    expect(shadow.updateView).toHaveBeenCalledTimes(1);
    const [vs, session, signal] = shadow.updateView.mock.calls[0];
    expect(vs.frameBudgetMs).toBe(42); // the store-prefix invariant
    expect(vs.slicePosition).toEqual([0, 0, 0, 7]);
    expect(session).toBeUndefined();
    expect(signal).toBeInstanceOf(AbortSignal);
    expect(signal.aborted).toBe(false);

    expect(foregroundLoader.updateView).not.toHaveBeenCalled();
  });

  it('chooses the progressive factory when n_additive_sublods > 1', async () => {
    graph.children = [makeNode('/splats', { n_additive_sublods: 4 })];
    registry.loaders.clear();
    prefetcher.prefetch(view, 10);
    await flushAsync();
    expect(factoryCalls.map((c) => c.helper)).toEqual(['createProgressiveGSplatsLoader']);
    expect(shadowLoaders.get('/splats')!.updateView).toHaveBeenCalledTimes(1);
  });

  it('reuses the cached shadow instance on subsequent passes', async () => {
    prefetcher.prefetch(view, 10);
    await flushAsync();
    prefetcher.prefetch({ ...view, slicePosition: [0, 0, 0, 8] }, 10);
    await flushAsync();
    // 2 nodes → 2 builds total (not 4); each shadow ran twice.
    expect(factoryCalls).toHaveLength(2);
    expect(shadowLoaders.get('/splats')!.updateView).toHaveBeenCalledTimes(2);
  });

  it('skips nodes whose extend_to_all covers all hidden dims, and no-hidden-dims views entirely', async () => {
    graph.children = [makeNode('/splats', { extend_to_all: ['t'] })];
    registry.loaders.clear();
    const dims = [
      { name: 'x' },
      { name: 'y' },
      { name: 'z' },
      { name: 't', discrete: true, step: 1 },
    ] as never;
    prefetcher.prefetch({ ...view, dimensions: dims }, 10);
    await flushAsync();
    expect(factoryCalls).toHaveLength(0); // derived.skip — no shadow built

    graph.children = [makeNode('/splats')];
    prefetcher.prefetch(
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      10
    );
    await flushAsync();
    expect(factoryCalls).toHaveLength(0); // no hidden dims — S-cache ineligible
  });

  it('persists across ticks: a prefetch while a batch is in flight is a no-op (not abort+restart)', async () => {
    // Batch 1 starts; inFlight > 0 synchronously. A cold LOD level outlives one
    // playback frame, so the next tick's prefetch must NOT abort+restart it —
    // that would never let a level complete + cache. The in-flight target
    // wins; the second call is dropped.
    prefetcher.prefetch(view, 10);
    prefetcher.prefetch({ ...view, slicePosition: [0, 0, 0, 8] }, 10); // in-flight → skipped
    await flushAsync();

    const shadow = shadowLoaders.get('/splats')!;
    expect(shadow.updateView).toHaveBeenCalledTimes(1); // only batch 1 ran
    expect(shadow.updateView.mock.calls[0][0].slicePosition).toEqual([0, 0, 0, 7]);
    const signal = shadow.updateView.mock.calls[0][2] as AbortSignal;
    expect(signal.aborted).toBe(false); // the skipped second call never aborted it

    // Playback end / dataset switch DOES tear the batch down.
    prefetcher.abortInFlight();
    expect(signal.aborted).toBe(true);
  });

  it('re-targets on the NEXT tick once the in-flight batch has settled', async () => {
    prefetcher.prefetch(view, 10);
    await flushAsync(); // batch 1 completes → inFlight gate reopens
    prefetcher.prefetch({ ...view, slicePosition: [0, 0, 0, 8] }, 10);
    await flushAsync();
    const shadow = shadowLoaders.get('/splats')!;
    expect(shadow.updateView).toHaveBeenCalledTimes(2); // batch 2 ran after batch 1 settled
    expect(shadow.updateView.mock.calls[1][0].slicePosition).toEqual([0, 0, 0, 8]);
  });

  it('supersedes a STALLED batch so a hung task cannot pin the gate for the session', () => {
    // A shadow fetch/build with no timeout could hang forever; its
    // `Promise.allSettled` would then never resolve and `inFlight` would stay
    // > 0, disabling prefetch for the rest of the session (the per-tick
    // foreground abort that used to self-correct this is gone). The stall guard
    // reclaims the gate. Synchronous test: the batch's tasks stay pending (no
    // flush), so `inFlight` stays > 0 the whole time; only the clock advances.
    let now = 1000;
    const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => now);
    const abortSpy = vi.spyOn(prefetcher, 'abortInFlight');
    try {
      prefetcher.prefetch(view, 10); // batch A: inFlight > 0, started at t=1000
      prefetcher.prefetch(view, 10); // still fresh → no-op, does NOT abort A
      expect(abortSpy).not.toHaveBeenCalled();

      now += 6000; // > MAX_BATCH_STALL_MS (5000): batch A is now stalled
      prefetcher.prefetch(view, 10); // stale → supersede
      expect(abortSpy).toHaveBeenCalledTimes(1);
    } finally {
      abortSpy.mockRestore();
      nowSpy.mockRestore();
    }
  });

  it('releaseShadows disposes every shadow and the next pass lazily rebuilds', async () => {
    prefetcher.prefetch(view, 10);
    await flushAsync();
    const shadow = shadowLoaders.get('/splats')!;

    prefetcher.releaseShadows();
    await flushAsync();
    expect(shadow.dispose).toHaveBeenCalledTimes(1);

    prefetcher.prefetch(view, 10);
    await flushAsync();
    expect(factoryCalls).toHaveLength(4); // 2 nodes × (initial + rebuild)
  });

  it('dispose makes further prefetch calls no-ops', async () => {
    prefetcher.dispose();
    prefetcher.prefetch(view, 10);
    await flushAsync();
    expect(factoryCalls).toHaveLength(0);
  });

  it('a failed shadow build is swallowed and retried on the next pass', async () => {
    graph.children = [makeNode('/splats', { n_additive_sublods: 4 })];
    registry.loaders.clear();
    failNextBuild = true;
    prefetcher.prefetch(view, 10);
    await flushAsync();
    expect(shadowLoaders.has('/splats')).toBe(false); // build rejected

    prefetcher.prefetch(view, 10); // retried — not poisoned
    await flushAsync();
    expect(shadowLoaders.get('/splats')!.updateView).toHaveBeenCalledTimes(1);
  });
});
