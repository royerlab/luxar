/**
 * Unit tests for `loadGSplatsNode` in scene-loader/nodes/load-gsplats-node.ts.
 *
 * The GSplats-specific invariant under test is the LOD branch:
 *   - `n_additive_sublods` is 0 or 1 → standard `createGSplatsLoader`
 *     path.
 *   - `n_additive_sublods > 1` → `createProgressiveGSplatsLoader`
 *     path, AND the parent's effective rendering attrs are composed up
 *     the scene-graph ancestry HERE (not inside the helper) so the LOD
 *     synthetic nodes inherit opacity/intensity/etc. — verify by
 *     checking that `applyEffectiveAttrs(node)` is called before the
 *     progressive helper.
 *
 * The extend_to_all skip fallback for GSplats uses an explicit 4-field
 * spread (rather than the identity that Points uses) — verify by
 * asserting the loader.loadGSplats argument has exactly the four
 * expected fields and is NOT the same object reference as ctx.viewState.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { processPointsData } from '../../../../../data/scene-loader/process/data-processor-points';
import * as THREE from 'three';

const createGSplatsLoaderMock = vi.fn();
const createProgressiveGSplatsLoaderMock = vi.fn();
vi.mock('../../../../../data/scene-loader/loaders/loader-factory', () => ({
  createGSplatsLoader: (...args: unknown[]) => createGSplatsLoaderMock(...args),
  createProgressiveGSplatsLoader: (...args: unknown[]) =>
    createProgressiveGSplatsLoaderMock(...args),
}));

import { loadGSplatsNode } from '../../../../../data/scene-loader/nodes/load-gsplats-node';
import { LoaderError } from '../../../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import { LoaderRegistry } from '../../../../../data/scene-loader/loaders/loader-registry';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode, ViewState } from '../../../../../data/data-loader-types';
import type {
  GSplatsDataLoader,
  GSplatsViewState,
  LoadedGSplatsData,
} from '../../../../../types/gsplats';
import type { StagedGSplatsCommit } from '../../../../../data/scene-loader/process/data-processor-gsplats';

// ============================================================================
// Local fixtures
// ============================================================================

function makeViewState(): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, 0],
    tolerance: [0, 0, 0, 1],
    dimensions: undefined,
  };
}

function makeSceneNode(attrs: Record<string, unknown> = {}): SceneNode {
  return {
    path: '/scene/g',
    type: 'gsplats',
    attrs: { n_splats: 50, ndim: 4, ...attrs },
    hasSpatialIndex: true,
    children: [],
  };
}

function makeGSplatsLoader(
  loadGSplats: (vs: GSplatsViewState) => Promise<LoadedGSplatsData>
): GSplatsDataLoader {
  return { loadGSplats } as unknown as GSplatsDataLoader;
}

function makePlaceholder(name: string): THREE.Mesh {
  const m = new THREE.Mesh();
  m.name = name;
  return m;
}

function makeCtx(overrides: Partial<NodeBuildCtx> = {}): NodeBuildCtx & {
  spies: {
    applyEffectiveAttrs: ReturnType<typeof vi.fn>;
    deriveNodeViewState: ReturnType<typeof vi.fn>;
    connectLoaderToMonitor: ReturnType<typeof vi.fn>;
    processGSplatsData: ReturnType<typeof vi.fn>;
    commitGSplatsGeometry: ReturnType<typeof vi.fn>;
    createEmptyGSplatsNode: ReturnType<typeof vi.fn>;
  };
} {
  const viewState = makeViewState();
  const createEmptyGSplatsNode = vi.fn((path: string) => makePlaceholder(path));
  const applyEffectiveAttrs = vi.fn((node: SceneNode) => node.attrs);
  const deriveNodeViewState = vi.fn(() => ({ skip: false as const, viewState }));
  const connectLoaderToMonitor = vi.fn();
  const processGSplatsData = vi.fn().mockResolvedValue(null);
  const commitGSplatsGeometry = vi.fn();

  const nodeFactory = {
    createEmptyPointsNode: vi.fn(),
    createEmptyLinesNode: vi.fn(),
    createEmptyGSplatsNode,
    applyTransform: vi.fn(),
    markPickingDirty: vi.fn(),
  } as unknown as NodeBuildCtx['nodeFactory'];

  const ctx: NodeBuildCtx = {
    registry: new LoaderRegistry(),
    nodeFactory,
    viewState,
    factoryDeps: {} as never,
    isDatasetLive: () => true,
    getViewVersion: () => 1,
    getLiveViewState: () => viewState,
    releaseLazyGSplats: vi.fn(),
    releaseLazyPoints: vi.fn(),
    releaseLazyLines: vi.fn(),
    kickRefinementIfIdle: vi.fn(),
    applyEffectiveAttrs,
    deriveNodeViewState,
    connectLoaderToMonitor,
    updatePointsGeometry: vi.fn(),
    processLinesData: vi.fn(),
    commitLinesGeometry: vi.fn(),
    processGSplatsData,
    processPointsData,
    commitPointsGeometry: vi.fn(),
    commitGSplatsGeometry,
    ...overrides,
  };
  return Object.assign(ctx, {
    spies: {
      applyEffectiveAttrs,
      deriveNodeViewState,
      connectLoaderToMonitor,
      processGSplatsData,
      commitGSplatsGeometry,
      createEmptyGSplatsNode,
    },
  });
}

beforeEach(() => {
  createGSplatsLoaderMock.mockReset();
  createProgressiveGSplatsLoaderMock.mockReset();
});

// ============================================================================
// Tests
// ============================================================================

describe('loadGSplatsNode — registration only after the initial load settles', () => {
  // Regression (activation race): registering BEFORE the expensive await let
  // a concurrent updateView sweep call loader.updateView while the initial
  // load was mid-flight on the same instance (shared accumulator +
  // _activeSignal). Registration must happen only once the load SETTLES —
  // and on failure too, so retryFailedLoader can still resolve the loader.
  it('does NOT register while the initial load is in flight; registers on success', async () => {
    let resolveLoad!: (d: unknown) => void;
    const pending = new Promise((res) => (resolveLoad = res));
    createGSplatsLoaderMock.mockReturnValue(makeGSplatsLoader(() => pending as never));
    const ctx = makeCtx();

    const promise = loadGSplatsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);
    await Promise.resolve(); // let the cheap half + the expensive await start
    expect(ctx.registry.gsplatLoaders.has('/scene/g')).toBe(false); // not yet in the sweep

    resolveLoad({ splatCount: 7 } as LoadedGSplatsData);
    await promise;
    expect(ctx.registry.gsplatLoaders.has('/scene/g')).toBe(true);
  });

  it('registers even when the initial load FAILS (loader stays retryable)', async () => {
    createGSplatsLoaderMock.mockReturnValue(
      makeGSplatsLoader(vi.fn().mockRejectedValue(new Error('network down')) as never)
    );
    const ctx = makeCtx();

    await expect(
      loadGSplatsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx)
    ).rejects.toThrow();
    expect(ctx.registry.gsplatLoaders.has('/scene/g')).toBe(true); // retry can find it
    expect(ctx.registry.failedLoaders.has('/scene/g')).toBe(true);
  });
});

describe('loadGSplatsNode — LOD branch', () => {
  it('uses single-LOD factory when n_additive_sublods is 0 / missing', async () => {
    const loader = makeGSplatsLoader(
      vi.fn().mockResolvedValue({ splatCount: 1 } as LoadedGSplatsData)
    );
    createGSplatsLoaderMock.mockReturnValue(loader);

    await loadGSplatsNode(makeSceneNode(), new THREE.Group(), {} as never, makeCtx());

    expect(createGSplatsLoaderMock).toHaveBeenCalledTimes(1);
    expect(createProgressiveGSplatsLoaderMock).not.toHaveBeenCalled();
  });

  it('uses single-LOD factory when n_additive_sublods === 1', async () => {
    const loader = makeGSplatsLoader(
      vi.fn().mockResolvedValue({ splatCount: 1 } as LoadedGSplatsData)
    );
    createGSplatsLoaderMock.mockReturnValue(loader);

    await loadGSplatsNode(
      makeSceneNode({ n_additive_sublods: 1 }),
      new THREE.Group(),
      {} as never,
      makeCtx()
    );

    expect(createGSplatsLoaderMock).toHaveBeenCalledTimes(1);
    expect(createProgressiveGSplatsLoaderMock).not.toHaveBeenCalled();
  });

  it('uses progressive factory when n_additive_sublods > 1', async () => {
    const loader = makeGSplatsLoader(
      vi.fn().mockResolvedValue({ splatCount: 1 } as LoadedGSplatsData)
    );
    createProgressiveGSplatsLoaderMock.mockResolvedValue(loader);

    await loadGSplatsNode(
      makeSceneNode({ n_additive_sublods: 3 }),
      new THREE.Group(),
      {} as never,
      makeCtx()
    );

    expect(createGSplatsLoaderMock).not.toHaveBeenCalled();
    expect(createProgressiveGSplatsLoaderMock).toHaveBeenCalledTimes(1);
  });

  it('multi-additive path passes applyEffectiveAttrs(node) to the progressive helper', async () => {
    const composedAttrs = { n_splats: 1000, opacity: 0.5, gamma: 1.2 };
    const loader = makeGSplatsLoader(
      vi.fn().mockResolvedValue({ splatCount: 1 } as LoadedGSplatsData)
    );
    createProgressiveGSplatsLoaderMock.mockResolvedValue(loader);

    const ctx = makeCtx();
    ctx.spies.applyEffectiveAttrs.mockImplementation(() => composedAttrs);

    const node = makeSceneNode({ n_additive_sublods: 4 });
    await loadGSplatsNode(node, new THREE.Group(), {} as never, ctx);

    // Helper signature: (node, nAdditive, parentEffectiveAttrs, deps).
    // The composed attrs (3rd arg) is THE invariant the LOD synthetic
    // nodes rely on to inherit ancestor opacity/intensity.
    expect(createProgressiveGSplatsLoaderMock).toHaveBeenCalledWith(
      node,
      4,
      composedAttrs,
      ctx.factoryDeps
    );
  });
});

describe('loadGSplatsNode — extend_to_all skip fallback', () => {
  it('builds a 4-field spread (not the same identity as ctx.viewState)', async () => {
    const loadGSplats = vi.fn().mockResolvedValue({ splatCount: 0 } as LoadedGSplatsData);
    createGSplatsLoaderMock.mockReturnValue(makeGSplatsLoader(loadGSplats));
    const ctx = makeCtx();
    ctx.spies.deriveNodeViewState.mockReturnValue({ skip: 'extend_to_all' });

    await loadGSplatsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    const passed = loadGSplats.mock.calls[0][0] as GSplatsViewState;
    // Same field VALUES…
    expect(passed.displayDims).toBe(ctx.viewState.displayDims);
    expect(passed.slicePosition).toBe(ctx.viewState.slicePosition);
    expect(passed.tolerance).toBe(ctx.viewState.tolerance);
    expect(passed.dimensions).toBe(ctx.viewState.dimensions);
    // …but a NEW object identity (4-field spread, unlike Points which
    // returns ctx.viewState by reference).
    expect(passed).not.toBe(ctx.viewState);
  });
});

describe('loadGSplatsNode — happy path commit flow', () => {
  it('routes data through processGSplatsData → commitGSplatsGeometry', async () => {
    const data = { splatCount: 7 } as LoadedGSplatsData;
    const staged = { path: '/scene/g' } as unknown as StagedGSplatsCommit;
    createGSplatsLoaderMock.mockReturnValue(makeGSplatsLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();
    ctx.spies.processGSplatsData.mockResolvedValue(staged);

    await loadGSplatsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(ctx.spies.processGSplatsData).toHaveBeenCalledTimes(1);
    expect(ctx.spies.commitGSplatsGeometry).toHaveBeenCalledWith(staged, undefined, 1);
  });

  it('stamps the DERIVE-time view version, not the commit-time version (lazy race)', async () => {
    // The decoupling exists so a deferred reload is stamped for the slice it was
    // ISSUED for, even if the global view version advances during the slow load
    // (a scrub mid-load). Bump the version INSIDE the awaited loadGSplats and
    // assert the commit stamps the pre-await value (the slice actually loaded) —
    // a mutant that captured ctx.getViewVersion() AFTER the await would stamp the
    // newer version and mark stale-slice data fresh. (Points/Lines share the
    // identical capture-before-await pattern.)
    let version = 5;
    const data = { splatCount: 7 } as LoadedGSplatsData;
    const staged = { path: '/scene/g' } as unknown as StagedGSplatsCommit;
    const loadGSplats = vi.fn().mockImplementation(async () => {
      version = 6; // a scrub advanced the global version while loading
      return data;
    });
    createGSplatsLoaderMock.mockReturnValue(makeGSplatsLoader(loadGSplats));
    const ctx = makeCtx({ getViewVersion: () => version });
    ctx.spies.processGSplatsData.mockResolvedValue(staged);

    await loadGSplatsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(ctx.spies.commitGSplatsGeometry).toHaveBeenCalledWith(staged, undefined, 5);
  });

  it('skips commit when processGSplatsData returns null', async () => {
    const data = { splatCount: 7 } as LoadedGSplatsData;
    createGSplatsLoaderMock.mockReturnValue(makeGSplatsLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();
    // ctx.processGSplatsData resolves null by default.

    const placeholder = await loadGSplatsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(placeholder).not.toBeNull();
    expect(ctx.spies.processGSplatsData).toHaveBeenCalled();
    expect(ctx.spies.commitGSplatsGeometry).not.toHaveBeenCalled();
  });
});

describe('loadGSplatsNode — failure-record clearing', () => {
  // A recovered lazy level must drop out of the outcome report and the
  // auto-retry budget, but only when its commit actually landed.
  it('clears a prior failure record when the commit lands', async () => {
    const data = { splatCount: 7 } as LoadedGSplatsData;
    const staged = { path: '/scene/g' } as unknown as StagedGSplatsCommit;
    createGSplatsLoaderMock.mockReturnValue(makeGSplatsLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();
    ctx.spies.processGSplatsData.mockResolvedValue(staged);
    ctx.registry.recordFailure('/scene/g', new Error('earlier 503'), 'Network');

    await loadGSplatsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(ctx.registry.failedLoaders.has('/scene/g')).toBe(false);
  });

  it('keeps the failure record when staged is null (load landed nowhere)', async () => {
    const data = { splatCount: 7 } as LoadedGSplatsData;
    createGSplatsLoaderMock.mockReturnValue(makeGSplatsLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx(); // processGSplatsData resolves null by default
    ctx.registry.recordFailure('/scene/g', new Error('earlier 503'), 'Network');

    await loadGSplatsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(ctx.registry.failedLoaders.has('/scene/g')).toBe(true);
  });
});

describe('loadGSplatsNode — error path', () => {
  it('records failure and rethrows LoaderError', async () => {
    const cause = new Error('validation expected ndim>3');
    createGSplatsLoaderMock.mockReturnValue(makeGSplatsLoader(vi.fn().mockRejectedValue(cause)));
    const ctx = makeCtx();
    const parent = new THREE.Group();

    await expect(loadGSplatsNode(makeSceneNode(), parent, {} as never, ctx)).rejects.toBeInstanceOf(
      LoaderError
    );

    expect(parent.children.length).toBe(1);
    expect(ctx.registry.failedLoaders.has('/scene/g')).toBe(true);
    expect(ctx.registry.failedLoaders.get('/scene/g')?.error).toBe(cause);
  });
});
