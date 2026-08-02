/**
 * Unit tests for `loadPointsNode` in scene-loader/nodes/load-points-node.ts.
 *
 * The initial-load path's three load-bearing invariants:
 *   1. Placeholder attached to parentThree BEFORE the async fetch — if
 *      `.add(placeholder)` happens after the await, the failure-recovery
 *      contract breaks (retry has no mesh to target) but the happy path
 *      still passes its tests.
 *   2. extend_to_all skip fallback uses the BASE viewState — on initial
 *      load we still want to construct the THREE node so future slice
 *      changes can populate it; the skip return only short-circuits on
 *      update/retry paths.
 *   3. On loader.loadPoints rejection, the failure is recorded AND a
 *      typed `LoaderError` is re-thrown so `loadLeafNode` can dispatch
 *      by kind and keep sibling nodes rendering.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { processPointsData } from '../../../../../data/scene-loader/process/data-processor-points';
import * as THREE from 'three';

// vi.mock the loader-factory module so we can capture createPointsLoader
// args without spinning up the real spatial-index loader (which would
// drag in zarr, WASM, the data monitor, etc.). Pattern matches
// `loader-factory.test.ts`.
const createPointsLoaderMock = vi.fn();
vi.mock('../../../../../data/scene-loader/loaders/loader-factory', () => ({
  createPointsLoader: (...args: unknown[]) => createPointsLoaderMock(...args),
  // Re-export the type-only interface — vitest ignores type imports at runtime
  // so this is just to satisfy `import type { LoaderFactoryDeps }` chains.
}));

import { loadPointsNode } from '../../../../../data/scene-loader/nodes/load-points-node';
import { LoaderError } from '../../../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import { LoaderRegistry } from '../../../../../data/scene-loader/loaders/loader-registry';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type {
  SceneNode,
  DataLoader,
  LoadedPointsData,
  ViewState,
} from '../../../../../data/data-loader-types';

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

function makeSceneNode(overrides: Partial<SceneNode> = {}): SceneNode {
  return {
    path: '/scene/p',
    type: 'points',
    attrs: { n_points: 42 },
    hasSpatialIndex: true,
    children: [],
    ...overrides,
  };
}

/** Minimal DataLoader stub — the helper only calls `.loadPoints`. */
function makePointsLoader(loadPoints: (vs: ViewState) => Promise<LoadedPointsData>): DataLoader {
  return { loadPoints } as unknown as DataLoader;
}

/** Real THREE.Mesh with `name` set — what NodeFactory.createEmptyPointsNode returns. */
function makePlaceholder(name: string): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = name;
  return mesh;
}

function makeCtx(overrides: Partial<NodeBuildCtx> = {}): NodeBuildCtx & {
  spies: {
    applyEffectiveAttrs: ReturnType<typeof vi.fn>;
    deriveNodeViewState: ReturnType<typeof vi.fn>;
    connectLoaderToMonitor: ReturnType<typeof vi.fn>;
    commitPointsGeometry: ReturnType<typeof vi.fn>;
    createEmptyPointsNode: ReturnType<typeof vi.fn>;
  };
} {
  const viewState = makeViewState();
  const createEmptyPointsNode = vi.fn((path: string) => makePlaceholder(path));
  const applyEffectiveAttrs = vi.fn((node: SceneNode) => node.attrs);
  const deriveNodeViewState = vi.fn(() => ({ skip: false as const, viewState }));
  const connectLoaderToMonitor = vi.fn();
  const commitPointsGeometry = vi.fn();

  const nodeFactory = {
    createEmptyPointsNode,
    createEmptyLinesNode: vi.fn(),
    createEmptyGSplatsNode: vi.fn(),
    applyTransform: vi.fn(),
    markPickingDirty: vi.fn(),
  } as unknown as NodeBuildCtx['nodeFactory'];

  const factoryDeps = {} as unknown as NodeBuildCtx['factoryDeps'];

  const ctx: NodeBuildCtx = {
    registry: new LoaderRegistry(),
    nodeFactory,
    viewState,
    factoryDeps,
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
    processLinesData: vi.fn(),
    commitLinesGeometry: vi.fn(),
    processGSplatsData: vi.fn(),
    processPointsData,
    commitPointsGeometry,
    commitGSplatsGeometry: vi.fn(),
    ...overrides,
  };
  return Object.assign(ctx, {
    spies: {
      applyEffectiveAttrs,
      deriveNodeViewState,
      connectLoaderToMonitor,
      commitPointsGeometry,
      createEmptyPointsNode,
    },
  });
}

beforeEach(() => {
  createPointsLoaderMock.mockReset();
});

// ============================================================================
// Tests
// ============================================================================

describe('loadPointsNode — placeholder-before-fetch invariant', () => {
  it('adds the placeholder to parentThree BEFORE awaiting loader.loadPoints', async () => {
    // A loadPoints that never resolves — lets us check the synchronous
    // state of parentThree right after the call returns the unsettled promise.
    let _resolve: (data: LoadedPointsData) => void = () => {};
    const pending = new Promise<LoadedPointsData>((r) => {
      _resolve = r;
    });
    const loader = makePointsLoader(() => pending);
    createPointsLoaderMock.mockReturnValue(loader);

    const ctx = makeCtx();
    const parent = new THREE.Group();

    const promise = loadPointsNode(makeSceneNode(), parent, {} as never, ctx);

    // The placeholder is attached synchronously by the cheap half, before the
    // function yields — so an initial-load failure leaves a recoverable scene.
    expect(parent.children.length).toBe(1);
    expect((parent.children[0] as THREE.Mesh).name).toBe('/scene/p');

    // The loader must NOT be registered while the fetch is in flight — a
    // concurrent updateView sweep calling loader.updateView mid-initial-load
    // would interleave the shared accumulator (the activation race).
    // Registration happens in the combined loader's finally, once the
    // expensive half settles.
    await Promise.resolve();
    expect(ctx.registry.loaders.has('/scene/p')).toBe(false);

    // Let the promise settle so the test cleans up — registration lands now.
    _resolve({ pointCount: 0 } as LoadedPointsData);
    await promise;
    expect(ctx.registry.loaders.has('/scene/p')).toBe(true);
  });
});

describe('loadPointsNode — happy path', () => {
  it('uses applyPartialExtendTolerance:true (Points variant)', async () => {
    const data = { pointCount: 5 } as LoadedPointsData;
    createPointsLoaderMock.mockReturnValue(makePointsLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();

    await loadPointsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(ctx.spies.deriveNodeViewState).toHaveBeenCalledWith(
      '/scene/p',
      { n_points: 42 },
      { applyPartialExtendTolerance: true }
    );
  });

  it('passes both composed attrs and raw leaf attrs to the points placeholder', async () => {
    const data = { pointCount: 0 } as LoadedPointsData;
    const loader = makePointsLoader(vi.fn().mockResolvedValue(data));
    createPointsLoaderMock.mockReturnValue(loader);
    const rawAttrs = {
      n_points: 42,
      colormap: 'viridis',
      has_scalars: true,
      intensity: 1.0,
      offset: 0.0,
    };
    const composedAttrs = {
      ...rawAttrs,
      intensity: 0.5,
      offset: 0.1,
    };
    const node = makeSceneNode({ attrs: rawAttrs });
    const ctx = makeCtx();
    ctx.spies.applyEffectiveAttrs.mockReturnValue(composedAttrs);

    await loadPointsNode(node, new THREE.Group(), {} as never, ctx);

    expect(ctx.spies.createEmptyPointsNode).toHaveBeenCalledWith(
      '/scene/p',
      composedAttrs,
      loader,
      rawAttrs
    );
  });

  it('commits via the ctx points process/commit pair on success', async () => {
    const data = { pointCount: 5 } as LoadedPointsData;
    createPointsLoaderMock.mockReturnValue(makePointsLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();

    const result = await loadPointsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(result).not.toBeNull();
    expect(ctx.spies.commitPointsGeometry).toHaveBeenCalledWith(
      { path: '/scene/p', data },
      undefined,
      1
    );
  });

  it('clears a prior failure record on a successful (re)load', async () => {
    // A recovered lazy level must drop out of the outcome report and the
    // auto-retry budget once its reload commits.
    const data = { pointCount: 5 } as LoadedPointsData;
    createPointsLoaderMock.mockReturnValue(makePointsLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();
    ctx.registry.recordFailure('/scene/p', new Error('earlier 503'), 'Network');

    await loadPointsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(ctx.registry.failedLoaders.has('/scene/p')).toBe(false);
  });
});

describe('loadPointsNode — extend_to_all skip fallback on initial load', () => {
  it('falls back to ctx.viewState (NOT the derived one) when derive returns skip', async () => {
    const loadPoints = vi.fn().mockResolvedValue({ pointCount: 1 } as LoadedPointsData);
    createPointsLoaderMock.mockReturnValue(makePointsLoader(loadPoints));
    const ctx = makeCtx();
    ctx.spies.deriveNodeViewState.mockReturnValue({ skip: 'extend_to_all' });

    await loadPointsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(loadPoints).toHaveBeenCalledTimes(1);
    // Initial-load uses ctx.viewState — the same object, not a copy.
    expect(loadPoints.mock.calls[0][0]).toBe(ctx.viewState);
  });
});

describe('loadPointsNode — pointCount === 0 path', () => {
  it('still returns the placeholder and commits the (empty) geometry', async () => {
    const data = { pointCount: 0 } as LoadedPointsData;
    createPointsLoaderMock.mockReturnValue(makePointsLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();

    const placeholder = await loadPointsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(placeholder).not.toBeNull();
    expect(placeholder!.name).toBe('/scene/p');
    // Future updateView() / retry calls need the empty commit to seed
    // the geometry — so the helper commits even on pointCount=0.
    expect(ctx.spies.commitPointsGeometry).toHaveBeenCalledWith(
      { path: '/scene/p', data },
      undefined,
      1
    );
  });
});

describe('loadPointsNode — registration only after the initial load settles', () => {
  // Regression (activation race): registering BEFORE the expensive await let
  // a concurrent updateView sweep call loader.updateView while the initial
  // load was mid-flight on the same instance (shared accumulator +
  // _activeSignal). Registration must happen only once the load SETTLES —
  // and on failure too, so retryFailedLoader can still resolve the loader.
  it('does NOT register while the initial load is in flight; registers on success', async () => {
    let resolveLoad!: (d: unknown) => void;
    const pending = new Promise((res) => (resolveLoad = res));
    createPointsLoaderMock.mockReturnValue(makePointsLoader(() => pending as never));
    const ctx = makeCtx();

    const promise = loadPointsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);
    await Promise.resolve(); // let the cheap half + the expensive await start
    expect(ctx.registry.loaders.has('/scene/p')).toBe(false); // not yet in the sweep

    resolveLoad({ pointCount: 5 } as LoadedPointsData);
    await promise;
    expect(ctx.registry.loaders.has('/scene/p')).toBe(true);
  });

  it('registers even when the initial load FAILS (loader stays retryable)', async () => {
    createPointsLoaderMock.mockReturnValue(
      makePointsLoader(vi.fn().mockRejectedValue(new Error('network down')) as never)
    );
    const ctx = makeCtx();

    await expect(
      loadPointsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx)
    ).rejects.toThrow();
    expect(ctx.registry.loaders.has('/scene/p')).toBe(true); // retry can find it
    expect(ctx.registry.failedLoaders.has('/scene/p')).toBe(true);
  });
});

describe('loadPointsNode — error path', () => {
  it('records failure, rethrows as LoaderError, and leaves the placeholder in the scene', async () => {
    const cause = new Error('network timeout');
    createPointsLoaderMock.mockReturnValue(makePointsLoader(vi.fn().mockRejectedValue(cause)));
    const ctx = makeCtx();
    const parent = new THREE.Group();

    await expect(loadPointsNode(makeSceneNode(), parent, {} as never, ctx)).rejects.toBeInstanceOf(
      LoaderError
    );

    // Placeholder stays — retry will populate it later.
    expect(parent.children.length).toBe(1);
    expect((parent.children[0] as THREE.Mesh).name).toBe('/scene/p');
    // Failure recorded with the original cause.
    expect(ctx.registry.failedLoaders.has('/scene/p')).toBe(true);
    expect(ctx.registry.failedLoaders.get('/scene/p')?.error).toBe(cause);
    // No commit on the failure path.
    expect(ctx.spies.commitPointsGeometry).not.toHaveBeenCalled();
  });

  it('classifies network errors as Network kind on the rethrown LoaderError', async () => {
    const cause = new Error('fetch failed: network unreachable');
    createPointsLoaderMock.mockReturnValue(makePointsLoader(vi.fn().mockRejectedValue(cause)));
    const ctx = makeCtx();

    try {
      await loadPointsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);
      expect.fail('expected throw');
    } catch (err) {
      expect(err).toBeInstanceOf(LoaderError);
      expect((err as LoaderError).kind).toBe('Network');
      expect((err as LoaderError).path).toBe('/scene/p');
    }
  });
});

describe('loadPointsNode — loader-factory + monitor wiring', () => {
  it('invokes createPointsLoaderHelper with (node, loc, factoryDeps) and wires the monitor', async () => {
    const data = { pointCount: 0 } as LoadedPointsData;
    const loader = makePointsLoader(vi.fn().mockResolvedValue(data));
    createPointsLoaderMock.mockReturnValue(loader);
    const ctx = makeCtx();
    const node = makeSceneNode();
    const loc = { fakeLoc: true } as never;

    await loadPointsNode(node, new THREE.Group(), loc, ctx);

    // Constructed with the same node + loc + factoryDeps the orchestrator passes.
    expect(createPointsLoaderMock).toHaveBeenCalledWith(node, loc, ctx.factoryDeps);
    // Wired to the monitor exactly once, with the freshly-constructed loader.
    expect(ctx.spies.connectLoaderToMonitor).toHaveBeenCalledWith('/scene/p', loader);
  });
});
