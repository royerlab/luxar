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
    updatePointsGeometry: ReturnType<typeof vi.fn>;
    createEmptyPointsNode: ReturnType<typeof vi.fn>;
  };
} {
  const viewState = makeViewState();
  const createEmptyPointsNode = vi.fn((path: string) => makePlaceholder(path));
  const applyEffectiveAttrs = vi.fn((node: SceneNode) => node.attrs);
  const deriveNodeViewState = vi.fn(() => ({ skip: false as const, viewState }));
  const connectLoaderToMonitor = vi.fn();
  const updatePointsGeometry = vi.fn();

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
    releaseLazyGSplats: vi.fn(),
    releaseLazyPoints: vi.fn(),
    releaseLazyLines: vi.fn(),
    applyEffectiveAttrs,
    deriveNodeViewState,
    connectLoaderToMonitor,
    updatePointsGeometry,
    processLinesData: vi.fn(),
    commitLinesGeometry: vi.fn(),
    processGSplatsData: vi.fn(),
    commitGSplatsGeometry: vi.fn(),
    ...overrides,
  };
  return Object.assign(ctx, {
    spies: {
      applyEffectiveAttrs,
      deriveNodeViewState,
      connectLoaderToMonitor,
      updatePointsGeometry,
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

    // The loader is registered before the fetch completes (so retry/update can
    // find it while the fetch is in flight). Registration lands one microtask
    // after the cheap split returns — the cheap/expensive split (mirroring the
    // gsplats loader) registers between the two halves, not synchronously — so
    // flush microtasks before asserting. The fetch promise (`pending`) is still
    // unresolved here, proving registration precedes the fetch.
    await vi.waitFor(() => expect(ctx.registry.loaders.has('/scene/p')).toBe(true));

    // Let the promise settle so the test cleans up.
    _resolve({ pointCount: 0 } as LoadedPointsData);
    await promise;
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

  it('commits via ctx.updatePointsGeometry on success', async () => {
    const data = { pointCount: 5 } as LoadedPointsData;
    createPointsLoaderMock.mockReturnValue(makePointsLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();

    const result = await loadPointsNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(result).not.toBeNull();
    expect(ctx.spies.updatePointsGeometry).toHaveBeenCalledWith('/scene/p', data);
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
    expect(ctx.spies.updatePointsGeometry).toHaveBeenCalledWith('/scene/p', data);
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
    expect(ctx.spies.updatePointsGeometry).not.toHaveBeenCalled();
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
