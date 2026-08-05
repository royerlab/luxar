/**
 * Unit tests for `loadLinesNode` in scene-loader/nodes/load-lines-node.ts.
 *
 * Mirrors load-points-node.test.ts but for the Lines variant. The
 * Lines-only invariant under test is `applyPartialExtendTolerance:
 * false` — Lines tolerate the un-overridden tolerance during the data
 * fetch because line bounds already encode their non-displayed spatial
 * extent; the partial-extend override would double-apply during
 * clipping.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { processPointsData } from '../../../../../data/scene-loader/process/data-processor-points';
import * as THREE from 'three';

const createLinesLoaderMock = vi.fn();
vi.mock('../../../../../data/scene-loader/loaders/loader-factory', () => ({
  createLinesLoader: (...args: unknown[]) => createLinesLoaderMock(...args),
}));

import { loadLinesNode } from '../../../../../data/scene-loader/nodes/load-lines-node';
import { LoaderError } from '../../../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import { LoaderRegistry } from '../../../../../data/scene-loader/loaders/loader-registry';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode, ViewState } from '../../../../../data/data-loader-types';
import type { LinesDataLoader, LoadedLinesData } from '../../../../../types/lines';
import type { StagedLinesCommit } from '../../../../../data/scene-loader/process/data-processor-lines';

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
    path: '/scene/l',
    type: 'lines',
    attrs: { n_segments: 10, n_vertices: 11 },
    hasSpatialIndex: true,
    children: [],
    ...overrides,
  };
}

function makeLinesLoader(loadLines: (vs: ViewState) => Promise<LoadedLinesData>): LinesDataLoader {
  return { loadLines } as unknown as LinesDataLoader;
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
    processLinesData: ReturnType<typeof vi.fn>;
    commitLinesGeometry: ReturnType<typeof vi.fn>;
    createEmptyLinesNode: ReturnType<typeof vi.fn>;
  };
} {
  const viewState = makeViewState();
  const createEmptyLinesNode = vi.fn((path: string) => makePlaceholder(path));
  const applyEffectiveAttrs = vi.fn((node: SceneNode) => node.attrs);
  const deriveNodeViewState = vi.fn(() => ({ skip: false as const, viewState }));
  const connectLoaderToMonitor = vi.fn();
  const processLinesData = vi.fn().mockResolvedValue(null);
  const commitLinesGeometry = vi.fn();

  const nodeFactory = {
    createEmptyPointsNode: vi.fn(),
    createEmptyLinesNode,
    createEmptyGSplatsNode: vi.fn(),
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
    releaseLazyGSplats: vi.fn(),
    releaseLazyPoints: vi.fn(),
    releaseLazyLines: vi.fn(),
    kickRefinementIfIdle: vi.fn(),
    applyEffectiveAttrs,
    deriveNodeViewState,
    connectLoaderToMonitor,
    processLinesData,
    commitLinesGeometry,
    processGSplatsData: vi.fn(),
    processPointsData,
    commitPointsGeometry: vi.fn(),
    commitGSplatsGeometry: vi.fn(),
    processMeshData: vi.fn(),
    commitMeshGeometry: vi.fn(),
    ...overrides,
  };
  return Object.assign(ctx, {
    spies: {
      applyEffectiveAttrs,
      deriveNodeViewState,
      connectLoaderToMonitor,
      processLinesData,
      commitLinesGeometry,
      createEmptyLinesNode,
    },
  });
}

beforeEach(() => {
  createLinesLoaderMock.mockReset();
});

// ============================================================================
// Tests
// ============================================================================

describe('loadLinesNode — registration only after the initial load settles', () => {
  // Regression (activation race): registering BEFORE the expensive await let
  // a concurrent updateView sweep call loader.updateView while the initial
  // load was mid-flight on the same instance (shared accumulator +
  // _activeSignal). Registration must happen only once the load SETTLES —
  // and on failure too, so retryFailedLoader can still resolve the loader.
  it('does NOT register while the initial load is in flight; registers on success', async () => {
    let resolveLoad!: (d: unknown) => void;
    const pending = new Promise((res) => (resolveLoad = res));
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(() => pending as never));
    const ctx = makeCtx();

    const promise = loadLinesNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);
    await Promise.resolve(); // let the cheap half + the expensive await start
    expect(ctx.registry.linesLoaders.has('/scene/l')).toBe(false); // not yet in the sweep

    resolveLoad({ segmentCount: 3 } as LoadedLinesData);
    await promise;
    expect(ctx.registry.linesLoaders.has('/scene/l')).toBe(true);
  });

  it('registers even when the initial load FAILS (loader stays retryable)', async () => {
    createLinesLoaderMock.mockReturnValue(
      makeLinesLoader(vi.fn().mockRejectedValue(new Error('network down')) as never)
    );
    const ctx = makeCtx();

    await expect(
      loadLinesNode(makeSceneNode(), new THREE.Group(), {} as never, ctx)
    ).rejects.toThrow();
    expect(ctx.registry.linesLoaders.has('/scene/l')).toBe(true); // retry can find it
    expect(ctx.registry.failedLoaders.has('/scene/l')).toBe(true);
  });
});

describe('loadLinesNode — placeholder-before-fetch invariant', () => {
  it('adds the placeholder BEFORE awaiting loader.loadLines', async () => {
    let _resolve: (data: LoadedLinesData) => void = () => {};
    const pending = new Promise<LoadedLinesData>((r) => {
      _resolve = r;
    });
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(() => pending));

    const ctx = makeCtx();
    const parent = new THREE.Group();
    const promise = loadLinesNode(makeSceneNode(), parent, {} as never, ctx);

    expect(parent.children.length).toBe(1);
    expect((parent.children[0] as THREE.Mesh).name).toBe('/scene/l');

    _resolve({ segmentCount: 0 } as LoadedLinesData);
    await promise;
  });
});

describe('loadLinesNode — happy path', () => {
  it('uses applyPartialExtendTolerance:false (Lines variant)', async () => {
    const data = { segmentCount: 4 } as LoadedLinesData;
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();

    await loadLinesNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(ctx.spies.deriveNodeViewState).toHaveBeenCalledWith(
      '/scene/l',
      { n_segments: 10, n_vertices: 11 },
      { applyPartialExtendTolerance: false }
    );
  });

  it('runs processLinesData → commitLinesGeometry on success', async () => {
    const data = { segmentCount: 4 } as LoadedLinesData;
    const staged = { path: '/scene/l' } as unknown as StagedLinesCommit;
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(vi.fn().mockResolvedValue(data)));

    const ctx = makeCtx();
    ctx.spies.processLinesData.mockResolvedValue(staged);

    const placeholder = await loadLinesNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(placeholder).not.toBeNull();
    expect(ctx.spies.processLinesData).toHaveBeenCalledWith('/scene/l', data, ctx.viewState);
    expect(ctx.spies.commitLinesGeometry).toHaveBeenCalledWith(staged, undefined, 1);
  });
});

describe('loadLinesNode — fully-extended node on initial load (#1157)', () => {
  it('loads with the derived extended-tolerance + pinned-slice view state for a fully-extended node', async () => {
    const loadLines = vi.fn().mockResolvedValue({ segmentCount: 0 } as LoadedLinesData);
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(loadLines));
    const ctx = makeCtx();
    // Even though lines opt out of the PARTIAL-extend override,
    // `deriveNodeViewState` computes the tolerance + slice pin unconditionally
    // for the full-extend case — so a fully-extended lines node loads its whole
    // extent instead of only the current-slice subset.
    const extendedViewState: ViewState = {
      ...makeViewState(),
      tolerance: [1e10, 1e10, 1e10, 1e10],
    };
    ctx.spies.deriveNodeViewState.mockReturnValue({
      skip: false,
      viewState: extendedViewState,
    });

    await loadLinesNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(loadLines.mock.calls[0][0]).toBe(extendedViewState);
  });
});

describe('loadLinesNode — processLinesData returns null', () => {
  it('skips commitLinesGeometry but still returns the placeholder', async () => {
    const data = { segmentCount: 4 } as LoadedLinesData;
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();
    // ctx.processLinesData defaults to resolving null in makeCtx.

    const placeholder = await loadLinesNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(placeholder).not.toBeNull();
    expect(ctx.spies.processLinesData).toHaveBeenCalledTimes(1);
    expect(ctx.spies.commitLinesGeometry).not.toHaveBeenCalled();
  });
});

describe('loadLinesNode — failure-record clearing', () => {
  // A recovered lazy level must drop out of the outcome report and the
  // auto-retry budget, but only when its commit actually landed.
  it('clears a prior failure record when the commit lands', async () => {
    const data = { segmentCount: 4 } as LoadedLinesData;
    const staged = { path: '/scene/l' } as unknown as StagedLinesCommit;
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();
    ctx.spies.processLinesData.mockResolvedValue(staged);
    ctx.registry.recordFailure('/scene/l', new Error('earlier 503'), 'Network');

    await loadLinesNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(ctx.registry.failedLoaders.has('/scene/l')).toBe(false);
  });

  it('keeps the failure record when staged is null (load landed nowhere)', async () => {
    const data = { segmentCount: 4 } as LoadedLinesData;
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx(); // processLinesData resolves null by default
    ctx.registry.recordFailure('/scene/l', new Error('earlier 503'), 'Network');

    await loadLinesNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    expect(ctx.registry.failedLoaders.has('/scene/l')).toBe(true);
  });
});

describe('loadLinesNode — segmentCount === 0 path', () => {
  it('still runs processLinesData (placeholder needs the empty commit)', async () => {
    const data = { segmentCount: 0 } as LoadedLinesData;
    const staged = { path: '/scene/l' } as unknown as StagedLinesCommit;
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(vi.fn().mockResolvedValue(data)));
    const ctx = makeCtx();
    ctx.spies.processLinesData.mockResolvedValue(staged);

    await loadLinesNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);

    // Lines does NOT short-circuit on 0 segments — unlike a guarded `if` —
    // because the placeholder needs the empty commit to seed its geometry.
    expect(ctx.spies.processLinesData).toHaveBeenCalled();
    expect(ctx.spies.commitLinesGeometry).toHaveBeenCalled();
  });
});

describe('loadLinesNode — error path', () => {
  it('records failure, rethrows LoaderError, placeholder stays attached', async () => {
    const cause = new Error('decode failed');
    createLinesLoaderMock.mockReturnValue(makeLinesLoader(vi.fn().mockRejectedValue(cause)));
    const ctx = makeCtx();
    const parent = new THREE.Group();

    await expect(loadLinesNode(makeSceneNode(), parent, {} as never, ctx)).rejects.toBeInstanceOf(
      LoaderError
    );

    expect(parent.children.length).toBe(1);
    expect(ctx.registry.failedLoaders.has('/scene/l')).toBe(true);
    expect(ctx.registry.failedLoaders.get('/scene/l')?.error).toBe(cause);
  });
});
