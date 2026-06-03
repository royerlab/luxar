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
    releaseLazyGSplats: vi.fn(),
    applyEffectiveAttrs,
    deriveNodeViewState,
    connectLoaderToMonitor,
    updatePointsGeometry: vi.fn(),
    processLinesData: vi.fn(),
    commitLinesGeometry: vi.fn(),
    processGSplatsData,
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
    expect(ctx.spies.commitGSplatsGeometry).toHaveBeenCalledWith(staged);
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
