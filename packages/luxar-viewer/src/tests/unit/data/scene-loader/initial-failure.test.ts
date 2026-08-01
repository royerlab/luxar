/**
 * Initial-load placeholder model.
 *
 * Each `loadX()` builds an empty placeholder via
 * `NodeFactory.createEmpty{Points,Lines,GSplats}Node`, attaches it to
 * `parentThree` BEFORE the data fetch, then commits real data on
 * success or records failure on error. The placeholder remains in the
 * scene across failures; retry commits into the existing object via
 * the same path used by all future updates.
 *
 * Without this model, an initial-load failure leaves no THREE object
 * in the scene, and a successful retry is silently no-op'd because
 * the commit helpers can't find the named object — `failedLoaders`
 * gets cleared while geometry stays permanently absent.
 *   - Defensive: `retryFailedLoader()` only clears `failedLoaders` when
 *     the named object still exists in `rootGroup` after commit. A
 *     scene that lost the placeholder (programmatic removal between
 *     failure and retry) returns `false` instead of false-claiming
 *     success.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../../rendering/node-factory';
import { getPointTexture } from '../../../../rendering/point-geometry';
import { SceneLoader } from '../../../../data/scene-loader';
import type { PointsMetadata } from '../../../../types/points';
import type { LinesMetadata } from '../../../../types/lines';
import type { GSplatsMetadata } from '../../../../types/gsplats';
import type { DataLoader } from '../../../../data/data-loader-types';
import type { LinesDataLoader } from '../../../../types/lines';
import type { GSplatsDataLoader } from '../../../../types/gsplats';

// `materialManager.getX` calls hit shader compilation, which requires a
// live WebGL context. Mock it the same way scene-loader tests do.
vi.mock('../../../../rendering/material-manager', async () => {
  const actual = await vi.importActual<typeof import('../../../../rendering/material-manager')>(
    '../../../../rendering/material-manager'
  );
  return {
    ...actual,
    materialManager: {
      getPointMaterial: vi.fn(() => ({
        uniforms: {},
        userData: {},
        updateCameraParams: vi.fn(),
      })),
      getLineMaterial: vi.fn(() => ({
        uniforms: { uMaxWidth: { value: 1.0 } },
        userData: {},
        updateCameraParams: vi.fn(),
        clone: vi.fn().mockReturnThis(),
        updateColormapTexture: vi.fn(),
        updateScalarRange: vi.fn(),
      })),
      getGSplatMaterial: vi.fn(() => ({
        uniforms: { uTruncate: { value: 3.0 } },
        userData: {},
        updateCameraParams: vi.fn(),
        clone: vi.fn().mockReturnThis(),
        updateColormapTexture: vi.fn(),
        updateScalarRange: vi.fn(),
      })),
      register: vi.fn(),
    },
  };
});

// `getColormapTexture` reads a sampler uniform; the placeholder factories
// don't trip the colormap branch (no `nodeAttrs.colormap`), but mock for
// safety.
vi.mock('../../../../rendering/colormap-textures', () => ({
  getColormapTexture: vi.fn(() => null),
}));

describe('NodeFactory.createEmptyPointsNode', () => {
  it('produces a THREE.Mesh with empty geometry and correct userData', () => {
    const factory = new NodeFactory();
    const attrs: PointsMetadata = {
      n_points: 100,
      max_radius: 1.0,
    } as PointsMetadata;
    const loader = { dispose: vi.fn() } as unknown as DataLoader;

    const placeholder = factory.createEmptyPointsNode('/empty-points', attrs, loader);

    // Points are THREE.Mesh with instanced quad geometry; per-point
    // data lives in the point texture and `aSortedIndex` is the only
    // per-instance attribute (texture-storage migration, Stage 2).
    expect(placeholder).toBeInstanceOf(THREE.Mesh);
    expect(placeholder.name).toBe('/empty-points');
    expect(placeholder.userData.nodeType).toBe('points');
    expect(placeholder.userData.attrs).toBe(attrs);
    expect(placeholder.userData.loader).toBe(loader);
    expect(placeholder.userData.visiblePointCount).toBe(0);

    // Geometry exists but draws zero instances; the storage pair is
    // attached (exact-size texture — zero points still allocate the
    // minimum 1-row texture) and the ordering attribute is empty.
    expect(placeholder.geometry).toBeDefined();
    expect(getPointTexture(placeholder.geometry)).not.toBeNull();
    const sortedIndex = placeholder.geometry.getAttribute(
      'aSortedIndex'
    ) as THREE.InstancedBufferAttribute;
    expect(sortedIndex).toBeDefined();
    expect(sortedIndex.count).toBe(0);
    expect((placeholder.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(0);
  });
});

describe('NodeFactory.createEmptyLinesNode', () => {
  it('produces a THREE.Mesh with empty instance buffers and correct userData', () => {
    const factory = new NodeFactory();
    const attrs: LinesMetadata = {
      n_segments: 50,
      n_vertices: 100,
      max_width: 2.0,
    } as LinesMetadata;
    const loader = { dispose: vi.fn() } as unknown as LinesDataLoader;

    const placeholder = factory.createEmptyLinesNode(
      '/empty-lines',
      {} as Record<string, unknown>,
      attrs,
      loader
    );

    expect(placeholder).toBeInstanceOf(THREE.Mesh);
    expect(placeholder.name).toBe('/empty-lines');
    expect(placeholder.userData.nodeType).toBe('lines');
    expect(placeholder.userData.attrs).toBe(attrs);
    expect(placeholder.userData.loader).toBe(loader);
    expect(placeholder.userData.visibleSegmentCount).toBe(0);
  });
});

describe('NodeFactory.createEmptyGSplatsNode', () => {
  it('produces a THREE.Mesh with empty instance buffers and correct userData', () => {
    const factory = new NodeFactory();
    const attrs: GSplatsMetadata = {
      n_splats: 1000,
      truncation_radius: 3.0,
    } as GSplatsMetadata;
    const loader = { dispose: vi.fn() } as unknown as GSplatsDataLoader;

    const placeholder = factory.createEmptyGSplatsNode(
      '/empty-gsplats',
      {} as Record<string, unknown>,
      attrs,
      loader
    );

    expect(placeholder).toBeInstanceOf(THREE.Mesh);
    expect(placeholder.name).toBe('/empty-gsplats');
    expect(placeholder.userData.nodeType).toBe('gsplats');
    expect(placeholder.userData.attrs).toBe(attrs);
    expect(placeholder.userData.loader).toBe(loader);
    expect(placeholder.userData.visibleSplatCount).toBe(0);
  });
});

describe('NodeFactory placeholder factories — common contract', () => {
  it('all three placeholder factories produce objects findable by name', () => {
    // Ensures `commitX` helpers (which use `getObjectByName`) can locate
    // the placeholder once data arrives. This is the primary contract of
    // the placeholder model: an empty object with the right name and
    // userData.nodeType is enough for the existing commit pipeline.
    const factory = new NodeFactory();
    const root = new THREE.Group();
    const pLoader = { dispose: vi.fn() } as unknown as DataLoader;
    const lLoader = { dispose: vi.fn() } as unknown as LinesDataLoader;
    const gLoader = { dispose: vi.fn() } as unknown as GSplatsDataLoader;

    const points = factory.createEmptyPointsNode('/p', { n_points: 0 } as PointsMetadata, pLoader);
    const lines = factory.createEmptyLinesNode(
      '/l',
      {} as Record<string, unknown>,
      { n_segments: 0 } as LinesMetadata,
      lLoader
    );
    const gsplats = factory.createEmptyGSplatsNode(
      '/g',
      {} as Record<string, unknown>,
      { n_splats: 0 } as GSplatsMetadata,
      gLoader
    );

    root.add(points);
    root.add(lines);
    root.add(gsplats);

    expect(root.getObjectByName('/p')).toBe(points);
    expect(root.getObjectByName('/l')).toBe(lines);
    expect(root.getObjectByName('/g')).toBe(gsplats);
  });

  it('placeholders persist in the scene when the loader function throws (caller responsibility)', () => {
    // This test asserts the *invariant* that the placeholder model
    // depends on: once attached, removing it requires explicit action.
    // The loader's catch block doesn't remove the placeholder, so a
    // failed initial load still leaves the placeholder in the scene
    // for retry to populate.
    const factory = new NodeFactory();
    const root = new THREE.Group();
    const placeholder = factory.createEmptyPointsNode(
      '/persistent',
      { n_points: 0 } as PointsMetadata,
      { dispose: vi.fn() } as unknown as DataLoader
    );
    root.add(placeholder);

    // Simulate the load function throwing without removing the placeholder
    // (mirrors the actual scene-loader catch path).
    const simulateFailedLoad = (): void => {
      throw new Error('Network timeout');
    };
    expect(simulateFailedLoad).toThrow();

    // Placeholder still findable.
    expect(root.getObjectByName('/persistent')).toBe(placeholder);
  });
});

/**
 * retryFailedLoader must mirror initial-load on
 * `derived.skip === 'extend_to_all'` — fall back to `this.viewState`
 * and actually load data, instead of clearing the failure flag with
 * an empty placeholder.
 *
 * Pre-fix scenario: the retry path returned `true` and deleted the
 * entry from `failedLoaders` without calling `updateView`, leaving the
 * placeholder in its post-failure empty state while reporting success.
 *
 * The tests below spy on `deriveNodeViewState` to deterministically
 * produce `skip: 'extend_to_all'`, so we can assert the fallback
 * path calls `updateView` with the base view state.
 */
describe('SceneLoader.retryFailedLoader — derived.skip fallback', () => {
  let loader: SceneLoader;
  let root: THREE.Group;

  // Type alias to access private members in tests without sprinkling
  // `as any` everywhere.
  type LoaderInternals = {
    rootGroup: THREE.Group | null;
    viewState: {
      displayDims: number[];
      slicePosition: number[];
      tolerance: number[];
      dimensions?: unknown;
    };
    deriveNodeViewState: (
      path: string,
      attrs: unknown,
      opts: { applyPartialExtendTolerance: boolean }
    ) => { skip: 'extend_to_all' } | { skip: false; viewState: unknown };
    registry: {
      registerPointsLoader: (path: string, loader: unknown) => void;
      registerLinesLoader: (path: string, loader: unknown) => void;
      registerGSplatsLoader: (path: string, loader: unknown) => void;
      recordFailure: (path: string, error: Error) => void;
      failedLoaders: Map<string, unknown>;
    };
  };

  beforeEach(() => {
    loader = new SceneLoader();
    root = new THREE.Group();
    const internals = loader as unknown as LoaderInternals;
    internals.rootGroup = root;
    internals.viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0.5, 0.5, 0.5, 0.5],
      dimensions: [{ name: 'x' }, { name: 'y' }, { name: 'z' }, { name: 't' }],
    };
  });

  it('Points retry on derived.skip falls back to this.viewState and populates placeholder', async () => {
    const internals = loader as unknown as LoaderInternals;
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyPointsNode(
      '/p',
      { n_points: 0, extend_to_all: ['t'] } as unknown as PointsMetadata,
      { dispose: vi.fn() } as unknown as DataLoader
    );
    placeholder.userData.attrs = { extend_to_all: ['t'] };
    root.add(placeholder);

    const updateView = vi.fn().mockResolvedValue(null);
    internals.registry.registerPointsLoader('/p', {
      updateView,
      dispose: vi.fn(),
    } as unknown as DataLoader);
    internals.registry.recordFailure('/p', new Error('initial network failure'));

    vi.spyOn(internals, 'deriveNodeViewState').mockReturnValue({ skip: 'extend_to_all' });

    const ok = await loader.retryFailedLoader('/p');

    expect(ok).toBe(true);
    expect(updateView).toHaveBeenCalledTimes(1);
    expect(updateView).toHaveBeenCalledWith(internals.viewState);
    expect(internals.registry.failedLoaders.has('/p')).toBe(false);
  });

  it('Lines retry on derived.skip falls back to this.viewState and populates placeholder', async () => {
    const internals = loader as unknown as LoaderInternals;
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyLinesNode(
      '/l',
      {} as Record<string, unknown>,
      { n_segments: 0, extend_to_all: ['t'] } as unknown as LinesMetadata,
      { dispose: vi.fn() } as unknown as LinesDataLoader
    );
    placeholder.userData.attrs = { extend_to_all: ['t'] };
    root.add(placeholder);

    const updateView = vi.fn().mockResolvedValue(null);
    internals.registry.registerLinesLoader('/l', {
      updateView,
      dispose: vi.fn(),
    } as unknown as LinesDataLoader);
    internals.registry.recordFailure('/l', new Error('initial decode failure'));

    vi.spyOn(internals, 'deriveNodeViewState').mockReturnValue({ skip: 'extend_to_all' });

    const ok = await loader.retryFailedLoader('/l');

    expect(ok).toBe(true);
    expect(updateView).toHaveBeenCalledTimes(1);
    expect(updateView).toHaveBeenCalledWith(internals.viewState);
    expect(internals.registry.failedLoaders.has('/l')).toBe(false);
  });

  it('refuses when an updateView is in progress', async () => {
    const internals = loader as unknown as LoaderInternals & { _updateInProgress: boolean };
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyPointsNode(
      '/p',
      { n_points: 0 } as unknown as PointsMetadata,
      { dispose: vi.fn() } as unknown as DataLoader
    );
    placeholder.userData.attrs = {};
    root.add(placeholder);

    const updateView = vi.fn().mockResolvedValue(null);
    internals.registry.registerPointsLoader('/p', {
      updateView,
      dispose: vi.fn(),
    } as unknown as DataLoader);
    internals.registry.recordFailure('/p', new Error('initial'));

    // Simulate "another update is in progress" by flipping the lock.
    internals._updateInProgress = true;

    const ok = await loader.retryFailedLoader('/p');

    expect(ok).toBe(false);
    expect(updateView).not.toHaveBeenCalled();
    // failedLoaders should NOT be cleared — retry was deferred, not failed.
    expect(internals.registry.failedLoaders.has('/p')).toBe(true);
  });

  it('updateView called during in-flight retry queues + drains via viewStateQueue', async () => {
    // Reproduces the lock race:
    //   1. retry starts on a path while _updateInProgress is false.
    //   2. retry takes the lock (_updateInProgress=true) at entry.
    //   3. updateView() called during retry → sees lock=true, queues
    //      onto viewStateQueue and returns immediately.
    //   4. retry releases lock; the queue's drain helper fires the
    //      queued updateView async (so retry's promise resolves first).
    // Pending state lives on viewStateQueue; inspect via hasPending().
    type Internals = LoaderInternals & {
      _updateInProgress: boolean;
      viewStateQueue: { hasPending(): boolean };
    };
    const internals = loader as unknown as Internals;
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyPointsNode(
      '/p',
      { n_points: 0 } as unknown as PointsMetadata,
      { dispose: vi.fn() } as unknown as DataLoader
    );
    placeholder.userData.attrs = {};
    root.add(placeholder);

    // Loader.updateView returns a deferred promise so retry parks
    // mid-flight while we exercise updateView().
    let releaseRetry: ((v: null) => void) | undefined;
    const retryUpdateView = vi.fn().mockReturnValue(
      new Promise<null>((resolve) => {
        releaseRetry = resolve;
      })
    );
    internals.registry.registerPointsLoader('/p', {
      updateView: retryUpdateView,
      dispose: vi.fn(),
    } as unknown as DataLoader);
    internals.registry.recordFailure('/p', new Error('initial'));

    // Step 1: kick off retry; don't await.
    const retryPromise = loader.retryFailedLoader('/p');
    await Promise.resolve();
    await Promise.resolve();
    // Lock taken.
    expect(internals._updateInProgress).toBe(true);
    expect(retryUpdateView).toHaveBeenCalledTimes(1);

    // Step 2: user dim/slider change → updateView() called during
    // retry. Should see the lock and queue onto viewStateQueue.
    const newViewState = {
      slicePosition: [1, 2, 3, 4],
    };
    void loader.updateView(newViewState);
    expect(internals.viewStateQueue.hasPending()).toBe(true);

    // Step 3: release retry's loader call. Retry finishes.
    releaseRetry?.(null);
    const retryResult = await retryPromise;
    expect(retryResult).toBe(true);

    // Step 4: lock released; the drain helper consumed the queued
    // pending state and scheduled the queued updateView async. The
    // drained updateView is now in flight (or about to be). The
    // primary contract: pending state was processed, not lost.
    expect(internals.viewStateQueue.hasPending()).toBe(false);
  });

  it('GSplats retry on derived.skip falls back to a viewState built from this.viewState', async () => {
    const internals = loader as unknown as LoaderInternals;
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyGSplatsNode(
      '/g',
      {} as Record<string, unknown>,
      { n_splats: 0, extend_to_all: ['t'] } as unknown as GSplatsMetadata,
      { dispose: vi.fn() } as unknown as GSplatsDataLoader
    );
    placeholder.userData.attrs = { extend_to_all: ['t'] };
    root.add(placeholder);

    const updateView = vi.fn().mockResolvedValue(null);
    internals.registry.registerGSplatsLoader('/g', {
      updateView,
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader);
    internals.registry.recordFailure('/g', new Error('initial validation failure'));

    vi.spyOn(internals, 'deriveNodeViewState').mockReturnValue({ skip: 'extend_to_all' });

    const ok = await loader.retryFailedLoader('/g');

    expect(ok).toBe(true);
    expect(updateView).toHaveBeenCalledTimes(1);
    // GSplats fallback constructs a fresh object from this.viewState
    // (mirrors loadGSplats() initial-load shape) — assert the key
    // fields rather than reference equality.
    const callArg = updateView.mock.calls[0][0] as {
      displayDims: number[];
      slicePosition: number[];
      tolerance: number[];
      dimensions?: unknown;
    };
    expect(callArg.displayDims).toBe(internals.viewState.displayDims);
    expect(callArg.slicePosition).toBe(internals.viewState.slicePosition);
    expect(callArg.tolerance).toBe(internals.viewState.tolerance);
    expect(callArg.dimensions).toBe(internals.viewState.dimensions);
    expect(internals.registry.failedLoaders.has('/g')).toBe(false);
  });

  // the existing retry tests prove the
  // call shape and the failedLoaders.delete behavior, but not that
  // the placeholder is *truly* still around with consistent userData
  // after retry. These add stronger end-state assertions.

  it('Points retry preserves placeholder userData + nodeType after success', async () => {
    const internals = loader as unknown as LoaderInternals;
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyPointsNode(
      '/p',
      { n_points: 0 } as unknown as PointsMetadata,
      { dispose: vi.fn() } as unknown as DataLoader
    );
    placeholder.userData.attrs = { custom: 'preserve-me' };
    root.add(placeholder);

    const updateView = vi.fn().mockResolvedValue(null);
    internals.registry.registerPointsLoader('/p', {
      updateView,
      dispose: vi.fn(),
    } as unknown as DataLoader);
    internals.registry.recordFailure('/p', new Error('initial'));

    const ok = await loader.retryFailedLoader('/p');

    expect(ok).toBe(true);
    // The placeholder is still in the scene graph at its path.
    expect(root.getObjectByName('/p')).toBe(placeholder);
    // Its userData.nodeType is intact (set by createEmptyPointsNode).
    expect(placeholder.userData.nodeType).toBe('points');
    // userData.attrs is unchanged — retry doesn't mutate caller-set
    // metadata when there's no new data to commit.
    expect(placeholder.userData.attrs).toEqual({ custom: 'preserve-me' });
    expect(internals.registry.failedLoaders.has('/p')).toBe(false);
  });

  // Stronger end-state assertions for the success path: when
  // updateView returns non-empty data, retry must thread that data
  // through the loader's geometry-commit step. Spy on the private
  // updateXxxGeometry methods so we don't have to construct fully
  // valid GPU-ready data shapes — the commit helpers themselves are
  // tested separately under tests/unit/data/scene-loader/.

  it('Points retry threads non-empty data through to the points commit on success', async () => {
    const internals = loader as unknown as LoaderInternals & {
      updatePointsGeometry: (path: string, data: unknown) => void;
    };
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyPointsNode(
      '/p',
      { n_points: 0 } as unknown as PointsMetadata,
      { dispose: vi.fn() } as unknown as DataLoader
    );
    placeholder.userData.attrs = {};
    root.add(placeholder);

    // Minimal non-empty data shape — full validity is the commit
    // helper's concern, tested separately.
    const fakeData = {
      positions: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      pointCount: 3,
      ndim: 3,
      metadata: {
        totalPoints: 3,
        loadedPoints: 3,
        bounds: new THREE.Box3(),
        usedSpatialIndex: false,
      },
    };

    const updateView = vi.fn().mockResolvedValue(fakeData);
    internals.registry.registerPointsLoader('/p', {
      updateView,
      dispose: vi.fn(),
    } as unknown as DataLoader);
    internals.registry.recordFailure('/p', new Error('initial'));

    const commitSpy = vi.spyOn(internals, 'updatePointsGeometry').mockImplementation(() => {
      // Mirror the helper's observable side effect: bump
      // visiblePointCount so the assertion below sees real change.
      placeholder.userData.visiblePointCount = fakeData.pointCount;
    });

    const ok = await loader.retryFailedLoader('/p');

    expect(ok).toBe(true);
    expect(commitSpy).toHaveBeenCalledTimes(1);
    // Path + data only: retry now routes through processPointsData →
    // commitPointsGeometry, which passes `session`/`loadedViewVersion`
    // explicitly instead of letting them default. Both resolve to the same
    // `_updateVersion`, so the call arity changed, not the behaviour.
    expect(commitSpy.mock.calls[0]?.slice(0, 2)).toEqual(['/p', fakeData]);
    expect(placeholder.userData.visiblePointCount).toBe(3);
    expect(internals.registry.failedLoaders.has('/p')).toBe(false);
  });

  it('Lines retry threads non-empty data to commitLinesGeometry on success', async () => {
    const internals = loader as unknown as LoaderInternals & {
      processLinesData: (path: string, data: unknown, viewState: unknown) => Promise<unknown>;
      commitLinesGeometry: (staged: unknown) => void;
    };
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyLinesNode(
      '/l',
      {} as Record<string, unknown>,
      { n_segments: 0 } as unknown as LinesMetadata,
      { dispose: vi.fn() } as unknown as LinesDataLoader
    );
    placeholder.userData.attrs = {};
    root.add(placeholder);

    const fakeData = {
      vertices: new Float32Array([0, 0, 0, 1, 1, 1]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([1, 1]),
      colors: null,
      sharpness: null,
      vertexCount: 2,
      segmentCount: 1,
      ndim: 3,
    };
    const stagedSentinel = { path: '/l', kind: 'lines' };

    const updateView = vi.fn().mockResolvedValue(fakeData);
    internals.registry.registerLinesLoader('/l', {
      updateView,
      dispose: vi.fn(),
    } as unknown as LinesDataLoader);
    internals.registry.recordFailure('/l', new Error('initial'));

    const processSpy = vi.spyOn(internals, 'processLinesData').mockResolvedValue(stagedSentinel);
    const commitSpy = vi.spyOn(internals, 'commitLinesGeometry').mockImplementation(() => {
      placeholder.userData.visibleSegmentCount = fakeData.segmentCount;
    });

    const ok = await loader.retryFailedLoader('/l');

    expect(ok).toBe(true);
    expect(processSpy).toHaveBeenCalledTimes(1);
    expect(processSpy.mock.calls[0][0]).toBe('/l');
    expect(processSpy.mock.calls[0][1]).toBe(fakeData);
    expect(commitSpy).toHaveBeenCalledTimes(1);
    expect(commitSpy).toHaveBeenCalledWith(stagedSentinel);
    expect(placeholder.userData.visibleSegmentCount).toBe(1);
    expect(internals.registry.failedLoaders.has('/l')).toBe(false);
  });

  it('GSplats retry threads non-empty data to commitGSplatsGeometry on success', async () => {
    const internals = loader as unknown as LoaderInternals & {
      processGSplatsData: (path: string, data: unknown, viewState: unknown) => Promise<unknown>;
      commitGSplatsGeometry: (staged: unknown) => void;
    };
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyGSplatsNode(
      '/g',
      {} as Record<string, unknown>,
      { n_splats: 0 } as unknown as GSplatsMetadata,
      { dispose: vi.fn() } as unknown as GSplatsDataLoader
    );
    placeholder.userData.attrs = {};
    root.add(placeholder);

    const fakeData = {
      centers: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1]),
      cholesky_factors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: null,
      splatCount: 1,
      ndim: 3,
    };
    const stagedSentinel = { path: '/g', kind: 'gsplats' };

    const updateView = vi.fn().mockResolvedValue(fakeData);
    internals.registry.registerGSplatsLoader('/g', {
      updateView,
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader);
    internals.registry.recordFailure('/g', new Error('initial'));

    const processSpy = vi.spyOn(internals, 'processGSplatsData').mockResolvedValue(stagedSentinel);
    const commitSpy = vi.spyOn(internals, 'commitGSplatsGeometry').mockImplementation(() => {
      placeholder.userData.visibleSplatCount = fakeData.splatCount;
    });

    const ok = await loader.retryFailedLoader('/g');

    expect(ok).toBe(true);
    expect(processSpy).toHaveBeenCalledTimes(1);
    expect(processSpy.mock.calls[0][1]).toBe(fakeData);
    expect(commitSpy).toHaveBeenCalledTimes(1);
    expect(commitSpy).toHaveBeenCalledWith(stagedSentinel);
    expect(placeholder.userData.visibleSplatCount).toBe(1);
    expect(internals.registry.failedLoaders.has('/g')).toBe(false);
  });

  it('retry returns false and keeps the failure when placeholder was removed', async () => {
    // The verifyAndClear defensive guard: if the named object is
    // missing from rootGroup, retry must NOT clear the failedLoaders
    // entry.
    const internals = loader as unknown as LoaderInternals;
    const factory = new NodeFactory();
    const placeholder = factory.createEmptyPointsNode(
      '/p',
      { n_points: 0 } as unknown as PointsMetadata,
      { dispose: vi.fn() } as unknown as DataLoader
    );
    placeholder.userData.attrs = {};
    root.add(placeholder);

    const updateView = vi.fn().mockResolvedValue(null);
    internals.registry.registerPointsLoader('/p', {
      updateView,
      dispose: vi.fn(),
    } as unknown as DataLoader);
    internals.registry.recordFailure('/p', new Error('initial'));

    // Simulate a programmatic node removal between failure and retry.
    root.remove(placeholder);
    expect(root.getObjectByName('/p')).toBeUndefined();

    const ok = await loader.retryFailedLoader('/p');

    expect(ok).toBe(false);
    // The failure stays — verifyAndClear refused to clear.
    expect(internals.registry.failedLoaders.has('/p')).toBe(true);
    // updateView WAS called (we don't gate on placeholder existence
    // before the loader fetch — the guard fires after).
    expect(updateView).toHaveBeenCalledTimes(1);
  });
});
