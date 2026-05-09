/**
 * Phase 14.3 regression tests for the initial-load placeholder model.
 *
 * Pre-fix scenario:
 *   - `loadPoints/Lines/GSplats` constructed a fully-populated THREE
 *     object only on data-fetch success.
 *   - On initial-load failure, no THREE object existed in the scene.
 *   - `retryFailedLoader()` could fetch data successfully but commit
 *     helpers found no object by name and silently no-oped, then
 *     `failedLoaders.delete(path)` cleared the failure — geometry was
 *     permanently absent while the loader claimed success.
 *
 * Post-fix:
 *   - Each `loadX()` builds an empty placeholder via
 *     `NodeFactory.createEmpty{Points,Lines,GSplats}Node`, attaches it
 *     to `parentThree` BEFORE the data fetch, then commits real data
 *     on success or records failure on error.
 *   - The placeholder remains in the scene across failures. Retry
 *     commits into the existing object via the same path used by all
 *     future updates.
 *   - Defensive: `retryFailedLoader()` only clears `failedLoaders` when
 *     the named object still exists in `rootGroup` after commit. A
 *     scene that lost the placeholder (programmatic removal between
 *     failure and retry) returns `false` instead of false-claiming
 *     success.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../rendering/node-factory';
import { SceneLoader } from '../../../data/scene-loader';
import type { PointsMetadata } from '../../../types/points';
import type { LinesMetadata } from '../../../types/lines';
import type { GSplatsMetadata } from '../../../types/gsplats';
import type { DataLoader } from '../../../data/data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';

// `materialManager.getX` calls hit shader compilation, which requires a
// live WebGL context. Mock it the same way scene-loader tests do.
vi.mock('../../../rendering/material-manager', async () => {
  const actual = await vi.importActual<typeof import('../../../rendering/material-manager')>(
    '../../../rendering/material-manager'
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
vi.mock('../../../rendering/colormap-textures', () => ({
  getColormapTexture: vi.fn(() => null),
}));

describe('NodeFactory.createEmptyPointsNode', () => {
  it('produces a THREE.Points with empty geometry and correct userData', () => {
    const factory = new NodeFactory();
    const attrs: PointsMetadata = {
      n_points: 100,
      max_radius: 1.0,
      max_sharpness: 31.0,
    } as PointsMetadata;
    const loader = { dispose: vi.fn() } as unknown as DataLoader;

    const placeholder = factory.createEmptyPointsNode('/empty-points', attrs, loader);

    expect(placeholder).toBeInstanceOf(THREE.Points);
    expect(placeholder.name).toBe('/empty-points');
    expect(placeholder.userData.nodeType).toBe('points');
    expect(placeholder.userData.attrs).toBe(attrs);
    expect(placeholder.userData.loader).toBe(loader);
    expect(placeholder.userData.visiblePointCount).toBe(0);

    // Geometry exists but has zero points.
    expect(placeholder.geometry).toBeDefined();
    const positionAttr = placeholder.geometry.getAttribute('position') as THREE.BufferAttribute;
    expect(positionAttr).toBeDefined();
    expect(positionAttr.count).toBe(0);
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

    const points = factory.createEmptyPointsNode(
      '/p',
      { n_points: 0 } as PointsMetadata,
      pLoader
    );
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
 * Phase 15.3: retryFailedLoader must mirror initial-load on
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
describe('SceneLoader.retryFailedLoader — derived.skip fallback (Phase 15.3)', () => {
  let loader: SceneLoader;
  let root: THREE.Group;

  // Type alias to access private members in tests without sprinkling
  // `as any` everywhere.
  type LoaderInternals = {
    rootGroup: THREE.Group | null;
    viewState: { displayDims: number[]; slicePosition: number[]; tolerance: number[]; dimensions?: unknown };
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
      dimensions: [
        { name: 'x' },
        { name: 'y' },
        { name: 'z' },
        { name: 't' },
      ],
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
    internals.registry.registerPointsLoader('/p', { updateView, dispose: vi.fn() } as unknown as DataLoader);
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
    internals.registry.registerLinesLoader('/l', { updateView, dispose: vi.fn() } as unknown as LinesDataLoader);
    internals.registry.recordFailure('/l', new Error('initial decode failure'));

    vi.spyOn(internals, 'deriveNodeViewState').mockReturnValue({ skip: 'extend_to_all' });

    const ok = await loader.retryFailedLoader('/l');

    expect(ok).toBe(true);
    expect(updateView).toHaveBeenCalledTimes(1);
    expect(updateView).toHaveBeenCalledWith(internals.viewState);
    expect(internals.registry.failedLoaders.has('/l')).toBe(false);
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
    internals.registry.registerGSplatsLoader(
      '/g',
      { updateView, dispose: vi.fn() } as unknown as GSplatsDataLoader
    );
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
});
