// @vitest-environment jsdom
/**
 * Comprehensive tests for SceneLoader
 *
 * Tests the orchestration of scene loading, spatial index integration,
 * hierarchical scene graph construction, and view updates.
 *
 * AUDIT NOTE (data.md W6 / W8 — open audit acknowledgment):
 *   The `loadScene` and `updateView` describe blocks lean heavily on
 *   `expect(scene).toBeDefined()` plus a single substantive check
 *   (scene.name, sceneDimensions, etc.). The substantive check pins
 *   the most load-bearing contract per test, but a fully rebalanced
 *   suite would either (a) load against the real zarr fixtures in
 *   `packages/luxar-viewer/tests/fixtures/` or (b) directly assert
 *   the assembled scene-graph shape (child counts, names, transform
 *   matrix values) for each branch. Both are structural refactors
 *   declared OUT OF SCOPE for this audit pass — kept visible here so
 *   the next test-quality pass can pick them up.
 *
 *   The W3 (material creation) and W4 (monitor integration) blocks
 *   below were strengthened in-place.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SceneLoader, type LoaderConfig, type ViewState } from '../../../data';
import type { ViewStateQueue } from '../../../data/scene-loader/view-state/view-state-queue';
import { releaseDepthSortNode } from '../../../rendering/depth-sort-coordinator';
import * as THREE from 'three';
import { getPointTexture } from '../../../rendering/point-geometry';
import { resolveLinePrimitiveForNode } from '../../../types/line-primitive';
import * as zarr from 'zarrita';
import { ArchiveFaultError } from '../../../cache/chunk-source';
import {
  LODGroupRegistry,
  type LODGroupChild,
  type LODGroupEntry,
} from '../../../scene/lod-group-registry';
import type { NodeBuildCtx } from '../../../data/scene-loader/nodes/build-ctx';
import { getLoadTimeline, resetLoadTimeline } from '../../../profiling/load-timeline';
import * as gsplatsRefinement from '../../../data/gsplats/lod-refinement';
import { log, Modules } from '../../../utils/log';

// THREE is NOT mocked here. The classes SceneLoader touches —
// Group / Points / Mesh / Box3 / Vector3 / Matrix4 /
// {,Instanced}Buffer{Geometry,Attribute} — are pure JS and run fine in
// jsdom; the WebGL-bound layer (renderers, shaders) is one level up.
// Earlier revisions kept a 165-line stand-in so individual constructor
// calls could be counted, but the resulting tests asserted on
// implementation details rather than behavior. The behavior assertions
// further down (transform validation, scene-graph shape, etc.) are
// stronger when run against real THREE.

// Mock zarrita (external dependency - network I/O for zarr stores)
vi.mock('zarrita', () => ({
  FetchStore: vi.fn(),
  withMaybeConsolidatedMetadata: vi.fn(),
  registry: {},
  root: vi.fn(),
  // `open.v2` / `open.v3` are pinned per-format siblings on the real module,
  // and the facade's v3-first root open calls `open.v3` directly. Alias all
  // three to ONE mock so a test that configures `open` still governs the root
  // open — separate stubs would silently resolve `undefined` instead.
  open: (() => {
    const openMock = vi.fn();
    return Object.assign(openMock, { v2: openMock, v3: openMock });
  })(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

// Mock material manager (depends on WebGL shader compilation - must be mocked).
// NOTE: shader-side material plumbing is intentionally untested in jsdom; the
// pure factory paths are covered separately in `material-manager.test.ts`.
vi.mock('../../../rendering/material-manager', () => ({
  materialManager: {
    getPointMaterial: vi.fn().mockReturnValue({
      uniforms: {},
      vertexShader: '',
      fragmentShader: '',
      userData: {},
      updateCameraParams: vi.fn(),
    }),
    getLineMaterial: vi.fn().mockReturnValue({
      uniforms: {},
      vertexShader: '',
      fragmentShader: '',
      userData: {},
      updateCameraParams: vi.fn(),
    }),
    createLinePickingMaterial: vi.fn().mockReturnValue({ userData: {} }),
    register: vi.fn(),
  },
  // The barrel re-exports the soft-dispose sentinel; supply a stand-in
  // Symbol so any barrel consumer resolves the import in jsdom even
  // though MaterialManager itself is mocked away. (The dispatcher in
  // `invalidate-render-object.ts` now imports the flag from its leaf
  // module, `material-manager/soft-dispose-flag.ts`, not this barrel.)
  SOFT_DISPOSE_FLAG: Symbol.for('luxar.material.softDispose.test-mock'),
}));

// The lazy-LOD demotion path (ctx.releaseLazyGSplats) must also drop the
// demoted level's depth-sort coordinator state (worker-side transferred
// centers). Partial mock via importOriginal so only releaseDepthSortNode is
// intercepted — SceneLoader's commit path imports noteDepthSortCommit from the
// same module and must keep the real implementation.
vi.mock('../../../rendering/depth-sort-coordinator', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../rendering/depth-sort-coordinator')>();
  return {
    ...actual,
    releaseDepthSortNode: vi.fn(),
  };
});

// SceneLoader now uses `notifier.toast` for the >16D scene-dimensions
// warning. Mock the notifier so the test can assert toast() was called.
const notifierMocks = vi.hoisted(() => ({
  toast: vi.fn(),
  error: vi.fn(),
  clearError: vi.fn(),
}));
vi.mock('../../../utils/cross-layer/notifier', () => ({
  notifier: {
    toast: notifierMocks.toast,
    error: notifierMocks.error,
    showHelp: vi.fn(),
    hideHelp: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
    clearError: notifierMocks.clearError,
    showSceneIdentityBanner: vi.fn(),
    hideSceneIdentityBanner: vi.fn(),
  },
}));

// NOTE: Previous mocks for DataLoadingMonitor ('../ui/data-loading-monitor'),
// PointsSpatialIndexLoader ('../data/points-spatial-index-loader'), and
// DataMonitorManager ('../data/data-monitor-manager') were removed because
// their paths were relative to the test file location (src/tests/unit/data/)
// and resolved to non-existent modules, making them dead code that never
// intercepted any real imports. The SceneLoader's actual imports resolve
// from src/data/ and are not affected by those mock paths.
//
// If mocking these becomes necessary in the future, use paths relative to
// the test file that resolve to the actual source modules, e.g.:
//   vi.mock('../../../data/points-spatial-index-loader', ...)
//   vi.mock('../../../data/data-monitor-manager', ...)

describe('SceneLoader', () => {
  let sceneLoader: SceneLoader;
  let mockStore: any;
  let mockRootLoc: any;
  let mockZarrGroup: any;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();

    // Setup mock store
    mockStore = {
      contents: vi.fn().mockResolvedValue([
        { path: '/', kind: 'group' },
        { path: '/points', kind: 'group' },
      ]),
    };

    // Setup mock zarr location
    mockRootLoc = {
      resolve: vi.fn().mockImplementation((_path) => ({
        resolve: vi.fn().mockImplementation((_subpath) => ({
          resolve: vi.fn(),
        })),
      })),
    };

    // Setup mock zarr group
    mockZarrGroup = {
      attrs: {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'z', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'time', unit: 's', range: [0, 10], display: false, step: 0.1 },
          ],
        },
      },
    };

    // Mock zarrita functions
    (zarr.FetchStore as any).mockImplementation(() => mockStore);
    (zarr as any).withMaybeConsolidatedMetadata.mockResolvedValue(mockStore);
    (zarr.root as any).mockReturnValue(mockRootLoc);
    (zarr.open as any).mockResolvedValue(mockZarrGroup);

    // Create SceneLoader instance
    sceneLoader = new SceneLoader();
  });

  afterEach(() => {
    sceneLoader.dispose();
  });

  describe('loadScene', () => {
    it('should load a scene with correct URL normalization', async () => {
      const url = 'http://localhost:8000/test.zarr';
      const scene = await sceneLoader.loadScene(url);

      // Verify scene loaded correctly (implementation may use FetchStore or MultiLevelCachingStore)
      expect((zarr as any).withMaybeConsolidatedMetadata).toHaveBeenCalled();
      expect(scene).toBeDefined();
      expect(scene.name).toBe('LuxarScene');
    });

    it('should initialize scene dimensions from metadata', async () => {
      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(scene.userData.sceneDimensions).toEqual(mockZarrGroup.attrs.scene_dimensions);
    });

    it('should handle missing scene dimensions gracefully', async () => {
      // [data.md/W6][P2] Strengthen: also pin (a) the scene is a real
      // THREE.Group (not just a truthy stub) and (b) that userData lacks
      // any sceneDimensions-adjacent keys — guards against a regression
      // that wrote the wrong key (e.g. `scene_dimensions`) and silently
      // satisfied the original toBeDefined() check.
      mockZarrGroup.attrs = {}; // No scene_dimensions

      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(scene).toBeDefined();
      expect(scene.name).toBe('LuxarScene');
      expect(scene.userData.sceneDimensions).toBeUndefined();
      // Defensive: confirm the snake_case mistake is not silently present.
      expect((scene.userData as Record<string, unknown>).scene_dimensions).toBeUndefined();
    });

    it('should build scene graph hierarchy correctly', async () => {
      // Setup hierarchical structure
      mockStore.contents.mockResolvedValue([
        { path: '/', kind: 'group' },
        { path: '/group1', kind: 'group' },
        { path: '/group1/points', kind: 'group' },
      ]);

      // Mock checking for spatial index
      mockRootLoc.resolve.mockImplementation((_path: any) => ({
        resolve: vi.fn().mockImplementation((subpath) => {
          if (subpath === 'spatial_index') {
            // Simulate spatial index exists for points
            if (_path.includes('points')) {
              return Promise.resolve({});
            }
            throw new Error('No spatial index');
          }
          return { resolve: vi.fn() };
        }),
      }));

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      // Verify scene graph was built
      expect(mockRootLoc.resolve).toHaveBeenCalled();
    });

    it('installs aggregate line load before constructing sibling materials', async () => {
      mockStore.contents.mockResolvedValue([
        { path: '/', kind: 'group' },
        { path: '/lines_a', kind: 'group' },
        { path: '/lines_b', kind: 'group' },
      ]);
      (zarr.open as any).mockImplementation((loc: any) =>
        Promise.resolve(
          loc === mockRootLoc ? mockZarrGroup : { attrs: { type: 'lines', n_segments: 1_200_000 } }
        )
      );

      const { materialManager } = await import('../../../rendering/material-manager');
      const getLineMaterial = materialManager.getLineMaterial as ReturnType<typeof vi.fn>;

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(resolveLinePrimitiveForNode({ nSegments: 100 })).toBe('screen-space');
      expect(getLineMaterial).toHaveBeenCalledTimes(2);
      expect(getLineMaterial.mock.calls.map(([config]) => config.primitive)).toEqual([
        'screen-space',
        'screen-space',
      ]);
    });

    it('should detect and log extend_to_all dimensions', async () => {
      const consoleSpy = vi.spyOn(console, 'log');

      // The root is a SCENE container (its type is 'scene'/absent); the
      // extend_to_all lives on the `/points` NODE. The root and node attrs
      // must differ — a leaf `type` on the root would make buildSceneGraph
      // (correctly) treat the file as a bare-leaf standalone node and skip
      // the child. mockRootLoc is the root location; resolved sub-locations
      // are the nodes.
      (zarr.open as any).mockImplementation((loc: any) =>
        Promise.resolve(
          loc === mockRootLoc
            ? mockZarrGroup // scene root (scene_dimensions, no leaf type)
            : { attrs: { type: 'points', extend_to_all: ['time', 'channel'] } }
        )
      );

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('extend_to_all'));
    });

    it('should handle missing spatial index gracefully for 3D points', async () => {
      // Setup points group without spatial index
      mockStore.contents.mockResolvedValue([
        { path: '/', kind: 'group' },
        { path: '/points', kind: 'group' },
      ]);

      // The root is a SCENE container; the `/points` NODE is the points leaf.
      // Root and node attrs must differ — a leaf `type` on the root would make
      // buildSceneGraph treat the file as a bare-leaf standalone node (marking
      // the whole store internal) and skip the child. mockRootLoc is the root.
      (zarr.open as any).mockImplementation((loc: any) =>
        Promise.resolve(
          loc === mockRootLoc
            ? mockZarrGroup // scene root (scene_dimensions, no leaf type)
            : { attrs: { type: 'points', n_points: 1000 } }
        )
      );

      // Mock resolve to simulate missing spatial index
      mockRootLoc.resolve.mockImplementation((_path: any) => ({
        resolve: vi.fn().mockImplementation((subpath) => {
          if (subpath === 'spatial_index') {
            throw new Error('Not found');
          }
          return { resolve: vi.fn() };
        }),
      }));

      // Should NOT throw an error - handles missing spatial index gracefully for 3D datasets
      // [data.md/W6][P2] Strengthen: previously only asserted scene was defined.
      // The production contract on the missing-spatial-index branch is
      // (a) the loadScene call resolves with a real LuxarScene root, and
      // (b) the SceneLoader still attempts to register a loader for the
      // points node — the actual `load-all` fallback lives inside the
      // PointsSpatialIndexLoader, not in SceneLoader. So we pin scene
      // shape AND the existence of the registered loader entry.
      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(scene).toBeDefined();
      expect(scene.name).toBe('LuxarScene');
      const loaders = (sceneLoader as any).loaders as Map<string, unknown>;
      expect(loaders.has('/points')).toBe(true);
    });
  });

  describe('updateView', () => {
    it('does not report a budget-truncated pass as the settled view', async () => {
      const committed = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 4],
        tolerance: [0, 0, 0, 0.5],
      } satisfies ViewState;

      await sceneLoader.updateView({ ...committed, frameBudgetMs: 8 });
      const stored = (sceneLoader as unknown as { viewState: ViewState }).viewState;
      expect(sceneLoader.isAtViewState(stored)).toBe(false);

      await sceneLoader.updateView(stored);
      expect(sceneLoader.isAtViewState(stored)).toBe(true);
    });

    beforeEach(async () => {
      // Load a scene first
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });

    it('should update all loaders with new view state', async () => {
      // A slice that DIFFERS from the one loadScene derives from the scene
      // dimensions (time midpoint 5, tolerance = step 0.1): re-submitting that
      // exact state is, by the version contract, a bump-free "Resyncing view"
      // pass rather than an "Updating view" one.
      const viewState: Partial<ViewState> = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 6],
        tolerance: [0, 0, 0, 0.1],
      };

      await sceneLoader.updateView(viewState);

      // Verify loaders were updated (check through mock)
      expect(console.log).toHaveBeenCalledWith(expect.stringContaining('Updating view'));
    });

    it('should handle loader update failures gracefully', async () => {
      const consoleSpy = vi.spyOn(console, 'error');

      // Make one loader fail - include dispose method
      const mockLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Update failed')),
        dispose: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/failing', mockLoader);

      const viewState: Partial<ViewState> = {
        displayDims: [0, 1, 2],
      };

      await sceneLoader.updateView(viewState);

      // Should log error with improved retry tracking format
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          '[❌] [SceneLoader] Failed to update /failing (attempt 1): Update failed'
        )
      );

      // Should track the failure
      expect(sceneLoader.hasFailures()).toBe(true);
      expect(sceneLoader.getFailedLoaders().size).toBe(1);
    });

    it('unwinds a progressive loader when its geometry commit fails', async () => {
      const rollbackToPassStart = vi.fn().mockReturnValue(1);
      const loaders = (sceneLoader as any).loaders as Map<string, unknown>;
      loaders.clear();
      loaders.set('/commit-fail', {
        updateView: vi.fn().mockResolvedValue({
          pointCount: 1,
          positions: new Float32Array([1, 2, 3]),
          metadata: { loadedPoints: 1 },
        }),
        rollbackToPassStart,
        dispose: vi.fn(),
      });
      vi.spyOn(sceneLoader as any, 'updatePointsGeometry').mockImplementation(() => {
        throw new Error('GPU commit failed');
      });

      await expect(sceneLoader.updateView({ displayDims: [0, 1, 2] })).rejects.toThrow(
        AggregateError
      );

      expect(rollbackToPassStart).toHaveBeenCalledOnce();
    });

    it('surfaces an archive fault once and preserves the last committed frame', async () => {
      const fault = new ArchiveFaultError(
        'The archive URL has expired. Refresh the page with a new URL.',
        'https://example.test/scene.zip'
      );
      const onArchiveFault = vi.fn();
      sceneLoader.onArchiveFault(onArchiveFault);
      sceneLoader.onArchiveFault(() => {
        throw new Error('consumer failure');
      });
      const removedListener = vi.fn();
      sceneLoader.onArchiveFault(removedListener)();
      const failingLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('loader wrapper', { cause: fault })),
        dispose: vi.fn(),
      };
      const successfulLoader = {
        updateView: vi.fn().mockResolvedValue({
          pointCount: 1,
          positions: new Float32Array([1, 2, 3]),
          metadata: { loadedPoints: 1 },
        }),
        dispose: vi.fn(),
      };
      const ordinaryFailureLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('ordinary node failure')),
        dispose: vi.fn(),
      };
      const commitSpy = vi.spyOn(sceneLoader as any, 'updatePointsGeometry');
      const prefetch = vi.fn();
      const releaseShadows = vi.fn();
      (sceneLoader as any)._slicePrefetcher = { prefetch, releaseShadows, dispose: vi.fn() };
      const loaders = (sceneLoader as any).loaders as Map<string, unknown>;
      loaders.clear();
      loaders.set('/ordinary-failure', ordinaryFailureLoader);

      await sceneLoader.updateView({ displayDims: [0, 1, 2] });

      expect(sceneLoader.getFailedLoaders().has('/ordinary-failure')).toBe(true);
      ordinaryFailureLoader.updateView.mockClear();

      loaders.set('/fault', failingLoader);
      loaders.set('/cached-success', successfulLoader);

      await sceneLoader.updateView({ slicePosition: [0, 0, 1] });

      expect(notifierMocks.error).toHaveBeenCalledOnce();
      expect(notifierMocks.error).toHaveBeenCalledWith(fault.message, { persistent: true });
      expect(sceneLoader.archiveFault).toBe(fault);
      expect(onArchiveFault).toHaveBeenCalledOnce();
      expect(onArchiveFault).toHaveBeenCalledWith(fault);
      expect(removedListener).not.toHaveBeenCalled();
      expect(sceneLoader.hasFailures()).toBe(false);
      expect(commitSpy).not.toHaveBeenCalled();
      expect(releaseShadows).toHaveBeenCalledOnce();

      await sceneLoader.updateView({ slicePosition: [0, 0, 3] });
      sceneLoader.prefetchSlice({ slicePosition: [0, 0, 3] }, 5);

      expect(failingLoader.updateView).toHaveBeenCalledOnce();
      expect(successfulLoader.updateView).toHaveBeenCalledOnce();
      expect(ordinaryFailureLoader.updateView).toHaveBeenCalledOnce();
      expect(notifierMocks.error).toHaveBeenCalledOnce();
      expect(onArchiveFault).toHaveBeenCalledOnce();
      expect(prefetch).not.toHaveBeenCalled();
      expect((sceneLoader as any)._updateInProgress).toBe(false);
    });

    it('notifies again when a cleared archive fault recurs', () => {
      const listener = vi.fn();
      const firstFault = new ArchiveFaultError('first expiry', 'scene.zip');
      const secondFault = new ArchiveFaultError('second expiry', 'scene.zip');
      const internals = sceneLoader as unknown as {
        reportArchiveFault(fault: ArchiveFaultError): void;
        clearArchiveFaultForRetry(): void;
      };
      sceneLoader.onArchiveFault(listener);

      internals.reportArchiveFault(firstFault);
      internals.clearArchiveFaultForRetry();
      internals.reportArchiveFault(secondFault);

      expect(listener).toHaveBeenCalledTimes(2);
      expect(listener).toHaveBeenNthCalledWith(1, firstFault);
      expect(listener).toHaveBeenNthCalledWith(2, secondFault);
      expect(sceneLoader.archiveFault).toBe(secondFault);
    });

    it('can replay the latched archive fault to a late subscriber', () => {
      const fault = new ArchiveFaultError('archive unavailable', 'scene.zip');
      (sceneLoader as any)._archiveFault = fault;
      const listener = vi.fn();
      const throwingListener = vi.fn(() => {
        throw new Error('listener boom');
      });
      let unsubscribeThrowingListener: (() => void) | undefined;

      const unsubscribe = sceneLoader.onArchiveFault(listener, { replayCurrent: true });
      expect(() => {
        unsubscribeThrowingListener = sceneLoader.onArchiveFault(throwingListener, {
          replayCurrent: true,
        });
      }).not.toThrow();

      expect(listener).toHaveBeenCalledOnce();
      expect(listener).toHaveBeenCalledWith(fault);
      expect(throwingListener).toHaveBeenCalledOnce();

      unsubscribe();
      unsubscribeThrowingListener!();
      (sceneLoader as any).notifyArchiveFault(fault);
      expect(listener).toHaveBeenCalledOnce();
      expect(throwingListener).toHaveBeenCalledOnce();
    });

    it('does not notify listeners added during archive-fault delivery twice', () => {
      const fault = new ArchiveFaultError('archive unavailable', 'scene.zip');
      const lateListener = vi.fn();
      sceneLoader.onArchiveFault(() => {
        sceneLoader.onArchiveFault(lateListener, { replayCurrent: true });
      });
      (sceneLoader as any)._archiveFault = fault;

      (sceneLoader as any).notifyArchiveFault(fault);

      expect(lateListener).toHaveBeenCalledOnce();
      expect(lateListener).toHaveBeenCalledWith(fault);
    });

    it('reports archive faults from node-build contexts exactly once', () => {
      const firstFault = new ArchiveFaultError('archive unavailable', 'scene.zip');
      const secondFault = new ArchiveFaultError('archive still unavailable', 'scene.zip');
      const releaseShadows = vi.fn();
      (sceneLoader as any)._slicePrefetcher = {
        prefetch: vi.fn(),
        releaseShadows,
        dispose: vi.fn(),
      };
      const internals = sceneLoader as unknown as {
        registry: { recordFailure(path: string, error: Error): void };
      };
      internals.registry.recordFailure('/lazy-failure', new Error('archive request failed'));
      expect(sceneLoader.hasFailures()).toBe(true);
      let failuresAtNotification: boolean | undefined;
      let releaseCallsAtNotification: number | undefined;
      const listener = vi.fn(() => {
        failuresAtNotification = sceneLoader.hasFailures();
        releaseCallsAtNotification = releaseShadows.mock.calls.length;
      });
      sceneLoader.onArchiveFault(listener);
      const ctx = (
        sceneLoader as unknown as { makeNodeBuildCtx(): Record<string, unknown> }
      ).makeNodeBuildCtx();
      const reportArchiveFault = ctx.reportArchiveFault as
        ((fault: ArchiveFaultError) => void) | undefined;

      reportArchiveFault!(firstFault);
      reportArchiveFault!(secondFault);

      expect(sceneLoader.archiveFault).toBe(firstFault);
      expect(listener).toHaveBeenCalledOnce();
      expect(listener).toHaveBeenCalledWith(firstFault);
      expect(failuresAtNotification).toBe(false);
      expect(releaseCallsAtNotification).toBe(1);
      expect(notifierMocks.error).toHaveBeenCalledOnce();
      expect(notifierMocks.error).toHaveBeenCalledWith(firstFault.message, { persistent: true });
      expect(sceneLoader.hasFailures()).toBe(false);
      expect(releaseShadows).toHaveBeenCalledOnce();
    });

    it('drains a superseded waiter when an archive fault stops the active pass', async () => {
      const fault = new ArchiveFaultError('archive unavailable', 'https://example.test/scene.zip');
      let rejectUpdate!: (error: Error) => void;
      const failingLoader = {
        updateView: vi.fn().mockImplementation(
          () =>
            new Promise<never>((_resolve, reject) => {
              rejectUpdate = reject;
            })
        ),
        dispose: vi.fn(),
      };
      const loaders = (sceneLoader as any).loaders as Map<string, unknown>;
      loaders.clear();
      loaders.set('/fault', failingLoader);

      const faultingPass = sceneLoader.updateView({ slicePosition: [0, 0, 1] });
      const queuedPass = sceneLoader.updateView({ slicePosition: [0, 0, 2] });
      rejectUpdate(fault);

      await Promise.all([faultingPass, queuedPass]);

      expect(failingLoader.updateView).toHaveBeenCalledOnce();
      expect(notifierMocks.error).toHaveBeenCalledOnce();
      expect((sceneLoader as any)._updateInProgress).toBe(false);
    });

    it('routes the updateView call through the loader even for empty results (loader decides skip)', async () => {
      // Renamed (was 'should not update geometry when no points are loaded')
      // — the only assertion was that the loader's updateView was called, NOT
      // that geometry-update was skipped. The test name lied about the contract
      // (data.md, C5). Honest contract pinned here: updateView is invoked once.
      const mockLoader = {
        updateView: vi.fn().mockResolvedValue({
          metadata: { loadedPoints: 0 },
        }),
        dispose: vi.fn(),
        getCacheStats: vi.fn().mockReturnValue({}),
        clearCache: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/empty', mockLoader);

      await sceneLoader.updateView({});

      expect(mockLoader.updateView).toHaveBeenCalledTimes(1);
    });
  });

  describe('updateView — version contract (bump only on a query-determinant change)', () => {
    // The LOD registry judges a level's freshness by EXACT equality of its
    // commit-time `loadedViewVersion` stamp with `currentViewVersion`, and lazy
    // lod_group levels are never re-stamped by the sweep. So a pass that does
    // not change the view (a partition resync, a retry re-run, a depth-sort
    // re-commit) must NOT bump the version — bumping invalidated every resident
    // fine level scene-wide and dropped all groups to coarse on camera motion.
    const base: Partial<ViewState> = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 3],
      tolerance: [0, 0, 0, 0.5],
    };

    beforeEach(async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      await sceneLoader.updateView(base);
    });

    it('updateView({}) (a reprocess) leaves currentViewVersion unchanged', async () => {
      const before = sceneLoader.currentViewVersion;
      await sceneLoader.updateView({});
      expect(sceneLoader.currentViewVersion).toBe(before);
      // requestReprocess is the public door for the same pass.
      sceneLoader.requestReprocess();
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(sceneLoader.currentViewVersion).toBe(before);
    });

    it('re-submitting an equal view state does not bump', async () => {
      const before = sceneLoader.currentViewVersion;
      await sceneLoader.updateView({ ...base });
      await sceneLoader.updateView({ slicePosition: [...base.slicePosition!] });
      expect(sceneLoader.currentViewVersion).toBe(before);
    });

    it.each([
      { field: 'slicePosition', next: { slicePosition: [0, 0, 0, 4] } },
      { field: 'displayDims', next: { displayDims: [0, 1, 3] } },
      { field: 'tolerance (displayed dim)', next: { tolerance: [1, 0, 0, 0.5] } },
    ])('a changed $field bumps the version by exactly one', async ({ next }) => {
      const before = sceneLoader.currentViewVersion;
      await sceneLoader.updateView(next);
      expect(sceneLoader.currentViewVersion).toBe(before + 1);
      // And re-submitting the now-current state is again a no-bump pass.
      await sceneLoader.updateView(next);
      expect(sceneLoader.currentViewVersion).toBe(before + 1);
    });

    it('a targeted resync sweeps only loaders under the target paths and forgets no baseline', async () => {
      const makeLoader = () => ({
        updateView: vi.fn().mockResolvedValue(null),
        dispose: vi.fn(),
      });
      const part0 = makeLoader();
      const part1 = makeLoader();
      const other = makeLoader();
      const loaders = (sceneLoader as unknown as { loaders: Map<string, unknown> }).loaders;
      loaders.set('/tiled/part_0/level_0', part0);
      loaders.set('/tiled/part_1/level_0', part1);
      loaders.set('/other', other);
      const forgetPathSpy = vi.spyOn(
        (sceneLoader as unknown as { viewStateQueue: ViewStateQueue }).viewStateQueue,
        'forgetPath'
      );
      const before = sceneLoader.currentViewVersion;

      sceneLoader.requestReprocess(['/tiled/part_1']);
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(part1.updateView).toHaveBeenCalledTimes(1);
      expect(part0.updateView).not.toHaveBeenCalled();
      expect(other.updateView).not.toHaveBeenCalled();
      expect(forgetPathSpy).not.toHaveBeenCalled();
      expect(sceneLoader.currentViewVersion).toBe(before);
    });

    it('a changed view ignores targeted resync options and sweeps every loader', async () => {
      const target = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
      const other = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
      const loaders = (sceneLoader as unknown as { loaders: Map<string, unknown> }).loaders;
      loaders.set('/tiled/part_1/level_0', target);
      loaders.set('/other', other);
      const before = sceneLoader.currentViewVersion;

      await sceneLoader.updateView(
        { slicePosition: [0, 0, 0, 9] },
        { resyncPaths: new Set(['/tiled/part_1']) }
      );

      expect(sceneLoader.currentViewVersion).toBe(before + 1);
      expect(target.updateView).toHaveBeenCalledTimes(1);
      expect(other.updateView).toHaveBeenCalledTimes(1);
    });

    it('a targeted resync preserves frame-budget refine debt until an untargeted pass', async () => {
      const stored = (sceneLoader as unknown as { viewState: ViewState }).viewState;

      await sceneLoader.updateView({ ...stored, frameBudgetMs: 8 });
      expect(sceneLoader.isAtViewState(stored)).toBe(false);

      sceneLoader.requestReprocess(['/tiled/part_1']);
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(sceneLoader.isAtViewState(stored)).toBe(false);

      sceneLoader.requestReprocess();
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(sceneLoader.isAtViewState(stored)).toBe(true);
    });

    it('a targeted resync arriving mid-pass neither aborts the pass nor is lost', async () => {
      let releaseGate!: () => void;
      const gate = new Promise<void>((resolve) => {
        releaseGate = resolve;
      });
      let capturedSignal: AbortSignal | undefined;
      const gated = {
        updateView: vi.fn(async (_vs: unknown, _session: unknown, signal?: AbortSignal) => {
          if (!capturedSignal) {
            capturedSignal = signal;
            await gate;
            signal?.throwIfAborted();
          }
          return null;
        }),
        dispose: vi.fn(),
      };
      const target = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
      const loaders = (sceneLoader as unknown as { loaders: Map<string, unknown> }).loaders;
      loaders.set('/node', gated);
      loaders.set('/tiled/part_1/level_0', target);

      // A real view change runs synchronously into the gated loader and parks.
      const p1 = sceneLoader.updateView({ slicePosition: [0, 0, 0, 7] });
      await Promise.resolve();
      expect(capturedSignal).toBeInstanceOf(AbortSignal);
      expect(target.updateView).toHaveBeenCalledTimes(1);

      // The resync must not take the supersede branch: no abort, no waiter.
      sceneLoader.requestReprocess(['/tiled/part_1']);
      await Promise.resolve();
      expect(capturedSignal?.aborted).toBe(false);

      releaseGate();
      await p1;
      // queueNext re-enters with the parked resync at the next frame (rAF in
      // jsdom), so poll rather than assume a timer granularity.
      await vi.waitFor(() => {
        expect(target.updateView).toHaveBeenCalledTimes(2); // full pass + targeted resync
      });
      expect(gated.updateView).toHaveBeenCalledTimes(1); // not a target — untouched
      expect((sceneLoader as unknown as { _updateInProgress: boolean })._updateInProgress).toBe(
        false
      );
    });

    it('a targeted resync arriving during a refinement HOLD queues its own pass instead of waiting', async () => {
      // A refinement run holds the serialization lock with no view pass in
      // flight. Nothing of updateView's `finally` will run, so the resync must
      // queue a pending state (which the refinement loop cancels into) rather
      // than park behind a pass that never ends. Before this, a re-entering
      // part sat on a stale slice until every ladder finished streaming.
      const target = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
      const other = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
      const loaders = (sceneLoader as unknown as { loaders: Map<string, unknown> }).loaders;
      loaders.set('/tiled/part_1/level_0', target);
      loaders.set('/other', other);
      const internals = sceneLoader as unknown as {
        _updateInProgress: boolean;
        _refining: boolean;
        _updateAbortController: AbortController | null;
        viewStateQueue: ViewStateQueue;
        takeQueuedResyncOpts(): { resyncPaths?: ReadonlySet<string> };
      };
      internals._updateInProgress = true;
      internals._refining = true;
      const controller = new AbortController();
      internals._updateAbortController = controller;
      expect(sceneLoader.isLoadPassInProgress()).toBe(false); // a hold is not a pass

      sceneLoader.requestReprocess(['/tiled/part_1']);
      await Promise.resolve();

      // Queued, not parked; the in-flight refinement is NOT aborted.
      expect(internals.viewStateQueue.hasPending()).toBe(true);
      expect(sceneLoader.isLoadPassInProgress()).toBe(true); // registry now defers
      expect(controller.signal.aborted).toBe(false);
      expect(target.updateView).not.toHaveBeenCalled();

      // The refinement loop's cancellation hand-off: release the lock and
      // re-enter with the pending state plus the queued resync opts.
      internals._updateInProgress = false;
      internals._refining = false;
      const pending = internals.viewStateQueue.takePending() ?? {};
      const before = sceneLoader.currentViewVersion;
      await sceneLoader.updateView(pending, internals.takeQueuedResyncOpts());

      expect(target.updateView).toHaveBeenCalledTimes(1);
      expect(other.updateView).not.toHaveBeenCalled();
      expect(sceneLoader.currentViewVersion).toBe(before);
      expect(internals.takeQueuedResyncOpts()).toEqual({}); // consumed exactly once
    });

    it('a real view change queued after a hold resync sweeps every loader', async () => {
      const target = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
      const other = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
      const loaders = (sceneLoader as unknown as { loaders: Map<string, unknown> }).loaders;
      loaders.set('/tiled/part_1/level_0', target);
      loaders.set('/other', other);
      const internals = sceneLoader as unknown as {
        _updateInProgress: boolean;
        _refining: boolean;
        _updateAbortController: AbortController | null;
        viewStateQueue: ViewStateQueue;
        takeQueuedResyncOpts(): { resyncPaths?: ReadonlySet<string> };
      };
      internals._updateInProgress = true;
      internals._refining = true;
      internals._updateAbortController = new AbortController();

      sceneLoader.requestReprocess(['/tiled/part_1']);
      await Promise.resolve();
      const queuedPass = sceneLoader.updateView({ slicePosition: [0, 0, 0, 9] });

      internals._updateInProgress = false;
      internals._refining = false;
      const pending = internals.viewStateQueue.takePending() ?? {};
      const before = sceneLoader.currentViewVersion;
      await sceneLoader.updateView(pending, internals.takeQueuedResyncOpts());
      await queuedPass;

      expect(sceneLoader.currentViewVersion).toBe(before + 1);
      expect(target.updateView).toHaveBeenCalledTimes(1);
      expect(other.updateView).toHaveBeenCalledTimes(1);
      expect(internals.takeQueuedResyncOpts()).toEqual({});
    });

    it('stashed resync paths never leak into a later, unrelated same-view pass', async () => {
      // A resync stashed during a hold may be drained by a path that does not
      // hand the paths over (the refinement run's final release, a retry's
      // resume). The stash must then be gone: a later `updateView({})` queued
      // behind a pass — the depth-sort reprocess whose whole purpose is to
      // re-commit stamp-less nodes — would otherwise re-enter narrowed to
      // loaders that have nothing to do with it.
      const target = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
      const other = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
      const loaders = (sceneLoader as unknown as { loaders: Map<string, unknown> }).loaders;
      loaders.set('/tiled/part_1/level_0', target);
      loaders.set('/other', other);
      const internals = sceneLoader as unknown as {
        _updateInProgress: boolean;
        _refining: boolean;
        _queuedResyncPaths: Set<string> | null;
        viewStateQueue: ViewStateQueue;
        reenterPending(state: Partial<ViewState>): Promise<void>;
      };
      internals._updateInProgress = true;
      internals._refining = true;
      sceneLoader.requestReprocess(['/tiled/part_1']);
      await Promise.resolve();
      expect(internals._queuedResyncPaths?.size).toBe(1);

      // Every drain site goes through `reenterPending`, which consumes the stash.
      internals._updateInProgress = false;
      internals._refining = false;
      const drained = internals.viewStateQueue.drain((state) => internals.reenterPending(state));
      expect(drained).toBe(true);
      await vi.waitFor(() => {
        expect(target.updateView).toHaveBeenCalledTimes(1);
      });
      expect(other.updateView).not.toHaveBeenCalled(); // the resync itself WAS targeted
      expect(internals._queuedResyncPaths).toBeNull();

      // Belt and braces: a stash left behind by any other route is dropped the
      // moment a pass starts, so an untargeted reprocess stays untargeted.
      internals._queuedResyncPaths = new Set(['/tiled/part_1']);
      await sceneLoader.updateView({});
      expect(other.updateView).toHaveBeenCalledTimes(1);
      expect(internals._queuedResyncPaths).toBeNull();
    });

    it('a second rising edge during the same hold merges into the stash instead of being dropped', async () => {
      const internals = sceneLoader as unknown as {
        _updateInProgress: boolean;
        _refining: boolean;
        _queuedResyncPaths: Set<string> | null;
        viewStateQueue: ViewStateQueue;
      };
      internals._updateInProgress = true;
      internals._refining = true;
      sceneLoader.requestReprocess(['/tiled/part_1']);
      sceneLoader.requestReprocess(['/tiled/part_3']); // our own `{}` is already pending
      await Promise.resolve();
      expect([...(internals._queuedResyncPaths ?? [])].sort()).toEqual([
        '/tiled/part_1',
        '/tiled/part_3',
      ]);
      expect(internals.viewStateQueue.hasPending()).toBe(true);
    });

    it.each([
      { order: 'untargeted reprocess FIRST, then a rising edge' },
      { order: 'rising edge FIRST, then an untargeted reprocess' },
    ])(
      'an untargeted reprocess queued during a hold is never narrowed ($order)',
      async ({ order }) => {
        // The depth-sort coordinator's `requestReprocess()` exists to re-commit
        // nodes whose stamps it just deleted, scene-wide. Whichever way it
        // interleaves with a partition rising edge during a refinement hold,
        // the hand-off must run a FULL sweep.
        const target = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
        const other = { updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() };
        const loaders = (sceneLoader as unknown as { loaders: Map<string, unknown> }).loaders;
        loaders.set('/tiled/part_1/level_0', target);
        loaders.set('/other', other);
        const internals = sceneLoader as unknown as {
          _updateInProgress: boolean;
          _refining: boolean;
          _updateAbortController: AbortController | null;
          viewStateQueue: ViewStateQueue;
          reenterPending(state: Partial<ViewState>): Promise<void>;
        };
        internals._updateInProgress = true;
        internals._refining = true;
        internals._updateAbortController = new AbortController();
        if (order.startsWith('untargeted')) {
          void sceneLoader.updateView({}); // depth-sort style: no paths → supersede branch
          sceneLoader.requestReprocess(['/tiled/part_1']);
        } else {
          sceneLoader.requestReprocess(['/tiled/part_1']);
          void sceneLoader.updateView({});
        }
        await Promise.resolve();
        expect(internals.viewStateQueue.hasPending()).toBe(true);

        internals._updateInProgress = false;
        internals._refining = false;
        const drained = internals.viewStateQueue.drain((state) => internals.reenterPending(state));
        expect(drained).toBe(true);
        await vi.waitFor(() => {
          expect(target.updateView).toHaveBeenCalledTimes(1);
          expect(other.updateView).toHaveBeenCalledTimes(1); // full sweep, not narrowed
        });
      }
    );

    it('a queued re-entry that lands on a latched archive fault flushes the parked waiters', async () => {
      // A lazy level can latch the fault OUTSIDE any pass. A waiter parked by a
      // superseded updateView must then settle (resolve-only) rather than hang
      // the dimension-animation pacing gate for the rest of the session.
      const internals = sceneLoader as unknown as {
        _archiveFault: ArchiveFaultError | null;
        _passWaiters: Array<() => void>;
      };
      let settled = false;
      internals._passWaiters.push(() => {
        settled = true;
      });
      internals._archiveFault = new ArchiveFaultError('container unreadable', 'test');
      await sceneLoader.updateView({ slicePosition: [0, 0, 0, 8] });
      expect(settled).toBe(true);
      expect(internals._passWaiters).toHaveLength(0);
    });
  });

  describe('updateView — dispose race', () => {
    it('resolves (does not hang) when called after dispose while the update lock is held', async () => {
      // Reproduce the dispose race: a refinement pass holds the update lock
      // (_updateInProgress) when dispose() runs, so a late updateView — e.g. an
      // in-flight dimension-animation tick landing during the dispose() await —
      // takes the "queue and park a waiter" branch. dispose() flushes waiters
      // once BEFORE teardown, and the refinement loop's isActive-return exit
      // hands off via noopReleaseLock without draining, so nothing ever resolves
      // that parked waiter. The _disposed guard at updateView's head must settle
      // it immediately instead (resolve-only, never reject).
      (sceneLoader as unknown as { _updateInProgress: boolean })._updateInProgress = true;
      await sceneLoader.dispose();

      await expect(
        Promise.race([
          sceneLoader.updateView({ slicePosition: [0, 0, 0, 1] }),
          new Promise<never>((_resolve, reject) =>
            setTimeout(() => reject(new Error('updateView hung after dispose')), 1000)
          ),
        ])
      ).resolves.toBeUndefined();
    });
  });

  // #1639 — `isLoadPassInProgress()` is what the debug snapshot's `isLoading`
  // reports, so its scope has to be exactly "data is still arriving for the view
  // the user asked for". The serialization lock is both too broad and too
  // narrow for that: the update tail and the post-load kick hand it to the
  // progressive-LOD refinement run, which only releases it after every additive
  // ladder has drained (long after the view committed), while a view-state
  // QUEUED behind that hold has not started loading at all yet. These tests pin
  // the subtraction and the addition, and pin that `isUpdateInProgress()`
  // (adaptive DPR, init pipeline) keeps its broad "the lock is held" meaning.
  describe('isLoadPassInProgress — refinement excluded, queued state included', () => {
    /** The two private flags the two accessors are composed from. */
    type LockFlags = { _updateInProgress: boolean; _refining: boolean };

    it('is true during a load pass and false during a refinement hold', () => {
      const flags = sceneLoader as unknown as LockFlags;

      expect(sceneLoader.isLoadPassInProgress()).toBe(false);
      expect(sceneLoader.isUpdateInProgress()).toBe(false);

      // An updateView sweep: lock held, not refining.
      flags._updateInProgress = true;
      expect(sceneLoader.isLoadPassInProgress()).toBe(true);
      expect(sceneLoader.isUpdateInProgress()).toBe(true);

      // The same lock, now handed to refinement — the load pass is over.
      flags._refining = true;
      expect(sceneLoader.isLoadPassInProgress()).toBe(false);
      // …but the broader accessor its existing consumers read is unchanged.
      expect(sceneLoader.isUpdateInProgress()).toBe(true);
    });

    it('scheduleGSplatsRefinement holds _refining for the whole run', async () => {
      const flags = sceneLoader as unknown as LockFlags;
      // Mirror the hand-off the real kick sites perform: the lock is already
      // held when the orchestrator is entered.
      flags._updateInProgress = true;

      const run = (
        sceneLoader as unknown as { scheduleGSplatsRefinement: () => Promise<void> }
      ).scheduleGSplatsRefinement();

      // Set SYNCHRONOUSLY, before the first await — otherwise a poll landing in
      // the gap between the hand-off and the first phase would read a load pass.
      expect(flags._refining).toBe(true);
      expect(sceneLoader.isLoadPassInProgress()).toBe(false);

      await run;

      // Cleared on the way out (the `finally`), so the next real load pass is
      // visible again.
      expect(flags._refining).toBe(false);
    });

    it('dispose leaves a torn-down loader reading idle, whatever state it was in', async () => {
      const flags = sceneLoader as unknown as LockFlags;
      const queue = (sceneLoader as unknown as { viewStateQueue: ViewStateQueue }).viewStateQueue;
      // Defence in depth, not a production bug fix: every SceneLoaderManager
      // disposal path detaches the loader before disposing it, so the aggregate
      // cannot observe a disposed loader anyway. This pins that a disposed
      // loader is self-consistently idle regardless — a phase that bails before
      // `finalReleaseLock` runs leaves the lock set, and nothing later clears
      // it.
      flags._updateInProgress = true;
      flags._refining = true;
      // …and a QUEUED view-state is the third input to the predicate, so a
      // disposal must clear it too: a disposed loader never runs its pending
      // pass, so a slot left filled reads busy forever. Parked through the real
      // supersede branch (the lock is held), which also leaves a pass waiter
      // for dispose() to flush.
      const parked = sceneLoader.updateView({ slicePosition: [1, 0, 0, 0] });
      expect(queue.hasPending()).toBe(true);

      await sceneLoader.dispose();

      expect(flags._updateInProgress).toBe(false);
      expect(flags._refining).toBe(false);
      expect(queue.hasPending()).toBe(false);
      expect(sceneLoader.isLoadPassInProgress()).toBe(false);
      expect(sceneLoader.isUpdateInProgress()).toBe(false);
      // dispose() flushes parked waiters (resolve-only), so this cannot hang.
      await parked;
    });

    it('counts a view-state queued behind a refinement hold as a load pass', async () => {
      const flags = sceneLoader as unknown as LockFlags;
      const internals = sceneLoader as unknown as {
        viewStateQueue: ViewStateQueue;
        resolvePassWaiters(): void;
      };
      const queue = internals.viewStateQueue;

      // The steady state right after a commit on any laddered dataset: the
      // update tail handed the lock to the refinement orchestrator.
      flags._updateInProgress = true;
      flags._refining = true;
      expect(sceneLoader.isLoadPassInProgress()).toBe(false);

      // A keyboard nav lands: driven through the REAL supersede branch (which
      // runs synchronously, before updateView's first await), so this pins the
      // writer as well as the predicate — updateView parks the state and
      // returns without touching either flag. The slice the user asked for has
      // NOT begun loading, so this must not read idle.
      const parked = sceneLoader.updateView({ slicePosition: [1, 0, 0] });
      expect(sceneLoader.isLoadPassInProgress()).toBe(true);

      // Cleared at the moment the next pass starts (queueNext / the refinement
      // loop's cancellation check / finalReleaseLock's drain all take it), so
      // the queued clause cannot latch busy once a pass is running.
      expect(queue.takePending()).not.toBeNull();
      expect(sceneLoader.isLoadPassInProgress()).toBe(false);

      // Unpark the caller (no real pass will run here) and drop the simulated
      // hold so the shared afterEach dispose sees a clean loader.
      internals.resolvePassWaiters();
      await parked;
      flags._updateInProgress = false;
      flags._refining = false;
    });
  });

  // The post-load refinement kick in `lifecycle/load-scene.ts` fires from
  // inside `loadScene`, holding the serialization lock. Its `.catch` is the
  // belt-and-braces double-fault path (each refinement loop releases the lock
  // in its own `finally`), and it must be symmetric with its two siblings —
  // `update-view/queue-next.ts` and `kickRefinementIfIdle` — because the
  // pending slot is routinely occupied in exactly this window: the init
  // pipeline's first `updateAllNDNodes` → `updateView` lands while the kick
  // holds the lock and parks its state. A slot nothing drains stranded the
  // user's initial slice, and now also latches `isLoadPassInProgress()` true.
  describe('post-load refinement kick — a rejection drains the pending slot', () => {
    it('releases the lock, drains the queued view-state and settles waiters', async () => {
      const internals = sceneLoader as unknown as {
        makeNodeBuildCtx(): unknown;
        scheduleGSplatsRefinement(): Promise<void>;
        gsplatLoaders: Map<string, unknown>;
        viewStateQueue: ViewStateQueue;
        _updateInProgress: boolean;
      };

      // Register a progressive loader mid-load so the post-load kick fires at
      // all: `makeNodeBuildCtx()` is called once, immediately before the
      // recursive node walk that would normally register it.
      const realMakeNodeBuildCtx = internals.makeNodeBuildCtx.bind(internals);
      vi.spyOn(internals, 'makeNodeBuildCtx').mockImplementation(() => {
        internals.gsplatLoaders.set('progressive-stub', { hasMoreLODs: true });
        return realMakeNodeBuildCtx();
      });

      const updateViewSpy = vi.spyOn(sceneLoader, 'updateView');
      const navState = { slicePosition: [1, 0, 0, 0] };
      let parked: Promise<void> | null = null;
      vi.spyOn(internals, 'scheduleGSplatsRefinement').mockImplementation(async () => {
        // The lock is held by the kick, so this takes updateView's supersede
        // branch and parks the state — the real init-pipeline race.
        parked = sceneLoader.updateView(navState);
        // Drop the stub so the drained re-entry is an ordinary (loader-free)
        // pass and cannot kick refinement a second time.
        internals.gsplatLoaders.clear();
        throw new Error('orchestrator glue died outside the refinement finallys');
      });

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      // The rejection handler runs on a microtask after the kick's promise
      // settles, and `drain` re-enters updateView one microtask later — a
      // macrotask hop flushes both.
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(parked).not.toBeNull();
      // Drained: the slot the parked nav filled is empty again…
      expect(internals.viewStateQueue.hasPending()).toBe(false);
      // …because it was re-entered as a fresh pass with that exact state.
      expect(updateViewSpy).toHaveBeenCalledTimes(2);
      expect(updateViewSpy.mock.calls[1][0]).toEqual(navState);
      // The parked waiter settles — here through the re-entered pass's own
      // queueNext (the handler only resolves waiters itself when nothing was
      // queued, so it can't release the pacing gate ahead of the commit the
      // caller asked for).
      await parked!;
      // The lock the kick took is released, and the drained pass released its
      // own — a torn-down double fault leaves the loader idle, not latched.
      await updateViewSpy.mock.results[1].value;
      expect(internals._updateInProgress).toBe(false);
      expect(sceneLoader.isLoadPassInProgress()).toBe(false);
    });
  });

  describe('updateView — does NOT abort the background deepen prefetch', () => {
    beforeEach(async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });

    function armPrefetcher(): { abortInFlight: ReturnType<typeof vi.fn> } {
      // Lazily create the prefetcher via the public API, then spy on it.
      sceneLoader.prefetchSlice({ slicePosition: [0, 0, 0, 6] }, 10);
      const prefetcher = (sceneLoader as unknown as { _slicePrefetcher: unknown })
        ._slicePrefetcher as { abortInFlight: () => void };
      expect(prefetcher).toBeTruthy();
      const abortSpy = vi.spyOn(prefetcher, 'abortInFlight');
      return { abortInFlight: abortSpy as unknown as ReturnType<typeof vi.fn> };
    }

    // The foreground now commits from the SliceCache without decoding fine
    // levels, so the background deepen must SURVIVE across ticks — a cold LOD
    // level outlives one frame, and a per-tick abort would never let it
    // complete + cache a level (playback quality could then never climb across
    // loops). The shadow runs on its own loader instances + worker decode with
    // cache-keyed writes, so it can't stall or corrupt a foreground tick.
    it('leaves an in-flight prefetch running at updateView entry (main branch)', async () => {
      const spy = armPrefetcher();
      await sceneLoader.updateView({ slicePosition: [0, 0, 0, 7] });
      expect(spy.abortInFlight).not.toHaveBeenCalled();
    });

    it('leaves the prefetch running in the QUEUED branch too', async () => {
      const spy = armPrefetcher();
      const internals = sceneLoader as unknown as {
        _updateInProgress: boolean;
        resolvePassWaiters(): void;
      };
      internals._updateInProgress = true; // simulate an in-flight pass
      const parked = sceneLoader.updateView({ slicePosition: [0, 0, 0, 8] });
      expect(spy.abortInFlight).not.toHaveBeenCalled();
      internals.resolvePassWaiters(); // unpark (the simulated pass "completes")
      await parked;
      internals._updateInProgress = false;
    });

    it('prefetchSlice never mutates the persistent view state (per-pass shadow copy)', () => {
      const before = JSON.stringify((sceneLoader as unknown as { viewState: unknown }).viewState);
      sceneLoader.prefetchSlice({ slicePosition: [9, 9, 9, 9], frameBudgetMs: 99 }, 10);
      expect(JSON.stringify((sceneLoader as unknown as { viewState: unknown }).viewState)).toBe(
        before
      );
    });

    it('releasePrefetchResources is a safe no-op before any prefetch', () => {
      expect(() => sceneLoader.releasePrefetchResources()).not.toThrow();
    });

    it('does NOT persist the transient `prefetch` directive into the view state', async () => {
      // Regression: `prefetch` (like `frameBudgetMs`) is a per-pass directive.
      // If it leaked into the persistent view state, every subsequent foreground
      // store would be pinned, silently defeating scan eviction.
      await sceneLoader.updateView({ slicePosition: [0, 0, 0, 4], prefetch: true });
      const vs = (sceneLoader as unknown as { viewState: { prefetch?: boolean } }).viewState;
      expect(vs.prefetch).toBeUndefined();
    });
  });

  describe('updateView — superseded loads abort (per-update AbortSignal)', () => {
    beforeEach(async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });

    // Three-geometry symmetry: the same supersede→abort contract must hold for
    // Points, Lines, and GSplats. Each registers its fake loader in the
    // matching registry map; the loader returns `null` on the winning pass so
    // the handler short-circuits before any processX/commit (no real mesh).
    const cases = [
      { type: 'points', map: 'loaders' },
      { type: 'lines', map: 'linesLoaders' },
      { type: 'gsplats', map: 'gsplatLoaders' },
    ] as const;

    it.each(cases)(
      'aborts the in-flight $type load when a newer view-state supersedes it; no false failure',
      async ({ map }) => {
        let capturedSignal: AbortSignal | undefined;
        let releaseFirst!: () => void;
        const firstGate = new Promise<void>((resolve) => {
          releaseFirst = resolve;
        });
        let calls = 0;

        const mockLoader = {
          loadPoints: vi.fn(),
          loadLines: vi.fn(),
          loadGSplats: vi.fn(),
          updateView: vi.fn(async (_vs: unknown, _session: unknown, signal?: AbortSignal) => {
            calls += 1;
            if (calls === 1) {
              // In-flight (to-be-superseded) load: park until released, then
              // bail exactly like zarrita's between-chunk throwIfAborted.
              capturedSignal = signal;
              await firstGate;
              signal?.throwIfAborted();
            }
            return null; // winning pass: null → handler returns before commit
          }),
          dispose: vi.fn(),
        };
        (sceneLoader as unknown as Record<string, Map<string, unknown>>)[map].set(
          '/node',
          mockLoader
        );

        const forgetPathSpy = vi.spyOn(
          (sceneLoader as unknown as { viewStateQueue: { forgetPath: (p: string) => void } })
            .viewStateQueue,
          'forgetPath'
        );

        // First update runs synchronously into loader.updateView and parks on
        // the gate (so _updateInProgress is true and the signal is captured).
        const p1 = sceneLoader.updateView({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 1],
          tolerance: [0, 0, 0, 0],
        });
        await Promise.resolve();

        // Second update supersedes the in-flight one → must abort its signal.
        // Do NOT await it yet: the queued promise now resolves only when the
        // winning pass completes, which can't happen until releaseFirst() —
        // awaiting here would deadlock (the pre-fix behavior resolved
        // immediately; that's exactly the pacing bug this guards against).
        const p2 = sceneLoader.updateView({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 2],
          tolerance: [0, 0, 0, 0],
        });
        await Promise.resolve();

        expect(capturedSignal).toBeInstanceOf(AbortSignal);
        expect(capturedSignal?.aborted).toBe(true);

        // Release the gated load; its AbortError must be classified as
        // superseded — NOT recorded as a loader failure, and the prefetch
        // baseline (forgetPath) must be left intact.
        releaseFirst();
        await p1;
        // Let queueNext re-enter with the winning state and settle; the
        // queued promise resolves once that winning pass commits.
        await new Promise((resolve) => setTimeout(resolve, 0));
        await p2;

        expect(sceneLoader.hasFailures()).toBe(false);
        expect(sceneLoader.getFailedLoaders().size).toBe(0);
        expect(forgetPathSpy).not.toHaveBeenCalledWith('/node');
      }
    );
  });

  describe('updateView — queued calls resolve on the winning pass (real pacing gate)', () => {
    beforeEach(async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });

    /** Register a points loader whose FIRST call parks on a gate; later calls resolve null. */
    function installGatedLoader(): {
      releaseFirst: () => void;
      updateView: ReturnType<typeof vi.fn>;
    } {
      let releaseFirst!: () => void;
      const firstGate = new Promise<void>((resolve) => {
        releaseFirst = resolve;
      });
      let calls = 0;
      const updateView = vi.fn(async (_vs: unknown, _s: unknown, signal?: AbortSignal) => {
        calls += 1;
        if (calls === 1) {
          await firstGate;
          signal?.throwIfAborted();
        }
        return null;
      });
      (sceneLoader as unknown as Record<string, Map<string, unknown>>)['loaders'].set('/node', {
        loadPoints: vi.fn(),
        updateView,
        dispose: vi.fn(),
      });
      return { releaseFirst, updateView };
    }

    const vs = (t: number) => ({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, t],
      tolerance: [0, 0, 0, 0],
    });

    it('queued promise stays PENDING until the winning pass completes, then resolves', async () => {
      const { releaseFirst } = installGatedLoader();

      const p1 = sceneLoader.updateView(vs(1));
      await Promise.resolve(); // let the first pass park on the gate

      let queuedResolved = false;
      const p2 = sceneLoader.updateView(vs(2)).then(() => {
        queuedResolved = true;
      });

      // Flush microtasks + a macrotask: pre-fix the queued branch resolved
      // immediately, so this assertion is the regression pin.
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(queuedResolved).toBe(false);

      releaseFirst();
      await p1;
      // queueNext re-enters with the winning state (rAF or its timeout
      // backstop), which completes and settles the waiter — awaiting the
      // queued promise itself is the deterministic wait.
      await p2;
      expect(queuedResolved).toBe(true);
    });

    it('multiple rapid queued calls all resolve when the latest-wins pass completes', async () => {
      const { releaseFirst, updateView } = installGatedLoader();

      const p1 = sceneLoader.updateView(vs(1));
      await Promise.resolve();
      const resolved = [false, false, false];
      const queued = [2, 3, 4].map((t, i) =>
        sceneLoader.updateView(vs(t)).then(() => {
          resolved[i] = true;
        })
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(resolved).toEqual([false, false, false]);

      releaseFirst();
      await p1;
      await Promise.all(queued);
      expect(resolved).toEqual([true, true, true]);

      // Latest-wins: only ONE winning pass ran for the three queued states
      // (first gated call + one re-entry), never one pass per queued call.
      expect(updateView.mock.calls.length).toBe(2);
    });

    it('a view-state queued during the FINAL refinement pass is drained and its waiter resolves', async () => {
      // Regression (deep-check round 3, HIGH — found by 8 independent
      // angles): finalReleaseLock was a bare `_updateInProgress = false`, so
      // a state queued DURING the last refinement pass (after the loop's
      // final loop-top pending check) was stranded, its parked pacing-gate
      // waiter never resolved, and playback froze permanently. The fix
      // makes finalReleaseLock mirror queueNext's contract (drain pending
      // into a fresh pass, else settle waiters).
      let queuedResolved = false;
      let raceFired = false;
      const linesLoader = {
        hasMoreLODs: true,
        loadLines: vi.fn(),
        updateView: vi.fn(async () => {
          // Simulate the race deterministically (ONCE): a tick arrives
          // DURING the final pass — the queued branch parks a waiter + sets
          // pending — and this pass completes the ladder.
          linesLoader.hasMoreLODs = false;
          if (!raceFired) {
            raceFired = true;
            void sceneLoader
              .updateView({
                displayDims: [0, 1, 2],
                slicePosition: [0, 0, 0, 9],
                tolerance: [0, 0, 0, 0],
              })
              .then(() => {
                queuedResolved = true;
              });
          }
          return null; // no data → no process/commit
        }),
        dispose: vi.fn(),
      };
      (sceneLoader as unknown as Record<string, Map<string, unknown>>)['linesLoaders'].set(
        '/lines',
        linesLoader
      );

      // Hold the lock exactly as queueNext's refinement branch does, then
      // run the real refinement orchestrator to completion.
      (sceneLoader as unknown as { _updateInProgress: boolean })._updateInProgress = true;
      await (
        sceneLoader as unknown as { scheduleGSplatsRefinement(): Promise<void> }
      ).scheduleGSplatsRefinement();

      // Post-fix: finalReleaseLock drains the stranded state; the re-entered
      // pass completes and settles the waiter. Pre-fix: this never resolves.
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(queuedResolved).toBe(true);
      expect((sceneLoader as unknown as { _updateInProgress: boolean })._updateInProgress).toBe(
        false
      );
    });

    it('dispose flushes queued-update waiters (no hang across dataset switches)', async () => {
      installGatedLoader(); // never released — pass stays in flight

      void sceneLoader.updateView(vs(1));
      await Promise.resolve();

      let queuedResolved = false;
      const p2 = sceneLoader.updateView(vs(2)).then(() => {
        queuedResolved = true;
      });
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(queuedResolved).toBe(false);

      await sceneLoader.dispose();
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(queuedResolved).toBe(true);
      await p2;
    });
  });

  describe('resource management', () => {
    it('should dispose all resources properly', async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      await sceneLoader.dispose();

      // Verify cleanup
      expect((sceneLoader as any).loaders.size).toBe(0);
      expect((sceneLoader as any)._zarrStore).toBeNull();
      expect((sceneLoader as any).rootGroup).toBeNull();
    });

    it('SceneLoader.dispose returns a Promise that resolves cleanly (async signature)', async () => {
      // Locks in commit 2.1's signature change. loadScene's call site
      // (commit 2.3) now uses `await this.dispose()` — we cannot directly
      // observe the await ordering in this test fixture (loadScene's
      // dispose path is gated on loaders.size > 0 and the jsdom mocks
      // don't populate spatial-index loaders), but a Promise return type
      // is the contract that lets that await work in production.
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      const result = sceneLoader.dispose();
      expect(result).toBeInstanceOf(Promise);
      await result;
      expect((sceneLoader as any)._zarrStore).toBeNull();
    });
  });

  describe('eager working-set admission', () => {
    it('shares one gate across node-build and retry contexts', () => {
      const internals = sceneLoader as unknown as {
        makeNodeBuildCtx(): { lineWorkingSetGate: unknown };
        makeRetryCtx(): { lineWorkingSetGate: unknown };
      };

      const firstBuild = internals.makeNodeBuildCtx();
      const secondBuild = internals.makeNodeBuildCtx();
      const retry = internals.makeRetryCtx();

      expect(firstBuild.lineWorkingSetGate).toBeDefined();
      expect(secondBuild.lineWorkingSetGate).toBe(firstBuild.lineWorkingSetGate);
      expect(retry.lineWorkingSetGate).toBe(firstBuild.lineWorkingSetGate);
    });
  });

  describe('releaseLazyGSplats — depth-sort release on LOD demotion', () => {
    // B4 fix: demoting a lazy gsplats LOD level must ALSO release the node's
    // depth-sort coordinator state (worker-side transferred centers,
    // 12 B/splat) — otherwise the SortWorker pins the demoted level's
    // centers until node disposal / dataset switch. Re-promotion
    // re-registers via the fresh commit's noteDepthSortCommit.
    it('releases the demoted mesh from the depth-sort coordinator', () => {
      const rootGroup = new THREE.Group();
      const mesh = new THREE.Mesh();
      mesh.name = '/lod/child_1';
      rootGroup.add(mesh);
      (sceneLoader as any).rootGroup = rootGroup;

      const ctx = (sceneLoader as any).makeNodeBuildCtx();
      ctx.releaseLazyGSplats('/lod/child_1');

      expect(vi.mocked(releaseDepthSortNode)).toHaveBeenCalledTimes(1);
      expect(vi.mocked(releaseDepthSortNode)).toHaveBeenCalledWith(mesh);
    });

    it('is a no-op (no call, no throw) when no mesh with that path exists', () => {
      (sceneLoader as any).rootGroup = new THREE.Group();

      const ctx = (sceneLoader as any).makeNodeBuildCtx();
      expect(() => ctx.releaseLazyGSplats('/lod/missing_child')).not.toThrow();

      expect(vi.mocked(releaseDepthSortNode)).not.toHaveBeenCalled();
    });
  });

  describe('monitor integration', () => {
    // data.md W4 fix [P2]: the previous three tests only asserted
    // `not.toThrow()` because the default SceneLoader is constructed without
    // a monitorFactory, so showMonitor/hideMonitor/toggleMonitor are no-ops.
    // We now inject a stub factory so the SceneLoader owns a real monitor port
    // and we can verify the delegation contract: each method must call its
    // counterpart on the monitor port exactly once.

    function makeMonitorStubs() {
      const show = vi.fn();
      const hide = vi.fn();
      const toggle = vi.fn();
      const dispose = vi.fn();
      const factory = vi.fn().mockReturnValue({
        show,
        hide,
        toggle,
        dispose,
        // Monitor port surface called from the dispose lifecycle.
        disconnectAllLoaders: vi.fn(),
        connectLoader: vi.fn(),
        notifyLoadStart: vi.fn(),
        notifyLoadEnd: vi.fn(),
        notifyError: vi.fn(),
      });
      return { show, hide, toggle, dispose, factory };
    }

    it('showMonitor delegates to the monitor port', () => {
      const stubs = makeMonitorStubs();
      const loader = new SceneLoader(
        {},
        'test',
        undefined,
        stubs.factory as unknown as ConstructorParameters<typeof SceneLoader>[3]
      );
      try {
        expect(stubs.factory).toHaveBeenCalledTimes(1);
        expect(stubs.factory).toHaveBeenCalledWith('test-monitor');

        loader.showMonitor();
        expect(stubs.show).toHaveBeenCalledTimes(1);
        expect(stubs.hide).not.toHaveBeenCalled();
        expect(stubs.toggle).not.toHaveBeenCalled();
      } finally {
        loader.dispose();
      }
    });

    it('hideMonitor delegates to the monitor port', () => {
      const stubs = makeMonitorStubs();
      const loader = new SceneLoader(
        {},
        'test',
        undefined,
        stubs.factory as unknown as ConstructorParameters<typeof SceneLoader>[3]
      );
      try {
        loader.hideMonitor();
        expect(stubs.hide).toHaveBeenCalledTimes(1);
        expect(stubs.show).not.toHaveBeenCalled();
        expect(stubs.toggle).not.toHaveBeenCalled();
      } finally {
        loader.dispose();
      }
    });

    it('toggleMonitor delegates to the monitor port', () => {
      const stubs = makeMonitorStubs();
      const loader = new SceneLoader(
        {},
        'test',
        undefined,
        stubs.factory as unknown as ConstructorParameters<typeof SceneLoader>[3]
      );
      try {
        loader.toggleMonitor();
        expect(stubs.toggle).toHaveBeenCalledTimes(1);
        expect(stubs.show).not.toHaveBeenCalled();
        expect(stubs.hide).not.toHaveBeenCalled();
      } finally {
        loader.dispose();
      }
    });

    it('show/hide/toggle are no-ops when no monitorFactory is injected', () => {
      // Defensive: the default SceneLoader has no monitor. The methods
      // must remain safe — must not throw and must not allocate a port.
      expect(() => sceneLoader.showMonitor()).not.toThrow();
      expect(() => sceneLoader.hideMonitor()).not.toThrow();
      expect(() => sceneLoader.toggleMonitor()).not.toThrow();
      expect((sceneLoader as unknown as { monitor: unknown }).monitor).toBeFalsy();
    });
  });

  describe('transform handling', () => {
    it('should apply column-major transforms to objects correctly', async () => {
      // Column-major translation matrix (translation at indices [12,13,14])
      mockZarrGroup.attrs = {
        type: 'points',
        transform: [
          1,
          0,
          0,
          0, // Column 0
          0,
          1,
          0,
          0, // Column 1
          0,
          0,
          1,
          0, // Column 2
          10,
          20,
          30,
          1, // Column 3 (translation)
        ],
      };

      // Behavior assertion: a column-major transform must load without
      // throwing. The negative path (row-major rejection) is the more
      // useful contract and is covered by the next test + the
      // transform-validation block further down.
      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).resolves.toBeDefined();
    });

    it('should reject row-major transforms', async () => {
      mockZarrGroup.attrs = {
        type: 'points',
        transform: [
          1,
          0,
          0,
          10, // Row 0 (tx at [3])
          0,
          1,
          0,
          20, // Row 1 (ty at [7])
          0,
          0,
          1,
          30, // Row 2 (tz at [11])
          0,
          0,
          0,
          1,
        ],
      };

      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).rejects.toThrow(
        /row-major/
      );
    });

    it('should reject invalid transform lengths', async () => {
      mockZarrGroup.attrs = {
        type: 'group',
        transform: [1, 2, 3], // Invalid length
      };

      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).rejects.toThrow(
        /Invalid transform length/
      );
    });
  });

  describe('retryAllFailedLoaders — deferred vs failed', () => {
    it('flags a lock-refused batch as deferred:true (nothing was retried)', async () => {
      // Regression: the deferred branch returned {succeeded:[], failed:<all>}
      // — byte-identical to a genuine all-failed batch — so the online
      // auto-retry and the monitor's Retry button reported "N still failing"
      // for retries that never ran.
      const internals = sceneLoader as unknown as {
        _updateInProgress: boolean;
        registry: { recordFailure(path: string, error: Error): void };
      };
      internals.registry.recordFailure('/points/p', new Error('network down'));
      internals._updateInProgress = true; // a main update holds the lock
      try {
        const result = await sceneLoader.retryAllFailedLoaders();
        expect(result).toEqual({ succeeded: [], failed: ['/points/p'], deferred: true });
        // Nothing was retried: the failure record must survive untouched.
        expect(sceneLoader.hasFailures()).toBe(true);
      } finally {
        internals._updateInProgress = false;
      }
    });

    it('a genuine batch result carries no deferred flag', async () => {
      // Empty batch (no failures): resolves immediately without the flag.
      const result = await sceneLoader.retryAllFailedLoaders();
      expect(result.deferred).toBeUndefined();
    });
  });

  describe('fire-and-forget re-entry failures', () => {
    it('logs a rejected requestReprocess update', async () => {
      const errorLog = vi.spyOn(log, 'error').mockImplementation(() => {});
      const failure = new Error('reprocess died');
      const updateView = vi.spyOn(sceneLoader, 'updateView').mockRejectedValue(failure);

      try {
        sceneLoader.requestReprocess();
        await Promise.resolve();

        expect(errorLog).toHaveBeenCalledWith(
          Modules.SCENE_LOADER,
          'View reprocess failed: reprocess died',
          failure
        );
      } finally {
        updateView.mockRestore();
        errorLog.mockRestore();
      }
    });

    it('logs a rejected targeted requestReprocess update', async () => {
      const errorLog = vi.spyOn(log, 'error').mockImplementation(() => {});
      const failure = new Error('targeted reprocess died');
      const updateView = vi.spyOn(sceneLoader, 'updateView').mockRejectedValue(failure);

      try {
        sceneLoader.requestReprocess(['/partition/part_1']);
        await Promise.resolve();

        expect(updateView).toHaveBeenCalledWith(
          {},
          { resyncPaths: new Set(['/partition/part_1']) }
        );
        expect(errorLog).toHaveBeenCalledWith(
          Modules.SCENE_LOADER,
          'View reprocess failed: targeted reprocess died',
          failure
        );
      } finally {
        updateView.mockRestore();
        errorLog.mockRestore();
      }
    });

    it('logs a rejected refinement-cancellation re-entry', async () => {
      vi.useFakeTimers();
      const pendingState = { slicePosition: [2, 1, 0] };
      const errorLog = vi.spyOn(log, 'error').mockImplementation(() => {});
      const failure = new Error('cancel re-entry died');
      const updateView = vi.spyOn(sceneLoader, 'updateView').mockRejectedValue(failure);
      const runRefinement = vi
        .spyOn(gsplatsRefinement, 'runGSplatsRefinement')
        .mockImplementation(async (ctx) => {
          ctx.retriggerUpdate(pendingState);
        });
      const internals = sceneLoader as unknown as {
        _updateInProgress: boolean;
        scheduleGSplatsRefinement(): Promise<void>;
      };
      internals._updateInProgress = true;

      try {
        await internals.scheduleGSplatsRefinement();
        await vi.runOnlyPendingTimersAsync();
        await Promise.resolve();

        expect(errorLog).toHaveBeenCalledWith(
          Modules.SCENE_LOADER,
          'Refinement cancellation re-entry failed: cancel re-entry died',
          failure
        );
      } finally {
        runRefinement.mockRestore();
        updateView.mockRestore();
        errorLog.mockRestore();
        vi.useRealTimers();
        internals._updateInProgress = false;
      }
    });
  });

  describe('getFailedLoadsProvider — reason mapping', () => {
    // #1055: the layers-panel error badge reads its tooltip from the provider's
    // getFailedReason, which folds error.message → classified kind → undefined.
    interface FailInternals {
      registry: { recordFailure(path: string, error: Error, kind?: string): void };
    }

    it('reports error.message, falls back to kind, else undefined for an unknown path', () => {
      const internals = sceneLoader as unknown as FailInternals;
      internals.registry.recordFailure('/points/a', new Error('Vertex index 3 not found'));
      // Empty message → the classified kind is reported instead.
      internals.registry.recordFailure('/points/b', new Error(''), 'Decode');

      const provider = sceneLoader.getFailedLoadsProvider();
      expect(provider.getFailedReason?.('/points/a')).toBe('Vertex index 3 not found');
      expect(provider.getFailedReason?.('/points/b')).toBe('Decode');
      expect(provider.getFailedReason?.('/points/missing')).toBeUndefined();
      // The provider reads the same live failed set.
      expect(provider.getFailedPaths().sort()).toEqual(['/points/a', '/points/b']);
    });

    it('supplies a classified reason when a loader throws a non-Error value', () => {
      interface LoadSceneInternals extends FailInternals {
        makeLoadSceneCtx(): { getFailedLoaderReasons(): string[] };
      }

      const internals = sceneLoader as unknown as LoadSceneInternals;
      internals.registry.recordFailure('/points/a', undefined as unknown as Error);

      expect(internals.makeLoadSceneCtx().getFailedLoaderReasons()).toEqual(['Unexpected']);
    });

    it('surfaces and retries an archive fault with no recorded node failure', async () => {
      const archiveFault = new ArchiveFaultError('archive unavailable', '/scene.zip');
      const current = { displayDims: [0, 1, 2], slicePosition: [3], tolerance: [0] };
      const blocked = { displayDims: [0, 1, 2], slicePosition: [4], tolerance: [0] };
      const internals = sceneLoader as unknown as {
        viewState: { displayDims: number[]; slicePosition: number[]; tolerance: number[] };
        viewStateQueue: { hasPending(): boolean };
        reportArchiveFault(fault: ArchiveFaultError): void;
      };
      await sceneLoader.updateView(current);
      internals.reportArchiveFault(archiveFault);
      await sceneLoader.updateView(blocked);
      expect(internals.viewState).toMatchObject(current);
      expect(internals.viewStateQueue.hasPending()).toBe(false);

      const updateViewSpy = vi.spyOn(sceneLoader, 'updateView');

      const provider = sceneLoader.getFailedLoadsProvider();
      expect(provider.getFailedPaths()).toEqual(['/scene.zip']);
      expect(provider.getFailedReason?.('/scene.zip')).toBe('archive unavailable');
      expect(sceneLoader.hasAutoRetryableFailures()).toBe(true);

      await expect(provider.retryAll()).resolves.toEqual({
        succeeded: ['/scene.zip'],
        failed: [],
      });
      expect(sceneLoader.archiveFault).toBeNull();
      expect(notifierMocks.clearError).toHaveBeenCalledOnce();
      expect(internals.viewStateQueue.hasPending()).toBe(false);
      expect(updateViewSpy).toHaveBeenCalledWith(internals.viewState);
    });

    it('auto-retries an archive fault with no recorded node failure', async () => {
      const internals = sceneLoader as unknown as {
        _archiveFault: ArchiveFaultError | null;
      };
      internals._archiveFault = new ArchiveFaultError('archive unavailable', '/scene.zip');

      await expect(sceneLoader.retryAllFailedLoaders({ onlyAutoRetryable: true })).resolves.toEqual(
        {
          succeeded: ['/scene.zip'],
          failed: [],
        }
      );
      expect(sceneLoader.archiveFault).toBeNull();
      expect(notifierMocks.clearError).toHaveBeenCalledOnce();
    });

    it('retries the surfaced archive fault path and reloads the current view', async () => {
      const current = { displayDims: [0, 1, 2], slicePosition: [3], tolerance: [0] };
      const internals = sceneLoader as unknown as {
        viewState: { displayDims: number[]; slicePosition: number[]; tolerance: number[] };
        reportArchiveFault(fault: ArchiveFaultError): void;
      };
      await sceneLoader.updateView(current);
      internals.reportArchiveFault(new ArchiveFaultError('archive unavailable', '/scene.zip'));
      const updateViewSpy = vi.spyOn(sceneLoader, 'updateView');

      await expect(sceneLoader.retryFailedLoader('/scene.zip')).resolves.toBe(true);
      expect(sceneLoader.archiveFault).toBeNull();
      expect(notifierMocks.clearError).toHaveBeenCalledOnce();
      expect(updateViewSpy).toHaveBeenCalledWith(internals.viewState);
    });

    it('surfaces and retries a latched anonymous deferred LOD branch', async () => {
      const camera = new THREE.Camera();
      const registry = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => ({ width: 800, height: 600 }),
        getDisplayDims: () => [0, 1, 2],
        hasArchiveFault: () => sceneLoader.archiveFault !== null,
      });
      const ensureLoaded = vi.fn();
      const eager: LODGroupChild = {
        object: new THREE.Group(),
        coverageFraction: 0,
        positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
      };
      const deferred: LODGroupChild = {
        object: new THREE.Group(),
        nodePath: '/lod/nested',
        coverageFraction: 0.5,
        positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
        ready: false,
        failed: true,
        permanentlyFailed: true,
        failureReason: 'archive expired',
        ensureLoaded,
      };
      const ensureOtherLoaded = vi.fn();
      const otherDeferred: LODGroupChild = {
        object: new THREE.Group(),
        nodePath: '/lod/other',
        coverageFraction: 0.75,
        positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
        ready: false,
        failed: true,
        permanentlyFailed: true,
        failureReason: 'archive expired',
        ensureLoaded: ensureOtherLoaded,
      };
      const entry: LODGroupEntry = {
        path: '/lod',
        groupObject: new THREE.Group(),
        children: [eager, deferred, otherDeferred],
        selectorMode: 'auto',
        defaultLevel: 0,
        activeChildIndex: 0,
      };
      registry.register(entry);
      (sceneLoader as any).lodGroupRegistry = registry;
      const archiveFault = new ArchiveFaultError('archive expired', '/scene.zip');
      (sceneLoader as any)._archiveFault = archiveFault;

      const provider = sceneLoader.getFailedLoadsProvider();
      expect(provider.getFailedPaths()).toEqual(['/lod/nested', '/lod/other']);
      expect(provider.getFailedReason?.('/lod/nested')).toBe('archive expired');

      await expect(sceneLoader.retryFailedLoader('/lod/nested')).resolves.toBe(true);
      expect(sceneLoader.archiveFault).toBeNull();
      expect(notifierMocks.clearError).toHaveBeenCalledOnce();
      expect(ensureLoaded).toHaveBeenCalledOnce();
      expect(ensureOtherLoaded).not.toHaveBeenCalled();
      expect(provider.getFailedPaths()).toEqual(['/lod/other']);
      expect(provider.getFailedReason?.('/lod/other')).toBe('archive expired');

      await expect(provider.retryAll()).resolves.toEqual({
        succeeded: ['/lod/other'],
        failed: [],
      });
      expect(ensureOtherLoaded).toHaveBeenCalledOnce();
      expect(otherDeferred.loading).toBe(true);
      expect(provider.getFailedPaths()).toEqual([]);
    });

    it('auto-retries an overlapping lazy loader once and clears the archive latch', async () => {
      const camera = new THREE.Camera();
      const registry = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => ({ width: 800, height: 600 }),
        getDisplayDims: () => [0, 1, 2],
        hasArchiveFault: () => sceneLoader.archiveFault !== null,
      });
      const ensureLoaded = vi.fn();
      const deferred: LODGroupChild = {
        object: new THREE.Group(),
        nodePath: '/lod/leaf_1',
        coverageFraction: 0.5,
        positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
        ready: false,
        failed: true,
        permanentlyFailed: true,
        failureReason: 'archive expired',
        ensureLoaded,
      };
      registry.register({
        path: '/lod',
        groupObject: new THREE.Group(),
        children: [
          {
            object: new THREE.Group(),
            coverageFraction: 0,
            positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
          },
          deferred,
        ],
        selectorMode: 'auto',
        defaultLevel: 0,
        activeChildIndex: 0,
      });
      const internals = sceneLoader as unknown as {
        lodGroupRegistry: LODGroupRegistry;
        registry: { recordFailure(path: string, error: Error, kind?: string): void };
        _archiveFault: ArchiveFaultError | null;
      };
      internals.lodGroupRegistry = registry;
      internals.registry.recordFailure('/lod/leaf_1', new Error('network down'), 'Network');
      internals._archiveFault = new ArchiveFaultError('archive expired', '/scene.zip');

      await expect(sceneLoader.retryAllFailedLoaders({ onlyAutoRetryable: true })).resolves.toEqual(
        { succeeded: ['/lod/leaf_1'], failed: [] }
      );
      expect(ensureLoaded).toHaveBeenCalledOnce();
      expect(sceneLoader.archiveFault).toBeNull();
      expect(deferred.permanentlyFailed).toBe(false);
    });

    it('auto-retries a lazy-only archive failure after connectivity returns', async () => {
      const camera = new THREE.Camera();
      const registry = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => ({ width: 800, height: 600 }),
        getDisplayDims: () => [0, 1, 2],
        hasArchiveFault: () => sceneLoader.archiveFault !== null,
      });
      const ensureLoaded = vi.fn();
      const deferred: LODGroupChild = {
        object: new THREE.Group(),
        nodePath: '/lod/nested',
        coverageFraction: 0.5,
        positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
        ready: false,
        failed: true,
        permanentlyFailed: true,
        failureReason: 'archive unavailable',
        ensureLoaded,
      };
      registry.register({
        path: '/lod',
        groupObject: new THREE.Group(),
        children: [
          {
            object: new THREE.Group(),
            coverageFraction: 0,
            positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
          },
          deferred,
        ],
        selectorMode: 'auto',
        defaultLevel: 0,
        activeChildIndex: 0,
      });
      const internals = sceneLoader as unknown as {
        lodGroupRegistry: LODGroupRegistry;
        _archiveFault: ArchiveFaultError | null;
      };
      internals.lodGroupRegistry = registry;
      internals._archiveFault = new ArchiveFaultError('archive unavailable', '/scene.zip');

      expect(sceneLoader.hasAutoRetryableFailures()).toBe(true);
      await expect(sceneLoader.retryAllFailedLoaders({ onlyAutoRetryable: true })).resolves.toEqual(
        { succeeded: ['/lod/nested'], failed: [] }
      );
      expect(sceneLoader.archiveFault).toBeNull();
      expect(ensureLoaded).toHaveBeenCalledOnce();
      expect(deferred.permanentlyFailed).toBe(false);
    });

    it('deduplicates a path present in both loader and lazy failure sets', async () => {
      const camera = new THREE.Camera();
      const registry = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => ({ width: 800, height: 600 }),
        getDisplayDims: () => [0, 1, 2],
        hasArchiveFault: () => sceneLoader.archiveFault !== null,
      });
      const ensureLoaded = vi.fn();
      registry.register({
        path: '/lod',
        groupObject: new THREE.Group(),
        children: [
          {
            object: new THREE.Group(),
            coverageFraction: 0,
            positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
          },
          {
            object: new THREE.Group(),
            nodePath: '/lod/leaf_1',
            coverageFraction: 0.5,
            positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
            ready: false,
            failed: true,
            permanentlyFailed: true,
            ensureLoaded,
          },
        ],
        selectorMode: 'auto',
        defaultLevel: 0,
        activeChildIndex: 0,
      });
      const internals = sceneLoader as unknown as {
        lodGroupRegistry: LODGroupRegistry;
        registry: { recordFailure(path: string, error: Error, kind?: string): void };
        _archiveFault: ArchiveFaultError | null;
      };
      internals.lodGroupRegistry = registry;
      internals.registry.recordFailure('/lod/leaf_1', new Error('network down'), 'Network');
      internals._archiveFault = new ArchiveFaultError('archive expired', '/scene.zip');

      await expect(sceneLoader.retryAllFailedLoaders()).resolves.toEqual({
        succeeded: ['/lod/leaf_1'],
        failed: [],
      });
      expect(ensureLoaded).toHaveBeenCalledOnce();
    });
  });

  describe('kickRefinementIfIdle — refinement after deferred-group activation', () => {
    interface KickInternals {
      _updateInProgress: boolean;
      _refining: boolean;
      _disposed: boolean;
      _archiveFault: ArchiveFaultError | null;
      _refinementKickPending: boolean;
      gsplatLoaders: Map<string, unknown>;
      loaders: Map<string, unknown>; // points
      linesLoaders: Map<string, unknown>;
      viewStateQueue: { setPending(s: unknown): void; hasPending(): boolean };
      scheduleGSplatsRefinement: () => Promise<void>;
    }

    /** Stub the orchestrator (instance property shadows the prototype method). */
    function stubOrchestrator(releaseLock = true) {
      const internals = sceneLoader as unknown as KickInternals;
      const spy = vi.fn(async () => {
        // The real orchestrator's final phase releases the lock on completion.
        if (releaseLock) internals._updateInProgress = false;
      });
      internals.scheduleGSplatsRefinement = spy;
      return { internals, spy };
    }

    it('takes the lock and schedules refinement once when idle and a loader has more LODs', () => {
      const { internals, spy } = stubOrchestrator(false);
      internals.gsplatLoaders.set('/g/part_0', { hasMoreLODs: true });
      try {
        sceneLoader.kickRefinementIfIdle();
        expect(spy).toHaveBeenCalledTimes(1);
        expect(internals._updateInProgress).toBe(true); // lock taken for the run
        // Re-entrant call while the run holds the lock must not double-fire
        // (it schedules a timer re-check instead).
        sceneLoader.kickRefinementIfIdle();
        expect(spy).toHaveBeenCalledTimes(1);
      } finally {
        internals._updateInProgress = false;
        internals.gsplatLoaders.clear();
      }
    });

    it('no-ops when no registered loader has more LODs', () => {
      const { internals, spy } = stubOrchestrator();
      internals.gsplatLoaders.set('/g/part_0', { hasMoreLODs: false });
      sceneLoader.kickRefinementIfIdle();
      expect(spy).not.toHaveBeenCalled();
      expect(internals._updateInProgress).toBe(false);
      internals.gsplatLoaders.clear();
    });

    it('refinementHoldReason: density gate first, then the last run’s residency budget, else null', () => {
      const caps = { blendable: 4, nonBlendable: 1 };
      const sample = { areaPx: 100, elements: 1_000_000, onScreen: true, blendable: true };
      type Internals = {
        refinementDensityGate: { deferred: Map<string, number> } | null;
        lastResidencyBudget: { isDeclined(path: string): boolean } | null;
      };
      const internals = sceneLoader as unknown as Internals;
      try {
        expect(sceneLoader.refinementHoldReason('/g/part_0')).toBeNull();
        internals.lastResidencyBudget = { isDeclined: (p) => p === '/g/part_0' };
        expect(sceneLoader.refinementHoldReason('/g/part_0')).toBe('budget');
        expect(sceneLoader.refinementHoldReason('/g/part_1')).toBeNull();
        sceneLoader.setRefinementDensityProvider(() => sample, caps);
        internals.refinementDensityGate!.deferred.set('/g/part_0', 1e9);
        // Both hold it: the camera-dependent reason wins.
        expect(sceneLoader.refinementHoldReason('/g/part_0')).toBe('density');
      } finally {
        sceneLoader.setRefinementDensityProvider(null, caps);
        internals.lastResidencyBudget = null;
      }
    });

    // Turning the density guard off at runtime clears the rung gate. The rungs
    // it was holding back have no other way to load: `resumeDensityDeferred
    // Refinement` has no gate left to consult, so the clear itself must kick.
    it('setRefinementDensityProvider(null) kicks refinement iff the old gate held rungs back', () => {
      const { internals, spy } = stubOrchestrator(false);
      internals.gsplatLoaders.set('/g/part_0', { hasMoreLODs: true });
      const caps = { blendable: 4, nonBlendable: 1 };
      const sample = { areaPx: 100, elements: 1_000_000, onScreen: true, blendable: true };
      type GateInternals = { refinementDensityGate: { deferred: Map<string, number> } | null };
      try {
        // Control arm: a gate with nothing deferred → clearing is silent.
        sceneLoader.setRefinementDensityProvider(() => sample, caps);
        sceneLoader.setRefinementDensityProvider(null, caps);
        expect(spy).not.toHaveBeenCalled();

        sceneLoader.setRefinementDensityProvider(() => sample, caps);
        (sceneLoader as unknown as GateInternals).refinementDensityGate!.deferred.set(
          '/g/part_0',
          1e9
        );
        sceneLoader.setRefinementDensityProvider(null, caps);
        expect(spy).toHaveBeenCalledTimes(1);
        expect((sceneLoader as unknown as GateInternals).refinementDensityGate).toBeNull();
      } finally {
        internals._updateInProgress = false;
        internals.gsplatLoaders.clear();
      }
    });

    // anyLoaderHasMoreLODs consults all THREE loader maps (gsplats/points/lines),
    // not just gsplats — a points- or lines-substitutive ladder must kick too.
    it.each([
      ['points', 'loaders' as const],
      ['lines', 'linesLoaders' as const],
    ])('kicks when only the %s loader map has more LODs', (_label, mapKey) => {
      const { internals, spy } = stubOrchestrator(false);
      const map = internals[mapKey];
      map.set('/g/part_0', { hasMoreLODs: true });
      try {
        sceneLoader.kickRefinementIfIdle();
        expect(spy).toHaveBeenCalledTimes(1);
      } finally {
        internals._updateInProgress = false;
        map.clear();
      }
    });

    it('schedules at most ONE re-check timer for repeated locked kicks (single-pending guard)', async () => {
      vi.useFakeTimers();
      const setTimeoutSpy = vi.spyOn(globalThis, 'setTimeout');
      const { internals, spy } = stubOrchestrator();
      internals.gsplatLoaders.set('/g/part_0', { hasMoreLODs: true });
      internals._updateInProgress = true; // an update holds the lock
      try {
        sceneLoader.kickRefinementIfIdle();
        sceneLoader.kickRefinementIfIdle(); // second locked kick — must be swallowed
        sceneLoader.kickRefinementIfIdle(); // third too
        expect(internals._refinementKickPending).toBe(true);
        expect(setTimeoutSpy).toHaveBeenCalledTimes(1); // ONE timer, not three
        // The lock frees; the single re-check fires and kicks exactly once.
        internals._updateInProgress = false;
        await vi.runOnlyPendingTimersAsync();
        expect(spy).toHaveBeenCalledTimes(1);
      } finally {
        setTimeoutSpy.mockRestore();
        vi.useRealTimers();
        internals._updateInProgress = false;
        internals._refinementKickPending = false;
        internals.gsplatLoaders.clear();
      }
    });

    it('re-checks on a timer while the lock is held, then kicks once it frees', async () => {
      vi.useFakeTimers();
      const { internals, spy } = stubOrchestrator();
      internals.gsplatLoaders.set('/g/part_0', { hasMoreLODs: true });
      internals._updateInProgress = true; // an update is mid-flight
      try {
        sceneLoader.kickRefinementIfIdle();
        expect(spy).not.toHaveBeenCalled(); // no double-acquire
        // Holder finishes; the pending re-check fires and kicks.
        internals._updateInProgress = false;
        await vi.runOnlyPendingTimersAsync();
        expect(spy).toHaveBeenCalledTimes(1);
      } finally {
        vi.useRealTimers();
        internals._updateInProgress = false;
        internals.gsplatLoaders.clear();
      }
    });

    it('treats a live refinement as busy even after the lock has opened', async () => {
      // `finalReleaseLock` clears `_updateInProgress` while `_refining` is still
      // set (the orchestrator's `finally` clears that one level up, when the
      // phase's await unwinds). A microtask queued at exactly that instant — a
      // deferred lod_group `ensureLoaded` continuation is one — must not read
      // the open lock as idle and start a SECOND refinement run on top of the
      // first; it re-checks on the timer instead.
      vi.useFakeTimers();
      const { internals, spy } = stubOrchestrator();
      internals.gsplatLoaders.set('/g/part_0', { hasMoreLODs: true });
      internals._updateInProgress = false; // lock already released…
      internals._refining = true; // …but the run is still draining
      try {
        sceneLoader.kickRefinementIfIdle();
        expect(spy).not.toHaveBeenCalled();
        expect(internals._refinementKickPending).toBe(true); // re-check armed
        // The first run finishes; the pending re-check kicks exactly once.
        internals._refining = false;
        await vi.runOnlyPendingTimersAsync();
        expect(spy).toHaveBeenCalledTimes(1);
      } finally {
        vi.useRealTimers();
        internals._refining = false;
        internals._refinementKickPending = false;
        internals._updateInProgress = false;
        internals.gsplatLoaders.clear();
      }
    });

    it('a pending re-check no-ops after dispose (no kick against a dead loader)', async () => {
      vi.useFakeTimers();
      const { internals, spy } = stubOrchestrator();
      internals.gsplatLoaders.set('/g/part_0', { hasMoreLODs: true });
      internals._updateInProgress = true;
      try {
        sceneLoader.kickRefinementIfIdle(); // schedules the re-check
        internals._updateInProgress = false;
        internals._disposed = true; // dataset switch tore the loader down
        await vi.runOnlyPendingTimersAsync();
        expect(spy).not.toHaveBeenCalled();
      } finally {
        vi.useRealTimers();
        internals._disposed = false;
        internals._updateInProgress = false;
        internals.gsplatLoaders.clear();
      }
    });

    it('a pending re-check no-ops after an archive fault', async () => {
      vi.useFakeTimers();
      const { internals, spy } = stubOrchestrator();
      internals.gsplatLoaders.set('/g/part_0', { hasMoreLODs: true });
      internals._updateInProgress = true;
      try {
        sceneLoader.kickRefinementIfIdle();
        internals._updateInProgress = false;
        internals._archiveFault = new ArchiveFaultError('archive unavailable', 'scene.zip');
        await vi.runOnlyPendingTimersAsync();
        expect(spy).not.toHaveBeenCalled();
      } finally {
        vi.useRealTimers();
        internals._archiveFault = null;
        internals._updateInProgress = false;
        internals.gsplatLoaders.clear();
      }
    });

    it('an already-scheduled refinement no-ops after an archive fault', async () => {
      const internals = sceneLoader as unknown as KickInternals;
      internals._updateInProgress = true;
      internals._archiveFault = new ArchiveFaultError('archive unavailable', 'scene.zip');
      try {
        await internals.scheduleGSplatsRefinement();
        expect(internals._updateInProgress).toBe(true);
        expect(internals._refining).toBe(false);
      } finally {
        internals._archiveFault = null;
        internals._updateInProgress = false;
      }
    });

    it('drains a view-state queued during the kick when the orchestrator rejects', async () => {
      // A concurrent updateView() parks its state via setPending while the kick
      // holds the lock. If the orchestrator glue rejects OUTSIDE the loops'
      // finally, the catch must release the lock AND drain — otherwise the
      // user's latest slice is stranded (queue-next.ts drains; this must too).
      const internals = sceneLoader as unknown as KickInternals;
      internals.gsplatLoaders.set('/g/part_0', { hasMoreLODs: true });
      const rejecting = vi.fn(async () => {
        throw { code: 'EORCHESTRATOR', retryable: false };
      });
      internals.scheduleGSplatsRefinement = rejecting;
      const errorLog = vi.spyOn(log, 'error').mockImplementation(() => {});
      const updateViewSpy = vi
        .spyOn(sceneLoader, 'updateView')
        .mockResolvedValue(undefined as never);
      try {
        const pending = { displayDims: [0, 1, 2], slicePosition: [3], tolerance: [0] };
        internals.viewStateQueue.setPending(pending);
        sceneLoader.kickRefinementIfIdle();
        expect(rejecting).toHaveBeenCalledTimes(1);
        // Let the rejection catch + drain's microtask settle.
        await new Promise((r) => setTimeout(r, 0));
        expect(internals._updateInProgress).toBe(false); // lock released
        expect(internals.viewStateQueue.hasPending()).toBe(false); // drained
        expect(updateViewSpy).toHaveBeenCalledWith(pending); // re-entered
        expect(errorLog).toHaveBeenCalledWith(
          Modules.SCENE_LOADER,
          'Deferred-activation refinement failed: {"code":"EORCHESTRATOR","retryable":false}'
        );
      } finally {
        errorLog.mockRestore();
        updateViewSpy.mockRestore();
        internals._updateInProgress = false;
        internals.gsplatLoaders.clear();
      }
    });

    it('preserves a plain-object error when reloading after an archive retry', async () => {
      const warningLog = vi.spyOn(log, 'warning').mockImplementation(() => {});
      const failure = { code: 'ERELOAD', retryable: true };
      const updateView = vi.spyOn(sceneLoader, 'updateView').mockRejectedValue(failure);
      const internals = sceneLoader as unknown as {
        resumeViewAfterRetry(hadArchiveFault: boolean): void;
      };

      try {
        internals.resumeViewAfterRetry(true);
        await Promise.resolve();

        expect(warningLog).toHaveBeenCalledWith(
          Modules.SCENE_LOADER,
          'Current view reload after archive retry failed: {"code":"ERELOAD","retryable":true}',
          failure
        );
      } finally {
        updateView.mockRestore();
        warningLog.mockRestore();
      }
    });
  });

  describe('error handling', () => {
    it('should handle store opening failures', async () => {
      resetLoadTimeline();
      (zarr as any).withMaybeConsolidatedMetadata.mockRejectedValue(
        new Error('Failed to open store')
      );

      await expect(sceneLoader.loadScene('http://invalid.url')).rejects.toThrow(
        'Failed to open store'
      );
      expect(getLoadTimeline().refinement.complete).toBe(true);
      expect(getLoadTimeline().milestones.refinementComplete).toBeUndefined();
    }, 15000);

    it('should handle enumeration failures gracefully', async () => {
      // [data.md/W6][P2] Strengthen: pin the fallback contract explicitly.
      // With no `contents()` method on the store, the loader cannot
      // enumerate children — production behavior is to fall back to the
      // root group only, NOT crash. Verify (a) we get a real LuxarScene
      // group, (b) no spatial-index loaders were registered (no children
      // to enumerate means no points/lines/gsplats loaders), (c) sceneDimensions
      // still comes through from the root attrs.
      mockStore.contents = undefined; // No contents method

      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(scene).toBeDefined();
      expect(scene.name).toBe('LuxarScene');
      expect(scene.userData.sceneDimensions).toEqual(mockZarrGroup.attrs.scene_dimensions);
      const loaders = (sceneLoader as any).loaders as Map<string, unknown>;
      expect(loaders.size).toBe(0);
    });
  });

  describe('config handling', () => {
    it('should accept and use loader configuration', () => {
      const config: LoaderConfig = {};

      const loader = new SceneLoader(config);
      expect((loader as any).config).toEqual(config);
      loader.dispose();
    });

    /**
     * A lines node whose eager admission charge is ~92 MiB: 300k vertices at
     * `10 × ndim + 70` bytes each, plus the 1:1 segment fallback at 220 B each
     * (see `nodes/load-children-concurrently.ts`).
     */
    function makeLineAdmissionNode(index: number) {
      return {
        path: `/lines/node_${index}`,
        type: 'lines' as const,
        attrs: { n_vertices: 300_000, ndim: 3 },
      };
    }

    /**
     * How many of `count` ~92 MiB line nodes the loader's session gate admits
     * at once. Read through the production `makeNodeBuildCtx()` so this also
     * pins that the ctx still carries the session gate. Charges are never
     * released — the loader is disposed right after.
     */
    async function concurrentLineAdmissions(loader: SceneLoader, count: number): Promise<number> {
      const internals = loader as unknown as { makeNodeBuildCtx(): NodeBuildCtx };
      const gate = internals.makeNodeBuildCtx().lineWorkingSetGate;
      let admitted = 0;
      for (let index = 0; index < count; index++) {
        void gate.acquire(makeLineAdmissionNode(index)).then(() => {
          admitted++;
        });
      }
      // Admission is decided synchronously inside `acquire`, so one microtask
      // turn is enough for every granted continuation to run.
      await Promise.resolve();
      await Promise.resolve();
      return admitted;
    }

    it('sizes line working-set admission from an explicit cache-pool override', async () => {
      // A 384 MiB pool resolves to a 128 MiB eager line budget, which seats one
      // ~92 MiB node. Ignoring the override (#2307) leaves a >=256 MiB budget,
      // which would admit both.
      const loader = new SceneLoader({ cacheBudgetMB: 384 });

      expect(await concurrentLineAdmissions(loader, 2)).toBe(1);
      loader.dispose();
    });

    it('sizes line admission from the device-class pool without an override', async () => {
      // No override and no `performance.memory` under jsdom: the gate still
      // takes the inferred device-class pool (>=1 GiB => >=341 MiB budget),
      // which seats three ~92 MiB nodes (~275 MiB); the 256 MiB fixed fallback
      // would seat only two.
      const loader = new SceneLoader({});

      expect(await concurrentLineAdmissions(loader, 3)).toBe(3);
      loader.dispose();
    });
  });

  describe('transform validation', () => {
    it('should accept column-major matrices (correct for THREE.js)', () => {
      // Column-major: translation at indices [12, 13, 14]
      const columnMajorTransform = [
        1,
        0,
        0,
        0, // Column 0: right vector
        0,
        1,
        0,
        0, // Column 1: up vector
        0,
        0,
        1,
        0, // Column 2: forward vector
        10,
        20,
        30,
        1, // Column 3: translation + w
      ];

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(columnMajorTransform)
      ).not.toThrow();
    });

    it('should throw for row-major matrices', () => {
      // Row-major: translation at indices [3, 7, 11] (WRONG for THREE.js)
      const rowMajorTransform = [
        1,
        0,
        0,
        10, // Row 0: right + tx
        0,
        1,
        0,
        20, // Row 1: up + ty
        0,
        0,
        1,
        30, // Row 2: forward + tz
        0,
        0,
        0,
        1, // Row 3: homogeneous
      ];

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(rowMajorTransform)
      ).toThrow(/row-major/);
    });

    it('should throw when translation is at wrong indices', () => {
      // Suspicious transform: non-zero at row-major positions [3, 7, 11], zero at column-major [12, 13, 14]
      const suspiciousTransform = [
        1,
        0,
        0,
        5, // Translation at [3] (row-major)
        0,
        1,
        0,
        10, // Translation at [7] (row-major)
        0,
        0,
        1,
        15, // Translation at [11] (row-major)
        0,
        0,
        0,
        1, // [12, 13, 14] are zero (column-major)
      ];

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(suspiciousTransform)
      ).toThrow(/matrix\.T\.ravel/);
    });

    it('should accept identity matrix', () => {
      const identityTransform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(identityTransform)
      ).not.toThrow();
    });

    it('should handle transforms with only rotation/scale (no translation)', () => {
      // Scale matrix: no translation, should pass validation
      const scaleTransform = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1];

      expect(() =>
        (sceneLoader as any).nodeFactory.validateTransformFormat(scaleTransform)
      ).not.toThrow();
    });
  });

  describe('material creation', () => {
    // data.md W3 fix [P2]: previously all six tests only asserted
    // `expect(material).toBeDefined()` + `expect(material.updateCameraParams).toBeDefined()`.
    // Both fields are pre-populated by the materialManager mock at the top
    // of this file, so the assertions held trivially for any input — a
    // mutation that swapped radiusScale, dropped blending
    // mode, or stopped honoring opacity/gamma defaults would still pass.
    //
    // The materialManager mock is the trust boundary (P3): we spy on
    // `getPointMaterial` to pin the EXACT argument bag the factory sends.
    // That ARGUMENT contract is what `createPointsMaterial` controls —
    // the returned material object's internals belong to materialManager
    // and are out of scope here.
    let materialManagerMock: { getPointMaterial: ReturnType<typeof vi.fn> };

    beforeEach(async () => {
      vi.clearAllMocks();
      // Re-resolve the mocked materialManager so we can inspect calls.
      const mod = await import('../../../rendering/material-manager');
      materialManagerMock = mod.materialManager as unknown as {
        getPointMaterial: ReturnType<typeof vi.fn>;
      };
    });

    it('forwards explicit radiusScale to the material manager', () => {
      (sceneLoader as any).nodeFactory.createPointsMaterial(
        { opacity: 1.0, gamma: 1.0, blending_mode: 'normal' as const },
        2.5
      );

      expect(materialManagerMock.getPointMaterial).toHaveBeenCalledTimes(1);
      expect(materialManagerMock.getPointMaterial).toHaveBeenCalledWith(
        expect.objectContaining({
          radiusScale: 2.5,
          opacity: 1.0,
          gamma: 1.0,
          blendingMode: 'normal',
        })
      );
    });

    it('propagates blendingMode distinctly for normal vs additive', () => {
      (sceneLoader as any).nodeFactory.createPointsMaterial(
        { blending_mode: 'normal' as const },
        1.0
      );
      (sceneLoader as any).nodeFactory.createPointsMaterial(
        { blending_mode: 'additive' as const },
        1.0
      );

      expect(materialManagerMock.getPointMaterial).toHaveBeenCalledTimes(2);
      const calls = materialManagerMock.getPointMaterial.mock.calls;
      expect(calls[0][0].blendingMode).toBe('normal');
      expect(calls[1][0].blendingMode).toBe('additive');
    });

    it('uses opacity=1.0 and gamma=1.0 as defaults when attrs omit them', () => {
      (sceneLoader as any).nodeFactory.createPointsMaterial({}, 1.0);

      // Note: the source defaults blending_mode to 'additive' when absent
      // (see create-points-node.ts line ~179).
      expect(materialManagerMock.getPointMaterial).toHaveBeenCalledWith(
        expect.objectContaining({
          opacity: 1.0,
          gamma: 1.0,
          blendingMode: 'additive',
        })
      );
    });

    it('passes through custom opacity and gamma values from attrs', () => {
      (sceneLoader as any).nodeFactory.createPointsMaterial({ opacity: 0.5, gamma: 2.2 }, 1.0);

      expect(materialManagerMock.getPointMaterial).toHaveBeenCalledWith(
        expect.objectContaining({
          opacity: 0.5,
          gamma: 2.2,
        })
      );
    });

    it('defaults radiusScale to 1.0 when callers omit it', () => {
      (sceneLoader as any).nodeFactory.createPointsMaterial({}); // No scale provided

      expect(materialManagerMock.getPointMaterial).toHaveBeenCalledWith(
        expect.objectContaining({
          radiusScale: 1.0,
        })
      );
    });
  });

  describe('error recovery', () => {
    beforeEach(async () => {
      vi.clearAllMocks();
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });

    it('should track failed loaders in failedLoaders map', async () => {
      const mockLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Network timeout')),
        dispose: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/failing_node', mockLoader);

      await sceneLoader.updateView({ displayDims: [0, 1, 2] });

      expect(sceneLoader.hasFailures()).toBe(true);
      const failures = sceneLoader.getFailedLoaders();
      expect(failures.size).toBe(1);
      expect(failures.has('/failing_node')).toBe(true);

      const failureInfo = failures.get('/failing_node');
      expect(failureInfo).toBeDefined();
      expect(failureInfo!.error.message).toBe('Network timeout');
      expect(failureInfo!.retryCount).toBe(0);
      expect(failureInfo!.timestamp).toBeGreaterThan(0);
    });

    it('should increment retryCount on repeated failures', async () => {
      const mockLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Persistent error')),
        dispose: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/persistent_failure', mockLoader);

      // First failure
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      let failures = sceneLoader.getFailedLoaders();
      expect(failures.get('/persistent_failure')!.retryCount).toBe(0);

      // Second failure
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      failures = sceneLoader.getFailedLoaders();
      expect(failures.get('/persistent_failure')!.retryCount).toBe(1);

      // Third failure
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      failures = sceneLoader.getFailedLoaders();
      expect(failures.get('/persistent_failure')!.retryCount).toBe(2);
    });

    it('should clear failures on successful load', async () => {
      const mockLoader = {
        updateView: vi
          .fn()
          .mockRejectedValueOnce(new Error('Temporary failure'))
          .mockResolvedValueOnce({
            positions: new Float32Array([1, 2, 3]),
            metadata: {
              totalPoints: 1,
              loadedPoints: 1,
              bounds: { clone: vi.fn().mockReturnThis() },
              ndim: 3,
              usedSpatialIndex: true,
            },
          }),
        dispose: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/recoverable', mockLoader);

      // First attempt fails
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      expect(sceneLoader.hasFailures()).toBe(true);

      // Second attempt succeeds
      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      expect(sceneLoader.hasFailures()).toBe(false);
      expect(sceneLoader.getFailedLoaders().size).toBe(0);
    });

    it('should continue loading other nodes when one fails', async () => {
      const failingLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Failure')),
        dispose: vi.fn(),
      };
      const successLoader = {
        updateView: vi.fn().mockResolvedValue({
          positions: new Float32Array([1, 2, 3]),
          metadata: {
            totalPoints: 1,
            loadedPoints: 1,
            bounds: { clone: vi.fn().mockReturnThis() },
            ndim: 3,
            usedSpatialIndex: true,
          },
        }),
        dispose: vi.fn(),
      };

      (sceneLoader as any).loaders.set('/failing', failingLoader);
      (sceneLoader as any).loaders.set('/success', successLoader);

      await sceneLoader.updateView({ displayDims: [0, 1, 2] });

      // Both should have been called
      expect(failingLoader.updateView).toHaveBeenCalled();
      expect(successLoader.updateView).toHaveBeenCalled();

      // Only one failure tracked
      expect(sceneLoader.getFailedLoaders().size).toBe(1);
      expect(sceneLoader.getFailedLoaders().has('/failing')).toBe(true);
    });

    it('should clear failures with clearFailures()', async () => {
      const mockLoader = {
        updateView: vi.fn().mockRejectedValue(new Error('Error')),
        dispose: vi.fn(),
      };
      (sceneLoader as any).loaders.set('/failed', mockLoader);

      await sceneLoader.updateView({ displayDims: [0, 1, 2] });
      expect(sceneLoader.hasFailures()).toBe(true);

      sceneLoader.clearFailures();
      expect(sceneLoader.hasFailures()).toBe(false);
      expect(sceneLoader.getFailedLoaders().size).toBe(0);
    });

    it('should warn user when multiple loaders fail', async () => {
      const consoleSpy = vi.spyOn(console, 'warn');

      const failingLoader1 = {
        updateView: vi.fn().mockRejectedValue(new Error('Error 1')),
        dispose: vi.fn(),
      };
      const failingLoader2 = {
        updateView: vi.fn().mockRejectedValue(new Error('Error 2')),
        dispose: vi.fn(),
      };

      (sceneLoader as any).loaders.set('/failing1', failingLoader1);
      (sceneLoader as any).loaders.set('/failing2', failingLoader2);

      await sceneLoader.updateView({ displayDims: [0, 1, 2] });

      // Should warn about multiple failures
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Some data could not be loaded')
      );
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('/failing1'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('/failing2'));
    });
  });

  describe('geometry updates', () => {
    beforeEach(async () => {
      vi.clearAllMocks();
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });

    /**
     * Drop a real point mesh into the loader's rootGroup so the
     * `getObjectByName(path)` lookup inside `commitPointsGeometry`
     * returns it. The placeholder carries interleaved-era attributes;
     * the commit path replaces the geometry with the texture-backed
     * layout (point texture + `aSortedIndex`) either way.
     */
    function attachPointsChild(name: string, oldCount: number): THREE.Mesh {
      const root = (sceneLoader as any).rootGroup as THREE.Group;
      const geom = new THREE.BufferGeometry();
      geom.setAttribute(
        'aCenter',
        new THREE.InstancedBufferAttribute(new Float32Array(oldCount * 3), 3)
      );
      geom.setAttribute(
        'aColor',
        new THREE.InstancedBufferAttribute(new Float32Array(oldCount * 3), 3)
      );
      geom.setAttribute(
        'aRadius',
        new THREE.InstancedBufferAttribute(new Float32Array(oldCount), 1)
      );
      geom.setAttribute(
        'aSharpness',
        new THREE.InstancedBufferAttribute(new Float32Array(oldCount), 1)
      );
      const points = new THREE.Mesh(geom);
      points.name = name;
      // commitPointsGeometry only writes `visiblePointCount` when the
      // node passes `isPointsUserData` (nodeType === 'points'). Mirror
      // what NodeFactory.createPointsNode would set up so the commit
      // path treats it as a real points node.
      points.userData = { nodeType: 'points', ndim: 3, visiblePointCount: oldCount };
      root.add(points);
      return points;
    }

    it('should update geometry with new points data', () => {
      const points = attachPointsChild('/test_points', 0);
      const newData = {
        positions: new Float32Array([4, 5, 6, 7, 8, 9]),
        colors: new Float32Array([1, 1, 1, 1, 1, 1]),
        radii: new Float32Array([0.5, 0.5]),
        sharpness: new Float32Array([2.0, 2.0]),
        pointCount: 2,
        ndim: 3,
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };

      (sceneLoader as any).updatePointsGeometry('/test_points', newData);

      // After the update the Points still has a (possibly recreated) geometry,
      // and its visiblePointCount reflects the new data.
      expect(points.geometry).toBeDefined();
      expect(points.userData.visiblePointCount).toBe(2);
    });

    it('writes new positions through whichever path the loader takes (pool or in-place)', () => {
      const points = attachPointsChild('/test_points', 1);
      const sameSizeData = {
        positions: new Float32Array([1, 2, 3]),
        colors: new Float32Array([1, 1, 1]),
        radii: new Float32Array([0.5]),
        sharpness: new Float32Array([2.0]),
        pointCount: 1,
        ndim: 3,
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: new THREE.Box3(),
          usedSpatialIndex: true,
        },
      };

      (sceneLoader as any).updatePointsGeometry('/test_points', sameSizeData);

      // Whether commit takes the buffer-pool path or the recreate path
      // (depends on whether _gpuBufferPool is wired up in this fixture),
      // the live point texture must reflect the new payload: center.xyz
      // is texel 0 of point 0 (see point-geometry.ts).
      const texture = getPointTexture(points.geometry);
      expect(texture).not.toBeNull();
      expect(Array.from((texture!.image.data as Float32Array).slice(0, 3))).toEqual([1, 2, 3]);
      expect(points.userData.visiblePointCount).toBe(1);
    });

    it('should update bounding box from metadata', () => {
      const points = attachPointsChild('/test_points', 1);
      const newBounds = new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(10, 10, 10));
      const newData = {
        positions: new Float32Array([1, 2, 3]),
        colors: new Float32Array([1, 1, 1]),
        radii: new Float32Array([0.5]),
        sharpness: new Float32Array([2.0]),
        pointCount: 1,
        ndim: 3,
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: newBounds,
          ndim: 3,
          usedSpatialIndex: true,
        },
      };

      (sceneLoader as any).updatePointsGeometry('/test_points', newData);

      // The same-size path computes bounding box from the geometry itself.
      // For the dispose-and-recreate path metadata bounds are cloned, but
      // either way the geometry ends up with a defined bounding box.
      expect(points.geometry.boundingBox).not.toBeNull();
    });

    it('should handle empty geometry updates (clearing points)', () => {
      attachPointsChild('/test_points', 1);

      const emptyData = {
        positions: new Float32Array([]), // Empty
        colors: new Float32Array([]),
        radii: new Float32Array([]),
        sharpness: new Float32Array([]),
        pointCount: 0,
        ndim: 4,
        metadata: {
          totalPoints: 1000,
          loadedPoints: 0, // No points visible at current slice
          bounds: new THREE.Box3(),
          ndim: 4,
          usedSpatialIndex: true,
        },
      };

      // Should not throw when updating to empty geometry
      expect(() => {
        (sceneLoader as any).updatePointsGeometry('/test_points', emptyData);
      }).not.toThrow();

      // With GPU buffer pool enabled, geometry is reused (not disposed)
      // The geometry is acquired from pool, updated in place, and reassigned
      // Disposal only happens on final cleanup, not on updates
    });
  });

  describe('scene dimensions initialization', () => {
    it('should initialize viewState from scene dimensions', async () => {
      mockZarrGroup.attrs = {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'z', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'time', unit: 's', range: [0, 10], display: false, step: 0.1 },
          ],
        },
      };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      const viewState = (sceneLoader as any).viewState;

      // Should have 3 displayed dimensions (x, y, z)
      expect(viewState.displayDims).toEqual([0, 1, 2]);

      // Should have slice position for all 4 dimensions
      expect(viewState.slicePosition.length).toBe(4);

      // Should have tolerance for all 4 dimensions
      expect(viewState.tolerance.length).toBe(4);
    });

    it('should handle validation with ViewStateManager', async () => {
      // [data.md/W6][P2] Strengthen: previously only asserted scene was
      // defined. The substantive contract is that the inverted-range
      // dimension (max < min) still survives into the loader's view state —
      // the implementation chooses to load rather than reject. Pin both
      // (a) scene is a real LuxarScene group, and (b) the inverted range
      // surfaces in the loader's saved viewState (slicePosition has the
      // right length for the 2 dims that were declared).
      const dimensionsWithIssues = {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [100, 0], display: true, step: 1 }, // max < min
          ],
        },
      };

      mockZarrGroup.attrs = dimensionsWithIssues;

      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(scene).toBeDefined();
      expect(scene.name).toBe('LuxarScene');
      const viewState = (sceneLoader as any).viewState;
      // ViewState has one entry per declared dim.
      expect(viewState.slicePosition.length).toBe(2);
      expect(viewState.tolerance.length).toBe(2);
    });

    it('should handle invalid dimensions gracefully', async () => {
      mockZarrGroup.attrs = {
        scene_dimensions: 'not_an_object', // Invalid type
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      // Should not throw, just log warning
      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).resolves.toBeDefined();

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Invalid scene_dimensions format')
      );
    });

    it('should handle missing dimensions array', async () => {
      mockZarrGroup.attrs = {
        scene_dimensions: {}, // Missing dimensions array
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Invalid scene_dimensions format')
      );
    });

    it('should handle dimensions with more than 3 displayed', async () => {
      mockZarrGroup.attrs = {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'z', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'time', unit: 's', range: [0, 10], display: true, step: 0.1 }, // 4 displayed
          ],
        },
      };

      // Should handle gracefully even with too many displayed dimensions
      // [data.md/W6][P2] Strengthen: pin the contract that the loader
      // capped displayed dims to 3 (the documented limit). With 4 dims
      // marked display=true, the saved viewState should still have
      // displayDims of length <= 3.
      const scene = await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(scene).toBeDefined();
      expect(scene.name).toBe('LuxarScene');
      const viewState = (sceneLoader as any).viewState;
      expect(viewState.displayDims.length).toBeLessThanOrEqual(3);
      // The first three displayed dimensions (indices 0, 1, 2) survive;
      // the 4th (index 3) is dropped from displayDims.
      expect(viewState.displayDims).toContain(0);
      expect(viewState.displayDims).toContain(1);
      expect(viewState.displayDims).toContain(2);
    });

    it('does NOT toast on a 16D scene (≤ WASM ceiling)', async () => {
      notifierMocks.toast.mockClear();
      const dims = Array.from({ length: 16 }, (_, i) => ({
        name: `d${i}`,
        unit: '',
        range: [0, 10] as [number, number],
        display: i < 3,
        step: 1,
      }));
      mockZarrGroup.attrs = { scene_dimensions: { dimensions: dims } };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(notifierMocks.toast).not.toHaveBeenCalled();
    });

    it('toasts on > 16D scenes warning about WASM fallback', async () => {
      notifierMocks.toast.mockClear();
      const dims = Array.from({ length: 18 }, (_, i) => ({
        name: `d${i}`,
        unit: '',
        range: [0, 10] as [number, number],
        display: i < 3,
        step: 1,
      }));
      mockZarrGroup.attrs = { scene_dimensions: { dimensions: dims } };

      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      expect(notifierMocks.toast).toHaveBeenCalledWith(
        expect.stringContaining('18 dimensions'),
        expect.any(Number)
      );
    });
  });

  describe('points data validation', () => {
    it('should validate empty datasets', () => {
      const emptyData = {
        positions: new Float32Array([]),
        metadata: {
          totalPoints: 0,
          loadedPoints: 0,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      // Should not throw, just log info
      expect(() => {
        (sceneLoader as any).nodeFactory.validateLoadedPointsData(emptyData);
      }).not.toThrow();

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Empty dataset detected'));
    });

    it('should stay silent for empty placeholder datasets (pre-fetch)', () => {
      const emptyData = {
        positions: new Float32Array([]),
        metadata: {
          totalPoints: 0,
          loadedPoints: 0,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const warnSpy = vi.spyOn(console, 'warn');

      // isPlaceholder=true: the empty placeholder built before the first
      // fetch is expected, so it must NOT emit the "empty dataset" warning
      // (otherwise it fires once per points node on every scene load).
      expect(() => {
        (sceneLoader as any).nodeFactory.validateLoadedPointsData(emptyData, true);
      }).not.toThrow();

      expect(warnSpy).not.toHaveBeenCalledWith(expect.stringContaining('Empty dataset detected'));
    });

    it('should detect malformed positions (not multiple of 3)', () => {
      const malformedData = {
        positions: new Float32Array([1, 2, 3, 4]), // Length 4, not divisible by 3
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      expect(() => {
        (sceneLoader as any).nodeFactory.validateLoadedPointsData(malformedData);
      }).toThrow('not divisible by 3');
    });

    it('should detect colors length mismatch', () => {
      const data = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]), // 2 points
        colors: new Float32Array([1, 0, 0]), // Only 1 color (should be 2)
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).nodeFactory.validateLoadedPointsData(data);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Colors length mismatch'),
        expect.any(Object)
      );
    });

    it('should detect radii length mismatch', () => {
      const data = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]), // 2 points
        radii: new Float32Array([0.5]), // Only 1 radius (should be 2)
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).nodeFactory.validateLoadedPointsData(data);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Radii length mismatch'),
        expect.any(Object)
      );
    });

    it('should detect sharpness length mismatch', () => {
      const data = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]), // 2 points
        sharpness: new Float32Array([2.0, 2.0, 2.0]), // 3 values (should be 2)
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).nodeFactory.validateLoadedPointsData(data);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Sharpness length mismatch'),
        expect.any(Object)
      );
    });

    it('should validate correct data without warnings', () => {
      const validData = {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]),
        colors: new Float32Array([1, 0, 0, 0, 1, 0]),
        radii: new Float32Array([0.5, 0.7]),
        sharpness: new Float32Array([2.0, 3.0]),
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
        },
      };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).nodeFactory.validateLoadedPointsData(validData);

      // Should not have any warnings (only info/log messages)
      const warnCalls = consoleSpy.mock.calls.filter((call) =>
        call.some((arg) => typeof arg === 'string' && arg.includes('mismatch'))
      );
      expect(warnCalls.length).toBe(0);
    });
  });

  describe('color mode validation', () => {
    it('should warn when metadata indicates HDR but array is Uint8', () => {
      const colors = new Uint8Array([255, 128, 64]);
      const metadata = { color_mode: 'hdr' };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).nodeFactory.validateColorMode(colors, metadata);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('metadata indicates HDR colors but array is Uint8Array')
      );
    });

    it('should not warn for Float32Array with HDR metadata', () => {
      const colors = new Float32Array([1.5, 2.0, 3.5]); // HDR values > 1.0
      const metadata = { color_mode: 'hdr' };

      const consoleSpy = vi.spyOn(console, 'warn');

      (sceneLoader as any).nodeFactory.validateColorMode(colors, metadata);

      // Should not have HDR/Uint8 mismatch warning
      const warnCalls = consoleSpy.mock.calls.filter((call) =>
        call.some((arg) => typeof arg === 'string' && arg.includes('Uint8Array'))
      );
      expect(warnCalls.length).toBe(0);
    });

    it('should suggest SDR mode when HDR values are in [0,1] range', () => {
      const colors = new Float32Array([0.5, 0.8, 1.0]); // All in [0,1]
      const metadata = { color_mode: 'hdr' };

      const consoleSpy = vi.spyOn(console, 'log');

      (sceneLoader as any).nodeFactory.validateColorMode(colors, metadata);

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Consider using SDR mode for better compression')
      );
    });

    it('should handle Uint16Array colors', () => {
      const colors = new Uint16Array([65535, 32768, 16384]);
      const metadata = {};

      const consoleSpy = vi.spyOn(console, 'log');

      // Should not throw, should log color type
      expect(() => {
        (sceneLoader as any).nodeFactory.validateColorMode(colors, metadata);
      }).not.toThrow();

      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Uint16Array'));
    });
  });
});
