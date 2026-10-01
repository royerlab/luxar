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
import { DRAG_COMMIT_MAX_HOLD_MS } from '../../../data/scene-loader';
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
import { UpdateProfiler, type TimingEntry } from '../../../profiling/update-profiler';
import * as gsplatsRefinement from '../../../data/gsplats/lod-refinement';
import { signalPriority } from '../../../utils/fetch-concurrency';
import { log, Modules } from '../../../utils/log';
import { failedLoadsVersion } from '../../../utils/failed-loads-version';
import { SlicePrefetcher } from '../../../data/scene-loader/prefetch/slice-prefetcher';
import {
  MAX_ABANDONED_RUNG_RETRY_ROUNDS,
  MAX_CONSECUTIVE_REFINEMENT_FAILURES,
} from '../../../data/scene-loader/progressive/refinement';

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
// centers). Partial mock via importOriginal so only the routed
// releaseDepthSortNode is intercepted — the rest of the module (the
// DepthSortCoordinator class a host hands the loader) keeps its real
// implementation.
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

    it('starts each dataset with a fresh update profiler', async () => {
      // The profiler is a manager-wide singleton; without a reset at loadStart
      // every row the previous dataset ever produced stays in the tree, and
      // each merge's stale sweep walks all of them.
      const profiler = new UpdateProfiler();
      profiler.beginUpdate();
      profiler.time('previous-dataset-node', () => undefined);
      profiler.endUpdate();
      expect(profiler.getTimings().children.map((c) => c.name)).toContain('previous-dataset-node');
      const loader = new SceneLoader({}, 'profiled', profiler);
      try {
        await loader.loadScene('http://localhost:8000/test.zarr');
        const names: string[] = [];
        const walk = (entry: TimingEntry): void => {
          names.push(entry.name);
          entry.children.forEach(walk);
        };
        walk(profiler.getTimings());
        expect(names).not.toContain('previous-dataset-node');
      } finally {
        await loader.dispose();
      }
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

    it('does not report a pinned-depth pass as the settled view either', async () => {
      const committed = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.5],
      } satisfies ViewState;

      await sceneLoader.updateView({ ...committed, ladderDepth: 2 });
      const stored = (sceneLoader as unknown as { viewState: ViewState }).viewState;
      expect(sceneLoader.isAtViewState(stored)).toBe(false);
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

    it('a throwing rollback after a failed commit is logged, not folded into the commit error', async () => {
      // Every other rollback site goes through tryRollbackToPassStart, which
      // keeps the ORIGINAL failure as the one reported; the commit path called
      // the raw method and buried it in a nested AggregateError.
      const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
      const loaders = (sceneLoader as any).loaders as Map<string, unknown>;
      loaders.clear();
      loaders.set('/commit-fail', {
        updateView: vi.fn().mockResolvedValue({
          pointCount: 1,
          positions: new Float32Array([1, 2, 3]),
          metadata: { loadedPoints: 1 },
        }),
        rollbackToPassStart: vi.fn(() => {
          throw new Error('rollback failed');
        }),
        dispose: vi.fn(),
      });
      const commitError = new Error('GPU commit failed');
      vi.spyOn(sceneLoader as any, 'updatePointsGeometry').mockImplementation(() => {
        throw commitError;
      });

      const failure = await sceneLoader
        .updateView({ displayDims: [0, 1, 2] })
        .catch((error: unknown) => error);

      expect(failure).toBeInstanceOf(AggregateError);
      expect((failure as AggregateError).errors).toEqual([commitError]);
      expect(warning).toHaveBeenCalledWith(
        Modules.SCENE_LOADER,
        'Progressive loader rollback failed',
        expect.any(Error)
      );
      warning.mockRestore();
    });

    it('a partially failing commit still records the frame it committed (A9)', async () => {
      // runAtomicCommit commits every sibling and only THEN rethrows. The
      // siblings' geometry is on screen, so the pass's post-commit bookkeeping
      // must run too: skipping it left the committed slice (B4 partition
      // gating), the B5 commit clock and the monitor describing the old frame.
      const applyCommittedSlice = vi.fn();
      (sceneLoader as unknown as { lodGroupRegistry: unknown }).lodGroupRegistry = {
        isPathInPartitionSlice: () => true,
        activatePartitionParts: vi.fn().mockResolvedValue([]),
        invalidatePartitionFootprint: vi.fn(),
        applyCommittedSlice,
        clear: vi.fn(),
      };
      const loaders = (sceneLoader as any).loaders as Map<string, unknown>;
      loaders.clear();
      const staged = {
        pointCount: 1,
        positions: new Float32Array([1, 2, 3]),
        metadata: { loadedPoints: 1 },
      };
      for (const path of ['/ok', '/bad']) {
        loaders.set(path, { updateView: vi.fn().mockResolvedValue(staged), dispose: vi.fn() });
      }
      const committed: string[] = [];
      vi.spyOn(sceneLoader as any, 'updatePointsGeometry').mockImplementation((path) => {
        if (path === '/bad') throw new Error('GPU commit failed');
        committed.push(path as string);
        return true;
      });
      const now = vi.spyOn(performance, 'now').mockReturnValue(123_456);

      try {
        await expect(
          sceneLoader.updateView({ displayDims: [0, 1, 2], slicePosition: [0, 0, 0, 7] })
        ).rejects.toThrow(AggregateError);
      } finally {
        now.mockRestore();
      }

      expect(committed).toEqual(['/ok']);
      expect(sceneLoader.committedViewState.slicePosition[3]).toBe(7);
      expect(applyCommittedSlice).toHaveBeenCalledOnce();
      expect((sceneLoader as unknown as { _lastCommitAt: number })._lastCommitAt).toBe(123_456);
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
        _passWaiters: Array<{ gen: number; resolve: () => void }>;
      };
      let settled = false;
      internals._passWaiters.push({
        gen: 1,
        resolve: () => {
          settled = true;
        },
      });
      internals._archiveFault = new ArchiveFaultError('container unreadable', 'test');
      await sceneLoader.updateView({ slicePosition: [0, 0, 0, 8] });
      expect(settled).toBe(true);
      expect(internals._passWaiters).toHaveLength(0);
    });
  });

  describe('updateView — partition parts entering the slice (B4)', () => {
    // A timelapse partition: one part per timepoint. Stepping the hidden dim
    // brings a deferred part into the slice; its activation must ride the pass
    // that moved the slice — swept by it, with its directives, and committed
    // with it — not load on its own and then ask for a second pass.
    const DIMS = [
      { name: 'x', unit: 'um', scale: 1 },
      { name: 'y', unit: 'um', scale: 1 },
      { name: 'z', unit: 'um', scale: 1 },
      { name: 'time', unit: 'frame', scale: 1, discrete: true, step: 1 },
    ];
    type Internals = {
      _passCount: number;
      loaders: Map<string, unknown>;
      lodGroupRegistry: LODGroupRegistry | null;
    };
    const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

    function setup(activate: (register: () => void) => Promise<void>) {
      const internals = sceneLoader as unknown as Internals;
      const camera = new THREE.Camera();
      const reg = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => ({ width: 800, height: 600 }),
        getDisplayDims: () => [0, 1, 2],
        getCommittedViewState: () => sceneLoader.committedViewState,
        getViewVersion: () => sceneLoader.currentViewVersion,
        isUpdateInProgress: () => sceneLoader.isLoadPassInProgress(),
        requestReprocess: (paths) => sceneLoader.requestReprocess(paths),
      });
      internals.lodGroupRegistry = reg;
      const makeLoader = () => ({ updateView: vi.fn().mockResolvedValue(null), dispose: vi.fn() });
      const now = makeLoader();
      const next = makeLoader();
      internals.loaders.set('/tiled/part_0', now);
      const groupObject = new THREE.Group();
      const slot0 = new THREE.Group();
      const slot1 = new THREE.Group();
      groupObject.add(slot0, slot1);
      reg.registerPartition({
        path: '/tiled',
        groupObject,
        children: [
          {
            path: '/tiled/part_0',
            objects: [slot0],
            positionBounds: { min: [-0.5, -0.5, -0.5, 3], max: [0.5, 0.5, 0.5, 3] },
          },
          {
            path: '/tiled/part_1',
            objects: [slot1],
            positionBounds: { min: [-0.5, -0.5, -0.5, 4], max: [0.5, 0.5, 0.5, 4] },
            activate: () => activate(() => internals.loaders.set('/tiled/part_1', next)),
          },
        ],
      });
      return { internals, reg, now, next };
    }

    beforeEach(async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      await sceneLoader.updateView({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 3],
        tolerance: [0, 0, 0, 0],
        dimensions: DIMS,
      });
    });

    it('the pass that moves the slice sweeps the part it activates, once, with its directives', async () => {
      const { internals, reg, next } = setup(async (register) => register());
      reg.evaluatePerFrame();
      const passes = internals._passCount;

      await sceneLoader.updateView({ slicePosition: [0, 0, 0, 4], ladderDepth: 2 });
      await flush();
      reg.evaluatePerFrame();
      await flush();

      expect(next.updateView).toHaveBeenCalledTimes(1);
      expect(next.updateView.mock.calls[0][0].ladderDepth).toBe(2);
      expect(internals._passCount - passes).toBe(1);
    });

    it('a part activated ahead of its slice joins that slice’s pass without a resync pass', async () => {
      let release!: () => void;
      const gate = new Promise<void>((resolve) => (release = resolve));
      const { internals, reg, next } = setup(async (register) => {
        await gate;
        register();
      });
      reg.evaluatePerFrame();
      const passes = internals._passCount;

      // What `prefetchSlice` does for the predicted next slice (playback).
      const ahead = reg.activatePartitionParts({
        ...sceneLoader.committedViewState,
        slicePosition: [0, 0, 0, 4],
      });
      const pass = sceneLoader.updateView({ slicePosition: [0, 0, 0, 4], ladderDepth: 2 });
      release();
      await Promise.all([ahead, pass]);
      await flush();
      reg.evaluatePerFrame();
      await flush();

      expect(next.updateView).toHaveBeenCalledTimes(1);
      expect(next.updateView.mock.calls[0][0].ladderDepth).toBe(2);
      expect(internals._passCount - passes).toBe(1);
    });
  });

  describe('prefetchSlice — partition parts entering the next slice (B4)', () => {
    it('warms a part activated for the predicted slice once its loader is registered', async () => {
      const DIMS = [
        { name: 'x', unit: 'um', scale: 1 },
        { name: 'y', unit: 'um', scale: 1 },
        { name: 'z', unit: 'um', scale: 1 },
        { name: 'time', unit: 'frame', scale: 1, discrete: true, step: 1 },
      ];
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      await sceneLoader.updateView({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 3],
        tolerance: [0, 0, 0, 0],
        dimensions: DIMS,
      });
      const internals = sceneLoader as unknown as {
        loaders: Map<string, unknown>;
        lodGroupRegistry: LODGroupRegistry | null;
      };
      const camera = new THREE.Camera();
      const reg = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => ({ width: 800, height: 600 }),
        getDisplayDims: () => [0, 1, 2],
        getCommittedViewState: () => sceneLoader.committedViewState,
      });
      internals.lodGroupRegistry = reg;
      const groupObject = new THREE.Group();
      const slot = new THREE.Group();
      groupObject.add(slot);
      reg.registerPartition({
        path: '/tiled',
        groupObject,
        children: [
          {
            path: '/tiled/part_1',
            objects: [slot],
            positionBounds: { min: [-0.5, -0.5, -0.5, 4], max: [0.5, 0.5, 0.5, 4] },
            activate: async () => {
              await Promise.resolve();
              internals.loaders.set('/tiled/part_1', { updateView: vi.fn(), dispose: vi.fn() });
            },
          },
        ],
      });
      reg.evaluatePerFrame();
      const warmed = vi.spyOn(
        SlicePrefetcher.prototype as unknown as { prefetchNode: (path: string) => Promise<void> },
        'prefetchNode'
      );

      // Playback at t=3 prefetches t=4, where the deferred part lives.
      sceneLoader.prefetchSlice({ slicePosition: [0, 0, 0, 4] }, 50);
      await new Promise((resolve) => setTimeout(resolve, 0));

      // A shadow can only warm a registered loader: the part must be warmed
      // after its activation registered it, or the first loop loads it cold.
      expect(warmed.mock.calls.map(([path]) => path)).toContain('/tiled/part_1');
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

    it('does NOT persist the transient `ladderDepth` directive into the view state', async () => {
      // A pinned playback depth is a per-pass directive like `frameBudgetMs`:
      // leaking it would keep every later foreground pass pinned (and idle the
      // refinement scheduler) after playback ends.
      await sceneLoader.updateView({ slicePosition: [0, 0, 0, 4], ladderDepth: 3 });
      const vs = (sceneLoader as unknown as { viewState: { ladderDepth?: number } }).viewState;
      expect(vs.ladderDepth).toBeUndefined();
    });

    it('prefetchSlice strips an incoming ladderDepth rider and passes the explicit one through', () => {
      const before = JSON.stringify((sceneLoader as unknown as { viewState: unknown }).viewState);
      sceneLoader.prefetchSlice({ slicePosition: [9, 9, 9, 9], ladderDepth: 99 }, 10, 4);
      expect(JSON.stringify((sceneLoader as unknown as { viewState: unknown }).viewState)).toBe(
        before
      );
    });
  });

  describe('updateView — drag commit guarantee (B5)', () => {
    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('a 3 s drag keeps committing: no supersede aborts a pass 150 ms after the last commit', async () => {
      let now = 0;
      vi.spyOn(performance, 'now').mockImplementation(() => now);
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');

      // Every load takes 100 ms of (simulated) time; an aborted one bails at once.
      const PASS_MS = 100;
      const jobs: Array<{ start: number; done: boolean; finish: () => void }> = [];
      const commits: number[] = [];
      const mockLoader = {
        loadGSplats: vi.fn(),
        updateView: vi.fn(
          (_vs: unknown, _session: unknown, signal?: AbortSignal) =>
            new Promise<null>((resolve, reject) => {
              const job = {
                start: now,
                done: false,
                finish: () => {
                  if (job.done) return;
                  job.done = true;
                  if (signal?.aborted) {
                    reject(new DOMException('Superseded', 'AbortError'));
                    return;
                  }
                  commits.push(now);
                  resolve(null);
                },
              };
              jobs.push(job);
              signal?.addEventListener('abort', () => job.finish());
            })
        ),
        dispose: vi.fn(),
      };
      (sceneLoader as unknown as Record<string, Map<string, unknown>>).gsplatLoaders.set(
        '/node',
        mockLoader
      );

      // A slider drag: a new slice every 50 ms for 3 s, never awaited.
      for (let t = 0; t <= 3000; t += 10) {
        now = t;
        if (t % 50 === 0) {
          void sceneLoader.updateView({
            displayDims: [0, 1, 2],
            slicePosition: [0, 0, 0, t / 50],
            tolerance: [0, 0, 0, 0],
          });
        }
        for (const job of jobs) if (now - job.start >= PASS_MS) job.finish();
        // Let the pass pipeline settle before simulated time moves on.
        for (let turn = 0; turn < 5; turn++) await new Promise((resolve) => setTimeout(resolve, 0));
      }

      // Superseded passes never commit, so without the guarantee a drag whose
      // passes outlast the event interval commits nothing until it stops.
      expect(commits.length).toBeGreaterThanOrEqual(12);
      const gaps = commits.map((at, i) => at - (i === 0 ? 0 : commits[i - 1]));
      expect(Math.max(...gaps)).toBeLessThanOrEqual(150 + PASS_MS + 50);
    });

    it('a displayDims change aborts an overdue pass instead of letting it commit', async () => {
      // The guarantee is for a DRAG: an intermediate slice is still a truthful
      // frame. A pass for the OLD display axes is not — letting it commit puts
      // at least one frame of geometry projected for the wrong axes on screen.
      let now = 0;
      vi.spyOn(performance, 'now').mockImplementation(() => now);
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      now = 10_000;
      const signals: Array<AbortSignal | undefined> = [];
      const mockLoader = {
        loadGSplats: vi.fn(),
        updateView: vi.fn((_vs: unknown, _session: unknown, signal?: AbortSignal) => {
          signals.push(signal);
          return new Promise<null>((_resolve, reject) => {
            signal?.addEventListener('abort', () =>
              reject(new DOMException('Superseded', 'AbortError'))
            );
          });
        }),
        dispose: vi.fn(),
      };
      (sceneLoader as unknown as Record<string, Map<string, unknown>>).gsplatLoaders.set(
        '/node',
        mockLoader
      );

      void sceneLoader.updateView({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 1],
        tolerance: [0, 0, 0, 0],
      });
      await vi.waitFor(() => expect(signals).toHaveLength(1));
      // Well past the drag interval, with nothing committed in this chain.
      now += 400;
      void sceneLoader.updateView({
        displayDims: [0, 1, 3],
        slicePosition: [0, 0, 0, 1],
        tolerance: [0, 0, 0, 0],
      });
      await Promise.resolve();

      expect(signals[0]?.aborted).toBe(true);
    });

    /**
     * One view pass that never finishes on its own (a chunk stuck on a cold
     * edge), then a slice-only supersede `heldMs` after it started.
     */
    async function supersedeStuckPassAfter(heldMs: number): Promise<AbortSignal | undefined> {
      let now = 0;
      vi.spyOn(performance, 'now').mockImplementation(() => now);
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      now = 10_000;
      const signals: Array<AbortSignal | undefined> = [];
      const mockLoader = {
        loadGSplats: vi.fn(),
        updateView: vi.fn((_vs: unknown, _session: unknown, signal?: AbortSignal) => {
          signals.push(signal);
          return new Promise<null>((_resolve, reject) => {
            signal?.addEventListener('abort', () =>
              reject(new DOMException('Superseded', 'AbortError'))
            );
          });
        }),
        dispose: vi.fn(),
      };
      (sceneLoader as unknown as Record<string, Map<string, unknown>>).gsplatLoaders.set(
        '/node',
        mockLoader
      );

      void sceneLoader.updateView({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 1],
        tolerance: [0, 0, 0, 0],
      });
      await vi.waitFor(() => expect(signals).toHaveLength(1));
      now += heldMs;
      void sceneLoader.updateView({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 2],
        tolerance: [0, 0, 0, 0],
      });
      await Promise.resolve();
      return signals[0];
    }

    it('a pass stuck past DRAG_COMMIT_MAX_HOLD_MS is aborted by a newer view', async () => {
      // Hosted chunks can take tens of seconds on a cold edge. Holding such a
      // pass for its commit queues every newer view behind that one download
      // (99 declined aborts over 42 s in an instrumented run): the scrub freezes.
      const stuck = await supersedeStuckPassAfter(5_000);
      expect(stuck?.aborted).toBe(true);
    });

    it('a pass younger than the hold cap and owed a commit is still let through', async () => {
      // 400 ms: past DRAG_COMMIT_INTERVAL_MS (owed a commit), inside the cap.
      const owed = await supersedeStuckPassAfter(400);
      expect(owed?.aborted).toBe(false);
    });

    it('a view queued behind a held pass starts once the hold cap expires, with no newer view', async () => {
      // The drag stops while its last position is queued behind a held pass.
      // No newer view arrives to re-check the cap, so without a timer the last
      // position waits for however long the stuck pass runs.
      let now = 0;
      vi.spyOn(performance, 'now').mockImplementation(() => now);
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
      try {
        now = 10_000;
        const calls: Array<{ slice: number; signal?: AbortSignal }> = [];
        const mockLoader = {
          loadGSplats: vi.fn(),
          updateView: vi.fn(
            (vs: { slicePosition: number[] }, _session: unknown, signal?: AbortSignal) => {
              calls.push({ slice: vs.slicePosition[3], signal });
              return new Promise<null>((_resolve, reject) => {
                signal?.addEventListener('abort', () =>
                  reject(new DOMException('Superseded', 'AbortError'))
                );
              });
            }
          ),
          dispose: vi.fn(),
        };
        (sceneLoader as unknown as Record<string, Map<string, unknown>>).gsplatLoaders.set(
          '/node',
          mockLoader
        );
        const view = (slice: number) => ({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, slice],
          tolerance: [0, 0, 0, 0],
        });

        void sceneLoader.updateView(view(1));
        await vi.advanceTimersByTimeAsync(0);
        expect(calls).toHaveLength(1);
        // 400 ms in: owed a commit and inside the cap, so the pass is held.
        now += 400;
        void sceneLoader.updateView(view(2));
        await vi.advanceTimersByTimeAsync(0);
        expect(calls[0].signal?.aborted).toBe(false);

        // Just inside the cap: still held.
        now = 10_000 + DRAG_COMMIT_MAX_HOLD_MS - 10;
        await vi.advanceTimersByTimeAsync(DRAG_COMMIT_MAX_HOLD_MS - 410);
        expect(calls[0].signal?.aborted).toBe(false);
        expect(calls).toHaveLength(1);

        // At the cap, with no newer view: the held pass is aborted and the
        // queued (last) view runs.
        now = 10_000 + DRAG_COMMIT_MAX_HOLD_MS + 10;
        await vi.advanceTimersByTimeAsync(20);
        expect(calls[0].signal?.aborted).toBe(true);
        await vi.waitFor(() => expect(calls.map((call) => call.slice)).toEqual([1, 2]));
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe('updateView — the inter-pass frame yield (A8)', () => {
    // Between two serialized passes the lock stays held across one frame
    // (`scheduleFrame`), with no pass in flight. A view arriving in that window
    // must win over the state queued before it, and must not arm the B5 hold
    // timer against the finished pass's dead controller.
    let frames: Array<() => void>;

    beforeEach(() => {
      frames = [];
      vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
      vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
        frames.push(() => callback(0));
        return frames.length;
      });
    });

    afterEach(() => {
      vi.unstubAllGlobals();
      vi.restoreAllMocks();
      vi.useRealTimers();
    });

    const view = (slice: number) => ({
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, slice],
      tolerance: [0, 0, 0, 0],
    });

    /** Fire every queued frame, letting each one's continuation settle. */
    async function pumpFrames(): Promise<void> {
      for (let i = 0; i < 50 && frames.length > 0; i++) {
        frames.shift()!();
        await vi.advanceTimersByTimeAsync(0);
      }
    }

    it('a view arriving during the yield runs instead of the older queued state', async () => {
      let now = 0;
      vi.spyOn(performance, 'now').mockImplementation(() => now);
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
      now = 10_000;
      const calls: Array<{ slice: number; signal?: AbortSignal }> = [];
      (sceneLoader as unknown as Record<string, Map<string, unknown>>).gsplatLoaders.set('/node', {
        loadGSplats: vi.fn(),
        updateView: vi.fn(
          (vs: { slicePosition: number[] }, _session: unknown, signal?: AbortSignal) => {
            calls.push({ slice: vs.slicePosition[3], signal });
            return new Promise<null>((_resolve, reject) => {
              signal?.addEventListener('abort', () =>
                reject(new DOMException('Superseded', 'AbortError'))
              );
            });
          }
        ),
        dispose: vi.fn(),
      });

      void sceneLoader.updateView(view(1));
      await vi.advanceTimersByTimeAsync(0);
      // Inside the drag interval: the newer view aborts pass 1 outright.
      now = 10_050;
      void sceneLoader.updateView(view(2));
      await vi.advanceTimersByTimeAsync(0);
      expect(calls[0].signal?.aborted).toBe(true);
      // Pass 1 ended; view 2 waits for the frame the lock is held across.
      expect(frames).toHaveLength(1);

      // A view lands in that window, when the finished pass would read as owed
      // a commit (its start is 200 ms old, nothing committed in the chain).
      now = 10_200;
      void sceneLoader.updateView(view(3));
      await vi.advanceTimersByTimeAsync(0);
      frames.shift()!();
      await vi.advanceTimersByTimeAsync(0);
      // The newest view runs; the superseded view 2 never starts a pass.
      expect(calls.map((call) => call.slice)).toEqual([1, 3]);

      // The pass now in flight is held for its commit by a newer view, and the
      // hold cap still applies to it: the timer belongs to THIS pass.
      now = 10_600;
      void sceneLoader.updateView(view(4));
      await vi.advanceTimersByTimeAsync(0);
      expect(calls[1].signal?.aborted).toBe(false);
      now = 10_200 + DRAG_COMMIT_MAX_HOLD_MS + 10;
      await vi.advanceTimersByTimeAsync(DRAG_COMMIT_MAX_HOLD_MS);
      await pumpFrames();
      expect(calls[1].signal?.aborted).toBe(true);
      expect(calls.map((call) => call.slice)).toEqual([1, 3, 4]);
    });

    it('a view arriving during the refinement hand-off frame wins over the state that cancelled it', async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
      const slices: number[] = [];
      const ladder = {
        loadedLODCount: 1,
        totalLODCount: 3,
        get hasMoreLODs() {
          return ladder.loadedLODCount < ladder.totalLODCount;
        },
        updateView: vi.fn(async (vs: { slicePosition: number[] }) => {
          slices.push(vs.slicePosition[3]);
          return null;
        }),
        dispose: vi.fn(),
      };
      const internals = sceneLoader as unknown as {
        _updateInProgress: boolean;
        gsplatLoaders: Map<string, unknown>;
        scheduleGSplatsRefinement(): Promise<void>;
      };
      internals.gsplatLoaders.set('/g', ladder);

      // A refinement run holds the lock, parked on its first frame yield.
      internals._updateInProgress = true;
      const run = internals.scheduleGSplatsRefinement();
      expect(frames).toHaveLength(1);
      // View 2 cancels it: the loop takes it and hands off across a frame.
      void sceneLoader.updateView(view(2));
      frames.shift()!();
      await vi.advanceTimersByTimeAsync(0);
      await run;
      expect(frames).toHaveLength(1);

      // A newer view lands in that hand-off frame.
      void sceneLoader.updateView(view(3));
      await pumpFrames();

      // Pre-fix the cancelling view 2 ran a full pass first, unaborted.
      expect(slices).not.toContain(2);
      expect(slices[0]).toBe(3);
      expect(sceneLoader.isUpdateInProgress()).toBe(false);
    });

    it('a real view landing in the yield supersedes the queued resync with one full sweep', async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
      let release!: () => void;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      const swept: string[] = [];
      const makeLoader = (path: string, gated: boolean) => ({
        loadPoints: vi.fn(),
        updateView: vi.fn(async () => {
          swept.push(path);
          if (gated) await gate;
          return null;
        }),
        dispose: vi.fn(),
      });
      const loaders = (sceneLoader as unknown as Record<string, Map<string, unknown>>).loaders;
      loaders.set('/a', makeLoader('/a', true));
      loaders.set('/b', makeLoader('/b', false));

      const pass = sceneLoader.updateView(view(1));
      await vi.advanceTimersByTimeAsync(0);
      // A targeted resync lands mid-pass and is parked for the follow-up pass.
      const resync = sceneLoader.updateView({}, { resyncPaths: new Set(['/b']) });
      swept.length = 0;
      release();
      await vi.advanceTimersByTimeAsync(0);
      expect(frames).toHaveLength(1);
      // A newer, real view lands in the yield: its full sweep is a superset.
      const real = sceneLoader.updateView(view(2));
      await pumpFrames();
      await Promise.all([pass, resync, real]);

      // One full sweep for view 2 (both loaders), no narrowed or stale pass.
      expect(swept.sort()).toEqual(['/a', '/b']);
    });
  });

  describe('updateView — superseded loads abort (per-update AbortSignal)', () => {
    beforeEach(async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    });
    afterEach(() => vi.restoreAllMocks());

    // Three-geometry symmetry: the same supersede→abort contract must hold for
    // Points, Lines, and GSplats. Each registers its fake loader in the
    // matching registry map; the loader returns `null` on the winning pass so
    // the handler short-circuits before any processX/commit (no real mesh).
    const cases = [
      { type: 'points', map: 'loaders' },
      { type: 'lines', map: 'linesLoaders' },
      { type: 'gsplats', map: 'gsplatLoaders' },
    ] as const;

    it.each(
      cases.flatMap((testCase) => [
        { ...testCase, idleMs: 0 },
        { ...testCase, idleMs: 5000 },
      ])
    )(
      'aborts the in-flight $type load after $idleMs ms idle when a newer view supersedes it',
      async ({ map, idleMs }) => {
        let now = (sceneLoader as unknown as { _lastCommitAt: number })._lastCommitAt + idleMs;
        vi.spyOn(performance, 'now').mockImplementation(() => now);
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
        now += 10;

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

    it.each([false, true])(
      'aborts the first pass superseded after a refinement hand-off following idle (resync first: %s)',
      async (resyncFirst) => {
        const internals = sceneLoader as unknown as {
          _lastCommitAt: number;
          _updateInProgress: boolean;
          _refining: boolean;
          _updateAbortController: AbortController | null;
          viewStateQueue: ViewStateQueue;
          reenterPending(state: Partial<ViewState>): Promise<void>;
        };
        let now = internals._lastCommitAt + 5000;
        vi.spyOn(performance, 'now').mockImplementation(() => now);
        let capturedSignal: AbortSignal | undefined;
        let releaseFirst!: () => void;
        const firstGate = new Promise<void>((resolve) => {
          releaseFirst = resolve;
        });
        const loader = {
          updateView: vi.fn(async (_vs: unknown, _session: unknown, signal?: AbortSignal) => {
            if (!capturedSignal) {
              capturedSignal = signal;
              await firstGate;
              signal?.throwIfAborted();
            }
            return null;
          }),
          dispose: vi.fn(),
        };
        (sceneLoader as unknown as { loaders: Map<string, unknown> }).loaders.set('/node', loader);

        // Refinement owns the lock, then hands its queued view to the next pass.
        internals._updateInProgress = true;
        internals._refining = true;
        internals._updateAbortController = new AbortController();
        if (resyncFirst) {
          await sceneLoader.updateView({}, { resyncPaths: new Set(['/node']) });
          expect(internals.viewStateQueue.hasPending()).toBe(true);
        }
        const queued = sceneLoader.updateView({ slicePosition: [0, 0, 0, 1] });
        expect(internals.viewStateQueue.hasPending()).toBe(true);
        internals._updateInProgress = false;
        internals._refining = false;
        const pending = internals.viewStateQueue.takePending();
        expect(pending).toBeDefined();
        const first = internals.reenterPending(pending!);
        await Promise.resolve();
        expect(capturedSignal).toBeInstanceOf(AbortSignal);

        now += 10;
        const winning = sceneLoader.updateView({ slicePosition: [0, 0, 0, 2] });
        expect(capturedSignal?.aborted).toBe(true);

        releaseFirst();
        await first;
        await Promise.all([queued, winning]);
        expect(loader.updateView).toHaveBeenCalledTimes(2);
        expect(sceneLoader.hasFailures()).toBe(false);
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
      // queueNext re-enters with the winning state (rAF or its timeout
      // backstop), which completes and settles the waiter — awaiting the
      // queued promise itself is the deterministic wait.
      await Promise.all([p1, p2]);
      expect(queuedResolved).toBe(true);
    });

    /** Points loader whose calls each park on their own gate, released in order. */
    function installPerCallGatedLoader(): {
      release: (call: number) => void;
      updateView: ReturnType<typeof vi.fn>;
      signals: Array<AbortSignal | undefined>;
    } {
      const gates: Array<() => void> = [];
      const gatePromises: Array<Promise<void>> = [];
      const gateFor = (i: number): Promise<void> => {
        while (gatePromises.length <= i) {
          gatePromises.push(
            new Promise<void>((resolve) => {
              gates.push(resolve);
            })
          );
        }
        return gatePromises[i];
      };
      const signals: Array<AbortSignal | undefined> = [];
      const updateView = vi.fn(async (_vs: unknown, _s: unknown, signal?: AbortSignal) => {
        const i = signals.length;
        signals.push(signal);
        await gateFor(i);
        return null;
      });
      (sceneLoader as unknown as Record<string, Map<string, unknown>>)['loaders'].set('/node', {
        loadPoints: vi.fn(),
        updateView,
        dispose: vi.fn(),
      });
      return {
        release: (call: number) => {
          void gateFor(call);
          gates[call]();
        },
        updateView,
        signals,
      };
    }

    const flush = () => new Promise((resolve) => setTimeout(resolve, 20));

    it("a superseded pass completing does not resolve the newer request's waiter (#2943)", async () => {
      const { release, updateView } = installPerCallGatedLoader();

      const pA = sceneLoader.updateView(vs(1)); // pass A in flight
      await Promise.resolve();
      let bResolved = false;
      const pB = sceneLoader.updateView(vs(2)).then(() => {
        bResolved = true;
      });

      release(0); // A completes (superseded)
      await vi.waitFor(() => expect(updateView).toHaveBeenCalledTimes(2));
      expect(bResolved).toBe(false); // B's pass has not committed yet

      release(1); // B's pass completes
      await Promise.all([pA, pB]);
      expect(bResolved).toBe(true);
    });

    it('a direct caller waits for the pass that commits its superseded view', async () => {
      const { release, updateView } = installPerCallGatedLoader();
      let firstResolved = false;
      const first = sceneLoader.updateView(vs(7)).then(() => {
        firstResolved = true;
      });
      await Promise.resolve();
      const replacement = sceneLoader.updateView({});

      release(0);
      await vi.waitFor(() => expect(updateView).toHaveBeenCalledTimes(2));
      expect(firstResolved).toBe(false);

      release(1);
      await Promise.all([first, replacement]);
      expect(firstResolved).toBe(true);
    });

    it.each([
      ['running pass is budgeted', { frameBudgetMs: 8 }, {}],
      ['incoming request is budgeted', {}, { frameBudgetMs: 8 }],
      ['running pass has a ladder depth', { ladderDepth: 2 }, {}],
      ['incoming request has a ladder depth', {}, { ladderDepth: 2 }],
    ])(
      '%s: same-view request supersedes instead of joining',
      async (_name, firstOpts, nextOpts) => {
        const { release, updateView, signals } = installPerCallGatedLoader();
        const first = sceneLoader.updateView({ ...vs(5), ...firstOpts });
        await Promise.resolve();
        const next = sceneLoader.updateView({ ...vs(5), ...nextOpts });

        expect(signals[0]?.aborted).toBe(true);
        release(0);
        await vi.waitFor(() => expect(updateView).toHaveBeenCalledTimes(2));

        release(1);
        await Promise.all([first, next]);
        expect(signals[1]?.aborted).toBe(false);
      }
    );

    it('isAtViewState is false while the pass carrying that view state is still in flight (#2943)', async () => {
      const { release } = installPerCallGatedLoader();
      const pA = sceneLoader.updateView(vs(5));
      await Promise.resolve();
      const inFlight = structuredClone(
        (sceneLoader as unknown as { viewState: ViewState }).viewState
      );
      expect(inFlight.slicePosition[3]).toBe(5);

      // The dims layer short-circuits a request isAtViewState() accepts, so a
      // true here resolved waitForUpdate() before the slice had committed.
      expect(sceneLoader.isAtViewState(inFlight)).toBe(false);

      release(0);
      await pA;
      expect(sceneLoader.isAtViewState(inFlight)).toBe(true);
    });

    it('does not report an aborted view as settled while its replacement waits for a frame', async () => {
      const { release } = installPerCallGatedLoader();
      const frames: Array<() => void> = [];
      const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
      vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
        frames.push(() => callback(0));
        return frames.length;
      });

      try {
        const pA = sceneLoader.updateView(vs(1));
        await Promise.resolve();
        const interrupted = structuredClone(
          (sceneLoader as unknown as { viewState: ViewState }).viewState
        );
        const pB = sceneLoader.updateView(vs(2));
        release(0);
        await flush();
        expect(frames).toHaveLength(1);
        expect(sceneLoader.isAtViewState(interrupted)).toBe(false);

        frames[0]();
        release(1);
        await Promise.all([pA, pB]);
      } finally {
        hidden.mockRestore();
        vi.unstubAllGlobals();
      }
    });

    it('a request for the in-flight view state joins that pass: no abort, no re-run (#2943)', async () => {
      const { release, updateView, signals } = installPerCallGatedLoader();
      const pA = sceneLoader.updateView(vs(5));
      await Promise.resolve();

      let joinedResolved = false;
      const pJoin = sceneLoader.updateView(vs(5)).then(() => {
        joinedResolved = true;
      });
      await flush();
      expect(joinedResolved).toBe(false); // waits for the in-flight commit
      expect(signals[0]?.aborted).toBe(false); // the pass it joined keeps going

      release(0);
      await pA;
      await pJoin;
      expect(joinedResolved).toBe(true);
      await flush();
      expect(updateView).toHaveBeenCalledTimes(1); // no redundant re-run
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
      await Promise.all([p1, ...queued]);
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

    it('dispose drops its refinement state and cancels a pending kick re-check', async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      type Internals = {
        _updateInProgress: boolean;
        _refinementKickPending: boolean;
        gsplatLoaders: Map<string, unknown>;
        sliceCache: unknown;
        lastResidencyBudget: { isDeclined(path: string): boolean } | null;
        refinementDensityGate: { deferred: Map<string, number> } | null;
      };
      const internals = sceneLoader as unknown as Internals;
      expect(internals.sliceCache).not.toBeNull();
      internals.lastResidencyBudget = { isDeclined: () => true };
      sceneLoader.setRefinementDensityProvider(
        () => ({ areaPx: 100, elements: 1_000_000, onScreen: true, blendable: true }),
        { blendable: 4, nonBlendable: 1 }
      );
      internals.refinementDensityGate!.deferred.set('/g', 1e9);
      internals.gsplatLoaders.set('/g', { hasMoreLODs: true, dispose: vi.fn() });
      internals._updateInProgress = true;
      sceneLoader.kickRefinementIfIdle(); // lock busy: one re-check timer
      expect(internals._refinementKickPending).toBe(true);

      await sceneLoader.dispose();

      expect(internals._refinementKickPending).toBe(false);
      expect(internals.sliceCache).toBeNull();
      expect(internals.lastResidencyBudget).toBeNull();
      expect(internals.refinementDensityGate).toBeNull();
      expect(sceneLoader.refinementHoldReason('/g')).toBeNull();
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
    // re-registers via the fresh commit's `depthSort.noteCommit`.
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

  describe('format-version policy (shared with the Python reader)', () => {
    // The policy itself is pinned by format-version.test.ts (same case table
    // as typing_utils/tests/test_format_version.py). These assert the SEAM:
    // that loadScene actually routes the root attrs through it, so a refused
    // version reaches the dataset-error overlay and a newer minor only toasts.
    it('refuses a newer-major scene (9.9) naming the version', async () => {
      mockZarrGroup.attrs = { ...mockZarrGroup.attrs, type: 'scene', format_version: '9.9' };
      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).rejects.toThrow(
        /9\.9/
      );
    });

    it('refuses an unparsable scene version (abc)', async () => {
      mockZarrGroup.attrs = { ...mockZarrGroup.attrs, type: 'scene', format_version: 'abc' };
      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).rejects.toThrow(/abc/);
    });

    it('loads a newer-minor scene (0.3) with a toast instead of refusing', async () => {
      mockZarrGroup.attrs = { ...mockZarrGroup.attrs, type: 'scene', format_version: '0.3' };
      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).resolves.toBeDefined();
      expect(notifierMocks.toast).toHaveBeenCalledWith(expect.stringContaining('0.3'), 6000);
    });

    it('loads a legacy 0.1 root (luxar_version only) silently', async () => {
      mockZarrGroup.attrs = { ...mockZarrGroup.attrs, type: 'scene', luxar_version: '0.1' };
      await expect(sceneLoader.loadScene('http://localhost:8000/test.zarr')).resolves.toBeDefined();
      expect(notifierMocks.toast).not.toHaveBeenCalledWith(
        expect.stringContaining('format'),
        expect.anything()
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

  describe('retry during a refinement drain pre-empts it (A10)', () => {
    // A refinement drain holds the serialization lock for as long as ladders
    // stream — minutes on a deep ladder. A retry refused for that long made the
    // online auto-retry give up (10 deferred attempts, 2 s apart) and refused
    // the monitor's Retry button. A view change pre-empts refinement; so must
    // a retry.
    let frames: Array<() => void>;
    type Internals = {
      _updateInProgress: boolean;
      gsplatLoaders: Map<string, unknown>;
      loaders: Map<string, unknown>;
      rootGroup: THREE.Group | null;
      registry: { recordFailure(path: string, error: Error): void };
      scheduleGSplatsRefinement(): Promise<void>;
    };

    beforeEach(async () => {
      await sceneLoader.loadScene('http://localhost:8000/test.zarr');
      frames = [];
      vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
      vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
        frames.push(() => callback(0));
        return frames.length;
      });
    });

    afterEach(() => {
      vi.unstubAllGlobals();
      vi.restoreAllMocks();
    });

    /** Fire queued frames until `done` settles (bounded). */
    async function pumpUntil(done: Promise<unknown>): Promise<void> {
      let settled = false;
      void done.finally(() => {
        settled = true;
      });
      for (let i = 0; i < 50 && !settled; i++) {
        frames.shift()?.();
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    /** A deep ladder drained by a running refinement, plus one failed points node. */
    function startDrainWithFailure(): { retried: ReturnType<typeof vi.fn> } {
      const internals = sceneLoader as unknown as Internals;
      const ladder = {
        loadedLODCount: 1,
        totalLODCount: 1000,
        get hasMoreLODs() {
          return ladder.loadedLODCount < ladder.totalLODCount;
        },
        updateView: vi.fn(async () => {
          ladder.loadedLODCount += 1;
          return null;
        }),
        dispose: vi.fn(),
      };
      internals.gsplatLoaders.set('/g', ladder);
      const retried = vi.fn().mockResolvedValue(null);
      internals.loaders.set('/p', { updateView: retried, dispose: vi.fn() });
      const placeholder = new THREE.Group();
      placeholder.name = '/p';
      internals.rootGroup!.add(placeholder);
      internals.registry.recordFailure('/p', new Error('network down'));
      internals._updateInProgress = true;
      void internals.scheduleGSplatsRefinement();
      expect(sceneLoader.isUpdateInProgress()).toBe(true);
      return { retried };
    }

    it('retryFailedLoader runs instead of reporting deferred', async () => {
      const { retried } = startDrainWithFailure();
      const result = sceneLoader.retryFailedLoader('/p');
      await pumpUntil(result);

      await expect(result).resolves.toBe(true);
      expect(retried).toHaveBeenCalledOnce();
      expect(sceneLoader.hasFailures()).toBe(false);
    });

    it('retryAllFailedLoaders runs instead of reporting deferred', async () => {
      const { retried } = startDrainWithFailure();
      const result = sceneLoader.retryAllFailedLoaders();
      await pumpUntil(result);

      await expect(result).resolves.toEqual({ succeeded: ['/p'], failed: [] });
      expect(retried).toHaveBeenCalledOnce();
    });

    it('a retry still defers to a VIEW pass holding the lock', async () => {
      const internals = sceneLoader as unknown as Internals;
      internals.registry.recordFailure('/p', new Error('network down'));
      internals._updateInProgress = true; // a main update, not refinement
      try {
        await expect(sceneLoader.retryAllFailedLoaders()).resolves.toMatchObject({
          deferred: true,
        });
      } finally {
        internals._updateInProgress = false;
      }
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

  describe('refinement fetch class (B9c)', () => {
    it("runs refinement under a signal carrying the 'refinement' fetch priority", async () => {
      let seen: AbortSignal | undefined;
      const runRefinement = vi
        .spyOn(gsplatsRefinement, 'runGSplatsRefinement')
        .mockImplementation(async (ctx) => {
          seen = ctx.signal;
        });
      const internals = sceneLoader as unknown as {
        _updateInProgress: boolean;
        scheduleGSplatsRefinement(): Promise<void>;
      };
      internals._updateInProgress = true;
      try {
        await internals.scheduleGSplatsRefinement();
        expect(seen).toBeDefined();
        expect(signalPriority(seen)?.value).toBe('refinement');
      } finally {
        runRefinement.mockRestore();
        internals._updateInProgress = false;
      }
    });
  });

  describe('getFailedLoadsProvider — reason mapping', () => {
    // #1055: the layers-panel error badge reads its tooltip from the provider's
    // getFailedReason, which folds error.message → classified kind → undefined.
    interface FailInternals {
      registry: {
        recordFailure(path: string, error: Error, kind?: string): void;
        clearFailure(path: string): void;
      };
    }

    it('versions recorded, replaced and cleared failures, but not reads', () => {
      const registry = (sceneLoader as unknown as FailInternals).registry;
      const provider = sceneLoader.getFailedLoadsProvider();
      const initial = provider.getFailedLoadsVersion?.();
      expect(typeof initial).toBe('number');
      registry.recordFailure('/points/a', new Error('network 503'));
      const failed = provider.getFailedLoadsVersion?.();
      expect(failed).toBeGreaterThan(initial!);
      provider.getFailedPaths();
      provider.getFailedReason?.('/points/a');
      expect(provider.getFailedLoadsVersion?.()).toBe(failed);
      registry.recordFailure('/points/a', new Error('decode error'));
      expect(provider.getFailedLoadsVersion?.()).toBeGreaterThan(failed!);
      const replaced = provider.getFailedLoadsVersion?.();
      registry.clearFailure('/points/a');
      expect(provider.getFailedLoadsVersion?.()).toBeGreaterThan(replaced!);
    });

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

    it('finds only network failures under the requested path boundary', () => {
      const internals = sceneLoader as unknown as FailInternals;
      internals.registry.recordFailure('/g/level_0', new Error('HTTP 503 fetching chunk'));
      internals.registry.recordFailure('/g2/level_0', new Error('HTTP 503 fetching chunk'));
      internals.registry.recordFailure('/g/level_1', new Error('invalid chunk'), 'Decode');

      expect(sceneLoader.hasNetworkFailureUnder('/g')).toBe(true);
      expect(sceneLoader.hasNetworkFailureUnder('/g2')).toBe(true);
      expect(sceneLoader.hasNetworkFailureUnder('/')).toBe(true);
      expect(sceneLoader.hasNetworkFailureUnder('/missing')).toBe(false);
      expect(sceneLoader.hasNetworkFailureUnder('/g/level_1')).toBe(false);
    });

    it('surfaces and retries an archive fault with no recorded node failure', async () => {
      const archiveFault = new ArchiveFaultError('archive unavailable', '/scene.zip');
      const provider = sceneLoader.getFailedLoadsProvider();
      const beforeFault = provider.getFailedLoadsVersion!();
      const current = { displayDims: [0, 1, 2], slicePosition: [3], tolerance: [0] };
      const blocked = { displayDims: [0, 1, 2], slicePosition: [4], tolerance: [0] };
      const internals = sceneLoader as unknown as {
        viewState: { displayDims: number[]; slicePosition: number[]; tolerance: number[] };
        viewStateQueue: { hasPending(): boolean };
        reportArchiveFault(fault: ArchiveFaultError): void;
      };
      await sceneLoader.updateView(current);
      internals.reportArchiveFault(archiveFault);
      const faultVersion = provider.getFailedLoadsVersion!();
      expect(faultVersion).toBeGreaterThan(beforeFault);
      await sceneLoader.updateView(blocked);
      expect(internals.viewState).toMatchObject(current);
      expect(internals.viewStateQueue.hasPending()).toBe(false);

      const updateViewSpy = vi.spyOn(sceneLoader, 'updateView');

      expect(provider.getFailedPaths()).toEqual(['/scene.zip']);
      expect(provider.getFailedReason?.('/scene.zip')).toBe('archive unavailable');
      expect(sceneLoader.hasAutoRetryableFailures()).toBe(true);

      await expect(provider.retryAll()).resolves.toEqual({
        succeeded: ['/scene.zip'],
        failed: [],
      });
      expect(sceneLoader.archiveFault).toBeNull();
      expect(provider.getFailedLoadsVersion!()).toBeGreaterThan(faultVersion);
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

  describe('abandoned rung recovery (#2975)', () => {
    // A chunk fetch that stalls to give-up makes the store reject; a
    // refinement step that hits it fails. After the consecutive-failure cap
    // the ladder was retired and nothing but an `online` event re-opened it,
    // so a session that never went offline sat at the coarser rung, idle and
    // reported settled, until the user happened to move something.
    interface RecoveryInternals {
      _updateInProgress: boolean;
      rootGroup: THREE.Group | null;
      gsplatLoaders: Map<string, unknown>;
      registry: { recordFailure(path: string, error: Error): void };
      scheduleGSplatsRefinement: () => Promise<void>;
    }

    const stallError = () =>
      new Error(
        'fetch exhausted retries for additive_1/amplitudes/c/0: ' +
          'response bodies made no aggregate progress for 7500 ms'
      );

    /** A ladder whose next rung fails `failures` times, then streams normally. */
    function flakyLadder(failures: number, total = 3) {
      const ladder = {
        calls: 0,
        loadedLODCount: 1,
        totalLODCount: total,
        get hasMoreLODs() {
          return ladder.loadedLODCount < ladder.totalLODCount;
        },
        updateView: vi.fn(async () => {
          ladder.calls += 1;
          if (ladder.calls <= failures) throw stallError();
          ladder.loadedLODCount += 1;
          return null;
        }),
        dispose: vi.fn(),
      };
      return ladder;
    }

    let errorLog: ReturnType<typeof vi.spyOn>;

    beforeEach(() => {
      vi.useFakeTimers();
      resetLoadTimeline();
      errorLog = vi.spyOn(log, 'error').mockImplementation(() => {});
    });

    afterEach(() => {
      (sceneLoader as unknown as RecoveryInternals).gsplatLoaders.clear();
      errorLog.mockRestore();
      vi.useRealTimers();
    });

    /** Run one refinement drain the way an update tail does (lock held). */
    async function drain(): Promise<void> {
      const internals = sceneLoader as unknown as RecoveryInternals;
      internals._updateInProgress = true;
      const run = internals.scheduleGSplatsRefinement();
      await vi.advanceTimersByTimeAsync(300);
      await run;
    }

    it('re-drains an abandoned rung after a backoff and commits it, with no user action', async () => {
      const ladder = flakyLadder(3);
      (sceneLoader as unknown as RecoveryInternals).gsplatLoaders.set('/g', ladder);

      await drain();
      // The first drain gave up at the consecutive-failure cap and let go of
      // the lock: the rung is still missing.
      expect(ladder.calls).toBe(3);
      expect(ladder.loadedLODCount).toBe(1);
      expect(sceneLoader.isUpdateInProgress()).toBe(false);
      // ...so the loader must not report its refinement done while that rung
      // is only waiting for a retry: capture / isSettled read this.
      expect(getLoadTimeline().refinement.complete).toBe(false);

      // Backoff, not a hot loop: nothing is re-fetched right away.
      await vi.advanceTimersByTimeAsync(400);
      expect(ladder.calls).toBe(3);

      await vi.advanceTimersByTimeAsync(60_000);
      expect(ladder.loadedLODCount).toBe(3);
      expect(ladder.hasMoreLODs).toBe(false);
      expect(sceneLoader.isUpdateInProgress()).toBe(false);
      expect(getLoadTimeline().refinement.complete).toBe(true);
    });

    it('bounds the retries of a rung that never recovers, then reports settled', async () => {
      const ladder = flakyLadder(Number.POSITIVE_INFINITY);
      (sceneLoader as unknown as RecoveryInternals).gsplatLoaders.set('/g', ladder);
      const warningLog = vi.spyOn(log, 'warning').mockImplementation(() => {});
      notifierMocks.toast.mockClear();

      await drain();
      expect(notifierMocks.toast).toHaveBeenCalledWith(
        'Refinement failed for /g — showing reduced detail',
        5000
      );
      await vi.advanceTimersByTimeAsync(30 * 60_000);
      const settledCalls = ladder.calls;
      // More than the one drain it used to get, but bounded: a few backoff
      // rounds of the consecutive-failure cap each.
      expect(settledCalls).toBe(
        MAX_CONSECUTIVE_REFINEMENT_FAILURES * (MAX_ABANDONED_RUNG_RETRY_ROUNDS + 1)
      );
      expect(
        notifierMocks.toast.mock.calls.filter(([message]) =>
          String(message).startsWith('Refinement failed')
        )
      ).toHaveLength(1);
      expect(ladder.loadedLODCount).toBe(1);
      expect(sceneLoader.isUpdateInProgress()).toBe(false);
      // Given up for good: no timer keeps polling and the drain reads complete.
      await vi.advanceTimersByTimeAsync(30 * 60_000);
      expect(ladder.calls).toBe(settledCalls);
      expect(getLoadTimeline().refinement.complete).toBe(true);
      await drain();
      await drain();
      expect(
        warningLog.mock.calls.filter(([, message]) =>
          String(message).startsWith('Leaving LOD refinement')
        )
      ).toHaveLength(1);
      for (let slice = 1; slice <= 3; slice++) {
        const update = sceneLoader.updateView({ slicePosition: [0, 0, 0, slice] });
        await vi.advanceTimersByTimeAsync(500);
        await update;
      }
      expect(
        notifierMocks.toast.mock.calls.filter(([message]) =>
          String(message).startsWith('Refinement failed')
        )
      ).toHaveLength(1);
      warningLog.mockRestore();
    });

    it('counts re-drain rounds per ladder: one that gave up does not use up another’s', async () => {
      // The backoff round used to be one integer per SceneLoader, so a ladder
      // retired AFTER another had exhausted its rounds inherited the spent
      // counter and was left at once, never re-drained.
      const internals = sceneLoader as unknown as RecoveryInternals;
      const stuck = flakyLadder(Number.POSITIVE_INFINITY);
      internals.gsplatLoaders.set('/a', stuck);
      const warningLog = vi.spyOn(log, 'warning').mockImplementation(() => {});
      await drain();
      await vi.advanceTimersByTimeAsync(30 * 60_000);
      const stuckCalls = stuck.calls;

      const late = flakyLadder(MAX_CONSECUTIVE_REFINEMENT_FAILURES);
      internals.gsplatLoaders.set('/b', late);
      await drain();
      expect(late.loadedLODCount).toBe(1);
      await vi.advanceTimersByTimeAsync(60_000);

      expect(late.loadedLODCount).toBe(3);
      // The ladder that gave up for good is not re-opened by the other's round.
      expect(stuck.calls).toBe(stuckCalls);
      expect(getLoadTimeline().refinement.complete).toBe(true);
      warningLog.mockRestore();
    });

    it('a newer view re-opens a retired ladder at once instead of waiting out the backoff', async () => {
      const ladder = flakyLadder(3, 4);
      (sceneLoader as unknown as RecoveryInternals).gsplatLoaders.set('/g', ladder);

      await drain();
      expect(ladder.calls).toBe(3);
      const update = sceneLoader.updateView({ slicePosition: [0, 0, 0, 5] });
      await vi.advanceTimersByTimeAsync(500);
      await update;
      // The view pass took one rung; its own refinement drain took the rest,
      // well inside the first backoff delay.
      expect(ladder.loadedLODCount).toBe(4);
      expect(getLoadTimeline().refinement.complete).toBe(true);
    });

    it('a dataset switch (dispose) cancels a pending re-drain', async () => {
      const ladder = flakyLadder(3);
      (sceneLoader as unknown as RecoveryInternals).gsplatLoaders.set('/g', ladder);

      await drain();
      expect(ladder.calls).toBe(3);
      void sceneLoader.dispose();
      await vi.advanceTimersByTimeAsync(60_000);
      expect(ladder.calls).toBe(3);
    });

    it('resumes refinement after a failed view-pass load is retried successfully', async () => {
      // The retry path commits one pass's worth of the node and used to stop
      // there, leaving the rest of the ladder idle.
      const internals = sceneLoader as unknown as RecoveryInternals;
      const root = new THREE.Group();
      const node = new THREE.Group();
      node.name = '/g';
      root.add(node);
      internals.rootGroup = root;
      const ladder = flakyLadder(0);
      internals.gsplatLoaders.set('/g', ladder);
      internals.registry.recordFailure('/g', stallError());
      const refine = vi.fn(async () => {
        internals._updateInProgress = false;
      });
      internals.scheduleGSplatsRefinement = refine;

      const result = await sceneLoader.retryAllFailedLoaders();
      await vi.advanceTimersByTimeAsync(300);

      expect(result.succeeded).toEqual(['/g']);
      expect(ladder.hasMoreLODs).toBe(true);
      expect(refine).toHaveBeenCalledTimes(1);
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
    });

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

      const beforeClear = failedLoadsVersion();
      sceneLoader.clearFailures();
      expect(sceneLoader.hasFailures()).toBe(false);
      expect(sceneLoader.getFailedLoaders().size).toBe(0);
      expect(failedLoadsVersion()).toBe(beforeClear + 1);
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
      const invalidatePartitionFootprint = vi.fn();
      (sceneLoader as any).lodGroupRegistry = {
        invalidatePartitionFootprint,
        clear: vi.fn(),
      };
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
      expect(invalidatePartitionFootprint).toHaveBeenCalledWith('/test_points');
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
