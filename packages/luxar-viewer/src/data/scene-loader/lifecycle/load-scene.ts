/**
 * Initial scene-load orchestrator.
 *
 * Sequence:
 *   1. Reset monitor + abort any in-flight worker tasks from a prior dataset.
 *   2. Dispose the previous loader (caching store, GPU pool, L0 cache,
 *      colormap LUTs, monitor closures) — awaited so stale writes drain.
 *   3. Fresh AbortController wired into the worker pool's signal.
 *   4. Reset the predictive-prefetch baseline (otherwise the first
 *      updateView extrapolates from the prior dataset's slicePosition).
 *   5. Set up L0 + L1/L2 caches; open the zarr root.
 *   6. Build the empty root THREE.Group + initialize scene dimensions
 *      from `scene_dimensions` metadata.
 *   7. Surface a toast when ndim > 16 (WASM ceiling — TS fallback works
 *      but is slower).
 *   8. Persist `viewer_config` + `position_bounds` onto the root group's
 *      userData for the UI to read.
 *   9. Build the scene graph (zarr group enumeration → SceneNode tree).
 *  10. Recursively load every leaf via `loadSceneNodes`.
 *  11. Load overlay configs (screen-space annotations).
 *  12. Wire post-load monitor providers (cache stats, loader maps, etc.).
 *  13. Schedule progressive GSplats LOD refinement when any multi-LOD
 *      loader still has higher LODs to fetch (the initial-load path
 *      only fetches LOD 0; without this kick, higher LODs would not
 *      load until the user's first updateView).
 *
 * The orchestrator (`SceneLoader.loadScene`) is a thin wrapper that
 * builds the ctx, calls this helper, and returns the root group. The
 * helper writes new resource references back through the ctx setters.
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules, LogEmoji } from '../../../utils/log';
import { notifier } from '../../../utils/cross-layer/notifier';
import { getWorkerPool } from '../../../workers/worker-pool';
import { ZarrSceneAttrs } from '../../../types/zarr';
import type { LoaderConfig, SceneNode, ViewState } from '../../data-loader-types';
import type { DataLoader } from '../../data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import type { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { UpdateProfiler } from '../../../profiling/update-profiler';
import type { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import type { DecompressedChunkCache } from '../../../cache/decompressed-chunk-cache';
import type { SceneLoaderMonitorPort } from '../../scene-loader-monitor-port';
import type { LODGroupRegistry } from '../../../scene/lod-group-registry';
import { setupCaches } from '../cache/cache-setup';
import { wireMonitorAfterLoad } from '../monitor/monitor-wiring';
import { loadOverlayConfigs } from '../../loaders';
import { buildSceneGraph } from '../nodes/build-scene-graph';
import { loadSceneNodes } from '../nodes/load-scene-nodes';
import type { NodeBuildCtx } from '../nodes/build-ctx';

/**
 * Narrow context the load-scene path needs. Captures the orchestrator's
 * mutable resource references via getter callbacks so the helper can
 * read the latest snapshot of `gpuBufferPool` / `monitor` / etc., and
 * uses explicit setters to write the new dataset-scoped resources back.
 */
export interface LoadSceneCtx {
  /** Loader configuration (cache flags, prefetch flags). */
  config: LoaderConfig;
  /** Current view state — read after `initializeSceneDimensions` for logging. */
  viewState: () => ViewState;
  /** Active per-type loader maps for the post-load monitor wiring. */
  loaders: Map<string, DataLoader>;
  linesLoaders: Map<string, LinesDataLoader>;
  gsplatLoaders: Map<string, GSplatsDataLoader>;
  /** Current GPU buffer pool reference (may be null when disabled). */
  gpuBufferPool: () => GPUBufferPool | null;
  /** Current monitor reference. */
  monitor: () => SceneLoaderMonitorPort | null;
  /** Profiler reference. */
  profiler: UpdateProfiler | null;
  /** LOD-group registry for the live LOD-progress provider (substitutive levels). */
  lodGroupRegistry: LODGroupRegistry | null;

  // Lifecycle callbacks the orchestrator owns:
  normalizeURL(url: string): string;
  /** Await any previous loader's teardown before constructing the new one. */
  dispose(): Promise<void>;
  /** Reset the predictive-prefetch baseline. */
  clearViewStatePrev(): void;
  /**
   * Validate `scene_dimensions` blob and update the loader's viewState.
   * Implementation in `initialize-scene-dimensions.ts`; the orchestrator
   * writes the result back to its own viewState field.
   */
  initializeSceneDimensions(sceneDims: unknown): void;
  /**
   * Build the NodeBuildCtx used by the recursive scene-graph walk.
   * Built once before `loadSceneNodes` runs so the leaves all see the
   * same viewState snapshot.
   */
  makeNodeBuildCtx(): NodeBuildCtx;
  /** Stats helper used by the post-load monitor wiring. */
  updateVisibleCountsInMonitor(): void;
  /** Kick the GSplats LOD refinement loop after initial load. */
  scheduleGSplatsRefinement(): void;

  // Resource-write setters — orchestrator nulls/sets its own fields.
  setDatasetAbortController(controller: AbortController | null): void;
  setCachingStore(store: MultiLevelCachingStore | null): void;
  setL0Cache(cache: DecompressedChunkCache | null): void;
  setZarrStore(store: zarr.Readable): void;
  setRootGroup(group: THREE.Group): void;
  setSceneGraph(graph: SceneNode): void;
  setUpdateInProgress(value: boolean): void;
  /** Current abort controller — `loadScene` aborts it before disposing. */
  getDatasetAbortController(): AbortController | null;
}

/**
 * Execute the full initial-load sequence and return the populated root
 * THREE.Group. Mutates the orchestrator's resource references via the
 * ctx setters.
 */
export async function loadScene(url: string, ctx: LoadSceneCtx): Promise<THREE.Group> {
  log.custom(LogEmoji.SCENE, Modules.SCENE_LOADER, `Loading scene from ${url}`);

  // Clear any existing loaders from monitor before loading new scene
  ctx.monitor()?.disconnectAllLoaders();

  // Abort any in-flight worker tasks queued by the previous dataset.
  // Doing this BEFORE `dispose()` settles already-racing
  // `runWithTimeout` callers immediately so they unwind without
  // waiting for the worker tasks to complete — the worker keeps
  // executing the WASM kernels to completion (no WASM cancellation),
  // but the results are dropped.
  const prevAbort = ctx.getDatasetAbortController();
  if (prevAbort) {
    prevAbort.abort();
    ctx.setDatasetAbortController(null);
  }
  getWorkerPool().setAbortSignal(undefined);

  // Dispose of any existing loaders. Awaited so the previous caching
  // store fully drains (prefetcher tear-down, OPFS metadata flush,
  // validation cancellation) before we construct the next one — without
  // this, rapid dataset switches let an old store's writes land after
  // the new store starts initialising.
  if (ctx.loaders.size > 0) {
    await ctx.dispose();
  }

  // Fresh abort source for THIS dataset; wire into the worker pool so
  // every subsequent `runWithTimeout` races against it.
  const datasetAbortController = new AbortController();
  ctx.setDatasetAbortController(datasetAbortController);
  getWorkerPool().setAbortSignal(datasetAbortController.signal);

  // S6: reset per-loader prefetch predictor state. Without this,
  // the first updateView on a new dataset would extrapolate from
  // the prior dataset's slicePosition, producing wild prefetch
  // targets.
  ctx.clearViewStatePrev();

  const cacheResult = await setupCaches(ctx.normalizeURL(url), {
    noCache: ctx.config.noCache,
    cacheDebug: ctx.config.cacheDebug,
    clearCache: ctx.config.clearCache,
    noPrefetch: ctx.config.noPrefetch,
    prefetchDebug: ctx.config.prefetchDebug,
  });
  ctx.setL0Cache(cacheResult.l0Cache);
  ctx.setCachingStore(cacheResult.cachingStore);
  const zarrStore = (await zarr.openStore(cacheResult.rawStore)) as zarr.Readable;
  ctx.setZarrStore(zarrStore);

  // Create root THREE.js group
  const rootGroup = new THREE.Group();
  rootGroup.name = 'LuxarScene';
  ctx.setRootGroup(rootGroup);

  // Load scene metadata
  const rootLoc = zarr.root(zarrStore);
  const rootZarrGroup = await zarr.open(rootLoc, { kind: 'group' });
  const sceneAttrs = rootZarrGroup.attrs as ZarrSceneAttrs;

  // Initialize scene dimensions - CRITICAL for extend_to_all feature
  if (sceneAttrs?.scene_dimensions) {
    ctx.initializeSceneDimensions(sceneAttrs.scene_dimensions);
    rootGroup.userData.sceneDimensions = sceneAttrs.scene_dimensions;

    // Log dimension initialization status for debugging
    const vs = ctx.viewState();
    const ndim = vs.dimensions?.length ?? 0;
    if (ndim > 0) {
      log.success(
        Modules.SCENE_LOADER,
        `Scene dimensions initialized: ${ndim} dimensions, ` +
          `displayed=[${vs.displayDims.join(', ')}]`
      );
    }

    // Surface a user-facing toast when the scene exceeds the WASM
    // 16-dim ceiling — the worker auto-falls-back to TS, which is
    // correct but slower, and silent fallback can confuse users
    // wondering why interaction feels sluggish.
    if (ndim > 16) {
      notifier.toast(
        `Scene has ${ndim} dimensions — WASM acceleration limited to 16D, using TypeScript fallback. ` +
          'Consider reducing dimensions for better performance.',
        5000
      );
    }
  } else {
    log.warning(
      Modules.SCENE_LOADER,
      'No scene_dimensions found in scene metadata. extend_to_all features will not work.'
    );
  }

  // Extract viewer_config if present (Python API scene defaults)
  if (sceneAttrs?.viewer_config) {
    rootGroup.userData.viewerConfig = sceneAttrs.viewer_config;
    log.info(
      Modules.SCENE_LOADER,
      `Viewer config found in zarr: ${Object.keys(sceneAttrs.viewer_config).join(', ')}`
    );
  }

  // Store scene-level position bounds (from Python compiler)
  // These bounds represent the full dataset extent, available immediately without loading points
  if (sceneAttrs?.position_bounds) {
    rootGroup.userData.positionBounds = sceneAttrs.position_bounds;
    log.info(
      Modules.SCENE_LOADER,
      `Scene bounds loaded: min=[${sceneAttrs.position_bounds.min.join(', ')}], ` +
        `max=[${sceneAttrs.position_bounds.max.join(', ')}]`
    );
  }

  // Build scene graph
  const sceneGraph = await buildSceneGraph(rootLoc, sceneAttrs, zarrStore);
  ctx.setSceneGraph(sceneGraph);

  // Load points / lines / gsplats / nested groups recursively
  await loadSceneNodes(sceneGraph, rootGroup, rootLoc, ctx.makeNodeBuildCtx());

  // Load overlay configs (screen-space annotations)
  const overlayConfigs = await loadOverlayConfigs(zarrStore, rootLoc);
  if (overlayConfigs.length > 0) {
    rootGroup.userData.overlayConfigs = overlayConfigs;
    // Store base URL for image fetching
    rootGroup.userData.zarrBaseUrl = ctx.normalizeURL(url);
  }

  // Post-load monitor-tab provider wiring (extracted to
  // scene-loader/monitor-wiring.ts).
  wireMonitorAfterLoad({
    monitor: ctx.monitor(),
    cachingStore: cacheResult.cachingStore,
    l0Cache: cacheResult.l0Cache,
    cacheTelemetryState: cacheResult.telemetryState,
    gpuBufferPool: ctx.gpuBufferPool(),
    profiler: ctx.profiler,
    loaders: ctx.loaders,
    linesLoaders: ctx.linesLoaders,
    gsplatLoaders: ctx.gsplatLoaders,
    lodGroupRegistry: ctx.lodGroupRegistry,
    sceneGraph,
    updateVisibleCounts: () => ctx.updateVisibleCountsInMonitor(),
  });

  log.success(Modules.SCENE_LOADER, 'Scene loaded successfully');

  // Schedule progressive LOD refinement after initial load.
  // Each per-type loader maps may include progressive loaders that
  // only emit LOD 0 on the initial load — the refinement loop drains
  // their remaining LODs frame-by-frame. Symmetric across Points /
  // Lines / GSplats.
  const hasMore = (loader: unknown) => (loader as { hasMoreLODs?: boolean }).hasMoreLODs === true;
  const gsplatsNeed = [...ctx.gsplatLoaders.values()].some(hasMore);
  const pointsNeed = [...ctx.loaders.values()].some(hasMore);
  const linesNeed = [...ctx.linesLoaders.values()].some(hasMore);

  if (gsplatsNeed || pointsNeed || linesNeed) {
    log.info(
      Modules.SCENE_LOADER,
      'Scheduling post-load progressive LOD refinement ' +
        `(points=${pointsNeed} lines=${linesNeed} gsplats=${gsplatsNeed})`
    );
    // Hold the serialization lock during refinement so any updateView()
    // calls queue as _pendingViewState (which naturally cancels the
    // refinement loops). Each scheduler is responsible for releasing
    // the lock when its loop completes; the SceneLoader's
    // `scheduleProgressiveRefinement` orchestrates the three.
    ctx.setUpdateInProgress(true);
    ctx.scheduleGSplatsRefinement();
  }

  return rootGroup;
}
